"""
Measure streaming real-time factor (RTF) for LaCoSENet checkpoints.

The script reports steady-state RTF over the real audio interval and a second
RTF that includes flush padding. Steady-state RTF is the value used for
continuous streaming because flush work only happens at stream end.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from omegaconf import OmegaConf

from src.utils import compute_lookahead_from_config


def set_cpu_threads(num_threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def load_model_config(chkpt_dir: Path):
    config_path = chkpt_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return OmegaConf.load(config_path)


def resolve_checkpoint_file(chkpt_dir: Path, chkpt_file: str) -> str:
    if chkpt_file != "auto":
        return chkpt_file

    states_path = chkpt_dir / "states.th"
    if not states_path.exists():
        raise FileNotFoundError(f"states.th not found for --chkpt_file auto: {states_path}")

    states = torch.load(states_path, map_location="cpu", weights_only=False)
    best_models = states.get("best_models", [])
    if not best_models:
        raise ValueError(f"No best_models entry found in: {states_path}")

    step = best_models[0]["steps"]
    return f"model_{step}.th"


def make_streaming_model(args, encoder_lookahead: int, decoder_lookahead: int):
    if args.use_onnx:
        from src.models.onnx_export.streaming_wrapper import ONNXLaCoSENet

        return ONNXLaCoSENet.from_checkpoint(
            chkpt_dir=str(args.chkpt_dir),
            chkpt_file=args.chkpt_file,
            chunk_size=args.chunk_size,
            encoder_lookahead=encoder_lookahead,
            decoder_lookahead=decoder_lookahead,
            onnx_path=args.onnx_path,
            force_export=args.force_export,
            use_reshape_free=not args.no_reshape_free,
            num_threads=args.num_threads,
            verbose=not args.quiet,
        )

    from src.models.streaming.lacosenet import LaCoSENet

    return LaCoSENet.from_checkpoint(
        chkpt_dir=str(args.chkpt_dir),
        chkpt_file=args.chkpt_file,
        chunk_size=args.chunk_size,
        encoder_lookahead=encoder_lookahead,
        decoder_lookahead=decoder_lookahead,
        use_reshape_free=not args.no_reshape_free,
        fold_bn=args.optimize_cpu,
        device=args.device,
        verbose=not args.quiet,
    )


def run_streaming_once(streaming_model, audio: torch.Tensor, include_flush: bool) -> float:
    streaming_model.reset_state()

    if include_flush:
        flush_size = streaming_model.samples_per_chunk * (streaming_model.total_lookahead + 2)
        audio = torch.cat([audio, torch.zeros(flush_size, device=audio.device)])

    step = streaming_model.output_samples_per_chunk
    if torch.cuda.is_available() and getattr(streaming_model, "device", torch.device("cpu")).type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(audio), step):
            chunk = audio[offset:offset + step]
            if len(chunk) == 0:
                continue
            streaming_model.process_samples(chunk)

    if torch.cuda.is_available() and getattr(streaming_model, "device", torch.device("cpu")).type == "cuda":
        torch.cuda.synchronize()

    return time.perf_counter() - start


def mean(values: List[float]) -> float:
    return float(sum(values) / len(values))


def json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    try:
        return OmegaConf.to_container(value, resolve=True)
    except Exception:
        return str(value)


def build_result(
    args,
    conf,
    streaming_model,
    audio_samples: int,
    steady_times: List[float],
    total_times: List[float],
) -> Dict[str, Any]:
    sample_rate = int(conf.get("sampling_rate", 16000))
    duration_sec = audio_samples / sample_rate
    steady_mean = mean(steady_times)
    total_mean = mean(total_times)

    return {
        "model": Path(args.chkpt_dir).parent.name,
        "seed": Path(args.chkpt_dir).name,
        "chkpt_dir": str(args.chkpt_dir),
        "chkpt_file": args.chkpt_file,
        "backend": "onnx" if args.use_onnx else "pytorch",
        "use_reshape_free": not args.no_reshape_free,
        "optimize_cpu": args.optimize_cpu,
        "chunk_size": args.chunk_size,
        "duration_sec": duration_sec,
        "audio_samples": audio_samples,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "num_threads": args.num_threads,
        "encoder_lookahead": streaming_model.encoder_lookahead,
        "decoder_lookahead": streaming_model.decoder_lookahead,
        "total_lookahead": streaming_model.total_lookahead,
        "latency_ms": streaming_model.latency_ms,
        "samples_per_chunk": streaming_model.samples_per_chunk,
        "output_samples_per_chunk": streaming_model.output_samples_per_chunk,
        "steady_state_seconds": steady_mean,
        "total_seconds": total_mean,
        "streaming_rtf": steady_mean / duration_sec,
        "streaming_rtf_total": total_mean / duration_sec,
        "steady_state_times": steady_times,
        "total_times": total_times,
        "streaming_config": json_safe(streaming_model.streaming_config),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Measure LaCoSENet streaming RTF")
    parser.add_argument("--chkpt_dir", type=Path, required=True)
    parser.add_argument("--chkpt_file", type=str, default="best.th")
    parser.add_argument("--chunk_size", type=int, default=1)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use_onnx", action="store_true")
    parser.add_argument("--onnx_path", type=str, default=None)
    parser.add_argument("--force_export", action="store_true")
    parser.add_argument("--optimize_cpu", action="store_true")
    parser.add_argument("--no_reshape_free", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_cpu_threads(args.num_threads)

    conf = load_model_config(args.chkpt_dir)
    args.chkpt_file = resolve_checkpoint_file(args.chkpt_dir, args.chkpt_file)
    encoder_lookahead, decoder_lookahead = compute_lookahead_from_config(conf.model)
    sample_rate = int(conf.get("sampling_rate", 16000))
    audio_samples = int(round(sample_rate * args.duration))

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    audio = torch.randn(audio_samples, generator=generator)
    if not args.use_onnx:
        audio = audio.to(args.device)

    streaming_model = make_streaming_model(args, encoder_lookahead, decoder_lookahead)

    for _ in range(args.warmup):
        run_streaming_once(streaming_model, audio, include_flush=True)

    steady_times = []
    total_times = []
    for _ in range(args.repeats):
        steady_times.append(run_streaming_once(streaming_model, audio, include_flush=False))
        total_times.append(run_streaming_once(streaming_model, audio, include_flush=True))

    result = build_result(args, conf, streaming_model, audio_samples, steady_times, total_times)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)

    if not args.quiet:
        print(f"Model: {result['model']}/{result['seed']} ({result['backend']})")
        print(
            "Latency: "
            f"{result['latency_ms']:.2f}ms "
            f"(enc={result['encoder_lookahead']}, dec={result['decoder_lookahead']})"
        )
        print(
            "Streaming RTF: "
            f"{result['streaming_rtf']:.3f} "
            f"(total_with_flush={result['streaming_rtf_total']:.3f})"
        )


if __name__ == "__main__":
    main()
