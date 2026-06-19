"""Benchmark ONNX lookahead distribution for streaming inference.

Compares three lookahead distributions (balanced, encoder-concentrated,
decoder-concentrated) across different chunk sizes using a single checkpoint.

Two measurement levels:
  Level 1: Isolated session.run() — pure NN graph execution time
  Level 2: Full streaming pipeline — STFT + state I/O + feature buffer + iSTFT

The difference (Level 2 - Level 1) quantifies per-chunk buffer/STFT overhead.

ONNX StatefulExportableNNCore runs encoder+TS_BLOCK+decoder as a single graph.
session.run() processes the same T frames through all stages:
  - Model A (enc=7, dec=7):  T = cs + max(1,7) + 7  (buffered mode)
  - Model B (enc=14, dec=0): T = cs + max(1,14)      (immediate mode)
  - Model C (enc=0, dec=14): T = cs + max(1,0) + 14  (buffered mode)

Key overhead difference: Model B (immediate) has no feature buffer numpy
concat/slicing, while A/C (buffered) incur per-chunk buffer management.

Usage:
    python -m src.benchmark_lookahead_distribution \\
        --chkpt_dir path/to/checkpoint \\
        --chkpt_file best.th \\
        --total_lookahead 14 \\
        --chunk_sizes 1,4,16,64 \\
        --warmup 3 --repeats 5 \\
        --num_threads 1 \\
        --output_dir results/rtf/lookahead_distribution/
"""

import argparse
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class TimingResult:
    mean_ms: float
    std_ms: float
    per_repeat: List[float]


def generate_dummy_audio(duration_sec: float, sample_rate: int = 16000) -> torch.Tensor:
    """Generate dummy audio tensor for RTF measurement."""
    return torch.randn(int(duration_sec * sample_rate))


def get_lookahead_configs(total_la: int) -> List[Tuple[str, int, int]]:
    """Return (name, enc_la, dec_la) for three distribution strategies."""
    half = total_la // 2
    return [
        ("balanced", half, total_la - half),
        ("enc-concent", total_la, 0),
        ("dec-concent", 0, total_la),
    ]


def compute_onnx_input_t(model) -> int:
    """Compute time dimension of ONNX model input during steady-state.

    Immediate mode (dec_la=0): T = chunk_size + input_lookahead_frames
    Buffered mode  (dec_la>0): T = chunk_size + input_lookahead_frames + decoder_lookahead
    """
    if model.decoder_lookahead == 0:
        return model.chunk_size + model.input_lookahead_frames
    else:
        return model.chunk_size + model.input_lookahead_frames + model.decoder_lookahead


def create_onnx_model(
    chkpt_dir: str,
    chkpt_file: str,
    chunk_size: int,
    enc_la: int,
    dec_la: int,
    num_threads: int,
):
    """Create ONNXLaCoSENet with controlled thread count.

    Returns (model, onnx_path).
    """
    import tempfile

    import onnxruntime as ort

    from src.models.onnx_export.streaming_wrapper import ONNXLaCoSENet

    temp_file = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    onnx_path = temp_file.name
    temp_file.close()

    model = ONNXLaCoSENet.from_checkpoint(
        chkpt_dir=chkpt_dir,
        chkpt_file=chkpt_file,
        chunk_size=chunk_size,
        encoder_lookahead=enc_la,
        decoder_lookahead=dec_la,
        use_reshape_free=True,
        onnx_path=onnx_path,
        force_export=True,
        verbose=False,
    )

    # Replace session with thread-controlled one
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = num_threads
    sess_options.inter_op_num_threads = 1
    model.session = ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    return model, onnx_path


def measure_session_run(
    model,
    warmup: int,
    repeats: int,
    iterations: int,
) -> TimingResult:
    """Level 1: Measure isolated session.run() per-call time (ms).

    Uses dummy mag/pha matching steady-state T and realistic state I/O
    (states updated between consecutive calls).
    """
    session = model.session
    onnx_t = compute_onnx_input_t(model)
    freq_size = model.freq_size

    dummy_mag = np.random.randn(1, freq_size, onnx_t).astype(np.float32)
    dummy_pha = np.random.randn(1, freq_size, onnx_t).astype(np.float32)

    state_names = model._state_names
    model.reset_state()
    init_states = [s.copy() for s in model._states]

    # est_mask, est_pha, states... (atan2 mode) or est_mask, pha_real, pha_imag, states...
    state_offset = 2 if model.phase_output_mode == "atan2" else 3

    per_repeat_ms = []
    for r in range(warmup + repeats):
        states = [np.zeros_like(s) for s in init_states]

        t0 = time.perf_counter()
        for _ in range(iterations):
            inputs = {"mag": dummy_mag, "pha": dummy_pha}
            for name, state in zip(state_names, states):
                inputs[name] = state
            outputs = session.run(None, inputs)
            states = list(outputs[state_offset:])
        elapsed = time.perf_counter() - t0

        if r >= warmup:
            per_repeat_ms.append(elapsed / iterations * 1000)

    return TimingResult(
        mean_ms=float(np.mean(per_repeat_ms)),
        std_ms=float(np.std(per_repeat_ms)),
        per_repeat=per_repeat_ms,
    )


def measure_pipeline(
    model,
    audio: torch.Tensor,
    warmup: int,
    repeats: int,
    sample_rate: int = 16000,
) -> Tuple[TimingResult, float]:
    """Level 2: Measure steady-state pipeline per-chunk time (ms) and RTF.

    Two-phase measurement:
      Phase 1 (untimed for per-chunk): Pre-fill until first output.
        Handles both input buffer accumulation and decoder feature buffering.
      Phase 2 (timed): Steady-state processing where every chunk produces output.

    RTF is computed over the full run (both phases) for real-world accuracy.

    Returns (steady_per_chunk_timing, rtf).
    """
    osp = model.output_samples_per_chunk
    audio_len = len(audio)
    total_audio_sec = audio_len / sample_rate
    audio_iters = math.ceil(audio_len / osp)

    flush = model.samples_per_chunk * (model.total_lookahead + 2)
    padded = torch.cat([audio, torch.zeros(flush)])

    # Pre-compute chunk start positions
    chunk_positions = list(range(0, len(padded), osp))

    per_repeat_ms = []
    per_repeat_rtf = []

    for r in range(warmup + repeats):
        model.reset_state()

        t_total_start = time.perf_counter()

        # Phase 1: Pre-fill until first output (timed for RTF only)
        prefill_count = 0
        with torch.inference_mode():
            for pos in chunk_positions:
                chunk = padded[pos:pos + osp]
                if len(chunk) == 0:
                    break
                result = model.process_samples(chunk)
                prefill_count += 1
                if result is not None:
                    break

        t_steady_start = time.perf_counter()

        # Phase 2: Steady-state (timed for per-chunk)
        steady_count = 0
        with torch.inference_mode():
            for pos in chunk_positions[prefill_count:]:
                chunk = padded[pos:pos + osp]
                if len(chunk) == 0:
                    break
                model.process_samples(chunk)
                steady_count += 1
                if prefill_count + steady_count >= audio_iters:
                    break

        t_audio_end = time.perf_counter()

        if r >= warmup:
            # RTF: total wall time including startup
            per_repeat_rtf.append((t_audio_end - t_total_start) / total_audio_sec)

            # Steady-state per-chunk: excludes pre-fill
            if steady_count > 0:
                per_repeat_ms.append(
                    (t_audio_end - t_steady_start) / steady_count * 1000
                )

    timing = TimingResult(
        mean_ms=float(np.mean(per_repeat_ms)) if per_repeat_ms else 0.0,
        std_ms=float(np.std(per_repeat_ms)) if per_repeat_ms else 0.0,
        per_repeat=per_repeat_ms,
    )
    rtf = float(np.mean(per_repeat_rtf)) if per_repeat_rtf else 0.0
    return timing, rtf


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX lookahead distribution for streaming inference"
    )
    parser.add_argument("--chkpt_dir", type=str, required=True)
    parser.add_argument("--chkpt_file", type=str, default="best.th")
    parser.add_argument("--total_lookahead", type=int, default=14)
    parser.add_argument("--chunk_sizes", type=str, default="1,4,16,64")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Duration of dummy audio in seconds")
    parser.add_argument("--iterations", type=int, default=200,
                        help="Iterations per repeat for Level 1 session.run() timing")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    # Thread control (set before any ONNX/torch computation)
    torch.set_num_threads(args.num_threads)
    os.environ["OMP_NUM_THREADS"] = str(args.num_threads)

    import onnxruntime as ort

    chunk_sizes = [int(x) for x in args.chunk_sizes.split(",")]
    configs = get_lookahead_configs(args.total_lookahead)
    audio = generate_dummy_audio(args.duration)
    hardware = platform.processor() or platform.machine()

    print(f"\n{'=' * 70}")
    print(f"  Lookahead Distribution Benchmark (ONNX)")
    print(f"{'=' * 70}")
    print(f"Checkpoint: {args.chkpt_dir}")
    print(f"Hardware: {hardware} ({args.num_threads} thread{'s' if args.num_threads > 1 else ''})")
    print(f"ONNX Runtime: {ort.__version__}")
    print(f"total_lookahead={args.total_lookahead}, duration={args.duration}s")
    print(f"warmup={args.warmup}, repeats={args.repeats}, L1_iters={args.iterations}")
    print()

    all_results = {}

    for cs in chunk_sizes:
        print(f"--- chunk_size = {cs} ---\n")
        cs_results = []

        for name, enc_la, dec_la in configs:
            print(f"  [{name}] enc={enc_la}, dec={dec_la} ... ", end="", flush=True)

            model, onnx_path = create_onnx_model(
                args.chkpt_dir, args.chkpt_file, cs, enc_la, dec_la, args.num_threads,
            )
            onnx_t = compute_onnx_input_t(model)

            # Level 1: isolated session.run()
            l1 = measure_session_run(model, args.warmup, args.repeats, args.iterations)

            # Level 2: full streaming pipeline
            l2, rtf = measure_pipeline(model, audio, args.warmup, args.repeats)

            overhead = l2.mean_ms - l1.mean_ms

            cs_results.append({
                "name": name,
                "enc_la": enc_la,
                "dec_la": dec_la,
                "onnx_t": onnx_t,
                "session_run": asdict(l1),
                "pipeline_per_chunk": asdict(l2),
                "rtf_steady": rtf,
                "overhead_ms": overhead,
            })

            # Cleanup temp ONNX file
            try:
                os.unlink(onnx_path)
            except OSError:
                pass

            print(f"done (L1={l1.mean_ms:.3f}ms, L2={l2.mean_ms:.3f}ms, RTF={rtf:.4f})")

        # Print comparison table
        print()
        print(
            f"{'Config':<14}| {'enc':>3} | {'dec':>3} | {'ONNX T':>6} | "
            f"{'session.run (ms)':>18} | {'pipeline (ms)':>16} | "
            f"{'RTF':>8} | {'overhead (ms)':>13}"
        )
        print("-" * 110)
        for r in cs_results:
            sr = r["session_run"]
            pp = r["pipeline_per_chunk"]
            print(
                f"{r['name']:<14}| {r['enc_la']:>3} | {r['dec_la']:>3} | "
                f"{r['onnx_t']:>6} | "
                f"{sr['mean_ms']:>7.3f} +/- {sr['std_ms']:<7.3f} | "
                f"{pp['mean_ms']:>6.3f} +/- {pp['std_ms']:<5.3f} | "
                f"{r['rtf_steady']:>8.4f} | "
                f"{r['overhead_ms']:>13.3f}"
            )
        print()

        all_results[str(cs)] = cs_results

    # Save JSON
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output = {
            "metadata": {
                "chkpt_dir": args.chkpt_dir,
                "chkpt_file": args.chkpt_file,
                "total_lookahead": args.total_lookahead,
                "hardware": hardware,
                "num_threads": args.num_threads,
                "onnxruntime_version": ort.__version__,
                "duration_sec": args.duration,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "l1_iterations": args.iterations,
            },
            "results": all_results,
        }

        json_path = output_dir / "benchmark_results.json"
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to: {json_path}")


if __name__ == "__main__":
    main()
