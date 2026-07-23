"""
Unified batch evaluation script for LaCoSENet experiments.

Three subcommands cover all evaluation modes:

  fullseq     — Full-sequence (non-streaming) evaluation (baseline).
  streaming   — Streaming evaluation with a single chunk_size.
  chunksweep  — Chunk-size sweep across multiple chunk_sizes.

Usage:
    python -m src.batch_evaluate fullseq --exp_pattern "*s2039" --device cuda
    python -m src.batch_evaluate streaming --exp_pattern "*s2039" --chunk_size 1 --device cuda
    python -m src.batch_evaluate chunksweep --experiments M1_12.5ms_s2039 --chunk_sizes 1 64 --device cuda

    # Split across 2 GPUs (fullseq / streaming):
    python -m src.batch_evaluate fullseq --exp_pattern "*s2039" --device cuda:0 --split 0/2 &
    python -m src.batch_evaluate fullseq --exp_pattern "*s2039" --device cuda:1 --split 1/2 &
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import logging
import torch
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets import load_dataset

from src.compute_metrics import compute_metrics
from src.data import VoiceBankDataset
from src.utils import bold, compute_lookahead_frames

METRICS_LIST = ["pesq", "stoi", "csig", "cbak", "covl", "segSNR"]


# ============================================================================
# Section 2: Shared utility functions
# ============================================================================

def find_best_checkpoint(exp_dir: Path) -> dict:
    """Load states.th and return best model info (step, valid_pesq_value)."""
    states_path = exp_dir / "states.th"
    if not states_path.exists():
        raise FileNotFoundError(f"states.th not found in {exp_dir}")

    states = torch.load(states_path, map_location="cpu", weights_only=False)
    best_models = states.get("best_models", [])
    if not best_models:
        raise ValueError(f"No best_models found in {exp_dir}/states.th")

    best = best_models[0]
    return {
        "step": best["steps"],
        "valid_pesq": best["valid_pesq_value"],
    }


def setup_logging(output_dir):
    """Configure logging to stderr only."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def load_test_dataset():
    """Load VoiceBank-DEMAND 16kHz test split from HuggingFace."""
    return load_dataset("JacobLinCool/VoiceBank-DEMAND-16k", split="test")


def create_data_loader(hf_dataset, conf, num_workers):
    """Create a DataLoader for evaluation (batch_size=1)."""
    ev_dataset = VoiceBankDataset(
        hf_dataset, segment=None, with_id=True, with_text=True,
    )
    return DataLoader(
        dataset=ev_dataset,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=True,
    )


def is_latency_expert_config(conf) -> bool:
    return conf.model.get("type", "backbone") == "latency_expert_backbone"


def get_config_latency_ids(conf, requested=None):
    if not is_latency_expert_config(conf):
        return [None]

    configured = list(conf.model.latency_expert.latency_ids)
    if requested:
        unknown = sorted(set(requested) - set(configured))
        if unknown:
            raise ValueError(f"Unknown latency ids {unknown}; configured ids are {configured}")
        return list(requested)

    return configured


def result_name_for_latency(exp_name, latency_id):
    return f"{exp_name}_{latency_id}" if latency_id else exp_name


def forward_model(model, noisy_com, latency_id=None):
    if latency_id is not None:
        return model(noisy_com, latency_id=latency_id)
    return model(noisy_com)


def find_experiments(base_dir, exp_pattern=None, exp_names=None, split=None):
    """Find experiment directories by glob pattern or explicit names.

    Returns list of (exp_name, exp_dir) tuples.
    """
    base_dir = Path(base_dir)

    def is_experiment_dir(path: Path) -> bool:
        return path.is_dir() and (path / ".hydra" / "config.yaml").exists()

    def display_name(path: Path) -> str:
        rel = path.relative_to(base_dir)
        if len(rel.parts) == 2:
            return f"{rel.parts[0]}_{rel.parts[1]}"
        return rel.as_posix().replace("/", "_")

    def explicit_candidates(name: str):
        yield base_dir / name

        # Current layout is results/experiments/<model>/<seed>, while many
        # sweep manifests use the compact <model>_<seed> form.
        if "/" not in name:
            model_name, sep, seed_name = name.rpartition("_")
            if sep and seed_name.startswith("s") and seed_name[1:].isdigit():
                yield base_dir / model_name / seed_name

    if exp_names is not None:
        # Explicit names (chunksweep mode)
        results = []
        for name in exp_names:
            exp_dir = next((d for d in explicit_candidates(name) if is_experiment_dir(d)), None)
            if exp_dir is None:
                logging.getLogger(__name__).error(f"Experiment directory not found: {base_dir / name}")
                continue
            results.append((display_name(exp_dir), exp_dir))
        return results

    # Glob pattern (fullseq / streaming mode)
    glob_patterns = [exp_pattern]
    if exp_pattern and "/" not in exp_pattern:
        glob_patterns.append(f"*/{exp_pattern}")
        if exp_pattern.startswith("*") and exp_pattern.lstrip("*"):
            glob_patterns.append(f"*/{exp_pattern.lstrip('*')}")

    seen = set()
    exp_dirs = []
    for pattern in glob_patterns:
        for exp_dir in sorted(base_dir.glob(pattern)):
            if not is_experiment_dir(exp_dir):
                continue
            key = exp_dir.resolve()
            if key in seen:
                continue
            seen.add(key)
            exp_dirs.append(exp_dir)

    if not exp_dirs:
        return []

    # Apply split if specified
    if split:
        split_idx, split_total = map(int, split.split("/"))
        n = len(exp_dirs)
        chunk_size = n // split_total
        remainder = n % split_total
        start = split_idx * chunk_size + min(split_idx, remainder)
        end = start + chunk_size + (1 if split_idx < remainder else 0)
        exp_dirs = exp_dirs[start:end]
        logging.getLogger(__name__).info(
            f"Split {split_idx}/{split_total}: processing experiments [{start}:{end}]"
        )

    return [(display_name(d), d) for d in exp_dirs]


def compute_streaming_lookahead(
    conf,
    chunk_size: int,
    latency_id=None,
    sample_rate=None,
):
    """Compute streaming lookahead info from model config.

    Returns dict with encoder_lookahead, decoder_lookahead, total_lookahead,
    latency_ms, hop_size, sample_rate, etc.
    """
    if is_latency_expert_config(conf):
        latency_ids = get_config_latency_ids(conf)
        if latency_id is None:
            latency_id = conf.model.latency_expert.get("default_latency_id", latency_ids[0])
        latency_config = conf.model.latency_expert
        future_frames = latency_config.get("future_frames")
        if future_frames is not None:
            from src.v2_contract import decoder_padding_ratios_for_future_frames

            enc_ratio = [1.0, 0.0]
            dec_ratio = decoder_padding_ratios_for_future_frames(
                dict(future_frames), depth=conf.model.get("dense_depth", 4)
            )[latency_id]
        else:
            enc_ratio = list(latency_config.encoder_padding_ratios[latency_id])
            dec_ratio = list(latency_config.decoder_padding_ratios[latency_id])
    else:
        enc_ratio = list(conf.model.encoder_padding_ratio)
        dec_ratio = list(conf.model.decoder_padding_ratio)
    depth = conf.model.get("dense_depth", 4)
    enc_la = compute_lookahead_frames(enc_ratio, depth)
    if is_latency_expert_config(conf) and conf.model.latency_expert.get("future_frames") is not None:
        dec_la = int(conf.model.latency_expert.future_frames[latency_id])
    else:
        dec_la = compute_lookahead_frames(dec_ratio, depth)

    # Match LaCoSENet streaming wrapper behavior:
    # - input_lookahead_frames = encoder lookahead (no min-2-frame hack needed)
    # - STFT future buffering (center=False) eliminates reflect padding requirement
    input_la = int(enc_la)
    total_la = input_la + dec_la

    sample_rate = int(sample_rate or conf.get("sampling_rate", 16000))
    sfi_config = conf.model.get("sfi", {})
    if sfi_config.get("enabled", False):
        from src.v2_contract import sfi_profile_from_config

        profile = sfi_profile_from_config(sfi_config, sample_rate)
        hop_size = profile.hop_len
        win_size = profile.win_len
    else:
        hop_size = conf.model.get("hop_len", 100)
        win_size = conf.model.get("win_len", 400)
    stft_center_delay_samples = win_size // 2  # STFT future buffering delay
    latency_ms = (total_la * hop_size + stft_center_delay_samples) / sample_rate * 1000

    return {
        "enc_ratio": enc_ratio,
        "dec_ratio": dec_ratio,
        "latency_id": latency_id,
        "encoder_lookahead": enc_la,
        "decoder_lookahead": dec_la,
        "input_lookahead": input_la,
        "total_lookahead": total_la,
        "latency_ms": latency_ms,
        "hop_size": hop_size,
        "win_size": win_size,
        "stft_center_delay_samples": stft_center_delay_samples,
        "sample_rate": sample_rate,
    }


# ============================================================================
# Section 3: Evaluation functions (per-mode)
# ============================================================================

def evaluate_fullseq_single(model, data_loader, stft_args, device, logger, latency_id=None):
    """Run full-sequence evaluation, return dict of metrics."""
    from src.stft import mag_pha_stft, mag_pha_istft

    model.eval()
    results = []

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            noisy, clean, _, _ = data

            noisy_com = mag_pha_stft(noisy, **stft_args)[2].to(device)
            clean_mag_hat, clean_pha_hat, _ = forward_model(
                model, noisy_com, latency_id=latency_id
            )
            clean_hat = mag_pha_istft(clean_mag_hat, clean_pha_hat, **stft_args)

            clean_np = clean.squeeze().detach().cpu().numpy()
            clean_hat_np = clean_hat.squeeze().detach().cpu().numpy()

            # Align lengths
            if len(clean_np) != len(clean_hat_np):
                length = min(len(clean_np), len(clean_hat_np))
                clean_np = clean_np[:length]
                clean_hat_np = clean_hat_np[:length]

            results.append(compute_metrics(clean_np, clean_hat_np))

            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{len(data_loader)} utterances")

    pesq, csig, cbak, covl, segSNR, stoi = np.mean(results, axis=0)
    return {
        "pesq": float(pesq),
        "stoi": float(stoi),
        "csig": float(csig),
        "cbak": float(cbak),
        "covl": float(covl),
        "segSNR": float(segSNR),
    }


def evaluate_streaming_single(
    streaming_model,
    data_loader,
    device,
    logger,
    shift_samples=0,
    tail_trim_samples=0,
):
    """Run streaming evaluation, return dict of metrics.

    Args:
        shift_samples: Number of leading samples to skip from enhanced signal
            before metric computation (OLA center shift compensation).
        tail_trim_samples: Number of trailing samples to remove from BOTH clean
            and enhanced signals before metric computation. Useful to exclude
            the tail boundary region affected by lookahead flush.
    """
    results = []

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            noisy, clean, _, _ = data

            enhanced = streaming_model.process_audio_fast(noisy.squeeze(0).to(device))

            clean_np = clean.squeeze().numpy()
            enhanced_np = enhanced.cpu().numpy()

            if shift_samples > 0:
                enhanced_np = enhanced_np[shift_samples:]

            # Align lengths
            length = min(len(clean_np), len(enhanced_np))
            clean_np = clean_np[:length]
            enhanced_np = enhanced_np[:length]

            if tail_trim_samples and tail_trim_samples > 0 and length > tail_trim_samples:
                clean_np = clean_np[:-tail_trim_samples]
                enhanced_np = enhanced_np[:-tail_trim_samples]

            results.append(compute_metrics(clean_np, enhanced_np))

            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{len(data_loader)} utterances")

    pesq, csig, cbak, covl, segSNR, stoi = np.mean(results, axis=0)
    return {
        "pesq": float(pesq),
        "stoi": float(stoi),
        "csig": float(csig),
        "cbak": float(cbak),
        "covl": float(covl),
        "segSNR": float(segSNR),
    }


# ============================================================================
# Section 4: Comparison / summary builders (per-mode)
# ============================================================================

def generate_streaming_comparison(streaming_results, fullseq_path, logger):
    """Compare streaming vs full-seq results, return comparison dict."""
    if not fullseq_path.exists():
        logger.info(f"Full-seq results not found at {fullseq_path}, skipping comparison.")
        return {}

    with open(fullseq_path, "r") as f:
        fullseq_results = json.load(f)

    comparison = {}
    for exp_name, s_result in streaming_results.items():
        if exp_name not in fullseq_results:
            continue

        f_metrics = fullseq_results[exp_name]["test_metrics"]
        s_metrics = s_result["test_metrics"]

        entry = {
            "fullseq_pesq": f_metrics["pesq"],
            "streaming_pesq": s_metrics["pesq"],
        }
        for metric in METRICS_LIST:
            entry[f"delta_{metric}"] = round(s_metrics[metric] - f_metrics[metric], 6)

        comparison[exp_name] = entry

    # Summary
    if comparison:
        delta_pesqs = [abs(v["delta_pesq"]) for v in comparison.values()]
        comparison["_summary"] = {
            "max_abs_delta_pesq": round(max(delta_pesqs), 6),
            "mean_abs_delta_pesq": round(float(np.mean(delta_pesqs)), 6),
        }

    return comparison


def build_chunksweep_comparison(all_results, fullseq_path, logger):
    """Build comparison: each (exp, chunk) vs full-sequence baseline.

    Returns (comparison_dict, summary_dict) or (None, None) if no baseline.
    """
    if not fullseq_path.exists():
        logger.info(f"Full-seq results not found at {fullseq_path}, skipping comparison.")
        return None, None

    with open(fullseq_path, "r") as f:
        fullseq_results = json.load(f)

    comparison = {}

    for exp_name, exp_data in all_results.items():
        if exp_name not in fullseq_results:
            continue

        f_metrics = fullseq_results[exp_name]["test_metrics"]
        chunk_results = exp_data["chunk_results"]

        entry = {"fullseq_pesq": f_metrics["pesq"], "chunk_deltas": {}}
        max_abs_deltas = {m: 0.0 for m in METRICS_LIST}

        for cs_str, c_metrics in chunk_results.items():
            deltas = {}
            for m in METRICS_LIST:
                delta = round(c_metrics[m] - f_metrics[m], 6)
                deltas[f"delta_{m}"] = delta
                max_abs_deltas[m] = max(max_abs_deltas[m], abs(delta))
            entry["chunk_deltas"][cs_str] = deltas

        entry["max_abs_delta_pesq"] = round(max_abs_deltas["pesq"], 6)
        comparison[exp_name] = entry

    # Global summary
    summary = None
    if comparison:
        all_max_delta_pesq = [v["max_abs_delta_pesq"] for v in comparison.values()]
        global_max = max(all_max_delta_pesq)
        summary = {
            "global_max_abs_delta_pesq": round(global_max, 6),
        }

    return comparison, summary


# ============================================================================
# Section 5: Mode handlers (main loops)
# ============================================================================

def run_fullseq(args):
    """Handler for 'fullseq' subcommand."""
    from src.stft import mag_pha_stft, mag_pha_istft  # noqa: F401
    from src.utils import load_model, load_checkpoint, get_stft_args_from_config

    output_dir = Path(args.output_dir)
    seed_tag = args.exp_pattern.replace("*", "").strip("_")
    split_tag = f"_part{args.split.split('/')[0]}" if args.split else ""

    logger = setup_logging(output_dir)

    # Find experiments
    base_dir = Path(args.exp_dir)
    experiments = find_experiments(base_dir, exp_pattern=args.exp_pattern, split=args.split)
    if not experiments:
        logger.error(f"No experiment directories found matching {args.exp_pattern} in {base_dir}")
        return

    logger.info(f"Found {len(experiments)} experiments to evaluate")

    # Load dataset once
    logger.info("Loading VoiceBank-DEMAND test set...")
    hf_dataset = load_test_dataset()
    logger.info(f"Test set loaded: {len(hf_dataset)} utterances")

    # Evaluate each experiment
    all_results = {}

    for exp_name, exp_dir in experiments:
        logger.info(bold(f"\n{'='*60}"))
        logger.info(bold(f"Evaluating: {exp_name}"))
        logger.info(bold(f"{'='*60}"))

        try:
            best_info = find_best_checkpoint(exp_dir)
            best_step = best_info["step"]
            chkpt_file = f"model_{best_step}.th"
            logger.info(f"Best model: step={best_step}, valid_pesq={best_info['valid_pesq']:.4f}")

            chkpt_path = exp_dir / chkpt_file
            if not chkpt_path.exists():
                logger.error(f"Checkpoint file not found: {chkpt_path}")
                continue

            config_path = exp_dir / ".hydra" / "config.yaml"
            if not config_path.exists():
                logger.error(f"Config file not found: {config_path}")
                continue

            conf = OmegaConf.load(config_path)
            ev_loader = create_data_loader(hf_dataset, conf, args.num_workers)

            stft_args = get_stft_args_from_config(conf.model)
            model = load_model(conf.model, args.device)
            model = load_checkpoint(model, str(exp_dir), chkpt_file, args.device)
            latency_ids = get_config_latency_ids(conf, requested=args.latency_ids)

            for latency_id in latency_ids:
                display_latency = latency_id or "default"
                logger.info(f"  Fullseq latency_id: {display_latency}")
                metrics = evaluate_fullseq_single(
                    model, ev_loader, stft_args, args.device, logger,
                    latency_id=latency_id,
                )

                result_name = result_name_for_latency(exp_name, latency_id)
                all_results[result_name] = {
                    "best_step": best_step,
                    "valid_pesq": best_info["valid_pesq"],
                    "latency_id": latency_id,
                    "test_metrics": metrics,
                }

                logger.info(
                    bold(f"Results [{display_latency}]: PESQ={metrics['pesq']:.4f}, STOI={metrics['stoi']:.4f}, "
                         f"CSIG={metrics['csig']:.4f}, CBAK={metrics['cbak']:.4f}, "
                         f"COVL={metrics['covl']:.4f}, segSNR={metrics['segSNR']:.4f}")
                )

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed to evaluate {exp_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        logger.error("No experiments were successfully evaluated.")
        return

    # JSON output
    json_path = output_dir / f"eval_results_{seed_tag}{split_tag}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {json_path}")

    # Print summary table
    logger.info(bold(f"\n{'='*80}"))
    logger.info(bold("SUMMARY"))
    logger.info(bold(f"{'='*80}"))
    header = f"{'Experiment':<25} {'LatID':>6} {'Step':>7} {'PESQ':>7} {'STOI':>7} {'CSIG':>7} {'CBAK':>7} {'COVL':>7}"
    logger.info(header)
    logger.info("-" * 80)
    for exp_name, result in all_results.items():
        m = result["test_metrics"]
        logger.info(
            f"{exp_name:<25} {str(result.get('latency_id') or '-'):>6} {result['best_step']:>7} "
            f"{m['pesq']:>7.4f} {m['stoi']:>7.4f} {m['csig']:>7.4f} "
            f"{m['cbak']:>7.4f} {m['covl']:>7.4f}"
        )


def run_streaming(args):
    """Handler for 'streaming' subcommand."""
    from src.models.streaming.lacosenet import LaCoSENet

    output_dir = Path(args.output_dir)
    seed_tag = args.exp_pattern.replace("*", "").strip("_")
    split_tag = f"_part{args.split.split('/')[0]}" if args.split else ""
    align_ola = getattr(args, "align_ola", False)

    logger = setup_logging(output_dir)

    if align_ola:
        logger.info("OLA alignment enabled: will compensate win_size//2 shift")

    # Find experiments
    base_dir = Path(args.exp_dir)
    experiments = find_experiments(base_dir, exp_pattern=args.exp_pattern, split=args.split)
    if not experiments:
        logger.error(f"No experiment directories found matching {args.exp_pattern} in {base_dir}")
        return

    logger.info(f"Found {len(experiments)} experiments to evaluate")

    # Load dataset once
    logger.info("Loading VoiceBank-DEMAND test set...")
    hf_dataset = load_test_dataset()
    logger.info(f"Test set loaded: {len(hf_dataset)} utterances")

    # Prepare output path (for incremental saves)
    mode_tag = "streaming_aligned" if align_ola else "streaming"
    json_path = output_dir / f"eval_results_{mode_tag}_{seed_tag}{split_tag}.json"
    fullseq_json_path = output_dir / f"eval_results_{seed_tag}.json"

    # Evaluate each experiment
    all_results = {}

    for exp_name, exp_dir in experiments:
        logger.info(bold(f"\n{'='*60}"))
        logger.info(bold(f"Evaluating: {exp_name}"))
        logger.info(bold(f"{'='*60}"))

        try:
            best_info = find_best_checkpoint(exp_dir)
            best_step = best_info["step"]
            chkpt_file = f"model_{best_step}.th"
            logger.info(f"Best model: step={best_step}, valid_pesq={best_info['valid_pesq']:.4f}")

            chkpt_path = exp_dir / chkpt_file
            if not chkpt_path.exists():
                logger.error(f"Checkpoint file not found: {chkpt_path}")
                continue

            config_path = exp_dir / ".hydra" / "config.yaml"
            if not config_path.exists():
                logger.error(f"Config file not found: {config_path}")
                continue

            conf = OmegaConf.load(config_path)
            latency_ids = get_config_latency_ids(conf, requested=args.latency_ids)

            ev_loader = create_data_loader(hf_dataset, conf, args.num_workers)

            for latency_id in latency_ids:
                display_latency = latency_id or "default"
                la = compute_streaming_lookahead(
                    conf, chunk_size=args.chunk_size, latency_id=latency_id
                )
                logger.info(f"  Latency id: {display_latency}")
                logger.info(f"  Padding ratio: enc={la['enc_ratio']}, dec={la['dec_ratio']}")
                logger.info(f"  Lookahead: enc={la['encoder_lookahead']}, dec={la['decoder_lookahead']}, total={la['total_lookahead']}")
                logger.info(f"  Latency: {la['latency_ms']:.2f}ms, chunk_size: {args.chunk_size}")

                streaming = LaCoSENet.from_checkpoint(
                    chkpt_dir=str(exp_dir),
                    chkpt_file=chkpt_file,
                    chunk_size=args.chunk_size,
                    encoder_lookahead=la["encoder_lookahead"],
                    decoder_lookahead=la["decoder_lookahead"],
                    latency_id=latency_id,
                    device=args.device,
                    verbose=False,
                )

                shift_samples = la["stft_center_delay_samples"] if align_ola else 0
                metrics = evaluate_streaming_single(
                    streaming, ev_loader, args.device, logger,
                    shift_samples=shift_samples,
                )

                result_name = result_name_for_latency(exp_name, latency_id)
                all_results[result_name] = {
                    "best_step": best_step,
                    "valid_pesq": best_info["valid_pesq"],
                    "latency_id": latency_id,
                    "chunk_size": args.chunk_size,
                    "encoder_lookahead": la["encoder_lookahead"],
                    "decoder_lookahead": la["decoder_lookahead"],
                    "latency_ms": la["latency_ms"],
                    "align_ola": align_ola,
                    "shift_samples": shift_samples,
                    "test_metrics": metrics,
                }

                logger.info(
                    bold(f"Results [{display_latency}]: PESQ={metrics['pesq']:.4f}, STOI={metrics['stoi']:.4f}, "
                         f"CSIG={metrics['csig']:.4f}, CBAK={metrics['cbak']:.4f}, "
                         f"COVL={metrics['covl']:.4f}, segSNR={metrics['segSNR']:.4f}")
                )

                # Incremental save — write after each latency id
                output_json = dict(all_results)
                comparison = generate_streaming_comparison(all_results, fullseq_json_path, logger)
                if comparison:
                    output_json["_comparison"] = comparison
                with open(json_path, "w") as f:
                    json.dump(output_json, f, indent=2)
                logger.info(f"  [{len(all_results)} results] saved → {json_path}")

                del streaming
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed to evaluate {exp_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        logger.error("No experiments were successfully evaluated.")
        return

    logger.info(f"\nFinal results saved to {json_path}")

    # Print summary table
    logger.info(bold(f"\n{'='*80}"))
    logger.info(bold("SUMMARY"))
    logger.info(bold(f"{'='*80}"))
    header = f"{'Experiment':<25} {'LatID':>6} {'Step':>7} {'Lat(ms)':>8} {'PESQ':>7} {'STOI':>7} {'CSIG':>7} {'CBAK':>7} {'COVL':>7}"
    logger.info(header)
    logger.info("-" * 80)
    for exp_name, result in all_results.items():
        m = result["test_metrics"]
        logger.info(
            f"{exp_name:<25} {str(result.get('latency_id') or '-'):>6} {result['best_step']:>7} "
            f"{result['latency_ms']:>8.2f} "
            f"{m['pesq']:>7.4f} {m['stoi']:>7.4f} {m['csig']:>7.4f} "
            f"{m['cbak']:>7.4f} {m['covl']:>7.4f}"
        )


def run_chunksweep(args):
    """Handler for 'chunksweep' subcommand."""
    from src.models.streaming.lacosenet import LaCoSENet
    from src.models.streaming.utils import prepare_streaming_model

    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir)

    logger.info(f"Experiments: {args.experiments}")
    logger.info(f"Chunk sizes: {args.chunk_sizes}")
    logger.info(f"Device: {args.device}")

    # Find experiments
    base_dir = Path(args.exp_dir)
    experiments = find_experiments(base_dir, exp_names=args.experiments)
    if not experiments:
        logger.error("No valid experiment directories found.")
        return

    # Load dataset once
    logger.info("Loading VoiceBank-DEMAND test set...")
    hf_dataset = load_test_dataset()
    logger.info(f"Test set loaded: {len(hf_dataset)} utterances")

    # Main loop: experiments (outer) x chunk_sizes (inner)
    all_results = {}

    for exp_name, exp_dir in experiments:
        logger.info(bold(f"\n{'='*60}"))
        logger.info(bold(f"Evaluating: {exp_name}"))
        logger.info(bold(f"{'='*60}"))

        try:
            best_info = find_best_checkpoint(exp_dir)
            best_step = best_info["step"]
            chkpt_file = f"model_{best_step}.th"
            logger.info(f"  Model loaded (step={best_step}, valid_pesq={best_info['valid_pesq']:.4f})")

            chkpt_path = exp_dir / chkpt_file
            if not chkpt_path.exists():
                logger.error(f"  Checkpoint file not found: {chkpt_path}")
                continue

            config_path = exp_dir / ".hydra" / "config.yaml"
            if not config_path.exists():
                logger.error(f"  Config file not found: {config_path}")
                continue

            conf = OmegaConf.load(config_path)

            sample_rate = conf.get("sampling_rate", 16000)
            ev_loader = create_data_loader(hf_dataset, conf, args.num_workers)

            # Load model ONCE per experiment
            model, metadata = prepare_streaming_model(
                chkpt_dir=str(exp_dir),
                chkpt_file=chkpt_file,
                use_stateful_conv=True,
                device=args.device,
                verbose=False,
            )
            model_args = metadata["model_args"]
            latency_id = args.latency_id
            if is_latency_expert_config(conf):
                valid_latency_ids = get_config_latency_ids(conf)
                if latency_id is None:
                    latency_id = conf.model.latency_expert.get("default_latency_id", valid_latency_ids[0])
                if latency_id not in valid_latency_ids:
                    raise ValueError(
                        f"Unknown latency_id '{latency_id}'; configured ids are {valid_latency_ids}"
                    )
                model.set_latency_id(latency_id)

            def get_model_arg(*names, default):
                for name in names:
                    if hasattr(model_args, "get"):
                        value = model_args.get(name, None)
                    else:
                        value = getattr(model_args, name, None)
                    if value is not None:
                        return value
                return default

            hop_size = get_model_arg("hop_len", "hop_size", default=100)
            n_fft = get_model_arg("fft_len", "n_fft", default=400)
            win_size = get_model_arg("win_len", "win_size", default=400)
            compress_factor = get_model_arg("compress_factor", default=0.3)
            freq_size = n_fft // 2 + 1

            # Sweep chunk_sizes
            chunk_results = {}

            for cs in args.chunk_sizes:
                la = compute_streaming_lookahead(conf, chunk_size=cs, latency_id=latency_id)
                logger.info(f"  [cs={cs}] Lookahead: enc={la['encoder_lookahead']}, dec={la['decoder_lookahead']}, "
                            f"total={la['total_lookahead']}, latency={la['latency_ms']:.2f}ms")

                lacosenet = LaCoSENet(
                    model=model,
                    chunk_size=cs,
                    encoder_lookahead=la["encoder_lookahead"],
                    decoder_lookahead=la["decoder_lookahead"],
                    hop_size=hop_size,
                    n_fft=n_fft,
                    win_size=win_size,
                    compress_factor=compress_factor,
                    sample_rate=sample_rate,
                    freq_size=freq_size,
                )

                shift_samples = la["stft_center_delay_samples"] if getattr(args, "align_ola", False) else 0
                metrics = evaluate_streaming_single(
                    lacosenet, ev_loader, args.device, logger,
                    shift_samples=shift_samples,
                )

                logger.info(
                    f"  [chunk_size={cs:>3}] "
                    f"PESQ={metrics['pesq']:.4f} STOI={metrics['stoi']:.4f} "
                    f"CSIG={metrics['csig']:.4f} CBAK={metrics['cbak']:.4f} "
                    f"COVL={metrics['covl']:.4f} segSNR={metrics['segSNR']:.4f}"
                )

                chunk_results[str(cs)] = metrics
                del lacosenet

            result_name = result_name_for_latency(exp_name, latency_id)
            all_results[result_name] = {
                "best_step": best_step,
                "latency_id": latency_id,
                "encoder_lookahead": la["encoder_lookahead"],
                "decoder_lookahead": la["decoder_lookahead"],
                "latency_ms": la["latency_ms"],
                "chunk_results": chunk_results,
            }

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed to evaluate {exp_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        logger.error("No experiments were successfully evaluated.")
        return

    # Comparison vs full-sequence
    first_exp = next(iter(all_results), args.experiments[0])
    seed_tag = ""
    for part in first_exp.split("_"):
        if part.startswith("s") and part[1:].isdigit():
            seed_tag = part
            break

    fullseq_path = output_dir / f"eval_results_{seed_tag}.json" if seed_tag else None
    comparison, summary = None, None
    if fullseq_path:
        comparison, summary = build_chunksweep_comparison(all_results, fullseq_path, logger)

    # Build output JSON
    output_json = dict(all_results)
    if comparison is not None:
        output_json["_comparison"] = comparison
    if summary is not None:
        output_json["_summary"] = summary

    json_path = output_dir / "eval_results_chunksweep.json"
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)
    logger.info(f"Results saved to {json_path}")

    # Print summary table
    logger.info(bold(f"\n{'='*90}"))
    logger.info(bold("SUMMARY"))
    logger.info(bold(f"{'='*90}"))
    header = (
        f"{'Experiment':<25} {'Lat(ms)':>8} {'CS':>5} "
        f"{'PESQ':>7} {'STOI':>7} {'CSIG':>7} {'CBAK':>7} {'COVL':>7} "
        f"{'δPESQ':>9}"
    )
    logger.info(header)
    logger.info("-" * 90)

    for exp_name, exp_data in all_results.items():
        for cs_str, metrics in exp_data["chunk_results"].items():
            delta_str = ""
            if comparison and exp_name in comparison:
                chunk_deltas = comparison[exp_name]["chunk_deltas"]
                if cs_str in chunk_deltas:
                    d = chunk_deltas[cs_str]["delta_pesq"]
                    delta_str = f"{d:+.6f}"
            logger.info(
                f"{exp_name:<25} {exp_data['latency_ms']:>8.2f} {cs_str:>5} "
                f"{metrics['pesq']:>7.4f} {metrics['stoi']:>7.4f} "
                f"{metrics['csig']:>7.4f} {metrics['cbak']:>7.4f} "
                f"{metrics['covl']:>7.4f} {delta_str:>9}"
            )

    # Cross-chunk consistency
    logger.info("")
    logger.info(bold("Cross-chunk consistency (within each experiment):"))
    for exp_name, exp_data in all_results.items():
        pesq_vals = [m["pesq"] for m in exp_data["chunk_results"].values()]
        max_spread = max(pesq_vals) - min(pesq_vals)
        logger.info(f"  {exp_name}: max PESQ spread = {max_spread:.6f}")

    if summary:
        logger.info("")
        logger.info(bold(f"Global max |δPESQ| vs full-seq: {summary['global_max_abs_delta_pesq']:.6f}"))


# ============================================================================
# Section 6: Argument parser + dispatch
# ============================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Unified batch evaluation for LaCoSENet experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s fullseq --exp_pattern "*s2039" --device cuda
  %(prog)s streaming --exp_pattern "*s2039" --chunk_size 1 --device cuda
  %(prog)s chunksweep --experiments M1_12.5ms_s2039 --chunk_sizes 1 64 --device cuda
""",
    )

    # Parent parser with common arguments
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--exp_dir", type=str, default="results/experiments",
                        help="Base directory containing experiment directories.")
    parent.add_argument("--output_dir", type=str, default="results/evaluation",
                        help="Directory to save evaluation results.")
    parent.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parent.add_argument("--num_workers", type=int, default=5)

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # --- fullseq ---
    p_full = subparsers.add_parser(
        "fullseq", parents=[parent],
        help="Full-sequence (non-streaming) evaluation",
    )
    p_full.add_argument("--exp_pattern", type=str, default="*s2039",
                        help="Glob pattern to match experiment directories.")
    p_full.add_argument("--split", type=str, default=None,
                        help="Split experiments: 'INDEX/TOTAL' (e.g., '0/2')")
    p_full.add_argument("--latency_ids", type=str, nargs="+", default=None,
                        help="Latency ids for latency-expert checkpoints (default: all configured ids).")
    p_full.set_defaults(func=run_fullseq)

    # --- streaming ---
    p_stream = subparsers.add_parser(
        "streaming", parents=[parent],
        help="Streaming evaluation (single chunk_size)",
    )
    p_stream.add_argument("--exp_pattern", type=str, default="*s2039",
                          help="Glob pattern to match experiment directories.")
    p_stream.add_argument("--chunk_size", type=int, default=1,
                          help="LaCoSENet chunk size in STFT frames.")
    p_stream.add_argument("--align_ola", action="store_true",
                          help="Compensate OLA center shift (win_size//2 samples)")
    p_stream.add_argument("--split", type=str, default=None,
                          help="Split experiments: 'INDEX/TOTAL' (e.g., '0/2')")
    p_stream.add_argument("--latency_ids", type=str, nargs="+", default=None,
                          help="Latency ids for latency-expert checkpoints (default: all configured ids).")
    p_stream.set_defaults(func=run_streaming)

    # --- chunksweep ---
    p_sweep = subparsers.add_parser(
        "chunksweep", parents=[parent],
        help="Chunk-size sweep across multiple chunk_sizes",
    )
    p_sweep.add_argument("--experiments", type=str, nargs="+", required=True,
                         help="Experiment directory names to evaluate.")
    p_sweep.add_argument("--chunk_sizes", type=int, nargs="+",
                         default=[1, 4, 16, 64, 160],
                         help="Chunk sizes (STFT frames) to sweep.")
    p_sweep.add_argument("--align_ola", action="store_true",
                         help="Compensate OLA center shift (win_size//2 samples)")
    p_sweep.add_argument("--latency_id", type=str, default=None,
                         help="Latency id for a latency-expert checkpoint (default: configured default).")
    p_sweep.set_defaults(func=run_chunksweep)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
