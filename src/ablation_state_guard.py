"""
Selective State Update (C3) ablation: guard ON vs guard OFF.

Evaluates the impact of disabling StateFramesContext (selective state update)
across different models and chunk sizes. When disabled, lookahead frames
contaminate streaming state buffers, causing quality degradation proportional
to the lookahead-to-chunk ratio.

Note on metrics:
    Streaming inference uses a different iSTFT frontend (manual OLA) than the
    offline path (torch.istft). For models with large lookahead, the mismatch
    concentrates near the tail boundary due to lookahead flush padding. Therefore,
    this ablation trims the last 30 frames (30 * hop_size samples) from both clean
    and enhanced signals before metric computation, to reduce tail-boundary
    artifacts unrelated to the state-guard mechanism being studied.

Usage:
    # Full matrix (5 models × 3 chunk_sizes × 2 conditions = 30 evaluations)
    python -m src.ablation_state_guard --device cuda

    # Quick sanity check (M1 + M7 only)
    python -m src.ablation_state_guard --models M1 M7 --chunk_sizes 1 64 --device cuda
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

from src.batch_evaluate import (
    find_best_checkpoint,
    compute_streaming_lookahead,
    load_test_dataset,
    create_data_loader,
    evaluate_streaming_single,
)
from src.models.streaming.lacosenet import LaCoSENet
from src.utils import bold

EXPERIMENT_MATRIX = {
    "M1": {"model_dir": "M1_12.5ms", "latency": "12.5ms"},
    "M2": {"model_dir": "M2_25.0ms", "latency": "25.0ms"},
    "M4": {"model_dir": "M4_50.0ms", "latency": "50.0ms"},
    "M6": {"model_dir": "M6_75.0ms", "latency": "75.0ms"},
    "M7": {"model_dir": "M7_200.0ms", "latency": "200.0ms"},
}
DEFAULT_CHUNK_SIZES = [1, 16, 64]
DEFAULT_OUTPUT_DIR = "results/evaluation/ablation_state_guard"


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def run_single_evaluation(exp_dir, chkpt_file, conf, chunk_size,
                          disable_guard, data_loader, device, logger):
    """Evaluate a single (model, chunk_size, guard) condition."""
    la = compute_streaming_lookahead(conf, chunk_size=chunk_size)

    streaming = LaCoSENet.from_checkpoint(
        chkpt_dir=str(exp_dir),
        chkpt_file=chkpt_file,
        chunk_size=chunk_size,
        encoder_lookahead=la["encoder_lookahead"],
        decoder_lookahead=la["decoder_lookahead"],
        device=device,
        verbose=False,
        disable_state_guard=disable_guard,
    )

    shift_samples = la["stft_center_delay_samples"]
    tail_trim_frames = 30
    tail_trim_samples = tail_trim_frames * la["hop_size"]
    metrics = evaluate_streaming_single(
        streaming, data_loader, device, logger,
        shift_samples=shift_samples,
        tail_trim_samples=tail_trim_samples,
    )

    del streaming
    torch.cuda.empty_cache()
    return metrics, la


def main():
    parser = argparse.ArgumentParser(
        description="Selective State Update (C3) ablation study",
    )
    parser.add_argument("--models", type=str, nargs="+",
                        default=list(EXPERIMENT_MATRIX.keys()),
                        help="Model keys to evaluate (default: all)")
    parser.add_argument("--chunk_sizes", type=int, nargs="+",
                        default=DEFAULT_CHUNK_SIZES,
                        help="Chunk sizes to sweep (default: 1 16 64)")
    parser.add_argument("--exp_dir", type=str, default="results/experiments",
                        help="Base experiment directory")
    parser.add_argument("--seed", type=str, default="s2039",
                        help="Experiment seed subdirectory to evaluate")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip evaluations whose result JSON already exists")
    args = parser.parse_args()

    logger = setup_logging()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = Path(args.exp_dir)

    # Validate model keys
    models = [m for m in args.models if m in EXPERIMENT_MATRIX]
    if not models:
        logger.error(f"No valid models. Choose from: {list(EXPERIMENT_MATRIX.keys())}")
        return

    logger.info(f"Models: {models}")
    logger.info(f"Chunk sizes: {args.chunk_sizes}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Device: {args.device}")

    # Load dataset once
    logger.info("Loading VoiceBank-DEMAND test set...")
    hf_dataset = load_test_dataset()
    logger.info(f"Test set loaded: {len(hf_dataset)} utterances")

    # Collect all results
    all_results = {}
    total = len(models) * len(args.chunk_sizes) * 2
    done = 0

    for model_key in models:
        info = EXPERIMENT_MATRIX[model_key]
        exp_dir = base_dir / info["model_dir"] / args.seed

        if not exp_dir.exists():
            logger.error(f"Experiment directory not found: {exp_dir}")
            continue

        logger.info(bold(f"\n{'='*60}"))
        logger.info(bold(f"Model: {model_key} ({info['latency']})"))
        logger.info(bold(f"{'='*60}"))

        # Load checkpoint info and config
        best_info = find_best_checkpoint(exp_dir)
        best_step = best_info["step"]
        chkpt_file = f"model_{best_step}.th"
        logger.info(f"Best checkpoint: step={best_step}, valid_pesq={best_info['valid_pesq']:.4f}")

        config_path = exp_dir / ".hydra" / "config.yaml"
        if not config_path.exists():
            logger.error(f"Config not found: {config_path}")
            continue

        conf = OmegaConf.load(config_path)
        ev_loader = create_data_loader(hf_dataset, conf, args.num_workers)

        model_results = {}

        for cs in args.chunk_sizes:
            for guard_label, disable_guard in [("guard_on", False), ("guard_off", True)]:
                tag = f"{model_key}_{info['latency']}_cs{cs}_{guard_label}"
                done += 1

                # Skip if result already exists
                json_path = output_dir / f"{tag}.json"
                if args.skip_existing and json_path.exists():
                    logger.info(f"\n  [{done}/{total}] {tag} — SKIP (exists)")
                    with open(json_path) as f:
                        result_entry = json.load(f)
                    all_results[tag] = result_entry
                    continue

                logger.info(f"\n  [{done}/{total}] {tag}")

                metrics, la = run_single_evaluation(
                    exp_dir, chkpt_file, conf, cs,
                    disable_guard, ev_loader, args.device, logger,
                )

                logger.info(
                    f"    PESQ={metrics['pesq']:.4f}  STOI={metrics['stoi']:.4f}  "
                    f"CSIG={metrics['csig']:.4f}  CBAK={metrics['cbak']:.4f}"
                )

                result_entry = {
                    "model": model_key,
                    "seed": args.seed,
                    "chkpt_file": chkpt_file,
                    "latency_ms": la["latency_ms"],
                    "chunk_size": cs,
                    "guard": guard_label,
                    "disable_state_guard": disable_guard,
                    "encoder_lookahead": la["encoder_lookahead"],
                    "decoder_lookahead": la["decoder_lookahead"],
                    "total_lookahead": la["total_lookahead"],
                    "tail_trim_frames": 30,
                    "test_metrics": metrics,
                }

                # Save individual result
                json_path = output_dir / f"{tag}.json"
                with open(json_path, "w") as f:
                    json.dump(result_entry, f, indent=2)

                model_results[tag] = result_entry
                all_results[tag] = result_entry

    if not all_results:
        logger.error("No evaluations completed.")
        return

    # Build summary: δ metrics for each (model, chunk_size)
    summary = {"conditions": [], "deltas": []}

    for model_key in models:
        info = EXPERIMENT_MATRIX[model_key]
        for cs in args.chunk_sizes:
            on_tag = f"{model_key}_{info['latency']}_cs{cs}_guard_on"
            off_tag = f"{model_key}_{info['latency']}_cs{cs}_guard_off"

            if on_tag not in all_results or off_tag not in all_results:
                continue

            on_metrics = all_results[on_tag]["test_metrics"]
            off_metrics = all_results[off_tag]["test_metrics"]

            delta_entry = {
                "model": model_key,
                "latency_ms": all_results[on_tag]["latency_ms"],
                "chunk_size": cs,
                "total_lookahead": all_results[on_tag]["total_lookahead"],
            }
            for m in ["pesq", "stoi", "csig", "cbak", "covl", "segSNR"]:
                delta_entry[f"delta_{m}"] = round(off_metrics[m] - on_metrics[m], 6)
                delta_entry[f"guard_on_{m}"] = round(on_metrics[m], 6)
                delta_entry[f"guard_off_{m}"] = round(off_metrics[m], 6)

            summary["deltas"].append(delta_entry)

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved to {summary_path}")

    # Print summary table
    logger.info(bold(f"\n{'='*100}"))
    logger.info(bold("STATE GUARD ABLATION SUMMARY"))
    logger.info(bold(f"{'='*100}"))
    header = (
        f"{'Model':<6} {'Lat(ms)':>8} {'LA':>4} {'CS':>4} "
        f"{'ON PESQ':>9} {'OFF PESQ':>9} {'δPESQ':>9} "
        f"{'ON STOI':>9} {'OFF STOI':>9} {'δSTOI':>9}"
    )
    logger.info(header)
    logger.info("-" * 100)

    for d in summary["deltas"]:
        logger.info(
            f"{d['model']:<6} {d['latency_ms']:>8.1f} {d['total_lookahead']:>4} {d['chunk_size']:>4} "
            f"{d['guard_on_pesq']:>9.4f} {d['guard_off_pesq']:>9.4f} {d['delta_pesq']:>+9.4f} "
            f"{d['guard_on_stoi']:>9.4f} {d['guard_off_stoi']:>9.4f} {d['delta_stoi']:>+9.4f}"
        )


if __name__ == "__main__":
    main()
