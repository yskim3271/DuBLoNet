"""
Verify training-inference equivalence: full-sequence vs chunk-wise streaming.

Compares model output spectrograms (est_mag, est_pha) between:
  1. Full-sequence forward pass (Backbone.forward on entire spectrogram)
  2. Chunk-by-chunk streaming (LaCoSENet.process_spectrogram_buffered)

Both paths receive the SAME input spectrogram (training STFT), isolating
model-level equivalence from STFT frontend differences.

A single test utterance suffices since this is a numerical identity check.

Usage:
    python -m src.verify_equivalence --device cuda
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import numpy as np
from omegaconf import OmegaConf
from datasets import load_dataset

from src.data import VoiceBankDataset
from src.stft import mag_pha_stft
from src.utils import load_model, load_checkpoint, get_stft_args_from_config
from src.models.streaming.lacosenet import LaCoSENet
from src.batch_evaluate import compute_streaming_lookahead, get_config_latency_ids

# ── Experiment registry ──────────────────────────────────────────────────
EXPERIMENTS = {
    "M1_12.5ms":  "results/experiments/M1_12.5ms",   # L_enc=0, L_dec=0
    "M4_50.0ms":  "results/experiments/M4_50.0ms",    # asymmetric
    "M7_200.0ms": "results/experiments/M7_200.0ms",   # symmetric (r_R=0.5)
}
SEED = "s2039"


def resolve_best_checkpoint(exp_seed_dir: Path) -> str:
    """Find the best checkpoint file from states.th."""
    states = torch.load(exp_seed_dir / "states.th", map_location="cpu", weights_only=False)
    best = states["best_models"][0]
    step = best["steps"]
    chkpt_file = f"model_{step}.th"
    assert (exp_seed_dir / chkpt_file).exists(), f"{chkpt_file} not found"
    return chkpt_file


def load_single_utterance():
    """Load one test utterance from VoiceBank-DEMAND."""
    hf_dataset = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k", split="test")
    dataset = VoiceBankDataset(hf_dataset, segment=None, with_id=True, with_text=True)
    noisy, clean, utt_id, _ = dataset[0]
    return noisy.unsqueeze(0), utt_id  # [1, 1, T]


def fullseq_forward(model, noisy_com, device, latency_id=None):
    """Full-sequence model forward → (est_mag, est_pha)."""
    model.eval()
    with torch.no_grad():
        if latency_id is not None:
            est_mag, est_pha, _ = model(noisy_com.to(device), latency_id=latency_id)
        else:
            est_mag, est_pha, _ = model(noisy_com.to(device))
    return est_mag, est_pha


def streaming_forward(streaming_model, noisy_com, chunk_size, enc_la):
    """Chunk-by-chunk streaming forward → (est_mag, est_pha).

    Feeds spectrogram slices through process_spectrogram_buffered,
    matching the chunking that LaCoSENet performs internally.
    """
    T_total = noisy_com.shape[2]  # [B, F, T, 2]
    all_mag, all_pha = [], []

    for t in range(0, T_total, chunk_size):
        t_end = min(t + chunk_size + enc_la, T_total)
        chunk = noisy_com[:, :, t:t_end, :]

        result = streaming_model.process_spectrogram_buffered(chunk)
        if result is not None:
            est_mag, est_pha = result
            all_mag.append(est_mag)
            all_pha.append(est_pha)

    if not all_mag:
        return None, None

    return torch.cat(all_mag, dim=2), torch.cat(all_pha, dim=2)


def verify_checkpoint(exp_name, exp_seed_dir, noisy, device, chunk_size=1, latency_id=None, chkpt_file=None):
    """Verify equivalence for a single checkpoint directory."""
    exp_seed_dir = Path(exp_seed_dir)
    conf = OmegaConf.load(exp_seed_dir / ".hydra" / "config.yaml")
    stft_args = get_stft_args_from_config(conf.model)
    if chkpt_file is None or chkpt_file == "auto":
        chkpt_file = resolve_best_checkpoint(exp_seed_dir)

    # ── Compute shared spectrogram (training STFT) ──
    noisy_com = mag_pha_stft(noisy, **stft_args)[2].to(device)  # [1, F, T, 2]
    T_total = noisy_com.shape[2]

    # ── Full-sequence model ──
    model = load_model(conf.model, device)
    model = load_checkpoint(model, str(exp_seed_dir), chkpt_file, device)
    mag_full, pha_full = fullseq_forward(model, noisy_com, device, latency_id=latency_id)
    mag_full = mag_full.cpu()
    pha_full = pha_full.cpu()
    del model
    if "cuda" in device:
        torch.cuda.empty_cache()

    # ── Streaming model ──
    la_info = compute_streaming_lookahead(conf, chunk_size, latency_id=latency_id)
    enc_la = la_info["encoder_lookahead"]
    dec_la = la_info["decoder_lookahead"]
    total_la = la_info["total_lookahead"]

    streaming = LaCoSENet.from_checkpoint(
        chkpt_dir=str(exp_seed_dir),
        chkpt_file=chkpt_file,
        chunk_size=chunk_size,
        encoder_lookahead=enc_la,
        decoder_lookahead=dec_la,
        latency_id=latency_id,
        fold_bn=False,
        device=device,
        verbose=False,
    )
    streaming.reset_state()
    mag_stream, pha_stream = streaming_forward(streaming, noisy_com, chunk_size, enc_la)
    del streaming
    if "cuda" in device:
        torch.cuda.empty_cache()

    if mag_stream is None:
        return {"exp_name": exp_name, "error": "No streaming output produced"}

    # ── Align and trim ──
    # Streaming frame k corresponds to full-seq frame k (no positional shift).
    # The lookahead delay only affects when the first output appears, not which
    # spectrogram position it represents.
    T_stream = mag_stream.shape[2]
    mag_full_aligned = mag_full[:, :, :T_stream]
    pha_full_aligned = pha_full[:, :, :T_stream]

    # Exclude tail: the last total_la frames of the input spectrogram may lack
    # sufficient future context in streaming (boundary effect).
    tail_margin = total_la
    safe_T = max(T_stream - tail_margin, 0) if tail_margin > 0 else T_stream

    mag_full_cmp = mag_full_aligned[:, :, :safe_T].detach().cpu().numpy()
    pha_full_cmp = pha_full_aligned[:, :, :safe_T].detach().cpu().numpy()
    mag_stream_cmp = mag_stream[:, :, :safe_T].detach().cpu().numpy()
    pha_stream_cmp = pha_stream[:, :, :safe_T].detach().cpu().numpy()

    mag_err = np.abs(mag_full_cmp - mag_stream_cmp)
    pha_err = np.abs(pha_full_cmp - pha_stream_cmp)

    return {
        "exp_name": exp_name,
        "latency_id": latency_id,
        "latency_ms": la_info["latency_ms"],
        "enc_la": enc_la,
        "dec_la": dec_la,
        "T_total": T_total,
        "T_compared": safe_T,
        "mag_max_err": float(mag_err.max()),
        "mag_mean_err": float(mag_err.mean()),
        "pha_max_err": float(pha_err.max()),
        "pha_mean_err": float(pha_err.mean()),
    }


def verify_one(exp_name, exp_dir, noisy, device, chunk_size=1):
    """Verify equivalence for a single legacy registry experiment."""
    exp_seed_dir = Path(exp_dir) / SEED
    return verify_checkpoint(exp_name, exp_seed_dir, noisy, device, chunk_size=chunk_size)


def main():
    parser = argparse.ArgumentParser(description="Verify full-seq vs streaming equivalence")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--chunk_size", type=int, default=1, help="Streaming chunk size")
    parser.add_argument("--chkpt_dir", type=str, default=None,
                        help="Checkpoint directory containing .hydra/config.yaml. If omitted, uses the built-in registry.")
    parser.add_argument("--chkpt_file", type=str, default="auto",
                        help="Checkpoint filename, or 'auto' to use states.th best model.")
    parser.add_argument("--latency_ids", type=str, nargs="+", default=None,
                        help="Latency ids for latency-expert checkpoints (default: all configured ids).")
    args = parser.parse_args()

    print("Loading test utterance...")
    noisy, utt_id = load_single_utterance()
    print(f"  Utterance: {utt_id}, samples: {noisy.shape[-1]}")
    print()

    print(f"{'Experiment':<18} {'τ (ms)':>7} {'L_e':>3} {'L_d':>3} "
          f"{'T_cmp':>5} {'Mag MaxErr':>11} {'Mag MeanErr':>12} "
          f"{'Pha MaxErr':>11} {'Pha MeanErr':>12}")
    print("-" * 100)

    if args.chkpt_dir:
        conf = OmegaConf.load(Path(args.chkpt_dir) / ".hydra" / "config.yaml")
        jobs = [
            (Path(args.chkpt_dir).name, Path(args.chkpt_dir), latency_id)
            for latency_id in get_config_latency_ids(conf, args.latency_ids)
        ]
    else:
        jobs = []
        for exp_name, exp_dir in EXPERIMENTS.items():
            seed_dir = Path(exp_dir) / SEED
            jobs.append((exp_name, seed_dir, None))

    for exp_name, seed_dir, latency_id in jobs:
        if not seed_dir.exists():
            print(f"{exp_name:<18} SKIPPED (directory not found)")
            continue

        result = verify_checkpoint(
            exp_name,
            seed_dir,
            noisy,
            args.device,
            args.chunk_size,
            latency_id=latency_id,
            chkpt_file=args.chkpt_file,
        )
        if "error" in result:
            print(f"{exp_name:<18} ERROR: {result['error']}")
            continue

        label = result["exp_name"]
        if result.get("latency_id"):
            label = f"{label}_{result['latency_id']}"
        print(f"{label:<18} {result['latency_ms']:>5.1f}ms "
              f"{result['enc_la']:>3} {result['dec_la']:>3} "
              f"{result['T_compared']:>5} "
              f"{result['mag_max_err']:>11.2e} {result['mag_mean_err']:>12.2e} "
              f"{result['pha_max_err']:>11.2e} {result['pha_mean_err']:>12.2e}")

    print()
    print("If max errors < 1e-5, training-inference equivalence is confirmed")
    print("(differences are floating-point accumulation order).")


if __name__ == "__main__":
    main()
