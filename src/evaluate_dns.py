"""Cross-dataset evaluation (e.g., DNS Challenge) for LaCo-SENet rebuttal.

Runs enhancement on a paired (noisy/clean) directory, computes intrusive
metrics (PESQ wideband, STOI, CSIG/CBAK/COVL/segSNR), and computes DNSMOS
P.835 (SIG/BAK/OVRL) via ``speechmos`` unless explicitly disabled.

Designed to mirror ``src/evaluate.py`` CLI so existing checkpoints load without
changes.

Example::

    python -m src.evaluate_dns \
        --model_config results/experiments/M1_12.5ms/s42/.hydra/config.yaml \
        --chkpt_dir   results/experiments/M1_12.5ms/s42 \
        --chkpt_file  best.th \
        --noisy_dir   /data/dns2020/no_reverb/noisy \
        --clean_dir   /data/dns2020/no_reverb/clean \
        --layout      dns2020 \
        --out_json    results/rebuttal/dns_eval/M1_s42.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.compute_metrics import compute_metrics
from src.data_dns import PairedWavDataset
from src.stft import mag_pha_stft, mag_pha_istft
from src.utils import LogProgress, bold, get_stft_args_from_config, load_checkpoint, load_model


def _try_import_dnsmos():
    """Import a DNSMOS implementation; return a callable or None."""
    # Preferred: speechmos.dnsmos (pip install speechmos)
    try:
        from speechmos import dnsmos as _dnsmos
        def _run(wav: np.ndarray, sr: int = 16000) -> Dict[str, float]:
            # speechmos requires wav values in [-1, 1]; our dataset applies power
            # normalization, so peak-normalize before scoring only when required
            # by the wrapper. The external DNSMOS scale policy is handled by
            # --dnsmos_scale before this callable is invoked.
            peak = float(np.max(np.abs(wav))) if wav.size else 0.0
            if peak > 1.0:
                wav = (wav / peak).astype(np.float32, copy=False)
            out = _dnsmos.run(wav, sr=sr)
            # speechmos returns keys: 'ovrl_mos', 'sig_mos', 'bak_mos', 'p808_mos'
            return {
                "sig": float(out["sig_mos"]),
                "bak": float(out["bak_mos"]),
                "ovrl": float(out["ovrl_mos"]),
                "p808": float(out.get("p808_mos", float("nan"))),
            }
        return _run
    except Exception:
        return None


def evaluate_paired(
    args: Any,
    model: torch.nn.Module,
    data_loader: DataLoader,
    logger: logging.Logger,
    stft_args: Dict[str, Any],
    dnsmos_fn=None,
    dnsmos_scale: str = "current",
) -> Dict[str, Any]:
    """Run enhancement + metric computation on a paired dataset.

    Args:
        dnsmos_scale: ``'current'`` scores the (power-)normalized signals fed to
            the model. ``'denorm'`` scores ``noisy_norm / norm_factor`` and
            ``enhanced_norm / norm_factor`` to undo the input-side power
            normalization, so DNSMOS is computed at the raw waveform scale
            while the model still consumes the training-time normalized input.

    Returns a dict of mean/std metrics plus per-sample records.
    """
    if dnsmos_scale not in ("current", "denorm"):
        raise ValueError(f"Unknown dnsmos_scale '{dnsmos_scale}'.")
    model.eval()
    iterator = LogProgress(logger, data_loader, name="DNS-like eval")

    per_sample: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch_idx, data in enumerate(iterator):
            noisy, clean, sample_id, _, norm_factor = data

            noisy_com = mag_pha_stft(noisy, **stft_args)[2].to(args.device)
            mag_hat, pha_hat, _ = model(noisy_com)
            enhanced = mag_pha_istft(mag_hat, pha_hat, **stft_args)

            clean_np = clean.squeeze().detach().cpu().numpy().astype(np.float32)
            noisy_np = noisy.squeeze().detach().cpu().numpy().astype(np.float32)
            enh_np = enhanced.squeeze().detach().cpu().numpy().astype(np.float32)

            L = min(len(clean_np), len(enh_np))
            clean_np = clean_np[:L]
            enh_np = enh_np[:L]
            noisy_trim = noisy_np[:L]

            # Intrusive metrics (clean vs enhanced) — independent of dnsmos_scale.
            try:
                pesq, csig, cbak, covl, ssnr, stoi = compute_metrics(clean_np, enh_np)
            except Exception as e:
                logger.warning(f"intrusive metric failed for {sample_id[0]}: {e}")
                pesq = csig = cbak = covl = ssnr = stoi = float("nan")

            nf = float(norm_factor.item() if hasattr(norm_factor, "item") else norm_factor)

            # DNSMOS (no-reference) on both noisy and enhanced for delta tracking.
            dns_noisy = dns_enh = None
            if dnsmos_fn is not None:
                if dnsmos_scale == "denorm":
                    inv = 1.0 / nf if nf != 0.0 else 1.0
                    noisy_for_dns = (noisy_trim * inv).astype(np.float32, copy=False)
                    enh_for_dns = (enh_np * inv).astype(np.float32, copy=False)
                else:
                    noisy_for_dns = noisy_trim
                    enh_for_dns = enh_np
                try:
                    dns_noisy = dnsmos_fn(noisy_for_dns, sr=16000)
                    dns_enh = dnsmos_fn(enh_for_dns, sr=16000)
                except Exception as e:
                    logger.warning(f"DNSMOS failed for {sample_id[0]}: {e}")

            rec = {
                "id": sample_id[0] if isinstance(sample_id, (list, tuple)) else str(sample_id),
                "norm_factor": nf,
                "pesq": float(pesq),
                "stoi": float(stoi),
                "csig": float(csig),
                "cbak": float(cbak),
                "covl": float(covl),
                "segSNR": float(ssnr),
            }
            if dns_enh is not None:
                rec.update({
                    "dnsmos_sig_enh": dns_enh["sig"],
                    "dnsmos_bak_enh": dns_enh["bak"],
                    "dnsmos_ovrl_enh": dns_enh["ovrl"],
                    "dnsmos_p808_enh": dns_enh["p808"],
                })
            if dns_noisy is not None:
                rec.update({
                    "dnsmos_sig_noisy": dns_noisy["sig"],
                    "dnsmos_bak_noisy": dns_noisy["bak"],
                    "dnsmos_ovrl_noisy": dns_noisy["ovrl"],
                    "dnsmos_p808_noisy": dns_noisy["p808"],
                })
            per_sample.append(rec)

    # Aggregate
    keys = [k for k in per_sample[0].keys() if k not in ("id", "norm_factor")]
    summary = {}
    for k in keys:
        vals = np.array([r[k] for r in per_sample], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0), "n": int(len(vals))}

    # Short log
    def _fmt(k: str) -> str:
        if k not in summary:
            return f"{k}=NA"
        s = summary[k]
        return f"{k}={s['mean']:.3f}±{s['std']:.3f}"

    log_keys = ["pesq", "stoi", "csig", "cbak", "covl", "dnsmos_ovrl_enh", "dnsmos_sig_enh", "dnsmos_bak_enh"]
    logger.info(bold("Aggregate: " + " | ".join(_fmt(k) for k in log_keys if k in summary)))
    if "dnsmos_ovrl_noisy" in summary and "dnsmos_ovrl_enh" in summary:
        d = summary["dnsmos_ovrl_enh"]["mean"] - summary["dnsmos_ovrl_noisy"]["mean"]
        logger.info(bold(f"ΔDNSMOS-OVRL (enh − noisy) = {d:+.3f}"))

    return {"summary": summary, "per_sample": per_sample}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--chkpt_dir", type=str, required=True)
    parser.add_argument("--chkpt_file", type=str, required=True)
    parser.add_argument("--noisy_dir", type=str, required=True)
    parser.add_argument("--clean_dir", type=str, required=True)
    parser.add_argument("--layout", type=str, default="simple", choices=["simple", "dns2020"])
    parser.add_argument("--out_json", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--skip_dnsmos", action="store_true", help="Disable DNSMOS even if available.")
    parser.add_argument(
        "--dnsmos_scale",
        choices=["current", "denorm"],
        default="current",
        help=(
            "DNSMOS scoring policy: 'current' = score the power-normalized "
            "noisy/enhanced signals (model input scale); 'denorm' = divide by "
            "the per-utterance norm_factor before scoring, so DNSMOS sees the "
            "raw-scale noisy and enhanced/norm_factor."
        ),
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for smoke tests.")
    parser.add_argument("--log_file", type=str, default=None)
    args = parser.parse_args()

    handlers = [logging.StreamHandler()]
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=handlers)
    logger = logging.getLogger(__name__)

    conf = OmegaConf.load(args.model_config)
    conf.device = args.device

    model = load_model(conf.model, args.device)
    model = load_checkpoint(model, args.chkpt_dir, args.chkpt_file, args.device)
    stft_args = get_stft_args_from_config(conf.model)

    ds = PairedWavDataset(
        noisy_dir=args.noisy_dir,
        clean_dir=args.clean_dir,
        sampling_rate=16000,
        power_normalize=True,
        layout=args.layout,
        with_id=True,
        with_text=True,
        with_norm_factor=True,
    )
    if args.max_samples is not None:
        ds.pairs = ds.pairs[: args.max_samples]
    loader = DataLoader(ds, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    dnsmos_fn = None if args.skip_dnsmos else _try_import_dnsmos()
    if dnsmos_fn is None and not args.skip_dnsmos:
        raise RuntimeError(
            "speechmos is required for this evaluation. Install it with "
            "`python -m pip install speechmos==0.0.1.1`, or pass "
            "`--skip_dnsmos` for an intrusive-metrics-only run."
        )

    logger.info(f"Dataset: paired ({args.noisy_dir} / {args.clean_dir}) layout={args.layout} n={len(ds)}")
    logger.info(f"Checkpoint: {args.chkpt_dir}/{args.chkpt_file}")
    logger.info(f"Device: {args.device}")
    logger.info(f"DNSMOS: {'enabled' if dnsmos_fn else 'skipped'} (scale={args.dnsmos_scale})")

    result = evaluate_paired(
        conf, model, loader, logger, stft_args, dnsmos_fn,
        dnsmos_scale=args.dnsmos_scale,
    )

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({
                "model_config": args.model_config,
                "chkpt": os.path.join(args.chkpt_dir, args.chkpt_file),
                "noisy_dir": args.noisy_dir,
                "clean_dir": args.clean_dir,
                "layout": args.layout,
                "dnsmos_scale": args.dnsmos_scale,
                "n_samples": len(result["per_sample"]),
                "summary": result["summary"],
                "per_sample": result["per_sample"],
            }, f, indent=2)
        logger.info(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
