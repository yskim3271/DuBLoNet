"""Check whether DNSMOS conclusions depend on evaluation-time scaling.

The model is always fed the training-time power-normalized noisy waveform.
This script compares two DNSMOS scoring policies:

1. current: score the normalized noisy/enhanced signals, matching
   ``src.evaluate_dns``.
2. denorm: score raw noisy and enhanced / norm_factor, i.e. undo the
   model-input power normalization before DNSMOS.

Both policies still apply the speechmos-required peak guard only when
``abs(wav).max() > 1``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from speechmos import dnsmos

from src.stft import mag_pha_istft, mag_pha_stft
from src.utils import get_stft_args_from_config, load_checkpoint, load_model


def load_wav(path: Path, sr: int = 16000) -> np.ndarray:
    wav, got_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if got_sr != sr:
        raise ValueError(f"Expected sr={sr}, got {got_sr}: {path}")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32, copy=False)


def pairs_simple(noisy_dir: Path, clean_dir: Path) -> List[Tuple[Path, Path, str]]:
    clean_map = {p.stem: p for p in clean_dir.glob("*.wav")}
    out = []
    for noisy_path in sorted(noisy_dir.glob("*.wav")):
        clean_path = clean_map.get(noisy_path.stem)
        if clean_path is not None:
            out.append((noisy_path, clean_path, noisy_path.stem))
    return out


def pairs_dns2020(noisy_dir: Path, clean_dir: Path) -> List[Tuple[Path, Path, str]]:
    pat = re.compile(r"fileid_(\d+)")
    clean_map = {}
    for p in clean_dir.glob("clean_fileid_*.wav"):
        m = pat.search(p.name)
        if m:
            clean_map[m.group(1)] = p
    out = []
    for noisy_path in sorted(noisy_dir.glob("*fileid_*.wav")):
        m = pat.search(noisy_path.name)
        if not m:
            continue
        fid = m.group(1)
        clean_path = clean_map.get(fid)
        if clean_path is not None:
            out.append((noisy_path, clean_path, f"fileid_{fid}"))
    return out


def choose_indices(n_total: int, n: int) -> Iterable[int]:
    n = min(n, n_total)
    if n <= 0:
        return []
    return np.linspace(0, n_total - 1, n, dtype=int).tolist()


def dnsmos_score(wav: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    wav = np.asarray(wav, dtype=np.float32)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    peak_guarded = False
    if peak > 1.0:
        wav = (wav / peak).astype(np.float32, copy=False)
        peak_guarded = True
    out = dnsmos.run(wav, sr=sr)
    return {
        "sig": float(out["sig_mos"]),
        "bak": float(out["bak_mos"]),
        "ovrl": float(out["ovrl_mos"]),
        "p808": float(out.get("p808_mos", float("nan"))),
        "peak_before_guard": peak,
        "peak_guarded": peak_guarded,
    }


def summarize(records: List[Dict]) -> Dict[str, Dict[str, float]]:
    keys = [
        "current_delta_ovrl",
        "denorm_delta_ovrl",
        "current_enh_ovrl",
        "denorm_enh_ovrl",
        "current_noisy_ovrl",
        "denorm_noisy_ovrl",
        "delta_shift_ovrl",
    ]
    out = {}
    for key in keys:
        vals = np.array([r[key] for r in records], dtype=np.float64)
        out[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0),
            "n": int(len(vals)),
        }
    for key in ["current_noisy_peak_guarded", "current_enh_peak_guarded", "denorm_noisy_peak_guarded", "denorm_enh_peak_guarded"]:
        vals = np.array([bool(r[key]) for r in records], dtype=np.float64)
        out[key] = {"rate": float(vals.mean()), "n": int(len(vals))}
    return out


def evaluate_dataset(
    name: str,
    pairs: List[Tuple[Path, Path, str]],
    n: int,
    model: torch.nn.Module,
    stft_args: Dict,
    device: str,
) -> Dict:
    records = []
    for idx in choose_indices(len(pairs), n):
        noisy_path, clean_path, sample_id = pairs[idx]
        noisy_raw = load_wav(noisy_path)
        clean_raw = load_wav(clean_path)
        length = min(len(noisy_raw), len(clean_raw))
        noisy_raw = noisy_raw[:length]
        norm_factor = math.sqrt(float(length) / max(float(np.sum(noisy_raw ** 2.0)), 1e-12))
        noisy_norm = (noisy_raw * norm_factor).astype(np.float32, copy=False)

        with torch.no_grad():
            noisy_t = torch.from_numpy(noisy_norm.copy()).unsqueeze(0)
            noisy_com = mag_pha_stft(noisy_t, **stft_args)[2].to(device)
            mag_hat, pha_hat, _ = model(noisy_com)
            enhanced_norm = mag_pha_istft(mag_hat, pha_hat, **stft_args)
        enhanced_norm_np = enhanced_norm.squeeze(0).detach().cpu().numpy().astype(np.float32)

        out_len = min(len(noisy_raw), len(enhanced_norm_np))
        noisy_raw_trim = noisy_raw[:out_len]
        noisy_norm_trim = noisy_norm[:out_len]
        enhanced_norm_trim = enhanced_norm_np[:out_len]
        enhanced_denorm = (enhanced_norm_trim / norm_factor).astype(np.float32, copy=False)

        cur_noisy = dnsmos_score(noisy_norm_trim)
        cur_enh = dnsmos_score(enhanced_norm_trim)
        den_noisy = dnsmos_score(noisy_raw_trim)
        den_enh = dnsmos_score(enhanced_denorm)

        cur_delta = cur_enh["ovrl"] - cur_noisy["ovrl"]
        den_delta = den_enh["ovrl"] - den_noisy["ovrl"]
        records.append({
            "dataset": name,
            "id": sample_id,
            "noisy_path": str(noisy_path),
            "norm_factor": norm_factor,
            "current_noisy_ovrl": cur_noisy["ovrl"],
            "current_enh_ovrl": cur_enh["ovrl"],
            "current_delta_ovrl": cur_delta,
            "denorm_noisy_ovrl": den_noisy["ovrl"],
            "denorm_enh_ovrl": den_enh["ovrl"],
            "denorm_delta_ovrl": den_delta,
            "delta_shift_ovrl": den_delta - cur_delta,
            "current_noisy_peak_guarded": cur_noisy["peak_guarded"],
            "current_enh_peak_guarded": cur_enh["peak_guarded"],
            "denorm_noisy_peak_guarded": den_noisy["peak_guarded"],
            "denorm_enh_peak_guarded": den_enh["peak_guarded"],
        })
    return {"summary": summarize(records), "per_sample": records}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_config", default=str(REPO_ROOT / "results/experiments/M1_12.5ms/s42/.hydra/config.yaml"))
    ap.add_argument("--chkpt_dir", default=str(REPO_ROOT / "results/experiments/M1_12.5ms/s42"))
    ap.add_argument("--chkpt_file", default=None, help="Defaults to the last Best Model in trainer.log.")
    ap.add_argument("--n", type=int, default=30, help="Samples per dataset.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_json", default=str(REPO_ROOT / "results/rebuttal/dnsmos_scale_check/scale_check_m1_s42.json"))
    args = ap.parse_args()

    chkpt_file = args.chkpt_file
    if chkpt_file is None:
        trainer_log = Path(args.chkpt_dir) / "trainer.log"
        best_step = None
        for line in trainer_log.read_text().splitlines():
            if "Best Model" in line:
                m = re.search(r"Steps:\s*(\d+)", line)
                if m:
                    best_step = m.group(1)
        if best_step is None:
            raise RuntimeError(f"Could not find Best Model line in {trainer_log}")
        chkpt_file = f"model_{best_step}.th"

    conf = OmegaConf.load(args.model_config)
    model = load_model(conf.model, args.device)
    model = load_checkpoint(model, args.chkpt_dir, chkpt_file, args.device)
    model.eval()
    stft_args = get_stft_args_from_config(conf.model)

    vb_pairs = pairs_simple(
        REPO_ROOT / "data/voicebank_test_wav/noisy",
        REPO_ROOT / "data/voicebank_test_wav/clean",
    )
    dns_pairs = pairs_dns2020(
        REPO_ROOT / "data/dns2020/DNS-Challenge/datasets/test_set/synthetic/no_reverb/noisy",
        REPO_ROOT / "data/dns2020/DNS-Challenge/datasets/test_set/synthetic/no_reverb/clean",
    )
    if not vb_pairs or not dns_pairs:
        raise RuntimeError(f"Missing pairs: voicebank={len(vb_pairs)}, dns2020={len(dns_pairs)}")

    result = {
        "model_config": args.model_config,
        "chkpt": str(Path(args.chkpt_dir) / chkpt_file),
        "n_per_dataset": args.n,
        "device": args.device,
        "datasets": {
            "voicebank": evaluate_dataset("voicebank", vb_pairs, args.n, model, stft_args, args.device),
            "dns2020": evaluate_dataset("dns2020", dns_pairs, args.n, model, stft_args, args.device),
        },
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print(f"wrote {out_path}")
    for name, data in result["datasets"].items():
        s = data["summary"]
        print(
            f"{name}: current_delta={s['current_delta_ovrl']['mean']:+.3f}, "
            f"denorm_delta={s['denorm_delta_ovrl']['mean']:+.3f}, "
            f"shift={s['delta_shift_ovrl']['mean']:+.3f}"
        )


if __name__ == "__main__":
    main()
