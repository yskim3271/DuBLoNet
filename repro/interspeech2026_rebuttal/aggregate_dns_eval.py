"""Aggregate DNS-style evaluation JSONs into a per-(model, dataset) summary.

Reads ``results/rebuttal/dns_eval/{D,C}_<MODEL>_<SEED>_<dataset>.json`` produced
by ``repro/interspeech2026_rebuttal/run_dns_eval_all.sh`` and emits:

  - markdown table to stdout (or --md-out)
  - CSV (--csv-out)
  - per-config aggregate JSON (--json-out)

Aggregation: mean and standard deviation over seeds, treating each seed's
per-utterance mean as one observation (matches the reporting convention used
in the paper, e.g., 3.35 +/- .02 over 3 seeds).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

FNAME_RE = re.compile(r"^(?P<kind>[CD])_(?P<model>M\d+_[\d\.]+ms)_(?P<seed>s\d+)_(?P<dataset>voicebank|dns2020)\.json$")

DATASET_ORDER = ["voicebank", "dns2020"]
DATASET_LABEL = {"voicebank": "VoiceBank+DEMAND", "dns2020": "DNS 2020 (synth, no_reverb)"}

# Model display order (latency ascending)
MODEL_ORDER = [
    "M1_12.5ms", "M2_25.0ms", "M3_37.5ms", "M4_50.0ms",
    "M5_62.5ms", "M6_75.0ms", "M9_100.0ms", "M13_150.0ms", "M7_200.0ms",
]

# Metrics to report (in order). Keep both intrusive and DNSMOS.
INTRUSIVE = ["pesq", "stoi", "csig", "cbak", "covl"]
DNSMOS_ENH = ["dnsmos_ovrl_enh", "dnsmos_sig_enh", "dnsmos_bak_enh", "dnsmos_p808_enh"]
DNSMOS_NOISY = ["dnsmos_ovrl_noisy", "dnsmos_sig_noisy", "dnsmos_bak_noisy", "dnsmos_p808_noisy"]


def parse_filename(path: Path):
    m = FNAME_RE.match(path.name)
    if not m:
        return None
    return m.groupdict()


def latency_ms_from_model(model: str) -> float:
    m = re.search(r"_([\d\.]+)ms", model)
    return float(m.group(1)) if m else float("inf")


def load_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def collect(root: Path) -> Tuple[Dict[Tuple[str, str], Dict[str, List[float]]], Dict[Tuple[str, str], List[str]]]:
    """Returns ({(model, dataset): {metric: [seed_means...]}}, {(model, dataset): [seeds]})."""
    out: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    seeds_seen: Dict[Tuple[str, str], set] = defaultdict(set)
    for path in sorted(root.glob("[CD]_*_voicebank.json")) + sorted(root.glob("[CD]_*_dns2020.json")):
        meta = parse_filename(path)
        if meta is None:
            continue
        d = load_json(path)
        summary = d.get("summary", {})
        key = (meta["model"], meta["dataset"])
        seeds_seen[key].add(meta["seed"])
        for metric, agg in summary.items():
            mean_val = agg.get("mean")
            if mean_val is None:
                continue
            out[key][metric].append(float(mean_val))
    return out, {k: sorted(v) for k, v in seeds_seen.items()}


def fmt(mean: float, std: float, decimals: int = 3) -> str:
    if np.isnan(mean):
        return "-"
    return f"{mean:.{decimals}f}+/-{std:.{decimals}f}"


def aggregate(seed_means: List[float]) -> Tuple[float, float, int]:
    if not seed_means:
        return float("nan"), float("nan"), 0
    arr = np.asarray(seed_means, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), 0
    mean = float(arr.mean())
    std = float(arr.std(ddof=1) if len(arr) > 1 else 0.0)
    return mean, std, len(arr)


def render_markdown(data: Dict[Tuple[str, str], Dict[str, List[float]]],
                     seeds_seen: Dict[Tuple[str, str], List[str]],
                     title: str = "") -> str:
    lines: List[str] = []
    if title:
        lines.append(f"# {title}\n")
    metric_groups = [
        ("Intrusive (clean vs enhanced)", INTRUSIVE, 3),
        ("DNSMOS P.835 - enhanced", DNSMOS_ENH, 3),
        ("DNSMOS P.835 - noisy (reference baseline)", DNSMOS_NOISY, 3),
    ]
    for dataset in DATASET_ORDER:
        models_present = [m for m in MODEL_ORDER if (m, dataset) in data]
        if not models_present:
            continue
        lines.append(f"## {DATASET_LABEL[dataset]}\n")
        for group_name, metrics, dec in metric_groups:
            lines.append(f"### {group_name}\n")
            header = ["Model", "tau (ms)", "n_seeds"] + metrics
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + "|".join(["---"] * len(header)) + "|")
            for model in models_present:
                key = (model, dataset)
                d = data[key]
                tau = latency_ms_from_model(model)
                n_seeds = len(seeds_seen.get(key, []))
                row = [model, f"{tau:.1f}", str(n_seeds)]
                for metric in metrics:
                    mean, std, _ = aggregate(d.get(metric, []))
                    row.append(fmt(mean, std, decimals=dec))
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
        lines.append("")
    # Delta DNSMOS row
    lines.append("## Delta DNSMOS OVRL (enhanced - noisy), per dataset\n")
    lines.append("| Model | tau (ms) | VoiceBank Delta | DNS Delta |")
    lines.append("|---|---|---|---|")
    all_models = [m for m in MODEL_ORDER if any((m, ds) in data for ds in DATASET_ORDER)]
    for model in all_models:
        tau = latency_ms_from_model(model)
        cells = [model, f"{tau:.1f}"]
        for ds in DATASET_ORDER:
            key = (model, ds)
            if key in data:
                enh = aggregate(data[key].get("dnsmos_ovrl_enh", []))[0]
                ny = aggregate(data[key].get("dnsmos_ovrl_noisy", []))[0]
                if not np.isnan(enh) and not np.isnan(ny):
                    cells.append(f"{enh - ny:+.3f}")
                else:
                    cells.append("-")
            else:
                cells.append("-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_comparison(
    data_cur: Dict[Tuple[str, str], Dict[str, List[float]]],
    data_den: Dict[Tuple[str, str], Dict[str, List[float]]],
    seeds_cur: Dict[Tuple[str, str], List[str]],
    seeds_den: Dict[Tuple[str, str], List[str]],
) -> str:
    """Side-by-side delta-DNSMOS-OVRL table: current vs denorm."""
    lines: List[str] = []
    lines.append("## Comparison: Delta DNSMOS-OVRL (enh - noisy) - current vs denorm\n")
    lines.append("Each cell is mean of seed-level per-utterance means; +/- is across-seed std (n_seeds shown). Shift = denorm - current.\n")
    header = ["Model", "tau (ms)", "Dataset", "n_seeds", "current Delta", "denorm Delta", "shift"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    all_models = [m for m in MODEL_ORDER if any((m, ds) in data_cur or (m, ds) in data_den for ds in DATASET_ORDER)]
    for model in all_models:
        tau = latency_ms_from_model(model)
        for ds in DATASET_ORDER:
            key = (model, ds)
            cur_dict = data_cur.get(key, {})
            den_dict = data_den.get(key, {})
            if not cur_dict and not den_dict:
                continue
            n_cur = len(seeds_cur.get(key, []))
            n_den = len(seeds_den.get(key, []))

            def _delta(d):
                enh_seeds = np.asarray(d.get("dnsmos_ovrl_enh", []), dtype=np.float64)
                ny_seeds = np.asarray(d.get("dnsmos_ovrl_noisy", []), dtype=np.float64)
                if enh_seeds.size == 0 or ny_seeds.size == 0:
                    return float("nan"), float("nan"), 0
                # Pair seed means, since both come from the same per-seed JSON.
                n = min(enh_seeds.size, ny_seeds.size)
                deltas = enh_seeds[:n] - ny_seeds[:n]
                m = float(deltas.mean())
                s = float(deltas.std(ddof=1) if n > 1 else 0.0)
                return m, s, n

            cur_m, cur_s, _ = _delta(cur_dict)
            den_m, den_s, _ = _delta(den_dict)
            shift = den_m - cur_m if (not np.isnan(cur_m) and not np.isnan(den_m)) else float("nan")
            row = [
                model,
                f"{tau:.1f}",
                DATASET_LABEL[ds],
                f"{n_cur}/{n_den}",
                f"{cur_m:+.3f}+/-{cur_s:.3f}" if not np.isnan(cur_m) else "-",
                f"{den_m:+.3f}+/-{den_s:.3f}" if not np.isnan(den_m) else "-",
                f"{shift:+.3f}" if not np.isnan(shift) else "-",
            ]
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def render_csv(data: Dict[Tuple[str, str], Dict[str, List[float]]],
                seeds_seen: Dict[Tuple[str, str], List[str]],
                out_path: Path,
                policy: str = "current") -> None:
    metrics_all = INTRUSIVE + DNSMOS_ENH + DNSMOS_NOISY
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["policy", "model", "tau_ms", "dataset", "n_seeds"]
        for m in metrics_all:
            header += [f"{m}_mean", f"{m}_std"]
        w.writerow(header)
        for dataset in DATASET_ORDER:
            for model in MODEL_ORDER:
                key = (model, dataset)
                if key not in data:
                    continue
                tau = latency_ms_from_model(model)
                row = [policy, model, f"{tau:.1f}", dataset, len(seeds_seen.get(key, []))]
                for metric in metrics_all:
                    mean, std, _ = aggregate(data[key].get(metric, []))
                    row += [
                        f"{mean:.6f}" if not np.isnan(mean) else "",
                        f"{std:.6f}" if not np.isnan(std) else "",
                    ]
                w.writerow(row)


def render_csv_dual(
    data_cur: Dict[Tuple[str, str], Dict[str, List[float]]],
    data_den: Dict[Tuple[str, str], Dict[str, List[float]]],
    seeds_cur: Dict[Tuple[str, str], List[str]],
    seeds_den: Dict[Tuple[str, str], List[str]],
    out_path: Path,
) -> None:
    """Write a wide CSV with both policies stacked, plus shift columns."""
    metrics_all = INTRUSIVE + DNSMOS_ENH + DNSMOS_NOISY
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["policy", "model", "tau_ms", "dataset", "n_seeds"]
        for m in metrics_all:
            header += [f"{m}_mean", f"{m}_std"]
        header += ["delta_dnsmos_ovrl_mean", "delta_dnsmos_ovrl_std"]
        w.writerow(header)
        for policy_name, data, seeds in (("current", data_cur, seeds_cur), ("denorm", data_den, seeds_den)):
            for dataset in DATASET_ORDER:
                for model in MODEL_ORDER:
                    key = (model, dataset)
                    if key not in data:
                        continue
                    tau = latency_ms_from_model(model)
                    row = [policy_name, model, f"{tau:.1f}", dataset, len(seeds.get(key, []))]
                    for metric in metrics_all:
                        mean, std, _ = aggregate(data[key].get(metric, []))
                        row += [
                            f"{mean:.6f}" if not np.isnan(mean) else "",
                            f"{std:.6f}" if not np.isnan(std) else "",
                        ]
                    enh_seeds = np.asarray(data[key].get("dnsmos_ovrl_enh", []), dtype=np.float64)
                    ny_seeds = np.asarray(data[key].get("dnsmos_ovrl_noisy", []), dtype=np.float64)
                    if enh_seeds.size and ny_seeds.size:
                        n = min(enh_seeds.size, ny_seeds.size)
                        d = enh_seeds[:n] - ny_seeds[:n]
                        d_mean = float(d.mean())
                        d_std = float(d.std(ddof=1) if n > 1 else 0.0)
                        row += [f"{d_mean:.6f}", f"{d_std:.6f}"]
                    else:
                        row += ["", ""]
                    w.writerow(row)


def render_json(data: Dict[Tuple[str, str], Dict[str, List[float]]],
                 seeds_seen: Dict[Tuple[str, str], List[str]],
                 out_path: Path) -> None:
    metrics_all = INTRUSIVE + DNSMOS_ENH + DNSMOS_NOISY
    out = {}
    for (model, dataset), d in data.items():
        rec = {"n_seeds": len(seeds_seen.get((model, dataset), [])), "metrics": {}}
        for metric in metrics_all:
            mean, std, n = aggregate(d.get(metric, []))
            rec["metrics"][metric] = {"mean": mean, "std": std, "n": n}
        out.setdefault(model, {})[dataset] = rec
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO_ROOT / "results/rebuttal/dns_eval"),
                    help="current scale results dir")
    ap.add_argument("--root-denorm", default=None,
                    help="denorm scale results dir; if set, render side-by-side comparison")
    ap.add_argument("--md-out", default=None)
    ap.add_argument("--csv-out", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    data, seeds_seen = collect(root)

    if args.root_denorm is None:
        md = render_markdown(data, seeds_seen)
        if args.md_out:
            Path(args.md_out).write_text(md)
            print(f"wrote {args.md_out}")
        else:
            print(md)
        if args.csv_out:
            render_csv(data, seeds_seen, Path(args.csv_out))
            print(f"wrote {args.csv_out}")
        if args.json_out:
            render_json(data, seeds_seen, Path(args.json_out))
            print(f"wrote {args.json_out}")
        return

    root_den = Path(args.root_denorm)
    data_den, seeds_den = collect(root_den)

    parts: List[str] = []
    parts.append("# DNS Cross-Dataset Evaluation - Current vs Denorm DNSMOS Policy\n")
    parts.append(f"- current root: `{root}` ({sum(len(v) for v in seeds_seen.values())} JSONs)\n")
    parts.append(f"- denorm root:  `{root_den}` ({sum(len(v) for v in seeds_den.values())} JSONs)\n")
    parts.append("")
    parts.append(render_comparison(data, data_den, seeds_seen, seeds_den))
    parts.append("\n---\n")
    parts.append(render_markdown(data, seeds_seen, title="Current scale (model-input scale, baseline sweep)"))
    parts.append("\n---\n")
    parts.append(render_markdown(data_den, seeds_den, title="Denorm scale (raw waveform scale)"))
    md = "\n".join(parts)
    if args.md_out:
        Path(args.md_out).write_text(md)
        print(f"wrote {args.md_out}")
    else:
        print(md)
    if args.csv_out:
        render_csv_dual(data, data_den, seeds_seen, seeds_den, Path(args.csv_out))
        print(f"wrote {args.csv_out}")
    if args.json_out:
        # Dual-policy JSON: nest under top-level "current" / "denorm" keys.
        out = {
            "current": {},
            "denorm": {},
        }
        metrics_all = INTRUSIVE + DNSMOS_ENH + DNSMOS_NOISY
        for policy_name, data_p, seeds_p in (("current", data, seeds_seen), ("denorm", data_den, seeds_den)):
            for (model, dataset), d in data_p.items():
                rec = {"n_seeds": len(seeds_p.get((model, dataset), [])), "metrics": {}}
                for metric in metrics_all:
                    mean, std, n = aggregate(d.get(metric, []))
                    rec["metrics"][metric] = {"mean": mean, "std": std, "n": n}
                out[policy_name].setdefault(model, {})[dataset] = rec
        Path(args.json_out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
