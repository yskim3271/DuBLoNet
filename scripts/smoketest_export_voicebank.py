"""Export VoiceBank+DEMAND test samples to paired noisy/clean wav dirs.

Used by ``src/evaluate_dns.py`` to reuse the DNS-style paired pipeline on
VoiceBank for rebuttal experiment D (DNSMOS on in-distribution test data).

Usage:
    # Full test set (824 pairs, ~500 MB):
    python scripts/smoketest_export_voicebank.py --n -1 --dest data/voicebank_full
    # First 5 pairs (smoke test):
    python scripts/smoketest_export_voicebank.py --n 5 --dest data/voicebank_smoketest
"""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="Number of pairs to export; -1 for all.")
    ap.add_argument("--dest", type=str, default="data/voicebank_smoketest")
    ap.add_argument("--dataset", type=str, default="JacobLinCool/VoiceBank-DEMAND-16k")
    ap.add_argument("--skip_existing", action="store_true", help="Skip pairs already present at dest.")
    args = ap.parse_args()

    noisy_dir = Path(args.dest) / "noisy"
    clean_dir = Path(args.dest) / "clean"
    noisy_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, split="test")
    total = len(ds)
    n = total if args.n is None or args.n < 0 else min(args.n, total)

    written = skipped = 0
    for i in tqdm(range(n), desc="export", total=n):
        item = ds[i]
        sid = item["id"]
        np_path = noisy_dir / f"{sid}.wav"
        cp_path = clean_dir / f"{sid}.wav"
        if args.skip_existing and np_path.exists() and cp_path.exists():
            skipped += 1
            continue
        noisy = item["noisy"]["array"].astype("float32")
        clean = item["clean"]["array"].astype("float32")
        sf.write(str(np_path), noisy, 16000, subtype="PCM_16")
        sf.write(str(cp_path), clean, 16000, subtype="PCM_16")
        written += 1

    print(f"Done. written={written} skipped={skipped} (total requested={n} / dataset size={total})")
    print(f"  noisy: {noisy_dir}\n  clean: {clean_dir}")


if __name__ == "__main__":
    main()
