#!/usr/bin/env bash
# Download DNS Challenge 2020 non-blind test set (synthetic/no_reverb).
# Pulls only the test_set directory via sparse git-lfs, not the full ~1 TB corpus.
#
# Usage:
#   bash scripts/download_dns_testset.sh <dest_dir>
#
# Output structure after success:
#   <dest_dir>/DNS-Challenge/datasets/test_set/synthetic/no_reverb/
#     ├── noisy/   (150 wavs, clnsp..._fileid_<N>.wav)
#     └── clean/   (150 wavs, clean_fileid_<N>.wav)
#
# Notes:
#   - Requires git, git-lfs.
#   - Total download size ≈ 200–400 MB (test_set subset only).
#   - First run takes ~5–15 min depending on bandwidth.

set -euo pipefail

DEST="${1:-data/dns2020}"
REPO_URL="https://github.com/microsoft/DNS-Challenge.git"
REPO_DIR="$DEST/DNS-Challenge"
BRANCH="interspeech2020/master"

if ! command -v git-lfs >/dev/null 2>&1; then
    echo "ERROR: git-lfs not found. Install with: sudo apt install git-lfs && git lfs install" >&2
    exit 1
fi

mkdir -p "$DEST"

if [ ! -d "$REPO_DIR/.git" ]; then
    echo "[1/4] Cloning DNS-Challenge repo (sparse, no-checkout)..."
    git clone --filter=blob:none --no-checkout "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
    git sparse-checkout init --cone
    git sparse-checkout set datasets/test_set/synthetic/no_reverb
else
    echo "[1/4] Repo already exists at $REPO_DIR, skipping clone."
    cd "$REPO_DIR"
fi

echo "[2/4] git-lfs install..."
git lfs install --local

echo "[3/4] Checking out $BRANCH sparse paths (this pulls LFS blobs)..."
git fetch origin "$BRANCH"
git checkout "$BRANCH"

echo "[4/4] Verifying..."
NO_REVERB="$REPO_DIR/datasets/test_set/synthetic/no_reverb"
if [ ! -d "$NO_REVERB" ]; then
    echo "ERROR: expected $NO_REVERB not found." >&2
    exit 1
fi

NOISY_COUNT=$(find "$NO_REVERB/noisy" -name "*.wav" 2>/dev/null | wc -l)
CLEAN_COUNT=$(find "$NO_REVERB/clean" -name "*.wav" 2>/dev/null | wc -l)
echo "  noisy wavs: $NOISY_COUNT"
echo "  clean wavs: $CLEAN_COUNT"

if [ "$NOISY_COUNT" -lt 100 ] || [ "$CLEAN_COUNT" -lt 100 ]; then
    echo "WARNING: expected ≥150 wavs each. If counts are low, ensure git-lfs pulled blobs." >&2
    echo "         Try: cd $REPO_DIR && git lfs pull --include='datasets/test_set/synthetic/no_reverb/**'" >&2
fi

echo
echo "DONE. Data located at:"
echo "  $NO_REVERB"
echo
echo "Next: run scripts/eval_dns.sh <chkpt_dir> <chkpt_file> $NO_REVERB"
