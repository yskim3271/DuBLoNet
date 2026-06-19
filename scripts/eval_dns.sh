#!/usr/bin/env bash
# Wrapper to evaluate a LaCo-SENet checkpoint on a DNS Challenge 2020 test set.
#
# Usage:
#   bash scripts/eval_dns.sh <chkpt_dir> <chkpt_file> <no_reverb_dir> [out_json] [extra args...]
#
# Example:
#   bash scripts/eval_dns.sh \
#       path/to/checkpoint \
#       best.th \
#       data/dns2020/DNS-Challenge/datasets/test_set/synthetic/no_reverb \
#       results/dns_eval/example.json

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <chkpt_dir> <chkpt_file> <no_reverb_dir> [out_json] [extra args...]" >&2
    exit 1
fi

CHKPT_DIR="$1"
CHKPT_FILE="$2"
NO_REVERB_DIR="$3"
OUT_JSON="${4:-}"
shift 3
if [ -n "${OUT_JSON}" ]; then shift || true; fi

MODEL_CONFIG="$CHKPT_DIR/.hydra/config.yaml"
NOISY_DIR="$NO_REVERB_DIR/noisy"
CLEAN_DIR="$NO_REVERB_DIR/clean"

if [ ! -f "$MODEL_CONFIG" ]; then
    echo "ERROR: model config not found at $MODEL_CONFIG" >&2
    exit 1
fi
if [ ! -d "$NOISY_DIR" ] || [ ! -d "$CLEAN_DIR" ]; then
    echo "ERROR: $NOISY_DIR or $CLEAN_DIR missing." >&2
    exit 1
fi

OUT_JSON_ARG=()
if [ -n "$OUT_JSON" ]; then
    OUT_JSON_ARG=(--out_json "$OUT_JSON")
fi

python -m src.evaluate_dns \
    --model_config "$MODEL_CONFIG" \
    --chkpt_dir    "$CHKPT_DIR" \
    --chkpt_file   "$CHKPT_FILE" \
    --noisy_dir    "$NOISY_DIR" \
    --clean_dir    "$CLEAN_DIR" \
    --layout       dns2020 \
    "${OUT_JSON_ARG[@]}" \
    "$@"
