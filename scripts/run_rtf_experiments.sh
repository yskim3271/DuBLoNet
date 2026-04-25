#!/bin/bash
# RTF measurement experiments for tracked latency models
# Section 4.1: cs=1 (fair comparison)
# Section 4.2: cs=total_la (natural operating point)
set -euo pipefail

cd "$(dirname "$0")/.."

SEED=${SEED:-s2039}
WARMUP=3
REPEATS=3
DURATION=2.0
THREADS=1
OUTDIR=results/rtf

# Model definitions: NAME CHKPT_FILE TOTAL_LA
# total_la = 2*R, chunk_sizes = [1, total_la]
declare -a MODELS=(
  "M1_12.5ms    auto  0"
  "M2_25.0ms    auto  2"
  "M3_37.5ms    auto  4"
  "M4_50.0ms    auto  6"
  "M5_62.5ms    auto  8"
  "M6_75.0ms    auto  10"
  "M9_100.0ms   auto  14"
  "M13_150.0ms  auto  22"
  "M7_200.0ms   auto  30"
)

echo "=== RTF Experiments: ${#MODELS[@]} models ==="
echo "Warmup=$WARMUP, Repeats=$REPEATS, Duration=${DURATION}s, Threads=$THREADS"
echo ""

for entry in "${MODELS[@]}"; do
  read -r NAME CKPT TLA <<< "$entry"
  DIR="results/experiments/${NAME}/${SEED}"

  # --- Section 4.1: cs=1 ---
  echo "[${NAME}] cs=1 ..."
  python -m src.measure_rtf \
    --chkpt_dir "$DIR" \
    --chkpt_file "$CKPT" \
    --use_onnx \
    --chunk_size 1 \
    --num_threads $THREADS \
    --warmup $WARMUP --repeats $REPEATS \
    --duration $DURATION \
    --output_json "${OUTDIR}/${NAME}_${SEED}_cs1.json"

  # --- Section 4.2: cs=total_la (skip if total_la=0) ---
  if [ "$TLA" -gt 0 ]; then
    echo "[${NAME}] cs=${TLA} ..."
    python -m src.measure_rtf \
      --chkpt_dir "$DIR" \
      --chkpt_file "$CKPT" \
      --use_onnx \
      --chunk_size "$TLA" \
      --num_threads $THREADS \
      --warmup $WARMUP --repeats $REPEATS \
      --duration $DURATION \
      --output_json "${OUTDIR}/${NAME}_${SEED}_cs${TLA}.json"
  fi

  echo ""
done

echo "=== Done. Results in ${OUTDIR}/ ==="
