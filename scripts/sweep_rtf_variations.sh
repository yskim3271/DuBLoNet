#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

CHKPT_DIR="results/experiments/M5_62.5ms/s7"
CHKPT_FILE="auto"
OUT_DIR="results/rtf/M5_variations"
CHUNK_SIZES=(1 8 16 32)

# Measurement conditions (from design doc Section 7)
WARMUP=3
REPEATS=3
DURATION=2.0
NUM_THREADS=1

mkdir -p "$OUT_DIR"

total=0
for rf in 0 1; do
  for cpu in 0 1; do
    for cs in "${CHUNK_SIZES[@]}"; do
      total=$((total + 1))
    done
  done
done
echo "Total runs: $total"

run=0
for rf in 0 1; do
  for cpu in 0 1; do
    for cs in "${CHUNK_SIZES[@]}"; do
      run=$((run + 1))
      tag="rf${rf}_cpu${cpu}_cs${cs}"
      out_json="${OUT_DIR}/${tag}.json"

      # Build flags
      flags=""
      if [ "$rf" -eq 0 ]; then
        flags="$flags --no_reshape_free"
      fi
      if [ "$cpu" -eq 1 ]; then
        flags="$flags --optimize_cpu"
      fi

      echo "=== [$run/$total] $tag ==="
      python -m src.measure_rtf \
        --chkpt_dir "$CHKPT_DIR" \
        --chkpt_file "$CHKPT_FILE" \
        --chunk_size "$cs" \
        --num_threads "$NUM_THREADS" \
        --warmup "$WARMUP" \
        --repeats "$REPEATS" \
        --duration "$DURATION" \
        $flags \
        --output_json "$out_json"
      echo ""
    done
  done
done

echo "All done. Results in $OUT_DIR/"
