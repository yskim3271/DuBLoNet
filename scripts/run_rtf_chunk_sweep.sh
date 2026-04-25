#!/bin/bash
# RTF Chunk Size Sweep — ONNX backend
# Measures RTF across dense chunk_size points for M1, M6, M7
# to reveal overhead amortization curves.
#
# Usage: bash scripts/run_rtf_chunk_sweep.sh

set -euo pipefail
cd "$(dirname "$0")/.."

OUTDIR="results/rtf/chunk_sweep_steady_state"
mkdir -p "$OUTDIR"

WARMUP=3
REPEATS=5
DURATION=2.0
NUM_THREADS=1

# Dense chunk_size points: 1-16 by 1, then 20,24,32,48,64
CHUNK_SIZES=(1 2 3 4 5 6 7 8 10 12 14 16 20 24 32 48 64)

# Models: (name, chkpt_dir, chkpt_file)
declare -A MODELS
MODELS[M1]="results/experiments/M1_12.5ms/s2039:model_279000.th"
MODELS[M6]="results/experiments/M6_75.0ms/s2039:model_163000.th"
MODELS[M7]="results/experiments/M7_200.0ms/s2039:model_184000.th"
MODEL_ORDER=(M1 M6 M7)

TOTAL=$(( ${#MODEL_ORDER[@]} * ${#CHUNK_SIZES[@]} ))
COUNT=0

for MODEL_NAME in "${MODEL_ORDER[@]}"; do
    IFS=: read -r CHKPT_DIR CHKPT_FILE <<< "${MODELS[$MODEL_NAME]}"

    for CS in "${CHUNK_SIZES[@]}"; do
        COUNT=$((COUNT + 1))
        OUTFILE="${OUTDIR}/${MODEL_NAME}_cs${CS}.json"

        # Skip if already measured
        if [ -f "$OUTFILE" ]; then
            echo "[${COUNT}/${TOTAL}] SKIP ${MODEL_NAME} cs=${CS} (exists)"
            continue
        fi

        echo "[${COUNT}/${TOTAL}] ${MODEL_NAME} cs=${CS} ..."
        python3 -m src.measure_rtf \
            --chkpt_dir "$CHKPT_DIR" \
            --chkpt_file "$CHKPT_FILE" \
            --use_onnx \
            --chunk_size "$CS" \
            --warmup "$WARMUP" \
            --repeats "$REPEATS" \
            --duration "$DURATION" \
            --num_threads "$NUM_THREADS" \
            --output_json "$OUTFILE"

        echo ""
    done
done

echo "=== Done. Results in ${OUTDIR}/ ==="
echo "Files:"
ls -1 "${OUTDIR}"/*.json 2>/dev/null | wc -l
echo "json files generated."
