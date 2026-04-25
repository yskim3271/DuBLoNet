#!/usr/bin/env bash
# Sweep DNS-style evaluation (PESQ/STOI/CSIG/CBAK/COVL + DNSMOS P.835) over all
# (model, seed) combinations on both VoiceBank+DEMAND test (D) and DNS Challenge
# 2020 synthetic no_reverb test (C). Resume-friendly: skips runs whose JSON
# already exists.
#
# Inputs (assumed):
#   results/experiments/<MODEL>/<SEED>/{.hydra/config.yaml, trainer.log, model_<step>.th}
#   data/voicebank_test_wav/{noisy,clean}
#   data/dns2020/DNS-Challenge/datasets/test_set/synthetic/no_reverb/{noisy,clean}
#
# Output:
#   results/rebuttal/dns_eval/D_<MODEL>_<SEED>_voicebank.{json,log}
#   results/rebuttal/dns_eval/C_<MODEL>_<SEED>_dns2020.{json,log}

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXP="$ROOT/results/experiments"
VB_NOISY="$ROOT/data/voicebank_test_wav/noisy"
VB_CLEAN="$ROOT/data/voicebank_test_wav/clean"
DNS_BASE="$ROOT/data/dns2020/DNS-Challenge/datasets/test_set/synthetic/no_reverb"
OUT="$ROOT/results/rebuttal/dns_eval"
mkdir -p "$OUT"

MODELS=(M1_12.5ms M2_25.0ms M3_37.5ms M4_50.0ms M5_62.5ms M6_75.0ms M9_100.0ms M13_150.0ms M7_200.0ms)
SEEDS=(s42 s7 s2039)

cd "$ROOT"

python - <<'PY'
try:
    from speechmos import dnsmos  # noqa: F401
except Exception as exc:
    raise SystemExit(
        "ERROR: speechmos is required for DNSMOS evaluation. "
        "Install with `python -m pip install speechmos==0.0.1.1`. "
        f"Import error: {exc}"
    )
PY

run_one() {
    local kind="$1" model="$2" seed="$3" dir="$4" ckpt="$5"
    local out_json="$6" out_log="$7" noisy_dir="$8" clean_dir="$9" layout="${10}"

    if [ -f "$out_json" ]; then
        echo "[$kind] $model/$seed → SKIP (already done: $out_json)"
        return 0
    fi
    echo "[$kind] $model/$seed → RUN ($ckpt)"
    python -m src.evaluate_dns \
        --model_config "$dir/.hydra/config.yaml" \
        --chkpt_dir    "$dir" \
        --chkpt_file   "$ckpt" \
        --noisy_dir    "$noisy_dir" \
        --clean_dir    "$clean_dir" \
        --layout       "$layout" \
        --out_json     "$out_json" \
        --log_file     "$out_log"
}

START=$(date +%s)
TOTAL=0; DONE=0; FAILED=0
for M in "${MODELS[@]}"; do
    for S in "${SEEDS[@]}"; do
        DIR="$EXP/$M/$S"
        if [ ! -d "$DIR" ]; then echo "SKIP missing dir: $DIR"; continue; fi
        BEST=$(grep "Best Model" "$DIR/trainer.log" 2>/dev/null | tail -1 | grep -oP "Steps:\s*\K\d+" || true)
        if [ -z "$BEST" ]; then echo "SKIP no Best Model line: $DIR"; continue; fi
        CKPT="model_${BEST}.th"
        if [ ! -f "$DIR/$CKPT" ]; then echo "SKIP missing ckpt: $DIR/$CKPT"; continue; fi

        # D: VoiceBank
        TOTAL=$((TOTAL+1))
        if run_one D "$M" "$S" "$DIR" "$CKPT" \
            "$OUT/D_${M}_${S}_voicebank.json" \
            "$OUT/D_${M}_${S}_voicebank.log" \
            "$VB_NOISY" "$VB_CLEAN" "simple"; then
            DONE=$((DONE+1))
        else
            FAILED=$((FAILED+1))
        fi

        # C: DNS 2020 no_reverb
        TOTAL=$((TOTAL+1))
        if run_one C "$M" "$S" "$DIR" "$CKPT" \
            "$OUT/C_${M}_${S}_dns2020.json" \
            "$OUT/C_${M}_${S}_dns2020.log" \
            "$DNS_BASE/noisy" "$DNS_BASE/clean" "dns2020"; then
            DONE=$((DONE+1))
        else
            FAILED=$((FAILED+1))
        fi
    done
done
END=$(date +%s)
ELAPSED=$((END - START))
printf "\nDone. total=%d done=%d failed=%d elapsed=%ds\n" "$TOTAL" "$DONE" "$FAILED" "$ELAPSED"
