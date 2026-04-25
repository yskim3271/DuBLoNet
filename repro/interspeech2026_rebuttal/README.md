# Interspeech 2026 Rebuttal Reproduction

This directory contains the tracked reproduction scripts for the rebuttal DNS
cross-dataset evaluation. The project-level `scripts/`, `paper_works/`, and
`results/` directories are intentionally ignored because they are used for local
experiments, manuscript work, and large generated artifacts.

## Scope

The scripts here reproduce the small, reviewable part of the rebuttal workflow:

- export the VoiceBank+DEMAND test split to paired wav directories;
- download the DNS Challenge 2020 synthetic no-reverb test set from the
  `interspeech2020/master` branch;
- run the 9 latency configs x 3 seeds x 2 datasets evaluation sweeps
  for both current-scale and denormalized DNSMOS scoring;
- aggregate per-run JSON files into markdown, CSV, and JSON summaries.

Large files remain outside git:

- `data/voicebank_test_wav/`
- `data/dns2020/`
- `results/experiments/`
- `results/rebuttal/dns_eval/`
- `results/rebuttal/dns_eval_denorm/`

## Commands

Run from the repository root:

```bash
conda activate fullcomplex

python repro/interspeech2026_rebuttal/export_voicebank_test.py \
    --n -1 \
    --dest data/voicebank_test_wav \
    --skip_existing

bash repro/interspeech2026_rebuttal/download_dns_testset.sh data/dns2020

bash repro/interspeech2026_rebuttal/run_dns_eval_all.sh

bash repro/interspeech2026_rebuttal/run_dns_eval_all_denorm.sh

python repro/interspeech2026_rebuttal/aggregate_dns_eval.py \
    --md-out results/rebuttal/dns_eval/summary.md \
    --csv-out results/rebuttal/dns_eval/summary.csv \
    --json-out results/rebuttal/dns_eval/summary.json

python repro/interspeech2026_rebuttal/aggregate_dns_eval.py \
    --root results/rebuttal/dns_eval \
    --root-denorm results/rebuttal/dns_eval_denorm \
    --md-out results/rebuttal/dns_eval_denorm/summary_current_vs_denorm.md \
    --csv-out results/rebuttal/dns_eval_denorm/summary_current_vs_denorm.csv \
    --json-out results/rebuttal/dns_eval_denorm/summary_current_vs_denorm.json
```

The sweep expects trained checkpoints under `results/experiments/<MODEL>/<SEED>/`
and writes generated outputs under `results/rebuttal/dns_eval/` and
`results/rebuttal/dns_eval_denorm/`.

## DNSMOS Scale Check

To compare DNSMOS on the current normalized evaluation signals against DNSMOS
after undoing the model-input power normalization:

```bash
CUDA_VISIBLE_DEVICES=1 python repro/interspeech2026_rebuttal/check_dnsmos_scale.py \
    --n 30 \
    --device cuda
```

The output JSON is written to `results/rebuttal/dnsmos_scale_check/`.

The scale policies are:

- `current`: model input is `noisy_norm`; DNSMOS scores `noisy_norm` and
  `enhanced_norm`.
- `denorm`: model input is still `noisy_norm`; DNSMOS scores `noisy_raw` and
  `enhanced_norm / norm_factor`.
- raw model input is intentionally not used for rebuttal numbers because it
  changes the model input distribution.

The denormalized DNSMOS policy is preferred for final rebuttal reporting.
