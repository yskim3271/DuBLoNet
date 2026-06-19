# LaCo-SENet

Latency-configurable streaming speech enhancement via asymmetric temporal
padding.

LaCo-SENet exposes a single padding-ratio hyperparameter that trades algorithmic
latency for enhancement quality while keeping the backbone architecture and
parameter budget fixed. The streaming implementation uses state buffers for past
context, lookahead buffers for future context, and selective state updates to
avoid future-frame leakage across chunks.

This repository contains the model, training pipeline, streaming wrappers, ONNX
export utilities, and evaluation entry points used for the LaCo-SENet
experiments. Paper-writing materials, local datasets, checkpoints, generated
results, and local utility scripts are not tracked in this code repository.

## VoiceBank+DEMAND Results

The table below summarizes the VoiceBank+DEMAND comparison reported in the
paper. Best causal results are shown in bold. Rows marked `(dagger)` have
indeterminate algorithmic latency because their reported unidirectional variants
still use non-causal symmetric encoder padding. The `(double-dagger)` row is the
symmetric-padding architecture upper bound and is excluded from causal ranking.

| Model | Year | tau (ms) | Params | PESQ | STOI | CSIG | CBAK | COVL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Noisy | - | - | - | 1.97 | .921 | 3.35 | 2.44 | 2.63 |
| RNNoise | 2018 | 10 | 0.06M | 2.33 | .922 | 3.40 | 2.51 | 2.84 |
| GaGNet | 2022 | ~10 | 5.94M | 2.94 | - | 4.26 | 3.45 | 3.59 |
| DFNet3 | 2023 | 40 | 2.13M | 3.17 | .944 | 4.34 | 3.61 | 3.77 |
| aTENNuate | 2025 | 46.5 | 0.84M | 3.27 | - | 4.57 | 2.85 | 3.96 |
| xLSTM-SENet (dagger) | 2025 | - | ~2.2M | 3.26 | .950 | 4.57 | 3.79 | 4.00 |
| SEMamba (dagger) | 2024 | - | 1.41M | 3.29 | .950 | - | - | - |
| LaCo-SENet (`L_enc=0`, `L_dec=0`) | 2026 | 12.5 | 1.37M | 3.35 +/- .02 | .952 +/- .000 | 4.61 +/- .01 | 3.71 +/- .01 | 4.05 +/- .02 |
| LaCo-SENet (`L_enc=1`, `L_dec=1`) | 2026 | 25.0 | 1.37M | 3.36 +/- .01 | .953 +/- .000 | 4.62 +/- .01 | 3.72 +/- .02 | 4.07 +/- .01 |
| LaCo-SENet (`L_enc=3`, `L_dec=3`) | 2026 | 50.0 | 1.37M | 3.40 +/- .02 | .953 +/- .001 | 4.63 +/- .02 | 3.72 +/- .01 | 4.09 +/- .02 |
| LaCo-SENet (`L_enc=5`, `L_dec=5`) | 2026 | 75.0 | 1.37M | **3.43 +/- .01** | **.954 +/- .001** | **4.66 +/- .02** | **3.78 +/- .00** | **4.12 +/- .02** |
| LaCo-SENet (double-dagger, `L_enc=15`, `L_dec=15`) | 2026 | 200.0 | 1.37M | 3.47 +/- .02 | .957 +/- .001 | 4.69 +/- .01 | 3.79 +/- .03 | 4.17 +/- .02 |
| PrimeK-Net | 2025 | - | 1.41M | 3.61 | - | 4.81 | 3.98 | 4.35 |

## Repository Layout

| Path | Purpose |
|------|---------|
| `src/` | Training, evaluation, STFT, metrics, and model implementation. |
| `src/models/streaming/` | Chunk-wise LaCo-SENet streaming modules and stateful layers. |
| `src/models/onnx_export/` | Exportable streaming core and ONNX Runtime wrapper. |
| `conf/config.yaml` | Default Hydra configuration for VoiceBank+DEMAND training. |
| `requirements.txt` | Python dependencies. |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The default training configuration uses the Hugging Face
`JacobLinCool/VoiceBank-DEMAND-16k` dataset.

## Training

```bash
python src/train.py
```

Hydra writes each run under `results/experiments/...` by default. Generated
checkpoints and result directories are intentionally ignored by Git.

To train a specific latency configuration, override the encoder and decoder
padding ratios:

```bash
python src/train.py \
  model.encoder_padding_ratio='[1.0,0.0]' \
  model.decoder_padding_ratio='[1.0,0.0]'
```

## Streaming RTF Measurement

```bash
python src/measure_rtf.py \
  --chkpt_dir /path/to/run \
  --chkpt_file auto \
  --chunk_size 8 \
  --use_onnx \
  --num_threads 1
```

`--chkpt_file auto` resolves the best model from the run metadata.

## Citation

```bibtex
@inproceedings{kim2026lacosenet,
  title     = {Latency-Configurable Streaming Speech Enhancement via Asymmetric Temporal Padding},
  author    = {Kim, Yunsik and Chung, Yoonyoung},
  booktitle = {Proc. Interspeech},
  year      = {2026}
}
```
