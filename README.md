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
