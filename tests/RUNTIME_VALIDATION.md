# LaCoSENet v2 runtime validation manifest

The local development machine intentionally has no PyTorch/CUDA project
runtime.  The following checks are specified but were not executed locally.
They must pass on the training server before any quality or efficiency claim.

## P0 — tensor correctness

- Instantiate both `lacosenet_v2_sfi_bn` and
  `lacosenet_v2_sfi_channel_ln`.
- Run forward and backward for all 7 rates x LA0/1/2/3.
- Assert finite outputs, losses, and gradients.
- Assert that only the selected decoder branch receives gradients.
- Assert that shared encoder and TS-core parameters receive gradients.
- Assert output frequency bins match the unpadded input, including
  22.05 kHz `442 -> 443 -> 442`.
- Assert the parameter count is identical for every active sample rate.

## P0 — data and optimization

- Check noisy/clean resampling preserves equal length and zero relative delay.
- Check one 28-step schedule cycle visits each operating point exactly once.
- Save and resume a checkpoint mid-cycle; compare subsequent operating points.
- Run a short generator-only overfit and confirm loss decreases in all cells.
- Do not enable MetricGAN/PESQ for non-16-kHz steps until the objective is
  replaced by a rate-aware implementation.

## P1 — streaming and export

- Compare full-sequence and streaming output at chunk sizes 1, 2, 4, and 8.
- Verify algorithmic latencies are 20/40/60/80 ms.
- Verify stream reset between utterances.
- Verify BatchNorm folding before/after maximum absolute error.
- Export one static artifact per rate and latency (28 initial artifacts).
- Compare PyTorch and ONNX Runtime output for every artifact.

## P1 — resource and research validation

- Measure peak training and inference memory, especially at 48 kHz.
- Measure CPU/GPU RTF and target-device online latency.
- Run the full normalization A/B with identical data order and seeds.
- Run URGENT evaluation and rate/latency specialists.
