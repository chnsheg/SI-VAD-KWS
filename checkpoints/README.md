# Deployment checkpoints

This directory contains curated SI-dscnn-kws v6.1 deployment artifacts.

Full-precision PyTorch checkpoints:

- `mobvoi_nihao_wenwen_v2_final_L5_C64_v6_1_fp32.pt`
- `mobvoi_nihao_wenwen_v3_final_L5_C64_v6_1_fp32.pt`
- `mobvoi_nihao_wenwen_v419_a_reclean_v2_fp32.pt` (current reclean-v2 delivery candidate)

The first two FP32 files use the v6.1 `{"state_dict": ...}` container contract. Their L5/C64
global-pooling backbones load strictly into the v6.1 32-frame, 10-coefficient model without
key conversion. They contain model state only; optimizer, scheduler, training history, and
temporary experiment state are excluded.

`mobvoi_nihao_wenwen_v419_a_reclean_v2_fp32.pt` is the current delivery candidate of the
reclean-v2 CE line. It uses the `{"model": ...}` container with 56 `backbone.*` entries,
the TorchMFCC triangular frontend (13 coefficients x 32 frames), and is exported through
`dscnn_kws/deploy_transfer/export_float_onnx.py`. Its measured deployment status, training
lineage, and known limitations are recorded in its `.metadata.json`.

Quantized ONNX deployment artifacts:

- `mobvoi_hi_xiaowen_binary_hardneg_L5_C64_c11_seed42_int8_qdq.onnx`
- `mobvoi_nihao_wenwen_binary_hardneg_L5_C64_c11_seed42_int8_qdq.onnx`

Each artifact has a matching `.metadata.json` file containing the SHA-256 digest, input/output
contract, architecture, source checkpoint provenance, and the recorded ONNX Runtime parity
scope where applicable. The ONNX models expect mono, normalized 16 kHz, one-second waveforms with shape `[1, 16000]`
and return two logits (`positive` is class index `0`).

These are deployment checkpoints, not training datasets. Re-training requires the original
dataset and manifests, which are intentionally excluded from this public repository.
