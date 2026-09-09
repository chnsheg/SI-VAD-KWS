# Deployment checkpoints

This directory contains the two curated SI-dscnn-kws v6.1 ONNX QDQ deployment artifacts:

- `mobvoi_hi_xiaowen_binary_hardneg_L5_C64_c11_seed42_int8_qdq.onnx`
- `mobvoi_nihao_wenwen_binary_hardneg_L5_C64_c11_seed42_int8_qdq.onnx`

Each model has a matching `.metadata.json` file containing the SHA-256 digest, input/output
contract, architecture, source checkpoint provenance, and the recorded ONNX Runtime parity
scope. The models expect mono, normalized 16 kHz, one-second waveforms with shape `[1, 16000]`
and return two logits (`positive` is class index `0`).

These are deployment checkpoints, not training datasets. Re-training requires the original
dataset and manifests, which are intentionally excluded from this public repository.
