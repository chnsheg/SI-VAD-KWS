# Runtime demo

`dscnn_kws.demo` provides the local microphone/WAV runner, VAD-KWS cascade, diagnostics,
telemetry and optimization CLI. It is intentionally kept independent from the training data;
provide model paths and manifests from the command line.

Install the demo dependencies and list available input devices:

```powershell
python -m pip install -r dscnn_kws/demo/requirements.txt
python -m dscnn_kws.demo.run_demo devices
```

For a local WAV or microphone run, pass the VAD and KWS model paths explicitly:

```powershell
python -m dscnn_kws.demo.run_demo listen `
  --vad-model path/to/vad.onnx `
  --kws-model checkpoints/mobvoi_nihao_wenwen_binary_hardneg_L5_C64_c11_seed42_int8_qdq.onnx
```

The runtime supports streaming MFCC extraction, VAD gating, KWS scoring, stateful history and
JSON telemetry. Model input/output contracts are documented beside the curated checkpoints.
