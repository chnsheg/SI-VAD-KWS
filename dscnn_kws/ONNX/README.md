# DSCNN / Bandpass ONNX 导出

本目录用于把 sweep 得到的 `best.pt` 批量导出为 ONNX。

## 当前默认配置

`export_sweep_best_to_onnx.py` 文件顶部有一组 `DEFAULT_*` 配置，可以直接在代码中修改。

当前默认是批量导出 **bandpass/Conv1D + PWL log 前端的完整模型**：

```text
export_mode = full
frontend = bandpass
dct_coeff = 10
bandpass_n_bands = 10
log_approx_mode = pwl
log_pwl_num_segments = 6
sample_rate = 16000
window_size_ms = 32
window_stride_ms = 32
```

默认输入目录：

```text
dscnn_kws/runs/bandpass_pwl_snr_scene_arch_sweep_best_models
```

默认输出目录：

```text
dscnn_kws/ONNX/models_bandpass_pwl_full
```

## 批量导出 bandpass PWL full ONNX

在项目根目录运行：

```bash
python dscnn_kws/ONNX/export_sweep_best_to_onnx.py
```

默认 ONNX 输入：

```text
waveform: [batch, 16000]
```

默认 ONNX 输出：

```text
logits: [batch, 2]
```

## 用命令行临时覆盖

指定输入目录：

```bash
python dscnn_kws/ONNX/export_sweep_best_to_onnx.py \
  --input_dir dscnn_kws/runs/bandpass_pwl_snr_scene_arch_sweep_best_models
```

指定单个模型：

```bash
python dscnn_kws/ONNX/export_sweep_best_to_onnx.py \
  --checkpoints dscnn_kws/runs/bandpass_pwl_snr_scene_arch_sweep_best_models/your_model.pt
```

显式写出 bandpass full 参数：

```bash
python dscnn_kws/ONNX/export_sweep_best_to_onnx.py \
  --export_mode full \
  --frontend bandpass \
  --dct_coeff 10 \
  --bandpass_n_bands 10 \
  --bandpass_f_min 200.0 \
  --bandpass_f_max 4000.0 \
  --bandpass_spacing log \
  --bandpass_kernel_size 63 \
  --bandpass_phase_count 1 \
  --log_approx_mode pwl \
  --log_pwl_num_segments 6 \
  --log_pwl_strategy uniform_logx \
  --log_pwl_gamma 1.0
```

## 导出 backbone ONNX

如果只想导出后端 DSCNN：

```bash
python dscnn_kws/ONNX/export_sweep_best_to_onnx.py \
  --export_mode backbone \
  --frontend bandpass
```

backbone 模式输入是已经算好的特征：

```text
features: [batch, 320]
```

其中 `320 = 32 time_steps * 10 bands`。

## 注意

bandpass PWL full 导出不使用 `torch.stft`，也不使用精确 `Log`，比 MFCC full ONNX 更适合部署。

bandpass 模式必须满足：

```text
dct_coeff == bandpass_n_bands
```

否则脚本会提前报错。
