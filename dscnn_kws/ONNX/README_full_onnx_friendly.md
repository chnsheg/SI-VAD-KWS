# 完整 MFCC + DSCNN 的 ONNX-Friendly 导出详解

本文档专门说明 `dscnn_kws/ONNX/export_full_onnx_friendly.py` 这条导出路线。它的目标是把一个训练好的 DSCNN KWS 模型导出为完整的 ONNX 模型，让 ONNX 输入直接是原始音频波形：

```text
waveform [batch, sample_rate]
-> ONNX-friendly MFCC frontend
-> selected MFCC features
-> DSCNN backbone
-> logits [batch, num_classes]
```

这条路线解决的是传统 `torch.stft` / 复数张量在 ONNX 导出中不够友好的问题。它不是 bandpass 前端导出路线；bandpass 的默认批量导出说明在 `dscnn_kws/ONNX/README.md`，对应脚本是 `dscnn_kws/ONNX/export_sweep_best_to_onnx.py`。

---

## 1. 这份 README 讲的是哪条导出路线

本文件对应：

```text
dscnn_kws/ONNX/onnx_friendly_mfcc.py
dscnn_kws/ONNX/export_full_onnx_friendly.py
```

核心关键词是：

```text
MFCC full ONNX
waveform -> MFCC -> DSCNN -> logits
ONNX-friendly real-valued DFT
avoid torch.stft complex export issue
```

也就是说，这条路线仍然是 MFCC 前端，只是把 MFCC 的 STFT 实现方式换成了 ONNX 更容易接受的实数卷积实现。

传统训练中的 MFCC 路径大致是：

```text
waveform
-> pre-emphasis
-> torch.stft
-> complex spectrum
-> power spectrum
-> Mel filterbank
-> log / dB
-> DCT
-> selected MFCC coefficients
-> DSCNN
```

而本 ONNX-friendly 路径是：

```text
waveform
-> pre-emphasis
-> real Conv1D DFT kernels
-> imag Conv1D DFT kernels
-> real^2 + imag^2 power spectrum
-> Mel filterbank
-> log / dB
-> DCT
-> selected MFCC coefficients
-> DSCNN
```

替换点非常明确：

```text
把 torch.stft 替换成 real-valued conv1d DFT。
```

它保留 MFCC 的整体数学链路，但避免了 ONNX 导出时的复数 STFT 问题。

---

## 2. 为什么需要 ONNX-Friendly MFCC

PyTorch 中的 `torch.stft(..., return_complex=True)` 会产生复数张量。训练和 PyTorch 推理时这没有问题，但导出 ONNX 时，复数类型和部分 STFT 相关算子经常会成为障碍。

常见问题包括：

- ONNX exporter 对 `torch.stft` 支持不稳定。
- 复数张量在 ONNX 图中不如实数张量通用。
- 不同 ONNX Runtime / 推理框架对复数算子的支持差异较大。
- 嵌入式部署通常更希望看到 Conv、MatMul、Log、Add、Mul 这类基础实数算子。

因此本工程新增了 `ONNXFriendlyMFCC`，它不调用 `torch.stft`，而是把 DFT 的实部和虚部分别预先构造成固定的一维卷积核：

```text
real_kernel: cos basis * Hann window
imag_kernel: -sin basis * Hann window
```

前向时用两次 `F.conv1d` 得到实部和虚部：

```python
real = F.conv1d(x, real_kernel, stride=hop_length)
imag = F.conv1d(x, imag_kernel, stride=hop_length)
power_spec = real * real + imag * imag
```

这样 ONNX 图里看到的就是普通实数卷积和普通实数算子，导出和后续部署都更直接。

---

## 3. ONNX 目录下相关文件总览

`dscnn_kws/ONNX` 当前有几类文件：

```text
dscnn_kws/ONNX/
├─ onnx_friendly_mfcc.py
├─ export_full_onnx_friendly.py
├─ export_sweep_best_to_onnx.py
├─ README_full_onnx_friendly.md
└─ README.md
```

它们的职责不同：

| 文件 | 主要用途 | 前端路线 |
| --- | --- | --- |
| `onnx_friendly_mfcc.py` | 定义 ONNX 友好的 MFCC 前端和 MFCC+DSCNN wrapper | MFCC |
| `export_full_onnx_friendly.py` | 导出完整 `waveform -> logits` 的 MFCC full ONNX | MFCC |
| `export_sweep_best_to_onnx.py` | 导出 sweep best 模型，支持 `backbone` 或 `full`，默认偏 bandpass | MFCC 或 bandpass，默认 bandpass |
| `README_full_onnx_friendly.md` | 本文档，详细解释 MFCC full ONNX-friendly 路线 | MFCC |
| `README.md` | ONNX 目录通用说明，当前重点是 bandpass/PWL full 导出 | 默认 bandpass |

理解这几个文件时，最重要的是区分两条导出路线：

```text
路线 A: MFCC full ONNX-friendly
  onnx_friendly_mfcc.py
  export_full_onnx_friendly.py
  README_full_onnx_friendly.md

路线 B: sweep best / bandpass full ONNX
  export_sweep_best_to_onnx.py
  README.md
```

---

## 4. `onnx_friendly_mfcc.py` 文件详解

这个文件提供两个类：

```python
class ONNXFriendlyMFCC(nn.Module)
class ONNXFriendlyMFCCDSCNN(nn.Module)
```

第一个类只负责 MFCC 前端；第二个类把 MFCC 前端和 DSCNN backbone 包在一起。

---

## 5. `ONNXFriendlyMFCC` 的作用

`ONNXFriendlyMFCC` 是 MFCC 前端的 ONNX 友好复现版本。它的输入是原始波形：

```text
x: [B, T] 或 [B, 1, T]
```

输出是 MFCC 特征：

```text
mfcc: [B, n_mfcc, time]
```

默认构造时使用：

```text
n_mfcc = 40
n_mels = 40
f_min = 20.0
f_max = sample_rate / 2
center = True
mel_filter_shape = triangular
output_scale = torchaudio_db
```

它保留了 MFCC 的主要环节：

```text
pre-emphasis
-> windowed DFT
-> power spectrum
-> Mel filterbank
-> log / dB
-> DCT
```

但 DFT 的实现不是 `torch.stft`，而是 `F.conv1d`。

---

## 6. DFT 卷积核如何构造

`ONNXFriendlyMFCC._build_dft_kernels(...)` 做了这件事：

```python
freq = torch.arange(n_fft // 2 + 1).unsqueeze(1)
time = torch.arange(n_fft).unsqueeze(0)
phase = 2.0 * math.pi * freq * time / float(n_fft)
real = torch.cos(phase) * window.unsqueeze(0)
imag = -torch.sin(phase) * window.unsqueeze(0)
return real.unsqueeze(1), imag.unsqueeze(1)
```

这里的 `freq` 是频率 bin：

```text
0, 1, 2, ..., n_fft / 2
```

`time` 是窗口内的采样点：

```text
0, 1, 2, ..., n_fft - 1
```

对每个频率 bin，都构造一条 cos 基函数和一条 -sin 基函数。乘上 Hann window 后，就得到和 STFT 类似的窗函数加权 DFT 基。

卷积核形状是：

```text
real_kernel: [n_fft // 2 + 1, 1, n_fft]
imag_kernel: [n_fft // 2 + 1, 1, n_fft]
```

例如 `sample_rate=16000`、`window_size_ms=32` 时：

```text
n_fft = 16000 * 32 / 1000 = 512
n_freqs = 512 // 2 + 1 = 257

real_kernel shape = [257, 1, 512]
imag_kernel shape = [257, 1, 512]
```

前向时：

```python
real = F.conv1d(x, real_kernel, stride=hop_length)
imag = F.conv1d(x, imag_kernel, stride=hop_length)
```

这等价于用实数卷积的方式计算每一帧、每一个频率 bin 上的 DFT 实部和虚部。

---

## 7. pre-emphasis 在 ONNX-friendly MFCC 中怎么做

`ONNXFriendlyMFCC` 内部实现了 `_pre_emphasis(...)`：

```python
first = x[:, :1]
rest = x[:, 1:] - coeff * x[:, :-1]
return torch.cat([first, rest], dim=1)
```

对应公式：

```text
y[0] = x[0]
y[t] = x[t] - coeff * x[t - 1]
```

默认：

```text
coeff = 0.97
```

这里把 pre-emphasis 放进 ONNX-friendly 前端内部，是为了导出的完整 ONNX 能从原始 waveform 开始，不依赖外部 Python 预处理。

注意：训练脚本 `MFCCDSCNN` 中的 pre-emphasis 是在 wrapper 里统一做的；而这个 ONNX-friendly wrapper 是为了导出完整模型，所以把 pre-emphasis 也放进导出的 ONNX 图中。

---

## 8. center padding 如何处理

训练时 `torchaudio.transforms.MFCC` 的 MelSpectrogram 配置中使用：

```text
center = True
```

为了尽量贴近训练时行为，`ONNXFriendlyMFCC` 也支持 `center=True`。前向时：

```python
if self.center:
    pad = self.n_fft // 2
    x = F.pad(x, (pad, pad), mode="reflect")
```

这会在波形左右各补 `n_fft // 2` 个采样点。然后再用 Conv1D DFT kernel 按 `hop_length` 取帧。

为什么这会影响时间步数？因为 center padding 会让首尾也能形成居中的窗口，因此时间帧数通常是：

```text
floor(T / hop_length) + 1
```

这和本工程 `calculate_time_steps(sample_rate, window_stride_ms)` 的估计保持一致。

---

## 9. power spectrum 如何计算

通过实部和虚部卷积得到：

```python
real = F.conv1d(...)
imag = F.conv1d(...)
```

随后计算功率谱：

```python
power_spec = real * real + imag * imag
```

形状大致是：

```text
power_spec: [B, n_fft // 2 + 1, time]
```

例如：

```text
sample_rate = 16000
window_size_ms = 32
window_stride_ms = 32
n_fft = 512
hop_length = 512
n_freqs = 257
time_steps = 16000 // 512 + 1 = 32

power_spec = [B, 257, 32]
```

---

## 10. Mel filterbank 和 DCT 从哪里来

`onnx_friendly_mfcc.py` 复用了工程内 MFCC 实现中的工具函数：

```python
from dscnn_kws.frontend.mfcc_torch import create_dct_matrix, create_mel_filterbank
```

也就是说，Mel 滤波器组和 DCT 矩阵不是重新写一套独立逻辑，而是复用：

```text
dscnn_kws/frontend/mfcc_torch.py
```

构造时：

```python
mel_fb = create_mel_filterbank(
    sample_rate=sample_rate,
    n_fft=n_fft,
    n_mels=n_mels,
    f_min=f_min,
    f_max=f_max,
    filter_shape=mel_filter_shape,
)
dct_mat = create_dct_matrix(n_mfcc=n_mfcc, n_mels=n_mels, norm="ortho")
```

前向时：

```python
mel_spec = torch.matmul(self.mel_fb.to(dtype=x.dtype).unsqueeze(0), power_spec)
...
mfcc = torch.matmul(self.dct_mat.to(dtype=x.dtype).unsqueeze(0), log_mel)
```

形状变化：

```text
power_spec: [B, n_freqs, time]
mel_fb:     [n_mels, n_freqs]
mel_spec:  [B, n_mels, time]

dct_mat:   [n_mfcc, n_mels]
mfcc:      [B, n_mfcc, time]
```

---

## 11. `mfcc_scale` 的两种模式

导出脚本提供：

```text
--mfcc_scale {torchaudio_db,natural_log}
```

这对应 `ONNXFriendlyMFCC(output_scale=...)`。

### 11.1 `torchaudio_db`

默认值是：

```text
torchaudio_db
```

代码：

```python
log_mel = 10.0 * torch.log10(torch.clamp(mel_spec, min=1e-10))
max_per_item = torch.amax(log_mel, dim=(1, 2), keepdim=True)
log_mel = torch.maximum(log_mel, max_per_item - 80.0)
```

这条路径用于尽量贴近 `torchaudio.transforms.MFCC` 默认链路中的 dB 缩放。它有两个关键点：

- 使用 `10 * log10(mel_spec)`，不是自然对数。
- 做 top-dB 截断，把过低能量限制在 `max - 80 dB`。

如果训练时使用的是 `mfcc_impl=torchaudio`，通常应该使用默认的：

```bash
--mfcc_scale torchaudio_db
```

### 11.2 `natural_log`

代码：

```python
log_mel = torch.log(torch.clamp(mel_spec + log_offset, min=log_input_clamp_min))
```

这条路径更接近工程内 `TorchMFCC` 的自然 log 实现：

```text
log(mel_spec + offset)
```

如果训练时使用的是：

```bash
--mfcc_impl torch
--log_approx_mode exact
```

可以考虑导出时使用：

```bash
--mfcc_scale natural_log
```

注意：`export_full_onnx_friendly.py` 目前没有暴露 `log_offset` 和 `log_input_clamp_min` 命令行参数给 `ONNXFriendlyMFCCDSCNN`，`ONNXFriendlyMFCC` 内部默认是：

```text
log_offset = 1e-6
log_input_clamp_min = 1e-12
```

---

## 12. `ONNXFriendlyMFCCDSCNN` 的作用

`ONNXFriendlyMFCCDSCNN` 是完整导出 wrapper：

```text
ONNXFriendlyMFCCDSCNN
├─ feature_extractor: ONNXFriendlyMFCC
└─ backbone: DSCNN
```

构造时：

```python
self.feature_extractor = ONNXFriendlyMFCC(
    sample_rate=sample_rate,
    n_mfcc=40,
    n_fft=n_fft,
    win_length=n_fft,
    hop_length=hop_length,
    n_mels=40,
    f_min=20.0,
    f_max=sample_rate / 2,
    center=True,
    mel_filter_shape=mel_filter_shape,
    output_scale=mfcc_scale,
)
self.backbone = backbone
```

前向时：

```python
mfcc = self.feature_extractor(waveform, ...)
mfcc = mfcc[:, : self.dct_coeff, :]
features = mfcc.permute(0, 2, 1).reshape(mfcc.size(0), -1)
return self.backbone(features)
```

因此完整 ONNX 图的输入输出是：

```text
input:
  waveform [batch, sample_rate]

output:
  logits [batch, label_count]
```

默认 `sample_rate=16000`、`dct_coeff=10`、`window_stride_ms=32` 时：

```text
time_steps = 16000 // 512 + 1 = 32
selected MFCC = [B, 10, 32]
flatten features = [B, 320]
logits = [B, 2]
```

---

## 13. `export_full_onnx_friendly.py` 文件详解

这个脚本负责：

1. 找到一个或多个 checkpoint。
2. 读取 PyTorch `state_dict`。
3. 推断 DSCNN 的层数、通道数和类别数。
4. 构造 DSCNN backbone。
5. 把 backbone 包进 `ONNXFriendlyMFCCDSCNN`。
6. 调用 `torch.onnx.export(...)`。
7. 可选运行 ONNX checker。
8. 可选运行 ONNX Runtime 对比。
9. 写出 manifest CSV。

脚本入口：

```bash
python dscnn_kws/ONNX/export_full_onnx_friendly.py
```

---

## 14. checkpoint 如何发现

`discover_checkpoints(args)` 的逻辑：

```python
if args.checkpoints:
    paths = [Path(p) for p in args.checkpoints]
elif args.input_dir:
    paths = sorted(Path(args.input_dir).glob(args.pattern))
else:
    sweep_dir = REPO_ROOT / "dscnn_kws" / "runs" / "sweep_best_models"
    if sweep_dir.exists():
        paths = sorted(sweep_dir.glob("*.pt"))
    else:
        paths = sorted((REPO_ROOT / "dscnn_kws" / "runs").glob("**/best.pt"))
```

优先级：

1. `--checkpoints` 显式指定一个或多个 `.pt` 文件。
2. `--input_dir` 指定目录，配合 `--pattern` 搜索。
3. 默认搜索 `dscnn_kws/runs/sweep_best_models/*.pt`。
4. 如果没有 sweep_best_models，则回退搜索 `dscnn_kws/runs/**/best.pt`。

常用方式：

```bash
python dscnn_kws/ONNX/export_full_onnx_friendly.py \
  --checkpoints dscnn_kws/runs/your_run/best.pt
```

或：

```bash
python dscnn_kws/ONNX/export_full_onnx_friendly.py \
  --input_dir dscnn_kws/runs/sweep_best_models
```

---

## 15. checkpoint 权重如何加载

`load_state_dict(path)` 支持两类格式：

1. 文件本身就是 `state_dict`。
2. 文件是 checkpoint dict，里面有 `"state_dict"` 字段。

逻辑：

```python
obj = torch.load(path, map_location="cpu", weights_only=True)
if isinstance(obj, dict) and "state_dict" in obj:
    obj = obj["state_dict"]
```

同时会移除 DataParallel 产生的前缀：

```python
if key.startswith("module."):
    key = key[len("module."):]
```

这样脚本既能处理普通 `best.pt`，也能处理带 `module.` 前缀的权重。

---

## 16. DSCNN 架构如何推断

脚本有两种推断方式。

### 16.1 从文件名推断

正则：

```python
ARCH_RE = re.compile(r"L(?P<layers>\d+)_C(?P<channels>\d+)", re.IGNORECASE)
```

如果 checkpoint 文件名或父目录名包含：

```text
L5_C64
L3_C16
L1_C4
```

就能推断：

```text
num_layers = 5
channels = 64
```

### 16.2 从 state_dict 推断

脚本会扫描：

```text
conv_layers.N.
backbone.conv_layers.N.
conv_layers.0.0.weight
final_fc.weight
```

用于推断：

- DSCNN 卷积层数量。
- 第一层通道数。
- 输出类别数。

如果命令行显式传入：

```bash
--layers 5 --channels 64
```

则命令行参数优先。

---

## 17. backbone 如何重建

脚本使用：

```python
time_steps = calculate_time_steps(args.sample_rate, args.window_stride_ms)
backbone = DSCNN(
    input_dim=time_steps * args.dct_coeff,
    label_count=label_count,
    model_size_info=make_model_size_info(num_layers, channels),
    dct_coeff=args.dct_coeff,
)
```

`make_model_size_info(num_layers, channels)` 的规则：

```python
info = [num_layers]
info += [channels, 10, 4, 2, 2]
for _ in range(num_layers - 1):
    info += [channels, 3, 3, 1, 1]
```

也就是：

- 第 1 个 DSCNN 卷积层使用 kernel `(10, 4)`、stride `(2, 2)`。
- 后续 depthwise separable 层使用 kernel `(3, 3)`、stride `(1, 1)`。
- 所有层通道数相同，为 `channels`。

这适合本工程 sweep 脚本产生的 `Lx_Cy` 系列模型。

如果你的训练模型不是这套 `make_model_size_info(...)` 规则，导出时就不能只靠 `Lx_Cy` 推断，需要修改脚本或显式适配真实 `model_size_info`。

---

## 18. ONNX 导出时的 dummy 输入

`export_one(...)` 中：

```python
dummy = torch.randn(args.batch_size, args.sample_rate, dtype=torch.float32)
```

默认：

```text
batch_size = 1
sample_rate = 16000
dummy shape = [1, 16000]
```

因此导出的 full ONNX 默认输入是 1 秒音频：

```text
waveform: [batch, 16000]
```

如果设置：

```bash
--sample_rate 8000
```

则 dummy 输入变成：

```text
waveform: [batch, 8000]
```

注意：这里的 `sample_rate` 同时被当作 1 秒 waveform 的采样点数。所以该脚本默认导出的是“固定 1 秒输入长度”的 KWS ONNX。

---

## 19. dynamic batch

默认：

```text
--dynamic_batch
```

对应：

```python
dynamic_axes = {
    "waveform": {0: "batch"},
    "logits": {0: "batch"},
}
```

这表示 ONNX 的 batch 维是动态的：

```text
waveform: [batch, sample_rate]
logits:   [batch, label_count]
```

但时间长度不是动态的。也就是说：

```text
batch 可以变，waveform length 仍应保持 sample_rate 个点。
```

如果不想使用动态 batch，可以加：

```bash
--no-dynamic_batch
```

---

## 20. ONNX checker 和 ONNX Runtime 对比

默认：

```text
--check_onnx=True
```

脚本会执行：

```python
onnx_model = onnx.load(str(onnx_path))
onnx.checker.check_model(onnx_model)
```

这只检查 ONNX 图是否合法，不代表 ONNX Runtime 数值一定和 PyTorch 完全一致。

如果希望做 ONNX Runtime 数值对比，需要额外加：

```bash
--check_onnxruntime
```

脚本会：

```python
session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
torch_out = model(dummy).detach().cpu().numpy()
ort_out = session.run(None, {"waveform": dummy.cpu().numpy()})[0]
max_abs_diff = max(abs(torch_out - ort_out))
```

结果会写进 manifest：

```text
onnxruntime_max_abs_diff
```

如果本地没有安装 `onnxruntime`，不要加 `--check_onnxruntime`。

---

## 21. 输出文件和 manifest

默认输出目录：

```text
dscnn_kws/ONNX/models_full
```

每个 checkpoint 会生成：

```text
{checkpoint_stem}_full_onnx_friendly.onnx
```

同时写出：

```text
full_onnx_friendly_manifest.csv
```

manifest 字段包括：

| 字段 | 含义 |
| --- | --- |
| `checkpoint` | 输入 checkpoint 路径 |
| `onnx` | 导出的 ONNX 文件路径 |
| `layers` | 推断/指定的 DSCNN 层数 |
| `channels` | 推断/指定的 DSCNN 通道数 |
| `label_count` | 输出类别数 |
| `sample_rate` | 采样率，也是 1 秒输入长度 |
| `dct_coeff` | 选取的 MFCC 系数数 |
| `time_steps` | 时间帧数 |
| `input_shape` | dummy 输入形状 |
| `mfcc_scale` | `torchaudio_db` 或 `natural_log` |
| `opset` | ONNX opset |
| `onnxruntime_max_abs_diff` | 可选 ORT 对比误差 |

---

## 22. 命令行参数总览

### 22.1 输入输出相关

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--input_dir` | `None` | 批量读取 checkpoint 的目录 |
| `--pattern` | `*.pt` | 配合 `--input_dir` 的 glob 模式 |
| `--checkpoints` | `None` | 显式指定一个或多个 checkpoint |
| `--output_dir` | `dscnn_kws/ONNX/models_full` | ONNX 输出目录 |
| `--opset` | `17` | ONNX opset 版本 |
| `--batch_size` | `1` | 导出时 dummy batch |
| `--dynamic_batch` | `True` | 是否把 batch 维设为动态 |
| `--check_onnx` | `True` | 是否运行 ONNX checker |
| `--check_onnxruntime` | `False` | 是否运行 ORT 对比 |

### 22.2 前端和模型相关

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--sample_rate` | `16000` | 采样率，也是 1 秒输入长度 |
| `--dct_coeff` | `10` | 选取前多少个 MFCC 系数 |
| `--window_size_ms` | `32` | MFCC DFT 窗长 |
| `--window_stride_ms` | `32` | MFCC hop |
| `--layers` | `None` | 手动指定 DSCNN 层数 |
| `--channels` | `None` | 手动指定 DSCNN 通道数 |
| `--pre_emphasis` | `True` | 是否在 ONNX 图内做预加重 |
| `--pre_emphasis_coeff` | `0.97` | 预加重系数 |
| `--mfcc_scale` | `torchaudio_db` | MFCC log/dB 缩放模式 |
| `--mel_filter_shape` | `triangular` | Mel 滤波器形状 |

---

## 23. 单个模型导出示例

在项目根目录运行：

```bash
python dscnn_kws/ONNX/export_full_onnx_friendly.py \
  --checkpoints dscnn_kws/runs/dscnn_mobvoi_hi_xiaowen_binary_hardneg_lr0.001_ep30_20260524_194213/best.pt
```

如果 checkpoint 文件名或目录名不包含 `Lx_Cy`，脚本会尝试从权重形状推断层数和通道数。

如果你想明确指定：

```bash
python dscnn_kws/ONNX/export_full_onnx_friendly.py \
  --checkpoints dscnn_kws/runs/your_run/best.pt \
  --layers 5 \
  --channels 64
```

---

## 24. 批量导出 sweep best

如果模型都在：

```text
dscnn_kws/runs/sweep_best_models
```

可以运行：

```bash
python dscnn_kws/ONNX/export_full_onnx_friendly.py \
  --input_dir dscnn_kws/runs/sweep_best_models
```

如果不传 `--input_dir`，脚本会优先扫描：

```text
dscnn_kws/runs/sweep_best_models/*.pt
```

如果不存在，则回退扫描：

```text
dscnn_kws/runs/**/best.pt
```

---

## 25. 使用 ONNX Runtime 做数值检查

```bash
python dscnn_kws/ONNX/export_full_onnx_friendly.py \
  --checkpoints dscnn_kws/runs/your_run/best.pt \
  --check_onnxruntime
```

如果成功，manifest 中会出现：

```text
onnxruntime_max_abs_diff
```

这个值表示同一个 dummy waveform 输入下，PyTorch 输出和 ONNX Runtime 输出之间的最大绝对差。

---

## 26. 8 kHz 模型导出示例

如果训练模型是 8 kHz：

```bash
python dscnn_kws/ONNX/export_full_onnx_friendly.py \
  --checkpoints dscnn_kws/runs/your_8k_run/best.pt \
  --sample_rate 8000 \
  --dct_coeff 13 \
  --window_size_ms 32 \
  --window_stride_ms 32
```

此时：

```text
waveform input = [batch, 8000]
hop_length = 8000 * 32 / 1000 = 256
time_steps = 8000 // 256 + 1 = 32
selected MFCC = [B, 13, 32]
features = [B, 416]
```

---

## 27. 如果训练时使用 `TorchMFCC`

如果训练命令使用过：

```bash
--mfcc_impl torch
```

尤其是自然 log 路线，可以导出时改成：

```bash
python dscnn_kws/ONNX/export_full_onnx_friendly.py \
  --checkpoints dscnn_kws/runs/your_run/best.pt \
  --mfcc_scale natural_log
```

但需要注意：

- `ONNXFriendlyMFCC` 是为了 ONNX 友好导出而重写的前端。
- 它不保证和 `TorchMFCC` 每一个数值细节完全 bit-exact。
- 如果结果敏感，应使用 `--check_onnxruntime` 先确认 ONNX 图和当前 PyTorch wrapper 一致，再用验证集检查模型精度。

---

## 28. 如果训练时使用 `torchaudio.transforms.MFCC`

如果训练时是默认：

```bash
--mfcc_impl torchaudio
```

建议导出时保持：

```bash
--mfcc_scale torchaudio_db
```

这是脚本默认值。它试图贴近 torchaudio MFCC 的 dB 路线：

```text
power/mel energy -> 10*log10 -> top_db style clamp -> DCT
```

不过仍然要注意：ONNX-friendly 前端不是直接调用 torchaudio，而是复刻等价链路。因此导出完成后最好用真实验证集做端到端精度确认。

---

## 29. 和普通 `TorchMFCC` 的区别

| 项目 | `TorchMFCC` | `ONNXFriendlyMFCC` |
| --- | --- | --- |
| 文件 | `dscnn_kws/frontend/mfcc_torch.py` | `dscnn_kws/ONNX/onnx_friendly_mfcc.py` |
| 主要用途 | 训练/实验中的 PyTorch MFCC 前端 | 完整 ONNX 导出 |
| STFT 实现 | `torch.stft(..., return_complex=True)` | 两组实数 `F.conv1d` DFT kernels |
| 是否产生复数张量 | 是 | 否 |
| Mel/DCT | 复用本工程工具 | 复用本工程工具 |
| log 模式 | exact/PWL 等工程内配置 | `torchaudio_db` 或 `natural_log` |
| ONNX 友好性 | `torch.stft` 可能受限 | 更友好 |
| 是否适合训练 | 可以 | 主要为导出设计 |

---

## 30. 和 bandpass 前端的区别

这是最容易混淆的部分。`ONNXFriendlyMFCC` 和 `bandpass` 都会用到 `Conv1D`，但它们的含义完全不同。

### 30.1 总体链路不同

MFCC ONNX-friendly：

```text
waveform
-> pre-emphasis
-> Conv1D DFT real/imag kernels
-> power spectrum
-> Mel filterbank
-> dB/log
-> DCT
-> selected MFCC coefficients
-> DSCNN
```

bandpass：

```text
waveform
-> pre-emphasis
-> Conv1D sinc FIR bandpass filterbank
-> square power
-> optional phase average
-> log/PWL log
-> selected bands
-> DSCNN
```

一句话区别：

```text
ONNXFriendlyMFCC 用 Conv1D 模拟 STFT/DFT，再继续做 Mel 和 DCT；
bandpass 用 Conv1D 直接做时域带通滤波器组，跳过 STFT、Mel 矩阵和 DCT。
```

### 30.2 Conv1D 卷积核含义不同

| 项目 | ONNXFriendlyMFCC | bandpass |
| --- | --- | --- |
| Conv1D 核类型 | DFT 基函数 | FIR 带通滤波器 |
| 核来源 | cos/sin basis * Hann window | sinc lowpass 相减 + Hamming window |
| 输出通道 | `n_fft//2 + 1` 个频率 bin，实部和虚部分开算 | `bandpass_n_bands` 个频带 |
| 物理含义 | 计算每帧频谱 | 直接检测各频带响应 |
| 是否有虚部 | 有 imag kernel | 无复数/虚部概念 |
| 后续处理 | Mel filterbank + DCT | square + log/PWL |

### 30.3 输出特征语义不同

| 项目 | ONNXFriendlyMFCC | bandpass |
| --- | --- | --- |
| 输出名字 | MFCC | log band energy |
| 频率维含义 | DCT 后的 cepstral coefficients | 带通滤波器频带 |
| 是否有 DCT | 有 | 无 |
| 是否有 Mel filterbank | 有 | 无 |
| 默认选取 | `mfcc[:, :dct_coeff, :]` | `bands[:, :dct_coeff, :]` |
| `dct_coeff` 含义 | 选取前多少个 MFCC 系数 | 必须等于 band 数，表示选取多少个频带 |

### 30.4 ONNX 导出脚本不同

| 路线 | 脚本 | README |
| --- | --- | --- |
| MFCC full ONNX-friendly | `dscnn_kws/ONNX/export_full_onnx_friendly.py` | `dscnn_kws/ONNX/README_full_onnx_friendly.md` |
| bandpass / sweep best ONNX | `dscnn_kws/ONNX/export_sweep_best_to_onnx.py` | `dscnn_kws/ONNX/README.md` |

### 30.5 默认配置不同

MFCC full ONNX-friendly 默认：

```text
frontend = MFCC ONNX-friendly
sample_rate = 16000
dct_coeff = 10
window_size_ms = 32
window_stride_ms = 32
mfcc_scale = torchaudio_db
output_dir = dscnn_kws/ONNX/models_full
```

bandpass ONNX 默认：

```text
frontend = bandpass
sample_rate = 16000
dct_coeff = 10
bandpass_n_bands = 10
bandpass_f_min = 200.0
bandpass_f_max = 4000.0
bandpass_spacing = log
bandpass_kernel_size = 63
bandpass_phase_count = 1
log_approx_mode = pwl
output_dir = dscnn_kws/ONNX/models_bandpass_pwl_full
```

### 30.6 部署复杂度不同

MFCC ONNX-friendly 虽然避免了 `torch.stft`，但仍保留：

```text
DFT real conv
DFT imag conv
power spectrum
Mel matrix multiply
log10 / dB clamp
DCT matrix multiply
```

bandpass 前端更短：

```text
FIR bandpass Conv1D
square
phase average
log/PWL
```

因此从硬件/嵌入式实现角度，bandpass 通常更直接；从“贴近传统 MFCC 训练模型”的角度，ONNX-friendly MFCC 更合适。

### 30.7 什么时候用哪条路线

使用 `export_full_onnx_friendly.py` 的场景：

- 你的 checkpoint 是 MFCC 前端训练出来的。
- 你希望 ONNX 输入直接是 waveform。
- 你希望尽量贴近 `torchaudio.transforms.MFCC`。
- 你遇到了 `torch.stft` / complex tensor 的 ONNX 导出问题。

使用 `export_sweep_best_to_onnx.py --frontend bandpass` 的场景：

- 你的 checkpoint 是 bandpass 前端训练出来的。
- 你使用了 `bandpass_n_bands`、`bandpass_kernel_size`、`log_approx_mode=pwl` 等参数。
- 你希望导出 bandpass/PWL full ONNX。
- 你更关注部署友好的 Conv1D bandpass 前端。

最重要的原则：

```text
训练时用什么前端，导出时就必须用同一类前端和同一组关键参数。
```

不要用 MFCC ONNX-friendly 脚本去导出 bandpass checkpoint，也不要用 bandpass 参数去解释 MFCC checkpoint。

---

## 31. 和 `BANDPASS_FRONTEND_EXPLANATION.md` 的关系

如果你想理解 bandpass 前端本身，包括：

- sinc FIR 带通核如何生成。
- `bandpass_n_bands` 和 `dct_coeff` 为什么要一致。
- `window_stride_ms` 如何决定时间步。
- `log/PWL` 在 bandpass 中的位置。
- bandpass 和 MFCC 的信号处理差异。

请看：

```text
dscnn_kws/BANDPASS_FRONTEND_EXPLANATION.md
```

本文档只负责解释：

```text
MFCC full ONNX-friendly 导出
```

以及它和 bandpass ONNX 路线的区别。

---

## 32. 常见问题

### 32.1 为什么叫 full ONNX

因为导出的 ONNX 包含前端和后端：

```text
waveform -> MFCC -> DSCNN -> logits
```

不是只导出 DSCNN backbone。

### 32.2 full ONNX 的输入是不是 MFCC 特征

不是。full ONNX 的输入是原始 waveform：

```text
waveform: [batch, sample_rate]
```

如果只导出 backbone，输入才会是已经算好的特征。

### 32.3 `dct_coeff=10` 是不是只算 10 个 MFCC

不是。`ONNXFriendlyMFCCDSCNN` 内部先算：

```text
n_mfcc = 40
```

然后再取：

```python
mfcc[:, :dct_coeff, :]
```

所以 `dct_coeff=10` 表示从 40 维 MFCC 中选取前 10 维送入 DSCNN。

### 32.4 这个脚本能不能导出 bandpass 模型

不建议。这个脚本的 wrapper 是 `ONNXFriendlyMFCCDSCNN`，它固定走 ONNX-friendly MFCC 前端。bandpass 模型请使用：

```bash
python dscnn_kws/ONNX/export_sweep_best_to_onnx.py \
  --frontend bandpass \
  --export_mode full
```

### 32.5 为什么 ONNX-friendly MFCC 里也用了 Conv1D，它还是 MFCC

因为这里的 Conv1D 只是 DFT 的实现手段。它的卷积核是 cos/sin 频谱基，用来代替 `torch.stft`，后面仍然有 Mel filterbank 和 DCT。

bandpass 前端里的 Conv1D 是 sinc FIR 带通滤波器，直接输出频带能量，后面没有 Mel filterbank 和 DCT。

### 32.6 导出后是否一定和训练时完全一致

不一定。它的目标是“ONNX 友好”和“尽量贴近 MFCC 训练链路”，但不是 bit-exact 复刻所有 torchaudio 内部细节。建议导出后做两类检查：

1. `--check_onnxruntime` 检查当前 PyTorch wrapper 和 ONNX Runtime 输出差异。
2. 用验证集/测试集跑端到端精度，检查导出模型是否保持可接受准确率。

---

## 33. 推荐工作流

### 33.1 MFCC 模型

如果 checkpoint 是 MFCC 训练得到的：

```bash
python dscnn_kws/ONNX/export_full_onnx_friendly.py \
  --checkpoints dscnn_kws/runs/your_mfcc_run/best.pt \
  --sample_rate 16000 \
  --dct_coeff 10 \
  --window_size_ms 32 \
  --window_stride_ms 32 \
  --mfcc_scale torchaudio_db \
  --check_onnxruntime
```

导出后检查：

```text
dscnn_kws/ONNX/models_full/*.onnx
dscnn_kws/ONNX/models_full/full_onnx_friendly_manifest.csv
```

### 33.2 bandpass 模型

如果 checkpoint 是 bandpass/PWL 训练得到的：

```bash
python dscnn_kws/ONNX/export_sweep_best_to_onnx.py \
  --export_mode full \
  --frontend bandpass \
  --checkpoints dscnn_kws/runs/bandpass_pwl_snr_scene_arch_sweep_best_models/your_model.pt \
  --sample_rate 16000 \
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
  --log_pwl_gamma 1.0 \
  --check_onnxruntime
```

---

## 34. 验证 ONNX 在十个场景、五种 SNR 下的精度

导出 ONNX 以后，还需要验证它在真实测试集和噪声场景下的精度。为此，本目录提供：

```text
dscnn_kws/ONNX/eval_full_onnx_snr_scene_acc.py
```

这个脚本用于评估已经导出的 full ONNX 模型：

```text
waveform -> ONNXFriendlyMFCC -> DSCNN -> logits
```

它会复用项目里的 `SpeechCommandDataset`，对 test manifest 中的样本在线混入指定 TAU 场景噪声，并在固定 SNR 下计算：

```text
acc
precision
recall
f1
num_samples
```

默认场景是 10 个 TAU scene：

```text
airport
bus
metro
metro_station
park
public_square
shopping_mall
street_pedestrian
street_traffic
tram
```

默认 SNR 是 5 个点：

```text
20, 10, 5, 0, -5 dB
```

因此每个 ONNX 模型会产生：

```text
10 scenes * 5 SNRs = 50 rows
```

如果你有 20 个 ONNX 模型，就会产生：

```text
20 * 50 = 1000 rows
```

### 34.1 在服务器上评估刚导出的 MFCC-friendly ONNX

你的 ONNX 目录是：

```text
/root/kws/dscnn_kws/dscnn_kws/ONNX/models_snr_scene_mfcc_friendly
```

推荐在项目根目录运行：

```bash
cd /root/kws/dscnn_kws

python dscnn_kws/ONNX/eval_full_onnx_snr_scene_acc.py \
  --onnx_dir /root/kws/dscnn_kws/dscnn_kws/ONNX/models_snr_scene_mfcc_friendly \
  --root /root/kws/dscnn_kws/dscnn_kws/data \
  --scene_test_root /root/kws/dscnn_kws/dscnn_kws/noise/tau \
  --sample_rate 16000 \
  --batch 128 \
  --num_workers 0 \
  --out_dir /root/kws/dscnn_kws \
  --out_prefix onnx_snr_scene_mfcc_friendly
```

输出文件：

```text
/root/kws/dscnn_kws/onnx_snr_scene_mfcc_friendly_grid_results.csv
/root/kws/dscnn_kws/onnx_snr_scene_mfcc_friendly_scene_summary.csv
/root/kws/dscnn_kws/onnx_snr_scene_mfcc_friendly_arch_summary.csv
```

### 34.2 三个输出 CSV 的含义

`*_grid_results.csv` 是最细粒度结果，一行对应：

```text
一个 ONNX 模型 + 一个 dataset + 一个 scene + 一个 SNR
```

主要字段：

| 字段 | 含义 |
| --- | --- |
| `dataset` | 数据集名，例如 `mobvoi_hi_xiaowen_binary_hardneg` |
| `arch` | 架构名，例如 `L5_C64` |
| `scene` | TAU 场景名 |
| `snr_db` | SNR |
| `acc` | 准确率 |
| `precision` | macro precision |
| `recall` | macro recall |
| `f1` | macro F1 |
| `num_samples` | test manifest 样本数 |
| `onnx` | 当前评估的 ONNX 文件 |

`*_scene_summary.csv` 按：

```text
dataset + arch + scene
```

聚合 5 个 SNR 点，给出每个场景的平均/最小/最大 ACC 和 F1。

`*_arch_summary.csv` 按：

```text
dataset + arch
```

聚合全部 10 个 scene 和 5 个 SNR 点，也就是每个模型 50 个评估点的总体表现。

### 34.3 dataset 如何匹配

脚本默认会从 ONNX 文件名中解析 dataset 和架构。例如：

```text
mobvoi_hi_xiaowen_binary_hardneg_L5_C64_layers5_channels64_params22530_noise_best_full_onnx_friendly.onnx
```

会解析成：

```text
dataset = mobvoi_hi_xiaowen_binary_hardneg
arch = L5_C64
layers = 5
channels = 64
expected_params = 22530
```

然后用：

```text
/root/kws/dscnn_kws/dscnn_kws/data/{dataset}/test_manifest.json
```

作为测试集。

如果文件名里无法解析 dataset，可以手动指定：

```bash
--dataset mobvoi_hi_xiaowen_binary_hardneg
```

但如果同一个目录里同时有两个 dataset 的 ONNX，通常不要加 `--dataset`，让脚本逐个从文件名解析即可。

### 34.4 ONNX Runtime 线程参数

脚本默认：

```text
--ort_intra_op_num_threads 1
--ort_inter_op_num_threads 1
```

这样可以减少 Docker/容器里常见的：

```text
pthread_setaffinity_np failed
```

如果你想让 ONNX Runtime 自己决定线程数，可以改成：

```bash
--ort_intra_op_num_threads 0 \
--ort_inter_op_num_threads 0
```

### 34.5 评估一个单独 ONNX 模型

如果只想先测试一个模型：

```bash
python dscnn_kws/ONNX/eval_full_onnx_snr_scene_acc.py \
  --onnx_models /root/kws/dscnn_kws/dscnn_kws/ONNX/models_snr_scene_mfcc_friendly/mobvoi_hi_xiaowen_binary_hardneg_L5_C64_layers5_channels64_params22530_noise_best_full_onnx_friendly.onnx \
  --root /root/kws/dscnn_kws/dscnn_kws/data \
  --scene_test_root /root/kws/dscnn_kws/dscnn_kws/noise/tau \
  --sample_rate 16000 \
  --out_dir /root/kws/dscnn_kws \
  --out_prefix onnx_single_L5_C64
```

这会只跑一个模型的 50 个评估点。

### 34.6 重要提醒

这个评估脚本验证的是：

```text
导出的 ONNX 模型在 test_manifest + TAU scene noise + fixed SNR 下的实际分类精度。
```

它比导出时的 `--check_onnxruntime` 更接近真实精度验证，因为 `--check_onnxruntime` 只比较一个随机 dummy 输入下：

```text
PyTorch ONNX-friendly wrapper vs ONNX Runtime
```

而这里会真正遍历测试集，并在线混入场景噪声。

---

## 35. 最后总结

`README_full_onnx_friendly.md` 对应的方案是：

```text
用实数 Conv1D DFT 重写 MFCC 前端，
绕开 torch.stft / complex ONNX 导出问题，
导出完整 waveform -> logits 的 DSCNN KWS ONNX。
```

它的完整链路是：

```text
waveform
-> pre-emphasis
-> real/imag Conv1D DFT
-> power spectrum
-> Mel filterbank
-> torchaudio_db 或 natural_log
-> DCT
-> selected MFCC coefficients
-> DSCNN
-> logits
```

它和 bandpass 前端的核心区别是：

```text
ONNX-friendly MFCC:
  Conv1D 是为了实现 DFT，后面仍然是 Mel + DCT。

bandpass:
  Conv1D 是 sinc FIR 带通滤波器组，直接输出 log band energy，
  不再使用 STFT、Mel filterbank 和 DCT。
```

因此，导出时一定先确认 checkpoint 的训练前端：

```text
MFCC checkpoint      -> export_full_onnx_friendly.py
bandpass checkpoint  -> export_sweep_best_to_onnx.py --frontend bandpass
```
