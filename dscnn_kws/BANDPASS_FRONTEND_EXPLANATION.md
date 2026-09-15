# Bandpass Conv1D 前端替换 MFCC 前端的实现详解

本文档说明本工程如何用 `bandpass` 前端替换传统 `MFCC` 前端。这里的 `bandpass` 前端也可以理解为一个基于 `Conv1D` 的时域滤波器组前端，但它不是普通意义上随网络一起随机初始化、自由学习的 1D CNN 层，而是一个带有明确信号处理含义的 FIR 带通滤波器组：

```text
waveform
-> pre-emphasis
-> fixed FIR bandpass Conv1D filterbank
-> square power
-> optional multi-phase average
-> log / PWL log
-> select dct_coeff bands
-> flatten
-> DSCNN / LSTM backbone
```

对应代码主线：

- `dscnn_kws/train.py`
  - `MFCCDSCNN` wrapper 负责在 `mfcc` 和 `bandpass` 前端之间切换。
  - `--frontend mfcc|bandpass` 是训练入口的前端选择开关。
  - `--dct_coeff` 决定送入 backbone 的频率维数量。
- `dscnn_kws/frontend/mfcc_torch.py`
  - 工程内的 PyTorch MFCC 复现路径。
  - 实现 `STFT -> power -> Mel filterbank -> log/PWL -> DCT`。
- `dscnn_kws/frontend/bandpass_torch.py`
  - Conv1D bandpass 前端实现。
  - 实现 `sinc FIR bandpass kernels -> F.conv1d -> square -> log/PWL`。
- `dscnn_kws/model/dscnn.py`
  - DSCNN backbone 不直接关心前端是 MFCC 还是 bandpass。
  - 它只要求输入最终被展平成 `[B, time_steps * dct_coeff]`。
- `dscnn_kws/model/lstm.py`
  - LSTM wrapper 也支持同样的 `frontend` 切换逻辑。
- `dscnn_kws/eval_fah_frr.py`
  - 评估时同样暴露 `--frontend bandpass` 和 bandpass 参数。
- `dscnn_kws/sweep_fixed_bandpass_noise_snr_scene_acc.py`
  - 固定使用 bandpass + PWL log 的噪声/SNR/场景鲁棒性 sweep 脚本。

---

## 1. 替换 MFCC 的总体思路

传统 KWS 中，模型通常不直接吃原始波形，而是先把 1 秒音频变成二维时频特征，例如 MFCC：

```text
waveform
-> pre-emphasis
-> framing / windowing
-> STFT
-> power spectrum
-> Mel filterbank
-> log
-> DCT
-> selected MFCC coefficients
-> backbone
```

本工程的 bandpass 前端保留了“从原始波形提取短时频带能量图”的核心目标，但把 MFCC 中最复杂的 `STFT + Mel + DCT` 链路替换成了时域 Conv1D 滤波器组：

```text
waveform
-> pre-emphasis
-> Conv1D bandpass filterbank
-> square power
-> log / PWL log
-> selected bands
-> backbone
```

也就是说，替换的本质不是“把整个模型改成 1D CNN”，而是：

```text
用一组时域 FIR 带通卷积核，替代 MFCC 前端里的频谱分析和滤波器组聚合部分。
```

这样做以后，后端 DSCNN/LSTM 仍然接收类似 `[频率维, 时间维]` 的二维特征图。对后端来说，它看到的仍然是：

```text
[B, dct_coeff, time_steps]
```

只是这个 `dct_coeff` 在 MFCC 路径里表示“选取前多少个 MFCC 系数”，在 bandpass 路径里表示“选取多少个带通频带”。为了接口兼容，工程强制要求：

```text
frontend == bandpass 时，dct_coeff 必须等于 bandpass_n_bands
```

这个检查在 `train.py` 和 bandpass sweep 脚本里都有。

---

## 2. 工程中如何切换前端

训练入口 `dscnn_kws/train.py` 的参数中有：

```text
--frontend {mfcc,bandpass}
--dct_coeff
--window_size_ms
--window_stride_ms

--bandpass_n_bands
--bandpass_f_min
--bandpass_f_max
--bandpass_spacing
--bandpass_kernel_size
--bandpass_phase_count

--pre_emphasis / --no-pre_emphasis
--pre_emphasis_coeff

--log_approx_mode {exact,pwl}
--log_offset
--log_input_clamp_min
```

在 `MFCCDSCNN.__init__()` 中，代码先计算窗口长度和帧移：

```python
n_fft = int(sample_rate * window_size_ms / 1000)
hop_length = int(sample_rate * window_stride_ms / 1000)
```

然后根据 `frontend` 选择不同的 `feature_extractor`：

```python
if frontend == "mfcc":
    self.feature_extractor = MFCC(...)      # 或 TorchMFCC(...)
else:
    self.feature_extractor = TorchBandpass(...)
```

其中 `TorchBandpass(...)` 接收的关键参数是：

```python
TorchBandpass(
    sample_rate=sample_rate,
    n_bands=bandpass_n_bands,
    frame_length=n_fft,
    frame_hop=hop_length,
    f_min=bandpass_f_min,
    f_max=bandpass_f_max,
    spacing=bandpass_spacing,
    kernel_size=bandpass_kernel_size,
    phase_count=bandpass_phase_count,
    log_approx_mode=log_approx_mode,
    ...
)
```

这说明工程层面的替换非常集中：不是改 dataset，不是改训练循环，也不是重写 DSCNN，而是在 wrapper 里把 `feature_extractor` 从 MFCC 换成 `TorchBandpass`。

---

## 3. 前向传播完整路径

`MFCCDSCNN.forward()` 是整个替换方案的接口枢纽。无论 `feature_extractor` 是 MFCC 还是 bandpass，后处理逻辑基本一致：

```python
def forward(self, x):
    if x.dim() == 3:
        x = x.squeeze(1)

    if self.pre_emphasis:
        x = apply_pre_emphasis(x, self.pre_emphasis_coeff)

    mfcc = self.feature_extractor(x)

    if self.training and self.spec_aug:
        ...

    mfcc = mfcc[:, : self.dct_coeff, :]
    mfcc = mfcc.permute(0, 2, 1).reshape(mfcc.size(0), -1)
    return self.backbone(mfcc)
```

虽然变量名仍然叫 `mfcc`，但当 `frontend=bandpass` 时，这个张量实际是 bandpass log-energy 特征：

```text
MFCC 路径:
  mfcc = [B, n_mfcc, time]

bandpass 路径:
  mfcc = [B, n_bands, time]
```

后面统一执行：

```text
[B, freq, time]
-> mfcc[:, :dct_coeff, :]
-> [B, dct_coeff, time]
-> permute(0, 2, 1)
-> [B, time, dct_coeff]
-> flatten
-> [B, time * dct_coeff]
-> DSCNN / LSTM
```

这个设计使得前端替换对后端是低侵入的。DSCNN 不需要知道特征来自 MFCC 还是 Conv1D bandpass，只需要 `input_dim = time_steps * dct_coeff` 对得上。

---

## 4. pre-emphasis 在哪里

本工程保留了传统语音前端常用的预加重。它不属于 Conv1D 卷积核本身，而是在进入 `feature_extractor` 之前统一执行，所以 MFCC 和 bandpass 都可以共享它。

代码在 `dscnn_kws/utils/audio.py`：

```python
def apply_pre_emphasis(x: torch.Tensor, coeff: float = 0.97) -> torch.Tensor:
    if coeff <= 0:
        return x
    ...
    y[:, :, 1:] = x[:, :, 1:] - coeff * x[:, :, :-1]
    y[:, :, 0] = x[:, :, 0]
    return y
```

公式为：

```text
y[0] = x[0]
y[t] = x[t] - alpha * x[t-1], t >= 1
```

默认：

```text
alpha = 0.97
```

它的作用：

- 增强高频成分。
- 抑制语音波形中低频能量过强的问题。
- 让后续滤波器组看到更平衡的频谱。
- 保持和传统 MFCC 语音处理流程的工程习惯一致。

需要注意的是，pre-emphasis 是 wrapper 里的公共预处理，不是 `TorchBandpass` 类内部做的。因此从结构上说：

```text
pre-emphasis + TorchBandpass
```

才是完整的 bandpass 前端链路。

---

## 5. MFCC 路径具体做了什么

为了理解替换点，先看工程内 `TorchMFCC` 的处理流程。`dscnn_kws/frontend/mfcc_torch.py` 中的 `TorchMFCC.forward()` 是：

```python
stft = torch.stft(...)
power_spec = stft.real.pow(2) + stft.imag.pow(2)
mel_spec = torch.matmul(mel_fb, power_spec)
log_mel = self._log_transform(mel_spec)
mfcc = torch.matmul(dct_mat, log_mel)
return mfcc
```

对应流程：

```text
waveform
-> STFT
-> power spectrum
-> Mel filterbank matrix multiply
-> log / PWL log
-> DCT matrix multiply
-> MFCC
```

这里的几个核心算子是：

- `torch.stft`
  - 把时域信号变成复数频谱。
- `power_spec`
  - 对复数频谱求能量。
- `mel_fb @ power_spec`
  - 用 Mel 滤波器组把 FFT bin 聚合成 Mel 频带。
- `log`
  - 压缩动态范围。
- `dct_mat @ log_mel`
  - 做离散余弦变换，得到 MFCC 系数。

MFCC 的输出形状是：

```text
[B, n_mfcc, time]
```

工程里通常先算 40 维 MFCC，然后通过：

```python
mfcc = mfcc[:, : self.dct_coeff, :]
```

选取前 `dct_coeff` 个系数送入后端。

---

## 6. bandpass 路径具体做了什么

`TorchBandpass` 的核心文件是 `dscnn_kws/frontend/bandpass_torch.py`。它由三部分组成：

1. 构造频带边界。
2. 根据频带边界生成 sinc FIR 带通卷积核。
3. 前向时用 `F.conv1d` 提取每个频带的短时能量，再做 log/PWL。

整体流程：

```text
waveform [B, T]
-> unsqueeze
-> [B, 1, T]
-> F.conv1d(x, bandpass_kernels, stride=frame_hop, padding=0)
-> [B, n_bands, time_conv]
-> square
-> adaptive align to target_time_steps
-> phase average
-> log / PWL log
-> [B, n_bands, target_time_steps]
```

---

## 7. 频带边界如何生成

频带边界由 `_build_band_edges(...)` 生成：

```python
if spacing == "log":
    edges = torch.logspace(math.log10(f_min), math.log10(f_max), steps=n_bands + 1)
else:
    edges = torch.linspace(f_min, f_max, steps=n_bands + 1)
```

参数含义：

| 参数 | 含义 |
| --- | --- |
| `n_bands` | 输出频带数，也就是 Conv1D 输出通道数 |
| `f_min` | 最低频率边界 |
| `f_max` | 最高频率边界，不能超过 Nyquist 频率 |
| `spacing` | 频带划分方式，支持 `log` 或 `linear` |

如果 `n_bands=10`，就会生成 11 个边界点，相邻两个边界点组成一个频带，因此得到 10 个带通滤波器。

如果 `spacing=log`，低频区域的频带更密，高频区域更宽，比较接近听觉系统对频率的非线性感知；如果 `spacing=linear`，每个频带在 Hz 轴上宽度相等。

---

## 8. sinc FIR 带通卷积核如何生成

带通滤波器由 `_sinc_bandpass_kernel(...)` 生成。核心思想是：

```text
bandpass(low, high) = lowpass(high) - lowpass(low)
```

代码中：

```python
h_high = 2.0 * high / fs * torch.sinc(2.0 * high * n / fs)
h_low = 2.0 * low / fs * torch.sinc(2.0 * low * n / fs)
h_bp = h_high - h_low
```

这里的 `torch.sinc(...)` 用来生成理想低通滤波器的时域冲激响应。两个低通相减，就得到一个只保留 `low_hz ~ high_hz` 的带通响应。

然后代码加 Hamming window：

```python
window = torch.hamming_window(kernel_size, periodic=False)
h_bp = h_bp * window
```

原因是理想 sinc 滤波器是无限长的，实际部署必须截断成有限长度 FIR。直接截断会带来明显旁瓣和振铃，Hamming window 可以让滤波器的频率响应更平滑。

最后做归一化：

```python
norm = torch.sum(torch.abs(h_bp), dim=1, keepdim=True).clamp_min(1e-12)
h_bp = h_bp / norm
```

归一化的作用是避免不同频带卷积核因为带宽或数值尺度不同而输出能量差异过大。

最终生成的卷积核形状是：

```text
[n_bands, 1, kernel_size]
```

例如：

```text
n_bands = 10
kernel_size = 63
bandpass_kernels shape = [10, 1, 63]
```

或者：

```text
n_bands = 16
kernel_size = 63
bandpass_kernels shape = [16, 1, 63]
```

---

### 8.1 补充问答：bandpass 的卷积权重到底怎么来的

#### 问题 1：bandpass 带通滤波器组是怎么实现的，卷积权重参数怎么得到？

本工程里的 `bandpass` 前端不是普通神经网络中随机初始化、再通过反向传播学习出来的 `Conv1D` 层。它的卷积权重来自一个明确的信号处理构造过程：

```text
确定频带边界
-> 每个频带生成一个 sinc FIR 带通滤波器
-> 加 Hamming window
-> 归一化
-> 作为 Conv1D weight 使用
```

以当前 bandpass sweep 默认配置为例：

```python
bandpass_n_bands = 10
bandpass_f_min = 200.0
bandpass_f_max = 4000.0
bandpass_spacing = "log"
bandpass_kernel_size = 63
```

代码会先在 `200 Hz ~ 4000 Hz` 之间生成 `n_bands + 1` 个频带边界。如果 `n_bands=10`，就会得到 11 个边界点：

```text
edge0, edge1, edge2, ..., edge10
```

每相邻两个边界组成一个频带：

```text
band 0: edge0 ~ edge1
band 1: edge1 ~ edge2
...
band 9: edge9 ~ edge10
```

所以最终会生成 10 个带通滤波器，对应 `Conv1D` 的 10 个输出通道。

对每个频带，代码使用：

```text
bandpass(low, high) = lowpass(high) - lowpass(low)
```

也就是：

```python
h_high = 2.0 * high / fs * torch.sinc(2.0 * high * n / fs)
h_low = 2.0 * low / fs * torch.sinc(2.0 * low * n / fs)
h_bp = h_high - h_low
```

直观理解是：

```text
lowpass(high) 允许 0 ~ high Hz 通过
lowpass(low)  允许 0 ~ low Hz 通过

两者相减后，只剩 low ~ high Hz
```

这样就得到了一个带通 FIR 滤波器。

随后代码会对 `h_bp` 乘 Hamming window：

```python
h_bp = h_bp * window
```

这是因为理想 sinc 滤波器本来是无限长的，实际只能截断成有限长度。直接截断会带来较明显的旁瓣和振铃，Hamming window 可以让频率响应更平滑。

最后做归一化：

```python
norm = torch.sum(torch.abs(h_bp), dim=1, keepdim=True).clamp_min(1e-12)
h_bp = h_bp / norm
```

归一化的目的是让不同频带的滤波器幅度尺度不要差太多。

#### 问题 2：带通滤波器是怎么变成卷积权重的？

FIR 带通滤波器本身就是一串时域系数，而 `Conv1D` 的卷积核本质上也是一串时域系数，所以它们可以直接对应。

比如一个长度为 63 的 FIR 带通滤波器可以写成：

```text
h = [h0, h1, h2, ..., h62]
```

它作用在输入波形上时，本质上就是滑动加权求和：

```text
y[t] = h0*x[t] + h1*x[t+1] + h2*x[t+2] + ... + h62*x[t+62]
```

这正是 `Conv1D` 做的事情：

```python
y = F.conv1d(x, weight)
```

因此代码只需要把所有带通滤波器的 FIR 系数整理成 PyTorch `conv1d` 需要的权重形状。

生成滤波器后，`kernels` 的原始形状可以理解为：

```text
[n_bands, kernel_size]
```

例如：

```text
[10, 63]
```

表示：

```text
10 个带通滤波器
每个滤波器 63 个 FIR 系数
```

然后代码做：

```python
return kernels.unsqueeze(1), edges
```

形状变成：

```text
[10, 1, 63]
```

这正好对应 PyTorch `F.conv1d` 的权重格式：

```text
[out_channels, in_channels, kernel_size]
```

也就是：

```text
out_channels = 10 个频带
in_channels  = 1 个原始 waveform 通道
kernel_size  = 63 点 FIR 滤波器
```

前向计算时：

```python
x_phase shape = [B, 1, T]
kernels shape = [10, 1, 63]
y = F.conv1d(x_phase, kernels, stride=self.frame_hop, padding=0)
y shape = [B, 10, time_steps]
```

于是每个输出通道就是一个频带的滤波响应：

```text
y[:, 0, :] = 第 0 个带通滤波器扫过 waveform 的响应
y[:, 1, :] = 第 1 个带通滤波器扫过 waveform 的响应
...
y[:, 9, :] = 第 9 个带通滤波器扫过 waveform 的响应
```

所以“带通滤波器变成卷积权重”的本质就是：

```text
带通滤波器的 FIR 系数 = Conv1D kernel 的 weight 数值
```

#### 问题 3：`h_bp` 就是 FIR 系数吗？

是的，`h_bp` 就是带通 FIR 滤波器的系数。

在 `_sinc_bandpass_kernel(...)` 中：

```python
h_high = 2.0 * high / fs * torch.sinc(2.0 * high * n / fs)
h_low = 2.0 * low / fs * torch.sinc(2.0 * low * n / fs)
h_bp = h_high - h_low
```

这里：

```text
h_high 是截止频率为 high 的低通 FIR 系数
h_low  是截止频率为 low 的低通 FIR 系数
h_bp   是 low ~ high 的带通 FIR 系数
```

后面的：

```python
h_bp = h_bp * window
h_bp = h_bp / norm
```

只是对 FIR 系数做窗函数平滑和幅度归一化。最终的 `h_bp` 仍然是 FIR 系数。

如果当前：

```python
bandpass_n_bands = 10
bandpass_kernel_size = 63
```

那么：

```text
h_bp shape = [10, 63]
```

含义是：

```text
10 个带通 FIR 滤波器
每个滤波器 63 个 FIR 系数
```

再经过：

```python
kernels = h_bp.unsqueeze(1)
```

就变成：

```text
[10, 1, 63]
```

并作为 `F.conv1d(...)` 的固定卷积权重使用。

需要再次强调：这些权重不是训练出来的参数，而是由下面这些配置确定性生成的：

```text
sample_rate
bandpass_n_bands
bandpass_f_min
bandpass_f_max
bandpass_spacing
bandpass_kernel_size
```

只要这些配置一致，每次重建模型都会得到同一组 bandpass 卷积权重。

---

## 9. 这里的 Conv1D 是什么结构

`TorchBandpass` 前向中真正做卷积的代码是：

```python
y = F.conv1d(x_phase, kernels, stride=self.frame_hop, padding=0)
```

因此它等价于如下 Conv1D 结构：

| 项目 | 值 |
| --- | --- |
| 输入形状 | `[B, 1, T]` |
| `in_channels` | `1` |
| `out_channels` | `bandpass_n_bands` |
| `kernel_size` | `bandpass_kernel_size`，默认 `63` |
| `stride` | `frame_hop = sample_rate * window_stride_ms / 1000` |
| `padding` | `0` |
| `bias` | 无 |
| `groups` | `1` |
| weight 形状 | `[bandpass_n_bands, 1, bandpass_kernel_size]` |

但是要特别注意：工程里不是使用 `nn.Conv1d(...)` 注册一个可训练层，而是把生成好的 FIR 核注册成 buffer：

```python
self.register_buffer("bandpass_kernels", kernels, persistent=False)
```

然后在 forward 中调用函数式卷积：

```python
kernels = self.bandpass_kernels.to(device=x.device, dtype=x.dtype)
y = F.conv1d(x_phase, kernels, stride=self.frame_hop, padding=0)
```

这意味着：

- bandpass 卷积核由 `f_min/f_max/n_bands/spacing/kernel_size` 决定。
- 默认不是随机初始化。
- 默认不是训练参数。
- optimizer 不会更新这些滤波器。
- checkpoint 中也不需要保存这些核，因为 `persistent=False`，重建模型时会根据参数重新生成。

所以这个前端更准确的名字是：

```text
fixed sinc-initialized FIR Conv1D bandpass filterbank
```

而不是：

```text
learnable 1D CNN frontend
```

---

## 10. `window_size_ms` 和 `window_stride_ms` 的真实作用

训练脚本对 MFCC 和 bandpass 使用同一组参数：

```text
--window_size_ms
--window_stride_ms
```

但它们在两条路径里的作用并不完全一样。

### 10.1 在 MFCC 路径中

MFCC 路径里：

```python
n_fft = int(sample_rate * window_size_ms / 1000)
hop_length = int(sample_rate * window_stride_ms / 1000)
```

`n_fft` / `win_length` 决定 STFT 每帧窗口长度，`hop_length` 决定帧移。

也就是说：

```text
window_size_ms  -> STFT window length
window_stride_ms -> STFT hop length
```

### 10.2 在 bandpass 路径中

bandpass 路径也会把：

```python
frame_length = n_fft
frame_hop = hop_length
```

传给 `TorchBandpass`。但是当前 `TorchBandpass.forward()` 里真正影响卷积采样步长的是：

```python
stride=self.frame_hop
```

而 `frame_length` 只是保存为成员变量，并没有参与卷积核长度或能量窗口计算。bandpass 的卷积核长度由单独的参数控制：

```text
bandpass_kernel_size
```

因此当前 bandpass 路径中：

```text
window_stride_ms -> 决定 Conv1D stride / 时间步数
window_size_ms   -> 保持接口兼容，目前不决定 bandpass kernel_size
bandpass_kernel_size -> 决定 FIR 滤波器长度
```

这是理解本工程实现时非常重要的一点。不要把 `window_size_ms=32` 误解成 bandpass 卷积核覆盖 32 ms。真正的 FIR 核长度是：

```text
bandpass_kernel_size samples
```

例如 `kernel_size=63`：

```text
8 kHz  下约 63 / 8000  = 7.875 ms
16 kHz 下约 63 / 16000 = 3.9375 ms
```

---

## 11. 时间步数如何计算

DSCNN 的输入维度由：

```python
time_steps = calculate_time_steps(sample_rate, window_stride_ms)
input_dim = time_steps * dct_coeff
```

`calculate_time_steps(...)` 的逻辑是：

```python
stride_samples = int(sample_rate * window_stride_ms / 1000)
audio_samples = int(sample_rate * audio_duration_ms / 1000)
return floor(audio_samples / stride_samples) + 1
```

默认 `audio_duration_ms=1000`，因此：

```text
time_steps = floor(1 秒采样点数 / frame_hop) + 1
```

在 `TorchBandpass.forward()` 中也有对应目标时间步：

```python
raw_len = x.size(-1)
target_time_steps = raw_len // self.frame_hop + 1
```

如果 `F.conv1d` 实际输出时间长度与目标不一致，代码使用：

```python
power = F.adaptive_avg_pool1d(power, output_size=target_time_steps)
```

强制对齐到目标时间步。

常见例子：

| sample_rate | window_stride_ms | frame_hop | 1 秒样本数 | time_steps |
| --- | ---: | ---: | ---: | ---: |
| 8000 | 32 | 256 | 8000 | `8000 // 256 + 1 = 32` |
| 16000 | 32 | 512 | 16000 | `16000 // 512 + 1 = 32` |
| 16000 | 16 | 256 | 16000 | `16000 // 256 + 1 = 63` |

所以在默认 1 秒音频、32 ms stride 下，bandpass 和 MFCC 都会被组织成 32 个时间步。

---

## 12. 能量平方和 log 压缩

Conv1D 输出 `y` 是每个带通滤波器对当前时间位置附近波形的响应：

```python
y = F.conv1d(...)
```

这个响应本身可能为正也可能为负，因此工程中对它平方：

```python
power = y.pow(2)
```

平方之后得到非负能量，含义是：

```text
当前时间位置，当前频带内的响应强度。
```

然后把不同 phase 的 power 平均：

```python
band_energy = torch.stack(phase_powers, dim=0).mean(dim=0)
```

最后进入 `_log_transform(...)`：

```python
x = torch.clamp(x + self.log_offset, min=self.log_input_clamp_min)
if self.log_approx_mode == "exact":
    return torch.log(x)
return apply_piecewise_linear(x, bp, slopes, intercepts)
```

默认是精确 log：

```text
log(power + 1e-6)
```

也支持 PWL 分段线性近似：

```text
--log_approx_mode pwl
--log_pwl_num_segments 6
--log_pwl_strategy uniform_logx
```

这和工程里对 MFCC 前端 log 的硬件友好化改造是一致的：MFCC 的 `TorchMFCC` 和 bandpass 的 `TorchBandpass` 都复用了 PWL log 思路。

log 的作用：

- 压缩动态范围。
- 让大能量和小能量差距不至于过分悬殊。
- 更接近传统 log-Mel / MFCC 特征的统计性质。
- 为后续 Q 格式、PWL、硬件实现提供更清晰的替换点。

---

## 13. `phase_count` 是什么

`TorchBandpass.forward()` 中有：

```python
for p in range(self.phase_count):
    offset = int(round(p * self.frame_hop / self.phase_count))
    x_phase = x[..., offset:]
    y = F.conv1d(x_phase, kernels, stride=self.frame_hop, padding=0)
    power = y.pow(2)
    ...
    phase_powers.append(power)

band_energy = torch.stack(phase_powers, dim=0).mean(dim=0)
```

如果 `phase_count=1`，只从原始起点做一次 stride 卷积。

如果 `phase_count>1`，则用不同 offset 起点做多次卷积，然后把能量平均。可以把它理解为一种多相采样平均：

```text
phase_count = 1:
  offset = 0

phase_count = 2:
  offset = 0
  offset = frame_hop / 2

phase_count = 4:
  offset = 0
  offset = frame_hop / 4
  offset = frame_hop / 2
  offset = 3 * frame_hop / 4
```

这样可以降低单一 stride 起点带来的相位敏感性，但计算量也会近似随 `phase_count` 增加。当前工程默认：

```text
bandpass_phase_count = 1
```

也就是说默认走最低计算量路径。

---

## 14. 输出形状如何和 DSCNN 对齐

以 bandpass 路径为例：

```text
输入 waveform:
  [B, T]

进入 TorchBandpass:
  [B, 1, T]

Conv1D 输出:
  [B, bandpass_n_bands, time_steps]

log-energy 输出:
  [B, bandpass_n_bands, time_steps]

截取 dct_coeff:
  [B, dct_coeff, time_steps]

permute:
  [B, time_steps, dct_coeff]

flatten:
  [B, time_steps * dct_coeff]
```

DSCNN 内部再恢复成二维特征图：

```python
x = x.reshape(batch_size, 1, self.input_time_size, self.input_frequency_size)
```

其中：

```text
input_time_size = input_dim // dct_coeff
input_frequency_size = dct_coeff
```

因此 DSCNN 实际看到的是：

```text
[B, 1, time_steps, dct_coeff]
```

举例 1：`train.py` 的通用 bandpass 配置，如果使用 8 kHz、32 ms stride、16 个 band：

```text
sample_rate = 8000
window_stride_ms = 32
frame_hop = 256
time_steps = 8000 // 256 + 1 = 32
bandpass_n_bands = 16
dct_coeff = 16

TorchBandpass output = [B, 16, 32]
Flatten output       = [B, 32 * 16] = [B, 512]
DSCNN input image    = [B, 1, 32, 16]
```

举例 2：`sweep_fixed_bandpass_noise_snr_scene_acc.py` 的默认配置是 16 kHz、10 个 band：

```text
sample_rate = 16000
window_stride_ms = 32
frame_hop = 512
time_steps = 16000 // 512 + 1 = 32
bandpass_n_bands = 10
dct_coeff = 10

TorchBandpass output = [B, 10, 32]
Flatten output       = [B, 32 * 10] = [B, 320]
DSCNN input image    = [B, 1, 32, 10]
```

---

## 15. 为什么要求 `dct_coeff == bandpass_n_bands`

MFCC 路径里，`dct_coeff` 的含义是：

```text
从 n_mfcc=40 中取前 dct_coeff 个 MFCC 系数。
```

bandpass 路径里，`bandpass_n_bands` 的含义是：

```text
Conv1D 带通滤波器组输出多少个频带。
```

后端 backbone 的 `input_dim` 是用：

```python
input_dim = time_steps * dct_coeff
```

提前构造好的。如果 `bandpass_n_bands` 和 `dct_coeff` 不一致，就会出现前端输出频率维和后端预期频率维不一致的问题。

虽然代码中有：

```python
mfcc = mfcc[:, : self.dct_coeff, :]
```

但如果 `dct_coeff > bandpass_n_bands`，切片也无法凭空产生更多频带；如果 `dct_coeff < bandpass_n_bands`，会丢掉一部分 bandpass 输出。为了避免这种隐式错误，工程在入口处直接检查：

```python
if args.frontend == "bandpass" and args.dct_coeff != args.bandpass_n_bands:
    raise ValueError(...)
```

因此使用 bandpass 时应该显式写成：

```bash
--frontend bandpass \
--dct_coeff 10 \
--bandpass_n_bands 10
```

或者：

```bash
--frontend bandpass \
--dct_coeff 16 \
--bandpass_n_bands 16
```

---

## 16. 通用训练入口中的默认参数

`dscnn_kws/train.py` 中和前端相关的默认参数包括：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--sample_rate` | `8000` | 通用训练默认采样率 |
| `--frontend` | `mfcc` | 默认仍走 MFCC |
| `--dct_coeff` | `13` | 默认取 13 个 MFCC 系数 |
| `--window_size_ms` | `32` | MFCC 路径中为 STFT 窗长 |
| `--window_stride_ms` | `32` | MFCC hop / bandpass stride |
| `--bandpass_n_bands` | `16` | bandpass 默认频带数 |
| `--bandpass_f_min` | `200.0` | bandpass 最低频率 |
| `--bandpass_f_max` | `4000.0` | bandpass 最高频率 |
| `--bandpass_spacing` | `log` | 频带按 log 或 linear 划分 |
| `--bandpass_kernel_size` | `63` | FIR Conv1D 核长度 |
| `--bandpass_phase_count` | `1` | 多相平均数量 |
| `--pre_emphasis` | `True` | 默认开启预加重 |
| `--pre_emphasis_coeff` | `0.97` | 预加重系数 |
| `--log_approx_mode` | `exact` | 默认精确 `torch.log` |
| `--log_offset` | `1e-6` | log 前加偏置 |
| `--log_input_clamp_min` | `1e-12` | log 输入下限 |

注意：由于 `train.py` 默认 `frontend=mfcc` 且 `dct_coeff=13`，如果切换为 `bandpass`，必须同步指定：

```bash
--dct_coeff 16 \
--bandpass_n_bands 16
```

否则 `dct_coeff=13` 和 `bandpass_n_bands=16` 不一致，会触发检查错误。

---

## 17. 固定 bandpass sweep 脚本中的默认参数

`dscnn_kws/sweep_fixed_bandpass_noise_snr_scene_acc.py` 是专门做 bandpass + PWL + 噪声鲁棒性实验的脚本。它的默认前端配置是：

```text
SAMPLE_RATE = 16000
DCT_COEFF = 10
WINDOW_SIZE_MS = 32
WINDOW_STRIDE_MS = 32

BANDPASS_N_BANDS = 10
BANDPASS_F_MIN = 200.0
BANDPASS_F_MAX = 4000.0
BANDPASS_SPACING = "log"
BANDPASS_KERNEL_SIZE = 63
BANDPASS_PHASE_COUNT = 1

LOG_APPROX_MODE = "pwl"
LOG_PWL_NUM_SEGMENTS = 6
LOG_PWL_STRATEGY = "uniform_logx"
LOG_PWL_GAMMA = 1.0
LOG_OFFSET = 1e-6
LOG_INPUT_CLAMP_MIN = 1e-12
```

这套脚本里的重要差异：

- 它默认 `sample_rate=16000`，不是 `train.py` 的 8000。
- 它默认 `dct_coeff=10` 且 `bandpass_n_bands=10`，已经满足 bandpass 约束。
- 它默认 `log_approx_mode=pwl`，更偏硬件友好部署。
- 它默认训练和验证都启用噪声增强/噪声评估，面向鲁棒性实验。
- 它调用 `python -m dscnn_kws.train` 时会强制传入 `--frontend bandpass`。

对应训练命令在脚本内部构造时包含：

```text
--frontend bandpass
--bandpass_n_bands ...
--bandpass_f_min ...
--bandpass_f_max ...
--bandpass_spacing ...
--bandpass_kernel_size ...
--bandpass_phase_count ...
--log_approx_mode ...
--window_size_ms ...
--window_stride_ms ...
```

因此这个脚本是当前工程里最完整的 bandpass 实验入口。

---

## 18. MFCC 和 bandpass 的逐项对比

| 维度 | MFCC 路径 | bandpass Conv1D 路径 |
| --- | --- | --- |
| 输入 | 原始 waveform | 原始 waveform |
| 预加重 | wrapper 中统一支持 | wrapper 中统一支持 |
| 频率分析 | STFT | 时域 FIR Conv1D |
| 滤波器组 | Mel filterbank 矩阵 | sinc 生成的带通卷积核 |
| 能量 | 复数频谱平方 | 卷积响应平方 |
| 频率尺度 | Mel | log/linear Hz 边界 |
| log | 支持 exact/PWL | 支持 exact/PWL |
| DCT | 有 | 无 |
| 输出含义 | MFCC 系数 | log band energy |
| 输出形状 | `[B, n_mfcc, time]` | `[B, n_bands, time]` |
| 后端输入 | `[B, time * dct_coeff]` | `[B, time * dct_coeff]` |
| 部署复杂度 | STFT + Mel + DCT 较复杂 | Conv1D + square + log 更直接 |
| 当前卷积核是否可训练 | 不适用 | 否，当前为固定 buffer |

---

## 19. 为什么这个替换是合理的

KWS 模型的目标是判断 1 秒左右音频中是否出现关键词。对这类任务，模型通常需要的是：

- 音素/音节相关的短时频谱结构。
- 不同时间位置上的能量变化。
- 不同频带之间的相对分布。
- 对幅度变化有一定鲁棒性的压缩表示。

MFCC 是一种成熟的手工设计特征，但 DSCNN/LSTM 并不一定必须依赖 DCT 后的 MFCC 系数。很多现代语音模型也直接使用 log-Mel filterbank 或更轻量的可学习/半可学习前端。

本工程的 bandpass 前端抓住了 MFCC 前端最关键的部分：

```text
短时频带能量 + log 动态范围压缩
```

它省掉了：

```text
STFT 复数频谱
Mel 矩阵乘法
DCT 矩阵乘法
```

但保留了：

```text
频带划分
短时响应
能量平方
log 压缩
二维 time-frequency 输入形态
```

因此从后端网络视角看，它仍然接收一个类似时频图的输入，只是频率维不再是 MFCC cepstral coefficients，而是 bandpass log-energy bands。

---

## 20. 为什么它更适合硬件/嵌入式部署

传统 MFCC 的硬件实现通常要处理：

- 分帧和窗函数。
- FFT/STFT。
- 复数乘加。
- 幅度平方。
- Mel filterbank 矩阵乘法。
- log。
- DCT。

这些步骤当然都能做硬件实现，但链路长、控制复杂、数值格式边界多。

bandpass 前端把主要计算变成：

```text
1D FIR convolution
square
average
log 或 PWL log
```

硬件友好点在于：

- FIR Conv1D 可以用移位寄存器 + MAC 阵列实现。
- 滤波器系数固定时，可以固化在 ROM/参数表中。
- `square` 是明确的定点乘法。
- `log` 可以用 PWL 分段线性近似。
- 输出维度和后端 DSCNN 输入接口保持稳定。

也就是说，这个前端为后续从 PyTorch 浮点模型走向 Q 格式仿真、bit-accurate fixed-point、Verilog/RTL 留出了更直接的路径。

需要注意的是：当前 PyTorch 代码仍是浮点实现，PWL log 只是让 log 更硬件友好；如果要严格硬件等价，还需要继续做定点系数量化、累加位宽、舍入/饱和策略和 bit-accurate 仿真。

---

## 21. 典型训练命令

### 21.1 使用通用训练入口跑 bandpass exact log

8 kHz、16 band、精确 log：

```bash
python -m dscnn_kws.train \
  --root ./dscnn_kws/data \
  --dataset speech_commands_v0.02_sr8k \
  --sample_rate 8000 \
  --frontend bandpass \
  --dct_coeff 16 \
  --bandpass_n_bands 16 \
  --bandpass_f_min 200.0 \
  --bandpass_f_max 4000.0 \
  --bandpass_spacing log \
  --bandpass_kernel_size 63 \
  --bandpass_phase_count 1 \
  --window_size_ms 32 \
  --window_stride_ms 32 \
  --log_approx_mode exact \
  --epoch 50 \
  --batch 256
```

### 21.2 使用通用训练入口跑 bandpass PWL log

```bash
python -m dscnn_kws.train \
  --root ./dscnn_kws/data \
  --dataset speech_commands_v0.02_sr8k \
  --sample_rate 8000 \
  --frontend bandpass \
  --dct_coeff 16 \
  --bandpass_n_bands 16 \
  --bandpass_f_min 200.0 \
  --bandpass_f_max 4000.0 \
  --bandpass_spacing log \
  --bandpass_kernel_size 63 \
  --bandpass_phase_count 1 \
  --window_size_ms 32 \
  --window_stride_ms 32 \
  --log_approx_mode pwl \
  --log_pwl_num_segments 6 \
  --log_pwl_strategy uniform_logx \
  --log_pwl_gamma 1.0 \
  --epoch 50 \
  --batch 256
```

### 21.3 使用 Mobvoi 16 kHz、10 band 配置

这更接近 `sweep_fixed_bandpass_noise_snr_scene_acc.py` 的默认配置：

```bash
python -m dscnn_kws.train \
  --root ./dscnn_kws/data \
  --dataset mobvoi_hi_xiaowen_binary_hardneg \
  --sample_rate 16000 \
  --frontend bandpass \
  --dct_coeff 10 \
  --bandpass_n_bands 10 \
  --bandpass_f_min 200.0 \
  --bandpass_f_max 4000.0 \
  --bandpass_spacing log \
  --bandpass_kernel_size 63 \
  --bandpass_phase_count 1 \
  --window_size_ms 32 \
  --window_stride_ms 32 \
  --log_approx_mode pwl \
  --log_pwl_num_segments 6 \
  --log_pwl_strategy uniform_logx \
  --log_pwl_gamma 1.0 \
  --allow_online_resample \
  --epoch 30 \
  --batch 256
```

### 21.4 使用固定 bandpass 噪声鲁棒性 sweep

```bash
python dscnn_kws/sweep_fixed_bandpass_noise_snr_scene_acc.py \
  --root ./dscnn_kws/data \
  --datasets mobvoi_hi_xiaowen_binary_hardneg mobvoi_nihao_wenwen_binary_hardneg \
  --sample_rate 16000 \
  --dct_coeff 10 \
  --bandpass_n_bands 10 \
  --log_approx_mode pwl
```

这个脚本会训练多个 DSCNN 架构，并在不同 SNR/TAU 场景下评估 ACC/F1，输出 CSV 和 best model。

---

## 22. 典型评估命令

`eval_fah_frr.py` 也支持同样的 bandpass 参数。评估 checkpoint 时，前端参数必须和训练时保持一致：

```bash
python -m dscnn_kws.eval_fah_frr \
  --root ./dscnn_kws/data \
  --dataset mobvoi_hi_xiaowen_binary_hardneg \
  --ckpt dscnn_kws/runs/xxx/best.pt \
  --sample_rate 16000 \
  --frontend bandpass \
  --dct_coeff 10 \
  --bandpass_n_bands 10 \
  --bandpass_f_min 200.0 \
  --bandpass_f_max 4000.0 \
  --bandpass_spacing log \
  --bandpass_kernel_size 63 \
  --bandpass_phase_count 1 \
  --window_size_ms 32 \
  --window_stride_ms 32 \
  --log_approx_mode pwl \
  --target_fah 1.0
```

如果训练时用了 `exact`，评估时也要用 `exact`；如果训练时用了 `pwl`，评估时也要用 `pwl`。如果训练时是 10 band，评估时也必须是 10 band。

---

## 23. 参数选择的实际含义

### 23.1 `bandpass_n_bands`

控制频率维大小，也是 Conv1D 输出通道数。

更大：

- 频率分辨率更高。
- 后端输入维度更大。
- 计算和参数压力可能增加。

更小：

- 前端更轻。
- 输入维度更小。
- 可能损失频率细节。

在本工程中，常见设置：

```text
10 bands: 适合和 MFCC 10 x 32 输入对齐，后端输入 320 维。
16 bands: 更细一些，后端输入 512 维。
```

### 23.2 `bandpass_f_min`

最低频率边界。默认 `200 Hz`。

原因是非常低频区域可能更多包含直流漂移、环境低频、麦克风低频噪声等，对关键词区分未必最有效。把最低频率抬到 200 Hz 可以让有限的 band 更集中在语音有效区域。

### 23.3 `bandpass_f_max`

最高频率边界。默认 `4000 Hz`。

对于 8 kHz 音频，Nyquist 频率就是 4000 Hz；对于 16 kHz 音频，4000 Hz 只使用到一半 Nyquist。这样做的直觉是关键词识别的主要语音信息集中在较低到中高频区域，限制到 4 kHz 可以降低高频噪声影响和计算压力。

如果希望利用 16 kHz 音频中 4 kHz 以上的信息，可以把它提高，但必须满足：

```text
bandpass_f_max <= sample_rate / 2
```

### 23.4 `bandpass_spacing`

支持：

```text
log
linear
```

`log` 更接近听觉尺度，低频更密；`linear` 每个 band 的 Hz 宽度相同。当前默认 `log`。

### 23.5 `bandpass_kernel_size`

控制 FIR 滤波器长度，默认 `63`。

它必须是奇数，因为代码要求线性相位 FIR：

```python
if kernel_size % 2 == 0:
    raise ValueError(...)
```

更大的 kernel：

- 频率选择性更好。
- 计算量更高。
- 延迟和缓存需求更大。

更小的 kernel：

- 计算更轻。
- 频率响应更粗。
- 低频窄带滤波能力可能变差。

### 23.6 `bandpass_phase_count`

默认 `1`。增大后会用多个 offset 做卷积并平均，降低 stride 起点敏感性，但计算量增加。

### 23.7 `log_approx_mode`

支持：

```text
exact
pwl
```

`exact` 使用 `torch.log`，适合训练基线和浮点评估。

`pwl` 使用分段线性近似，更适合硬件部署思路。它需要：

```text
breakpoints
slopes
intercepts
```

如果没有提供 JSON，代码会用默认采样范围自动拟合一组 PWL 参数。

---

## 24. 当前实现的边界和注意事项

### 24.1 当前 bandpass 卷积核固定，不参与训练

因为核是 buffer，不是 `nn.Parameter`。这保证了前端可解释、部署更直接，但也意味着模型不能自动学习更适合数据集的滤波器形状。

如果后续希望做 learnable frontend，可以考虑：

- 把 `bandpass_kernels` 改成 `nn.Parameter`。
- 或者只学习每个 band 的 low/high 边界，再动态生成 sinc kernels。
- 或者在固定 bandpass 后面加轻量可训练 1D/1x1 mixing。

但这会改变当前“固定可解释前端”的设计。

### 24.2 当前 `frame_length` 没有参与 bandpass 能量窗口

`TorchBandpass` 保存了 `frame_length`，但 forward 里没有用它做传统意义上的 framing/windowing。时间采样主要由 `frame_hop` 和 `adaptive_avg_pool1d` 对齐控制。

如果希望更接近传统短时能量统计，可以进一步设计：

- 每个 hop 内做局部平均池化。
- 或者 Conv1D stride 更小，再用 frame window 聚合。
- 或者使用 depthwise temporal pooling 模拟帧内能量。

当前实现为了轻量化，采用的是 stride Conv1D 后直接平方和对齐。

### 24.3 当前没有 DCT

bandpass 输出是 log band energy，不是 MFCC cepstrum。变量名仍然叫 `mfcc` 是历史命名复用，不代表语义仍是 MFCC。

更准确的说法：

```text
MFCC path output: cepstral coefficients
bandpass path output: log bandpass energies
```

### 24.4 训练和评估参数必须一致

由于 bandpass kernels 是根据参数重建的，checkpoint 里不保存这些 kernels。因此加载模型评估时，必须传入和训练时一致的：

```text
sample_rate
window_stride_ms
dct_coeff
bandpass_n_bands
bandpass_f_min
bandpass_f_max
bandpass_spacing
bandpass_kernel_size
bandpass_phase_count
log_approx_mode
log_pwl 参数
pre_emphasis 设置
```

否则即使 backbone 权重相同，前端生成的特征也会变，评估结果不可比。

---

## 25. 一句话总结

本工程的 bandpass 前端不是简单地把 MFCC 换成一个普通 1D CNN，而是实现了一个：

```text
pre-emphasis
+ fixed sinc FIR Conv1D bandpass filterbank
+ square power
+ optional multi-phase average
+ exact/PWL log compression
```

的轻量 KWS 前端。它把传统 MFCC 中的 `STFT + Mel filterbank + DCT` 替换为更硬件友好的 `Conv1D filterbank + energy + log`，同时保持输出形状仍然能被原来的 DSCNN/LSTM backbone 接收。

最关键的工程约束是：

```text
--frontend bandpass 时：
  --dct_coeff 必须等于 --bandpass_n_bands
```

最关键的结构参数是：

```text
Conv1D:
  in_channels  = 1
  out_channels = bandpass_n_bands
  kernel_size  = bandpass_kernel_size
  stride       = sample_rate * window_stride_ms / 1000
  padding      = 0
  bias         = False
  weights      = fixed sinc FIR kernels, shape [bandpass_n_bands, 1, bandpass_kernel_size]
```

在 16 kHz、10 band、32 ms stride 的 sweep 默认配置下，最终送入 DSCNN 的特征是：

```text
TorchBandpass output = [B, 10, 32]
DSCNN flattened input = [B, 320]
DSCNN internal image  = [B, 1, 32, 10]
```

这就是本工程用 bandpass Conv1D 前端替换 MFCC 前端的完整实现逻辑。
