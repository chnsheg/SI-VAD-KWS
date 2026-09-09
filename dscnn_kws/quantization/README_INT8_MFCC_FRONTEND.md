# INT8 MFCC Frontend + INT8 QAT Backbone

本说明对应新增代码：

```text
dscnn_kws/frontend/int8_mfcc_frontend.py
dscnn_kws/quantization/qat_int8_mfcc_frontend_noise_snr_scene.py
```

这套流程用于在噪声场景训练模型的基础上：

```text
噪声场景 best.pt
  -> DSCNN backbone 做 INT8 QAT fine-tune
  -> convert 得到 PyTorch INT8 quantized backbone
  -> 用校准集导出 INT8 MFCC 前端各阶段 scale
  -> 用 INT8 MFCC 前端 + INT8 backbone 做 tau_test 和 10 scene x 5 SNR 测试
```

## 1. 这次前端量化做了什么

`Int8MFCCFrontend` 是一个整数感知的 MFCC Python 参考实现：

```text
waveform
  -> int8 quantize
  -> pre-emphasis
  -> int8 requantize
  -> framing + Hann window int8 coefficient
  -> integer DFT, int64 accumulate
  -> power
  -> integer Mel filterbank, int64 accumulate
  -> log / PWL log
  -> int8 requantize
  -> integer DCT, int64 accumulate
  -> int8 requantize
  -> dequantized float MFCC tensor for backbone input
```

注意：当前脚本默认已经把 `log` 这一步设为 PWL，用来更接近后续硬件里的 LUT/PWL log：

```bash
--int8_mfcc_log_approx_mode pwl \
--int8_mfcc_log_pwl_num_segments 8
```

最终上 Verilog 时，DFT/Mel/DCT 可以按这里的 int8 系数和 int64/定点累加逻辑实现；log 建议用 LUT 或 PWL 单独实现。

默认不把 `power` 和 `mel` 这两个能量中间量强制重映射到 int8，因为它们动态范围很大，且 `mel` 后面马上进入 log。如果在 log 前用全局线性 int8 压缩能量，很多小能量会变成 0，容易导致 MFCC 近似常数、模型退化。当前默认是：

```text
INT8_MFCC_REQUANTIZE_POWER False
INT8_MFCC_REQUANTIZE_MEL   False
```

如果你想做更激进的“每个阶段都 int8”实验，可以显式开启：

```bash
--int8_mfcc_requantize_power \
--int8_mfcc_requantize_mel
```

但这通常会明显损伤 MFCC 前端。

## 2. 这里的 INT8 分别指什么

这套脚本里有两个地方都叫 INT8，但含义不完全一样：

```text
QAT 阶段:
  TorchMFCC + PWL 前端 fake quant
  + DSCNN backbone INT8 QAT fake quant

最终测试阶段:
  Int8MFCCFrontend 整数感知前端
  + convert 后的 PyTorch INT8 quantized backbone
```

也就是说，最终测试时的 `int8_mfcc_int8_backbone` 不是简单地“全模型所有张量永远都是 int8”。更准确的说法是：

- 前端用手写的 `Int8MFCCFrontend`，在关键阶段把张量和固定系数量化到 INT8 网格，并用较宽 accumulator 做矩阵乘/累加。
- Backbone 用 PyTorch eager quantization 的 `convert()`，把 DSCNN backbone 变成 PyTorch 支持的 INT8 quantized Conv/Linear/ReLU 等模块。
- 前端输出给 backbone 前，仍会反量化成 float MFCC tensor；随后 backbone 内部的 `QuantStub` 会把它重新量化成 PyTorch quantized tensor。

### 2.1 前端 INT8 指什么

`Int8MFCCFrontend` 中的 INT8 主要指 **前端各阶段张量和固定系数被约束到 8-bit 整数网格**。代码为了方便 PyTorch 计算，很多整数张量实际存成 `torch.int32` 或 `torch.int64`，但数值范围按 INT8 限制。

默认量化范围：

```text
有符号阶段: qmin=-128, qmax=127
非负阶段:   qmin=0,    qmax=127
```

各阶段大致如下：

```text
waveform:
  qrange = [-128, 127]
  默认 scale = 1 / 127
  含义: 输入波形先量化到 INT8 网格

preemphasis:
  qrange = [-128, 127]
  默认 scale = 2 / 127
  含义: pre-emphasis 后再次量化

window / DFT / Mel / DCT 固定系数:
  qrange = [-128, 127]
  coeff_bits = 8
  存储: Python 中保存为 int32 tensor
  含义: Hann window、DFT cos/sin、Mel filterbank、DCT matrix 都按对称 INT8 系数量化

windowed:
  qrange = [-128, 127]
  scale = 校准/observer 得到
  含义: waveform frame 乘 window 后重新量化

power:
  qrange = [0, 127]
  默认只统计 scale，不强制 requantize
  原因: power 动态范围很大，过早压到 INT8 容易把小能量压成 0

mel:
  qrange = [0, 127]
  默认只统计 scale，不强制 requantize
  原因: mel energy 后面马上进入 log，过早 INT8 压缩会严重损伤动态范围

log_mel:
  qrange = [-128, 127]
  scale = 校准/observer 得到
  含义: log 或 PWL log 输出后量化

mfcc:
  qrange = [-128, 127]
  scale = 校准/observer 得到
  含义: DCT 后的 MFCC 输出量化，然后反量化为 float32 返回给 backbone
```

默认：

```text
INT8_MFCC_COEFF_BITS       8
INT8_MFCC_REQUANTIZE_POWER False
INT8_MFCC_REQUANTIZE_MEL   False
```

所以默认前端不是“每一个中间量都强制 int8”。更准确地说：

```text
waveform / preemphasis / windowed / log_mel / mfcc:
  做 INT8 requantize

power / mel:
  默认保持宽动态范围中间量，只统计 scale，不压回 INT8
```

### 2.2 前端中间结果是什么数值类型

`Int8MFCCFrontend` 是 Python 参考实现，不是最终 RTL。因此它会用较宽的 PyTorch dtype 避免溢出，方便对齐数值：

```text
量化后的 waveform/preemphasis/windowed/log_mel/mfcc:
  数值范围按 INT8 限制
  Python 存储常用 int32

windowed_acc:
  int64
  来自 waveform/frame INT8 x window INT8

DFT real_acc / imag_acc:
  int64
  来自 windowed INT8 x DFT cos/sin INT8 的累加

power_acc:
  int64
  来自 real_acc^2 + imag_acc^2

mel_acc:
  int64
  来自 power 或 power_q 与 Mel INT8 系数的矩阵乘

log / PWL log:
  在 dequantized mel float 上计算
  然后 log_mel 再量化到 INT8 网格

mfcc_acc:
  int64
  来自 log_mel INT8 x DCT INT8 的矩阵乘

最终 frontend 输出:
  mfcc_q * mfcc_scale
  即 dequantized float32 MFCC tensor
```

后续如果写 Verilog，不一定要照搬 `int64`，而是要根据最大输入幅度、乘法项数量、是否截断/饱和来设计 accumulator 位宽。这里的 `int64` 是为了让 Python 参考实现尽量不被溢出污染。

### 2.3 Backbone INT8 指什么

Backbone 的 INT8 来自 PyTorch QAT + `convert()`：

```text
prepare_qat:
  插入 fake quant / observer
  训练时权重仍是 float 参数
  forward 中模拟量化误差

convert:
  把 backbone 转成 PyTorch quantized modules
  Conv/Linear 权重被打包成 INT8 quantized weights
  激活在 quantized ops 之间以 PyTorch quantized tensor 形式传递
```

当前脚本使用：

```python
get_default_qat_qconfig(args.backend)
```

默认 backend 是 `fbgemm`。在这种配置下，通常可以这样理解：

```text
Backbone 输入:
  前端输出的 float32 MFCC
  进入 QuantStub 后量化成 PyTorch quantized activation

Backbone 权重:
  convert 后是 qint8 quantized weights
  Conv/Linear 权重通常使用 per-channel scale
  具体 scale / zero_point 由 PyTorch observer 统计得到

Backbone 激活:
  通常是 quint8 affine quantized activation
  每个 quantized op 有自己的 activation scale / zero_point
  具体 qmin/qmax 由 PyTorch backend/qconfig 决定

Conv/Linear 乘加中间结果:
  通常使用 int32 accumulator
  再根据输出 scale / zero_point requantize 到下一层 activation

Bias:
  PyTorch quantized module 中通常仍以 float32 形式保存
  内部计算时等效到 input_scale * weight_scale 对应的累加域

Backbone 输出:
  DeQuantStub 把 quantized logits 反量化成 float32
```

所以 backbone 的 INT8 是 PyTorch 真正支持的 quantized op 路径；它比前端的 fake quant 更接近实际 INT8 推理。但它仍然依赖 PyTorch 后端实现，具体 packed weight 格式和 kernel 细节不等同于你将来自己写的 Verilog。

### 2.4 一句话对应关系

```text
前端固定系数:
  INT8 网格，Python 中多用 int32 保存

前端阶段激活:
  INT8 网格，部分阶段默认 requantize，power/mel 默认不强制 requantize

前端矩阵乘/累加:
  int64 accumulator，随后乘 scale 反量化或 requantize

前端最终输出:
  float32 MFCC tensor

Backbone 权重:
  convert 后的 PyTorch qint8 quantized weights

Backbone 激活:
  PyTorch quantized activation，通常是 quint8 affine

Backbone Conv/Linear 累加:
  通常 int32 accumulator

Backbone 最终输出:
  DeQuantStub 后的 float32 logits
```

## 3. MFCC 前端算子级位宽规定

本节把当前 `Int8MFCCFrontend` 默认配置进一步落成硬件/RTL 视角的位宽草案。默认参数为：

```text
sample_rate 16000
n_fft       512
win_length  512
hop_length  512
n_freq      257  # n_fft / 2 + 1
n_mels      40
n_mfcc      40
coeff_bits  8
```

记号：

```text
S8  = signed 8-bit,  range [-128, 127]
U8  = unsigned 8-bit, range [0, 255]
U7  = unsigned 7-bit, range [0, 127]
S24 = signed 24-bit accumulator
U48 = unsigned 48-bit accumulator
U64 = unsigned 64-bit accumulator
```

这里的位宽分成两类：

```text
逻辑位宽:
  算子数值真实需要的范围，例如 Mel 系数实际非负，可按 U7 看待。

存储位宽:
  为了复用存储/ROM，也可以统一存成 S8。非负系数虽然存成 S8，但 RTL 乘法时建议按 U7/U8 解释。
```

### 3.1 默认推荐位宽表

| 阶段 | 算子 | 输入位宽 | 系数位宽 | 乘积/中间位宽 | 累加位宽 | 输出位宽 | 说明 |
|---|---|---:|---:|---:|---:|---:|---|
| ADC/归一化后波形 | waveform quantize | float/PCM | - | - | - | S8 | 当前代码按 `scale=1/127` 量化到 `waveform_q` |
| 预加重 | `y[n]=x[n]-0.97*x[n-1]` | S8 | `alpha_q` U8, 建议 Q0.7 | S16 | S16 | S8 | 预加重后用 `preemphasis_scale` 饱和回 S8 |
| 分帧/补边 | reflect pad + unfold | S8 | - | - | - | S8 | 只搬移数据，不改变位宽 |
| 加窗 | frame x Hann | S8 | Hann U7/S8 | S16 | - | S8 | 乘积经 `windowed_scale` 重新量化为 `windowed_q` |
| DFT real | windowed x cos | S8 | cos S8 | S16 | S24 | S24 | 512 点累加，`512 * 128 * 127` 可放入 S24 |
| DFT imag | windowed x -sin | S8 | sin S8 | S16 | S24 | S24 | 同 real |
| 功率谱 | `real^2 + imag^2` | S24 | - | U48 | U48 | U48 | 严格最坏情况约需 47 bit，建议用 U48 留 1 bit 余量 |
| Power requant 可选 | power -> power_q | U48 | - | scale/requant | - | U8 | 仅开启 `--int8_mfcc_requantize_power` 时使用；默认不开 |
| Mel 默认路径 | power x Mel filterbank | U48 | Mel U7/S8 | U55 | U64 | U64 | 默认不先压缩 power，Mel 累加建议 U64 |
| Mel 激进路径 | power_q x Mel filterbank | U8 | Mel U7/S8 | U15 | U24 | U24 | 开启 power requant 后可显著降低 Mel 累加位宽 |
| Mel requant 可选 | mel -> mel_q | U64 或 U24 | - | scale/requant | - | U8 | 仅开启 `--int8_mfcc_requantize_mel` 时使用；默认不开 |
| Log/PWL log | log(mel) 或 PWL log | U64 默认 / U8 可选 | PWL 参数建议 S32 Q16.16 | S64 或内部定点 | - | S8 | 当前 Python 在 dequantized float 上算 log，RTL 建议独立 LUT/PWL，输出固定为 `log_mel_q` S8 |
| DCT | log_mel x DCT | S8 | DCT S8 | S16 | S21 | S21 | 40 mel bin 累加，`40 * 128 * 127` 需要 S21 |
| MFCC 输出量化 | mfcc -> mfcc_q | S21 对应反量化值 | - | scale/requant | - | S8 | `mfcc_q` 是前端最终 INT8 MFCC |
| 给 PyTorch backbone | dequantize | S8 | `mfcc_scale` | float32 | - | float32 | 当前代码返回 `mfcc_q * mfcc_scale`；若纯 RTL 后端，可直接传 S8 + scale |

默认建议的 RTL 主路径可以简化成：

```text
waveform_q        S8
preemphasis_q     S8
windowed_q        S8
dft_real/imag     S24
power_acc         U48
mel_acc           U64
log_mel_q         S8
mfcc_acc          S21
mfcc_q            S8
```

如果后续明确要做“全阶段强制 INT8 中间量”的低资源版本，则主路径变成：

```text
waveform_q        S8
preemphasis_q     S8
windowed_q        S8
dft_real/imag     S24
power_q           U8
mel_acc           U24
mel_q             U8
log_mel_q         S8
mfcc_acc          S21
mfcc_q            S8
```

但这个低资源版本会牺牲 power/mel 的动态范围，和当前默认实验结论不完全等价。

### 3.2 固定系数位宽

| 系数 | 数学范围 | 逻辑位宽 | 建议存储 | scale 来源 |
|---|---:|---:|---:|---|
| pre-emphasis `alpha=0.97` | `[0, 1]` | U8 Q0.7 | U8 | 可固定为 `round(0.97 * 128)=124` |
| Hann window | `[0, 1]` | U7 | S8/ROM8 | `window_scale` |
| DFT cos | `[-1, 1]` | S8 | S8/ROM8 | `dft_cos_scale` |
| DFT -sin | `[-1, 1]` | S8 | S8/ROM8 | `dft_sin_scale` |
| Mel filterbank | `[0, 1]` 稀疏非负 | U7 | S8/ROM8 | `mel_coeff_scale` |
| DCT matrix | 有正有负 | S8 | S8/ROM8 | `dct_coeff_scale` |

系数 ROM 统一 8 bit 最省事；计算时对 Hann/Mel 这类非负系数按 unsigned 解释，可以少 1 bit 符号扩展压力。

### 3.3 Requantize 规则

所有需要压回 8 bit 的阶段都按同一类规则处理：

```text
q = round(x / scale)
q = clamp(q, qmin, qmax)
```

对应位宽规定：

| 阶段 | qmin/qmax | 输出位宽 | scale 来源 |
|---|---:|---:|---|
| waveform | `[-128, 127]` | S8 | 固定 `1/127` |
| preemphasis | `[-128, 127]` | S8 | 固定 `2/127` |
| windowed | `[-128, 127]` | S8 | 校准集 observer |
| power 可选 | `[0, 127]` | U8 | 校准集 observer |
| mel 可选 | `[0, 127]` | U8 | 校准集 observer |
| log_mel | `[-128, 127]` | S8 | 校准集 observer |
| mfcc | `[-128, 127]` | S8 | 校准集 observer |

硬件中可以把 `1/scale` 或相邻阶段的 scale ratio 预先转成定点乘子。建议先用 S32 Q16.16 或 S32 Q0.31 做 requant multiplier；乘法结果右移后再饱和到 S8/U8。最终具体选 Q16.16 还是 Q0.31，应以导出的 `*_int8_mfcc_scales.json` 中 scale 的实际范围为准。

### 3.4 位宽公式，便于改参数后重算

如果以后把 `n_fft`、`n_mels` 或 `coeff_bits` 改掉，可以按下面公式重算累加器位宽：

```text
signed_acc_bits(num_terms, in_absmax, coeff_absmax)
  = 1 + ceil(log2(num_terms * in_absmax * coeff_absmax + 1))

unsigned_acc_bits(num_terms, in_max, coeff_max)
  = ceil(log2(num_terms * in_max * coeff_max + 1))
```

套到当前默认配置：

```text
DFT:
  num_terms    = 512
  in_absmax    = 128
  coeff_absmax = 127
  signed_acc_bits = 24

Power:
  real/imag S24
  real^2 + imag^2 建议 U48

Mel 默认路径:
  power_acc U48
  mel_coeff U7
  freq_terms = 257
  建议 U64

DCT:
  num_terms    = 40
  in_absmax    = 128
  coeff_absmax = 127
  signed_acc_bits = 21
```

注意：Mel filterbank 实际是稀疏三角滤波器，不是 257 个频点全为最大值；所以 U64 是保守上界。后续如果要压面积，可以按每个 mel bin 的非零 tap 数和最大系数重新做 per-bin accumulator 位宽。

### 3.5 当前 Python 参考实现和 RTL 的差异

当前 Python 代码为了避免数值调试时被溢出干扰，很多地方直接用 `torch.int64`：

```text
windowed_acc / dft_acc / power_acc / mel_acc / mfcc_acc:
  Python 中多用 int64
```

RTL 不需要全部照搬 int64。按上面的默认位宽，真正关键的是：

```text
DFT accumulator 不能低于 S24
Power accumulator 建议 U48
默认 Mel accumulator 建议 U64
DCT accumulator 不能低于 S21
```

如果采用 `power_q` 和 `mel_q` 的激进压缩路径，Mel 后半段位宽会明显下降，但需要重新跑精度评估确认损失。

## 4. 默认输入输出

默认输入 checkpoint 目录：

```text
/root/kws/dscnn_kws/dscnn_kws/runs/snr_scene_arch_sweep_best_models
```

默认数据目录：

```text
/root/kws/dscnn_kws/dscnn_kws/data
```

默认输出：

```text
dscnn_kws/quantization/qat_int8_mfcc_frontend_models/
dscnn_kws/quantization/qat_int8_mfcc_frontend_train_results.csv
dscnn_kws/quantization/qat_int8_mfcc_frontend_grid_results.csv
```

每个模型会输出：

```text
*_qat_prepared_best.pt
*_qat_int8_backbone.pt
*_int8_mfcc_scales.json
*_int8_mfcc_int8_backbone.pt
```

其中 `*_int8_mfcc_scales.json` 是前端各阶段 scale 和系数 scale，后续做硬件验证时很重要。

## 5. 快速试跑

在服务器项目外层目录运行：

```bash
cd /root/kws/dscnn_kws
```

先试跑一个模型：

```bash
python dscnn_kws/quantization/qat_int8_mfcc_frontend_noise_snr_scene.py \
  --limit 1 \
  --qat_epochs 3 \
  --batch 128 \
  --gpu 1
```

完整跑默认目录：

```bash
python dscnn_kws/quantization/qat_int8_mfcc_frontend_noise_snr_scene.py \
  --qat_epochs 10 \
  --batch 256 \
  --num_workers 8 \
  --gpu 1
```

如果只跑某一个 checkpoint：

```bash
python dscnn_kws/quantization/qat_int8_mfcc_frontend_noise_snr_scene.py \
  --checkpoints /root/kws/dscnn_kws/dscnn_kws/runs/snr_scene_arch_sweep_best_models/<model>.pt \
  --dataset mobvoi_hi_xiaowen_binary_hardneg \
  --qat_epochs 5
```

## 6. 默认噪声设置

QAT 训练：

```text
train_noise_roots  ./dscnn_kws/noise/lists/tau_train.txt
train_noise_prob   0.8
train_snr          -5 ~ 20 dB
```

验证和普通测试：

```text
validation noise   ./dscnn_kws/noise/lists/tau_valid.txt, 5 dB
test noise         ./dscnn_kws/noise/lists/tau_test.txt, 5 dB
```

场景测试：

```text
scene_test_root    ./dscnn_kws/noise/tau
scene_names        airport bus metro metro_station park public_square shopping_mall street_pedestrian street_traffic tram
test_snrs          20 10 5 0 -5
```

前端 scale 标定默认使用：

```text
calibration_split       validation
calibration_noise_roots ./dscnn_kws/noise/lists/tau_valid.txt
calibration_snr         -5 ~ 20 dB
```

这样 scale 会覆盖完整 SNR 范围。如果想严格和 validation 的 5 dB 一致：

```bash
--calibration_snr_min_db 5 \
--calibration_snr_max_db 5
```

## 7. 关键参数

最常改的是脚本开头的这些变量：

```text
QAT_EPOCHS
LR
BATCH
INPUT_DIR
OUTPUT_DIR
INT8_MFCC_MEL_LOG_MODE
INT8_MFCC_LOG_APPROX_MODE
INT8_MFCC_REQUANTIZE_POWER
INT8_MFCC_REQUANTIZE_MEL
LOG_APPROX_MODE
CALIBRATION_SNR_MIN_DB / CALIBRATION_SNR_MAX_DB
```

当前默认会让两个阶段都使用 PWL：

```text
MFCC_IMPL                 torch
LOG_APPROX_MODE           pwl
LOG_PWL_NUM_SEGMENTS      8
INT8_MFCC_MEL_LOG_MODE    natural_log
INT8_MFCC_LOG_APPROX_MODE pwl
INT8_MFCC_REQUANTIZE_POWER False
INT8_MFCC_REQUANTIZE_MEL   False
SAMPLE_RATE               16000
DCT_COEFF                 10
WINDOW_SIZE_MS            32
WINDOW_STRIDE_MS          32
PRE_EMPHASIS              True
```

如果 checkpoint 来自原来的 `sweep_fixed_dscnn_noise_snr_scene_acc.py`，原始训练前端通常是 `torchaudio`。新脚本会优先尝试完整加载 state_dict；如果前端 buffer 不匹配，会自动只加载 DSCNN backbone 权重，然后用 `TorchMFCC + PWL` 前端继续 QAT fine-tune。

如果你只是想模拟原始 torchaudio MFCC，而不是两个阶段都 PWL，可以改回：

```bash
--mfcc_impl torchaudio \
--int8_mfcc_mel_log_mode torchaudio_db \
--int8_mfcc_log_approx_mode exact
```

## 8. 和旧 QAT 脚本的区别

旧脚本 `qat_int8_noise_snr_scene.py`：

```text
frontend: 只在前端路径几个张量上做 fake-quant
backbone: QAT 后 convert 成 PyTorch INT8 quantized ops
```

新脚本：

```text
QAT 阶段: 使用 TorchMFCC + PWL，并在前端路径加 fake-quant 约束训练
最终测试: 把前端替换成 Int8MFCCFrontend + PWL，再接 convert 后的 INT8 backbone
```

所以新脚本测到的是更接近“前端也真正按 int8 网格运行”的精度。



