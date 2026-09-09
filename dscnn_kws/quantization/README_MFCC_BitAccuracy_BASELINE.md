• 先给一版 baseline，后面可以逐项改。

  输入与加窗

  audio_sram / pcm input: signed int8
  MFCC internal sample: signed 10 or 12
  Hann coeff: unsigned 16, Q0.16
  window product raw: signed 28  // 12 x 16
  windowed sample: signed 18     // round/shift 后送 FFT

  我建议 int8 PCM -> signed 12，留 4 bit 余量给前端缩放/可选预加重；如果不做预加重，10 bit 也够。

  FFT

  FFT input: signed 18 complex
  FFT internal Re/Im: signed 20
  Twiddle coeff: signed 16, Q1.15
  Complex multiply raw:
    20 x 16 -> 36
    real/imag combine -> 37
  Butterfly add/sub + stage scaling:
    output back to signed 20
  FFT stage scaling:
    每级或按 schedule 右移，避免 9 级增长
  RFFT output: signed 20 complex

  Power

  Re/Im: signed 20
  Re^2 raw: unsigned 40
  Im^2 raw: unsigned 40
  Re^2 + Im^2 raw: unsigned 41
  power output: unsigned 32 or 40

  这里建议 power 内部保持 41bit，输出到 Mel 前可以不急着压到 32。如果 Mel accumulator 想简单，保留：

  POWER_W = 41

  Mel
  当前 rectangular Mel，单个 bin 最多命中 2 个 band，但每个 band 最多累计 32 个 bin：

  Mel accumulator: POWER_W + 5 = 46
  mel_acc[0..39]: unsigned 46
  Mel drain to log: unsigned 46

  如果想减 PWL log 复杂度，可以在 Mel drain 处做 block floating 或固定右移：

  mel_log_input: unsigned 32

  但要记录 shift，因为会影响 log 偏移。

  PWL Log

  log input: unsigned 32 or 46
  PWL breakpoint compare: same width as log input
  PWL slope coeff: signed 16 or 24
  PWL intercept: signed 32
  a*x raw:
    32 x 16 -> 48   // 若先压到 32
  log_mel output: signed 24 or signed 32

  建议第一版：

  PWL_IN_W = 32
  LOG_W = 24

  DCT

  log_buf[40]: signed 24
  DCT coeff: signed 8 or 16

  如果用 int8 DCT coeff：

  24 x 8 -> 32
  sum 40 terms -> +6 bit = 38
  DCT acc: signed 40
  MFCC output requant: signed int8

  如果用 int16 DCT coeff：

  24 x 16 -> 40
  sum 40 terms -> 46
  DCT acc: signed 48

  成本优先我建议：

  DCT coeff = int8
  DCT acc = signed 40
  MFCC output = signed int8

  推荐 baseline 汇总

  PCM_W              = 8 signed
  MFCC_SAMPLE_W      = 12 signed
  HANN_COEFF_W       = 16 unsigned
  FFT_IN_W           = 18 signed
  FFT_DATA_W         = 20 signed
  TWIDDLE_W          = 16 signed
  POWER_W            = 41 unsigned
  MEL_ACC_W          = 46 unsigned
  PWL_IN_W           = 32 unsigned  // Mel drain 处右移/饱和
  LOG_W              = 24 signed
  DCT_COEFF_W        = 8 signed
  DCT_ACC_W          = 40 signed
  MFCC_OUT_W         = 8 signed

  我建议我们下一步先确认两个关键点：Mel 46bit -> PWL 32bit 这里是否接受固定右移/饱和；以及 DCT coeff 用 int8 还
  是 int16。

## Baseline 安全位宽逐 stage 推导

本节解释上面 baseline 中每一个位宽是如何从定点运算规则、最坏情况增长和工程 guard bit 推出来的。

需要先明确一点：这里的 baseline 不是“理论最小位宽”，而是“第一版硬件友好、安全、不容易溢出的保守位宽”。因此有些地方会明显比数学最小需求更宽，例如：

```text
MFCC_SAMPLE_W = 12
FFT_IN_W      = 18
POWER_W       = 41
MEL_ACC_W     = 46
DCT_ACC_W     = 40
```

这些位宽的目标是先保证数值链路可信，后续再通过 bit-width sweep 找最小安全配置。

### 1. 基本位宽规则

#### 1.1 signed / unsigned 范围

如果一个整数是 signed W bit，采用二进制补码，则范围是：

```text
S(W): [-2^(W-1), 2^(W-1)-1]
```

例如：

```text
S8  = [-128, 127]
S12 = [-2048, 2047]
S18 = [-131072, 131071]
S20 = [-524288, 524287]
```

如果一个整数是 unsigned W bit，则范围是：

```text
U(W): [0, 2^W - 1]
```

例如：

```text
U16 = [0, 65535]
U32 = [0, 4294967295]
U41 = [0, 2199023255551]
U46 = [0, 70368744177663]
```

#### 1.2 乘法位宽增长

两个整数相乘时，保守位宽可以直接相加：

```text
W_raw = W_a + W_b
```

例如：

```text
S12 x U16 -> S28
S20 x S16 -> S36
S24 x S8  -> S32
```

这不是说真实数值一定会用满所有 bit，而是说硬件乘法器的 raw product 用这个宽度最安全。

#### 1.3 加法 / 累加位宽增长

两个同宽数相加，最坏会多 1 bit：

```text
S(W) + S(W) -> S(W+1)
U(W) + U(W) -> U(W+1)
```

如果累加 N 个同号最坏情况的 W bit 数，则需要额外：

```text
ceil(log2(N))
```

所以：

```text
sum_N U(W) -> U(W + ceil(log2(N)))
sum_N S(W) -> S(W + ceil(log2(N)))
```

例如：

```text
sum 32 terms -> +5 bit
sum 40 terms -> +6 bit, because ceil(log2(40)) = 6
```

#### 1.4 fixed-point 系数的位宽与数值缩放

系数经常不是以 float 存储，而是以整数 + 隐含小数点存储。

例如 FFT twiddle：

```text
TWIDDLE_W = S16, Q1.15
```

表示：

```text
twiddle_real_value ~= twiddle_int / 2^15
```

如果：

```text
cos(pi/4) ~= 0.70710678
```

则：

```text
twiddle_int = round(0.70710678 * 32768) = 23170
23170 / 32768 ~= 0.707092
```

这个量化误差会进入算法误差，但不会改变 raw multiplier 的位宽规则：

```text
data_int S20 x twiddle_int S16 -> raw S36
```

#### 1.5 round / shift / saturate 的作用

很多 stage 的 raw 结果会比最终存储位宽宽得多。硬件通常会执行：

```text
1. raw accumulator
2. rounding
3. right shift 或 scale requant
4. saturate / clip
5. 存入下一 stage 位宽
```

例如：

```text
window product raw S28 -> round/shift -> FFT input S18
FFT butterfly raw wider than S20 -> stage scaling -> internal S20
DCT accumulator S40 -> requant -> MFCC output S8
```

下面分别给出三个具体数字例子。数字只是为了说明位宽收敛方式，真实硬件中具体 shift、round 模式和 scale 需要与 Python golden model / RTL 保持一致。

例 1：`window product raw S28 -> round/shift -> FFT input S18`

假设加窗前内部 sample 是：

```text
sample_q = 1000          // S12, 合法范围 [-2048, 2047]
```

Hann 系数使用 U16 表示，假设某个窗系数约为 0.75：

```text
hann_q = round(0.75 * 65535) = 49151  // U16
```

乘法 raw product 为：

```text
raw = sample_q * hann_q
    = 1000 * 49151
    = 49,151,000
```

位宽上：

```text
S12 x U16 -> S28
```

S28 的正范围上界约为：

```text
2^27 - 1 = 134,217,727
```

所以 `49,151,000` 可以安全放在 S28 raw product 中。

但是 FFT 输入只希望保留到 S18：

```text
S18 range = [-131072, 131071]
```

如果直接把 `49,151,000` 写入 S18，必然溢出。因此需要右移。由于 raw product 中包含 Hann U16 系数的小数位，而输出 S18 比原始 S12 多保留约 6 bit：

```text
Hann fractional bits ~= 16
FFT extra precision  = 18 - 12 = 6
right_shift          = 16 - 6 = 10
```

于是可以做：

```text
fft_in_q = round(raw / 2^10)
         = round(49,151,000 / 1024)
         = 47,999
```

`47,999` 可以安全放入 S18。它对应的真实加窗结果约为：

```text
fft_in_q / 2^6 ~= 47,999 / 64 ~= 749.98
```

而理想结果是：

```text
1000 * 0.75 = 750
```

所以这一步的含义是：

```text
S28 raw product 很宽，用来安全承接乘法；
S18 FFT input 是经过 round/shift 后的存储格式，用来控制 FFT 数据通路成本。
```

例 2：`FFT butterfly raw wider than S20 -> stage scaling -> internal S20`

假设某一级 FFT butterfly 中，两个复数分量在某个 real lane 上分别为：

```text
x = 400,000   // S20, 合法
t = 350,000   // S20, 合法
```

S20 的范围是：

```text
[-524288, 524287]
```

butterfly 做：

```text
out0_raw = x + t = 750,000
out1_raw = x - t = 50,000
```

其中 `out0_raw = 750,000` 已经超过 S20 正上界 `524,287`。如果没有 stage scaling，写回 S20 时只能饱和成：

```text
out0_sat = 524,287
```

这会产生明显非线性失真。

因此 FFT 每一级或某些级会按 schedule 做右移。例如这一层设置：

```text
stage_shift = 1
```

则：

```text
out0_q = round(out0_raw / 2^1)
       = round(750,000 / 2)
       = 375,000

out1_q = round(out1_raw / 2^1)
       = round(50,000 / 2)
       = 25,000
```

两者都能安全写回 S20：

```text
375,000 in S20
25,000  in S20
```

这就是：

```text
butterfly add/sub 可能临时超过 S20；
stage scaling 通过右移把每级输出重新压回 S20；
代价是牺牲一部分低位精度。
```

对于 512 点 FFT，总共有 9 级 butterfly。如果每级都可能增长 1 bit，而完全不缩放，则最坏会增长 9 bit。因此 stage shift schedule 是 RTL FFT 中非常关键的数值设计参数。

例 3：`DCT accumulator S40 -> requant -> MFCC output S8`

DCT 阶段单项乘法是：

```text
log_mel_q:  S24
dct_coeff:  S8
```

所以单项 raw product 是：

```text
S24 x S8 -> S32
```

一个 MFCC 系数需要累加 40 项：

```text
dct_acc = sum_{m=0}^{39} log_mel_q[m] * dct_coeff_q[i,m]
```

累加 40 项需要额外：

```text
ceil(log2(40)) = 6 bit
```

因此理论累加器约为：

```text
S32 + 6 = S38
```

baseline 取 S40，是在 S38 基础上再留 2 bit guard。

假设某个 MFCC 维度的 DCT accumulator 输出为：

```text
dct_acc = 123,456,789   // S40, 合法
```

最终 backbone 只接受 S8 MFCC：

```text
MFCC_OUT_W = S8
S8 range = [-128, 127]
```

所以不能直接把 `123,456,789` 写成 S8，而必须 requant。假设某个通道的 requant 等效为右移 20 bit：

```text
mfcc_q = round(dct_acc / 2^20)
       = round(123,456,789 / 1,048,576)
       = 118
```

`118` 可以安全放入 S8。

如果另一个样本产生：

```text
dct_acc = 200,000,000
```

则：

```text
round(200,000,000 / 2^20) = 191
```

但 S8 最大只能表示 127，所以必须 saturate：

```text
mfcc_q = saturate_S8(191) = 127
```

这说明：

```text
DCT_ACC_W = S40 用来安全承接宽累加；
MFCC_OUT_W = S8 是网络输入接口；
两者之间必须有 per-tensor 或 per-channel requant scale，并且要统计 saturation_ratio。
```

因此，“raw 位宽”和“stage 输出位宽”必须分开看。

### 2. PCM_W = 8 signed

输入音频来自 `audio_sram / pcm input`，baseline 假设硬件输入就是 signed int8：

```text
PCM_W = 8 signed
```

因此输入整数范围是：

```text
pcm_q in [-128, 127]
```

这是外部接口约束，不是由 MFCC 内部计算推出来的。后续所有内部位宽都从这个 S8 输入开始推导。

如果把 `pcm_q` 理解为归一化音频：

```text
pcm_float ~= pcm_q * pcm_scale
```

则 `pcm_scale` 是数据 scale，决定真实幅度；但硬件整数数据通路首先看到的是 S8。

### 3. MFCC_SAMPLE_W = 12 signed

MFCC 内部 sample 位宽用于承接 PCM 扩展、可选预加重、以及进入加窗前的中间采样。

baseline 采用：

```text
MFCC_SAMPLE_W = 12 signed
```

#### 3.1 如果不做 pre-emphasis

如果只是把 S8 PCM 搬到内部 sample buffer，那么数学上 S8 已经足够：

```text
pcm_q in [-128, 127]
```

甚至 S9/S10 都已经有余量。

但实际硬件中通常会希望保留一些 guard bits 或 fractional bits，用于：

```text
1. 前端内部缩放
2. 输入增益调整
3. 后续 pre-emphasis
4. 避免早期量化过粗
```

因此选择 S12，等价于在 S8 基础上多留 4 bit：

```text
12 - 8 = 4 guard/fraction bits
```

#### 3.2 如果做 pre-emphasis

pre-emphasis 公式是：

```text
y[n] = x[n] - alpha * x[n-1]
alpha ~= 0.97
```

如果硬件中显式实现 `alpha = 0.97`，建议把它作为固定系数处理，而不是 float：

```text
PREEMPH_COEFF_W    = 16 unsigned
PREEMPH_COEFF_FRAC = 16
alpha_q = round(0.97 * 2^16) = 63570
alpha_q / 2^16 ~= 0.9700012207
```

如果 pre-emphasis 直接作用在 S8 PCM 上，则乘法 raw 位宽为：

```text
S8 x U16 -> S24
```

硬件可写成：

```text
pre_acc = (x[n] << 16) - alpha_q * x[n-1]
pre_tmp = round(pre_acc / 2^16)
```

如果系统先把 PCM 扩展到 S12 再做 pre-emphasis，则更保守的 raw 位宽是：

```text
S12 x U16 -> S28
```

无论采用哪种顺序，pre-emphasis stage 的输出都会再 requant / saturate 到：

```text
MFCC_SAMPLE_W = S12
```

如果 `x[n]` 和 `x[n-1]` 都来自 S8 PCM，则最坏绝对值上界为：

```text
|y[n]| <= |x[n]| + alpha * |x[n-1]|
       <= 128 + 0.97 * 128
       = 252.16
```

要表示 `[-252.16, 252.16]`，理论上 S9 就够：

```text
S9 = [-256, 255]
```

但是 S9 只是刚好覆盖整数幅度，没有给后续缩放、舍入和不同输入 scale 留余量。S12 的范围是：

```text
S12 = [-2048, 2047]
```

相对于 pre-emphasis 最坏幅度约 252，仍有约 3 bit guard：

```text
2048 / 252.16 ~= 8.12 ~= 2^3
```

所以：

```text
MFCC_SAMPLE_W = 12 signed
```

是一个明显偏安全的内部 sample baseline。若最终硬件不做 pre-emphasis，或者输入动态范围已经严格受控，后续可以考虑压到 S10。

### 4. HANN_COEFF_W = 16 unsigned

Hann 窗系数满足：

```text
0 <= hann[n] <= 1
```

因此它不需要 signed。baseline 使用：

```text
HANN_COEFF_W = 16 unsigned
```

可以理解为 U16 定点系数。常见解释方式有两种：

```text
方式 A: hann_float ~= hann_int / 65535
方式 B: hann_float ~= hann_int / 65536
```

二者都属于 U16 窗系数表，只是满幅 1.0 的处理略有差别。当前 Python bit-accurate 高精度路径里采用的是接近：

```text
scale = 1 / (2^16 - 1)
```

也就是方式 A。硬件实现时只要 RTL、Python golden model 和导出的系数表保持一致即可。

### 5. window product raw = S28

加窗公式：

```text
windowed[n] = sample[n] * hann[n]
```

输入 sample 是：

```text
sample_q: S12
```

Hann 系数是：

```text
hann_q: U16
```

因此 raw product 的保守乘法位宽是：

```text
S12 x U16 -> S(12 + 16) = S28
```

所以：

```text
window product raw: signed 28
```

注意，从真实数值角度，Hann 系数不超过 1，因此加窗不会放大幅度：

```text
|windowed_float[n]| <= |sample_float[n]|
```

但 raw integer product 仍然有 16 bit 系数小数位，所以硬件乘法器输出需要 S28。随后必须通过 round/shift 把 raw product 送到 FFT 输入格式。

### 6. FFT_IN_W = 18 signed

baseline 选择：

```text
FFT_IN_W = 18 signed
```

这一步不是数学上必须从 S28 保留到 S18，而是工程上在以下两件事之间折中：

```text
1. 不希望把 S28 raw product 全部传入 FFT，成本太高。
2. 不希望完全回到 S12，避免加窗后立刻损失过多低位信息。
```

可以把它理解为：

```text
window product raw S28
  -> round/shift
  -> windowed sample S18
```

如果 Hann 是 U16 小数系数，那么 raw S28 中包含了大约 16 bit 的系数小数信息。输出 S18 相当于比原始 S12 sample 多保留约 6 bit 精度：

```text
18 - 12 = 6
```

这 6 bit 可以视为加窗后的 fractional precision / guard precision。它让 FFT 输入比纯 S12 更平滑，但又不会把 FFT 主数据通路扩得太大。

因此：

```text
FFT_IN_W = 18 signed
```

是一个安全的 FFT 输入接口宽度。

### 7. TWIDDLE_W = 16 signed, Q1.15

FFT twiddle 是：

```text
W_N^k = exp(-j * 2*pi*k/N)
      = cos(theta) - j sin(theta)
```

其中：

```text
cos(theta), sin(theta) in [-1, 1]
```

因此使用 signed 16 bit 的 Q1.15 非常自然：

```text
TWIDDLE_W = 16 signed
TWIDDLE_FRAC = 15
twiddle_float ~= twiddle_q / 2^15
```

S16 Q1.15 的量化步长是：

```text
delta = 2^-15 ~= 3.0518e-5
```

round-to-nearest 的单个系数量化误差上界约为：

```text
|error| <= 0.5 * 2^-15 ~= 1.5259e-5
```

对于 KWS MFCC 前端，这个误差通常远小于后续 log/DCT/网络所能容忍的扰动，因此 S16 Q1.15 是一个常见且安全的 twiddle baseline。

### 8. FFT_DATA_W = 20 signed

baseline 规定 FFT 内部 Re/Im 和 RFFT 输出为：

```text
FFT internal Re/Im = S20
RFFT output        = S20 complex
```

这里要区分三种位宽：

```text
1. FFT 输入存储位宽: S18
2. FFT butterfly 主数据通路存储位宽: S20
3. 复乘 raw 中间位宽: S36 / S37
```

#### 8.1 为什么内部从 S18 扩到 S20

FFT 每一级 butterfly 都有：

```text
out0 = x + t
out1 = x - t
```

两个同量级数相加，最坏会增长 1 bit。S18 输入如果完全不扩展，第一层加减就可能溢出。S20 给了：

```text
20 - 18 = 2 guard bits
```

这允许早期 butterfly 有一定增长空间，也给 round/shift 前后的误差留余量。

#### 8.2 复乘 raw 位宽

FFT 复乘：

```text
(a + j b) * (c + j d)

real = a*c - b*d
imag = a*d + b*c
```

其中：

```text
a, b: FFT data S20
c, d: twiddle S16 Q1.15
```

单个乘法：

```text
S20 x S16 -> S36
```

real/imag combine 是两个 S36 product 的加减：

```text
a*c - b*d -> S37
a*d + b*c -> S37
```

所以：

```text
Complex multiply raw:
  20 x 16 -> 36
  real/imag combine -> 37
```

这个 S37 是局部 raw 中间结果，不是每一级最终存储宽度。

#### 8.3 Q1.15 小数位对齐

因为 twiddle 是 Q1.15，复乘 raw 结果带有 15 bit 小数缩放。要回到 FFT data 的数值尺度，需要右移 15 bit：

```text
mul_aligned ~= round(raw_mul / 2^15)
```

如果从纯位宽看：

```text
S37 raw >> 15 -> 约 S22
```

再经过 butterfly 加减，仍可能超过 S20。因此每一级必须有 stage scaling 和 saturate/clip 规则。

#### 8.4 512 点 FFT 的 9 级增长

512 点 radix-2 FFT 有：

```text
log2(512) = 9 stages
```

每一级 butterfly 最坏可能增长 1 bit。若完全不做缩放，则 9 级最坏增长：

```text
+9 bit
```

如果主数据通路是 S20，不缩放时最坏可能需要接近：

```text
S20 + 9 = S29
```

这显然不符合 `FFT_DATA_W = S20` 的存储目标。

因此 baseline 要求：

```text
FFT stage scaling:
  每级或按 schedule 右移，避免 9 级增长
```

常见策略包括：

```text
策略 A: 每级都右移 1 bit
  total_shift = 9
  最保守，最不容易溢出，但低位损失较大。

策略 B: 按 schedule 右移
  例如 stage_shift = [1,0,1,1,0,1,1,1,0]
  total_shift < 9 或按实际动态范围设置，精度更好，但需要验证溢出风险。
```

因此：

```text
FFT_DATA_W = 20 signed
```

不是说 raw FFT 永远只需要 20 bit，而是说每一级 butterfly 经过规定的 round/shift/saturate 后，都重新写回 S20。

### 9. POWER_W = 41 unsigned

FFT 输出：

```text
Re: S20
Im: S20
```

Power 计算：

```text
power = Re^2 + Im^2
```

S20 乘 S20 的硬件 raw product 宽度是：

```text
S20 x S20 -> S40
```

平方结果一定非负，因此可以看成 unsigned raw：

```text
Re^2 raw: U40
Im^2 raw: U40
```

两个 U40 相加，最坏增长 1 bit：

```text
U40 + U40 -> U41
```

所以：

```text
POWER_W = 41 unsigned
```

这是从 multiplier raw width 出发的保守上界。严格从数值最大值看，S20 的最大绝对值约为 `2^19`，平方最高约 `2^38`，单个平方的数学最小表示可以小于 U40；但硬件乘法器 raw product 通常自然给出 40 bit，继续用 U41 承接双平方和最简单、最安全。

### 10. MEL_ACC_W = 46 unsigned

当前 Mel 使用 rectangular / square-wave filter：

```text
mel_filter_shape = rectangular
```

因此 Mel 滤波不是三角权重乘法，而是对若干 FFT power bin 做选择性累加：

```text
mel[m] = sum_{k in band_m} power[k]
```

每个 `power[k]` 是：

```text
U41
```

如果某个 Mel band 最多累加 K 个 bin，则 accumulator 需要：

```text
MEL_ACC_W = POWER_W + ceil(log2(K))
```

baseline 估计每个 band 最多约 32 个 bin：

```text
K <= 32
ceil(log2(32)) = 5
```

因此：

```text
MEL_ACC_W = 41 + 5 = 46 unsigned
```

这就是：

```text
MEL_ACC_W = 46
```

需要注意：

```text
rectangular Mel 中，一个 FFT bin 可能被多个 band 覆盖；
但对单个 Mel accumulator 来说，位宽只由该 band 内累加的 bin 数 K 决定。
```

如果以后改回 triangular Mel，则 Mel stage 会变成：

```text
power[k] * mel_coeff[m,k]
```

那就需要额外定义 `MEL_COEFF_W` 和乘法 raw 位宽；当前 rectangular Mel 不需要这一步。

### 11. PWL_IN_W = 32 unsigned

Mel accumulator 是：

```text
mel_acc: U46
```

如果把 U46 直接送入 PWL log，则 PWL breakpoint compare、segment select 和 slope multiply 都会非常宽。例如：

```text
PWL slope S16 x PWL input U46 -> raw 62 bit
```

硬件成本偏高。

因此 baseline 在 Mel drain 到 PWL log 之前做一次压缩：

```text
mel_acc U46 -> round/shift/saturate -> pwl_input U32
```

得到：

```text
PWL_IN_W = 32 unsigned
```

从纯位宽看，如果要从 U46 压到 U32，至少需要丢掉或缩放：

```text
46 - 32 = 14 bit
```

所以可以理解为存在一个 Mel drain shift：

```text
pwl_input_q = saturate_U32(round(mel_acc / 2^mel_drain_shift))
```

其中：

```text
mel_drain_shift >= 14
```

或者由 calibration 得到等价的 data scale / requant 参数。

这一步必须记录 shift 或 scale，因为 log 的输入绝对幅度会影响：

```text
log(x + offset)
```

如果 shift 改变，但 PWL breakpoint、offset、scale 不同步更新，log_mel 特征会整体漂移。

因此：

```text
PWL_IN_W = 32
```

是一个硬件复杂度和动态范围之间的折中：它不是对 U46 的无损保存，而是要求通过 calibration / scale / shift 保证进入 log 的有效范围稳定。

### 12. LOG_W = 24 signed

log 输出是 signed，因为：

```text
log(x) < 0, when 0 < x < 1
log(x) > 0, when x > 1
```

所以：

```text
LOG_W = signed
```

PWL log 输出可以抽象为：

```text
log_mel_float ~= log(pwl_input_float + offset)
log_mel_q     = round(log_mel_float / log_mel_scale)
```

若使用 signed `LOG_W`，则需要：

```text
2^(LOG_W-1)-1 >= max(|log_min|, |log_max|) / log_mel_scale
```

当前 baseline 使用：

```text
LOG_W = 24 signed
```

S24 范围为：

```text
[-8388608, 8388607]
```

结合当前 bit-accurate 实验中常见的 log_mel scale 约 `2e-6`，S24 可覆盖的真实 log 范围大约是：

```text
8388607 * 2e-6 ~= 16.8
```

实际 PWL log 的有效范围通常在十几这个量级内，例如低能量端可能接近 `log(1e-8) ~= -18.4`，高能量端为正但较小。最终是否完全覆盖，取决于：

```text
1. log_offset
2. log_input_clamp_min
3. Mel drain scale
4. PWL breakpoint 范围
5. log_mel_scale
```

因此 `LOG_W = 24` 是一个合理的第一版安全位宽，但它不是完全脱离 scale 的数学常数。后续如果调整 PWL 输入 scale 或 log offset，需要重新检查 S24 的 saturation ratio。

### 13. DCT_COEFF_W = 8 signed

MFCC 最后一步是 DCT：

```text
mfcc[i] = sum_{m=0}^{39} dct_coeff[i,m] * log_mel[m]
```

DCT 系数来自正交 DCT matrix。对于 `n_mels = 40`，常见 ortho DCT 系数范围远小于 1，因此可以用 signed int8 定点近似。

baseline 选择：

```text
DCT_COEFF_W = 8 signed
```

当前 Python 高精度 bit-accurate 路径里，DCT coeff 使用近似：

```text
dct_coeff_float ~= dct_coeff_q / 2^7
```

即 S8 Q1.7：

```text
DCT_COEFF_SCALE = 1 / 128 = 0.0078125
```

S8 Q1.7 的表示范围约为：

```text
[-1.0, 0.9921875]
```

足够覆盖 DCT matrix 系数。round-to-nearest 的单个系数量化误差上界是：

```text
0.5 / 128 = 0.00390625
```

使用 S8 的好处是 DCT 乘法成本低：

```text
LOG_W S24 x DCT_COEFF_W S8 -> raw S32
```

如果使用 S16 DCT coeff，则：

```text
S24 x S16 -> raw S40
```

后续累加器会明显变宽。因此 baseline 成本优先选择 S8。

### 14. DCT_ACC_W = 40 signed

DCT 输入：

```text
log_mel[m]: S24
```

DCT 系数：

```text
dct_coeff[i,m]: S8
```

单项乘法 raw 位宽：

```text
S24 x S8 -> S32
```

每个 MFCC 系数需要累加 40 个 Mel 项：

```text
sum 40 terms
```

累加 40 项最坏需要额外：

```text
ceil(log2(40)) = 6 bit
```

因此数学上的累加器保守宽度是：

```text
S32 + 6 = S38
```

baseline 取：

```text
DCT_ACC_W = 40 signed
```

也就是在 S38 基础上再加 2 bit guard：

```text
40 - 38 = 2 guard bits
```

这 2 bit guard 用于覆盖：

```text
1. DCT coefficient 量化误差导致的局部偏移
2. rounding/shift 前的中间增长
3. 不同输入场景下 log_mel 动态范围变化
4. 后续 requant 到 MFCC_OUT_W 之前的安全余量
```

所以：

```text
DCT_ACC_W = 40
```

是从 `S24 x S8 -> S32` 和 `sum 40 -> +6 bit` 推出来的保守安全位宽。

### 15. MFCC_OUT_W = 8 signed

最终输出给 backbone 的 MFCC 特征保持 int8：

```text
MFCC_OUT_W = 8 signed
```

也就是：

```text
mfcc_q in [-128, 127]
```

从 DCT accumulator 到 MFCC output 需要 requant：

```text
mfcc_q = saturate_S8(round(dct_acc_float / mfcc_scale))
```

实际实验中更推荐 per-channel scale：

```text
mfcc_scale[i], i = 0..39
```

原因是不同 MFCC 维度的动态范围差异很大。若使用单一 per-tensor scale，则大动态范围维度会决定整体 scale，小动态范围维度会损失分辨率。per-channel scale 可以让 40 个 MFCC 维度分别充分利用 S8 范围。

因此：

```text
MFCC_OUT_W = 8
```

不是因为 DCT accumulator 的数学结果只需要 8 bit，而是因为网络输入接口希望保持 int8；中间宽度由 `DCT_ACC_W = S40` 保证，最终通过校准 scale 压到 S8。

### 16. Baseline 位宽推导总表

| stage | 输入/操作 | raw 位宽推导 | baseline 输出位宽 | 说明 |
|---|---|---:|---:|---|
| PCM | 外部音频输入 | 外部接口 | S8 | audio_sram / PCM input |
| internal sample | PCM + optional pre-emphasis | S8 最坏 pre-emphasis 约需 S9 | S12 | 留 3-4 bit guard/fraction |
| Hann coeff | `0 <= hann <= 1` | U16 coeff table | U16 | Q0.16 / U16 table |
| window product | S12 sample x U16 Hann | 12 + 16 = 28 | raw S28 | raw product |
| FFT input | window raw round/shift | S28 drain | S18 | 比 S12 多保留约 6 bit |
| twiddle | cos/sin in [-1,1] | S16 Q1.15 | S16 | 系数量化步长 2^-15 |
| FFT complex multiply | S20 data x S16 twiddle | 20 + 16 = 36; combine -> 37 | raw S37 | 局部 raw，不是 stage 输出 |
| FFT butterfly | add/sub + stage scaling | radix-2 9 stages may grow +9 | S20 | 每级或 schedule 右移压回 S20 |
| power | Re/Im S20 square | S20 x S20 -> U40; sum two -> U41 | U41 | `Re^2 + Im^2` |
| Mel | rectangular sum up to 32 bins | U41 + ceil(log2(32)) = U46 | U46 | 方波 Mel 无小数系数乘法 |
| PWL input | Mel drain | U46 -> shift/saturate | U32 | 降低 PWL 硬件成本 |
| log output | PWL log | signed log dynamic range | S24 | 需结合 log scale/offset 验证 |
| DCT coeff | DCT matrix | S8 Q1.7 | S8 | 成本优先 |
| DCT accumulator | S24 log x S8 coeff, sum 40 | 24 + 8 + ceil(log2(40)) = 38 | S40 | 额外 2 bit guard |
| MFCC output | DCT acc requant | S40 -> calibrated scale | S8 | backbone 输入 int8 |

### 17. 哪些位宽最可能仍有压缩空间

从上述推导可以看出，baseline 中有几类位宽明显偏安全：

```text
MFCC_SAMPLE_W = 12
FFT_IN_W      = 18
POWER_W       = 41
MEL_ACC_W     = 46
PWL_IN_W      = 32
LOG_W         = 24
DCT_ACC_W     = 40
```

其中：

```text
POWER_W / MEL_ACC_W / DCT_ACC_W
```

主要来自最坏情况累加上界，通常比较保守。

```text
PWL_IN_W / LOG_W
```

则强依赖 calibration scale、log offset 和输入分布。

```text
FFT_IN_W / FFT_DATA_W
```

还依赖最终 RTL FFT 的 stage shift schedule。

因此后续 bit-width sweep 的合理顺序是：

```text
1. 先固定算法和 scale，验证 baseline 完全可信。
2. 再逐 stage 缩位宽，观察 feature drift 和噪声场景精度。
3. 最后把 FFT radix-2 stage scaling、PWL 参数定点化、scale multiplier/shift 全部纳入硬件 golden model。
```

## 已落地的 Python 参考实现

当前已经新增：

```text
dscnn_kws/frontend/bit_accuracy_mfcc_frontend.py
dscnn_kws/quantization/qat_bit_accuracy_mfcc_frontend_noise_snr_scene.py
```

`BitAccuracyMFCCFrontend` 按上面的 baseline 固定位宽实现第一版整数参考：

```text
PCM input        S8
internal sample  S12
Hann coeff       U16 Q0.16
windowed sample  S18
DFT/FFT contract S20 complex output
Power            U41
Mel accumulator  U46
PWL/log input    U32
Log output       S24
DCT coeff        S8
DCT accumulator  S40
MFCC output      S8
```

注意：这一版为了先固定 bit-accuracy 接口，频域部分使用固定缩放的整数 DFT matrix 作为 Python reference；后续可以把该块替换为 radix-2 FFT，但保持 `S18 -> S20 complex -> U41 power` 的接口不变。

Mel 滤波当前统一使用方波/矩形滤波：

```text
mel_filter_shape = rectangular
```

新 QAT 脚本也把 `--mel_filter_shape` 限制为 `rectangular`，避免这组实验混入三角 Mel。

## 噪声场景 QAT + 测试命令

在服务器项目外层目录运行：

```bash
cd /root/kws/dscnn_kws

python dscnn_kws/quantization/qat_bit_accuracy_mfcc_frontend_noise_snr_scene.py \
  --limit 1 \
  --qat_epochs 3 \
  --batch 128 \
  --gpu 1
```

完整跑默认 checkpoint 目录：

```bash
python dscnn_kws/quantization/qat_bit_accuracy_mfcc_frontend_noise_snr_scene.py \
  --qat_epochs 10 \
  --batch 256 \
  --num_workers 8 \
  --gpu 1
```

如果只跑某个 checkpoint：

```bash
python dscnn_kws/quantization/qat_bit_accuracy_mfcc_frontend_noise_snr_scene.py \
  --checkpoints /root/kws/dscnn_kws/dscnn_kws/runs/snr_scene_arch_sweep_best_models/<model>.pt \
  --dataset mobvoi_hi_xiaowen_binary_hardneg \
  --qat_epochs 5 \
  --batch 256 \
  --gpu 1
```

默认输出不会覆盖旧 INT8 MFCC 结果：

```text
dscnn_kws/quantization/qat_bit_accuracy_mfcc_frontend_models/
dscnn_kws/quantization/qat_bit_accuracy_mfcc_frontend_train_results.csv
dscnn_kws/quantization/qat_bit_accuracy_mfcc_frontend_grid_results.csv
```

每个模型会保存：

```text
*_qat_prepared_best.pt
*_qat_int8_backbone.pt
*_bit_accuracy_mfcc_config.json
*_bit_accuracy_mfcc_int8_backbone.pt
```

其中 `*_bit_accuracy_mfcc_config.json` 会记录 baseline 位宽、Mel 方波设置、校准集观测范围和最终 `mfcc_output_scale`，后续写 Verilog/testbench 时应优先对齐这个 JSON。
