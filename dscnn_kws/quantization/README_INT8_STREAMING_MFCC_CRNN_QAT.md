# INT8 StreamingMFCC + CRNN QAT 使用说明

本文档对应新增脚本：

```text
dscnn_kws/quantization/qat_int8_streaming_mfcc_crnn_noise_snr_scene.py
dscnn_kws/frontend/int8_streaming_mfcc_frontend.py
```

目标是把流式 KWS 的这一条路径做成 INT8 量化实验流程：

```text
waveform
  -> INT8 StreamingMFCC frontend
  -> INT8 CRNN backbone reference
  -> TAU noise scene x SNR grid evaluation
```

它面向 `StreamingMFCC + CRNN-GRU` 模型，而不是旧的 DSCNN flatten-backbone 模型。脚本会从已有 CRNN checkpoint 出发，进行噪声增强 QAT fine-tune，随后校准前端和 CRNN backbone 的 INT8 scale，并直接在 TAU 噪声场景网格上测试量化后的模型。

## 1. 这份脚本做什么

整体流程如下：

```text
CRNN best.pt
  -> 重建 StreamingKWSModel
  -> 加载 CRNN backbone / full model 权重
  -> 构造 QATStreamingMFCCCRNNModel
  -> 在 TAU train noise 上做 INT8 fake-quant QAT
  -> 用 TAU valid noise 校准 INT8 StreamingMFCC 和 INT8 CRNN activation scale
  -> 保存 QAT checkpoint、INT8 reference checkpoint、scale json
  -> 用 INT8 StreamingMFCC + INT8 CRNN reference 跑 tau_test 5 dB
  -> 用 INT8 StreamingMFCC + INT8 CRNN reference 跑 10 scene x 5 SNR grid
```

默认噪声策略与前面 DSCNN 的噪声场景实验保持一致：

```text
train noise: ./dscnn_kws/noise/lists/tau_train.txt
valid noise: ./dscnn_kws/noise/lists/tau_valid.txt
test noise:  ./dscnn_kws/noise/lists/tau_test.txt

train SNR: [-5, 20] dB, noise_prob=0.8
valid/test list SNR: 5 dB, noise_prob=1.0
scene grid SNR: 20 / 10 / 5 / 0 / -5 dB
scene grid scenes: TAU 10 scenes
```

## 2. 推荐运行命令

在 Linux 服务器项目根目录运行，例如 `/root/kws/dscnn_kws`：

```bash
python dscnn_kws/quantization/qat_int8_streaming_mfcc_crnn_noise_snr_scene.py \
  --input_dir ./dscnn_kws/runs/streaming_kt7/streaming_crnn_noise_snr_scene_sweep_kt7_best_models \
  --root /root/kws/dscnn_kws/dscnn_kws/data \
  --gpu 1 \
  --batch 256 \
  --num_workers 8
```

先只测试一个 checkpoint：

```bash
python dscnn_kws/quantization/qat_int8_streaming_mfcc_crnn_noise_snr_scene.py \
  --input_dir ./dscnn_kws/runs/streaming_kt7/streaming_crnn_noise_snr_scene_sweep_kt7_best_models \
  --root /root/kws/dscnn_kws/dscnn_kws/data \
  --limit 1 \
  --qat_epochs 2 \
  --calibration_batches 10
```

如果 checkpoint 文件名无法推断 dataset，可以显式指定：

```bash
python dscnn_kws/quantization/qat_int8_streaming_mfcc_crnn_noise_snr_scene.py \
  --checkpoints ./dscnn_kws/runs/streaming_kt7/xxx_best.pt \
  --dataset mobvoi_nihao_wenwen_binary_hardneg \
  --root /root/kws/dscnn_kws/dscnn_kws/data
```

如果想显式覆盖 CRNN 结构：

```bash
python dscnn_kws/quantization/qat_int8_streaming_mfcc_crnn_noise_snr_scene.py \
  --checkpoints ./dscnn_kws/runs/streaming_kt7/xxx_best.pt \
  --dataset mobvoi_hi_xiaowen_binary_hardneg \
  --cnn_channels 40,40,40,40,40 \
  --kernel_time 7 \
  --kernel_freq 3 \
  --gru_hidden 48 \
  --root /root/kws/dscnn_kws/dscnn_kws/data
```

通常不需要手动传 `--cnn_channels`、`--kernel_time`、`--kernel_freq`、`--gru_hidden`，脚本会优先从 checkpoint 的 `state_dict` 推断；文件名里带 `_ch40-40-40-40-40_gru48_` 时，也能辅助推断。

## 3. 默认输入和输出

默认输入目录：

```text
./dscnn_kws/runs/streaming_kt7/streaming_crnn_noise_snr_scene_sweep_kt7_best_models
```

默认输出目录：

```text
./dscnn_kws/quantization/qat_int8_streaming_mfcc_crnn_models
```

默认 CSV：

```text
./dscnn_kws/quantization/qat_int8_streaming_mfcc_crnn_train_results.csv
./dscnn_kws/quantization/qat_int8_streaming_mfcc_crnn_grid_results.csv
```

每个 checkpoint 会保存这些文件：

```text
{stem}_streaming_crnn_qat_prepared_best.pt
{stem}_int8_streaming_mfcc_int8_crnn.pt
{stem}_int8_streaming_mfcc_scales.json
{stem}_int8_crnn_activation_scales.json
```

含义：

| 文件 | 含义 |
| --- | --- |
| `*_streaming_crnn_qat_prepared_best.pt` | QAT 阶段的 float 参数 + fake-quant module state，用于继续 QAT 或复现实验 |
| `*_int8_streaming_mfcc_int8_crnn.pt` | 最终 INT8 reference 推理模型，包含前端 scale、CRNN scale、INT8 weight state |
| `*_int8_streaming_mfcc_scales.json` | StreamingMFCC 前端各阶段 scale、观测范围、PWL log 配置 |
| `*_int8_crnn_activation_scales.json` | CRNN backbone 各层 activation scale |
| `qat_int8_streaming_mfcc_crnn_train_results.csv` | 每个模型的 QAT/校准/测试摘要 |
| `qat_int8_streaming_mfcc_crnn_grid_results.csv` | 量化模型在 TAU scene x SNR 网格上的逐点结果 |

## 4. 量化范围

这套脚本覆盖两部分：

```text
前端:
  Int8StreamingMFCCFrontend

Backbone:
  CRNN = causal depthwise-separable CNN + GRU + FC
```

### 4.1 前端 INT8

`Int8StreamingMFCCFrontend` 参考旧的 `Int8MFCCFrontend`，但帧切分语义改成和 `StreamingMFCC` 对齐：

```text
center=False
causal framing
left history padding
flush_tail=True
```

也就是说，对于 1 秒、16 kHz、32 ms hop 的输入，它会生成与 `StreamingMFCC` 对齐的帧序列，而不是 `torch.stft(center=True)` 那种带未来上下文的前端。

前端主要步骤：

```text
waveform
  -> INT8 quantize
  -> optional pre-emphasis
  -> INT8 requantize
  -> causal framing
  -> Hann window INT8 coefficient
  -> integer DFT coefficient matmul
  -> power spectrum
  -> integer Mel filterbank matmul
  -> natural log or PWL log
  -> INT8 log_mel requantize
  -> integer DCT coefficient matmul
  -> INT8 MFCC requantize
  -> dequantized float MFCC tensor
```

默认配置：

```text
INT8_MFCC_LOG_APPROX_MODE       = "pwl"
INT8_MFCC_LOG_PWL_NUM_SEGMENTS  = 8
INT8_MFCC_LOG_PWL_STRATEGY      = "uniform_logx"
INT8_MFCC_COEFF_BITS            = 8
INT8_MFCC_REQUANTIZE_POWER      = False
INT8_MFCC_REQUANTIZE_MEL        = False
```

`power` 和 `mel` 默认不强制 requantize 到 INT8，只统计范围并保留较宽动态范围。原因是这两个能量域动态范围很大，过早压到 INT8 容易把小能量压成 0，导致 log/MFCC 退化。想做更激进的“每个中间阶段都 INT8”实验时，可以打开：

```bash
--int8_mfcc_requantize_power \
--int8_mfcc_requantize_mel
```

但这通常会明显影响 MFCC 数值质量。

### 4.2 QAT 阶段的 CRNN INT8

脚本没有直接依赖 PyTorch eager `convert()` 去转换 GRU。原因是 PyTorch 对 GRU 的 eager-mode QAT/INT8 支持不像 Conv/Linear 那么直接，直接套 DSCNN 那条 `QuantStub -> prepare_qat -> convert` 路径并不适合这个 CRNN。

这里采用显式 fake-quant 的方式：

```text
QATStreamingCRNNBackbone
  -> CNN block input fake quant
  -> depthwise weight fake quant
  -> depthwise output fake quant
  -> pointwise weight fake quant
  -> pointwise output fake quant
  -> freq-pool output fake quant
  -> GRU input fake quant
  -> GRU weight_ih / weight_hh fake quant
  -> GRU gate fake quant
  -> GRU hidden fake quant
  -> FC input fake quant
  -> FC weight fake quant
  -> logits fake quant
```

GRU forward 被展开成显式 gate 计算：

```text
r_t = sigmoid(W_ir x_t + b_ir + W_hr h_{t-1} + b_hr)
z_t = sigmoid(W_iz x_t + b_iz + W_hz h_{t-1} + b_hz)
n_t = tanh(W_in x_t + b_in + r_t * (W_hn h_{t-1} + b_hn))
h_t = (1 - z_t) * n_t + z_t * h_{t-1}
```

这样做的好处是可以明确控制 GRU 输入、权重、gate、hidden 的 fake-quant 点，更接近后续硬件定点化时要关注的边界。

### 4.3 最终评测阶段的 INT8 reference

最终评测模型是：

```text
Int8StreamingMFCCInt8CRNNModel
  feature_extractor = Int8StreamingMFCCFrontend
  backbone = Int8ReferenceCRNNBackbone
```

`Int8ReferenceCRNNBackbone` 做的是 INT8 reference 推理：

```text
float tensor
  -> clamp/round 到 INT8 网格
  -> dequantize 回 float
  -> PyTorch functional conv/gru/fc 运算
```

权重也会按 INT8 网格量化，并在保存的 checkpoint 里额外写入：

```text
int8_weight_state
```

这不是最终 RTL，也不是 PyTorch backend 的真正 quantized kernel；它是一个可运行、可校准、可导出 scale/weight 的 INT8 数值参考。后续写 Verilog 时，应把这个 reference 进一步落到 bit-accurate 定点模型，再决定 accumulator 位宽、舍入、饱和和 LUT/PWL log 的具体实现。

## 5. 为什么前端输出还是 float

`Int8StreamingMFCCFrontend` 返回的是：

```text
mfcc_q * mfcc_scale
```

也就是已经限制在 INT8 网格上的 dequantized float tensor。这样做有两个目的：

1. 方便在 PyTorch 里继续接 CRNN reference，不依赖自定义 int8 kernel。
2. 数值上仍然可以看作“这个 MFCC tensor 只能取 INT8 scale 网格上的值”。

所以不要把它理解成“前端完全没有量化”。它已经经过了 INT8 clamp/round/requantize，只是为了在 PyTorch 里可运行，最后以 float 形式承载量化后的值。

## 6. 重要参数说明

### 6.1 输入与输出

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--input_dir` | `./dscnn_kws/runs/streaming_kt7/..._best_models` | 批量读取 CRNN checkpoint |
| `--pattern` | `*.pt` | checkpoint 匹配模式 |
| `--checkpoints` | `None` | 显式指定一个或多个 checkpoint |
| `--output_dir` | `./dscnn_kws/quantization/qat_int8_streaming_mfcc_crnn_models` | 保存模型和 scale json |
| `--train_results_csv` | `./dscnn_kws/quantization/qat_int8_streaming_mfcc_crnn_train_results.csv` | QAT 摘要 CSV |
| `--grid_results_csv` | `./dscnn_kws/quantization/qat_int8_streaming_mfcc_crnn_grid_results.csv` | scene x SNR 网格结果 |
| `--limit` | `0` | 只跑前 N 个 checkpoint，0 表示不限制 |

### 6.2 数据与运行环境

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--root` | `/root/kws/dscnn_kws/dscnn_kws/data` | 数据集根目录 |
| `--dataset` | `None` | 默认从 checkpoint 文件名推断，失败时必须显式指定 |
| `--batch` | `256` | batch size |
| `--num_workers` | `8` | dataloader workers |
| `--gpu` | `1` | `>0` 且 CUDA 可用时使用 GPU 做 QAT |
| `--seed` | `42` | 随机种子 |

### 6.3 QAT

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--qat_epochs` | `10` | QAT fine-tune epoch 数 |
| `--lr` | `5e-5` | Adam 学习率 |
| `--weight_decay` | `1e-6` | 权重衰减 |
| `--freeze_bn_after_epoch` | `3` | 第 4 个 epoch 开始冻结 BN 统计 |
| `--disable_observer_after_epoch` | `5` | 第 6 个 epoch 开始关闭 fake-quant observer |
| `--backend` | `fbgemm` | 保留给量化 backend 选择；本脚本主要使用显式 fake-quant/reference |

### 6.4 CRNN 结构

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--cnn_channels` | `None` | 默认从 checkpoint 推断，例如 `40,40,40,40,40` |
| `--kernel_time` | `None` | 默认从 depthwise conv weight 推断 |
| `--kernel_freq` | `None` | 默认从 depthwise conv weight 推断 |
| `--gru_hidden` | `None` | 默认从 GRU weight 推断 |
| `--gru_layers` | `None` | 默认从 checkpoint 推断 |
| `--dropout` | `0.2` | 重建模型时的 dropout |

### 6.5 StreamingMFCC 与 INT8 前端

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--sample_rate` | `16000` | 采样率 |
| `--dct_coeff` | `10` | CRNN 输入使用前 10 维 MFCC |
| `--window_size_ms` | `32` | 窗长 |
| `--window_stride_ms` | `32` | hop |
| `--pre_emphasis` | `True` | 前端里做 pre-emphasis |
| `--pre_emphasis_coeff` | `0.97` | pre-emphasis 系数 |
| `--mfcc_center` | `False` | 流式路径默认不使用未来上下文 |
| `--streaming_mfcc` | `True` | 源模型使用 StreamingMFCC |
| `--mel_filter_shape` | `triangular` | Mel filterbank 形状 |
| `--log_approx_mode` | `pwl` | QAT 浮点前端使用 PWL log |
| `--int8_mfcc_log_approx_mode` | `pwl` | INT8 前端使用 PWL log |
| `--int8_mfcc_coeff_bits` | `8` | window/DFT/Mel/DCT 系数位宽 |
| `--int8_mfcc_requantize_power` | `False` | 是否强制 power 阶段 requantize |
| `--int8_mfcc_requantize_mel` | `False` | 是否强制 mel 阶段 requantize |

### 6.6 校准

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--calibration_split` | `validation` | 用哪个 split 校准 scale |
| `--calibration_noise_roots` | `./dscnn_kws/noise/lists/tau_valid.txt` | 校准噪声源 |
| `--calibration_noise_prob` | `1.0` | 校准时始终加噪 |
| `--calibration_snr_min_db` | `-5.0` | 校准 SNR 下限 |
| `--calibration_snr_max_db` | `20.0` | 校准 SNR 上限 |
| `--calibration_batches` | `0` | 0 表示用完整 calibration loader |

## 7. CSV 字段怎么看

### 7.1 train results

`qat_int8_streaming_mfcc_crnn_train_results.csv` 每行对应一个 checkpoint，核心字段：

| 字段 | 含义 |
| --- | --- |
| `dataset` | 数据集名称 |
| `arch` | 推断出的 CRNN 架构，例如 `C40-40-40-40-40_H48` |
| `cnn_channels` | CNN channel 列表 |
| `kernel_time`, `kernel_freq` | causal depthwise conv kernel |
| `gru_hidden`, `gru_layers` | GRU 结构 |
| `source_checkpoint` | 原始 checkpoint |
| `source_load_scope` | full model 加载或只加载 backbone |
| `best_epoch` | QAT 过程中 valid acc 最好的 epoch |
| `best_valid_acc` | 最好 valid acc |
| `qat_test_tau_list_acc` | QAT fake-quant 模型在 tau_test 5 dB 上的 acc |
| `int8_test_tau_list_acc` | INT8 reference 模型在 tau_test 5 dB 上的 acc |
| `frontend_scale_json` | 前端 scale 文件 |
| `backbone_scale_json` | CRNN activation scale 文件 |
| `qat_checkpoint` | QAT checkpoint |
| `inference_checkpoint` | INT8 reference checkpoint |

### 7.2 grid results

`qat_int8_streaming_mfcc_crnn_grid_results.csv` 每行对应一个：

```text
dataset x arch x scene x snr_db
```

核心字段：

| 字段 | 含义 |
| --- | --- |
| `scene` | TAU scene，例如 `airport`、`shopping_mall` |
| `snr_db` | 测试 SNR |
| `acc`, `precision`, `recall`, `f1` | 分类指标 |
| `num_samples` | 该测试点样本数 |
| `usable_noise_files` | 该 scene 可用 wav 数 |
| `inference_checkpoint` | 用于测试的 INT8 reference checkpoint |

## 8. 常见问题

### 8.1 报错：Cannot infer dataset

脚本会尝试从文件名推断 dataset。若 checkpoint 文件名不符合类似下面的模式：

```text
mobvoi_nihao_wenwen_binary_hardneg_C40x5_H48_ch40-40-40-40-40_gru48_...
```

请显式传：

```bash
--dataset mobvoi_nihao_wenwen_binary_hardneg
```

### 8.2 报错：Cannot infer CRNN cnn_channels

checkpoint 的 key 需要包含：

```text
backbone.cnn.0.pointwise.weight
backbone.cnn.0.depthwise.weight
backbone.gru.weight_hh_l0
backbone.fc.weight
```

如果这是只保存了部分模块、或者 key 前缀不同的 checkpoint，需要先整理成 `StreamingKWSModel.state_dict()` 风格，或者修改脚本里的 `infer_model_shape_from_state_dict()`。

### 8.3 噪声文件数量为 0

脚本启动时会打印：

```text
[INFO] train_noise_roots=..., usable_noise_files=...
```

如果 train/valid/test 噪声源可用数量为 0，会直接报错。请检查：

```text
./dscnn_kws/noise/lists/tau_train.txt
./dscnn_kws/noise/lists/tau_valid.txt
./dscnn_kws/noise/lists/tau_test.txt
./dscnn_kws/noise/tau
```

`.txt` 里可以写相对路径或绝对路径；相对路径按 `.txt` 文件所在目录解析。

### 8.4 本地 Windows 不能跑完整验证

完整 QAT 和 scene-grid 评测依赖：

```text
torch
sklearn
tqdm
真实数据集
TAU 噪声 wav
CUDA 或足够快的 CPU
```

如果本地没有 `torch`，只能做语法检查：

```bash
python -m py_compile \
  dscnn_kws/frontend/int8_streaming_mfcc_frontend.py \
  dscnn_kws/quantization/qat_int8_streaming_mfcc_crnn_noise_snr_scene.py
```

真实数值和准确率要在 Linux/conda 服务器环境上验证。

### 8.5 这是不是最终硬件实现

不是。当前脚本是 INT8 QAT + INT8 reference 推理，用于验证量化策略和噪声鲁棒性。它已经比“只在输出 hook 上 fake-quant”更接近硬件，但仍不是 bit-accurate RTL。

后续硬件路线建议：

```text
PyTorch float CRNN
  -> 当前 INT8 fake-quant / reference
  -> bit-accurate Python fixed-point model
  -> Verilog / RTL module
  -> FPGA / ASIC 验证
```

特别需要继续明确的硬件细节包括：

```text
DFT/Mel/DCT accumulator 位宽
power accumulator 位宽
PWL/LUT log 输入输出格式
GRU gate sigmoid/tanh 近似方式
hidden state scale 是否逐层/逐门固定
rounding 策略
saturation 策略
per-tensor 或 per-channel weight scale
```

## 9. 建议实验顺序

第一步，先跑一个 checkpoint、短 QAT、少量校准 batch：

```bash
python dscnn_kws/quantization/qat_int8_streaming_mfcc_crnn_noise_snr_scene.py \
  --input_dir ./dscnn_kws/runs/streaming_kt7/streaming_crnn_noise_snr_scene_sweep_kt7_best_models \
  --root /root/kws/dscnn_kws/dscnn_kws/data \
  --limit 1 \
  --qat_epochs 2 \
  --calibration_batches 10
```

第二步，确认输出文件存在，并检查：

```text
qat_test_tau_list_acc
int8_test_tau_list_acc
grid acc/f1
frontend scale json
backbone activation scale json
```

第三步，再恢复完整配置：

```bash
python dscnn_kws/quantization/qat_int8_streaming_mfcc_crnn_noise_snr_scene.py \
  --input_dir ./dscnn_kws/runs/streaming_kt7/streaming_crnn_noise_snr_scene_sweep_kt7_best_models \
  --root /root/kws/dscnn_kws/dscnn_kws/data \
  --qat_epochs 10 \
  --calibration_batches 0
```

第四步，对比量化前 CRNN scene-grid CSV 与本脚本的 `qat_int8_streaming_mfcc_crnn_grid_results.csv`，重点看：

```text
平均 acc drop
低 SNR, especially 0 / -5 dB drop
shopping_mall / street_traffic / airport 等困难 scene drop
不同 CRNN channel / hidden 配置的量化敏感性
```

## 10. 与旧 INT8 MFCC DSCNN 脚本的区别

旧脚本：

```text
dscnn_kws/quantization/qat_int8_mfcc_frontend_noise_snr_scene.py
```

主要用于：

```text
TorchMFCC / Int8MFCCFrontend
  -> DSCNN flatten backbone
  -> PyTorch eager quantized Conv/Linear backbone
```

当前脚本用于：

```text
StreamingMFCC / Int8StreamingMFCCFrontend
  -> CRNN = causal depthwise-separable CNN + GRU + FC
  -> explicit fake-quant QAT
  -> INT8 reference CRNN backbone
```

最关键差别：

```text
1. 前端帧切分必须对齐 StreamingMFCC 的 causal/flush-tail 语义。
2. Backbone 里有 GRU，不能简单沿用 DSCNN 的 eager convert 路径。
3. 最终 CRNN INT8 是 reference 数值路径，不是 PyTorch quantized GRU kernel。
4. 输出 checkpoint 额外保存 int8_weight_state，方便后续定点/RTL 继续使用。
```


# 一些问题：

## 1.所以现在做的不是只止步于fake quant，而是实现了int8的真量化和测试对吧

对，但要精确分两层说：

现在不是只止步于 QAT fake quant。脚本后半段确实会构造并测试一个 **INT8 reference 推理模型**：

```text
Int8StreamingMFCCFrontend
+ Int8ReferenceCRNNBackbone
```

它会把前端和 CRNN backbone 的权重/激活约束到 INT8 网格，保存 scale json 和 `int8_weight_state`，并用这个量化后的 reference 模型跑 `tau_test` 和 TAU scene x SNR 测试。

但它还不是最终硬件意义上的“全程 int8 kernel / RTL bit-accurate 实现”。当前实现为了在 PyTorch 里可运行，很多地方是：

```text
float tensor
-> round/clamp 到 int8 网格
-> dequantize 回 float
-> 用 PyTorch functional conv / GRU / FC 计算
```

所以更准确的表述是：

```text
不是只做 fake quant；
已经有 INT8 数值参考模型和噪声场景测试；
但还不是 bit-accurate Verilog/硬件实现。
```

如果你论文或报告里写，可以写成：“完成了 StreamingMFCC 到 CRNN backbone 的 INT8 QAT 与 INT8 reference inference evaluation”。

## 2.现在int8 streamingMFCC的power和log是不是和int8 MFCC一样都还是int64

不是完全一样地说“power 和 log 都是 int64”。

现在 `Int8StreamingMFCCFrontend` 基本沿用了 `Int8MFCCFrontend` 的数值策略：

```text
DFT real_acc / imag_acc: int64
power_acc = real_acc^2 + imag_acc^2: int64
mel_acc: int64
mfcc_acc: int64
```

所以 **power 的 accumulator 是 int64**。但默认情况下：

```text
INT8_MFCC_REQUANTIZE_POWER = False
INT8_MFCC_REQUANTIZE_MEL = False
```

也就是说 power/mel 默认不会强制压回 int8，而是：

```text
power_acc(int64) -> power_float
mel_acc(int64) -> mel_float
```

然后进入 log。

**log 不是 int64。** 当前 log / PWL log 是在 dequantized 的 `mel_for_log` float tensor 上算的：

```text
log_mel_float = self._log_transform(mel_for_log)
log_q, log_scale = self._quantize(log_mel_float, "log_mel")
```

所以 log 阶段是：

```text
float log/PWL log -> 再量化成 INT8 log_mel 网格
```

一句话总结：

```text
power: int64 accumulator，然后默认转 float 进入 mel/log
log: float/PWL 计算，不是 int64；log 输出再 requantize 到 INT8
```

这点和旧 `Int8MFCCFrontend` 是一致的。

## 3.bit-accurate 定点模型是什么

`bit-accurate 定点模型`就是一个**在 Python/C 里逐位模拟硬件会怎么算的参考模型**。

它和现在的 INT8 reference 不一样。现在很多地方是：

```text
round/clamp 到 INT8 网格
-> dequantize 回 float
-> 用 PyTorch float 算 conv / GRU / log
```

而 bit-accurate 定点模型会规定每一步到底是什么整数格式、多少位、怎么舍入、怎么饱和。例如：

```text
输入: int8
权重: int8
乘法: int8 * int8 -> int16
累加: int32
bias: int32
右移: >> shift
舍入: round-to-nearest 或 truncate
饱和: clamp 到 int8
输出: int8
```

它要做到：**Python 模型输出的每一个整数值，都和未来 Verilog/RTL 仿真输出完全一致**。

举个 CRNN 里的例子：

```text
conv_acc = sum(input_int8 * weight_int8) + bias_int32
conv_shifted = round_shift(conv_acc, output_shift)
conv_out = saturate_int8(conv_shifted)
```

这里 `round_shift` 怎么四舍五入，溢出怎么截断，bias 用什么 scale，activation scale 怎么接下一层，都必须写死。这样才叫 bit-accurate。

所以关系是：

```text
QAT fake quant:
  训练时模拟量化误差

INT8 reference:
  验证大体 INT8 scale/权重/激活策略是否可行

bit-accurate 定点模型:
  精确模拟硬件每一步整数运算，作为 Verilog 对照答案

RTL/Verilog:
  硬件实现
```

简单说：**bit-accurate 定点模型就是硬件实现前的“整数标准答案”。**

## 4.解释一下“最终 CRNN INT8 是 reference 数值路径，不是 PyTorch quantized GRU kernel”

这句话的意思是：

当前脚本最后测试的 CRNN，**不是调用 PyTorch 官方已经写好的 INT8 GRU 算子**，而是我们自己写了一条“数值上按 INT8 约束”的参考推理路径。

也就是现在的 `Int8ReferenceCRNNBackbone` 大概这样做：

```text
weight float
-> round/clamp 成 int8 网格
-> dequantize 回 float

activation float
-> round/clamp 成 int8 网格
-> dequantize 回 float

然后用 PyTorch 的 F.conv2d / F.linear / sigmoid / tanh 等函数继续算
```

所以它能模拟：

```text
如果权重和激活只能取 INT8 网格上的值，模型精度会怎样？
```

但它不是：

```text
真正的 torch quantized GRU kernel
```

也不是：

```text
底层全程 int8 x int8 -> int32 accumulator -> requantize 的整数算子
```

为什么要这样？因为 PyTorch 对 Conv/Linear 的 INT8 quantized kernel 支持比较成熟，但对你这个自定义 CRNN 里的 GRU，尤其是我们想控制 gate、hidden、activation scale 的情况，直接用 PyTorch eager quantization 不方便。所以脚本自己展开 GRU gate，做 INT8 网格约束，用它作为可测试的 reference。

一句话：

```text
reference 数值路径 = 用 PyTorch float 算子承载 INT8 量化后的数值
PyTorch quantized GRU kernel = PyTorch 底层真正的量化 GRU 算子
```

当前是前者，不是后者。

## 5.那我之前做的DSCNN的int8是reference模型吗

不完全是。你之前 DSCNN 的 INT8 更接近 **混合形式**：

```text
INT8 MFCC frontend: reference 数值路径
DSCNN backbone: PyTorch eager quantization convert 后的 quantized backbone
```

具体说：

如果指这个脚本：

```text
dscnn_kws/quantization/qat_int8_mfcc_frontend_noise_snr_scene.py
```

那么最终模型是：

```text
Int8MFCCFrontend
+ quantized_qat_model.backbone
```

其中：

**1. 前端 `Int8MFCCFrontend` 是 reference 模型**

它和现在的 `Int8StreamingMFCCFrontend` 类似：

```text
INT8 clamp/round
-> int64 accumulator 做 DFT/Mel/DCT
-> log 用 float/PWL
-> 最后 dequantized float MFCC 输出
```

所以前端不是 PyTorch 官方 quantized kernel，而是我们自己写的 INT8 数值参考路径。

**2. DSCNN backbone 不是 reference，而是 PyTorch quantized backbone**

DSCNN backbone 走的是：

```text
prepare_qat()
-> QAT
-> convert()
```

Conv/Linear 这些层会被 PyTorch 转成 quantized module。也就是说，DSCNN backbone 比现在 CRNN 的 GRU 更“真 PyTorch INT8 kernel”一些。

但它仍然不是硬件 bit-accurate RTL，因为 PyTorch quantized kernel 内部的 scale、zero point、bias 处理、requantize、backend 行为，不等同于你未来自己写的 Verilog。

所以一句话：

```text
之前 DSCNN INT8:
  前端是 INT8 reference
  backbone 是 PyTorch quantized INT8

现在 CRNN INT8:
  前端是 INT8 reference
  backbone 也是自己写的 INT8 reference，尤其因为 GRU 没走 PyTorch quantized GRU kernel
```

这就是两者最大的区别。

## 6.“它和现在的 INT8 reference 不一样。现在很多地方是：round/clamp 到 INT8 网格,dequantize 回 float,用 PyTorch float 算 conv / GRU / log”  那现在不就是做的fake quant吗?

是的，你这个判断很敏锐：**从计算形式上看，现在的 CRNN INT8 reference 很像 fake quant。**

更准确地说，当前有两类“fake”：

```text
训练阶段:
  QAT fake quant
  用 observer + FakeQuantize 模拟量化误差，参数仍是 float

测试阶段:
  INT8 reference fake-quantized inference
  用固定 scale 做 round/clamp，再 dequantize 成 float，用 float 算子承载计算
```

所以如果严格按“有没有全程整数 kernel”来定义，当前 CRNN backbone 还不是真正整数推理。它不是：

```text
int8 activation * int8 weight
-> int32 accumulator
-> integer requantize
-> int8 activation
```

它现在更像：

```text
把每层输入/权重/hidden 限制到 INT8 可表示值
但乘加本身仍由 PyTorch float 执行
```

那为什么我之前说“不只是 fake quant”？因为它已经比训练期 QAT fake quant多做了几件事：

```text
固定 scale 校准
保存 scale json
保存 int8_weight_state
用最终量化 reference 模型跑 tau_test 和 scene x SNR
```

但你要是问“它是不是已经是真正 int8 integer arithmetic inference？”答案是：

```text
不是。
```

最严谨的表述应该改成：

```text
当前 CRNN 实现了 INT8 QAT + 固定 scale 的 fake-quantized reference inference；
还没有实现 bit-accurate int8 integer arithmetic inference。
```

之前 README 里“最终 INT8 reference”这个说法容易让人误会成“真 int8 算子”，确实可以改得更清楚。

## 7.那现在跑出的结果具有可信度吗，我最后真的实现bit-accurate模型后，精度会掉很多吗

有可信度，但要把它放在正确的位置上看。

现在跑出的结果可信的是：

```text
如果权重、前端输出、CRNN 中间激活、GRU hidden 都被限制到 INT8 可表示网格，
模型在噪声场景下大概还能不能工作。
```

它对“量化敏感性”和“QAT 后精度是否大体可保住”是有参考价值的。尤其你已经固定 scale、跑了 tau_test 和 scene x SNR，比单纯训练时 fake quant 更有意义。

但它不完全可信的是：

```text
真实整数硬件实现后，逐层 bit-level 数值是否完全一样。
```

因为现在还没有严格模拟：

```text
int8 * int8 -> int32 accumulator
bias int32 scale
右移 / requantize
rounding 策略
saturation 策略
sigmoid/tanh LUT 或 PWL 近似
GRU gate 的定点格式
log 的 LUT/PWL 定点误差
```

所以最后 bit-accurate 后精度会不会掉很多，取决于这些细节。我的判断是：

```text
如果 scale 选得合理、accumulator 位宽够、log/sigmoid/tanh 近似做得不粗糙，
一般不会“突然掉很多”。

但 CRNN 的 GRU 比 DSCNN 更敏感，尤其 hidden state 会跨时间递推，
所以如果 hidden/gate scale、sigmoid/tanh 近似、requantize 策略做差，
低 SNR 下可能明显掉点。
```

最可能导致额外掉精度的地方是：

```text
1. GRU hidden 每帧递推，量化误差会累积
2. sigmoid/tanh 如果 LUT/PWL 太粗，会改变 gate 行为
3. log/MFCC 前端如果 power/mel 过早压 int8，会损失小能量细节
4. accumulator 位宽不够或饱和太频繁
5. 每层 scale 用得太粗，比如所有 gate 共用一个不合适的 scale
```

所以现在结果可以作为“上限附近的量化可行性评估”。真正 bit-accurate 后，最好预期：

```text
轻微下降是正常的；
如果下降很大，通常说明定点格式/scale/LUT/饱和策略需要调，而不是模型本身一定不行。
```

我建议下一步不是直接写 Verilog，而是先做一个 **bit-accurate Python fixed-point CRNN inference**，只跑一小批样本，对比现在 reference 的 logits/预测差异。这样能在写硬件前把掉点来源定位清楚。

## 8.再解释下fake-quant的概念，我怎么感觉和你说的reference是一样的处理啊，只不过是阶段不一样，一个是训练时的概念，一个是部署后推理的概念

你理解得基本对：**fake-quant 和现在这个 reference 的核心操作确实很像**，都是：

```text
float
-> 按 scale / zero_point 量化到整数网格
-> 再反量化回 float
-> 后续仍用 float 算子计算
```

所以它们的共同点是：**数值被量化约束了，但计算内核不是真整数乘加。**

区别主要在“目的”和“状态”。

**1. QAT fake-quant**

训练时用：

```text
float weight / activation
-> fake quant
-> float op
-> backward
```

它的目的不是部署，而是让模型在训练时提前适应量化误差。observer 还会统计范围，scale 可能还在更新。

可以理解成：

```text
训练时的量化噪声模拟器
```

**2. 当前 INT8 reference inference**

推理/测试时用：

```text
固定 scale
-> round/clamp 到 INT8 网格
-> dequantize 回 float
-> float op
```

它的目的不是训练，而是评估“如果所有关键张量只能取 INT8 网格值，准确率大概怎样”。scale 已经固定，权重也会导出 int8 版本。

可以理解成：

```text
部署前的量化数值参考模型
```

所以你说的“只不过阶段不一样”是对的，但还可以补一句：

```text
QAT fake-quant: 为了训练适应量化
reference inference: 为了用固定量化参数评估/导出
```

它们都不等于真正的：

```text
int8 * int8 -> int32 accumulator -> requantize -> int8
```

真正整数推理要求乘法、累加、bias、右移、舍入、饱和也都按整数规则执行。当前 reference 只保证“输入值和权重值落在 INT8 网格上”，不保证“中间乘加过程也和硬件逐 bit 一致”。

一个很短的类比：

```text
fake-quant / reference:
  我把数字先四舍五入成 8-bit 能表示的值，
  但后面还是用计算器按小数算。

bit-accurate integer:
  我从头到尾都只用整数、小数点位置、移位、饱和来算。
```

所以你的直觉是正确的：**当前 reference 本质上仍是 fake-quantized inference，只是固定 scale、用于测试和导出。**

## 9.那我之前做DSCNN量化时，设计的INT8 MFCC前端又和原有的MFCC前端有什么区别，我本来用原有的MFCC前端不就做的fake quant吗，换成INT8 MFCC前端后也是fake quant，reference推理

对，你这个问题抓到核心了：**换成 INT8 MFCC 前端后，它仍然不是 bit-accurate 整数硬件实现，也仍然属于 reference / fake-quantized inference。**

但它和“原有 MFCC 前端 + fake quant”区别很大，主要区别在 **fake quant 插入的位置**。

原有 MFCC 前端做 fake quant，大致是：

```text
waveform
-> float pre-emphasis
-> float STFT / FFT
-> float power
-> float Mel
-> float log
-> float DCT
-> 得到 float MFCC
-> 对 MFCC 输出 fake quant
-> 送入 backbone
```

也就是说，**MFCC 内部计算本身还是高精度 float**。量化误差主要只发生在前端输出边界。

而 `Int8MFCCFrontend` / `Int8StreamingMFCCFrontend` 是：

```text
waveform quantize
-> pre-emphasis 后 requantize
-> window 系数量化
-> DFT cos/sin 系数量化
-> Mel filterbank 系数量化
-> log_mel requantize
-> DCT 系数量化
-> MFCC 输出 requantize
```

所以它虽然最后还是：

```text
int8 网格值 -> dequantize float -> 继续 PyTorch 计算
```

但它模拟的是：**MFCC 前端内部很多关键阶段也受到 INT8 系数/activation scale 的约束**。

最关键区别可以这样理解：

```text
原有 MFCC + fake quant:
  只问 “float MFCC 输出被量化后，模型还能不能用？”

INT8 MFCC frontend:
  进一步问 “如果 MFCC 前端内部的 window / DFT / Mel / DCT 等也按 INT8 约束，
  最后得到的 MFCC 还能不能用？”
```

所以你说“换成 INT8 MFCC 前端后也是 fake quant / reference 推理”是对的。更准确的结论是：

```text
它不是从 fake quant 变成真正整数硬件；
而是从“只在 MFCC 输出边界 fake quant”
推进到“MFCC 内部多阶段 INT8 reference 建模”。
```

这一步的价值在于：它比原有 float MFCC 更接近后续硬件前端。否则如果只用原有 MFCC 前端，结果可能过于乐观，因为真实硬件里 DFT/Mel/DCT 系数和中间量不会一直是 float。