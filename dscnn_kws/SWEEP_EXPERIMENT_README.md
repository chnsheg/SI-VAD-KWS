# DSCNN-KWS 三套 Sweep 实验流程说明

本文档根据当前仓库中的三个实验脚本整理：

```text
dscnn_kws/sweep_dscnn_acc.py
dscnn_kws/sweep_dscnn_noise_acc.py
dscnn_kws/sweep_fixed_dscnn_noise_snr_scene_acc.py
```

目标是把项目里从源代码、数据集、模型结构、训练、验证、测试、保存模型、生成结果表格到最终汇总分析的完整流程讲清楚，并说明各个环节中主要代码分别负责什么。

## 1. 总览

这三个脚本都围绕同一个核心任务：

```text
训练多个 DSCNN 结构，在两个 Mobvoi 二分类唤醒词数据集上比较模型大小与识别效果。
```

当前默认二分类标签来自 `dscnn_kws/configs.py`：

```text
CLASS_LIST = ["positive", "negative"]
CLASS_ENCODING = {"positive": 0, "negative": 1}
```

含义是：

- `positive`：当前关键词，也就是目标唤醒词。
- `negative`：非当前关键词，包括普通非唤醒词，也可能包括另一个相近唤醒词作为 hard negative。

三套 sweep 的区别：

| 脚本 | 主要目的 | 训练是否加噪声 | 验证/测试是否加噪声 | 是否做 TAU 场景/SNR 网格 |
| --- | --- | --- | --- | --- |
| `sweep_dscnn_acc.py` | clean 基线结构搜索 | 否 | 否 | 否 |
| `sweep_dscnn_noise_acc.py` | 在线 TAU 噪声增强训练，并可选按场景测试 | 是 | 是 | 可选 `--per_scene_test` |
| `sweep_fixed_dscnn_noise_snr_scene_acc.py` | 固定 SNR、固定 TAU 场景网格的鲁棒性评估 | 是，或 `--skip_train` 跳过训练 | 训练后的 test 使用固定 5 dB 列表；额外做 scene x SNR 网格 | 是，默认 10 个场景 x 5 个 SNR |

三套脚本本身不直接实现神经网络训练细节，而是批量调用：

```text
python -m dscnn_kws.train ...
```

真正的单次训练、验证、测试由这些文件完成：

```text
dscnn_kws/train.py
dscnn_kws/engine/trainer.py
dscnn_kws/data/dataset.py
dscnn_kws/model/dscnn.py
```

可以把三个 sweep 脚本理解为“实验调度器”，把 `train.py` 理解为“单次训练入口”。

## 2. 关键源文件分工

### 2.1 `dscnn_kws/sweep_dscnn_acc.py`

这是 clean 基线实验脚本。它负责：

- 定义默认数据集列表。
- 定义要遍历的 DSCNN 结构列表。
- 为每个结构生成 `model_size_info`。
- 调用 `python -m dscnn_kws.train` 训练模型。
- 强制关闭训练噪声和评估噪声。
- 从训练输出文本里解析 `best_valid_acc`、`test_loss`、`test_acc`、`precision`、`recall`、`f1`。
- 把每次训练得到的 `best.pt` 复制到统一目录。
- 持续写出 CSV、XLSX 和 shape report。
- 按准确率阈值寻找满足条件且参数量最小的模型。

核心函数：

```text
make_model_size_info()
  把 layers/channels 转成 DSCNN 构造函数需要的 model_size_info。

expected_params()
  按固定 DSCNN 结构公式估算参数量。

shape_trace()
  推导每个结构从 waveform 到 logits 的张量形状。

write_shape_report()
  把每个结构的形状变化写到 sweep_dscnn_shapes.txt。

write_xlsx()
  不依赖 openpyxl，手工生成一个简单 XLSX 文件。

copy_best_model()
  从 train.py 输出中找到 save_dir，再复制 save_dir/best.pt。

run_one()
  对一个 dataset + 一个 architecture 执行一次完整训练。

summarize()
  保存已有全部结果，并打印阈值筛选摘要。

main()
  双层循环遍历 ARCHS 和 DATASETS。
```

### 2.2 `dscnn_kws/sweep_dscnn_noise_acc.py`

这是在线噪声增强版本。它和 clean 脚本结构相似，但额外负责：

- 设置 TAU 噪声列表。
- 训练时开启 `--noise_aug`。
- 验证/测试时开启 `--eval_noise_aug`。
- 可通过环境变量或命令行覆盖噪声路径。
- 启动前检查噪声文件是否可用。
- 可选在每个 best checkpoint 后做 per-scene 测试。

核心函数：

```text
split_path_env()
  读取环境变量中的噪声路径，按 os.pathsep 拆成列表。

count_usable_noise_files()
  统计目录或 txt 列表中可用的 wav 噪声文件数。

build_scene_eval_loader()
  为 per-scene 测试构造 test DataLoader，固定使用 test_manifest.json。

build_scene_eval_model()
  用 checkpoint 重建 DSCNN + MFCCDSCNN 模型。

eval_scene_acc()
  计算 per-scene 测试中的 acc / precision / recall / f1。

run_scene_tests()
  对一个 checkpoint 遍历多个 TAU 场景并测试。

apply_cli_overrides()
  根据命令行参数覆盖 train/valid/test 噪声列表。

validate_noise_roots()
  确认噪声文件存在，否则提前报错。

run_one()
  对一个 dataset + architecture 做一次带噪声训练与测试。

summarize_scene_results()
  保存 per-scene 测试结果。
```

### 2.3 `dscnn_kws/sweep_fixed_dscnn_noise_snr_scene_acc.py`

这是目前最完整的噪声鲁棒性实验脚本。它不仅训练模型，还在训练后按固定 TAU 场景和固定 SNR 网格做额外测试。

它负责：

- 支持命令行指定数据集、结构、训练参数、噪声路径、输出前缀。
- 支持遍历所有内置结构，也支持只跑指定结构。
- 支持 `--skip_train --ckpt`，即不重新训练，直接拿已有 checkpoint 做 scene/SNR 测试。
- 训练时使用 TAU train 噪声列表。
- 验证和训练结束时的 test 使用固定验证 SNR。
- 额外对 test split 做 `scene x snr_db` 网格评估。
- 输出训练结果、网格结果、按场景汇总、按结构汇总。

核心函数：

```text
configure_outputs()
  根据 --out_prefix 设置输出 CSV 文件名和 best model 目录。

make_model_size_info()
  生成固定 DSCNN 结构描述。

expected_params()
  估算固定 DSCNN 参数量。

count_usable_noise_files()
  检查 TAU 噪声目录或 txt 列表中可用 wav 数量。

copy_best_model()
  复制 train.py 保存的 best.pt 到统一目录。

train_one()
  对一个 dataset + architecture 运行带噪声训练。

build_model()
  从 checkpoint 重建用于 scene/SNR 测试的模型。

build_eval_loader()
  为指定 scene 和 snr_db 构造 test DataLoader。

eval_acc()
  计算 scene/SNR 网格上的 acc / precision / recall / f1。

run_snr_scene_grid()
  遍历所有 scene 和 SNR，生成细粒度网格结果。

save_train_results()
  写训练阶段结果 CSV。

save_grid_results()
  写 scene/SNR 逐点结果 CSV。

summarize_scene_grid()
  把逐点结果聚合为 scene_summary 和 arch_summary。

save_summary_results()
  写聚合后的 summary CSV。

selected_archs()
  根据命令行参数选择要跑的结构。

main()
  串起输出配置、噪声检查、训练、网格测试和结果保存。
```

### 2.4 `dscnn_kws/train.py`

这是单次训练入口。三个 sweep 脚本都会通过子进程调用它。

它负责：

- 解析训练参数。
- 读取数据集路径。
- 构造 train/validation/test DataLoader。
- 根据参数构造 DSCNN 或 LSTM backbone。
- 构造 MFCC 或 bandpass 前端。
- 包装成 `MFCCDSCNN` 或 `MFCCLSTM`。
- 设置优化器、学习率调度器。
- 创建保存目录。
- 调用 `Trainer.fit()` 执行训练、验证和最终测试。

关键类和函数：

```text
MFCCDSCNN
  前端特征提取 + DSCNN backbone 的整体模型。
  输入 waveform，输出 logits。

parse_args()
  定义所有训练参数，例如 sample_rate、epoch、batch、noise_aug、model_size_info。

build_optimizer_scheduler()
  根据参数创建 Adam/SGD 和 cos/step scheduler。

main()
  单次训练完整入口。
```

### 2.5 `dscnn_kws/engine/trainer.py`

这是训练循环和指标计算实现。

它负责：

- 每个 epoch 跑训练。
- 每个 epoch 后跑 validation。
- 根据 validation accuracy 保存 `best.pt`。
- 训练结束保存 `last.pt`。
- 加载 `best.pt` 并在 test split 上输出最终测试指标。

关键函数：

```text
_run_train_epoch()
  单个 epoch 训练：forward、loss、backward、optimizer.step。

_run_eval()
  验证/测试：关闭梯度，计算 loss、acc、macro precision、macro recall、macro f1。

fit()
  完整训练流程：train -> valid -> save best -> final test。
```

最终输出格式：

```text
[TEST] loss=... acc=... precision=... recall=... f1=...
```

三个 sweep 脚本就是通过正则表达式解析这一行，提取测试指标。

### 2.6 `dscnn_kws/data/dataset.py`

这是数据读取和在线噪声混合的核心。

它负责：

- 读取 manifest。
- 根据 `audio_filepath` 找到实际 wav。
- 加载音频并统一为单声道 float32。
- 按目标 sample rate 检查或在线重采样。
- 把音频补齐或裁剪为 1 秒。
- 训练时做随机时间裁剪。
- 验证/测试时做居中裁剪。
- 如果开启 noise augmentation，则按 SNR 混入噪声。
- 构造 train/validation/test DataLoader。

关键函数：

```text
SpeechCommandDataset._load_speech_dataset()
  读取 train_manifest.json / validation_manifest.json / test_manifest.json。

SpeechCommandDataset._load_noise_dataset()
  从 _background_noise_、噪声目录或 txt 列表加载噪声 wav 路径。

SpeechCommandDataset._load_audio()
  加载单条语音样本，做重采样、补零、裁剪和噪声混合。

SpeechCommandDataset._apply_noise()
  选择噪声片段，按指定 SNR 混到语音上。

build_dataloaders()
  为 train/validation/test 创建三个 DataLoader。
```

### 2.7 `dscnn_kws/model/dscnn.py`

这是 DSCNN backbone 的定义。

它负责：

- 解析 `model_size_info`。
- 构造第一层普通 `Conv2d + BatchNorm + ReLU`。
- 构造后续 depthwise separable conv block。
- 做 adaptive average pooling。
- 做 dropout。
- 用 final linear 输出二分类 logits。

关键类和函数：

```text
DepthwiseSeparableConv2d
  后续 DSCNN block：depthwise conv + pointwise conv。

DSCNN
  完整 DSCNN backbone。

calculate_time_steps()
  根据 sample_rate 和 window_stride_ms 估计 MFCC 时间帧数。
```

## 3. 数据集来源与组织方式

### 3.1 默认数据集

三个 sweep 脚本默认都使用这两个数据集：

```text
mobvoi_hi_xiaowen_binary_hardneg
mobvoi_nihao_wenwen_binary_hardneg
```

默认根目录：

```text
dscnn_kws/data
```

因此预期路径是：

```text
dscnn_kws/data/mobvoi_hi_xiaowen_binary_hardneg/
dscnn_kws/data/mobvoi_nihao_wenwen_binary_hardneg/
```

每个数据集目录中需要有：

```text
train_manifest.json
validation_manifest.json
test_manifest.json
```

这些 manifest 是 JSON Lines 格式，也就是一行一个 JSON 对象。

典型字段：

```json
{"audio_filepath": ".../xxx.wav", "command": "positive"}
{"audio_filepath": ".../yyy.wav", "command": "negative"}
```

`SpeechCommandDataset` 读取时只关心：

```text
audio_filepath
command
```

如果 `command` 在 `CLASS_LIST` 中，就保留原类名；否则会映射为 `"unknown"`。当前二分类任务只定义了 `positive` 和 `negative`，所以 manifest 应该只包含这两个类别，否则会找不到 `unknown` 的 class encoding。

### 3.2 hard negative 的含义

`*_hardneg` 表示 negative 中包含更难的负样本。比如：

- 在 `hi_xiaowen` 任务中，`hi_xiaowen` 是 positive。
- 普通非唤醒词是 negative。
- 另一个唤醒词 `nihao_wenwen` 也可以被当作 negative。

这样训练出来的模型不只会区分“唤醒词 vs 普通语音”，还会区分两个相近唤醒词。

### 3.3 train / validation / test 的用法

三个 split 在代码中的用途：

```text
train_manifest.json
  用于训练参数更新。

validation_manifest.json
  每个 epoch 后评估，用于选择 best.pt。

test_manifest.json
  训练完成后最终测试；也用于额外 scene/SNR grid 测试。
```

在 `build_dataloaders()` 中：

```text
train_dataset:
  is_training=True
  shuffle=True
  drop_last=True

valid_dataset:
  is_training=False
  shuffle=False
  drop_last=False

test_dataset:
  is_training=False
  shuffle=False
  drop_last=False
```

训练集会打乱并丢掉最后一个不满 batch 的 batch；验证和测试会保留全部样本。

## 4. 噪声数据来源与组织方式

### 4.1 TAU 噪声列表

噪声训练和噪声测试默认使用：

```text
dscnn_kws/noise/lists/tau_train.txt
dscnn_kws/noise/lists/tau_valid.txt
dscnn_kws/noise/lists/tau_test.txt
```

这些 txt 文件中每行指向一个 wav 噪声文件。空行和以 `#` 开头的行会被忽略。

相关代码：

```text
count_usable_noise_files()
  统计 txt 或目录中可用 wav。

SpeechCommandDataset._load_noise_dataset()
  真正加载噪声路径列表。
```

### 4.2 TAU 场景目录

按场景测试时默认使用：

```text
dscnn_kws/noise/tau/<scene>/*.wav
```

默认场景：

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

`sweep_fixed_dscnn_noise_snr_scene_acc.py` 会对每个 scene 再遍历多个 SNR：

```text
20 dB, 10 dB, 5 dB, 0 dB, -5 dB
```

### 4.3 在线混噪逻辑

在线混噪发生在 `SpeechCommandDataset._apply_noise()` 中。

逻辑是：

1. 如果 `noise_aug=False` 或没有噪声文件，直接返回原始 waveform。
2. 按 `noise_prob` 决定当前样本是否加噪。
3. 随机选择一个噪声 wav。
4. 如果噪声短于 1 秒，重复拼接到足够长。
5. 从噪声中随机截取 1 秒片段。
6. 在 `[noise_snr_min_db, noise_snr_max_db]` 区间随机采样 SNR。
7. 调用 `_mix_at_snr()` 按目标 SNR 调整噪声 RMS 并混入语音。
8. 如果峰值太大，会缩放到不超过 0.99，最后 clamp 到 `[-1, 1]`。

固定 SNR 测试时，代码把：

```text
noise_snr_min_db = snr_db
noise_snr_max_db = snr_db
```

因此每个样本都按同一个 SNR 混噪。

## 5. 模型结构 Sweep

### 5.1 ARCHS 列表

三个脚本都使用同一组固定结构：

```text
L5_C64, L5_C48, L5_C32, L5_C24, L5_C16,
L4_C16, L3_C16,
L5_C12, L4_C12, L3_C12,
L5_C8, L4_C8, L3_C8,
L2_C16, L2_C12, L2_C8,
L5_C6, L4_C6, L3_C6, L2_C6,
L1_C8, L3_C4, L2_C4, L1_C6, L1_C4
```

命名规则：

```text
L<num_layers>_C<channels>
```

例如：

```text
L5_C64 = 5 层 DSCNN，每层输出通道数 64
L1_C4  = 1 层 DSCNN，输出通道数 4
```

### 5.2 `model_size_info`

`make_model_size_info(num_layers, channels)` 生成如下结构描述：

```text
[
  num_layers,
  channels, 10, 4, 2, 2,
  channels, 3, 3, 1, 1,
  channels, 3, 3, 1, 1,
  ...
]
```

含义：

```text
第一层:
  channels 输出通道
  kernel_t = 10
  kernel_f = 4
  stride_t = 2
  stride_f = 2

后续每层:
  channels 输出通道
  kernel_t = 3
  kernel_f = 3
  stride_t = 1
  stride_f = 1
```

在 `DSCNN.__init__()` 中：

- 第 0 层使用普通 `Conv2d`。
- 第 1 层及以后使用 `DepthwiseSeparableConv2d`。

### 5.3 参数量估算

三个脚本都用：

```text
expected_params(num_layers, channels)
```

公式：

```text
C = channels
N = num_layers
K = num_classes

expected_params = (N - 1) * C * C + (42 + 13 * (N - 1) + K) * C + K
```

当前 `K = 2`。

大致来源：

```text
第一层 Conv:       10 * 4 * C = 40C
第一层 BN:         2C
每个 DS block:     C*C + 9C + 4C = C*C + 13C
最终 Linear:       C*K + K
```

这个公式适用于这三套 sweep 里的固定 DSCNN 结构，不代表任意模型的真实参数量。真实 PyTorch 参数数由 `train.py` 中：

```text
parameter_number(model) = sum(p.numel() for p in model.parameters())
```

打印为：

```text
[INFO] device=..., params=...
```

sweep 脚本会把这个日志里的 `params=...` 解析为 `printed_params`。

## 6. 单次训练过程

三个 sweep 脚本最终都会构造一个命令：

```text
python -m dscnn_kws.train ...
```

`train.py` 的主流程：

1. 解析命令行参数。
2. 设置随机种子。
3. 准备 device。
4. 构造数据路径：`root / dataset`。
5. 构造 train/validation/test DataLoader。
6. 计算 MFCC 时间帧数。
7. 构造 DSCNN backbone。
8. 构造 MFCCDSCNN wrapper。
9. 构造 optimizer 和 scheduler。
10. 创建保存目录。
11. 交给 `Trainer.fit()` 执行训练。

### 6.1 输入音频处理

`SpeechCommandDataset` 输出：

```text
waveform shape = [1, sample_rate]
```

当前 sweep 默认：

```text
sample_rate = 16000
sample_length = 16000
```

所以每条样本是 1 秒音频。

处理细节：

- 多声道音频会求平均变成单声道。
- 采样率不匹配时，如果允许 online resample，就重采样；否则严格模式会报错。
- 短于 1 秒会右侧补零。
- 训练时会前后 pad 10%，然后随机裁剪 1 秒。
- 验证/测试时如果长于 1 秒，会居中裁剪。

### 6.2 MFCC 前端

`MFCCDSCNN.forward()` 中：

1. 输入 waveform。
2. 可选 pre-emphasis。
3. 计算 MFCC 或 bandpass 特征。
4. 默认取前 `dct_coeff` 个 MFCC 系数。
5. 转置并 flatten。
6. 输入 DSCNN backbone。

三个 sweep 默认：

```text
frontend = mfcc
sample_rate = 16000
dct_coeff = 10
window_size_ms = 32
window_stride_ms = 32
```

`calculate_time_steps()` 计算：

```text
time_steps = floor(16000 / (16000 * 32 / 1000)) + 1
           = floor(16000 / 512) + 1
           = 31 + 1
           = 32
```

因此输入 DSCNN 前的 MFCC 特征通常可理解为：

```text
32 帧 x 10 个系数 = 320 维
```

### 6.3 DSCNN backbone

DSCNN 输入：

```text
[B, 320]
```

在 `DSCNN.forward()` 中 reshape 为：

```text
[B, 1, 32, 10]
```

然后进入卷积层：

- 第一层普通 2D Conv。
- 后续层 depthwise separable Conv。
- Adaptive average pool 到 `[B, C, 1, 1]`。
- Dropout。
- Linear 输出 `[B, 2]` logits。

### 6.4 训练循环

`Trainer.fit()` 中每个 epoch：

1. `_run_train_epoch()` 训练一轮。
2. `_run_eval(valid_loader)` 验证一轮。
3. 调用 scheduler 更新学习率。
4. 如果当前 `valid_acc` 超过历史最好，就保存 `best.pt`。

训练结束：

1. 保存 `last.pt`。
2. 加载 `best.pt`。
3. 在 test_loader 上跑 `_run_eval()`。
4. 打印最终 `[TEST] ...` 指标。

### 6.5 指标计算

普通分类测试使用：

```text
preds = argmax(logits, dim=1)
```

指标：

```text
loss      = CrossEntropyLoss 的 batch 平均
acc       = correct / total
precision = sklearn precision_score(..., average="macro", zero_division=0)
recall    = sklearn recall_score(..., average="macro", zero_division=0)
f1        = sklearn f1_score(..., average="macro", zero_division=0)
```

二分类下，macro 指标表示对 `positive` 和 `negative` 两个类别分别计算，再取平均。

## 7. 实验一：clean sweep

脚本：

```bash
python dscnn_kws/sweep_dscnn_acc.py
```

### 7.1 默认设置

```text
ROOT = .\dscnn_kws\data
DATASETS = mobvoi_hi_xiaowen_binary_hardneg, mobvoi_nihao_wenwen_binary_hardneg
EPOCH = 30
BATCH = 128
SAMPLE_RATE = 16000
GPU = 0
NUM_WORKERS = 0
DCT_COEFF = 10
WINDOW_SIZE_MS = 32
WINDOW_STRIDE_MS = 32
```

### 7.2 训练命令特点

`run_one()` 构造的命令包含：

```text
--no-noise_aug
--no-eval_noise_aug
```

因此：

- train 不加噪声。
- validation 不加噪声。
- test 不加噪声。

这是最干净的结构基线，用于回答：

```text
在没有噪声扰动时，不同 DSCNN 大小能达到什么准确率？
```

### 7.3 执行顺序

`main()` 中：

```text
for arch in ARCHS:
    for dataset in DATASETS:
        run_one(dataset, arch)
        summarize(results)
```

也就是先遍历结构，再遍历两个数据集。

每完成一次训练就立即调用 `summarize()` 保存已有结果，避免中途停止导致已完成结果丢失。

### 7.4 输出文件

clean sweep 输出：

```text
sweep_dscnn_acc_results.csv
sweep_dscnn_acc_results.xlsx
sweep_dscnn_shapes.txt
dscnn_kws/runs/sweep_best_models/*.pt
```

CSV 字段：

```text
dataset
arch
layers
channels
mfcc_output
expected_params
printed_params
best_valid_acc
test_loss
test_acc
precision
recall
f1
returncode
train_save_dir
best_model_saved_as
```

字段含义：

- `dataset`：当前数据集名。
- `arch`：结构名，例如 `L5_C64`。
- `layers`：DSCNN 层数。
- `channels`：每层通道数。
- `mfcc_output`：MFCC 特征形状，例如 `10x32`。
- `expected_params`：脚本公式估算参数量。
- `printed_params`：`train.py` 打印的实际 PyTorch 参数量。
- `best_valid_acc`：训练过程中最高 validation accuracy。
- `test_loss`：加载 best checkpoint 后的 test loss。
- `test_acc`：加载 best checkpoint 后的 test accuracy。
- `precision/recall/f1`：test split 上的 macro 指标。
- `returncode`：训练子进程退出码。
- `train_save_dir`：`train.py` 原始保存目录。
- `best_model_saved_as`：复制后的统一 best checkpoint 路径。

### 7.5 阈值摘要

`summarize()` 中默认看三个阈值：

```text
0.92, 0.95, 0.98
```

它会做两类筛选：

1. 对每个 dataset，按参数量从小到大找第一个 `test_acc >= threshold` 的结构。
2. 对两个 dataset 共同满足阈值的结构，找参数量最小的结构。

这个摘要用于快速回答：

```text
满足某个准确率目标时，最小模型是哪一个？
```

## 8. 实验二：在线噪声增强 sweep

脚本：

```bash
python dscnn_kws/sweep_dscnn_noise_acc.py
```

### 8.1 默认设置

主要训练参数：

```text
EPOCH = 30
BATCH = 128
SAMPLE_RATE = 16000
GPU = 1
NUM_WORKERS = 8
DCT_COEFF = 10
WINDOW_SIZE_MS = 32
WINDOW_STRIDE_MS = 32
```

噪声参数：

```text
TRAIN_NOISE_PROB = 0.8
TRAIN_SNR_MIN_DB = -5.0
TRAIN_SNR_MAX_DB = 20.0
EVAL_NOISE_PROB = 1.0
EVAL_SNR_MIN_DB = 0.0
EVAL_SNR_MAX_DB = 20.0
```

默认噪声列表：

```text
TRAIN_NOISE_ROOTS = ./dscnn_kws/noise/lists/tau_train.txt
VALID_NOISE_ROOTS = ./dscnn_kws/noise/lists/tau_valid.txt
TEST_NOISE_ROOTS  = ./dscnn_kws/noise/lists/tau_test.txt
```

### 8.2 噪声路径覆盖方式

脚本支持两种覆盖。

第一种是环境变量：

```bash
TAU_NOISE_ROOTS=./dscnn_kws/noise/tau/airport python dscnn_kws/sweep_dscnn_noise_acc.py
```

或者分别指定：

```bash
TAU_TRAIN_NOISE_ROOTS=...
TAU_VALID_NOISE_ROOTS=...
TAU_TEST_NOISE_ROOTS=...
python dscnn_kws/sweep_dscnn_noise_acc.py
```

第二种是命令行：

```bash
python dscnn_kws/sweep_dscnn_noise_acc.py \
  --train_noise_roots ./dscnn_kws/noise/lists/tau_train.txt \
  --valid_noise_roots ./dscnn_kws/noise/lists/tau_valid.txt \
  --test_noise_roots ./dscnn_kws/noise/lists/tau_test.txt
```

如果传：

```bash
--noise_roots <path>
```

则 train/valid/test 都使用同一套噪声。

### 8.3 噪声检查

`main()` 一开始会调用：

```text
validate_noise_roots("TRAIN_NOISE_ROOTS", TRAIN_NOISE_ROOTS)
validate_noise_roots("VALID_NOISE_ROOTS", VALID_NOISE_ROOTS)
validate_noise_roots("TEST_NOISE_ROOTS", TEST_NOISE_ROOTS)
```

如果某组噪声没有可用 wav，会直接报错停止。

这一步避免训练到一半才发现噪声路径错了。

### 8.4 训练命令特点

`run_one()` 构造的命令包含：

```text
--noise_aug
--eval_noise_aug
--train_noise_roots ...
--valid_noise_roots ...
--test_noise_roots ...
--noise_aug_prob 0.8
--noise_snr_min_db -5.0
--noise_snr_max_db 20.0
--eval_noise_aug_prob 1.0
--eval_noise_snr_min_db 0.0
--eval_noise_snr_max_db 20.0
```

因此：

- train：80% 概率加噪，SNR 随机在 -5 到 20 dB。
- validation：100% 加噪，SNR 随机在 0 到 20 dB。
- test：100% 加噪，SNR 随机在 0 到 20 dB。

### 8.5 可选 per-scene 测试

如果加：

```bash
python dscnn_kws/sweep_dscnn_noise_acc.py --per_scene_test
```

每个训练完成后的 best checkpoint 会额外调用：

```text
run_scene_tests()
```

对默认 10 个 TAU scene 分别测试。

这里的 per-scene 测试使用：

```text
test_manifest.json
noise_aug=True
noise_prob=1.0
SNR 在 0 到 20 dB 随机采样
seed = 300000 + scene_idx * 10007
```

它和第三个脚本的固定 SNR 网格不同：这里每个 scene 内 SNR 是随机区间，不是固定枚举 `20/10/5/0/-5`。

### 8.6 输出文件

主训练输出：

```text
sweep_dscnn_noise_acc_results.csv
sweep_dscnn_noise_acc_results.xlsx
sweep_dscnn_noise_shapes.txt
dscnn_kws/runs/sweep_noise_best_models/*.pt
```

如果开启 `--per_scene_test`，额外输出：

```text
sweep_dscnn_noise_scene_results.csv
sweep_dscnn_noise_scene_results.xlsx
```

主 CSV 比 clean 版本多了噪声配置字段：

```text
train_noise_roots
valid_noise_roots
test_noise_roots
train_noise_prob
train_snr_min_db
train_snr_max_db
eval_noise_prob
eval_snr_min_db
eval_snr_max_db
```

per-scene CSV 字段：

```text
dataset
arch
layers
channels
scene
scene_noise_root
usable_noise_files
eval_noise_prob
eval_snr_min_db
eval_snr_max_db
expected_params
test_acc
precision
recall
f1
num_samples
ckpt
```

## 9. 实验三：固定 SNR / TAU 场景网格 sweep

脚本：

```bash
python dscnn_kws/sweep_fixed_dscnn_noise_snr_scene_acc.py
```

这是三套脚本里最适合做鲁棒性分析的一套。

### 9.1 默认设置

```text
ROOT = ./dscnn_kws/data
DEFAULT_DATASETS = mobvoi_hi_xiaowen_binary_hardneg, mobvoi_nihao_wenwen_binary_hardneg
EPOCH = 30
BATCH = 256
SAMPLE_RATE = 16000
GPU = 1
NUM_WORKERS = 8
DCT_COEFF = 10
WINDOW_SIZE_MS = 32
WINDOW_STRIDE_MS = 32
```

训练噪声：

```text
TRAIN_NOISE_PROB = 0.8
TRAIN_SNR_MIN_DB = -5.0
TRAIN_SNR_MAX_DB = 20.0
```

验证和训练结束时的 test 噪声：

```text
VALID_NOISE_PROB = 1.0
VALID_SNR_DB = 5.0
```

额外 grid 测试默认 SNR：

```text
20.0, 10.0, 5.0, 0.0, -5.0
```

### 9.2 输出配置

脚本支持：

```bash
--out_prefix snr_scene_arch_sweep
```

`configure_outputs()` 会生成：

```text
<out_prefix>_train_results.csv
<out_prefix>_grid_results.csv
<out_prefix>_scene_summary.csv
<out_prefix>_arch_summary.csv
dscnn_kws/runs/<out_prefix>_best_models/
```

默认：

```text
snr_scene_arch_sweep_train_results.csv
snr_scene_arch_sweep_grid_results.csv
snr_scene_arch_sweep_scene_summary.csv
snr_scene_arch_sweep_arch_summary.csv
dscnn_kws/runs/snr_scene_arch_sweep_best_models/
```

可以用：

```bash
--no-write_outputs
```

关闭 CSV 写出，适合只想看控制台输出的调试场景。

### 9.3 结构选择

默认跑全部 `ARCHS` 和全部默认数据集。

只跑部分内置结构：

```bash
python dscnn_kws/sweep_fixed_dscnn_noise_snr_scene_acc.py \
  --arch_names L5_C64 L3_C16
```

只跑一个自定义结构：

```bash
python dscnn_kws/sweep_fixed_dscnn_noise_snr_scene_acc.py \
  --single_arch \
  --arch_name L3_C16 \
  --num_layers 3 \
  --channels 16
```

直接传完整 `model_size_info`：

```bash
python dscnn_kws/sweep_fixed_dscnn_noise_snr_scene_acc.py \
  --single_arch \
  --arch_name custom \
  --num_layers 3 \
  --channels 16 \
  --model_size_info 3 16 10 4 2 2 16 3 3 1 1 16 3 3 1 1
```

`selected_archs()` 负责把命令行参数转换成待实验结构列表。

### 9.4 训练阶段

如果不传 `--skip_train`，每个 dataset + architecture 会调用：

```text
train_one()
```

训练命令包含：

```text
--noise_aug
--eval_noise_aug
--train_noise_roots tau_train.txt
--valid_noise_roots tau_valid.txt
--test_noise_roots tau_test.txt
--noise_aug_prob 0.8
--noise_snr_min_db -5.0
--noise_snr_max_db 20.0
--eval_noise_aug_prob 1.0
--eval_noise_snr_min_db 5.0
--eval_noise_snr_max_db 5.0
```

因此：

- train：80% 概率混噪，SNR 在 -5 到 20 dB 随机。
- validation：100% 混噪，固定 5 dB。
- train.py 最终 test：100% 混噪，固定 5 dB。

训练完成后：

1. 从日志解析所有 validation acc。
2. 得到 `best_valid_acc = max(valid_accs)`。
3. 从 `[TEST] ...` 日志解析 `test_acc_on_tau_test_list` 和 `f1_on_tau_test_list`。
4. 调用 `copy_best_model()` 复制 best checkpoint。
5. 保存到 train CSV。

### 9.5 跳过训练，直接评估已有 checkpoint

可以使用：

```bash
python dscnn_kws/sweep_fixed_dscnn_noise_snr_scene_acc.py \
  --skip_train \
  --ckpt dscnn_kws/runs/snr_scene_arch_sweep_best_models/xxx.pt \
  --single_arch \
  --arch_name L3_C16 \
  --num_layers 3 \
  --channels 16 \
  --datasets mobvoi_hi_xiaowen_binary_hardneg
```

限制：

```text
--skip_train 当前只支持一个 dataset 和一个 architecture。
```

如果同时给多个结构或多个数据集，代码会主动报错。

### 9.6 scene x SNR 网格测试

训练完成并拿到 checkpoint 后，脚本调用：

```text
run_snr_scene_grid()
```

默认遍历：

```text
10 个 TAU scene x 5 个 SNR = 50 个测试点
```

每个测试点：

1. 使用同一个 `test_manifest.json`。
2. 使用当前 scene 目录作为噪声源。
3. 设置 `noise_prob=1.0`。
4. 设置 `noise_snr_min_db = snr_db`。
5. 设置 `noise_snr_max_db = snr_db`。
6. 使用 deterministic noise。
7. 用当前 checkpoint 跑完整 test split。
8. 输出 `acc / precision / recall / f1 / num_samples`。

每个 scene/SNR 的随机种子：

```text
seed = 500000 + scene_idx * 10007 + snr_idx * 101
```

这保证相同 scene/SNR 配置下的噪声选择和截取是可复现的。

### 9.7 grid CSV

逐点结果保存到：

```text
snr_scene_arch_sweep_grid_results.csv
```

字段：

```text
dataset
arch
layers
channels
expected_params
scene
snr_db
scene_noise_root
usable_noise_files
acc
precision
recall
f1
num_samples
ckpt
```

一行代表：

```text
某个 dataset + 某个 arch + 某个 scene + 某个 SNR 的测试结果。
```

### 9.8 scene summary

`summarize_scene_grid()` 会按：

```text
dataset + arch + layers + channels + expected_params + scene
```

分组。

也就是说，对同一个模型在同一个 TAU scene 下，把不同 SNR 的结果聚合起来。

输出：

```text
snr_scene_arch_sweep_scene_summary.csv
```

字段：

```text
dataset
arch
layers
channels
expected_params
scene
num_snr_points
mean_acc
min_acc
max_acc
mean_f1
min_f1
max_f1
snrs
```

含义：

- `num_snr_points`：该 scene 下有多少个 SNR 测试点。
- `mean_acc`：该 scene 下多个 SNR 的平均准确率。
- `min_acc`：该 scene 下最差 SNR 的准确率。
- `max_acc`：该 scene 下最好 SNR 的准确率。
- `mean_f1/min_f1/max_f1`：同理，对 f1 做统计。
- `snrs`：参与统计的 SNR 列表。

### 9.9 arch summary

`summarize_scene_grid()` 还会按：

```text
dataset + arch + layers + channels + expected_params
```

分组。

也就是说，对同一个模型在所有 scene 和所有 SNR 下的结果聚合。

输出：

```text
snr_scene_arch_sweep_arch_summary.csv
```

字段：

```text
dataset
arch
layers
channels
expected_params
num_eval_points
mean_acc
min_acc
max_acc
mean_f1
min_f1
max_f1
```

含义：

- `num_eval_points`：总测试点数，默认最多是 50。
- `mean_acc`：整体平均鲁棒性。
- `min_acc`：所有 scene/SNR 中最坏工况准确率。
- `max_acc`：最好工况准确率。
- `mean_f1/min_f1/max_f1`：同理，对 f1 做统计。

实际选模型时，`min_acc` 和 `min_f1` 很重要，因为它们代表最差噪声条件下模型是否还能工作。

## 10. 三套实验的结果关系

可以把三套脚本理解为逐步加难：

### 10.1 clean baseline

使用：

```text
sweep_dscnn_acc.py
```

回答：

```text
没有噪声干扰时，模型容量和准确率之间是什么关系？
```

适合先确定：

- 模型结构是否正常。
- 小模型是否有足够表达能力。
- clean 数据集本身是否可学。

### 10.2 noise augmentation

使用：

```text
sweep_dscnn_noise_acc.py
```

回答：

```text
训练和评估都加入 TAU 噪声后，模型表现如何？
```

适合观察：

- 噪声增强是否提升鲁棒性。
- 哪些结构在噪声条件下仍能保持准确率。
- per-scene 随机 SNR 测试下不同场景是否有明显差异。

### 10.3 fixed scene/SNR grid

使用：

```text
sweep_fixed_dscnn_noise_snr_scene_acc.py
```

回答：

```text
在具体 TAU 场景和具体 SNR 条件下，模型鲁棒性曲线如何？
```

适合最终分析：

- 哪个 scene 最难。
- 哪个 SNR 是性能断崖点。
- 某个模型的平均性能和最坏性能。
- 在参数量约束下，哪个模型最稳。

## 11. 从源文件到最终结果的完整链路

完整链路如下：

```text
原始 wav 数据
  ↓
manifest 生成脚本把 wav 标成 positive / negative
  ↓
train_manifest.json / validation_manifest.json / test_manifest.json
  ↓
SpeechCommandDataset 读取 manifest 和 wav
  ↓
可选在线混入 TAU noise
  ↓
DataLoader 组成 batch
  ↓
MFCCDSCNN 前端提取 MFCC
  ↓
DSCNN backbone 输出 logits
  ↓
CrossEntropyLoss 训练
  ↓
每个 epoch 在 validation split 上选 best.pt
  ↓
加载 best.pt 在 test split 上输出 [TEST] 指标
  ↓
sweep 脚本解析日志，复制 best.pt
  ↓
写 CSV / XLSX / shape report
  ↓
可选额外 scene/SNR grid 测试
  ↓
生成 grid results / scene summary / arch summary
```

每个环节对应代码：

```text
manifest 读取:
  dscnn_kws/data/dataset.py
  SpeechCommandDataset._load_speech_dataset()

音频加载和裁剪:
  dscnn_kws/data/dataset.py
  SpeechCommandDataset._load_audio()

噪声加载:
  dscnn_kws/data/dataset.py
  SpeechCommandDataset._load_noise_dataset()

在线混噪:
  dscnn_kws/data/dataset.py
  SpeechCommandDataset._apply_noise()
  _mix_at_snr()

DataLoader:
  dscnn_kws/data/dataset.py
  build_dataloaders()

前端特征:
  dscnn_kws/train.py
  MFCCDSCNN.forward()

DSCNN 模型:
  dscnn_kws/model/dscnn.py
  DSCNN
  DepthwiseSeparableConv2d

单次训练入口:
  dscnn_kws/train.py
  main()

训练和验证循环:
  dscnn_kws/engine/trainer.py
  Trainer.fit()
  _run_train_epoch()
  _run_eval()

clean 实验调度:
  dscnn_kws/sweep_dscnn_acc.py
  run_one()
  summarize()
  main()

在线噪声实验调度:
  dscnn_kws/sweep_dscnn_noise_acc.py
  validate_noise_roots()
  run_one()
  run_scene_tests()
  summarize()
  summarize_scene_results()
  main()

固定 scene/SNR 实验调度:
  dscnn_kws/sweep_fixed_dscnn_noise_snr_scene_acc.py
  train_one()
  run_snr_scene_grid()
  summarize_scene_grid()
  save_summary_results()
  main()
```

## 12. 推荐实验顺序

建议按这个顺序跑：

### Step 1：确认 clean 基线

```bash
python dscnn_kws/sweep_dscnn_acc.py
```

观察：

```text
sweep_dscnn_acc_results.csv
sweep_dscnn_shapes.txt
dscnn_kws/runs/sweep_best_models/
```

先确认 clean 条件下模型能正常收敛。

### Step 2：跑在线噪声训练

```bash
python dscnn_kws/sweep_dscnn_noise_acc.py
```

如果想额外看随机 SNR 下每个场景：

```bash
python dscnn_kws/sweep_dscnn_noise_acc.py --per_scene_test
```

观察：

```text
sweep_dscnn_noise_acc_results.csv
sweep_dscnn_noise_scene_results.csv
dscnn_kws/runs/sweep_noise_best_models/
```

### Step 3：跑固定 scene/SNR 网格

```bash
python dscnn_kws/sweep_fixed_dscnn_noise_snr_scene_acc.py
```

如果已经有 checkpoint，只想评估一个模型：

```bash
python dscnn_kws/sweep_fixed_dscnn_noise_snr_scene_acc.py \
  --skip_train \
  --ckpt dscnn_kws/runs/snr_scene_arch_sweep_best_models/<model>.pt \
  --single_arch \
  --arch_name L3_C16 \
  --num_layers 3 \
  --channels 16 \
  --datasets mobvoi_hi_xiaowen_binary_hardneg
```

观察：

```text
snr_scene_arch_sweep_train_results.csv
snr_scene_arch_sweep_grid_results.csv
snr_scene_arch_sweep_scene_summary.csv
snr_scene_arch_sweep_arch_summary.csv
dscnn_kws/runs/snr_scene_arch_sweep_best_models/
```

## 13. 如何阅读最终结果

### 13.1 先看 returncode

所有 sweep 训练结果中：

```text
returncode = 0
```

才表示训练子进程正常结束。

如果非 0，相关指标可能为空，不应纳入统计。

### 13.2 再看 best_valid_acc

`best_valid_acc` 表示训练过程中 validation split 的最高准确率。

它用于判断训练是否正常，也用于选择 `best.pt`。

但最终泛化表现要看 test 指标。

### 13.3 clean/noise 主表看 test_acc 和 f1

clean 和 noise 主表里重点看：

```text
test_acc
precision
recall
f1
expected_params
printed_params
```

如果 test_acc 高但 f1 低，说明可能类别表现不均衡。

### 13.4 固定 scene/SNR 先看 arch_summary

`arch_summary` 适合做模型总体比较：

```text
mean_acc
min_acc
mean_f1
min_f1
expected_params
```

建议不要只看 `mean_acc`。如果部署环境复杂，`min_acc` 更能反映最坏情况风险。

### 13.5 再看 scene_summary 找薄弱场景

`scene_summary` 能回答：

```text
哪个 TAU 场景最难？
```

如果某个 scene 的 `min_acc` 或 `min_f1` 明显低，说明模型对该场景噪声不稳。

### 13.6 最后看 grid_results 找断崖 SNR

`grid_results` 最细，能回答：

```text
在 airport 的 0 dB 时准确率是多少？
在 bus 的 -5 dB 时 f1 是否崩掉？
同一个模型从 20 dB 到 -5 dB 的下降曲线怎样？
```

做图或详细分析时主要用这个文件。

## 14. 三个脚本的主要输出对照

| 脚本 | 主结果 | 模型目录 | 额外结果 |
| --- | --- | --- | --- |
| `sweep_dscnn_acc.py` | `sweep_dscnn_acc_results.csv` | `dscnn_kws/runs/sweep_best_models/` | `sweep_dscnn_acc_results.xlsx`, `sweep_dscnn_shapes.txt` |
| `sweep_dscnn_noise_acc.py` | `sweep_dscnn_noise_acc_results.csv` | `dscnn_kws/runs/sweep_noise_best_models/` | `sweep_dscnn_noise_scene_results.csv` when `--per_scene_test` |
| `sweep_fixed_dscnn_noise_snr_scene_acc.py` | `<prefix>_train_results.csv`, `<prefix>_grid_results.csv` | `dscnn_kws/runs/<prefix>_best_models/` | `<prefix>_scene_summary.csv`, `<prefix>_arch_summary.csv` |

## 15. 常见注意事项

### 15.1 数据路径必须匹配

默认 root 是：

```text
dscnn_kws/data
```

如果数据不在这里，需要改脚本常量或传参数。第三个脚本支持：

```bash
--root /path/to/data
```

前两个脚本主要通过脚本内常量控制。

### 15.2 噪声路径必须可用

噪声脚本会检查噪声 wav 数量。如果 txt 里的路径是相对路径，会相对 txt 文件所在目录解析。

### 15.3 sample rate 要匹配

三个 sweep 默认：

```text
sample_rate = 16000
```

训练命令都传了：

```text
--allow_online_resample
--no-verify_sample_rate
```

因此允许在线重采样，并跳过预先采样率检查。这提高了兼容性，但也意味着读取时会有额外重采样开销。

### 15.4 clean 和 noise 结果不能简单横比

clean 测试不加噪声，noise 测试加噪声。noise 下准确率低是正常的。

合理比较方式：

- 同一测试条件下比较不同结构。
- 同一结构比较 clean 与 noise，用于观察噪声损失。
- 不要把 clean 的 test_acc 当作噪声部署性能。

### 15.5 `sweep_dscnn_noise_acc.py` 的 per-scene 和固定 grid 不一样

`sweep_dscnn_noise_acc.py --per_scene_test`：

```text
每个 scene 内 SNR 在 0 到 20 dB 随机。
```

`sweep_fixed_dscnn_noise_snr_scene_acc.py`：

```text
每个 scene 明确测试 20, 10, 5, 0, -5 dB。
```

所以固定 grid 更适合做严谨的 SNR 曲线分析。

### 15.6 best checkpoint 是按 validation acc 选择

`Trainer.fit()` 保存 best 的规则是：

```text
if valid_m.acc > best_acc:
    save best.pt
```

因此最终 test 结果是：

```text
validation 最佳 checkpoint 在 test 上的表现
```

不是 test 上挑出来的最好结果。

## 16. 快速定位代码

如果你想改实验范围：

```text
改数据集:
  sweep_dscnn_acc.py -> DATASETS
  sweep_dscnn_noise_acc.py -> DATASETS
  sweep_fixed_dscnn_noise_snr_scene_acc.py -> --datasets 或 DEFAULT_DATASETS

改结构列表:
  ARCHS
  或第三个脚本用 --arch_names / --single_arch

改训练 epoch/batch/GPU:
  EPOCH / BATCH / GPU / NUM_WORKERS
  或第三个脚本用 --epoch / --batch / --gpu / --num_workers

改噪声强度:
  TRAIN_SNR_MIN_DB / TRAIN_SNR_MAX_DB
  EVAL_SNR_MIN_DB / EVAL_SNR_MAX_DB
  VALID_SNR_DB
  或第三个脚本用对应命令行参数

改输出文件名:
  前两个脚本改 OUT_CSV / OUT_XLSX / BEST_MODEL_DIR
  第三个脚本用 --out_prefix
```

如果你想改单次训练逻辑：

```text
训练循环:
  dscnn_kws/engine/trainer.py

模型结构:
  dscnn_kws/model/dscnn.py

前端特征:
  dscnn_kws/train.py -> MFCCDSCNN
  dscnn_kws/frontend/

数据增强和混噪:
  dscnn_kws/data/dataset.py

指标计算:
  dscnn_kws/engine/trainer.py -> _run_eval()
  scene/SNR 额外评估看 sweep_fixed_dscnn_noise_snr_scene_acc.py -> eval_acc()
```

## 17. 一句话总结

这三个脚本共同构成了当前项目的实验主线：

```text
sweep_dscnn_acc.py
  建 clean 基线，找到无噪声条件下参数量和准确率的关系。

sweep_dscnn_noise_acc.py
  加在线 TAU 噪声训练，观察噪声增强后的整体性能和可选场景表现。

sweep_fixed_dscnn_noise_snr_scene_acc.py
  在训练后做固定 TAU 场景 x 固定 SNR 网格测试，得到最适合鲁棒性分析和模型选型的最终结果表。
```

