# Noise Training / Validation / Test Guide

本文档说明当前项目中噪声数据的组织方式、在线加噪逻辑、单模型训练命令，以及两个噪声相关 sweep 脚本的最新实验设置。

当前主要噪声实验脚本：

```text
dscnn_kws/sweep_dscnn_noise_acc.py
dscnn_kws/sweep_fixed_dscnn_noise_snr_scene_acc.py
```

它们都基于同一套训练入口和数据加载逻辑：

```text
dscnn_kws/train.py
dscnn_kws/data/dataset.py
```

推荐在项目外层目录运行，也就是可以直接执行 `python -m dscnn_kws.train` 的目录：

```bash
cd /path/to/dscnn_kws
```

## 1. 当前实验目标

当前噪声实验围绕 Mobvoi 二分类关键词识别：

```text
positive: 当前唤醒词
negative: 非当前唤醒词，包括 hard negative
```

默认数据集：

```text
mobvoi_hi_xiaowen_binary_hardneg
mobvoi_nihao_wenwen_binary_hardneg
```

默认数据根目录：

```text
dscnn_kws/data
```

每个数据集目录需要包含：

```text
dscnn_kws/data/<dataset_name>/
  train_manifest.json
  validation_manifest.json
  test_manifest.json
```

manifest 是 JSON Lines 格式，每行一条样本，核心字段：

```json
{"audio_filepath": ".../xxx.wav", "command": "positive"}
{"audio_filepath": ".../yyy.wav", "command": "negative"}
```

## 2. 噪声目录结构

当前默认使用 TAU 噪声，并按 train/validation/test 列表文件划分：

```text
dscnn_kws/noise/
  lists/
    tau_train.txt
    tau_valid.txt
    tau_test.txt
  tau/
    airport/
    bus/
    metro/
    metro_station/
    park/
    public_square/
    shopping_mall/
    street_pedestrian/
    street_traffic/
    tram/
```

列表文件中的每一行是一个 wav 噪声文件路径。空行和以 `#` 开头的行会被忽略。

如果列表里的路径是相对路径，会相对该 txt 文件所在目录解析。例如：

```text
../tau/airport/example.wav
../tau/bus/example.wav
```

## 3. TAU 场景中文说明

| 目录名 | 中文含义 |
| --- | --- |
| `airport` | 机场 |
| `bus` | 公交车 / 巴士 |
| `metro` | 地铁车厢 / 地铁内 |
| `metro_station` | 地铁站 |
| `park` | 公园 |
| `public_square` | 公共广场 |
| `shopping_mall` | 购物中心 / 商场 |
| `street_pedestrian` | 步行街 / 行人街道 |
| `street_traffic` | 车流街道 / 交通道路 |
| `tram` | 有轨电车 |

其中 `metro` 更偏地铁车厢内环境，`metro_station` 是地铁站环境；`street_pedestrian` 偏人行街道，`street_traffic` 偏车辆交通噪声。

### 3.1 TAU train/valid/test 划分方式

当前 `tau_train.txt`、`tau_valid.txt`、`tau_test.txt` 由脚本生成：

```text
dscnn_kws/noise/make_tau_split_lists.py
```

默认命令：

```bash
python dscnn_kws/noise/make_tau_split_lists.py
```

默认输入输出：

```text
输入 TAU 根目录:
  dscnn_kws/noise/tau

输出列表目录:
  dscnn_kws/noise/lists

输出文件:
  tau_train.txt
  tau_valid.txt
  tau_test.txt
```

默认比例：

```text
train : valid : test = 0.8 : 0.1 : 0.1
```

默认随机种子：

```text
seed = 42
```

划分不是把所有 TAU wav 混在一起后整体随机切分，而是按 scene 分层切分。也就是说，每个 TAU 场景内部先单独打乱，再按 80/10/10 切成 train、valid、test，最后把所有场景的 train 合并成 `tau_train.txt`，所有场景的 valid 合并成 `tau_valid.txt`，所有场景的 test 合并成 `tau_test.txt`。

这样做的目的：

```text
train/valid/test 都覆盖 10 个 TAU 场景，避免某个 split 缺少某类场景噪声。
```

场景识别规则在 `infer_scene()` 中：

1. 如果 wav 路径的上级目录名包含场景名，就用目录名判断。
2. 否则从 wav 文件名开头判断，例如 `airport-...wav`、`metro_station-...wav`。
3. 判断时较长的场景名优先，因此 `metro_station` 不会被误归到 `metro`。

每个场景内部的切分数量由 `split_scene_files()` 计算：

```text
n_train = int(n * train_ratio)
n_valid = int(n * valid_ratio)
n_test  = n - n_train - n_valid
```

如果某个场景文件数很少，脚本会做保护：

- `n >= 3` 时，至少保证 train 和 valid 各有 1 个，并保留 test。
- `n == 2` 时，train 1 个，valid 0 个，test 1 个。
- `n == 1` 时，train 1 个，valid/test 为空。

当前仓库中已生成的 TAU 列表实际数量是：

| split | 总 wav 数 | 每个场景 wav 数 |
| --- | ---: | ---: |
| `tau_train.txt` | 18430 | 每个场景 1843 |
| `tau_valid.txt` | 2300 | 每个场景 230 |
| `tau_test.txt` | 2310 | 每个场景 231 |

也就是说，当前 TAU 噪声数据总数为：

```text
18430 + 2300 + 2310 = 23040 wav
```

整体比例约为：

```text
train 79.99%
valid 9.98%
test  10.03%
```

之所以不是精确的 80/10/10，是因为每个 scene 内部要按整数文件数切分。当前每个场景共有 2304 个 wav：

```text
train = int(2304 * 0.8) = 1843
valid = int(2304 * 0.1) = 230
test  = 2304 - 1843 - 230 = 231
```

如果需要重新生成不同划分比例，可以运行：

```bash
python dscnn_kws/noise/make_tau_split_lists.py \
  --tau_root ./dscnn_kws/noise/tau \
  --out_dir ./dscnn_kws/noise/lists \
  --train_ratio 0.8 \
  --valid_ratio 0.1 \
  --seed 42
```

其中 test 比例不需要单独传入，脚本会自动使用：

```text
test_ratio = 1 - train_ratio - valid_ratio
```

## 4. 在线加噪的代码位置

在线加噪不额外生成 noisy wav 文件，而是在 DataLoader 读取样本时动态混噪。

核心代码：

```text
dscnn_kws/data/dataset.py
```

关键函数：

```text
SpeechCommandDataset._load_noise_dataset()
  从 _background_noise_、噪声目录或 txt 列表中收集 wav 噪声文件。

SpeechCommandDataset._load_noise_audio()
  加载噪声音频，必要时重采样，并转成单声道 float32。

SpeechCommandDataset._apply_noise()
  决定当前样本是否加噪，随机选择噪声片段，随机采样 SNR，并混到语音上。

_mix_at_snr()
  按目标 SNR 调整噪声 RMS，然后与语音相加。

build_dataloaders()
  分别为 train、validation、test 创建 Dataset 和 DataLoader。
```

## 5. 在线混噪逻辑

对每条语音样本，`_apply_noise()` 的流程是：

1. 如果 `noise_aug=False`，直接返回原始语音。
2. 如果没有可用噪声文件，直接返回原始语音。
3. 按 `noise_prob` 决定当前样本是否加噪。
4. 随机选择一个噪声 wav。
5. 如果噪声短于 1 秒，就重复拼接到足够长。
6. 从噪声中随机截取 1 秒片段。
7. 在 `[noise_snr_min_db, noise_snr_max_db]` 中随机采样一个 SNR。
8. 调整噪声能量后与语音混合。
9. 如果混合后峰值过大，会缩放到不超过 0.99。
10. 最后 clamp 到 `[-1, 1]`。

SNR 混合公式核心是：

```text
target_noise_rms = speech_rms / (10 ** (snr_db / 20))
scaled_noise = noise * (target_noise_rms / noise_rms)
mixed = speech + scaled_noise
```

SNR 含义：

```text
20 dB   较干净
10 dB   常见噪声环境
5 dB    困难但现实的噪声环境
0 dB    语音和噪声能量接近
-5 dB   强压力测试
```

## 6. 训练/验证/测试加噪参数

以下参数由 `python -m dscnn_kws.train` 支持。

### 训练集是否加噪

```bash
--noise_aug
--no-noise_aug
```

### 验证和测试集是否加噪

```bash
--eval_noise_aug
--no-eval_noise_aug
```

### 噪声来源

```bash
--noise_roots <path1> <path2> ...
```

train/validation/test 共用同一批噪声。

也可以分别指定：

```bash
--train_noise_roots <path1> ...
--valid_noise_roots <path1> ...
--test_noise_roots <path1> ...
```

如果同时传入，分 split 的参数优先级高于 `--noise_roots`。

路径可以是目录，也可以是 txt 列表文件。

### 训练集加噪概率

```bash
--noise_aug_prob 0.8
```

当前噪声 sweep 默认训练加噪概率是 `0.8`。

### 训练集 SNR

```bash
--noise_snr_min_db -5
--noise_snr_max_db 20
```

当前噪声 sweep 默认训练 SNR 范围是 `-5 ~ 20 dB`。

### 验证/测试集加噪概率

```bash
--eval_noise_aug_prob 1.0
```

当前噪声评估默认验证/测试样本全部加噪。

如果不传该参数，`build_dataloaders()` 会沿用 `--noise_aug_prob`。

### 验证/测试集 SNR

随机区间评估：

```bash
--eval_noise_snr_min_db 0
--eval_noise_snr_max_db 20
```

固定 SNR 评估：

```bash
--eval_noise_snr_min_db 5
--eval_noise_snr_max_db 5
```

如果不传这两个参数，会沿用训练集的 `--noise_snr_min_db` 和 `--noise_snr_max_db`。

### 在线重采样

```bash
--allow_online_resample
```

Mobvoi 语音和 TAU 噪声采样率可能不同。当前 sweep 脚本都开启该参数，并同时关闭预检查：

```bash
--allow_online_resample
--no-verify_sample_rate
```

这样可以提高兼容性，但会在读取时增加重采样开销。

## 7. 当前 clean baseline 设置

clean baseline 不加外部噪声，主要脚本是：

```text
dscnn_kws/sweep_dscnn_acc.py
```

单模型等价命令示例：

```bash
python -m dscnn_kws.train \
  --root ./dscnn_kws/data \
  --dataset mobvoi_hi_xiaowen_binary_hardneg \
  --sample_rate 16000 \
  --epoch 30 \
  --batch 128 \
  --gpu 0 \
  --num_workers 0 \
  --dct_coeff 10 \
  --window_size_ms 32 \
  --window_stride_ms 32 \
  --allow_online_resample \
  --no-verify_sample_rate \
  --no-noise_aug \
  --no-eval_noise_aug
```

含义：

```text
train:      不加噪声
validation: 不加噪声
test:       不加噪声
```

输出来自 `train.py` 最终打印：

```text
[TEST] loss=... acc=... precision=... recall=... f1=...
```

## 8. 当前在线噪声增强 sweep

脚本：

```text
dscnn_kws/sweep_dscnn_noise_acc.py
```

运行：

```bash
python dscnn_kws/sweep_dscnn_noise_acc.py
```

### 默认数据集

```text
mobvoi_hi_xiaowen_binary_hardneg
mobvoi_nihao_wenwen_binary_hardneg
```

### 默认噪声来源

```text
train noise:      ./dscnn_kws/noise/lists/tau_train.txt
validation noise: ./dscnn_kws/noise/lists/tau_valid.txt
test noise:       ./dscnn_kws/noise/lists/tau_test.txt
```

注意：当前默认已经不是整个 `./dscnn_kws/noise/tau` 目录，而是 train/valid/test 三个列表文件。

### 默认训练参数

```text
epoch             30
batch             128
sample_rate       16000
gpu               1
num_workers       8
dct_coeff         10
window_size_ms    32
window_stride_ms  32
```

### 默认噪声参数

```text
train:
  noise_aug       True
  noise_prob      0.8
  snr             -5 ~ 20 dB 随机
  deterministic   False

validation/test:
  eval_noise_aug  True
  noise_prob      1.0
  snr             0 ~ 20 dB 随机
  deterministic   True
```

验证集随机种子在 `build_dataloaders()` 中是：

```text
seed + 100000
```

测试集随机种子是：

```text
seed + 200000
```

### 输出文件

```text
sweep_dscnn_noise_acc_results.csv
sweep_dscnn_noise_acc_results.xlsx
sweep_dscnn_noise_shapes.txt
dscnn_kws/runs/sweep_noise_best_models/
```

主 CSV 字段包括：

```text
dataset
arch
layers
channels
mfcc_output
train_noise_roots
valid_noise_roots
test_noise_roots
train_noise_prob
train_snr_min_db
train_snr_max_db
eval_noise_prob
eval_snr_min_db
eval_snr_max_db
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

### 覆盖噪声路径

所有 split 使用同一噪声源：

```bash
python dscnn_kws/sweep_dscnn_noise_acc.py \
  --noise_roots ./dscnn_kws/noise/tau/airport
```

分别覆盖 train/valid/test：

```bash
python dscnn_kws/sweep_dscnn_noise_acc.py \
  --train_noise_roots ./dscnn_kws/noise/lists/tau_train.txt \
  --valid_noise_roots ./dscnn_kws/noise/lists/tau_valid.txt \
  --test_noise_roots ./dscnn_kws/noise/lists/tau_test.txt
```

也支持环境变量：

```bash
TAU_NOISE_ROOTS=./dscnn_kws/noise/tau/airport python dscnn_kws/sweep_dscnn_noise_acc.py
```

Linux 下多个路径使用冒号分隔，Windows 下使用分号分隔，因为代码使用 `os.pathsep`：

```bash
TAU_NOISE_ROOTS=./dscnn_kws/noise/tau/airport:./dscnn_kws/noise/tau/bus python dscnn_kws/sweep_dscnn_noise_acc.py
```

### 可选 per-scene 测试

运行：

```bash
python dscnn_kws/sweep_dscnn_noise_acc.py --per_scene_test
```

每个训练完成后的 best checkpoint 会额外在 10 个 TAU 场景上测试。

per-scene 测试设置：

```text
split            test_manifest.json
noise_prob       1.0
snr              0 ~ 20 dB 随机
seed             300000 + scene_idx * 10007
scene_test_root  ./dscnn_kws/noise/tau
```

输出：

```text
sweep_dscnn_noise_scene_results.csv
sweep_dscnn_noise_scene_results.xlsx
```

注意：这里的 SNR 是 `0 ~ 20 dB` 随机区间，不是固定 SNR 网格。严谨的固定 SNR 分析请使用下一节的脚本。

## 9. 当前固定 SNR / TAU 场景网格 sweep

脚本：

```text
dscnn_kws/sweep_fixed_dscnn_noise_snr_scene_acc.py
```

运行：

```bash
python dscnn_kws/sweep_fixed_dscnn_noise_snr_scene_acc.py
```

这个脚本会先训练模型，再用训练得到的 best checkpoint 做固定 `scene x SNR` 网格测试。

### 默认数据集

```text
mobvoi_hi_xiaowen_binary_hardneg
mobvoi_nihao_wenwen_binary_hardneg
```

### 默认训练参数

```text
epoch             30
batch             256
sample_rate       16000
gpu               1
num_workers       8
dct_coeff         10
window_size_ms    32
window_stride_ms  32
```

### 默认训练噪声

```text
train_noise_roots     ./dscnn_kws/noise/lists/tau_train.txt
valid_noise_roots     ./dscnn_kws/noise/lists/tau_valid.txt
test_noise_roots      ./dscnn_kws/noise/lists/tau_test.txt
train_noise_prob      0.8
train_snr             -5 ~ 20 dB 随机
valid_noise_prob      1.0
valid_snr_db          5.0
```

训练命令中 validation 和训练结束时的 test 都使用固定 5 dB：

```text
--eval_noise_snr_min_db 5.0
--eval_noise_snr_max_db 5.0
```

因此 train.py 打印的：

```text
[TEST] loss=... acc=... precision=... recall=... f1=...
```

对应的是：

```text
test_manifest.json + tau_test.txt + 固定 5 dB 噪声
```

脚本把其中的 `acc` 和 `f1` 保存为：

```text
test_acc_on_tau_test_list
f1_on_tau_test_list
```

### 默认 TAU scene/SNR 网格

场景：

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

SNR：

```text
20.0, 10.0, 5.0, 0.0, -5.0
```

默认每个模型会额外做：

```text
10 scenes x 5 SNR = 50 个测试点
```

每个测试点使用：

```text
split            test_manifest.json
noise_prob       1.0
noise_roots      ./dscnn_kws/noise/tau/<scene>
snr              当前 snr_db 固定值
seed             500000 + scene_idx * 10007 + snr_idx * 101
```

### 输出文件

默认输出：

```text
snr_scene_arch_sweep_train_results.csv
snr_scene_arch_sweep_grid_results.csv
snr_scene_arch_sweep_scene_summary.csv
snr_scene_arch_sweep_arch_summary.csv
dscnn_kws/runs/snr_scene_arch_sweep_best_models/
```

可以通过 `--out_prefix` 修改输出前缀：

```bash
python dscnn_kws/sweep_fixed_dscnn_noise_snr_scene_acc.py \
  --out_prefix my_noise_exp
```

输出会变成：

```text
my_noise_exp_train_results.csv
my_noise_exp_grid_results.csv
my_noise_exp_scene_summary.csv
my_noise_exp_arch_summary.csv
dscnn_kws/runs/my_noise_exp_best_models/
```

### 只跑部分结构

跑内置结构中的几个：

```bash
python dscnn_kws/sweep_fixed_dscnn_noise_snr_scene_acc.py \
  --arch_names L5_C64 L3_C16
```

只跑一个结构：

```bash
python dscnn_kws/sweep_fixed_dscnn_noise_snr_scene_acc.py \
  --single_arch \
  --arch_name L3_C16 \
  --num_layers 3 \
  --channels 16
```

### 跳过训练，只评估已有 checkpoint

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

限制：

```text
--skip_train 当前只支持一个 dataset 和一个 architecture。
```

## 10. 单模型训练命令示例

下面命令等价于“带 TAU 列表噪声训练 + 验证/测试固定 5 dB”的单模型版本：

```bash
python -m dscnn_kws.train \
  --root ./dscnn_kws/data \
  --dataset mobvoi_hi_xiaowen_binary_hardneg \
  --sample_rate 16000 \
  --epoch 30 \
  --batch 256 \
  --gpu 1 \
  --num_workers 8 \
  --dct_coeff 10 \
  --window_size_ms 32 \
  --window_stride_ms 32 \
  --allow_online_resample \
  --no-verify_sample_rate \
  --noise_aug \
  --eval_noise_aug \
  --train_noise_roots ./dscnn_kws/noise/lists/tau_train.txt \
  --valid_noise_roots ./dscnn_kws/noise/lists/tau_valid.txt \
  --test_noise_roots ./dscnn_kws/noise/lists/tau_test.txt \
  --noise_aug_prob 0.8 \
  --noise_snr_min_db -5 \
  --noise_snr_max_db 20 \
  --eval_noise_aug_prob 1.0 \
  --eval_noise_snr_min_db 5 \
  --eval_noise_snr_max_db 5 \
  --model_size_info 5 64 10 4 2 2 64 3 3 1 1 64 3 3 1 1 64 3 3 1 1 64 3 3 1 1
```

这里的 `model_size_info` 是 L5_C64：

```text
5
64 10 4 2 2
64 3 3 1 1
64 3 3 1 1
64 3 3 1 1
64 3 3 1 1
```

## 11. FAH/FRR 固定 SNR 评估

如果要评估 FAH/FRR，可使用：

```text
dscnn_kws/eval_fah_frr.py
```

示例：已有 checkpoint，在 TAU airport 的 5 dB 噪声下评估：

```bash
python -m dscnn_kws.eval_fah_frr \
  --root ./dscnn_kws/data \
  --dataset mobvoi_hi_xiaowen_binary_hardneg \
  --ckpt ./dscnn_kws/runs/<run_name>/best.pt \
  --sample_rate 16000 \
  --batch 256 \
  --gpu 1 \
  --num_workers 0 \
  --eval_noise_aug \
  --valid_noise_roots ./dscnn_kws/noise/tau/airport \
  --test_noise_roots ./dscnn_kws/noise/tau/airport \
  --noise_aug_prob 1.0 \
  --noise_snr_min_db 5 \
  --noise_snr_max_db 5 \
  --dct_coeff 10 \
  --window_size_ms 32 \
  --window_stride_ms 32 \
  --target_fah 0.5 1.0 2.0
```

注意：`eval_fah_frr.py` 当前使用 `--noise_aug_prob`、`--noise_snr_min_db`、`--noise_snr_max_db` 控制 validation/test 噪声，不使用 `--eval_noise_snr_min_db` 这一组参数。

## 12. Cross-domain 测试

如果要观察跨噪声域泛化，可以用 TAU 训练，然后用 DEMAND 或 MUSAN 测试。

TAU 训练示例：

```bash
python -m dscnn_kws.train \
  --root ./dscnn_kws/data \
  --dataset mobvoi_hi_xiaowen_binary_hardneg \
  --sample_rate 16000 \
  --epoch 30 \
  --batch 256 \
  --gpu 1 \
  --num_workers 8 \
  --allow_online_resample \
  --no-verify_sample_rate \
  --noise_aug \
  --eval_noise_aug \
  --train_noise_roots ./dscnn_kws/noise/lists/tau_train.txt \
  --valid_noise_roots ./dscnn_kws/noise/lists/tau_valid.txt \
  --test_noise_roots ./dscnn_kws/noise/lists/tau_test.txt \
  --noise_aug_prob 0.8 \
  --noise_snr_min_db -5 \
  --noise_snr_max_db 20 \
  --eval_noise_aug_prob 1.0 \
  --eval_noise_snr_min_db 5 \
  --eval_noise_snr_max_db 5
```

用 DEMAND 测试时，把 `eval_fah_frr.py` 的测试噪声换成：

```bash
--test_noise_roots ./dscnn_kws/noise/demand
```

用 MUSAN noise 测试：

```bash
--test_noise_roots ./dscnn_kws/noise/musan/noise
```

用 MUSAN speech/music 做更强干扰：

```bash
--test_noise_roots ./dscnn_kws/noise/musan/speech ./dscnn_kws/noise/musan/music
```

## 13. 如何阅读噪声实验结果

### `sweep_dscnn_noise_acc_results.csv`

这个文件回答：

```text
随机 SNR 噪声增强训练后，每个结构在带噪 validation/test 设置下表现如何？
```

重点字段：

```text
expected_params
printed_params
best_valid_acc
test_acc
precision
recall
f1
```

### `sweep_dscnn_noise_scene_results.csv`

只有 `--per_scene_test` 时生成。

这个文件回答：

```text
每个 TAU 场景在 0~20 dB 随机 SNR 下的测试表现如何？
```

### `snr_scene_arch_sweep_grid_results.csv`

这个文件最细，每一行是：

```text
一个 dataset + 一个 arch + 一个 scene + 一个 snr_db
```

适合画 SNR 曲线、找性能断崖点。

### `snr_scene_arch_sweep_scene_summary.csv`

按：

```text
dataset + arch + scene
```

聚合多个 SNR，重点看：

```text
mean_acc
min_acc
mean_f1
min_f1
```

### `snr_scene_arch_sweep_arch_summary.csv`

按：

```text
dataset + arch
```

聚合所有 scene 和 SNR，适合最终选型。

重点看：

```text
expected_params
num_eval_points
mean_acc
min_acc
mean_f1
min_f1
```

`mean_*` 代表平均鲁棒性，`min_*` 代表最坏工况表现。部署选型时不要只看平均值。

## 14. 当前推荐实验顺序

1. 先跑 clean baseline：

```bash
python dscnn_kws/sweep_dscnn_acc.py
```

2. 再跑在线噪声增强 sweep：

```bash
python dscnn_kws/sweep_dscnn_noise_acc.py
```

3. 最后跑固定 SNR / TAU scene 网格：

```bash
python dscnn_kws/sweep_fixed_dscnn_noise_snr_scene_acc.py
```

如果只想快速验证一个模型的鲁棒性，用第三个脚本的 `--skip_train` 直接评估已有 checkpoint。

## 15. 常见问题

### 指定了噪声但没有实际加噪

检查：

```text
noise_aug 或 eval_noise_aug 是否打开
noise_roots 路径是否存在
txt 列表里的 wav 路径是否正确
wav 文件大小是否大于 44 字节
```

两个 sweep 脚本都会在启动时调用 `count_usable_noise_files()`，如果可用 wav 数为 0 会直接报错。

### clean 和 noise 结果能否直接比较

可以看同一结构在 clean 与 noise 条件下的下降幅度，但不要把 clean accuracy 当作噪声部署性能。

更合理的比较方式：

```text
同一测试条件下比较不同结构
同一结构比较 clean / random noise / fixed scene-SNR
最终部署选型看 fixed scene-SNR 的 mean/min 指标
```

### `sweep_dscnn_noise_acc.py --per_scene_test` 和固定网格有什么区别

`--per_scene_test`：

```text
每个 scene 使用 0~20 dB 随机 SNR。
```

`sweep_fixed_dscnn_noise_snr_scene_acc.py`：

```text
每个 scene 明确测试 20, 10, 5, 0, -5 dB。
```

固定网格更适合做严谨分析。

### GPU 参数怎么理解

项目里：

```text
--gpu 0  表示不用 GPU
--gpu 1  表示使用 1 张 GPU
```

启动日志中如果看到：

```text
[INFO] device=cuda
```

说明正在使用 GPU。

## 16. 实验记录模板

建议记录：

```text
model   dataset       train_noise       valid_noise       test_noise       train_snr  eval_snr  scene_grid        acc/f1
L5_C64  hi_xiaowen    tau_train.txt     tau_valid.txt     tau_test.txt     -5~20      5         no                ...
L3_C16  hi_xiaowen    tau_train.txt     tau_valid.txt     tau_test.txt     -5~20      5         yes 20/10/5/0/-5 ...
L5_C64  nihao_wenwen  tau_train.txt     tau_valid.txt     tau_test.txt     -5~20      5         yes 20/10/5/0/-5 ...
```

最终分析建议回答：

```text
1. clean baseline 的准确率是多少？
2. 随机噪声测试下准确率和 f1 下降多少？
3. 固定 5 dB tau_test 上模型是否稳定？
4. scene/SNR 网格中最难的场景是什么？
5. -5 dB 或 0 dB 时是否出现性能断崖？
6. 在满足目标 mean/min acc 或 f1 的前提下，最小模型是哪一个？
```
