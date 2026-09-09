# Mobvoi Hotword Dataset Processing Guide

本文档说明本项目中 Mobvoi hotword 数据集的组织方式、各个数据准备脚本的处理逻辑，以及如何把生成的 manifest 用于 DSCNN 训练、ACC/F1 评估和 FAH/FRR 评估。

推荐在项目外层目录运行命令，也就是可以直接执行 `python -m dscnn_kws.train` 的目录：

```bash
cd /path/to/dscnn_kws
```

## 1. 数据集在本项目中的任务形式

当前仓库把 Mobvoi hotword 处理成二分类关键词检测任务：

```text
positive: 当前目标唤醒词
negative: 非当前目标唤醒词，包括普通 negative，也可以包括另一个唤醒词作为 hard negative
```

当前 `dscnn_kws/configs.py` 中的类别也是二分类：

```python
CLASS_LIST = [
    "positive",
    "negative",
]
```

Mobvoi 数据集中本项目使用的两个唤醒词 id 为：

| `keyword_id` | 数据集目录名中的名称 | 含义 |
| ---: | --- | --- |
| `0` | `hi_xiaowen` | Hi Xiaowen |
| `1` | `nihao_wenwen` | Nihao Wenwen |

也就是说，训练 `mobvoi_hi_xiaowen_*` 时，`hi_xiaowen` 是 positive；训练 `mobvoi_nihao_wenwen_*` 时，`nihao_wenwen` 是 positive。

## 2. 原始数据目录约定

Mobvoi 原始数据默认放在 `dscnn_kws/data/mobvoi_hotwords_raw/` 下，几个脚本都按下面的路径查找：

```text
dscnn_kws/data/
  mobvoi_hotwords_raw/
    mobvoi_hotword_dataset_resources/
      p_train.json
      p_dev.json
      p_test.json
      n_train.json
      n_dev.json
      n_test.json
    mobvoi_hotword_dataset/
      mobvoi_hotword_dataset/
        .../*.wav
```

其中：

- `p_*.json` 是 positive metadata，里面通过 `keyword_id` 区分两个唤醒词。
- `n_*.json` 是 non-hotword negative metadata，通常 `keyword_id` 为 `-1`。
- `utt_id` 用于和 wav 文件名匹配。准备脚本会用 `Path(utt_id).stem` 查找同名 `.wav`。
- `train/dev/test` 会映射成本项目统一使用的 `train/validation/test`。

如果想先查看 ModelScope 上 Mobvoi 数据集对象的结构，可以运行：

```bash
python dscnn_kws/data/inspect_mobvoi.py
```

这个脚本只负责打印数据集结构，不会自动生成本项目训练用的 manifest。

## 3. Manifest 格式

本项目训练和评估只需要每个数据集目录里有三个 JSON Lines 文件：

```text
<dataset_dir>/
  train_manifest.json
  validation_manifest.json
  test_manifest.json
```

每一行是一条样本记录：

```json
{"audio_filepath": "../mobvoi_hotwords_raw/mobvoi_hotword_dataset/mobvoi_hotword_dataset/xxx.wav", "command": "positive"}
{"audio_filepath": "../mobvoi_hotwords_raw/mobvoi_hotword_dataset/mobvoi_hotword_dataset/yyy.wav", "command": "negative"}
```

字段说明：

- `audio_filepath`: wav 路径。准备脚本写的是相对于输出数据集目录的路径，避免复制整个原始音频目录。
- `command`: 类别名，当前 Mobvoi 二分类任务只应为 `positive` 或 `negative`。

`dscnn_kws/data/dataset.py` 读取 manifest 后会解析音频路径。它会尝试相对于数据集父目录、仓库根目录、数据集目录和 manifest 所在目录查找 wav，所以只要 manifest 中的相对路径和实际目录关系没有被破坏，就不需要复制原始 wav。

## 4. Mobvoi 准备脚本说明

### 4.1 `prepare_mobvoi_manifest.py`

这是最早的粗粒度二分类版本，输出目录为：

```text
dscnn_kws/data/mobvoi_hotwords_binary/
```

处理逻辑：

- 同时使用 `p_train/dev/test.json` 中的所有 positive，不区分 `keyword_id`。
- 使用 `n_train/dev/test.json` 中的 non-hotword 样本作为 negative。
- `NEG_TO_POS_RATIO = 1.0` 时，每个 split 都把 negative 下采样到与 positive 数量相同。
- 输出 `train_manifest.json`、`validation_manifest.json`、`test_manifest.json`。

适用场景：

- 只想做一个“任意 Mobvoi 唤醒词 vs 非唤醒词”的粗略二分类实验。
- 不推荐作为当前主要实验入口，因为它没有区分两个唤醒词。

运行：

```bash
python dscnn_kws/data/prepare_mobvoi_manifest.py
```

### 4.2 `prepare_mobvoi_per_keyword_manifest.py`

这是单关键词二分类版本，输出两个数据集：

```text
dscnn_kws/data/mobvoi_hi_xiaowen_binary/
dscnn_kws/data/mobvoi_nihao_wenwen_binary/
```

处理逻辑：

- 对每个 `keyword_id` 单独构建一个数据集。
- 当前关键词的 positive 作为 `positive`。
- `n_train/dev/test.json` 中的 non-hotword 作为 `negative`。
- 另一个唤醒词默认不会被当成 negative。
- `NEG_TO_POS_RATIO = 1.0` 时，train/validation/test 都做正负 1:1 平衡。

适用场景：

- 做单关键词 ACC/F1 实验。
- 负样本只希望使用原始 non-hotword，不希望另一个唤醒词干扰当前任务。

运行：

```bash
python dscnn_kws/data/prepare_mobvoi_per_keyword_manifest.py
```

### 4.3 `prepare_mobvoi_per_keyword_manifest_fah.py`

这是单关键词 FAH/FRR 版本，输出两个数据集：

```text
dscnn_kws/data/mobvoi_hi_xiaowen_binary_fah/
dscnn_kws/data/mobvoi_nihao_wenwen_binary_fah/
```

处理逻辑：

- 当前关键词的 positive 作为 `positive`。
- `n_train/dev/test.json` 中的 non-hotword 作为 `negative`。
- 训练集按 `TRAIN_NEG_TO_POS_RATIO = 1.0` 做正负平衡。
- validation/test 默认保留全部 negative，不再强行平衡，用于更接近 FAH/FRR 的评估。
- `INCLUDE_OTHER_KEYWORD_AS_NEGATIVE = False` 时，另一个唤醒词不会进入 negative；如果改成 `True`，另一个唤醒词也会作为 hard negative。

适用场景：

- 已有单关键词模型，需要在较多 negative 上估计误触发情况。
- 需要用 `dscnn_kws/eval_fah_frr.py` 评估 FAH/FRR。

运行：

```bash
python dscnn_kws/data/prepare_mobvoi_per_keyword_manifest_fah.py
```

### 4.4 `prepare_mobvoi_hardneg_manifests.py`

这是当前更推荐使用的 Mobvoi manifest 生成脚本。它一次生成 ACC/F1 和 FAH/FRR 两套 hard-negative 数据集：

```text
dscnn_kws/data/mobvoi_hi_xiaowen_binary_hardneg/
dscnn_kws/data/mobvoi_nihao_wenwen_binary_hardneg/
dscnn_kws/data/mobvoi_hi_xiaowen_binary_fah_hardneg/
dscnn_kws/data/mobvoi_nihao_wenwen_binary_fah_hardneg/
```

处理逻辑：

- 当前关键词的 positive 作为 `positive`。
- negative 由两部分组成：
  - `n_train/dev/test.json` 中的 non-hotword。
  - 另一个唤醒词的 positive，也就是 hard negative。
- 对 `_binary_hardneg` 数据集：
  - train/validation/test 都做正负 1:1 平衡。
  - 适合看 ACC/F1。
- 对 `_binary_fah_hardneg` 数据集：
  - train 做正负 1:1 平衡。
  - validation/test 保留全部 negative。
  - 适合做 FAH/FRR。
- `TRAIN_NEG_TO_POS_RATIO = 1.0` 控制需要平衡时的 negative 数量。
- 脚本固定 `random.seed(42)`，因此 negative 下采样是可复现的。

推荐运行：

```bash
python dscnn_kws/data/prepare_mobvoi_hardneg_manifests.py
```

## 5. 生成数据集后的目录

运行推荐脚本后，典型目录如下：

```text
dscnn_kws/data/
  mobvoi_hi_xiaowen_binary_hardneg/
    train_manifest.json
    validation_manifest.json
    test_manifest.json
  mobvoi_nihao_wenwen_binary_hardneg/
    train_manifest.json
    validation_manifest.json
    test_manifest.json
  mobvoi_hi_xiaowen_binary_fah_hardneg/
    train_manifest.json
    validation_manifest.json
    test_manifest.json
  mobvoi_nihao_wenwen_binary_fah_hardneg/
    train_manifest.json
    validation_manifest.json
    test_manifest.json
```

可以用下面命令快速检查 manifest 是否生成：

```bash
ls dscnn_kws/data/mobvoi_hi_xiaowen_binary_hardneg
head -n 3 dscnn_kws/data/mobvoi_hi_xiaowen_binary_hardneg/train_manifest.json
```

Windows PowerShell 下可以用：

```powershell
Get-ChildItem dscnn_kws\data\mobvoi_hi_xiaowen_binary_hardneg
Get-Content dscnn_kws\data\mobvoi_hi_xiaowen_binary_hardneg\train_manifest.json -TotalCount 3
```

## 6. `dataset.py` 如何读取 Mobvoi

训练入口 `dscnn_kws/train.py` 会调用：

```python
build_dataloaders(data_path, CLASS_LIST, CLASS_ENCODING, args)
```

其中 `data_path` 由下面两个参数拼出来：

```text
data_path = <root>/<dataset>
```

例如：

```bash
--root ./dscnn_kws/data \
--dataset mobvoi_hi_xiaowen_binary_hardneg
```

`SpeechCommandDataset` 的主要处理流程：

1. 读取 `train_manifest.json`、`validation_manifest.json` 或 `test_manifest.json`。
2. 把 `command` 映射到 `CLASS_ENCODING`，当前为 `positive -> 0`、`negative -> 1`。
3. 用 `torchaudio.load()` 加载 wav。
4. 如果是多通道音频，转成单通道平均值。
5. 默认严格检查采样率是否等于 `--sample_rate`。
6. 每条样本被处理成 1 秒长度：
   - 不足 1 秒则右侧补 0。
   - 训练时先在两侧补 10% 长度，再随机裁剪 1 秒。
   - 验证和测试时居中裁剪 1 秒。
7. 返回 waveform 和 label，后续由 `train.py` 中的 MFCC 或 bandpass frontend 提取特征。

采样率相关参数：

- `--sample_rate`: 训练和特征提取使用的采样率。
- `--verify_sample_rate`: 训练开始前抽样检查 manifest 中 wav 的采样率，默认开启。
- `--strict_sample_rate`: 加载音频时严格要求采样率匹配，默认开启。
- `--allow_online_resample`: 如果原始 wav 与 `--sample_rate` 不一致，允许在线重采样。

如果不确定原始 Mobvoi wav 的采样率，建议先保持 `--verify_sample_rate` 开启。报错中会直接显示实际采样率和期望采样率。

## 7. 训练 clean Mobvoi baseline

因为 `train.py` 目前默认 `--noise_aug` 为开启，如果只想训练干净 Mobvoi baseline，请显式关闭训练和评估噪声增强：

```bash
python -m dscnn_kws.train \
  --root ./dscnn_kws/data \
  --dataset mobvoi_hi_xiaowen_binary_hardneg \
  --sample_rate 16000 \
  --batch 256 \
  --epoch 50 \
  --gpu 1 \
  --no-noise_aug \
  --no-eval_noise_aug
```

另一个唤醒词：

```bash
python -m dscnn_kws.train \
  --root ./dscnn_kws/data \
  --dataset mobvoi_nihao_wenwen_binary_hardneg \
  --sample_rate 16000 \
  --batch 256 \
  --epoch 50 \
  --gpu 1 \
  --no-noise_aug \
  --no-eval_noise_aug
```

如果你的 Mobvoi wav 是 8 kHz，请把 `--sample_rate 16000` 改成 `--sample_rate 8000`。如果想保留某个统一实验采样率但原始 wav 不一致，可以加：

```bash
--allow_online_resample
```

## 8. 使用在线噪声增强训练 Mobvoi

本项目不需要提前生成带噪 Mobvoi wav。`dataset.py` 支持在线随机混噪：

- `--noise_aug`: 训练集启用噪声增强。
- `--eval_noise_aug`: validation/test 也启用噪声增强。
- `--noise_roots`: train/validation/test 共用的噪声目录或 `.txt` 列表。
- `--train_noise_roots`、`--valid_noise_roots`、`--test_noise_roots`: 按 split 指定噪声来源。
- `--noise_aug_prob`: 每条样本被混噪的概率。
- `--noise_snr_min_db`、`--noise_snr_max_db`: 训练混噪 SNR 范围。
- `--eval_noise_aug_prob`、`--eval_noise_snr_min_db`、`--eval_noise_snr_max_db`: 单独控制验证和测试混噪。

噪声来源可以是目录，也可以是 `.txt` 列表。`.txt` 中每行一个 wav 路径，空行和 `#` 开头的注释会被忽略；相对路径按 `.txt` 文件所在目录解析。扫描时会跳过重复路径、非 wav 文件和过小的无效 wav。

示例：使用 TAU train/valid/test 列表在线增强训练：

```bash
python -m dscnn_kws.train \
  --root ./dscnn_kws/data \
  --dataset mobvoi_hi_xiaowen_binary_hardneg \
  --sample_rate 16000 \
  --batch 256 \
  --epoch 50 \
  --gpu 1 \
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

训练集混噪是随机的；validation/test 使用 `deterministic_noise=True`，同一 index 在同一 seed 下会得到可复现的噪声选择、裁剪位置和 SNR。

## 9. FAH/FRR 评估

FAH/FRR 评估建议使用 `_binary_fah_hardneg` 数据集，因为它的 validation/test 保留了全部 negative，更适合估计误触发。

先用对应的 hardneg 训练集训练模型，或者直接用已有 checkpoint。然后运行：

```bash
python dscnn_kws/eval_fah_frr.py \
  --root ./dscnn_kws/data \
  --dataset mobvoi_hi_xiaowen_binary_fah_hardneg \
  --ckpt ./dscnn_kws/runs/<run_name>/best.pt \
  --sample_rate 16000 \
  --batch 128 \
  --gpu 0 \
  --target_fah 0.1 0.5 1.0
```

带固定 SNR 噪声的 FAH/FRR 评估：

```bash
python dscnn_kws/eval_fah_frr.py \
  --root ./dscnn_kws/data \
  --dataset mobvoi_hi_xiaowen_binary_fah_hardneg \
  --ckpt ./dscnn_kws/runs/<run_name>/best.pt \
  --sample_rate 16000 \
  --batch 128 \
  --gpu 0 \
  --eval_noise_aug \
  --valid_noise_roots ./dscnn_kws/noise/lists/tau_valid.txt \
  --test_noise_roots ./dscnn_kws/noise/lists/tau_test.txt \
  --noise_aug_prob 1.0 \
  --noise_snr_min_db 5 \
  --noise_snr_max_db 5 \
  --target_fah 0.1 0.5 1.0
```

## 10. 连续两帧确认训练 manifest

`--pair_objective` 只接受独立的 `--pair_train_manifest` JSONL。每行必须显式指向同一音频源中的一段连续区间，不能把旧单窗 manifest 或 shuffled batch 中的两条记录拼成 pair。

16 kHz、1 秒窗口、96 ms hop 的正例行示例：

```json
{"format":"kws_confirmation_pair_v1","audio_filepath":"audio/session_7.wav","command":"positive","role":"captured_positive","source_split":"train","source_id":"mic243/session_7","sample_rate":16000,"span_start_sample":32000,"span_num_samples":17536,"window_samples":16000,"hop_samples":1536,"active_start_sample":35000,"active_end_sample":43800}
```

负例使用同一 schema，但必须省略 `active_start_sample` 和 `active_end_sample`：

```json
{"format":"kws_confirmation_pair_v1","audio_filepath":"audio/bus_1.wav","command":"negative","role":"tau_noise_negative","source_split":"train","source_id":"tau/bus_1","sample_rate":16000,"span_start_sample":96000,"span_num_samples":17536,"window_samples":16000,"hop_samples":1536}
```

字段约束：

- `span_start_sample`、`active_start_sample` 和 `active_end_sample` 都是源文件内的绝对 sample offset。
- `span_num_samples` 必须恰好等于 `window_samples + hop_samples`。loader 从 `span_start_sample` 和 `span_start_sample + hop_samples` 切出两帧，并返回这两个绝对 offset。
- `source_split` 记录源数据的固定划分；`source_id` 必须在该划分内稳定标识同一条原始录音，供 source-balanced CVaR 使用，不能用随机 record ID 代替；`role` 保留 positive、captured、false-wake 或 TAU 等训练来源类别。
- 正例 active span 必须完整位于两帧公共重叠区 `[span_start_sample + hop_samples, span_start_sample + window_samples)`，确保两帧都包含完整唤醒词。
- 多声道音频固定使用 channel 0。pair loader 禁止在线重采样、在线噪声增强和独立窗口 jitter；增强后的连续 span 应离线生成并写入 manifest。
- `--pair_negative_cvar_fraction` 在每个 source 内选择最坏位置；`--pair_negative_source_cvar_fraction` 再在每个 domain 内选择最坏 source。后者默认 `1.0`，只有显式调低时才会把梯度集中到少数最高风险 source。
- 当 pair manifest 严重偏向负例时，应将 `--pair_frame_ce_weight` 设为 `0`，依靠平衡的基础 mixture CE 保持分类能力，避免辅助 CE 无差别下压所有 wake 分数。

训练入口示例：

```bash
python -m dscnn_kws.train \
  --pair_objective \
  --pair_train_manifest /data/kws/pairs/train.jsonl \
  --pair_hop_ms 96 \
  --sample_rate 16000 \
  --offline_augmented_dataset \
  --no-noise_aug
```

validation/test 仍使用原单窗 manifest，以便旧指标与历史 checkpoint 保持可比。

当前 DDP 下的 CVaR 是 rank-local surrogate；构造 manifest 和 batch 时应让每个 noise source 在单个 batch 内提供多个连续 pair，并避免把同一 source 的记录切散到过多 rank。部署 gate 仍须使用独立长流音频计算全局 FAH，不能用训练 CVaR 代替。

## 11. Sweep 脚本和 Mobvoi 数据集

干净 ACC/F1 架构搜索脚本 `dscnn_kws/sweep_dscnn_acc.py` 使用 hardneg 数据集，并且会显式传入：

```text
--no-noise_aug --no-eval_noise_aug
```

这样可以保证 clean baseline 不会因为 `train.py` 默认开启 `noise_aug` 而被悄悄改变。

噪声增强架构搜索脚本 `dscnn_kws/sweep_dscnn_noise_acc.py` 默认数据集为：

```text
mobvoi_hi_xiaowen_binary_hardneg
mobvoi_nihao_wenwen_binary_hardneg
```

默认噪声列表为：

```text
./dscnn_kws/noise/lists/tau_train.txt
./dscnn_kws/noise/lists/tau_valid.txt
./dscnn_kws/noise/lists/tau_test.txt
```

因此，运行 sweep 前通常需要先确保：

```bash
python dscnn_kws/data/prepare_mobvoi_hardneg_manifests.py
```

已经成功生成推荐的 hardneg 数据集。

## 12. 常见问题

### 12.1 找不到 wav

报错通常类似：

```text
Failed to resolve audio path from manifest
```

检查：

- `mobvoi_hotwords_raw/` 是否仍在 `dscnn_kws/data/` 下。
- manifest 中的 `audio_filepath` 是否仍然能相对于数据集目录找到原始 wav。
- 是否移动了生成后的 manifest 目录，但没有同时保持它与原始 wav 目录的相对关系。

### 12.2 采样率不匹配

报错通常类似：

```text
Sample-rate mismatch: ..., got 16000, expected 8000
```

处理方式：

- 如果想按原始采样率训练，把命令中的 `--sample_rate` 改成实际采样率。
- 如果想统一到某个采样率，加 `--allow_online_resample`。
- 如果只是临时跳过启动前抽样检查，可以加 `--no-verify_sample_rate`，但加载时仍可能因 `--strict_sample_rate` 报错。

### 12.3 clean baseline 结果突然变了

`train.py` 默认 `--noise_aug=True`。干净实验应显式加：

```bash
--no-noise_aug --no-eval_noise_aug
```

### 12.4 ACC/F1 数据集和 FAH/FRR 数据集怎么选

简单规则：

| 目的 | 推荐数据集 |
| --- | --- |
| 训练和普通 ACC/F1 测试 | `mobvoi_<keyword>_binary_hardneg` |
| FAH/FRR 评估 | `mobvoi_<keyword>_binary_fah_hardneg` |
| 不使用另一个唤醒词作为 hard negative | `mobvoi_<keyword>_binary` 或 `mobvoi_<keyword>_binary_fah` |
| 两个唤醒词合并成一个 positive 类 | `mobvoi_hotwords_binary` |

## 13. 推荐最小流程

从原始 Mobvoi 数据到一次 clean baseline 的最小流程：

```bash
cd /path/to/dscnn_kws

python dscnn_kws/data/prepare_mobvoi_hardneg_manifests.py

python -m dscnn_kws.train \
  --root ./dscnn_kws/data \
  --dataset mobvoi_hi_xiaowen_binary_hardneg \
  --sample_rate 16000 \
  --batch 256 \
  --epoch 50 \
  --gpu 1 \
  --no-noise_aug \
  --no-eval_noise_aug
```

训练完成后，用 FAH/FRR 数据集评估：

```bash
python dscnn_kws/eval_fah_frr.py \
  --root ./dscnn_kws/data \
  --dataset mobvoi_hi_xiaowen_binary_fah_hardneg \
  --ckpt ./dscnn_kws/runs/<run_name>/best.pt \
  --sample_rate 16000 \
  --batch 128 \
  --gpu 0 \
  --target_fah 0.1 0.5 1.0
```
