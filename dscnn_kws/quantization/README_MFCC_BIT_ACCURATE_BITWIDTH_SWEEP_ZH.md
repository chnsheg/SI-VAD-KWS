# Bit-Accurate MFCC 前端 Bit-Width Sweep 运行指南

本文档说明如何基于当前 `bit_accurate` MFCC 前端链路，逐步进行 bit-width sweep，寻找各个 stage 算子的最小安全位宽。

本轮实验的目标不是只跑出一个准确率结果，而是得到一套有证据链的最终策略：

```text
accuracy_first_recommended.json  // 准确率优先的主推荐位宽策略
aggressive_candidate.json        // 更激进压缩的对照位宽策略
```

后续代码和实验逻辑以英文 plan 为准；本文档是运行说明和中文解释。

## 0. 总体思路

本轮 sweep 分为四个阶段：

```text
阶段 1：大规模 feature screening
阶段 2：从 feature screening 中挑选不超过 50 个 full QAT 配置
阶段 3：对选出的配置跑 10 epoch full QAT + 完整噪声场景 grid
阶段 4：根据 single-stage QAT 结果生成组合策略，并跑最终组合验证
```

为什么不是直接做全组合 sweep：

- stage 数量多，所有位宽全组合会爆炸。
- 很多明显不安全的位宽，可以通过前端 feature 误差提前筛掉。
- QAT 虽然不慢，但完整噪声场景 grid 测试有成本，所以 full QAT 配置总数控制在 50 个以内。

本轮实验固定以下硬件边界：

```text
PCM_W      = 8 signed
MFCC_OUT_W = 8 signed
Mel filter = rectangular
log        = PWL
backbone   = INT8 QAT
```

也就是说，本轮 sweep 主要优化中间 stage 的位宽。

## 1. 输出目录

本轮实验使用全新目录，避免覆盖之前 v1/v2 结果：

```text
dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep
```

目录结构如下：

```text
bit_accurate_mfcc_experiments_v3_bitwidth_sweep/
  feature_screening/
    bitwidth_feature_screening_plan.csv
    stage_feature_loss_raw.csv
    stage_feature_loss_summary.csv
    feature_candidate_status.csv

  single_stage_qat/
    full_qat_candidate_plan.csv
    single_stage_qat_commands.txt
    overrides/
    <candidate_id>/
      train_results.csv
      grid_results.csv
      models/

  combined_candidates/
    single_stage_qat_decision.csv
    accuracy_first_recommended.json
    aggressive_candidate.json
    final_bitwidth_strategy_summary.csv
    combined_qat_commands.txt
    accuracy_first_recommended/
      train_results.csv
      grid_results.csv
      models/
    aggressive_candidate/
      train_results.csv
      grid_results.csv
      models/
```

注意：脚本不会自动生成中文最终报告。最终报告需要根据 CSV/JSON 结果手动整理。

## 2. Stage 名称与位宽字段

命令行里使用硬件风格字段名，例如：

```text
LOG_W=20
DCT_ACC_W=32
TWIDDLE_W=12
```

内部会自动映射到 Python frontend 的 stage 名称。

| 命令行字段 | 内部 stage | 含义 |
|---|---|---|
| `PCM_W` | `pcm` | 前端输入 PCM 量化，固定 S8 |
| `MFCC_SAMPLE_W` | `preemphasis` | pre-emphasis 后的采样值位宽 |
| `HANN_COEFF_W` | `hann_coeff` | Hann window 系数量化位宽 |
| `FFT_IN_W` | `windowed` | 加窗后送入 DFT/FFT 的数据位宽 |
| `TWIDDLE_W` | `twiddle_coeff` | DFT/FFT 旋转因子位宽 |
| `FFT_DATA_W` | `fft_data` | 频域复数 real/imag 数据位宽 |
| `POWER_W` | `power` | power spectrum 位宽 |
| `MEL_ACC_W` | `mel` | Mel 滤波累加输出位宽 |
| `PWL_IN_W` | `pwl_input` | PWL log 输入位宽 |
| `LOG_W` | `log_mel` | PWL log 输出位宽 |
| `DCT_COEFF_W` | `dct_coeff` | DCT 系数量化位宽 |
| `DCT_ACC_W` | `dct` | DCT 累加输出位宽 |
| `MFCC_OUT_W` | `mfcc` | MFCC 输出位宽，固定 S8 |

本轮默认 sweep 范围：

```text
MFCC_SAMPLE_W:      12, 11, 10, 9, 8
HANN_COEFF_W:       16, 14, 12, 10, 8, 6
FFT_IN_W:           18, 17, 16, 15, 14, 13, 12, 11, 10
TWIDDLE_W:          16, 15, 14, 13, 12, 11, 10, 9, 8
FFT_DATA_W:         20, 19, 18, 17, 16, 15, 14, 13, 12
POWER_W:            41, 38, 36, 34, 32, 30, 28, 26, 24
MEL_ACC_W:          46, 44, 42, 40, 38, 36, 34, 32, 30
PWL_IN_W:           32, 30, 28, 26, 24, 22, 20, 18, 16
LOG_W:              24, 22, 20, 18, 16, 14, 12
DCT_COEFF_W:        8, 7, 6, 5, 4
DCT_ACC_W:          40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20
```

## 3. 第 0 步：确认 baseline 已经存在

本轮 bit-width sweep 以 `hardware_coeff_baseline` 为比较基准。

先确认以下文件存在：

```bash
ls dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_coeff_quant/hardware_coeff_baseline_grid_results.csv
ls dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_coeff_quant/hardware_coeff_baseline_train_results.csv
```

如果这两个文件已经存在，可以直接进入第 1 步。

如果不存在，先运行 baseline：

```bash
python dscnn_kws/quantization/qat_bit_accurate_mfcc_accuracy_first.py \
  --pattern "*L5_C64*.pt" \
  --qat_epochs 10 \
  --constraint_profile hardware_coeff_baseline \
  --output_dir dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_coeff_quant/hardware_coeff_baseline_models \
  --train_results_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_coeff_quant/hardware_coeff_baseline_train_results.csv \
  --grid_results_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_coeff_quant/hardware_coeff_baseline_grid_results.csv
```

这一步完成后，你应该看到：

```text
hardware_coeff_baseline_train_results.csv
hardware_coeff_baseline_grid_results.csv
hardware_coeff_baseline_models/
```

后续所有候选位宽都会和这个 baseline 对比。

## 4. 第 1 步：生成 feature screening 计划

这一步只生成计划，不真正跑前端。

命令：

```bash
python dscnn_kws/quantization/screen_bit_accurate_mfcc_bitwidths.py \
  --datasets mobvoi_hi_xiaowen_binary_hardneg mobvoi_nihao_wenwen_binary_hardneg \
  --batches 8 \
  --output_dir dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/feature_screening \
  --plan_only
```

输出文件：

```text
dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/feature_screening/bitwidth_feature_screening_plan.csv
```

打开这个文件，重点看：

| 字段 | 含义 |
|---|---|
| `candidate_id` | 候选配置 ID |
| `candidate_type` | `single_stage` 或 `pairwise` |
| `target_fields` | 被修改的硬件字段 |
| `stage_bit_overrides` | 实际覆盖的位宽 |
| `notes` | 人类可读说明 |

例子：

```text
single_LOG_W_20
stage_bit_overrides = {"LOG_W": 20}
```

表示只把 `LOG_W` 从 baseline 的 24 bit 压到 20 bit，其他 stage 不变。

## 5. 第 2 步：运行大规模 feature screening

确认计划没问题后，运行真正的 feature screening：

```bash
python dscnn_kws/quantization/screen_bit_accurate_mfcc_bitwidths.py \
  --datasets mobvoi_hi_xiaowen_binary_hardneg mobvoi_nihao_wenwen_binary_hardneg \
  --batches 8 \
  --output_dir dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/feature_screening
```

这一步不训练模型，只跑 MFCC 前端 forward。它会比较：

```text
reference: hardware_coeff_baseline
candidate: hardware_coeff_baseline + 某个 stage 的位宽覆盖
```

它同时记录三类信息：

```text
1. 本 stage 自己的量化误差
2. 误差传播到后续 stage 后的变化
3. 最终 mfcc_int8 特征是否仍然和 baseline 高度一致
```

输出文件：

```text
stage_feature_loss_raw.csv
stage_feature_loss_summary.csv
feature_candidate_status.csv
```

### 5.1 查看 raw 结果

`stage_feature_loss_raw.csv` 是最细粒度结果。

一行大致表示：

```text
某个 candidate
某个 dataset
某个 measured_stage
在一个 batch 上的误差
```

重点字段：

| 字段 | 含义 |
|---|---|
| `candidate_id` | 候选配置 |
| `target_fields` | 被压缩的位宽字段 |
| `measured_stage` | 当前被观察的 stage |
| `mae` | 平均绝对误差 |
| `rmse` | 均方根误差 |
| `relative_rmse` | 相对 RMSE |
| `max_abs_error` | 最大绝对误差 |
| `cosine` | 余弦相似度 |
| `pearson` | Pearson 相关系数 |
| `zero_ratio` | 零值比例 |
| `saturation_ratio` | 饱和比例 |

通常不用手动逐行看 raw 文件，除非某个候选很异常。

### 5.2 查看 summary 结果

`stage_feature_loss_summary.csv` 是按 candidate 和 measured stage 汇总后的结果。

优先看这些 stage：

```text
fft_complex
power
rectangular_mel
pwl_log_mel
dct_acc
mfcc_int8
```

如果某个候选只改了 `TWIDDLE_W`，但你发现：

```text
twiddle_coeff 误差很小
power 误差变大
mfcc_int8 cosine 明显下降
```

说明旋转因子误差被后面的平方和累加放大了，这个候选要谨慎。

### 5.3 查看候选状态表

最重要的是：

```text
feature_candidate_status.csv
```

这个文件会把每个 candidate 分成：

```text
green   // feature 层面稳定，优先进入 full QAT
yellow  // 有轻微漂移，但值得尝试
red     // feature 层面明显不安全，不进入 full QAT
```

重点字段：

| 字段 | 含义 |
|---|---|
| `feature_status` | `green` / `yellow` / `red` |
| `mfcc_cosine` | 最终 MFCC 与 baseline 的相似度 |
| `mfcc_relative_rmse` | 最终 MFCC 相对误差 |
| `downstream_max_relative_rmse` | 后续 stage 最大相对误差 |
| `max_saturation_ratio` | 最大饱和比例 |
| `compression_score` | 压缩程度，越大代表压得越多 |

一般判断：

```text
green:
  可以优先进入 full QAT

yellow:
  作为激进候选或边界候选进入 full QAT

red:
  默认不进入 full QAT
```

## 6. 第 3 步：从 feature screening 里规划 full QAT 候选

这一步读取：

```text
feature_candidate_status.csv
```

然后挑选不超过 50 个配置进入 full QAT。

命令：

```bash
python dscnn_kws/quantization/plan_bit_accurate_mfcc_bitwidth_qat.py \
  --feature_status_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/feature_screening/feature_candidate_status.csv \
  --output_dir dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/single_stage_qat \
  --input_dir /root/kws/dscnn_kws/dscnn_kws/runs/snr_scene_arch_sweep_best_models \
  --pattern "*L5_C64*.pt" \
  --arch L5_C64 \
  --qat_epochs 10 \
  --max_configs 50
```

输出文件：

```text
single_stage_qat/full_qat_candidate_plan.csv
single_stage_qat/single_stage_qat_commands.txt
single_stage_qat/overrides/*.json
```

### 6.1 查看 full_qat_candidate_plan.csv

重点看：

| 字段 | 含义 |
|---|---|
| `candidate_id` | 即将跑 QAT 的候选 |
| `candidate_type` | `single_stage` 或 `pairwise` |
| `target_fields` | 压缩了哪些字段 |
| `feature_status` | 来自 feature screening 的状态 |
| `stage_bit_overrides` | 实际位宽覆盖 |
| `selection_reason` | 为什么选入 full QAT |

候选数量应小于等于 50。

如果你觉得候选太多，可以重新运行并调小：

```bash
--max_configs 40
--max_single_stage_configs 32
--max_pairwise_configs 8
```

如果你觉得候选太少，可以增加：

```bash
--per_stage_green 3
--per_stage_yellow 2
```

但总数仍建议控制在 50 以内。

## 7. 第 4 步：运行 single-stage full QAT

推荐方式是直接让 planning 脚本执行：

```bash
python dscnn_kws/quantization/plan_bit_accurate_mfcc_bitwidth_qat.py \
  --feature_status_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/feature_screening/feature_candidate_status.csv \
  --output_dir dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/single_stage_qat \
  --input_dir /root/kws/dscnn_kws/dscnn_kws/runs/snr_scene_arch_sweep_best_models \
  --pattern "*L5_C64*.pt" \
  --arch L5_C64 \
  --qat_epochs 10 \
  --max_configs 50 \
  --run
```

这会顺序执行 `full_qat_candidate_plan.csv` 中的候选。

每个候选会产生一个独立目录，例如：

```text
single_stage_qat/single_LOG_W_20/
  train_results.csv
  grid_results.csv
  models/
```

其中：

```text
train_results.csv
```

记录 QAT 训练、tau list 测试、fakequant/hard frontend 对齐结果。

```text
grid_results.csv
```

记录完整 TAU noise scene/SNR grid 的结果。

### 7.1 如果中途断了怎么办

如果中途断掉，先看已经完成了哪些目录：

```bash
ls dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/single_stage_qat
```

一个候选完成的标志是：

```text
<candidate_id>/train_results.csv
<candidate_id>/grid_results.csv
```

如果某个候选目录没有 `grid_results.csv`，说明它没有完整完成。

你可以打开：

```text
single_stage_qat/single_stage_qat_commands.txt
```

找到未完成的那一行，单独复制运行。

注意：手动复制命令时，`--pattern "*L5_C64*.pt"` 要保留引号，避免 shell 展开通配符。

### 7.2 手动跑单个候选的例子

如果你只想手动验证一个候选，例如 `LOG_W=20`：

```bash
python dscnn_kws/quantization/qat_bit_accurate_mfcc_accuracy_first.py \
  --pattern "*L5_C64*.pt" \
  --qat_epochs 10 \
  --constraint_profile hardware_coeff_baseline \
  --stage_bit_overrides LOG_W=20 \
  --bitwidth_sweep_id manual_LOG_W_20 \
  --output_dir dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/manual_LOG_W_20/models \
  --train_results_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/manual_LOG_W_20/train_results.csv \
  --grid_results_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/manual_LOG_W_20/grid_results.csv
```

如果要同时改多个 stage：

```bash
python dscnn_kws/quantization/qat_bit_accurate_mfcc_accuracy_first.py \
  --pattern "*L5_C64*.pt" \
  --qat_epochs 10 \
  --constraint_profile hardware_coeff_baseline \
  --stage_bit_overrides LOG_W=20 DCT_ACC_W=32 PWL_IN_W=24 \
  --bitwidth_sweep_id manual_LOG20_DCT32_PWL24 \
  --output_dir dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/manual_LOG20_DCT32_PWL24/models \
  --train_results_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/manual_LOG20_DCT32_PWL24/train_results.csv \
  --grid_results_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/manual_LOG20_DCT32_PWL24/grid_results.csv
```

## 8. 第 5 步：汇总 single-stage QAT 结果

所有 full QAT 候选跑完后，执行：

```bash
python dscnn_kws/quantization/summarize_bit_accurate_mfcc_bitwidth_qat.py \
  --qat_plan_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/single_stage_qat/full_qat_candidate_plan.csv \
  --qat_root dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/single_stage_qat \
  --output_dir dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/combined_candidates \
  --baseline_grid_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_coeff_quant/hardware_coeff_baseline_grid_results.csv \
  --baseline_train_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_coeff_quant/hardware_coeff_baseline_train_results.csv
```

输出：

```text
combined_candidates/single_stage_qat_decision.csv
combined_candidates/accuracy_first_recommended.json
combined_candidates/aggressive_candidate.json
combined_candidates/final_bitwidth_strategy_summary.csv
combined_candidates/combined_qat_commands.txt
```

### 8.1 查看 single_stage_qat_decision.csv

这个文件最重要。

它会把每个 QAT 候选判定为：

```text
pass
borderline
fail
missing_grid_results
```

判定标准：

```text
pass:
  avg_acc_drop_pp <= 0.3
  low_snr_acc_drop_pp <= 1.0
  fake_hard_gap_pp <= 0.2

borderline:
  avg_acc_drop_pp <= 0.5
  low_snr_acc_drop_pp <= 1.5

fail:
  超过 borderline 条件

missing_grid_results:
  没有找到该 candidate 的 grid_results.csv
```

重点字段：

| 字段 | 含义 |
|---|---|
| `avg_acc` | 候选平均准确率 |
| `baseline_avg_acc` | baseline 平均准确率 |
| `avg_acc_drop_pp` | 相对 baseline 平均掉点，单位是百分点 |
| `low_snr_acc` | 低 SNR 平均准确率 |
| `baseline_low_snr_acc` | baseline 低 SNR 平均准确率 |
| `low_snr_acc_drop_pp` | 低 SNR 掉点 |
| `fake_hard_gap_pp` | fakequant frontend 和 hard frontend 的 tau list 差距 |
| `qat_status` | 最终判定 |

如果看到 `missing_grid_results`，说明对应候选还没有完整跑完，需要回到第 4 步补跑。

### 8.2 查看 accuracy_first_recommended.json

这个文件的规则是：

```text
每个 stage 选择 pass 中最低的位宽。
```

它是主推荐策略。

例子：

```json
{
  "strategy": "accuracy_first_recommended",
  "stage_bit_overrides": {
    "LOG_W": 20,
    "DCT_ACC_W": 32
  }
}
```

实际文件里会包含所有被选择的 stage 位宽。

### 8.3 查看 aggressive_candidate.json

这个文件的规则是：

```text
每个 stage 选择 pass 或 borderline 中最低的位宽。
```

它不是默认推荐方案，而是压缩上限参考。

用途：

```text
1. 看硬件还能不能继续省
2. 观察哪些 stage 一压就掉点
3. 给后续硬件面积/功耗评估一个对照方案
```

## 9. 第 6 步：运行组合策略 full QAT

第 5 步会生成：

```text
combined_candidates/combined_qat_commands.txt
```

里面有两条命令：

```text
accuracy_first_recommended
aggressive_candidate
```

也可以直接用下面两条命令运行。

### 9.1 跑 accuracy_first 推荐组合

```bash
python dscnn_kws/quantization/qat_bit_accurate_mfcc_accuracy_first.py \
  --constraint_profile hardware_coeff_baseline \
  --stage_bit_overrides_json dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/combined_candidates/accuracy_first_recommended.json \
  --bitwidth_sweep_id accuracy_first_recommended \
  --pattern "*L5_C64*.pt" \
  --qat_epochs 10 \
  --output_dir dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/combined_candidates/accuracy_first_recommended/models \
  --train_results_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/combined_candidates/accuracy_first_recommended/train_results.csv \
  --grid_results_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/combined_candidates/accuracy_first_recommended/grid_results.csv
```

### 9.2 跑 aggressive 对照组合

```bash
python dscnn_kws/quantization/qat_bit_accurate_mfcc_accuracy_first.py \
  --constraint_profile hardware_coeff_baseline \
  --stage_bit_overrides_json dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/combined_candidates/aggressive_candidate.json \
  --bitwidth_sweep_id aggressive_candidate \
  --pattern "*L5_C64*.pt" \
  --qat_epochs 10 \
  --output_dir dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/combined_candidates/aggressive_candidate/models \
  --train_results_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/combined_candidates/aggressive_candidate/train_results.csv \
  --grid_results_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/combined_candidates/aggressive_candidate/grid_results.csv
```

跑完后应该有：

```text
combined_candidates/accuracy_first_recommended/train_results.csv
combined_candidates/accuracy_first_recommended/grid_results.csv

combined_candidates/aggressive_candidate/train_results.csv
combined_candidates/aggressive_candidate/grid_results.csv
```

## 10. 第 7 步：检查组合策略是否通过

组合策略需要重新检查，因为：

```text
单个 stage 降位宽都安全，不代表多个 stage 一起降位宽仍然安全。
```

先看 `train_results.csv`：

```text
combined_candidates/accuracy_first_recommended/train_results.csv
combined_candidates/aggressive_candidate/train_results.csv
```

重点字段：

| 字段 | 含义 |
|---|---|
| `fakequant_qat_test_tau_list_acc` | QAT fakequant 模型 tau list 准确率 |
| `quantized_backbone_fakequant_frontend_test_tau_list_acc` | INT8 backbone + fakequant frontend 准确率 |
| `bit_accurate_mfcc_int8_backbone_test_tau_list_acc` | hard bit-accurate frontend + INT8 backbone 准确率 |

如果后三者非常接近，说明：

```text
训练前端、量化 backbone、hard bit-accurate frontend 三者链路对齐良好。
```

再看 `grid_results.csv`：

```text
combined_candidates/accuracy_first_recommended/grid_results.csv
combined_candidates/aggressive_candidate/grid_results.csv
```

重点看：

```text
1. 平均 acc 是否接近 hardware_coeff_baseline
2. 0 dB / -5 dB 是否明显掉点
3. 哪些 scene 最敏感
```

当前自动汇总脚本不会生成中文最终报告，因此这一步需要你根据 CSV 做最终解释。

## 11. 如果 accuracy_first 组合掉点超标怎么办

如果 `accuracy_first_recommended` 单独 stage 都是 pass，但组合后掉点超标，说明多个 stage 的误差叠加了。

优先回退这些敏感 stage：

```text
TWIDDLE_W
FFT_DATA_W
LOG_W
DCT_ACC_W
PWL_IN_W
POWER_W
MEL_ACC_W
```

其次回退：

```text
FFT_IN_W
MFCC_SAMPLE_W
DCT_COEFF_W
HANN_COEFF_W
```

回退方式：

1. 打开：

```text
combined_candidates/accuracy_first_recommended.json
```

2. 把敏感 stage 的位宽提高一档。

例如：

```text
LOG_W: 18 -> 20
DCT_ACC_W: 28 -> 32
FFT_DATA_W: 16 -> 18
```

3. 保存成新文件，例如：

```text
combined_candidates/accuracy_first_repaired.json
```

4. 重新跑组合 QAT：

```bash
python dscnn_kws/quantization/qat_bit_accurate_mfcc_accuracy_first.py \
  --constraint_profile hardware_coeff_baseline \
  --stage_bit_overrides_json dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/combined_candidates/accuracy_first_repaired.json \
  --bitwidth_sweep_id accuracy_first_repaired \
  --pattern "*L5_C64*.pt" \
  --qat_epochs 10 \
  --output_dir dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/combined_candidates/accuracy_first_repaired/models \
  --train_results_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/combined_candidates/accuracy_first_repaired/train_results.csv \
  --grid_results_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/combined_candidates/accuracy_first_repaired/grid_results.csv
```

## 12. 推荐的完整运行顺序

如果从头到尾跑，按下面顺序执行。

### 12.1 确认 baseline

```bash
ls dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_coeff_quant/hardware_coeff_baseline_grid_results.csv
ls dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_coeff_quant/hardware_coeff_baseline_train_results.csv
```

### 12.2 生成 feature screening 计划

```bash
python dscnn_kws/quantization/screen_bit_accurate_mfcc_bitwidths.py \
  --datasets mobvoi_hi_xiaowen_binary_hardneg mobvoi_nihao_wenwen_binary_hardneg \
  --batches 8 \
  --output_dir dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/feature_screening \
  --plan_only
```

### 12.3 运行 feature screening

```bash
python dscnn_kws/quantization/screen_bit_accurate_mfcc_bitwidths.py \
  --datasets mobvoi_hi_xiaowen_binary_hardneg mobvoi_nihao_wenwen_binary_hardneg \
  --batches 8 \
  --output_dir dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/feature_screening
```

### 12.4 规划 full QAT 候选

```bash
python dscnn_kws/quantization/plan_bit_accurate_mfcc_bitwidth_qat.py \
  --feature_status_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/feature_screening/feature_candidate_status.csv \
  --output_dir dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/single_stage_qat \
  --input_dir /root/kws/dscnn_kws/dscnn_kws/runs/snr_scene_arch_sweep_best_models \
  --pattern "*L5_C64*.pt" \
  --arch L5_C64 \
  --qat_epochs 10 \
  --max_configs 50
```

### 12.5 运行 full QAT 候选

```bash
python dscnn_kws/quantization/plan_bit_accurate_mfcc_bitwidth_qat.py \
  --feature_status_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/feature_screening/feature_candidate_status.csv \
  --output_dir dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/single_stage_qat \
  --input_dir /root/kws/dscnn_kws/dscnn_kws/runs/snr_scene_arch_sweep_best_models \
  --pattern "*L5_C64*.pt" \
  --arch L5_C64 \
  --qat_epochs 10 \
  --max_configs 50 \
  --run
```

### 12.6 汇总 single-stage QAT 并生成组合策略

```bash
python dscnn_kws/quantization/summarize_bit_accurate_mfcc_bitwidth_qat.py \
  --qat_plan_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/single_stage_qat/full_qat_candidate_plan.csv \
  --qat_root dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/single_stage_qat \
  --output_dir dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/combined_candidates \
  --baseline_grid_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_coeff_quant/hardware_coeff_baseline_grid_results.csv \
  --baseline_train_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_coeff_quant/hardware_coeff_baseline_train_results.csv
```

### 12.7 跑 accuracy_first 组合

```bash
python dscnn_kws/quantization/qat_bit_accurate_mfcc_accuracy_first.py \
  --constraint_profile hardware_coeff_baseline \
  --stage_bit_overrides_json dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/combined_candidates/accuracy_first_recommended.json \
  --bitwidth_sweep_id accuracy_first_recommended \
  --pattern "*L5_C64*.pt" \
  --qat_epochs 10 \
  --output_dir dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/combined_candidates/accuracy_first_recommended/models \
  --train_results_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/combined_candidates/accuracy_first_recommended/train_results.csv \
  --grid_results_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/combined_candidates/accuracy_first_recommended/grid_results.csv
```

### 12.8 跑 aggressive 组合

```bash
python dscnn_kws/quantization/qat_bit_accurate_mfcc_accuracy_first.py \
  --constraint_profile hardware_coeff_baseline \
  --stage_bit_overrides_json dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/combined_candidates/aggressive_candidate.json \
  --bitwidth_sweep_id aggressive_candidate \
  --pattern "*L5_C64*.pt" \
  --qat_epochs 10 \
  --output_dir dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/combined_candidates/aggressive_candidate/models \
  --train_results_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/combined_candidates/aggressive_candidate/train_results.csv \
  --grid_results_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep/combined_candidates/aggressive_candidate/grid_results.csv
```

## 13. 最终应该得到什么结论

跑完后，你应该能够回答这些问题：

```text
1. 每个 stage 的最低 pass 位宽是多少？
2. 每个 stage 的 aggressive 位宽是多少？
3. 哪些 stage 最敏感？
4. 哪些 stage 可以大胆压缩？
5. accuracy_first 组合是否仍然满足准确率护栏？
6. aggressive 组合相比 accuracy_first 多省了多少位宽，又多掉了多少点？
7. fakequant 前端、hard bit-accurate 前端、INT8 backbone 是否仍然对齐？
```

最终推荐策略以：

```text
combined_candidates/accuracy_first_recommended.json
```

为主。

如果 aggressive 组合掉点也很小，可以进一步讨论是否把 aggressive 中的部分 stage 合入最终硬件策略。

