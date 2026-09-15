# 位精确 MFCC 前端：精度优先设计

本文定义了一套面向位精确 MFCC 前端的精度优先设计方案。所选方法结合：

- 方案 1：优先进行特征对齐
- 方案 3：逐步收紧硬件约束

目标不是一开始就立即最小化所有中间位宽。第一目标是在目标部署最关键的四个硬约束下恢复精度上限：

- 前端语音输入为有符号 INT8 PCM
- 矩形 Mel 滤波
- PWL 形式的 log 近似
- 有符号 INT8 MFCC 输出

在确认这一精度上限之后，再逐步收紧中间位宽和定点运算约束。

## 目标

以全精度 MFCC 模型为起点，然后训练并评估位精确 MFCC 前端与 INT8 主干网络路径。

主要精度目标：

- 与 `snr_scene_arch_sweep_grid_results.csv` 进行对比
- 在公共评估单元格上，平均准确率下降应不超过 1 个百分点

第一阶段范围：

- 架构：`L5_C64`
- 数据集：
  - `mobvoi_hi_xiaowen_binary_hardneg`
  - `mobvoi_nihao_wenwen_binary_hardneg`
- 评估：使用与现有 TAU 噪声场景扫描相同的场景和 SNR 网格
- Mel 形状：矩形
- log 形式：PWL log
- 前端输入：有符号 INT8 PCM
- 最终前端输出：有符号 INT8 MFCC

## 设计原则

当前直接位精确前端的结果，相比普通 INT8 MFCC 前端出现了更大的精度下降。这意味着主要风险并不在矩形 Mel 本身，而在于缩放、移位、舍入、饱和、log/DCT 近似，以及训练/推理前端差异所累积产生的不匹配。

因此，该设计将问题拆分为三个层次：

1. 特征精度上限：
   保持中间计算为宽位宽或接近浮点的形式，仅强制使用有符号 INT8 PCM 输入、矩形 Mel、PWL log 和最终 INT8 MFCC 输出。
2. 可训练伪量化前端：
   加入使用直通估计器（STE）的可微量化算子，并从全精度检查点开始训练。
3. 整数推理前端：
   将通过验证的伪量化约束契约转换为确定性的整数前端，并验证两者一致性。

只有当前一层通过精度和特征对齐检查后，才应继续收紧下一层。

## 命名约定与历史实验边界

本轮方案统一使用正确命名：

```text
bit-accurate   // 文档描述
bit_accurate   // Python 文件、目录、CSV/JSON/Markdown 产物
BitAccurate    // Python 类名
```

之前粗糙实验中误用了 `bit_accuracy` / `BitAccuracy` 命名。那些脚本和前端文件只作为历史结果对照，不作为本轮实验的实现基础。本轮新增的前端、QAT 脚本、评估脚本、输出目录和 spec 文件都应重新命名并重新实现，避免与旧实验混淆。

旧命名只允许出现在“历史结果对照”语境中，例如读取旧 CSV 来解释为什么需要新方案；训练、QAT、推理和导出流程不得导入或继承旧的 `bit_accuracy` 脚本。

## 前端变体

### 1. 精度上限前端

目的：回答这样一个问题——当中间精度不是瓶颈时，有符号 INT8 PCM 输入、矩形 Mel、PWL log 加 INT8 MFCC 输出能否保持精度。

行为：

- 波形输入遵循与基线一致的输入归一化约定
- 前端入口处将语音输入量化为有符号 INT8 PCM；这是从阶段 1 开始就启用的硬件边界约束
- Mel 滤波器使用矩形形状
- FFT、功率、Mel 累加和 DCT 使用高精度计算
- log 必须使用 PWL 形式；精度上限阶段只放宽 PWL 输入/输出位宽，不回退到 exact log
- 最终 MFCC 输出量化为有符号 INT8
- 训练期间的量化应使用直通估计器（STE）

推荐名称：

- `BitAccurateMFCCHighPrecisionFrontend`

该前端是第一道门槛。如果它无法达到目标，那么此时继续缩减中间位宽没有意义。

### 2. 伪量化位精确前端

目的：在训练模型时，让模型暴露于最终位精确前端将产生的相同量化效应。

行为：

- 每个阶段均可选择性启用量化、裁剪、舍入和缩放约束
- 反向传播期间，量化使用 STE
- 前向过程记录饱和率、零值比例、最小值/最大值和分布漂移等统计信息
- 输出形状和值域契约与最终整数前端一致

推荐名称：

- `BitAccurateMFCCFakeQuantFrontend`

这是 QAT 前端。主干网络在训练期间就能看到前端失真，而不是仅在训练结束后才遇到这些失真。

### 3. 整数位精确前端

目的：为面向硬件的契约提供确定性推理参考实现。

行为：

- 不需要梯度
- 使用整数运算、显式移位、显式舍入和饱和处理
- 使用从伪量化实验中导出的、已经验证的前端规范
- 生成供主干网络使用的最终有符号 INT8 MFCC 张量

推荐的新目标文件：

- `dscnn_kws/frontend/bit_accurate_mfcc_frontend.py`

推荐类名：

- `BitAccurateMFCCFrontend`

本轮应重新实现该整数参考路径。旧的 `dscnn_kws/frontend/bit_accuracy_mfcc_frontend.py` 只作为历史粗糙实验参考，不作为本轮前端的父类、导入依赖或配置来源。最终实现应由导出的规范驱动，而不是依赖散落在代码中的手工调优常量。

## 阶段契约

每个阶段都必须具备明确的数值契约。最终验证后的设计应生成类似下表的结果，但其中每个选择都必须附带实验依据：

```text
PCM_W              = 8 signed
MFCC_SAMPLE_W      = 12 signed
HANN_COEFF_W       = 16 unsigned
FFT_IN_W           = 18 signed
FFT_DATA_W         = 20 signed
TWIDDLE_W          = 16 signed
POWER_W            = 41 unsigned
MEL_ACC_W          = 46 unsigned
PWL_IN_W           = 32 unsigned
LOG_W              = 24 signed
DCT_COEFF_W        = 8 signed
DCT_ACC_W          = 40 signed
MFCC_OUT_W         = 8 signed
```

最终规范应包含的不只是位宽：

```text
PCM_W
MFCC_SAMPLE_W
PREEMPH_COEFF_W
HANN_COEFF_W
FFT_IN_W
FFT_DATA_W
TWIDDLE_W
FFT_STAGE_SHIFT
POWER_W
MEL_COEFF_W
MEL_ACC_W
MEL_DRAIN_SHIFT
PWL_IN_W
LOG_APPROX_MODE
LOG_PWL_NUM_SEGMENTS
LOG_PWL_BREAKPOINTS
LOG_PWL_SLOPES
LOG_PWL_INTERCEPTS
LOG_W
LOG_FRAC_BITS
DCT_COEFF_W
DCT_ACC_W
DCT_SHIFT
MFCC_OUT_W
MFCC_OUT_SCALE
ROUNDING_MODE
SATURATION_MODE
```

对于每个字段，报告中应记录：

- 选定的位宽或 Q 格式
- 有符号/无符号属性
- 缩放因子或小数位数
- 移位量
- 舍入规则
- 饱和范围
- 观测到的最小值/最大值
- 饱和率
- 零值比例
- 相对于参考实现的特征误差
- 相对于全精度模型的准确率变化

## 特征对齐

特征对齐是组合方案中的第一部分。它应在长时间 QAT 扫描之前执行。

参考前端：

- 基线模型使用的现有全精度 `TorchMFCC` 路径

候选前端阶段：

```text
waveform
pcm_or_scaled_sample
preemphasis_or_sample_scale
windowed_frame
fft_input
fft_complex
power
rectangular_mel
pwl_log_mel
dct_acc
mfcc_prequant
mfcc_int8
```

对于每个阶段，计算：

- MAE
- RMSE
- 最大绝对误差
- 余弦相似度
- Pearson 相关系数
- 均值/标准差/最小值/最大值
- 饱和率
- 零值比例
- 按 MFCC 系数统计的误差
- 按时间帧统计的 MFCC 误差

最重要的对齐目标并不是实现与浮点 MFCC 的逐比特相等。真正的目标是识别前端在哪个阶段开始产生足够大的分布差异，以至于主干网络无法恢复。

## 实验阶段

### 阶段 0：基线整理

输入：

- 全精度网格结果：`snr_scene_arch_sweep_grid_results.csv`
- INT8 MFCC QAT 结果：
  `dscnn_kws/quantization/qat_int8_mfcc_frontend_grid_results.csv`
- 历史粗糙实验结果（只读对照，不作为本轮实现依赖）：
  `dscnn_kws/quantization/qat_bit_accuracy_mfcc_frontend_grid_results.csv`

输出：

- 公共单元格对比表
- 按数据集和 SNR 汇总的基线结果
- 为 `L5_C64` 选定的全精度检查点路径

该阶段用于确保之后的所有实验都比较相同的数据集、架构、场景、SNR 和检查点系列。

### 阶段 1：精度上限

约束配置：

```text
mel_filter_shape      = rectangular
log_approx_mode       = pwl
pcm_input             = signed INT8
mfcc_output           = signed INT8
intermediate_widths   = high precision
pwl_log_io            = high precision
dct_coeff             = high precision or at least >= 16 bits
power/mel accum       = no effective clipping
```

训练：

- 从全精度 `L5_C64` 检查点初始化
- 使用精度上限前端
- 前端输入固定为有符号 INT8 PCM；中间算子可以先保持宽位宽
- log 算子固定为 PWL；只允许先使用较宽的 PWL 输入/输出位宽
- 在启用前端 INT8 输出伪量化的情况下训练
- 按照现有 INT8 MFCC 流程对主干网络进行 QAT

验收标准：

- 相对于全精度基线，平均下降应不超过 1 个百分点
- 如果未通过，应先检查 MFCC 输出缩放、C0 处理、log 归一化和各系数分布，再考虑缩减位宽

### 阶段 2：前端感知 QAT

约束配置：

```text
mel_filter_shape    = rectangular
log_approx_mode     = pwl
pcm_input           = signed INT8
mfcc_output         = signed INT8
frontend_quant      = differentiable fake quant
backbone_quant      = INT8 QAT
```

训练时应让主干网络看到与推理阶段相同的 MFCC 动态范围和裁剪行为。该阶段是真正的精度优先 QAT 基线。

默认 MFCC 输出缩放：

- 首先使用逐系数 MFCC 输出缩放
- 然后测试逐张量缩放，作为硬件简化方案

原因：

- 逐系数缩放能提供更现实的精度上限
- 逐张量缩放成本更低，但可能过度压缩低能量系数

验收标准：

- 相对于全精度模型，平均下降不超过 1 个百分点
- 阶段 4 完成后，整数前端与伪量化前端之间的准确率差距不超过 0.5 个百分点

### 阶段 3：逐步收紧约束

阶段 1 和阶段 2 通过后，以受控方式逐步降低硬件成本。

推荐顺序：

1. DCT 系数位宽：
   `16 -> 12 -> 10 -> 8`
2. DCT 累加器与最终移位：
   选择满足低饱和率和稳定精度的最小位宽
3. PWL log 输出位宽、小数位数和查表/分段参数：
   扫描 `LOG_W`、`LOG_FRAC_BITS`、`PWL_IN_W` 和 PWL 分段参数；不要回退到 exact log
4. Mel 输出缩减：
   扫描 `MEL_DRAIN_SHIFT`
5. 功率和 Mel 累加器位宽：
   仅在观察真实最小值/最大值和饱和率后再缩减
6. FFT 和旋转因子位宽：
   最后再收紧，因为这里的误差会影响所有后续阶段

不要一次改变多个阶段。每次扫描应只有一个主要变量，其他阶段契约保持固定。

每一步缩减的验收标准：

- 相对于上一个已接受配置，平均准确率下降应较小
- 不应出现大范围低 SNR 崩溃
- 饱和率应保持可解释且稳定
- 特征漂移应能明确定位到受影响的具体阶段

### 阶段 4：整数推理一致性

伪量化配置通过验收后，将其导出为规范并运行整数前端。

检查项：

- 伪量化前端输出与整数前端输出的对比
- 整数前端加已训练主干网络，与伪量化前端加已训练主干网络之间的对比
- 按数据集、SNR 和场景统计准确率差距

验收标准：

- 平均准确率差距不超过 0.5 个百分点
- 任一场景/SNR 的差距超过 5 个百分点时，必须报告并解释
- 阶段级不匹配应能够追溯到舍入、移位、饱和或查表/PWL 近似

### 阶段 5：已验证位宽规范

最终输出是一份经过验证的前端契约，而不仅仅是一张手写表格。

推荐输出文件：

```text
dscnn_kws/quantization/bit_accurate_mfcc_experiments/
  L5_C64_accuracy_first_summary.md
  L5_C64_bit_accurate_mfcc_validated_spec.json
  L5_C64_bit_accurate_mfcc_validated_spec.md
  feature_alignment_summary.csv
  constraint_sweep_summary.csv
  integer_parity_summary.csv
```

最终 Markdown 规范应包含：

- 位宽表
- Q 格式表
- 移位表
- PWL log 分段表，包括断点、斜率、截距和输入/输出缩放
- 舍入和饱和规则
- 各阶段最小值/最大值统计
- 饱和率和零值比例
- 特征对齐指标
- 与全精度和 INT8 MFCC QAT 的准确率对比
- 面向硬件实现的最终建议

## 建议脚本

以下是实现阶段建议交付的脚本。

```text
dscnn_kws/frontend/bit_accurate_mfcc_high_precision.py
dscnn_kws/frontend/bit_accurate_mfcc_fakequant.py
dscnn_kws/frontend/bit_accurate_mfcc_frontend.py
dscnn_kws/quantization/analyze_bit_accurate_mfcc_stages.py
dscnn_kws/quantization/sweep_bit_accurate_mfcc_constraints.py
dscnn_kws/quantization/qat_bit_accurate_mfcc_accuracy_first.py
dscnn_kws/quantization/eval_bit_accurate_mfcc_integer_parity.py
```

## 胎教级逐步运行指南

下面的命令默认都在项目根目录执行，也就是当前目录下能看到 `dscnn_kws/quantization`。如果你在 PowerShell 里运行，把命令里的反斜杠 `\` 换成反引号，或者直接写成一整行。

如果服务器上的数据或 checkpoint 路径不是脚本默认值，就额外加：

```bash
--root /root/kws/dscnn_kws/dscnn_kws/data
--input_dir /root/kws/dscnn_kws/dscnn_kws/runs/snr_scene_arch_sweep_best_models
```

### 第 0 步：确认环境和脚本能被 Python 解析

先确认训练环境里有 PyTorch：

```bash
python -c "import torch; print(torch.__version__)"
```

再确认新脚本没有语法错误：

```bash
python -m py_compile \
  dscnn_kws/frontend/bit_accurate_mfcc_high_precision.py \
  dscnn_kws/frontend/bit_accurate_mfcc_fakequant.py \
  dscnn_kws/frontend/bit_accurate_mfcc_frontend.py \
  dscnn_kws/quantization/analyze_bit_accurate_mfcc_stages.py \
  dscnn_kws/quantization/qat_bit_accurate_mfcc_accuracy_first.py \
  dscnn_kws/quantization/sweep_bit_accurate_mfcc_constraints.py \
  dscnn_kws/quantization/eval_bit_accurate_mfcc_integer_parity.py
```

怎么看结果：

- 没有输出，说明语法检查通过。
- 如果报 `ModuleNotFoundError: torch`，说明当前环境不是训练环境，需要先切换到安装了 PyTorch/torchaudio 的环境。
- 如果报某个 `.py` 文件语法错误，先修这个错误，不要进入后续训练。

### 第 1 步：先做 stage-level 特征对齐检查

这一步不训练模型，只看前端每一阶段的输出是否明显跑偏。第一轮用 `upper_bound_s8_mfcc`，也就是：

```text
S8 PCM input + rectangular Mel + PWL log + high precision intermediate + S8 MFCC output
```

命令：

```bash
python dscnn_kws/quantization/analyze_bit_accurate_mfcc_stages.py \
  --datasets mobvoi_hi_xiaowen_binary_hardneg mobvoi_nihao_wenwen_binary_hardneg \
  --constraint_profile upper_bound_s8_mfcc \
  --output_dir dscnn_kws/quantization/bit_accurate_mfcc_experiments/stage_upper_bound \
  --batches 8
```

输出文件：

```text
dscnn_kws/quantization/bit_accurate_mfcc_experiments/stage_upper_bound/
  upper_bound_s8_mfcc_feature_alignment_summary.csv
  <dataset>_upper_bound_s8_mfcc_stage_analysis_spec.json
  <dataset>_upper_bound_s8_mfcc_stage_stats.json
```

这一步具体在做什么：

1. 脚本会构造两个 MFCC 前端。
2. 第一个是参考前端：同样使用矩形 Mel 和 PWL log，但关闭所有 stage quant，用来代表“当前硬件形式下的高精度参考”。
3. 第二个是候选前端：使用 `--constraint_profile` 指定的量化约束。例如 `upper_bound_s8_mfcc` 会启用 S8 PCM 输入和 S8 MFCC 输出，但中间 FFT、power、Mel、log、DCT 先保持高精度。
4. 脚本从指定数据集里取若干 batch 的语音，分别送进参考前端和候选前端。
5. 对每一个 stage 的输出做逐元素比较，统计误差、相关性、均值/方差变化。
6. 最后把每个 batch、每个 stage 的比较结果写入 CSV，并把候选前端观测到的 scale、min/max、饱和率等写入 JSON。

这一步比较的不是模型分类准确率，也不会训练 backbone。它的作用是提前回答：

- 输入 S8 PCM 会不会马上造成严重失真？
- Mel 能量在量化后是否仍然稳定？
- PWL log 会不会把低能量区域的误差放大？
- DCT/MFCC 输出分布有没有明显偏移？
- 目前的 scale 是否导致大量饱和？

注意：这里的参考前端不是原始 `torchaudio.transforms.MFCC`，而是“矩形 Mel + PWL log + 高精度中间计算”的参考路径。这样做是为了隔离当前实验里的量化约束影响，而不是把 torchaudio、三角 Mel、exact log 的差异混在一起。

命令参数含义：

- `--datasets`：一次分析哪些数据集。这里同时跑 `hi_xiaowen` 和 `nihao_wenwen`。
- `--dataset`：只分析单个数据集时使用。如果已经用了 `--datasets`，通常不用再写。
- `--constraint_profile`：候选前端启用哪组量化约束。
- `upper_bound_s8_mfcc`：第一阶段上限配置，启用 S8 PCM 输入和 S8 MFCC 输出，中间大部分 stage 保持高精度。
- `frontend_fakequant`：进一步启用 PWL 输入、log_mel、DCT、MFCC 等 stage 的 fake quant。
- `wide_bit_accurate`：与 `frontend_fakequant` 类似，用于保留较宽位宽的 bit-accurate 训练版本。
- `hardware_baseline`：打开所有当前 baseline stage quant，是最接近完整硬件约束、也最容易掉点的配置。
- `--output_dir`：输出目录。建议每一次实验都用不同目录，避免覆盖旧结果。
- `--batches`：每个数据集分析多少个 batch。`8` 表示每个数据集取 8 个 batch。设为 `0` 表示跑完整个 split。
- `--batch`：每个 batch 的样本数，默认 `128`。
- `--split`：用哪个数据 split，默认 `validation`。可选 `train`、`validation`、`test`。
- `--noise_roots`：噪声列表或噪声目录，默认使用 TAU valid 噪声列表。
- `--noise_prob`：加噪概率，默认 `1.0`。
- `--snr_min_db` / `--snr_max_db`：加噪 SNR 范围。默认覆盖 `-5 dB` 到 `20 dB`。
- `--gpu`：是否使用 GPU。`0` 表示 CPU，`1` 表示如果有 CUDA 就用 GPU。
- `--sample_rate`：采样率，默认 `16000`。
- `--window_size_ms` / `--window_stride_ms`：MFCC 帧长和帧移，默认都是 `32 ms`。
- `--pre_emphasis` / `--no-pre_emphasis`：是否启用预加重，默认启用。
- `--pre_emphasis_coeff`：预加重系数，默认 `0.97`。
- `--log_pwl_num_segments`：PWL log 分段数，默认 `8` 段。
- `--log_pwl_strategy`：PWL log 断点策略，默认 `uniform_logx`。
- `--log_pwl_gamma`：PWL 拟合时的 powerlaw 参数，默认 `1.0`。
- `--log_offset`：Mel 能量进 log 前加的 offset，默认 `1e-6`。
- `--log_input_clamp_min`：log 输入下限，默认 `1e-12`。

CSV 里每个 stage 代表什么：

- `waveform`：原始浮点语音输入。
- `pcm_or_scaled_sample`：前端入口处的 S8 PCM 量化/反量化结果，是检查输入量化损失的关键 stage。
- `preemphasis_or_sample_scale`：预加重后的语音。
- `fft_complex`：STFT/FFT 的复数频域输出。
- `power`：功率谱，等于实部平方加虚部平方。
- `rectangular_mel`：方波/矩形 Mel 滤波后的 Mel 能量。
- `pwl_log_mel`：PWL log 之后的 log-Mel 特征。
- `dct_acc`：DCT 后的 MFCC 累加结果。
- `mfcc_prequant`：最终 MFCC 输出量化前的结果。当前实现里通常和 `dct_acc` 一致。
- `mfcc_int8`：S8 MFCC 输出量化/反量化后的结果，是 backbone 实际看到的前端输出。

CSV 里每一列代表什么：

- `dataset`：数据集名称。
- `stage`：当前比较的是前端哪个阶段。
- `status`：比较状态。`ok` 表示形状一致并完成比较；`shape_mismatch` 表示参考和候选输出形状不同。
- `numel`：参与比较的元素总数。
- `mae`：平均绝对误差，越小越好。
- `rmse`：均方根误差，比 `mae` 更敏感于少数大误差，越小越好。
- `max_abs_error`：最大绝对误差，用来发现极端异常点。
- `mean_error`：候选输出减参考输出的平均值。长期偏正或偏负说明有系统性偏移。
- `reference_mean`：参考前端该 stage 输出均值。
- `candidate_mean`：候选前端该 stage 输出均值。
- `reference_std`：参考前端该 stage 输出标准差。
- `candidate_std`：候选前端该 stage 输出标准差。
- `cosine`：余弦相似度，越接近 `1` 越好。它更关注整体方向/形状是否一致。
- `pearson`：Pearson 相关系数，越接近 `1` 越好。它更关注线性相关性。

怎么读 CSV：

- 先看 `status`，必须全是 `ok`。
- 再按 stage 顺序看误差在哪里突然变大。
- 如果 `pcm_or_scaled_sample` 就很差，说明 S8 PCM 输入约束已经有问题。
- 如果 `rectangular_mel` 之前都很好，但 `pwl_log_mel` 开始变差，说明 log 对低能量误差有放大，优先看 PWL log 和 log 输入范围。
- 如果 `pwl_log_mel` 很好，但 `dct_acc` 或 `mfcc_int8` 才变差，优先看 DCT 系数、DCT 累加、MFCC 输出 scale。
- `candidate_std / reference_std` 如果明显小于 `1`，说明候选特征被压缩；如果明显大于 `1`，说明量化或 log 可能放大了动态范围。

JSON 文件怎么看：

- `<dataset>_upper_bound_s8_mfcc_stage_analysis_spec.json`：保存候选前端配置和观测到的 scale。
- 里面的 `config.stage_quant.pcm` 可以确认输入是不是 S8、是否启用、scale 是多少。
- 里面的 `config.stage_quant.mfcc` 可以确认 MFCC 输出是不是 S8、是否逐系数 scale。
- 里面的 `observed_absmax` 是每个量化 stage 观察到的最大绝对值，用来推导 scale。
- `<dataset>_upper_bound_s8_mfcc_stage_stats.json`：保存每个 stage 的统计量。
- `min/max`：当前 stage 输出范围。
- `mean/std`：当前 stage 分布均值和标准差。
- `zero_ratio`：输出等于 0 的比例。比例异常高时，可能出现过度裁剪或量化过粗。
- `saturation_ratio`：落在量化上下限的比例。只会出现在启用了量化的 stage，比如 `pcm` 和 `mfcc`。
- `saturation_ratio` 很小通常没问题；如果达到百分之几甚至更高，就要优先检查 scale。

怎么看输出：

- 先打开 `upper_bound_s8_mfcc_feature_alignment_summary.csv`。
- 每一行是一个 stage，例如 `pcm_or_scaled_sample`、`rectangular_mel`、`pwl_log_mel`、`mfcc_int8`。
- `status` 必须是 `ok`。如果出现 `shape_mismatch`，说明前端形状契约就错了，不能进入 QAT。
- 重点看 `mfcc_int8`、`pwl_log_mel`、`rectangular_mel`。
- `cosine` 越接近 1 越好，`mae`、`rmse`、`max_abs_error` 越小越好。
- 再打开 `<dataset>_upper_bound_s8_mfcc_stage_stats.json`，看各阶段的 `min/max/mean/std/zero_ratio/saturation_ratio`。
- 如果 `pcm` 或 `mfcc` 的 `saturation_ratio` 很大，优先检查输入或输出 scale，而不是直接训练。

什么时候进入下一步：

- 没有 `shape_mismatch`。
- `mfcc_int8` 的分布没有明显离谱，例如均值/标准差和参考相比不是数量级差异。
- `pcm` 输入没有大量饱和。
- 如果这一步已经明显异常，先停下来调前端 scale/PWL/log 配置，不要跑长 QAT。

### 第 2 步：跑一个极小 QAT smoke test

这一步只验证数据路径、checkpoint、QAT、spec 导出、场景评估全链路能跑通。先用 1 个 checkpoint 和 1 个 epoch。

命令：

```bash
python dscnn_kws/quantization/qat_bit_accurate_mfcc_accuracy_first.py \
  --pattern "*L5_C64*.pt" \
  --limit 1 \
  --qat_epochs 1 \
  --constraint_profile upper_bound_s8_mfcc \
  --output_dir dscnn_kws/quantization/bit_accurate_mfcc_experiments/phase1_upper_bound_smoke_models \
  --train_results_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments/phase1_upper_bound_smoke_train_results.csv \
  --grid_results_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments/phase1_upper_bound_smoke_grid_results.csv
```

输出文件：

```text
dscnn_kws/quantization/bit_accurate_mfcc_experiments/
  phase1_upper_bound_smoke_train_results.csv
  phase1_upper_bound_smoke_grid_results.csv
  phase1_upper_bound_smoke_models/
    <stem>_bit_accurate_mfcc_spec.json
    <stem>_bit_accurate_qat_prepared_best.pt
    <stem>_bit_accurate_qat_int8_backbone.pt
    <stem>_bit_accurate_mfcc_int8_backbone.pt
```

怎么看输出：

- `phase1_upper_bound_smoke_train_results.csv` 应该至少有 1 行。
- 这一行里的 `bit_accurate_mfcc_spec_json` 应该指向一个真实存在的 JSON。
- `fakequant_qat_test_tau_list_acc` 是 fake-quant 前端加训练中 backbone 的测试集结果。
- `quantized_backbone_fakequant_frontend_test_tau_list_acc` 是 fake-quant 前端加 INT8 backbone 的结果。
- `bit_accurate_mfcc_int8_backbone_test_tau_list_acc` 是 hard bit-accurate 前端加 INT8 backbone 的最终推理结果。
- smoke test 只看能不能跑通，不用纠结 1 epoch 的准确率。

什么时候进入下一步：

- 脚本没有报错。
- train CSV 和 grid CSV 都生成了。
- spec JSON 生成了。
- `bit_accurate_mfcc_int8_backbone_test_tau_list_acc` 不是空值或 NaN。

#### 一个问题：“这说明当前 bit-accurate 前端的训练/导出/测试链路本身没有明显问题”是什么意思

这里的意思可以拆成一句话：

**QAT 训练时“模拟出来的硬件前端”，和测试时“真正按 bit-accurate 规则跑出来的前端”，在模型最终识别结果上几乎没有差异。**

也就是说，我们现在不是只看准确率高不高，而是在检查这条链路有没有“训练时看到的是 A，部署时实际变成 B”的问题。

这次冒烟测试里有三种测试结果：

```text
fakequant_qat_tau_list_acc           = 0.9609
int8_backbone_fakequant_frontend_acc = 0.9610
bit_accurate_mfcc_int8_backbone_acc  = 0.9610
```

可以理解成三个阶段：

**1. fakequant 前端 + QAT 模型**

这是训练时的视角。

前端还是 PyTorch float Tensor 在跑，但里面人为加入了：

```text
round / clamp / saturate / scale / int8 输出约束 / PWL log 近似
```

也就是说，它不是完全真实整数计算，但会模拟硬件量化误差，让 backbone 在训练时提前适应这些失真。

结果是：

```text
0.9609
```

**2. fakequant 前端 + INT8 backbone**

这一步把 backbone 真正换成 INT8 推理形式，但前端仍然用 fakequant 模拟。

它主要检查：

```text
backbone 从 QAT 形态转换成 INT8 之后，有没有额外掉点
```

结果是：

```text
0.9610
```

和第一步几乎一样，所以说明 backbone 的 INT8 转换没出明显问题。

**3. hard bit-accurate 前端 + INT8 backbone**

这是最接近目标硬件部署的形式。

前端不再只是 fakequant 模拟，而是尽量按照设计里的 bit-accurate 规则跑，例如：

```text
输入 S8 PCM
中间定点位宽
饱和
右移
PWL log
最终 S8 MFCC 输出
```

然后接 INT8 backbone。

结果还是：

```text
0.9610
```

这就很关键了。

它说明：**训练时用的 fakequant 前端，和最终 hard bit-accurate 前端，在模型看来基本是同一个分布。**

换句话说，模型训练时适应到的量化误差，测试时真的出现了，而且没有突然变成另一种误差。

所以我说“链路本身没有明显问题”，具体指的是这些风险目前没有暴露出来：

```text
前端输出 scale 不一致
MFCC 输出范围不一致
signed / unsigned 搞反
int8 饱和位置不对
PWL log 和训练时模拟不一致
backbone INT8 转换后精度崩掉
hard frontend 和 fakequant frontend 分布差异很大
shape / layout / channel 顺序不一致
```

如果这些地方有明显 bug，通常会看到类似这种现象：

```text
fakequant_qat_acc                 = 96%
int8_backbone_fakequant_acc       = 95% 或 96%
bit_accurate_mfcc_int8_acc        = 80% / 70% / 甚至更低
```

也就是 hard bit-accurate 一接上去就明显掉点。

但现在三个结果几乎一样，所以说明：

```text
fakequant frontend 训练假设成立
INT8 backbone 转换稳定
hard bit-accurate frontend 没有明显实现偏差
```

不过要注意一点：这不等于“最终准确率已经没问题”。

它只说明 **QAT 链路内部是一致的**。  
从全精度 baseline 到现在的 bit-accurate 约束，仍然有掉点：

```text
相比全精度 baseline 平均约 -1.82 pp
```

这个掉点更可能来自真实约束本身，比如：

```text
输入 PCM 强制 S8
PWL log 近似
MFCC 输出 S8
低 SNR 下 log 对小能量扰动敏感
```

而不是来自脚本接错、scale 错、前后端不匹配这类工程问题。

所以一句话总结就是：

**这次冒烟测试证明“训练时模拟的 bit-accurate 前端”和“测试时真正 hard 的 bit-accurate 前端”已经对上了；下一步要优化的是约束本身带来的准确率损失，而不是先怀疑实现链路。**

### 第 3 步：跑 Phase 1 精度上限实验

这一步是第一轮真正实验。约束仍然是：

```text
S8 PCM input + rectangular Mel + PWL log + high precision intermediate + S8 MFCC output
```

命令：

```bash
python dscnn_kws/quantization/qat_bit_accurate_mfcc_accuracy_first.py \
  --pattern "*L5_C64*.pt" \
  --qat_epochs 10 \
  --constraint_profile upper_bound_s8_mfcc \
  --output_dir dscnn_kws/quantization/bit_accurate_mfcc_experiments/phase1_upper_bound_models \
  --train_results_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments/phase1_upper_bound_train_results.csv \
  --grid_results_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments/phase1_upper_bound_grid_results.csv
```

输出文件：

```text
dscnn_kws/quantization/bit_accurate_mfcc_experiments/
  phase1_upper_bound_train_results.csv
  phase1_upper_bound_grid_results.csv
  phase1_upper_bound_models/
    *_bit_accurate_mfcc_spec.json
    *_bit_accurate_qat_prepared_best.pt
    *_bit_accurate_qat_int8_backbone.pt
    *_bit_accurate_mfcc_int8_backbone.pt
```

怎么看 train CSV：

- `best_valid_acc`：QAT 微调期间最好的验证集准确率。
- `fakequant_qat_test_tau_list_acc`：训练态 fake-quant 前端结果。
- `quantized_backbone_fakequant_frontend_test_tau_list_acc`：fake-quant 前端加 INT8 backbone 结果。
- `bit_accurate_mfcc_int8_backbone_test_tau_list_acc`：最终 hard bit-accurate 前端加 INT8 backbone 结果。
- 如果 `fakequant_qat_test_tau_list_acc` 很好，但 `bit_accurate_mfcc_int8_backbone_test_tau_list_acc` 明显差，说明 fake-quant 和 hard 前端不一致，应先跑第 6 步的一致性检查。

怎么看 grid CSV：

- 每一行是一个 `dataset + arch + scene + snr_db` 单元格。
- `acc` 是最终 hard bit-accurate 前端加 INT8 backbone 的场景噪声准确率。
- 先看低 SNR，例如 `0 dB` 和 `-5 dB`，这里最容易暴露前端量化问题。

和全精度 baseline 对比：

如果 `snr_scene_arch_sweep_grid_results.csv` 在项目根目录，可以直接运行：

```bash
python -c "import pandas as pd; keys=['dataset','arch','scene','snr_db']; fp=pd.read_csv('snr_scene_arch_sweep_grid_results.csv'); bit=pd.read_csv('dscnn_kws/quantization/bit_accurate_mfcc_experiments/phase1_upper_bound_grid_results.csv'); m=fp.merge(bit,on=keys,suffixes=('_fp','_bit')); scale=100 if max(fp['acc'].max(),bit['acc'].max())<=1.5 else 1; m['delta_pp']=(m['acc_bit']-m['acc_fp'])*scale; print(m['delta_pp'].describe()); print(m.groupby(['dataset','arch'])['delta_pp'].mean())"
```

怎么看这条对比命令：

- `delta_pp` 是 bit-accurate 相对全精度的准确率变化，单位是百分点。
- 平均值接近 0 最好。
- 如果平均 `delta_pp >= -1.0`，说明阶段 1 精度上限基本通过。
- 如果平均掉点超过 1 个百分点，先不要进入更窄位宽实验，应回到第 1 步看哪个 stage 漂移最大。

什么时候进入下一步：

- 平均掉点不超过 1 个百分点。
- 没有大面积低 SNR 崩溃。
- fake-quant 前端结果和 hard bit-accurate 前端结果差距不大，建议平均差距不超过 0.5 个百分点。

#### 一个问题：现在QAT阶段的前端和网络都是什么样的

现在 QAT 阶段的模型可以理解成这个结构：

```text
原始 waveform
   ↓
BitAccurateMFCCFakeQuantFrontend
   ↓
取前 10 维 MFCC
   ↓
transpose + flatten
   ↓
QuantStub
   ↓
DSCNN backbone，带 QAT fake quant
   ↓
DeQuantStub
   ↓
logits
```

也就是：

```text
QAT 阶段 = fakequant bit-accurate MFCC 前端 + QAT DSCNN backbone
```

前端这部分现在是：

```python
BitAccurateMFCCFakeQuantFrontend
```

它不是 torchaudio MFCC，也不是之前那个粗糙的 `bit_accuracy` 前端。它的设计约束是：

```text
输入 waveform 先模拟 signed INT8 PCM
Mel 滤波使用 rectangular / 方波 Mel
log 使用 PWL log，不用 exact log
最终 MFCC 输出模拟 signed INT8
训练时量化使用 STE
```

对于你现在冒烟测试用的配置：

```text
--constraint_profile upper_bound_s8_mfcc
```

实际启用的硬约束主要是：

```text
pcm  : enabled, signed int8
mfcc : enabled, signed int8, 默认 per-channel scale
```

中间的这些阶段目前在 `upper_bound_s8_mfcc` 下大多还是高精度：

```text
preemphasis : 暂不强制量化
windowed    : 暂不强制量化
power       : 暂不强制量化
mel         : 暂不强制量化
pwl_input   : 暂不强制量化
log_mel     : 暂不强制量化
dct         : 暂不强制量化
```

所以这个阶段的意图是：**先固定输入 S8、输出 S8、方波 Mel、PWL log，看 bit-accurate 前端的精度上限。**

backbone 这部分是：

```python
DSCNN backbone
```

它来自原来的全精度 checkpoint，只加载 backbone 权重。然后脚本会做：

```text
1. fuse backbone 里的 Conv / BN / ReLU
2. 用 QuantizedBackboneWrapper 包起来
3. 在 backbone 前后加 QuantStub / DeQuantStub
4. 给 backbone 设置 get_default_qat_qconfig(fbgemm)
5. prepare_qat
```

所以训练时 backbone 内部是 PyTorch QAT 形式：

```text
权重 fake quant
activation fake quant
Conv/BN/ReLU 融合后的 QAT module
```

但是注意：**前端没有交给 PyTorch eager quantization 去 convert。**

也就是说，前端不是 PyTorch 的 INT8 kernel，而是我们自己写的：

```text
round / clamp / scale / STE / PWL log / stage quant
```

训练完成后会有三种模型视角：

```text
1. fakequant_qat
   BitAccurateMFCCFakeQuantFrontend + QAT backbone

2. int8_backbone_fakequant_frontend
   BitAccurateMFCCFakeQuantFrontend + converted INT8 backbone

3. bit_accurate_mfcc_int8_backbone
   BitAccurateMFCCFrontend hard frontend + converted INT8 backbone
```

其中第 3 个就是最接近最终部署目标的形式：

```text
hard bit-accurate MFCC 前端 + INT8 DSCNN backbone
```

一句话概括：

**现在 QAT 阶段的前端是带 STE 的 bit-accurate fakequant MFCC，网络是从全精度 checkpoint 加载来的 DSCNN backbone，并经过 PyTorch QAT 准备成 INT8 backbone。前端负责让模型看到硬件 MFCC 失真，backbone 负责适应 INT8 网络量化。**

#### 又一个问题：BitAccurateMFCCFakeQuantFrontendz在现阶段和BitAccurateMFCCFrontend hard frontend是一致的对吧

是的，**在你当前这个阶段，二者在前向行为上应该是高度一致的**，但要加一个很重要的限定：

```text
训练阶段：BitAccurateMFCCFakeQuantFrontend 用 STE
推理/测试阶段：BitAccurateMFCCFrontend hard frontend 不用 STE
```

也就是说，**它们的前向数值目标是一致的，反向传播行为不一样。**

更具体地说：

```text
BitAccurateMFCCFakeQuantFrontend
```

用于 QAT 训练。它在 forward 里也会做：

```text
round / clamp / scale / PWL log / S8 PCM / S8 MFCC
```

但训练时用 STE：

```python
return x + (y - x).detach()
```

含义是：

```text
前向看到的是量化后的 y
反向梯度近似当成从 x 直接传过来
```

这样模型可以训练。

而：

```text
BitAccurateMFCCFrontend
```

用于最终 hard 推理。它不需要反向传播，所以直接返回 hard quant/dequant 后的结果：

```text
return y
```

所以二者关系是：

```text
训练时 fakequant frontend ≈ 部署时 hard frontend
```

但不是“代码完全一样”，而是：

```text
同一套配置
同一套 scale/spec
同一套矩形 Mel
同一套 PWL log
同一套输入 S8 / 输出 S8 约束
不同的梯度处理方式
```

这也是为什么冒烟测试里这三个数几乎一样：

```text
fakequant_qat_tau_list_acc              = 0.9609
int8_backbone_fakequant_frontend_acc    = 0.9610
bit_accurate_mfcc_int8_backbone_acc     = 0.9610
```

它说明：**QAT 阶段用来训练的 fakequant 前端，和校准导出的 hard frontend，在模型最终预测上基本对齐了。**

不过要注意一个小细节：在训练过程中，`BitAccurateMFCCFakeQuantFrontend` 的 scale 可能还在通过 observer 更新；而 `BitAccurateMFCCFrontend hard frontend` 是训练后校准并导出 spec 之后固定下来的。

所以更准确的说法是：

```text
QAT 训练时：
  FakeQuantFrontend 是可训练友好的近似 hard frontend

校准导出后：
  FakeQuantFrontend 和 HardFrontend 应该在前向数值上基本一致

最终部署/测试时：
  使用 HardFrontend + INT8 backbone
```

一句话总结：

**是的，现阶段它们设计目标上是一致的；区别只在于 fakequant 版本训练时用 STE 保梯度，hard 版本推理时直接执行固定的 bit-accurate 前向。**

### 第 4 步：跑 Phase 2 前端感知 QAT

如果 Phase 1 通过，再打开更多前端伪量化约束：

```text
S8 PCM input + PWL input quant + log_mel quant + DCT quant + S8 MFCC output
```

命令：

```bash
python dscnn_kws/quantization/qat_bit_accurate_mfcc_accuracy_first.py \
  --pattern "*L5_C64*.pt" \
  --qat_epochs 10 \
  --constraint_profile frontend_fakequant \
  --output_dir dscnn_kws/quantization/bit_accurate_mfcc_experiments/phase2_frontend_fakequant_models \
  --train_results_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments/phase2_frontend_fakequant_train_results.csv \
  --grid_results_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments/phase2_frontend_fakequant_grid_results.csv
```

怎么看输出：

- 看法和第 3 步完全相同。
- 重点比较 `phase2_frontend_fakequant_grid_results.csv` 和 `phase1_upper_bound_grid_results.csv`。
- 如果 Phase 2 比 Phase 1 明显掉点，说明新打开的 `pwl_input/log_mel/dct` 约束太强，应回到 stage 分析检查这些 stage 的误差。

什么时候进入下一步：

- Phase 2 相对全精度平均掉点仍不超过 1 个百分点，或者虽然略超但你能明确定位原因。
- Phase 2 相对 Phase 1 没有突然大幅掉点。
- hard bit-accurate 前端和 fake-quant 前端之间没有明显断层。

### 第 5 步：生成并运行逐步收紧约束 sweep

这一步的目的不是再验证一个单独模型，而是把前端约束从“比较宽松”逐步推到“更接近硬件”，观察每次新增约束会额外带来多少掉点。

可以把它理解成一组回归实验：

```text
Phase 1 upper_bound_s8_mfcc
  只验证：S8 PCM 输入 + 方波 Mel + PWL log + S8 MFCC 输出

Phase 2 frontend_fakequant / wide_bit_accurate
  在 Phase 1 基础上，额外打开 PWL 输入、log_mel、DCT 等中间 stage 的伪量化

Phase 3 hardware_baseline
  在当前代码支持的范围内，打开所有 baseline stage quant，观察更完整硬件约束下的精度
```

所以第 5 步真正要回答的是：

```text
继续把前端中间算子定点化之后，准确率从 Phase 1/Phase 2 掉了多少？
掉点主要是否集中在低 SNR？
fake-quant 前端和 hard bit-accurate 前端是否仍然一致？
```

#### 第 5 步会跑哪些 profile

`sweep_bit_accurate_mfcc_constraints.py` 默认会生成 4 组实验命令：

```text
upper_bound_s8_mfcc
frontend_fakequant
wide_bit_accurate
hardware_baseline
```

各 profile 当前含义如下：

| profile | 主要作用 | 当前实际启用的前端量化点 |
|---|---|---|
| `upper_bound_s8_mfcc` | 精度上限基线 | `pcm`、`mfcc` |
| `frontend_fakequant` | 前端中间宽位宽伪量化 | `pcm`、`pwl_input`、`log_mel`、`dct`、`mfcc` |
| `wide_bit_accurate` | 目前与 `frontend_fakequant` 基本等价 | `pcm`、`pwl_input`、`log_mel`、`dct`、`mfcc` |
| `hardware_baseline` | 打开当前代码支持的所有 baseline stage quant | `pcm`、`preemphasis`、`power`、`mel`、`pwl_input`、`log_mel`、`dct`、`mfcc` |
| `hardware_coeff_baseline` | 进一步补齐系数量化参考路径 | `pcm`、`preemphasis`、`hann_coeff`、`windowed`、`twiddle_coeff`、`fft_data`、`power`、`mel`、`pwl_input`、`log_mel`、`dct_coeff`、`dct`、`mfcc` |

注意：当前代码里 Hann window 仍然是通过 `torch.stft(..., window=torch.hann_window(...))` 在 STFT 内部完成的，所以这一步即使跑 `hardware_baseline`，也还没有真正验证：

```text
HANN_COEFF_W = 16 unsigned
FFT_IN_W / windowed = 18 signed
```

也就是说：

```text
Hann 窗：有使用
Hann 系数量化成 U16：当前没有
加窗后样本量化成 S18：当前没有
```

如果后续要验证这两个约束，需要先把前端改成显式 framing + Hann U16 系数量化 + windowed S18 量化 + 显式频域参考实现。否则第 5 步的 `hardware_baseline` 只能代表“当前代码已经接入的硬件 stage quant”，还不是完整硬件 MFCC 前端。

补充说明：现在新增的 `hardware_coeff_baseline` 就是为了解决这个缺口。它不再用 `torch.stft` 的内部 float Hann 路径，而是改成：

```text
显式 framing
  -> Hann coeff U16
  -> windowed sample S18
  -> DFT/twiddle coeff S16
  -> FFT/RFFT data S20
  -> power U41
  -> Mel U46
  -> PWL input U32
  -> log_mel S24
  -> DCT coeff S8
  -> DCT acc S40
  -> MFCC out S8
```

这里的频域部分使用显式 DFT/twiddle matrix reference 来验证 `TWIDDLE_W = 16 signed`，不是逐级 radix-2 butterfly 的硬件时序模型。因此它能验证系数量化和接口位宽对模型精度的影响，但还不是最终 RTL 级 FFT 时序等价实现。

#### 先生成 sweep 计划

先生成 sweep 计划，不真正训练：

```bash
python dscnn_kws/quantization/sweep_bit_accurate_mfcc_constraints.py \
  --arch L5_C64 \
  --input_dir /root/kws/dscnn_kws/dscnn_kws/runs/snr_scene_arch_sweep_best_models \
  --pattern "*L5_C64*.pt" \
  --qat_epochs 10 \
  --output_dir dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_coeff_quant/constraint_sweeps
```

输出文件：

```text
dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_coeff_quant/constraint_sweeps/
  constraint_sweep_plan.csv
  constraint_sweep_commands.txt
```

怎么看输出：

- 先打开 `constraint_sweep_plan.csv`，确认每一行的 `constraint_profile` 是否符合预期。
- 再打开 `constraint_sweep_commands.txt`，确认它生成的训练命令没有跑错路径。
- 注意：这里的 `--datasets` 主要用于记录实验计划；实际跑哪些模型由 `--input_dir` 和 `--pattern` 找到的 checkpoint 决定。

`constraint_sweep_plan.csv` 只是一张实验计划表，每一行对应一个 profile。重点看这些列：

- `sweep_id`：这组实验的编号。
- `constraint_profile`：这组实验使用哪套前端约束。
- `output_dir`：这组实验的模型、spec 和结果 CSV 会写到哪里。
- `notes`：这组 profile 的简短说明。

`constraint_sweep_commands.txt` 是脚本自动拼出来的命令列表。你可以先不加 `--run`，只检查这些命令；确认路径正确后，再决定是让 sweep 脚本一次性跑完，还是手动复制其中某一条单独跑。

如果你已经单独跑过 Phase 1 和 Phase 2，不想重复训练，可以不使用 sweep 的 `--run`，而是直接跑 `hardware_baseline`：

```bash
python dscnn_kws/quantization/qat_bit_accurate_mfcc_accuracy_first.py \
  --pattern "*L5_C64*.pt" \
  --qat_epochs 10 \
  --constraint_profile hardware_baseline \
  --output_dir dscnn_kws/quantization/bit_accurate_mfcc_experiments/phase3_hardware_baseline_models \
  --train_results_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments/phase3_hardware_baseline_train_results.csv \
  --grid_results_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments/phase3_hardware_baseline_grid_results.csv
```

如果你想把所有 profile 都放在同一个 sweep 目录里统一管理，就继续使用下面的 `--run` 命令。

#### 新增 v2：补齐 Hann / windowed / twiddle / DCT coeff 量化

如果目标是先补齐这些以前没有真实验证的约束：

```text
HANN_COEFF_W = 16 unsigned
FFT_IN_W / windowed = 18 signed
TWIDDLE_W = 16 signed
DCT_COEFF_W = 8 signed
```

请不要写到旧目录：

```text
dscnn_kws/quantization/bit_accurate_mfcc_experiments
```

而是使用新的 v2 目录：

```text
dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_coeff_quant
```

先跑 stage-level 检查：

```bash
python dscnn_kws/quantization/analyze_bit_accurate_mfcc_stages.py \
  --datasets mobvoi_hi_xiaowen_binary_hardneg mobvoi_nihao_wenwen_binary_hardneg \
  --constraint_profile hardware_coeff_baseline \
  --output_dir dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_coeff_quant/stage_hardware_coeff_baseline \
  --batches 8
```

这一步重点看输出 spec / stats 里是否出现这些 stage：

```text
hann_coeff
windowed
twiddle_coeff
fft_data
dct_coeff
```

然后先跑 1 epoch 冒烟测试：

```bash
python dscnn_kws/quantization/qat_bit_accurate_mfcc_accuracy_first.py \
  --pattern "*L5_C64*.pt" \
  --limit 1 \
  --qat_epochs 1 \
  --constraint_profile hardware_coeff_baseline \
  --output_dir dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_coeff_quant/hardware_coeff_baseline_smoke_models \
  --train_results_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_coeff_quant/hardware_coeff_baseline_smoke_train_results.csv \
  --grid_results_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_coeff_quant/hardware_coeff_baseline_smoke_grid_results.csv
```

如果冒烟测试通过，再跑完整 QAT：

```bash
python dscnn_kws/quantization/qat_bit_accurate_mfcc_accuracy_first.py \
  --pattern "*L5_C64*.pt" \
  --qat_epochs 10 \
  --constraint_profile hardware_coeff_baseline \
  --output_dir dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_coeff_quant/hardware_coeff_baseline_models \
  --train_results_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_coeff_quant/hardware_coeff_baseline_train_results.csv \
  --grid_results_csv dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_coeff_quant/hardware_coeff_baseline_grid_results.csv
```

这个 v2 结果应优先和旧 `hardware_baseline` 比较。如果 v2 掉点很小，说明补齐 Hann / windowed / twiddle / DCT coeff 后仍然稳定；如果 v2 明显掉点，就要优先检查 `windowed`、`fft_data`、`twiddle_coeff`、`dct_coeff` 的 `saturation_ratio` 和量化误差。

#### 真正运行 sweep

确认没问题后，再真正运行：

```bash
python dscnn_kws/quantization/sweep_bit_accurate_mfcc_constraints.py \
  --arch L5_C64 \
  --input_dir /root/kws/dscnn_kws/dscnn_kws/runs/snr_scene_arch_sweep_best_models \
  --pattern "*L5_C64*.pt" \
  --qat_epochs 10 \
  --output_dir dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_coeff_quant/constraint_sweeps \
  --run
```

加上 `--run` 后，脚本会按 `constraint_sweep_commands.txt` 里的顺序依次运行 QAT。每个 profile 会有自己的输出子目录，例如：

```text
dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_coeff_quant/constraint_sweeps/
  upper_bound_s8_mfcc/
    train_results.csv
    grid_results.csv
    *_bit_accurate_mfcc_spec.json
    *_bit_accurate_qat_prepared_best.pt
    *_bit_accurate_qat_int8_backbone.pt
    *_bit_accurate_mfcc_int8_backbone.pt

  frontend_fakequant/
    train_results.csv
    grid_results.csv
    ...

  wide_bit_accurate/
    train_results.csv
    grid_results.csv
    ...

  hardware_baseline/
    train_results.csv
    grid_results.csv
    ...
```

#### 每组输出怎么看

先看每个 profile 的 `train_results.csv`：

- `best_epoch`：验证集最好的是第几轮。
- `best_valid_acc`：QAT 过程中最好的验证集准确率。
- `fakequant_qat_test_tau_list_acc`：训练形态模型在 tau list 上的测试准确率。
- `quantized_backbone_fakequant_frontend_test_tau_list_acc`：INT8 backbone + fakequant 前端的准确率。
- `bit_accurate_mfcc_int8_backbone_test_tau_list_acc`：hard bit-accurate 前端 + INT8 backbone 的准确率。

这里最重要的是最后三列是否接近。如果差距很大，说明 fake-quant 训练前端和 hard bit-accurate 推理前端没有对齐，应该先查 scale、rounding、observer/spec 导出，而不是直接讨论准确率。

然后看每个 profile 的 `grid_results.csv`：

- `acc`：当前场景和 SNR 下的准确率。
- `scene`：噪声场景，例如 `shopping_mall`、`street_traffic`。
- `snr_db`：测试信噪比，通常低 SNR 更容易暴露前端量化问题。
- `constraint_profile`：确认这张表确实来自你想看的 profile。
- `bit_accurate_mfcc_spec_json`：这次 hard 前端使用的 spec。

推荐比较顺序：

```text
hardware_baseline vs frontend_fakequant
hardware_baseline vs upper_bound_s8_mfcc
hardware_baseline vs 全精度 baseline
hardware_baseline vs 旧 INT8 MFCC QAT
```

如果 `hardware_baseline` 只比 `frontend_fakequant` 低很少，例如 0.3 个百分点以内，说明当前硬件 stage 约束基本安全。如果差距明显，例如超过 1 个百分点，就要优先看低 SNR 和 spec 里的 stage 饱和率。

#### 第 5 步重点看什么

第 5 步不是只看一个平均准确率。建议按这个顺序判断：

1. 先看 fake/hard 是否对齐。
2. 再看 `hardware_baseline` 相对 `frontend_fakequant` 的平均掉点。
3. 再看 `-5 dB` 和 `0 dB` 是否明显恶化。
4. 最后打开 `*_bit_accurate_mfcc_spec.json`，看 `preemphasis`、`power`、`mel`、`pwl_input`、`log_mel`、`dct`、`mfcc` 的 `saturation_ratio`。

如果某个 stage 的饱和率明显偏高，或者 `grid_results.csv` 里只有低 SNR 场景明显掉点，通常说明这个 stage 的 scale 或位宽需要单独调整。

什么时候进入下一步：

- 选择一个准确率和硬件约束都可接受的 profile。
- 如果 `hardware_baseline` 掉点太大，就退回 `frontend_fakequant` 或 `wide_bit_accurate`，然后只针对失败 stage 单独调位宽。

### 第 6 步：做 fake-quant 与 hard bit-accurate 前端一致性检查

先从训练结果 CSV 里找到 spec 路径：

```text
bit_accurate_mfcc_spec_json
```

假设这个路径是：

```text
dscnn_kws/quantization/bit_accurate_mfcc_experiments/phase1_upper_bound_models/<stem>_bit_accurate_mfcc_spec.json
```

运行：

```bash
python dscnn_kws/quantization/eval_bit_accurate_mfcc_integer_parity.py \
  --spec dscnn_kws/quantization/bit_accurate_mfcc_experiments/phase1_upper_bound_models/<stem>_bit_accurate_mfcc_spec.json \
  --datasets mobvoi_hi_xiaowen_binary_hardneg mobvoi_nihao_wenwen_binary_hardneg \
  --output_dir dscnn_kws/quantization/bit_accurate_mfcc_experiments/integer_parity_phase1
```

输出文件：

```text
dscnn_kws/quantization/bit_accurate_mfcc_experiments/integer_parity_phase1/
  integer_parity_summary.csv
  integer_parity_summary.json
```

怎么看输出：

- `integer_parity_summary.csv` 是逐 batch 的特征差异。
- `integer_parity_summary.json` 是汇总。
- 重点看 `mae_mean`、`rmse_mean`、`max_abs_error_max`、`cosine_mean`。
- fake-quant 和 hard 前端来自同一份 spec，理想情况下误差应非常小。
- 如果这里误差大，说明训练前端和推理前端没有对齐，优先检查 rounding、scale、observer 导出和 spec 加载。

什么时候可以认为通过：

- `mae_mean` 和 `rmse_mean` 很小。
- `cosine_mean` 接近 1。
- QAT train CSV 里的 fake-quant 测试结果和 hard bit-accurate 测试结果平均差距不超过 0.5 个百分点。

### 第 7 步：确定最终候选 spec

当某一轮实验满足：

- 相对全精度平均掉点不超过 1 个百分点
- 低 SNR 没有大范围崩溃
- fake-quant 与 hard bit-accurate 一致性通过
- stage 统计没有不可解释的大量饱和

就把这一轮的：

```text
*_bit_accurate_mfcc_spec.json
*_grid_results.csv
*_train_results.csv
integer_parity_summary.json
```

作为当前候选结果。后续如果要继续缩位宽，只从这个候选 spec/profile 出发，不要从已经失败的 profile 继续往下缩。

### 一句话决策表

```text
第 1 步 stage 分析失败
=> 先修前端 scale/PWL/log/shape，不跑 QAT

第 2 步 smoke test 失败
=> 先修路径、数据集、checkpoint、环境

第 3 步 Phase 1 掉点 > 1 pp
=> 不进入更窄位宽，回查 S8 PCM、PWL log、S8 MFCC 输出分布

第 3 步 Phase 1 通过
=> 进入 Phase 2，打开更多前端 fake-quant 约束

Phase 2 明显掉点
=> 查 pwl_input/log_mel/dct 哪个 stage 漂移最大

fake-quant 好但 hard bit-accurate 差
=> 跑第 6 步 integer parity，查 fake/hard 实现差异

某个 profile 准确率和 parity 都通过
=> 记录该 spec，作为下一轮缩位宽起点
```

## 默认决策

除非实验结果明确表明应采用其他方案，否则使用以下默认设置。

```text
First model                 = L5_C64
Datasets                    = hi_xiaowen + nihao_wenwen
Mel filter                  = rectangular
Log implementation          = PWL log
Frontend input              = signed INT8 PCM
Training start              = full-precision checkpoint
First-stage frontend        = S8 PCM input + high precision intermediate + S8 MFCC output
First-stage MFCC scale      = per-coefficient scale
Hardware simplification     = test per-tensor scale after upper bound passes
DCT coefficients            = include the same coefficients as the existing model
Primary metric              = average accuracy delta on common grid cells
Secondary metric            = low-SNR and per-scene robustness
```

## 失败诊断

如果阶段 1 失败，按以下顺序排查：

1. 将最终 MFCC S8 分布与浮点 MFCC 分布进行比较。
2. 检查逐张量输出缩放是否裁剪了重要系数。
3. 检查 C0 和 log 能量处理。
4. 将矩形 Mel 高精度输出与之前的矩形 Mel 扫描结果进行比较。
5. 检查 PWL log 的断点、斜率、截距、输入裁剪和输出缩放是否导致特征均值变化过大。
6. 首先冻结前端进行训练；如有必要，再解冻前端或仅调节缩放参数。

如果阶段 2 通过但阶段 4 失败，则问题不在模型容量，而在伪量化与整数推理之间存在实现不匹配。应优先检查舍入、移位顺序、饱和位置以及查表/PWL 近似。

如果逐步缩减约束失败，只回退最后修改的阶段，并保留上一个已接受配置作为硬件候选方案。

## 最终预期输出

是的，组合使用方案 1 和方案 3 的流程，最终应生成类似当前基线位宽列表的表格。不同之处在于，最终表格中的配置将由实验结果支撑：

```text
PCM_W              = 8 signed
MFCC_SAMPLE_W      = 12 signed
HANN_COEFF_W       = 16 unsigned
FFT_IN_W           = 18 signed
FFT_DATA_W         = 20 signed
TWIDDLE_W          = 16 signed
POWER_W            = 41 unsigned
MEL_ACC_W          = 46 unsigned
PWL_IN_W           = 32 unsigned
LOG_W              = 24 signed
DCT_COEFF_W        = 8 signed
DCT_ACC_W          = 40 signed
MFCC_OUT_W         = 8 signed
```

最终接受的版本可能继续使用这些数值，也可能根据扫描结果选择更宽或更窄的位宽。关键在于，每个数值都必须由以下依据支撑：

- 阶段级特征对齐
- 饱和率和动态范围统计
- QAT 准确率
- 整数推理一致性








# 一个问题：
##
之前按dscnn_kws\quantization\README_MFCC_BIT_ACCURATE_QAT_DESIGN_ZH.md这个方案进行的事情，是不是就是基于我给出的baseline，即dscnn_kws\quantization\README_MFCC_BitAccuracy_BASELINE.md，来一步步的增加约束直至和baseline中的位宽设置一致，来逐步观察最终量化精度的变化，并且对比更换前端/网络前后的结果来判定是否有链路问题，从而判定最终量化的结果可不可信，但是这个方案都是基于baseline来的，所以位宽都只是通过计算得来的宽松配置，保证数值不会爆，要想部署到硬件上，还需进一步的优化bit-accurate模型各个stage的位宽、scale等，也就是通过dscnn_kws\quantization\README_MFCC_BIT_ACCURATE_BITWIDTH_SWEEP_ZH.md中的方案来进行
##

对，你这个理解基本完全正确，而且总结得很到位。

可以更精确地拆成两层：

**第一层：QAT design 方案做的是“链路可信性验证”**

`README_MFCC_BIT_ACCURATE_QAT_DESIGN_ZH.md` 这套方案，本质上是：

```text
以 README_MFCC_BitAccuracy_BASELINE.md 里的位宽表为硬件参考 baseline
-> 从较宽松/较少约束开始
-> 逐步加入 INT8 PCM、矩形 Mel、PWL log、INT8 MFCC 输出
-> 再加入中间 stage 量化
-> 再加入 Hann / windowed / twiddle / FFT data / DCT coeff 等约束
-> 观察每一步的精度变化
-> 对比 fakequant frontend、hard bit-accurate frontend、INT8 backbone 的结果是否对齐
```

它主要回答的是：

```text
这条 bit-accurate MFCC + INT8 backbone 的训练/导出/测试链路是不是可信？
```

也就是：

```text
前端换了以后是否还能训练？
fakequant 前端和 hard 前端是否一致？
导出的 spec 是否能复现训练时的行为？
backbone INT8 后是否和训练期结果一致？
精度下降是前端数值约束导致的，还是链路实现有 bug？
```

所以这一步更像是 **验证框架和实验链路**，不是最终硬件最优位宽搜索。

**第二层：baseline 位宽本身不是最终最优硬件方案**

`README_MFCC_BitAccuracy_BASELINE.md` 里的那些位宽，比如：

```text
POWER_W   = 41
MEL_ACC_W = 46
DCT_ACC_W = 40
```

它们更像是通过数值范围推导出来的 **安全上界/宽松配置**。

它们的特点是：

```text
优点：不容易溢出，数值安全，适合先验证链路
缺点：不一定硬件友好，可能浪费 bit、寄存器、BRAM、DSP/逻辑资源
```

所以它不能直接等价于：

```text
最终 FPGA/ASIC 部署最优位宽
```

它只能说明：

```text
在这套安全位宽下，链路能不能跑通，精度能不能保持，fake/hard 是否对齐。
```

**第三层：bit-width sweep 才是“硬件部署优化”**

`README_MFCC_BIT_ACCURATE_BITWIDTH_SWEEP_ZH.md` 这套方案做的是下一步：

```text
以 hardware_coeff_baseline 为可信起点
-> 对每个 stage 做位宽压缩
-> 先用 feature screening 看数值误差和传播损失
-> 再挑选不超过 50 个候选做 full QAT
-> 找每个 stage 的最低安全位宽
-> 组合成 accuracy_first 和 aggressive 两套策略
-> 再跑组合 QAT/grid 验证
```

它主要回答的是：

```text
在保证精度基本不掉的情况下，每个 stage 到底可以压到多少 bit？
哪些 stage 敏感？
哪些 stage 可以大胆压？
最终硬件部署应该选哪套位宽？
```

所以你可以把两份文档的关系理解成：

```text
README_MFCC_BIT_ACCURATE_QAT_DESIGN_ZH.md
= 先证明 bit-accurate QAT 链路可信

README_MFCC_BIT_ACCURATE_BITWIDTH_SWEEP_ZH.md
= 在可信链路上做硬件位宽优化
```

一句话概括就是：

```text
前一个方案解决“这条量化链路可信吗”；
后一个方案解决“可信之后，怎样把它压到更适合硬件部署”。
```

这也是很正规的工程节奏。先不要一上来追最小位宽，而是先用宽松、安全的 baseline 建立可信链路；链路可信后，再做 bit-width / scale / saturation / rounding 的逐级优化。