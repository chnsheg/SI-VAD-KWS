# DSCNN-KWS 测试指标说明

本文档说明本仓库中模型训练、测试、结构 sweep、噪声评估、量化评估和 FAH/FRR 评估里各项指标的计算方式与含义。

当前主任务配置在 `dscnn_kws/configs.py` 中是二分类：

```text
0: positive
1: negative
```

普通分类指标使用 `argmax(logits)` 得到预测类别；FAH/FRR 指标使用 `softmax(logits)[:, positive_index]` 作为唤醒词置信度，再按阈值判断是否触发。

## 1. 指标来源

主要实现位置如下：

```text
dscnn_kws/engine/trainer.py
  训练、验证、最终测试的 loss / acc / precision / recall / f1

dscnn_kws/quantization/quantize_sweep_best_models.py
  Q16.16 量化后 clean / noise / scene-SNR 评估的 loss / acc / precision / recall / f1 / num_samples

dscnn_kws/eval_fah_frr.py
  指定目标 FAH 时的 threshold / FAH / FRR / ACC / TP / FN / FP / TN

dscnn_kws/sweep_dscnn_acc.py
dscnn_kws/sweep_dscnn_noise_acc.py
dscnn_kws/sweep_fixed_dscnn_noise_snr_scene_acc.py
  批量训练与汇总 CSV 字段，例如 best_valid_acc、test_acc、mean_acc、min_acc、max_acc

dscnn_kws/nas/evaluator.py
dscnn_kws/nas/retrain_topk.py
dscnn_kws/nas/constraints.py
  NAS 搜索/重训中的 acc、loss、mults、params
```

## 2. 基础分类流程

对每个 batch：

```python
logits = model(waveform)
preds = torch.argmax(logits, dim=1)
```

也就是说，常规测试不使用手工阈值，而是直接选择 `logits` 最大的类别。二分类时：

- `preds == 0` 表示预测为 `positive`
- `preds == 1` 表示预测为 `negative`

`logits` 是模型输出的未归一化分数。普通分类指标不需要先做 `softmax`，因为 `argmax(logits)` 和 `argmax(softmax(logits))` 的结果相同。

## 3. loss

### 计算方式

训练、验证和普通测试使用 `torch.nn.CrossEntropyLoss`。

在 `Trainer` 中：

```text
batch_loss = CrossEntropyLoss(logits, labels)
loss = 所有 batch_loss 的平均值
```

代码中是：

```text
total_loss += loss.item()
reported_loss = total_loss / len(loader)
```

注意：这里是“按 batch 平均”，不是先把每个样本的 loss 全部累加再除以总样本数。由于 `CrossEntropyLoss` 默认会先对一个 batch 内样本求平均，所以当每个 batch 大小一致时，两者几乎等价；如果最后一个 batch 较小，它和其它 batch 仍然占相同权重，会产生很小差异。

### label smoothing 的影响

在 `dscnn_kws/engine/trainer.py` 中，训练、验证和最终测试共用同一个 criterion：

```text
CrossEntropyLoss(label_smoothing=args.label_smoothing)
```

并且 `label_smoothing` 会被限制在 `[0.0, 0.2]`。

因此如果训练时指定了 `--label_smoothing`，最终 `[TEST] loss=...` 也是带 label smoothing 的 loss。量化脚本 `quantize_sweep_best_models.py` 中的评估则使用默认 `CrossEntropyLoss()`，不带 label smoothing。

### 含义

`loss` 衡量模型输出分布与真实标签之间的差距，越低通常越好。它不仅关心预测是否正确，也关心模型对正确类别的置信度。

常见解读：

- `acc` 高但 `loss` 高：多数样本预测对了，但模型置信度可能不稳定，或者少数错例非常自信。
- `loss` 低但 `acc` 提升不明显：模型概率校准有所改善，但类别边界未明显改变。
- 不同 label smoothing 设置下的 `loss` 不宜直接横向比较。

## 4. acc / test_acc / valid_acc

### 计算方式

```text
correct = 预测类别等于真实标签的样本数
total = 样本总数
acc = correct / total
```

公式：

```text
accuracy = (TP + TN) / (TP + TN + FP + FN)
```

在二分类里：

- TP：真实 `positive` 且预测 `positive`
- FN：真实 `positive` 但预测 `negative`
- FP：真实 `negative` 但预测 `positive`
- TN：真实 `negative` 且预测 `negative`

### 字段名对应关系

不同脚本里同一个指标可能有不同字段名：

```text
acc
  通用准确率字段，训练器、量化评估、scene-SNR grid 中常见

test_acc
  sweep_dscnn_acc.py / sweep_dscnn_noise_acc.py 从 [TEST] 日志解析出的测试集准确率

test_acc_on_tau_test_list
  sweep_fixed_dscnn_noise_snr_scene_acc.py 中，使用 TAU test 噪声列表时的测试准确率

valid_acc
  单个 epoch 的验证准确率，通常只打印在训练日志中

best_valid_acc
  训练过程中所有 epoch 里最高的验证准确率
```

### 含义

`acc` 是最直观的整体正确率，越高越好。但在正负样本比例不均衡时，单看 `acc` 可能有误导性。例如负样本很多时，模型即使很少触发 `positive`，也可能获得很高准确率，但唤醒词漏检会很严重。

因此：

- 平衡测试集上，`acc` 可以作为主指标之一。
- 非平衡测试集上，应同时看 `precision`、`recall`、`f1`，唤醒任务还应看 `FAH/FRR`。

## 5. precision

### 计算方式

本仓库使用 sklearn：

```python
precision_score(all_labels, all_preds, average="macro", zero_division=0)
```

对每个类别分别计算：

```text
precision_c = TP_c / (TP_c + FP_c)
```

然后做宏平均：

```text
macro_precision = 所有类别 precision_c 的算术平均
```

二分类时：

```text
macro_precision = (precision_positive + precision_negative) / 2
```

如果某个类别从未被预测出来，导致 `TP_c + FP_c = 0`，`zero_division=0` 会把该类别 precision 记为 0。

### 含义

`precision` 表示“模型预测为某类的样本里，有多少是真的该类”。对于 `positive` 类：

```text
positive precision = 预测触发唤醒的样本中，真正是唤醒词的比例
```

它主要反映误触发风险。`positive precision` 越低，说明把负样本误判成唤醒词的比例越高。

由于仓库输出的是 `macro precision`，它同时平均了 `positive` 和 `negative` 两个类别的 precision，不是单独的正类 precision。

## 6. recall

### 计算方式

本仓库使用 sklearn：

```python
recall_score(all_labels, all_preds, average="macro", zero_division=0)
```

对每个类别分别计算：

```text
recall_c = TP_c / (TP_c + FN_c)
```

然后做宏平均：

```text
macro_recall = 所有类别 recall_c 的算术平均
```

二分类时：

```text
macro_recall = (recall_positive + recall_negative) / 2
```

其中：

```text
recall_positive = TP / (TP + FN)
recall_negative = TN / (TN + FP)
```

### 含义

`recall` 表示“真实属于某类的样本里，有多少被模型找出来”。对于 `positive` 类：

```text
positive recall = 真实唤醒词样本中，被正确触发的比例
```

它主要反映漏唤醒风险。`positive recall` 越低，说明真实唤醒词越容易被漏掉。

仓库输出的是 `macro recall`，不是单独的正类 recall。二分类场景下，`macro recall` 等于正类召回率和负类召回率的平均。

## 7. f1

### 计算方式

本仓库使用 sklearn：

```python
f1_score(all_labels, all_preds, average="macro", zero_division=0)
```

对每个类别分别计算：

```text
f1_c = 2 * precision_c * recall_c / (precision_c + recall_c)
```

然后做宏平均：

```text
macro_f1 = 所有类别 f1_c 的算术平均
```

二分类时：

```text
macro_f1 = (f1_positive + f1_negative) / 2
```

### 含义

`f1` 是 precision 和 recall 的调和平均，适合观察误报和漏报之间的折中。它比 `acc` 更能暴露类别不均衡带来的问题。

常见解读：

- `precision` 高、`recall` 低：模型触发很谨慎，误触发少，但漏掉较多唤醒词。
- `precision` 低、`recall` 高：模型容易触发，唤醒词找得多，但误触发也多。
- `f1` 高：precision 和 recall 的综合表现较均衡。

## 8. 混淆矩阵计数

FAH/FRR 脚本会显式输出：

```text
TP, FN, FP, TN
```

普通分类测试脚本没有直接打印混淆矩阵，但 `acc`、`precision`、`recall`、`f1` 都是由预测标签和真实标签间接计算出来的。

二分类定义如下：

| 真实标签 | 预测标签 | 计数 |
| --- | --- | --- |
| positive | positive | TP |
| positive | negative | FN |
| negative | positive | FP |
| negative | negative | TN |

这些计数可以推导常见指标：

```text
accuracy = (TP + TN) / (TP + FN + FP + TN)
positive precision = TP / (TP + FP)
positive recall = TP / (TP + FN)
positive f1 = 2TP / (2TP + FP + FN)
negative recall = TN / (TN + FP)
```

## 9. FAH / FRR

FAH/FRR 是唤醒词系统更贴近实际部署的指标，位于 `dscnn_kws/eval_fah_frr.py`。

### score

脚本先收集正类概率：

```text
score = softmax(logits)[positive_index]
```

然后用阈值 `threshold` 判断：

```text
score >= threshold  -> 触发 positive
score < threshold   -> 不触发，判为 negative
```

这和普通 `argmax` 分类不同。FAH/FRR 允许你为了减少误触发而提高阈值，或为了减少漏唤醒而降低阈值。

### threshold

阈值在 validation 集上选择，不在 test 集上直接调参。流程是：

1. 收集 validation positive/negative 的 positive score。
2. 根据目标 `target_fah` 和 validation 负样本时长计算允许的最大误触发数。
3. 选择一个阈值，使 validation 上的 FAH 不超过目标值。
4. 用同一个阈值评估 test 集，输出 test FAH/FRR/ACC。

允许的最大误触发数：

```text
neg_hours = len(valid_neg) * window_sec / 3600
max_fp = floor(target_fah * neg_hours)
```

### FAH

FAH 是 False Alarms per Hour，每小时误触发次数：

```text
FAH = FP / negative_hours
negative_hours = negative_sample_count * window_sec / 3600
```

含义：

- FAH 越低，误唤醒越少。
- `FAH = 1.0` 表示平均每小时约 1 次误触发。
- 负样本时长越短，FAH 的分辨率越粗。

脚本会打印：

```text
FAH resolution = 3600 / (len(neg) * window_sec)
```

这表示在当前负样本数量下，增加或减少 1 个 FP 会让 FAH 变化多少。比如负样本总时长只有 0.5 小时，那么 1 个 FP 对应 `2 FAH`，无法稳定评估 `FAH <= 0.5` 这种目标。

### FRR

FRR 是 False Rejection Rate，漏唤醒率：

```text
FRR = FN / positive_sample_count
```

含义：

- FRR 越低，真实唤醒词被漏掉的比例越低。
- `FRR = 0.05` 表示 5% 的真实唤醒词没有触发。

### FAH 和 FRR 的权衡

提高阈值通常会：

- 降低 FP，从而降低 FAH。
- 增加 FN，从而提高 FRR。

降低阈值通常会：

- 增加 FP，从而提高 FAH。
- 降低 FN，从而降低 FRR。

所以 FAH/FRR 不是独立优化的两个数，而是同一阈值下的 trade-off。

## 10. 噪声测试指标

噪声相关测试仍然使用同一套分类指标：

```text
loss / acc / precision / recall / f1
```

区别在于测试样本会在线叠加噪声。仓库里主要有几类噪声评估：

```text
clean_test
  不加噪声，使用 test_manifest.json。

noise_validation_tau_valid_list
  validation split，使用 tau_valid.txt 噪声列表，固定 SNR。

noise_test_tau_test_list
  test split，使用 tau_test.txt 噪声列表，固定 SNR。

noise_scene_snr_test
  test split，按 TAU scene 和 SNR 网格分别测试。
```

量化脚本中的 `noise_scene_snr_test` 默认组合：

```text
scene: airport, bus, metro, metro_station, park, public_square,
       shopping_mall, street_pedestrian, street_traffic, tram
snr_db: 20, 10, 5, 0, -5
```

每个 scene/SNR 都会独立输出一组 `acc / precision / recall / f1`。SNR 越低，噪声越强，指标通常越低。

## 11. mean_acc / min_acc / max_acc

`sweep_fixed_dscnn_noise_snr_scene_acc.py` 会对 scene-SNR 网格结果做汇总。

### scene_summary

按：

```text
dataset + arch + scene
```

分组，统计该 scene 下不同 SNR 点的：

```text
mean_acc = 该 scene 所有 SNR 的 acc 平均值
min_acc  = 该 scene 所有 SNR 的最低 acc
max_acc  = 该 scene 所有 SNR 的最高 acc

mean_f1  = 该 scene 所有 SNR 的 f1 平均值
min_f1   = 该 scene 所有 SNR 的最低 f1
max_f1   = 该 scene 所有 SNR 的最高 f1
```

### arch_summary

按：

```text
dataset + arch
```

分组，统计该模型在所有 scene 和所有 SNR 点上的：

```text
mean_acc / min_acc / max_acc
mean_f1  / min_f1  / max_f1
```

含义：

- `mean_acc` / `mean_f1`：平均鲁棒性。
- `min_acc` / `min_f1`：最坏工况表现，通常比平均值更能暴露部署风险。
- `max_acc` / `max_f1`：最好工况表现，通常用于了解上限，不适合作为唯一选型指标。

## 12. num_samples

```text
num_samples = 当前评估 loader 中参与评估的样本数
```

它来自：

```text
total += labels.numel()
```

含义：

- 用于确认不同实验是否在同样规模的数据上比较。
- 如果某次测试 `num_samples` 异常偏小，指标波动会更大。
- 在 scene/SNR grid 中，同一 dataset 的不同 scene/SNR 通常样本数相同，因为语音样本来自同一个 test manifest，只是叠加的噪声不同。

## 13. best_valid_acc / best_epoch / test_acc_at_best

### best_valid_acc

训练过程中每个 epoch 都会在 validation 集评估一次。`best_valid_acc` 是这些验证准确率中的最大值：

```text
best_valid_acc = max(valid_acc over epochs)
```

在 `Trainer.fit()` 中，只有当 `valid_m.acc > self.best_acc` 时才保存 `best.pt`。

含义：

- 用于选择 checkpoint。
- 比最后一个 epoch 的 validation acc 更适合做模型选择。
- 不应把它当作最终泛化性能，最终性能应看 test split。

### best_epoch

NAS 重训脚本中记录 `best_valid_acc` 出现的 epoch：

```text
best_epoch = 验证集 acc 最高的 epoch 编号
```

### test_acc_at_best

NAS 重训脚本会加载 validation 最佳 epoch 的权重，再在 test 集上评估：

```text
test_acc_at_best = best checkpoint 在 test split 上的 acc
```

含义：

- 它不是“test 集上所有 epoch 的最高 acc”。
- 它是“按 validation 选出的模型，在 test 上的结果”。
- 这是更规范的模型选择方式，因为 test 没有参与选 epoch。

## 14. expected_params / printed_params / params

### printed_params

训练脚本启动时打印：

```text
params = sum(p.numel() for p in model.parameters())
```

这是 PyTorch 模型中可训练参数张量的总元素数。

### expected_params

固定 DSCNN sweep 脚本里使用解析公式估计参数量。该公式适用于这些脚本生成的固定结构：

```text
第 1 层: Conv2d, kernel=10x4, stride=2x2
后续层: Depthwise separable Conv2d, kernel=3x3, stride=1x1
所有层 channels 相同
num_classes = 2
```

公式：

```text
C = channels
N = num_layers
K = num_classes

expected_params = (N - 1) * C * C + (42 + 13 * (N - 1) + K) * C + K
```

当 `K = 2` 时：

```text
expected_params = (N - 1) * C^2 + (44 + 13 * (N - 1)) * C + 2
```

其中大致包括：

```text
第一层 Conv:        40C
第一层 BatchNorm:    2C
每个 DS block:       C^2 + 13C
最终 Linear:         C*K + K
```

注意：`expected_params` 是固定 sweep 配置的公式字段，不一定适用于任意 `model_size_info`、LSTM、NAS 模型或带可训练前端的模型。任意模型应优先看 `printed_params` 或实际 `sum(p.numel())`。

### NAS params

NAS 中 `params` 来自 `dscnn_kws/nas/constraints.py` 的结构代价估计。它按搜索空间中的层类型估算参数量：

```text
conv2d:   c_out * c_in * kernel_t * kernel_f
dsconv2d: c_in * kernel_t * kernel_f + c_out * c_in
dsconv1d: c_in * kernel_f + c_out * c_in
eca:      eca_kernel
```

NAS 的 `params` 是结构比较用的估计值，不完全等同于 PyTorch 模型里所有可训练参数，特别是 BatchNorm、分类头或前端参数是否计入，要看对应 NAS 模型和估算逻辑。

## 15. mults / total_mults / frontend_mults / backbone_mults

NAS 搜索中还会统计乘法次数：

```text
total_mults    = frontend_mults + backbone_mults
frontend_mults = MFCC 前端估算乘法次数
backbone_mults = NAS backbone 各层估算乘法次数
```

这些值来自 `dscnn_kws/nas/constraints.py`。

含义：

- `mults` 越低，理论计算量越小。
- 它是乘法次数估计，不是严格 FLOPs，也不是端到端实测延迟。
- 实际部署速度还受内存访问、算子实现、硬件并行度、缓存、量化方式影响。

NAS 中用 `acc / mults / params` 一起做 Pareto 比较：

- `acc` 越高越好。
- `mults` 越低越好。
- `params` 越低越好。

## 16. quantized_size_bytes

量化脚本保存 Q 格式 checkpoint 后记录：

```text
quantized_size_bytes = quantized_checkpoint 文件大小，单位 byte
```

它来自：

```text
path.stat().st_size
```

含义：

- 反映保存到磁盘的 PyTorch checkpoint 文件大小。
- 它不等同于裸权重大小，因为 checkpoint 还包含 metadata、字典结构、pickle/zip 序列化开销等。
- 如果要估算硬件 ROM/RAM 裸权重占用，应基于 `state_dict_qint` 中权重数量和每个整数位宽另算。

## 17. quant_format / scale

量化脚本默认使用 Q16.16：

```text
integer_bits = 16
fractional_bits = 16
total_bits = 32
scale = 2^16 = 65536
```

量化公式：

```text
q = clamp(round(x * scale), qmin, qmax)
x_q = q / scale
```

其中：

```text
qmin = -2^31
qmax =  2^31 - 1
```

量化评估时，模型权重会先 fake-quantize，前向中部分中间输出也会做 round/clamp/dequantize 仿真。指标仍然按普通分类方式计算。

## 18. eval_kind / split / scene / snr_db

量化和噪声 grid CSV 中常见字段：

```text
eval_kind
  当前评估类型，例如 clean_test、noise_test_tau_test_list、noise_scene_snr_test。

split
  使用的数据划分，通常是 validation 或 test。

scene
  TAU 噪声场景名。clean 或固定噪声列表测试时为空。

snr_db
  当前叠加噪声的信噪比。值越低，噪声越强。clean 测试时为空。

noise_roots / scene_noise_root
  当前使用的噪声文件列表或噪声目录。

usable_noise_files
  实际可用的 wav 噪声文件数量。脚本会忽略不存在或长度异常的文件。
```

## 19. returncode

sweep 脚本通过子进程运行训练命令，并记录：

```text
returncode = 子进程退出码
```

含义：

- `0`：训练命令正常结束。
- 非 `0`：训练命令异常退出。

如果 `returncode != 0`，对应行的 `test_acc / f1 / best_model_saved_as` 等字段可能为空，不应直接纳入统计分析。

## 20. 如何选择主要指标

### clean 分类实验

优先看：

```text
test_acc
macro_f1
test_loss
```

如果测试集正负样本是 1:1，`test_acc` 很直观；如果样本不均衡，更应看 `macro_f1`。

### 噪声鲁棒性实验

优先看：

```text
mean_acc / mean_f1
min_acc / min_f1
各 scene + snr_db 下的 acc/f1
```

`mean_*` 代表平均鲁棒性，`min_*` 代表最坏工况。部署选型时通常不能只看平均值。

### 唤醒词部署实验

优先看：

```text
FAH
FRR
threshold
TP / FN / FP / TN
```

原因是实际唤醒系统更关心：

- 每小时误触发多少次。
- 真实唤醒词漏掉多少。
- 在目标误触发约束下，漏唤醒是否可接受。

### 资源受限模型选择

优先看：

```text
acc / f1
expected_params 或 params
mults
quantized_size_bytes
```

推荐按约束筛选：

1. 先过滤 `returncode != 0` 或指标为空的实验。
2. 再按最低可接受 `acc / f1 / FAH / FRR` 筛掉不合格模型。
3. 在合格模型中选择 `params`、`mults` 或 `quantized_size_bytes` 更小的模型。

## 21. 常见误区

### 误区 1：acc 高就一定适合唤醒词部署

不一定。若负样本远多于正样本，模型只要倾向预测 negative 就能得到较高 acc，但可能漏掉很多唤醒词。部署时要看 FAH/FRR。

### 误区 2：macro recall 就是 positive recall

不是。当前输出的 `recall` 是宏平均：

```text
(positive recall + negative recall) / 2
```

如果需要单独的正类召回率，需要额外输出 per-class metrics 或混淆矩阵。

### 误区 3：FAH 可以在很短的负样本集上精确评估

不行。FAH 的分辨率取决于负样本总时长：

```text
FAH resolution = 1 / negative_hours
```

负样本越短，FAH 波动越大。

### 误区 4：expected_params 等于所有模型的真实参数量

不一定。`expected_params` 是固定 DSCNN sweep 的公式字段。任意结构请看实际 PyTorch 参数数或 NAS 对应的 `params` 定义。

### 误区 5：quantized_size_bytes 等于硬件权重存储量

不等于。它是 PyTorch checkpoint 文件大小，包含序列化和 metadata 开销。硬件存储量要按导出的整数权重数量和位宽计算。

## 22. 建议补充的调试输出

当前普通测试没有直接输出 per-class 指标和混淆矩阵。如果后续需要更细分析，建议在评估函数中增加：

```text
confusion_matrix(all_labels, all_preds, labels=[positive_index, negative_index])
classification_report(all_labels, all_preds, target_names=CLASS_LIST)
positive_precision
positive_recall
negative_recall
```

这样可以更直接地区分“漏唤醒”和“误唤醒”，尤其适合分析正负样本不均衡的数据集。
