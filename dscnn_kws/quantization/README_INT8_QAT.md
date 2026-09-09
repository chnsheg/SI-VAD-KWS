# INT8 QAT Noise Scene Evaluation

本文件说明如何在已有噪声场景训练模型的基础上做 INT8 量化感知训练（QAT），再将 QAT 后模型转换为 INT8 量化模型，并在 10 个 TAU 场景、5 个 SNR 下测试量化后精度。

新增脚本：

```text
dscnn_kws/quantization/qat_int8_noise_snr_scene.py
```

这个脚本不会修改已有训练、sweep、PTQ 或 Q16.16 文件。

## 1. 做了什么

流程如下：

```text
已有噪声场景 best.pt
  -> 重建原始模型
  -> 加载 checkpoint
  -> 对前端路径插入 INT8 fake-quant
  -> 对 DSCNN backbone 插入 INT8 QAT fake-quant
  -> 使用噪声训练集 fine-tune 若干 epoch
  -> 用 tau_valid.txt + 5 dB 选择 best QAT epoch
  -> convert 得到 PyTorch INT8 quantized backbone
  -> 用 tau_test.txt + 5 dB 测试
  -> 用 10 scene x 5 SNR 测试
```

默认 QAT 范围：

```text
frontend: INT8 fake-quant QAT
DSCNN backbone: INT8 QAT + convert to PyTorch quantized ops
```

也就是说，脚本会对前端路径插入 INT8 fake-quant 量化点，让前端也在量化误差下参与 QAT fine-tune；同时 DSCNN backbone 会使用 PyTorch eager-mode QAT，并在训练后 `convert` 成 INT8 quantized ops。

前端默认量化点包括：

```text
waveform 输入
pre-emphasis 输出
feature_extractor 输出
flatten 后、送入 DSCNN backbone 前的 feature
```

需要注意：PyTorch eager-mode 不能直接把 `torch.stft`、`torch.log`、PWL log、functional `F.conv1d` bandpass 这些前端内部算子自动 convert 成真正整数 kernel。因此这里的前端量化是 **INT8 fake-quant QAT**，用于训练和评估前端张量被 INT8 网格约束后的精度；backbone 则会真正 convert 成 PyTorch INT8 quantized 模块。![alt text](image.png)

## 2. 默认输入输出

默认输入目录：

```text
/root/kws/dscnn_kws/dscnn_kws/runs/snr_scene_arch_sweep_best_models
```

默认数据目录：

```text
/root/kws/dscnn_kws/dscnn_kws/data
```

默认输出：

```text
dscnn_kws/quantization/qat_int8_noise_snr_scene_models/
dscnn_kws/quantization/qat_int8_noise_snr_scene_train_results.csv
dscnn_kws/quantization/qat_int8_noise_snr_scene_grid_results.csv
```

每个输入 checkpoint 会生成两类模型文件：

```text
*_qat_prepared_best.pt
  QAT fake-quant prepared 模型的 best state_dict

*_qat_int8_quantized.pt
  convert 后的 INT8 quantized 模型 state_dict
```

## 3. 默认噪声设置

QAT fine-tune 使用和噪声场景 sweep 一致的噪声训练设置：

```text
train_noise_roots  ./dscnn_kws/noise/lists/tau_train.txt
train_noise_prob   0.8
train_snr          -5 ~ 20 dB
```

验证和普通测试：

```text
validation:
  noise_roots      ./dscnn_kws/noise/lists/tau_valid.txt
  noise_prob       1.0
  snr_db           5.0

test:
  noise_roots      ./dscnn_kws/noise/lists/tau_test.txt
  noise_prob       1.0
  snr_db           5.0
```

场景/SNR 网格测试：

```text
scene_test_root    ./dscnn_kws/noise/tau
scene_names        airport bus metro metro_station park public_square shopping_mall street_pedestrian street_traffic tram
test_snrs          20 10 5 0 -5
```

scene/SNR 随机种子与之前 sweep 脚本一致：

```text
500000 + scene_idx * 10007 + snr_idx * 101
```

## 4. 运行 MFCC 噪声模型 QAT

在项目外层目录运行：

```bash
cd /root/kws/dscnn_kws
```

快速试跑一个模型：

```bash
python dscnn_kws/quantization/qat_int8_noise_snr_scene.py \
  --limit 1 \
  --qat_epochs 3
```

完整运行默认目录中的所有 checkpoint：

```bash
python dscnn_kws/quantization/qat_int8_noise_snr_scene.py \
  --qat_epochs 5 \
  --batch 128 \
  --num_workers 8 \
  --gpu 1
```

默认会启用前端 INT8 fake-quant QAT。如果只想复现实验中“只量化 backbone”的旧行为，可显式关闭：

```bash
python dscnn_kws/quantization/qat_int8_noise_snr_scene.py \
  --no-quantize_frontend \
  --qat_epochs 5
```

如果只跑某个 checkpoint：

```bash
python dscnn_kws/quantization/qat_int8_noise_snr_scene.py \
  --checkpoints /root/kws/dscnn_kws/dscnn_kws/runs/snr_scene_arch_sweep_best_models/<model>.pt \
  --dataset mobvoi_hi_xiaowen_binary_hardneg \
  --qat_epochs 5
```

如果 checkpoint 文件名不能推断 arch，可以显式指定：

```bash
--layers 5 --channels 64
```

## 5. 运行 projected bandpass 噪声模型 QAT

如果输入 checkpoint 来自 `sweep_fixed_projected_bandpass_noise_snr_scene_acc.py`，使用：

```bash
python dscnn_kws/quantization/qat_int8_noise_snr_scene.py \
  --model_family projected_bandpass \
  --input_dir ./dscnn_kws/runs/projected_bandpass40_to10_pwl_snr_scene_arch_sweep_best_models \
  --log_approx_mode pwl \
  --log_pwl_num_segments 8 \
  --bandpass_internal_bands 40 \
  --dct_coeff 10 \
  --bandpass_f_min 80 \
  --bandpass_f_max 6000 \
  --bandpass_kernel_size 255 \
  --bandpass_phase_count 4 \
  --projection_init dct \
  --trainable_projection \
  --qat_epochs 5
```

关键是：**前端参数必须和训练该 checkpoint 时一致**。包括：

```text
dct_coeff
bandpass_internal_bands
bandpass_f_min / f_max
bandpass_spacing
bandpass_kernel_size
bandpass_phase_count
projection_init
trainable_projection
log_approx_mode
log_pwl_num_segments
```

## 6. 运行普通 bandpass 噪声模型 QAT

如果 checkpoint 是普通 `MFCCDSCNN` wrapper 的 `frontend=bandpass` 模型：

```bash
python dscnn_kws/quantization/qat_int8_noise_snr_scene.py \
  --model_family standard \
  --frontend bandpass \
  --input_dir ./dscnn_kws/runs/bandpass_pwl_snr_scene_arch_sweep_best_models \
  --dct_coeff 10 \
  --bandpass_n_bands 10 \
  --bandpass_f_min 80 \
  --bandpass_f_max 6000 \
  --bandpass_kernel_size 255 \
  --bandpass_phase_count 4 \
  --log_approx_mode pwl \
  --log_pwl_num_segments 6 \
  --qat_epochs 5
```

`--frontend bandpass` 时要求：

```text
dct_coeff == bandpass_n_bands
```

## 7. QAT 训练参数

默认：

```text
qat_epochs                    5
lr                            1e-4
weight_decay                  1e-6
backend                       fbgemm
freeze_bn_after_epoch         2
disable_observer_after_epoch  3
quantize_frontend             True
```

含义：

```text
freeze_bn_after_epoch
  前几个 epoch 允许 QAT fused BN 继续更新统计量，之后冻结。

disable_observer_after_epoch
  前几个 epoch 允许 fake-quant observer 继续更新 scale，之后固定 scale。

quantize_frontend
  对 waveform、pre-emphasis 输出、前端输出、backbone 输入做 INT8 fake-quant QAT。
```

如果模型较小或数据较少，可以先用：

```bash
--qat_epochs 3 --lr 5e-5
```

如果量化后仍掉点明显，可以尝试：

```bash
--qat_epochs 8 --lr 1e-4
```

## 8. CSV 字段

训练结果 CSV：

```text
qat_int8_noise_snr_scene_train_results.csv
```

主要字段：

```text
dataset
arch
layers / channels
source_checkpoint
best_epoch
best_valid_acc
qat_test_tau_list_acc
quantized_test_tau_list_acc
qat_prepared_checkpoint
quantized_checkpoint
backend
model_family
frontend
quantize_frontend
```

场景/SNR 测试 CSV：

```text
qat_int8_noise_snr_scene_grid_results.csv
```

主要字段：

```text
dataset
arch
scene
snr_db
acc
precision
recall
f1
num_samples
source_checkpoint
quantized_checkpoint
```

每个模型理论上会产生：

```text
10 scenes x 5 SNR = 50 行 grid 结果
```

## 9. 和 PTQ 脚本的区别

之前的 `quantize_sweep_best_models_int8.py` 是 PTQ / fake quant evaluation：

```text
加载 best.pt
校准 scale
直接测试
不训练
```

本脚本是 QAT：

```text
加载 best.pt
插入前端 fake-quant + backbone QAT fake-quant
继续训练若干 epoch
用验证集选择 best
convert 为 INT8 quantized 模型
再做场景/SNR 测试
```

如果 PTQ 在 clean 或某些场景下掉点较大，QAT 通常能恢复一部分精度。

## 10. 注意事项

- 训练阶段可以使用 GPU；convert 后的 INT8 quantized backbone 在 CPU 上评估。
- 前端参与 INT8 fake-quant QAT，但前端内部 STFT/log/PWL/functional bandpass 不会被 PyTorch 自动转换成真正整数 kernel。
- 这个脚本使用 PyTorch eager-mode QAT + 前端 fake-quant，不是手写 Verilog bit-accurate int8 仿真。
- 如果要做硬件级验证，仍需要根据 quantized 模型中的 scale、zero point、packed weights 设计导出和 RTL/testbench 对齐流程。
