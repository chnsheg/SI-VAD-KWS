# INT8 前端 + INT4 Backbone Fake-Quant QAT

本实验新增脚本：

```bash
dscnn_kws/quantization/qat_mixed_int8_frontend_int4_backbone_noise_snr_scene.py
```

它不修改之前的 INT8、INT4、INT3 或 per-channel 脚本。目标是验证一种更温和的混合量化方案：

- 前端路径：INT8 fake quant
- Backbone 激活：INT4 fake quant
- Backbone Conv/Linear 权重：INT4 fake quant，默认 per-channel symmetric
- 评估：五种 SNR x 十种 TAU scene
- 不做 PyTorch `convert()`，只做 fake-quant QAT 和 fake-quant 推理评估

## 为什么要这样做

之前“前端 + backbone 全部 INT4/INT3”的结果掉点很大，说明低比特对 MFCC 前端路径非常敏感。这个混合方案把前端恢复到 INT8，只让 backbone 承受 INT4 压缩，可以回答一个关键问题：

```text
掉点主要来自前端低比特，还是 backbone INT4 本身也无法承受？
```

如果这个方案明显好于全 INT4，说明前端低比特是主要瓶颈。  
如果仍然大幅掉点，说明 backbone INT4 也需要进一步策略，例如首尾层保留 INT8、只量化权重、或更长/渐进式 QAT。

## 默认配置

脚本顶部可以直接修改：

```python
FRONTEND_BIT_WIDTH = 8
BACKBONE_BIT_WIDTH = 4
BACKBONE_WEIGHT_QSCHEME = "per_channel"  # per_channel or per_tensor

QAT_EPOCHS = 30
LR = 5e-5
WEIGHT_DECAY = 1e-6
FREEZE_BN_AFTER_EPOCH = 8
DISABLE_OBSERVER_AFTER_EPOCH = 18
QUANTIZE_FRONTEND = True
WEIGHT_CH_AXIS = 0
```

其中 `QUANTIZE_FRONTEND=True` 表示前端路径仍然插入 fake quant，只是 bit-width 是 INT8，而不是 INT4。

## 量化点

前端路径的 INT8 fake quant 插在：

- waveform 输入
- pre-emphasis 输出
- MFCC/bandpass frontend 输出
- flatten 后送入 backbone 之前

Backbone 的 INT4 fake quant 包括：

- backbone 输入 fake quant
- conv/linear 激活 fake quant
- conv/linear 权重 fake quant
- backbone 输出 fake quant

默认权重是 `per_channel_symmetric`，激活是 `per_tensor_symmetric`。

## 直接运行

```bash
python dscnn_kws/quantization/qat_mixed_int8_frontend_int4_backbone_noise_snr_scene.py
```

默认输入目录：

```bash
/root/kws/dscnn_kws/dscnn_kws/runs/snr_scene_arch_sweep_best_models_full
```

默认输出：

```bash
dscnn_kws/quantization/qat_frontend_int8_backbone_int4_noise_snr_scene_models/
dscnn_kws/quantization/qat_frontend_int8_backbone_int4_noise_snr_scene_train_results.csv
dscnn_kws/quantization/qat_frontend_int8_backbone_int4_noise_snr_scene_grid_results.csv
```

## Smoke Test

先只跑一个 checkpoint，确认环境和路径没问题：

```bash
python dscnn_kws/quantization/qat_mixed_int8_frontend_int4_backbone_noise_snr_scene.py \
  --limit 1 \
  --qat_epochs 3 \
  --freeze_bn_after_epoch 1 \
  --disable_observer_after_epoch 2
```

这只用于检查流程，不代表最终精度。

## 正式实验建议

```bash
python dscnn_kws/quantization/qat_mixed_int8_frontend_int4_backbone_noise_snr_scene.py \
  --qat_epochs 30 \
  --freeze_bn_after_epoch 8 \
  --disable_observer_after_epoch 18 \
  --backbone_weight_qscheme per_channel
```

如果想确认 per-channel 是否真的有帮助，可以再跑一版 per-tensor：

```bash
python dscnn_kws/quantization/qat_mixed_int8_frontend_int4_backbone_noise_snr_scene.py \
  --qat_epochs 30 \
  --freeze_bn_after_epoch 8 \
  --disable_observer_after_epoch 18 \
  --backbone_weight_qscheme per_tensor \
  --output_dir ./dscnn_kws/quantization/qat_frontend_int8_backbone_int4_pertensor_noise_snr_scene_models \
  --train_results_csv ./dscnn_kws/quantization/qat_frontend_int8_backbone_int4_pertensor_noise_snr_scene_train_results.csv \
  --grid_results_csv ./dscnn_kws/quantization/qat_frontend_int8_backbone_int4_pertensor_noise_snr_scene_grid_results.csv
```

## 应该如何比较

重点比较这些结果：

```bash
qat_int4_noise_snr_scene_grid_results.csv
qat_int4_per_channel_noise_snr_scene_grid_results.csv
qat_frontend_int8_backbone_int4_noise_snr_scene_grid_results.csv
```

如果混合方案有效，通常应该看到：

- 明显优于“前端 + backbone 全 INT4”
- 高 SNR 场景恢复更明显
- `L5_C48`、`L5_C64` 等较大模型更容易恢复
- 低 SNR 仍会下降，但不应像全 INT4 那样整体塌缩

## 注意

这个脚本仍然是 fake-quant 评估：

- 训练和推理仍跑 float 算子
- fake quant 会模拟量化、反量化、舍入和截断误差
- 不会生成真正 INT4 PyTorch kernel
- `state_dict_qint` 和 `quant_specs` 只是后续硬件/定点实现的参考导出
