# INT4/INT3 Per-Channel QAT 说明

本目录新增脚本：

```bash
dscnn_kws/quantization/qat_lowbit_per_channel_noise_snr_scene.py
```

它是在原 `qat_lowbit_noise_snr_scene.py` 之外新增的一套实验文件，不修改之前的脚本和结果。

## 这版做了什么

这版仍然用于噪声场景下的 INT4/INT3 QAT fake quant 评估，评估设置保持和 `sweep_fixed_dscnn_noise_snr_scene_acc.py` 一致：

- 训练噪声：`tau_train.txt`
- 验证噪声：`tau_valid.txt`
- 测试噪声：`tau_test.txt`
- 五种 SNR：`20, 10, 5, 0, -5 dB`
- 十种 TAU scene：`airport, bus, metro, metro_station, park, public_square, shopping_mall, street_pedestrian, street_traffic, tram`

和旧 low-bit 脚本相比，主要变化有两个：

1. **Conv/Linear 权重改成 per-channel symmetric 低比特 fake quant**

   - 原脚本：权重和激活都是 `per_tensor_symmetric`
   - 新脚本：权重是 `per_channel_symmetric`，激活仍是 `per_tensor_symmetric`
   - 默认 `WEIGHT_CH_AXIS = 0`，也就是按输出通道分别统计 scale

2. **QAT 默认训练更久**

   默认配置改成：

```python
QAT_EPOCHS = 30
LR = 5e-5
FREEZE_BN_AFTER_EPOCH = 8
DISABLE_OBSERVER_AFTER_EPOCH = 18
QUANTIZE_FRONTEND = True
```

其中 `QUANTIZE_FRONTEND = True` 表示仍然保持“前端 + backbone 都做同 bit-width 量化模拟”。前端路径包括 waveform、pre-emphasis 后、MFCC 输出、backbone 输入的低比特 fake quant。

## 重要边界

这里的 per-channel 主要作用在 **backbone 的 Conv/Linear 权重** 上。原因是 per-channel weight quantization 是低比特卷积里最常见、也最有效的做法。

前端 waveform/MFCC/features 和网络激活仍然是 per-tensor activation fake quant。也就是说：

- 权重：per-channel symmetric INT4/INT3 fake quant
- 激活：per-tensor symmetric INT4/INT3 fake quant
- 前端路径：per-tensor symmetric INT4/INT3 fake quant
- 仍然不是 PyTorch `convert()` 后的真正 INT4/INT3 kernel

PyTorch eager quantization 后端主要支持 INT8 kernel。INT4/INT3 这里用于 QAT 训练和量化损失评估，同时导出 `state_dict_qint` 和 `quant_specs`，方便后续硬件/定点实现参考。

## 直接运行 INT4

```bash
python dscnn_kws/quantization/qat_lowbit_per_channel_noise_snr_scene.py \
  --bit_width 4
```

默认输入目录：

```bash
/root/kws/dscnn_kws/dscnn_kws/runs/snr_scene_arch_sweep_best_models_full
```

默认输出：

```bash
dscnn_kws/quantization/qat_int4_per_channel_noise_snr_scene_models/
dscnn_kws/quantization/qat_int4_per_channel_noise_snr_scene_train_results.csv
dscnn_kws/quantization/qat_int4_per_channel_noise_snr_scene_grid_results.csv
```

## 直接运行 INT3

```bash
python dscnn_kws/quantization/qat_lowbit_per_channel_noise_snr_scene.py \
  --bit_width 3
```

默认输出会自动切换为：

```bash
dscnn_kws/quantization/qat_int3_per_channel_noise_snr_scene_models/
dscnn_kws/quantization/qat_int3_per_channel_noise_snr_scene_train_results.csv
dscnn_kws/quantization/qat_int3_per_channel_noise_snr_scene_grid_results.csv
```

## 只跑一个 checkpoint 做 smoke test

```bash
python dscnn_kws/quantization/qat_lowbit_per_channel_noise_snr_scene.py \
  --bit_width 4 \
  --limit 1 \
  --qat_epochs 3 \
  --freeze_bn_after_epoch 1 \
  --disable_observer_after_epoch 2
```

这只用于检查脚本、数据路径和 checkpoint 加载是否正常，不代表最终精度。

## 建议的正式实验

INT4：

```bash
python dscnn_kws/quantization/qat_lowbit_per_channel_noise_snr_scene.py \
  --bit_width 4 \
  --qat_epochs 30 \
  --freeze_bn_after_epoch 8 \
  --disable_observer_after_epoch 18
```

INT3：

```bash
python dscnn_kws/quantization/qat_lowbit_per_channel_noise_snr_scene.py \
  --bit_width 3 \
  --qat_epochs 40 \
  --freeze_bn_after_epoch 10 \
  --disable_observer_after_epoch 24
```

INT3 更激进，建议比 INT4 训练更久。

## 如何判断有没有改善

重点比较以下文件：

```bash
qat_int4_noise_snr_scene_grid_results.csv
qat_int4_per_channel_noise_snr_scene_grid_results.csv

qat_int3_noise_snr_scene_grid_results.csv
qat_int3_per_channel_noise_snr_scene_grid_results.csv
```

如果 per-channel 有效，通常会看到：

- INT4 平均准确率明显高于旧 per-tensor 版本
- 高 SNR 下恢复更明显
- 大模型如 `L5_C48`, `L5_C64` 更容易恢复
- INT3 可能仍然损失较大，但不应像 per-tensor 那样普遍接近随机

## 代码顶部可直接改的关键参数

```python
BIT_WIDTH = 4
QAT_EPOCHS = 30
LR = 5e-5
WEIGHT_DECAY = 1e-6
FREEZE_BN_AFTER_EPOCH = 8
DISABLE_OBSERVER_AFTER_EPOCH = 18
QUANTIZE_FRONTEND = True
WEIGHT_CH_AXIS = 0
```

如果只想看 backbone 低比特损失，可以临时改：

```python
QUANTIZE_FRONTEND = False
```

但本次默认仍然保持前端和 backbone 都做同 bit-width 量化模拟。
