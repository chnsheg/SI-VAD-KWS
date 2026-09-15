# INT4 / INT3 QAT Noise Scene Evaluation

低比特 QAT 现在只有一个脚本：

```text
dscnn_kws/quantization/qat_lowbit_noise_snr_scene.py
```

在脚本顶部可以直接改：

```python
BIT_WIDTH = 4  # 3 or 4
```

也可以运行时用 `--bit_width` 覆盖。

## 1. 做了什么

流程参考 `qat_int8_noise_snr_scene.py`：

```text
噪声场景 best.pt
  -> 重建原模型
  -> 加载 checkpoint
  -> 前端路径插入 low-bit fake-quant
  -> backbone 插入 low-bit fake-quant QAT
  -> 噪声训练集 fine-tune
  -> tau_valid.txt + 5 dB 选 best epoch
  -> tau_test.txt + 5 dB 测试
  -> 10 scene x 5 SNR 测试
  -> 导出 fake-quant 模型和 qint state_dict
```

量化范围：

```text
INT4: [-8, 7]
INT3: [-4, 3]
```

## 2. 重要区别

PyTorch eager-mode 量化后端提供的是 INT8 quantized conv/linear kernel，不提供 INT4/INT3 的 `convert` 后 CPU kernel。

所以这个脚本做的是：

```text
INT4/INT3 signed symmetric fake-quant QAT
保留 fake-quant float 模型做评估
额外导出 state_dict_qint 和 quant_specs
```

不是：

```text
真正 INT4/INT3 kernel 推理
真正 convert 成 PyTorch INT4/INT3 quantized module
硬件 bit-accurate 推理
```

## 3. 顶部常用配置

最常改的是：

```python
BIT_WIDTH = 4
INPUT_DIR = "/root/kws/dscnn_kws/dscnn_kws/runs/snr_scene_arch_sweep_best_models"
OUTPUT_DIR = f"./dscnn_kws/quantization/qat_int{BIT_WIDTH}_noise_snr_scene_models"
TRAIN_RESULTS_CSV = f"./dscnn_kws/quantization/qat_int{BIT_WIDTH}_noise_snr_scene_train_results.csv"
GRID_RESULTS_CSV = f"./dscnn_kws/quantization/qat_int{BIT_WIDTH}_noise_snr_scene_grid_results.csv"
QAT_EPOCHS = 10
LR = 5e-5
BATCH = 256
GPU = 1
```

噪声设置默认和 noise scene sweep 一致：

```text
train_noise_roots  ./dscnn_kws/noise/lists/tau_train.txt
train_snr          -5 ~ 20 dB
valid/test         tau_valid.txt / tau_test.txt at 5 dB
scene grid         10 TAU scenes x 20/10/5/0/-5 dB
```

## 4. 运行 INT4

方式一：修改脚本顶部：

```python
BIT_WIDTH = 4
```

然后运行：

```bash
cd /root/kws/dscnn_kws

python dscnn_kws/quantization/qat_lowbit_noise_snr_scene.py \
  --limit 1 \
  --qat_epochs 3 \
  --batch 128 \
  --gpu 1
```

方式二：不改文件，直接覆盖：

```bash
python dscnn_kws/quantization/qat_lowbit_noise_snr_scene.py \
  --bit_width 4 \
  --limit 1 \
  --qat_epochs 3 \
  --batch 128 \
  --gpu 1
```

## 5. 运行 INT3

方式一：修改脚本顶部：

```python
BIT_WIDTH = 3
```

然后运行：

```bash
cd /root/kws/dscnn_kws

python dscnn_kws/quantization/qat_lowbit_noise_snr_scene.py \
  --limit 1 \
  --qat_epochs 3 \
  --batch 128 \
  --gpu 1
```

方式二：不改文件，直接覆盖：

```bash
python dscnn_kws/quantization/qat_lowbit_noise_snr_scene.py \
  --bit_width 3 \
  --limit 1 \
  --qat_epochs 3 \
  --batch 128 \
  --gpu 1
```

## 6. 输出文件

默认输出会随 `BIT_WIDTH` 自动改变：

```text
dscnn_kws/quantization/qat_int4_noise_snr_scene_models/
dscnn_kws/quantization/qat_int4_noise_snr_scene_train_results.csv
dscnn_kws/quantization/qat_int4_noise_snr_scene_grid_results.csv
```

或者：

```text
dscnn_kws/quantization/qat_int3_noise_snr_scene_models/
dscnn_kws/quantization/qat_int3_noise_snr_scene_train_results.csv
dscnn_kws/quantization/qat_int3_noise_snr_scene_grid_results.csv
```

每个 checkpoint 会保存：

```text
*_qat_int4_prepared_best.pt / *_qat_int3_prepared_best.pt
*_qat_int4_fake_quant.pt   / *_qat_int3_fake_quant.pt
```

`*_fake_quant.pt` 包含：

```text
state_dict
state_dict_qint
quant_specs
metadata
```

`state_dict_qint` 用 `torch.int8` 容器保存 INT4/INT3 整数值，真实有效范围由 `quant_specs` 中的 `qmin/qmax` 指明。
