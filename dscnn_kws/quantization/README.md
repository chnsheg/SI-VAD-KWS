# Q16.16 Quantization for DSCNN Best Models

本目录用于对 sweep 得到的 DSCNN best models 做 Q16.16 定点仿真量化，并在 clean 场景和噪声场景下重新测试。

量化脚本：

```text
dscnn_kws/quantization/quantize_sweep_best_models.py
```

默认使用两个服务器结果目录：

```text
/root/kws/dscnn_kws/dscnn_kws/runs/sweep_best_models
/root/kws/dscnn_kws/dscnn_kws/runs/snr_scene_arch_sweep_best_models
```

其中：

```text
sweep_best_models
  clean sweep 的 best models
  Q16.16 后做 clean test

snr_scene_arch_sweep_best_models
  sweep_fixed_dscnn_noise_snr_scene_acc.py 保存的噪声训练 best models
  Q16.16 后按同一套噪声验证/测试设置评估
```

## 1. Q16.16 方式

脚本使用 signed 32-bit Q16.16：

```text
integer_bits    16
fractional_bits 16
scale           65536
int range       [-2147483648, 2147483647]
real range      [-32768.0, 32767.99998474121]
```

量化采用定点仿真：

```text
q = clamp(round(x * 65536), int32_min, int32_max)
x_q = q / 65536
```

保存的 checkpoint 同时包含：

```text
state_dict_qint
  int32 Q16.16 权重/BN buffer，可用于后续硬件或 C 侧导出

state_dict_qfloat
  qint 反量化后的 float 权重，用于 PyTorch 中复现实验
```

评估时会对模型输入、DSCNN backbone 中的 Conv / BN / ReLU / Pool / Linear 输出做 Q16.16 round-clamp-dequantize 仿真。默认不量化 MFCC 前端内部输出；如果希望连前端部分的叶子模块输出也做仿真，可加 `--quantize_frontend`。

## 2. 运行位置

推荐在项目外层目录运行，也就是可以直接执行 `python -m dscnn_kws.train` 的目录：

```bash
cd /root/kws/dscnn_kws
```

默认数据目录：

```text
/root/kws/dscnn_kws/dscnn_kws/data
```

需要具备：

```text
dscnn_kws/data/<dataset_name>/
  train_manifest.json
  validation_manifest.json
  test_manifest.json

dscnn_kws/noise/lists/tau_valid.txt
dscnn_kws/noise/lists/tau_test.txt
dscnn_kws/noise/tau/<scene>/*.wav
```

## 3. 一键运行 clean + noise

```bash
python dscnn_kws/quantization/quantize_sweep_best_models.py
```

默认输出：

```text
dscnn_kws/quantization/q16_16_sweep_best_models/
dscnn_kws/quantization/q16_16_snr_scene_arch_sweep_best_models/
dscnn_kws/quantization/q16_16_clean_results.csv
dscnn_kws/quantization/q16_16_noise_snr_scene_results.csv
```

## 4. 只跑 clean 或 noise

只处理 clean best models：

```bash
python dscnn_kws/quantization/quantize_sweep_best_models.py \
  --mode clean
```

只处理噪声场景 best models：

```bash
python dscnn_kws/quantization/quantize_sweep_best_models.py \
  --mode noise_snr_scene
```

快速检查流程时，每类只跑 1 个 checkpoint：

```bash
python dscnn_kws/quantization/quantize_sweep_best_models.py \
  --mode both \
  --limit 1
```

## 5. Clean 测试

clean profile 的默认输入：

```text
/root/kws/dscnn_kws/dscnn_kws/runs/sweep_best_models
```

测试设置：

```text
split          test_manifest.json
noise_aug      False
sample_rate    16000
dct_coeff      10
window         32 ms
stride         32 ms
```

输出 CSV：

```text
dscnn_kws/quantization/q16_16_clean_results.csv
```

每个模型对应一行 `clean_test` 结果。

## 6. 噪声验证和测试

noise profile 的默认输入：

```text
/root/kws/dscnn_kws/dscnn_kws/runs/snr_scene_arch_sweep_best_models
```

它和 `sweep_fixed_dscnn_noise_snr_scene_acc.py` 保持一致，包含两部分评估。

第一部分是固定列表验证/测试：

```text
validation split:
  manifest       validation_manifest.json
  noise_roots    ./dscnn_kws/noise/lists/tau_valid.txt
  noise_prob     1.0
  snr_db         5.0
  random_seed    seed + 100000

test split:
  manifest       test_manifest.json
  noise_roots    ./dscnn_kws/noise/lists/tau_test.txt
  noise_prob     1.0
  snr_db         5.0
  random_seed    seed + 200000
```

第二部分是 TAU scene × SNR 网格测试：

```text
scene_test_root  ./dscnn_kws/noise/tau
scenes           airport bus metro metro_station park public_square shopping_mall street_pedestrian street_traffic tram
test_snrs        20 10 5 0 -5
split            test_manifest.json
noise_prob       1.0
```

每个 scene/SNR 的随机种子与原 sweep 脚本一致：

```text
500000 + scene_idx * 10007 + snr_idx * 101
```

输出 CSV：

```text
dscnn_kws/quantization/q16_16_noise_snr_scene_results.csv
```

其中每个模型会包含：

```text
1 行 noise_validation_tau_valid_list
1 行 noise_test_tau_test_list
50 行 noise_scene_snr_test
```

如果某个 scene 目录没有可用 wav，会跳过该 scene。

## 7. 常用参数

指定 batch 和 workers：

```bash
python dscnn_kws/quantization/quantize_sweep_best_models.py \
  --mode both \
  --batch 256 \
  --num_workers 8
```

如果服务器路径不同：

```bash
python dscnn_kws/quantization/quantize_sweep_best_models.py \
  --root /path/to/dscnn_kws/data \
  --clean_input_dir /path/to/sweep_best_models \
  --noise_input_dir /path/to/snr_scene_arch_sweep_best_models
```

修改 Q 格式，例如 Q8.24：

```bash
python dscnn_kws/quantization/quantize_sweep_best_models.py \
  --integer_bits 8 \
  --fractional_bits 24 \
  --total_bits 32
```

连前端叶子模块输出也做 Q16.16 仿真：

```bash
python dscnn_kws/quantization/quantize_sweep_best_models.py \
  --mode both \
  --quantize_frontend
```

## 8. CSV 字段

主要字段：

```text
profile                 clean 或 noise_snr_scene
dataset                 数据集名称
checkpoint              原始 best model
quantized_checkpoint    Q16.16 checkpoint
quant_format            Q16.16
scale                   65536
layers / channels       DSCNN 结构
expected_params         期望参数量
eval_kind               clean_test / noise_validation_tau_valid_list / noise_test_tau_test_list / noise_scene_snr_test
split                   validation 或 test
scene                   TAU scene，clean 和固定列表测试为空
snr_db                  噪声 SNR，clean 为空
noise_roots             当前测试使用的噪声路径
usable_noise_files      可用噪声 wav 数量
loss
acc
precision
recall
f1
num_samples
quantized_size_bytes
```

## 9. 单 checkpoint 调试

如果 checkpoint 文件名不能推断 dataset，可以单独指定：

```bash
mkdir -p /tmp/kws_q16_one
cp /root/kws/dscnn_kws/dscnn_kws/runs/sweep_best_models/<model>.pt /tmp/kws_q16_one/

python dscnn_kws/quantization/quantize_sweep_best_models.py \
  --mode clean \
  --clean_input_dir /tmp/kws_q16_one \
  --dataset mobvoi_hi_xiaowen_binary_hardneg \
  --limit 1
```

## 10. 注意事项

- 这个脚本评估的是 Q16.16 定点仿真精度，不是 PyTorch int8 后端量化。
- 推理仍在 PyTorch float 算子中运行，只是在权重和指定中间输出处插入 Q16.16 round/clamp/dequantize。
- Q16.16 精度较高，预期 accuracy drop 通常很小；如果希望看到更明显的定点误差，可以尝试 Q8.24、Q12.20 或减少 fractional bits。
- 如果训练 sweep 使用了非默认前端参数，量化测试时也要传入相同参数，例如 `--dct_coeff`、`--window_stride_ms`、`--frontend` 等。
- 脚本默认 CPU 评估，不需要 GPU。
