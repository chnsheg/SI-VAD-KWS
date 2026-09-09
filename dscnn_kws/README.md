# DSCNN-KWS（精简版）

这个目录是从 TorchKWS 中拆分出的 **仅 DSCNN** 训练工程。

## 特性

- 只保留 DSCNN 模型训练链路（无多模型分发接口）
- 包含完整流程：训练 / 验证 / 测试 / best&last 模型保存
- 默认数据根目录：`./dataset`

## 目录

```text
dscnn_kws/
├─ train.py
├─ configs.py
├─ frontend/
│  ├─ mfcc_torch.py
│  ├─ pwl_fit_utils.py
│  ├─ fit_log_pwl.py
│  ├─ run_log_pwl_grid.py
│  ├─ run_log_pwl_gamma_scan_seg4.py
│  ├─ artifacts/
│  └─ README.md
├─ model/dscnn.py
├─ data/dataset.py
├─ engine/trainer.py
└─ utils/
```

## 安装依赖

```bash
pip install -r dscnn_kws/requirements.txt
```

## 运行训练

```bash
python -m dscnn_kws.train --dataset speech_commands_v0.02_sr8k --sample_rate 8000
```

## 默认即“旧工程行为基线”

当前默认配置已对齐旧 TorchKWS dscnn 的有效训练行为（你已验证 test 可到 0.92+）：

```bash
python -m dscnn_kws.train \
  --dataset speech_commands_v0.02_sr8k \
  --sample_rate 8000 \
  --window_size_ms 32 \
  --window_stride_ms 32 \
  --epoch 50 \
  --num_workers 8 \
  --opt adam
```

默认关键设置：
- scheduler → `cos` (`CosineAnnealingWarmRestarts`, `t0=10`, `t_mult=1`, `eta_min=lr*0.01`)
- `pre_emphasis=True`, `pre_emphasis_coeff=0.97`
- `non_deterministic=True`（更接近旧工程训练行为）

同时启动时会打印：
- `split_stats`（train/valid/test 样本数与 unknown/silence 数量）

## 逐项优化原则（单变量A/B，无收益就删除）

在默认基线之上一次只加一项，先短程对照（如 15~20 epoch），确认 test 提升再保留代码。

### 当前可选优化：SpecAugment（本轮新增）

说明：
- 仅在训练时对 MFCC 做频率/时间遮挡；验证/测试自动关闭；
- 默认关闭（`--no-spec_aug`），不影响旧版基线复现；
- 建议与基线做单变量A/B：先 20 epoch 快筛，再 50 epoch 复验。

示例（开启 SpecAugment）：

```bash
python -m dscnn_kws.train --dataset speech_commands_v0.02_sr8k --sample_rate 8000 --window_size_ms 32 --window_stride_ms 32 --epoch 20 --num_workers 8 --opt adam --spec_aug
```

可调参数：
- `--spec_aug_freq_mask_param`（默认 4）
- `--spec_aug_time_mask_param`（默认 3）
- `--spec_aug_num_freq_masks`（默认 2）
- `--spec_aug_num_time_masks`（默认 2）

若你使用 16k 数据，可改成：

```bash
python -m dscnn_kws.train --dataset speech_commands_v0.02 --sample_rate 16000
```

## Log 函数线性分段近似实验（硬件部署模拟）

本工程已支持在 `TorchMFCC` 中把 `log()` 替换为分段线性函数（PWL）：

- `--log_approx_mode exact`：原始对数（基线）
- `--log_approx_mode pwl`：分段线性近似

### 1) 直接用默认拟合规则训练（不依赖外部拟合文件）

```bash
python -m dscnn_kws.train --dataset speech_commands_v0.02_sr8k --sample_rate 8000 --mfcc_impl torch --log_approx_mode pwl --log_pwl_num_segments 6 --log_pwl_strategy uniform_logx
```

### 2) 基于真实数据采样拟合 PWL 参数（不推荐quantile效果一般）

```bash
python -m dscnn_kws.frontend.fit_log_pwl --dataset speech_commands_v0.02_sr8k --sample_rate 8000 --num_segments 6 --strategy quantile --max_batches 120 --max_points 300000 --save_json dscnn_kws/frontend/artifacts/log_pwl_fit_quantile_seg6.json
```

拟合结果 JSON 中包含：
- `breakpoints`, `slopes`, `intercepts`
- `fit_mae`, `fit_max_ae`

### 3) 训练时加载拟合好的 PWL 系数

```bash
python -m dscnn_kws.train --dataset speech_commands_v0.02_sr8k --sample_rate 8000 --mfcc_impl torch --log_approx_mode pwl --log_pwl_fit_json dscnn_kws/frontend/artifacts/log_pwl_fit_quantile_seg6.json
```

### 4) 批量网格实验（段数 × 策略 × seed）

```bash
python -m dscnn_kws.frontend.run_log_pwl_grid --dataset speech_commands_v0.02_sr8k --sample_rate 8000 --epochs 20 --segments 2 4 6 8 12 --strategies uniform_logx quantile --seeds 42 43 44 --extra_train_args "--num_workers 8"
```

会生成 CSV：`dscnn_kws/frontend/artifacts/log_pwl_grid_results.csv`。

### 5) 固定 4 段扫描 gamma（每次可跑满 50 epochs）

```bash
python -m dscnn_kws.frontend.run_log_pwl_gamma_scan_seg4 --dataset speech_commands_v0.02_sr8k --sample_rate 8000 --epochs 50 --seed 42 --gammas 0.5 0.6 0.8 1.0 1.2 1.5 2.0 2.5 --extra_train_args "--num_workers 8"
```

会生成 CSV：`dscnn_kws/frontend/artifacts/log_pwl_gamma_scan_seg4.csv`（按 acc 排序）。

## 训练参数速查（与 frontend 相关）

以下参数位于 `python -m dscnn_kws.train ...`：

- `--mfcc_impl {torchaudio,torch}`  
  选择官方 MFCC 或项目内复现 MFCC。
- `--mel_filter_shape {triangular,rectangular}`  
  Mel 滤波器形状。
- `--log_approx_mode {exact,pwl}`  
  `exact` 为原始 `log`；`pwl` 为分段线性近似。
- `--log_pwl_num_segments`  
  PWL 段数（常用 4/6/8/12）。
- `--log_pwl_strategy {uniform_logx,quantile,powerlaw}`  
  分段策略：
  - `uniform_logx`：log 域等距分段；
  - `quantile`：按样本分位数分段；
  - `powerlaw`：幂律分段（由 gamma 控制密度）。
- `--log_pwl_gamma`  
  仅对 `powerlaw` 生效。`gamma=1` 时接近标准 log 等距，`>1` 更偏低值区，`<1` 更偏高值区。
- `--log_pwl_fit_json`  
  读取外部拟合好的 `breakpoints/slopes/intercepts`。
- `--log_offset`  
  log 前偏置，避免接近 0 数值不稳定（常用 `1e-6 ~ 1e-5`）。
- `--log_input_clamp_min`  
  输入到 log/PWL 前的最小裁剪值。

> frontend 子目录脚本与参数详解见：`dscnn_kws/frontend/README.md`
