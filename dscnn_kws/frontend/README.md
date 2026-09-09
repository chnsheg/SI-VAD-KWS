# frontend 说明（MFCC复现 / Log分段近似 / 实验脚本）

本目录用于：
- 复现 PyTorch 版 MFCC 前端；
- 将 `log()` 替换为分段线性（PWL）以模拟硬件部署；
- 提供拟合脚本与批量实验脚本。

---

## 文件结构

```text
frontend/
├─ __init__.py
├─ mfcc_torch.py
├─ pwl_fit_utils.py
├─ fit_log_pwl.py
├─ run_log_pwl_grid.py
├─ run_log_pwl_gamma_scan_seg4.py
└─ artifacts/
```

### 1) `mfcc_torch.py`

核心前端实现：
- Mel滤波器构造（`triangular` / `rectangular`）；
- DCT矩阵构造；
- `TorchMFCC` 前向：`STFT -> Power -> Mel -> log/PWL -> DCT`；
- PWL运行时应用工具：
  - `apply_piecewise_linear(...)`
  - `load_log_pwl_json(...)`

### 2) `pwl_fit_utils.py`

用途：
- 提供 log 到 PWL 的拟合核心函数；
- 被 `fit_log_pwl.py`（离线拟合）和 `mfcc_torch.py`（默认参数构造）共同复用。

核心函数：
- `fit_piecewise_linear_log_from_samples(...)`

`fit_piecewise_linear_log_from_samples` 支持分段策略：
- `uniform_logx`：在 log(x) 轴等距分段
- `quantile`：按样本分位数分段
- `powerlaw`：按幂律分段（由 `gamma` 控制）

`powerlaw` 断点公式：

`b_i = x_min * (x_max / x_min) ^ ((i / K) ^ gamma)`

其中 `K` 为段数，`i=0..K`。

---

### 2) `fit_log_pwl.py`

用途：从训练集采样真实 mel 能量分布，拟合 log 的 PWL 参数并导出 JSON。

#### 常用命令

```bash
python -m dscnn_kws.frontend.fit_log_pwl \
  --dataset speech_commands_v0.02_sr8k \
  --sample_rate 8000 \
  --num_segments 6 \
  --strategy uniform_logx \
  --save_json dscnn_kws/frontend/artifacts/log_pwl_fit_uniform_seg6.json
```

#### 参数说明

- `--root`：数据根目录（默认 `./dataset`）
- `--dataset`：数据集目录名（如 `speech_commands_v0.02_sr8k`）
- `--sample_rate`：采样率
- `--window_size_ms` / `--window_stride_ms`：STFT窗口与步长（毫秒）
- `--batch`：采样 DataLoader 批大小
- `--num_workers` / `--prefetch_factor`：DataLoader 并行参数
- `--noise_aug`：采样时是否启用噪声增强（默认关闭，建议保持关闭）
- `--allow_online_resample` / `--strict_sample_rate`：采样率处理策略
- `--gpu`：仅用于兼容 dataloader 的 `pin_memory` 判断
- `--max_batches`：最多采样多少个 batch
- `--max_points`：最多保留多少 mel 点用于拟合
- `--num_segments`：分段数
- `--strategy`：`uniform_logx | quantile | powerlaw`
- `--gamma`：仅 `powerlaw` 有效
- `--log_offset`：log 前偏置
- `--sample_seed`：采样随机种子
- `--save_json`：输出 JSON 路径（默认 `dscnn_kws/frontend/artifacts/log_pwl_fit.json`）

### 3) `artifacts/`

用途：
- 存放拟合结果 JSON、扫描结果 CSV 等实验产物；
- 避免与脚本源码同级混放。

---

### 4) `run_log_pwl_grid.py`

用途：批量跑网格实验（段数 × 策略 × seed），并汇总到 CSV。

#### 常用命令

```bash
python -m dscnn_kws.frontend.run_log_pwl_grid \
  --dataset speech_commands_v0.02_sr8k \
  --sample_rate 8000 \
  --epochs 20 \
  --segments 2 4 6 8 12 \
  --strategies uniform_logx quantile \
  --seeds 42 43 44 \
  --extra_train_args "--num_workers 8"
```

#### 参数说明

- `--python`：训练命令使用的 python 可执行文件
- `--dataset` / `--sample_rate`：训练数据配置
- `--epochs`：每次实验训练 epoch
- `--segments`：扫描段数列表
- `--strategies`：扫描策略列表
- `--seeds`：随机种子列表
- `--extra_train_args`：透传给 `dscnn_kws.train` 的额外参数
- `--save_csv`：结果 CSV 输出路径

---

### 5) `run_log_pwl_gamma_scan_seg4.py`

用途：**固定4段**，扫描 gamma（powerlaw 分段），适合你当前实验目标。

#### 常用命令（每个gamma跑满50 epochs）

```bash
python -m dscnn_kws.frontend.run_log_pwl_gamma_scan_seg4 \
  --dataset speech_commands_v0.02_sr8k \
  --sample_rate 8000 \
  --epochs 50 \
  --seed 42 \
  --gammas 0.5 0.6 0.8 1.0 1.2 1.5 2.0 2.5 \
  --extra_train_args "--num_workers 8"
```

#### 参数说明

- `--python`：训练命令使用的 python 可执行文件
- `--dataset` / `--sample_rate`：训练数据配置
- `--epochs`：每个 gamma 的训练轮数（默认 50）
- `--seed`：固定单seed扫描
- `--gammas`：待扫描 gamma 列表
- `--log_offset`：透传给训练脚本的 log 偏置
- `--extra_train_args`：透传给训练脚本的额外参数
- `--save_csv`：结果 CSV 输出路径

输出 CSV 包含：`gamma, loss, acc, precision, recall, f1`，并按 `acc` 降序保存。

---

## 与 `dscnn_kws.train` 的参数对应

以下参数直接影响 frontend 行为：

- `--mfcc_impl`：`torchaudio` 或 `torch`
- `--mel_filter_shape`：`triangular` 或 `rectangular`
- `--log_approx_mode`：`exact` 或 `pwl`
- `--log_pwl_num_segments`
- `--log_pwl_strategy`：`uniform_logx | quantile | powerlaw`
- `--log_pwl_gamma`：仅 powerlaw 生效
- `--log_pwl_fit_json`：加载外部拟合参数
- `--log_offset`
- `--log_input_clamp_min`

建议流程：
1. 先跑 `exact`（torch基线）；
2. 再跑 `pwl + uniform_logx`；
3. 若固定4段掉点明显，用 `run_log_pwl_gamma_scan_seg4.py` 扫 gamma。
