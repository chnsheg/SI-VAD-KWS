# NAS 使用说明（dscnn_kws/nas）

本目录实现了一个面向 1s@8k KWS 的 NAS 原型：

- 搜索方式：随机初始化 + 进化搜索
- 多目标：准确率（越高越好）+ 乘法次数 mults（越低越好）+ Params（越低越好）
- 约束方式：硬约束（如 mult 上限）+ Pareto 前沿筛选

---

## 目录与文件说明

- `search_space.py`
  - 定义架构基因（LayerGene、NASArchitecture）
  - 定义候选算子空间：`conv2d / dsconv2d / dsconv1d / eca`
  - 随机采样与变异
  - 前两层采用“稳妥 stride 设计”（首层大步长 + 第二层修正）

- `constraints.py`
  - 逐层 shape / mults / Params 估计
  - 硬约束检查（mults、参数量、stride/kernel、通道跳变等）

- `model.py`
  - 将 NAS 架构基因实例化为可训练网络（`NASKWSModel`）
  - 内含 `DSConv2d`、`DSConv1dAs2d`、`ECA2d`

- `evaluator.py`
  - 单候选训练评估（默认每候选训练若干 epoch）
  - 返回验证集精度与损失

- `pareto.py`
  - 非支配排序（Pareto front）

- `evolution.py`
  - 搜索主流程（init + evolution）
  - 每个候选打印日志（进度、原因、acc/mults/params）
  - 周期性输出摘要（有效数、拒绝数、Pareto数量、当前最好候选）
  - 导出 `search_results.jsonl`

- `run_search.py`
  - 命令行入口
  - 负责保存：`config.json / summary.json / best.json / pareto.json / topk.json`

- `__init__.py`
  - 模块导出

---

## 输出文件说明（`--nas_out_dir`）

- `config.json`：本次运行参数快照（便于复现）
- `search_results.jsonl`：每个候选完整记录（含 reason、结构）
- `topk.json`：按 acc 排序的 Top-K 候选
- `pareto.json`：最终 Pareto 前沿候选
- `best.json`：最佳候选（当前规则为 topk[0]）
- `summary.json`：总体统计（总候选数、有效/拒绝、拒绝原因计数、best 指标）

---

## 运行命令

### 1) 冒烟测试（快速验证流程）

```bash
python -m dscnn_kws.nas.run_search \
  --dataset speech_commands_v0.02_sr8k \
  --sample_rate 8000 \
  --gpu 1 \
  --nas_num_layers 6 \
  --nas_init_samples 20 \
  --nas_total_samples 40 \
  --nas_epochs_per_candidate 2 \
  --nas_topk 5 \
  --nas_log_interval 10 \
  --nas_out_dir dscnn_kws/nas/runs/smoke_v2
```

### 2) 标准搜索（与你的设定一致）

```bash
python -m dscnn_kws.nas.run_search \
  --dataset speech_commands_v0.02_sr8k \
  --sample_rate 8000 \
  --gpu 1 \
  --nas_num_layers 6 \
  --nas_init_samples 120 \
  --nas_total_samples 300 \
  --nas_epochs_per_candidate 10 \
  --nas_topk 20 \
  --nas_mult_limit 2200000 \
  --nas_mult_limit_parent 3000000 \
  --nas_sampling_max_tries 30 \
  --nas_log_interval 20 \
  --nas_out_dir dscnn_kws/nas/runs/standard
```

---

## 参数说明（核心可调）

- `--nas_num_layers`：搜索网络层数
- `--nas_init_samples`：随机初始化候选数
- `--nas_total_samples`：总候选评估数（含 init + evolution）
- `--nas_epochs_per_candidate`：每候选训练轮数（你当前建议 10）
- `--nas_mutation_prob`：进化变异概率
- `--nas_topk`：导出 Top-K 数量（你当前建议 20）
- `--nas_mult_limit`：乘法次数硬约束（默认 2.2M，最终候选必须满足）
- `--nas_mult_limit_parent`：母体缓冲阈值（允许少量超最终阈值但参与进化）
- `--nas_param_limit`：保留参数位（当前自由搜索模式下不作为淘汰条件）
- `--nas_sampling_max_tries`：可行性优先采样最大尝试次数
- `--nas_t_target`：前端目标时间维（默认 32）
- `--nas_f_target`：前端目标频率维（当前实现默认 16）
- `--nas_log_interval`：日志摘要输出间隔
- `--nas_out_dir`：结果目录

训练相关通用参数：`--batch --lr --weight_decay --gpu --seed ...`

---

## 当前实现注意事项

1. 当前 NAS 模型将波形重排为二维输入 `[B,1,T',F]`（默认 `F=16`）后再做搜索。
2. `dsconv1d` 在当前实现中为“沿频率维卷积”的 2D 兼容算子。
3. `build_dataloaders` 的 split 统计 INFO 已注释，减少 NAS 过程噪声输出。
4. 搜索默认以 `2.2M` 乘法次数为目标上限，`3.0M` 仅作母体缓冲，不会进入最终 `topk`。
5. 当前为 **MFCC-only + mult约束** 模式：前端搜索 `window/stride/n_mfcc`，主干在 MFCC 输出上搜索。

---

## 当前统计口径

当前统一统计的是 **乘法次数（mults）**，不是 MAC/FLOP。

### MFCC 前端
- Hann窗：`n_frames * n_fft`
- FFT：`n_frames * (n_fft / 2) * log2(n_fft) * 4`
- 功率谱：`n_frames * (n_fft // 2 + 1) * 2`
- MEL：按当前实现记为 `0`
- LOG：`n_frames * (n_fft // 2 + 1)`
- DCT：`n_frames * n_mfcc * (n_fft // 2 + 1)`

### 主干算子
- `conv2d / dsconv2d / dsconv1d / eca` 均只统计乘法次数
- `params` 仍统计卷积/注意力核参数，不含 BN

---

## 推荐后续步骤

1. 先跑 `smoke_v2`，检查：
   - 控制台是否出现候选日志 + 周期摘要
   - 输出目录是否生成 6 类文件
2. 再跑 `standard`，得到 Top20/Pareto
3. 针对 `topk.json` 做全量重训（建议另写 `retrain_topk.py`）
4. 比较重训后指标，筛选最终电路实现候选（优先小参数 + 低MAC + 稳定精度）

---

## 搜索后完整复训（按主训练策略）

如果你已经用搜索命令生成了 `topk.json`，可直接执行：

```bash
python -m dscnn_kws.nas.retrain_topk \
  --topk_json dscnn_kws/nas/runs/smoke_free_mac/topk.json \
  --dataset speech_commands_v0.02_sr8k \
  --sample_rate 8000 \
  --gpu 1 \
  --epoch 60 \
  --batch 256 \
  --lr 0.001 \
  --weight_decay 1e-6 \
  --out_dir dscnn_kws/nas/runs/smoke_free_mac/retrain60
```

输出内容：
- `retrain_results.json`：每个候选的 `best_valid_acc / best_epoch / test_acc_at_best`
- `retrain_config.json`：本次复训参数
- `<uid>.pt`：每个候选在最佳验证精度下的模型权重

---

## MFCC-only 搜索空间说明

当前 NAS 仅搜索 MFCC 前端与主干网络，不再使用 waveform reshape。

- `mfcc_window_ms ∈ {16, 32, 64, 128}`
- `mfcc_stride_ms ∈ {8, 16, 32, 64, 128}`，并始终满足 `stride <= window`
- `mfcc_n_mfcc ∈ {10, 11, 12, 13, 14, 15}`

其中：
- `window_size_ms` 取 2 的幂；
- `n_fft` 在模型内部按 `win_length` 自动提升到最近的 2 的幂；
- 主干网络的输入时间维由 `stride` 决定，输入频率维由 `n_mfcc` 决定。

python -m dscnn_kws.nas.run_search --dataset speech_commands_v0.02_sr8k --sample_rate 8000 --gpu 1 --nas_num_layers 6 --nas_init_samples 120 --nas_total_samples 300 --nas_epochs_per_candidate 10 --nas_topk 20 --nas_mult_limit 2200000 --nas_mult_limit_parent 3000000 --nas_log_interval 20 --nas_out_dir dscnn_kws/nas/runs/mfcc_only_mult

python -m dscnn_kws.nas.retrain_topk --topk_json .\dscnn_kws\nas\runs\mfcc_only_standard\topk.json --dataset speech_commands_v0.02_sr8k --sample_rate 8000 --gpu 1 --epoch 60 --batch 256 --lr 0.001 --weight_decay 1e-6 --out_dir dscnn_kws/nas/runs/mfcc_only_standard/retrain60