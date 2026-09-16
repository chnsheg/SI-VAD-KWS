# SI-VAD-KWS

联合语音活动检测（VAD）与唤醒词检测（KWS）研究/部署代码库。VAD 基准来自
`vadbench`，KWS 保持上游 SI-dscnn-kws 的 `dscnn_kws` 目录结构，并收录当前已验证的
DSCNN、流式 CRNN、MFCC 前端、ONNX 导出、QAT 量化和部署比对实现。

本仓库为**交接发布仓库**：只收录可复现的核心实现与精选部署产物。数据集、原始
音频、训练 runs、缓存、扫描结果、AI 过程文件（计划/规范/临时实验目录/废弃测试）
一律不入库。

## 目录

```text
configs/              VAD 训练与评估配置
vadbench/             VAD 算法、特征、数据清单和指标实现
dscnn_kws/            KWS 主包：训练、数据管线、前端、模型、量化、流式、demo
dscnn_kws/deploy_transfer/  当前指定的部署导出/链路比对工具链（含前端匹配铁律）
dscnn_kws/tests/      KWS 数据管线与训练器测试（pytest）
tools/                四卡训练启动器与 v3 数据生成交接脚本
checkpoints/          精选部署 checkpoint 及其元数据（白名单入库）
tests/                VAD 合约测试
paper/                VAD 相关公开论文
```

## 链路总览（前端 → 模型 → 训练 → 评估 → 导出 → 量化 → 部署）

**前端与模型系谱**（接手必读，两条线的训练前端不同，不可混用）：

| 模型线 | 训练前端 | 模型 | 说明 |
|---|---|---|---|
| reclean 线（v2/v3/v24/i 系/v344/v419…） | TorchMFCC **三角** mel（浮点，pre_emphasis 0.97，32 ms 窗/步，40 三角 mel[20,8000]，log，DCT 取 13 维 → `[13,32]`） | DSCNN L5/C64（22530 参数，2 类） | 全语料从零 CE 训练，保持跨说话人泛化；当前交付线 |
| model_hi_xiaowen 家族（v6_1/v6_2/0911/0914 及其 QAT） | **strict 整数矩形带** MFCC（平台整数前端） | 同上 | 平台侧训练 + QAT 线；`checkpoints/` 内两个 `*_int8_qdq.onnx` 属于该线 |

**QAT 前端匹配铁律**：给模型做 QAT/量化时，spec 必须复刻该模型**自己训练时**的前端
（reclean 线 → 三角；整数线 → strict 矩形）。反例教训：v6.1 浮点模型（三角）被用
strict 矩形 spec 量化，量化后召回因分布饱和"虚高"，真实场景误唤醒恶化 1.4–4×。
详见 `dscnn_kws/deploy_transfer/README.md`。

**训练**：`dscnn_kws.train`（支持 packed mixture：base pack + raw anchors + 硬负
pack + 低信噪比正例 pack，margin/DAAT 目标可选，当前 v419 配方为 `loss_type=ce`）。
四卡启动器 `tools/run_kws_reclean_four_gpu.py`。数据管线 `dscnn_kws/data/`
（reclean 数据集构建、packed PCM、硬负挖掘、Mobvoi/GSC 清单生成、噪声工具）。
训练 batch 必须保持严格 1:1 正负平衡。

**评估（官方协议）**：1 s 窗口、96 ms hop、双窗确认、softmax 正类分数、**不用
debounce 掩盖误唤醒**。指标定义见 `dscnn_kws/METRICS_README.md`；部署式评估
`dscnn_kws/eval_deployment_kws.py`、FAH/FRR 评估 `dscnn_kws/eval_fah_frr.py`、
浮点/INT8 链滑窗比对 `dscnn_kws/deploy_transfer/slidewin_compare.py`。

**导出**：一律使用 `dscnn_kws/deploy_transfer/`（浮点全量 ONNX 与 strict 整数
INT8 ONNX 两条链，契约均为 `waveform [1,16000] → logits [1,2]`，positive 为
class 0），不要回退到早期自写导出脚本。

## 当前状态（2026-09-15 交接基线）

- **reclean 线交付候选：v419**（`checkpoints/mobvoi_nihao_wenwen_v419_a_reclean_v2_fp32.pt`）。
  标准阈值 0.5 下实测：cs 召回 99.4%、zh 91.0%（yd 用一声"hi"读法，不唤醒是正确
  行为，不计入目标）；真实场景 FAH ≈ 1982/h，**误唤醒压到接近 0 的工作仍在进行**，
  这是项目收口的硬性要求。训练时已排除 cs/zh/yd 真人录音，防止按人拟合；模型必须
  泛化到任意新说话人，拒绝按人重训。
- **QAT 线现状**：v6_2 浮点 + phase_b c11_seed44 为当前最优配对（@0.5 命中 97.5%，
  建议交付阈值 0.5–0.55）；v6_1 QAT 因前端错配误唤醒恶化 1.4–4×，不可交付；0911
  候选存在浮点分数平坦化问题，需先解决再调 QAT。QAT 产物存于工作机
  `kws_deploy_transfer/models_qat/`（不入库）。
- **已知结论（避免重走弯路）**：PCEN/混合 PCEN 前端被否（唤醒崩塌，能量先验是
  -15 dB 唤醒机制的关键）；DAAT 域对抗无效（对照实验证实）；三窗确认结构上消灭
  不了 pub/road 噪声墙；i22_a65 的低 FAH 与说话人特化是同一权重结构，二者不可兼得，
  只能从零重训恢复；正例噪声混合 SNR ≥5 dB 会让语音域 FAH 爆炸（<-5 dB 混合是
  压误唤醒的承重成分）；TAU 硬负挖掘必须密集（≥0.3 全窗、每文件封顶 20）。

## 训练服务器（训练与数据所在）

训练与数据在内部 GPU 服务器上执行（本仓库不含数据；主机地址与账号走内部渠道交接，
不写入仓库）。目录布局约定：

```text
vad_kws_datasets/                      数据集根（kws_reclean_aug_v1 等）
kws_packs/                             packed mixture / 硬负 / 低 SNR 正例 pack
kws_training_runs/                     训练 runs（kws_reclean_v2/v419_a_20260915 等）
vad_kws_sources/reclean_v3/kws_trainer_20260910_v21bucket
                                       训练器快照（已同步入本仓库 dscnn_kws/）
```

使用规则：数据一律放 home 盘，不放 `/dev/shm`（共享机器会被清理）；不要用 GPU0
（他人进程占用）；Python 用 demucs 环境（含 torchaudio）。

## 环境与快速检查

VAD 依赖位于根目录 `pyproject.toml`；KWS 依赖位于 `dscnn_kws/requirements.txt`。
建议 Python 3.10+、PyTorch、torchaudio、NumPy、SciPy、scikit-learn 和 soundfile。

```powershell
python -m unittest discover -s tests          # VAD 合约测试
python -m pytest dscnn_kws/tests -q           # KWS 数据管线/训练器测试
python -m compileall vadbench dscnn_kws       # 语法检查
python -m vadbench.cli list-algorithms        # VAD 算法列表
```

KWS 训练入口（本地冒烟；真实训练在服务端四卡执行）：

```powershell
python -m dscnn_kws.train --root .\dataset --dataset speech_commands_v0.02 --sample_rate 16000 --epoch 1 --num_workers 0
```

部署 checkpoint 的输入/输出契约、SHA-256 和验证范围见 `checkpoints/*.metadata.json`。

## 来源与范围

KWS 基础工程来源于 [iCharose/SI-dscnn-kws](https://github.com/iCharose/SI-dscnn-kws)
（上游原始说明见 `dscnn_kws/README_original.md`），本仓库只提交可复现的核心实现和
精选部署产物；本地 AI 辅助过程、任务清单、执行规范、临时实验目录和废弃测试均未
纳入版本控制。工作区里的 `analysis/`、`tmp/`、`tools/`（工作机顶层临时分析工具）、
`deliveries/` 等目录为过程材料，不属于本仓库。
