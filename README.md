# SI-VAD-KWS

联合语音活动检测（VAD）与唤醒词检测（KWS）研究/部署代码库。VAD 基准来自
`vadbench`，KWS 保持上游 SI-dscnn-kws 的 `dscnn_kws` 目录结构，并收录当前已验证的
DSCNN、流式 CRNN、MFCC 前端、ONNX 导出和量化实现。

## 目录

```text
configs/              VAD 训练与评估配置
vadbench/             VAD 算法、特征、数据清单和指标实现
dscnn_kws/            KWS 训练、推理、前端、模型、流式与部署代码
checkpoints/          精选的 ONNX 部署 checkpoint 及其元数据
tests/                VAD 合约测试
paper/                VAD 相关公开论文
```

数据集、原始音频、训练 runs、缓存和大规模扫描结果不随仓库分发。请先准备本地数据，
再通过 `dscnn_kws/data` 中的清单生成器建立训练/验证/测试清单。

## 环境

VAD 依赖位于根目录 `pyproject.toml`；KWS 依赖位于 `dscnn_kws/requirements.txt`。
建议使用 Python 3.10+、PyTorch、torchaudio、NumPy、SciPy、scikit-learn 和 soundfile。

## 快速检查

```powershell
python -m unittest discover -s tests
python -m compileall vadbench dscnn_kws
python -m vadbench.cli list-algorithms
```

KWS 训练入口：

```powershell
python -m dscnn_kws.train --root .\dataset --dataset speech_commands_v0.02 --sample_rate 16000 --epoch 1 --num_workers 0
```

部署 checkpoint 的输入/输出契约、SHA-256 和验证范围见 `checkpoints/*.metadata.json`。

## 来源与范围

KWS 基础工程来源于 [iCharose/SI-dscnn-kws](https://github.com/iCharose/SI-dscnn-kws)，
本仓库只提交可复现的核心实现和精选部署产物；本地 AI 辅助过程、任务清单、执行规范、
临时实验目录和废弃测试均未纳入版本控制。
