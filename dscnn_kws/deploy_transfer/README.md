# 部署导出工具链（deploy transfer）

本目录是当前指定的模型导出/部署比对工具链，取代早期自写的"strict 整数前端 + ONNX"
导出脚本。项目约定：**对外交付模型的导出一律使用本目录脚本**，不得回退到旧的自写
导出链路。

## 两条导出链

| 脚本 | 前端 | 后端 | 适用模型线 |
|---|---|---|---|
| `export_float_onnx.py` | TorchMFCC 浮点逐算子精确复刻（pre_emphasis 0.97 → reflect pad → hann DFT → 40 三角 mel[20,8000] → log → ortho DCT 前 13 维 → `[1,13,32]`） | 浮点 DSCNN（L5/C64） | reclean 线全部浮点模型（v2/v3/v24/i 系/v344/v419…，训练前端均为三角 TorchMFCC） |
| `export_v6_1_strict_int8_onnx.py` + `onnx_strict_integer_mfcc.py` | strict 整数矩形带 MFCC（QAT 训练所用整数前端的逐算子复刻） | QDQ QAT INT8 backbone | strict 整数前端线（hi_xiaowen 家族 c11 系等） |

两条链的输入/输出契约一致：`waveform [1, 16000]` float32（单声道、归一化 16 kHz、
1 s）→ `logits [1, 2]`（positive 为 class index 0），与 demo 应用契约对齐。

## 前端匹配铁律（QAT/量化必读）

**给模型做 QAT/量化时，spec 必须复刻该模型自己训练时使用的前端**：

- reclean 线（三角 TorchMFCC）模型 → QAT 必须用三角 mel spec；
- strict 整数线（矩形带）模型 → QAT 必须用 strict 整数矩形 spec。

反例（v6.1 事故）：三角浮点模型被用 strict 整数矩形 spec 送去 QAT，结果量化后
召回率因分布饱和而"虚高"，真实场景误唤醒全面恶化 1.4–4 倍。前端不匹配属于系统性
错误，不会在训练集指标上暴露。

## 滑窗比对（官方评估协议）

- `slidewin_compare.py`：在 mic243 等 243 段实采录音上对比"浮点链 vs strict INT8
  链"。协议为 1 s 窗口、96 ms hop、逐窗 softmax positive 分数，统计各阈值下的
  最长连续高分段、高分碎片数（锯齿指标）、峰值、一阶差分，以及按标注的命中
  （标注中点 ±0.4 s 内存在过阈值窗口）。
- `plot_slidewin_compare.py`：将比对 JSON 画成分数时间线对比图。

注意：`slidewin_compare.py` 顶部写死了服务端路径（`/home/chensheng/...`）用于
`sys.path` 注入，换环境时需按本仓库路径调整这两个常量；其余脚本为自包含实现，
仅依赖 torch / onnx / numpy / matplotlib。

## 用法

```powershell
# 浮点链导出（reclean 线 checkpoint → full ONNX）
python export_float_onnx.py --checkpoint <best.pt> --output_dir <dir>

# strict 整数 INT8 链导出（QAT QDQ backbone → INT8 ONNX）
python export_v6_1_strict_int8_onnx.py --help

# 滑窗比对（服务端执行，输入为 243 段实采录音 + 两条链的模型）
python slidewin_compare.py --help
```

导出后务必用 `checkpoints/*.metadata.json` 的格式记录 SHA-256、输入/输出契约与
验证范围，再进入交付流程。
