# v6.1 直接 INT8 ONNX 导出

`export_v6_1_int8_qoperator.py` 将 **v6.1 已转换的 FBGEMM INT8 checkpoint** 导出为 ONNX opset 17，并把导出结果与 PyTorch v6.1 FBGEMM 参考逐码值验证。它不会把 checkpoint 重新加载到浮点 DSCNN 后再导出。

默认模式是 `exact`，也是 v6.1 的全阶段 bit-exact 导出路径：模型保留 checkpoint 中的 INT8 权重码值、逐通道 scale/zero point、输入/输出量化参数及 bias，并显式复现 FBGEMM 的重定标、round-to-even、ReLU 饱和和池化规则。

## 前提与输入契约

- 在本项目根目录运行脚本；Python 环境必须可导入 `torch`、`onnx`、`onnxruntime`，并且 PyTorch 必须提供 `fbgemm` 量化后端。
- `--checkpoint` 必须是 v6.1 转换后的 INT8 backbone checkpoint，且包含可严格加载的 `state_dict`；不能传入 QAT prepared checkpoint 或浮点 checkpoint。
- `--spec` 必须是与 checkpoint 配套的严格整数 MFCC JSON 规范。二者作为一个不可拆分的版本对待，报告和 ONNX 元数据会分别记录 SHA-256。
- 模型 I/O 固定为 `waveform: float32[1, 16000] -> logits: float32[1, 2]`，即一秒、16 kHz、batch=1；不支持动态 batch 或动态时长。

## 导出 v6.1 hi_xiaowen

在项目根目录下执行。以下 checkpoint/spec 是当前 v6.1 `hi_xiaowen` 的 `c11_stage_margin_scale_u16/seed_42` 配对产物，所有路径均相对于项目根目录：

```powershell
python dscnn_kws/ONNX/export_v6_1_int8_qoperator.py --checkpoint dscnn_kws/quantization/bit_accurate_mfcc_experiments_v6_1_strict_scale_calibration/qat_runs/phase_a/c11_stage_margin_scale_u16/seed_42/mobvoi_hi_xiaowen_binary_hardneg_L5_C64_layers5_channels64_params22530_noise_best_bit_accurate_mfcc_int8_backbone.pt --spec dscnn_kws/quantization/bit_accurate_mfcc_experiments_v6_1_strict_scale_calibration/qat_runs/phase_a/c11_stage_margin_scale_u16/seed_42/mobvoi_hi_xiaowen_binary_hardneg_L5_C64_layers5_channels64_params22530_noise_best_bit_accurate_mfcc_spec.json --output artifacts/v6_1_hi_xiaowen_exact_int8.onnx --report artifacts/v6_1_hi_xiaowen_exact_int8.report.json
```

CLI 参数如下：

- `--checkpoint`、`--spec`、`--output`、`--report`：必填。
- `--parity-samples N`：验证语料条数，默认 `19`，且不得小于 `9`。
- `--keep-staging`：失败或成功后保留临时 staging ONNX；默认清理。
- `--mode {exact,qoperator}`：默认 `exact`。

`--output` 与 `--report` 不能解析为同一个文件，发生冲突时命令直接以退出码 `2` 结束，不改写已有 ONNX。

## 两种模式

### `exact`（默认）

这是 bit-exact 导出模式。其 ONNX 采用标准 ONNX 语法，并直接嵌入 checkpoint 的 10 份 INT8 权重码值（9 个卷积和 1 个全连接）及其量化参数。由于 ORT CPU 没有可用的 `ConvInteger` 执行实现，backbone 的累加使用标准浮点 `Conv`/`Gemm`：先将 UINT8/INT8 码值中心化，再计算有界整数乘加。导出器会证明每层累加上界小于 `2^24`，因此这些整数在 float32 中可精确表示；这不是浮点模型导出，也不改变 checkpoint 的权重码值或量化语义。

精确图不保留 `QLinearConv`、`QLinearGlobalAveragePool` 或 `QGemm`；它使用 9 个上述受限 `Conv`、1 个 `Gemm`，并且仅在最终 INT8 logits code 后执行一次 `DequantizeLinear`。

### `qoperator`（仅诊断）

`--mode qoperator` 让 ONNX Runtime 以扩展图优化 materialize 常规 QOperator 图，包含 9 个 `QLinearConv` 和 1 个 `QLinearGlobalAveragePool`。它仅用于定位与比较普通量化 ONNX 的差异；该模式只检查最终 float32 logits word，不检查 13 个中间 code 阶段。针对本 v6.1 FBGEMM checkpoint，预期不能通过该门禁，失败时不会发布候选文件；无论其结果如何，都不能作为全阶段 bit-exact 的部署依据。

例如诊断命令只需在上例末尾加入：

```powershell
--mode qoperator
```

## 实时严格缓存 bundle

`export_v6_1_int8_runtime_bundle.py` 一次导出实时级联所需的三个标准 ONNX 图及其独立认证报告。
它仍直接加载同一个已转换的 FBGEMM INT8 checkpoint，不经过浮点 DSCNN 回退，也不需要自定义算子或
动态库：

```powershell
python dscnn_kws/ONNX/export_v6_1_int8_runtime_bundle.py `
  --checkpoint dscnn_kws\quantization\bit_accurate_mfcc_experiments_v6_1_strict_scale_calibration\qat_runs\phase_a\c11_stage_margin_scale_u16\seed_42\mobvoi_hi_xiaowen_binary_hardneg_L5_C64_layers5_channels64_params22530_noise_best_bit_accurate_mfcc_int8_backbone.pt `
  --spec dscnn_kws\quantization\bit_accurate_mfcc_experiments_v6_1_strict_scale_calibration\qat_runs\phase_a\c11_stage_margin_scale_u16\seed_42\mobvoi_hi_xiaowen_binary_hardneg_L5_C64_layers5_channels64_params22530_noise_best_bit_accurate_mfcc_spec.json `
  --output-dir artifacts `
  --stem v6_1_hi_xiaowen_exact
```

成功时写入：

- `v6_1_hi_xiaowen_exact_frontend.onnx`：`waveform float32[1,16000] -> strict_mfcc_codes uint8[1,320]`。
- `v6_1_hi_xiaowen_exact_backbone.onnx`：`strict_mfcc_codes uint8[1,320] -> logits float32[1,2]`。
- `v6_1_hi_xiaowen_exact_frame_repair.onnx`：`waveform float32[1,16000] ->` 五个边界帧的 50 个严格 code。
- `..._split.report.json`、`..._frame_repair.report.json` 与 `..._runtime_bundle.report.json`。

bundle 要求至少 19 条认证波形，且 split/repair 的 checkpoint SHA-256、spec SHA-256 和来源完整
严格 ONNX SHA-256 必须完全一致，否则不发布 runtime bundle。demo 只能在这些报告为零 mismatch 且
ONNX I/O/工件哈希均匹配时启用 `split_cached`。

缓存的适用条件是相邻两个 1 s KWS 窗口严格连续且相差 1536 个采样点（96 ms）。在当前严格前端的
`n_fft=512`、`hop=512`、`center=true` 语义下，前窗口第 4--30 帧可直接成为下一窗口第 1--27 帧；
受 reflect padding、预加重边界和新 PCM 影响的第 0、28--31 帧由 repair 图重算。首个窗口、任意
不连续窗口或 reset 后均回退到完整严格 frontend。这是对代码缓存的优化，不是对量化参数、舍入规则或
backbone 的修改。

## 严格发布门禁

默认 `exact` 用确定性的 19 条固定形状波形验证：静音、正弦、噪声、正负极值、超范围饱和、交替极性和 PCM/requantization 边界候选，以及固定随机种子样本。它在 ONNX Runtime `CPUExecutionProvider` 上与 PyTorch v6.1 FBGEMM 参考比较：

- 全部 13 个 UINT8 code 阶段：`input_quant`、9 个卷积输出、`global_avg_pool`、`fc_input`、`final_fc`。
- 最终 `logits` 的 float32 数值，以及每个 float32 的原始 32-bit word。

发布成功要求所有阶段 code mismatch 为零、`float_word_mismatch_count == 0`、`logit_mismatch_count == 0` 且 `max_abs_error == 0.0`。任一条件失败，候选文件不会替换 `--output`；若目标已有 ONNX，会保持原文件不变。CLI 仍会向 `--report` 写入失败报告并返回退出码 `1`。

为防止并发导出互相覆盖，脚本在发布前为输出路径获取非阻塞的同名锁文件 `.<output-name>.lock`。已有活动持有该锁时会失败；遗留的锁文件本身不会阻止后续导出，锁由操作系统释放。

## 产物与报告

`--output` 成功后生成 ONNX，包含以下模型元数据：

- `format`：`ONNX opset 17 direct INT8 parameters with exact FBGEMM lowering`
- `exporter_version`
- `source_checkpoint_sha256`、`source_spec_sha256`
- `verification_scope`
- `checkpoint_parameter_source` 和最终 INT8 `exact_logit_codes` 张量名

`--report` 始终输出 JSON。成功报告包括 checkpoint/spec/ONNX 的路径和 SHA-256、候选 ONNX SHA-256、`artifact_published`、PyTorch/ONNX/ORT 版本、模式、图算子统计、10 个 INT8 权重张量与直接 checkpoint 参数哈希、固定 I/O、每阶段 mismatch 计数、严格 parity 结果，以及 `float32_accumulator_bounds`。后者逐层给出在当前 checkpoint 的中心化码值域中可达到的绝对累加上界；`exact` 只有所有这类上界严格小于 `2^24` 时才会导出。失败报告至少包含失败状态、路径、`strict_parity_passed: false`、异常类型与异常消息；在已经完成比较的失败中也会保留候选哈希、图统计和 mismatch 信息。

## NPU 与目标平台

本导出器的 bit-exact 结论仅覆盖导出时的 PyTorch v6.1 FBGEMM 参考与 ONNX Runtime `CPUExecutionProvider`。尽管 `exact` 图只使用标准 ONNX 语法，报告会标记 `requires_target_bit_exact_certification: true`，且不会把它声明为可移植的通用高性能图。

目标 NPU 可能不接受该图、改变 `Conv`/`Gemm` 的数值路径，或融合/替换 `Round`、`Clip` 等算子。因此每个 NPU 编译器、固件和优化配置都必须以同一批输入重新做 13 个 code 阶段和最终 float32 word 的逐项认证；未通过时不得宣称 bit-exact，也不能以 QOperator 诊断图替代 `exact` 图。

同样的限制适用于 runtime bundle：其 frontend code、repair code 和 backbone float32 logits 在 ORT CPU
认证通过，不能推出任意 NPU 的 bit-exact。目标侧应保留/导出对应中间 code，在目标编译后的模型上用同一
认证语料逐项比较；若目标无法暴露该调试输出，至少要将其视为数值一致性而非 bit-exact 发布。
