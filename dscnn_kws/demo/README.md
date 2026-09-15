# VAD-KWS Terminal Demo

这个目录提供 Windows 终端下的实时 VAD-KWS 级联演示：默认推荐使用最新
`causal-crnn-vad-kws-realneg` VAD ONNX。`pc-deployment` 对实时麦克风 PCM 使用语音频段仅上拉 AGC，
再维护连续 16 kHz PCM：默认每
32 ms 计算能量并在需要时执行 VAD；VAD 确认人声后，从保留的 1.5 秒 PCM 历史按 96 ms 对齐的
左边界开始提交所有完整的 KWS 窗口，直至追上实时输入。KWS 模型输入始终固定为 1 秒 / 16000
样本，后续每积累一个 KWS 周期继续提交一个窗口。两个异步 worker 可并行推理，但结果严格按
PCM 提交顺序回写控制器。能量、VAD、KWS 与确认计数共同组成四级门控。

默认 VAD 模型及其同目录元数据为：

```text
D:\VAD-KWS\handover_artifacts\models\causal-crnn-vad-kws-realneg\model.onnx
```

该 ONNX 是从 `D:\VAD-KWS\VAD\runs\causal_crnn_vad_kws_aishell4_realneg\checkpoints\best.pt`
导出的认证产物；其元数据中记录的检查点 SHA-256 与该训练运行目录一致。该目录保存的是
PyTorch 检查点而非 ONNX 文件，因此 ONNX Runtime 启动参数必须继续指向上面的 `model.onnx`，
不能直接传入 `best.pt`。

也可用以下 20h 模型做 A/B 现场测试；两个 model ID 都会继续执行完整的元数据哈希、
特征契约和 ONNX I/O 校验：

```text
D:\VAD-KWS\handover_artifacts\models\causal-crnn-vad-kws-20h\model.onnx
```

KWS 使用仓库产物：

```text
artifacts\v6_1_hi_xiaowen_exact_int8.onnx
```

默认使用这个完整严格模型。也支持普通的 frontend/backbone split；两种模式都在每次
KWS 调用时独立处理当前完整 1 秒 PCM 窗口。

## 安装与设备

WAV 重放不需要额外声卡依赖。实时麦克风录音安装 `sounddevice`：

```powershell
D:\VAD-KWS\.envs\vadbench-py311-cpu\python.exe -m pip install -r dscnn_kws\demo\requirements.txt
```

列出 Windows 中所有可作为输入的设备：

```powershell
D:\VAD-KWS\.envs\vadbench-py311-cpu\python.exe dscnn_kws\demo\run_demo.py devices
```

耳机麦克风只要出现在该列表中，就和普通麦克风一样使用，不需要 USB/声卡适配层。
`--device` 可接受列表中的数字索引或完整设备名。
若旧式终端将设备名中的特殊字符显示为 `\x..`，直接使用每行左侧的数字索引即可。

## 运行

一键启动实时麦克风和诊断页面：

```powershell
.\dscnn_kws\demo\start_realtime_demo.ps1
```

终端会在服务实际启动后打印 `http://127.0.0.1:19374/`。选择耳机麦克风时：

```powershell
.\dscnn_kws\demo\start_realtime_demo.ps1 -Device 1
```

脚本默认使用 realneg VAD、仓库 v6.1 INT8 KWS、`--web` 与端口 `19374`；其余
参数可在末尾透传，例如 `-Device 1 --kws-threshold 0.85`。端口被占用时 demo
会报告错误并退出，不会静默改用另一个端口。

默认使用 Windows 默认麦克风：

```powershell
D:\VAD-KWS\.envs\vadbench-py311-cpu\python.exe dscnn_kws\demo\run_demo.py listen `
  --vad-model D:\VAD-KWS\handover_artifacts\models\causal-crnn-vad-kws-realneg\model.onnx `
  --kws-model artifacts\v6_1_hi_xiaowen_exact_int8.onnx
```

选择耳机麦克风：

```powershell
D:\VAD-KWS\.envs\vadbench-py311-cpu\python.exe dscnn_kws\demo\run_demo.py listen `
  --vad-model D:\VAD-KWS\handover_artifacts\models\causal-crnn-vad-kws-realneg\model.onnx `
  --kws-model artifacts\v6_1_hi_xiaowen_exact_int8.onnx `
  --device "Headset Microphone"
```

用 20h VAD 进行现场对比时，仅替换 `--vad-model` 路径：

```powershell
D:\VAD-KWS\.envs\vadbench-py311-cpu\python.exe dscnn_kws\demo\run_demo.py listen `
  --vad-model D:\VAD-KWS\handover_artifacts\models\causal-crnn-vad-kws-20h\model.onnx `
  --kws-model artifacts\v6_1_hi_xiaowen_exact_int8.onnx `
  --device 1
```

不使用 PowerShell launcher 时，可使用普通 split KWS。三个 KWS 参数必须作为同一组
提供，CLI 会在打开设备前验证模型哈希、ONNX I/O 和认证报告：

```powershell
D:\VAD-KWS\.envs\vadbench-py311-cpu\python.exe dscnn_kws\demo\run_demo.py listen `
  --vad-model D:\VAD-KWS\handover_artifacts\models\causal-crnn-vad-kws-realneg\model.onnx `
  --kws-frontend artifacts\v6_1_hi_xiaowen_exact_frontend.onnx `
  --kws-backbone artifacts\v6_1_hi_xiaowen_exact_backbone.onnx `
  --kws-split-report artifacts\v6_1_hi_xiaowen_exact_split.report.json
```

重放 WAV 用于可复现实验，不导入或要求 `sounddevice`：

```powershell
D:\VAD-KWS\.envs\vadbench-py311-cpu\python.exe dscnn_kws\demo\run_demo.py listen `
  --vad-model D:\VAD-KWS\handover_artifacts\models\causal-crnn-vad-kws-realneg\model.onnx `
  --kws-model artifacts\v6_1_hi_xiaowen_exact_int8.onnx `
  --input-wav .\example.wav `
  --session-root .\sessions\replay
```

v6.1 `hi_xiaowen` 的训练标签固定为 `positive -> 0`、`negative -> 1`，因此默认
`--kws-positive-index` 为 `0`。接入标签顺序不同的模型时，必须显式覆盖该参数：

```powershell
--kws-positive-index 0
```

`pc-deployment` 默认四级门控参数为：最新 32 ms 的 `rms_dbfs > -33`，VAD `prob > 0.8` 且连续
3 个 32 ms tick，KWS `prob >= 0.50` 且连续 2 个 96 ms tick；默认 KWS 历史为 1500 ms。主 96 ms
窗口的原始分数落入阈值下方 0.15 的灰区时，会额外评估 `+16 ms` 相位；检测到 voiced-quiet-voiced
能量谷时才会同时评估 `-16 ms`。相位补偿后的越阈结果仍必须由稍后一个独立主窗口确认，才会唤醒。
运行中
修改 VAD 周期会同步把 energy 设为 1 倍、KWS 设为 3 倍，因此 VAD=100 ms 时为
Energy=100 ms、KWS=300 ms；之后可单独修改 KWS 周期且不会改变 VAD。确认帧数仍分别为 3 和 2。进入
KWS 后，能量下降沿会启动 1 秒 hangover 计时，但计时结束不会关闭 KWS；只有 VAD 已
确认静音后连续 3 秒未再次确认人声，才关闭 KWS。`pc-deployment` 唤醒后会在当前音频边界
自动 rearm：丢弃上一轮模型状态与未完成作业，但保留持续更新的 PCM 缓存，因此下一轮不会退回到
上一次唤醒时的旧音频，也不会等待重新积累一段历史。

运行中可直接输入以下命令：`status`、`devices`、`reset`、`pause`、`resume`、`quit`。
`pause` 会全面停止麦克风/WAV 输入、PCM 缓存、VAD 和 KWS；`resume` 会清空实时
诊断缓存，并重置 PCM 转换器、级联窗口、确认计数和队列后重新采集，因此暂停前后的
音频不会组成同一个 1 秒模型输入窗口。状态行还显示当前 Energy/VAD/KWS 周期；KWS 分数
和 KWS 推理耗时只在 `kws_active` 时显示。状态切换与唤醒会额外打印事件行。

仓库已安装为 Python 包或使用常规 Python 环境时，等价入口是
`python -m dscnn_kws.demo`。当前项目的 `vadbench-py311-cpu` 使用隔离的
`python311._pth`，因此推荐始终使用上面的 `run_demo.py` launcher。

## 实时诊断页面

在 `listen` 后加入 `--web`，终端会打印固定的本机 URL
`http://127.0.0.1:19374/`。录音、VAD/KWS
分数、有效阈值、状态区间和唤醒标记使用同一个单调 `captured_ns` 时间轴。系统保留页面和模型共用的
16 kHz 单声道 PCM；实时麦克风页面、播放和下载则共用采集回调收到的设备格式 WAV 时间轴（双通道 48 kHz/24-bit
设备将显示、播放并导出为双通道 48 kHz/24-bit PCM WAV），不经本程序重采样或混音。下载不会在服务端生成文件。
选区超出历史缓存时，页面会提示重新选择。

实时能量门从开启变为关闭后，VAD 默认仍持续计算 1 秒，用于保留因果模型在语音尾部才输出的分数；这段保持期本身不会运行 KWS，只有 VAD 按既有阈值和确认次数确认人声后才会打开 KWS 门。

### 麦克风输入与自动增益

实时设备优先请求最多 2 个输入声道；下载选区会保留回调收到的双声道原始设备 PCM（例如
48 kHz/24-bit PCM WAV）。模型仍只使用第 0 声道并转换为 16 kHz 单声道输入；不做双声道融合、
阵列选通或 MVDR，因此不会把采集声道混入既有 VAD/KWS 模型路径。

页面的 `级联` 区可分别设置 VAD/KWS 阈值、能量门、VAD 门、VAD/KWS 调度周期与 KWS 历史；默认值为
`0.80/0.50`、两个门均开启、`32/96 ms`、`1500 ms`。KWS 历史可在 1000 到 2000 ms 间以 100 ms
设置。这些值保存在独立的 `localStorage` 项中，点击
`应用级联设置` 后才会提交。周期是模型调用调度周期，不会改变 VAD 训练时的 25 ms/10 ms
log-Mel 帧或 KWS 固定的一秒输入窗口；配置切换同样会在音频安全边界清空级联状态。

当前 `pc-deployment` 的实时麦克风、WAV 回放和浏览器导入分析均暂时使用原始 PCM，AGC 不参与
VAD/KWS 推理；48 kHz 输入仍经过状态保持的抗混叠降采样。AGC 实现和页面参数保留用于后续
独立 A/B 实验，重新启用前必须先完成距离、噪声和误唤醒回归评估。

输入状态行中的 `增益`、`峰值`、`KWS 队列`、`过载`、`过期` 都是诊断值：队列是待执行 KWS
窗口数，过载是满队列时有序记为失败的窗口数，过期是配置切换、暂停或丢帧后被 generation
丢弃的旧结果数。它们不会参与时间轴坐标计算。

实时麦克风路径使用两个独立 KWS ONNX worker；每个 ONNX session 的 intra/inter-op 线程数均为
1，避免两个 worker 再各自创建大线程池。主线程始终持续执行重采样、能量和因果 VAD；
worker 完成顺序不同也会按 KWS 提交顺序、原始 `captured_ns` 写回诊断时间轴。队列满会显式报告
过载，而不会静默跳过音频或修改当前 KWS 调度周期。暂停的导入/重分析会等待已经提交的 KWS
任务完成后再报告结束。

`导入 WAV` 兼容旧的 16 kHz 单声道 PCM WAV，也接受标准 16/24/32-bit PCM 单声道或双声道 WAV；
双声道导入只取第 0 通道进入既有单声道重采样与模型路径。导入或`重新分析选区`会使用已经
应用的运行时配置，先淘汰当前保留的实时音频与诊断结果，再把命令音频作为新的时间轴段分析。
导入和重分析都不会在服务器端创建音频文件；持久化遥测仍必须显式指定 `--record-telemetry`。
实时采集暂停时，`导入 WAV` 与 `重新分析选区`仍可执行：它们不会恢复或读取已暂停的麦克风/WAV
输入，但同样会替换诊断时间轴。

页面始终以 `int32` 容器请求 WASAPI 独占模式并关闭自动转换，设备列表只显示 WASAPI 输入端点；
如果独占流无法打开，服务会返回具体的 PortAudio 错误，不会回退到共享模式。Windows“音频增强”及
Realtek/Nahimic/DTS 等 APO 仍由录音设备属性和厂商软件控制，须在系统中按需关闭。任何 WASAPI
路径都不能证明驱动或硬件 ADC 之前没有 DSP；需要该保证时使用厂商 ASIO/SDK 或硬件的无 DSP
采集模式。

```powershell
D:\VAD-KWS\.envs\vadbench-py311-cpu\python.exe dscnn_kws\demo\run_demo.py listen `
  --vad-model D:\VAD-KWS\handover_artifacts\models\causal-crnn-vad-kws-realneg\model.onnx `
  --kws-model artifacts\v6_1_hi_xiaowen_exact_int8.onnx `
  --device 1 `
  --web
```

默认保留最近 120 秒已转换的单声道 16 kHz PCM（约 7.7 MB 内存），并在后续音频到达时
淘汰最早内容。可调整时长或显式覆盖端口；端口 `0` 仍可要求系统自动选择，历史范围为
1 到 3600 秒：

```powershell
--web-history-seconds 120 --web-port 0
```

浏览器页面包含 `暂停`/`继续` 控件和六条同尺度轨道：模型输入 16 kHz 单声道 PCM、能量、VAD 分数与阈值、KWS
分数与阈值、KWS 实际 1 秒模型输入窗口及其提交时刻、级联状态与唤醒标记。摘要栏显示已加载
模型、固定输入大小、当前 hop 和历史长度。回放和下载均为原始 PCM。摘要栏明确显示
`VAD 3 / KWS 2` 的确认要求。暂停时已有
曲线和回放音频冻结可查看；恢复后开始新的实时缓存段。KWS 曲线的空档表示该时刻没有调用
KWS，并不表示 KWS 分数为零。评分点包含对应的能量门、VAD 确认状态、hangover 与无语音
定时器字段，悬浮信息对应其模型输入窗口末端。`自动跟随` 可跟随最新数据；要保留并回放或
下载历史选区时，先由用户关闭该选项，再拖动时间轴或调整选择范围。任何选择操作都不会暂停采集
或提交运行时配置。

页面的 `VAD 周期` 与 `KWS 周期` 是运行时调度的两个时间设置。VAD 改动会自动把 KWS
改为三倍；只改 KWS 时 VAD 保持不变。提交整数毫秒值（最小 10）后，页面会把这对周期
保存在浏览器本地存储，并在音频线程的安全边界应用。KWS 为 VAD 的整数倍时继续使用 VAD
tick 计数路线；其他值按 16 kHz PCM 的精确样本边界调度。切换会清空 PCM 历史、VAD CNN/GRU
状态、特征缓存、确认计数和定时器，随后等待新的连续 1 秒音频；不会改变 VAD 模型的 25 ms
分帧或 10 ms hop。

该功能只绑定 `127.0.0.1`，不请求浏览器麦克风权限。默认采集不会创建会话目录或写入
`events.jsonl`、`manifest.json`、`summary.json` 或触发音频；状态化 VAD 所需的适配 ONNX
只会写入进程临时目录，并在 demo 停止时清理。选择音频仅在 demo 运行期间通过本机
`/api/audio` 返回，停止 demo 后缓存和 HTTP 服务都会释放。

## 会话记录

仅在 `listen` 显式指定 `--record-telemetry` 时，demo 才会在 `--session-root` 下创建独立目录，包含：

- `manifest.json`：模型路径/哈希、CPU provider、解析后的配置和输入来源。
- `events.jsonl`：设备打开、控制命令、丢帧、VAD/KWS 调用、状态切换、唤醒、错误与停止事件。
- `summary.json`：运行时长、事件与调用计数，以及队列、前端、ORT 和总处理时延的 P50/P95/P99。
- `trigger-*.wav`：仅同时指定 `--record-telemetry --save-trigger-audio` 时产生。保存唤醒时对应的原始 PCM 窗口；异步 KWS 也按提交时的时间戳快照保存。

默认 JSONL 不保存原始 PCM，且默认根本不创建 JSONL。需要保留诊断记录与唤醒片段时，使用：

```powershell
D:\VAD-KWS\.envs\vadbench-py311-cpu\python.exe dscnn_kws\demo\run_demo.py listen `
  --vad-model D:\VAD-KWS\handover_artifacts\models\causal-crnn-vad-kws-realneg\model.onnx `
  --kws-model artifacts\v6_1_hi_xiaowen_exact_int8.onnx `
  --record-telemetry --save-trigger-audio
```

`--save-trigger-audio` 不能单独使用；没有 `--record-telemetry` 时启动会被拒绝。启用持久化或保存
触发音频前，应确认本地音频留存策略符合使用场景。

## 运行范围

VAD 前端按照模型元数据复现 16 kHz、25 ms 帧、10 ms hop、64-bin causal
log-Mel 与窗口内归一化。VAD/KWS 都在 CPU `ONNX Runtime` 上验证。v6.1 KWS 的
逐 logit/中间 INT8 code 对齐保证仅覆盖导出报告中定义的 PyTorch v6.1 FBGEMM 与
ONNX Runtime CPU 确定性语料；其他执行提供方或目标设备必须逐目标
重新认证，不能从本 demo 推断 bit-exact。

控制器的默认 32 ms/96 ms 是业务调度周期，而非模型前端帧长；页面可将 VAD/KWS 设置为任意
有效的独立周期，默认联动关系为三倍。VAD 始终使用训练时的 25 ms/10 ms causal log-Mel，KWS
始终使用 v6.1 的严格整数前端，并且 KWS 不使用 VAD 特征。每个 KWS 调用都对最新完整 PCM 窗口
独立执行前端和模型，因而支持运行时调度变化。

PC deployment 控制器保留用于 VAD 的尾随 1 秒 PCM 窗口，并独立保留默认 1.5 秒的 KWS 历史。
VAD 启动时先用整个窗口从零状态顺序更新因果状态，但仅将窗口末尾属于当前 32 ms tick 的
posterior 归约为该 tick 分数；后续每个 tick 对 feature cache 实际新产生的 posterior 取最大值。
25 ms 帧与 10 ms hop 不整除 32 ms，因此每个 tick 会产生 3 或 4 个 posterior。KWS 模型每次
接收完整 1 秒 PCM 并输出单个二分类分数；96 ms 只规定相邻完整窗口的左边界步长，不代表等待
96 ms 后才开始推理。VAD 确认后，控制器立即将已有历史中的完整窗口按同一游标快速提交，并在
worker 推理期间继续接收 PCM；处理完成后再从未提交的游标继续追赶，所以不会因为历史回溯而丢失
已经缓存的实时窗口。能量门关闭后不再提交新的 KWS 窗口，已授权且排队中的结果仍按提交顺序
交给原有确认规则处理。

传入 `--kws-frame-repair` 或 `--kws-frame-repair-report` 会在启动时被拒绝：该缓存路线只对
旧的 96 ms 相邻窗口认证，不能用于可调度的 demo。指定 v6.1 checkpoint/spec、PyTorch FBGEMM
与 ONNX Runtime `CPUExecutionProvider` 之外的执行提供方或目标设备，仍须对自己的编译产物重新
完成 code 与 float32-word 认证。
