# Streaming CRNN-GRU KWS 改造方案

本文给出一个不改动现有 DSCNN 代码的实验路线：在 `dscnn_kws/streaming/` 下新增一套流式友好的 CRNN 原型，CNN 部分使用深度可分离卷积，RNN 部分使用 GRU。

## 1. 当前问题

现有 DSCNN 推理方式更接近整窗分类：

```text
1 秒 waveform
-> MFCC / bandpass 整段特征
-> reshape 为 time x freq 图
-> DSCNN 完整卷积
-> logits / argmax
```

如果硬件部署时每 32 ms 或更短间隔滑动一次 1 秒窗口，那么相邻窗口大约 96% 以上的音频是重叠的。每次都重新计算完整 1 秒，会重复计算前端特征、卷积特征和最终分类，功耗会明显高于真正流式结构。

流式 KWS 应该变成：

```text
每来一个新 hop 的音频
-> 只计算新的一帧特征
-> CNN 只更新新时间步附近的卷积结果
-> GRU hidden 继承历史状态
-> 输出当前时刻的关键词分数
```

## 2. 总体结构

建议先做一个 DS-CNN + GRU 的 CRNN：

```text
waveform chunk
-> streaming frontend frame
-> feature frame [F]
-> causal depthwise separable CNN
-> frequency pooling
-> GRU hidden update
-> FC logits
-> optional trigger threshold
```

训练时可以仍然用 1 秒样本，以便复用现有 manifest、noise augmentation、Trainer 和评估流程。部署时改成逐帧执行。

## 3. 模型结构建议

默认输入特征已经和之前的 `L5_C64` DSCNN sweep 对齐：

- `sample_rate = 16000`
- `window_size_ms = 32`
- `window_stride_ms = 32`
- `frontend = mfcc`
- `dct_coeff = 10`

默认 CRNN 的输入配置与 `L5_C64` 对齐；CNN 保持 5 层，但为了让参数量不超过 `L5_C64`，默认每层使用 24 通道：

```text
Input features: [B, T, F]

DSConv2D block 1: 1  -> 24, kernel_time=5, kernel_freq=3, causal time padding
DSConv2D block 2: 24 -> 24, kernel_time=5, kernel_freq=3, causal time padding
DSConv2D block 3: 24 -> 24, kernel_time=5, kernel_freq=3, causal time padding
DSConv2D block 4: 24 -> 24, kernel_time=5, kernel_freq=3, causal time padding
DSConv2D block 5: 24 -> 24, kernel_time=5, kernel_freq=3, causal time padding

mean over frequency
GRU: input=24, hidden=64, layers=1
FC: 64 -> 2
```

这里 CNN 不做时间下采样，原因是第一版要保持最简单的逐帧状态更新。后续如果为了更低功耗，可以每隔 2 或 3 帧更新一次 GRU，或者加入 stride，但硬件控制和标签对齐会复杂一些。

## 4. 为什么 CNN 要做 causal

普通 2D 卷积如果使用对称 padding，会在时间维看到未来帧：

```text
frame t 的卷积使用 frame t-2, t-1, t, t+1, t+2
```

这不适合真实流式，因为 `t+1, t+2` 在当前时刻还没到。新原型里的 `CausalDepthwiseSeparableConv2d` 使用左侧时间 padding：

```text
frame t 的卷积只使用 frame t-4, t-3, t-2, t-1, t
```

这样离线训练和在线推理的时序依赖是一致的。

## 5. GRU 如何减少重复计算

DSCNN 的整窗分类需要每次处理完整 `[T, F]` 特征图。GRU 的状态可以保留过去的信息：

```text
h_t = GRU(x_t, h_{t-1})
```

在线部署时，硬件只需要保存：

- CNN 每层最近 `kernel_time - 1` 帧缓存
- GRU hidden state
- 当前帧 logits

不需要每次把过去 1 秒的特征重新卷一遍。

## 6. 新增文件

当前新增文件都位于 `dscnn_kws/streaming/`：

```text
dscnn_kws/streaming/__init__.py
dscnn_kws/frontend/streaming_mfcc.py
dscnn_kws/streaming/streaming_crnn.py
dscnn_kws/streaming/train_streaming_crnn.py
dscnn_kws/streaming/README_STREAMING_CRNN_GRU.md
```

旧的 `train.py`、`model/dscnn.py`、`quantization/` 文件不需要改动。

## 7. 第一阶段实验

先用完整 1 秒样本训练 CRNN-GRU，验证它是否能接近 DSCNN 的准确率：

```bash
python -m dscnn_kws.streaming.train_streaming_crnn \
  --root ./dataset \
  --dataset mobvoi_xiaowen_binary \
  --epoch 50 \
  --batch 256 \
  --gpu 1 \
  --frontend mfcc \
  --sample_rate 16000 \
  --dct_coeff 10 \
  --window_size_ms 32 \
  --window_stride_ms 32 \
  --streaming_mfcc \
  --cnn_channels 24,24,24,24,24 \
  --gru_hidden 64
```

如果使用你的噪声增强路径，可以继续传入：

```bash
--train_noise_roots /path/to/train_noise.txt \
--valid_noise_roots /path/to/valid_noise.txt \
--test_noise_roots /path/to/test_noise.txt \
--noise_snr_min_db -5 \
--noise_snr_max_db 20
```

## 8. 第二阶段流式一致性验证

训练完成后要验证两件事：

1. `model(waveform)` 的整段输出正常。
2. `extract_features(waveform)` 后逐帧调用 `backbone.forward_stream_frame()`，最后一帧 logits 与整段 CNN-GRU 输出接近。

由于 BatchNorm、浮点顺序和前端差异，数值可能不是逐 bit 相等，但分类结果和 logit 应该接近。若要进一步硬件化，建议将 BatchNorm fold 到卷积权重中。

现在新增了一个真正按音频 chunk 推理的脚本：

```text
dscnn_kws/streaming/eval_streaming_crnn_true_streaming.py
```

它不会把整段 waveform 直接送进 `model(waveform)` 作为主路径，而是：

```text
1s dataset waveform
-> split into 512-sample chunks
-> StreamingMFCC.forward_stream_chunk()
-> backbone.forward_stream_frame()
-> CNN cache + GRU hidden state
-> final logits
```

Clean test 示例：

```bash
python -m dscnn_kws.streaming.eval_streaming_crnn_true_streaming \
  --root ./dscnn_kws/data \
  --dataset mobvoi_hi_xiaowen_binary_hardneg \
  --ckpt dscnn_kws/runs/streaming/streaming_crnn_clean_sweep_best_models/<best>.pt \
  --batch 256 --gpu 1 --num_workers 8 \
  --sample_rate 16000 \
  --dct_coeff 10 \
  --window_size_ms 32 \
  --window_stride_ms 32 \
  --chunk_samples 512 \
  --cnn_channels 24,24,24,24,24 \
  --gru_hidden 64 \
  --allow_online_resample \
  --no-strict_sample_rate
```

Noise test 示例：

```bash
python -m dscnn_kws.streaming.eval_streaming_crnn_true_streaming \
  --root ./dscnn_kws/data \
  --dataset mobvoi_hi_xiaowen_binary_hardneg \
  --ckpt dscnn_kws/runs/streaming/streaming_crnn_noise_snr_scene_sweep_best_models/<best>.pt \
  --batch 256 --gpu 1 --num_workers 8 \
  --sample_rate 16000 \
  --dct_coeff 10 \
  --window_size_ms 32 \
  --window_stride_ms 32 \
  --chunk_samples 512 \
  --cnn_channels 24,24,24,24,24 \
  --gru_hidden 64 \
  --eval_noise_aug \
  --test_noise_roots ./dscnn_kws/noise/lists/tau_test.txt \
  --eval_noise_aug_prob 1.0 \
  --eval_noise_snr_min_db 5.0 \
  --eval_noise_snr_max_db 5.0 \
  --allow_online_resample \
  --no-strict_sample_rate
```

默认会额外计算一次整段离线路径作为对照，并打印：

```text
[TRUE_STREAM_TEST] acc=...
[OFFLINE_COMPARE] offline_acc=... pred_agree=... max_abs_logit_diff=...
```

其中 `TRUE_STREAM_TEST` 才是真正逐 chunk 推理得到的结果。结果会追加保存到：

```text
dscnn_kws/streaming/results/streaming_crnn_true_streaming_eval.csv
```

### 8.1 批量真流式评估 noise sweep 模型

如果要把之前 noise 场景下 sweep 出来的所有 best checkpoint 都按真正流式方式重新评估，可以使用：

```text
dscnn_kws/streaming/batch_eval_true_streaming_noise_sweep.py
```

默认会读取：

```text
dscnn_kws/streaming/results/streaming_crnn_noise_snr_scene_sweep_train_results.csv
```

其中的 `best_model_saved_as`、`arch`、`cnn_channels`、`gru_hidden` 字段会用于自动加载模型和恢复网络配置。然后每个模型都会跑十个 TAU 场景和五个 SNR：

```text
airport, bus, metro, metro_station, park, public_square,
shopping_mall, street_pedestrian, street_traffic, tram

20, 10, 5, 0, -5 dB
```

推荐命令：

```bash
python -m dscnn_kws.streaming.batch_eval_true_streaming_noise_sweep \
  --root ./dscnn_kws/data \
  --batch 256 --gpu 1 --num_workers 8 \
  --chunk_samples 512 \
  --scene_test_root ./dscnn_kws/noise/tau \
  --allow_online_resample \
  --no-strict_sample_rate
```

如果只想先测试一个数据集或一个结构，可以加：

```bash
--datasets mobvoi_hi_xiaowen_binary_hardneg \
--archs C24x5_H64
```

默认也会计算整段离线路径作为对照。若只关心真流式结果、想加速，可以加：

```bash
--no-compare_offline
```

最终会生成三个汇总表：

```text
dscnn_kws/streaming/results/streaming_crnn_noise_snr_scene_true_streaming_grid_results.csv
dscnn_kws/streaming/results/streaming_crnn_noise_snr_scene_true_streaming_scene_summary.csv
dscnn_kws/streaming/results/streaming_crnn_noise_snr_scene_true_streaming_arch_summary.csv
```

## 9. 第三阶段真正流式前端

当前训练仍然可以用完整 waveform 提高效率，但 MFCC 特征计算已经可以走 `StreamingMFCC` 的 causal framing。真正硬件部署时，还需要把这个 PyTorch 流式前端映射成定点/RTL 实现：

### MFCC 路线

每个 hop 缓存 `win_length` 个音频点，只对最新窗口计算：

```text
ring buffer audio window
-> window multiply
-> DFT / power
-> mel filterbank
-> log or PWL log
-> DCT
-> one MFCC frame
```

注意训练原型里建议 `mfcc_center=False`，这样不会依赖未来采样点。

### bandpass 路线

更硬件友好的路线是使用当前项目已有的 bandpass Conv1D 前端思想：

```text
audio sample stream
-> FIR bandpass filters
-> frame energy accumulation
-> log / PWL log
-> one band-energy frame
```

这条路线比 STFT/MFCC 更容易做成低功耗流式硬件。

## 10. 硬件状态设计

每个时间步需要保存：

```text
frontend audio ring buffer
cnn_cache[layer] = previous kernel_time - 1 feature maps
gru_hidden[layer] = hidden vector
```

以默认配置估算：

- CNN cache 只保存很少的时间帧，而不是完整 1 秒。
- GRU hidden 只有 64 维。
- 每个 hop 只计算一个新帧对应的 CNN 和 GRU。

这正是相比滑动 1 秒 DSCNN 降低重复计算的核心。

## 11. 触发逻辑

模型输出仍然是两个 logits。普通分类可以继续：

```text
argmax(logits)
```

实际唤醒部署建议使用 FAH/FRR 那套思想：

```text
score = softmax(logits)[positive]
trigger = score >= threshold
```

另外建议加一个简单去抖策略：

```text
连续 N 帧超过阈值才触发
触发后进入 cooldown
```

这样能减少短时噪声造成的误触发。

## 12. 推荐路线

建议按这个顺序推进：

1. 训练 `StreamingKWSModel`，对比当前 DSCNN 的 clean ACC/F1。
2. 加入 TAU/MUSAN/DEMAND 噪声增强，对比噪声场景 SNR 网格。
3. 做逐帧 streaming consistency check。
4. 用 FAH/FRR 数据集选择触发阈值。
5. 做 QAT 或定点仿真。
6. 把前端、CNN cache、GRU hidden 明确映射到 RTL。

第一版不要急着完全硬件化。先确认 DS-CNN-GRU 的准确率和鲁棒性能接受，再把状态缓存和定点细节往硬件方向收紧。

## 13. 当前 StreamingMFCC 前端

`dscnn_kws/frontend/streaming_mfcc.py` 已经新增 `StreamingMFCC`。它有两种路径：

```text
训练/离线评估:
waveform [B, T]
-> StreamingMFCC.forward()
-> causal frames
-> FFT / power / Mel / log / DCT
-> MFCC [B, n_mfcc, frames]

在线部署:
new audio chunk
-> StreamingMFCC.forward_stream_chunk()
-> zero or more new MFCC frames
-> CRNN backbone.forward_stream_frame()
```

默认 `flush_tail=True`，所以 16 kHz、1 秒、32 ms hop 下会输出 `ceil(16000 / 512) = 32` 帧，和之前 `L5_C64` 使用的 `10 x 32` 输入规模对齐。

这个前端不使用 `torch.stft(center=True)`，而是使用 causal hop framing：

```text
history + current hop -> one frame
```

当 `window_size_ms == window_stride_ms == 32` 时，每帧就是当前 512 个采样点；当窗口大于 hop 时，会自动保留前一段 history，仍然不需要未来采样点。

训练脚本默认使用 `StreamingMFCC`：

```bash
--streaming_mfcc
```

如果需要回退到旧的整段 `TorchMFCC` 做消融，可以传：

```bash
--no-streaming_mfcc
```

## 14. Clean / Noise 训练与 5x10 场景测试命令

下面命令假设在项目根目录执行：

```bash
cd /root/kws/dscnn_kws
DATA_ROOT=./dscnn_kws/data
DATASET=mobvoi_hi_xiaowen_binary_hardneg
```

如果要跑另一个关键词数据集，把 `DATASET` 换成：

```bash
DATASET=mobvoi_nihao_wenwen_binary_hardneg
```

### 14.1 Clean 训练 + Clean 测试

`train_streaming_crnn.py` 训练结束后会自动加载 best checkpoint 并在 test set 上测试。这里关闭训练噪声和评估噪声：

```bash
python -m dscnn_kws.streaming.train_streaming_crnn \
  --root "$DATA_ROOT" \
  --dataset "$DATASET" \
  --epoch 30 \
  --batch 256 \
  --gpu 1 \
  --num_workers 8 \
  --sample_rate 16000 \
  --frontend mfcc \
  --streaming_mfcc \
  --dct_coeff 10 \
  --window_size_ms 32 \
  --window_stride_ms 32 \
  --cnn_channels 24,24,24,24,24 \
  --gru_hidden 64 \
  --no-noise_aug \
  --no-eval_noise_aug \
  --save_root "./dscnn_kws/streaming/runs/clean_${DATASET}"
```

训练完成后，clean best checkpoint 可以这样取：

```bash
CLEAN_RUN=$(ls -td ./dscnn_kws/streaming/runs/clean_${DATASET}/streaming_crnn_mfcc_${DATASET}_* | head -n 1)
CLEAN_CKPT="${CLEAN_RUN}/best.pt"
echo "$CLEAN_CKPT"
```

### 14.2 Noise 训练 + Tau valid/test 5 dB 测试

这里训练时使用 TAU train list，SNR 在 `-5..20 dB` 随机采样；valid/test 固定使用 5 dB，分别对应 `tau_valid.txt` 和 `tau_test.txt`：

```bash
python -m dscnn_kws.streaming.train_streaming_crnn \
  --root "$DATA_ROOT" \
  --dataset "$DATASET" \
  --epoch 30 \
  --batch 256 \
  --gpu 1 \
  --num_workers 8 \
  --sample_rate 16000 \
  --frontend mfcc \
  --streaming_mfcc \
  --dct_coeff 10 \
  --window_size_ms 32 \
  --window_stride_ms 32 \
  --cnn_channels 24,24,24,24,24 \
  --gru_hidden 64 \
  --noise_aug \
  --eval_noise_aug \
  --train_noise_roots ./dscnn_kws/noise/lists/tau_train.txt \
  --valid_noise_roots ./dscnn_kws/noise/lists/tau_valid.txt \
  --test_noise_roots ./dscnn_kws/noise/lists/tau_test.txt \
  --noise_aug_prob 0.8 \
  --noise_snr_min_db -5 \
  --noise_snr_max_db 20 \
  --eval_noise_aug_prob 1.0 \
  --eval_noise_snr_min_db 5 \
  --eval_noise_snr_max_db 5 \
  --save_root "./dscnn_kws/streaming/runs/noise_${DATASET}"
```

训练完成后，noise best checkpoint 可以这样取：

```bash
NOISE_RUN=$(ls -td ./dscnn_kws/streaming/runs/noise_${DATASET}/streaming_crnn_mfcc_${DATASET}_* | head -n 1)
NOISE_CKPT="${NOISE_RUN}/best.pt"
echo "$NOISE_CKPT"
```

### 14.3 Noise 模型做 5 个 SNR x 10 个 TAU 场景测试

这个步骤使用新增的 `eval_streaming_crnn_snr_scene.py`，加载上一步的 `NOISE_CKPT`，在十个 TAU 场景和 `20/10/5/0/-5 dB` 上测试：

```bash
python -m dscnn_kws.streaming.eval_streaming_crnn_snr_scene \
  --root "$DATA_ROOT" \
  --dataset "$DATASET" \
  --ckpt "$NOISE_CKPT" \
  --batch 256 \
  --gpu 1 \
  --num_workers 8 \
  --sample_rate 16000 \
  --frontend mfcc \
  --streaming_mfcc \
  --dct_coeff 10 \
  --window_size_ms 32 \
  --window_stride_ms 32 \
  --cnn_channels 24,24,24,24,24 \
  --gru_hidden 64 \
  --scene_test_root ./dscnn_kws/noise/tau \
  --test_snrs 20 10 5 0 -5 \
  --out_csv "./dscnn_kws/streaming/results/${DATASET}_streaming_crnn_noise_snr_scene_grid_results.csv"
```

默认十个场景为：

```text
airport, bus, metro, metro_station, park, public_square,
shopping_mall, street_pedestrian, street_traffic, tram
```

如果只想临时测试部分场景，可以加：

```bash
--scene_names airport shopping_mall street_traffic
```

## 15. Streaming CRNN Sweep 脚本

现在提供两个 sweep 脚本：

```text
dscnn_kws/streaming/sweep_streaming_crnn_clean.py
dscnn_kws/streaming/sweep_streaming_crnn_noise_snr_scene.py
```

两个脚本都会 sweep 5 个 CRNN 配置：

```text
C24x5_H64        cnn_channels=24,24,24,24,24  gru_hidden=64
C32x5_H48        cnn_channels=32,32,32,32,32  gru_hidden=48
C24x5_H48        cnn_channels=24,24,24,24,24  gru_hidden=48
C16x5_H64        cnn_channels=16,16,16,16,16  gru_hidden=64
C32_32_48_H48    cnn_channels=32,32,48        gru_hidden=48
```

输出统一放在：

```text
./dscnn_kws/streaming/results
```

训练过程目录和复制出的 best checkpoint 统一放在：

```text
./dscnn_kws/runs/streaming
```

### 15.1 Clean Sweep

clean sweep 只做 clean 训练、clean valid、clean test：

```bash
python -m dscnn_kws.streaming.sweep_streaming_crnn_clean \
  --root ./dscnn_kws/data \
  --datasets mobvoi_hi_xiaowen_binary_hardneg mobvoi_nihao_wenwen_binary_hardneg \
  --epoch 30 \
  --batch 256 \
  --gpu 1 \
  --num_workers 8
```

主要输出：

```text
./dscnn_kws/streaming/results/streaming_crnn_clean_sweep_results.csv
./dscnn_kws/runs/streaming/streaming_crnn_clean_sweep_train_runs/
./dscnn_kws/runs/streaming/streaming_crnn_clean_sweep_best_models/
```

### 15.2 Noise + SNR Scene Sweep

noise sweep 的训练配置与之前 DSCNN noise 脚本保持一致：

- train: `tau_train.txt`, 随机 SNR `-5..20 dB`, `noise_aug_prob=0.8`
- valid/test: `tau_valid.txt` / `tau_test.txt`, 固定 `5 dB`
- grid test: 10 个 TAU 场景 x `20/10/5/0/-5 dB`

```bash
python -m dscnn_kws.streaming.sweep_streaming_crnn_noise_snr_scene \
  --root ./dscnn_kws/data \
  --datasets mobvoi_hi_xiaowen_binary_hardneg mobvoi_nihao_wenwen_binary_hardneg \
  --epoch 30 \
  --batch 256 \
  --gpu 1 \
  --num_workers 8 \
  --train_noise_roots ./dscnn_kws/noise/lists/tau_train.txt \
  --valid_noise_roots ./dscnn_kws/noise/lists/tau_valid.txt \
  --test_noise_roots ./dscnn_kws/noise/lists/tau_test.txt \
  --scene_test_root ./dscnn_kws/noise/tau
```

主要输出：

```text
./dscnn_kws/streaming/results/streaming_crnn_noise_snr_scene_sweep_train_results.csv
./dscnn_kws/streaming/results/streaming_crnn_noise_snr_scene_sweep_grid_results.csv
./dscnn_kws/streaming/results/streaming_crnn_noise_snr_scene_sweep_scene_summary.csv
./dscnn_kws/streaming/results/streaming_crnn_noise_snr_scene_sweep_arch_summary.csv
./dscnn_kws/runs/streaming/streaming_crnn_noise_snr_scene_sweep_train_runs/
./dscnn_kws/runs/streaming/streaming_crnn_noise_snr_scene_sweep_best_models/
```

如果只想先验证训练流程，不跑 5x10 网格测试，可以加：

```bash
--skip_grid
```

## 16. 结构详解图

当前更详细的“整段离线调用 vs 真正流式调用”对照图位于：

```text
dscnn_kws/streaming/figures/imagegen_streaming_mfcc_crnn_offline_vs_true_streaming_17x6.png
```

TorchMFCC 前端与 StreamingMFCC 前端的关键差异对比图位于：

```text
dscnn_kws/streaming/figures/imagegen_torchmfcc_vs_streamingmfcc_frontend_17x6.png
```

上一版流式前端 + CRNN 完整结构图保留在：

```text
dscnn_kws/streaming/figures/imagegen_streaming_mfcc_crnn_structure_17x6.png
```

另外保留了一版脚本绘制的可复现结构图：

```text
dscnn_kws/streaming/figures/streaming_mfcc_crnn_inference_structure_17x6.png
dscnn_kws/streaming/draw_streaming_crnn_structure.py
```

图中默认配置为 `C24x5_H64`：

```text
StreamingMFCC: 16 kHz, win=512, hop=512, n_mels=40, n_mfcc=40, select first 10
CRNN CNN: 5 个 causal depthwise-separable Conv2d block, channels=24
GRU: input_size=24, hidden_size=64, num_layers=1
FC: 64 -> 2
total params = 21627
```
