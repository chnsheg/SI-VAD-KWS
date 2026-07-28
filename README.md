# VAD Research Benchmark

Audio-only research benchmark for voice activity detection (VAD).

The first version focuses on a small end-to-end workflow:

1. Prepare a tiny YESNO-derived synthetic VAD dataset.
2. Train or fit several traditional and lightweight neural VAD algorithms.
3. Evaluate all algorithms through one manifest and prediction contract.
4. Save metrics, per-file frame probabilities, and speech segments under `runs/`.

Use the existing `eis` environment directly:

```powershell
C:\myApps\Miniconda\envs\eis\python.exe -m vadbench.cli list-algorithms
C:\myApps\Miniconda\envs\eis\python.exe -m vadbench.cli prepare yesno-synth --download
C:\myApps\Miniconda\envs\eis\python.exe -m vadbench.cli eval --config configs/eval/energy_yesno.yaml
C:\myApps\Miniconda\envs\eis\python.exe -m vadbench.cli train --config configs/train/tiny_cnn_yesno.yaml
C:\myApps\Miniconda\envs\eis\python.exe -m vadbench.cli eval --config configs/eval/tiny_cnn_yesno.yaml
```

`conda run -n eis` is intentionally avoided because this local machine reported a Python `site` initialization encoding error through that path, while the direct interpreter path imports PyTorch and torchaudio correctly.

## AVA-Speech

AVA-Speech is the intended larger dataset path. Cache extracted audio, frame labels, and log-mel features on the faster C drive:

```powershell
C:\myApps\Miniconda\envs\eis\python.exe -m vadbench.cli prepare ava-speech --download-labels --cache-root C:\vadbench_cache\ava_speech --media-root C:\path\to\ava_media --max-videos 5
C:\myApps\Miniconda\envs\eis\python.exe -m vadbench.cli train --config configs/train/tiny_cnn_ava.yaml
C:\myApps\Miniconda\envs\eis\python.exe -m vadbench.cli eval --config configs/eval/tiny_cnn_ava.yaml
```

`ffmpeg` is required for audio extraction. Online YouTube extraction also requires `yt-dlp`; if `--use-yt-dlp` is used without it, the CLI prints the exact install command. The default flow stores only the 15:00-30:00 mono 16 kHz WAV excerpts and cached features, not full videos.

## Data Contract

Manifests are JSONL files with one item per audio file:

```json
{"id": "train_0000", "audio_path": "audio/train/train_0000.wav", "label_path": "labels/train/train_0000.npy", "split": "train", "sample_rate": 16000, "duration_sec": 8.0, "frame_hop_ms": 10.0, "source": "YESNO-synth-VAD"}
```

Labels are 10 ms frame-level `.npy` arrays with `0` for non-speech and `1` for speech. Algorithm predictions use the same frame grid and return `float32` speech probabilities. AVA manifests may also include `feature_path`, `video_id`, `chunk_start_sec`, `chunk_end_sec`, and `label_source`.

## AVA Paper v2 Protocol

The v2 AVA path reports paper-style metrics in addition to frame F1:

- `paper_metrics.json`: `TPR@FPR=0.315`, `AUROC`, and per-class TPR for clean/noise/music speech.
- `metrics.json`: current frame F1/precision/recall/accuracy plus embedded paper metrics.
- AVA manifests may include `class_label_path` with `0=NO_SPEECH`, `1=CLEAN_SPEECH`, `2=SPEECH_WITH_MUSIC`, `3=SPEECH_WITH_NOISE`.

The current cache has 61 automatically available AVA videos. A retry of the remaining official IDs without cookies did not add videos because YouTube reported unavailable/private/age-restricted/copyright-blocked content. To recover more videos, provide browser cookies or local media:

```powershell
C:\myApps\Miniconda\envs\eis\python.exe -m vadbench.cli prepare ava-speech --label-csv C:\Users\20193\vadbench_cache\ava_speech_full\labels\ava_speech_labels_v1.csv --cache-root C:\Users\20193\vadbench_cache\ava_speech_full --manifest-out manifests\ava_speech_paper_manifest.jsonl --use-yt-dlp --cookies-from-browser chrome
C:\myApps\Miniconda\envs\eis\python.exe -m vadbench.cli train --config configs/train/marblenet_3x2x64_ava_paper_v2.yaml
C:\myApps\Miniconda\envs\eis\python.exe -m vadbench.cli eval --config configs/eval/marblenet_3x2x64_ava_paper_v2.yaml
```

## Algorithms

Traditional:

- `energy_adaptive`
- `zcr_energy`
- `spectral_gate`
- `mfcc_gmm`
- `kaldi_energy`
- `sohn_hmm`
- `rvad_fast`
- `spectral_flux_ltsd`
- `webrtc_vad`

Lightweight neural:

- `tiny_mel_cnn`
- `marblenet_lite`
- `attn_tcn_lite`
- `marblenet_3x2x64`
- `cnn_td_like`
- `crnn_vad`
- `self_attentive_vad`

Teacher-student support is limited to the `pseudo-label` command in v1. It generates soft-label manifests from an existing model but does not reproduce large AudioSet/VoxCeleb training.

## Tests

The project uses standard-library `unittest` so no extra test dependency is required:

```powershell
C:\myApps\Miniconda\envs\eis\python.exe -m unittest discover -s tests
```
