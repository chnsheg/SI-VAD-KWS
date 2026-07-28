from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchaudio


def ensure_mono(waveform: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return a mono float32 waveform in shape [samples]."""
    if isinstance(waveform, torch.Tensor):
        array = waveform.detach().cpu().float().numpy()
    else:
        array = np.asarray(waveform, dtype=np.float32)

    if array.ndim == 2:
        if array.shape[0] <= array.shape[1]:
            array = array.mean(axis=0)
        else:
            array = array.mean(axis=1)
    elif array.ndim != 1:
        raise ValueError(f"Expected 1D or 2D waveform, got shape {array.shape}")

    return np.asarray(array, dtype=np.float32)


def load_audio(path: str | Path, target_sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    path = Path(path)
    waveform, sample_rate = torchaudio.load(str(path))
    waveform = waveform.float()
    if target_sample_rate is not None and sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    return ensure_mono(waveform), int(sample_rate)


def save_audio(path: str | Path, waveform: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(waveform, dtype=np.float32)
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak > 0.99:
        array = array / peak * 0.99
    sf.write(str(path), array, sample_rate)

