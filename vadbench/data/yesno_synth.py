from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio

from vadbench.audio import ensure_mono, save_audio
from vadbench.features import rms_zcr, sample_mask_to_frame_labels
from vadbench.manifest import ManifestRecord, write_manifest


@dataclass
class YesNoSynthConfig:
    raw_root: Path = Path("data/raw")
    out_dir: Path = Path("data/yesno_synth")
    download: bool = False
    sample_rate: int = 16000
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    seed: int = 7
    train_samples: int = 64
    val_samples: int = 16
    test_samples: int = 16
    duration_min_sec: float = 6.0
    duration_max_sec: float = 10.0
    noise_amp: float = 0.003


def prepare_yesno_synth(config: YesNoSynthConfig) -> Path:
    rng = random.Random(config.seed)
    np_rng = np.random.default_rng(config.seed)
    raw_root = Path(config.raw_root)
    out_dir = Path(config.out_dir)
    raw_root.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = torchaudio.datasets.YESNO(str(raw_root), download=config.download)
    source_items = [_load_item(dataset, idx, config.sample_rate) for idx in range(len(dataset))]
    split_to_indices = _split_indices(len(source_items), rng)
    split_counts = {"train": config.train_samples, "val": config.val_samples, "test": config.test_samples}

    all_records: list[ManifestRecord] = []
    for split, indices in split_to_indices.items():
        chunks = _collect_chunks([source_items[idx] for idx in indices], config)
        if not chunks:
            raise RuntimeError(f"No speech chunks found for split {split}")
        for sample_idx in range(split_counts[split]):
            duration = rng.uniform(config.duration_min_sec, config.duration_max_sec)
            sample_id = f"{split}_{sample_idx:04d}"
            waveform, mask = _compose_sample(chunks, duration, config, rng, np_rng)
            labels = sample_mask_to_frame_labels(mask, config.sample_rate, config.frame_ms, config.hop_ms)
            audio_rel = Path("audio") / split / f"{sample_id}.wav"
            label_rel = Path("labels") / split / f"{sample_id}.npy"
            save_audio(out_dir / audio_rel, waveform, config.sample_rate)
            (out_dir / label_rel).parent.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / label_rel, labels.astype(np.uint8))
            all_records.append(
                ManifestRecord(
                    id=sample_id,
                    audio_path=audio_rel.as_posix(),
                    label_path=label_rel.as_posix(),
                    split=split,
                    sample_rate=config.sample_rate,
                    duration_sec=float(len(waveform) / config.sample_rate),
                    frame_hop_ms=config.hop_ms,
                    source="YESNO-synth-VAD",
                )
            )

    manifest_path = out_dir / "manifest.jsonl"
    write_manifest(all_records, manifest_path)
    metadata = {
        "source": "torchaudio.datasets.YESNO",
        "sample_rate": config.sample_rate,
        "frame_ms": config.frame_ms,
        "hop_ms": config.hop_ms,
        "seed": config.seed,
        "splits": split_counts,
        "note": "Synthetic VAD labels are generated from inserted speech chunks and are intended for smoke tests.",
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return manifest_path


def _load_item(dataset, idx: int, target_sample_rate: int) -> np.ndarray:
    waveform, sample_rate, _labels = dataset[idx]
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform.float(), sample_rate, target_sample_rate)
    return ensure_mono(waveform)


def _split_indices(total: int, rng: random.Random) -> dict[str, list[int]]:
    indices = list(range(total))
    rng.shuffle(indices)
    train_end = max(1, int(total * 0.70))
    val_end = max(train_end + 1, int(total * 0.85))
    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }


def _collect_chunks(items: list[np.ndarray], config: YesNoSynthConfig) -> list[np.ndarray]:
    chunks: list[np.ndarray] = []
    for waveform in items:
        chunks.extend(_extract_speech_chunks(waveform, config.sample_rate, config.frame_ms, config.hop_ms))
    return chunks


def _extract_speech_chunks(
    waveform: np.ndarray,
    sample_rate: int,
    frame_ms: float,
    hop_ms: float,
) -> list[np.ndarray]:
    rms, _ = rms_zcr(waveform, sample_rate, frame_ms, hop_ms)
    if len(rms) == 0:
        return []
    threshold = max(float(np.percentile(rms, 65)), float(np.max(rms)) * 0.10)
    active = rms > threshold
    active = _smooth_binary(active, fill_gap=4, min_run=4)
    hop = int(round(sample_rate * hop_ms / 1000.0))
    frame = int(round(sample_rate * frame_ms / 1000.0))
    chunks: list[np.ndarray] = []
    for start_frame, end_frame in _active_runs(active):
        start = max(0, start_frame * hop - int(0.03 * sample_rate))
        end = min(len(waveform), (end_frame - 1) * hop + frame + int(0.03 * sample_rate))
        if end - start < int(0.12 * sample_rate):
            continue
        chunk = np.asarray(waveform[start:end], dtype=np.float32)
        peak = float(np.max(np.abs(chunk))) if chunk.size else 0.0
        if peak > 1e-5:
            chunks.append(chunk / peak * 0.7)
    if chunks:
        return chunks
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    return [waveform / peak * 0.7] if peak > 1e-5 else []


def _smooth_binary(mask: np.ndarray, fill_gap: int, min_run: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    for value, start, end in _runs(out):
        if value:
            continue
        if start > 0 and end < len(out) and end - start <= fill_gap:
            out[start:end] = True
    for value, start, end in _runs(out):
        if value and end - start < min_run:
            out[start:end] = False
    return out


def _runs(mask: np.ndarray) -> list[tuple[bool, int, int]]:
    if len(mask) == 0:
        return []
    runs: list[tuple[bool, int, int]] = []
    start = 0
    value = bool(mask[0])
    for idx in range(1, len(mask)):
        current = bool(mask[idx])
        if current != value:
            runs.append((value, start, idx))
            start = idx
            value = current
    runs.append((value, start, len(mask)))
    return runs


def _active_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(start, end) for value, start, end in _runs(mask) if value]


def _compose_sample(
    chunks: list[np.ndarray],
    duration_sec: float,
    config: YesNoSynthConfig,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    total_samples = int(round(duration_sec * config.sample_rate))
    waveform = np_rng.normal(0.0, config.noise_amp, total_samples).astype(np.float32)
    speech_mask = np.zeros(total_samples, dtype=bool)
    cursor = int(round(rng.uniform(0.15, 0.65) * config.sample_rate))
    while cursor < total_samples:
        chunk = np.asarray(rng.choice(chunks), dtype=np.float32)
        max_remaining = total_samples - cursor
        if max_remaining <= int(0.15 * config.sample_rate):
            break
        if len(chunk) > max_remaining:
            chunk = chunk[:max_remaining]
        end = cursor + len(chunk)
        gain = rng.uniform(0.55, 0.95)
        waveform[cursor:end] += chunk * gain
        speech_mask[cursor:end] = True
        silence = rng.uniform(0.18, 0.85)
        cursor = end + int(round(silence * config.sample_rate))

    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 0.98:
        waveform = waveform / peak * 0.98
    return waveform.astype(np.float32), speech_mask

