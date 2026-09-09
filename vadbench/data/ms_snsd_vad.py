from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vadbench.audio import load_audio, save_audio
from vadbench.features import frame_count, log_mel_spectrogram, sample_mask_to_frame_labels
from vadbench.manifest import ManifestRecord, write_manifest


MS_SNSD_GIT_URL = "https://github.com/microsoft/MS-SNSD.git"
_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}

MS_SNSD_EVENT_CLASSES = {
    "silence": 0,
    "low_background_noise": 1,
    "noise_only_event": 2,
    "hard_negative_event": 3,
    "clean_speech": 4,
    "speech_with_noise": 5,
}
_SPEECH_EVENT_IDS = {MS_SNSD_EVENT_CLASSES["clean_speech"], MS_SNSD_EVENT_CLASSES["speech_with_noise"]}
_HARD_NEGATIVE_KEYWORDS = {
    "airconditioner",
    "copy",
    "copymachine",
    "door",
    "keyboard",
    "mechanical",
    "munching",
    "shuttingdoor",
    "typing",
    "vacuum",
}


@dataclass
class MSSNSDVADConfig:
    raw_root: Path = Path("data/raw/ms-snsd")
    out_dir: Path = Path("data/ms_snsd_vad_v2")
    download: bool = False
    protocol: str = "v2"
    sample_rate: int = 16000
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    n_mels: int = 64
    clip_sec: float = 10.0
    total_hours: float = 20.0
    snr_levels: list[float] = field(default_factory=lambda: [0.0, 5.0, 10.0, 15.0, 20.0])
    seed: int = 7
    precompute_features: bool = True
    speech_min_sec: float = 0.35
    speech_max_sec: float = 2.2
    silence_min_sec: float = 0.08
    silence_max_sec: float = 0.75
    target_speech_ratio: float = 0.45
    speech_ratio_min: float = 0.35
    speech_ratio_max: float = 0.55
    hard_negative_ratio: float = 0.25
    silence_ratio: float = 0.15
    low_noise_ratio: float = 0.15
    exclude_noise_keywords: list[str] = field(
        default_factory=lambda: ["Babble", "AirportAnnouncement", "AirportAnnouncements", "Neighbor"]
    )


def prepare_ms_snsd_vad(config: MSSNSDVADConfig) -> Path:
    rng = random.Random(config.seed)
    np_rng = np.random.default_rng(config.seed)
    raw_root = Path(config.raw_root)
    out_dir = Path(config.out_dir)
    protocol = str(config.protocol).lower()
    if protocol not in {"v1", "v2"}:
        raise ValueError(f"Unsupported MS-SNSD VAD protocol: {config.protocol}")
    if config.download:
        _ensure_ms_snsd_repo(raw_root)
    clean_files, noise_files = scan_ms_snsd_sources(raw_root)
    if len(clean_files) < 1:
        raise RuntimeError(f"No clean speech wav files found under {raw_root}")
    if len(noise_files) < 1:
        raise RuntimeError(f"No noise wav files found under {raw_root}")

    noise_pool = _build_noise_pool(noise_files, config)
    usable_noise = noise_pool["usable"]
    if len(usable_noise) < 1:
        raise RuntimeError("No usable noise files remain after applying exclude keywords")
    hard_noise = noise_pool["hard_negative"] or usable_noise

    split_to_clean = split_clean_files(clean_files, config.seed)
    split_hours = {"train": config.total_hours * 0.8, "val": config.total_hours * 0.1, "test": config.total_hours * 0.1}
    records: list[ManifestRecord] = []
    source_cursor = {split: 0 for split in split_to_clean}
    noise_cursor = 0
    hard_noise_cursor = 0
    noise_order = list(usable_noise)
    hard_noise_order = list(hard_noise)
    rng.shuffle(noise_order)
    rng.shuffle(hard_noise_order)

    for split in ("train", "val", "test"):
        target_samples = int(round(split_hours[split] * 3600.0 * config.sample_rate))
        samples_per_clip = int(round(config.clip_sec * config.sample_rate))
        clip_count = max(1, int(math.ceil(target_samples / max(samples_per_clip, 1)))) if target_samples > 0 else 0
        clean_pool = split_to_clean[split]
        for clip_idx in range(clip_count):
            clean_sequence: list[np.ndarray] = []
            while sum(len(item) for item in clean_sequence) < samples_per_clip * 2 and len(clean_sequence) < 8:
                source = clean_pool[source_cursor[split] % len(clean_pool)]
                source_cursor[split] += 1
                waveform, _ = load_audio(source, config.sample_rate)
                chunks = _extract_non_silent_chunks(waveform, config.sample_rate, config, rng)
                clean_sequence.extend(chunks or [_normalize_peak(waveform)])

            noise_source = noise_order[noise_cursor % len(noise_order)]
            noise_cursor += 1
            noise, _ = load_audio(noise_source, config.sample_rate)
            hard_noise_source = hard_noise_order[hard_noise_cursor % len(hard_noise_order)]
            hard_noise_cursor += 1
            hard_noise_waveform, _ = load_audio(hard_noise_source, config.sample_rate)
            snr_db = float(rng.choice(config.snr_levels))
            sample_id = f"{split}_{clip_idx:06d}"

            if protocol == "v1":
                waveform, speech_mask, speech_ratio = _compose_vad_clip(clean_sequence, noise, snr_db, config, rng, np_rng)
                event_mask = np.where(
                    speech_mask,
                    MS_SNSD_EVENT_CLASSES["speech_with_noise"],
                    MS_SNSD_EVENT_CLASSES["noise_only_event"],
                ).astype(np.uint8)
                label_source = f"protocol=v1,snr={snr_db:g}dB,speech_ratio={speech_ratio:.3f}"
            else:
                waveform, speech_mask, event_mask = _compose_vad_clip_v2(
                    clean_sequence,
                    noise,
                    hard_noise_waveform,
                    snr_db,
                    config,
                    rng,
                    np_rng,
                )
                speech_ratio = float(np.mean(speech_mask))
                hard_ratio = float(np.mean(event_mask == MS_SNSD_EVENT_CLASSES["hard_negative_event"]))
                label_source = f"protocol=v2,snr={snr_db:g}dB,speech_ratio={speech_ratio:.3f},hard_negative={hard_ratio:.3f}"

            labels = sample_mask_to_frame_labels(speech_mask, config.sample_rate, config.frame_ms, config.hop_ms)
            class_labels = _event_mask_to_frame_labels(event_mask, config)
            class_labels = class_labels[: len(labels)]
            audio_rel = Path("audio") / split / f"{sample_id}.wav"
            label_rel = Path("labels") / split / f"{sample_id}.npy"
            class_label_rel = Path("event_labels") / split / f"{sample_id}.npy"
            feature_rel = Path("features") / split / f"{sample_id}.logmel{config.n_mels}.npy"
            save_audio(out_dir / audio_rel, waveform, config.sample_rate)
            (out_dir / label_rel).parent.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / label_rel, labels.astype(np.uint8))
            (out_dir / class_label_rel).parent.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / class_label_rel, class_labels.astype(np.uint8))
            feature_path: str | None = None
            if config.precompute_features:
                features = log_mel_spectrogram(
                    waveform,
                    config.sample_rate,
                    n_mels=config.n_mels,
                    frame_ms=config.frame_ms,
                    hop_ms=config.hop_ms,
                    normalize=True,
                )
                features = features[: len(labels)]
                (out_dir / feature_rel).parent.mkdir(parents=True, exist_ok=True)
                np.save(out_dir / feature_rel, features.astype(np.float32))
                feature_path = feature_rel.as_posix()
            records.append(
                ManifestRecord(
                    id=sample_id,
                    audio_path=audio_rel.as_posix(),
                    label_path=label_rel.as_posix(),
                    split=split,
                    sample_rate=config.sample_rate,
                    duration_sec=float(len(waveform) / config.sample_rate),
                    frame_hop_ms=config.hop_ms,
                    source="MS-SNSD-derived-VAD",
                    feature_path=feature_path,
                    label_source=label_source,
                    class_label_path=class_label_rel.as_posix(),
                )
            )

    manifest_path = out_dir / "manifest.jsonl"
    write_manifest(records, manifest_path)
    _write_metadata(config, clean_files, noise_files, records, noise_pool)
    return manifest_path


def scan_ms_snsd_sources(raw_root: str | Path) -> tuple[list[Path], list[Path]]:
    root = Path(raw_root)
    clean_files: list[Path] = []
    noise_files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _AUDIO_EXTENSIONS:
            continue
        lower_parts = [part.lower() for part in path.parts]
        lower_name = path.name.lower()
        is_noise = any(part in {"noise", "noises"} or part.startswith("noise_") for part in lower_parts) or "noise" in lower_name
        is_clean = any(part in {"cleanspeech", "clean_speech", "clean"} or part.startswith("clean_") for part in lower_parts)
        if is_noise:
            noise_files.append(path)
        elif is_clean:
            clean_files.append(path)
    return sorted(clean_files), sorted(noise_files)


def split_clean_files(clean_files: list[Path], seed: int = 7) -> dict[str, list[Path]]:
    if not clean_files:
        raise ValueError("clean_files must be non-empty")
    rng = np.random.default_rng(seed)
    order = [clean_files[int(idx)] for idx in rng.permutation(len(clean_files))]
    n = len(order)
    if n == 1:
        return {"train": order, "val": order, "test": order}
    if n == 2:
        return {"train": [order[0]], "val": [order[1]], "test": [order[1]]}
    train_end = max(1, int(round(n * 0.8)))
    train_end = min(train_end, n - 2)
    val_end = max(train_end + 1, int(round(n * 0.9)))
    val_end = min(val_end, n - 1)
    return {"train": order[:train_end], "val": order[train_end:val_end], "test": order[val_end:]}


def _build_noise_pool(noise_files: list[Path], config: MSSNSDVADConfig) -> dict[str, list[Path]]:
    excluded_keywords = [keyword.lower() for keyword in config.exclude_noise_keywords]
    usable: list[Path] = []
    excluded: list[Path] = []
    hard_negative: list[Path] = []
    for path in noise_files:
        stem = path.stem.lower()
        if any(keyword in stem for keyword in excluded_keywords):
            excluded.append(path)
            continue
        usable.append(path)
        if any(keyword in stem for keyword in _HARD_NEGATIVE_KEYWORDS):
            hard_negative.append(path)
    return {"usable": sorted(usable), "excluded": sorted(excluded), "hard_negative": sorted(hard_negative)}


def _ensure_ms_snsd_repo(raw_root: Path) -> None:
    if (raw_root / ".git").exists() or (raw_root / "CleanSpeech").exists():
        return
    raw_root.parent.mkdir(parents=True, exist_ok=True)
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to download MS-SNSD. Install git or provide --raw-root with an existing checkout.")
    subprocess.run([git, "clone", "--depth", "1", MS_SNSD_GIT_URL, str(raw_root)], check=True)


def _extract_non_silent_chunks(
    waveform: np.ndarray,
    sample_rate: int,
    config: MSSNSDVADConfig,
    rng: random.Random,
) -> list[np.ndarray]:
    waveform = _normalize_peak(waveform)
    if len(waveform) == 0:
        return []
    min_len = int(round(config.speech_min_sec * sample_rate))
    max_len = max(min_len, int(round(config.speech_max_sec * sample_rate)))
    rms = float(np.sqrt(np.mean(waveform * waveform) + 1e-12))
    if rms < 1e-5:
        return []
    chunks: list[np.ndarray] = []
    cursor = 0
    while cursor + min_len <= len(waveform):
        length = int(round(rng.uniform(config.speech_min_sec, config.speech_max_sec) * sample_rate))
        length = min(max(length, min_len), max_len, len(waveform) - cursor)
        chunk = waveform[cursor : cursor + length]
        if float(np.sqrt(np.mean(chunk * chunk) + 1e-12)) > rms * 0.15:
            chunks.append(chunk.astype(np.float32))
        cursor += max(length, min_len)
    return chunks or [waveform.astype(np.float32)]


def _compose_vad_clip(
    speech_chunks: list[np.ndarray],
    noise: np.ndarray,
    snr_db: float,
    config: MSSNSDVADConfig,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    total_samples = int(round(config.clip_sec * config.sample_rate))
    background = _sample_noise(noise, total_samples, rng, np_rng)
    speech_only = np.zeros(total_samples, dtype=np.float32)
    speech_mask = np.zeros(total_samples, dtype=bool)
    cursor = int(round(rng.uniform(config.silence_min_sec, config.silence_max_sec) * config.sample_rate))
    placed = 0
    while cursor < total_samples:
        chunk = np.asarray(rng.choice(speech_chunks), dtype=np.float32)
        if len(chunk) == 0:
            break
        remaining = total_samples - cursor
        if remaining < int(round(config.speech_min_sec * config.sample_rate)):
            break
        if len(chunk) > remaining:
            chunk = chunk[:remaining]
        gain = rng.uniform(0.55, 1.0)
        end = cursor + len(chunk)
        speech_only[cursor:end] += _normalize_peak(chunk) * gain
        speech_mask[cursor:end] = True
        placed += 1
        silence = rng.uniform(config.silence_min_sec, config.silence_max_sec)
        cursor = end + int(round(silence * config.sample_rate))
    if placed == 0:
        chunk = _normalize_peak(np.asarray(speech_chunks[0], dtype=np.float32))
        end = min(total_samples, len(chunk))
        speech_only[:end] = chunk[:end]
        speech_mask[:end] = True

    speech_rms = float(np.sqrt(np.mean(speech_only[speech_mask] ** 2) + 1e-12)) if np.any(speech_mask) else 0.01
    noise_rms = float(np.sqrt(np.mean(background * background) + 1e-12))
    target_noise_rms = speech_rms / (10.0 ** (snr_db / 20.0))
    background = background * (target_noise_rms / max(noise_rms, 1e-8))
    waveform = speech_only + background
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 0.98:
        waveform = waveform / peak * 0.98
    return waveform.astype(np.float32), speech_mask, float(np.mean(speech_mask))


def _compose_vad_clip_v2(
    speech_chunks: list[np.ndarray],
    noise: np.ndarray,
    hard_noise: np.ndarray,
    snr_db: float,
    config: MSSNSDVADConfig,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_samples = int(round(config.clip_sec * config.sample_rate))
    waveform = np.zeros(total_samples, dtype=np.float32)
    speech_mask = np.zeros(total_samples, dtype=bool)
    event_mask = np.full(total_samples, MS_SNSD_EVENT_CLASSES["silence"], dtype=np.uint8)

    target_speech = int(round(total_samples * config.target_speech_ratio))
    target_speech = int(np.clip(target_speech, total_samples * config.speech_ratio_min, total_samples * config.speech_ratio_max))
    non_speech_samples = max(0, total_samples - target_speech)
    desired_hard = int(round(total_samples * config.hard_negative_ratio))
    desired_silence = int(round(total_samples * config.silence_ratio))
    desired_low = int(round(total_samples * config.low_noise_ratio))
    # Keep a real noise-only class even when the requested ratios fill the
    # whole clip. This prevents v2 from collapsing to speech / silence / hard
    # negatives only.
    min_noise_only = min(non_speech_samples, int(round(total_samples * 0.10)))
    available_for_other_negatives = max(0, non_speech_samples - min_noise_only)
    desired_other = max(1, desired_hard + desired_silence + desired_low)
    scale = min(1.0, available_for_other_negatives / desired_other)
    target_hard = int(round(desired_hard * scale))
    target_silence = int(round(desired_silence * scale))
    target_low = int(round(desired_low * scale))
    target_noise = max(0, total_samples - target_speech - target_hard - target_silence - target_low)

    speech_with_noise = int(round(target_speech * 0.7))
    clean_speech = max(0, target_speech - speech_with_noise)
    events: list[tuple[str, int]] = []
    events.extend(_split_duration("speech_with_noise", speech_with_noise, config.sample_rate, 0.35, 1.6, rng))
    events.extend(_split_duration("clean_speech", clean_speech, config.sample_rate, 0.35, 1.3, rng))
    events.extend(_split_duration("hard_negative_event", target_hard, config.sample_rate, 0.20, 1.4, rng))
    events.extend(_split_duration("noise_only_event", target_noise, config.sample_rate, 0.20, 1.4, rng))
    events.extend(_split_duration("low_background_noise", target_low, config.sample_rate, 0.25, 1.2, rng))
    events.extend(_split_duration("silence", target_silence, config.sample_rate, 0.20, 1.0, rng))
    rng.shuffle(events)

    cursor = 0
    for event_name, duration in events:
        if cursor >= total_samples:
            break
        duration = min(duration, total_samples - cursor)
        if duration <= 0:
            continue
        end = cursor + duration
        if event_name == "clean_speech":
            speech = _sample_speech_chunk(speech_chunks, duration, rng)
            waveform[cursor:end] += _normalize_peak(speech, target=rng.uniform(0.45, 0.85))
            speech_mask[cursor:end] = True
        elif event_name == "speech_with_noise":
            speech = _sample_speech_chunk(speech_chunks, duration, rng)
            speech = _normalize_peak(speech, target=rng.uniform(0.45, 0.85))
            bg = _sample_noise(noise, duration, rng, np_rng)
            waveform[cursor:end] += speech + _scale_noise_to_snr(bg, speech, snr_db)
            speech_mask[cursor:end] = True
        elif event_name == "hard_negative_event":
            event = _sample_noise(hard_noise, duration, rng, np_rng)
            waveform[cursor:end] += _normalize_peak(event, target=rng.uniform(0.12, 0.45))
        elif event_name == "noise_only_event":
            event = _sample_noise(noise, duration, rng, np_rng)
            waveform[cursor:end] += _normalize_peak(event, target=rng.uniform(0.05, 0.30))
        elif event_name == "low_background_noise":
            event = _sample_noise(noise, duration, rng, np_rng)
            waveform[cursor:end] += _normalize_peak(event, target=rng.uniform(0.008, 0.05))
        elif event_name == "silence" and rng.random() < 0.35:
            waveform[cursor:end] += np_rng.normal(0.0, rng.uniform(0.0002, 0.002), duration).astype(np.float32)
        event_mask[cursor:end] = MS_SNSD_EVENT_CLASSES[event_name]
        cursor = end

    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 0.98:
        waveform = waveform / peak * 0.98
    return waveform.astype(np.float32), speech_mask, event_mask.astype(np.uint8)


def _split_duration(event_name: str, total_samples: int, sample_rate: int, min_sec: float, max_sec: float, rng: random.Random) -> list[tuple[str, int]]:
    if total_samples <= 0:
        return []
    min_samples = max(1, int(round(min_sec * sample_rate)))
    max_samples = max(min_samples, int(round(max_sec * sample_rate)))
    remaining = int(total_samples)
    output: list[tuple[str, int]] = []
    while remaining > 0:
        duration = remaining if remaining <= max_samples else rng.randint(min_samples, min(max_samples, remaining))
        output.append((event_name, duration))
        remaining -= duration
    return output


def _sample_speech_chunk(speech_chunks: list[np.ndarray], duration: int, rng: random.Random) -> np.ndarray:
    parts: list[np.ndarray] = []
    remaining = duration
    while remaining > 0:
        chunk = np.asarray(rng.choice(speech_chunks), dtype=np.float32)
        if len(chunk) == 0:
            break
        if len(chunk) > remaining:
            start = rng.randint(0, len(chunk) - remaining)
            parts.append(chunk[start : start + remaining])
            remaining = 0
        else:
            parts.append(chunk)
            remaining -= len(chunk)
    if not parts:
        return np.zeros(duration, dtype=np.float32)
    out = np.concatenate(parts).astype(np.float32)
    if len(out) < duration:
        out = np.pad(out, (0, duration - len(out)))
    return out[:duration]


def _scale_noise_to_snr(noise: np.ndarray, speech: np.ndarray, snr_db: float) -> np.ndarray:
    speech_rms = float(np.sqrt(np.mean(speech * speech) + 1e-12))
    noise_rms = float(np.sqrt(np.mean(noise * noise) + 1e-12))
    target_noise_rms = speech_rms / (10.0 ** (snr_db / 20.0))
    return (noise * (target_noise_rms / max(noise_rms, 1e-8))).astype(np.float32)


def _event_mask_to_frame_labels(event_mask: np.ndarray, config: MSSNSDVADConfig) -> np.ndarray:
    count = frame_count(len(event_mask), config.sample_rate, config.frame_ms, config.hop_ms)
    frame_length = max(1, int(round(config.sample_rate * config.frame_ms / 1000.0)))
    hop_length = max(1, int(round(config.sample_rate * config.hop_ms / 1000.0)))
    required = (count - 1) * hop_length + frame_length
    padded = np.pad(event_mask.astype(np.uint8), (0, max(0, required - len(event_mask))), constant_values=MS_SNSD_EVENT_CLASSES["silence"])
    labels = np.empty(count, dtype=np.uint8)
    for idx in range(count):
        frame = padded[idx * hop_length : idx * hop_length + frame_length]
        labels[idx] = int(np.argmax(np.bincount(frame, minlength=max(MS_SNSD_EVENT_CLASSES.values()) + 1)))
    return labels


def _sample_noise(noise: np.ndarray, total_samples: int, rng: random.Random, np_rng: np.random.Generator) -> np.ndarray:
    noise = np.asarray(noise, dtype=np.float32)
    if len(noise) == 0:
        return np_rng.normal(0.0, 0.003, total_samples).astype(np.float32)
    if len(noise) >= total_samples:
        start = rng.randint(0, len(noise) - total_samples)
        return noise[start : start + total_samples].astype(np.float32)
    repeats = int(math.ceil(total_samples / len(noise)))
    tiled = np.tile(noise, repeats)
    return tiled[:total_samples].astype(np.float32)


def _normalize_peak(waveform: np.ndarray, target: float = 0.7) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32)
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak <= 1e-8:
        return waveform.astype(np.float32)
    return (waveform / peak * target).astype(np.float32)


def _write_metadata(
    config: MSSNSDVADConfig,
    clean_files: list[Path],
    noise_files: list[Path],
    records: list[ManifestRecord],
    noise_pool: dict[str, list[Path]],
) -> None:
    out_dir = Path(config.out_dir)
    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    speech_frames = 0
    total_frames = 0
    event_frame_counts: Counter[str] = Counter()
    for record in records:
        counts[record.split] = counts.get(record.split, 0) + 1
        labels = np.load(out_dir / record.label_path)
        speech_frames += int(labels.sum())
        total_frames += int(labels.size)
        if record.class_label_path:
            class_labels = np.load(out_dir / record.class_label_path).astype(np.uint8)
            for name, value in MS_SNSD_EVENT_CLASSES.items():
                event_frame_counts[name] += int(np.sum(class_labels == value))
    noise_type_counts = Counter(_noise_type(path) for path in noise_pool["usable"])
    metadata = {
        "source": "MS-SNSD-derived-VAD",
        "protocol": config.protocol,
        "ms_snsd_url": MS_SNSD_GIT_URL,
        "sample_rate": config.sample_rate,
        "frame_ms": config.frame_ms,
        "hop_ms": config.hop_ms,
        "n_mels": config.n_mels,
        "clip_sec": config.clip_sec,
        "total_hours_requested": config.total_hours,
        "snr_levels": config.snr_levels,
        "seed": config.seed,
        "target_speech_ratio": config.target_speech_ratio,
        "speech_ratio_min": config.speech_ratio_min,
        "speech_ratio_max": config.speech_ratio_max,
        "hard_negative_ratio": config.hard_negative_ratio,
        "silence_ratio": config.silence_ratio,
        "low_noise_ratio": config.low_noise_ratio,
        "clean_file_count": len(clean_files),
        "noise_file_count": len(noise_files),
        "usable_noise_file_count": len(noise_pool["usable"]),
        "excluded_noise_file_count": len(noise_pool["excluded"]),
        "hard_negative_noise_file_count": len(noise_pool["hard_negative"]),
        "excluded_noise_keywords": config.exclude_noise_keywords,
        "noise_type_counts": dict(sorted(noise_type_counts.items())),
        "event_classes": MS_SNSD_EVENT_CLASSES,
        "event_frame_counts": dict(sorted(event_frame_counts.items())),
        "event_type_counts": dict(sorted(event_frame_counts.items())),
        "split_clip_counts": counts,
        "speech_frame_ratio": float(speech_frames / max(total_frames, 1)),
        "silence_frame_ratio": float(event_frame_counts["silence"] / max(total_frames, 1)),
        "noise_only_frame_ratio": float(event_frame_counts["noise_only_event"] / max(total_frames, 1)),
        "hard_negative_frame_ratio": float(event_frame_counts["hard_negative_event"] / max(total_frames, 1)),
        "low_noise_frame_ratio": float(event_frame_counts["low_background_noise"] / max(total_frames, 1)),
        "note": "Protocol v2 composes human speech, silence, low noise, noise-only events, and hard negative non-speech events for VAD training.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def _noise_type(path: Path) -> str:
    stem = path.stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem
