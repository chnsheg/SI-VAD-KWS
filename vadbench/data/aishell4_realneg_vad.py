from __future__ import annotations

import csv
import json
import math
import re
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torchaudio

from vadbench.audio import save_audio
from vadbench.features import frame_count, log_mel_spectrogram, sample_mask_to_frame_labels
from vadbench.manifest import ManifestRecord, write_manifest


AISHELL4_URLS = {
    "train_L.tar.gz": "https://www.openslr.org/resources/111/train_L.tar.gz",
    "train_M.tar.gz": "https://www.openslr.org/resources/111/train_M.tar.gz",
    "train_S.tar.gz": "https://www.openslr.org/resources/111/train_S.tar.gz",
    "test.tar.gz": "https://www.openslr.org/resources/111/test.tar.gz",
}
AISHELL4_ARCHIVE_BY_SUBSET = {
    "train_l": "train_L.tar.gz",
    "train_m": "train_M.tar.gz",
    "train_s": "train_S.tar.gz",
    "test": "test.tar.gz",
}

FSD50K_URLS = {
    "FSD50K.dev_audio.z01": "https://zenodo.org/api/records/4060432/files/FSD50K.dev_audio.z01/content",
    "FSD50K.dev_audio.z02": "https://zenodo.org/api/records/4060432/files/FSD50K.dev_audio.z02/content",
    "FSD50K.dev_audio.z03": "https://zenodo.org/api/records/4060432/files/FSD50K.dev_audio.z03/content",
    "FSD50K.dev_audio.z04": "https://zenodo.org/api/records/4060432/files/FSD50K.dev_audio.z04/content",
    "FSD50K.dev_audio.z05": "https://zenodo.org/api/records/4060432/files/FSD50K.dev_audio.z05/content",
    "FSD50K.dev_audio.zip": "https://zenodo.org/records/4060432/files/FSD50K.dev_audio.zip?download=1",
    "FSD50K.eval_audio.z01": "https://zenodo.org/api/records/4060432/files/FSD50K.eval_audio.z01/content",
    "FSD50K.eval_audio.zip": "https://zenodo.org/records/4060432/files/FSD50K.eval_audio.zip?download=1",
    "FSD50K.ground_truth.zip": "https://zenodo.org/records/4060432/files/FSD50K.ground_truth.zip?download=1",
    "FSD50K.metadata.zip": "https://zenodo.org/records/4060432/files/FSD50K.metadata.zip?download=1",
}
FSD50K_SPLIT_ZIP_PARTS = {
    "dev": [
        "FSD50K.dev_audio.z01",
        "FSD50K.dev_audio.z02",
        "FSD50K.dev_audio.z03",
        "FSD50K.dev_audio.z04",
        "FSD50K.dev_audio.z05",
        "FSD50K.dev_audio.zip",
    ],
    "eval": [
        "FSD50K.eval_audio.z01",
        "FSD50K.eval_audio.zip",
    ],
}
FSD50K_FILE_SIZES = {
    "FSD50K.dev_audio.z01": 3221225472,
    "FSD50K.dev_audio.z02": 3221225472,
    "FSD50K.dev_audio.z03": 3221225472,
    "FSD50K.dev_audio.z04": 3221225472,
    "FSD50K.dev_audio.z05": 3221225472,
    "FSD50K.dev_audio.zip": 2306663327,
    "FSD50K.eval_audio.z01": 3221225472,
    "FSD50K.eval_audio.zip": 3037675767,
    "FSD50K.ground_truth.zip": 334701,
    "FSD50K.metadata.zip": 6700838,
}

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}
AISHELL4_EVENT_CLASSES = {"non_speech": 0, "speech": 1, "fsd50k_hard_negative": 2}
DEFAULT_FSD50K_EXCLUDE_KEYWORDS = [
    "speech",
    "conversation",
    "narration",
    "singing",
    "vocal",
    "human voice",
    "child speech",
    "children speaking",
    "crowd",
    "choir",
    "chant",
    "babbling",
]


@dataclass(frozen=True)
class SpeechInterval:
    recording_id: str
    start_sec: float
    end_sec: float
    speaker: str = ""


@dataclass
class Aishell4RealnegVADConfig:
    aishell4_root: Path = Path("data/raw/aishell4")
    fsd50k_root: Path = Path("data/raw/fsd50k")
    out_dir: Path = Path("data/aishell4_realneg_vad")
    download_aishell4: bool = False
    download_fsd50k: bool = False
    sample_rate: int = 16000
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    n_mels: int = 64
    chunk_sec: float = 10.0
    negative_ratio: float = 0.35
    mic_channel: int = 0
    seed: int = 7
    max_hours: float | None = None
    target_hours: float | None = None
    aishell4_hours: float | None = None
    fsd50k_hours: float | None = None
    aishell4_subsets: list[str] = field(default_factory=lambda: ["train_L", "train_M", "train_S"])
    aishell4_room_ratios: dict[str, float] = field(default_factory=lambda: {"L": 0.25, "M": 0.45, "S": 0.30})
    use_official_aishell4_test: bool = False
    fsd50k_train_source: str = "dev"
    fsd50k_test_source: str = "eval"
    precompute_features: bool = True
    fsd50k_exclude_keywords: list[str] = field(default_factory=lambda: list(DEFAULT_FSD50K_EXCLUDE_KEYWORDS))


@dataclass(frozen=True)
class FSD50KItem:
    audio_path: Path
    item_id: str
    labels: tuple[str, ...]
    split: str | None = None
    source_split: str | None = None


def prepare_aishell4_realneg_vad(config: Aishell4RealnegVADConfig) -> Path:
    rng = np.random.default_rng(config.seed)
    aishell_root = Path(config.aishell4_root)
    fsd_root = Path(config.fsd50k_root)
    out_dir = Path(config.out_dir)
    _ensure_aishell4_archives(aishell_root, config)
    _ensure_fsd50k_archives(fsd_root, config)

    intervals = scan_aishell4_intervals(aishell_root)
    audio_by_recording = scan_aishell4_audio(aishell_root)
    selected_audio = _select_aishell4_audio(audio_by_recording, intervals, config.mic_channel, config)
    if not selected_audio:
        raise RuntimeError(
            f"No AISHELL-4 audio with matching speaker activity annotations found under {aishell_root}. "
            "Provide extracted AISHELL-4 train/test directories or use --download-aishell4."
        )

    split_by_recording = split_aishell4_recordings(selected_audio, config.seed, config.use_official_aishell4_test)
    records: list[ManifestRecord] = []
    source_counts: Counter[str] = Counter()
    event_frame_counts: Counter[str] = Counter()
    split_seconds: Counter[str] = Counter()
    max_seconds_by_split = _target_seconds_by_split(_resolve_aishell4_hours(config), config.max_hours)
    room_seconds: Counter[tuple[str, str]] = Counter()
    room_targets = _aishell4_room_targets(max_seconds_by_split, config.aishell4_room_ratios)

    recording_order = _deterministic_permutation(sorted(selected_audio), rng)
    for recording_id in recording_order:
        split = split_by_recording[recording_id]
        audio_path = selected_audio[recording_id]
        room = aishell4_room_type(audio_path) or "unknown"
        recording_intervals = intervals[recording_id]
        for chunk_index, chunk_start in enumerate(_chunk_starts(audio_path, config.chunk_sec)):
            if max_seconds_by_split is not None and split_seconds[split] >= max_seconds_by_split[split]:
                break
            if _room_target_reached(split, room, room_seconds, room_targets):
                break
            waveform = _load_audio_chunk(audio_path, chunk_start, config)
            labels = intervals_to_frame_labels(recording_intervals, chunk_start, chunk_start + config.chunk_sec, config)
            class_labels = labels.astype(np.uint8)
            sample_id = f"aishell4_{recording_id}_{chunk_index:06d}"
            record = _write_example(
                out_dir=out_dir,
                sample_id=sample_id,
                split=split,
                waveform=waveform,
                labels=labels,
                class_labels=class_labels,
                sample_rate=config.sample_rate,
                frame_hop_ms=config.hop_ms,
                source="AISHELL4",
                label_source="aishell4-speaker-activity-union",
                video_id=recording_id,
                chunk_start_sec=chunk_start,
                chunk_end_sec=chunk_start + config.chunk_sec,
                config=config,
            )
            records.append(record)
            split_seconds[split] += float(record.duration_sec)
            room_seconds[(split, room)] += float(record.duration_sec)
            source_counts[record.source] += 1
            event_frame_counts["aishell4_speech"] += int(labels.sum())
            event_frame_counts["aishell4_non_speech"] += int(labels.size - labels.sum())

    fsd_items, excluded_fsd = scan_fsd50k_nonspeech(fsd_root, config.fsd50k_exclude_keywords)
    fsd_records = _prepare_fsd50k_records(fsd_items, records, out_dir, config, rng)
    for record in fsd_records:
        labels = np.load(out_dir / record.label_path)
        records.append(record)
        source_counts[record.source] += 1
        event_frame_counts["fsd50k_hard_negative"] += int(labels.size)

    if not records:
        raise RuntimeError("No records were generated for AISHELL-4 + FSD50K VAD")
    manifest_path = out_dir / "manifest.jsonl"
    write_manifest(records, manifest_path)
    _write_metadata(config, records, out_dir, selected_audio, intervals, fsd_items, excluded_fsd, source_counts, event_frame_counts)
    return manifest_path


def scan_aishell4_audio(root: str | Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in Path(root).rglob("*"):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        recording_id = normalize_recording_id(path.stem)
        grouped[recording_id].append(path)
    return {key: sorted(paths) for key, paths in grouped.items()}


def scan_aishell4_intervals(root: str | Path) -> dict[str, list[SpeechInterval]]:
    root = Path(root)
    intervals: dict[str, list[SpeechInterval]] = defaultdict(list)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        parsed: list[SpeechInterval] = []
        if suffix == ".rttm":
            parsed = parse_rttm(path)
        elif suffix == ".textgrid":
            parsed = parse_textgrid(path)
        elif suffix in {".lab", ".txt", ".csv", ".tsv"}:
            parsed = parse_generic_activity_file(path)
        for interval in parsed:
            if interval.end_sec > interval.start_sec:
                intervals[normalize_recording_id(interval.recording_id)].append(interval)
    return {key: merge_intervals(value) for key, value in intervals.items()}


def parse_rttm(path: str | Path) -> list[SpeechInterval]:
    intervals: list[SpeechInterval] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5 or parts[0].upper() != "SPEAKER":
                continue
            recording_id = parts[1]
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7] if len(parts) > 7 else ""
            intervals.append(SpeechInterval(recording_id, start, start + duration, speaker))
    return intervals


def parse_textgrid(path: str | Path) -> list[SpeechInterval]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    recording_id = Path(path).stem
    intervals: list[SpeechInterval] = []
    xmin: float | None = None
    xmax: float | None = None
    label = ""
    for line in text:
        stripped = line.strip()
        if stripped.startswith("xmin ="):
            xmin = _safe_float(stripped.split("=", 1)[1])
        elif stripped.startswith("xmax ="):
            xmax = _safe_float(stripped.split("=", 1)[1])
        elif stripped.startswith("text ="):
            label = stripped.split("=", 1)[1].strip().strip('"').strip()
            if xmin is not None and xmax is not None and _is_speech_textgrid_label(label):
                intervals.append(SpeechInterval(recording_id, xmin, xmax, label))
            xmin = None
            xmax = None
            label = ""
    return intervals


def parse_generic_activity_file(path: str | Path) -> list[SpeechInterval]:
    path = Path(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        parsed = _parse_activity_csv(path)
        if parsed:
            return parsed
    intervals: list[SpeechInterval] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[\s,]+", line)
            numbers = [_safe_float(part) for part in parts]
            numeric_positions = [idx for idx, value in enumerate(numbers) if value is not None]
            if len(numeric_positions) < 2:
                continue
            start = float(numbers[numeric_positions[0]])
            end_or_dur = float(numbers[numeric_positions[1]])
            end = end_or_dur if end_or_dur > start else start + end_or_dur
            recording_id = parts[0] if numeric_positions[0] > 0 else path.stem
            speaker = parts[-1] if parts[-1] and _safe_float(parts[-1]) is None else ""
            intervals.append(SpeechInterval(recording_id, start, end, speaker))
    return intervals


def intervals_to_frame_labels(
    intervals: list[SpeechInterval],
    chunk_start_sec: float,
    chunk_end_sec: float,
    config: Aishell4RealnegVADConfig,
) -> np.ndarray:
    samples = int(round((chunk_end_sec - chunk_start_sec) * config.sample_rate))
    mask = np.zeros(samples, dtype=bool)
    for interval in intervals:
        start = max(float(interval.start_sec), float(chunk_start_sec))
        end = min(float(interval.end_sec), float(chunk_end_sec))
        if end <= start:
            continue
        start_sample = max(0, int(round((start - chunk_start_sec) * config.sample_rate)))
        end_sample = min(samples, int(round((end - chunk_start_sec) * config.sample_rate)))
        if end_sample > start_sample:
            mask[start_sample:end_sample] = True
    return sample_mask_to_frame_labels(mask, config.sample_rate, config.frame_ms, config.hop_ms)


def merge_intervals(intervals: list[SpeechInterval]) -> list[SpeechInterval]:
    if not intervals:
        return []
    recording_id = intervals[0].recording_id
    ordered = sorted(intervals, key=lambda item: (item.start_sec, item.end_sec))
    merged: list[SpeechInterval] = []
    cur_start = float(ordered[0].start_sec)
    cur_end = float(ordered[0].end_sec)
    for interval in ordered[1:]:
        if interval.start_sec <= cur_end:
            cur_end = max(cur_end, float(interval.end_sec))
        else:
            merged.append(SpeechInterval(recording_id, cur_start, cur_end, "union"))
            cur_start = float(interval.start_sec)
            cur_end = float(interval.end_sec)
    merged.append(SpeechInterval(recording_id, cur_start, cur_end, "union"))
    return merged


def split_recording_ids(recording_ids: list[str], seed: int = 7) -> dict[str, str]:
    if not recording_ids:
        return {}
    rng = np.random.default_rng(seed)
    order = [recording_ids[int(idx)] for idx in rng.permutation(len(recording_ids))]
    if len(order) == 1:
        return {order[0]: "train"}
    if len(order) == 2:
        return {order[0]: "train", order[1]: "test"}
    train_end = max(1, int(round(len(order) * 0.8)))
    train_end = min(train_end, len(order) - 2)
    val_end = max(train_end + 1, int(round(len(order) * 0.9)))
    val_end = min(val_end, len(order) - 1)
    out: dict[str, str] = {}
    for recording_id in order[:train_end]:
        out[recording_id] = "train"
    for recording_id in order[train_end:val_end]:
        out[recording_id] = "val"
    for recording_id in order[val_end:]:
        out[recording_id] = "test"
    return out


def split_aishell4_recordings(
    selected_audio: dict[str, Path],
    seed: int = 7,
    use_official_test: bool = False,
) -> dict[str, str]:
    if not use_official_test:
        return split_recording_ids(sorted(selected_audio), seed)
    train_val_ids = sorted(
        recording_id for recording_id, path in selected_audio.items() if _aishell4_subset_name(path) != "test"
    )
    test_ids = sorted(recording_id for recording_id, path in selected_audio.items() if _aishell4_subset_name(path) == "test")
    rng = np.random.default_rng(seed)
    order = [train_val_ids[int(idx)] for idx in rng.permutation(len(train_val_ids))]
    if len(order) <= 1:
        out = {recording_id: "train" for recording_id in order}
    else:
        train_end = max(1, int(round(len(order) * 0.9)))
        train_end = min(train_end, len(order) - 1)
        out = {recording_id: "train" for recording_id in order[:train_end]}
        out.update({recording_id: "val" for recording_id in order[train_end:]})
    out.update({recording_id: "test" for recording_id in test_ids})
    return out


def aishell4_room_type(path_or_id: str | Path) -> str | None:
    text = str(path_or_id).replace("\\", "/")
    parts = [part.lower() for part in Path(text).parts]
    joined = "/".join(parts)
    if "train_l" in joined or re.search(r"(^|[_/-])l([_/-]|$)", joined):
        return "L"
    if "train_m" in joined or re.search(r"(^|[_/-])m([_/-]|$)", joined):
        return "M"
    if "train_s" in joined or re.search(r"(^|[_/-])s([_/-]|$)", joined):
        return "S"
    return None


def scan_fsd50k_nonspeech(root: str | Path, exclude_keywords: list[str] | None = None) -> tuple[list[FSD50KItem], list[dict[str, str]]]:
    root = Path(root)
    if not root.exists():
        raise RuntimeError(f"FSD50K root does not exist: {root}")
    exclude_keywords = [item.lower() for item in (exclude_keywords or DEFAULT_FSD50K_EXCLUDE_KEYWORDS)]
    audio_by_stem = {path.stem: path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS}
    mid_to_label = _load_fsd50k_vocabulary(root)
    rows = _load_fsd50k_metadata_rows(root)
    if not rows:
        raise RuntimeError(
            f"No FSD50K metadata CSV with fname/labels fields found under {root}. "
            "Download FSD50K.ground_truth.zip and FSD50K.metadata.zip or provide an extracted FSD50K root."
        )
    included: list[FSD50KItem] = []
    excluded: list[dict[str, str]] = []
    seen_included: set[tuple[str | None, str]] = set()
    seen_excluded: set[str] = set()
    for row in rows:
        item_id = _row_value(row, ["fname", "filename", "file_name", "audio_id", "id"])
        if not item_id:
            continue
        stem = Path(item_id).stem
        audio_path = audio_by_stem.get(stem)
        if audio_path is None:
            continue
        labels = _labels_from_fsd50k_row(row, mid_to_label)
        label_text = " ".join(labels).lower()
        if any(keyword in label_text for keyword in exclude_keywords):
            if stem not in seen_excluded:
                excluded.append({"id": stem, "labels": ";".join(labels), "reason": "speech-like-label"})
                seen_excluded.add(stem)
            continue
        split = _row_value(row, ["split", "subset", "set"])
        source_split = fsd50k_source_split(audio_path)
        dedupe_key = (source_split, stem)
        if dedupe_key in seen_included:
            continue
        seen_included.add(dedupe_key)
        included.append(
            FSD50KItem(
                audio_path=audio_path,
                item_id=stem,
                labels=tuple(labels),
                split=split or None,
                source_split=source_split,
            )
        )
    return sorted(included, key=lambda item: item.item_id), excluded


def fsd50k_source_split(path: str | Path) -> str | None:
    lower = str(path).replace("\\", "/").lower()
    if "eval_audio" in lower or "/eval/" in lower or "\\eval\\" in lower:
        return "eval"
    if "dev_audio" in lower or "/dev/" in lower or "\\dev\\" in lower:
        return "dev"
    return None


def normalize_recording_id(value: str) -> str:
    stem = Path(value).stem
    patterns = [
        r"([_-](?:ch|channel|mic|microphone)?\d+)$",
        r"([._-]CH\d+)$",
        r"([._-]Mic\d+)$",
    ]
    normalized = stem
    for pattern in patterns:
        normalized = re.sub(pattern, "", normalized, flags=re.IGNORECASE)
    return normalized


def _select_aishell4_audio(
    audio_by_recording: dict[str, list[Path]],
    intervals: dict[str, list[SpeechInterval]],
    mic_channel: int,
    config: Aishell4RealnegVADConfig,
) -> dict[str, Path]:
    selected_subsets = {_normalize_aishell4_subset(item) for item in config.aishell4_subsets if item}
    if config.use_official_aishell4_test:
        selected_subsets.add("test")
    selected: dict[str, Path] = {}
    for recording_id, recording_intervals in intervals.items():
        if not recording_intervals:
            continue
        candidates = [
            path
            for path in audio_by_recording.get(recording_id, [])
            if _aishell4_subset_name(path) is None or _aishell4_subset_name(path) in selected_subsets
        ]
        if not candidates:
            continue
        selected[recording_id] = candidates[min(max(0, mic_channel), len(candidates) - 1)]
    return selected


def _chunk_starts(audio_path: Path, chunk_sec: float) -> list[float]:
    info = torchaudio.info(str(audio_path))
    duration = float(info.num_frames / info.sample_rate)
    if duration <= 0:
        return []
    count = max(1, int(math.ceil(duration / chunk_sec)))
    return [idx * float(chunk_sec) for idx in range(count)]


def _load_audio_chunk(audio_path: Path, chunk_start_sec: float, config: Aishell4RealnegVADConfig) -> np.ndarray:
    info = torchaudio.info(str(audio_path))
    src_sr = int(info.sample_rate)
    frame_offset = int(round(chunk_start_sec * src_sr))
    num_frames = int(round(config.chunk_sec * src_sr))
    waveform, sample_rate = torchaudio.load(str(audio_path), frame_offset=frame_offset, num_frames=num_frames)
    waveform = waveform.float()
    if waveform.ndim == 2 and waveform.shape[0] > 1:
        channel = min(max(0, int(config.mic_channel)), waveform.shape[0] - 1)
        waveform = waveform[channel : channel + 1]
    if sample_rate != config.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, config.sample_rate)
    mono = waveform.mean(dim=0).cpu().numpy().astype(np.float32)
    target_samples = int(round(config.chunk_sec * config.sample_rate))
    if len(mono) < target_samples:
        mono = np.pad(mono, (0, target_samples - len(mono)))
    return mono[:target_samples].astype(np.float32)


def _write_example(
    out_dir: Path,
    sample_id: str,
    split: str,
    waveform: np.ndarray,
    labels: np.ndarray,
    class_labels: np.ndarray,
    sample_rate: int,
    frame_hop_ms: float,
    source: str,
    label_source: str,
    video_id: str,
    chunk_start_sec: float,
    chunk_end_sec: float,
    config: Aishell4RealnegVADConfig,
) -> ManifestRecord:
    audio_rel = Path("audio") / split / f"{sample_id}.wav"
    label_rel = Path("labels") / split / f"{sample_id}.npy"
    class_rel = Path("event_labels") / split / f"{sample_id}.npy"
    feature_rel = Path("features") / split / f"{sample_id}.logmel{config.n_mels}.npy"
    save_audio(out_dir / audio_rel, waveform, sample_rate)
    (out_dir / label_rel).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / label_rel, labels.astype(np.uint8))
    (out_dir / class_rel).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / class_rel, class_labels.astype(np.uint8))
    feature_path: str | None = None
    if config.precompute_features:
        features = log_mel_spectrogram(
            waveform,
            sample_rate,
            n_mels=config.n_mels,
            frame_ms=config.frame_ms,
            hop_ms=config.hop_ms,
            normalize=True,
        )
        features = features[: len(labels)]
        (out_dir / feature_rel).parent.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / feature_rel, features.astype(np.float32))
        feature_path = feature_rel.as_posix()
    return ManifestRecord(
        id=sample_id,
        audio_path=audio_rel.as_posix(),
        label_path=label_rel.as_posix(),
        split=split,
        sample_rate=sample_rate,
        duration_sec=float(len(waveform) / sample_rate),
        frame_hop_ms=frame_hop_ms,
        source=source,
        feature_path=feature_path,
        video_id=video_id,
        chunk_start_sec=chunk_start_sec,
        chunk_end_sec=chunk_end_sec,
        label_source=label_source,
        class_label_path=class_rel.as_posix(),
    )


def _prepare_fsd50k_records(
    fsd_items: list[FSD50KItem],
    aishell_records: list[ManifestRecord],
    out_dir: Path,
    config: Aishell4RealnegVADConfig,
    rng: np.random.Generator,
) -> list[ManifestRecord]:
    if not fsd_items:
        return []
    fsd_seconds_by_split = _target_seconds_by_split(_resolve_fsd50k_hours(config), None)
    if fsd_seconds_by_split is None and (not aishell_records or config.negative_ratio <= 0.0):
        return []
    items_by_split = _split_fsd50k_items(fsd_items, config, rng)
    output: list[ManifestRecord] = []
    for split in ("train", "val", "test"):
        if fsd_seconds_by_split is not None:
            target_seconds = fsd_seconds_by_split[split]
        else:
            split_counts = Counter(record.split for record in aishell_records)
            target_count = int(round(split_counts[split] * config.negative_ratio / max(1.0 - config.negative_ratio, 1e-6)))
            target_seconds = target_count * float(config.chunk_sec)
        candidates = items_by_split.get(split) or fsd_items
        if not candidates or target_seconds <= 0:
            continue
        order = [candidates[int(idx)] for idx in rng.permutation(len(candidates))]
        cursor = 0
        written_seconds = 0.0
        while written_seconds < target_seconds and cursor < len(order):
            item = order[cursor]
            cursor += 1
            for chunk_idx, chunk_start in enumerate(_chunk_starts(item.audio_path, config.chunk_sec)):
                if written_seconds >= target_seconds:
                    break
                waveform = _load_audio_chunk(item.audio_path, chunk_start, config)
                labels = np.zeros(frame_count(len(waveform), config.sample_rate, config.frame_ms, config.hop_ms), dtype=np.uint8)
                class_labels = np.full(labels.shape, AISHELL4_EVENT_CLASSES["fsd50k_hard_negative"], dtype=np.uint8)
                sample_id = f"fsd50k_{split}_{item.item_id}_{chunk_idx:04d}"
                output.append(
                    _write_example(
                        out_dir=out_dir,
                        sample_id=sample_id,
                        split=split,
                        waveform=waveform,
                        labels=labels,
                        class_labels=class_labels,
                        sample_rate=config.sample_rate,
                        frame_hop_ms=config.hop_ms,
                        source="FSD50K-hard-negative",
                        label_source="fsd50k-nonspeech-negative",
                        video_id=item.item_id,
                        chunk_start_sec=chunk_start,
                        chunk_end_sec=chunk_start + config.chunk_sec,
                        config=config,
                    )
                )
                written_seconds += float(config.chunk_sec)
    return output


def _split_fsd50k_items(
    fsd_items: list[FSD50KItem],
    config: Aishell4RealnegVADConfig,
    rng: np.random.Generator,
) -> dict[str, list[FSD50KItem]]:
    train_source = _normalize_fsd_source(config.fsd50k_train_source)
    test_source = _normalize_fsd_source(config.fsd50k_test_source)
    train_val_items = [item for item in fsd_items if _source_matches(item.source_split, train_source)]
    test_items = [item for item in fsd_items if _source_matches(item.source_split, test_source)]
    by_split: dict[str, list[FSD50KItem]] = defaultdict(list)
    unspecified_train_val: list[FSD50KItem] = []
    for item in train_val_items:
        split = _normalize_split(item.split)
        if split in {"train", "val"}:
            by_split[split].append(item)
        else:
            unspecified_train_val.append(item)
    if unspecified_train_val:
        order = [unspecified_train_val[int(idx)] for idx in rng.permutation(len(unspecified_train_val))]
        val_count = max(1, int(round(len(order) * 0.1))) if len(order) > 1 else 0
        by_split["val"].extend(order[:val_count])
        by_split["train"].extend(order[val_count:])
    for item in test_items:
        by_split["test"].append(item)
    if not by_split["test"]:
        for item in fsd_items:
            if _normalize_split(item.split) == "test":
                by_split["test"].append(item)
    return {split: sorted(items, key=lambda item: item.item_id) for split, items in by_split.items()}


def _write_metadata(
    config: Aishell4RealnegVADConfig,
    records: list[ManifestRecord],
    out_dir: Path,
    selected_audio: dict[str, Path],
    intervals: dict[str, list[SpeechInterval]],
    fsd_items: list[FSD50KItem],
    excluded_fsd: list[dict[str, str]],
    source_counts: Counter[str],
    event_frame_counts: Counter[str],
) -> None:
    split_counts = Counter(record.split for record in records)
    split_seconds = Counter({split: 0.0 for split in ("train", "val", "test")})
    source_clip_counts: Counter[str] = Counter()
    source_seconds: Counter[str] = Counter()
    room_clip_counts: Counter[str] = Counter()
    room_recording_ids: dict[str, set[str]] = defaultdict(set)
    source_split_clip_counts: Counter[str] = Counter()
    for record in records:
        split_seconds[record.split] += float(record.duration_sec)
        source_clip_counts[record.source] += 1
        source_seconds[record.source] += float(record.duration_sec)
        source_split_clip_counts[f"{record.split}:{record.source}"] += 1
        if record.source == "AISHELL4":
            room = aishell4_room_type(record.video_id or "") or _room_from_record_path(selected_audio.get(record.video_id or "")) or "unknown"
            room_clip_counts[room] += 1
            if record.video_id:
                room_recording_ids[room].add(record.video_id)
    total_frames = sum(event_frame_counts.values())
    metadata = {
        "source": "AISHELL4-realneg-VAD",
        "sample_rate": config.sample_rate,
        "frame_ms": config.frame_ms,
        "hop_ms": config.hop_ms,
        "n_mels": config.n_mels,
        "chunk_sec": config.chunk_sec,
        "negative_ratio": config.negative_ratio,
        "target_hours": config.target_hours,
        "aishell4_hours": config.aishell4_hours,
        "fsd50k_hours": config.fsd50k_hours,
        "aishell4_subsets": config.aishell4_subsets,
        "aishell4_room_ratios": dict(sorted(_normalize_room_ratios(config.aishell4_room_ratios).items())),
        "use_official_aishell4_test": config.use_official_aishell4_test,
        "fsd50k_train_source": config.fsd50k_train_source,
        "fsd50k_test_source": config.fsd50k_test_source,
        "mic_channel": config.mic_channel,
        "seed": config.seed,
        "max_hours": config.max_hours,
        "event_classes": AISHELL4_EVENT_CLASSES,
        "aishell4_recording_count": len(selected_audio),
        "room_recording_counts": {room: len(ids) for room, ids in sorted(room_recording_ids.items())},
        "aishell4_interval_count": sum(len(items) for items in intervals.values()),
        "fsd50k_included_item_count": len(fsd_items),
        "fsd50k_excluded_item_count": len(excluded_fsd),
        "fsd50k_exclude_keywords": config.fsd50k_exclude_keywords,
        "split_clip_counts": dict(sorted(split_counts.items())),
        "split_hours": {split: seconds / 3600.0 for split, seconds in sorted(split_seconds.items())},
        "source_clip_counts": dict(sorted(source_counts.items())),
        "source_hours": {source: seconds / 3600.0 for source, seconds in sorted(source_seconds.items())},
        "source_split_clip_counts": dict(sorted(source_split_clip_counts.items())),
        "room_clip_counts": dict(sorted(room_clip_counts.items())),
        "event_frame_counts": dict(sorted(event_frame_counts.items())),
        "speech_frame_ratio": float(event_frame_counts["aishell4_speech"] / max(total_frames, 1)),
        "fsd50k_hard_negative_frame_ratio": float(event_frame_counts["fsd50k_hard_negative"] / max(total_frames, 1)),
        "aishell4_url": "https://www.openslr.org/111/",
        "fsd50k_url": "https://zenodo.org/records/4060432",
        "note": "AISHELL-4 speaker activity intervals are unioned into binary VAD labels; FSD50K non-speech clips are hard-negative all-zero labels.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    with (out_dir / "fsd50k_excluded_preview.json").open("w", encoding="utf-8") as handle:
        json.dump(excluded_fsd[:500], handle, indent=2, sort_keys=True)


def _room_from_record_path(path: Path | None) -> str | None:
    return aishell4_room_type(path) if path is not None else None


def _parse_activity_csv(path: Path) -> list[SpeechInterval]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if not reader.fieldnames:
                return []
            fieldnames = {name.lower(): name for name in reader.fieldnames}
            start_key = _first_existing(fieldnames, ["start", "start_time", "start_sec", "begin", "tbeg"])
            end_key = _first_existing(fieldnames, ["end", "end_time", "end_sec", "stop"])
            dur_key = _first_existing(fieldnames, ["duration", "dur", "tdur"])
            rec_key = _first_existing(fieldnames, ["recording_id", "recording", "file_id", "file", "utt", "meeting_id", "session"])
            speaker_key = _first_existing(fieldnames, ["speaker", "speaker_id", "spk"])
            if start_key is None or (end_key is None and dur_key is None):
                return []
            out: list[SpeechInterval] = []
            for row in reader:
                start = _safe_float(row.get(start_key, ""))
                end = _safe_float(row.get(end_key, "")) if end_key else None
                duration = _safe_float(row.get(dur_key, "")) if dur_key else None
                if start is None:
                    continue
                final_end = end if end is not None else start + float(duration or 0.0)
                recording_id = row.get(rec_key, "") if rec_key else path.stem
                speaker = row.get(speaker_key, "") if speaker_key else ""
                out.append(SpeechInterval(recording_id or path.stem, float(start), float(final_end), speaker))
            return out
    except csv.Error:
        return []


def _load_fsd50k_vocabulary(root: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in root.rglob("*.csv"):
        if "vocab" not in path.name.lower():
            continue
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                mid = _row_value(row, ["mids", "mid", "audioset_id"])
                label = _row_value(row, ["label", "labels", "display_name", "name"])
                if mid and label:
                    mapping[mid] = label
    return mapping


def _load_fsd50k_metadata_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in root.rglob("*.csv"):
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            try:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue
                fields = {field.lower() for field in reader.fieldnames}
                if not fields.intersection({"fname", "filename", "file_name", "audio_id", "id"}):
                    continue
                if not fields.intersection({"labels", "label", "tags", "mids", "mid"}):
                    continue
                rows.extend(dict(row) for row in reader)
            except csv.Error:
                continue
    return rows


def _labels_from_fsd50k_row(row: dict[str, str], mid_to_label: dict[str, str]) -> tuple[str, ...]:
    raw = _row_value(row, ["labels", "label", "tags", "display_name", "name"])
    labels = _split_label_field(raw)
    mids = _split_label_field(_row_value(row, ["mids", "mid", "audioset_id"]))
    labels.extend(mid_to_label.get(mid, mid) for mid in mids)
    return tuple(label for label in labels if label)


def _split_label_field(value: str | None) -> list[str]:
    if not value:
        return []
    value = value.strip().strip("[]")
    parts = re.split(r"[,;|]", value)
    return [part.strip().strip("'\"").replace("_", " ") for part in parts if part.strip()]


def _row_value(row: dict[str, str], keys: list[str]) -> str | None:
    lowered = {key.lower(): value for key, value in row.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _first_existing(fieldnames: dict[str, str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in fieldnames:
            return fieldnames[candidate]
    return None


def _is_speech_textgrid_label(label: str) -> bool:
    if not label:
        return False
    lower = label.lower()
    return lower not in {"sil", "sp", "silence", "noise", "non-speech", "nonspeech", "<sil>", "<noise>"}


def _safe_float(value: object) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _normalize_split(value: str | None) -> str | None:
    if value is None:
        return None
    lower = value.lower()
    if lower in {"train", "training", "dev"}:
        return "train"
    if lower in {"val", "valid", "validation"}:
        return "val"
    if lower in {"test", "eval", "evaluation"}:
        return "test"
    return None


def _target_seconds_by_split(hours: float | None, fallback_max_hours: float | None) -> dict[str, float] | None:
    selected_hours = hours if hours is not None else fallback_max_hours
    if selected_hours is None or selected_hours <= 0:
        return None
    total = float(selected_hours) * 3600.0
    return {"train": total * 0.8, "val": total * 0.1, "test": total * 0.1}


def _max_seconds_by_split(max_hours: float | None) -> dict[str, float] | None:
    return _target_seconds_by_split(max_hours, None)


def _resolve_aishell4_hours(config: Aishell4RealnegVADConfig) -> float | None:
    if config.aishell4_hours is not None:
        return float(config.aishell4_hours)
    if config.target_hours is not None:
        return float(config.target_hours) * max(0.0, 1.0 - float(config.negative_ratio))
    return None


def _resolve_fsd50k_hours(config: Aishell4RealnegVADConfig) -> float | None:
    if config.fsd50k_hours is not None:
        return float(config.fsd50k_hours)
    if config.target_hours is not None:
        return float(config.target_hours) * max(0.0, float(config.negative_ratio))
    return None


def _aishell4_room_targets(
    split_seconds: dict[str, float] | None,
    room_ratios: dict[str, float],
) -> dict[tuple[str, str], float] | None:
    if split_seconds is None:
        return None
    normalized = _normalize_room_ratios(room_ratios)
    if not normalized:
        return None
    targets: dict[tuple[str, str], float] = {}
    for split, seconds in split_seconds.items():
        if split == "test":
            continue
        for room, ratio in normalized.items():
            targets[(split, room)] = float(seconds) * float(ratio)
    return targets


def _room_target_reached(
    split: str,
    room: str,
    room_seconds: Counter[tuple[str, str]],
    room_targets: dict[tuple[str, str], float] | None,
) -> bool:
    if room_targets is None:
        return False
    target = room_targets.get((split, room))
    if target is None or target <= 0:
        return False
    return room_seconds[(split, room)] >= target


def _normalize_room_ratios(room_ratios: dict[str, float]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for key, value in room_ratios.items():
        room = str(key).strip().upper()
        if room not in {"L", "M", "S"}:
            continue
        ratio = max(0.0, float(value))
        if ratio > 0.0:
            totals[room] = totals.get(room, 0.0) + ratio
    total = sum(totals.values())
    if total <= 0.0:
        return {}
    return {room: value / total for room, value in totals.items()}


def _deterministic_permutation(items: list[str], rng: np.random.Generator) -> list[str]:
    return [items[int(idx)] for idx in rng.permutation(len(items))] if items else []


def _normalize_aishell4_subset(value: str) -> str:
    lower = str(value).strip().lower()
    if lower in {"l", "large", "train_l", "train-l"}:
        return "train_l"
    if lower in {"m", "medium", "train_m", "train-m"}:
        return "train_m"
    if lower in {"s", "small", "train_s", "train-s"}:
        return "train_s"
    if lower in {"test", "eval"}:
        return "test"
    return lower


def _aishell4_subset_name(path: str | Path) -> str | None:
    lower = str(path).replace("\\", "/").lower()
    for name in ("train_l", "train_m", "train_s", "test"):
        if re.search(rf"(^|/){re.escape(name)}(/|$)", lower):
            return name
    return None


def _normalize_fsd_source(value: str | None) -> str | None:
    if value is None:
        return None
    lower = str(value).strip().lower()
    if lower in {"dev", "train", "training"}:
        return "dev"
    if lower in {"eval", "test", "evaluation"}:
        return "eval"
    if lower in {"any", "all", ""}:
        return None
    return lower


def _source_matches(source_split: str | None, requested: str | None) -> bool:
    return requested is None or source_split is None or source_split == requested


def _ensure_aishell4_archives(root: Path, config: Aishell4RealnegVADConfig) -> None:
    root.mkdir(parents=True, exist_ok=True)
    subsets = {_normalize_aishell4_subset(item) for item in config.aishell4_subsets if item}
    if config.use_official_aishell4_test:
        subsets.add("test")
    if not subsets:
        subsets = {"train_l", "train_m", "train_s"}
    filenames = [AISHELL4_ARCHIVE_BY_SUBSET[subset] for subset in sorted(subsets) if subset in AISHELL4_ARCHIVE_BY_SUBSET]
    for filename in filenames:
        url = AISHELL4_URLS.get(filename)
        archive = root / filename
        if config.download_aishell4 and url is not None:
            _download_file(url, archive)
        if archive.exists():
            _extract_archive(archive, root)


def _ensure_fsd50k_archives(root: Path, config: Aishell4RealnegVADConfig) -> None:
    root.mkdir(parents=True, exist_ok=True)
    sources = {_normalize_fsd_source(config.fsd50k_train_source), _normalize_fsd_source(config.fsd50k_test_source)}
    metadata_filenames = ["FSD50K.ground_truth.zip", "FSD50K.metadata.zip"]
    for filename in metadata_filenames:
        archive = root / filename
        if config.download_fsd50k:
            _download_file(FSD50K_URLS[filename], archive, expected_size=FSD50K_FILE_SIZES.get(filename))
        if archive.exists():
            _extract_archive(archive, root)
    audio_sources: set[str] = set()
    if None in sources or "dev" in sources:
        audio_sources.add("dev")
    if None in sources or "eval" in sources:
        audio_sources.add("eval")
    for source in sorted(audio_sources):
        filenames = FSD50K_SPLIT_ZIP_PARTS[source]
        strict = config.download_fsd50k or _resolve_fsd50k_hours(config) is not None
        for filename in filenames:
            archive = root / filename
            if config.download_fsd50k:
                _download_file(FSD50K_URLS[filename], archive, expected_size=FSD50K_FILE_SIZES.get(filename))
        _extract_fsd50k_split_zip(root, source, strict=strict)


def _extract_fsd50k_split_zip(root: Path, source: str, strict: bool = True) -> None:
    expected_dir = root / ("FSD50K.dev_audio" if source == "dev" else "FSD50K.eval_audio")
    if expected_dir.exists() and any(path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS for path in expected_dir.rglob("*")):
        return
    marker = root / f".extracted_FSD50K.{source}_audio.split.done"
    if marker.exists():
        return
    parts = [root / filename for filename in FSD50K_SPLIT_ZIP_PARTS[source]]
    missing = [path.name for path in parts if not path.exists()]
    incomplete = [
        path.name
        for path in parts
        if path.exists()
        and FSD50K_FILE_SIZES.get(path.name) is not None
        and path.stat().st_size != FSD50K_FILE_SIZES[path.name]
    ]
    if missing or incomplete:
        if not strict:
            return
        detail = []
        if missing:
            detail.append(f"missing parts: {', '.join(missing)}")
        if incomplete:
            detail.append(f"incomplete parts: {', '.join(incomplete)}")
        raise RuntimeError(
            f"FSD50K {source} audio is a split ZIP and has {', '.join(detail)}. "
            "Rerun prepare with --download-fsd50k, or download all listed parts from Zenodo into the FSD50K root."
        )
    seven_zip = _find_7zip()
    if seven_zip is not None:
        subprocess.run(
            [str(seven_zip), "x", str(parts[-1]), f"-o{root}", "-y"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    else:
        combined = root / f"FSD50K.{source}_audio.combined.zip"
        try:
            zip_tool = _find_zip_tool()
            if zip_tool is not None:
                subprocess.run(
                    [str(zip_tool), "-s", "0", str(parts[-1]), "--out", str(combined)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            else:
                with combined.open("wb") as out_handle:
                    for part in parts:
                        with part.open("rb") as in_handle:
                            shutil.copyfileobj(in_handle, out_handle, length=1024 * 1024 * 16)
            _extract_archive(combined, root)
        finally:
            if combined.exists():
                combined.unlink()
    marker.write_text("ok", encoding="utf-8")


def _download_aishell4(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for filename, url in AISHELL4_URLS.items():
        archive = root / filename
        _download_file(url, archive)
        _extract_archive(archive, root)


def _download_fsd50k(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for filename, url in FSD50K_URLS.items():
        archive = root / filename
        _download_file(url, archive, expected_size=FSD50K_FILE_SIZES.get(filename))
        _extract_archive(archive, root)


def _download_file(url: str, path: Path, expected_size: int | None = None) -> None:
    if path.exists() and path.stat().st_size > 0:
        if expected_size is None or path.stat().st_size == expected_size:
            return
        path.unlink()
    elif path.exists():
        path.unlink()
    part_path = path.with_suffix(path.suffix + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    resume_at = part_path.stat().st_size if part_path.exists() else 0
    if expected_size is not None and resume_at > expected_size:
        part_path.unlink()
        resume_at = 0
    request = urllib.request.Request(url)
    if resume_at > 0:
        request.add_header("Range", f"bytes={resume_at}-")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            mode = "ab" if resume_at > 0 and getattr(response, "status", 200) == 206 else "wb"
            if mode == "wb" and resume_at > 0:
                resume_at = 0
            with part_path.open(mode) as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024 * 8)
    except Exception:
        if part_path.exists() and part_path.stat().st_size <= 0:
            part_path.unlink()
        raise
    if not part_path.exists() or part_path.stat().st_size <= 0:
        raise RuntimeError(f"Downloaded file is empty: {path.name}")
    if expected_size is not None and part_path.stat().st_size != expected_size:
        actual = part_path.stat().st_size
        raise RuntimeError(f"Downloaded file has wrong size: {path.name}, expected {expected_size}, got {actual}")
    part_path.replace(path)


def _extract_archive(path: Path, out_dir: Path) -> None:
    marker = out_dir / f".extracted_{path.name}.done"
    if marker.exists():
        return
    if path.suffix == ".zip":
        try:
            with zipfile.ZipFile(path) as archive:
                archive.extractall(out_dir)
        except zipfile.BadZipFile as exc:
            _extract_zip_with_external_tool(path, out_dir, exc)
    elif path.name.endswith(".tar.gz") or path.name.endswith(".tgz"):
        with tarfile.open(path, "r:gz") as archive:
            archive.extractall(out_dir)
    marker.write_text("ok", encoding="utf-8")


def _extract_zip_with_external_tool(path: Path, out_dir: Path, original_error: Exception) -> None:
    seven_zip = _find_7zip()
    unzip_tool = _find_unzip_tool()
    if seven_zip is None and unzip_tool is None:
        raise RuntimeError(
            f"Cannot extract {path.name} with Python zipfile ({original_error}). "
            "Install 7-Zip or add unzip.exe to PATH, then rerun prepare."
        ) from original_error
    if seven_zip is not None:
        command = [str(seven_zip), "x", str(path), f"-o{out_dir}", "-y"]
    else:
        command = [str(unzip_tool), "-o", str(path), "-d", str(out_dir)]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _find_7zip() -> str | None:
    for name in ("7z", "7za", "7zr"):
        found = shutil.which(name)
        if found:
            return found
    candidates = [
        Path(r"C:\Program Files\7-Zip\7z.exe"),
        Path(r"C:\Program Files (x86)\7-Zip\7z.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _find_zip_tool() -> str | None:
    for name in ("zip",):
        found = shutil.which(name)
        if found:
            return found
    candidates = [Path(r"D:\app\texlive\2025\bin\windows\zip.exe")]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _find_unzip_tool() -> str | None:
    for name in ("unzip",):
        found = shutil.which(name)
        if found:
            return found
    candidates = [Path(r"D:\app\texlive\2025\bin\windows\unzip.exe")]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None
