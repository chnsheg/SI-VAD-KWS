from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class ManifestRecord:
    id: str
    audio_path: str
    label_path: str
    split: str
    sample_rate: int
    duration_sec: float
    frame_hop_ms: float
    source: str
    feature_path: str | None = None
    video_id: str | None = None
    chunk_start_sec: float | None = None
    chunk_end_sec: float | None = None
    label_source: str | None = None
    class_label_path: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "ManifestRecord":
        return cls(
            id=str(data["id"]),
            audio_path=str(data["audio_path"]),
            label_path=str(data["label_path"]),
            split=str(data["split"]),
            sample_rate=int(data["sample_rate"]),
            duration_sec=float(data["duration_sec"]),
            frame_hop_ms=float(data["frame_hop_ms"]),
            source=str(data["source"]),
            feature_path=str(data["feature_path"]) if data.get("feature_path") is not None else None,
            video_id=str(data["video_id"]) if data.get("video_id") is not None else None,
            chunk_start_sec=float(data["chunk_start_sec"]) if data.get("chunk_start_sec") is not None else None,
            chunk_end_sec=float(data["chunk_end_sec"]) if data.get("chunk_end_sec") is not None else None,
            label_source=str(data["label_source"]) if data.get("label_source") is not None else None,
            class_label_path=str(data["class_label_path"]) if data.get("class_label_path") is not None else None,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def resolve_audio(self, base_dir: str | Path) -> Path:
        return _resolve(self.audio_path, base_dir)

    def resolve_label(self, base_dir: str | Path) -> Path:
        return _resolve(self.label_path, base_dir)

    def resolve_feature(self, base_dir: str | Path) -> Path | None:
        if self.feature_path is None:
            return None
        return _resolve(self.feature_path, base_dir)

    def resolve_class_label(self, base_dir: str | Path) -> Path | None:
        if self.class_label_path is None:
            return None
        return _resolve(self.class_label_path, base_dir)


def _resolve(path: str, base_dir: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(base_dir) / candidate


def read_manifest(path: str | Path, split: str | None = None) -> list[ManifestRecord]:
    path = Path(path)
    records: list[ManifestRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = ManifestRecord.from_dict(json.loads(line))
            except Exception as exc:
                raise ValueError(f"Invalid manifest line {line_no} in {path}: {exc}") from exc
            if split is None or record.split == split:
                records.append(record)
    return records


def write_manifest(records: Iterable[ManifestRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=True, sort_keys=True) + "\n")


def validate_manifest(path: str | Path, check_files: bool = True) -> list[ManifestRecord]:
    path = Path(path)
    base_dir = path.parent
    records = read_manifest(path)
    if not records:
        raise ValueError(f"Manifest is empty: {path}")
    seen: set[str] = set()
    for record in records:
        if record.id in seen:
            raise ValueError(f"Duplicate manifest id: {record.id}")
        seen.add(record.id)
        if record.sample_rate <= 0:
            raise ValueError(f"{record.id}: invalid sample_rate")
        if record.duration_sec <= 0:
            raise ValueError(f"{record.id}: invalid duration_sec")
        if record.frame_hop_ms <= 0:
            raise ValueError(f"{record.id}: invalid frame_hop_ms")
        if check_files:
            audio_path = record.resolve_audio(base_dir)
            label_path = record.resolve_label(base_dir)
            if not audio_path.exists():
                raise FileNotFoundError(f"{record.id}: missing audio {audio_path}")
            if not label_path.exists():
                raise FileNotFoundError(f"{record.id}: missing label {label_path}")
            labels = np.load(label_path)
            if labels.ndim != 1:
                raise ValueError(f"{record.id}: labels must be 1D")
            class_label_path = record.resolve_class_label(base_dir)
            if class_label_path is not None:
                if not class_label_path.exists():
                    raise FileNotFoundError(f"{record.id}: missing class label {class_label_path}")
                class_labels = np.load(class_label_path)
                if class_labels.ndim != 1:
                    raise ValueError(f"{record.id}: class labels must be 1D")
                if len(class_labels) != len(labels):
                    raise ValueError(f"{record.id}: class labels and binary labels must have the same length")
            feature_path = record.resolve_feature(base_dir)
            if feature_path is not None:
                if not feature_path.exists():
                    raise FileNotFoundError(f"{record.id}: missing feature {feature_path}")
                features = np.load(feature_path)
                if features.ndim != 2:
                    raise ValueError(f"{record.id}: features must be 2D")
    return records
