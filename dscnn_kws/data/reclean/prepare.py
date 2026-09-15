from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import soundfile as sf

from .audio import canonicalize_wav, load_mono_float32, save_pcm16_atomic, sha256_file
from .segment import detect_positive_boundary, split_false_wake_segments


def _active_runs(mask) -> list[list[int]]:
    indices = mask.nonzero(as_tuple=False).flatten().tolist()
    if not indices:
        return []
    runs: list[list[int]] = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            runs.append([start, previous + 1])
            start = index
        previous = index
    runs.append([start, previous + 1])
    return runs


def _prepared_destination(output_root: Path, role: str, source_sha256: str, suffix: str = "") -> Path:
    return output_root / role / f"{source_sha256[:16]}{suffix}.wav"


def _base_row(entry: dict[str, Any], prepared_path: Path, metadata) -> dict[str, Any]:
    return {
        "role": str(entry["role"]),
        "source_label": entry.get("source_label"),
        "source_split": entry.get("split"),
        "scene": entry.get("scene"),
        "source_path": str(Path(entry["path"]).expanduser().resolve()),
        "source_sha256": metadata.source_sha256,
        "prepared_path": str(prepared_path.resolve()),
        "prepared_sha256": metadata.output_sha256,
        "sample_rate": metadata.output_sample_rate,
        "frames": metadata.output_frames,
    }


def prepare_sources(inventory_path: Path, output_root: Path) -> list[dict[str, Any]]:
    """Canonicalize catalogued sources and persist clean boundaries for recipe planning."""
    inventory_path = Path(inventory_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    raw_inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    files = raw_inventory.get("files")
    if not isinstance(files, list):
        raise ValueError(f"Inventory has no files list: {inventory_path}")

    rows: list[dict[str, Any]] = []
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("Inventory files entries must be objects")
        role = str(entry.get("role", ""))
        source_path = Path(str(entry["path"])).expanduser().resolve()
        if role == "rir" and entry.get("already_normalized"):
            normalized_sha256 = str(entry.get("normalized_sha256", "")).lower()
            if len(normalized_sha256) != 64 or any(character not in "0123456789abcdef" for character in normalized_sha256):
                raise ValueError(f"Catalogued normalized RIR has an invalid SHA-256: {source_path}")
            try:
                info = sf.info(source_path)
            except RuntimeError as error:
                raise ValueError(f"Unreadable catalogued normalized RIR: {source_path}: {error}") from error
            if info.frames <= 0 or (info.samplerate, info.channels, info.subtype) != (16000, 1, "PCM_16"):
                raise ValueError(f"Catalogued normalized RIR is not 16 kHz mono PCM_16: {source_path}")
            rows.append(
                {
                    "role": "rir",
                    "source_label": entry.get("source_label"),
                    "source_split": entry.get("split"),
                    "scene": entry.get("scene"),
                    "source_path": str(source_path),
                    "source_sha256": normalized_sha256,
                    "prepared_path": str(source_path),
                    "prepared_sha256": normalized_sha256,
                    "sample_rate": 16000,
                    "frames": int(info.frames),
                    "source_kind": "rir",
                }
            )
            continue
        source_hash = sha256_file(source_path)
        if role == "false_wake":
            source_destination = _prepared_destination(output_root, "false_wake_sources", source_hash)
            metadata = canonicalize_wav(source_path, source_destination, sample_rate=16000)
            waveform, sample_rate, _ = load_mono_float32(source_destination)
            if sample_rate != 16000:
                raise RuntimeError("Canonical false-wake source has an unexpected sample rate")
            for index, span in enumerate(split_false_wake_segments(waveform, sample_rate=16000)):
                segment_destination = _prepared_destination(output_root, "false_wake_segments", source_hash, f"-{index:04d}")
                save_pcm16_atomic(segment_destination, waveform[:, span.start : span.end], sample_rate=16000)
                rows.append(
                    {
                        "role": "false_wake",
                        "source_kind": "false_wake",
                        "source_label": "negative",
                        "source_split": entry.get("split"),
                        "parent_source_path": str(source_path),
                        "parent_source_sha256": metadata.source_sha256,
                        "source_offset_start": span.start,
                        "source_offset_end": span.end,
                        "prepared_path": str(segment_destination.resolve()),
                        "prepared_sha256": sha256_file(segment_destination),
                        "sample_rate": 16000,
                        "frames": span.end - span.start,
                    }
                )
            continue

        destination = _prepared_destination(output_root, role, source_hash)
        metadata = canonicalize_wav(source_path, destination, sample_rate=16000)
        row = _base_row(entry, destination, metadata)
        if role == "mobvoi_speech":
            row["source_kind"] = "speech"
            if row["source_label"] == "positive":
                waveform, sample_rate, _ = load_mono_float32(destination)
                boundary = detect_positive_boundary(waveform, sample_rate)
                row.update(
                    {
                        "active_start": boundary.start,
                        "active_end": boundary.end,
                        "active_center": boundary.center,
                        "active_runs": _active_runs(boundary.active_mask),
                        "boundary_audit_status": boundary.audit_status,
                    }
                )
        elif role == "noise":
            row["source_kind"] = "noise"
        elif role == "rir":
            row["source_kind"] = "rir"
        rows.append(row)

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "prepared_sources.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output_root / "prepare_report.json").write_text(
        json.dumps({"schema_version": 1, "inventory_path": str(inventory_path), "prepared_rows": len(rows)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return rows
