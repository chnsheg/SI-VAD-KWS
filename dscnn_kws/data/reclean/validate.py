from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import soundfile as sf

from .audio import sha256_file


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    errors: tuple[str, ...]
    counts: dict[str, int]


def _read_metadata_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL record {path}:{line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"Metadata record is not an object: {path}:{line_number}")
        rows.append(row)
    return rows


def _source_kind(row: Mapping[str, object]) -> str:
    recipe = row.get("recipe")
    if isinstance(recipe, Mapping):
        return str(recipe.get("source_kind", "speech"))
    return "speech"


def validate_manifest_quotas(
    metadata_path: Path,
    expected_total: int,
    expected_quotas: Mapping[str, int] | None = None,
) -> ValidationReport:
    errors: list[str] = []
    counts: Counter[str] = Counter()
    ids: set[str] = set()
    for row in _read_metadata_rows(Path(metadata_path)):
        example_id = str(row.get("example_id", ""))
        if not example_id:
            errors.append("Metadata row has no example_id")
            continue
        if example_id in ids:
            errors.append(f"Duplicate example_id: {example_id}")
        ids.add(example_id)
        label = str(row.get("label", ""))
        kind = _source_kind(row)
        if label == "positive":
            counts["positive"] += 1
        elif label == "negative" and kind == "false_wake":
            counts["false_wake_negative"] += 1
        elif label == "negative" and kind == "pure_noise":
            counts["pure_noise_negative"] += 1
        elif label == "negative":
            counts["speech_negative"] += 1
        else:
            errors.append(f"Invalid label for {example_id}: {label}")

    total = sum(counts.values())
    if total != expected_total:
        errors.append(f"Manifest total is {total}, expected {expected_total}")
    negative_total = counts["speech_negative"] + counts["false_wake_negative"] + counts["pure_noise_negative"]
    if counts["positive"] != negative_total:
        errors.append(f"Positive/negative imbalance: {counts['positive']} != {negative_total}")
    if expected_quotas is not None:
        for key, expected in expected_quotas.items():
            if counts[key] != expected:
                errors.append(f"Quota {key} is {counts[key]}, expected {expected}")
    return ValidationReport(ok=not errors, errors=tuple(errors), counts=dict(counts))


def validate_dataset(
    root: Path,
    expected_sample_rate: int = 16000,
    expected_frames: int = 16000,
    verify_hashes: bool = True,
) -> ValidationReport:
    root = Path(root)
    errors: list[str] = []
    counts: Counter[str] = Counter()
    audio_files = sorted((root / "audio").rglob("*.wav")) if (root / "audio").is_dir() else []
    if not audio_files:
        errors.append("No generated WAV files found under audio/")
    for path in audio_files:
        try:
            info = sf.info(path)
        except RuntimeError as error:
            errors.append(f"Unreadable WAV {path}: {error}")
            continue
        if (info.samplerate, info.channels, info.subtype, info.frames) != (expected_sample_rate, 1, "PCM_16", expected_frames):
            errors.append(
                f"Wrong format {path}: {(info.samplerate, info.channels, info.subtype, info.frames)}"
            )
        counts["audio_files"] += 1

    metadata_files = sorted((root / "metadata").glob("shard-*.jsonl")) if (root / "metadata").is_dir() else []
    metadata_rows: list[dict[str, object]] = []
    for metadata_file in metadata_files:
        metadata_rows.extend(_read_metadata_rows(metadata_file))
    if not metadata_rows and audio_files:
        errors.append("Generated audio has no completion metadata")
    ids: set[str] = set()
    for row in metadata_rows:
        example_id = str(row.get("example_id", ""))
        output_path = Path(str(row.get("output_path", "")))
        if example_id in ids:
            errors.append(f"Duplicate example_id: {example_id}")
        ids.add(example_id)
        if not output_path.is_file():
            errors.append(f"Metadata output is missing: {output_path}")
            continue
        if verify_hashes and row.get("output_sha256") != sha256_file(output_path):
            errors.append(f"Output hash mismatch: {output_path}")
    counts["metadata_rows"] = len(metadata_rows)
    if metadata_rows and len(metadata_rows) != len(audio_files):
        errors.append(f"Audio/metadata count mismatch: {len(audio_files)} != {len(metadata_rows)}")
    return ValidationReport(ok=not errors, errors=tuple(errors), counts=dict(counts))


def _write_training_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_dataset_manifests(output_root: Path, prepared_rows: list[dict[str, object]]) -> dict[str, int]:
    """Create the manifests expected by the existing trainer without moving WAV files."""
    output_root = Path(output_root).expanduser().resolve()
    metadata_rows: list[dict[str, object]] = []
    for metadata_path in sorted((output_root / "metadata").glob("shard-*.jsonl")):
        metadata_rows.extend(_read_metadata_rows(metadata_path))
    train_rows = [
        {"audio_filepath": str(row["output_path"]), "command": str(row["label"])}
        for row in metadata_rows
    ]
    eval_rows: dict[str, list[dict[str, object]]] = {"validation": [], "test": []}
    for row in prepared_rows:
        split = str(row.get("source_split", ""))
        if row.get("source_kind") != "speech" or split not in eval_rows:
            continue
        label = str(row.get("source_label", ""))
        if label not in {"positive", "negative"}:
            raise ValueError(f"Invalid evaluation source label: {label}")
        eval_rows[split].append({"audio_filepath": str(row["prepared_path"]), "command": label})
    if not train_rows or not eval_rows["validation"] or not eval_rows["test"]:
        raise ValueError("Cannot build training manifests without generated train and prepared validation/test speech")
    _write_training_manifest(output_root / "train_manifest.json", train_rows)
    _write_training_manifest(output_root / "validation_manifest.json", eval_rows["validation"])
    _write_training_manifest(output_root / "test_manifest.json", eval_rows["test"])
    return {"train": len(train_rows), "validation": len(eval_rows["validation"]), "test": len(eval_rows["test"])}
