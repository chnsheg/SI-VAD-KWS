"""Read-only eligibility checks for using V2 packed corpora in a V3 role."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


_ROLE_LABELS = {
    "base_positive": 0,
    "raw_positive": 0,
    "base_negative": 1,
    "raw_negative": 1,
    "false_wake_hard_negative": 1,
    "captured_environment_negative": 1,
    "tau_environment_negative": 1,
}
_KNOWN_REJECTED_V2_ROLES = {
    "base_positive": "V2 base positives lack the complete per-record V3 scene/start provenance required for reuse",
    "base_negative": "V2 base negatives include repeated pure-noise recipes and incomplete captured-scene coverage",
    "false_wake_hard_negative": "V2 hard negatives use sparse candidates and insufficient false-wake coverage",
}


@dataclass(frozen=True)
class ReuseDecision:
    role: str
    manifest_path: str
    eligible: bool
    reasons: tuple[str, ...]
    record_count: int
    label_counts: dict[str, int]
    source_manifest_sha256: str | None
    quality: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _read_json(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read packed manifest: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Packed manifest must be an object: {path}")
    return value


def _integer_field(manifest: Mapping[str, object], key: str) -> int:
    value = manifest.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Packed manifest has invalid {key}: {value!r}")
    return value


def _path_field(manifest: Mapping[str, object], key: str, manifest_path: Path) -> Path:
    value = manifest.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Packed manifest is missing {key}")
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_manifest_path(manifest: Mapping[str, object], manifest_file: Path) -> Path | None:
    value = manifest.get("source_manifest")
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if not path.is_absolute():
        path = manifest_file.parent / path
    return path.resolve()


def _with_reasons(decision: ReuseDecision, reasons: list[str]) -> ReuseDecision:
    combined = tuple(dict.fromkeys((*decision.reasons, *reasons)))
    return replace(decision, eligible=not combined, reasons=combined)


def _with_quality(
    decision: ReuseDecision,
    quality: Mapping[str, object],
    reasons: list[str],
) -> ReuseDecision:
    updated = _with_reasons(decision, reasons)
    return replace(updated, quality=dict(quality))


def audit_v2_packed_role(
    manifest_path: Path | str,
    *,
    role: str,
    required_label: int | None = None,
) -> ReuseDecision:
    """Audit only packed metadata/labels and return a fail-closed reuse decision.

    This function opens no V2 audio or shard for writing.  A malformed packed
    corpus is an input-integrity error; known semantic V2 defects are recorded
    as an ineligible decision so the caller can publish an audit and generate
    a replacement role instead.
    """
    if role not in _ROLE_LABELS:
        raise ValueError(f"Unsupported V3 reuse role: {role}")
    if required_label is not None and required_label not in (0, 1):
        raise ValueError("required_label must be 0, 1, or None")

    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = _read_json(manifest_file)
    if manifest.get("format") != "packed_pcm16_v1":
        raise ValueError(f"Unsupported packed format: {manifest.get('format')!r}")
    record_count = _integer_field(manifest, "record_count")
    if _integer_field(manifest, "sample_rate") != 16000:
        raise ValueError("Reusable V2 pack must be 16 kHz")
    if _integer_field(manifest, "record_samples") != 16000:
        raise ValueError("Reusable V2 pack must contain complete one-second records")

    labels_path = _path_field(manifest, "labels", manifest_file)
    try:
        labels = labels_path.read_bytes()
    except OSError as error:
        raise ValueError(f"Unable to read packed labels: {labels_path}") from error
    if len(labels) != record_count:
        raise ValueError(f"Packed labels length is {len(labels)}, expected {record_count}")
    invalid = sorted(set(labels).difference({0, 1}))
    if invalid:
        raise ValueError(f"Packed labels must be binary, found values {invalid}")
    label_counts = {"positive": int(labels.count(0)), "negative": int(labels.count(1))}

    jitter_path = _path_field(manifest, "jitter_ms", manifest_file)
    try:
        jitter_bytes = jitter_path.stat().st_size
    except OSError as error:
        raise ValueError(f"Unable to stat packed jitter metadata: {jitter_path}") from error
    if jitter_bytes != record_count * 2:
        raise ValueError(f"Packed jitter length is {jitter_bytes}, expected {record_count * 2}")

    source_manifest_sha256: str | None = None
    reasons: list[str] = []
    source_value = manifest.get("source_manifest")
    expected_source_hash = manifest.get("source_manifest_sha256")
    if not isinstance(source_value, str) or not source_value.strip():
        reasons.append("packed manifest has no source manifest provenance")
    elif not isinstance(expected_source_hash, str) or len(expected_source_hash) != 64:
        reasons.append("packed manifest has no valid source manifest SHA-256")
    else:
        source_path = _source_manifest_path(manifest, manifest_file)
        assert source_path is not None
        if not source_path.is_file():
            reasons.append(f"source manifest does not exist: {source_path}")
        else:
            source_manifest_sha256 = _sha256(source_path)
            if source_manifest_sha256 != expected_source_hash.lower():
                reasons.append("source manifest SHA-256 does not match packed manifest")

    target_label = _ROLE_LABELS[role] if required_label is None else required_label
    target_name = "positive" if target_label == 0 else "negative"
    if label_counts[target_name] == 0:
        reasons.append(f"packed corpus has no {target_name} records for role {role}")
    known_reason = _KNOWN_REJECTED_V2_ROLES.get(role)
    if known_reason is not None:
        reasons.append(known_reason)

    return ReuseDecision(
        role=role,
        manifest_path=str(manifest_file),
        eligible=not reasons,
        reasons=tuple(reasons),
        record_count=record_count,
        label_counts=label_counts,
        source_manifest_sha256=source_manifest_sha256,
    )


def audit_v2_role_provenance(manifest_path: Path | str, *, role: str) -> ReuseDecision:
    """Require train-only raw-anchor provenance before V3 may reference it.

    V2 packed record order has no embedded row provenance.  Therefore V3 uses
    only the small raw-anchor corpus for external references, whose source
    manifest has one train-only source row per packed record.  The much larger
    V2 base corpus is deliberately rejected by :func:`audit_v2_packed_role`.
    """
    decision = audit_v2_packed_role(manifest_path, role=role)
    if role not in {"raw_positive", "raw_negative"}:
        return decision

    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = _read_json(manifest_file)
    source_path = _source_manifest_path(manifest, manifest_file)
    if source_path is None or not source_path.is_file():
        return _with_reasons(decision, ["raw-anchor source manifest is unavailable"])

    rows: list[Mapping[str, object]] = []
    try:
        with source_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"row {line_number} is not an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return _with_reasons(decision, [f"raw-anchor source manifest is malformed: {error}"])

    reasons: list[str] = []
    if len(rows) != decision.record_count:
        reasons.append(
            f"raw-anchor source manifest has {len(rows)} rows, expected {decision.record_count}"
        )
    allowed_roles = {"raw_positive": "positive", "raw_negative": "negative"}
    for row in rows:
        split = str(row.get("source_split", row.get("split", ""))).strip().casefold()
        if split != "train":
            reasons.append("raw-anchor source manifest is not train-only")
            break
        source_role = row.get("source_role")
        command = row.get("command", row.get("label"))
        if source_role not in allowed_roles or command != allowed_roles.get(source_role):
            reasons.append("raw-anchor source role and command label do not match")
            break
    return _with_reasons(decision, reasons)


def _raw_positive_rms_quality(manifest_path: Path, *, minimum_rms_dbfs: float) -> tuple[dict[str, object], list[str]]:
    if not np.isfinite(minimum_rms_dbfs):
        raise ValueError("minimum_rms_dbfs must be finite")
    manifest = _read_json(manifest_path)
    record_count = _integer_field(manifest, "record_count")
    record_samples = _integer_field(manifest, "record_samples")
    labels_path = _path_field(manifest, "labels", manifest_path)
    labels = np.fromfile(labels_path, dtype=np.uint8)
    if labels.size != record_count:
        raise ValueError("Packed labels changed while auditing raw-positive quality")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("Packed raw-anchor manifest has no shards")

    positive_rms: list[float] = []
    positive_indexes: list[int] = []
    start = 0
    for shard in shards:
        if not isinstance(shard, dict):
            raise ValueError("Packed raw-anchor shard entry must be an object")
        records = shard.get("records")
        name = shard.get("path")
        if isinstance(records, bool) or not isinstance(records, int) or records < 1:
            raise ValueError("Packed raw-anchor shard has invalid record count")
        if not isinstance(name, str) or not name:
            raise ValueError("Packed raw-anchor shard has no path")
        shard_path = (manifest_path.parent / name).resolve()
        expected_size = records * record_samples * 2
        if shard_path.stat().st_size != expected_size:
            raise ValueError(f"Packed raw-anchor shard has unexpected size: {shard_path}")
        mapped = np.memmap(shard_path, mode="r", dtype="<i2", shape=(records, record_samples))
        try:
            for offset in range(0, records, 256):
                stop = min(records, offset + 256)
                local_labels = labels[start + offset : start + stop]
                positive_mask = local_labels == 0
                if not bool(positive_mask.any()):
                    continue
                samples = np.asarray(mapped[offset:stop], dtype=np.float32) / 32767.0
                rms = np.sqrt(np.mean(np.square(samples), axis=1))
                positive_rms.extend(float(value) for value in rms[positive_mask])
                positive_indexes.extend(
                    int(start + offset + index)
                    for index in np.flatnonzero(positive_mask)
                )
        finally:
            mapped._mmap.close()
        start += records
    if start != record_count:
        raise ValueError("Packed raw-anchor shard counts do not match manifest")
    if not positive_rms:
        raise ValueError("Packed raw-anchor corpus has no positive records")

    rms_dbfs = np.asarray(positive_rms, dtype=np.float64)
    levels_dbfs = 20.0 * np.log10(np.maximum(rms_dbfs, 1e-12))
    excluded = [
        int(index)
        for index, level in zip(positive_indexes, levels_dbfs)
        if level < minimum_rms_dbfs
    ]
    below = len(excluded)
    quality: dict[str, object] = {
        "positive_rms_count": int(levels_dbfs.size),
        "minimum_rms_dbfs": float(minimum_rms_dbfs),
        "below_minimum_rms_count": below,
        "excluded_record_indices": excluded,
        "observed_min_rms_dbfs": float(levels_dbfs.min()),
        "observed_p01_rms_dbfs": float(np.quantile(levels_dbfs, 0.01)),
        "observed_median_rms_dbfs": float(np.quantile(levels_dbfs, 0.50)),
    }
    reasons = [] if below == 0 else [f"raw-positive RMS gate failed for {below} records"]
    return quality, reasons


def audit_v2_raw_anchor_for_reuse(
    manifest_path: Path | str,
    *,
    role: str,
    minimum_rms_dbfs: float = -50.0,
) -> ReuseDecision:
    """Apply train provenance and full packed-RMS quality gates to raw anchors."""
    if role not in {"raw_positive", "raw_negative"}:
        raise ValueError("Raw-anchor reuse only supports raw_positive or raw_negative")
    decision = audit_v2_role_provenance(manifest_path, role=role)
    if role == "raw_negative":
        return decision
    quality, reasons = _raw_positive_rms_quality(
        Path(manifest_path).expanduser().resolve(),
        minimum_rms_dbfs=minimum_rms_dbfs,
    )
    return _with_quality(decision, quality, reasons)


def write_filtered_v2_role_reference(destination: Path | str, decision: ReuseDecision) -> Path:
    """Publish a V3-only selector that excludes bad V2 raw-positive records.

    The selector contains record indexes, not audio.  Training continues to
    mmap the original V2 packed shards in read-only mode.
    """
    if decision.role != "raw_positive":
        raise ValueError("Only raw_positive may publish a filtered V2 role reference")
    excluded_raw = decision.quality.get("excluded_record_indices", [])
    if not isinstance(excluded_raw, list) or any(
        isinstance(index, bool) or not isinstance(index, int) for index in excluded_raw
    ):
        raise ValueError("raw-positive quality audit has invalid excluded record indexes")
    excluded = sorted(set(int(index) for index in excluded_raw))
    positive_count = int(decision.label_counts.get("positive", 0))
    if any(index < 0 or index >= decision.record_count for index in excluded):
        raise ValueError("raw-positive excluded record index is outside the packed corpus")
    remaining_reasons = [
        reason for reason in decision.reasons if not reason.startswith("raw-positive RMS gate failed")
    ]
    if remaining_reasons:
        raise ValueError("Cannot reference a raw-positive pack that failed provenance or packed-format gates")
    retained_count = positive_count - len(excluded)
    if retained_count < 1:
        raise ValueError("Filtered raw-positive reference would be empty")

    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                {
                    "schema_version": 1,
                    "role": "raw_positive",
                    "storage": "external_read_only",
                    "external_manifest": decision.manifest_path,
                    "target_label": 0,
                    "record_count": retained_count,
                    "excluded_record_indices": excluded,
                    "quality": decision.quality,
                },
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return output


def write_reuse_audit(destination: Path | str, decisions: Sequence[ReuseDecision]) -> Path:
    """Atomically publish V2 reuse decisions outside all read-only V2 inputs."""
    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                {
                    "schema_version": 1,
                    "decisions": [decision.as_dict() for decision in decisions],
                },
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return output
