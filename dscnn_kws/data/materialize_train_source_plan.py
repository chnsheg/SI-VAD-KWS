"""Materialize train-only source-plan blocks into canonical continuous WAVs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as audio_functional


NORMALIZED_SOURCE_PLAN_FORMAT = "normalized_train_source_plan_v1"


@dataclass(frozen=True)
class SourcePlanMaterializationResult:
    output_root: Path
    normalized_source_plan: Path
    audit_path: Path
    block_count: int


class SourcePlanValidationError(ValueError):
    """Raised after a fail-closed source-plan validation audit is published."""


@dataclass(frozen=True)
class PreparedProvenanceIndex:
    manifest_path: Path
    manifest_sha256: str
    rows_by_path: dict[Path, dict[str, Any]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_int(value: object, *, minimum: int = 0) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return None
    return int(value)


def _strict_sha256(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        return None
    return normalized


def _resolve_path(value: object, *, relative_to: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _load_prepared_provenance_manifest(path: Path) -> PreparedProvenanceIndex:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows_by_path: dict[Path, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"Prepared provenance manifest is not valid UTF-8: {path}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on prepared provenance manifest line {line_number}: {path}") from error
        if not isinstance(row, dict):
            raise ValueError(f"Prepared provenance manifest line {line_number} must be an object")
        prepared_path = _resolve_path(row.get("prepared_path"), relative_to=path.parent)
        prepared_sha256 = _strict_sha256(row.get("prepared_sha256"))
        source_sha256 = _strict_sha256(row.get("source_sha256"))
        parent_source_sha256 = _strict_sha256(row.get("parent_source_sha256"))
        sample_rate = _strict_int(row.get("sample_rate"), minimum=1)
        frames = _strict_int(row.get("frames"), minimum=1)
        if prepared_path is None:
            raise ValueError(f"Prepared provenance manifest line {line_number} has no valid prepared_path")
        if prepared_sha256 is None:
            raise ValueError(f"Prepared provenance manifest line {line_number} has no valid prepared_sha256")
        if row.get("source_sha256") is not None and source_sha256 is None:
            raise ValueError(f"Prepared provenance manifest line {line_number} has invalid source_sha256")
        if row.get("parent_source_sha256") is not None and parent_source_sha256 is None:
            raise ValueError(f"Prepared provenance manifest line {line_number} has invalid parent_source_sha256")
        if source_sha256 is None and parent_source_sha256 is None:
            raise ValueError(
                f"Prepared provenance manifest line {line_number} has no source_sha256 or parent_source_sha256"
            )
        if (
            source_sha256 is not None
            and parent_source_sha256 is not None
            and source_sha256 != parent_source_sha256
        ):
            raise ValueError(
                f"Prepared provenance manifest line {line_number} has conflicting source identities"
            )
        if sample_rate is None or frames is None:
            raise ValueError(f"Prepared provenance manifest line {line_number} has invalid audio metadata")
        if prepared_path in rows_by_path:
            previous_line = rows_by_path[prepared_path]["line_number"]
            raise ValueError(
                f"Duplicate resolved prepared_path in provenance manifest lines {previous_line} and "
                f"{line_number}: {prepared_path}"
            )
        source_identity_key = "source_sha256" if source_sha256 is not None else "parent_source_sha256"
        source_identity_sha256 = source_sha256 or parent_source_sha256
        rows_by_path[prepared_path] = {
            "line_number": line_number,
            "prepared_path": str(prepared_path),
            "prepared_sha256": prepared_sha256,
            "source_identity_key": source_identity_key,
            "source_identity_sha256": source_identity_sha256,
            "sample_rate": sample_rate,
            "frames": frames,
        }
    return PreparedProvenanceIndex(
        manifest_path=path,
        manifest_sha256=_sha256_file(path),
        rows_by_path=rows_by_path,
    )


def _publish_no_replace(temporary_path: Path, output_path: Path) -> None:
    """Atomically publish a same-filesystem temporary file without overwrite."""

    if output_path.exists():
        raise FileExistsError(output_path)
    try:
        os.link(temporary_path, output_path)
    except FileExistsError:
        raise
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _atomic_write_bytes_no_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_no_replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _atomic_write_json_no_replace(path: Path, payload: object) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write_bytes_no_replace(path, encoded)


def _write_pcm16_wav_no_replace(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp.wav", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        payload = waveform.detach().cpu().to(torch.float32).clamp(-1.0, 1.0).squeeze(0).numpy()
        sf.write(temporary_path, payload, sample_rate, subtype="PCM_16", format="WAV")
        # Windows requires a writable descriptor for fsync.
        with temporary_path.open("r+b") as handle:
            os.fsync(handle.fileno())
        info = sf.info(temporary_path)
        expected = (sample_rate, 1, "PCM_16", int(waveform.shape[1]))
        actual = (int(info.samplerate), int(info.channels), info.subtype, int(info.frames))
        if actual != expected:
            raise RuntimeError(f"Canonical WAV verification failed for {path}: {actual} != {expected}")
        _publish_no_replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _block_value(plan: dict[str, Any], block: dict[str, Any], key: str) -> object:
    return block[key] if key in block else plan.get(key)


def _source_id(
    plan: dict[str, Any],
    block: dict[str, Any],
    *,
    verified_source_sha256: str,
) -> str:
    value = _block_value(plan, block, "source_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    declared_hash = _block_value(plan, block, "source_sha256")
    if isinstance(declared_hash, str) and declared_hash.strip():
        return declared_hash.strip()
    return verified_source_sha256


def _audit_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(record["status"]) for record in records)
    return {
        "blocks_seen": len(records),
        "train_blocks_seen": sum(record.get("input_split") == "train" for record in records),
        "emitted_blocks": statuses["emitted"],
        "skipped_non_train_blocks": statuses["skipped_non_train"],
        "rejected_blocks": sum(count for status, count in statuses.items() if status.startswith("rejected_")),
        "statuses": dict(sorted(statuses.items())),
    }


def _load_block_channel_zero(
    source_path: Path,
    *,
    start_sample: int,
    end_sample: int,
    source_sample_rate: int,
    target_sample_rate: int,
) -> torch.Tensor:
    with sf.SoundFile(source_path, "r") as handle:
        handle.seek(start_sample)
        samples = handle.read(end_sample - start_sample, dtype="float32", always_2d=True)
    if samples.shape[0] != end_sample - start_sample or samples.shape[1] < 1:
        raise RuntimeError(f"Short read from {source_path}: {samples.shape}")
    waveform = torch.from_numpy(np.asarray(samples).T.copy()).to(torch.float32)
    if source_sample_rate != target_sample_rate:
        # The resampler treats the leading dimensions independently. Select
        # channel 0 only after resampling to match the deployment contract.
        waveform = audio_functional.resample(waveform, source_sample_rate, target_sample_rate)
    waveform = waveform.narrow(0, 0, 1)
    if not bool(torch.isfinite(waveform).all()):
        raise ValueError(f"Non-finite audio samples in {source_path}")
    return waveform.clamp(-1.0, 1.0).contiguous()


def materialize_train_source_plan(
    source_plan: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    *,
    prepared_provenance_manifest: str | os.PathLike[str] | None = None,
    sample_rate: int = 16_000,
    minimum_frames: int = 17_536,
) -> SourcePlanMaterializationResult:
    """Publish canonical WAVs and a train-only normalized source plan.

    The output root is an immutable run directory: it must not exist before
    the call. Non-train blocks are audited and skipped. Any invalid train
    block fails the complete preflight, publishes only a failure audit, and
    never publishes a normalized source plan.
    """

    for name, value in (("sample_rate", sample_rate), ("minimum_frames", minimum_frames)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    source_plan_path = Path(source_plan).expanduser().resolve()
    output_root_path = Path(output_root).expanduser().resolve()
    normalized_path = output_root_path / "normalized_source_plan.json"
    audit_path = output_root_path / "materialization_audit.json"
    if not source_plan_path.is_file():
        raise FileNotFoundError(source_plan_path)
    if output_root_path.exists():
        raise FileExistsError(f"Output root already exists; refusing overwrite: {output_root_path}")

    provenance_index: PreparedProvenanceIndex | None = None
    if prepared_provenance_manifest is not None:
        provenance_path = Path(prepared_provenance_manifest).expanduser().resolve()
        provenance_index = _load_prepared_provenance_manifest(provenance_path)

    try:
        payload = json.loads(source_plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid source plan JSON: {source_plan_path}") from error
    plans = payload.get("plans") if isinstance(payload, dict) else None
    if not isinstance(plans, list):
        raise ValueError("source_plan.json must contain a plans list")

    source_hash_cache: dict[Path, str] = {}
    candidates: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    plans_seen = 0
    for plan_index, plan in enumerate(plans):
        if not isinstance(plan, dict) or not isinstance(plan.get("blocks"), list):
            raise ValueError(f"Plan {plan_index} must be an object containing a blocks list")
        plans_seen += 1
        for input_block_index, block in enumerate(plan["blocks"]):
            record: dict[str, Any] = {
                "plan_index": plan_index,
                "input_block_index": input_block_index,
                "input_split": block.get("split") if isinstance(block, dict) else None,
            }
            audit_records.append(record)
            if not isinstance(block, dict):
                record["status"] = "rejected_invalid_block"
                continue
            if block.get("split") != "train":
                record["status"] = "skipped_non_train"
                continue

            source_path = _resolve_path(block.get("path"), relative_to=source_plan_path.parent)
            record["original_path"] = block.get("path")
            record["resolved_original_path"] = str(source_path) if source_path is not None else None
            if source_path is None or not source_path.is_file():
                record["status"] = "rejected_missing_source"
                continue
            try:
                info = sf.info(source_path)
            except (OSError, RuntimeError, ValueError):
                record["status"] = "rejected_unreadable_source"
                continue
            source_rate = int(info.samplerate)
            source_frames = int(info.frames)
            source_channels = int(info.channels)
            record.update(
                {
                    "original_sample_rate": source_rate,
                    "original_frames": source_frames,
                    "original_channels": source_channels,
                }
            )
            if source_rate < 1 or source_frames < 1 or source_channels < 1:
                record["status"] = "rejected_invalid_audio_metadata"
                continue
            declared_rate = _block_value(plan, block, "sample_rate")
            if declared_rate is not None and _strict_int(declared_rate, minimum=1) != source_rate:
                record["status"] = "rejected_declared_sample_rate_mismatch"
                continue

            start = _strict_int(block.get("allowed_start"))
            end = _strict_int(block.get("allowed_end"), minimum=1)
            record["original_allowed_start"] = block.get("allowed_start")
            record["original_allowed_end"] = block.get("allowed_end")
            if start is None or end is None or not 0 <= start < end <= source_frames:
                record["status"] = "rejected_invalid_bounds"
                continue
            expected_frames = math.ceil((end - start) * sample_rate / source_rate)
            record["expected_materialized_frames"] = expected_frames
            if expected_frames < minimum_frames:
                record["status"] = "rejected_too_short"
                continue

            verified_hash = source_hash_cache.get(source_path)
            if verified_hash is None:
                verified_hash = _sha256_file(source_path)
                source_hash_cache[source_path] = verified_hash
            declared_hash = _block_value(plan, block, "source_sha256")
            record["declared_original_source_sha256"] = declared_hash
            record["verified_original_file_sha256"] = verified_hash
            verification_method = "direct_file_sha256"
            record["source_verification_method"] = verification_method
            verified_source_identity_hash = verified_hash
            provenance_row: dict[str, Any] | None = None
            if declared_hash is not None:
                normalized_declared_hash = _strict_sha256(declared_hash)
                if normalized_declared_hash == verified_hash:
                    verified_source_identity_hash = verified_hash
                elif provenance_index is None:
                    # Preserve the legacy failure category when no provenance
                    # source was supplied for a prepared-file mismatch.
                    record["status"] = "rejected_source_hash_mismatch"
                    continue
                else:
                    verification_method = "prepared_provenance_manifest"
                    record["source_verification_method"] = verification_method
                    provenance_row = provenance_index.rows_by_path.get(source_path)
                    if provenance_row is None:
                        record["status"] = "rejected_missing_prepared_provenance"
                        continue
                    record.update(
                        {
                            "prepared_provenance_manifest": str(provenance_index.manifest_path),
                            "prepared_provenance_manifest_sha256": provenance_index.manifest_sha256,
                            "prepared_provenance_line_number": provenance_row["line_number"],
                            "prepared_provenance_source_identity_key": provenance_row["source_identity_key"],
                            "prepared_provenance_source_identity_sha256": provenance_row[
                                "source_identity_sha256"
                            ],
                            "prepared_provenance_prepared_sha256": provenance_row["prepared_sha256"],
                            "prepared_provenance_sample_rate": provenance_row["sample_rate"],
                            "prepared_provenance_frames": provenance_row["frames"],
                        }
                    )
                    if (
                        normalized_declared_hash is None
                        or provenance_row["source_identity_sha256"] != normalized_declared_hash
                    ):
                        record["status"] = "rejected_prepared_provenance_source_identity_mismatch"
                        continue
                    if provenance_row["prepared_sha256"] != verified_hash:
                        record["status"] = "rejected_prepared_provenance_content_hash_mismatch"
                        continue
                    if provenance_row["sample_rate"] != source_rate:
                        record["status"] = "rejected_prepared_provenance_sample_rate_mismatch"
                        continue
                    if provenance_row["frames"] != source_frames:
                        record["status"] = "rejected_prepared_provenance_frame_count_mismatch"
                        continue
                    verified_source_identity_hash = normalized_declared_hash

            record["source_verification_method"] = verification_method
            record["verified_original_source_sha256"] = verified_source_identity_hash
            record["verified_original_prepared_sha256"] = verified_hash

            source_id = _source_id(plan, block, verified_source_sha256=verified_hash)
            record["source_id"] = source_id
            record["status"] = "eligible"
            candidates.append(
                {
                    "plan": plan,
                    "block": block,
                    "record": record,
                    "source_path": source_path,
                    "source_rate": source_rate,
                    "source_channels": source_channels,
                    "verified_hash": verified_hash,
                    "declared_hash": declared_hash,
                    "verification_method": verification_method,
                    "verified_source_identity_hash": verified_source_identity_hash,
                    "provenance_row": provenance_row,
                    "source_id": source_id,
                    "start": start,
                    "end": end,
                    "expected_frames": expected_frames,
                }
            )

    rejected = [record for record in audit_records if str(record.get("status", "")).startswith("rejected_")]
    audit: dict[str, Any] = {
        "schema_version": 1,
        "status": "failed_validation" if rejected or not candidates else "materializing",
        "input": {
            "source_plan": str(source_plan_path),
            "source_plan_sha256": _sha256_file(source_plan_path),
            "prepared_provenance_manifest": (
                str(provenance_index.manifest_path) if provenance_index is not None else None
            ),
            "prepared_provenance_manifest_sha256": (
                provenance_index.manifest_sha256 if provenance_index is not None else None
            ),
        },
        "config": {
            "sample_rate": sample_rate,
            "minimum_frames": minimum_frames,
            "channel_policy": "resample_each_channel_then_select_channel_0",
            "overwrite_policy": "forbidden",
        },
        "plans_seen": plans_seen,
        "counts": _audit_counts(audit_records),
        "records": audit_records,
    }
    if not candidates:
        audit["errors"] = ["no valid train blocks"]
    elif rejected:
        audit["errors"] = [f"{len(rejected)} train blocks failed validation"]
    if rejected or not candidates:
        output_root_path.mkdir(parents=True, exist_ok=False)
        _atomic_write_json_no_replace(audit_path, audit)
        raise SourcePlanValidationError("; ".join(audit["errors"]))

    output_root_path.mkdir(parents=True, exist_ok=False)
    audio_root = output_root_path / "audio"
    normalized_by_plan: dict[int, list[dict[str, Any]]] = {}
    try:
        for candidate in candidates:
            plan = candidate["plan"]
            block = candidate["block"]
            record = candidate["record"]
            digest = hashlib.sha256(
                (
                    f"{record['plan_index']}\0{record['input_block_index']}\0{candidate['source_id']}\0"
                    f"{candidate['start']}\0{candidate['end']}\0{candidate['verified_hash']}"
                ).encode("utf-8")
            ).hexdigest()[:20]
            output_path = audio_root / f"block-{record['plan_index']:05d}-{record['input_block_index']:06d}-{digest}.wav"
            waveform = _load_block_channel_zero(
                candidate["source_path"],
                start_sample=candidate["start"],
                end_sample=candidate["end"],
                source_sample_rate=candidate["source_rate"],
                target_sample_rate=sample_rate,
            )
            actual_frames = int(waveform.shape[1])
            if actual_frames < minimum_frames:
                record["status"] = "rejected_too_short_after_resample"
                raise RuntimeError(f"Materialized block is too short: {actual_frames} < {minimum_frames}")
            if actual_frames != candidate["expected_frames"]:
                record["status"] = "rejected_resampled_frame_count_mismatch"
                raise RuntimeError(
                    f"Unexpected resampled frame count: {actual_frames} != {candidate['expected_frames']}"
                )
            _write_pcm16_wav_no_replace(output_path, waveform, sample_rate)
            output_hash = _sha256_file(output_path)
            record.update(
                {
                    "status": "emitted",
                    "materialized_path": str(output_path),
                    "materialized_frames": actual_frames,
                    "materialized_sha256": output_hash,
                }
            )
            normalized_block: dict[str, Any] = {
                "split": "train",
                "path": str(output_path),
                "allowed_start": 0,
                "allowed_end": actual_frames,
                "sample_rate": sample_rate,
                "domain": _block_value(plan, block, "domain"),
                "scene": _block_value(plan, block, "scene"),
                "source_id": candidate["source_id"],
                "source_sha256": output_hash,
                "materialized_sha256": output_hash,
                "original_path": block.get("path"),
                "resolved_original_path": str(candidate["source_path"]),
                "original_source_sha256": (
                    candidate["declared_hash"] or candidate["verified_source_identity_hash"]
                ),
                "verified_original_source_sha256": candidate["verified_source_identity_hash"],
                "verified_original_prepared_sha256": candidate["verified_hash"],
                "original_source_verification_method": candidate["verification_method"],
                "original_allowed_start": candidate["start"],
                "original_allowed_end": candidate["end"],
                "original_sample_rate": candidate["source_rate"],
                "original_channels": candidate["source_channels"],
                "input_plan_index": record["plan_index"],
                "input_block_index": record["input_block_index"],
            }
            if candidate["provenance_row"] is not None:
                normalized_block.update(
                    {
                        "prepared_provenance_manifest": str(provenance_index.manifest_path),
                        "prepared_provenance_manifest_sha256": provenance_index.manifest_sha256,
                        "prepared_provenance_line_number": candidate["provenance_row"]["line_number"],
                    }
                )
            for key in ("role", "block_index", "block_start", "block_end"):
                value = _block_value(plan, block, key)
                if value is not None:
                    normalized_block[f"original_{key}" if key.startswith("block_") else key] = value
            normalized_by_plan.setdefault(int(record["plan_index"]), []).append(normalized_block)
    except Exception as error:
        audit["status"] = "failed_materialization"
        audit["counts"] = _audit_counts(audit_records)
        audit["errors"] = [f"{type(error).__name__}: {error}"]
        _atomic_write_json_no_replace(audit_path, audit)
        raise

    normalized_plans = []
    for plan_index in sorted(normalized_by_plan):
        original_plan = plans[plan_index]
        plan_row: dict[str, Any] = {"input_plan_index": plan_index, "blocks": normalized_by_plan[plan_index]}
        for key in ("domain", "scene", "role", "source_id"):
            value = original_plan.get(key)
            if value is not None:
                plan_row[key] = value
        normalized_plans.append(plan_row)
    normalized_payload = {
        "schema_version": 1,
        "format": NORMALIZED_SOURCE_PLAN_FORMAT,
        "sample_rate": sample_rate,
        "minimum_block_frames": minimum_frames,
        "source_split": "train",
        "prepared_provenance_manifest": (
            str(provenance_index.manifest_path) if provenance_index is not None else None
        ),
        "prepared_provenance_manifest_sha256": (
            provenance_index.manifest_sha256 if provenance_index is not None else None
        ),
        "plans": normalized_plans,
    }
    _atomic_write_json_no_replace(normalized_path, normalized_payload)
    audit["status"] = "completed"
    audit["counts"] = _audit_counts(audit_records)
    audit["output"] = {
        "normalized_source_plan": str(normalized_path),
        "normalized_source_plan_sha256": _sha256_file(normalized_path),
        "audio_root": str(audio_root),
    }
    _atomic_write_json_no_replace(audit_path, audit)
    return SourcePlanMaterializationResult(
        output_root=output_root_path,
        normalized_source_plan=normalized_path,
        audit_path=audit_path,
        block_count=len(candidates),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize train-only source-plan blocks")
    parser.add_argument("--source-plan", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--prepared-provenance-manifest")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--minimum-frames", type=int, default=17_536)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> SourcePlanMaterializationResult:
    args = parse_args(argv)
    return materialize_train_source_plan(
        args.source_plan,
        args.output_root,
        prepared_provenance_manifest=args.prepared_provenance_manifest,
        sample_rate=args.sample_rate,
        minimum_frames=args.minimum_frames,
    )


if __name__ == "__main__":
    main()
