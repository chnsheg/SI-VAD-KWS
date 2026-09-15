"""Recoverable V3 migration for changed clean-slot render completions."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from itertools import zip_longest
from .recipes import build_recipe
from pathlib import Path
from typing import Any, Iterable


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSON at {path}:{line_number}")
            yield row


def _request_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        example_id = row.get("example_id")
        if not isinstance(example_id, str) or not example_id:
            raise ValueError(f"request has no example_id: {path}")
        if example_id in rows:
            raise ValueError(f"duplicate request example_id: {example_id}")
        rows[example_id] = row
    return rows


def _canonical(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _is_clean(row: dict[str, Any]) -> bool:
    recipe = row.get("recipe")
    return isinstance(recipe, dict) and recipe.get("augmentation_group") == "clean_time_placement"


def _changed_clean_ids(old_requests: Path, new_requests: Path) -> set[str]:
    old_rows = _request_rows(old_requests)
    new_rows = _request_rows(new_requests)
    if old_rows.keys() != new_rows.keys():
        missing = len(old_rows.keys() ^ new_rows.keys())
        raise ValueError(f"request example-id sets differ: {missing}")
    changed = {example_id for example_id in old_rows if _canonical(old_rows[example_id]) != _canonical(new_rows[example_id])}
    non_clean = sorted(
        example_id
        for example_id in changed
        if not _is_clean(old_rows[example_id]) or not _is_clean(new_rows[example_id])
    )
    if non_clean:
        raise ValueError(f"non-clean request changed: {non_clean[0]}")
    return changed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle, source.open("rb") as input_handle:
            shutil.copyfileobj(input_handle, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def migrate_clean_completions(*, old_requests: Path, new_requests: Path, role_root: Path, quarantine_root: Path) -> dict[str, int]:
    """Archive changed clean completions and retain every unchanged completion."""
    changed_ids = _changed_clean_ids(Path(old_requests), Path(new_requests))
    archived_count = 0
    retained_count = 0
    metadata_root = Path(role_root) / "rendered" / "metadata"
    for metadata_path in sorted(metadata_root.glob("shard-*.jsonl")):
        retained_rows: list[dict[str, Any]] = []
        archived_rows: list[dict[str, Any]] = []
        for row in _iter_jsonl(metadata_path):
            example_id = row.get("example_id")
            if not isinstance(example_id, str) or not example_id:
                raise ValueError(f"completion has no example_id: {metadata_path}")
            if example_id not in changed_ids:
                retained_rows.append(row)
                retained_count += 1
                continue
            output_path = row.get("output_path")
            output_hash = row.get("output_sha256")
            if not isinstance(output_path, str) or not isinstance(output_hash, str):
                raise ValueError(f"completion lacks output provenance: {example_id}")
            source = Path(output_path)
            if not source.is_file() or _sha256(source) != output_hash:
                raise ValueError(f"completion output hash mismatch: {example_id}")
            archived_audio = Path(quarantine_root) / "audio" / metadata_path.stem / source.name
            if archived_audio.exists():
                if _sha256(archived_audio) != output_hash:
                    raise ValueError(f"quarantine hash mismatch: {example_id}")
            else:
                _atomic_copy(source, archived_audio)
                if _sha256(archived_audio) != output_hash:
                    raise ValueError(f"quarantine hash mismatch: {example_id}")
            archived = dict(row)
            archived["archived_from_output_path"] = str(source)
            archived["output_path"] = str(archived_audio.resolve())
            archived_rows.append(archived)
            archived_count += 1
        if archived_rows:
            archive_metadata = Path(quarantine_root) / "metadata" / metadata_path.name
            _atomic_write_jsonl(archive_metadata, archived_rows)
            _atomic_write_jsonl(metadata_path, retained_rows)
    return {
        "changed_request_count": len(changed_ids),
        "archived_clean_count": archived_count,
        "retained_completion_count": retained_count,
    }


def _rewrite_clean_pair(
    positive: dict[str, Any],
    negative: dict[str, Any],
    *,
    global_seed: int,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    counterpart_id = positive.get("counterpart_id")
    if not isinstance(counterpart_id, str) or counterpart_id != negative.get("counterpart_id"):
        raise ValueError("counterpart rows are not aligned")
    if positive.get("source_role") != "base_positive" or negative.get("source_role") != "base_negative":
        raise ValueError(f"unexpected counterpart roles: {counterpart_id}")
    positive_recipe = positive.get("recipe")
    negative_recipe = negative.get("recipe")
    if not isinstance(positive_recipe, dict) or not isinstance(negative_recipe, dict):
        raise ValueError(f"counterpart has no recipe: {counterpart_id}")
    if positive_recipe.get("augmentation_group") != negative_recipe.get("augmentation_group"):
        raise ValueError(f"counterpart augmentation groups differ: {counterpart_id}")
    if positive_recipe.get("augmentation_group") != "clean_time_placement":
        return positive, negative, False
    source_id = positive_recipe.get("source_id")
    source_sha256 = positive_recipe.get("source_sha256")
    split = positive_recipe.get("split")
    slot = positive_recipe.get("slot")
    source_kind = positive_recipe.get("source_kind")
    if not all(isinstance(value, str) and value for value in (source_id, source_sha256, split, source_kind)):
        raise ValueError(f"clean counterpart has incomplete positive provenance: {counterpart_id}")
    if isinstance(slot, bool) or not isinstance(slot, int):
        raise ValueError(f"clean counterpart has invalid slot: {counterpart_id}")
    recipe = build_recipe(
        global_seed,
        split,
        source_id,
        source_sha256,
        slot,
        label="positive",
        source_kind=source_kind,
    )
    rms = recipe.active_rms_dbfs
    if rms is None:
        raise ValueError(f"clean counterpart selected no RMS: {counterpart_id}")
    jitter_ms = recipe.jitter_ms
    rewritten_positive = dict(positive)
    rewritten_negative = dict(negative)
    rewritten_positive_recipe = dict(positive_recipe)
    rewritten_negative_recipe = dict(negative_recipe)
    rewritten_positive_recipe["active_rms_dbfs"] = rms
    rewritten_negative_recipe["active_rms_dbfs"] = rms
    rewritten_positive_recipe["jitter_ms"] = jitter_ms
    rewritten_negative_recipe["jitter_ms"] = jitter_ms
    rewritten_positive["recipe"] = rewritten_positive_recipe
    rewritten_negative["recipe"] = rewritten_negative_recipe
    changed = _canonical(rewritten_positive) != _canonical(positive) or _canonical(rewritten_negative) != _canonical(negative)
    return rewritten_positive, rewritten_negative, changed


def write_deduplicated_counterfactual_requests(*, old_root: Path, destination_root: Path, global_seed: int) -> dict[str, int]:
    """Publish deduplicated base-role requests without re-sampling augmentation inputs."""
    old_root = Path(old_root)
    destination = Path(destination_root)
    if destination.exists():
        raise FileExistsError(destination)
    positive_path = old_root / "base_positive.requests.jsonl"
    negative_path = old_root / "base_negative.requests.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    pair_count = 0
    changed_pair_count = 0
    try:
        output_positive = staging / positive_path.name
        output_negative = staging / negative_path.name
        with output_positive.open("w", encoding="utf-8", newline="\n") as positive_handle, output_negative.open("w", encoding="utf-8", newline="\n") as negative_handle:
            positive_rows = _iter_jsonl(positive_path)
            negative_rows = _iter_jsonl(negative_path)
            for pair_count, pair in enumerate(zip_longest(positive_rows, negative_rows), start=1):
                positive, negative = pair
                if positive is None or negative is None:
                    raise ValueError("counterpart request lengths differ")
                rewritten_positive, rewritten_negative, changed = _rewrite_clean_pair(positive, negative, global_seed=global_seed)
                positive_handle.write(json.dumps(rewritten_positive, ensure_ascii=False, sort_keys=True) + "\n")
                negative_handle.write(json.dumps(rewritten_negative, ensure_ascii=False, sort_keys=True) + "\n")
                changed_pair_count += int(changed)
            positive_handle.flush()
            negative_handle.flush()
            os.fsync(positive_handle.fileno())
            os.fsync(negative_handle.fileno())
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "pair_count": pair_count,
        "changed_pair_count": changed_pair_count,
        "changed_record_count": changed_pair_count * 2,
    }
