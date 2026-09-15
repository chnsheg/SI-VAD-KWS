"""Atomic role-local PCM publication for the V3 mixture trainer."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import uuid

import soundfile as sf
import torch

from .packed_mixture import CompositePackedDataset
from .packed_pcm import PackedPcmDataset, PackedPcmShardWriter


_LABELS = {"positive": 0, "negative": 1}


@dataclass(frozen=True)
class RoleSpec:
    name: str
    label: str
    metadata_paths: tuple[Path, ...]
    pack_root: Path
    expected_records: int | None = None


@dataclass(frozen=True)
class RolePackResult:
    name: str
    manifest_path: Path
    record_count: int


def _read_metadata(spec: RoleSpec):
    if spec.label not in _LABELS:
        raise ValueError(f"Unsupported role label for {spec.name}: {spec.label}")
    for metadata_path in spec.metadata_paths:
        with Path(metadata_path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid metadata JSON for {spec.name}: {metadata_path}:{line_number}") from error
                if not isinstance(row, dict) or row.get("label") != spec.label:
                    raise ValueError(f"Unexpected label for {spec.name}: {metadata_path}:{line_number}")
                output_path = row.get("output_path")
                if not isinstance(output_path, str) or not output_path:
                    raise ValueError(f"Missing output_path for {spec.name}: {metadata_path}:{line_number}")
                recipe = row.get("recipe", {})
                jitter_ms = recipe.get("jitter_ms", 0) if isinstance(recipe, dict) else 0
                if jitter_ms is None:
                    jitter_ms = 0
                elif isinstance(jitter_ms, float) and jitter_ms.is_integer():
                    jitter_ms = int(jitter_ms)
                elif isinstance(jitter_ms, int) and jitter_ms < 0:
                    # The packed format stores a non-negative provenance field;
                    # the rendered waveform already contains the signed offset.
                    jitter_ms = 0
                if isinstance(jitter_ms, bool) or not isinstance(jitter_ms, int) or not 0 <= jitter_ms <= 65_535:
                    raise ValueError(f"Invalid jitter_ms for {spec.name}: {metadata_path}:{line_number}")
                yield Path(output_path), jitter_ms


def _published_result(spec: RoleSpec) -> RolePackResult | None:
    manifest_path = Path(spec.pack_root) / "manifest.json"
    if not manifest_path.is_file():
        return None
    dataset = PackedPcmDataset(manifest_path)
    try:
        if spec.expected_records is not None and dataset.record_count != spec.expected_records:
            raise ValueError(f"Published {spec.name} pack has {dataset.record_count} records, expected {spec.expected_records}")
        if dataset.sample_rate != 16_000 or dataset.record_samples != 16_000:
            raise ValueError(f"Published {spec.name} pack violates the 16 kHz one-second contract")
        if len(dataset.label_indices(_LABELS[spec.label])) != dataset.record_count:
            raise ValueError(f"Published {spec.name} pack has an unexpected label")
        return RolePackResult(spec.name, manifest_path.resolve(), dataset.record_count)
    finally:
        dataset.close()


def publish_role_pack(spec: RoleSpec) -> RolePackResult:
    """Build a role pack off-path, then atomically publish its manifest and shards."""
    published = _published_result(spec)
    if published is not None:
        return published
    pack_root = Path(spec.pack_root).resolve()
    if pack_root.exists():
        raise ValueError(f"Refusing to replace unpublished role pack directory: {pack_root}")
    pack_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = pack_root.parent / f".{pack_root.name}.{uuid.uuid4().hex}.tmp"
    writer = PackedPcmShardWriter(temporary_root, sample_rate=16_000, shard_records=16_384)
    try:
        for output_path, jitter_ms in _read_metadata(spec):
            waveform, sample_rate = sf.read(output_path, dtype="float32", always_2d=False)
            if sample_rate != 16_000:
                raise ValueError(f"{spec.name} audio is {sample_rate} Hz, expected 16000: {output_path}")
            writer.add(torch.as_tensor(waveform, dtype=torch.float32).reshape(1, -1), label=_LABELS[spec.label], jitter_ms=jitter_ms)
        if spec.expected_records is not None and writer.record_count != spec.expected_records:
            raise ValueError(f"Role {spec.name} has {writer.record_count} records, expected {spec.expected_records}")
        manifest_path = writer.finalize(source_manifest=",".join(str(path) for path in spec.metadata_paths), metadata={"role": spec.name})
        result = _published_result(RoleSpec(spec.name, spec.label, spec.metadata_paths, temporary_root, spec.expected_records))
        assert result is not None and result.manifest_path == manifest_path.resolve()
        os.replace(temporary_root, pack_root)
        return RolePackResult(spec.name, pack_root / "manifest.json", result.record_count)
    except Exception:
        writer.close()
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def probe_v3_role_manifests(manifests: dict[str, Path | str]) -> None:
    dataset = CompositePackedDataset.from_v3_manifests(manifests)
    try:
        for role_index, role in enumerate(dataset.roles):
            waveform, label, _, observed_name = dataset[(role_index, int(role.indexes[0]))]
            if waveform.shape != (1, 16_000) or label != role.label or observed_name != role.name:
                raise ValueError(f"V3 role probe failed for {role.name}")
    finally:
        dataset.close()
