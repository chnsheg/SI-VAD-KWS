from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from dscnn_kws.data.packed_pcm import PackedPcmShardWriter
from dscnn_kws.data.reclean.v3_reuse import (
    audit_v2_raw_anchor_for_reuse,
    audit_v2_packed_role,
    audit_v2_role_provenance,
    write_filtered_v2_role_reference,
    write_reuse_audit,
)


def write_packed_fixture(root: Path, *, labels: bytes, source_rows: list[dict[str, object]] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    source_manifest = root / "source.jsonl"
    source_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in (source_rows or [{"source_split": "train"}])),
        encoding="utf-8",
    )
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "packed_pcm16_v1",
                "labels": "labels.u8",
                "jitter_ms": "jitter_ms.u16le",
                "record_count": len(labels),
                "record_samples": 16000,
                "sample_rate": 16000,
                "source_manifest": str(source_manifest),
                "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    (root / "labels.u8").write_bytes(labels)
    (root / "jitter_ms.u16le").write_bytes(b"\x00\x00" * len(labels))
    return manifest


def test_reuse_rejects_a_pack_with_nonbinary_labels(tmp_path: Path) -> None:
    manifest = write_packed_fixture(tmp_path, labels=b"\x00\x02")

    with pytest.raises(ValueError, match="binary"):
        audit_v2_packed_role(manifest, role="raw_positive")


def test_reuse_rejects_known_v2_negative_and_hard_negative_roles(tmp_path: Path) -> None:
    base = write_packed_fixture(tmp_path / "base", labels=b"\x00\x01")
    hard = write_packed_fixture(tmp_path / "hard", labels=b"\x01")

    assert audit_v2_packed_role(base, role="base_positive").eligible is False
    assert audit_v2_packed_role(base, role="base_negative").eligible is False
    assert audit_v2_packed_role(hard, role="false_wake_hard_negative").eligible is False


def test_reuse_audit_never_mutates_v2_input(tmp_path: Path) -> None:
    raw = write_packed_fixture(tmp_path / "raw", labels=b"\x00\x01")
    before = raw.read_bytes()

    audit_v2_packed_role(raw, role="raw_positive")

    assert raw.read_bytes() == before


def test_raw_anchor_provenance_requires_train_split_and_matching_role(tmp_path: Path) -> None:
    manifest = write_packed_fixture(
        tmp_path,
        labels=b"\x00\x01",
        source_rows=[
            {"source_split": "train", "source_role": "raw_positive", "command": "positive"},
            {"source_split": "train", "source_role": "raw_negative", "command": "negative"},
        ],
    )

    assert audit_v2_role_provenance(manifest, role="raw_positive").eligible is True
    assert audit_v2_role_provenance(manifest, role="raw_negative").eligible is True


def test_raw_anchor_provenance_rejects_held_out_source(tmp_path: Path) -> None:
    manifest = write_packed_fixture(
        tmp_path,
        labels=b"\x00",
        source_rows=[{"source_split": "test", "source_role": "raw_positive", "command": "positive"}],
    )

    decision = audit_v2_role_provenance(manifest, role="raw_positive")

    assert decision.eligible is False
    assert any("train-only" in reason for reason in decision.reasons)


def test_reuse_audit_publishes_atomically_without_v2_writes(tmp_path: Path) -> None:
    raw = write_packed_fixture(tmp_path / "v2", labels=b"\x00\x01")
    decision = audit_v2_role_provenance(raw, role="raw_positive")
    output = tmp_path / "v3" / "v2_reuse_audit.json"

    write_reuse_audit(output, [decision])

    published = json.loads(output.read_text(encoding="utf-8"))
    assert published["schema_version"] == 1
    assert published["decisions"] == json.loads(json.dumps([decision.as_dict()]))
    assert raw.exists()
    assert not list(output.parent.glob(".v2_reuse_audit.json.*.tmp"))


def write_pcm_packed_fixture(root: Path, *, positive_waveform: torch.Tensor) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    source_manifest = root / "source.jsonl"
    source_manifest.write_text(
        '{"source_split":"train","source_role":"raw_positive","command":"positive"}\n'
        '{"source_split":"train","source_role":"raw_negative","command":"negative"}\n',
        encoding="utf-8",
    )
    writer = PackedPcmShardWriter(root, shard_records=2)
    writer.add(positive_waveform, label=0, jitter_ms=0)
    writer.add(torch.full((1, 16000), 0.1), label=1, jitter_ms=0)
    return writer.finalize(
        source_manifest=str(source_manifest),
        metadata={"source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest()},
    )


def test_raw_positive_reuse_rejects_a_silent_packed_record(tmp_path: Path) -> None:
    manifest = write_pcm_packed_fixture(tmp_path, positive_waveform=torch.zeros((1, 16000)))

    decision = audit_v2_raw_anchor_for_reuse(manifest, role="raw_positive", minimum_rms_dbfs=-50.0)

    assert decision.eligible is False
    assert any("RMS" in reason for reason in decision.reasons)
    assert decision.quality["below_minimum_rms_count"] == 1


def test_raw_positive_reuse_accepts_a_non_silent_packed_record(tmp_path: Path) -> None:
    manifest = write_pcm_packed_fixture(tmp_path, positive_waveform=torch.full((1, 16000), 0.1))

    decision = audit_v2_raw_anchor_for_reuse(manifest, role="raw_positive", minimum_rms_dbfs=-50.0)

    assert decision.eligible is True
    assert decision.quality["below_minimum_rms_count"] == 0


def test_filtered_raw_positive_reference_excludes_only_silent_indices(tmp_path: Path) -> None:
    root = tmp_path / "v2"
    root.mkdir()
    source_manifest = root / "source.jsonl"
    source_manifest.write_text(
        '{"source_split":"train","source_role":"raw_positive","command":"positive"}\n'
        '{"source_split":"train","source_role":"raw_positive","command":"positive"}\n'
        '{"source_split":"train","source_role":"raw_negative","command":"negative"}\n',
        encoding="utf-8",
    )
    writer = PackedPcmShardWriter(root, shard_records=3)
    writer.add(torch.zeros((1, 16000)), label=0, jitter_ms=0)
    writer.add(torch.full((1, 16000), 0.1), label=0, jitter_ms=0)
    writer.add(torch.full((1, 16000), 0.1), label=1, jitter_ms=0)
    manifest = writer.finalize(
        source_manifest=str(source_manifest),
        metadata={"source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest()},
    )
    decision = audit_v2_raw_anchor_for_reuse(manifest, role="raw_positive", minimum_rms_dbfs=-50.0)
    reference = tmp_path / "v3" / "raw_positive.reference.json"

    write_filtered_v2_role_reference(reference, decision)

    published = json.loads(reference.read_text(encoding="utf-8"))
    assert published["storage"] == "external_read_only"
    assert published["target_label"] == 0
    assert published["excluded_record_indices"] == [0]
    assert published["record_count"] == 1
    assert manifest.exists()
