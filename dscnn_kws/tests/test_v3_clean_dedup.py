from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dscnn_kws.data.reclean import v3_clean_dedup
from dscnn_kws.data.reclean.v3_clean_dedup import migrate_clean_completions, write_deduplicated_counterfactual_requests


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _request(example_id: str, group: str, rms: float) -> dict[str, object]:
    return {
        "example_id": example_id,
        "recipe": {"augmentation_group": group, "active_rms_dbfs": rms},
    }


def test_migration_archives_changed_clean_completion_and_retains_environment(tmp_path: Path) -> None:
    role_root = tmp_path / "role"
    metadata_path = role_root / "rendered" / "metadata" / "shard-00.jsonl"
    metadata_path.parent.mkdir(parents=True)
    clean_audio = role_root / "rendered" / "audio" / "train" / "shard-00" / "positive" / "clean.wav"
    environment_audio = role_root / "rendered" / "audio" / "train" / "shard-00" / "positive" / "environment.wav"
    clean_audio.parent.mkdir(parents=True)
    clean_audio.write_bytes(b"clean")
    environment_audio.write_bytes(b"environment")
    clean_hash = hashlib.sha256(clean_audio.read_bytes()).hexdigest()
    environment_hash = hashlib.sha256(environment_audio.read_bytes()).hexdigest()
    _write_jsonl(
        metadata_path,
        [
            {"example_id": "clean", "output_path": str(clean_audio), "output_sha256": clean_hash},
            {"example_id": "environment", "output_path": str(environment_audio), "output_sha256": environment_hash},
        ],
    )
    old_requests = tmp_path / "old.requests.jsonl"
    new_requests = tmp_path / "new.requests.jsonl"
    _write_jsonl(old_requests, [_request("clean", "clean_time_placement", -34.0), _request("environment", "environment", -34.0)])
    _write_jsonl(new_requests, [_request("clean", "clean_time_placement", -30.0), _request("environment", "environment", -34.0)])

    result = migrate_clean_completions(
        old_requests=old_requests,
        new_requests=new_requests,
        role_root=role_root,
        quarantine_root=tmp_path / "quarantine",
    )

    assert result == {"changed_request_count": 1, "archived_clean_count": 1, "retained_completion_count": 1}
    assert clean_audio.exists()
    assert environment_audio.exists()
    active_rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    assert active_rows == [{"example_id": "environment", "output_path": str(environment_audio), "output_sha256": environment_hash}]
    archived_rows = [
        json.loads(line)
        for line in (tmp_path / "quarantine" / "metadata" / "shard-00.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert archived_rows[0]["example_id"] == "clean"
    archived_audio = Path(archived_rows[0]["output_path"])
    assert archived_audio.read_bytes() == b"clean"
    assert archived_rows[0]["archived_from_output_path"] == str(clean_audio)


def test_migration_rejects_a_changed_non_clean_request(tmp_path: Path) -> None:
    old_requests = tmp_path / "old.requests.jsonl"
    new_requests = tmp_path / "new.requests.jsonl"
    _write_jsonl(old_requests, [_request("environment", "environment", -34.0)])
    _write_jsonl(new_requests, [_request("environment", "environment", -30.0)])

    try:
        migrate_clean_completions(
            old_requests=old_requests,
            new_requests=new_requests,
            role_root=tmp_path / "role",
            quarantine_root=tmp_path / "quarantine",
        )
    except ValueError as error:
        assert "non-clean" in str(error)
    else:
        raise AssertionError("expected migration to reject a changed non-clean request")


def test_migration_rejects_corrupt_new_quarantine_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    role_root = tmp_path / "role"
    metadata_path = role_root / "rendered" / "metadata" / "shard-00.jsonl"
    metadata_path.parent.mkdir(parents=True)
    clean_audio = role_root / "rendered" / "audio" / "train" / "shard-00" / "positive" / "clean.wav"
    clean_audio.parent.mkdir(parents=True)
    clean_audio.write_bytes(b"clean")
    clean_hash = hashlib.sha256(clean_audio.read_bytes()).hexdigest()
    active_row = {"example_id": "clean", "output_path": str(clean_audio), "output_sha256": clean_hash}
    _write_jsonl(metadata_path, [active_row])
    old_requests = tmp_path / "old.requests.jsonl"
    new_requests = tmp_path / "new.requests.jsonl"
    _write_jsonl(old_requests, [_request("clean", "clean_time_placement", -34.0)])
    _write_jsonl(new_requests, [_request("clean", "clean_time_placement", -30.0)])

    def corrupt_copy(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"corrupt")

    monkeypatch.setattr(v3_clean_dedup, "_atomic_copy", corrupt_copy)

    with pytest.raises(ValueError, match="quarantine hash mismatch: clean"):
        migrate_clean_completions(
            old_requests=old_requests,
            new_requests=new_requests,
            role_root=role_root,
            quarantine_root=tmp_path / "quarantine",
        )

    assert [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()] == [active_row]
    assert not (tmp_path / "quarantine" / "metadata" / "shard-00.jsonl").exists()


def _counterpart_request(
    example_id: str,
    counterpart_id: str,
    role: str,
    label: str,
    slot: int,
    group: str,
    rms: float,
    source_sha256: str,
) -> dict[str, object]:
    return {
        "example_id": example_id,
        "counterpart_id": counterpart_id,
        "source_role": role,
        "label": label,
        "recipe": {
            "split": "train",
            "source_id": source_sha256,
            "source_sha256": source_sha256,
            "slot": slot,
            "label": label,
            "source_kind": "speech",
            "augmentation_group": group,
            "active_rms_dbfs": rms,
            "jitter_ms": 0,
        },
    }


def test_request_rewrite_changes_only_clean_pairs_and_keeps_pair_rms_equal(tmp_path: Path) -> None:
    old_root = tmp_path / "old"
    old_root.mkdir()
    positive_rows = []
    negative_rows = []
    for slot in range(4):
        counterpart_id = f"pair-{slot}"
        positive_rows.append(
            _counterpart_request(f"{counterpart_id}-positive", counterpart_id, "base_positive", "positive", slot, "clean_time_placement", -34.0, "positive-source")
        )
        negative_rows.append(
            _counterpart_request(f"{counterpart_id}-negative", counterpart_id, "base_negative", "negative", slot, "clean_time_placement", -34.0, f"negative-{slot}")
        )
    positive_rows.append(_counterpart_request("environment-positive", "environment", "base_positive", "positive", 4, "environment", -30.0, "positive-source"))
    negative_rows.append(_counterpart_request("environment-negative", "environment", "base_negative", "negative", 4, "environment", -30.0, "negative-environment"))
    _write_jsonl(old_root / "base_positive.requests.jsonl", positive_rows)
    _write_jsonl(old_root / "base_negative.requests.jsonl", negative_rows)

    destination = tmp_path / "deduplicated"
    result = write_deduplicated_counterfactual_requests(old_root=old_root, destination_root=destination, global_seed=20260831)

    assert result["pair_count"] == 5
    rewritten_positive = [json.loads(line) for line in (destination / "base_positive.requests.jsonl").read_text(encoding="utf-8").splitlines()]
    rewritten_negative = [json.loads(line) for line in (destination / "base_negative.requests.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len({row["recipe"]["active_rms_dbfs"] for row in rewritten_positive[:4]}) == 4
    assert len({row["recipe"]["jitter_ms"] for row in rewritten_positive[:4]}) == 4
    assert [row["recipe"]["active_rms_dbfs"] for row in rewritten_positive] == [row["recipe"]["active_rms_dbfs"] for row in rewritten_negative]
    assert [row["recipe"]["jitter_ms"] for row in rewritten_positive] == [row["recipe"]["jitter_ms"] for row in rewritten_negative]
    assert all(row["recipe"]["jitter_ms"] != 0 for row in rewritten_positive[:4])
    assert rewritten_positive[-1] == positive_rows[-1]
    assert rewritten_negative[-1] == negative_rows[-1]
