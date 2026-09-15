from __future__ import annotations

import json
from pathlib import Path

import pytest
import soundfile as sf

from dscnn_kws.data.v3_handoff import RoleSpec, publish_role_pack


def _write_metadata(path: Path, *, rows: int, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio_path = path.parent / "sample.wav"
    sf.write(audio_path, [0.0] * 16_000, 16_000, subtype="PCM_16")
    with path.open("w", encoding="utf-8") as handle:
        for index in range(rows):
            handle.write(
                json.dumps(
                    {
                        "output_path": str(audio_path),
                        "label": label,
                        "sample_rate": 16_000,
                        "recipe": {"jitter_ms": index},
                    }
                )
                + "\n"
            )


def test_publish_role_pack_does_not_publish_partial_manifest(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.jsonl"
    _write_metadata(metadata_path, rows=1, label="negative")
    spec = RoleSpec("base_negative", "negative", (metadata_path,), tmp_path / "pack", expected_records=2)

    with pytest.raises(ValueError, match="expected 2"):
        publish_role_pack(spec)

    assert not (spec.pack_root / "manifest.json").exists()


def test_publish_role_pack_writes_readable_pcm_manifest(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.jsonl"
    _write_metadata(metadata_path, rows=2, label="positive")
    spec = RoleSpec("base_positive", "positive", (metadata_path,), tmp_path / "pack", expected_records=2)

    result = publish_role_pack(spec)

    assert result.record_count == 2
    assert result.manifest_path.is_file()
