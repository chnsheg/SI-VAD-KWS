import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dscnn_kws.data.build_reclean_training_manifests import (
    build_reclean_training_manifests,
    resolve_current_training_manifests,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _metadata_row(path: Path, label: str, *, jitter_ms: int = 200) -> dict[str, object]:
    return {
        "example_id": path.stem,
        "output_path": str(path),
        "label": label,
        "recipe": {"online_window_jitter_max_ms": jitter_ms},
    }


def _write_clean_manifest(path: Path, rows: list[dict[str, str]]) -> str:
    content = "".join(json.dumps(row) + "\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return content


def test_builds_immutable_training_splits_from_streamed_metadata_and_clean_manifests(tmp_path):
    corpus_root = tmp_path / "corpus"
    metadata_root = corpus_root / "metadata"
    rows_by_shard = (
        [_metadata_row(tmp_path / "audio" / "positive-0.wav", "positive")],
        [_metadata_row(tmp_path / "audio" / "negative-0.wav", "negative", jitter_ms=999)],
        [_metadata_row(tmp_path / "audio" / "positive-1.wav", "positive")],
        [_metadata_row(tmp_path / "audio" / "negative-1.wav", "negative")],
    )
    for shard_index, rows in enumerate(rows_by_shard):
        _write_jsonl(metadata_root / f"shard-{shard_index:02d}.jsonl", rows)

    source_root = corpus_root / "source_manifests"
    validation_content = _write_clean_manifest(
        source_root / "validation_manifest.json",
        [{"audio_filepath": "/clean/validation.wav", "command": "positive"}],
    )
    test_content = _write_clean_manifest(
        source_root / "test_manifest.json",
        [
            {"audio_filepath": "/clean/test-positive.wav", "command": "positive"},
            {"audio_filepath": "/clean/test-negative.wav", "command": "negative"},
        ],
    )

    result = build_reclean_training_manifests(
        corpus_root,
        expected_train_count=4,
        expected_validation_count=1,
        expected_test_count=2,
    )

    assert result.train_manifest_path.parent.name.startswith("generation-")
    assert result.validation_manifest_path.parent == result.train_manifest_path.parent
    assert result.test_manifest_path.parent == result.train_manifest_path.parent
    assert (result.train_count, result.validation_count, result.test_count) == (4, 1, 2)
    train_rows = [json.loads(line) for line in result.train_manifest_path.read_text(encoding="utf-8").splitlines()]
    assert train_rows == [
        {
            "audio_filepath": str((tmp_path / "audio" / "positive-0.wav").resolve()),
            "command": "positive",
            "online_window_jitter_max_ms": 200,
        },
        {
            "audio_filepath": str((tmp_path / "audio" / "negative-0.wav").resolve()),
            "command": "negative",
            "online_window_jitter_max_ms": 0,
        },
        {
            "audio_filepath": str((tmp_path / "audio" / "positive-1.wav").resolve()),
            "command": "positive",
            "online_window_jitter_max_ms": 200,
        },
        {
            "audio_filepath": str((tmp_path / "audio" / "negative-1.wav").resolve()),
            "command": "negative",
            "online_window_jitter_max_ms": 0,
        },
    ]
    assert result.validation_manifest_path.read_text(encoding="utf-8") == validation_content
    assert result.test_manifest_path.read_text(encoding="utf-8") == test_content


@pytest.mark.parametrize(
    "row, message",
    [
        ({"label": "positive", "recipe": {"online_window_jitter_max_ms": 200}}, "output_path"),
        ({"output_path": "/audio.wav", "label": "unknown"}, "label"),
        (
            {"output_path": "/audio.wav", "label": "positive", "recipe": {"online_window_jitter_max_ms": 100}},
            "online_window_jitter_max_ms",
        ),
    ],
)
def test_rejects_invalid_metadata_without_replacing_existing_training_manifest(tmp_path, row, message):
    corpus_root = tmp_path / "corpus"
    row = dict(row)
    if "output_path" in row:
        row["output_path"] = str(tmp_path / "audio.wav")
    _write_jsonl(corpus_root / "metadata" / "shard-00.jsonl", [row])
    for shard_index in range(1, 4):
        _write_jsonl(corpus_root / "metadata" / f"shard-{shard_index:02d}.jsonl", [])
    _write_clean_manifest(corpus_root / "source_manifests" / "validation_manifest.json", [])
    _write_clean_manifest(corpus_root / "source_manifests" / "test_manifest.json", [])
    target = corpus_root / "training_manifests" / "train_manifest.json"
    target.parent.mkdir(parents=True)
    target.write_text("previous manifest\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        build_reclean_training_manifests(
            corpus_root,
            expected_train_count=1,
            expected_validation_count=0,
            expected_test_count=0,
        )

    assert target.read_text(encoding="utf-8") == "previous manifest\n"


def test_rejects_count_mismatches_before_writing_outputs(tmp_path):
    corpus_root = tmp_path / "corpus"
    for shard_index in range(4):
        _write_jsonl(corpus_root / "metadata" / f"shard-{shard_index:02d}.jsonl", [])
    _write_clean_manifest(corpus_root / "source_manifests" / "validation_manifest.json", [])
    _write_clean_manifest(corpus_root / "source_manifests" / "test_manifest.json", [])

    with pytest.raises(ValueError, match="Training manifest count is 0, expected 1"):
        build_reclean_training_manifests(
            corpus_root,
            expected_train_count=1,
            expected_validation_count=0,
            expected_test_count=0,
        )

    assert not (corpus_root / "training_manifests" / "current.json").exists()


def test_rejects_relative_metadata_output_path(tmp_path):
    corpus_root = tmp_path / "corpus"
    _write_jsonl(
        corpus_root / "metadata" / "shard-00.jsonl",
        [{"output_path": "audio/positive.wav", "label": "positive", "recipe": {"online_window_jitter_max_ms": 200}}],
    )
    for shard_index in range(1, 4):
        _write_jsonl(corpus_root / "metadata" / f"shard-{shard_index:02d}.jsonl", [])
    _write_clean_manifest(corpus_root / "source_manifests" / "validation_manifest.json", [])
    _write_clean_manifest(corpus_root / "source_manifests" / "test_manifest.json", [])

    with pytest.raises(ValueError, match="absolute output_path"):
        build_reclean_training_manifests(
            corpus_root,
            expected_train_count=1,
            expected_validation_count=0,
            expected_test_count=0,
        )


def test_build_does_not_resolve_metadata_audio_paths(tmp_path, monkeypatch):
    corpus_root = tmp_path / "corpus"
    _write_jsonl(
        corpus_root / "metadata" / "shard-00.jsonl",
        [_metadata_row(tmp_path / "audio" / "positive.wav", "positive")],
    )
    for shard_index in range(1, 4):
        _write_jsonl(corpus_root / "metadata" / f"shard-{shard_index:02d}.jsonl", [])
    _write_clean_manifest(corpus_root / "source_manifests" / "validation_manifest.json", [])
    _write_clean_manifest(corpus_root / "source_manifests" / "test_manifest.json", [])

    def unexpected_resolve(*args, **kwargs):
        raise AssertionError("manifest construction must not resolve audio paths")

    monkeypatch.setattr(Path, "resolve", unexpected_resolve)

    result = build_reclean_training_manifests(
        corpus_root,
        expected_train_count=1,
        expected_validation_count=0,
        expected_test_count=0,
    )

    assert result.train_manifest_path.read_text(encoding="utf-8")


def test_second_publish_switches_json_pointer_and_preserves_first_generation(tmp_path):
    corpus_root = tmp_path / "corpus"
    metadata = corpus_root / "metadata" / "shard-00.jsonl"
    _write_jsonl(metadata, [_metadata_row(tmp_path / "audio" / "first.wav", "positive")])
    for shard_index in range(1, 4):
        _write_jsonl(corpus_root / "metadata" / f"shard-{shard_index:02d}.jsonl", [])
    _write_clean_manifest(corpus_root / "source_manifests" / "validation_manifest.json", [])
    _write_clean_manifest(corpus_root / "source_manifests" / "test_manifest.json", [])

    first_result = build_reclean_training_manifests(
        corpus_root,
        expected_train_count=1,
        expected_validation_count=0,
        expected_test_count=0,
    )
    first_train_content = first_result.train_manifest_path.read_text(encoding="utf-8")
    _write_jsonl(metadata, [_metadata_row(tmp_path / "audio" / "second.wav", "positive")])

    second_result = build_reclean_training_manifests(
        corpus_root,
        expected_train_count=1,
        expected_validation_count=0,
        expected_test_count=0,
    )
    resolved = resolve_current_training_manifests(corpus_root)
    pointer_path = corpus_root / "training_manifests" / "current.json"

    assert second_result.train_manifest_path != first_result.train_manifest_path
    assert first_result.train_manifest_path.read_text(encoding="utf-8") == first_train_content
    assert resolved.train_manifest_path == second_result.train_manifest_path
    assert "second.wav" in resolved.train_manifest_path.read_text(encoding="utf-8")
    assert json.loads(pointer_path.read_text(encoding="utf-8")) == {"generation": second_result.train_manifest_path.parent.name}
    assert pointer_path.is_file()
    assert not (corpus_root / "training_manifests" / "current").exists()
    assert not any(path.is_symlink() for path in (corpus_root / "training_manifests").iterdir())


def test_failed_current_pointer_replace_leaves_previous_generation_readable(tmp_path, monkeypatch):
    corpus_root = tmp_path / "corpus"
    metadata = corpus_root / "metadata" / "shard-00.jsonl"
    _write_jsonl(metadata, [_metadata_row(tmp_path / "audio" / "first.wav", "positive")])
    for shard_index in range(1, 4):
        _write_jsonl(corpus_root / "metadata" / f"shard-{shard_index:02d}.jsonl", [])
    _write_clean_manifest(corpus_root / "source_manifests" / "validation_manifest.json", [])
    _write_clean_manifest(corpus_root / "source_manifests" / "test_manifest.json", [])
    first_result = build_reclean_training_manifests(
        corpus_root,
        expected_train_count=1,
        expected_validation_count=0,
        expected_test_count=0,
    )
    previous_train_content = first_result.train_manifest_path.read_text(encoding="utf-8")
    _write_jsonl(metadata, [_metadata_row(tmp_path / "audio" / "second.wav", "positive")])

    import importlib

    builder_module = importlib.import_module("dscnn_kws.data.build_reclean_training_manifests")
    actual_replace = os.replace

    def fail_current_switch(source, destination):
        if Path(destination) == corpus_root / "training_manifests" / "current.json":
            raise OSError("simulated current switch failure")
        return actual_replace(source, destination)

    monkeypatch.setattr(builder_module.os, "replace", fail_current_switch)

    with pytest.raises(OSError, match="simulated current switch failure"):
        build_reclean_training_manifests(
            corpus_root,
            expected_train_count=1,
            expected_validation_count=0,
            expected_test_count=0,
        )

    resolved = resolve_current_training_manifests(corpus_root)
    assert resolved.train_manifest_path.read_text(encoding="utf-8") == previous_train_content
    generations = [path for path in (corpus_root / "training_manifests").iterdir() if path.name.startswith("generation-")]
    assert len(generations) == 2


@pytest.mark.parametrize("generation", ["../generation-other", "generation-other/../outside", "/absolute-generation", "generation\\outside"])
def test_current_pointer_rejects_nonrelative_generation_paths(tmp_path, generation):
    pointer_path = tmp_path / "corpus" / "training_manifests" / "current.json"
    pointer_path.parent.mkdir(parents=True)
    pointer_path.write_text(json.dumps({"generation": generation}), encoding="utf-8")

    with pytest.raises(ValueError, match="relative generation"):
        resolve_current_training_manifests(tmp_path / "corpus")


def test_module_cli_is_warning_free():
    package_root = Path(__file__).resolve().parents[3]
    command = (
        "import runpy, sys; "
        f"sys.path.insert(0, {str(package_root)!r}); "
        "sys.argv = ['build_reclean_training_manifests', '--help']; "
        "runpy.run_module('dscnn_kws.data.build_reclean_training_manifests', run_name='__main__')"
    )

    result = subprocess.run([sys.executable, "-c", command], text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr
    assert "--corpus-root" in result.stdout
    assert "RuntimeWarning" not in result.stderr
