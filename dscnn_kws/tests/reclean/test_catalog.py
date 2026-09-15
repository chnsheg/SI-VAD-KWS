from collections import Counter
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest
import soundfile as sf
import numpy as np

from dscnn_kws.data.reclean import catalog
from dscnn_kws.data.build_reclean_augmented_dataset import build_parser
from dscnn_kws.data.reclean.catalog import (
    CAPTURED_SCENES,
    TAU_SCENES,
    assign_false_wake_splits,
    build_noise_catalog,
    read_mobvoi_manifest_rows,
    validate_noise_scene_roots,
)


def test_scene_choice_is_uniform_by_scene_not_file_count(tmp_path):
    catalog = build_noise_catalog({"tau": [tmp_path / "a.wav"] * 100, "wind": [tmp_path / "b.wav"]})

    counts = Counter(catalog.choose_scene(random.Random(seed)).scene for seed in range(2000))

    assert abs(counts["tau"] - counts["wind"]) < 160


def test_false_wake_split_never_splits_one_source_recording():
    rows = assign_false_wake_splits(["a.wav", "a.wav", "b.wav", "c.wav", "d.wav"], seed=42)

    assert len({row.split for row in rows if row.source_id == "a.wav"}) == 1


def test_noise_root_validation_excludes_office_and_requires_all_remaining_scenes(tmp_path):
    roots = {scene: tmp_path for scene in (*TAU_SCENES, *CAPTURED_SCENES)}
    assert len(roots) == 15
    assert "office" not in roots
    validate_noise_scene_roots(roots)
    roots["office"] = tmp_path

    with pytest.raises(ValueError, match="unexpected scenes: office"):
        validate_noise_scene_roots(roots)


def test_dataset_cli_wrapper_runs_under_isolated_python():
    wrapper = Path(__file__).resolve().parents[3] / "run_reclean_augmented_dataset.py"

    result = subprocess.run(
        [sys.executable, str(wrapper), "inventory", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--office-root" not in result.stdout


def test_mobvoi_manifest_inventory_retains_source_label(tmp_path):
    audio = tmp_path / "source.wav"
    sf.write(audio, np.zeros(16000), 16000)
    manifest = tmp_path / "train_manifest.json"
    manifest.write_text(json.dumps({"audio_filepath": "source.wav", "command": "positive"}) + "\n", encoding="utf-8")

    rows = read_mobvoi_manifest_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0].source_label == "positive"


def test_catalog_exposes_fast_inventory_api():
    assert callable(getattr(catalog, "build_mobvoi_hard_negative_manifests", None))
    assert callable(getattr(catalog, "fast_inventory_sources", None))


def test_fast_inventory_cli_accepts_official_mobvoi_roots():
    args = build_parser().parse_args(
        [
            "fast-inventory",
            "--mobvoi-resource-root",
            "resources",
            "--mobvoi-audio-root",
            "mobvoi-wavs",
            "--target-keyword-id",
            "0",
            "--tau-root",
            "tau",
            "--kindgarden-root",
            "kindgarden",
            "--livingroom-root",
            "livingroom",
            "--pub-root",
            "pub",
            "--road-root",
            "road",
            "--wind-root",
            "wind",
            "--false-wake-root",
            "false-wake",
            "--rir-root",
            "rirs",
            "--output-root",
            "output",
        ]
    )

    assert args.command == "fast-inventory"
    assert args.target_keyword_id == 0


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(160), 16000, subtype="PCM_16")


def _write_mobvoi_resources(resource_root: Path, audio_root: Path) -> None:
    split_entries = {
        "train": (
            [
                {"utt_id": "target-train-a", "keyword_id": 0},
                {"utt_id": "target-train-b", "keyword_id": 0},
                {"utt_id": "other-train", "keyword_id": 1},
            ],
            [{"utt_id": "nonhotword-train", "keyword_id": -1}],
        ),
        "dev": (
            [
                {"utt_id": "target-dev", "keyword_id": 0},
                {"utt_id": "other-dev", "keyword_id": 1},
            ],
            [{"utt_id": "nonhotword-dev", "keyword_id": -1}],
        ),
        "test": (
            [
                {"utt_id": "target-test", "keyword_id": 0},
                {"utt_id": "other-test", "keyword_id": 1},
            ],
            [{"utt_id": "nonhotword-test", "keyword_id": -1}],
        ),
    }
    resource_root.mkdir(parents=True)
    for split, (positives, negatives) in split_entries.items():
        (resource_root / f"p_{split}.json").write_text(json.dumps(positives), encoding="utf-8")
        (resource_root / f"n_{split}.json").write_text(json.dumps(negatives), encoding="utf-8")
        for item in (*positives, *negatives):
            _write_wav(audio_root / f"{item['utt_id']}.wav")


def test_mobvoi_official_resources_build_deterministic_balanced_hard_negative_manifests(tmp_path):
    resources = tmp_path / "resources"
    audio_root = tmp_path / "mobvoi_wavs"
    output_root = tmp_path / "output"
    _write_mobvoi_resources(resources, audio_root)

    manifests = catalog.build_mobvoi_hard_negative_manifests(resources, audio_root, output_root, target_keyword_id=0)
    first_train = manifests["train"].read_text(encoding="utf-8")
    catalog.build_mobvoi_hard_negative_manifests(resources, audio_root, output_root, target_keyword_id=0)
    train_rows = [json.loads(line) for line in first_train.splitlines()]

    assert first_train == manifests["train"].read_text(encoding="utf-8")
    assert len(train_rows) == 4
    assert sum(row["command"] == "positive" for row in train_rows) == 2
    assert sum(row["command"] == "negative" for row in train_rows) == 2
    assert {Path(row["audio_filepath"]).name for row in train_rows if row["command"] == "negative"} == {
        "other-train.wav",
        "nonhotword-train.wav",
    }
    assert all(Path(row["audio_filepath"]).is_absolute() for row in train_rows)


def test_fast_inventory_uses_generated_mobvoi_manifests_without_source_hashes(tmp_path):
    resources = tmp_path / "resources"
    mobvoi_root = tmp_path / "mobvoi_wavs"
    _write_mobvoi_resources(resources, mobvoi_root)

    tau_root = tmp_path / "tau"
    for scene in TAU_SCENES:
        _write_wav(tau_root / scene / f"{scene}.wav")
    captured_roots = {}
    for scene in CAPTURED_SCENES:
        root = tmp_path / "captured" / scene
        _write_wav(root / f"{scene}.wav")
        captured_roots[scene] = root
    false_wake_root = tmp_path / "false_wake"
    _write_wav(false_wake_root / "recording.wav")
    rir_root = tmp_path / "rirs"
    rir_path = rir_root / "rir-000.wav"
    _write_wav(rir_path)
    rir_hash = "a" * 64
    (rir_root / "rir_catalog.json").write_text(
        json.dumps({"items": [{"normalized_path": str(rir_path), "normalized_sha256": rir_hash}]}),
        encoding="utf-8",
    )

    report = catalog.fast_inventory_sources(
        mobvoi_resource_root=resources,
        mobvoi_audio_root=mobvoi_root,
        target_keyword_id=0,
        tau_root=tau_root,
        captured_scene_roots=captured_roots,
        false_wake_root=false_wake_root,
        rir_root=rir_root,
        output_root=tmp_path / "output",
        require_free_gib=0,
    )

    assert report["noise_scene_count"] == 15
    assert all("sha256" not in row for row in report["files"])
    rir_rows = [row for row in report["files"] if row["role"] == "rir"]
    assert rir_rows == [
        {
            "role": "rir",
            "path": str(rir_path.resolve()),
            "already_normalized": True,
            "normalized_sha256": rir_hash,
        }
    ]
    assert (tmp_path / "output" / "inventory.json").is_file()
    assert (tmp_path / "output" / "source_manifests" / "train_manifest.json").is_file()
