import json

from dscnn_kws.data.reclean.validate import build_dataset_manifests, validate_dataset, validate_manifest_quotas


def test_validator_rejects_wrong_audio_format(tmp_path):
    report = validate_dataset(tmp_path, expected_sample_rate=16000, expected_frames=16000)

    assert not report.ok


def test_validator_accepts_balanced_quota_manifest(tmp_path):
    metadata = tmp_path / "metadata.jsonl"
    rows = (
        [{"example_id": f"p-{index}", "label": "positive", "recipe": {"source_kind": "speech"}} for index in range(10)]
        + [{"example_id": f"s-{index}", "label": "negative", "recipe": {"source_kind": "speech"}} for index in range(7)]
        + [{"example_id": "f-0", "label": "negative", "recipe": {"source_kind": "false_wake"}}]
        + [{"example_id": f"n-{index}", "label": "negative", "recipe": {"source_kind": "pure_noise"}} for index in range(2)]
    )
    metadata.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report = validate_manifest_quotas(metadata, expected_total=20)

    assert report.ok


def test_dataset_manifest_builder_uses_generated_train_and_prepared_evaluation_sources(tmp_path):
    output = tmp_path / "dataset"
    metadata = output / "metadata" / "shard-00.jsonl"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        json.dumps({"example_id": "train-0", "label": "positive", "output_path": str(output / "audio" / "train-0.wav")}) + "\n",
        encoding="utf-8",
    )
    prepared = [
        {"source_kind": "speech", "source_split": "validation", "source_label": "negative", "prepared_path": "/tmp/valid.wav"},
        {"source_kind": "speech", "source_split": "test", "source_label": "positive", "prepared_path": "/tmp/test.wav"},
    ]

    counts = build_dataset_manifests(output, prepared)

    assert counts == {"train": 1, "validation": 1, "test": 1}
    assert '"command": "positive"' in (output / "train_manifest.json").read_text(encoding="utf-8")
