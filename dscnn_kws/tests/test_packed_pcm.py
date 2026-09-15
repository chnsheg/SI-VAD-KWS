from __future__ import annotations

import json

import pytest
import soundfile as sf
import torch

from dscnn_kws.data.packed_pcm import PackedPcmDataset, PackedPcmShardWriter
from dscnn_kws.data.pack_reclean_pcm import pack_training_manifest


def test_packed_pcm_round_trip_keeps_pcm16_samples_and_record_metadata(tmp_path):
    writer = PackedPcmShardWriter(tmp_path, sample_rate=16_000, shard_records=2)
    writer.add(torch.tensor([[0.0, 0.5, -0.5]], dtype=torch.float32), label=0, jitter_ms=200)
    writer.add(torch.tensor([[1.0, -1.0]], dtype=torch.float32), label=1, jitter_ms=0)

    index_path = writer.finalize(source_manifest="/data/train_manifest.jsonl")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    dataset = PackedPcmDataset(index_path)

    waveform, label = dataset[0]

    assert payload["format"] == "packed_pcm16_v1"
    assert payload["source_manifest"] == "/data/train_manifest.jsonl"
    assert payload["record_count"] == 2
    assert waveform.dtype == torch.int16
    assert waveform.shape == (1, 16_000)
    assert label == 0
    assert dataset.jitter_ms_at(0) == 200
    assert waveform[0, 1].item() / 32767.0 == pytest.approx(0.5, abs=1 / 32767)
    assert waveform[0, 2].item() / 32767.0 == pytest.approx(-0.5, abs=1 / 32767)
    assert torch.count_nonzero(waveform[:, 3:]) == 0

    second_waveform, second_label = dataset[1]
    assert second_label == 1
    assert dataset.jitter_ms_at(1) == 0
    assert second_waveform[0, 0].item() == 32767
    assert second_waveform[0, 1].item() == -32767


def test_packed_pcm_rejects_non_binary_label_without_publishing_manifest(tmp_path):
    writer = PackedPcmShardWriter(tmp_path, sample_rate=16_000, shard_records=2)

    with pytest.raises(ValueError, match="binary"):
        writer.add(torch.zeros(1, 16_000), label=2, jitter_ms=0)

    assert not (tmp_path / "manifest.json").exists()


def test_pack_training_manifest_streams_rows_and_preserves_positive_jitter(tmp_path):
    first_audio = tmp_path / "source" / "first.wav"
    second_audio = tmp_path / "source" / "second.wav"
    first_audio.parent.mkdir(parents=True)
    sf.write(first_audio, [0.0, 0.25, -0.25], 16_000, subtype="PCM_16")
    sf.write(second_audio, [0.0, -0.5, 0.5], 16_000, subtype="PCM_16")
    source_manifest = tmp_path / "train.jsonl"
    source_manifest.write_text(
        "\n".join(
            [
                json.dumps({"audio_filepath": str(first_audio), "command": "positive", "online_window_jitter_max_ms": 200}),
                json.dumps({"audio_filepath": str(second_audio), "command": "negative", "online_window_jitter_max_ms": 200}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = pack_training_manifest(source_manifest, tmp_path / "packed", expected_count=2, shard_records=1)
    dataset = PackedPcmDataset(result.index_path)
    payload = json.loads(result.index_path.read_text(encoding="utf-8"))

    assert result.record_count == 2
    assert payload["source_manifest_sha256"]
    assert dataset.jitter_ms_at(0) == 200
    assert dataset.jitter_ms_at(1) == 0


def test_pack_training_manifest_does_not_publish_manifest_when_count_is_wrong(tmp_path):
    audio_path = tmp_path / "source.wav"
    sf.write(audio_path, [0.0], 16_000, subtype="PCM_16")
    source_manifest = tmp_path / "train.jsonl"
    source_manifest.write_text(
        json.dumps({"audio_filepath": str(audio_path), "command": "positive"}) + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "packed"

    with pytest.raises(ValueError, match="expected"):
        pack_training_manifest(source_manifest, output_root, expected_count=2, shard_records=1)

    assert not (output_root / "manifest.json").exists()
