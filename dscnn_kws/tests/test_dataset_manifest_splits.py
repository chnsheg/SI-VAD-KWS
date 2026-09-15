from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

import dscnn_kws.data.dataset as dataset_module
import dscnn_kws.utils.audio as audio_module
from dscnn_kws.data.dataset import SpeechCommandDataset, build_dataloaders
from dscnn_kws.train import normalize_offline_data_args, parse_args
from dscnn_kws.utils.audio import verify_dataset_sample_rate


CLASS_LIST = ["positive", "negative", "unknown", "silence"]
CLASS_ENCODING = {label: index for index, label in enumerate(CLASS_LIST)}


def _write_wav(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    waveform = np.linspace(-0.25, 0.25, 8000, dtype=np.float32)
    sf.write(path, waveform, 8000, subtype="PCM_16")
    return path


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _dataset(manifest_path: Path, *, is_training: bool, online_window_jitter_ms: int | None = None):
    return SpeechCommandDataset(
        dataset_path=str(manifest_path.parent),
        json_filename=str(manifest_path),
        is_training=is_training,
        class_list=CLASS_LIST,
        class_encoding=CLASS_ENCODING,
        sample_rate=8000,
        noise_aug=False,
        online_window_jitter_ms=online_window_jitter_ms,
    )


def _loader_args(**overrides):
    values = {
        "train_noise_roots": None,
        "valid_noise_roots": None,
        "test_noise_roots": None,
        "noise_roots": None,
        "eval_noise_aug_prob": None,
        "eval_noise_snr_min_db": None,
        "eval_noise_snr_max_db": None,
        "noise_aug_prob": 0.0,
        "noise_snr_min_db": -5.0,
        "noise_snr_max_db": 20.0,
        "sample_rate": 8000,
        "noise_aug": False,
        "eval_noise_aug": False,
        "seed": 7,
        "allow_online_resample": False,
        "strict_sample_rate": True,
        "online_window_jitter_ms": None,
        "num_workers": 0,
        "prefetch_factor": 2,
        "batch": 1,
        "gpu": 0,
        "distributed": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_training_manifest_jitters_only_positive_exact_windows(tmp_path, monkeypatch):
    positive = _write_wav(tmp_path / "audio" / "positive.wav")
    negative = _write_wav(tmp_path / "audio" / "negative.wav")
    manifest = _write_manifest(
        tmp_path / "train.jsonl",
        [
            {"audio_filepath": str(positive), "command": "positive", "online_window_jitter_max_ms": 200},
            {"audio_filepath": str(negative), "command": "negative", "online_window_jitter_max_ms": 200},
        ],
    )
    calls = []
    original_jitter = dataset_module.apply_online_window_jitter

    def record_jitter(waveform, sample_rate, max_jitter_ms, generator=None):
        calls.append(max_jitter_ms)
        return original_jitter(waveform, sample_rate, max_jitter_ms, generator)

    monkeypatch.setattr(dataset_module, "apply_online_window_jitter", record_jitter)
    dataset = _dataset(manifest, is_training=True)

    dataset[0]
    dataset[1]

    assert calls == [200]


def test_legacy_manifest_uses_constructor_jitter_for_positive_records_only(tmp_path, monkeypatch):
    positive = _write_wav(tmp_path / "audio" / "positive.wav")
    negative = _write_wav(tmp_path / "audio" / "negative.wav")
    manifest = _write_manifest(
        tmp_path / "legacy.jsonl",
        [
            {"audio_filepath": str(positive), "command": "positive"},
            {"audio_filepath": str(negative), "command": "negative"},
        ],
    )
    calls = []
    original_jitter = dataset_module.apply_online_window_jitter

    def record_jitter(waveform, sample_rate, max_jitter_ms, generator=None):
        calls.append(max_jitter_ms)
        return original_jitter(waveform, sample_rate, max_jitter_ms, generator)

    monkeypatch.setattr(dataset_module, "apply_online_window_jitter", record_jitter)
    dataset = _dataset(manifest, is_training=True, online_window_jitter_ms=200)

    dataset[0]
    dataset[1]

    assert calls == [200]


def test_evaluation_manifest_never_applies_record_jitter(tmp_path, monkeypatch):
    positive = _write_wav(tmp_path / "audio" / "positive.wav")
    manifest = _write_manifest(
        tmp_path / "validation.jsonl",
        [{"audio_filepath": str(positive), "command": "positive", "online_window_jitter_max_ms": 200}],
    )
    calls = []

    def record_jitter(*args, **kwargs):
        calls.append(args)
        return args[0], 0

    monkeypatch.setattr(dataset_module, "apply_online_window_jitter", record_jitter)
    dataset = _dataset(manifest, is_training=False)

    dataset[0]

    assert calls == []


def test_build_dataloaders_uses_explicit_manifests_and_legacy_default_when_attrs_missing(tmp_path):
    data_path = tmp_path / "dataset"
    train_audio = _write_wav(tmp_path / "explicit" / "train.wav")
    validation_audio = _write_wav(tmp_path / "explicit" / "validation.wav")
    test_audio = _write_wav(tmp_path / "explicit" / "test.wav")
    train_manifest = _write_manifest(
        tmp_path / "explicit" / "train.jsonl", [{"audio_filepath": str(train_audio), "command": "positive"}]
    )
    validation_manifest = _write_manifest(
        tmp_path / "explicit" / "validation.jsonl", [{"audio_filepath": str(validation_audio), "command": "negative"}]
    )
    test_manifest = _write_manifest(
        tmp_path / "explicit" / "test.jsonl", [{"audio_filepath": str(test_audio), "command": "positive"}]
    )
    explicit_args = _loader_args(
        train_manifest=str(train_manifest),
        validation_manifest=str(validation_manifest),
        test_manifest=str(test_manifest),
    )

    train_loader, validation_loader, test_loader = build_dataloaders(
        str(data_path), CLASS_LIST, CLASS_ENCODING, explicit_args
    )

    assert train_loader.dataset.json_filename == str(train_manifest.resolve())
    assert validation_loader.dataset.json_filename == str(validation_manifest.resolve())
    assert test_loader.dataset.json_filename == str(test_manifest.resolve())

    default_train = _write_manifest(
        data_path / "train_manifest.json", [{"audio_filepath": str(train_audio), "command": "positive"}]
    )
    default_validation = _write_manifest(
        data_path / "validation_manifest.json", [{"audio_filepath": str(validation_audio), "command": "negative"}]
    )
    default_test = _write_manifest(
        data_path / "test_manifest.json", [{"audio_filepath": str(test_audio), "command": "positive"}]
    )
    legacy_args = _loader_args()

    train_loader, validation_loader, test_loader = build_dataloaders(
        str(data_path), CLASS_LIST, CLASS_ENCODING, legacy_args
    )

    assert train_loader.dataset.json_filename == str(default_train.resolve())
    assert validation_loader.dataset.json_filename == str(default_validation.resolve())
    assert test_loader.dataset.json_filename == str(default_test.resolve())


def test_offline_mode_keeps_global_jitter_disabled():
    args = SimpleNamespace(
        offline_augmented_dataset=True,
        noise_aug=True,
        eval_noise_aug=True,
        online_window_jitter_ms=None,
    )

    normalize_offline_data_args(args)

    assert args.noise_aug is False
    assert args.eval_noise_aug is False
    assert args.online_window_jitter_ms is None


def test_train_parser_accepts_explicit_split_manifests(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--train_manifest",
            "train.jsonl",
            "--validation_manifest",
            "validation.jsonl",
            "--test_manifest",
            "test.jsonl",
        ],
    )

    args = parse_args()

    assert (args.train_manifest, args.validation_manifest, args.test_manifest) == (
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
    )


def test_sample_rate_verification_uses_explicit_split_manifests(tmp_path):
    data_path = tmp_path / "dataset-without-default-manifests"
    train_audio = _write_wav(tmp_path / "explicit" / "train.wav")
    validation_audio = _write_wav(tmp_path / "explicit" / "validation.wav")
    test_audio = _write_wav(tmp_path / "explicit" / "test.wav")
    manifests = {
        "train": _write_manifest(
            tmp_path / "explicit" / "train.jsonl", [{"audio_filepath": str(train_audio), "command": "positive"}]
        ),
        "validation": _write_manifest(
            tmp_path / "explicit" / "validation.jsonl",
            [{"audio_filepath": str(validation_audio), "command": "negative"}],
        ),
        "test": _write_manifest(
            tmp_path / "explicit" / "test.jsonl", [{"audio_filepath": str(test_audio), "command": "positive"}],
        ),
    }

    verify_dataset_sample_rate(
        data_path=str(data_path),
        expected_sample_rate=8000,
        manifest_paths={name: str(path) for name, path in manifests.items()},
    )


def test_distributed_eval_sampler_covers_nondivisible_split_without_padding():
    dataset = range(21_282)
    sampler_type = getattr(dataset_module, "DistributedEvalSampler")
    shards = [list(sampler_type(dataset, rank=rank, world_size=4)) for rank in range(4)]

    assert sum(len(shard) for shard in shards) == 21_282
    assert sorted(index for shard in shards for index in shard) == list(range(21_282))


def test_distributed_train_loader_keeps_each_rank_tail_batch(monkeypatch):
    class TinyDataset(torch.utils.data.Dataset):
        def __init__(self, **_kwargs):
            self.items = list(range(12))

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return torch.tensor(self.items[index]), self.items[index]

    monkeypatch.setattr(dataset_module, "SpeechCommandDataset", TinyDataset)
    observed = []
    for rank in range(4):
        train_loader, _, _ = build_dataloaders(
            "unused",
            CLASS_LIST,
            CLASS_ENCODING,
            _loader_args(distributed=True, rank=rank, world_size=4, batch=2),
        )
        rank_items = [int(item) for batch, _ in train_loader for item in batch]
        assert len(rank_items) == 3
        observed.extend(rank_items)

    assert sorted(observed) == list(range(12))


@pytest.mark.parametrize(
    ("command", "jitter_ms"),
    [("negative", 200), ("positive", 0)],
)
def test_training_preserves_exact_windows_without_positive_jitter(tmp_path, monkeypatch, command, jitter_ms):
    audio_path = _write_wav(tmp_path / "audio" / f"{command}.wav")
    manifest = _write_manifest(
        tmp_path / "train.jsonl",
        [
            {
                "audio_filepath": str(audio_path),
                "command": command,
                "online_window_jitter_max_ms": jitter_ms,
            }
        ],
    )
    expected, _ = dataset_module.torchaudio.load(str(audio_path))

    def unexpected_crop(*args, **kwargs):
        raise AssertionError("exact non-jittered training windows must not be randomly cropped")

    monkeypatch.setattr(dataset_module.torch, "randint", unexpected_crop)
    waveform, _ = _dataset(manifest, is_training=True)[0]

    assert torch.equal(waveform, expected)


def test_sample_rate_verification_reservoir_samples_jsonl_without_retaining_full_manifest(tmp_path, monkeypatch):
    audio_paths = [_write_wav(tmp_path / "audio" / f"{index}.wav") for index in range(32)]
    manifest = _write_manifest(
        tmp_path / "train.jsonl",
        [{"audio_filepath": str(path), "command": "positive"} for path in audio_paths],
    )
    real_json_loads = audio_module.json.loads
    real_info = audio_module.torchaudio.info

    class TrackingRow(dict):
        live = 0
        max_live = 0

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            type(self).live += 1
            type(self).max_live = max(type(self).max_live, type(self).live)

        def __del__(self):
            type(self).live -= 1

    def tracked_loads(line):
        return TrackingRow(real_json_loads(line))

    inspected_paths = []

    def tracked_info(audio_path):
        inspected_paths.append(audio_path)
        return real_info(audio_path)

    monkeypatch.setattr(audio_module.json, "loads", tracked_loads)
    monkeypatch.setattr(audio_module.torchaudio, "info", tracked_info)
    manifest_paths = {"train": str(manifest), "validation": str(manifest), "test": str(manifest)}

    verify_dataset_sample_rate(
        data_path=str(tmp_path / "dataset"),
        expected_sample_rate=8000,
        sample_per_split=3,
        random_seed=17,
        manifest_paths=manifest_paths,
    )
    first_run_paths = list(inspected_paths)
    first_run_max_live = TrackingRow.max_live
    gc.collect()
    TrackingRow.max_live = 0
    inspected_paths.clear()
    verify_dataset_sample_rate(
        data_path=str(tmp_path / "dataset"),
        expected_sample_rate=8000,
        sample_per_split=3,
        random_seed=17,
        manifest_paths=manifest_paths,
    )

    assert first_run_max_live <= 4
    assert TrackingRow.max_live <= 4
    assert len(first_run_paths) == 9
    assert inspected_paths == first_run_paths


def test_build_dataloaders_supports_legacy_namespace_without_data_options(tmp_path):
    data_path = tmp_path / "dataset"
    train_audio = _write_wav(data_path / "audio" / "train.wav")
    validation_audio = _write_wav(data_path / "audio" / "validation.wav")
    test_audio = _write_wav(data_path / "audio" / "test.wav")
    _write_manifest(
        data_path / "train_manifest.json",
        [{"audio_filepath": str(train_audio), "command": "positive"}],
    )
    _write_manifest(
        data_path / "validation_manifest.json",
        [{"audio_filepath": str(validation_audio), "command": "negative"}],
    )
    _write_manifest(
        data_path / "test_manifest.json",
        [{"audio_filepath": str(test_audio), "command": "positive"}],
    )

    train_loader, validation_loader, test_loader = build_dataloaders(
        str(data_path), CLASS_LIST, CLASS_ENCODING, SimpleNamespace()
    )

    assert train_loader.batch_size == 256
    assert train_loader.num_workers == 0
    assert train_loader.dataset.noise_aug is True
    assert validation_loader.dataset.noise_aug is False
    assert test_loader.dataset.noise_aug is False


def test_offline_mode_clears_preexisting_global_jitter():
    args = SimpleNamespace(
        offline_augmented_dataset=True,
        noise_aug=True,
        eval_noise_aug=True,
        online_window_jitter_ms=200,
    )

    normalize_offline_data_args(args)

    assert args.online_window_jitter_ms is None
