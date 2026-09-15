from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import soundfile as sf
import torch

from dscnn_kws.data.dataset import DistributedBlockBatchSampler, PackedTrainingDataset, build_dataloaders
from dscnn_kws.data.packed_mixture import (
    CompositePackedDataset,
    StratifiedCompositeBatchSampler,
    V3_ROLE_QUOTAS_PER_TWENTY,
)
from dscnn_kws.data.packed_pcm import PackedPcmShardWriter
import dscnn_kws.train as train_module


CLASS_LIST = ["positive", "negative"]
CLASS_ENCODING = {label: index for index, label in enumerate(CLASS_LIST)}


def _write_manifest(path: Path, audio_path: Path, command: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"audio_filepath": str(audio_path), "command": command}) + "\n", encoding="utf-8")
    return path


def _packed_index(root: Path, records: int = 32) -> Path:
    writer = PackedPcmShardWriter(root, sample_rate=16_000, shard_records=8)
    for index in range(records):
        writer.add(torch.full((1, 16_000), index / 100.0), label=index % 2, jitter_ms=200 if index % 2 == 0 else 0)
    return writer.finalize(source_manifest="/immutable/train.jsonl")


def _single_label_packed_index(root: Path, *, label: int, records: int = 16) -> Path:
    writer = PackedPcmShardWriter(root, sample_rate=16_000, shard_records=32)
    for index in range(records):
        writer.add(torch.full((1, 16_000), index / 1_000.0), label=label, jitter_ms=200 if label == 0 else 0)
    return writer.finalize(source_manifest=f"/immutable/{root.name}.jsonl")


def test_block_batch_sampler_is_rank_disjoint_and_every_batch_is_physically_contiguous():
    samplers = [
        DistributedBlockBatchSampler(dataset_size=64, batch_size=4, block_records=16, rank=rank, world_size=4, seed=7)
        for rank in range(4)
    ]
    for sampler in samplers:
        sampler.set_epoch(3)

    batches_by_rank = [list(sampler) for sampler in samplers]
    indices_by_rank = [{index for batch in batches for index in batch} for batches in batches_by_rank]

    assert all(len(batches) == 4 for batches in batches_by_rank)
    assert all(max(batch) // 16 == min(batch) // 16 for batches in batches_by_rank for batch in batches)
    assert all(indices_by_rank[left].isdisjoint(indices_by_rank[right]) for left in range(4) for right in range(left + 1, 4))
    assert set.union(*indices_by_rank) == set(range(64))


def test_packed_train_loader_enforces_one_worker_and_one_prefetch_batch(tmp_path):
    audio_path = tmp_path / "held_out.wav"
    sf.write(audio_path, [0.0], 16_000, subtype="PCM_16")
    train_loader, validation_loader, test_loader = build_dataloaders(
        str(tmp_path / "dataset"),
        CLASS_LIST,
        CLASS_ENCODING,
        SimpleNamespace(
            packed_train_index=str(_packed_index(tmp_path / "packed")),
            packed_block_records=8,
            train_manifest=str(_write_manifest(tmp_path / "train.jsonl", audio_path, "positive")),
            validation_manifest=str(_write_manifest(tmp_path / "validation.jsonl", audio_path, "negative")),
            test_manifest=str(_write_manifest(tmp_path / "test.jsonl", audio_path, "positive")),
            sample_rate=16_000,
            noise_aug=False,
            eval_noise_aug=False,
            num_workers=0,
            prefetch_factor=8,
            batch=4,
            gpu=0,
            distributed=True,
            rank=1,
            world_size=4,
            seed=9,
        ),
    )

    assert isinstance(train_loader.dataset, PackedTrainingDataset)
    assert train_loader.num_workers == 1
    assert train_loader.prefetch_factor == 1
    assert isinstance(train_loader.batch_sampler, DistributedBlockBatchSampler)
    assert validation_loader.num_workers == 0
    assert test_loader.num_workers == 0


def test_v3_mixture_loader_enforces_the_seven_role_contract(tmp_path):
    manifests = {
        role: _single_label_packed_index(
            tmp_path / role,
            label=0 if role in {"base_positive", "raw_positive"} else 1,
        )
        for role in V3_ROLE_QUOTAS_PER_TWENTY
    }
    audio_path = tmp_path / "held_out.wav"
    sf.write(audio_path, [0.0], 16_000, subtype="PCM_16")
    train_loader, _, _ = build_dataloaders(
        str(tmp_path / "dataset"),
        CLASS_LIST,
        CLASS_ENCODING,
        SimpleNamespace(
            mixture_v3_role=[f"{role}={manifest}" for role, manifest in manifests.items()],
            mixture_base_pack="",
            mixture_raw_anchor_pack="",
            mixture_hard_negative_pack="",
            mixture_steps_per_epoch=1,
            packed_train_index="",
            train_manifest=str(_write_manifest(tmp_path / "train.jsonl", audio_path, "positive")),
            validation_manifest=str(_write_manifest(tmp_path / "validation.jsonl", audio_path, "negative")),
            test_manifest=str(_write_manifest(tmp_path / "test.jsonl", audio_path, "positive")),
            sample_rate=16_000,
            noise_aug=False,
            eval_noise_aug=False,
            num_workers=0,
            prefetch_factor=8,
            batch=20,
            gpu=0,
            distributed=False,
            rank=0,
            world_size=1,
            seed=9,
        ),
    )

    assert isinstance(train_loader.dataset, CompositePackedDataset)
    assert train_loader.dataset.quotas == V3_ROLE_QUOTAS_PER_TWENTY
    assert isinstance(train_loader.batch_sampler, StratifiedCompositeBatchSampler)
    roles = [train_loader.dataset[key][3] for key in next(iter(train_loader.batch_sampler))]
    assert Counter(roles) == V3_ROLE_QUOTAS_PER_TWENTY


def test_v3_mixture_reuses_only_negative_records_from_mixed_v2_raw_anchor_pack(tmp_path):
    manifests = {
        role: _single_label_packed_index(
            tmp_path / role,
            label=0 if role in {"base_positive", "raw_positive"} else 1,
        )
        for role in V3_ROLE_QUOTAS_PER_TWENTY
    }
    manifests["raw_negative"] = _packed_index(tmp_path / "v2_raw_anchors", records=32)

    dataset = CompositePackedDataset.from_v3_manifests(manifests)
    raw_negative = next(role for role in dataset.roles if role.name == "raw_negative")

    assert len(raw_negative.indexes) == 16
    assert all(raw_negative.dataset[index][1] == 1 for index in raw_negative.indexes)


def test_v3_mixture_reuses_read_only_raw_positive_reference_with_exclusions(tmp_path):
    manifests = {
        role: _single_label_packed_index(
            tmp_path / role,
            label=0 if role in {"base_positive", "raw_positive"} else 1,
        )
        for role in V3_ROLE_QUOTAS_PER_TWENTY
    }
    raw_anchor = _packed_index(tmp_path / "v2_raw_anchors", records=32)
    reference = tmp_path / "raw_positive.reference.json"
    reference.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "storage": "external_read_only",
                "target_label": 0,
                "external_manifest": str(raw_anchor),
                "excluded_record_indices": [2, 6],
            }
        ),
        encoding="utf-8",
    )
    manifests["raw_positive"] = reference

    dataset = CompositePackedDataset.from_v3_manifests(manifests)
    raw_positive = next(role for role in dataset.roles if role.name == "raw_positive")

    assert 2 not in raw_positive.indexes
    assert 6 not in raw_positive.indexes
    assert len(raw_positive.indexes) == 14


def test_v3_false_wake_shortfall_allows_explicit_global_replacement_only(tmp_path):
    manifests = {
        role: _single_label_packed_index(
            tmp_path / role,
            label=0 if role in {"base_positive", "raw_positive"} else 1,
        )
        for role in V3_ROLE_QUOTAS_PER_TWENTY
    }
    manifests["false_wake_hard_negative"] = _single_label_packed_index(
        tmp_path / "false_wake_hard_negative",
        label=1,
        records=3,
    )
    dataset = CompositePackedDataset.from_v3_manifests(manifests)

    with pytest.raises(ValueError, match="cannot fill one global update without duplication"):
        StratifiedCompositeBatchSampler(
            dataset,
            batch_size=20,
            rank=0,
            world_size=2,
            seed=9,
            steps_per_epoch=1,
        )

    samplers = [
        StratifiedCompositeBatchSampler(
            dataset,
            batch_size=20,
            rank=rank,
            world_size=2,
            seed=9,
            steps_per_epoch=1,
            allow_global_replacement_roles={"false_wake_hard_negative"},
        )
        for rank in range(2)
    ]
    global_batch = [key for sampler in samplers for key in next(iter(sampler))]
    role_keys = {
        role.name: [record_index for role_index, record_index in global_batch if role_index == index]
        for index, role in enumerate(dataset.roles)
    }

    assert len(role_keys["false_wake_hard_negative"]) == 4
    assert len(set(role_keys["false_wake_hard_negative"])) == 3
    assert all(
        len(record_indexes) == len(set(record_indexes))
        for name, record_indexes in role_keys.items()
        if name != "false_wake_hard_negative"
    )


def test_v3_loader_forwards_explicit_false_wake_replacement_policy(tmp_path):
    manifests = {
        role: _single_label_packed_index(
            tmp_path / role,
            label=0 if role in {"base_positive", "raw_positive"} else 1,
        )
        for role in V3_ROLE_QUOTAS_PER_TWENTY
    }
    manifests["false_wake_hard_negative"] = _single_label_packed_index(
        tmp_path / "false_wake_hard_negative",
        label=1,
        records=3,
    )
    audio_path = tmp_path / "held_out.wav"
    sf.write(audio_path, [0.0], 16_000, subtype="PCM_16")

    train_loader, _, _ = build_dataloaders(
        str(tmp_path / "dataset"),
        CLASS_LIST,
        CLASS_ENCODING,
        SimpleNamespace(
            mixture_v3_role=[f"{role}={manifest}" for role, manifest in manifests.items()],
            mixture_v3_allow_replacement_role=["false_wake_hard_negative"],
            mixture_base_pack="",
            mixture_raw_anchor_pack="",
            mixture_hard_negative_pack="",
            mixture_steps_per_epoch=1,
            packed_train_index="",
            train_manifest=str(_write_manifest(tmp_path / "train.jsonl", audio_path, "positive")),
            validation_manifest=str(_write_manifest(tmp_path / "validation.jsonl", audio_path, "negative")),
            test_manifest=str(_write_manifest(tmp_path / "test.jsonl", audio_path, "positive")),
            sample_rate=16_000,
            noise_aug=False,
            eval_noise_aug=False,
            num_workers=0,
            prefetch_factor=8,
            batch=20,
            gpu=0,
            distributed=True,
            rank=0,
            world_size=2,
            seed=9,
        ),
    )

    assert isinstance(train_loader.batch_sampler, StratifiedCompositeBatchSampler)
    assert len(next(iter(train_loader.batch_sampler))) == 20


def test_v3_loader_forwards_explicit_captured_environment_replacement_policy(tmp_path):
    manifests = {
        role: _single_label_packed_index(
            tmp_path / role,
            label=0 if role in {"base_positive", "raw_positive"} else 1,
        )
        for role in V3_ROLE_QUOTAS_PER_TWENTY
    }
    manifests["captured_environment_negative"] = _single_label_packed_index(
        tmp_path / "captured_environment_negative",
        label=1,
        records=3,
    )
    audio_path = tmp_path / "held_out.wav"
    sf.write(audio_path, [0.0], 16_000, subtype="PCM_16")

    train_loader, _, _ = build_dataloaders(
        str(tmp_path / "dataset"),
        CLASS_LIST,
        CLASS_ENCODING,
        SimpleNamespace(
            mixture_v3_role=[f"{role}={manifest}" for role, manifest in manifests.items()],
            mixture_v3_allow_replacement_role=["captured_environment_negative"],
            mixture_base_pack="",
            mixture_raw_anchor_pack="",
            mixture_hard_negative_pack="",
            mixture_steps_per_epoch=1,
            packed_train_index="",
            train_manifest=str(_write_manifest(tmp_path / "train.jsonl", audio_path, "positive")),
            validation_manifest=str(_write_manifest(tmp_path / "validation.jsonl", audio_path, "negative")),
            test_manifest=str(_write_manifest(tmp_path / "test.jsonl", audio_path, "positive")),
            sample_rate=16_000,
            noise_aug=False,
            eval_noise_aug=False,
            num_workers=0,
            prefetch_factor=8,
            batch=20,
            gpu=0,
            distributed=True,
            rank=0,
            world_size=2,
            seed=9,
        ),
    )

    assert isinstance(train_loader.batch_sampler, StratifiedCompositeBatchSampler)
    assert len(next(iter(train_loader.batch_sampler))) == 20


def test_train_parser_accepts_explicit_v3_false_wake_replacement_policy(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--mixture-v3-allow-replacement-role",
            "false_wake_hard_negative",
        ],
    )

    args = train_module.parse_args()

    assert args.mixture_v3_allow_replacement_role == ["false_wake_hard_negative"]
