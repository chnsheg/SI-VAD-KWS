import argparse

import pytest
import torch
from torch.utils.data import Dataset

from dscnn_kws.data.dataset import apply_online_window_jitter, build_train_sampler
from dscnn_kws.engine.trainer import epoch_metrics_from_totals
from dscnn_kws.train import normalize_offline_data_args


class TinyDataset(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        return index


def test_rank_samplers_are_disjoint():
    dataset = TinyDataset()
    left = build_train_sampler(dataset, rank=0, world_size=2, seed=42)
    right = build_train_sampler(dataset, rank=1, world_size=2, seed=42)

    assert set(iter(left)).isdisjoint(set(iter(right)))


def test_offline_corpus_forces_runtime_noise_off():
    args = argparse.Namespace(offline_augmented_dataset=True, noise_aug=True, eval_noise_aug=True)

    normalize_offline_data_args(args)

    assert args.noise_aug is False
    assert args.eval_noise_aug is False


def test_distributed_epoch_metrics_use_global_confusion_totals():
    metrics = epoch_metrics_from_totals(
        loss_sum=torch.tensor(2.0),
        total=torch.tensor(10.0),
        correct=torch.tensor(7.0),
        confusion=torch.tensor([[5.0, 1.0], [2.0, 2.0]]),
    )

    assert metrics.loss == pytest.approx(0.2)
    assert metrics.acc == pytest.approx(0.7)
    assert metrics.precision == pytest.approx((5 / 7 + 2 / 3) / 2)


def test_online_window_jitter_uses_a_random_offset_inside_plus_minus_200ms():
    waveform = torch.arange(16000, dtype=torch.float32).view(1, -1)
    jittered, offset = apply_online_window_jitter(
        waveform,
        sample_rate=16000,
        max_jitter_ms=200,
        generator=torch.Generator().manual_seed(7),
    )

    assert jittered.shape == (1, 16000)
    assert -3200 <= offset <= 3200
    assert offset != 0
