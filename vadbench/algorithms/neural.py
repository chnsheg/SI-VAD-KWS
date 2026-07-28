from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from vadbench.algorithms.base import VADAlgorithm
from vadbench.audio import load_audio
from vadbench.features import align_length, log_mel_spectrogram, mfcc_features
from vadbench.frame_prediction import FramePrediction
from vadbench.manifest import ManifestRecord
from vadbench.metrics import choose_best_threshold


class TinyMelCNN(nn.Module):
    def __init__(self, n_mels: int = 64, channels: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_mels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(channels, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x.transpose(1, 2)).squeeze(1)
        return y


class TimeChannelSeparableBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, groups=channels),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, groups=channels),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if y.shape[-1] != x.shape[-1]:
            y = y[..., : x.shape[-1]]
        return x + y


class MarbleNetLite(nn.Module):
    def __init__(self, n_mels: int = 64, channels: int = 64) -> None:
        super().__init__()
        self.prologue = nn.Sequential(nn.Conv1d(n_mels, channels, kernel_size=11, padding=5), nn.BatchNorm1d(channels), nn.ReLU())
        self.blocks = nn.Sequential(
            TimeChannelSeparableBlock(channels, 13),
            TimeChannelSeparableBlock(channels, 15),
            TimeChannelSeparableBlock(channels, 17),
            TimeChannelSeparableBlock(channels, 29, dilation=2),
        )
        self.epilogue = nn.Sequential(
            nn.Conv1d(channels, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(128, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.prologue(x.transpose(1, 2))
        y = self.blocks(y)
        return self.epilogue(y).squeeze(1)


class MarbleNet3x2x64(nn.Module):
    def __init__(self, n_mels: int = 64, channels: int = 64) -> None:
        super().__init__()
        self.prologue = nn.Sequential(nn.Conv1d(n_mels, channels, kernel_size=11, padding=5), nn.BatchNorm1d(channels), nn.ReLU())
        self.blocks = nn.Sequential(
            TimeChannelSeparableBlock(channels, 13),
            TimeChannelSeparableBlock(channels, 13),
            TimeChannelSeparableBlock(channels, 15),
            TimeChannelSeparableBlock(channels, 15),
            TimeChannelSeparableBlock(channels, 17),
            TimeChannelSeparableBlock(channels, 17),
        )
        self.epilogue = nn.Sequential(
            nn.Conv1d(channels, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(128, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(128, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.prologue(x.transpose(1, 2))
        y = self.blocks(y)
        return self.epilogue(y).squeeze(1)


class CNNTDLike(nn.Module):
    def __init__(self, n_mels: int = 64, channels: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_mels, channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Conv1d(channels, channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels * 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Conv1d(channels * 2, channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels * 2),
            nn.ReLU(),
            nn.Conv1d(channels * 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).squeeze(1)


class CRNNVAD(nn.Module):
    def __init__(self, n_mels: int = 64, channels: int = 64) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(channels, channels, num_layers=1, batch_first=True, bidirectional=True)
        self.out = nn.Sequential(nn.Dropout(0.15), nn.Linear(channels * 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        y, _ = self.rnn(y)
        return self.out(y).squeeze(-1)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, groups: int = 1, bias: bool = False) -> None:
        super().__init__()
        self.time_pad = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.time_pad, 0)))


class CausalDSConv1dBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.08) -> None:
        super().__init__()
        self.depthwise = CausalConv1d(channels, channels, kernel_size=kernel_size, groups=channels, bias=False)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.act(self.norm(self.pointwise(self.depthwise(x)))))


class CausalCRNNVAD(nn.Module):
    def __init__(self, n_mels: int = 64, channels: int = 32, hidden_size: int = 40, dropout: float = 0.12) -> None:
        super().__init__()
        self.frontend = nn.Sequential(
            CausalConv1d(n_mels, channels, kernel_size=5, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            CausalDSConv1dBlock(channels, kernel_size=5, dropout=dropout),
            CausalDSConv1dBlock(channels, kernel_size=5, dropout=dropout),
        )
        self.rnn = nn.GRU(channels, hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.frontend(x.transpose(1, 2)).transpose(1, 2)
        y, _ = self.rnn(y)
        return self.out(y).squeeze(-1)


class CausalCRNNVADMicro(CausalCRNNVAD):
    def __init__(self, n_mels: int = 64) -> None:
        super().__init__(n_mels=n_mels, channels=24, hidden_size=20, dropout=0.10)


class CausalCRNNVADNano(CausalCRNNVAD):
    def __init__(self, n_mels: int = 64) -> None:
        super().__init__(n_mels=n_mels, channels=16, hidden_size=12, dropout=0.08)


class CausalCRNNVADTiny(CausalCRNNVAD):
    def __init__(self, n_mels: int = 64) -> None:
        super().__init__(n_mels=n_mels, channels=20, hidden_size=16, dropout=0.09)


class CausalCRNNVADKWS(CausalCRNNVAD):
    def __init__(self, n_mels: int = 64) -> None:
        super().__init__(n_mels=n_mels, channels=32, hidden_size=40, dropout=0.12)


class SelfAttentiveVAD(nn.Module):
    def __init__(self, n_mels: int = 64, channels: int = 96, heads: int = 4) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Conv1d(n_mels, channels, kernel_size=5, padding=2), nn.BatchNorm1d(channels), nn.ReLU())
        self.attn1 = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(channels, channels * 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(channels * 2, channels))
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.out = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.input(x.transpose(1, 2)).transpose(1, 2)
        attn, _ = self.attn1(y, y, y, need_weights=False)
        y = self.norm1(y + attn)
        y = self.norm2(y + self.ffn(y))
        return self.out(y).squeeze(-1)


class CausalConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.time_pad = kernel_size[0] - 1
        self.freq_pad_left = kernel_size[1] // 2
        self.freq_pad_right = kernel_size[1] - 1 - self.freq_pad_left
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.freq_pad_left, self.freq_pad_right, self.time_pad, 0))
        return self.conv(x)


class DSCNN2DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int], freq_stride: int = 1, dropout: float = 0.05) -> None:
        super().__init__()
        self.depthwise = CausalConv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=(1, freq_stride),
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)
        self.use_residual = in_channels == out_channels and freq_stride == 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(x)
        y = self.pointwise(y)
        y = self.dropout(self.act(self.norm(y)))
        if self.use_residual and y.shape == x.shape:
            y = y + x
        return y


class DSCNNVAD(nn.Module):
    def __init__(self, n_mels: int = 64, channels: int = 12) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            CausalConv2d(1, channels, kernel_size=(5, 3), stride=(1, 2), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            DSCNN2DBlock(channels, channels, kernel_size=(5, 3), freq_stride=1),
            DSCNN2DBlock(channels, channels + 4, kernel_size=(5, 3), freq_stride=2),
            DSCNN2DBlock(channels + 4, channels + 4, kernel_size=(5, 3), freq_stride=1),
            DSCNN2DBlock(channels + 4, channels * 2, kernel_size=(3, 3), freq_stride=2),
            DSCNN2DBlock(channels * 2, channels * 2, kernel_size=(3, 3), freq_stride=1),
        )
        self.head = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv1d(channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.unsqueeze(1)
        y = self.stem(y)
        y = self.blocks(y)
        y = y.mean(dim=-1)
        return self.head(y).squeeze(1)


class DSCNNVADTiny(DSCNNVAD):
    def __init__(self, n_mels: int = 64) -> None:
        super().__init__(n_mels=n_mels, channels=8)


class DSCNNVADSmall(DSCNNVAD):
    def __init__(self, n_mels: int = 64) -> None:
        super().__init__(n_mels=n_mels, channels=12)


class DSCNNVADMedium(DSCNNVAD):
    def __init__(self, n_mels: int = 64) -> None:
        super().__init__(n_mels=n_mels, channels=16)


class DSCNNVADLarge(DSCNNVAD):
    def __init__(self, n_mels: int = 64) -> None:
        super().__init__(n_mels=n_mels, channels=24)


class DSCNNVADKWSMatch(DSCNNVAD):
    def __init__(self, n_mels: int = 64) -> None:
        super().__init__(n_mels=n_mels, channels=40)


class CausalDSCNNGRUVADKWS(nn.Module):
    def __init__(self, n_mels: int = 64, channels: int = 24, hidden_size: int = 48) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            CausalConv2d(1, channels, kernel_size=(5, 3), stride=(1, 2), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            DSCNN2DBlock(channels, channels, kernel_size=(5, 3), freq_stride=1),
            DSCNN2DBlock(channels, channels + 4, kernel_size=(5, 3), freq_stride=2),
            DSCNN2DBlock(channels + 4, channels + 4, kernel_size=(5, 3), freq_stride=1),
            DSCNN2DBlock(channels + 4, channels * 2, kernel_size=(3, 3), freq_stride=2),
            DSCNN2DBlock(channels * 2, channels * 2, kernel_size=(3, 3), freq_stride=1),
        )
        self.rnn = nn.GRU(channels * 2, hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.out = nn.Sequential(nn.Dropout(0.08), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem(x.unsqueeze(1))
        y = self.blocks(y).mean(dim=-1).transpose(1, 2)
        y, _ = self.rnn(y)
        return self.out(y).squeeze(-1)


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        padding = dilation * 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if y.shape[-1] != x.shape[-1]:
            y = y[..., : x.shape[-1]]
        return self.act(x + y)


class AttnTCNLite(nn.Module):
    def __init__(self, n_mels: int = 64, channels: int = 64, heads: int = 4) -> None:
        super().__init__()
        self.input = nn.Conv1d(n_mels, channels, kernel_size=3, padding=1)
        self.tcn = nn.Sequential(TCNBlock(channels, 1), TCNBlock(channels, 2), TCNBlock(channels, 4))
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.output = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.input(x.transpose(1, 2))
        y = self.tcn(y).transpose(1, 2)
        attn_out, _ = self.attn(y, y, y, need_weights=False)
        y = self.norm(y + attn_out)
        return self.output(y).squeeze(-1)


class ManifestFrameDataset(Dataset):
    def __init__(
        self,
        records: Sequence[ManifestRecord],
        base_dir: str | Path,
        sample_rate: int,
        n_mels: int,
        frame_ms: float,
        hop_ms: float,
        feature_type: str = "logmel",
        augment: dict | None = None,
    ) -> None:
        self.records = list(records)
        self.base_dir = Path(base_dir)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        self.feature_type = feature_type
        self.augment = augment or {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        record = self.records[idx]
        labels = np.load(record.resolve_label(self.base_dir)).astype(np.float32)
        feature_path = _resolve_feature_for_type(record, self.base_dir, self.feature_type)
        if feature_path is not None and feature_path.exists():
            features = np.load(feature_path).astype(np.float32)
        else:
            waveform, sample_rate = load_audio(record.resolve_audio(self.base_dir), self.sample_rate)
            if self.feature_type == "mfcc":
                features = mfcc_features(waveform, sample_rate, self.n_mels, self.n_mels, self.frame_ms, self.hop_ms, normalize=True)
            else:
                features = log_mel_spectrogram(waveform, sample_rate, self.n_mels, self.frame_ms, self.hop_ms, normalize=True)
        length = min(len(features), len(labels))
        features = features[:length]
        labels = labels[:length]
        segment_frames = int(self.augment.get("segment_frames", 0) or 0)
        if segment_frames > 0 and length > segment_frames:
            start = int(np.random.default_rng().integers(0, length - segment_frames + 1))
            end = start + segment_frames
            features = features[start:end]
            labels = labels[start:end]
            length = segment_frames
        features, labels = _augment_pair(features, labels, self.augment)
        mask = np.ones(length, dtype=np.float32)
        return (
            torch.from_numpy(features.astype(np.float32)),
            torch.from_numpy(labels.astype(np.float32)),
            torch.from_numpy(mask),
            record.id,
        )


def collate_frames(batch):
    max_len = max(item[0].shape[0] for item in batch)
    n_mels = batch[0][0].shape[1]
    features = torch.zeros(len(batch), max_len, n_mels, dtype=torch.float32)
    labels = torch.zeros(len(batch), max_len, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.float32)
    ids: list[str] = []
    for idx, (feat, lab, msk, item_id) in enumerate(batch):
        length = feat.shape[0]
        features[idx, :length] = feat
        labels[idx, :length] = lab
        mask[idx, :length] = msk
        ids.append(item_id)
    return features, labels, mask, ids


def _resolve_feature_for_type(record: ManifestRecord, base_dir: str | Path, feature_type: str) -> Path | None:
    feature_path = record.resolve_feature(base_dir)
    if feature_path is None:
        return None
    if feature_type == "mfcc":
        mfcc_path = feature_path.with_name(feature_path.name.replace(".logmel64.npy", ".mfcc64.npy"))
        if mfcc_path.exists():
            return mfcc_path
    if feature_type == "logmel":
        return feature_path
    return None


def _augment_pair(features: np.ndarray, labels: np.ndarray, augment: dict) -> tuple[np.ndarray, np.ndarray]:
    if not augment:
        return features.astype(np.float32), labels.astype(np.float32)
    out = np.asarray(features, dtype=np.float32).copy()
    out_labels = np.asarray(labels, dtype=np.float32).copy()
    rng = np.random.default_rng()
    if augment.get("gain_db", 0):
        gain = rng.uniform(-float(augment["gain_db"]), float(augment["gain_db"]))
        out = out + gain / 20.0
    if augment.get("time_shift_frames", 0):
        shift = int(rng.integers(-int(augment["time_shift_frames"]), int(augment["time_shift_frames"]) + 1))
        out = np.roll(out, shift, axis=0)
        out_labels = np.roll(out_labels, shift, axis=0)
    if augment.get("noise_std", 0):
        out = out + rng.normal(0.0, float(augment["noise_std"]), size=out.shape).astype(np.float32)
    freq_masks = int(augment.get("freq_masks", 0))
    time_masks = int(augment.get("time_masks", 0))
    max_freq = int(augment.get("max_freq_mask", 8))
    max_time = int(augment.get("max_time_mask", 20))
    for _ in range(freq_masks):
        width = int(rng.integers(1, max(2, max_freq + 1)))
        start = int(rng.integers(0, max(1, out.shape[1] - width + 1)))
        out[:, start : start + width] = 0.0
    for _ in range(time_masks):
        width = int(rng.integers(1, max(2, max_time + 1)))
        start = int(rng.integers(0, max(1, out.shape[0] - width + 1)))
        out[start : start + width, :] = 0.0
    return out.astype(np.float32), out_labels.astype(np.float32)


def _augment_features(features: np.ndarray, augment: dict) -> np.ndarray:
    labels = np.zeros((len(features),), dtype=np.float32)
    out, _ = _augment_pair(features, labels, augment)
    return out


def _make_record_sampler(records: Sequence[ManifestRecord], base_dir: str | Path) -> WeightedRandomSampler:
    weights: list[float] = []
    for record in records:
        try:
            labels = np.load(record.resolve_label(base_dir)).astype(np.float32)
            speech_ratio = float(np.mean(labels)) if len(labels) else 0.0
        except Exception:
            speech_ratio = 0.5
        balance = abs(speech_ratio - 0.5)
        weights.append(float(1.0 + 2.0 * balance))
    if not weights:
        weights = [1.0]
    return WeightedRandomSampler(weights, num_samples=len(records), replacement=True)


def _resolve_pos_weight(params: dict, records: Sequence[ManifestRecord], base_dir: str | Path, device: str) -> torch.Tensor | None:
    value = params.get("pos_weight", None)
    if value in (None, False):
        return None
    if value != "auto":
        return torch.tensor(float(value), dtype=torch.float32, device=device)
    positives = 0
    total = 0
    for record in records:
        labels = np.load(record.resolve_label(base_dir)).astype(np.uint8)
        positives += int(labels.sum())
        total += int(labels.size)
    negatives = max(total - positives, 1)
    positives = max(positives, 1)
    weight = float(np.clip(negatives / positives, 0.25, 4.0))
    return torch.tensor(weight, dtype=torch.float32, device=device)


class TorchFrameAlgorithm(VADAlgorithm):
    name = "torch_frame"
    requires_training = True
    model_cls = TinyMelCNN

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: float = 25.0,
        hop_ms: float = 10.0,
        n_mels: int = 64,
        device: str = "auto",
        training: dict | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(frame_hop_ms=hop_ms)
        self.sample_rate = int(sample_rate)
        self.frame_ms = float(frame_ms)
        self.hop_ms = float(hop_ms)
        self.n_mels = int(n_mels)
        self.feature_type = str(kwargs.get("feature_type", "logmel"))
        self.device_name = _resolve_device(device)
        self.training = training or {}
        self.model = self.model_cls(n_mels=self.n_mels).to(self.device_name)

    def model_stats(self, frames: int = 1) -> dict[str, float]:
        return estimate_model_stats(self.model, n_mels=self.n_mels, frames=frames)

    def fit(
        self,
        train_manifest: Sequence[ManifestRecord],
        val_manifest: Sequence[ManifestRecord],
        base_dir: str | Path,
        run_dir: str | Path | None = None,
        **kwargs: object,
    ) -> dict[str, float]:
        params = dict(self.training)
        params.update(kwargs.get("training", {}) if isinstance(kwargs.get("training"), dict) else {})
        epochs = int(params.get("epochs", 1))
        batch_size = int(params.get("batch_size", 4))
        lr = float(params.get("lr", 1e-3))
        weight_decay = float(params.get("weight_decay", 1e-5))
        num_workers = int(params.get("num_workers", 0))
        class_balanced_sampling = bool(params.get("class_balanced_sampling", False))
        early_stop_patience = int(params.get("early_stop_patience", 0))
        log_every_batches = int(params.get("log_every_batches", 0) or 0)
        max_train_batches = int(params.get("max_train_batches_per_epoch", 0) or 0)

        train_ds = ManifestFrameDataset(
            train_manifest,
            base_dir,
            self.sample_rate,
            self.n_mels,
            self.frame_ms,
            self.hop_ms,
            self.feature_type,
            params.get("augment", {}) if isinstance(params.get("augment"), dict) else {},
        )
        val_ds = ManifestFrameDataset(val_manifest, base_dir, self.sample_rate, self.n_mels, self.frame_ms, self.hop_ms, self.feature_type)
        sampler = _make_record_sampler(train_manifest, base_dir) if class_balanced_sampling else None
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_frames,
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_frames)

        optim_name = str(params.get("optimizer", "adamw")).lower()
        if optim_name == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=float(params.get("momentum", 0.9)), weight_decay=weight_decay)
        else:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = _make_scheduler(optimizer, params, epochs)
        pos_weight = _resolve_pos_weight(params, train_manifest, base_dir, self.device_name)
        criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
        best_state = copy.deepcopy(self.model.state_dict())
        best_f1 = -1.0
        best_threshold = 0.5
        epochs_without_improvement = 0
        epoch_history: list[dict[str, float]] = []
        checkpoint_path = Path(run_dir) / "checkpoints" / "best.pt" if run_dir is not None else None
        history_path = Path(run_dir) / "history.json" if run_dir is not None else None
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_frames = 0.0
            epoch_batches = len(train_loader) if max_train_batches <= 0 else min(len(train_loader), max_train_batches)
            for batch_index, (features, labels, mask, _ids) in enumerate(train_loader, start=1):
                features = features.to(self.device_name)
                labels = labels.to(self.device_name)
                mask = mask.to(self.device_name)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(features)
                loss_map = criterion(logits, labels) * mask
                loss = loss_map.sum() / mask.sum().clamp_min(1.0)
                loss.backward()
                optimizer.step()
                total_loss += float(loss_map.sum().detach().cpu())
                total_frames += float(mask.sum().detach().cpu())
                if log_every_batches > 0 and batch_index % log_every_batches == 0:
                    running_loss = total_loss / max(total_frames, 1.0)
                    print(f"epoch {epoch}/{epochs} batch {batch_index}/{epoch_batches} train_loss={running_loss:.6f}", flush=True)
                if max_train_batches > 0 and batch_index >= max_train_batches:
                    break

            val_labels, val_scores = self._predict_dataset(val_loader)
            threshold, val_metrics = choose_best_threshold(val_labels, val_scores)
            self.threshold = threshold
            train_loss = total_loss / max(total_frames, 1.0)
            history = {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_f1": float(val_metrics["f1"]),
                "val_threshold": float(threshold),
            }
            epoch_history.append(history)
            print(
                f"epoch {epoch}/{epochs} train_loss={train_loss:.6f} "
                f"val_f1={val_metrics['f1']:.4f} threshold={threshold:.3f}",
                flush=True,
            )
            if val_metrics["f1"] > best_f1:
                best_f1 = float(val_metrics["f1"])
                best_threshold = float(threshold)
                best_state = copy.deepcopy(self.model.state_dict())
                if checkpoint_path is not None:
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    self.save_checkpoint(checkpoint_path)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if history_path is not None:
                history_path.parent.mkdir(parents=True, exist_ok=True)
                with history_path.open("w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "epochs": epoch_history,
                            "best_epoch": float(max(epoch_history, key=lambda item: item["val_f1"])["epoch"]) if epoch_history else 0.0,
                            "best_val_f1": float(best_f1),
                            "best_threshold": float(best_threshold),
                        },
                        handle,
                        indent=2,
                    )
            if scheduler is not None:
                scheduler.step()
            if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
                print(f"early stopping after {epoch} epochs", flush=True)
                break
        self.model.load_state_dict(best_state)
        self.threshold = best_threshold
        return {
            "epochs": epoch_history,
            "best_epoch": float(max(epoch_history, key=lambda item: item["val_f1"])["epoch"]) if epoch_history else 0.0,
            "best_val_f1": float(best_f1),
            "best_threshold": float(best_threshold),
        }

    def _predict_dataset(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        labels_all: list[np.ndarray] = []
        scores_all: list[np.ndarray] = []
        with torch.no_grad():
            for features, labels, mask, _ids in loader:
                features = features.to(self.device_name)
                logits = self.model(features)
                scores = torch.sigmoid(logits).detach().cpu().numpy()
                mask_np = mask.numpy().astype(bool)
                labels_np = labels.numpy()
                for idx in range(scores.shape[0]):
                    valid = mask_np[idx]
                    labels_all.append(labels_np[idx][valid])
                    scores_all.append(scores[idx][valid])
        if not labels_all:
            return np.array([], dtype=np.uint8), np.array([], dtype=np.float32)
        return np.concatenate(labels_all).astype(np.uint8), np.concatenate(scores_all).astype(np.float32)

    def predict(self, waveform: np.ndarray, sample_rate: int, source_id: str | None = None) -> FramePrediction:
        if self.feature_type == "mfcc":
            features = mfcc_features(waveform, sample_rate, self.n_mels, self.n_mels, self.frame_ms, self.hop_ms, normalize=True)
        else:
            features = log_mel_spectrogram(waveform, sample_rate, self.n_mels, self.frame_ms, self.hop_ms, normalize=True)
        return self.predict_features(features, source_id=source_id)

    def predict_features(self, features: np.ndarray, source_id: str | None = None) -> FramePrediction:
        tensor = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(self.device_name)
        self.model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(self.model(tensor)).squeeze(0).detach().cpu().numpy()
        scores = align_length(scores, len(features), pad_value=0.0)
        return FramePrediction(scores=scores, frame_hop_ms=self.hop_ms, source_id=source_id)

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": self.name,
                "state_dict": self.model.state_dict(),
                "threshold": self.threshold,
                "sample_rate": self.sample_rate,
                "frame_ms": self.frame_ms,
                "hop_ms": self.hop_ms,
                "n_mels": self.n_mels,
                "feature_type": self.feature_type,
                "model_stats": self.model_stats(frames=1),
            },
            str(path),
        )

    def load_checkpoint(self, path: str | Path) -> None:
        try:
            checkpoint = torch.load(str(path), map_location=self.device_name, weights_only=True)
        except TypeError:
            checkpoint = torch.load(str(path), map_location=self.device_name)
        if "state_dict" not in checkpoint:
            raise ValueError(f"Invalid checkpoint: {path}")
        self.sample_rate = int(checkpoint.get("sample_rate", self.sample_rate))
        self.frame_ms = float(checkpoint.get("frame_ms", self.frame_ms))
        self.hop_ms = float(checkpoint.get("hop_ms", self.hop_ms))
        self.n_mels = int(checkpoint.get("n_mels", self.n_mels))
        self.feature_type = str(checkpoint.get("feature_type", self.feature_type))
        self.threshold = checkpoint.get("threshold", self.threshold)
        self.model = self.model_cls(n_mels=self.n_mels).to(self.device_name)
        self.model.load_state_dict(checkpoint["state_dict"])


class TinyMelCNNAlgorithm(TorchFrameAlgorithm):
    name = "tiny_mel_cnn"
    model_cls = TinyMelCNN


class MarbleNetLiteAlgorithm(TorchFrameAlgorithm):
    name = "marblenet_lite"
    model_cls = MarbleNetLite


class AttnTCNLiteAlgorithm(TorchFrameAlgorithm):
    name = "attn_tcn_lite"
    model_cls = AttnTCNLite


class MarbleNet3x2x64Algorithm(TorchFrameAlgorithm):
    name = "marblenet_3x2x64"
    model_cls = MarbleNet3x2x64


class CNNTDLikeAlgorithm(TorchFrameAlgorithm):
    name = "cnn_td_like"
    model_cls = CNNTDLike


class CRNNVADAlgorithm(TorchFrameAlgorithm):
    name = "crnn_vad"
    model_cls = CRNNVAD


class CausalCRNNVADMicroAlgorithm(TorchFrameAlgorithm):
    name = "causal_crnn_vad_micro"
    model_cls = CausalCRNNVADMicro


class CausalCRNNVADNanoAlgorithm(TorchFrameAlgorithm):
    name = "causal_crnn_vad_nano"
    model_cls = CausalCRNNVADNano


class CausalCRNNVADTinyAlgorithm(TorchFrameAlgorithm):
    name = "causal_crnn_vad_tiny"
    model_cls = CausalCRNNVADTiny


class CausalCRNNVADKWSAlgorithm(TorchFrameAlgorithm):
    name = "causal_crnn_vad_kws"
    model_cls = CausalCRNNVADKWS


class SelfAttentiveVADAlgorithm(TorchFrameAlgorithm):
    name = "self_attentive_vad"
    model_cls = SelfAttentiveVAD


class DSCNNVADSmallAlgorithm(TorchFrameAlgorithm):
    name = "dscnn_vad_small"
    model_cls = DSCNNVADSmall


class DSCNNVADTinyAlgorithm(TorchFrameAlgorithm):
    name = "dscnn_vad_tiny"
    model_cls = DSCNNVADTiny


class DSCNNVADMediumAlgorithm(TorchFrameAlgorithm):
    name = "dscnn_vad_medium"
    model_cls = DSCNNVADMedium


class DSCNNVADLargeAlgorithm(TorchFrameAlgorithm):
    name = "dscnn_vad_large"
    model_cls = DSCNNVADLarge


class DSCNNVADKWSMatchAlgorithm(TorchFrameAlgorithm):
    name = "dscnn_vad_kws_match"
    model_cls = DSCNNVADKWSMatch


class CausalDSCNNGRUVADKWSAlgorithm(TorchFrameAlgorithm):
    name = "causal_dscnn_gru_vad_kws"
    model_cls = CausalDSCNNGRUVADKWS


def estimate_model_stats(model: nn.Module, n_mels: int = 64, frames: int = 1) -> dict[str, float]:
    params = sum(param.numel() for param in model.parameters())
    macs_total = 0
    hooks = []

    def conv1d_hook(module: nn.Conv1d, _inputs, output) -> None:
        nonlocal macs_total
        batch, out_channels, out_len = output.shape
        kernel = module.kernel_size[0]
        in_channels = module.in_channels // module.groups
        macs_total += int(batch * out_channels * out_len * in_channels * kernel)

    def conv2d_hook(module: nn.Conv2d, _inputs, output) -> None:
        nonlocal macs_total
        batch, out_channels, out_h, out_w = output.shape
        kh, kw = module.kernel_size
        in_channels = module.in_channels // module.groups
        macs_total += int(batch * out_channels * out_h * out_w * in_channels * kh * kw)

    def linear_hook(module: nn.Linear, _inputs, output) -> None:
        nonlocal macs_total
        macs_total += int(output.numel() * module.in_features)

    def gru_hook(module: nn.GRU, inputs, _output) -> None:
        nonlocal macs_total
        x = inputs[0]
        batch = int(x.shape[0])
        seq = int(x.shape[1]) if module.batch_first else int(x.shape[0])
        directions = 2 if module.bidirectional else 1
        input_size = module.input_size
        hidden_size = module.hidden_size
        for _layer in range(module.num_layers):
            macs_total += batch * seq * directions * 3 * hidden_size * (input_size + hidden_size)
            input_size = hidden_size * directions

    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            hooks.append(module.register_forward_hook(conv1d_hook))
        elif isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv2d_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, nn.GRU):
            hooks.append(module.register_forward_hook(gru_hook))
    was_training = model.training
    device = next(model.parameters(), torch.zeros(1)).device
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, max(1, frames), n_mels, dtype=torch.float32, device=device))
    for hook in hooks:
        hook.remove()
    if was_training:
        model.train()
    return {
        "params": float(params),
        "macs_total": float(macs_total),
        "macs_per_frame": float(macs_total / max(1, frames)),
        "profile_frames": float(max(1, frames)),
    }


def _make_scheduler(optimizer: torch.optim.Optimizer, params: dict, epochs: int):
    name = str(params.get("scheduler", "none")).lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=float(params.get("min_lr", 1e-5)))
    if name in {"warmup_hold_decay", "whd"}:
        warmup = max(1, int(round(epochs * float(params.get("warmup_ratio", 0.05)))))
        hold = max(warmup, int(round(epochs * (float(params.get("warmup_ratio", 0.05)) + float(params.get("hold_ratio", 0.45))))))
        min_lr = float(params.get("min_lr", 1e-4))
        base_lrs = [group["lr"] for group in optimizer.param_groups]

        def lr_lambda(epoch: int) -> float:
            step = epoch + 1
            if step <= warmup:
                return step / warmup
            if step <= hold:
                return 1.0
            progress = min(1.0, (step - hold) / max(1, epochs - hold))
            return max(min_lr / max(base_lrs[0], 1e-12), (1.0 - progress) ** 2)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return None


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device
