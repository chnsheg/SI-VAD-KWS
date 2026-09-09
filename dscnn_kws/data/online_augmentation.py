"""On-the-fly waveform mixing and role-balanced KWS sampling.

The module deliberately stores only source paths.  Noise windows are decoded and
mixed when a sample is requested, so an experiment does not materialize a large
augmented corpus on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
from typing import Collection, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import Dataset, Sampler


ONLINE_AUGMENT_PROFILE = "nihao_wenwen_v1"
SUPPORTED_SAMPLE_ROLES = ("positive", "negative", "hard_negative")


@dataclass(frozen=True)
class SnrBand:
    low_db: float
    high_db: float
    weight: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.low_db, self.high_db, self.weight)):
            raise ValueError("SNR band values must be finite")
        if self.low_db > self.high_db:
            raise ValueError("SNR band low_db must not exceed high_db")
        if self.weight <= 0.0:
            raise ValueError("SNR band weight must be positive")


class SnrDistribution:
    """Weighted piecewise-uniform SNR distribution."""

    def __init__(self, bands: Sequence[SnrBand]):
        if not bands:
            raise ValueError("At least one SNR band is required")
        self.bands = tuple(bands)
        self._weights = tuple(band.weight for band in self.bands)

    def sample(self, rng: random.Random) -> float:
        band = rng.choices(self.bands, weights=self._weights, k=1)[0]
        return rng.uniform(band.low_db, band.high_db)


@dataclass(frozen=True)
class RoleAugmentationRule:
    mix_probability: float
    snr: SnrDistribution

    def __post_init__(self) -> None:
        if not math.isfinite(self.mix_probability) or not 0.0 <= self.mix_probability <= 1.0:
            raise ValueError("mix_probability must be in [0, 1]")


def default_nihao_wenwen_rules() -> dict[str, RoleAugmentationRule]:
    """Training policy emphasizing the -15 dB deployment boundary.

    Hard negatives retain more clean presentations than ordinary examples so
    their phonetic boundary is not hidden by noise on every visit.
    """

    return {
        "positive": RoleAugmentationRule(
            0.90,
            SnrDistribution(
                (
                    SnrBand(-15.0, -10.0, 0.35),
                    SnrBand(-10.0, 0.0, 0.30),
                    SnrBand(0.0, 10.0, 0.20),
                    SnrBand(10.0, 25.0, 0.15),
                )
            ),
        ),
        "negative": RoleAugmentationRule(
            0.65,
            SnrDistribution((SnrBand(-15.0, 0.0, 0.40), SnrBand(0.0, 20.0, 0.60))),
        ),
        "hard_negative": RoleAugmentationRule(
            0.35,
            SnrDistribution((SnrBand(-10.0, 5.0, 0.35), SnrBand(5.0, 20.0, 0.65))),
        ),
    }


def parse_snr_bands(entries: Sequence[str] | None, fallback: SnrDistribution) -> SnrDistribution:
    """Parse repeatable ``LOW:HIGH:WEIGHT`` command-line values."""

    if not entries:
        return fallback
    bands: list[SnrBand] = []
    for entry in entries:
        fields = str(entry).split(":")
        if len(fields) != 3:
            raise ValueError(f"SNR band must be LOW:HIGH:WEIGHT, got: {entry}")
        try:
            bands.append(SnrBand(*(float(field) for field in fields)))
        except ValueError as error:
            raise ValueError(f"Invalid SNR band: {entry}") from error
    return SnrDistribution(bands)


def active_rms(
    waveform: torch.Tensor,
    *,
    frame_length: int = 400,
    hop_length: int = 200,
    activity_db: float = -35.0,
) -> torch.Tensor:
    """Estimate speech RMS from active frames, excluding zero padding."""

    signal = waveform.to(torch.float32).reshape(-1)
    if signal.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=signal.device)
    frame_length = max(1, min(int(frame_length), signal.numel()))
    hop_length = max(1, min(int(hop_length), frame_length))
    if signal.numel() < frame_length:
        signal = F.pad(signal, (0, frame_length - signal.numel()))
    remainder = max(0, signal.numel() - frame_length)
    frame_count = 1 + math.ceil(remainder / hop_length)
    total_length = frame_length + (frame_count - 1) * hop_length
    if signal.numel() < total_length:
        signal = F.pad(signal, (0, total_length - signal.numel()))
    frame_energy = signal.unfold(0, frame_length, hop_length).square().mean(dim=-1)
    peak_energy = frame_energy.max()
    if float(peak_energy.item()) <= 1e-12:
        return torch.zeros((), dtype=torch.float32, device=signal.device)
    threshold = peak_energy * (10.0 ** (float(activity_db) / 10.0))
    return frame_energy[frame_energy >= threshold].mean().sqrt()


def rms(waveform: torch.Tensor) -> torch.Tensor:
    signal = waveform.to(torch.float32)
    return signal.square().mean().clamp_min(1e-12).sqrt()


def mix_waveforms_at_snr(
    clean: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float,
    *,
    clean_reference_rms: torch.Tensor | float | None = None,
    peak_limit: float = 0.99,
) -> torch.Tensor:
    """Mix equal-shaped tensors at a measured SNR without hard clipping."""

    if clean.shape != noise.shape:
        raise ValueError("clean and noise must have the same shape")
    if not math.isfinite(float(snr_db)):
        raise ValueError("snr_db must be finite")
    if not 0.0 < peak_limit <= 1.0:
        raise ValueError("peak_limit must be in (0, 1]")
    clean = clean.to(torch.float32)
    noise = noise.to(device=clean.device, dtype=torch.float32)
    reference = active_rms(clean) if clean_reference_rms is None else torch.as_tensor(
        clean_reference_rms, device=clean.device, dtype=torch.float32
    )
    if float(reference.item()) <= 1e-8:
        return clean.clone()
    centered_noise = noise - noise.mean()
    noise_level = rms(centered_noise)
    target_noise_rms = reference / (10.0 ** (float(snr_db) / 20.0))
    mixed = clean + centered_noise * (target_noise_rms / noise_level)
    peak = mixed.abs().max()
    if float(peak.item()) > peak_limit:
        mixed = mixed * (peak_limit / peak)
    return mixed


def normalize_role(role: str) -> str:
    normalized = str(role).strip().lower()
    if normalized == "positive" or normalized.endswith("_positive"):
        return "positive"
    if "hard_negative" in normalized or normalized in {"phonetic", "confusable"}:
        return "hard_negative"
    return "negative"


def _iter_wavs(root: Path) -> Iterator[Path]:
    if root.is_file() and root.suffix.lower() == ".wav":
        yield root
        return
    if root.is_file():
        for raw_line in root.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            path = Path(line).expanduser()
            if not path.is_absolute():
                path = root.parent / path
            if path.is_file() and path.suffix.lower() == ".wav":
                yield path.resolve()
        return
    if root.is_dir():
        yield from sorted(path.resolve() for path in root.rglob("*.wav") if path.is_file())


class NoiseCatalog:
    """In-memory path index with weighted noise-domain selection."""

    def __init__(self, domain_paths: Mapping[str, Sequence[Path | str]], domain_weights: Mapping[str, float] | None = None):
        cleaned: dict[str, tuple[str, ...]] = {}
        for raw_name, paths in domain_paths.items():
            name = str(raw_name).strip()
            if not name or name in cleaned:
                raise ValueError("Noise domain names must be unique and non-empty")
            unique = tuple(dict.fromkeys(os.path.abspath(os.fspath(path)) for path in paths))
            if not unique:
                raise ValueError(f"Noise domain is empty: {name}")
            cleaned[name] = unique
        if not cleaned:
            raise ValueError("At least one non-empty noise domain is required")
        supplied = dict(domain_weights or {})
        unknown = set(supplied).difference(cleaned)
        if unknown:
            raise ValueError(f"Noise weights name unknown domains: {', '.join(sorted(unknown))}")
        weights = {name: float(supplied.get(name, 1.0)) for name in cleaned}
        if any(not math.isfinite(value) or value <= 0.0 for value in weights.values()):
            raise ValueError("Noise domain weights must be finite and positive")
        self.domain_paths = cleaned
        self.domain_weights = weights
        self._domains = tuple(cleaned)

    @classmethod
    def from_entries(
        cls,
        entries: Sequence[str],
        *,
        weight_entries: Sequence[str] | None = None,
    ) -> "NoiseCatalog":
        roots: dict[str, list[Path]] = {}
        for entry in entries:
            name, separator, raw_path = str(entry).partition("=")
            if not separator or not name.strip() or not raw_path.strip():
                raise ValueError(f"Noise domain must be NAME=PATH, got: {entry}")
            paths = list(_iter_wavs(Path(raw_path).expanduser().resolve()))
            roots.setdefault(name.strip(), []).extend(paths)
        weights: dict[str, float] = {}
        for entry in weight_entries or ():
            name, separator, raw_weight = str(entry).partition("=")
            if not separator or not name.strip() or name.strip() in weights:
                raise ValueError(f"Noise domain weight must be unique NAME=WEIGHT, got: {entry}")
            try:
                weights[name.strip()] = float(raw_weight)
            except ValueError as error:
                raise ValueError(f"Invalid noise domain weight: {entry}") from error
        return cls(roots, weights)

    def sample(self, rng: random.Random) -> tuple[str, str]:
        domain = rng.choices(
            self._domains,
            weights=[self.domain_weights[name] for name in self._domains],
            k=1,
        )[0]
        return domain, rng.choice(self.domain_paths[domain])


class OnlineWaveformAugmenter:
    """Decode one noise window and mix it into a clean source waveform."""

    def __init__(
        self,
        catalog: NoiseCatalog,
        *,
        sample_rate: int,
        rules: Mapping[str, RoleAugmentationRule] | None = None,
        allow_resample: bool = False,
        silence_noise_dbfs: tuple[float, float] = (-36.0, -18.0),
        seed: int | None = None,
    ):
        if sample_rate < 1:
            raise ValueError("sample_rate must be positive")
        self.catalog = catalog
        self.sample_rate = int(sample_rate)
        self.rules = dict(rules or default_nihao_wenwen_rules())
        if set(self.rules) != set(SUPPORTED_SAMPLE_ROLES):
            raise ValueError("Augmentation rules must cover positive, negative, and hard_negative")
        low_dbfs, high_dbfs = (float(value) for value in silence_noise_dbfs)
        if not math.isfinite(low_dbfs) or not math.isfinite(high_dbfs) or low_dbfs > high_dbfs:
            raise ValueError("silence_noise_dbfs must be a finite ordered range")
        self.silence_noise_dbfs = (low_dbfs, high_dbfs)
        self.allow_resample = bool(allow_resample)
        self._rng = random.Random(seed) if seed is not None else None

    def _random(self) -> random.Random:
        return self._rng if self._rng is not None else random

    def _load_noise_window(self, path: str, samples: int, rng: random.Random) -> torch.Tensor:
        info = torchaudio.info(path)
        if info.sample_rate == self.sample_rate and info.num_frames >= samples:
            offset = rng.randint(0, int(info.num_frames) - samples)
            noise, sample_rate = torchaudio.load(path, frame_offset=offset, num_frames=samples)
        else:
            noise, sample_rate = torchaudio.load(path)
            if sample_rate != self.sample_rate:
                if not self.allow_resample:
                    raise ValueError(
                        f"Noise sample-rate mismatch: {path}, got {sample_rate}, expected {self.sample_rate}"
                    )
                noise = AF.resample(noise, sample_rate, self.sample_rate)
        if noise.shape[0] > 1:
            noise = noise[rng.randrange(noise.shape[0]) :][:1]
        if noise.shape[1] < samples:
            if noise.shape[1] == 0:
                raise ValueError(f"Noise file has no samples: {path}")
            repeats = math.ceil((samples + noise.shape[1]) / noise.shape[1])
            noise = noise.repeat(1, repeats)
            offset = rng.randint(0, noise.shape[1] - samples)
            noise = noise[:, offset : offset + samples]
        elif noise.shape[1] > samples:
            offset = rng.randint(0, noise.shape[1] - samples)
            noise = noise[:, offset : offset + samples]
        return noise.to(torch.float32).clamp(-1.0, 1.0)

    def _sample_noise(self, samples: int, rng: random.Random) -> torch.Tensor:
        last_error: Exception | None = None
        for _ in range(8):
            _, path = self.catalog.sample(rng)
            try:
                noise = self._load_noise_window(path, samples, rng)
            except (OSError, RuntimeError, ValueError) as error:
                last_error = error
                continue
            centered = noise - noise.mean()
            if float(rms(centered).item()) > 1e-6:
                return centered
        raise RuntimeError("Unable to load a non-silent noise window") from last_error

    def __call__(self, waveform: torch.Tensor, role: str) -> torch.Tensor:
        if waveform.ndim != 2 or waveform.shape[0] != 1:
            raise ValueError("online augmentation expects mono [1, samples] waveform")
        rng = self._random()
        normalized_role = normalize_role(role)
        rule = self.rules[normalized_role]
        if rng.random() >= rule.mix_probability:
            return waveform
        noise = self._sample_noise(waveform.shape[1], rng).to(waveform.device)
        clean_level = active_rms(waveform)
        if float(clean_level.item()) <= 1e-8:
            target_dbfs = rng.uniform(*self.silence_noise_dbfs)
            target_rms = 10.0 ** (target_dbfs / 20.0)
            return (noise * (target_rms / rms(noise))).clamp(-0.99, 0.99)
        return mix_waveforms_at_snr(
            waveform,
            noise,
            rule.snr.sample(rng),
            clean_reference_rms=clean_level,
        )


def manifest_sample_roles(manifest_path: str | Path, positive_commands: Collection[str] = ("positive",)) -> list[str]:
    """Read sampling roles without changing the model's class labels."""

    positive_commands = set(positive_commands)
    roles: list[str] = []
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            command = str(row.get("command", "unknown"))
            explicit = row.get("sampling_role")
            if explicit is None:
                explicit = "hard_negative" if bool(row.get("hard_negative", False)) else command
            elif str(explicit).strip().lower() not in SUPPORTED_SAMPLE_ROLES:
                raise ValueError(f"Unknown sampling_role at line {line_number}: {explicit}")
            role = normalize_role(str(explicit))
            if command in positive_commands and role != "positive":
                raise ValueError(f"Positive row has a non-positive sampling role at line {line_number}")
            if command not in positive_commands and role == "positive":
                raise ValueError(f"Non-positive row has a positive sampling role at line {line_number}")
            roles.append(role)
    return roles


class OnlineAugmentedDataset(Dataset):
    """Dataset decorator that preserves the wrapped loader's tuple contract."""

    def __init__(
        self,
        dataset: Dataset,
        augmenter: OnlineWaveformAugmenter,
        *,
        sample_roles: Sequence[str] | None = None,
    ):
        self.dataset = dataset
        self.augmenter = augmenter
        self.sample_roles = tuple(normalize_role(role) for role in sample_roles) if sample_roles is not None else None
        if self.sample_roles is not None and len(self.sample_roles) != len(dataset):
            raise ValueError("sample_roles length must match the wrapped dataset")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name: str):
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)

    def __getitem__(self, key):
        item = self.dataset[key]
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise TypeError("Wrapped dataset must return waveform and label")
        role = None
        if len(item) >= 4 and isinstance(item[3], str):
            role = item[3]
        elif self.sample_roles is not None:
            if not isinstance(key, int):
                raise TypeError("Index-addressed sampling roles require integer dataset keys")
            role = self.sample_roles[key]
        else:
            role = "positive" if int(item[1]) == 0 else "negative"
        values = list(item)
        values[0] = self.augmenter(values[0], role)
        return tuple(values)


class StratifiedRoleBatchSampler(Sampler[list[int]]):
    """Emit exact per-rank positive/negative/hard-negative batch quotas."""

    def __init__(
        self,
        sample_roles: Sequence[str],
        *,
        quotas: Mapping[str, int],
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        steps_per_epoch: int,
    ):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be positive")
        parsed_quotas = {normalize_role(name): value for name, value in quotas.items()}
        if len(parsed_quotas) != len(quotas) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in parsed_quotas.values()
        ):
            raise ValueError("Role quotas must be unique positive integers")
        pools = {name: [] for name in parsed_quotas}
        for index, raw_role in enumerate(sample_roles):
            role = normalize_role(raw_role)
            if role in pools:
                pools[role].append(index)
        empty = [name for name, pool in pools.items() if not pool]
        if empty:
            raise ValueError(f"Sampling role is empty: {', '.join(empty)}")
        for name, quota in parsed_quotas.items():
            if len(pools[name]) < quota * world_size:
                raise ValueError(f"Sampling role {name} cannot fill one global batch without duplication")
        self.sample_roles = tuple(normalize_role(role) for role in sample_roles)
        self.quotas = parsed_quotas
        self.pools = pools
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.steps_per_epoch = int(steps_per_epoch)
        self.epoch = 0

    @property
    def batch_size(self) -> int:
        return sum(self.quotas.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        orders = {name: rng.sample(pool, len(pool)) for name, pool in self.pools.items()}
        cursors = {name: 0 for name in self.pools}
        for step in range(self.steps_per_epoch):
            batch: list[int] = []
            for name, quota in self.quotas.items():
                global_count = quota * self.world_size
                order = orders[name]
                cursor = cursors[name]
                selected: list[int] = []
                while len(selected) < global_count:
                    take = min(global_count - len(selected), len(order) - cursor)
                    selected.extend(order[cursor : cursor + take])
                    cursor += take
                    if cursor == len(order):
                        # At a cycle boundary avoid repeating an item already in
                        # this global update while retaining the unconsumed pool.
                        selected_set = set(selected)
                        next_order = rng.sample(self.pools[name], len(self.pools[name]))
                        order = [index for index in next_order if index not in selected_set] + [
                            index for index in next_order if index in selected_set
                        ]
                        cursor = 0
                        orders[name] = order
                start = self.rank * quota
                batch.extend(selected[start : start + quota])
                cursors[name] = cursor
            random.Random(self.seed + self.epoch * 1_000_003 + step * 97 + self.rank).shuffle(batch)
            yield batch


def parse_role_quotas(entries: Sequence[str] | None, *, batch_size: int, has_hard_negatives: bool) -> dict[str, int]:
    if entries:
        quotas: dict[str, int] = {}
        for entry in entries:
            name, separator, raw_count = str(entry).partition("=")
            role = name.strip().lower()
            if role not in SUPPORTED_SAMPLE_ROLES:
                raise ValueError(f"Unknown online sampling role: {name}")
            if not separator or role in quotas:
                raise ValueError(f"Role quota must be unique ROLE=COUNT, got: {entry}")
            try:
                quotas[role] = int(raw_count)
            except ValueError as error:
                raise ValueError(f"Invalid role quota: {entry}") from error
        if sum(quotas.values()) != batch_size:
            raise ValueError("Online role quotas must sum to the per-rank batch size")
        return quotas
    if batch_size < 2:
        raise ValueError("Default online role sampling requires batch_size >= 2")
    positive = batch_size // 2
    if has_hard_negatives:
        hard_negative = max(1, batch_size // 5)
        return {"positive": positive, "negative": batch_size - positive - hard_negative, "hard_negative": hard_negative}
    return {"positive": positive, "negative": batch_size - positive}
