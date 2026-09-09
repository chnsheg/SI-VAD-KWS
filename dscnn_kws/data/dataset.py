from __future__ import annotations

import json
import math
import os
import random

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from .confirmation_pair import ConfirmationPairDataset, DistributedClassStratifiedPairBatchSampler
from .online_augmentation import (
    ONLINE_AUGMENT_PROFILE,
    NoiseCatalog,
    OnlineAugmentedDataset,
    OnlineWaveformAugmenter,
    RoleAugmentationRule,
    StratifiedRoleBatchSampler,
    default_nihao_wenwen_rules,
    manifest_sample_roles,
    parse_role_quotas,
    parse_snr_bands,
)
from .packed_mixture import CompositePackedDataset, StratifiedCompositeBatchSampler, parse_v3_role_quotas
from .packed_pcm import PackedPcmDataset


def _normalize_manifest_audio_path(raw_path: str) -> str:
    normalized = os.path.normpath(str(raw_path).strip())
    normalized_slash = normalized.replace("\\", "/")
    if normalized_slash.startswith("./"):
        normalized_slash = normalized_slash[2:]
    legacy_prefix = "TorchKWS/dataset/"
    if normalized_slash.startswith(legacy_prefix):
        normalized_slash = normalized_slash[len(legacy_prefix) :]
    return os.path.normpath(normalized_slash)


def _split_label_counter(dataset_list):
    counter = {}
    for _, label in dataset_list:
        counter[label] = counter.get(label, 0) + 1
    return counter


def _rms(waveform: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(waveform * waveform) + 1e-12)


def _mix_at_snr(speech: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    target_noise_rms = _rms(speech) / (10.0 ** (float(snr_db) / 20.0))
    scaled_noise = noise * (target_noise_rms / _rms(noise))
    mixed = speech + scaled_noise
    peak = mixed.abs().max()
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)
    return mixed.clamp(-1.0, 1.0)


def build_train_sampler(dataset: Dataset, rank: int, world_size: int, seed: int) -> DistributedSampler:
    return DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed, drop_last=True)


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation data across ranks without padding duplicate samples."""

    def __init__(self, dataset: Dataset, rank: int, world_size: int):
        if world_size < 1:
            raise ValueError("world_size must be positive")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        return max(0, (len(self.dataset) - self.rank + self.world_size - 1) // self.world_size)


class DistributedBlockBatchSampler(Sampler[list[int]]):
    """Shuffle disk blocks by epoch while keeping individual batches contiguous."""

    def __init__(
        self,
        *,
        dataset_size: int,
        batch_size: int,
        block_records: int,
        rank: int,
        world_size: int,
        seed: int,
    ):
        if dataset_size < 1:
            raise ValueError("dataset_size must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if block_records < batch_size or block_records % batch_size:
            raise ValueError("block_records must be a positive multiple of batch_size")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.block_records = int(block_records)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _global_batches(self) -> list[list[int]]:
        blocks = list(range(0, self.dataset_size, self.block_records))
        random.Random(self.seed + self.epoch).shuffle(blocks)
        batches: list[list[int]] = []
        for start in blocks:
            stop = min(start + self.block_records, self.dataset_size)
            stop -= (stop - start) % self.batch_size
            for batch_start in range(start, stop, self.batch_size):
                batches.append(list(range(batch_start, batch_start + self.batch_size)))
        usable_batch_count = len(batches) - (len(batches) % self.world_size)
        return batches[:usable_batch_count]

    def __iter__(self):
        return iter(self._global_batches()[self.rank :: self.world_size])

    def __len__(self) -> int:
        return len(self._global_batches()) // self.world_size


def apply_online_window_jitter(
    waveform: torch.Tensor,
    sample_rate: int,
    max_jitter_ms: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, int]:
    """Translate an exact one-second waveform at training time without rescaling it."""
    if waveform.ndim != 2 or waveform.shape[1] != sample_rate:
        raise ValueError("online window jitter expects [channels, sample_rate] audio")
    if max_jitter_ms < 0:
        raise ValueError("max_jitter_ms must be non-negative")
    max_jitter_samples = round(sample_rate * max_jitter_ms / 1000)
    if max_jitter_samples == 0:
        return waveform, 0
    offset = int(torch.randint(-max_jitter_samples, max_jitter_samples + 1, (1,), generator=generator).item())
    padded = F.pad(waveform, (max_jitter_samples, max_jitter_samples))
    start = max_jitter_samples + offset
    return padded.narrow(1, start, sample_rate), offset


class SpeechCommandDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        json_filename: str,
        is_training: bool,
        class_list: list[str],
        class_encoding: dict[str, int],
        sample_rate: int = 8000,
        noise_aug: bool = True,
        noise_roots: list[str] | None = None,
        noise_prob: float = 0.8,
        noise_snr_min_db: float = -5.0,
        noise_snr_max_db: float = 20.0,
        deterministic_noise: bool = False,
        random_seed: int = 42,
        allow_online_resample: bool = False,
        strict_sample_rate: bool = True,
        online_window_jitter_ms: int | None = None,
    ):
        super().__init__()
        self.classes = class_list
        self.class_encoding = class_encoding
        self.dataset_path = os.path.abspath(dataset_path)
        self.json_filename = os.path.abspath(json_filename)
        self.is_training = is_training
        self.sampling_rate = int(sample_rate)
        self.sample_length = self.sampling_rate
        self.noise_aug = bool(noise_aug)
        self.noise_roots = list(noise_roots or [])
        self.noise_prob = min(max(float(noise_prob), 0.0), 1.0)
        self.noise_snr_min_db = float(noise_snr_min_db)
        self.noise_snr_max_db = float(noise_snr_max_db)
        self.deterministic_noise = bool(deterministic_noise)
        self.random_seed = int(random_seed)
        self.allow_online_resample = bool(allow_online_resample)
        self.strict_sample_rate = bool(strict_sample_rate)
        self.online_window_jitter_ms = None if online_window_jitter_ms is None else int(online_window_jitter_ms)

        if self.noise_snr_min_db > self.noise_snr_max_db:
            raise ValueError("noise_snr_min_db must be <= noise_snr_max_db")
        if self.online_window_jitter_ms is not None and self.online_window_jitter_ms < 0:
            raise ValueError("online_window_jitter_ms must be non-negative")

        self.noise_path = os.path.join(self.dataset_path, "_background_noise_")
        self.noise_dataset = self._load_noise_dataset()
        self.speech_dataset = self._load_speech_dataset()

    def _resolve_audio_path(self, speech_path: str) -> str:
        normalized = _normalize_manifest_audio_path(speech_path)
        if os.path.isabs(normalized):
            return normalized

        package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        repo_root = os.path.dirname(package_root)
        dataset_parent = os.path.dirname(self.dataset_path)
        manifest_dir = os.path.dirname(self.json_filename)
        basename = os.path.basename(normalized)

        candidates = [
            os.path.normpath(os.path.join(dataset_parent, normalized)),
            os.path.normpath(os.path.join(repo_root, normalized)),
            os.path.normpath(os.path.join(self.dataset_path, basename)),
            os.path.normpath(os.path.join(self.dataset_path, normalized)),
            os.path.normpath(os.path.join(manifest_dir, normalized)),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            "Failed to resolve audio path from manifest. "
            f"raw='{speech_path}', normalized='{normalized}', candidates={candidates}"
        )

    def _resolve_noise_root(self, noise_root: str) -> str:
        normalized = os.path.normpath(str(noise_root).strip())
        if os.path.isabs(normalized):
            return normalized

        package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        repo_root = os.path.dirname(package_root)
        candidates = [
            os.path.normpath(os.path.join(os.getcwd(), normalized)),
            os.path.normpath(os.path.join(package_root, normalized)),
            os.path.normpath(os.path.join(repo_root, normalized)),
            os.path.normpath(os.path.join(self.dataset_path, normalized)),
        ]
        for path in candidates:
            if os.path.isdir(path) or os.path.isfile(path):
                return path
        return candidates[0]

    def _load_noise_dataset(self):
        noise_dataset = []
        roots = []
        if os.path.isdir(self.noise_path):
            roots.append(self.noise_path)
        roots.extend(self._resolve_noise_root(root) for root in self.noise_roots)

        seen = set()

        def add_noise_file(path: str):
            path = os.path.normpath(path)
            if path in seen or not path.lower().endswith(".wav"):
                return
            try:
                if os.path.getsize(path) <= 44:
                    return
            except OSError:
                return
            seen.add(path)
            noise_dataset.append(path)

        for noise_root in roots:
            if os.path.isfile(noise_root):
                list_dir = os.path.dirname(noise_root)
                with open(noise_root, "r", encoding="utf-8") as f:
                    for line in f:
                        item = line.strip()
                        if not item or item.startswith("#"):
                            continue
                        if not os.path.isabs(item):
                            item = os.path.join(list_dir, item)
                        add_noise_file(item)
                continue
            if not os.path.isdir(noise_root):
                continue
            for root, _, filenames in sorted(os.walk(noise_root, followlinks=True)):
                for fn in sorted(filenames):
                    add_noise_file(os.path.join(root, fn))
        return noise_dataset

    def _load_speech_dataset(self):
        with open(self.json_filename, "r", encoding="utf-8") as f:
            json_data = [json.loads(line) for line in f if line.strip()]

        dataset_list = []
        for item in json_data:
            category = item["command"] if item["command"] in self.classes else "unknown"
            jitter_ms = item.get("online_window_jitter_max_ms", self.online_window_jitter_ms)
            if jitter_ms is None:
                jitter_ms = 0
            if isinstance(jitter_ms, bool) or not isinstance(jitter_ms, int) or jitter_ms < 0:
                raise ValueError("online_window_jitter_max_ms must be a non-negative integer")
            dataset_list.append([self._resolve_audio_path(item["audio_filepath"]), category, jitter_ms])
        return dataset_list

    def _load_noise_audio(self, noise_path: str) -> torch.Tensor:
        noise, noise_sr = torchaudio.load(noise_path)
        if noise_sr != self.sampling_rate:
            if self.allow_online_resample:
                noise = AF.resample(noise, noise_sr, self.sampling_rate)
            elif self.strict_sample_rate:
                raise ValueError(
                    f"Noise sample-rate mismatch: {noise_path}, got {noise_sr}, expected {self.sampling_rate}"
                )
        if noise.shape[0] > 1:
            noise = noise[:1]
        return noise.to(torch.float32).clamp(-1.0, 1.0)

    def _random_state(self, index: int | None):
        if self.deterministic_noise and index is not None:
            return random.Random(self.random_seed + index * 1009)
        return random

    def _apply_noise(self, waveform: torch.Tensor, index: int | None) -> torch.Tensor:
        if not self.noise_aug or not self.noise_dataset:
            return waveform

        rng = self._random_state(index)
        if rng.random() >= self.noise_prob:
            return waveform

        for _ in range(5):
            noise_path = rng.choice(self.noise_dataset)
            try:
                noise = self._load_noise_audio(noise_path)
                break
            except Exception:
                continue
        else:
            return waveform

        if noise.shape[1] < self.sample_length:
            repeat_count = math.ceil(self.sample_length / max(1, noise.shape[1]))
            noise = noise.repeat(1, repeat_count)

        offset = rng.randint(0, noise.shape[1] - self.sample_length)
        noise = noise.narrow(1, offset, self.sample_length)
        snr_db = rng.uniform(self.noise_snr_min_db, self.noise_snr_max_db)
        return _mix_at_snr(waveform, noise, snr_db)

    def _load_audio(
        self,
        speech_path: str,
        speech_category: str,
        record_jitter_ms: int,
        index: int | None = None,
    ) -> torch.Tensor:
        waveform, orig_sr = torchaudio.load(speech_path)
        if orig_sr != self.sampling_rate:
            if self.allow_online_resample:
                waveform = AF.resample(waveform, orig_sr, self.sampling_rate)
            elif self.strict_sample_rate:
                raise ValueError(f"Sample-rate mismatch: {speech_path}, got {orig_sr}, expected {self.sampling_rate}")

        if waveform.shape[0] > 1:
            waveform = waveform[:1]
        waveform = waveform.to(torch.float32).clamp(-1.0, 1.0)
        is_exact_length = waveform.shape[1] == self.sample_length

        if waveform.shape[1] < self.sample_length:
            waveform = F.pad(waveform, [0, self.sample_length - waveform.shape[1]])

        if self.is_training:
            if (
                is_exact_length
                and speech_category == "positive"
                and record_jitter_ms > 0
            ):
                waveform, _ = apply_online_window_jitter(
                    waveform,
                    sample_rate=self.sampling_rate,
                    max_jitter_ms=record_jitter_ms,
                )
            elif not is_exact_length:
                pad_length = int(waveform.shape[1] * 0.1)
                waveform = F.pad(waveform, [pad_length, pad_length])
                offset = torch.randint(0, waveform.shape[1] - self.sample_length + 1, size=(1,)).item()
                waveform = waveform.narrow(1, offset, self.sample_length)
        elif waveform.shape[1] > self.sample_length:
            offset = (waveform.shape[1] - self.sample_length) // 2
            waveform = waveform.narrow(1, offset, self.sample_length)

        return self._apply_noise(waveform, index)

    def __len__(self):
        return len(self.speech_dataset)

    def __getitem__(self, index):
        speech_path, speech_category, record_jitter_ms = self.speech_dataset[index]
        label = self.class_encoding[speech_category]
        if speech_category == "silence":
            waveform = torch.zeros(1, self.sample_length)
            waveform = self._apply_noise(waveform, index)
        else:
            waveform = self._load_audio(speech_path, speech_category, record_jitter_ms, index)
        return waveform, label


class PackedTrainingDataset(PackedPcmDataset):
    """Packed train samples carry jitter metadata for vectorized device-side use."""

    def __getitem__(self, index):
        waveform, label = super().__getitem__(index)
        return waveform, label, self.jitter_ms_at(index)


def build_dataloaders(data_path: str, class_list, class_encoding, args):
    def arg(name, default):
        return getattr(args, name, default)

    train_manifest = arg("train_manifest", "") or os.path.join(data_path, "train_manifest.json")
    validation_manifest = arg("validation_manifest", "") or os.path.join(data_path, "validation_manifest.json")
    skip_test = bool(arg("skip_test", False))
    test_manifest = "" if skip_test else arg("test_manifest", "") or os.path.join(data_path, "test_manifest.json")
    noise_roots = arg("noise_roots", None)
    train_noise_roots = arg("train_noise_roots", None) or noise_roots
    valid_noise_roots = arg("valid_noise_roots", None) or noise_roots
    test_noise_roots = arg("test_noise_roots", None) or noise_roots
    noise_aug_prob = arg("noise_aug_prob", 0.8)
    noise_snr_min_db = arg("noise_snr_min_db", -5.0)
    noise_snr_max_db = arg("noise_snr_max_db", 20.0)
    eval_noise_aug_prob = arg("eval_noise_aug_prob", None)
    eval_noise_snr_min_db = arg("eval_noise_snr_min_db", None)
    eval_noise_snr_max_db = arg("eval_noise_snr_max_db", None)
    if eval_noise_aug_prob is None:
        eval_noise_aug_prob = noise_aug_prob
    if eval_noise_snr_min_db is None:
        eval_noise_snr_min_db = noise_snr_min_db
    if eval_noise_snr_max_db is None:
        eval_noise_snr_max_db = noise_snr_max_db
    sample_rate = arg("sample_rate", 8000)
    noise_aug = arg("noise_aug", True)
    eval_noise_aug = arg("eval_noise_aug", False)
    seed = arg("seed", 42)
    allow_online_resample = arg("allow_online_resample", False)
    strict_sample_rate = arg("strict_sample_rate", True)
    online_window_jitter_ms = arg("online_window_jitter_ms", None)
    online_augment_profile = str(arg("online_augment_profile", "none") or "none")
    online_noise_domains = list(arg("online_noise_domain", []) or [])
    online_noise_domain_weights = list(arg("online_noise_domain_weight", []) or [])
    online_positive_snr_bands = list(arg("online_positive_snr_band", []) or [])
    online_negative_snr_bands = list(arg("online_negative_snr_band", []) or [])
    online_hard_negative_snr_bands = list(arg("online_hard_negative_snr_band", []) or [])
    online_role_quotas = list(arg("online_role_quota", []) or [])
    online_steps_per_epoch = int(arg("online_steps_per_epoch", 0))
    num_workers = arg("num_workers", 0)
    prefetch_factor = arg("prefetch_factor", 4)
    packed_loader_workers = int(arg("packed_loader_workers", 1))
    packed_prefetch_factor = int(arg("packed_prefetch_factor", 1))
    if packed_loader_workers < 1:
        raise ValueError("packed_loader_workers must be positive")
    if packed_prefetch_factor < 1:
        raise ValueError("packed_prefetch_factor must be positive")
    batch = arg("batch", 256)
    gpu = arg("gpu", 0)
    packed_train_index = arg("packed_train_index", "")
    packed_block_records = arg("packed_block_records", 16_384)
    mixture_base_pack = arg("mixture_base_pack", "")
    mixture_raw_anchor_pack = arg("mixture_raw_anchor_pack", "")
    mixture_hard_negative_pack = arg("mixture_hard_negative_pack", "")
    mixture_steps_per_epoch = arg("mixture_steps_per_epoch", 0)
    raw_v3_roles = arg("mixture_v3_role", [])
    if raw_v3_roles is None:
        raw_v3_roles = []
    if isinstance(raw_v3_roles, str):
        raw_v3_roles = [raw_v3_roles]
    raw_v3_quotas = arg("mixture_v3_quota", [])
    if raw_v3_quotas is None:
        raw_v3_quotas = []
    if isinstance(raw_v3_quotas, str):
        raw_v3_quotas = [raw_v3_quotas]
    v3_quotas = parse_v3_role_quotas(raw_v3_quotas)
    raw_v3_replacement_roles = arg("mixture_v3_allow_replacement_role", [])
    if raw_v3_replacement_roles is None:
        raw_v3_replacement_roles = []
    if isinstance(raw_v3_replacement_roles, str):
        raw_v3_replacement_roles = [raw_v3_replacement_roles]
    v3_replacement_roles = tuple(str(name) for name in raw_v3_replacement_roles)
    if len(set(v3_replacement_roles)) != len(v3_replacement_roles) or any(not name for name in v3_replacement_roles):
        raise ValueError("V3 replacement-enabled roles must be unique non-empty names")
    replacement_eligible_roles = {"false_wake_hard_negative", "captured_environment_negative"}
    unsupported_replacement_roles = set(v3_replacement_roles).difference(replacement_eligible_roles)
    if unsupported_replacement_roles:
        names = ", ".join(sorted(unsupported_replacement_roles))
        allowed = ", ".join(sorted(replacement_eligible_roles))
        raise ValueError(f"Only {allowed} may allow global replacement, got: {names}")
    v3_manifests: dict[str, str] = {}
    for raw_role in raw_v3_roles:
        name, separator, manifest = str(raw_role).partition("=")
        if not separator or not name or not manifest or name in v3_manifests:
            raise ValueError("V3 mixture roles must be unique name=manifest pairs")
        v3_manifests[name] = manifest
    mixture_pack_paths = (
        mixture_base_pack,
        mixture_raw_anchor_pack,
        mixture_hard_negative_pack,
    )
    has_v2_mixture_pack = any(mixture_pack_paths)
    has_v3_mixture_pack = bool(v3_manifests)
    has_mixture_pack = has_v2_mixture_pack or has_v3_mixture_pack
    advanced_online_augmentation = online_augment_profile != "none"

    if advanced_online_augmentation and online_augment_profile != ONLINE_AUGMENT_PROFILE:
        raise ValueError(f"Unsupported online augmentation profile: {online_augment_profile}")
    if advanced_online_augmentation and bool(arg("offline_augmented_dataset", False)):
        raise ValueError("Online augmentation cannot be combined with offline_augmented_dataset")
    if advanced_online_augmentation and not online_noise_domains:
        online_noise_domains = [f"noise_{index}={root}" for index, root in enumerate(train_noise_roots or ())]
    if advanced_online_augmentation and not online_noise_domains:
        raise ValueError("Online augmentation requires --online-noise-domain or --train_noise_roots")

    if has_v2_mixture_pack and not all(mixture_pack_paths):
        raise ValueError("v2 mixture training requires base, raw-anchor, and hard-negative packs")
    if has_v2_mixture_pack and has_v3_mixture_pack:
        raise ValueError("v2 and V3 mixture packs cannot be combined")
    if v3_replacement_roles and not has_v3_mixture_pack:
        raise ValueError("V3 replacement-enabled roles require V3 mixture packs")
    if has_mixture_pack and packed_train_index:
        raise ValueError("mixture packs cannot be combined with packed_train_index")
    if has_mixture_pack and int(batch) % 20:
        raise ValueError("mixture batch must be divisible by 20")
    if has_mixture_pack and int(mixture_steps_per_epoch) < 1:
        raise ValueError("v2 mixture_steps_per_epoch must be positive")

    if has_v3_mixture_pack:
        train_dataset = CompositePackedDataset.from_v3_manifests(v3_manifests, quotas=v3_quotas)
        role_prefix = "V3"
    elif has_v2_mixture_pack:
        train_dataset = CompositePackedDataset.from_manifests(
            base_manifest=mixture_base_pack,
            raw_anchor_manifest=mixture_raw_anchor_pack,
            hard_negative_manifest=mixture_hard_negative_pack,
        )
        role_prefix = "v2"
    else:
        role_prefix = ""
    if has_mixture_pack:
        for role in train_dataset.roles:
            if role.dataset.sample_rate != sample_rate:
                raise ValueError(
                    f"{role_prefix} packed role {role.name} sample rate is {role.dataset.sample_rate}, expected {sample_rate}"
                )
    elif packed_train_index:
        train_dataset = PackedTrainingDataset(packed_train_index)
        if train_dataset.sample_rate != sample_rate:
            raise ValueError(
                f"packed train sample rate is {train_dataset.sample_rate}, expected {sample_rate}"
            )
    else:
        train_dataset = SpeechCommandDataset(
            dataset_path=data_path,
            json_filename=train_manifest,
            is_training=True,
            class_list=class_list,
            class_encoding=class_encoding,
            sample_rate=sample_rate,
            noise_aug=noise_aug and not advanced_online_augmentation,
            noise_roots=train_noise_roots,
            noise_prob=noise_aug_prob,
            noise_snr_min_db=noise_snr_min_db,
            noise_snr_max_db=noise_snr_max_db,
            deterministic_noise=False,
            random_seed=seed,
            allow_online_resample=allow_online_resample,
            strict_sample_rate=strict_sample_rate,
            online_window_jitter_ms=online_window_jitter_ms,
        )
    online_sample_roles = None
    if advanced_online_augmentation:
        rules = default_nihao_wenwen_rules()
        rules = {
            "positive": RoleAugmentationRule(
                rules["positive"].mix_probability,
                parse_snr_bands(online_positive_snr_bands, rules["positive"].snr),
            ),
            "negative": RoleAugmentationRule(
                rules["negative"].mix_probability,
                parse_snr_bands(online_negative_snr_bands, rules["negative"].snr),
            ),
            "hard_negative": RoleAugmentationRule(
                rules["hard_negative"].mix_probability,
                parse_snr_bands(online_hard_negative_snr_bands, rules["hard_negative"].snr),
            ),
        }
        catalog = NoiseCatalog.from_entries(
            online_noise_domains,
            weight_entries=online_noise_domain_weights,
        )
        augmenter = OnlineWaveformAugmenter(
            catalog,
            sample_rate=sample_rate,
            rules=rules,
            allow_resample=allow_online_resample,
        )
        if not packed_train_index and not has_mixture_pack:
            online_sample_roles = manifest_sample_roles(train_manifest)
        train_dataset = OnlineAugmentedDataset(
            train_dataset,
            augmenter,
            sample_roles=online_sample_roles,
        )
    valid_dataset = SpeechCommandDataset(
        dataset_path=data_path,
        json_filename=validation_manifest,
        is_training=False,
        class_list=class_list,
        class_encoding=class_encoding,
        sample_rate=sample_rate,
        noise_aug=eval_noise_aug,
        noise_roots=valid_noise_roots,
        noise_prob=eval_noise_aug_prob,
        noise_snr_min_db=eval_noise_snr_min_db,
        noise_snr_max_db=eval_noise_snr_max_db,
        deterministic_noise=True,
        random_seed=seed + 100000,
        allow_online_resample=allow_online_resample,
        strict_sample_rate=strict_sample_rate,
        online_window_jitter_ms=None,
    )
    test_dataset = None
    if not skip_test:
        test_dataset = SpeechCommandDataset(
            dataset_path=data_path,
            json_filename=test_manifest,
            is_training=False,
            class_list=class_list,
            class_encoding=class_encoding,
            sample_rate=sample_rate,
            noise_aug=eval_noise_aug,
            noise_roots=test_noise_roots,
            noise_prob=eval_noise_aug_prob,
            noise_snr_min_db=eval_noise_snr_min_db,
            noise_snr_max_db=eval_noise_snr_max_db,
            deterministic_noise=True,
            random_seed=seed + 200000,
            allow_online_resample=allow_online_resample,
            strict_sample_rate=strict_sample_rate,
            online_window_jitter_ms=None,
        )

    packed_training = bool(packed_train_index or has_mixture_pack)
    train_num_workers = packed_loader_workers if packed_training else num_workers
    eval_num_workers = max(0, num_workers // 2)

    distributed = bool(getattr(args, "distributed", False))
    rank = int(getattr(args, "rank", 0))
    world_size = int(getattr(args, "world_size", 1))
    train_sampler = build_train_sampler(train_dataset, rank, world_size, seed) if distributed and not packed_training else None
    valid_sampler = DistributedEvalSampler(valid_dataset, rank=rank, world_size=world_size) if distributed else None
    test_sampler = (
        DistributedEvalSampler(test_dataset, rank=rank, world_size=world_size)
        if distributed and test_dataset is not None
        else None
    )
    if has_mixture_pack:
        train_loader_kwargs = {
            "batch_sampler": StratifiedCompositeBatchSampler(
                train_dataset,
                batch_size=batch,
                rank=rank,
                world_size=world_size,
                seed=seed,
                steps_per_epoch=mixture_steps_per_epoch,
                allow_global_replacement_roles=v3_replacement_roles if has_v3_mixture_pack else (),
            ),
            "num_workers": train_num_workers,
            "pin_memory": gpu > 0 or distributed,
            "persistent_workers": True,
            "generator": torch.Generator().manual_seed(int(seed) + rank),
        }
    elif packed_train_index:
        train_loader_kwargs = {
            "batch_sampler": DistributedBlockBatchSampler(
                dataset_size=len(train_dataset),
                batch_size=batch,
                block_records=int(packed_block_records),
                rank=rank,
                world_size=world_size,
                seed=seed,
            ),
            "num_workers": train_num_workers,
            "pin_memory": gpu > 0 or distributed,
            "persistent_workers": True,
            "generator": torch.Generator().manual_seed(int(seed) + rank),
        }
    elif advanced_online_augmentation:
        if online_sample_roles is None:
            raise ValueError("Online role sampling requires manifest-backed training data")
        has_hard_negatives = "hard_negative" in online_sample_roles
        quotas = parse_role_quotas(
            online_role_quotas,
            batch_size=int(batch),
            has_hard_negatives=has_hard_negatives,
        )
        steps_per_epoch = online_steps_per_epoch or max(1, math.ceil(len(train_dataset) / int(batch)))
        train_loader_kwargs = {
            "batch_sampler": StratifiedRoleBatchSampler(
                online_sample_roles,
                quotas=quotas,
                rank=rank,
                world_size=world_size,
                seed=seed,
                steps_per_epoch=steps_per_epoch,
            ),
            "num_workers": train_num_workers,
            "pin_memory": gpu > 0 or distributed,
            "persistent_workers": train_num_workers > 0,
            "generator": torch.Generator().manual_seed(int(seed) + rank),
        }
    else:
        train_loader_kwargs = {
            "batch_size": batch,
            "shuffle": train_sampler is None,
            "sampler": train_sampler,
            "drop_last": False,
            "num_workers": train_num_workers,
            "pin_memory": gpu > 0 or distributed,
            "persistent_workers": train_num_workers > 0,
            "generator": torch.Generator().manual_seed(int(seed) + rank),
        }
    eval_loader_kwargs = {
        "batch_size": batch,
        "shuffle": False,
        "sampler": valid_sampler,
        "drop_last": False,
        "num_workers": eval_num_workers,
        "pin_memory": gpu > 0 or distributed,
        "persistent_workers": eval_num_workers > 0,
        "generator": torch.Generator().manual_seed(int(seed) + rank + 100000),
    }
    if train_num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = packed_prefetch_factor if packed_training else max(2, prefetch_factor)
        train_loader_kwargs["multiprocessing_context"] = "spawn"
    if eval_num_workers > 0:
        eval_loader_kwargs["prefetch_factor"] = max(2, prefetch_factor)
        eval_loader_kwargs["multiprocessing_context"] = "spawn"

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    valid_loader = DataLoader(valid_dataset, **eval_loader_kwargs)
    test_loader = None
    if test_dataset is not None:
        test_loader_kwargs = dict(eval_loader_kwargs)
        test_loader_kwargs["sampler"] = test_sampler
        test_loader = DataLoader(test_dataset, **test_loader_kwargs)
    return train_loader, valid_loader, test_loader


def build_confirmation_pair_loader(
    class_encoding,
    args,
    *,
    steps_per_epoch: int | None = None,
) -> DataLoader | None:
    """Build the optional auxiliary pair loader without changing base CE data."""

    pair_objective = bool(getattr(args, "pair_objective", False))
    manifest_path = str(getattr(args, "pair_train_manifest", "") or "")
    if pair_objective != bool(manifest_path):
        raise ValueError("--pair_objective and --pair_train_manifest must be enabled together")
    if not pair_objective:
        return None

    sample_rate = int(getattr(args, "sample_rate", 8_000))
    hop_ms = float(getattr(args, "pair_hop_ms", 96.0))
    if not math.isfinite(hop_ms):
        raise ValueError("pair_hop_ms must be finite")
    hop_samples = round(float(sample_rate) * hop_ms / 1000.0)
    if hop_samples < 1:
        raise ValueError("pair_hop_ms must produce at least one sample")
    dataset = ConfirmationPairDataset(
        manifest_path,
        class_encoding=class_encoding,
        sample_rate=sample_rate,
        window_samples=sample_rate,
        hop_samples=hop_samples,
        expected_source_split="train",
    )

    distributed = bool(getattr(args, "distributed", False))
    rank = int(getattr(args, "rank", 0))
    world_size = int(getattr(args, "world_size", 1))
    seed = int(getattr(args, "seed", 42))
    configured_pair_batch = getattr(args, "pair_batch", None)
    pair_batch = int(configured_pair_batch) if configured_pair_batch is not None else int(getattr(args, "batch", 256))
    if pair_batch < 1:
        raise ValueError("pair_batch must be positive")
    configured_positive_count = getattr(args, "pair_positive_per_batch", None)
    positive_per_batch = (
        int(configured_positive_count)
        if configured_positive_count is not None
        else 0
    )
    if positive_per_batch < 0:
        raise ValueError("pair_positive_per_batch must be non-negative")
    if float(getattr(args, "pair_tail_ranking_weight", 0.0)) > 0.0 and positive_per_batch == 0:
        raise ValueError("pair tail ranking requires pair_positive_per_batch greater than zero")
    configured_workers = getattr(args, "pair_num_workers", None)
    num_workers = (
        int(configured_workers) if configured_workers is not None else int(getattr(args, "num_workers", 0))
    )
    if num_workers < 0:
        raise ValueError("pair_num_workers must be non-negative")
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": int(getattr(args, "gpu", 0)) > 0 or distributed,
        "persistent_workers": num_workers > 0,
        "generator": torch.Generator().manual_seed(seed + rank + 300_000),
    }
    if positive_per_batch:
        kwargs["batch_sampler"] = DistributedClassStratifiedPairBatchSampler(
            dataset,
            batch_size=pair_batch,
            positive_per_batch=positive_per_batch,
            rank=rank,
            world_size=world_size,
            seed=seed + 300_000,
            steps_per_epoch=steps_per_epoch,
        )
    else:
        sampler = build_train_sampler(dataset, rank, world_size, seed + 300_000) if distributed else None
        kwargs.update(
            batch_size=pair_batch,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=False,
        )
    if num_workers > 0:
        kwargs["prefetch_factor"] = max(2, int(getattr(args, "prefetch_factor", 4)))
        kwargs["multiprocessing_context"] = "spawn"
    return DataLoader(dataset, **kwargs)
