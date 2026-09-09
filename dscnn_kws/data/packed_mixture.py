"""Composite packed PCM training data with exact per-rank role mixtures."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Collection, Iterator, Mapping
import random

import torch
from torch.utils.data import Dataset, Sampler

from .packed_pcm import PackedPcmDataset


ROLE_QUOTAS_PER_TWENTY = {
    "base_positive": 7,
    "raw_positive": 3,
    "base_negative": 5,
    "raw_negative": 3,
    "hard_negative": 2,
}

V3_ROLE_QUOTAS_PER_TWENTY = {
    "base_positive": 7,
    "raw_positive": 3,
    "base_negative": 3,
    "raw_negative": 2,
    "false_wake_hard_negative": 2,
    "captured_environment_negative": 2,
    "tau_environment_negative": 1,
}
V3_PHONETIC_HARD_NEGATIVE_ROLE = "phonetic_hard_negative"
V3_STRUCTURED_ACOUSTIC_NEGATIVE_ROLE = "structured_acoustic_negative"
V3_OPTIONAL_ROLE_NAMES = (
    V3_PHONETIC_HARD_NEGATIVE_ROLE,
    V3_STRUCTURED_ACOUSTIC_NEGATIVE_ROLE,
)
V3_EXTENDED_ROLE_NAMES = (*V3_ROLE_QUOTAS_PER_TWENTY, *V3_OPTIONAL_ROLE_NAMES)

_V3_POSITIVE_ROLES = frozenset({"base_positive", "raw_positive"})
_V3_MIXED_LABEL_REUSE_ROLES = frozenset({"raw_positive", "raw_negative"})

LOCALITY_STEPS_PER_BLOCK = 4


def parse_v3_role_quotas(entries: Collection[str] | None) -> dict[str, int]:
    """Parse a complete, class-balanced v3 role quota override."""
    if not entries:
        return dict(V3_ROLE_QUOTAS_PER_TWENTY)
    parsed: dict[str, int] = {}
    for entry in entries:
        name, separator, raw_value = str(entry).partition("=")
        if not separator or not name or not raw_value or name in parsed:
            raise ValueError("V3 role quotas must be unique name=count pairs")
        try:
            value = int(raw_value)
        except ValueError as error:
            raise ValueError(f"V3 role quota must be an integer: {entry}") from error
        parsed[name] = value
    supported = set(V3_EXTENDED_ROLE_NAMES)
    required = set(V3_ROLE_QUOTAS_PER_TWENTY)
    required.update(name for name in V3_OPTIONAL_ROLE_NAMES if name in parsed)
    unknown = set(parsed).difference(supported)
    if unknown:
        raise ValueError(f"V3 role quota contains unknown roles: {', '.join(sorted(unknown))}")
    missing = required.difference(parsed)
    if missing:
        raise ValueError(f"V3 role quota is missing roles: {', '.join(sorted(missing))}")
    if any(value < 1 for value in parsed.values()):
        raise ValueError("V3 role quotas must be positive")
    if sum(parsed.values()) != 20:
        raise ValueError("V3 role quotas must sum to twenty")
    positive_slots = sum(parsed[name] for name in _V3_POSITIVE_ROLES)
    if positive_slots != 10:
        raise ValueError("V3 role quotas must be class balanced with ten positive slots")
    return {name: parsed[name] for name in V3_EXTENDED_ROLE_NAMES if name in required}


@dataclass(frozen=True)
class PackedRole:
    name: str
    dataset: PackedPcmDataset
    label: int
    indexes: array


class CompositePackedDataset(Dataset):
    """Address packed train roles through an explicit mixture contract."""

    def __init__(self, roles: list[PackedRole], *, quotas: Mapping[str, int] = ROLE_QUOTAS_PER_TWENTY):
        if {role.name for role in roles} != set(quotas):
            raise ValueError("Composite pack roles must match the mixture contract")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in quotas.values()):
            raise ValueError("Composite pack quotas must be positive integers")
        if sum(quotas.values()) != 20:
            raise ValueError("Composite pack quotas must sum to twenty")
        self.roles = tuple(roles)
        self.quotas = dict(quotas)
        self._datasets = tuple(dict.fromkeys(role.dataset for role in roles))

    @classmethod
    def from_manifests(
        cls,
        base_manifest: Path | str,
        raw_anchor_manifest: Path | str,
        hard_negative_manifest: Path | str,
    ) -> "CompositePackedDataset":
        base = PackedPcmDataset(base_manifest)
        raw = PackedPcmDataset(raw_anchor_manifest)
        hard = PackedPcmDataset(hard_negative_manifest)
        roles = [
            PackedRole("base_positive", base, 0, base.label_indices(0)),
            PackedRole("raw_positive", raw, 0, raw.label_indices(0)),
            PackedRole("base_negative", base, 1, base.label_indices(1)),
            PackedRole("raw_negative", raw, 1, raw.label_indices(1)),
            PackedRole("hard_negative", hard, 1, hard.label_indices(1)),
        ]
        if any(not role.indexes for role in roles):
            missing = [role.name for role in roles if not role.indexes]
            raise ValueError(f"Composite pack has an empty role: {', '.join(missing)}")
        if hard.label_indices(0):
            raise ValueError("Hard-negative pack must contain negative labels only")
        return cls(roles)

    @classmethod
    def from_v3_manifests(
        cls,
        manifests: Mapping[str, Path | str],
        *,
        quotas: Mapping[str, int] | None = None,
    ) -> "CompositePackedDataset":
        resolved_quotas = parse_v3_role_quotas(
            None if quotas is None else [f"{name}={value}" for name, value in quotas.items()]
        )
        if set(manifests) != set(resolved_quotas):
            raise ValueError("V3 manifests must name exactly the V3 mixture roles")
        roles: list[PackedRole] = []
        for name in resolved_quotas:
            expected_label = 0 if name in _V3_POSITIVE_ROLES else 1
            dataset, indexes = cls._load_v3_role_source(name, manifests[name], expected_label)
            if not indexes:
                raise ValueError(f"V3 packed role is empty: {name}")
            other_label = 1 - expected_label
            if name not in _V3_MIXED_LABEL_REUSE_ROLES and dataset.label_indices(other_label):
                raise ValueError(f"V3 packed role must contain only label {expected_label}: {name}")
            roles.append(PackedRole(name, dataset, expected_label, indexes))
        return cls(roles, quotas=resolved_quotas)

    @staticmethod
    def _load_v3_role_source(
        name: str,
        source_path: Path | str,
        expected_label: int,
    ) -> tuple[PackedPcmDataset, array]:
        path = Path(source_path).expanduser().resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Unable to read V3 role source for {name}: {path}") from error
        if isinstance(payload, dict) and payload.get("format") == "packed_pcm16_v1":
            if payload.get("intended_use") == "evaluation_only":
                raise ValueError(f"V3 role {name} is evaluation-only and cannot enter a training mixture")
            dataset = PackedPcmDataset(path)
            return dataset, dataset.label_indices(expected_label)
        if name != "raw_positive":
            raise ValueError(f"V3 role {name} must use a packed_pcm16_v1 manifest")
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("raw_positive reference has an unsupported schema")
        if payload.get("storage") != "external_read_only":
            raise ValueError("raw_positive reference must use external_read_only storage")
        if payload.get("target_label") != expected_label:
            raise ValueError("raw_positive reference target label is invalid")
        external_manifest = payload.get("external_manifest")
        if not isinstance(external_manifest, str) or not external_manifest:
            raise ValueError("raw_positive reference has no external manifest")
        dataset = PackedPcmDataset(Path(external_manifest).expanduser().resolve())
        excluded = payload.get("excluded_record_indices", [])
        if not isinstance(excluded, list) or any(
            isinstance(index, bool) or not isinstance(index, int) for index in excluded
        ):
            raise ValueError("raw_positive reference exclusions must be integer indexes")
        if len(set(excluded)) != len(excluded) or any(not 0 <= index < len(dataset) for index in excluded):
            raise ValueError("raw_positive reference exclusions are invalid")
        excluded_set = set(excluded)
        indexes = array("I", (index for index in dataset.label_indices(expected_label) if index not in excluded_set))
        return dataset, indexes

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self._datasets)

    def __getitem__(self, key: tuple[int, int]) -> tuple[torch.Tensor, int, int, str]:
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("CompositePackedDataset keys must be (role_index, record_index)")
        role_index, record_index = key
        role = self.roles[int(role_index)]
        waveform, label = role.dataset[int(record_index)]
        if label != role.label:
            raise ValueError(f"Packed role {role.name} returned label {label}")
        return waveform, label, role.dataset.jitter_ms_at(int(record_index)), role.name

    def close(self) -> None:
        for dataset in self._datasets:
            dataset.close()


class StratifiedCompositeBatchSampler(Sampler[list[tuple[int, int]]]):
    """Deterministically emit exact packed-role quotas to one DDP rank."""

    def __init__(
        self,
        dataset: CompositePackedDataset,
        *,
        batch_size: int,
        rank: int,
        world_size: int,
        seed: int,
        steps_per_epoch: int,
        allow_global_replacement_roles: Collection[str] = (),
    ):
        quota_total = sum(dataset.quotas.values())
        if batch_size < quota_total or batch_size % quota_total:
            raise ValueError("mixture batch_size must be divisible by the role quota total")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.steps_per_epoch = int(steps_per_epoch)
        self.epoch = 0
        self._roles = {role.name: (role_index, role) for role_index, role in enumerate(dataset.roles)}
        self._replacement_roles = frozenset(str(name) for name in allow_global_replacement_roles)
        unknown_replacement_roles = self._replacement_roles.difference(self._roles)
        if unknown_replacement_roles:
            unknown = ", ".join(sorted(unknown_replacement_roles))
            raise ValueError(f"Unknown replacement-enabled role: {unknown}")
        self._quotas = {
            name: self.batch_size * quota // quota_total for name, quota in dataset.quotas.items()
        }
        for name, quota in self._quotas.items():
            _, role = self._roles[name]
            if len(role.indexes) < quota * self.world_size and name not in self._replacement_roles:
                raise ValueError(f"Role {name} cannot fill one global update without duplication")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _role_order(self, role_index: int, role: PackedRole) -> array:
        """Shuffle groups of contiguous role indexes without randomizing each record."""
        quota = self._quotas[role.name]
        global_count = quota * self.world_size
        block_records = global_count * LOCALITY_STEPS_PER_BLOCK
        blocks = list(range(0, len(role.indexes), block_records))
        random.Random(self.seed + self.epoch * 1009 + role_index * 9176).shuffle(blocks)

        order = array("I")
        for start in blocks:
            stop = min(start + block_records, len(role.indexes))
            full_stop = stop - (stop - start) % global_count
            order.extend(role.indexes[start:full_stop])
            if full_stop < stop:
                tail = role.indexes[full_stop:stop]
                order.extend(tail)
                order.extend(role.indexes[start : start + global_count - len(tail)])
        return order

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        orders = {
            name: self._role_order(role_index, role)
            for name, (role_index, role) in self._roles.items()
        }
        for step in range(self.steps_per_epoch):
            batch: list[tuple[int, int]] = []
            for name, quota in self._quotas.items():
                role_index, role = self._roles[name]
                order = orders[name]
                global_count = quota * self.world_size
                start = (step * global_count) % len(order)
                rank_start = start + self.rank * quota
                batch.extend((role_index, int(order[(rank_start + offset) % len(order)])) for offset in range(quota))
            random.Random(self.seed + self.epoch * 1_000_003 + step * 97 + self.rank).shuffle(batch)
            yield batch
