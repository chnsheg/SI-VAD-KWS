"""Manifest-backed adjacent-window examples for confirmation-aware KWS."""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import torch
import torchaudio
from torch.utils.data import Dataset, Sampler


CONFIRMATION_PAIR_FORMAT = "kws_confirmation_pair_v1"


def extract_adjacent_window_pairs(
    waveform: torch.Tensor,
    *,
    window_samples: int,
    hop_samples: int,
    starts: torch.Tensor | Sequence[int] | None = None,
    random_offset: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Cut exact adjacent windows from one continuous span per batch item.

    The input must contain real context through ``start + window + hop``.  The
    function never pads or rolls a one-second record because either operation
    would fabricate the second deployment window.
    """

    if window_samples <= 0:
        raise ValueError("window_samples must be positive")
    if hop_samples <= 0:
        raise ValueError("hop_samples must be positive")
    if starts is not None and random_offset:
        raise ValueError("starts and random_offset are mutually exclusive")
    unbatched = waveform.ndim == 2
    if unbatched:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 3:
        raise ValueError("waveform must have shape [channels, samples] or [batch, channels, samples]")

    batch_size, channels, sample_count = waveform.shape
    maximum_start = sample_count - window_samples - hop_samples
    if maximum_start < 0:
        raise ValueError(
            "continuous waveform is too short for an adjacent pair: "
            f"got {sample_count}, need at least {window_samples + hop_samples} samples"
        )
    if starts is None:
        if random_offset and maximum_start > 0:
            starts_tensor = torch.randint(
                maximum_start + 1,
                (batch_size,),
                device=waveform.device,
                generator=generator,
            )
        else:
            starts_tensor = torch.zeros(batch_size, device=waveform.device, dtype=torch.long)
    else:
        starts_tensor = torch.as_tensor(starts, device=waveform.device, dtype=torch.long)
        if starts_tensor.ndim == 0:
            starts_tensor = starts_tensor.expand(batch_size)
        if starts_tensor.shape != (batch_size,):
            raise ValueError(f"starts must contain one offset per batch item, got {tuple(starts_tensor.shape)}")
    if bool(((starts_tensor < 0) | (starts_tensor > maximum_start)).any()):
        raise ValueError(f"starts must be within [0, {maximum_start}]")

    frame_offsets = torch.tensor((0, hop_samples), device=waveform.device, dtype=torch.long)
    positions = starts_tensor[:, None, None] + frame_offsets[None, :, None]
    positions = positions + torch.arange(window_samples, device=waveform.device)[None, None, :]
    gather_index = positions[:, :, None, :].expand(batch_size, 2, channels, window_samples)
    pairs = torch.gather(
        waveform[:, None, :, :].expand(batch_size, 2, channels, sample_count),
        dim=3,
        index=gather_index,
    )
    return pairs[0] if unbatched else pairs


def _required_int(row: dict, key: str, *, row_number: int, minimum: int = 0) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"pair manifest row {row_number}: {key} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class ConfirmationPairRecord:
    audio_path: str
    label: int
    command: str
    role: str
    source_split: str
    source_id: str
    sample_rate: int
    span_start_sample: int
    span_num_samples: int
    window_samples: int
    hop_samples: int
    active_start_sample: int | None
    active_end_sample: int | None


class ConfirmationPairDataset(Dataset):
    """Read explicit continuous spans and return exact two-frame examples.

    Each JSONL row uses ``kws_confirmation_pair_v1`` and identifies an audio
    source plus an absolute source offset.  Positive active-span timestamps are
    required and must lie wholly inside the intersection of both windows.
    """

    def __init__(
        self,
        manifest_path: str | os.PathLike[str],
        *,
        class_encoding: dict[str, int],
        sample_rate: int,
        window_samples: int,
        hop_samples: int,
        expected_source_split: str = "train",
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path).resolve()
        self.class_encoding = dict(class_encoding)
        self.sample_rate = int(sample_rate)
        self.window_samples = int(window_samples)
        self.hop_samples = int(hop_samples)
        self.expected_source_split = str(expected_source_split).strip()
        if self.sample_rate <= 0 or self.window_samples <= 0 or self.hop_samples <= 0:
            raise ValueError("sample_rate, window_samples, and hop_samples must be positive")
        if not self.expected_source_split:
            raise ValueError("expected_source_split must be a non-empty string")
        if self.class_encoding.get("positive") != 0 or self.class_encoding.get("negative") != 1:
            raise ValueError("confirmation pair training requires positive=0 and negative=1")
        self.records: list[ConfirmationPairRecord] = []
        self._source_info: dict[str, tuple[int, int]] = {}
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for row_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"pair manifest row {row_number}: invalid JSON") from error
                self.records.append(self._parse_record(row, row_number=row_number))
        if not self.records:
            raise ValueError("confirmation pair manifest is empty")

    def _resolve_audio_path(self, raw_path: object, *, row_number: int) -> str:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"pair manifest row {row_number}: audio_filepath must be a non-empty string")
        normalized = os.path.normpath(raw_path.strip())
        if os.path.isabs(normalized):
            path = normalized
        else:
            path = os.path.normpath(os.path.join(self.manifest_path.parent, normalized))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"pair manifest row {row_number}: audio file not found: {path}")
        return os.path.abspath(path)

    def _parse_record(self, row: object, *, row_number: int) -> ConfirmationPairRecord:
        if not isinstance(row, dict):
            raise ValueError(f"pair manifest row {row_number}: row must be a JSON object")
        if row.get("format") != CONFIRMATION_PAIR_FORMAT:
            raise ValueError(
                f"pair manifest row {row_number}: format must be {CONFIRMATION_PAIR_FORMAT!r}"
            )
        command = row.get("command")
        if command not in ("positive", "negative"):
            raise ValueError(f"pair manifest row {row_number}: command must be 'positive' or 'negative'")
        role = row.get("role")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"pair manifest row {row_number}: role must be a non-empty string")
        source_id = row.get("source_id")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"pair manifest row {row_number}: source_id must be a non-empty string")
        source_split = row.get("source_split")
        if not isinstance(source_split, str) or not source_split.strip():
            raise ValueError(f"pair manifest row {row_number}: source_split must be a non-empty string")
        source_split = source_split.strip()
        if source_split != self.expected_source_split:
            raise ValueError(
                f"pair manifest row {row_number}: source_split must be {self.expected_source_split!r}, "
                f"got {source_split!r}"
            )
        row_sample_rate = _required_int(row, "sample_rate", row_number=row_number, minimum=1)
        span_start = _required_int(row, "span_start_sample", row_number=row_number)
        span_samples = _required_int(row, "span_num_samples", row_number=row_number, minimum=1)
        row_window = _required_int(row, "window_samples", row_number=row_number, minimum=1)
        row_hop = _required_int(row, "hop_samples", row_number=row_number, minimum=1)
        if row_sample_rate != self.sample_rate:
            raise ValueError(
                f"pair manifest row {row_number}: sample_rate={row_sample_rate}, expected {self.sample_rate}"
            )
        if row_window != self.window_samples or row_hop != self.hop_samples:
            raise ValueError(
                f"pair manifest row {row_number}: window/hop=({row_window}, {row_hop}), "
                f"expected ({self.window_samples}, {self.hop_samples})"
            )
        required_span = row_window + row_hop
        if span_samples != required_span:
            raise ValueError(
                f"pair manifest row {row_number}: span_num_samples={span_samples}, "
                f"must equal window_samples + hop_samples ({required_span})"
            )

        active_start = row.get("active_start_sample")
        active_end = row.get("active_end_sample")
        if command == "positive":
            active_start = _required_int(row, "active_start_sample", row_number=row_number)
            active_end = _required_int(row, "active_end_sample", row_number=row_number, minimum=1)
            common_start = span_start + row_hop
            common_end = span_start + row_window
            if not common_start <= active_start < active_end <= common_end:
                raise ValueError(
                    f"pair manifest row {row_number}: positive active span [{active_start}, {active_end}) "
                    f"must be fully inside both windows' overlap [{common_start}, {common_end})"
                )
        elif active_start is not None or active_end is not None:
            raise ValueError(f"pair manifest row {row_number}: negative rows must not declare an active span")

        return ConfirmationPairRecord(
            audio_path=self._resolve_audio_path(row.get("audio_filepath"), row_number=row_number),
            label=self.class_encoding[command],
            command=command,
            role=role.strip(),
            source_split=source_split,
            source_id=source_id.strip(),
            sample_rate=row_sample_rate,
            span_start_sample=span_start,
            span_num_samples=span_samples,
            window_samples=row_window,
            hop_samples=row_hop,
            active_start_sample=active_start,
            active_end_sample=active_end,
        )

    def __len__(self) -> int:
        return len(self.records)

    def _verified_source_info(self, record: ConfirmationPairRecord) -> tuple[int, int]:
        cached = self._source_info.get(record.audio_path)
        if cached is None:
            info = torchaudio.info(record.audio_path)
            cached = (int(info.sample_rate), int(info.num_frames))
            self._source_info[record.audio_path] = cached
        actual_rate, frame_count = cached
        if actual_rate != record.sample_rate:
            raise ValueError(
                f"pair source sample-rate mismatch: {record.audio_path}, got {actual_rate}, "
                f"manifest says {record.sample_rate}"
            )
        span_end = record.span_start_sample + record.span_num_samples
        if span_end > frame_count:
            raise ValueError(
                f"pair span [{record.span_start_sample}, {span_end}) exceeds "
                f"{frame_count} frames in {record.audio_path}"
            )
        return cached

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        self._verified_source_info(record)
        required_samples = record.window_samples + record.hop_samples
        waveform, actual_rate = torchaudio.load(
            record.audio_path,
            frame_offset=record.span_start_sample,
            num_frames=required_samples,
        )
        if int(actual_rate) != record.sample_rate or waveform.shape[1] != required_samples:
            raise ValueError(f"failed to read the declared continuous pair span from {record.audio_path}")
        if waveform.shape[0] < 1:
            raise ValueError(f"pair source has no audio channels: {record.audio_path}")
        waveform = waveform.narrow(0, 0, 1).to(torch.float32).clamp(-1.0, 1.0)
        pair = extract_adjacent_window_pairs(
            waveform,
            window_samples=record.window_samples,
            hop_samples=record.hop_samples,
        )
        return {
            "waveform": pair,
            "labels": record.label,
            "source_splits": record.source_split,
            "source_ids": record.source_id,
            "pair_roles": record.role,
            "pair_offsets": torch.tensor(
                (record.span_start_sample, record.span_start_sample + record.hop_samples),
                dtype=torch.int64,
            ),
        }


class DistributedClassStratifiedPairBatchSampler(Sampler[list[int]]):
    """Emit deterministic per-rank pair batches with fixed class balance.

    Negative records are visited round-robin by role and then source.  Every
    rank derives the same global batch before taking its disjoint shard, so a
    training update cannot duplicate a record across ranks when every domain
    has enough distinct records to fill one global update.
    """

    def __init__(
        self,
        dataset: ConfirmationPairDataset,
        *,
        batch_size: int,
        positive_per_batch: int,
        rank: int,
        world_size: int,
        seed: int,
        steps_per_epoch: int | None = None,
    ) -> None:
        if batch_size < 2:
            raise ValueError("pair batch_size must be at least two")
        if not 0 < positive_per_batch < batch_size:
            raise ValueError("positive_per_batch must be in [1, batch_size)")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.positive_per_batch = int(positive_per_batch)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.epoch = 0
        self._positive_indexes = [
            index for index, record in enumerate(dataset.records) if record.label == 0
        ]
        negative_groups: dict[str, dict[str, list[int]]] = {}
        for index, record in enumerate(dataset.records):
            if record.label == 0:
                continue
            negative_groups.setdefault(record.role, {}).setdefault(record.source_id, []).append(index)
        if not self._positive_indexes or not negative_groups:
            raise ValueError("stratified pair sampling requires positive and negative records")
        self._negative_groups = negative_groups

        global_positive = self.positive_per_batch * self.world_size
        global_negative = (self.batch_size - self.positive_per_batch) * self.world_size
        if len(self._positive_indexes) < global_positive:
            raise ValueError("positive pair pool cannot fill one global update without duplication")
        negative_count = sum(
            len(indexes)
            for sources in self._negative_groups.values()
            for indexes in sources.values()
        )
        if negative_count < global_negative:
            raise ValueError("negative pair pool cannot fill one global update without duplication")
        per_domain_required = math.ceil(global_negative / len(self._negative_groups))
        for domain, sources in self._negative_groups.items():
            if sum(len(indexes) for indexes in sources.values()) < per_domain_required:
                raise ValueError(
                    f"negative pair domain {domain!r} cannot fill one balanced global update"
                )

        if steps_per_epoch is None:
            steps_per_epoch = math.ceil(negative_count / global_negative)
        if isinstance(steps_per_epoch, bool) or not isinstance(steps_per_epoch, int) or steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be a positive integer")
        self.steps_per_epoch = int(steps_per_epoch)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    @staticmethod
    def _cycling_indexes(indexes: Sequence[int], *, seed: int) -> Iterator[int]:
        cycle = 0
        while True:
            order = list(indexes)
            random.Random(seed + cycle * 104_729).shuffle(order)
            yield from order
            cycle += 1

    @staticmethod
    def _draw_unique(iterator: Iterator[int], used: set[int], *, attempts: int) -> int:
        for _ in range(attempts):
            index = next(iterator)
            if index not in used:
                used.add(index)
                return index
        raise ValueError("pair sampler could not construct a duplicate-free global update")

    def _negative_domain_iterator(self, domain: str, *, domain_index: int) -> Iterator[int]:
        sources = sorted(self._negative_groups[domain])
        random.Random(self.seed + self.epoch * 1_000_003 + domain_index * 9_176).shuffle(sources)
        source_iterators = {
            source: self._cycling_indexes(
                self._negative_groups[domain][source],
                seed=self.seed + self.epoch * 1_000_003 + domain_index * 9_176 + index * 65_537,
            )
            for index, source in enumerate(sources)
        }
        cursor = 0
        while True:
            source = sources[cursor % len(sources)]
            yield next(source_iterators[source])
            cursor += 1

    def __iter__(self) -> Iterator[list[int]]:
        epoch_seed = self.seed + self.epoch * 1_000_003
        positive_iterator = self._cycling_indexes(self._positive_indexes, seed=epoch_seed + 17)
        domains = sorted(self._negative_groups)
        random.Random(epoch_seed + 31).shuffle(domains)
        negative_iterators = {
            domain: self._negative_domain_iterator(domain, domain_index=index)
            for index, domain in enumerate(domains)
        }
        global_positive_count = self.positive_per_batch * self.world_size
        local_negative_count = self.batch_size - self.positive_per_batch
        global_negative_count = local_negative_count * self.world_size
        maximum_attempts = max(1, len(self.dataset) * 2)

        for step in range(self.steps_per_epoch):
            used: set[int] = set()
            global_positive = [
                self._draw_unique(positive_iterator, used, attempts=maximum_attempts)
                for _ in range(global_positive_count)
            ]
            global_negative: list[int] = []
            domain_start = (step * global_negative_count) % len(domains)
            for offset in range(global_negative_count):
                domain = domains[(domain_start + offset) % len(domains)]
                global_negative.append(
                    self._draw_unique(
                        negative_iterators[domain],
                        used,
                        attempts=maximum_attempts,
                    )
                )

            positive_start = self.rank * self.positive_per_batch
            negative_start = self.rank * local_negative_count
            batch = global_positive[
                positive_start : positive_start + self.positive_per_batch
            ] + global_negative[negative_start : negative_start + local_negative_count]
            random.Random(epoch_seed + step * 97 + self.rank).shuffle(batch)
            yield batch
