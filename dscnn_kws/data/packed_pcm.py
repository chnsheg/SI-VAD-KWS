"""Lossless fixed-width PCM16 storage for high-throughput KWS training."""

from __future__ import annotations

import bisect
from array import array
import json
import mmap
import os
import struct
import tempfile
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Mapping

import torch
from torch.utils.data import Dataset


PACKED_PCM_FORMAT = "packed_pcm16_v1"
PCM16_SCALE = 32767


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _atomic_write_json(path: Path, value: object) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write_bytes(path, encoded)


def canonical_pcm16(waveform: torch.Tensor, *, sample_rate: int) -> torch.Tensor:
    """Convert a waveform to one fixed PCM16 second using channel 0.

    Packed records are already at ``sample_rate`` (there is no source-rate
    argument here), so this function performs only shape/length canonicalizing.
    For multi-channel input we intentionally retain channel 0, matching the
    streaming/evaluation audio contract; averaging channels can create phase
    cancellation and train/deployment skew.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    value = waveform.detach().to(device="cpu", dtype=torch.float32)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2:
        raise ValueError("waveform must have shape [samples] or [channels, samples]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("waveform must contain only finite values")
    if value.shape[0] > 1:
        value = value.narrow(0, 0, 1)
    if value.shape[1] < sample_rate:
        value = torch.nn.functional.pad(value, (0, sample_rate - value.shape[1]))
    elif value.shape[1] > sample_rate:
        offset = (value.shape[1] - sample_rate) // 2
        value = value.narrow(1, offset, sample_rate)
    value = value.clamp(-1.0, 1.0)
    scaled = value * PCM16_SCALE
    rounded = torch.where(scaled >= 0, torch.floor(scaled + 0.5), torch.ceil(scaled - 0.5))
    return rounded.to(torch.int16).contiguous()


@dataclass(frozen=True)
class PackedPcmShard:
    path: Path
    records: int
    start_index: int


class PackedPcmShardWriter:
    """Append fixed-width PCM records and publish a manifest only when complete."""

    def __init__(self, output_root: Path | str, *, sample_rate: int = 16_000, shard_records: int = 16_384):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if shard_records <= 0:
            raise ValueError("shard_records must be positive")
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.sample_rate = int(sample_rate)
        self.shard_records = int(shard_records)
        self._labels = bytearray()
        self._jitter = bytearray()
        self._shards: list[dict[str, object]] = []
        self._shard_index = 0
        self._records_in_current_shard = 0
        self._current_handle: BinaryIO | None = None
        self._current_temporary_path: Path | None = None

    @property
    def record_count(self) -> int:
        return len(self._labels)

    def _open_current_shard(self) -> None:
        if self._current_handle is not None:
            return
        filename = f"shard-{self._shard_index:05d}.pcm16le"
        temporary_path = self.output_root / f".{filename}.{uuid.uuid4().hex}.tmp"
        self._current_handle = temporary_path.open("wb")
        self._current_temporary_path = temporary_path

    def _flush_current_shard(self) -> None:
        if self._current_handle is None or self._current_temporary_path is None:
            return
        self._current_handle.flush()
        os.fsync(self._current_handle.fileno())
        self._current_handle.close()
        filename = f"shard-{self._shard_index:05d}.pcm16le"
        final_path = self.output_root / filename
        os.replace(self._current_temporary_path, final_path)
        self._shards.append({"path": filename, "records": self._records_in_current_shard})
        self._shard_index += 1
        self._records_in_current_shard = 0
        self._current_handle = None
        self._current_temporary_path = None

    def add(self, waveform: torch.Tensor, *, label: int, jitter_ms: int) -> None:
        if isinstance(label, bool) or int(label) not in (0, 1):
            raise ValueError("label must be a binary integer (0 or 1)")
        if isinstance(jitter_ms, bool) or not isinstance(jitter_ms, int) or not 0 <= jitter_ms <= 65_535:
            raise ValueError("jitter_ms must be an integer in [0, 65535]")
        pcm16 = canonical_pcm16(waveform, sample_rate=self.sample_rate)
        self._open_current_shard()
        assert self._current_handle is not None
        self._current_handle.write(pcm16.numpy().tobytes(order="C"))
        self._labels.append(int(label))
        self._jitter.extend(struct.pack("<H", int(jitter_ms)))
        self._records_in_current_shard += 1
        if self._records_in_current_shard >= self.shard_records:
            self._flush_current_shard()

    def finalize(
        self,
        *,
        source_manifest: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Path:
        if self.record_count == 0:
            raise ValueError("cannot publish an empty packed corpus")
        self._flush_current_shard()
        labels_name = "labels.u8"
        jitter_name = "jitter_ms.u16le"
        _atomic_write_bytes(self.output_root / labels_name, bytes(self._labels))
        _atomic_write_bytes(self.output_root / jitter_name, bytes(self._jitter))
        manifest_path = self.output_root / "manifest.json"
        payload: dict[str, object] = {
            "format": PACKED_PCM_FORMAT,
            "sample_rate": self.sample_rate,
            "record_samples": self.sample_rate,
            "record_count": self.record_count,
            "shard_records": self.shard_records,
            "pcm_encoding": "s16le",
            "labels": labels_name,
            "jitter_ms": jitter_name,
            "shards": self._shards,
            "source_manifest": source_manifest,
        }
        for key, value in (metadata or {}).items():
            if key in payload:
                raise ValueError(f"packed manifest metadata cannot replace reserved key: {key}")
            payload[key] = value
        _atomic_write_json(manifest_path, payload)
        return manifest_path

    def close(self) -> None:
        if self._current_handle is not None:
            self._current_handle.close()
            self._current_handle = None
        if self._current_temporary_path is not None:
            try:
                self._current_temporary_path.unlink()
            except FileNotFoundError:
                pass
            self._current_temporary_path = None

    def __del__(self) -> None:
        self.close()


class PackedPcmDataset(Dataset):
    """Read a packed PCM corpus through small per-process read-only mmap caches."""

    def __init__(self, manifest_path: Path | str, *, max_open_shards: int = 2):
        self.manifest_path = Path(manifest_path).resolve()
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("format") != PACKED_PCM_FORMAT:
            raise ValueError(f"Unsupported packed PCM format: {payload.get('format')!r}")
        self.sample_rate = int(payload["sample_rate"])
        self.record_samples = int(payload["record_samples"])
        self.record_count = int(payload["record_count"])
        self.max_open_shards = max(1, int(max_open_shards))
        self._root = self.manifest_path.parent
        self._labels_path = self._root / str(payload["labels"])
        self._jitter_path = self._root / str(payload["jitter_ms"])
        self._shards: list[PackedPcmShard] = []
        start_index = 0
        for row in payload["shards"]:
            records = int(row["records"])
            if records <= 0:
                raise ValueError("packed shard record count must be positive")
            self._shards.append(PackedPcmShard(self._root / str(row["path"]), records, start_index))
            start_index += records
        if start_index != self.record_count:
            raise ValueError(f"packed shard total is {start_index}, expected {self.record_count}")
        self._starts = [shard.start_index for shard in self._shards]
        self._labels_handle: BinaryIO | None = None
        self._labels_map: mmap.mmap | None = None
        self._jitter_handle: BinaryIO | None = None
        self._jitter_map: mmap.mmap | None = None
        self._open_shards: OrderedDict[int, tuple[BinaryIO, mmap.mmap]] = OrderedDict()

    def __len__(self) -> int:
        return self.record_count

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_labels_handle"] = None
        state["_labels_map"] = None
        state["_jitter_handle"] = None
        state["_jitter_map"] = None
        state["_open_shards"] = OrderedDict()
        return state

    @staticmethod
    def _mapped_file(path: Path) -> tuple[BinaryIO, mmap.mmap]:
        handle = path.open("rb")
        try:
            return handle, mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ)
        except Exception:
            handle.close()
            raise

    def _ensure_metadata_maps(self) -> None:
        if self._labels_map is None:
            self._labels_handle, self._labels_map = self._mapped_file(self._labels_path)
        if self._jitter_map is None:
            self._jitter_handle, self._jitter_map = self._mapped_file(self._jitter_path)
        assert self._labels_map is not None and self._jitter_map is not None
        if len(self._labels_map) != self.record_count:
            raise ValueError("packed label index has an unexpected size")
        if len(self._jitter_map) != self.record_count * 2:
            raise ValueError("packed jitter index has an unexpected size")

    def _shard_for_index(self, index: int) -> tuple[int, PackedPcmShard]:
        if index < 0:
            index += self.record_count
        if not 0 <= index < self.record_count:
            raise IndexError(index)
        shard_index = bisect.bisect_right(self._starts, index) - 1
        return shard_index, self._shards[shard_index]

    def _shard_map(self, shard_index: int, shard: PackedPcmShard) -> mmap.mmap:
        opened = self._open_shards.pop(shard_index, None)
        if opened is None:
            opened = self._mapped_file(shard.path)
        self._open_shards[shard_index] = opened
        while len(self._open_shards) > self.max_open_shards:
            _, (old_handle, old_map) = self._open_shards.popitem(last=False)
            old_map.close()
            old_handle.close()
        mapped = opened[1]
        expected_bytes = shard.records * self.record_samples * 2
        if len(mapped) != expected_bytes:
            raise ValueError(f"packed shard has {len(mapped)} bytes, expected {expected_bytes}: {shard.path}")
        return mapped

    def jitter_ms_at(self, index: int) -> int:
        self._ensure_metadata_maps()
        if index < 0:
            index += self.record_count
        if not 0 <= index < self.record_count:
            raise IndexError(index)
        assert self._jitter_map is not None
        return int(struct.unpack_from("<H", self._jitter_map, index * 2)[0])

    def label_indices(self, label: int) -> array:
        """Return compact record indexes for one binary label, validating metadata."""
        if label not in (0, 1):
            raise ValueError("label must be 0 or 1")
        self._ensure_metadata_maps()
        assert self._labels_map is not None
        indexes = array("I")
        for index, raw_value in enumerate(self._labels_map):
            value = raw_value[0] if isinstance(raw_value, bytes) else int(raw_value)
            if value not in (0, 1):
                raise ValueError(f"packed label index contains invalid value {value} at {index}")
            if value == label:
                indexes.append(index)
        return indexes

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        self._ensure_metadata_maps()
        shard_index, shard = self._shard_for_index(index)
        if index < 0:
            index += self.record_count
        record_index = index - shard.start_index
        byte_count = self.record_samples * 2
        offset = record_index * byte_count
        mapped = self._shard_map(shard_index, shard)
        waveform = torch.frombuffer(bytearray(mapped[offset : offset + byte_count]), dtype=torch.int16).reshape(
            1, self.record_samples
        )
        assert self._labels_map is not None
        return waveform, int(self._labels_map[index])

    def close(self) -> None:
        for handle, mapped in self._open_shards.values():
            mapped.close()
            handle.close()
        self._open_shards.clear()
        if self._labels_map is not None:
            self._labels_map.close()
            self._labels_map = None
        if self._labels_handle is not None:
            self._labels_handle.close()
            self._labels_handle = None
        if self._jitter_map is not None:
            self._jitter_map.close()
            self._jitter_map = None
        if self._jitter_handle is not None:
            self._jitter_handle.close()
            self._jitter_handle = None

    def __del__(self) -> None:
        self.close()
