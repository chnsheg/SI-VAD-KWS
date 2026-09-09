"""In-memory diagnostics and localhost dashboard for the VAD-KWS demo."""

from __future__ import annotations

import io
import json
import math
import socket
import threading
import time
import wave
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from .cascade import CascadeEvent
from .contracts import TimingSchedule
from .input_frontend import InputFrontend
from .runtime_profile import DEFAULT_RUNTIME_PROFILE, RuntimeProfile


_RAW_AUDIO_KEYS = frozenset({"audio", "pcm", "raw_pcm", "waveform", "samples"})
_AGC_PREVIEW_BLOCK_SAMPLES = 1600
_MIN_TARGET_RMS_DBFS = -30.0
_MAX_TARGET_RMS_DBFS = -18.0
_DASHBOARD_STATE_MIN_DURATION_NS = 120_000_000
_WAVEFORM_LINE_POINT_LIMIT = 2_048
_ANALYSIS_SAMPLE_RATE = 16_000
_MAX_DASHBOARD_IMPORT_SAMPLE_RATE = 48_000
_DASHBOARD_IMPORT_CHANNEL_COUNTS = frozenset((1, 2))
_DASHBOARD_IMPORT_SAMPLE_WIDTHS = frozenset((2, 3, 4))
_DASHBOARD_IMPORT_WAV_HEADER_ALLOWANCE_BYTES = 65_536
_DASHBOARD_IMPORT_BODY_HARD_CAP_BYTES = 64 * 1024 * 1024


def _fits_analysis_duration(
    *,
    frame_count: int,
    sample_rate: int,
    maximum_samples: int,
) -> bool:
    """Check source duration against a 16 kHz analysis-history capacity."""

    return frame_count * _ANALYSIS_SAMPLE_RATE <= maximum_samples * sample_rate


def _maximum_dashboard_import_body_bytes(maximum_samples: int) -> int:
    """Bound an upload using the largest supported source PCM shape."""

    maximum_source_frames = (
        maximum_samples * _MAX_DASHBOARD_IMPORT_SAMPLE_RATE // _ANALYSIS_SAMPLE_RATE
    )
    duration_limit = (
        maximum_source_frames
        * max(_DASHBOARD_IMPORT_CHANNEL_COUNTS)
        * max(_DASHBOARD_IMPORT_SAMPLE_WIDTHS)
        + _DASHBOARD_IMPORT_WAV_HEADER_ALLOWANCE_BYTES
    )
    return min(duration_limit, _DASHBOARD_IMPORT_BODY_HARD_CAP_BYTES)


def _schedule_fields(schedule: TimingSchedule) -> dict[str, int]:
    return {
        "energy_period_ms": schedule.energy_period_ms,
        "vad_period_ms": schedule.vad_period_ms,
        "kws_period_ms": schedule.kws_period_ms,
    }


def _render_agc_preview(
    pcm: np.ndarray,
    *,
    profile: RuntimeProfile,
    target_rms_dbfs: float,
) -> np.ndarray:
    """Render retained raw PCM through a fresh copy of the deployed frontend."""

    values = np.asarray(pcm)
    if values.dtype != np.float32 or values.ndim != 1:
        raise ValueError("PCM must be a float32 mono array")
    if not np.all(np.isfinite(values)):
        raise ValueError("PCM must contain only finite values")
    if values.size == 0:
        return np.empty(0, dtype=np.float32)
    if not isinstance(profile, RuntimeProfile):
        raise ValueError("profile must be a RuntimeProfile")
    frontend = InputFrontend(profile.frontend_profile, target_rms_dbfs=target_rms_dbfs)
    blocks = [
        frontend.process(values[start : start + _AGC_PREVIEW_BLOCK_SAMPLES])[0]
        for start in range(0, values.size, _AGC_PREVIEW_BLOCK_SAMPLES)
    ]
    return np.concatenate(blocks).astype(np.float32, copy=False)


def _waveform_envelope(
    values: np.ndarray,
    *,
    audio_start_ns: int | None,
    sample_rate: int,
) -> list[dict[str, object]]:
    """Summarize a PCM timeline with buckets that retain its exact endpoints."""

    if audio_start_ns is None or values.size == 0:
        return []
    bucket_count = min(480, max(1, math.ceil(values.size / 160)))
    boundaries = np.linspace(0, values.size, num=bucket_count + 1, dtype=np.int64)
    return [
        {
            "start_ns": audio_start_ns + int(start) * 1_000_000_000 // sample_rate,
            "end_ns": audio_start_ns + int(end) * 1_000_000_000 // sample_rate,
            "min": float(np.min(values[int(start) : int(end)])),
            "max": float(np.max(values[int(start) : int(end)])),
        }
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]


def _device_waveform_envelope(
    values: np.ndarray,
    *,
    audio_start_ns: int | None,
    sample_rate: int,
) -> list[dict[str, object]]:
    """Summarize all device channels without changing the captured samples."""

    samples = np.asarray(values)
    if samples.ndim != 2:
        raise ValueError("device waveform must be channel-last samples")
    if audio_start_ns is None or samples.shape[0] == 0:
        return []
    if np.issubdtype(samples.dtype, np.integer):
        normalized = samples.astype(np.float64) / 2147483648.0
    else:
        normalized = samples.astype(np.float64, copy=False)
    bucket_count = min(480, max(1, math.ceil(samples.shape[0] / 160)))
    boundaries = np.linspace(0, samples.shape[0], num=bucket_count + 1, dtype=np.int64)
    return [
        {
            "start_ns": audio_start_ns + int(start) * 1_000_000_000 // sample_rate,
            "end_ns": audio_start_ns + int(end) * 1_000_000_000 // sample_rate,
            "min": float(np.min(normalized[int(start) : int(end)])),
            "max": float(np.max(normalized[int(start) : int(end)])),
        }
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]


@dataclass(frozen=True)
class _RawDeviceChunk:
    samples: np.ndarray
    start_ns: int
    end_ns: int


def _raw_device_waveform_envelope(
    chunks: deque[_RawDeviceChunk],
    *,
    audio_start_ns: int,
    sample_rate: int,
) -> list[dict[str, object]]:
    """Render a bounded envelope directly from retained device chunks."""

    if not chunks:
        return []
    audio_end_ns = chunks[-1].end_ns
    total_frames = sum(chunk.samples.shape[0] for chunk in chunks)
    bucket_count = min(480, max(1, math.ceil(total_frames / 160)))
    span_ns = max(1, audio_end_ns - audio_start_ns)
    minimums = np.full(bucket_count, np.inf, dtype=np.float64)
    maximums = np.full(bucket_count, -np.inf, dtype=np.float64)
    for chunk in chunks:
        chunk_frames = chunk.samples.shape[0]
        chunk_span_ns = max(1, chunk.end_ns - chunk.start_ns)
        first_bucket = max(0, (chunk.start_ns - audio_start_ns) * bucket_count // span_ns)
        last_bucket = min(
            bucket_count - 1,
            max(0, (chunk.end_ns - audio_start_ns) * bucket_count - 1) // span_ns,
        )
        for bucket in range(int(first_bucket), int(last_bucket) + 1):
            bucket_start_ns = audio_start_ns + bucket * span_ns // bucket_count
            bucket_end_ns = audio_start_ns + (bucket + 1) * span_ns // bucket_count
            overlap_start_ns = max(chunk.start_ns, bucket_start_ns)
            overlap_end_ns = min(chunk.end_ns, bucket_end_ns)
            if overlap_end_ns <= overlap_start_ns:
                continue
            frame_start = (overlap_start_ns - chunk.start_ns) * chunk_frames // chunk_span_ns
            frame_end = max(
                frame_start + 1,
                (overlap_end_ns - chunk.start_ns) * chunk_frames // chunk_span_ns,
            )
            values = chunk.samples[int(frame_start) : int(min(chunk_frames, frame_end))]
            if np.issubdtype(values.dtype, np.integer):
                minimums[bucket] = min(minimums[bucket], float(np.min(values)) / 2147483648.0)
                maximums[bucket] = max(maximums[bucket], float(np.max(values)) / 2147483648.0)
            else:
                minimums[bucket] = min(minimums[bucket], float(np.min(values)))
                maximums[bucket] = max(maximums[bucket], float(np.max(values)))
    return [
        {
            "start_ns": audio_start_ns + bucket * span_ns // bucket_count,
            "end_ns": audio_start_ns + (bucket + 1) * span_ns // bucket_count,
            "min": float(minimums[bucket]),
            "max": float(maximums[bucket]),
        }
        for bucket in range(bucket_count)
        if np.isfinite(minimums[bucket])
    ]


def _waveform_line_points(
    values: np.ndarray,
    *,
    audio_start_ns: int | None,
    sample_rate: int,
) -> list[dict[str, object]]:
    """Return evenly-spaced PCM samples for a conventional line waveform."""

    samples = np.asarray(values)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2:
        raise ValueError("waveform samples must be one- or two-dimensional")
    if audio_start_ns is None or samples.shape[0] == 0:
        return []
    indices = np.linspace(
        0,
        samples.shape[0] - 1,
        num=min(_WAVEFORM_LINE_POINT_LIMIT, samples.shape[0]),
        dtype=np.int64,
    )
    selected = samples[indices]
    if np.issubdtype(selected.dtype, np.integer):
        selected = selected.astype(np.float64) / 2147483648.0
    else:
        selected = selected.astype(np.float64, copy=False)
    return [
        {
            "captured_ns": audio_start_ns + int(index) * 1_000_000_000 // sample_rate,
            "values": [float(value) for value in frame],
        }
        for index, frame in zip(indices, selected, strict=True)
    ]


def _raw_device_waveform_line_points(
    chunks: deque[_RawDeviceChunk],
) -> list[dict[str, object]]:
    """Sample retained device PCM without first concatenating the live ring."""

    total_frames = sum(chunk.samples.shape[0] for chunk in chunks)
    if total_frames == 0:
        return []
    target_indices = np.linspace(
        0,
        total_frames - 1,
        num=min(_WAVEFORM_LINE_POINT_LIMIT, total_frames),
        dtype=np.int64,
    )
    result: list[dict[str, object]] = []
    chunk_offset = 0
    target_offset = 0
    for chunk in chunks:
        frame_count = chunk.samples.shape[0]
        next_offset = chunk_offset + frame_count
        while (
            target_offset < target_indices.size
            and int(target_indices[target_offset]) < next_offset
        ):
            local_index = int(target_indices[target_offset]) - chunk_offset
            frame = chunk.samples[local_index]
            if np.issubdtype(frame.dtype, np.integer):
                frame_values = frame.astype(np.float64) / 2147483648.0
            else:
                frame_values = frame.astype(np.float64, copy=False)
            result.append(
                {
                    "captured_ns": chunk.start_ns
                    + local_index * (chunk.end_ns - chunk.start_ns) // frame_count,
                    "values": [float(value) for value in frame_values],
                }
            )
            target_offset += 1
        chunk_offset = next_offset
    return result


class DiagnosticRangeError(ValueError):
    """Raised when an audio request has been evicted from the live buffer."""

    def __init__(
        self,
        available_start_ns: int | None,
        available_end_ns: int | None,
        *,
        message: str = "requested audio range is no longer available",
    ) -> None:
        super().__init__(message)
        self.available_start_ns = available_start_ns
        self.available_end_ns = available_end_ns


class RealtimeCaptureControl:
    """Thread-safe requested capture state shared by the CLI and dashboard."""

    def __init__(self) -> None:
        self._paused = False
        self._generation = 0
        self._condition = threading.Condition()

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return {"paused": self._paused, "generation": self._generation}

    def pause(self) -> dict[str, object]:
        return self._set_paused(True)

    def resume(self) -> dict[str, object]:
        return self._set_paused(False)

    def wait_for_change(self, generation: int, timeout: float) -> dict[str, object]:
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ValueError("generation must be a nonnegative integer")
        if not isinstance(timeout, (int, float)) or not math.isfinite(float(timeout)) or timeout < 0:
            raise ValueError("timeout must be a nonnegative finite number")
        with self._condition:
            self._condition.wait_for(lambda: self._generation != generation, timeout=float(timeout))
            return {"paused": self._paused, "generation": self._generation}

    def _set_paused(self, paused: bool) -> dict[str, object]:
        with self._condition:
            if self._paused != paused:
                self._paused = paused
                self._generation += 1
                self._condition.notify_all()
            return {"paused": self._paused, "generation": self._generation}


class RealtimeCaptureModeControl:
    """Serialize shared/exclusive capture requests at a PCM-safe boundary."""

    _MODES = frozenset({"exclusive", "shared"})

    def __init__(self, *, initial_mode: str = "exclusive") -> None:
        self._require_mode(initial_mode)
        self._selected_mode = initial_mode
        self._requested_mode: str | None = None
        self._claimed_generation: int | None = None
        self._generation = 0
        self._error: str | None = None
        self._condition = threading.Condition()

    @classmethod
    def _require_mode(cls, mode: object) -> None:
        if not isinstance(mode, str) or mode not in cls._MODES:
            raise ValueError("capture_mode must be exclusive or shared")

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                "available": True,
                "selected_mode": self._selected_mode,
                "requested_mode": self._requested_mode,
                "generation": self._generation,
                "error": self._error,
            }

    def request(self, mode: str) -> dict[str, object]:
        self._require_mode(mode)
        with self._condition:
            if self._claimed_generation is not None and mode != self._requested_mode:
                raise ValueError("capture mode switch is already in progress")
            if mode != self._selected_mode and self._requested_mode != mode:
                self._requested_mode = mode
                self._claimed_generation = None
                self._generation += 1
                self._error = None
                self._condition.notify_all()
            return {"requested_mode": mode, "generation": self._generation}

    def claim_pending(self) -> dict[str, object] | None:
        with self._condition:
            if self._requested_mode is None or self._claimed_generation == self._generation:
                return None
            self._claimed_generation = self._generation
            return {"requested_mode": self._requested_mode, "generation": self._generation}

    def mark_applied(self, mode: str, *, generation: int) -> dict[str, object]:
        return self._finish(mode=mode, generation=generation, error=None)

    def mark_failed(self, mode: str, *, generation: int, error: str) -> dict[str, object]:
        if not isinstance(error, str) or not error:
            raise ValueError("capture mode switch error must be nonempty")
        return self._finish(mode=mode, generation=generation, error=error)

    def _finish(self, *, mode: str, generation: int, error: str | None) -> dict[str, object]:
        self._require_mode(mode)
        with self._condition:
            if (
                self._requested_mode != mode
                or self._claimed_generation != generation
                or self._generation != generation
            ):
                raise ValueError("capture mode request is no longer current")
            if error is None:
                self._selected_mode = mode
            self._requested_mode = None
            self._claimed_generation = None
            self._error = error
            self._condition.notify_all()
            return self.snapshot()

class RealtimeDeviceControl:
    """Serialize live input-device changes for application at an audio boundary."""

    def __init__(self, *, devices: list[Mapping[str, object]], selected_index: int) -> None:
        if isinstance(selected_index, bool) or not isinstance(selected_index, int) or selected_index < 0:
            raise ValueError("selected_index must be a nonnegative integer")
        normalized: list[dict[str, object]] = []
        indexes: set[int] = set()
        for device in devices:
            if not isinstance(device, Mapping):
                raise ValueError("devices must contain mappings")
            index = device.get("index")
            name = device.get("name")
            sample_rate = device.get("default_sample_rate")
            channels = device.get("max_input_channels")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError("device index must be a nonnegative integer")
            if index in indexes:
                raise ValueError("device indexes must be unique")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("device name must be nonempty")
            if (
                isinstance(sample_rate, bool)
                or not isinstance(sample_rate, (int, float))
                or not math.isfinite(float(sample_rate))
                or float(sample_rate) <= 0.0
            ):
                raise ValueError("device default_sample_rate must be positive and finite")
            if isinstance(channels, bool) or not isinstance(channels, int) or channels < 1:
                raise ValueError("device max_input_channels must be a positive integer")
            indexes.add(index)
            normalized.append(
                {
                    "index": index,
                    "name": name,
                    "default_sample_rate": float(sample_rate),
                    "max_input_channels": channels,
                }
            )
        if selected_index not in indexes:
            raise ValueError("selected input device is unavailable")
        self._devices = tuple(normalized)
        self._indexes = frozenset(indexes)
        self._selected_index = selected_index
        self._requested_index: int | None = None
        self._claimed_generation: int | None = None
        self._generation = 0
        self._error: str | None = None
        self._condition = threading.Condition()

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                "available": True,
                "devices": [dict(device) for device in self._devices],
                "selected_index": self._selected_index,
                "requested_index": self._requested_index,
                "generation": self._generation,
                "error": self._error,
            }

    def select(self, index: int) -> dict[str, int]:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("device_index must be a nonnegative integer")
        with self._condition:
            if index not in self._indexes:
                raise ValueError("requested input device is unavailable")
            if self._claimed_generation is not None and index != self._requested_index:
                raise ValueError("input device switch is already in progress")
            if index != self._selected_index:
                if self._requested_index != index:
                    self._requested_index = index
                    self._claimed_generation = None
                    self._generation += 1
                    self._error = None
                    self._condition.notify_all()
            return {"requested_index": index, "generation": self._generation}

    def claim_pending(self) -> dict[str, int] | None:
        with self._condition:
            if self._requested_index is None or self._claimed_generation == self._generation:
                return None
            self._claimed_generation = self._generation
            return {"requested_index": self._requested_index, "generation": self._generation}

    def mark_applied(self, index: int, *, generation: int) -> dict[str, object]:
        return self._finish(index=index, generation=generation, error=None)

    def mark_failed(self, index: int, *, generation: int, error: str) -> dict[str, object]:
        if not isinstance(error, str) or not error:
            raise ValueError("device switch error must be nonempty")
        return self._finish(index=index, generation=generation, error=error)

    def _finish(self, *, index: int, generation: int, error: str | None) -> dict[str, object]:
        with self._condition:
            if (
                self._requested_index != index
                or self._claimed_generation != generation
                or self._generation != generation
            ):
                raise ValueError("input device request is no longer current")
            if error is None:
                self._selected_index = index
            self._requested_index = None
            self._claimed_generation = None
            self._error = error
            self._condition.notify_all()
            return self.snapshot()


class RealtimeInputControl:
    """Thread-safe AGC enablement and RMS target control for live input."""

    def __init__(
        self,
        *,
        conditioner_enabled: bool = False,
        target_rms_dbfs: float = DEFAULT_RUNTIME_PROFILE.default_target_rms_dbfs,
    ) -> None:
        _require_bool("conditioner_enabled", conditioner_enabled)
        self._conditioner_enabled = conditioner_enabled
        self._validate_target_rms_dbfs(target_rms_dbfs)
        self._target_rms_dbfs = float(target_rms_dbfs)
        self._generation = 0
        self._condition = threading.Condition()

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                "conditioner_enabled": self._conditioner_enabled,
                "target_rms_dbfs": self._target_rms_dbfs,
                "generation": self._generation,
            }

    def set_target_rms_dbfs(self, target_rms_dbfs: float) -> dict[str, object]:
        """Update only the target while preserving the requested AGC mode."""

        self._validate_target_rms_dbfs(target_rms_dbfs)
        with self._condition:
            if self._target_rms_dbfs != float(target_rms_dbfs):
                self._target_rms_dbfs = float(target_rms_dbfs)
                self._generation += 1
                self._condition.notify_all()
            return {
                "conditioner_enabled": self._conditioner_enabled,
                "target_rms_dbfs": self._target_rms_dbfs,
                "generation": self._generation,
            }

    def set_config(
        self,
        *,
        conditioner_enabled: bool,
        target_rms_dbfs: float,
    ) -> dict[str, object]:
        """Apply one complete live-input configuration atomically."""

        _require_bool("conditioner_enabled", conditioner_enabled)
        requested_enabled = conditioner_enabled
        self._validate_target_rms_dbfs(target_rms_dbfs)
        requested_target = float(target_rms_dbfs)
        with self._condition:
            if (self._conditioner_enabled, self._target_rms_dbfs) != (
                requested_enabled,
                requested_target,
            ):
                self._conditioner_enabled = requested_enabled
                self._target_rms_dbfs = requested_target
                self._generation += 1
                self._condition.notify_all()
            return {
                "conditioner_enabled": self._conditioner_enabled,
                "target_rms_dbfs": self._target_rms_dbfs,
                "generation": self._generation,
            }

    @classmethod
    def _validate_target_rms_dbfs(cls, target_rms_dbfs: object) -> None:
        if (
            isinstance(target_rms_dbfs, bool)
            or not isinstance(target_rms_dbfs, (int, float))
            or not math.isfinite(float(target_rms_dbfs))
            or not float(target_rms_dbfs).is_integer()
            or not _MIN_TARGET_RMS_DBFS <= float(target_rms_dbfs) <= _MAX_TARGET_RMS_DBFS
        ):
            raise ValueError(
                f"target_rms_dbfs must be an integer within [{_MIN_TARGET_RMS_DBFS}, {_MAX_TARGET_RMS_DBFS}]"
            )


class AnalysisBusyError(RuntimeError):
    """Raised when the dashboard already owns an analysis command."""


@dataclass(frozen=True)
class RealtimeAnalysisRequest:
    """One immutable PCM command awaiting CLI-thread analysis."""

    request_id: int
    source: str
    pcm: np.ndarray
    sample_rate: int = 16000


class RealtimeAnalysisControl:
    """Serialize imported and selected-range analysis without sharing model state."""

    def __init__(self, *, maximum_samples: int) -> None:
        if isinstance(maximum_samples, bool) or not isinstance(maximum_samples, int) or maximum_samples < 1:
            raise ValueError("maximum_samples must be a positive integer")
        self.maximum_samples = maximum_samples
        self._next_request_id = 1
        self._queued: RealtimeAnalysisRequest | None = None
        self._running: RealtimeAnalysisRequest | None = None
        self._status: dict[str, object] = {
            "request_id": None,
            "source": None,
            "sample_count": 0,
            "state": "idle",
            "message": "Ready",
        }
        self._condition = threading.Condition()

    def enqueue(self, *, source: str, pcm: np.ndarray, sample_rate: int = 16000) -> dict[str, object]:
        if source not in {"import", "reanalysis"}:
            raise ValueError("analysis source must be import or reanalysis")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("analysis sample_rate must be a positive integer")
        values = np.asarray(pcm)
        if (
            values.dtype != np.float32
            or values.ndim not in (1, 2)
            or values.shape[0] < 1
            or (values.ndim == 2 and values.shape[1] not in (1, 2))
        ):
            raise ValueError("analysis PCM must be bounded float32 mono/stereo audio")
        if not _fits_analysis_duration(
            frame_count=values.shape[0],
            sample_rate=sample_rate,
            maximum_samples=self.maximum_samples,
        ):
            raise ValueError("analysis duration exceeds the diagnostic history window")
        if not np.all(np.isfinite(values)):
            raise ValueError("analysis PCM must contain only finite values")
        copied = values.copy()
        copied.setflags(write=False)
        with self._condition:
            if self._queued is not None or self._running is not None:
                raise AnalysisBusyError("an analysis command is already queued or running")
            request = RealtimeAnalysisRequest(
                request_id=self._next_request_id,
                source=source,
                pcm=copied,
                sample_rate=sample_rate,
            )
            self._next_request_id += 1
            self._queued = request
            self._status = self._status_for(request, state="queued", message=f"{source} queued")
            self._condition.notify_all()
            return dict(self._status)

    def claim_next(self) -> RealtimeAnalysisRequest | None:
        with self._condition:
            if self._queued is None:
                return None
            request = self._queued
            self._queued = None
            self._running = request
            self._status = self._status_for(request, state="running", message=f"{request.source} running")
            self._condition.notify_all()
            return request

    def complete(self, request_id: int) -> dict[str, object]:
        return self._finish(request_id, state="complete", message="analysis complete")

    def defer(self, request_id: int) -> dict[str, object]:
        """Return one claimed request to the pending slot without changing its identity."""

        with self._condition:
            request = self._running
            if request is None or request.request_id != request_id:
                raise ValueError("analysis request is not running")
            self._running = None
            self._queued = request
            self._status = self._status_for(request, state="queued", message=f"{request.source} queued")
            self._condition.notify_all()
            return dict(self._status)

    def fail(self, request_id: int, message: str) -> dict[str, object]:
        if not isinstance(message, str) or not message:
            raise ValueError("analysis failure message must be nonempty")
        return self._finish(request_id, state="failed", message=message)

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return dict(self._status)

    def _finish(self, request_id: int, *, state: str, message: str) -> dict[str, object]:
        with self._condition:
            request = self._running
            if request is None or request.request_id != request_id:
                raise ValueError("analysis request is not running")
            self._running = None
            self._status = self._status_for(request, state=state, message=message)
            self._condition.notify_all()
            return dict(self._status)

    @staticmethod
    def _status_for(request: RealtimeAnalysisRequest, *, state: str, message: str) -> dict[str, object]:
        return {
            "request_id": request.request_id,
            "source": request.source,
            "sample_count": int(request.pcm.shape[0]),
            "sample_rate": request.sample_rate,
            "state": state,
            "message": message,
        }


class RealtimeThresholdControl:
    """Thread-safe requested runtime thresholds and gate modes for one session."""

    def __init__(
        self,
        *,
        vad_threshold: float,
        kws_threshold: float,
        energy_enabled: bool = True,
        vad_enabled: bool = True,
        vad_period_ms: int = 32,
        kws_period_ms: int | None = None,
        kws_lookback_ms: int = 1500,
    ) -> None:
        _require_probability("vad_threshold", vad_threshold)
        _require_probability("kws_threshold", kws_threshold)
        _require_bool("energy_enabled", energy_enabled)
        _require_bool("vad_enabled", vad_enabled)
        schedule = TimingSchedule.from_periods(
            vad_period_ms,
            vad_period_ms * 3 if kws_period_ms is None else kws_period_ms,
        )
        lookback_ms = _require_kws_lookback_ms(kws_lookback_ms)
        self._vad_threshold = float(vad_threshold)
        self._kws_threshold = float(kws_threshold)
        self._energy_enabled = energy_enabled
        self._vad_enabled = vad_enabled
        self._schedule = schedule
        self._kws_lookback_ms = lookback_ms
        self._generation = 0
        self._condition = threading.Condition()

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                "vad_threshold": self._vad_threshold,
                "kws_threshold": self._kws_threshold,
                "energy_enabled": self._energy_enabled,
                "vad_enabled": self._vad_enabled,
                **_schedule_fields(self._schedule),
                "kws_lookback_ms": self._kws_lookback_ms,
                "generation": self._generation,
            }

    def set_thresholds(self, *, vad_threshold: float, kws_threshold: float) -> dict[str, object]:
        """Update only thresholds, preserving gate modes for compatibility."""

        _require_probability("vad_threshold", vad_threshold)
        _require_probability("kws_threshold", kws_threshold)
        requested_vad = float(vad_threshold)
        requested_kws = float(kws_threshold)
        with self._condition:
            if (self._vad_threshold, self._kws_threshold) != (requested_vad, requested_kws):
                self._vad_threshold = requested_vad
                self._kws_threshold = requested_kws
                self._generation += 1
                self._condition.notify_all()
            return {
                "vad_threshold": self._vad_threshold,
                "kws_threshold": self._kws_threshold,
                "generation": self._generation,
            }

    def set_runtime_config(
        self,
        *,
        vad_threshold: float,
        kws_threshold: float,
        energy_enabled: bool,
        vad_enabled: bool,
        vad_period_ms: int,
        kws_period_ms: int | None = None,
        kws_lookback_ms: int = 1500,
    ) -> dict[str, object]:
        _require_probability("vad_threshold", vad_threshold)
        _require_probability("kws_threshold", kws_threshold)
        _require_bool("energy_enabled", energy_enabled)
        _require_bool("vad_enabled", vad_enabled)
        schedule = TimingSchedule.from_periods(
            vad_period_ms,
            vad_period_ms * 3 if kws_period_ms is None else kws_period_ms,
        )
        lookback_ms = _require_kws_lookback_ms(kws_lookback_ms)
        requested_vad = float(vad_threshold)
        requested_kws = float(kws_threshold)
        with self._condition:
            if (
                self._vad_threshold,
                self._kws_threshold,
                self._energy_enabled,
                self._vad_enabled,
                self._schedule,
                self._kws_lookback_ms,
            ) != (
                requested_vad,
                requested_kws,
                energy_enabled,
                vad_enabled,
                schedule,
                lookback_ms,
            ):
                self._vad_threshold = requested_vad
                self._kws_threshold = requested_kws
                self._energy_enabled = energy_enabled
                self._vad_enabled = vad_enabled
                self._schedule = schedule
                self._kws_lookback_ms = lookback_ms
                self._generation += 1
                self._condition.notify_all()
            return {
                "vad_threshold": self._vad_threshold,
                "kws_threshold": self._kws_threshold,
                "energy_enabled": self._energy_enabled,
                "vad_enabled": self._vad_enabled,
                **_schedule_fields(self._schedule),
                "kws_lookback_ms": self._kws_lookback_ms,
                "generation": self._generation,
            }


class RealtimeDiagnosticStore:
    """Thread-safe rolling PCM and decision history with a monotonic time axis."""

    def __init__(
        self,
        *,
        history_seconds: int,
        vad_threshold: float,
        kws_threshold: float,
        vad_confirmations: int,
        kws_confirmations: int,
        sample_rate: int = 16000,
        vad_period_ms: int = 32,
        kws_period_ms: int | None = None,
        kws_lookback_ms: int = 1500,
        input_profile: RuntimeProfile = DEFAULT_RUNTIME_PROFILE,
    ) -> None:
        if isinstance(history_seconds, bool) or not isinstance(history_seconds, int) or history_seconds <= 0:
            raise ValueError("history_seconds must be a positive integer")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        for name, value in (
            ("vad_confirmations", vad_confirmations),
            ("kws_confirmations", kws_confirmations),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._validate_probability("vad_threshold", vad_threshold)
        self._validate_probability("kws_threshold", kws_threshold)
        if not isinstance(input_profile, RuntimeProfile):
            raise ValueError("input_profile must be a RuntimeProfile")
        schedule = TimingSchedule.from_periods(
            vad_period_ms,
            vad_period_ms * 3 if kws_period_ms is None else kws_period_ms,
            sample_rate_hz=sample_rate,
        )
        lookback_ms = _require_kws_lookback_ms(kws_lookback_ms, sample_rate=sample_rate)
        self.history_seconds = history_seconds
        self.sample_rate = sample_rate
        self._capacity = history_seconds * sample_rate
        self._thresholds = {
            "vad": float(vad_threshold),
            "kws": float(kws_threshold),
            "energy_dbfs": float(input_profile.energy_threshold_dbfs),
        }
        self._gates = {"energy_enabled": True, "vad_enabled": True}
        self._schedule = schedule
        self._kws_lookback_ms = lookback_ms
        self._confirmations = {"vad": vad_confirmations, "kws": kws_confirmations}
        self._input_profile = input_profile
        self._pcm = np.empty(0, dtype=np.float32)
        self._raw_device = np.empty((0, 1), dtype=np.float32)
        self._raw_device_chunks: deque[_RawDeviceChunk] = deque()
        self._raw_device_frames = 0
        self._raw_device_sample_rate: int | None = None
        self._raw_device_channels = 1
        self._raw_device_start_ns: int | None = None
        self._raw_device_end_ns: int | None = None
        self._audio_start_ns: int | None = None
        self._audio_end_ns: int | None = None
        self._events: deque[dict[str, object]] = deque()
        self._state_at_history_start = "idle"
        self._latest: dict[str, object] = {"state": "idle", "queue_depth": 0, "drop_count": 0}
        self._input_status: dict[str, object] = {}
        self._model_identity: dict[str, object] = {
            "kws_model_path": None,
            "kws_model_sha256": None,
            "kws_input_ms": 1000,
            "kws_input_samples": 16_000,
        }
        self._generation = 0
        self._condition = threading.Condition()

    @property
    def capacity_samples(self) -> int:
        return self._capacity

    def append_pcm(self, pcm: np.ndarray, *, captured_ns: int) -> None:
        values = np.asarray(pcm)
        if values.dtype != np.float32 or values.ndim != 1 or values.size == 0:
            raise ValueError("PCM must be a nonempty float32 mono array")
        if not np.all(np.isfinite(values)):
            raise ValueError("PCM must contain only finite values")
        self._validate_timestamp(captured_ns)
        with self._condition:
            if self._audio_end_ns is not None and captured_ns < self._audio_end_ns:
                raise ValueError("PCM capture timestamps must be monotonic")
            self._pcm = np.concatenate((self._pcm, values))
            if self._pcm.size > self._capacity:
                self._pcm = self._pcm[-self._capacity :].copy()
            self._audio_end_ns = captured_ns
            self._audio_start_ns = captured_ns - self._pcm.size * 1_000_000_000 // self.sample_rate
            self._prune_events_locked()
            self._advance_generation_locked()

    def append_raw_device(self, samples: np.ndarray, *, sample_rate: int, captured_ns: int) -> None:
        source = np.asarray(samples)
        is_integer = np.issubdtype(source.dtype, np.integer)
        values = source.copy() if is_integer else np.asarray(source, dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 1:
            raise ValueError("raw device audio must be nonempty channel-last samples")
        if sample_rate <= 0 or not np.all(np.isfinite(values)):
            raise ValueError("raw device audio has invalid sample rate or samples")
        self._validate_timestamp(captured_ns)
        with self._condition:
            if self._raw_device_sample_rate not in (None, sample_rate) or (
                self._raw_device_chunks and self._raw_device_channels != values.shape[1]
            ):
                raise ValueError("raw device format changed during one stream")
            self._raw_device_sample_rate = int(sample_rate)
            self._raw_device_channels = values.shape[1]
            start_ns = captured_ns - values.shape[0] * 1_000_000_000 // sample_rate
            self._raw_device_chunks.append(
                _RawDeviceChunk(samples=values, start_ns=start_ns, end_ns=captured_ns)
            )
            self._raw_device_frames += values.shape[0]
            capacity = max(1, round(self.history_seconds * sample_rate))
            while self._raw_device_frames > capacity and self._raw_device_chunks:
                oldest = self._raw_device_chunks[0]
                excess = self._raw_device_frames - capacity
                if oldest.samples.shape[0] <= excess:
                    self._raw_device_chunks.popleft()
                    self._raw_device_frames -= oldest.samples.shape[0]
                    continue
                retained = oldest.samples[int(excess) :].copy()
                retained_start_ns = oldest.start_ns + excess * 1_000_000_000 // sample_rate
                self._raw_device_chunks[0] = _RawDeviceChunk(
                    samples=retained,
                    start_ns=retained_start_ns,
                    end_ns=oldest.end_ns,
                )
                self._raw_device_frames -= excess
            self._raw_device_end_ns = captured_ns
            self._raw_device_start_ns = self._raw_device_chunks[0].start_ns
            self._advance_generation_locked()

    def set_raw_device_audio(self, samples: np.ndarray, *, sample_rate: int, end_ns: int) -> None:
        """Replace the download-only source track with one imported WAV payload."""
        self.clear_raw_device_audio()
        self.append_raw_device(samples, sample_rate=sample_rate, captured_ns=end_ns)

    def clear_raw_device_audio(self) -> None:
        with self._condition:
            self._raw_device = np.empty((0, 1), dtype=np.float32)
            self._raw_device_chunks.clear()
            self._raw_device_frames = 0
            self._raw_device_sample_rate = None
            self._raw_device_channels = 1
            self._raw_device_start_ns = None
            self._raw_device_end_ns = None
            self._advance_generation_locked()

    def append_event(self, event: CascadeEvent, runtime: Mapping[str, object]) -> None:
        if not isinstance(event, CascadeEvent):
            raise ValueError("event must be a CascadeEvent")
        self._validate_timestamp(event.captured_ns)
        record = {"kind": event.kind, "captured_ns": event.captured_ns}
        record.update(_json_safe_mapping(event.fields))
        runtime_record = _json_safe_mapping(runtime)
        with self._condition:
            self._events.append(record)
            self._latest.update(runtime_record)
            for key in (
                "energy_dbfs",
                "energy_gate",
                "energy_gate_effective",
                "energy_gate_enabled",
                "vad_gate_enabled",
            ):
                if key in record:
                    self._latest[key] = record[key]
            if "state" in record:
                self._latest["state"] = record["state"]
            if event.kind == "state_transition":
                state = record.get("to_state")
                if isinstance(state, str):
                    self._latest["state"] = state
            self._prune_events_locked()
            self._advance_generation_locked()

    def clear(self) -> None:
        """Discard retained PCM/events so resumed capture starts a fresh segment."""

        with self._condition:
            self._pcm = np.empty(0, dtype=np.float32)
            self._raw_device = np.empty((0, 1), dtype=np.float32)
            self._raw_device_chunks.clear()
            self._raw_device_frames = 0
            self._raw_device_sample_rate = None
            self._raw_device_channels = 1
            self._raw_device_start_ns = None
            self._raw_device_end_ns = None
            self._audio_start_ns = None
            self._audio_end_ns = None
            self._events.clear()
            self._state_at_history_start = "idle"
            self._latest = {"state": "idle", "queue_depth": 0, "drop_count": 0}
            self._advance_generation_locked()

    def set_thresholds(self, *, vad_threshold: float, kws_threshold: float) -> None:
        """Publish effective session thresholds after the audio loop accepts them."""

        _require_probability("vad_threshold", vad_threshold)
        _require_probability("kws_threshold", kws_threshold)
        with self._condition:
            self._thresholds["vad"] = float(vad_threshold)
            self._thresholds["kws"] = float(kws_threshold)
            self._advance_generation_locked()

    def set_runtime_config(
        self,
        *,
        vad_threshold: float,
        kws_threshold: float,
        energy_enabled: bool,
        vad_enabled: bool,
        vad_period_ms: int,
        kws_period_ms: int | None = None,
        kws_lookback_ms: int = 1500,
    ) -> None:
        """Publish all effective runtime settings after a safe audio-boundary transition."""

        _require_probability("vad_threshold", vad_threshold)
        _require_probability("kws_threshold", kws_threshold)
        _require_bool("energy_enabled", energy_enabled)
        _require_bool("vad_enabled", vad_enabled)
        schedule = TimingSchedule.from_periods(
            vad_period_ms,
            vad_period_ms * 3 if kws_period_ms is None else kws_period_ms,
            sample_rate_hz=self.sample_rate,
        )
        lookback_ms = _require_kws_lookback_ms(
            kws_lookback_ms, sample_rate=self.sample_rate
        )
        with self._condition:
            self._thresholds["vad"] = float(vad_threshold)
            self._thresholds["kws"] = float(kws_threshold)
            self._gates = {"energy_enabled": energy_enabled, "vad_enabled": vad_enabled}
            self._schedule = schedule
            self._kws_lookback_ms = lookback_ms
            self._advance_generation_locked()

    def runtime_config(self) -> dict[str, object]:
        """Return the currently published effective runtime configuration."""

        with self._condition:
            return {
                "vad_threshold": self._thresholds["vad"],
                "kws_threshold": self._thresholds["kws"],
                **self._gates,
                **_schedule_fields(self._schedule),
                "kws_lookback_ms": self._kws_lookback_ms,
            }

    def set_model_identity(self, *, kws_model_path: str, kws_model_sha256: str) -> None:
        if not isinstance(kws_model_path, str) or not kws_model_path:
            raise ValueError("kws_model_path must be a nonempty string")
        if not isinstance(kws_model_sha256, str) or len(kws_model_sha256) != 64:
            raise ValueError("kws_model_sha256 must be a SHA-256 string")
        with self._condition:
            self._model_identity = {
                "kws_model_path": kws_model_path,
                "kws_model_sha256": kws_model_sha256,
                "kws_input_ms": 1000,
                "kws_input_samples": 16_000,
            }
            self._advance_generation_locked()

    def notify_update(self) -> None:
        """Wake SSE clients after metadata outside the diagnostic store changes."""

        with self._condition:
            self._advance_generation_locked()

    def set_input_status(self, status: Mapping[str, object]) -> None:
        """Publish applied spatial, AGC, and executor diagnostics without changing the timeline."""

        if not isinstance(status, Mapping):
            raise ValueError("input status must be a mapping")
        safe_status = _json_safe_mapping(status)
        with self._condition:
            if safe_status != self._input_status:
                self._input_status = safe_status
                self._advance_generation_locked()

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            start_ns = self._audio_start_ns
            end_ns = self._audio_end_ns
            retained = list(self._events)
            events = [
                record
                for record in retained
                if start_ns is None or int(record["captured_ns"]) >= start_ns
            ]
            vad_points = [
                self._score_point(record)
                for record in events
                if record["kind"] == "vad_decision"
            ]
            kws_points = [
                self._score_point(record)
                for record in events
                if record["kind"] == "kws_call"
            ]
            wakes = [
                {
                    "captured_ns": record["captured_ns"],
                    "score": record.get("score"),
                    "confirmations": record.get("confirmations"),
                }
                for record in events
                if record["kind"] == "wake"
            ]
            return {
                "generation": self._generation,
                "sample_rate": self.sample_rate,
                "available_start_ns": start_ns,
                "available_end_ns": end_ns,
                "thresholds": {
                    "vad": self._thresholds["vad"],
                    "kws": self._thresholds["kws"],
                },
                "runtime_config": {
                    "vad_threshold": self._thresholds["vad"],
                    "kws_threshold": self._thresholds["kws"],
                    **self._gates,
                    **_schedule_fields(self._schedule),
                    "kws_lookback_ms": self._kws_lookback_ms,
                },
                "confirmations_required": dict(self._confirmations),
                "latest": dict(self._latest),
                "input_status": dict(self._input_status),
                "model_identity": dict(self._model_identity),
                "waveform": self._waveform_envelope_locked(),
                "vad_points": vad_points,
                "kws_points": kws_points,
                "state_intervals": self._state_intervals(events, start_ns, end_ns),
                "wake_markers": wakes,
            }

    def dashboard_snapshot(
        self,
        *,
        max_score_points: int = 960,
        agc_preview_target_rms_dbfs: float | None = None,
    ) -> dict[str, object]:
        """Return a bounded, drawing-oriented view without full event metadata."""

        if (
            isinstance(max_score_points, bool)
            or not isinstance(max_score_points, int)
            or max_score_points < 2
        ):
            raise ValueError("max_score_points must be an integer of at least 2")
        if agc_preview_target_rms_dbfs is not None and (
            isinstance(agc_preview_target_rms_dbfs, bool)
            or not isinstance(agc_preview_target_rms_dbfs, (int, float))
            or not math.isfinite(float(agc_preview_target_rms_dbfs))
        ):
            raise ValueError("agc_preview_target_rms_dbfs must be finite or None")
        with self._condition:
            has_raw_device_track = (
                self._raw_device_sample_rate is not None
                and bool(self._raw_device_chunks)
                and self._raw_device_start_ns is not None
                and self._raw_device_end_ns is not None
            )
            start_ns = self._raw_device_start_ns if has_raw_device_track else self._audio_start_ns
            end_ns = self._raw_device_end_ns if has_raw_device_track else self._audio_end_ns
            display_sample_rate = (
                int(self._raw_device_sample_rate) if has_raw_device_track else self.sample_rate
            )
            display_channels = self._raw_device_channels if has_raw_device_track else 1
            display_waveform = (
                _raw_device_waveform_line_points(self._raw_device_chunks)
                if has_raw_device_track
                else _waveform_line_points(
                    self._pcm,
                    audio_start_ns=start_ns,
                    sample_rate=display_sample_rate,
                )
            )
            preview_pcm = (
                self._pcm.copy() if agc_preview_target_rms_dbfs is not None else None
            )
            events = [
                record
                for record in self._events
                if start_ns is None or int(record["captured_ns"]) >= start_ns
            ]
            schedule = _schedule_fields(self._schedule)
            vad_points = self._decimate_points(
                self._dashboard_points_with_gaps(
                    [
                        record
                        for record in events
                        if record["kind"] == "vad_decision"
                    ],
                    self._dashboard_score_point,
                    cadence_ms=schedule["vad_period_ms"],
                ),
                max_score_points,
                value_key="score",
            )
            kws_points = self._decimate_points(
                self._dashboard_points_with_gaps(
                    [
                        record
                        for record in events
                        if record["kind"] == "kws_call"
                    ],
                    self._dashboard_score_point,
                    cadence_ms=schedule["kws_period_ms"],
                ),
                max_score_points,
                value_key="score",
            )
            energy_points = self._decimate_points(
                self._dashboard_points_with_gaps(
                    [
                        record
                        for record in events
                        if record["kind"] in ("energy_decision", "vad_decision")
                        and isinstance(record.get("energy_dbfs"), (int, float))
                    ],
                    self._dashboard_energy_point,
                    cadence_ms=schedule["energy_period_ms"],
                ),
                max_score_points,
                value_key="energy_dbfs",
            )
            state_intervals = self._coalesced_dashboard_state_intervals(
                self._state_intervals(events, start_ns, end_ns)
            )
            wakes = [
                {
                    "captured_ns": record["captured_ns"],
                    "score": record.get("score"),
                    "confirmations": record.get("confirmations"),
                }
                for record in events
                if record["kind"] == "wake"
            ]
            payload = {
                "generation": self._generation,
                "sample_rate": display_sample_rate,
                "channels": display_channels,
                "display_track": "raw_device" if has_raw_device_track else "model_input",
                "available_start_ns": start_ns,
                "available_end_ns": end_ns,
                "thresholds": dict(self._thresholds),
                "runtime_config": {
                    "vad_threshold": self._thresholds["vad"],
                    "kws_threshold": self._thresholds["kws"],
                    **self._gates,
                    **_schedule_fields(self._schedule),
                    "kws_lookback_ms": self._kws_lookback_ms,
                },
                "confirmations_required": dict(self._confirmations),
                "latest": dict(self._latest),
                "input_status": dict(self._input_status),
                "model_identity": dict(self._model_identity),
                "waveform": display_waveform,
                "vad_points": vad_points,
                "kws_points": kws_points,
                "energy_points": energy_points,
                "state_intervals": state_intervals,
                "wake_markers": wakes,
            }
        if preview_pcm is not None:
            preview = _render_agc_preview(
                preview_pcm,
                profile=self._input_profile,
                target_rms_dbfs=float(agc_preview_target_rms_dbfs),
            )
            payload["agc_preview_waveform"] = _waveform_line_points(
                preview,
                audio_start_ns=start_ns,
                sample_rate=self.sample_rate,
            )
        return payload

    def pcm_range(self, start_ns: int, end_ns: int) -> np.ndarray:
        self._validate_timestamp(start_ns)
        self._validate_timestamp(end_ns)
        if start_ns >= end_ns:
            raise ValueError("audio start_ns must be earlier than end_ns")
        with self._condition:
            selected = self._selected_pcm_locked(start_ns, end_ns)
        if selected.size == 0:
            raise ValueError("audio range resolves to zero samples")
        return selected

    def analysis_range(self, start_ns: int, end_ns: int) -> tuple[np.ndarray, int]:
        """Return the visible audio selection for model reanalysis."""

        self._validate_timestamp(start_ns)
        self._validate_timestamp(end_ns)
        if start_ns >= end_ns:
            raise ValueError("audio start_ns must be earlier than end_ns")
        with self._condition:
            if self._raw_device_sample_rate is not None and self._raw_device_chunks:
                selected = self._selected_raw_device_locked(start_ns, end_ns)
                sample_rate = int(self._raw_device_sample_rate)
            else:
                selected = self._selected_pcm_locked(start_ns, end_ns)
                sample_rate = self.sample_rate
        if selected.size == 0:
            raise ValueError("audio range resolves to zero samples")
        if np.issubdtype(selected.dtype, np.integer):
            selected = selected.astype(np.float32) / 2147483648.0
        return np.asarray(selected, dtype=np.float32), sample_rate

    def audio_bytes(self, start_ns: int, end_ns: int) -> bytes:
        with self._condition:
            if self._raw_device_sample_rate is not None and self._raw_device_chunks:
                selected = self._selected_raw_device_locked(start_ns, end_ns)
                return _pcm_wav(selected, self._raw_device_sample_rate, sample_width=3)
            selected = self._selected_pcm_locked(start_ns, end_ns)
        return _pcm16_wav(selected, self.sample_rate)

    def agc_preview_pcm(self, *, target_rms_dbfs: float) -> np.ndarray:
        """Recalculate a current-settings AGC preview from retained raw PCM."""

        with self._condition:
            values = self._pcm.copy()
        return _render_agc_preview(
            values,
            profile=self._input_profile,
            target_rms_dbfs=target_rms_dbfs,
        )

    def agc_preview_waveform(self, *, target_rms_dbfs: float) -> list[dict[str, object]]:
        """Return aligned line-waveform samples for a transient AGC preview."""

        with self._condition:
            values = self._pcm.copy()
            start_ns = self._audio_start_ns
        preview = _render_agc_preview(
            values,
            profile=self._input_profile,
            target_rms_dbfs=target_rms_dbfs,
        )
        return _waveform_line_points(preview, audio_start_ns=start_ns, sample_rate=self.sample_rate)

    def agc_preview_audio_bytes(
        self,
        start_ns: int,
        end_ns: int,
        *,
        target_rms_dbfs: float,
    ) -> bytes:
        """Render one playback range without retaining processed PCM."""

        self._validate_timestamp(start_ns)
        self._validate_timestamp(end_ns)
        if start_ns >= end_ns:
            raise ValueError("audio start_ns must be earlier than end_ns")
        with self._condition:
            available_start = self._audio_start_ns
            available_end = self._audio_end_ns
            if (
                available_start is None
                or available_end is None
                or start_ns < available_start
                or end_ns > available_end
            ):
                raise DiagnosticRangeError(available_start, available_end)
            start_index = (start_ns - available_start) * self.sample_rate // 1_000_000_000
            end_index = (end_ns - available_start) * self.sample_rate // 1_000_000_000
            history = self._pcm[: int(end_index)].copy()
        preview = _render_agc_preview(
            history,
            profile=self._input_profile,
            target_rms_dbfs=target_rms_dbfs,
        )
        selected = preview[int(start_index) : int(end_index)]
        if selected.size == 0:
            raise ValueError("audio range resolves to zero samples")
        return _pcm16_wav(selected, self.sample_rate)

    def wait_for_update(self, generation: int, timeout: float) -> int:
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ValueError("generation must be a nonnegative integer")
        if not isinstance(timeout, (int, float)) or not math.isfinite(float(timeout)) or timeout < 0:
            raise ValueError("timeout must be a nonnegative finite number")
        with self._condition:
            self._condition.wait_for(lambda: self._generation != generation, timeout=float(timeout))
            return self._generation

    def _prune_events_locked(self) -> None:
        if self._audio_start_ns is None:
            return
        while self._events and int(self._events[0]["captured_ns"]) < self._audio_start_ns:
            removed = self._events.popleft()
            if removed["kind"] == "state_transition":
                state = removed.get("to_state")
                if isinstance(state, str):
                    self._state_at_history_start = state

    def _selected_pcm_locked(self, start_ns: int, end_ns: int) -> np.ndarray:
        available_start = self._audio_start_ns
        available_end = self._audio_end_ns
        if (
            available_start is None
            or available_end is None
            or start_ns < available_start
            or end_ns > available_end
        ):
            raise DiagnosticRangeError(available_start, available_end)
        start_index = (start_ns - available_start) * self.sample_rate // 1_000_000_000
        end_index = (end_ns - available_start) * self.sample_rate // 1_000_000_000
        return self._pcm[int(start_index) : int(end_index)].copy()

    def _selected_raw_device_locked(self, start_ns: int, end_ns: int) -> np.ndarray:
        available_start = self._raw_device_start_ns
        available_end = self._raw_device_end_ns
        if available_start is None or available_end is None or start_ns < available_start or end_ns > available_end:
            raise DiagnosticRangeError(available_start, available_end)
        rate = int(self._raw_device_sample_rate)
        selected: list[np.ndarray] = []
        cursor_ns = start_ns
        tolerated_ns = max(1, 1_000_000_000 // rate)
        for chunk in self._raw_device_chunks:
            if chunk.end_ns <= start_ns:
                continue
            if chunk.start_ns >= end_ns:
                break
            if chunk.start_ns - cursor_ns > tolerated_ns:
                raise DiagnosticRangeError(
                    available_start,
                    available_end,
                    message="requested audio range crosses a capture gap",
                )
            overlap_start_ns = max(start_ns, chunk.start_ns)
            overlap_end_ns = min(end_ns, chunk.end_ns)
            if overlap_end_ns <= overlap_start_ns:
                continue
            span_ns = max(1, chunk.end_ns - chunk.start_ns)
            left = (overlap_start_ns - chunk.start_ns) * chunk.samples.shape[0] // span_ns
            right = (overlap_end_ns - chunk.start_ns) * chunk.samples.shape[0] // span_ns
            right = max(left + 1, min(chunk.samples.shape[0], right))
            if right > left:
                selected.append(chunk.samples[left:right])
            cursor_ns = overlap_end_ns
        if end_ns - cursor_ns > tolerated_ns:
            raise DiagnosticRangeError(
                available_start,
                available_end,
                message="requested audio range crosses a capture gap",
            )
        if not selected:
            return np.empty((0, self._raw_device_channels), dtype=np.float32)
        if len(selected) == 1:
            return selected[0].copy()
        return np.concatenate(selected, axis=0)

    def _advance_generation_locked(self) -> None:
        self._generation += 1
        self._condition.notify_all()

    def _waveform_envelope_locked(self) -> list[dict[str, object]]:
        return _waveform_envelope(
            self._pcm,
            audio_start_ns=self._audio_start_ns,
            sample_rate=self.sample_rate,
        )

    def _state_intervals(
        self, events: list[dict[str, object]], start_ns: int | None, end_ns: int | None
    ) -> list[dict[str, object]]:
        if start_ns is None or end_ns is None or end_ns <= start_ns:
            return []
        state = self._state_at_history_start
        cursor = start_ns
        intervals: list[dict[str, object]] = []
        for record in events:
            if record["kind"] != "state_transition":
                continue
            transition_ns = int(record["captured_ns"])
            if transition_ns > cursor:
                intervals.append({"state": state, "start_ns": cursor, "end_ns": transition_ns})
            next_state = record.get("to_state")
            if isinstance(next_state, str):
                state = next_state
            cursor = max(cursor, transition_ns)
        if cursor < end_ns:
            intervals.append({"state": state, "start_ns": cursor, "end_ns": end_ns})
        return intervals

    @staticmethod
    def _coalesced_dashboard_state_intervals(
        intervals: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Hide brief visual-only state reversals without changing recorded events."""

        displayed = [dict(interval) for interval in intervals]
        index = 1
        while index < len(displayed) - 1:
            previous = displayed[index - 1]
            current = displayed[index]
            following = displayed[index + 1]
            duration_ns = int(current["end_ns"]) - int(current["start_ns"])
            if (
                duration_ns < _DASHBOARD_STATE_MIN_DURATION_NS
                and previous["state"] == following["state"]
            ):
                previous["end_ns"] = following["end_ns"]
                del displayed[index : index + 2]
                index = max(1, index - 1)
                continue
            index += 1
        if len(displayed) > 1:
            trailing = displayed[-1]
            duration_ns = int(trailing["end_ns"]) - int(trailing["start_ns"])
            if duration_ns < _DASHBOARD_STATE_MIN_DURATION_NS:
                displayed[-2]["end_ns"] = trailing["end_ns"]
                displayed.pop()
        return displayed

    @staticmethod
    def _score_point(record: Mapping[str, object]) -> dict[str, object]:
        return {
            "captured_ns": record["captured_ns"],
            "score": record.get("score"),
            "model_input_start_ns": record.get("model_input_start_ns"),
            "model_input_end_ns": record.get("model_input_end_ns"),
            "state": record.get("state"),
            "kws_gate_open": record.get("kws_gate_open"),
            "energy_dbfs": record.get("energy_dbfs"),
            "energy_gate": record.get("energy_gate"),
            "energy_hangover_remaining_ms": record.get("energy_hangover_remaining_ms"),
            "vad_positive_count": record.get("vad_positive_count"),
            "vad_silence_count": record.get("vad_silence_count"),
            "vad_confirmed_state": record.get("vad_confirmed_state"),
            "kws_positive_count": record.get("kws_positive_count"),
            "vad_no_speech_remaining_ms": record.get("vad_no_speech_remaining_ms"),
        }

    @staticmethod
    def _dashboard_score_point(record: Mapping[str, object]) -> dict[str, object]:
        return {
            "captured_ns": record["captured_ns"],
            "score": record.get("score"),
            "model_input_start_ns": record.get("model_input_start_ns"),
            "model_input_end_ns": record.get("model_input_end_ns"),
        }

    @staticmethod
    def _dashboard_energy_point(record: Mapping[str, object]) -> dict[str, object]:
        return {
            "captured_ns": record["captured_ns"],
            "energy_dbfs": record.get("energy_dbfs"),
            "energy_gate": record.get("energy_gate"),
            "energy_gate_effective": record.get("energy_gate_effective"),
        }

    @staticmethod
    def _dashboard_points_with_gaps(
        records: list[dict[str, object]],
        point_factory: Callable[[Mapping[str, object]], dict[str, object]],
        *,
        cadence_ms: int,
    ) -> list[dict[str, object]]:
        gap_limit_ns = max(1_000_000_000, cadence_ms * 1_500_000)
        previous_ns: int | None = None
        points: list[dict[str, object]] = []
        for record in records:
            point = point_factory(record)
            captured_ns = int(point["captured_ns"])
            point["gap_before"] = previous_ns is not None and captured_ns - previous_ns > gap_limit_ns
            points.append(point)
            previous_ns = captured_ns
        return points

    @staticmethod
    def _decimate_points(
        points: list[dict[str, object]], max_points: int, *, value_key: str
    ) -> list[dict[str, object]]:
        if len(points) <= max_points:
            return points
        bucket_count = max(1, max_points // 2)
        boundaries = np.linspace(0, len(points), num=bucket_count + 1, dtype=np.int64)
        mandatory = {0, len(points) - 1}
        mandatory.update(index for index, point in enumerate(points) if point["gap_before"])
        if len(mandatory) > max_points:
            return [points[index] for index in sorted(mandatory)]
        selected_indices = set(mandatory)
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            start_index = int(start)
            end_index = int(end)
            if start_index >= end_index:
                continue
            candidates = range(start_index, end_index)
            low_index = min(candidates, key=lambda index: float(points[index][value_key]))
            high_index = max(range(start_index, end_index), key=lambda index: float(points[index][value_key]))
            selected_indices.update((low_index, high_index))
        extras = [index for index in sorted(selected_indices) if index not in mandatory]
        available_extras = max_points - len(mandatory)
        if len(extras) > available_extras:
            positions = np.linspace(0, len(extras) - 1, num=available_extras, dtype=np.int64)
            extras = [extras[int(position)] for position in positions]
        kept = sorted(mandatory) + extras
        return [points[index] for index in sorted(kept)]

    @staticmethod
    def _validate_probability(name: str, value: object) -> None:
        _require_probability(name, value)

    @staticmethod
    def _validate_timestamp(value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("timestamps must be nonnegative integers")


def _pcm_wav(samples: np.ndarray, sample_rate: int, *, sample_width: int = 3) -> bytes:
    source = np.asarray(samples)
    if np.issubdtype(source.dtype, np.integer) and sample_width == 3:
        values = source
        if values.ndim == 1:
            values = values[:, None]
        integer = (values.astype(np.int64) >> 8)
        packed = integer.reshape(-1)
        raw_array = np.empty((packed.size, 3), dtype=np.uint8)
        raw_array[:, 0] = packed & 0xFF
        raw_array[:, 1] = (packed >> 8) & 0xFF
        raw_array[:, 2] = (packed >> 16) & 0xFF
        raw = raw_array.tobytes()
        channels = values.shape[1]
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(sample_width)
            handle.setframerate(sample_rate)
            handle.writeframes(raw)
        return buffer.getvalue()
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("WAV samples must be nonempty channel-last audio")
    if sample_width not in (2, 3, 4):
        raise ValueError("WAV sample width must be 2, 3, or 4 bytes")
    clipped = np.clip(values, -1.0, 1.0)
    scale = float((1 << (sample_width * 8 - 1)) - 1)
    integer = np.rint(clipped * scale).astype(np.int64)
    if sample_width == 2:
        raw = integer.astype("<i2").tobytes()
    elif sample_width == 4:
        raw = integer.astype("<i4").tobytes()
    else:
        packed = integer.astype("<i4").reshape(-1)
        raw_array = np.empty((packed.size, 3), dtype=np.uint8)
        raw_array[:, 0] = packed & 0xFF
        raw_array[:, 1] = (packed >> 8) & 0xFF
        raw_array[:, 2] = (packed >> 16) & 0xFF
        raw = raw_array.tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(values.shape[1])
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
        handle.writeframes(raw)
    return buffer.getvalue()


def _pcm16_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    return _pcm_wav(samples, sample_rate, sample_width=2)


def decode_dashboard_wav(payload: bytes, *, maximum_samples: int) -> np.ndarray:
    """Decode only the WAV shape emitted by the dashboard download endpoint."""

    if isinstance(maximum_samples, bool) or not isinstance(maximum_samples, int) or maximum_samples < 1:
        raise ValueError("maximum_samples must be a positive integer")
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("import body is required")
    try:
        with wave.open(io.BytesIO(payload), "rb") as handle:
            format_values = (
                handle.getcomptype(),
                handle.getnchannels(),
                handle.getframerate(),
                handle.getsampwidth(),
            )
            if format_values != ("NONE", 1, 16000, 2):
                raise ValueError("import must be a 16 kHz mono PCM WAV")
            frame_count = handle.getnframes()
            if frame_count < 1 or frame_count > maximum_samples:
                raise ValueError("import duration exceeds the diagnostic history window")
            raw = handle.readframes(frame_count)
    except (EOFError, wave.Error) as error:
        raise ValueError("import must be a 16 kHz mono PCM WAV") from error
    if len(raw) != frame_count * 2:
        raise ValueError("import WAV data is truncated")
    return (np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0).copy()


def decode_dashboard_audio(payload: bytes, *, maximum_samples: int) -> tuple[np.ndarray, int]:
    """Decode standard PCM WAV for import, preserving channels and rate."""
    if isinstance(maximum_samples, bool) or not isinstance(maximum_samples, int) or maximum_samples < 1:
        raise ValueError("maximum_samples must be a positive integer")
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("import body is required")
    try:
        with wave.open(io.BytesIO(payload), "rb") as handle:
            channels = handle.getnchannels()
            sample_rate = handle.getframerate()
            sample_width = handle.getsampwidth()
            frame_count = handle.getnframes()
            compression = handle.getcomptype()
            if compression != "NONE":
                raise ValueError("import must be an uncompressed PCM WAV")
            if (
                channels not in _DASHBOARD_IMPORT_CHANNEL_COUNTS
                or sample_rate <= 0
                or sample_width not in _DASHBOARD_IMPORT_SAMPLE_WIDTHS
            ):
                raise ValueError("import must be mono/stereo PCM WAV with 16/24/32-bit samples")
            if sample_rate > _MAX_DASHBOARD_IMPORT_SAMPLE_RATE:
                raise ValueError("import sample rate must be at most 48 kHz")
            if frame_count < 1 or not _fits_analysis_duration(
                frame_count=frame_count,
                sample_rate=sample_rate,
                maximum_samples=maximum_samples,
            ):
                raise ValueError("import duration exceeds the diagnostic history window")
            raw = handle.readframes(frame_count)
    except (EOFError, wave.Error) as error:
        raise ValueError("import must be a standard PCM WAV") from error
    if len(raw) != frame_count * channels * sample_width:
        raise ValueError("import WAV data is truncated or compressed")
    if sample_width == 2:
        values = np.frombuffer(raw, dtype="<i2").astype(np.float32).reshape(-1, channels) / 32768.0
    elif sample_width == 4:
        values = np.frombuffer(raw, dtype="<i4").astype(np.float32).reshape(-1, channels) / 2147483648.0
    else:
        bytes_view = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        packed = (
            bytes_view[:, 0].astype(np.int32)
            | (bytes_view[:, 1].astype(np.int32) << 8)
            | (bytes_view[:, 2].astype(np.int32) << 16)
        )
        packed = np.where((packed & 0x800000) != 0, packed - 0x1000000, packed)
        values = (packed.astype(np.float32) / 8388608.0).reshape(-1, channels)
    if not np.all(np.isfinite(values)):
        raise ValueError("import WAV contains non-finite samples")
    return values.astype(np.float32, copy=False), int(sample_rate)


def _require_probability(name: str, value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be in [0, 1]")


def _require_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")


def _require_kws_lookback_ms(value: object, *, sample_rate: int = 16_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1_000 <= value <= 2_000:
        raise ValueError("kws_lookback_ms must be an integer in [1000, 2000]")
    if (sample_rate * value) % 1000:
        raise ValueError("kws_lookback_ms must resolve to an integer PCM sample count")
    return value


def _json_safe_mapping(values: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise ValueError("diagnostic metadata keys must be strings")
        if key.lower() in _RAW_AUDIO_KEYS:
            raise ValueError("diagnostic metadata must not contain raw audio")
        result[key] = _json_safe_value(value)
    return result


def _json_safe_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("diagnostic metadata floats must be finite")
        return result
    if isinstance(value, Mapping):
        return _json_safe_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    raise ValueError(f"diagnostic metadata does not support {type(value).__name__}")


class _ExclusiveLoopbackHttpServer(ThreadingHTTPServer):
    """Refuse address reuse so a stale dashboard cannot split browser traffic."""

    allow_reuse_address = False

    def server_bind(self) -> None:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class RealtimeDashboardServer:
    """Serve a diagnostic store to a local browser without extra dependencies."""

    def __init__(
        self,
        store: RealtimeDiagnosticStore,
        *,
        control: RealtimeCaptureControl | None = None,
        threshold_control: RealtimeThresholdControl | None = None,
        input_control: RealtimeInputControl | None = None,
        capture_mode_control: RealtimeCaptureModeControl | None = None,
        analysis_control: RealtimeAnalysisControl | None = None,
        port: int = 0,
    ) -> None:
        if not isinstance(store, RealtimeDiagnosticStore):
            raise ValueError("store must be a RealtimeDiagnosticStore")
        if control is not None and not isinstance(control, RealtimeCaptureControl):
            raise ValueError("control must be a RealtimeCaptureControl")
        if threshold_control is not None and not isinstance(
            threshold_control, RealtimeThresholdControl
        ):
            raise ValueError("threshold_control must be a RealtimeThresholdControl")
        if input_control is not None and not isinstance(input_control, RealtimeInputControl):
            raise ValueError("input_control must be a RealtimeInputControl")
        if capture_mode_control is not None and not isinstance(
            capture_mode_control, RealtimeCaptureModeControl
        ):
            raise ValueError("capture_mode_control must be a RealtimeCaptureModeControl")
        if analysis_control is not None and not isinstance(analysis_control, RealtimeAnalysisControl):
            raise ValueError("analysis_control must be a RealtimeAnalysisControl")
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("port must be an integer in [0, 65535]")
        self._store = store
        self._control = control or RealtimeCaptureControl()
        initial_runtime_config = store.runtime_config()
        self._threshold_control = threshold_control or RealtimeThresholdControl(
            vad_threshold=float(initial_runtime_config["vad_threshold"]),
            kws_threshold=float(initial_runtime_config["kws_threshold"]),
            energy_enabled=bool(initial_runtime_config["energy_enabled"]),
            vad_enabled=bool(initial_runtime_config["vad_enabled"]),
            vad_period_ms=int(initial_runtime_config["vad_period_ms"]),
            kws_period_ms=int(initial_runtime_config["kws_period_ms"]),
        )
        self._input_control = input_control or RealtimeInputControl()
        self._device_control: RealtimeDeviceControl | None = None
        self._capture_mode_control = capture_mode_control
        self._analysis_control = analysis_control or RealtimeAnalysisControl(
            maximum_samples=store.capacity_samples
        )
        self._runtime_config_persistor: Callable[[Mapping[str, object]], None] | None = None
        self._port = port
        self._server: _ExclusiveLoopbackHttpServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("dashboard server has not started")
        return f"http://127.0.0.1:{self._server.server_address[1]}/"

    def start(self) -> None:
        if self._server is not None:
            raise RuntimeError("dashboard server is already running")
        server = _ExclusiveLoopbackHttpServer(
            ("127.0.0.1", self._port),
            _request_handler(
                self._store,
                self._control,
                self._threshold_control,
                self._input_control,
                self._device_control,
                self._capture_mode_control,
                self._analysis_control,
                self._runtime_config_persistor,
            ),
        )
        server.daemon_threads = True
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.1},
            daemon=True,
            name="vad-kws-dashboard",
        )
        self._thread.start()

    def set_device_control(self, control: RealtimeDeviceControl | None) -> None:
        if control is not None and not isinstance(control, RealtimeDeviceControl):
            raise ValueError("device_control must be a RealtimeDeviceControl")
        if self._server is not None:
            raise RuntimeError("input device control must be configured before dashboard start")
        self._device_control = control

    def set_runtime_config_persistor(
        self, persistor: Callable[[Mapping[str, object]], None] | None
    ) -> None:
        """Configure durable settings storage before serving browser controls."""

        if persistor is not None and not callable(persistor):
            raise ValueError("runtime_config_persistor must be callable")
        if self._server is not None:
            raise RuntimeError("runtime config persistor must be configured before dashboard start")
        self._runtime_config_persistor = persistor

    def set_capture_mode_control(self, control: RealtimeCaptureModeControl | None) -> None:
        if control is not None and not isinstance(control, RealtimeCaptureModeControl):
            raise ValueError("capture_mode_control must be a RealtimeCaptureModeControl")
        if self._server is not None:
            raise RuntimeError("capture mode control must be configured before dashboard start")
        self._capture_mode_control = control

    def stop(self) -> None:
        server = self._server
        thread = self._thread
        self._server = None
        self._thread = None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join(timeout=3.0)


def _request_handler(
    store: RealtimeDiagnosticStore,
    control: RealtimeCaptureControl,
    threshold_control: RealtimeThresholdControl,
    input_control: RealtimeInputControl,
    device_control: RealtimeDeviceControl | None,
    capture_mode_control: RealtimeCaptureModeControl | None,
    analysis_control: RealtimeAnalysisControl,
    runtime_config_persistor: Callable[[Mapping[str, object]], None] | None,
) -> type[BaseHTTPRequestHandler]:
    class DashboardRequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch(send_body=True)

        def do_HEAD(self) -> None:  # noqa: N802
            self._dispatch(send_body=False)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/import":
                self._queue_import()
                return
            if parsed.path != "/api/control":
                self._send_json(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "method not allowed"}, send_body=True)
                return
            try:
                payload = self._control_payload()
            except ValueError as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)}, send_body=True)
                return
            action = payload["action"]
            if action == "pause":
                result = control.pause()
                status = HTTPStatus.OK
            elif action == "resume":
                result = control.resume()
                status = HTTPStatus.OK
            elif action == "clear_diagnostics":
                store.clear()
                result = {"cleared": True}
                status = HTTPStatus.OK
            elif action == "reanalyze_selection":
                try:
                    pcm, sample_rate = store.analysis_range(
                        payload["start_ns"], payload["end_ns"]
                    )
                    result = analysis_control.enqueue(
                        source="reanalysis", pcm=pcm, sample_rate=sample_rate
                    )
                except DiagnosticRangeError as error:
                    self._send_range_error(error, send_body=True)
                    return
                except AnalysisBusyError as error:
                    self._send_json(HTTPStatus.CONFLICT, {"error": str(error)}, send_body=True)
                    return
                except ValueError as error:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)}, send_body=True)
                    return
                status = HTTPStatus.ACCEPTED
            elif action == "set_thresholds":
                try:
                    result = threshold_control.set_thresholds(
                        vad_threshold=payload["vad_threshold"],
                        kws_threshold=payload["kws_threshold"],
                    )
                except ValueError as error:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)}, send_body=True)
                    return
                status = HTTPStatus.OK
            elif action == "set_input_config":
                try:
                    result = input_control.set_config(
                        conditioner_enabled=payload["conditioner_enabled"],
                        target_rms_dbfs=payload["target_rms_dbfs"],
                    )
                except ValueError as error:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)}, send_body=True)
                    return
                status = HTTPStatus.OK
            elif action == "select_input_device":
                if device_control is None:
                    self._send_json(
                        HTTPStatus.CONFLICT,
                        {"error": "input device switching is unavailable for WAV replay"},
                        send_body=True,
                    )
                    return
                try:
                    result = device_control.select(payload["device_index"])
                except ValueError as error:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)}, send_body=True)
                    return
                status = HTTPStatus.ACCEPTED
            elif action == "set_capture_mode":
                if capture_mode_control is None:
                    self._send_json(
                        HTTPStatus.CONFLICT,
                        {"error": "capture mode switching is unavailable for WAV replay"},
                        send_body=True,
                    )
                    return
                try:
                    result = capture_mode_control.request(str(payload["capture_mode"]))
                except ValueError as error:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)}, send_body=True)
                    return
                status = HTTPStatus.ACCEPTED
            else:
                try:
                    result = threshold_control.set_runtime_config(
                        vad_threshold=payload["vad_threshold"],
                        kws_threshold=payload["kws_threshold"],
                        energy_enabled=payload["energy_enabled"],
                        vad_enabled=payload["vad_enabled"],
                        vad_period_ms=payload["vad_period_ms"],
                        kws_period_ms=payload["kws_period_ms"],
                        kws_lookback_ms=payload["kws_lookback_ms"],
                    )
                except ValueError as error:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)}, send_body=True)
                    return
                if runtime_config_persistor is not None:
                    try:
                        runtime_config_persistor(result)
                    except OSError as error:
                        self._send_json(
                            HTTPStatus.INTERNAL_SERVER_ERROR,
                            {"error": f"could not persist runtime config: {error}"},
                            send_body=True,
                        )
                        return
                status = HTTPStatus.OK
            store.notify_update()
            self._send_json(status, result, send_body=True)

        def do_PUT(self) -> None:  # noqa: N802
            self._send_json(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "method not allowed"}, send_body=True)

        def do_DELETE(self) -> None:  # noqa: N802
            self._send_json(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "method not allowed"}, send_body=True)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _dispatch(self, *, send_body: bool) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_bytes(
                    HTTPStatus.OK,
                    "text/html; charset=utf-8",
                    _DASHBOARD_HTML.encode("utf-8"),
                    send_body=send_body,
                )
                return
            if parsed.path == "/api/snapshot":
                payload = store.dashboard_snapshot()
                payload["capture"] = control.snapshot()
                payload["requested_runtime_config"] = threshold_control.snapshot()
                payload["analysis"] = analysis_control.snapshot()
                payload["input_device"] = (
                    device_control.snapshot()
                    if device_control is not None
                    else {"available": False, "devices": [], "selected_index": None, "requested_index": None, "generation": 0, "error": None}
                )
                payload["capture_mode"] = (
                    capture_mode_control.snapshot()
                    if capture_mode_control is not None
                    else {
                        "available": False,
                        "selected_mode": None,
                        "requested_mode": None,
                        "generation": 0,
                        "error": None,
                    }
                )
                self._send_json(HTTPStatus.OK, payload, send_body=send_body)
                return
            if parsed.path == "/api/devices":
                payload = (
                    device_control.snapshot()
                    if device_control is not None
                    else {"available": False, "devices": [], "selected_index": None, "requested_index": None, "generation": 0, "error": None}
                )
                self._send_json(HTTPStatus.OK, payload, send_body=send_body)
                return
            if parsed.path == "/api/audio":
                self._send_audio(parse_qs(parsed.query, keep_blank_values=True), send_body=send_body)
                return
            if parsed.path == "/api/agc-preview":
                self._send_agc_preview(
                    parse_qs(parsed.query, keep_blank_values=True), send_body=send_body
                )
                return
            if parsed.path == "/api/events":
                self._send_events(send_body=send_body)
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"}, send_body=send_body)

        def _control_payload(self) -> dict[str, object]:
            raw_length = self.headers.get("Content-Length")
            try:
                content_length = int(raw_length) if raw_length is not None else 0
            except ValueError as error:
                raise ValueError("Content-Length must be an integer") from error
            if content_length <= 0:
                raise ValueError("control request body is required")
            try:
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("control request body must be JSON") from error
            if not isinstance(payload, dict) or not isinstance(payload.get("action"), str):
                raise ValueError("control request must contain action")
            action = payload["action"]
            if action in ("pause", "resume") and set(payload) == {"action"}:
                return {"action": action}
            if action == "clear_diagnostics" and set(payload) == {"action"}:
                return {"action": action}
            if action == "set_thresholds" and set(payload) == {
                "action",
                "vad_threshold",
                "kws_threshold",
            }:
                return payload
            if action == "set_runtime_config" and set(payload) == {
                "action",
                "vad_threshold",
                "kws_threshold",
                "energy_enabled",
                "vad_enabled",
                "vad_period_ms",
                "kws_period_ms",
                "kws_lookback_ms",
            }:
                return payload
            if action == "set_input_config" and set(payload) == {
                "action",
                "conditioner_enabled",
                "target_rms_dbfs",
            }:
                _require_bool("conditioner_enabled", payload["conditioner_enabled"])
                return payload
            if action == "select_input_device" and set(payload) == {"action", "device_index"}:
                device_index = payload["device_index"]
                if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
                    raise ValueError("device_index must be a nonnegative integer")
                return payload
            if action == "set_capture_mode" and set(payload) == {"action", "capture_mode"}:
                if not isinstance(payload["capture_mode"], str):
                    raise ValueError("capture_mode must be a string")
                return payload
            if action == "reanalyze_selection" and set(payload) == {
                "action",
                "start_ns",
                "end_ns",
            }:
                return payload
            raise ValueError("control action or fields are invalid")

        def _queue_import(self) -> None:
            try:
                raw_length = self.headers.get("Content-Length")
                try:
                    content_length = int(raw_length) if raw_length is not None else 0
                except ValueError as error:
                    self.close_connection = True
                    raise ValueError("Content-Length must be an integer") from error
                if content_length <= 0:
                    self.close_connection = True
                    raise ValueError("import body is required")
                if content_length > _maximum_dashboard_import_body_bytes(
                    analysis_control.maximum_samples
                ):
                    self.close_connection = True
                    raise ValueError("import duration exceeds the diagnostic history window")
                payload = self.rfile.read(content_length)
                if self.headers.get_content_type() != "audio/wav":
                    raise ValueError("import Content-Type must be audio/wav")
                pcm, sample_rate = decode_dashboard_audio(
                    payload,
                    maximum_samples=analysis_control.maximum_samples,
                )
                result = analysis_control.enqueue(source="import", pcm=pcm, sample_rate=sample_rate)
            except AnalysisBusyError as error:
                self._send_json(HTTPStatus.CONFLICT, {"error": str(error)}, send_body=True)
                return
            except ValueError as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)}, send_body=True)
                return
            store.notify_update()
            self._send_json(HTTPStatus.ACCEPTED, result, send_body=True)

        def _send_audio(self, query: dict[str, list[str]], *, send_body: bool) -> None:
            try:
                start_ns = _required_query_int(query, "start_ns")
                end_ns = _required_query_int(query, "end_ns")
                if "track" in query:
                    raise ValueError("audio export supports raw PCM only")
                payload = store.audio_bytes(start_ns, end_ns)
            except DiagnosticRangeError as error:
                self._send_range_error(error, send_body=send_body)
                return
            except ValueError as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)}, send_body=send_body)
                return
            self._send_bytes(HTTPStatus.OK, "audio/wav", payload, send_body=send_body)

        def _send_agc_preview(self, query: dict[str, list[str]], *, send_body: bool) -> None:
            try:
                start_ns = _required_query_int(query, "start_ns")
                end_ns = _required_query_int(query, "end_ns")
                input_config = input_control.snapshot()
                payload = store.agc_preview_audio_bytes(
                    start_ns,
                    end_ns,
                    target_rms_dbfs=float(input_config["target_rms_dbfs"]),
                )
            except DiagnosticRangeError as error:
                self._send_range_error(error, send_body=send_body)
                return
            except ValueError as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)}, send_body=send_body)
                return
            self._send_bytes(HTTPStatus.OK, "audio/wav", payload, send_body=send_body)

        def _send_range_error(self, error: DiagnosticRangeError, *, send_body: bool) -> None:
            self._send_json(
                HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                {
                    "error": str(error),
                    "available_start_ns": error.available_start_ns,
                    "available_end_ns": error.available_end_ns,
                },
                send_body=send_body,
            )

        def _send_events(self, *, send_body: bool) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            if not send_body:
                return
            generation = int(store.snapshot()["generation"])
            try:
                while True:
                    payload = json.dumps({"generation": generation}, separators=(",", ":")).encode("utf-8")
                    self.wfile.write(b"event: update\n")
                    self.wfile.write(b"data: " + payload + b"\n\n")
                    self.wfile.flush()
                    generation = store.wait_for_update(generation, timeout=1.0)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, TimeoutError):
                self.close_connection = True
                return

        def _send_json(self, status: HTTPStatus, payload: Mapping[str, object], *, send_body: bool) -> None:
            body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
            self._send_bytes(status, "application/json; charset=utf-8", body, send_body=send_body)

        def _send_bytes(self, status: HTTPStatus, content_type: str, body: bytes, *, send_body: bool) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if self.close_connection:
                self.send_header("Connection", "close")
            self.end_headers()
            if send_body:
                self.wfile.write(body)

    return DashboardRequestHandler


def _required_query_int(query: Mapping[str, list[str]], name: str) -> int:
    values = query.get(name)
    if values is None or len(values) != 1 or not values[0]:
        raise ValueError(f"{name} query parameter is required")
    try:
        value = int(values[0], 10)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VAD-KWS 实时诊断</title>
<style>
:root { color-scheme:light; font-family:"Segoe UI",Arial,sans-serif; background:#f2f5f4; color:#1d2933; }
* { box-sizing:border-box; }
body { margin:0; min-width:320px; }
header { min-height:48px; display:flex; justify-content:space-between; align-items:center; gap:16px; padding:10px 20px; background:#fff; border-bottom:1px solid #d8e1de; }
h1 { margin:0; font-size:17px; font-weight:700; }
.connection { font-size:12px; color:#61716a; white-space:nowrap; }
.connection::before { content:""; display:inline-block; width:7px; height:7px; margin-right:6px; border-radius:50%; background:#9aa9a3; }
.connection.online { color:#08786c; }.connection.online::before { background:#00897b; }
main { max-width:1440px; margin:0 auto; padding:12px 20px 20px; }
  .summary { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); border:1px solid #d8e1de; border-radius:6px; overflow:hidden; background:#d8e1de; gap:1px; }
.metric { min-width:0; min-height:58px; padding:9px 12px; background:#fff; }
.metric-label { color:#66766f; font-size:10px; font-weight:700; }.metric-value { margin-top:4px; font-size:13px; line-height:1.28; font-variant-numeric:tabular-nums; font-weight:700; overflow-wrap:anywhere; }
.workspace { margin-top:12px; border:1px solid #d8e1de; border-radius:6px; background:#fff; overflow:hidden; }
  .canvas-wrap { overflow-x:auto; padding:10px; }.timeline { display:block; min-width:760px; width:100%; height:471px; touch-action:none; cursor:crosshair; }
.controls { border-top:1px solid #d8e1de; }
.control-row { display:flex; align-items:center; gap:14px; min-height:52px; padding:9px 12px; border-bottom:1px solid #e1e8e5; }
.control-group { display:flex; align-items:center; gap:8px; min-width:0; }.control-group-label { color:#65756e; font-size:11px; font-weight:700; white-space:nowrap; }
.selection-group { flex:1 1 620px; min-width:0; display:grid; grid-template-columns:auto auto minmax(180px,1fr) auto minmax(180px,1fr); align-items:center; gap:7px; color:#4c5b55; font-size:12px; }
.control-group-playback { flex:0 0 auto; margin-left:auto; }
.control-group-data { margin-left:auto; padding-left:12px; border-left:1px solid #d8e1de; }
#clear-diagnostics { color:#a12a24; border-color:#d3a4a0; } #clear-diagnostics:hover:not(:disabled) { background:#fff4f2; border-color:#b84940; }
input[type="range"] { min-width:80px; width:100%; accent-color:#00897b; }
input[type="number"], select { width:62px; min-height:30px; border:1px solid #9bacA5; border-radius:4px; padding:3px 5px; background:#fff; color:#23322d; font:inherit; font-variant-numeric:tabular-nums; }
#input-device { width:min(360px,calc(100vw - 120px)); }
button { min-height:30px; border:1px solid #9baca5; border-radius:4px; padding:4px 9px; background:#fff; color:#23322d; font:inherit; font-size:12px; cursor:pointer; white-space:nowrap; }
button:hover:not(:disabled) { background:#edf7f4; border-color:#00897b; } button:disabled { cursor:not-allowed; opacity:.46; }
.toggle { color:#3d4c46; font-size:12px; white-space:nowrap; }.toggle input { accent-color:#00897b; }
.runtime-config { display:flex; align-items:center; flex-wrap:wrap; gap:10px 14px; min-width:0; padding:10px 12px; background:#f8faf9; border-top:1px solid #d8e1de; }
.runtime-field { display:grid; grid-template-columns:auto 64px; align-items:center; gap:6px; color:#3d4c46; font-size:12px; }
.notice { min-height:24px; padding:7px 2px 0; color:#64736d; font-size:12px; }
@media (max-width:1120px) { .control-row { flex-wrap:wrap; } .selection-group { flex-basis:100%; } .control-group-playback { margin-left:0; } .control-group-data { margin-left:auto; } .runtime-config { flex-wrap:wrap; } }
@media (max-width:680px) { main { padding:10px 12px 16px; } header { padding:10px 12px; } .summary { grid-template-columns:repeat(2,minmax(100px,1fr)); } .control-row { align-items:flex-start; gap:9px; } .selection-group { grid-template-columns:minmax(0,1fr); gap:5px; } .selection-group input[type="range"] { min-width:0; } .control-group { flex-wrap:wrap; } .control-group-playback { width:100%; } .control-group-data { margin-left:0; padding-left:0; border-left:0; } .runtime-config { align-items:flex-start; gap:8px; } }
</style>
</head>
<body>
<header><h1>VAD-KWS 实时诊断</h1><div id="connection" class="connection">正在连接</div></header>
<main>
  <section class="summary" aria-label="当前级联状态">
    <div class="metric"><div class="metric-label">级联状态</div><div id="state" class="metric-value">-</div></div>
    <div class="metric"><div class="metric-label">VAD / KWS</div><div id="scores" class="metric-value">-</div></div>
    <div class="metric"><div class="metric-label">门控</div><div id="energy" class="metric-value">-</div></div>
    <div class="metric"><div class="metric-label">队列 / 丢帧</div><div id="queue" class="metric-value">-</div></div>
    <div class="metric"><div class="metric-label">采集</div><div id="capture" class="metric-value">-</div></div>
  </section>
  <section class="workspace">
    <div class="canvas-wrap"><canvas id="timeline" class="timeline" aria-label="原始设备 PCM、能量、VAD、KWS 与状态时间轴"></canvas></div>
    <div class="controls">
      <div class="control-row control-row-selection">
        <div class="control-group selection selection-group">
          <label class="toggle"><input id="follow" type="checkbox" checked> 自动跟随</label>
          <span>起点</span><input id="start" type="range"><span>终点</span><input id="end" type="range">
        </div>
        <div class="control-group control-group-playback">
          <span class="control-group-label">回放</span>
          <button id="play" type="button" disabled>播放</button><button id="download" type="button" disabled>下载</button><button id="stop" type="button" disabled>停止</button>
        </div>
      </div>
      <div class="control-row control-row-actions">
        <input id="import-file" type="file" accept="audio/wav" hidden>
        <div class="control-group"><span class="control-group-label">分析</span><button id="import-audio" type="button" title="导入单/双声道 16/24/32-bit PCM WAV">导入 WAV</button><button id="reanalyze" type="button" disabled>重新分析选区</button></div>
        <div class="control-group"><span class="control-group-label">输入设备</span><select id="input-device" aria-label="输入设备"></select><button id="apply-input-device" type="button">切换</button></div>
        <div class="control-group"><label class="toggle"><input id="exclusive-capture" type="checkbox" checked> 独占采集</label><span id="input-capture-status" class="control-group-label">-</span></div>
        <div class="control-group"><span class="control-group-label">采集</span><button id="pause" type="button">暂停</button><button id="resume" type="button" disabled>继续</button></div>
        <div class="control-group control-group-data"><span class="control-group-label">数据</span><button id="clear-diagnostics" type="button" disabled>清空时间轴</button></div>
      </div>
    </div>
    <div class="runtime-config" aria-label="级联设置">
      <span class="control-group-label">级联</span>
      <label class="runtime-field">VAD 阈值 <input id="vad-threshold" type="number" min="0" max="1" step="0.01" value="0.80"></label>
      <label class="runtime-field">KWS 阈值 <input id="kws-threshold" type="number" min="0" max="1" step="0.01" value="0.50"></label>
      <label class="toggle"><input id="energy-gate" type="checkbox" checked> 能量门</label>
      <label class="toggle"><input id="vad-gate" type="checkbox" checked> VAD 门</label>
      <label class="runtime-field">VAD 周期 <input id="vad-period-ms" type="number" min="10" step="1" value="32"></label>
      <label class="runtime-field">KWS 周期 <input id="kws-period-ms" type="number" min="10" step="1" value="96"></label>
      <label class="runtime-field">KWS 历史 <input id="kws-lookback-ms" type="number" min="1000" max="2000" step="100" value="1500"></label>
      <button id="apply-runtime-config" type="button">应用级联设置</button>
    </div>
  </section>
  <div id="notice" class="notice">等待转换后的音频</div>
</main>
<script>
(() => {
  const canvas = document.getElementById("timeline");
  const ctx = canvas.getContext("2d");
  const timelineWrap = canvas.closest(".canvas-wrap");
  const follow = document.getElementById("follow");
  const startInput = document.getElementById("start");
  const endInput = document.getElementById("end");
  const importFileInput = document.getElementById("import-file");
  const importButton = document.getElementById("import-audio");
  const reanalyzeButton = document.getElementById("reanalyze");
  const playButton = document.getElementById("play");
  const downloadButton = document.getElementById("download");
  const stopButton = document.getElementById("stop");
  const pauseButton = document.getElementById("pause");
  const resumeButton = document.getElementById("resume");
  const inputDeviceSelect = document.getElementById("input-device");
  const applyInputDeviceButton = document.getElementById("apply-input-device");
  const exclusiveCaptureInput = document.getElementById("exclusive-capture");
  const inputCaptureStatus = document.getElementById("input-capture-status");
  const clearDiagnosticsButton = document.getElementById("clear-diagnostics");
  const vadThresholdInput = document.getElementById("vad-threshold");
  const kwsThresholdInput = document.getElementById("kws-threshold");
  const energyGateInput = document.getElementById("energy-gate");
  const vadGateInput = document.getElementById("vad-gate");
  const vadPeriodInput = document.getElementById("vad-period-ms");
  const kwsPeriodInput = document.getElementById("kws-period-ms");
  const kwsLookbackInput = document.getElementById("kws-lookback-ms");
  const applyRuntimeConfigButton = document.getElementById("apply-runtime-config");
  const notice = document.getElementById("notice");
  const connection = document.getElementById("connection");
  let snapshot = null;
  let snapshotEpoch = 0;
  let selection = null;
  let selectionViewport = null;
  let dragStart = null;
  let audio = null;
  let objectUrl = null;
  let downloading = false;
  let downloadNotice = null;
  let clearingDiagnostics = false;
  let playhead = null;
  let refreshTimer = null;
  let refreshInFlight = false;
  let refreshPending = false;
  let renderFrame = null;
  let lastCanvasSize = null;
  let runtimeConfigDirty = false;
  let runtimeConfigApplying = false;
  let runtimeConfigPending = false;
  let analysisSubmitting = false;
  let inputDeviceSwitching = false;
  let inputDeviceSelectionDirty = false;
  let captureModeSwitching = false;
  let pendingCaptureMode = null;
  const snapshotRefreshIntervalMs = 250;
  const lanes = { raw:[32,105], energy:[120,190], vad:[205,275], kws:[290,360], state:[375,445] };
  const laneLabels = { raw:"设备输入 PCM", energy:"能量", vad:"VAD", kws:"KWS", state:"状态" };
  const stateColors = { idle:"#e5e7eb", vad_candidate:"#f6c453", kws_active:"#14a38b", wake_latched:"#4d86c9" };
  const stateLabels = { idle:"静音", vad_candidate:"VAD 候选", kws_active:"KWS 活跃", wake_latched:"已唤醒" };
  const runtimeConfigStorageKey = "vad-kws.runtime-config.v2";

  function value(id, text) { const element=document.getElementById(id); if (element.textContent !== text) element.textContent = text; }
  function score(number) { return Number.isFinite(Number(number)) ? Number(number).toFixed(3) : "-"; }
  function relativeTime(ns) { return snapshot && snapshot.available_start_ns !== null ? ((Number(ns) - Number(snapshot.available_start_ns)) / 1e9).toFixed(2) + " 秒" : "-"; }
  function currentRange() { return snapshot && snapshot.available_start_ns !== null ? [Number(snapshot.available_start_ns), Number(snapshot.available_end_ns)] : null; }
  function clamp(value, low, high) { return Math.max(low, Math.min(high, value)); }
  function selectionViewportForFractions(startFraction, endFraction, range) {
    const duration = Math.max(1, range[1] - range[0]);
    const minimumFraction = Math.min(1, 100_000_000 / duration);
    let start = clamp(Number(startFraction), 0, 1);
    let end = clamp(Number(endFraction), 0, 1);
    if (end < start) [start, end] = [end, start];
    if (end - start < minimumFraction) {
      if (start + minimumFraction <= 1) end = start + minimumFraction;
      else { end = 1; start = Math.max(0, end - minimumFraction); }
    }
    return { start, end };
  }
  function defaultSelectionViewport(range) {
    const duration = Math.max(1, range[1] - range[0]);
    return selectionViewportForFractions(1 - Math.min(1, 3_000_000_000 / duration), 1, range);
  }
  function selectionFromViewport(range) {
    if (!selectionViewport) return null;
    const viewport = selectionViewportForFractions(selectionViewport.start, selectionViewport.end, range);
    const [low, high] = range;
    const duration = high - low;
    return {
      start: clamp(Math.round(low + viewport.start * duration), low, high),
      end: clamp(Math.round(low + viewport.end * duration), low, high),
    };
  }
  function viewportForSelection(start, end, range) {
    const [low, high] = range;
    const duration = Math.max(1, high - low);
    return selectionViewportForFractions((start - low) / duration, (end - low) / duration, range);
  }
  function analysisMessage(message) {
    const messages = {
      "analysis queued": "分析已排队",
      "analysis running": "正在分析",
      "analysis complete": "分析完成",
      "analysis failed": "分析失败",
      "import queued": "导入分析已排队",
      "import running": "正在导入分析",
      "reanalysis queued": "选区重分析已排队",
      "reanalysis running": "正在重分析选区",
    };
    return messages[String(message)] || String(message);
  }

  function loadRuntimeConfig() {
    try {
      const stored = JSON.parse(localStorage.getItem(runtimeConfigStorageKey) || "{}");
      const vad = Number(stored && stored.vad_threshold);
      const kws = Number(stored && stored.kws_threshold);
      const vadPeriod = Number(stored && stored.vad_period_ms);
      const kwsPeriod = Number(stored && stored.kws_period_ms);
      const kwsLookback = Number(stored && stored.kws_lookback_ms);
      if (!stored || !Number.isFinite(vad) || vad < 0 || vad > 1 || !Number.isFinite(kws) || kws < 0 || kws > 1 || !Number.isInteger(vadPeriod) || vadPeriod < 10 || !Number.isInteger(kwsPeriod) || kwsPeriod < 10 || !Number.isInteger(kwsLookback) || kwsLookback < 1000 || kwsLookback > 2000 || typeof stored.energy_enabled !== "boolean" || typeof stored.vad_enabled !== "boolean") return null;
      return {vad_threshold:vad, kws_threshold:kws, energy_enabled:stored.energy_enabled, vad_enabled:stored.vad_enabled, vad_period_ms:vadPeriod, kws_period_ms:kwsPeriod, kws_lookback_ms:kwsLookback};
    } catch (_) {
      return null;
    }
  }

  function saveRuntimeConfig(config) {
    localStorage.setItem(runtimeConfigStorageKey, JSON.stringify(config));
  }

  function requestedRuntimeConfig() {
    const vad = Number(vadThresholdInput.value);
    const kws = Number(kwsThresholdInput.value);
    const vadPeriod = Number(vadPeriodInput.value);
    const kwsPeriod = Number(kwsPeriodInput.value);
    const kwsLookback = Number(kwsLookbackInput.value);
    if (!Number.isFinite(vad) || vad < 0 || vad > 1) throw new Error("VAD 阈值必须在 0 到 1 之间");
    if (!Number.isFinite(kws) || kws < 0 || kws > 1) throw new Error("KWS 阈值必须在 0 到 1 之间");
    if (!Number.isInteger(vadPeriod) || vadPeriod < 10) throw new Error("VAD 周期必须是不小于 10 ms 的整数");
    if (!Number.isInteger(kwsPeriod) || kwsPeriod < 10) throw new Error("KWS 周期必须是不小于 10 ms 的整数");
    if (!Number.isInteger(kwsLookback) || kwsLookback < 1000 || kwsLookback > 2000) throw new Error("KWS 历史必须是 1000 到 2000 ms 的整数");
    return {vad_threshold:vad, kws_threshold:kws, energy_enabled:energyGateInput.checked, vad_enabled:vadGateInput.checked, vad_period_ms:vadPeriod, kws_period_ms:kwsPeriod, kws_lookback_ms:kwsLookback};
  }

  function sameRuntimeConfig(left, right) {
    if (!left || !right) return false;
    return Number(left.vad_threshold) === Number(right.vad_threshold)
      && Number(left.kws_threshold) === Number(right.kws_threshold)
      && Boolean(left.energy_enabled) === Boolean(right.energy_enabled)
      && Boolean(left.vad_enabled) === Boolean(right.vad_enabled)
      && Number(left.vad_period_ms) === Number(right.vad_period_ms)
      && Number(left.kws_period_ms) === Number(right.kws_period_ms)
      && Number(left.kws_lookback_ms) === Number(right.kws_lookback_ms);
  }

  function updateRuntimeConfigPending() {
    const requested = snapshot && snapshot.requested_runtime_config;
    const effective = snapshot && snapshot.runtime_config;
    runtimeConfigPending = Boolean(requested) && !sameRuntimeConfig(requested, effective);
  }

  async function refresh() {
    if (refreshInFlight) { refreshPending = true; return; }
    refreshInFlight = true;
    const snapshotEpochAtRequest = snapshotEpoch;
    try {
      const [response, deviceResponse] = await Promise.all([
        fetch("/api/snapshot", { cache:"no-store" }),
        fetch("/api/devices", { cache:"no-store" }),
      ]);
      if (!response.ok) throw new Error("snapshot " + response.status);
      if (!deviceResponse.ok) throw new Error("devices " + deviceResponse.status);
      const nextSnapshot = await response.json();
      nextSnapshot.input_device = await deviceResponse.json();
      if (snapshotEpochAtRequest !== snapshotEpoch) return;
      const accepted = !snapshot || Number(nextSnapshot.generation) >= Number(snapshot.generation);
      if (accepted) {
        const pageScrollTop = window.scrollY;
        const timelineScrollLeft = timelineWrap ? timelineWrap.scrollLeft : 0;
        snapshot = nextSnapshot;
        updateRuntimeConfigPending();
        updateControls();
        updateAnalysisControls();
        updateSummary();
        updateRuntimeConfigInputs();
        updateInputDeviceControl();
        updateCaptureModeStatus();
        scheduleRender();
        requestAnimationFrame(() => {
          if (window.scrollY !== pageScrollTop) window.scrollTo(window.scrollX, pageScrollTop);
          if (timelineWrap && timelineWrap.scrollLeft !== timelineScrollLeft) timelineWrap.scrollLeft = timelineScrollLeft;
        });
      }
      connection.textContent = "在线";
      connection.classList.add("online");
    } catch (error) {
      connection.textContent = "连接已断开";
      connection.classList.remove("online");
      notice.textContent = String(error);
    } finally {
      refreshInFlight = false;
      if (refreshPending) { refreshPending = false; refreshSoon(snapshotRefreshIntervalMs); }
    }
  }

  function refreshSoon(delay = snapshotRefreshIntervalMs) {
    if (refreshInFlight) { refreshPending = true; return; }
    if (refreshTimer !== null) return;
    refreshTimer = setTimeout(() => { refreshTimer = null; refresh(); }, delay);
  }

  function scheduleRender() {
    if (renderFrame !== null) return;
    renderFrame = requestAnimationFrame(() => {
      renderFrame = null;
      render();
    });
  }

  function updateSummary() {
    const latest = snapshot.latest || {};
    const thresholds = snapshot.thresholds || {};
    const lastVad = snapshot.vad_points.length ? snapshot.vad_points[snapshot.vad_points.length - 1].score : null;
    const lastKws = snapshot.kws_points.length ? snapshot.kws_points[snapshot.kws_points.length - 1].score : null;
    const vad_threshold = thresholds.vad;
    const kws_threshold = thresholds.kws;
    const energyDbfs = latest.energy_dbfs;
    const energyThreshold = thresholds.energy_dbfs;
    const capture = snapshot.capture || {};
    const runtimeConfig = snapshot.runtime_config || {};
    const noSpeechRemaining = Number(latest.vad_no_speech_remaining_ms);
    const stateDetail = latest.state === "kws_active" && Number.isFinite(noSpeechRemaining)
      ? " · 无人声 " + String(noSpeechRemaining) + " ms"
      : "";
    value("state", (stateLabels[latest.state] || latest.state || "-") + stateDetail);
    value("scores", "VAD " + score(lastVad) + " / " + score(vad_threshold) + " · KWS " + score(lastKws) + " / " + score(kws_threshold));
    const energyStatus = latest.energy_gate === null || latest.energy_gate === undefined
      ? "-"
      : (latest.energy_gate ? "通过" : "阻断");
    const hangoverRemaining = Number(latest.energy_hangover_remaining_ms ?? latest.hangover_remaining_ms);
    const hangoverText = Number.isFinite(hangoverRemaining)
      ? String(Math.max(0, hangoverRemaining)) + " ms"
      : "-";
    const kwsGateStatus = latest.kws_gate_open === null || latest.kws_gate_open === undefined
      ? "-"
      : (latest.kws_gate_open ? "开启" : "关闭");
    const confirmations = snapshot.confirmations_required || {};
    value("energy", "能量 " + (Number.isFinite(Number(energyDbfs)) ? Number(energyDbfs).toFixed(1) : "-") + " / " + Number(energyThreshold ?? -33).toFixed(0) + " dB · " + energyStatus + " · KWS门 " + kwsGateStatus + " · 保持 " + hangoverText + " · 确认 V" + String(confirmations.vad ?? "-") + "/K" + String(confirmations.kws ?? "-"));
    value("queue", String(latest.queue_depth ?? 0) + " / " + String(latest.drop_count ?? 0) + " · E" + String(runtimeConfig.energy_period_ms ?? "-") + "/V" + String(runtimeConfig.vad_period_ms ?? "-") + "/K" + String(runtimeConfig.kws_period_ms ?? "-") + " ms");
    const range = currentRange();
    const inputStatus = snapshot.input_status || {};
    const captureMode = String(inputStatus.capture_mode || "");
    const captureModeLabel = captureMode === "exclusive" ? "独占" : "独占未建立";
    value("capture", (capture.paused ? "已暂停" : "采集中") + " · " + captureModeLabel + " · " + (range ? ((range[1] - range[0]) / 1e9).toFixed(1) + " 秒" : "0.0 秒"));
  }

  function updateRuntimeConfigInputs() {
    const config = snapshot && snapshot.runtime_config ? snapshot.runtime_config : {};
    if (runtimeConfigDirty || runtimeConfigApplying || !Number.isFinite(Number(config.vad_threshold))) return;
    setValueIfChanged(vadThresholdInput, Number(config.vad_threshold).toFixed(2));
    setValueIfChanged(kwsThresholdInput, Number(config.kws_threshold).toFixed(2));
    energyGateInput.checked = config.energy_enabled !== false;
    vadGateInput.checked = config.vad_enabled !== false;
    setValueIfChanged(vadPeriodInput, config.vad_period_ms);
    setValueIfChanged(kwsPeriodInput, config.kws_period_ms);
    setValueIfChanged(kwsLookbackInput, config.kws_lookback_ms);
  }

  function updateInputDeviceControl() {
    const inputDevice = snapshot && snapshot.input_device ? snapshot.input_device : null;
    const devices = inputDevice && Array.isArray(inputDevice.devices) ? inputDevice.devices : [];
    const selectedIndex = inputDevice && Number(inputDevice.selected_index);
    const optionsChanged = inputDeviceSelect.options.length !== devices.length || devices.some((device, index) => {
      const option = inputDeviceSelect.options[index];
      return !option || option.value !== String(device.index) || option.textContent !== (String(device.name) + " (" + Number(device.default_sample_rate).toFixed(0) + " Hz)");
    });
    if (optionsChanged) {
      inputDeviceSelect.replaceChildren(...devices.map((device) => {
        const option = document.createElement("option");
        option.value = String(device.index);
        option.textContent = String(device.name) + " (" + Number(device.default_sample_rate).toFixed(0) + " Hz)";
        return option;
      }));
    }
    if (!inputDeviceSelectionDirty && Number.isFinite(selectedIndex)) inputDeviceSelect.value = String(selectedIndex);
    if (inputDeviceSelectionDirty && Number(inputDeviceSelect.value) === selectedIndex && inputDevice.requested_index === null) inputDeviceSelectionDirty = false;
    const disabled = !inputDevice || inputDevice.available !== true || !devices.length || inputDeviceSwitching || inputDevice.requested_index !== null;
    inputDeviceSelect.disabled = disabled;
    applyInputDeviceButton.disabled = disabled || Number(inputDeviceSelect.value) === selectedIndex;
    if (inputDevice && inputDevice.error) notice.textContent = "输入设备切换失败: " + String(inputDevice.error);
  }

  function updateCaptureModeStatus() {
    const inputStatus = snapshot && snapshot.input_status ? snapshot.input_status : {};
    const modeControl = snapshot && snapshot.capture_mode ? snapshot.capture_mode : {};
    const captureMode = String(inputStatus.capture_mode || "");
    const selectedMode = String(modeControl.selected_mode || captureMode || "");
    const switchPending = modeControl.requested_mode !== null && modeControl.requested_mode !== undefined;
    if (pendingCaptureMode !== null && (
      modeControl.error || selectedMode === pendingCaptureMode || String(modeControl.requested_mode || "") === pendingCaptureMode
    )) pendingCaptureMode = null;
    if (!captureModeSwitching && !switchPending && pendingCaptureMode === null && modeControl.available === true) {
      exclusiveCaptureInput.checked = selectedMode === "exclusive";
    }
    exclusiveCaptureInput.disabled = modeControl.available !== true || captureModeSwitching || switchPending || pendingCaptureMode !== null;
    inputCaptureStatus.textContent = captureMode === "exclusive"
      ? "独占"
      : captureMode === "shared"
        ? "共享"
        : "-";
    inputCaptureStatus.title = inputStatus.capture_format_reason
      ? String(inputStatus.capture_format_reason)
      : "";
    if (modeControl.error) notice.textContent = "采集模式切换失败: " + String(modeControl.error);
  }

  function setValueIfChanged(input, value) { const next=String(value); if (input.value !== next) input.value = next; }
  function setPropertyIfChanged(input, name, value) { const next=String(value); if (input[name] !== next) input[name] = next; }

  function updateAnalysisControls() {
    const analysis = snapshot && snapshot.analysis ? snapshot.analysis : {};
    const busy = analysisSubmitting || analysis.state === "queued" || analysis.state === "running";
    const blocked = !snapshot || runtimeConfigApplying || busy;
    importButton.disabled = blocked;
    importFileInput.disabled = blocked;
    reanalyzeButton.disabled = blocked || !selection || selection.end <= selection.start;
    if (analysis.state && analysis.state !== "idle" && analysis.message) notice.textContent = analysisMessage(analysis.message);
  }

  function updateControls() {
    const capture = snapshot && snapshot.capture ? snapshot.capture : {};
    pauseButton.disabled = !snapshot || Boolean(capture.paused);
    resumeButton.disabled = !snapshot || !Boolean(capture.paused);
    clearDiagnosticsButton.disabled = !snapshot || clearingDiagnostics;
    applyRuntimeConfigButton.disabled = !snapshot || runtimeConfigApplying;
    const range = currentRange();
    if (!range) { selection = null; selectionViewport = null; playButton.disabled = true; downloadButton.disabled = true; return; }
    const [low, high] = range;
    if (follow.checked || !selectionViewport) selectionViewport = defaultSelectionViewport(range);
    selection = selectionFromViewport(range);
    setPropertyIfChanged(startInput, "min", low); setPropertyIfChanged(startInput, "max", high); setPropertyIfChanged(startInput, "step", "100000000");
    setPropertyIfChanged(endInput, "min", low); setPropertyIfChanged(endInput, "max", high); setPropertyIfChanged(endInput, "step", "100000000");
    setValueIfChanged(startInput, selection.start); setValueIfChanged(endInput, selection.end);
    const selectionValid = selection.end > selection.start;
    playButton.disabled = !selectionValid;
    downloadButton.disabled = !selectionValid || downloading;
    if (downloadNotice !== null) { notice.textContent = downloadNotice; return; }
    notice.textContent = runtimeConfigPending
      ? "级联设置正在等待音频边界"
      : "选区 " + relativeTime(selection.start) + " - " + relativeTime(selection.end);
  }

  function setSelection(start, end) {
    const range = currentRange(); if (!range) return;
    const [low, high] = range;
    selection = { start:clamp(Math.round(start), low, high), end:clamp(Math.round(end), low, high) };
    if (selection.end <= selection.start) selection.end = Math.min(high, selection.start + 100_000_000);
    selectionViewport = viewportForSelection(selection.start, selection.end, range);
    selection = selectionFromViewport(range);
    downloadNotice = null;
    updateControls(); updateAnalysisControls(); scheduleRender();
  }

  function setSelectionViewport(startFraction, endFraction) {
    const range = currentRange(); if (!range) return;
    selectionViewport = selectionViewportForFractions(startFraction, endFraction, range);
    selection = selectionFromViewport(range);
    downloadNotice = null;
    updateControls(); updateAnalysisControls(); scheduleRender();
  }

  function resizeCanvasIfNeeded() {
    const rect = canvas.getBoundingClientRect(); const ratio = window.devicePixelRatio || 1;
    const nextSize = { width:Math.round(rect.width * ratio), height:Math.round(rect.height * ratio), ratio };
    if (!lastCanvasSize || nextSize.width !== lastCanvasSize.width || nextSize.height !== lastCanvasSize.height || nextSize.ratio !== lastCanvasSize.ratio) {
      canvas.width = nextSize.width; canvas.height = nextSize.height;
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      lastCanvasSize = nextSize;
    }
    return { width:rect.width, height:rect.height };
  }

  function render() {
    const size = resizeCanvasIfNeeded(); const width = size.width; const height = size.height;
    ctx.clearRect(0, 0, width, height); ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, width, height);
    if (!snapshot || snapshot.available_start_ns === null) { ctx.fillStyle="#6b7280"; ctx.font="13px Segoe UI"; ctx.fillText("等待设备输入音频", 18, 36); return; }
    const [start, end] = currentRange(); const plotLeft = 66; const plotRight = width - 16;
    const x = (time) => plotLeft + (Number(time) - start) / Math.max(1, end - start) * (plotRight - plotLeft);
    ctx.font = "11px Segoe UI"; ctx.textBaseline = "middle";
    for (const [name, lane] of Object.entries(lanes)) {
      const top = lane[0], bottom = lane[1];
      ctx.fillStyle="#f8faf9"; ctx.fillRect(plotLeft, top, plotRight - plotLeft, bottom - top);
      ctx.strokeStyle="#d9dfdc"; ctx.lineWidth=1; ctx.strokeRect(plotLeft, top, plotRight - plotLeft, bottom - top);
      ctx.fillStyle="#57615d"; ctx.fillText(laneLabels[name], 10, top + 13);
    }
    ctx.strokeStyle="#e1e7e3"; ctx.fillStyle="#6b7280";
    for (let tick = 0; tick <= 6; tick++) { const px = plotLeft + tick / 6 * (plotRight - plotLeft); ctx.beginPath(); ctx.moveTo(px, lanes.raw[0]); ctx.lineTo(px, lanes.state[1]); ctx.stroke(); ctx.fillText(((end-start)/1e9*tick/6).toFixed(1)+"秒", px-10, lanes.state[1] + 20); }
    drawWaveform(x, plotLeft, plotRight, lanes.raw, snapshot.waveform, "#00897b", audioTrackLabel()); drawEnergy(x); drawScores(x, lanes.vad, snapshot.vad_points, snapshot.thresholds.vad, "#1565c0"); drawScores(x, lanes.kws, snapshot.kws_points, snapshot.thresholds.kws, "#c62828"); drawStates(x);
    if (selection && selectionViewport) { const left=plotLeft + selectionViewport.start * (plotRight - plotLeft), right=plotLeft + selectionViewport.end * (plotRight - plotLeft); ctx.fillStyle="rgba(0,137,123,.12)"; ctx.fillRect(left, lanes.raw[0], right-left, lanes.state[1]-lanes.raw[0]); ctx.strokeStyle="#00897b"; ctx.setLineDash([4,3]); ctx.strokeRect(left, lanes.raw[0], right-left, lanes.state[1]-lanes.raw[0]); ctx.setLineDash([]); }
    if (playhead !== null) { ctx.strokeStyle="#202124"; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(x(playhead), lanes.raw[0]); ctx.lineTo(x(playhead), lanes.state[1]); ctx.stroke(); }
  }

  function drawWaveform(x, plotLeft, plotRight, lane, waveform, color, label) {
    const center=(lane[0]+lane[1])/2;
    const waveformPeak = waveform.reduce((peak, point) => Math.max(peak, ...(Array.isArray(point.values) ? point.values.map((value) => Math.abs(Number(value)) || 0) : [0])), 0.08);
    const scale=(lane[1]-lane[0])/2 / Math.min(1, waveformPeak);
    const channelCount = Math.max(1, Number(snapshot.channels || 1));
    const colors = [color, "#b45309", "#6d28d9", "#2563eb"];
    for (let channel=0; channel<channelCount; channel++) {
      ctx.strokeStyle=colors[channel % colors.length]; ctx.lineWidth=1; ctx.beginPath(); let hasPrevious=false;
      for (const point of waveform) {
        const values = Array.isArray(point.values) ? point.values : [];
        const sample = Number(values[channel]);
        const px=x(point.captured_ns);
        if (!Number.isFinite(sample) || px < plotLeft || px > plotRight) continue;
        const py=center - clamp(sample, -1, 1)*scale;
        if (!hasPrevious) ctx.moveTo(px,py); else ctx.lineTo(px,py);
        hasPrevious=true;
      }
      ctx.stroke();
    }
    ctx.fillStyle="#6b7280"; ctx.fillText(label, plotLeft+6, lane[0]+12);
  }

  function audioTrackLabel() {
    const source = snapshot.display_track === "raw_device" ? "设备输入 PCM" : "模型输入 PCM";
    const channels = Math.max(1, Number(snapshot.channels || 1));
    const rateKhz = (Number(snapshot.sample_rate) / 1000).toFixed(1).replace(/\.0$/, "");
    return source + " (" + channels + " 声道, " + rateKhz + " kHz)";
  }

  function drawEnergy(x) {
    const lane=lanes.energy, floor=-80, ceiling=0;
    const y=(dbfs) => lane[1] - (clamp(Number(dbfs), floor, ceiling)-floor)/(ceiling-floor)*(lane[1]-lane[0]);
    const threshold = Number(snapshot.thresholds.energy_dbfs ?? -33);
    ctx.strokeStyle="#8b9590"; ctx.setLineDash([3,3]); ctx.beginPath(); ctx.moveTo(x(snapshot.available_start_ns), y(threshold)); ctx.lineTo(x(snapshot.available_end_ns), y(threshold)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle="#59635f"; ctx.fillText("阈值 " + threshold.toFixed(0) + " dBFS", x(snapshot.available_start_ns)+6, y(threshold)-7);
    ctx.strokeStyle="#00897b"; ctx.lineWidth=2; ctx.beginPath(); let hasPrevious=false;
    for (const point of (snapshot.energy_points || [])) { if (!Number.isFinite(Number(point.energy_dbfs))) continue; const time=Number(point.captured_ns), px=x(time), py=y(point.energy_dbfs); if (!hasPrevious || point.gap_before) ctx.moveTo(px,py); else ctx.lineTo(px,py); hasPrevious=true; }
    ctx.stroke(); ctx.fillStyle="#6b7280"; ctx.fillText("能量门原始单声道 dBFS", x(snapshot.available_start_ns)+6, lane[0]+12);
  }

  function drawScores(x, lane, points, threshold, color) {
    const top=lane[0], bottom=lane[1], y=(scoreValue) => bottom - Number(scoreValue)*(bottom-top);
    ctx.strokeStyle="#8b9590"; ctx.setLineDash([3,3]); ctx.beginPath(); ctx.moveTo(x(snapshot.available_start_ns), y(threshold)); ctx.lineTo(x(snapshot.available_end_ns), y(threshold)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle="#59635f"; ctx.fillText("阈值 " + Number(threshold).toFixed(2), x(snapshot.available_start_ns)+6, y(threshold)-7);
    ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath(); let hasPrevious=false;
    for (const point of points) { if (!Number.isFinite(Number(point.score))) continue; const time=Number(point.model_input_end_ns || point.captured_ns); const px=x(time), py=y(point.score); if (!hasPrevious || point.gap_before) ctx.moveTo(px,py); else ctx.lineTo(px,py); hasPrevious=true; }
    ctx.stroke();
    ctx.fillStyle="#6b7280"; ctx.fillText("1.0", x(snapshot.available_start_ns)+4, top+10); ctx.fillText("0.0", x(snapshot.available_start_ns)+4, bottom-8);
  }

  function drawStates(x) {
    const lane=lanes.state;
    for (const interval of snapshot.state_intervals) { const left=x(interval.start_ns), right=x(interval.end_ns); ctx.fillStyle=stateColors[interval.state] || "#9ca3af"; ctx.fillRect(left, lane[0], Math.max(1,right-left), lane[1]-lane[0]); if (right-left>62) { ctx.fillStyle=interval.state === "silence" ? "#374151" : "#ffffff"; ctx.fillText(stateLabels[interval.state] || interval.state, left+6, (lane[0]+lane[1])/2); } }
    for (const wake of snapshot.wake_markers) { const px=x(wake.captured_ns); ctx.strokeStyle="#c62828"; ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(px,lane[0]); ctx.lineTo(px,lane[1]); ctx.stroke(); ctx.fillStyle="#c62828"; ctx.fillText("唤醒", px+4, lane[0]+11); }
  }

  function beginManualSelection() { follow.checked=false; }
  function viewportForPointer(event) { const rect=canvas.getBoundingClientRect(); return clamp((event.clientX-rect.left-66)/Math.max(1,rect.width-82),0,1); }
  canvas.addEventListener("pointerdown", (event) => { if (!snapshot || snapshot.available_start_ns === null) return; beginManualSelection(); dragStart=viewportForPointer(event); canvas.setPointerCapture(event.pointerId); setSelectionViewport(dragStart, dragStart); });
  canvas.addEventListener("pointermove", (event) => { if (dragStart === null) return; const current=viewportForPointer(event); setSelectionViewport(Math.min(dragStart,current), Math.max(dragStart,current)); });
  canvas.addEventListener("pointerup", () => { dragStart=null; });
  canvas.addEventListener("pointerleave", updateControls);
  startInput.addEventListener("input", () => { beginManualSelection(); setSelection(Number(startInput.value), Number(endInput.value)); });
  endInput.addEventListener("input", () => { beginManualSelection(); setSelection(Number(startInput.value), Number(endInput.value)); });
  follow.addEventListener("change", () => { if (follow.checked) { updateControls(); scheduleRender(); } });
  window.addEventListener("resize", scheduleRender);

  function stopPlayback() { if (audio) { audio.pause(); audio = null; } if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl=null; } playhead=null; stopButton.disabled=true; scheduleRender(); }
  stopButton.addEventListener("click", stopPlayback);
  function markRuntimeConfigDirty() {
    runtimeConfigDirty = true;
    try {
      saveRuntimeConfig(requestedRuntimeConfig());
    } catch (_) {
      // Leave validation feedback to the explicit apply action.
    }
  }
  async function submitRuntimeConfig() {
    let config;
    try {
      config = {action:"set_runtime_config", ...requestedRuntimeConfig()};
    } catch (error) {
      notice.textContent = String(error);
      return;
    }
    runtimeConfigApplying = true;
    updateControls();
    updateAnalysisControls();
    try {
      const response = await fetch("/api/control", {
        method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(config), cache:"no-store"
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "级联设置请求失败 " + response.status);
      runtimeConfigDirty = false;
      runtimeConfigPending = true;
      notice.textContent = "级联设置正在等待音频边界";
      refreshSoon(0);
    } catch (error) {
      notice.textContent = String(error);
    } finally {
      runtimeConfigApplying = false;
      updateControls();
      updateAnalysisControls();
    }
  }
  for (const controlInput of [vadThresholdInput, kwsThresholdInput, energyGateInput, vadGateInput, vadPeriodInput, kwsPeriodInput, kwsLookbackInput]) controlInput.addEventListener("input", markRuntimeConfigDirty);
  applyRuntimeConfigButton.addEventListener("click", submitRuntimeConfig);
  const storedRuntimeConfig = loadRuntimeConfig();
  if (storedRuntimeConfig) {
    setValueIfChanged(vadThresholdInput, storedRuntimeConfig.vad_threshold.toFixed(2));
    setValueIfChanged(kwsThresholdInput, storedRuntimeConfig.kws_threshold.toFixed(2));
    energyGateInput.checked = storedRuntimeConfig.energy_enabled;
    vadGateInput.checked = storedRuntimeConfig.vad_enabled;
    setValueIfChanged(vadPeriodInput, storedRuntimeConfig.vad_period_ms);
    setValueIfChanged(kwsPeriodInput, storedRuntimeConfig.kws_period_ms);
    setValueIfChanged(kwsLookbackInput, storedRuntimeConfig.kws_lookback_ms);
    runtimeConfigDirty = true;
    void submitRuntimeConfig();
  }
  async function setCapture(action) {
    try {
      const response = await fetch("/api/control", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({action}),
        cache:"no-store"
      });
      if (!response.ok) throw new Error("采集控制请求失败 " + response.status);
      await response.json();
      refreshSoon();
    } catch (error) {
      notice.textContent = String(error);
    }
  }
  pauseButton.addEventListener("click", () => setCapture("pause"));
  resumeButton.addEventListener("click", () => setCapture("resume"));
  async function submitCaptureMode() {
    const captureMode = exclusiveCaptureInput.checked ? "exclusive" : "shared";
    pendingCaptureMode = captureMode;
    captureModeSwitching = true;
    updateCaptureModeStatus();
    try {
      const response = await fetch("/api/control", {
        method:"POST", headers:{"Content-Type":"application/json"},
        body:JSON.stringify({action:"set_capture_mode", capture_mode:captureMode}), cache:"no-store"
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "采集模式切换请求失败 " + response.status);
      notice.textContent = "采集模式正在切换";
      refreshSoon(0);
    } catch (error) {
      pendingCaptureMode = null;
      notice.textContent = String(error);
      refreshSoon(0);
    } finally {
      captureModeSwitching = false;
      updateCaptureModeStatus();
    }
  }
  exclusiveCaptureInput.addEventListener("change", submitCaptureMode);
  inputDeviceSelect.addEventListener("change", () => { inputDeviceSelectionDirty = true; updateInputDeviceControl(); });
  applyInputDeviceButton.addEventListener("click", async () => {
    const deviceIndex = Number(inputDeviceSelect.value);
    if (!Number.isInteger(deviceIndex) || deviceIndex < 0) return;
    inputDeviceSwitching = true;
    updateInputDeviceControl();
    try {
      const response = await fetch("/api/control", {
        method:"POST", headers:{"Content-Type":"application/json"},
        body:JSON.stringify({action:"select_input_device", device_index:deviceIndex}), cache:"no-store"
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "输入设备切换请求失败 " + response.status);
      inputDeviceSelectionDirty = false;
      notice.textContent = "输入设备正在切换";
      refreshSoon(0);
    } catch (error) {
      notice.textContent = String(error);
    } finally {
      inputDeviceSwitching = false;
      updateInputDeviceControl();
    }
  });
  async function clearDiagnostics() {
    if (!window.confirm("确定清空当前时间轴中的音频与诊断数据吗？")) return;
    clearingDiagnostics = true;
    updateControls();
    try {
      const response = await fetch("/api/control", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({action:"clear_diagnostics"}),
        cache:"no-store",
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "清空请求失败 " + response.status);
      stopPlayback();
      selection = null;
      selectionViewport = null;
      snapshotEpoch += 1;
      snapshot = null;
      notice.textContent = "时间轴已清空";
      refreshSoon(0);
    } catch (error) {
      notice.textContent = String(error);
    } finally {
      clearingDiagnostics = false;
      updateControls();
      updateAnalysisControls();
    }
  }
  clearDiagnosticsButton.addEventListener("click", clearDiagnostics);
  async function submitAnalysis(url, options) {
    analysisSubmitting = true;
    updateAnalysisControls();
    try {
      const response = await fetch(url, options);
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "分析请求失败 " + response.status);
      notice.textContent = analysisMessage(result.message || result.state);
      refreshSoon(0);
    } catch (error) {
      notice.textContent = String(error);
    } finally {
      analysisSubmitting = false;
      updateAnalysisControls();
    }
  }
  importButton.addEventListener("click", () => importFileInput.click());
  importFileInput.addEventListener("change", () => {
    const file = importFileInput.files && importFileInput.files[0];
    importFileInput.value = "";
    if (!file) return;
    submitAnalysis("/api/import", {method:"POST", headers:{"Content-Type":"audio/wav"}, body:file, cache:"no-store"});
  });
  reanalyzeButton.addEventListener("click", () => {
    if (!selection) return;
    submitAnalysis("/api/control", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({action:"reanalyze_selection", start_ns:selection.start, end_ns:selection.end}), cache:"no-store"
    });
  });
  playButton.addEventListener("click", async () => {
    if (!selection) return; stopPlayback(); playButton.disabled=true;
    const endpoint = "/api/audio";
    try { const response=await fetch(endpoint+"?start_ns="+selection.start+"&end_ns="+selection.end, {cache:"no-store"}); if (!response.ok) throw new Error(response.status === 416 ? "选区音频已超出缓存" : "音频请求失败 "+response.status); objectUrl=URL.createObjectURL(await response.blob()); audio=new Audio(objectUrl); audio.addEventListener("timeupdate", () => { playhead=selection.start+audio.currentTime*1e9; scheduleRender(); }); audio.addEventListener("ended", stopPlayback); await audio.play(); stopButton.disabled=false; } catch (error) { notice.textContent=String(error); stopPlayback(); } finally { playButton.disabled=false; }
  });
  async function downloadSelectedAudio() {
    if (!selection || downloading) return;
    const start = selection.start;
    const end = selection.end;
    downloading = true;
    downloadNotice = "正在准备原始设备 WAV";
    updateControls();
    try {
      const response = await fetch("/api/audio?start_ns=" + start + "&end_ns=" + end, {cache:"no-store"});
      if (!response.ok) throw new Error(response.status === 416 ? "原始设备音频已超出缓存，请选择最新可用区间" : "音频请求失败 " + response.status);
      const downloadUrl = URL.createObjectURL(await response.blob());
      try {
        const anchor = document.createElement("a");
        anchor.href = downloadUrl;
        anchor.download = "vad-kws-" + start + "-" + end + ".wav";
        try {
          document.body.appendChild(anchor);
          anchor.click();
        } finally {
          anchor.remove();
        }
      } finally {
        setTimeout(() => URL.revokeObjectURL(downloadUrl), 60_000);
      }
      downloadNotice = "已开始下载原始设备 WAV";
    } catch (error) {
      downloadNotice = String(error);
    } finally {
      downloading = false;
      updateControls();
    }
  }
  downloadButton.addEventListener("click", downloadSelectedAudio);
  const events = new EventSource("/api/events"); events.addEventListener("update", () => refreshSoon()); events.onerror = () => { connection.textContent="正在重连"; connection.classList.remove("online"); };
  refresh(); setInterval(refreshSoon, 3000);
})();
</script>
</body>
</html>"""
