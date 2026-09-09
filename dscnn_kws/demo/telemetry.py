"""Structured session logging and latency aggregation for the terminal demo."""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class TelemetryError(RuntimeError):
    """Raised when a log record would violate the demo telemetry contract."""


_RAW_AUDIO_KEYS = frozenset({"audio", "pcm", "raw_pcm", "waveform", "samples"})
_LATENCY_FIELDS = (
    "queue_delay_ms",
    "frontend_ms",
    "onnx_ms",
    "state_ms",
    "total_ms",
    "end_to_end_ms",
)


class NullSessionLogger:
    """Drop telemetry events when durable session logging is disabled."""

    def record(self, event: str, **fields: object) -> None:
        """Discard an event without validating or persisting it."""

    def close(self, stopped_ns: int | None = None) -> dict[str, object]:
        """Return no session summary because no session was recorded."""

        return {}


def percentile_ms(values: Sequence[float], percentile: float) -> float | None:
    """Return nearest-rank percentile, retaining useful values for tiny samples."""

    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be in (0, 100]")
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) and value >= 0.0 for value in ordered):
        raise ValueError("latencies must be finite nonnegative numbers")
    rank = max(1, math.ceil(percentile / 100.0 * len(ordered)))
    return ordered[rank - 1]


class SessionLogger:
    """Append-only event writer that returns a durable session summary on close."""

    def __init__(self, session_path: Path, started_ns: int, events_handle: Any) -> None:
        self.session_path = session_path
        self.manifest_path = session_path / "manifest.json"
        self.events_path = session_path / "events.jsonl"
        self.summary_path = session_path / "summary.json"
        self._started_ns = started_ns
        self._events_handle = events_handle
        self._lock = threading.Lock()
        self._counts: Counter[str] = Counter()
        self._latencies: dict[str, list[float]] = defaultdict(list)
        self._dropped_audio_chunks = 0
        self._speech_duration_ms = 0.0
        self._summary: dict[str, object] | None = None

    @classmethod
    def create(
        cls, session_root: Path, manifest: Mapping[str, object], *, started_ns: int | None = None
    ) -> "SessionLogger":
        root = Path(session_root)
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise TelemetryError(f"could not create session root: {root}") from error
        session_path = root / f"session-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        try:
            session_path.mkdir()
        except OSError as error:
            raise TelemetryError(f"could not create session directory: {session_path}") from error
        effective_started_ns = time.perf_counter_ns() if started_ns is None else started_ns
        if isinstance(effective_started_ns, bool) or not isinstance(effective_started_ns, int) or effective_started_ns < 0:
            raise TelemetryError("started_ns must be a nonnegative integer")
        manifest_payload = dict(manifest)
        manifest_payload["started_monotonic_ns"] = effective_started_ns
        _validate_json_value(manifest_payload)
        manifest_path = session_path / "manifest.json"
        _atomic_json_write(manifest_path, manifest_payload)
        try:
            events_handle = (session_path / "events.jsonl").open("x", encoding="utf-8", newline="\n")
        except OSError as error:
            raise TelemetryError(f"could not create events file: {session_path}") from error
        return cls(session_path, effective_started_ns, events_handle)

    def record(self, event: str, **fields: object) -> None:
        if not isinstance(event, str) or not event.strip():
            raise TelemetryError("event must be a nonempty string")
        _validate_json_value(fields)
        dropped_audio_chunks = _validated_dropped_audio_chunks(event, fields)
        speech_duration_ms = _validated_speech_duration_ms(event, fields)
        payload = {"event": event, "monotonic_ns": time.perf_counter_ns(), **fields}
        with self._lock:
            if self._summary is not None:
                raise TelemetryError("cannot record an event after the session is closed")
            try:
                self._events_handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
                self._events_handle.flush()
            except OSError as error:
                raise TelemetryError(f"could not append event to {self.events_path}") from error
            self._counts[event] += 1
            self._dropped_audio_chunks += dropped_audio_chunks
            if speech_duration_ms is not None:
                self._speech_duration_ms = max(self._speech_duration_ms, speech_duration_ms)
            for field in _LATENCY_FIELDS:
                value = fields.get(field)
                if value is not None:
                    self._latencies[field].append(float(value))

    def close(self, stopped_ns: int | None = None) -> dict[str, object]:
        with self._lock:
            if self._summary is not None:
                return dict(self._summary)
            effective_stopped_ns = time.perf_counter_ns() if stopped_ns is None else stopped_ns
            if (
                isinstance(effective_stopped_ns, bool)
                or not isinstance(effective_stopped_ns, int)
                or effective_stopped_ns < self._started_ns
            ):
                raise TelemetryError("stopped_ns must be no earlier than started_ns")
            try:
                self._events_handle.flush()
                self._events_handle.close()
            except OSError as error:
                raise TelemetryError(f"could not close events file: {self.events_path}") from error
            summary: dict[str, object] = {
                "started_monotonic_ns": self._started_ns,
                "stopped_monotonic_ns": effective_stopped_ns,
                "duration_ms": (effective_stopped_ns - self._started_ns) / 1_000_000.0,
                "event_counts": dict(sorted(self._counts.items())),
                "vad_calls": self._counts["vad_decision"],
                "kws_calls": self._counts["kws_call"],
                "wake_count": self._counts["wake"],
                "audio_drops": self._counts["audio_drop"],
                "processed_audio_chunks": self._counts["audio_chunk"],
                "dropped_audio_chunks": self._dropped_audio_chunks,
                "captured_audio_chunks": self._counts["audio_chunk"] + self._dropped_audio_chunks,
                "speech_duration_ms": self._speech_duration_ms,
                "error_count": self._counts["error"],
                "latency_ms": {
                    name: {
                        "p50": percentile_ms(values, 50.0),
                        "p95": percentile_ms(values, 95.0),
                        "p99": percentile_ms(values, 99.0),
                    }
                    for name, values in sorted(self._latencies.items())
                },
            }
            _atomic_json_write(self.summary_path, summary)
            self._summary = summary
            return dict(summary)


def _validate_json_value(value: object, *, key: str | None = None) -> None:
    if key is not None and key.lower() in _RAW_AUDIO_KEYS:
        raise TelemetryError("raw PCM/audio fields must not be written to JSONL")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TelemetryError("JSON numeric values must be finite")
        return
    if isinstance(value, Mapping):
        for nested_key, nested_value in value.items():
            if not isinstance(nested_key, str):
                raise TelemetryError("JSON mapping keys must be strings")
            _validate_json_value(nested_value, key=nested_key)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_json_value(item)
        return
    raise TelemetryError(f"JSON serialization does not support {type(value).__name__}")


def _validated_dropped_audio_chunks(event: str, fields: Mapping[str, object]) -> int:
    if event != "audio_drop":
        return 0
    dropped = fields.get("dropped")
    if isinstance(dropped, bool) or not isinstance(dropped, int) or dropped < 1:
        raise TelemetryError("audio_drop events require a positive integer dropped field")
    return dropped


def _validated_speech_duration_ms(event: str, fields: Mapping[str, object]) -> float | None:
    if event != "vad_decision" or "speech_active_total_ms" not in fields:
        return None
    value = fields["speech_active_total_ms"]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise TelemetryError("speech_active_total_ms must be a finite nonnegative number")
    return float(value)


def _atomic_json_write(path: Path, payload: Mapping[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_name, path)
    except OSError as error:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise TelemetryError(f"could not write JSON file: {path}") from error
