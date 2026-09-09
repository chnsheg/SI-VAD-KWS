"""Deterministic four-gate VAD/KWS streaming controller."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable, Protocol

import numpy as np

from .contracts import CascadeConfig, TimingSchedule, VadContract
from .kws_executor import KwsJob, KwsResult


class CascadeState(StrEnum):
    """Externally visible controller states from the four-gate contract."""

    IDLE = "idle"
    VAD_CANDIDATE = "vad_candidate"
    KWS_ACTIVE = "kws_active"
    WAKE_LATCHED = "wake_latched"

    # Source-compatible aliases for callers that only need a coarse status.
    SILENCE = "idle"
    SPEECH_CANDIDATE = "vad_candidate"
    SPEECH_ACTIVE = "kws_active"
    COOLDOWN = "wake_latched"


class VadConfirmedState(StrEnum):
    SILENCE = "silence"
    SPEECH = "speech"


@dataclass(frozen=True)
class CascadeEvent:
    kind: str
    captured_ns: int
    fields: dict[str, object] = field(default_factory=dict)


class VadScoreRunner(Protocol):
    def latest_score(self, features: np.ndarray) -> float:
        """Return the newest VAD probability for a complete feature window."""


class KwsScoreRunner(Protocol):
    def score(self, waveform: np.ndarray) -> float:
        """Return the positive KWS probability for one complete PCM window."""


class KwsExecutionLane(Protocol):
    def submit(self, job: KwsJob) -> None: ...

    def collect_ordered(self, *, generation: int) -> list[KwsResult]: ...

    def invalidate(self, *, generation: int) -> None: ...

    def wait_for_generation(self, *, generation: int, timeout: float) -> bool: ...


FeatureExtractor = Callable[[np.ndarray, VadContract], np.ndarray]


class VadFeatureCache(Protocol):
    """Own cached causal VAD features for the active speech timeline."""

    def rebuild(self, waveform: np.ndarray) -> np.ndarray:
        """Build one continuous history and return its bootstrap features."""

    def append_pcm(self, pcm: np.ndarray) -> None:
        """Append continuous PCM while the stateful VAD path is active."""

    def take_new_features(self) -> np.ndarray:
        """Return each newly completed normalized feature frame once."""

    def reset(self) -> None:
        """Discard retained feature history after a state boundary."""


class PcmRingBuffer:
    """Fixed-size mono PCM history retaining the newest continuous samples."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._values = np.empty(capacity, dtype=np.float32)
        self._count = 0
        self._start = 0

    @property
    def sample_count(self) -> int:
        return self._count

    @property
    def is_full(self) -> bool:
        return self._count == self._capacity

    def append(self, pcm: np.ndarray) -> None:
        values = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if values.size >= self._capacity:
            self._values[:] = values[-self._capacity :]
            self._count = self._capacity
            self._start = 0
            return
        for value in values:
            if self._count < self._capacity:
                write_index = (self._start + self._count) % self._capacity
                self._values[write_index] = value
                self._count += 1
            else:
                self._values[self._start] = value
                self._start = (self._start + 1) % self._capacity

    def latest(self) -> np.ndarray:
        if not self.is_full:
            raise RuntimeError("PCM ring buffer does not yet contain a full one-second window")
        return self.snapshot()

    def snapshot(self) -> np.ndarray:
        """Return the retained PCM in chronological order, including partial history."""

        if self._count == 0:
            raise RuntimeError("PCM ring buffer is empty")
        if self._start == 0:
            return self._values[: self._count].copy()
        end = self._start + self._count
        if end <= self._capacity:
            return self._values[self._start : end].copy()
        return np.concatenate((self._values[self._start :], self._values[: end % self._capacity])).astype(
            np.float32, copy=False
        )

    def clear(self) -> None:
        self._count = 0
        self._start = 0


class CascadeEngine:
    """Apply energy, VAD, KWS and confirmation gates on a 16 kHz PCM timeline."""

    def __init__(
        self,
        *,
        config: CascadeConfig,
        vad_contract: VadContract,
        vad_runner: VadScoreRunner,
        kws_runner: KwsScoreRunner,
        kws_executor: KwsExecutionLane | None = None,
        feature_extractor: FeatureExtractor | None = None,
        vad_feature_cache: VadFeatureCache | None = None,
    ) -> None:
        config.validate()
        if vad_contract.sample_rate != config.sample_rate_hz:
            raise ValueError("cascade sample rate must match the VAD contract")
        if feature_extractor is None and vad_feature_cache is None:
            raise ValueError("either feature_extractor or vad_feature_cache is required")
        self.config = config
        self.vad_contract = vad_contract
        self._vad_runner = vad_runner
        self._kws_runner = kws_runner
        self._kws_executor = kws_executor
        self._kws_generation = 0
        self._kws_sequence = 0
        self._feature_extractor = feature_extractor
        self._vad_feature_cache = vad_feature_cache
        self._vad_cache_active = False
        self._effective_schedule = TimingSchedule.from_periods(
            config.vad_period_ms,
            config.kws_period_ms,
            sample_rate_hz=config.sample_rate_hz,
        )
        self._vad_tick_samples = config.sample_rate_hz * self._effective_schedule.vad_period_ms // 1000
        self._kws_tick_samples = config.sample_rate_hz * self._effective_schedule.kws_period_ms // 1000
        self._kws_uses_vad_ticks = (
            self._effective_schedule.kws_period_ms % self._effective_schedule.vad_period_ms == 0
        )
        self._kws_ticks_per_period = self._effective_schedule.kws_period_ms // self._effective_schedule.vad_period_ms
        self._window_samples = config.sample_rate_hz * config.window_ms // 1000
        self._kws_lookback_samples = config.sample_rate_hz * config.kws_lookback_ms // 1000
        self._ring = PcmRingBuffer(self._window_samples)
        self._kws_history = PcmRingBuffer(self._kws_lookback_samples)
        self._samples_until_vad_tick = self._vad_tick_samples
        self._last_tick_ns: int | None = None

        frame_samples = round(vad_contract.sample_rate * vad_contract.frame_ms / 1000.0)
        hop_samples = round(vad_contract.sample_rate * vad_contract.hop_ms / 1000.0)
        feature_frames = (self._window_samples - frame_samples) // hop_samples + 1
        self._vad_trailing_samples = self._window_samples - (
            (feature_frames - 1) * hop_samples + frame_samples
        )

        self.state = CascadeState.IDLE
        self.vad_confirmed_state = VadConfirmedState.SILENCE
        self.kws_call_count = 0
        self.vad_call_count = 0
        self.speech_active_ms = 0
        self.latest_vad_score: float | None = None
        self.latest_kws_score: float | None = None
        self.latest_vad_onnx_ms: float | None = None
        self.latest_kws_onnx_ms: float | None = None
        self._energy_gate_open = False
        self._energy_hangover_deadline_ns: int | None = None
        self._vad_energy_tail_deadline_ns: int | None = None
        self._vad_positive_count = 0
        self._vad_silence_count = 0
        self._wake_silence_count = 0
        self._kws_confirmation_count = 0
        self._kws_gate_open = False
        self._next_kws_window_end_ns: int | None = None
        self._diagnostic_next_kws_ns: int | None = None
        self._vad_no_speech_deadline_ns: int | None = None
        self._vad_threshold = float(config.vad_threshold)
        self._kws_threshold = float(config.kws_threshold)
        self._energy_gate_enabled = True
        self._vad_gate_enabled = True

    @property
    def kws_gate_open(self) -> bool:
        return self._kws_gate_open

    @property
    def effective_thresholds(self) -> dict[str, float]:
        """Return the session thresholds currently used by the two probability gates."""

        return {"vad_threshold": self._vad_threshold, "kws_threshold": self._kws_threshold}

    def set_runtime_thresholds(self, *, vad_threshold: float, kws_threshold: float) -> None:
        """Set a validated pair; the caller performs the accompanying safe reset."""

        self._vad_threshold = _require_probability("vad_threshold", vad_threshold)
        self._kws_threshold = _require_probability("kws_threshold", kws_threshold)

    @property
    def effective_schedule(self) -> dict[str, int]:
        """Return the periods currently used by the controller."""

        return {
            "energy_period_ms": self._effective_schedule.energy_period_ms,
            "vad_period_ms": self._effective_schedule.vad_period_ms,
            "kws_period_ms": self._effective_schedule.kws_period_ms,
        }

    def set_runtime_schedule(self, *, vad_period_ms: int, kws_period_ms: int) -> None:
        """Apply explicit periods; the caller must reset at a safe audio boundary."""

        schedule = TimingSchedule.from_periods(
            vad_period_ms, kws_period_ms, sample_rate_hz=self.config.sample_rate_hz
        )
        self._effective_schedule = schedule
        self._vad_tick_samples = self.config.sample_rate_hz * schedule.vad_period_ms // 1000
        self._kws_tick_samples = self.config.sample_rate_hz * schedule.kws_period_ms // 1000
        self._kws_uses_vad_ticks = schedule.kws_period_ms % schedule.vad_period_ms == 0
        self._samples_until_vad_tick = self._vad_tick_samples
        self._kws_ticks_per_period = schedule.kws_period_ms // schedule.vad_period_ms
        self._next_kws_window_end_ns = None

    def set_runtime_vad_period(self, vad_period_ms: int) -> None:
        """Keep the source-compatible default three-times KWS route."""

        self.set_runtime_schedule(vad_period_ms=vad_period_ms, kws_period_ms=vad_period_ms * 3)

    @property
    def effective_gates(self) -> dict[str, bool]:
        """Return the session gate modes currently used by the controller."""

        return {
            "energy_enabled": self._energy_gate_enabled,
            "vad_enabled": self._vad_gate_enabled,
        }

    def set_runtime_gates(self, *, energy_enabled: bool, vad_enabled: bool) -> None:
        """Set gate modes; the caller performs the accompanying safe reset."""

        self._energy_gate_enabled = _require_bool("energy_enabled", energy_enabled)
        self._vad_gate_enabled = _require_bool("vad_enabled", vad_enabled)

    @property
    def confirmation_count(self) -> int:
        return self._kws_confirmation_count

    @property
    def vad_positive_count(self) -> int:
        return self._vad_positive_count

    @property
    def vad_silence_count(self) -> int:
        return self._vad_silence_count

    @property
    def energy_hangover_remaining_ms(self) -> int:
        return self._remaining_ms(self._energy_hangover_deadline_ns)

    @property
    def kws_hangover_remaining_ms(self) -> int:
        """Compatibility name for diagnostics written before the four-gate upgrade."""

        return self.energy_hangover_remaining_ms

    @property
    def vad_no_speech_remaining_ms(self) -> int:
        return self._remaining_ms(self._vad_no_speech_deadline_ns)

    def push_pcm(self, pcm: np.ndarray, captured_ns: int) -> list[CascadeEvent]:
        """Process live PCM using the configured cascade gates."""

        return self._push_pcm(pcm, captured_ns, diagnostic_full_scoring=False)

    def push_diagnostic_pcm(self, pcm: np.ndarray, captured_ns: int) -> list[CascadeEvent]:
        """Process replay PCM with the same gate ordering as live capture."""

        return self.push_pcm(pcm, captured_ns)

    def drain_kws_results(self, *, wait: bool = False, timeout: float = 0.0) -> list[CascadeEvent]:
        """Commit ready asynchronous KWS results in their captured-time order."""

        if self._kws_executor is None:
            return []
        if wait:
            wait_for_generation = getattr(self._kws_executor, "wait_for_generation", None)
            if not callable(wait_for_generation):
                raise RuntimeError("KWS execution lane cannot wait for an offline analysis result")
            if not wait_for_generation(generation=self._kws_generation, timeout=timeout):
                raise RuntimeError("timed out waiting for KWS analysis results")
        events: list[CascadeEvent] = []
        for result in self._kws_executor.collect_ordered(generation=self._kws_generation):
            if not self._kws_gate_open:
                break
            if result.error is not None or result.score is None:
                events.append(
                    CascadeEvent(
                        "kws_error",
                        result.captured_ns,
                        {
                            "error": result.error or "KWS score missing",
                            "window_end_ns": self._result_window_end_ns(result),
                        },
                    )
                )
                continue
            self._commit_kws_score(result, events)
        return events

    def _push_pcm(
        self, pcm: np.ndarray, captured_ns: int, *, diagnostic_full_scoring: bool
    ) -> list[CascadeEvent]:
        """Append normalized PCM and execute each completed effective controller tick."""

        values = np.asarray(pcm)
        if values.dtype != np.float32 or values.ndim != 1 or values.size == 0:
            raise ValueError("PCM must be a nonempty float32 mono array")
        if not np.all(np.isfinite(values)):
            raise ValueError("PCM must contain only finite values")
        if isinstance(captured_ns, bool) or not isinstance(captured_ns, int) or captured_ns < 0:
            raise ValueError("captured_ns must be a nonnegative integer")

        source_start_ns = captured_ns - values.size * 1_000_000_000 // self.config.sample_rate_hz
        cursor = 0
        events = self.drain_kws_results()
        while cursor < values.size:
            current_end_ns = source_start_ns + cursor * 1_000_000_000 // self.config.sample_rate_hz
            if self._kws_schedule_is_due(current_end_ns):
                events.extend(
                    self._advance_kws_schedule(
                        captured_ns=current_end_ns,
                        vad_feature_ms=0.0,
                        started_ns=time.perf_counter_ns(),
                    )
                )
                continue
            until_kws_tick = self._samples_until_next_kws_window(current_end_ns)
            take = min(
                self._samples_until_vad_tick,
                values.size - cursor,
                until_kws_tick if until_kws_tick is not None else values.size - cursor,
            )
            had_full_window = self._ring.is_full
            appended = values[cursor : cursor + take]
            self._ring.append(appended)
            self._kws_history.append(appended)
            if self._vad_cache_active and self._vad_feature_cache is not None:
                self._vad_feature_cache.append_pcm(appended)
            cursor += take
            self._samples_until_vad_tick -= take
            first_complete_window = not had_full_window and self._ring.is_full
            if first_complete_window:
                boundary_ns = source_start_ns + cursor * 1_000_000_000 // self.config.sample_rate_hz
                self._last_tick_ns = boundary_ns
                events.extend(self._on_vad_tick(boundary_ns, diagnostic_full_scoring=diagnostic_full_scoring))
                self._samples_until_vad_tick = self._vad_tick_samples
            elif self._samples_until_vad_tick == 0:
                if self._ring.is_full:
                    boundary_ns = source_start_ns + cursor * 1_000_000_000 // self.config.sample_rate_hz
                    self._last_tick_ns = boundary_ns
                    events.extend(self._on_vad_tick(boundary_ns, diagnostic_full_scoring=diagnostic_full_scoring))
                self._samples_until_vad_tick = self._vad_tick_samples
            boundary_ns = source_start_ns + cursor * 1_000_000_000 // self.config.sample_rate_hz
            events.extend(
                self._advance_kws_schedule(
                    captured_ns=boundary_ns,
                    vad_feature_ms=0.0,
                    started_ns=time.perf_counter_ns(),
                )
            )
        return events

    def reset(self, captured_ns: int) -> list[CascadeEvent]:
        """Discard every time-dependent state after a discontinuity or manual reset."""

        if isinstance(captured_ns, bool) or not isinstance(captured_ns, int) or captured_ns < 0:
            raise ValueError("captured_ns must be a nonnegative integer")
        self._kws_generation += 1
        self._kws_sequence = 0
        if self._kws_executor is not None:
            reset_history = getattr(self._kws_executor, "reset_history", None)
            if callable(reset_history):
                reset_history()
            self._kws_executor.invalidate(generation=self._kws_generation)
        self._ring.clear()
        self._kws_history.clear()
        self._samples_until_vad_tick = self._vad_tick_samples
        self._next_kws_window_end_ns = None
        self._last_tick_ns = captured_ns
        self._energy_gate_open = False
        self._energy_hangover_deadline_ns = None
        self._vad_energy_tail_deadline_ns = None
        self._vad_positive_count = 0
        self._vad_silence_count = 0
        self._wake_silence_count = 0
        self.speech_active_ms = 0
        self.vad_confirmed_state = VadConfirmedState.SILENCE
        self._vad_no_speech_deadline_ns = None
        self._close_kws_gate(clear_confirmation=True)
        self._diagnostic_next_kws_ns = None
        self._reset_kws_runner()
        self._reset_vad_pipeline()
        self.latest_vad_score = None
        self.latest_kws_score = None
        self.latest_vad_onnx_ms = None
        self.latest_kws_onnx_ms = None
        events: list[CascadeEvent] = []
        self._transition(CascadeState.IDLE, captured_ns, events, reason="reset", force=True)
        return events

    def latest_window(self) -> np.ndarray:
        return self._ring.latest()

    def _on_vad_tick(self, captured_ns: int, *, diagnostic_full_scoring: bool) -> list[CascadeEvent]:
        started_ns = time.perf_counter_ns()
        vad_waveform = self._ring.latest()
        kws_waveform = (
            self._kws_history.latest()[: self._window_samples].copy()
            if self._kws_history.is_full
            else None
        )
        energy_dbfs = _rms_dbfs(vad_waveform[-self._vad_tick_samples :])
        energy_pass = energy_dbfs > self.config.energy_threshold_dbfs
        events: list[CascadeEvent] = []
        self._advance_energy_gate(energy_pass, captured_ns, energy_dbfs, events)

        if not self._vad_gate_enabled:
            events.extend(
                self._on_kws_only_tick(
                    captured_ns,
                    energy_dbfs,
                    energy_pass,
                    started_ns,
                    [],
                )
            )
            if diagnostic_full_scoring:
                self._append_diagnostic_vad_score(
                    vad_waveform, captured_ns, energy_dbfs, energy_pass, started_ns, events
                )
                if self.state is not CascadeState.KWS_ACTIVE:
                    if kws_waveform is not None:
                        self._append_diagnostic_kws_score(kws_waveform, captured_ns, started_ns, events)
            return events

        tail_active = (
            self._vad_energy_tail_deadline_ns is not None
            and captured_ns < self._vad_energy_tail_deadline_ns
        )
        if (
            self.state is CascadeState.IDLE
            and self._energy_gate_enabled
            and not energy_pass
            and not tail_active
        ):
            events.append(
                CascadeEvent(
                    "energy_decision",
                    captured_ns,
                    self._gate_fields(energy_dbfs, energy_pass, total_ms=self._elapsed_ms(started_ns)),
                )
            )
            if diagnostic_full_scoring:
                self._append_diagnostic_vad_score(
                    vad_waveform, captured_ns, energy_dbfs, energy_pass, started_ns, events
                )
                if kws_waveform is not None:
                    self._append_diagnostic_kws_score(kws_waveform, captured_ns, started_ns, events)
            return events

        feature_started_ns = time.perf_counter_ns()
        feature_cache_replay = False
        if self.state is CascadeState.IDLE and self._vad_feature_cache is not None:
            self._reset_vad_runner()
            features = self._vad_feature_cache.rebuild(vad_waveform)
            self._vad_cache_active = True
            feature_cache_replay = True
        elif self._vad_feature_cache is not None:
            features = self._vad_feature_cache.take_new_features()
        else:
            assert self._feature_extractor is not None
            features = self._feature_extractor(vad_waveform, self.vad_contract)
        feature_ms = self._elapsed_ms(feature_started_ns)
        if features.dtype != np.float32 or features.ndim != 2 or features.shape[0] == 0:
            raise RuntimeError("VAD feature path did not produce complete float32 frames")
        if self.state is CascadeState.IDLE:
            self._transition(
                CascadeState.VAD_CANDIDATE,
                captured_ns,
                events,
                reason="energy_gate_open" if self._energy_gate_enabled else "vad_gate_enabled",
            )
        vad_started_ns = time.perf_counter_ns()
        vad_score = float(self._vad_runner.latest_score(features))
        vad_ort_ms = self._elapsed_ms(vad_started_ns)
        if not math.isfinite(vad_score):
            raise RuntimeError("VAD runner returned a non-finite score")
        self.latest_vad_score = vad_score
        self.latest_vad_onnx_ms = vad_ort_ms
        self.vad_call_count += 1

        self._advance_vad_state(vad_score, captured_ns, events)
        if self.state is CascadeState.KWS_ACTIVE:
            self.speech_active_ms += self._effective_schedule.vad_period_ms
        window_start_ns = captured_ns - self.config.window_ms * 1_000_000
        vad_window_end_ns = (
            captured_ns
            - self._vad_trailing_samples * 1_000_000_000 // self.config.sample_rate_hz
        )
        state_ms = 0.0
        events.append(
            CascadeEvent(
                "vad_decision",
                captured_ns,
                {
                    "score": vad_score,
                    "frontend_ms": feature_ms,
                    "feature_cache_hit": self._vad_feature_cache is not None and not feature_cache_replay,
                    "feature_cache_deferred_replay": feature_cache_replay,
                    "feature_frame_count": int(features.shape[0]),
                    "vad_onnx_ms": vad_ort_ms,
                    "onnx_ms": vad_ort_ms,
                    "state_ms": state_ms,
                    "total_ms": self._elapsed_ms(started_ns),
                    "model_input_start_ns": window_start_ns,
                    "model_input_end_ns": vad_window_end_ns,
                    "window_end_ns": vad_window_end_ns,
                    **self._gate_fields(energy_dbfs, energy_pass),
                    "speech_active_total_ms": self.speech_active_ms,
                },
            )
        )
        if diagnostic_full_scoring and self.state is not CascadeState.KWS_ACTIVE and kws_waveform is not None:
            self._append_diagnostic_kws_score(kws_waveform, captured_ns, started_ns, events)
        return events

    def _append_diagnostic_vad_score(
        self,
        waveform: np.ndarray,
        captured_ns: int,
        energy_dbfs: float,
        energy_pass: bool,
        started_ns: int,
        events: list[CascadeEvent],
    ) -> None:
        self._reset_vad_runner()
        feature_started_ns = time.perf_counter_ns()
        if self._vad_feature_cache is not None:
            features = self._vad_feature_cache.rebuild(waveform)
        else:
            assert self._feature_extractor is not None
            features = self._feature_extractor(waveform, self.vad_contract)
        feature_ms = self._elapsed_ms(feature_started_ns)
        if features.dtype != np.float32 or features.ndim != 2 or features.shape[0] == 0:
            raise RuntimeError("VAD feature path did not produce complete float32 frames")
        vad_started_ns = time.perf_counter_ns()
        vad_score = float(self._vad_runner.latest_score(features))
        vad_ort_ms = self._elapsed_ms(vad_started_ns)
        if not math.isfinite(vad_score):
            raise RuntimeError("VAD runner returned a non-finite score")
        self.latest_vad_score = vad_score
        self.latest_vad_onnx_ms = vad_ort_ms
        self.vad_call_count += 1
        window_start_ns = captured_ns - self.config.window_ms * 1_000_000
        vad_window_end_ns = (
            captured_ns
            - self._vad_trailing_samples * 1_000_000_000 // self.config.sample_rate_hz
        )
        events.append(
            CascadeEvent(
                "diagnostic_vad_score",
                captured_ns,
                {
                    "score": vad_score,
                    "frontend_ms": feature_ms,
                    "vad_onnx_ms": vad_ort_ms,
                    "onnx_ms": vad_ort_ms,
                    "total_ms": self._elapsed_ms(started_ns),
                    "model_input_start_ns": window_start_ns,
                    "model_input_end_ns": vad_window_end_ns,
                    "window_end_ns": vad_window_end_ns,
                    "diagnostic_only": True,
                    **self._gate_fields(energy_dbfs, energy_pass),
                },
            )
        )

    def _append_diagnostic_kws_score(
        self,
        waveform: np.ndarray,
        captured_ns: int,
        started_ns: int,
        events: list[CascadeEvent],
    ) -> None:
        if (
            self._diagnostic_next_kws_ns is not None
            and captured_ns < self._diagnostic_next_kws_ns
        ):
            return
        kws_started_ns = time.perf_counter_ns()
        kws_score = float(self._kws_runner.score(waveform))
        kws_ort_ms = self._elapsed_ms(kws_started_ns)
        if not math.isfinite(kws_score):
            raise RuntimeError("KWS runner returned a non-finite score")
        self.latest_kws_score = kws_score
        self.latest_kws_onnx_ms = kws_ort_ms
        self.kws_call_count += 1
        self._diagnostic_next_kws_ns = captured_ns + self._effective_schedule.kws_period_ms * 1_000_000
        events.append(
            CascadeEvent(
                "diagnostic_kws_score",
                captured_ns,
                {
                    "score": kws_score,
                    "kws_backbone_ms": kws_ort_ms,
                    "onnx_ms": kws_ort_ms,
                    "total_ms": self._elapsed_ms(started_ns),
                    "window_end_ns": captured_ns,
                    "model_input_start_ns": captured_ns - self.config.window_ms * 1_000_000,
                    "model_input_end_ns": captured_ns,
                    "diagnostic_only": True,
                    **self._gate_fields(None, None),
                },
            )
        )

    def _advance_energy_gate(
        self, energy_pass: bool, captured_ns: int, energy_dbfs: float, events: list[CascadeEvent]
    ) -> None:
        if (
            self._energy_gate_enabled
            and self.state is CascadeState.KWS_ACTIVE
            and self._energy_gate_open
            and not energy_pass
        ):
            self._energy_hangover_deadline_ns = captured_ns + self.config.energy_hangover_ms * 1_000_000
            events.append(
                CascadeEvent(
                    "energy_hangover_started",
                    captured_ns,
                    {
                        "energy_dbfs": energy_dbfs,
                        "energy_hangover_remaining_ms": self.config.energy_hangover_ms,
                    },
                )
            )
        elif energy_pass:
            self._energy_hangover_deadline_ns = None
        if energy_pass and self._energy_gate_enabled:
            self._vad_energy_tail_deadline_ns = (
                captured_ns + self.config.vad_energy_tail_ms * 1_000_000
            )
        self._energy_gate_open = energy_pass

    def _on_kws_only_tick(
        self,
        captured_ns: int,
        energy_dbfs: float,
        energy_pass: bool,
        started_ns: int,
        events: list[CascadeEvent],
    ) -> list[CascadeEvent]:
        """Run KWS without VAD, retaining the physical energy telemetry."""

        if self.state is CascadeState.WAKE_LATCHED:
            self._transition(CascadeState.IDLE, captured_ns, events, reason="kws_only_rearm")

        energy_permits_kws = (
            not self._energy_gate_enabled
            or energy_pass
            or self.energy_hangover_remaining_ms > 0
        )
        if energy_permits_kws:
            if self.state is CascadeState.IDLE:
                self._open_kws_gate()
                self._transition(
                    CascadeState.KWS_ACTIVE,
                    captured_ns,
                    events,
                    reason="kws_only_energy_open" if self._energy_gate_enabled else "kws_only_always_on",
                )
        elif self.state is CascadeState.KWS_ACTIVE:
            self._close_kws_gate(clear_confirmation=True)
            self._transition(CascadeState.IDLE, captured_ns, events, reason="energy_hangover_expired")

        events.append(
            CascadeEvent(
                "energy_decision",
                captured_ns,
                self._gate_fields(energy_dbfs, energy_pass, total_ms=self._elapsed_ms(started_ns)),
            )
        )
        return events

    def _advance_vad_state(
        self, vad_score: float, captured_ns: int, events: list[CascadeEvent]
    ) -> None:
        vad_positive = vad_score > self._vad_threshold
        was_confirmed = self.vad_confirmed_state
        if vad_positive:
            self._vad_positive_count += 1
            self._vad_silence_count = 0
            if self._vad_positive_count >= self.config.vad_confirmations:
                self.vad_confirmed_state = VadConfirmedState.SPEECH
        else:
            self._vad_positive_count = 0
            self._vad_silence_count += 1
            if self._vad_silence_count >= self.config.vad_confirmations:
                self.vad_confirmed_state = VadConfirmedState.SILENCE

        confirmed_changed = was_confirmed is not self.vad_confirmed_state
        if confirmed_changed:
            events.append(
                CascadeEvent(
                    "vad_confirmed_state",
                    captured_ns,
                    {
                        "from_state": was_confirmed.value,
                        "to_state": self.vad_confirmed_state.value,
                        "positive_count": self._vad_positive_count,
                        "silence_count": self._vad_silence_count,
                    },
                )
            )

        if self.state is CascadeState.VAD_CANDIDATE:
            if self.vad_confirmed_state is VadConfirmedState.SPEECH:
                self._open_kws_gate()
                self._transition(CascadeState.KWS_ACTIVE, captured_ns, events, reason="vad_speech_confirmed")
            elif self._vad_silence_count >= self.config.vad_confirmations:
                self._transition(CascadeState.IDLE, captured_ns, events, reason="vad_candidate_silence_confirmed")
            return

        if self.state is CascadeState.KWS_ACTIVE:
            if confirmed_changed and self.vad_confirmed_state is VadConfirmedState.SPEECH:
                self._vad_no_speech_deadline_ns = None
            elif confirmed_changed and self.vad_confirmed_state is VadConfirmedState.SILENCE:
                self._vad_no_speech_deadline_ns = (
                    captured_ns + self.config.vad_no_speech_timeout_ms * 1_000_000
                )
                events.append(
                    CascadeEvent(
                        "vad_no_speech_started",
                        captured_ns,
                        {
                            "deadline_ns": self._vad_no_speech_deadline_ns,
                            "timeout_ms": self.config.vad_no_speech_timeout_ms,
                        },
                    )
                )
            if (
                self._vad_no_speech_deadline_ns is not None
                and captured_ns >= self._vad_no_speech_deadline_ns
            ):
                self._close_kws_gate(clear_confirmation=True)
                self._vad_no_speech_deadline_ns = None
                self._transition(CascadeState.IDLE, captured_ns, events, reason="vad_no_speech_timeout")
            return

        if self.state is CascadeState.WAKE_LATCHED:
            if vad_positive:
                self._wake_silence_count = 0
            else:
                self._wake_silence_count += 1
                if self._wake_silence_count >= self.config.wake_silence_confirmations:
                    self._close_kws_gate(clear_confirmation=True)
                    self._vad_no_speech_deadline_ns = None
                    self._transition(CascadeState.IDLE, captured_ns, events, reason="wake_silence_confirmed")
                    self._wake_silence_count = 0

    def _advance_kws_schedule(
        self,
        *,
        captured_ns: int,
        vad_feature_ms: float,
        started_ns: int,
    ) -> list[CascadeEvent]:
        """Score every due KWS window in PCM-time order without waiting for wall time."""

        events: list[CascadeEvent] = []
        if not self._kws_gate_open or self._kws_history.sample_count < self._window_samples:
            return events
        if self._next_kws_window_end_ns is None:
            # Anchor the initial catch-up window at the retained history's left edge.
            # This preserves the KWS input phase when lookback minus window length
            # is not an exact multiple of the configured hop (for example 500 / 96 ms).
            self._next_kws_window_end_ns = (
                captured_ns
                - (self.config.kws_lookback_ms - self.config.window_ms) * 1_000_000
            )
        period_ns = self._effective_schedule.kws_period_ms * 1_000_000
        while self._kws_gate_open and self._next_kws_window_end_ns <= captured_ns:
            window_end_ns = self._next_kws_window_end_ns
            waveform = self._kws_window_ending_at(
                window_end_ns=window_end_ns,
                available_end_ns=captured_ns,
            )
            if waveform is None:
                break
            next_window_end_ns = window_end_ns + period_ns
            self._run_kws_at_boundary(
                waveform,
                captured_ns,
                window_end_ns - self.config.window_ms * 1_000_000,
                vad_feature_ms,
                started_ns,
                events,
                window_end_ns=window_end_ns,
            )
            if self._kws_gate_open:
                self._next_kws_window_end_ns = next_window_end_ns
        return events

    def _kws_schedule_is_due(self, captured_ns: int) -> bool:
        return (
            self._kws_gate_open
            and self._next_kws_window_end_ns is not None
            and self._next_kws_window_end_ns <= captured_ns
        )

    def _samples_until_next_kws_window(self, captured_ns: int) -> int | None:
        if not self._kws_gate_open or self._next_kws_window_end_ns is None:
            return None
        remaining_ns = self._next_kws_window_end_ns - captured_ns
        if remaining_ns <= 0:
            return 0
        return max(
            1,
            remaining_ns * self.config.sample_rate_hz // 1_000_000_000,
        )

    def _kws_window_ending_at(
        self, *, window_end_ns: int, available_end_ns: int
    ) -> np.ndarray | None:
        history = self._kws_history.snapshot()
        trailing_ns = available_end_ns - window_end_ns
        if trailing_ns < 0:
            return None
        trailing_samples = trailing_ns * self.config.sample_rate_hz // 1_000_000_000
        end_index = history.size - trailing_samples
        start_index = end_index - self._window_samples
        if start_index < 0 or end_index > history.size:
            return None
        waveform = history[start_index:end_index]
        if waveform.shape != (self._window_samples,):
            return None
        return waveform.copy()

    def _run_kws_at_boundary(
        self,
        waveform: np.ndarray,
        captured_ns: int,
        window_start_ns: int,
        vad_feature_ms: float,
        started_ns: int,
        events: list[CascadeEvent],
        *,
        window_end_ns: int | None = None,
    ) -> None:
        effective_window_end_ns = captured_ns if window_end_ns is None else window_end_ns
        if self._kws_executor is not None:
            self._kws_executor.submit(
                KwsJob(
                    sequence=self._kws_sequence,
                    generation=self._kws_generation,
                    captured_ns=captured_ns,
                    waveform=waveform.copy(),
                    window_end_ns=effective_window_end_ns,
                )
            )
            self._kws_sequence += 1
            return
        kws_started_ns = time.perf_counter_ns()
        kws_score = float(self._kws_runner.score(waveform))
        kws_ort_ms = self._elapsed_ms(kws_started_ns)
        if not math.isfinite(kws_score):
            raise RuntimeError("KWS runner returned a non-finite score")
        self.latest_kws_score = kws_score
        self.latest_kws_onnx_ms = kws_ort_ms
        self.kws_call_count += 1
        events.append(
            CascadeEvent(
                "kws_call",
                captured_ns,
                {
                    "score": kws_score,
                    "frontend_ms": vad_feature_ms,
                    "kws_frontend_ms": 0.0,
                    "kws_backbone_ms": kws_ort_ms,
                    "onnx_ms": kws_ort_ms,
                    "total_ms": self._elapsed_ms(started_ns),
                    "window_end_ns": effective_window_end_ns,
                    "model_input_start_ns": window_start_ns,
                    "model_input_end_ns": effective_window_end_ns,
                    **self._gate_fields(None, None),
                },
            )
        )
        if kws_score > self._kws_threshold:
            self._kws_confirmation_count += 1
            if self._kws_confirmation_count >= self.config.kws_confirmations:
                events.append(
                    CascadeEvent(
                        "wake",
                        captured_ns,
                        {
                            "score": kws_score,
                            "confirmations": self._kws_confirmation_count,
                            "window_end_ns": effective_window_end_ns,
                        },
                    )
                )
                self._close_kws_gate(clear_confirmation=False)
                self._transition(CascadeState.WAKE_LATCHED, captured_ns, events, reason="kws_confirmed")
                self._wake_silence_count = 0
                return
        else:
            self._kws_confirmation_count = 0

    def _commit_kws_score(self, result: KwsResult, events: list[CascadeEvent]) -> None:
        """Apply the legacy threshold state machine to an ordered worker result."""

        assert result.score is not None
        window_end_ns = self._result_window_end_ns(result)
        self.latest_kws_score = result.score
        self.latest_kws_onnx_ms = result.inference_ms
        self.kws_call_count += 1
        events.append(
            CascadeEvent(
                "kws_call",
                result.captured_ns,
                {
                    "score": result.score,
                    "frontend_ms": 0.0,
                    "kws_frontend_ms": 0.0,
                    "kws_backbone_ms": result.inference_ms,
                    "onnx_ms": result.inference_ms,
                    "total_ms": result.inference_ms,
                    "window_end_ns": window_end_ns,
                    "model_input_start_ns": window_end_ns - self.config.window_ms * 1_000_000,
                    "model_input_end_ns": window_end_ns,
                    **self._gate_fields(None, None),
                },
            )
        )
        if result.score > self._kws_threshold:
            self._kws_confirmation_count += 1
            if self._kws_confirmation_count >= self.config.kws_confirmations:
                events.append(
                    CascadeEvent(
                        "wake",
                        result.captured_ns,
                        {
                            "score": result.score,
                            "confirmations": self._kws_confirmation_count,
                            "window_end_ns": window_end_ns,
                        },
                    )
                )
                self._close_kws_gate(clear_confirmation=False)
                self._transition(CascadeState.WAKE_LATCHED, result.captured_ns, events, reason="kws_confirmed")
                self._wake_silence_count = 0
                return
        else:
            self._kws_confirmation_count = 0

    @staticmethod
    def _result_window_end_ns(result: KwsResult) -> int:
        return result.captured_ns if result.window_end_ns is None else result.window_end_ns

    def _open_kws_gate(self) -> None:
        self._kws_gate_open = True
        if self._kws_executor is not None:
            discard_submitted_windows = getattr(self._kws_executor, "discard_submitted_windows", None)
            if callable(discard_submitted_windows):
                discard_submitted_windows()
        if self._kws_executor is not None and self._kws_sequence:
            self._kws_generation += 1
            self._kws_sequence = 0
            self._kws_executor.invalidate(generation=self._kws_generation)
        self._next_kws_window_end_ns = None
        self._kws_confirmation_count = 0
        self._vad_no_speech_deadline_ns = None

    def _close_kws_gate(self, *, clear_confirmation: bool) -> None:
        self._kws_gate_open = False
        self._next_kws_window_end_ns = None
        if self._kws_executor is not None and self._kws_sequence:
            self._kws_generation += 1
            self._kws_sequence = 0
            self._kws_executor.invalidate(generation=self._kws_generation)
        self._energy_hangover_deadline_ns = None
        if clear_confirmation:
            self._kws_confirmation_count = 0

    def _reset_vad_runner(self) -> None:
        reset = getattr(self._vad_runner, "reset", None)
        if callable(reset):
            reset()

    def _reset_vad_pipeline(self) -> None:
        """Clear cached features and optional CNN/GRU state at an explicit boundary."""

        if self._vad_cache_active:
            self._deactivate_streaming_vad()
            return
        if self._vad_feature_cache is not None:
            self._vad_feature_cache.reset()
        self._reset_vad_runner()

    def _deactivate_streaming_vad(self) -> None:
        if not self._vad_cache_active:
            return
        if self._vad_feature_cache is not None:
            self._vad_feature_cache.reset()
        self._reset_vad_runner()
        self._vad_cache_active = False

    def _reset_kws_runner(self) -> None:
        """Clear optional KWS streaming state without constraining legacy runners."""

        reset = getattr(self._kws_runner, "reset", None)
        if callable(reset):
            reset()

    def _transition(
        self,
        state: CascadeState,
        captured_ns: int,
        events: list[CascadeEvent],
        *,
        reason: str,
        force: bool = False,
    ) -> None:
        if not force and self.state is state:
            return
        previous = self.state
        if state is CascadeState.IDLE and previous is not CascadeState.IDLE:
            self._deactivate_streaming_vad()
        self.state = state
        events.append(
            CascadeEvent(
                "state_transition",
                captured_ns,
                {"from_state": previous.value, "to_state": state.value, "reason": reason},
            )
        )

    def _gate_fields(
        self, energy_dbfs: float | None, energy_pass: bool | None, *, total_ms: float | None = None
    ) -> dict[str, object]:
        fields: dict[str, object] = {
            "energy_dbfs": energy_dbfs,
            "energy_gate": energy_pass,
            "energy_gate_enabled": self._energy_gate_enabled,
            "vad_gate_enabled": self._vad_gate_enabled,
            "energy_gate_effective": (
                None
                if energy_pass is None
                else (energy_pass or not self._energy_gate_enabled)
            ),
            "energy_hangover_remaining_ms": self.energy_hangover_remaining_ms,
            "vad_positive_count": self._vad_positive_count,
            "vad_silence_count": self._vad_silence_count,
            "vad_confirmed_state": self.vad_confirmed_state.value,
            "kws_positive_count": self._kws_confirmation_count,
            "kws_gate_open": self._kws_gate_open,
            "vad_no_speech_remaining_ms": self.vad_no_speech_remaining_ms,
            "state": self.state.value,
        }
        if total_ms is not None:
            fields["total_ms"] = total_ms
        return fields

    def _remaining_ms(self, deadline_ns: int | None) -> int:
        if deadline_ns is None or self._last_tick_ns is None:
            return 0
        return max(0, math.ceil((deadline_ns - self._last_tick_ns) / 1_000_000.0))

    @staticmethod
    def _elapsed_ms(started_ns: int) -> float:
        return (time.perf_counter_ns() - started_ns) / 1_000_000.0


def _rms_dbfs(samples: np.ndarray) -> float:
    values = np.asarray(samples, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float32), dtype=np.float32)))
    return 20.0 * math.log10(max(rms, np.finfo(np.float32).eps))


def _require_probability(name: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be in [0, 1]")
    return float(value)


def _require_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value
