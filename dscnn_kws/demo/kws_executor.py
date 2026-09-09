"""Bounded background KWS inference with ordered result delivery."""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


class KwsScoreRunner(Protocol):
    def score(self, waveform: np.ndarray) -> float:
        """Return one finite keyword probability for a 16 kHz one-second waveform."""


@dataclass(frozen=True)
class KwsJob:
    sequence: int
    generation: int
    captured_ns: int
    waveform: np.ndarray
    window_end_ns: int | None = None
    phase_probe_waveforms: tuple[tuple[int, np.ndarray], ...] = ()
    phase_probe_lower: float | None = None
    phase_probe_threshold: float | None = None
    slot_sequence: int | None = None
    phase_offset_ms: int = 0
    decision_kind: str = "logical_slot"
    phase_confirmation_of_window_end_ns: int | None = None
    phase_confirmation_phase_offset_ms: int | None = None
    phase_refinement_primary_score: float | None = None
    phase_refinement_neighbors: tuple[tuple[int, tuple[tuple[int, np.ndarray], ...]], ...] = ()
    phase_refinement_of_window_end_ns: int | None = None


@dataclass(frozen=True)
class KwsResult:
    sequence: int
    generation: int
    captured_ns: int
    score: float | None
    inference_ms: float
    error: str | None
    window_end_ns: int | None = None
    primary_score: float | None = None
    phase_scores: tuple[tuple[int, float], ...] = ()
    decision_kind: str = "logical_slot"
    phase_confirmation_of_window_end_ns: int | None = None
    phase_confirmation_phase_offset_ms: int | None = None
    phase_refinement_of_window_end_ns: int | None = None


class RealtimeKwsExecutor:
    """Score fixed KWS windows without blocking the serial PCM/VAD controller."""

    def __init__(
        self,
        runner_factory: Callable[[], KwsScoreRunner],
        *,
        worker_count: int = 2,
        queue_capacity: int = 8,
    ) -> None:
        if not callable(runner_factory):
            raise ValueError("runner_factory must be callable")
        if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count < 1:
            raise ValueError("worker_count must be a positive integer")
        if isinstance(queue_capacity, bool) or not isinstance(queue_capacity, int) or queue_capacity < 1:
            raise ValueError("queue_capacity must be a positive integer")
        self._runner_factory = runner_factory
        self._jobs: queue.Queue[KwsJob | None] = queue.Queue(maxsize=queue_capacity)
        self._confirmation_jobs: queue.SimpleQueue[KwsJob] = queue.SimpleQueue()
        self._completed: queue.SimpleQueue[KwsResult] = queue.SimpleQueue()
        self._confirmation_completed: queue.SimpleQueue[KwsResult] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._active_generation: int | None = None
        self._next_sequence = 0
        self._ready: dict[int, KwsResult] = {}
        self._closed = False
        self._submitted = 0
        self._submitted_by_generation: dict[int, int] = {}
        self._completed_count = 0
        self._completed_by_generation: dict[int, int] = {}
        self._overload_count = 0
        self._stale_count = 0
        self._failure_count = 0
        self._probe_inference_count = 0
        self._confirmation_inference_count = 0
        self._last_inference_ms: float | None = None
        self._completion_condition = threading.Condition(self._lock)
        self._threads = [
            threading.Thread(target=self._worker, daemon=True, name=f"vad-kws-kws-{index}")
            for index in range(worker_count)
        ]
        for thread in self._threads:
            thread.start()

    def submit(self, job: KwsJob) -> None:
        """Queue a copied KWS window, or record an ordered overload result."""

        copied = _validate_job(job)
        with self._lock:
            if self._closed:
                raise RuntimeError("KWS executor is closed")
            if self._active_generation is None:
                self._active_generation = copied.generation
                self._next_sequence = copied.sequence
            elif copied.generation != self._active_generation:
                raise ValueError("job generation is not active")
            self._submitted += 1
            self._submitted_by_generation[copied.generation] = (
                self._submitted_by_generation.get(copied.generation, 0) + 1
            )
        try:
            if copied.decision_kind in {"phase_confirmation", "phase_refinement"}:
                self._confirmation_jobs.put(copied)
                return
            self._jobs.put_nowait(copied)
        except queue.Full:
            with self._completion_condition:
                self._overload_count += 1
                self._completed_count += 1
                self._failure_count += 1
                self._completed_by_generation[copied.generation] = (
                    self._completed_by_generation.get(copied.generation, 0) + 1
                )
                self._completion_condition.notify_all()
            self._completed.put(
                KwsResult(
                    sequence=copied.sequence,
                    generation=copied.generation,
                    captured_ns=copied.captured_ns,
                    score=None,
                    inference_ms=0.0,
                    error="kws_executor_overload",
                    window_end_ns=copied.window_end_ns,
                    primary_score=None,
                    phase_scores=(),
                )
            )

    def collect_phase_confirmations(self, *, generation: int) -> list[KwsResult]:
        """Return completed priority confirmation work for one live generation."""

        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ValueError("generation must be a nonnegative integer")
        result: list[KwsResult] = []
        while True:
            try:
                value = self._confirmation_completed.get_nowait()
            except queue.Empty:
                return result
            if value.generation == generation:
                result.append(value)
            else:
                with self._lock:
                    self._stale_count += 1

    def wait_for_generation(self, *, generation: int, timeout: float) -> bool:
        """Wait for jobs submitted before this call to complete, without collecting them."""

        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ValueError("generation must be a nonnegative integer")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout):
            raise ValueError("timeout must be a finite number")
        if timeout < 0:
            raise ValueError("timeout must be nonnegative")
        deadline = time.monotonic() + float(timeout)
        with self._completion_condition:
            submitted = self._submitted_by_generation.get(generation, 0)
            while self._completed_by_generation.get(generation, 0) < submitted:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._completion_condition.wait(remaining)
            return True

    def collect_ordered(self, *, generation: int) -> list[KwsResult]:
        """Return each completed result once, in submission order for one live generation."""

        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ValueError("generation must be a nonnegative integer")
        with self._lock:
            if self._active_generation is None:
                self._active_generation = generation
                self._next_sequence = 0
            elif generation != self._active_generation:
                if generation < self._active_generation:
                    return []
                self._active_generation = generation
                self._next_sequence = 0
                self._ready.clear()
            self._drain_completed_locked(generation)
            result: list[KwsResult] = []
            while self._next_sequence in self._ready:
                result.append(self._ready.pop(self._next_sequence))
                self._next_sequence += 1
            return result

    def invalidate(self, *, generation: int) -> None:
        """Make every queued or in-flight result from prior generations stale."""

        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ValueError("generation must be a nonnegative integer")
        with self._lock:
            if self._active_generation is None or generation > self._active_generation:
                self._active_generation = generation
                self._next_sequence = 0
                self._ready.clear()

    def snapshot(self) -> dict[str, int | float | None]:
        with self._lock:
            return {
                "submitted": self._submitted,
                "completed": self._completed_count,
                "queued": self._jobs.qsize(),
                "overload_count": self._overload_count,
                "stale_count": self._stale_count,
                "failure_count": self._failure_count,
                "probe_inference_count": self._probe_inference_count,
                "confirmation_inference_count": self._confirmation_inference_count,
                "last_inference_ms": self._last_inference_ms,
            }

    def close(self) -> None:
        """Stop workers before their ONNX sessions are released by the caller."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
        for _ in self._threads:
            while True:
                try:
                    self._jobs.put(None, timeout=0.1)
                    break
                except queue.Full:
                    continue
        for thread in self._threads:
            thread.join(timeout=3.0)

    def _worker(self) -> None:
        try:
            runner = self._runner_factory()
        except Exception as error:
            runner = None
            startup_error = _error_text(error)
        else:
            startup_error = None
        while True:
            try:
                job = self._confirmation_jobs.get_nowait()
            except queue.Empty:
                try:
                    job = self._jobs.get(timeout=0.010)
                except queue.Empty:
                    continue
                if job is None:
                    return
            started_ns = time.perf_counter_ns()
            score: float | None = None
            primary_score: float | None = None
            phase_scores: tuple[tuple[int, float], ...] = ()
            error_text = startup_error
            if runner is not None:
                try:
                    if job.decision_kind == "phase_refinement":
                        assert job.phase_refinement_primary_score is not None
                        primary_score = job.phase_refinement_primary_score
                        scored = [(0, primary_score)]
                        for offset_ms, probe_waveform in job.phase_probe_waveforms:
                            probe_score = float(runner.score(probe_waveform))
                            if not math.isfinite(probe_score):
                                raise RuntimeError("KWS runner returned a non-finite score")
                            scored.append((offset_ms, probe_score))
                        best_seed_score = max(score for _, score in scored[1:])
                        if best_seed_score >= job.phase_probe_threshold:
                            scored_neighbor_offsets: set[int] = set()
                            for selected_offset_ms, selected_score in tuple(scored[1:]):
                                if selected_score != best_seed_score:
                                    continue
                                for anchor_offset_ms, neighbors in job.phase_refinement_neighbors:
                                    if anchor_offset_ms != selected_offset_ms:
                                        continue
                                    if anchor_offset_ms in scored_neighbor_offsets:
                                        break
                                    for offset_ms, probe_waveform in neighbors:
                                        probe_score = float(runner.score(probe_waveform))
                                        if not math.isfinite(probe_score):
                                            raise RuntimeError("KWS runner returned a non-finite score")
                                        scored.append((offset_ms, probe_score))
                                    scored_neighbor_offsets.add(anchor_offset_ms)
                                    break
                    else:
                        primary_score = float(runner.score(job.waveform))
                        if not math.isfinite(primary_score):
                            raise RuntimeError("KWS runner returned a non-finite score")
                        scored = [(0, primary_score)]
                        if (
                            job.phase_probe_lower is not None
                            and job.phase_probe_threshold is not None
                            and job.phase_probe_lower <= primary_score < job.phase_probe_threshold
                        ):
                            for offset_ms, probe_waveform in job.phase_probe_waveforms:
                                probe_score = float(runner.score(probe_waveform))
                                if not math.isfinite(probe_score):
                                    raise RuntimeError("KWS runner returned a non-finite score")
                                scored.append((offset_ms, probe_score))
                    phase_scores = tuple(scored)
                    score = max(value for _, value in phase_scores)
                    with self._lock:
                        self._probe_inference_count += max(0, len(phase_scores) - 1)
                    if job.decision_kind in {"phase_confirmation", "phase_refinement"}:
                        with self._lock:
                            self._confirmation_inference_count += 1
                except Exception as error:
                    score = None
                    error_text = _error_text(error)
            inference_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
            result = KwsResult(
                sequence=job.sequence,
                generation=job.generation,
                captured_ns=job.captured_ns,
                score=score,
                inference_ms=inference_ms,
                error=error_text,
                window_end_ns=job.window_end_ns,
                primary_score=primary_score,
                phase_scores=phase_scores,
                decision_kind=job.decision_kind,
                phase_confirmation_of_window_end_ns=job.phase_confirmation_of_window_end_ns,
                phase_confirmation_phase_offset_ms=job.phase_confirmation_phase_offset_ms,
                phase_refinement_of_window_end_ns=job.phase_refinement_of_window_end_ns,
            )
            with self._completion_condition:
                self._completed_count += 1
                self._completed_by_generation[job.generation] = (
                    self._completed_by_generation.get(job.generation, 0) + 1
                )
                self._last_inference_ms = inference_ms
                if error_text is not None:
                    self._failure_count += 1
                self._completion_condition.notify_all()
            if job.decision_kind in {"phase_confirmation", "phase_refinement"}:
                self._confirmation_completed.put(result)
            else:
                self._completed.put(result)

    def _drain_completed_locked(self, generation: int) -> None:
        while True:
            try:
                result = self._completed.get_nowait()
            except queue.Empty:
                return
            if result.generation != generation:
                self._stale_count += 1
                continue
            self._ready[result.sequence] = result


def _validate_job(job: KwsJob) -> KwsJob:
    if not isinstance(job, KwsJob):
        raise ValueError("job must be a KwsJob")
    window_end_ns = job.captured_ns if job.window_end_ns is None else job.window_end_ns
    for name, value in (
        ("sequence", job.sequence),
        ("generation", job.generation),
        ("captured_ns", job.captured_ns),
        ("window_end_ns", window_end_ns),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    waveform = np.asarray(job.waveform)
    if waveform.dtype != np.float32 or waveform.shape != (16_000,) or not np.all(np.isfinite(waveform)):
        raise ValueError("KWS waveform must be finite float32 [16000]")
    probes: list[tuple[int, np.ndarray]] = []
    seen_offsets = {0}
    for offset_ms, probe_waveform in job.phase_probe_waveforms:
        if isinstance(offset_ms, bool) or not isinstance(offset_ms, int) or offset_ms == 0:
            raise ValueError("KWS phase probe offset must be a nonzero integer")
        if offset_ms in seen_offsets:
            raise ValueError("KWS phase probe offsets must be unique")
        probe = np.asarray(probe_waveform)
        if probe.dtype != np.float32 or probe.shape != (16_000,) or not np.all(np.isfinite(probe)):
            raise ValueError("KWS phase probe waveform must be finite float32 [16000]")
        seen_offsets.add(offset_ms)
        probes.append((offset_ms, probe.copy()))
    if job.decision_kind not in {"logical_slot", "phase_confirmation", "phase_refinement"}:
        raise ValueError("KWS decision_kind must be logical_slot, phase_confirmation, or phase_refinement")
    if job.decision_kind == "phase_confirmation":
        if (
            probes
            or job.phase_probe_lower is not None
            or job.phase_probe_threshold is not None
        ):
            raise ValueError("KWS phase confirmation job must not include phase probes")
        if (
            job.phase_confirmation_of_window_end_ns is None
            or job.phase_confirmation_phase_offset_ms is None
        ):
            raise ValueError("KWS phase confirmation provenance is required")
        if (
            isinstance(job.phase_confirmation_of_window_end_ns, bool)
            or not isinstance(job.phase_confirmation_of_window_end_ns, int)
            or job.phase_confirmation_of_window_end_ns >= window_end_ns
        ):
            raise ValueError("KWS phase confirmation source window must precede confirmation")
        if (
            isinstance(job.phase_confirmation_phase_offset_ms, bool)
            or not isinstance(job.phase_confirmation_phase_offset_ms, int)
            or job.phase_confirmation_phase_offset_ms == 0
        ):
            raise ValueError("KWS phase confirmation requires a nonzero phase offset")
    elif job.decision_kind == "phase_refinement":
        if (
            job.phase_refinement_primary_score is None
            or job.phase_refinement_of_window_end_ns is None
            or not probes
            or job.phase_probe_lower is not None
            or job.phase_probe_threshold is None
        ):
            raise ValueError("KWS phase refinement requires source score, probes, threshold, and provenance")
        if (
            not math.isfinite(float(job.phase_refinement_primary_score))
            or not 0.0 <= float(job.phase_refinement_primary_score) <= 1.0
        ):
            raise ValueError("KWS phase refinement source score must be a probability")
        if (
            isinstance(job.phase_refinement_of_window_end_ns, bool)
            or not isinstance(job.phase_refinement_of_window_end_ns, int)
            or job.phase_refinement_of_window_end_ns >= window_end_ns
        ):
            raise ValueError("KWS phase refinement source window must precede refinement")
        probe_offsets = {offset for offset, _ in probes}
        neighbors_by_anchor: dict[int, tuple[tuple[int, np.ndarray], ...]] = {}
        for anchor_offset, neighbor_values in job.phase_refinement_neighbors:
            if anchor_offset not in probe_offsets or anchor_offset == 0:
                raise ValueError("KWS phase refinement neighbors require a seed probe anchor")
            if anchor_offset in neighbors_by_anchor or not neighbor_values:
                raise ValueError("KWS phase refinement neighbor anchors must be unique and nonempty")
            checked_neighbors: list[tuple[int, np.ndarray]] = []
            for neighbor_offset, neighbor_waveform in neighbor_values:
                if (
                    isinstance(neighbor_offset, bool)
                    or not isinstance(neighbor_offset, int)
                    or neighbor_offset == anchor_offset
                    or abs(neighbor_offset - anchor_offset) > 8
                ):
                    raise ValueError("KWS phase refinement neighbor must be adjacent to its seed")
                neighbor = np.asarray(neighbor_waveform)
                if (
                    neighbor.dtype != np.float32
                    or neighbor.shape != (16_000,)
                    or not np.all(np.isfinite(neighbor))
                ):
                    raise ValueError("KWS phase refinement neighbor waveform must be finite float32 [16000]")
                checked_neighbors.append((neighbor_offset, neighbor.copy()))
            neighbors_by_anchor[anchor_offset] = tuple(checked_neighbors)
        if set(neighbors_by_anchor) != probe_offsets:
            raise ValueError("KWS phase refinement must provide neighbors for every seed probe")
    elif (
        job.phase_confirmation_of_window_end_ns is not None
        or job.phase_confirmation_phase_offset_ms is not None
    ):
        raise ValueError("only a KWS phase confirmation may carry confirmation provenance")
    for name, value in (("phase_probe_lower", job.phase_probe_lower), ("phase_probe_threshold", job.phase_probe_threshold)):
        if value is not None and (not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0):
            raise ValueError(f"{name} must be a probability")
    if (
        job.decision_kind != "phase_refinement"
        and (job.phase_probe_lower is None) != (job.phase_probe_threshold is None)
    ):
        raise ValueError("phase probe bounds must be provided together")
    return KwsJob(
        sequence=job.sequence,
        generation=job.generation,
        captured_ns=job.captured_ns,
        waveform=waveform.copy(),
        window_end_ns=window_end_ns,
        phase_probe_waveforms=tuple(probes),
        phase_probe_lower=job.phase_probe_lower,
        phase_probe_threshold=job.phase_probe_threshold,
        slot_sequence=job.slot_sequence,
        phase_offset_ms=job.phase_offset_ms,
        decision_kind=job.decision_kind,
        phase_confirmation_of_window_end_ns=job.phase_confirmation_of_window_end_ns,
        phase_confirmation_phase_offset_ms=job.phase_confirmation_phase_offset_ms,
        phase_refinement_primary_score=job.phase_refinement_primary_score,
        phase_refinement_neighbors=tuple(
            (anchor, tuple((offset, waveform.copy()) for offset, waveform in neighbors))
            for anchor, neighbors in (
                job.phase_refinement_neighbors
                if job.decision_kind == "phase_refinement"
                else ()
            )
        ),
        phase_refinement_of_window_end_ns=job.phase_refinement_of_window_end_ns,
    )


def _error_text(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"
