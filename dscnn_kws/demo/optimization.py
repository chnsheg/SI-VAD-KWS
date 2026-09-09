"""Offline replay primitives for VAD-then-KWS microphone tuning."""

from __future__ import annotations

import wave
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping, Protocol, Sequence

import numpy as np
import onnxruntime as ort

from dscnn_kws.ONNX.export_vad_stateful import export_stateful_vad

from .cascade import CascadeEngine
from .contracts import CascadeConfig, load_vad_contract, validate_kws_model
from .features import StreamingVadFeatureCache
from .input_frontend import FrontendProfile
from .input_frontend import InputFrontend
from .runners import KwsOnnxRunner, StatefulVadOnnxRunner


_SAMPLE_RATE = 16_000
_SEGMENT_FRAME_SAMPLES = 512
_SEGMENT_MAX_GAP_FRAMES = 14
_SEGMENT_MIN_FRAMES = 3


@dataclass(frozen=True)
class SourceSpec:
    """A continuous microphone WAV that satisfies the deployed PCM contract."""

    name: str
    path: Path

    def load_pcm(self) -> np.ndarray:
        if not self.path.is_file():
            raise ValueError(f"source WAV does not exist: {self.path}")
        try:
            with wave.open(str(self.path), "rb") as reader:
                if reader.getnchannels() != 1 or reader.getframerate() != _SAMPLE_RATE:
                    raise ValueError("source WAV must be mono 16 kHz")
                if reader.getsampwidth() != 2 or reader.getcomptype() != "NONE":
                    raise ValueError("source WAV must be uncompressed signed PCM16")
                pcm16 = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
        except (wave.Error, OSError) as error:
            raise ValueError(f"source WAV could not be read: {self.path}") from error
        if pcm16.size == 0:
            raise ValueError("source WAV must not be empty")
        return (pcm16.astype(np.float32) / 32768.0).copy()


@dataclass(frozen=True)
class FrontendTuning:
    profile: FrontendProfile
    target_rms_dbfs: float


@dataclass(frozen=True)
class CascadeTuning:
    energy_threshold_dbfs: float
    energy_hangover_ms: int
    vad_energy_tail_ms: int
    vad_period_ms: int
    kws_period_ms: int
    vad_threshold: float
    vad_confirmations: int
    kws_threshold: float

    def cascade_config(self, *, kws_positive_index: int, queue_capacity: int) -> CascadeConfig:
        config = CascadeConfig(
            energy_period_ms=self.vad_period_ms,
            energy_threshold_dbfs=self.energy_threshold_dbfs,
            energy_hangover_ms=self.energy_hangover_ms,
            vad_energy_tail_ms=self.vad_energy_tail_ms,
            vad_period_ms=self.vad_period_ms,
            vad_threshold=self.vad_threshold,
            vad_confirmations=self.vad_confirmations,
            kws_period_ms=self.kws_period_ms,
            kws_positive_index=kws_positive_index,
            kws_threshold=self.kws_threshold,
            kws_confirmations=2,
            queue_capacity=queue_capacity,
        )
        config.validate()
        return config


@dataclass(frozen=True)
class Candidate:
    name: str
    frontend: FrontendTuning
    cascade: CascadeTuning


@dataclass(frozen=True)
class ModelPaths:
    vad_model: Path
    kws_model: Path
    stateful_vad_model: Path | None = None


@dataclass(frozen=True)
class ReplayMetrics:
    """Utterance-attributed outcomes emitted by one exact cascade replay."""

    utterance_count: int
    unique_wakes: int
    background_wakes: int
    background_high_kws: int = 0
    vad_activated: int = 0
    kws_entered: int = 0
    kws_calls: int = 0
    p95_wake_latency_ms: float = float("nan")


@dataclass(frozen=True)
class ReplayReport:
    source: SourceSpec
    candidate: Candidate
    utterances: tuple[tuple[float, float], ...]
    metrics: ReplayMetrics
    events: tuple[dict[str, object], ...]


class CandidateEvaluator(Protocol):
    def candidates(self, *, seed: int, budget: int) -> Sequence[Candidate]: ...

    def evaluate(self, candidate: Candidate) -> Mapping[str, ReplayMetrics]: ...


class RecordingEvaluator:
    """Evaluate every candidate independently against the two formal sources."""

    def __init__(self, sources: Sequence[SourceSpec], models: ModelPaths) -> None:
        if len(sources) != 2:
            raise ValueError("exactly two continuous recording sources are required")
        names = [source.name for source in sources]
        if len(set(names)) != len(names):
            raise ValueError("recording source names must be unique")
        self._sources = tuple(sources)
        self._models = models

    def candidates(self, *, seed: int, budget: int) -> Sequence[Candidate]:
        return generate_candidates(seed=seed, budget=budget)

    def evaluate(self, candidate: Candidate) -> Mapping[str, ReplayMetrics]:
        return {
            source.name: replay_source(source, candidate, self._models).metrics
            for source in self._sources
        }


@dataclass(frozen=True)
class CandidateRejection:
    candidate: Candidate
    reports: Mapping[str, ReplayMetrics]
    reason: str


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: Candidate
    reports: Mapping[str, ReplayMetrics]


@dataclass(frozen=True)
class SearchResult:
    selected: Candidate | None
    accepted: tuple[CandidateEvaluation, ...]
    rejected: tuple[CandidateRejection, ...]


def search_candidates(
    evaluator: CandidateEvaluator,
    baseline: Mapping[str, ReplayMetrics],
    *,
    seed: int,
    budget: int,
) -> SearchResult:
    """Select only candidates that improve or retain recall on every source."""

    if budget < 1:
        raise ValueError("budget must be positive")
    if not baseline:
        raise ValueError("baseline metrics must not be empty")
    accepted: list[CandidateEvaluation] = []
    rejected: list[CandidateRejection] = []
    for candidate in evaluator.candidates(seed=seed, budget=budget):
        reports = evaluator.evaluate(candidate)
        reasons = _candidate_regressions(reports, baseline)
        if reasons:
            rejected.append(CandidateRejection(candidate, reports, "; ".join(reasons)))
        else:
            accepted.append(CandidateEvaluation(candidate, reports))
    accepted.sort(key=_candidate_rank, reverse=True)
    return SearchResult(
        selected=accepted[0].candidate if accepted else None,
        accepted=tuple(accepted),
        rejected=tuple(rejected),
    )


def generate_candidates(*, seed: int, budget: int) -> tuple[Candidate, ...]:
    """Create a seeded, stratified broad search over causal input and gates."""

    if budget < len(FrontendProfile):
        raise ValueError(f"budget must be at least {len(FrontendProfile)} to cover all input profiles")
    rng = np.random.default_rng(seed)
    profiles = list(FrontendProfile)
    rng.shuffle(profiles)
    profile_choices = tuple(profiles[index % len(profiles)] for index in range(budget))
    targets = _latin_hypercube(rng, budget, -26.0, -12.0)
    energy_thresholds = _latin_hypercube(rng, budget, -42.0, -30.0)
    vad_thresholds = _latin_hypercube(rng, budget, 0.62, 0.82)
    kws_thresholds = _latin_hypercube(rng, budget, 0.72, 0.84)
    vad_periods = _sample_choices(rng, budget, (16, 20, 24, 32, 40, 48, 64))
    kws_periods = _sample_choices(rng, budget, (64, 80, 96, 112, 128, 160, 192))
    vad_confirmations = _sample_choices(rng, budget, (2, 3, 4))
    candidates: list[Candidate] = []
    for index in range(budget):
        candidates.append(
            Candidate(
                name=f"broad-{index:02d}-{profile_choices[index].value}",
                frontend=FrontendTuning(
                    profile_choices[index], round(float(targets[index]), 3)
                ),
                cascade=CascadeTuning(
                    energy_threshold_dbfs=round(float(energy_thresholds[index]), 3),
                    energy_hangover_ms=1_000,
                    vad_energy_tail_ms=1_000,
                    vad_period_ms=int(vad_periods[index]),
                    kws_period_ms=int(kws_periods[index]),
                    vad_threshold=round(float(vad_thresholds[index]), 4),
                    vad_confirmations=int(vad_confirmations[index]),
                    kws_threshold=round(float(kws_thresholds[index]), 4),
                ),
            )
        )
    return tuple(candidates)


def _latin_hypercube(
    rng: np.random.Generator, count: int, low: float, high: float
) -> np.ndarray:
    bins = rng.permutation(count)
    return low + (bins + rng.random(count)) * (high - low) / count


def _sample_choices(
    rng: np.random.Generator, count: int, choices: Sequence[int]
) -> np.ndarray:
    repeated = np.resize(np.asarray(choices, dtype=np.int64), count)
    return rng.permutation(repeated)


def _candidate_regressions(
    reports: Mapping[str, ReplayMetrics], baseline: Mapping[str, ReplayMetrics]
) -> list[str]:
    missing = sorted(set(baseline).difference(reports))
    if missing:
        return [f"missing source metrics: {', '.join(missing)}"]
    reasons: list[str] = []
    for source, baseline_metrics in baseline.items():
        candidate_metrics = reports[source]
        if _recall(candidate_metrics) < _recall(baseline_metrics):
            reasons.append(f"recall dropped on {source}")
        if candidate_metrics.background_wakes > baseline_metrics.background_wakes:
            reasons.append(f"background wakes increased on {source}")
        if candidate_metrics.background_high_kws > baseline_metrics.background_high_kws:
            reasons.append(f"background high KWS increased on {source}")
    return reasons


def _candidate_rank(evaluation: CandidateEvaluation) -> tuple[float, float, int, int, float, int, str]:
    metrics = tuple(evaluation.reports.values())
    recalls = tuple(_recall(metric) for metric in metrics)
    latency = sum(
        metric.p95_wake_latency_ms
        if isfinite(metric.p95_wake_latency_ms)
        else 1_000_000_000.0
        for metric in metrics
    )
    return (
        min(recalls),
        sum(recalls),
        sum(metric.vad_activated for metric in metrics),
        sum(metric.kws_entered for metric in metrics),
        -latency,
        -sum(metric.kws_calls for metric in metrics),
        evaluation.candidate.name,
    )


def _recall(metrics: ReplayMetrics) -> float:
    return metrics.unique_wakes / metrics.utterance_count if metrics.utterance_count else 0.0


def replay_source(source: SourceSpec, candidate: Candidate, models: ModelPaths) -> ReplayReport:
    """Run one candidate through the deployed stateful VAD-then-KWS cascade."""

    raw_pcm = source.load_pcm()
    if models.stateful_vad_model is not None:
        stateful_path = models.stateful_vad_model
        if not stateful_path.is_file():
            raise ValueError(f"stateful VAD model does not exist: {stateful_path}")
        return _replay_loaded_source(raw_pcm, source, candidate, models, stateful_path)
    with TemporaryDirectory(prefix="vad-kws-stateful-") as directory:
        stateful_path = Path(directory) / "vad_stateful.onnx"
        export_stateful_vad(models.vad_model, stateful_path)
        return _replay_loaded_source(raw_pcm, source, candidate, models, stateful_path)


def _replay_loaded_source(
    raw_pcm: np.ndarray,
    source: SourceSpec,
    candidate: Candidate,
    models: ModelPaths,
    stateful_vad_path: Path,
) -> ReplayReport:
    vad_contract = load_vad_contract(models.vad_model)
    kws_contract = validate_kws_model(models.kws_model)
    config = candidate.cascade.cascade_config(kws_positive_index=0, queue_capacity=32)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    vad_session = ort.InferenceSession(
        str(stateful_vad_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    kws_session = ort.InferenceSession(
        str(models.kws_model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    engine = CascadeEngine(
        config=config,
        vad_contract=vad_contract,
        vad_runner=StatefulVadOnnxRunner(vad_session, n_mels=vad_contract.n_mels),
        kws_runner=KwsOnnxRunner(kws_session, kws_contract, positive_index=config.kws_positive_index),
        vad_feature_cache=StreamingVadFeatureCache(vad_contract),
    )
    frontend = InputFrontend(candidate.frontend.profile, target_rms_dbfs=candidate.frontend.target_rms_dbfs)
    records: list[dict[str, object]] = []
    for start in range(0, raw_pcm.size, 1_600):
        end = min(start + 1_600, raw_pcm.size)
        inference_pcm, _ = frontend.process(raw_pcm[start:end])
        events = engine.push_pcm(inference_pcm, end * 1_000_000_000 // _SAMPLE_RATE)
        records.extend(
            {"kind": event.kind, "time": event.captured_ns / 1_000_000_000, **event.fields}
            for event in events
        )
    utterances = tuple(segment_utterances(raw_pcm))
    metrics = score_events(
        records,
        utterances,
        duration_s=raw_pcm.size / _SAMPLE_RATE,
        kws_threshold=config.kws_threshold,
    )
    return ReplayReport(source, candidate, utterances, metrics, tuple(records))


def segment_utterances(
    pcm: np.ndarray,
    threshold_dbfs: float = -43.0,
) -> list[tuple[float, float]]:
    """Find stable speech regions in the raw recording for metric attribution.

    This detector is deliberately independent of the candidate cascade.  It is
    only an annotation source: 32 ms RMS frames are joined across gaps up to
    448 ms, which keeps syllables of one wake phrase together.
    """

    values = np.asarray(pcm)
    if values.dtype != np.float32 or values.ndim != 1 or values.size == 0:
        raise ValueError("PCM must be nonempty float32 mono")
    padded_count = int(np.ceil(values.size / _SEGMENT_FRAME_SAMPLES))
    padded = np.pad(
        values,
        (0, padded_count * _SEGMENT_FRAME_SAMPLES - values.size),
    )
    frames = padded.reshape(padded_count, _SEGMENT_FRAME_SAMPLES)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    levels = 20.0 * np.log10(np.maximum(rms, 1e-12))
    active = levels > float(threshold_dbfs)

    ranges: list[tuple[int, int]] = []
    start: int | None = None
    last_active: int | None = None
    for index, is_active in enumerate(active):
        if not bool(is_active):
            continue
        if start is None:
            start = index
        elif last_active is not None and index - last_active - 1 > _SEGMENT_MAX_GAP_FRAMES:
            ranges.append((start * _SEGMENT_FRAME_SAMPLES, (last_active + 1) * _SEGMENT_FRAME_SAMPLES))
            start = index
        last_active = index
    if start is not None and last_active is not None:
        ranges.append((start * _SEGMENT_FRAME_SAMPLES, (last_active + 1) * _SEGMENT_FRAME_SAMPLES))

    duration_limit = _SEGMENT_MIN_FRAMES * _SEGMENT_FRAME_SAMPLES
    return [
        (start / _SAMPLE_RATE, min(end, values.size) / _SAMPLE_RATE)
        for start, end in ranges
        if end - start >= duration_limit
    ]


def score_events(
    events: Sequence[Mapping[str, object]],
    utterances: Sequence[tuple[float, float]],
    *,
    duration_s: float,
    kws_threshold: float = 0.75,
) -> ReplayMetrics:
    """Attribute each wake to one utterance, counting repetitions only once."""

    if duration_s < 0.0:
        raise ValueError("duration_s must not be negative")
    if not 0.0 <= kws_threshold <= 1.0:
        raise ValueError("kws_threshold must be within [0, 1]")
    scored_utterances = [
        (start_s, end_s)
        for start_s, end_s in utterances
        if 1.0 <= start_s <= end_s <= duration_s
    ]

    def matched_utterance(time_s: float) -> int | None:
        return next(
            (
                index
                for index, (start_s, end_s) in enumerate(scored_utterances)
                if start_s <= time_s <= end_s + 0.8
            ),
            None,
        )

    assigned: set[int] = set()
    vad_activated: set[int] = set()
    kws_entered: set[int] = set()
    wake_latencies_ms: list[float] = []
    background_wakes = 0
    background_high_kws = 0
    kws_calls = 0
    for event in events:
        time_s = event.get("time")
        if not isinstance(time_s, (int, float)) or not isfinite(float(time_s)):
            raise ValueError("event time must be finite and numeric")
        time_s = float(time_s)
        matched = matched_utterance(time_s)
        kind = event.get("kind")
        if kind == "vad_confirmed_state" and event.get("to_state") == "speech":
            if matched is not None:
                vad_activated.add(matched)
        elif kind == "kws_call":
            kws_calls += 1
            if matched is not None:
                kws_entered.add(matched)
            else:
                score = event.get("score")
                if isinstance(score, (int, float)) and isfinite(float(score)) and score > kws_threshold:
                    background_high_kws += 1
        elif kind == "wake":
            if matched is None:
                background_wakes += 1
            elif matched not in assigned:
                assigned.add(matched)
                wake_latencies_ms.append((time_s - scored_utterances[matched][1]) * 1000.0)
    return ReplayMetrics(
        utterance_count=len(scored_utterances),
        unique_wakes=len(assigned),
        background_wakes=background_wakes,
        background_high_kws=background_high_kws,
        vad_activated=len(vad_activated),
        kws_entered=len(kws_entered),
        kws_calls=kws_calls,
        p95_wake_latency_ms=(
            float(np.percentile(np.asarray(wake_latencies_ms), 95))
            if wake_latencies_ms
            else float("nan")
        ),
    )
