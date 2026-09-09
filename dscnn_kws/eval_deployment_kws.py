"""Explicit-manifest KWS threshold calibration and streaming deployment evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as audio_functional

from dscnn_kws.configs import CLASS_ENCODING
from dscnn_kws.data.dataset import _normalize_manifest_audio_path
from dscnn_kws.data.hard_negative_mining import load_scoring_config, scoring_namespace
from dscnn_kws.eval_fah_frr import (
    build_eval_loader,
    build_model,
    choose_threshold_for_target_fah,
    collect_scores,
    eval_at_threshold,
)
from dscnn_kws.engine.deployment_gate import DeploymentEvidence, DevelopmentDeploymentGate


_PROTECTED_CALIBRATION_SPLITS = {"test", "held_out_test"}
_ALLOWED_HELD_OUT_EVIDENCE_SPLITS = {"validation", "held_out_validation", "held_out_dev"}
_REQUIRED_DEPLOYMENT_MANIFESTS = (
    "calibration",
    "captured_positive",
    "false_wake",
    "long_negative",
)


@dataclass(frozen=True)
class CalibrationResult:
    threshold: float
    metrics: dict[str, float | int]


@dataclass(frozen=True)
class StreamingMetrics:
    false_alarms: int
    evaluated_hours: float
    fah: float
    scored_windows: int


@dataclass(frozen=True)
class StreamingRecordingMetrics:
    """Streaming negative evidence summarized without retaining window scores."""

    audio_path: str
    duration_seconds: float
    false_alarms: int
    positive_windows: int
    scored_windows: int


def iter_jsonl(path: Path | str) -> Iterable[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Manifest row must be an object at {path}:{line_number}")
            yield row


def _manifest_rows(path: Path | str) -> list[dict[str, object]]:
    manifest = Path(path)
    if not manifest.is_file():
        raise ValueError(f"Explicit manifest does not exist: {manifest}")
    rows = list(iter_jsonl(manifest))
    if not rows:
        raise ValueError(f"Explicit manifest is empty: {manifest}")
    return rows


def _source_split(row: Mapping[str, object]) -> str:
    return str(row.get("source_split", row.get("split", ""))).strip().casefold()


def _require_held_out_evidence_split(manifest_path: Path | str, domain: str) -> None:
    invalid_splits = {
        _source_split(row)
        for row in _manifest_rows(manifest_path)
        if _source_split(row) not in _ALLOWED_HELD_OUT_EVIDENCE_SPLITS
    }
    if invalid_splits:
        observed = ", ".join(sorted(split or "missing" for split in invalid_splits))
        raise ValueError(
            f"{domain} deployment evidence must use explicit held-out development source_split "
            f"({', '.join(sorted(_ALLOWED_HELD_OUT_EVIDENCE_SPLITS))}); got {observed}"
        )


def require_non_test_calibration(manifest_path: Path | str) -> None:
    """Reject threshold calibration whose provenance contains a locked test split."""
    protected_splits = {_source_split(row) for row in _manifest_rows(manifest_path)} & _PROTECTED_CALIBRATION_SPLITS
    if protected_splits:
        raise ValueError("Test manifests cannot calibrate a deployment threshold")
    _require_held_out_evidence_split(manifest_path, "calibration")


def require_explicit_deployment_manifests(manifests: Mapping[str, str | Path | None]) -> dict[str, Path]:
    """Require all evidence domains to declare held-out development provenance.

    ``validation`` is the supported development-held-out split.  Named held-out
    development/validation splits are accepted as equivalent provenance.  Train
    and locked-test provenance are never deployment-best evidence.
    """
    resolved: dict[str, Path] = {}
    for name in _REQUIRED_DEPLOYMENT_MANIFESTS:
        raw_path = manifests.get(name)
        if raw_path is None or not str(raw_path).strip():
            raise ValueError(f"Deployment-best evidence requires explicit {name} manifest")
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError(f"Deployment-best evidence manifest does not exist: {name}={path}")
        if not _manifest_rows(path):
            raise ValueError(f"Deployment-best evidence manifest is empty: {name}={path}")
        _require_held_out_evidence_split(path, name)
        resolved[name] = path
    return resolved


def _validate_score_vector(scores: Sequence[float], name: str) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError(f"{name} scores must be a non-empty one-dimensional sequence")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} scores must be finite")
    return values


def _expected_labels(rows: Sequence[Mapping[str, object]], positive_index: int, negative_index: int) -> np.ndarray:
    expected: list[int] = []
    for row in rows:
        command = row.get("command", row.get("label"))
        if command == "positive":
            expected.append(positive_index)
        elif command == "negative":
            expected.append(negative_index)
        else:
            raise ValueError("Explicit deployment manifests require positive or negative command labels")
    return np.asarray(expected, dtype=np.int64)


def _validate_manifest_domain(
    manifest_path: Path | str,
    scores: Sequence[float],
    *,
    positive_index: int,
    negative_index: int,
    expected_command: str | None = None,
    observed_labels: Sequence[int] | None = None,
    scores_are_per_manifest_row: bool = True,
) -> tuple[list[dict[str, object]], np.ndarray]:
    rows = _manifest_rows(manifest_path)
    score_values = _validate_score_vector(scores, str(manifest_path))
    if scores_are_per_manifest_row and len(rows) != len(score_values):
        raise ValueError(f"Score count does not match manifest rows: {manifest_path}")
    labels = _expected_labels(rows, positive_index, negative_index)
    if expected_command is not None and any(row.get("command", row.get("label")) != expected_command for row in rows):
        raise ValueError(f"{manifest_path} must contain only {expected_command} rows")
    if observed_labels is not None:
        observed = np.asarray(observed_labels, dtype=np.int64)
        if observed.ndim != 1 or len(observed) != len(labels) or not np.array_equal(observed, labels):
            raise ValueError(f"Model labels do not match explicit manifest labels: {manifest_path}")
    return rows, labels


def select_calibration_threshold(
    *,
    scores: Sequence[float],
    labels: Sequence[int],
    manifest_path: Path | str,
    target_fah: float,
    window_seconds: float,
    positive_index: int,
    negative_index: int,
    min_positive_recall: float,
) -> CalibrationResult:
    """Choose the lowest FAH-compliant threshold and enforce the recall gate."""
    if not math.isfinite(target_fah) or target_fah < 0:
        raise ValueError("target_fah must be finite and non-negative")
    if not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be finite and positive")
    if not math.isfinite(min_positive_recall) or not 0 <= min_positive_recall <= 1:
        raise ValueError("min_positive_recall must be in [0, 1]")
    require_non_test_calibration(manifest_path)
    _, expected = _validate_manifest_domain(
        manifest_path,
        scores,
        positive_index=positive_index,
        negative_index=negative_index,
        observed_labels=labels,
    )
    score_values = _validate_score_vector(scores, "calibration")
    positive_scores = score_values[expected == positive_index]
    negative_scores = score_values[expected == negative_index]
    if not len(positive_scores) or not len(negative_scores):
        raise ValueError("Calibration manifest must contain both positive and negative samples")
    threshold = choose_threshold_for_target_fah(positive_scores, negative_scores, target_fah, window_seconds)
    if not math.isfinite(threshold):
        # The legacy helper uses -inf when a target permits every negative
        # alarm.  Keep that operating point while retaining strict JSON and
        # streaming metric compatibility.
        lowest_negative = float(np.min(negative_scores))
        threshold = float(np.nextafter(lowest_negative, -np.inf))
        if not math.isfinite(threshold):
            threshold = lowest_negative
    metrics = eval_at_threshold(positive_scores, negative_scores, threshold, window_seconds)
    positive_recall = 1.0 - float(metrics["frr"])
    if positive_recall + 1e-12 < min_positive_recall:
        raise ValueError(
            f"No threshold satisfies target_fah={target_fah} and min_positive_recall={min_positive_recall}"
        )
    return CalibrationResult(threshold=float(threshold), metrics=_json_scalars(metrics))


def streaming_false_alarm_metrics(
    scores: Sequence[float],
    threshold: float,
    hop_seconds: float,
    debounce_seconds: float,
) -> StreamingMetrics:
    """Count positive triggers, suppressing follow-on triggers during debounce."""
    values = _validate_score_vector(scores, "streaming")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if not math.isfinite(hop_seconds) or hop_seconds <= 0:
        raise ValueError("hop_seconds must be finite and positive")
    if not math.isfinite(debounce_seconds) or debounce_seconds < 0:
        raise ValueError("debounce_seconds must be finite and non-negative")

    next_alarm_time = -math.inf
    false_alarms = 0
    for index, score in enumerate(values):
        timestamp = index * hop_seconds
        if score >= threshold and timestamp >= next_alarm_time:
            false_alarms += 1
            next_alarm_time = timestamp + debounce_seconds
    evaluated_hours = len(values) * hop_seconds / 3600.0
    return StreamingMetrics(
        false_alarms=false_alarms,
        evaluated_hours=evaluated_hours,
        fah=false_alarms / max(evaluated_hours, 1e-12),
        scored_windows=len(values),
    )


def iter_sliding_audio_windows(
    audio_path: Path | str,
    *,
    sample_rate: int,
    hop_seconds: float,
    window_seconds: float = 1.0,
) -> Iterable[tuple[float, torch.Tensor]]:
    """Read only full one-second production windows.

    Recordings shorter than a window yield no score and no alarm. Their complete
    wall-clock duration still contributes to the streaming FAH denominator.
    Incomplete tail audio is never zero-padded into an evaluable window.
    """
    if sample_rate < 1:
        raise ValueError("sample_rate must be positive")
    if not math.isfinite(hop_seconds) or hop_seconds <= 0:
        raise ValueError("hop_seconds must be finite and positive")
    if not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be finite and positive")
    with sf.SoundFile(str(audio_path), "r") as handle:
        source_rate = int(handle.samplerate)
        total_frames = int(handle.frames)
        if source_rate < 1 or total_frames < 1:
            raise ValueError(f"Unreadable or empty streaming audio: {audio_path}")
        source_window_frames = max(1, round(window_seconds * source_rate))
        target_window_frames = max(1, round(window_seconds * sample_rate))
        window_index = 0
        while True:
            start_frame = round(window_index * hop_seconds * source_rate)
            if start_frame + source_window_frames > total_frames:
                return
            handle.seek(start_frame)
            samples = handle.read(source_window_frames, dtype="float32", always_2d=True)
            # Production input contract: resample each channel independently,
            # then select channel 0.  Mixing stereo before resampling can cause
            # phase cancellation and makes FAH depend on evaluator-only audio
            # preprocessing.
            waveform = torch.from_numpy(np.asarray(samples).T.copy()).to(torch.float32)
            if waveform.ndim != 2 or waveform.shape[0] < 1 or waveform.shape[1] < 1:
                raise ValueError(f"Invalid streaming audio shape: {tuple(waveform.shape)}")
            if source_rate != sample_rate:
                waveform = audio_functional.resample(waveform, source_rate, sample_rate)
            waveform = waveform.narrow(0, 0, 1)
            waveform = waveform.to(torch.float32).clamp(-1.0, 1.0)
            if waveform.shape[1] > target_window_frames:
                waveform = waveform.narrow(1, 0, target_window_frames)
            if waveform.shape[1] != target_window_frames:
                raise ValueError(f"Failed to form a full streaming window: {audio_path}")
            yield (start_frame / source_rate, waveform)
            window_index += 1


def resolve_manifest_audio_path(
    row: Mapping[str, object],
    manifest_path: Path | str,
    *,
    dataset_path: Path | str | None = None,
) -> Path:
    """Resolve explicit-manifest audio with the dataset reader's legacy search order."""
    value = row.get("audio_filepath", row.get("audio_path"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Explicit manifest is missing audio_filepath: {manifest_path}")
    manifest = Path(manifest_path).resolve()
    normalized = _normalize_manifest_audio_path(value)
    if Path(normalized).is_absolute():
        return Path(normalized)
    dataset_root = Path(dataset_path).resolve() if dataset_path is not None else manifest.parent
    package_root = Path(__file__).resolve().parent
    repo_root = package_root.parent
    basename = Path(normalized).name
    candidates = (
        dataset_root.parent / normalized,
        repo_root / normalized,
        dataset_root / basename,
        dataset_root / normalized,
        manifest.parent / normalized,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Failed to resolve audio path from explicit manifest. "
        f"raw={value!r}, normalized={normalized!r}, candidates={[str(path) for path in candidates]}"
    )


def _normalize_streaming_score_sequences(
    scores: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    name: str,
) -> tuple[np.ndarray, ...]:
    """Preserve recording boundaries; a flat vector remains one recording."""
    if isinstance(scores, np.ndarray):
        if scores.ndim == 1:
            return (_validate_score_vector(scores, name),)
        if scores.ndim == 2:
            return tuple(_validate_score_vector(row, f"{name}[{index}]") for index, row in enumerate(scores))
        raise ValueError(f"{name} scores must be one or two dimensional")
    values = list(scores)
    if not values:
        raise ValueError(f"{name} scores must not be empty")
    if all(np.isscalar(value) for value in values):
        return (_validate_score_vector(values, name),)
    return tuple(
        _validate_score_vector(value, f"{name}[{index}]")
        for index, value in enumerate(values)
    )


def _validate_streaming_manifest_domain(
    manifest_path: Path | str,
    scores: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    *,
    positive_index: int,
    negative_index: int,
    expected_command: str,
) -> tuple[list[dict[str, object]], tuple[np.ndarray, ...]]:
    rows = _manifest_rows(manifest_path)
    labels = _expected_labels(rows, positive_index, negative_index)
    if any(row.get("command", row.get("label")) != expected_command for row in rows):
        raise ValueError(f"{manifest_path} must contain only {expected_command} rows")
    sequences = _normalize_streaming_score_sequences(scores, str(manifest_path))
    if len(sequences) != len(rows):
        raise ValueError(f"Streaming recording count does not match manifest rows: {manifest_path}")
    return rows, sequences


def _combined_scores(score_sequences: Sequence[np.ndarray]) -> np.ndarray:
    return np.concatenate(score_sequences)


def aggregate_streaming_recordings(records: Sequence[StreamingRecordingMetrics]) -> StreamingMetrics:
    """Aggregate precomputed recording counts using real audio duration as FAH exposure."""
    if not records:
        raise ValueError("Streaming evidence must include at least one recording")
    for record in records:
        if not math.isfinite(record.duration_seconds) or record.duration_seconds < 0:
            raise ValueError("Streaming recording duration must be finite and non-negative")
        if min(record.false_alarms, record.positive_windows, record.scored_windows) < 0:
            raise ValueError("Streaming recording counts must be non-negative")
        if record.false_alarms > record.positive_windows or record.positive_windows > record.scored_windows:
            raise ValueError("Streaming recording counts are inconsistent")
    false_alarms = sum(record.false_alarms for record in records)
    evaluated_hours = sum(record.duration_seconds for record in records) / 3600.0
    return StreamingMetrics(
        false_alarms=false_alarms,
        evaluated_hours=evaluated_hours,
        fah=false_alarms / max(evaluated_hours, 1e-12),
        scored_windows=sum(record.scored_windows for record in records),
    )


def _recording_metrics_from_scores(
    scores: np.ndarray,
    *,
    threshold: float,
    hop_seconds: float,
    debounce_seconds: float,
    audio_path: str,
) -> StreamingRecordingMetrics:
    """Compatibility adapter for unit callers that supply a single score sequence."""
    metrics = streaming_false_alarm_metrics(scores, threshold, hop_seconds, debounce_seconds)
    return StreamingRecordingMetrics(
        audio_path=audio_path,
        duration_seconds=len(scores) * hop_seconds,
        false_alarms=metrics.false_alarms,
        positive_windows=int(np.sum(scores >= threshold)),
        scored_windows=len(scores),
    )


def _validate_streaming_recording_domain(
    manifest_path: Path | str,
    records: Sequence[StreamingRecordingMetrics],
    *,
    positive_index: int,
    negative_index: int,
    expected_command: str,
) -> tuple[StreamingRecordingMetrics, ...]:
    rows = _manifest_rows(manifest_path)
    _expected_labels(rows, positive_index, negative_index)
    if any(row.get("command", row.get("label")) != expected_command for row in rows):
        raise ValueError(f"{manifest_path} must contain only {expected_command} rows")
    if len(records) != len(rows):
        raise ValueError(f"Streaming recording count does not match manifest rows: {manifest_path}")
    # Validate counts and duration at the input boundary, before report assembly.
    aggregate_streaming_recordings(records)
    return tuple(records)


def _streaming_records_from_input(
    manifest_path: Path | str,
    score_or_records: Sequence[float] | Sequence[Sequence[float]] | Sequence[StreamingRecordingMetrics] | np.ndarray,
    *,
    threshold: float,
    hop_seconds: float,
    debounce_seconds: float,
    positive_index: int,
    negative_index: int,
) -> tuple[StreamingRecordingMetrics, ...]:
    values = list(score_or_records) if not isinstance(score_or_records, np.ndarray) else None
    if values and all(isinstance(value, StreamingRecordingMetrics) for value in values):
        return _validate_streaming_recording_domain(
            manifest_path,
            values,
            positive_index=positive_index,
            negative_index=negative_index,
            expected_command="negative",
        )
    _, sequences = _validate_streaming_manifest_domain(
        manifest_path,
        score_or_records,
        positive_index=positive_index,
        negative_index=negative_index,
        expected_command="negative",
    )
    return tuple(
        _recording_metrics_from_scores(
            scores,
            threshold=threshold,
            hop_seconds=hop_seconds,
            debounce_seconds=debounce_seconds,
            audio_path=f"{manifest_path}:{index}",
        )
        for index, scores in enumerate(sequences)
    )


@torch.no_grad()
def evaluate_streaming_manifest(
    model,
    manifest_path: Path | str,
    *,
    device: torch.device,
    sample_rate: int,
    batch_size: int,
    hop_seconds: float,
    debounce_seconds: float,
    threshold: float,
    positive_index: int,
    dataset_path: Path | str | None = None,
) -> list[StreamingRecordingMetrics]:
    """Score long audio incrementally and retain only per-recording aggregates."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if not math.isfinite(hop_seconds) or hop_seconds <= 0:
        raise ValueError("hop_seconds must be finite and positive")
    if not math.isfinite(debounce_seconds) or debounce_seconds < 0:
        raise ValueError("debounce_seconds must be finite and non-negative")
    manifest = Path(manifest_path)
    rows = _manifest_rows(manifest)
    records: list[StreamingRecordingMetrics] = []
    for row in rows:
        audio_path = resolve_manifest_audio_path(row, manifest, dataset_path=dataset_path)
        info = sf.info(str(audio_path))
        if info.samplerate < 1 or info.frames < 0:
            raise ValueError(f"Unreadable streaming audio: {audio_path}")
        duration_seconds = float(info.frames) / float(info.samplerate)
        windows: list[tuple[float, torch.Tensor]] = []
        false_alarms = 0
        positive_windows = 0
        scored_windows = 0
        next_alarm_time = -math.inf

        def score_pending() -> None:
            nonlocal false_alarms, positive_windows, scored_windows, next_alarm_time
            if not windows:
                return
            waveform = torch.stack([window for _, window in windows], dim=0).to(device)
            logits = model(waveform)
            scores = torch.softmax(logits.float(), dim=1)[:, positive_index].detach().cpu().tolist()
            for (start_seconds, _), score in zip(windows, scores):
                scored_windows += 1
                if score >= threshold:
                    positive_windows += 1
                    if start_seconds >= next_alarm_time:
                        false_alarms += 1
                        next_alarm_time = start_seconds + debounce_seconds
            windows.clear()

        for start_seconds, waveform in iter_sliding_audio_windows(
            audio_path, sample_rate=sample_rate, hop_seconds=hop_seconds
        ):
            windows.append((start_seconds, waveform))
            if len(windows) == batch_size:
                score_pending()
        score_pending()
        records.append(
            StreamingRecordingMetrics(
                audio_path=str(audio_path),
                duration_seconds=duration_seconds,
                false_alarms=false_alarms,
                positive_windows=positive_windows,
                scored_windows=scored_windows,
            )
        )
    return records


def _curve_points(scores: np.ndarray, labels: np.ndarray, positive_index: int) -> list[dict[str, float]]:
    thresholds = [float(np.nextafter(np.max(scores), np.inf)), *[float(value) for value in np.unique(scores)[::-1]], float(np.nextafter(np.min(scores), -np.inf))]
    points: list[dict[str, float]] = []
    positive_count = int(np.sum(labels == positive_index))
    negative_count = len(labels) - positive_count
    for threshold in thresholds:
        predicted_positive = scores >= threshold
        true_positive = int(np.sum(predicted_positive & (labels == positive_index)))
        false_positive = int(np.sum(predicted_positive & (labels != positive_index)))
        true_positive_rate = true_positive / max(1, positive_count)
        false_positive_rate = false_positive / max(1, negative_count)
        points.append(
            {
                "threshold": threshold,
                "true_positive_rate": true_positive_rate,
                "false_positive_rate": false_positive_rate,
                "false_negative_rate": 1.0 - true_positive_rate,
            }
        )
    return points


def _json_scalars(value):
    if isinstance(value, Mapping):
        return {str(key): _json_scalars(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_scalars(item) for item in value]
    return value


def _interval(values: Sequence[float], *, confidence_level: float) -> dict[str, float]:
    lower_quantile = (1.0 - confidence_level) / 2.0
    upper_quantile = 1.0 - lower_quantile
    return {
        "lower": float(np.quantile(values, lower_quantile)),
        "upper": float(np.quantile(values, upper_quantile)),
        "confidence_level": float(confidence_level),
    }


def _bootstrap_calibration_intervals(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    *,
    threshold: float,
    window_seconds: float,
    resamples: int,
    rng: np.random.Generator,
    confidence_level: float,
) -> dict[str, dict[str, float]]:
    frr_values: list[float] = []
    fah_values: list[float] = []
    for _ in range(resamples):
        pos = positive_scores[rng.integers(0, len(positive_scores), len(positive_scores))]
        neg = negative_scores[rng.integers(0, len(negative_scores), len(negative_scores))]
        metrics = eval_at_threshold(pos, neg, threshold, window_seconds)
        frr_values.append(float(metrics["frr"]))
        fah_values.append(float(metrics["fah"]))
    return {"frr": _interval(frr_values, confidence_level=confidence_level), "fah": _interval(fah_values, confidence_level=confidence_level)}


def _bootstrap_captured_positive_interval(
    scores: np.ndarray,
    *,
    threshold: float,
    resamples: int,
    rng: np.random.Generator,
    confidence_level: float,
) -> dict[str, dict[str, float]]:
    frr_values = [
        float(np.mean(scores[rng.integers(0, len(scores), len(scores))] < threshold))
        for _ in range(resamples)
    ]
    return {"frr": _interval(frr_values, confidence_level=confidence_level)}


def _bootstrap_streaming_interval(
    records: Sequence[StreamingRecordingMetrics],
    *,
    resamples: int,
    rng: np.random.Generator,
    confidence_level: float,
) -> dict[str, dict[str, float]]:
    fah_values: list[float] = []
    records = tuple(records)
    for _ in range(resamples):
        indexes = rng.integers(0, len(records), len(records))
        sampled_records = [records[int(index)] for index in indexes]
        fah_values.append(aggregate_streaming_recordings(sampled_records).fah)
    return {"fah": _interval(fah_values, confidence_level=confidence_level)}


def _negative_confusion(scores: np.ndarray, threshold: float) -> dict[str, int]:
    false_positive = int(np.sum(scores >= threshold))
    return {"tp": 0, "fn": 0, "fp": false_positive, "tn": int(len(scores) - false_positive)}


def _negative_confusion_from_recordings(records: Sequence[StreamingRecordingMetrics]) -> dict[str, int]:
    false_positive = sum(record.positive_windows for record in records)
    scored_windows = sum(record.scored_windows for record in records)
    return {"tp": 0, "fn": 0, "fp": false_positive, "tn": scored_windows - false_positive}


def _false_alarm_rate(confusion: Mapping[str, int]) -> float:
    return int(confusion["fp"]) / max(1, int(confusion["fp"]) + int(confusion["tn"]))


def build_deployment_report(
    *,
    calibration_scores: Sequence[float],
    calibration_labels: Sequence[int],
    calibration_manifest: Path | str,
    captured_positive_scores: Sequence[float],
    captured_positive_manifest: Path | str,
    false_wake_scores: Sequence[float] | Sequence[Sequence[float]] | Sequence[StreamingRecordingMetrics] | np.ndarray,
    false_wake_manifest: Path | str,
    long_negative_scores: Sequence[float] | Sequence[Sequence[float]] | Sequence[StreamingRecordingMetrics] | np.ndarray,
    long_negative_manifest: Path | str,
    target_fah: float,
    min_positive_recall: float,
    window_seconds: float,
    hop_seconds: float,
    debounce_seconds: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    positive_index: int,
    negative_index: int,
    confidence_level: float = 0.95,
    deployment_gate: DevelopmentDeploymentGate | None = None,
    captured_triggers: int | None = None,
) -> dict[str, object]:
    """Build all deployment evidence from precomputed scores without model duplication."""
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    if not math.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    manifests = require_explicit_deployment_manifests(
        {
            "calibration": calibration_manifest,
            "captured_positive": captured_positive_manifest,
            "false_wake": false_wake_manifest,
            "long_negative": long_negative_manifest,
        }
    )
    calibration = select_calibration_threshold(
        scores=calibration_scores,
        labels=calibration_labels,
        manifest_path=manifests["calibration"],
        target_fah=target_fah,
        window_seconds=window_seconds,
        positive_index=positive_index,
        negative_index=negative_index,
        min_positive_recall=min_positive_recall,
    )
    calibration_values = _validate_score_vector(calibration_scores, "calibration")
    calibration_labels_array = np.asarray(calibration_labels, dtype=np.int64)
    threshold = calibration.threshold
    captured_rows, _ = _validate_manifest_domain(
        manifests["captured_positive"],
        captured_positive_scores,
        positive_index=positive_index,
        negative_index=negative_index,
        expected_command="positive",
    )
    del captured_rows
    false_wake_records = _streaming_records_from_input(
        manifests["false_wake"],
        false_wake_scores,
        threshold=threshold,
        hop_seconds=hop_seconds,
        debounce_seconds=debounce_seconds,
        positive_index=positive_index,
        negative_index=negative_index,
    )
    long_negative_records = _streaming_records_from_input(
        manifests["long_negative"],
        long_negative_scores,
        threshold=threshold,
        hop_seconds=hop_seconds,
        debounce_seconds=debounce_seconds,
        positive_index=positive_index,
        negative_index=negative_index,
    )
    captured_values = _validate_score_vector(captured_positive_scores, "captured_positive")
    captured_tp = int(np.sum(captured_values >= threshold))
    captured_fn = int(len(captured_values) - captured_tp)
    false_wake_streaming = aggregate_streaming_recordings(false_wake_records)
    long_negative_streaming = aggregate_streaming_recordings(long_negative_records)
    curve = _curve_points(calibration_values, calibration_labels_array, positive_index)
    rng = np.random.default_rng(bootstrap_seed)
    calibration_positive = calibration_values[calibration_labels_array == positive_index]
    calibration_negative = calibration_values[calibration_labels_array == negative_index]
    confidence_intervals = {
        "calibration": _bootstrap_calibration_intervals(
            calibration_positive,
            calibration_negative,
            threshold=threshold,
            window_seconds=window_seconds,
            resamples=bootstrap_resamples,
            rng=rng,
            confidence_level=confidence_level,
        ),
        "captured_positive": _bootstrap_captured_positive_interval(
            captured_values,
            threshold=threshold,
            resamples=bootstrap_resamples,
            rng=rng,
            confidence_level=confidence_level,
        ),
        "false_wake": _bootstrap_streaming_interval(
            false_wake_records,
            resamples=bootstrap_resamples,
            rng=rng,
            confidence_level=confidence_level,
        ),
        "long_negative": _bootstrap_streaming_interval(
            long_negative_records,
            resamples=bootstrap_resamples,
            rng=rng,
            confidence_level=confidence_level,
        ),
    }
    calibration_confusion = {
        key: int(calibration.metrics[key])
        for key in ("tp", "fn", "fp", "tn")
    }
    calibration_metrics = {**calibration.metrics, "far": _false_alarm_rate(calibration_confusion)}
    false_wake_confusion = _negative_confusion_from_recordings(false_wake_records)
    long_negative_confusion = _negative_confusion_from_recordings(long_negative_records)
    if deployment_gate is None:
        deployment_best = {
            "eligible": True,
            "reason": "All required explicit evidence manifests are present; quality gates remain external.",
        }
    elif captured_triggers is None:
        # A gate must never infer captured-environment alarms from the positive
        # manifest.  Callers using an explicit gate must provide this separate
        # negative-domain count, otherwise selection fails closed.
        deployment_best = {
            "eligible": False,
            "reason": "missing_captured_triggers",
            "reasons": ["missing_captured_triggers"],
            "gate": deployment_gate.to_dict(),
        }
    else:
        tau_upper = float(confidence_intervals["long_negative"]["fah"]["upper"])
        evidence = DeploymentEvidence(
            positive_recall=float(captured_tp / max(1, len(captured_values))),
            captured_triggers=int(captured_triggers),
            false_wake_triggers=int(false_wake_streaming.false_alarms),
            max_tau_fah_upper=tau_upper,
            false_wake_fah_upper=float(confidence_intervals["false_wake"]["fah"]["upper"]),
        )
        eligible, reasons = deployment_gate.check(evidence)
        deployment_best = {
            "eligible": bool(eligible),
            "reason": "all_deployment_gates_pass" if eligible else "deployment_gate_failed",
            "reasons": list(reasons),
            "gate": deployment_gate.to_dict(),
            "evidence": evidence.__dict__,
        }

    return _json_scalars(
        {
            "schema_version": 1,
            "manifests": {name: str(path.resolve()) for name, path in manifests.items()},
            "threshold": {
                "value": threshold,
                "target_fah": target_fah,
                "min_positive_recall": min_positive_recall,
            },
            "roc": [
                {
                    "threshold": point["threshold"],
                    "true_positive_rate": point["true_positive_rate"],
                    "false_positive_rate": point["false_positive_rate"],
                }
                for point in curve
            ],
            "det": [
                {
                    "threshold": point["threshold"],
                    "false_positive_rate": point["false_positive_rate"],
                    "false_negative_rate": point["false_negative_rate"],
                }
                for point in curve
            ],
            "domains": {
                "calibration": {
                    "metrics": calibration_metrics,
                    "confusion": calibration_confusion,
                },
                "captured_positive": {
                    "score_count": int(len(captured_values)),
                    "frr": captured_fn / len(captured_values),
                    "confusion": {"tp": captured_tp, "fn": captured_fn, "fp": 0, "tn": 0},
                },
                "false_wake": {
                    "score_count": int(false_wake_streaming.scored_windows),
                    "far": _false_alarm_rate(false_wake_confusion),
                    "confusion": false_wake_confusion,
                    "streaming": asdict(false_wake_streaming),
                },
                "long_negative": {
                    "score_count": int(long_negative_streaming.scored_windows),
                    "far": _false_alarm_rate(long_negative_confusion),
                    "confusion": long_negative_confusion,
                    "streaming": asdict(long_negative_streaming),
                },
            },
            "confidence_intervals": confidence_intervals,
            "deployment_best": deployment_best,
        }
    )


def write_deployment_report(output_path: Path | str, report: Mapping[str, object]) -> None:
    """Atomically publish a strict JSON deployment report."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(_json_scalars(report), handle, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _evaluation_args(checkpoint_path: Path, *, batch: int, gpu: int, num_workers: int, seed: int) -> argparse.Namespace:
    config = load_scoring_config(checkpoint_path)
    args = scoring_namespace(config.values, checkpoint_path)
    args.batch = int(batch)
    args.gpu = int(gpu)
    args.num_workers = int(num_workers)
    args.seed = int(seed)
    args.eval_noise_aug = False
    args.noise_roots = None
    args.valid_noise_roots = None
    args.test_noise_roots = None
    args.noise_aug_prob = 0.0
    args.noise_snr_min_db = -5.0
    args.noise_snr_max_db = 20.0
    return args


def _score_one_second_manifest(
    model,
    manifest_path: Path,
    *,
    device: torch.device,
    args,
    dataset_path: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    loader = build_eval_loader(
        str(Path(dataset_path) if dataset_path is not None else manifest_path.parent),
        "validation",
        args,
        manifest_path=str(manifest_path),
    )
    return collect_scores(model, loader, device, CLASS_ENCODING["positive"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--captured-positive-manifest", type=Path, required=True)
    parser.add_argument("--false-wake-manifest", type=Path, required=True)
    parser.add_argument("--long-negative-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-fah", type=float, default=1.0)
    parser.add_argument("--min-positive-recall", type=float, default=0.93)
    parser.add_argument("--hop-seconds", type=float, default=0.25)
    parser.add_argument("--debounce-seconds", type=float, default=0.5)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()

    positive_index = CLASS_ENCODING["positive"]
    negative_index = CLASS_ENCODING["negative"]
    require_explicit_deployment_manifests(
        {
            "calibration": args.calibration_manifest,
            "captured_positive": args.captured_positive_manifest,
            "false_wake": args.false_wake_manifest,
            "long_negative": args.long_negative_manifest,
        }
    )
    require_non_test_calibration(args.calibration_manifest)
    evaluation_args = _evaluation_args(args.ckpt, batch=args.batch, gpu=args.gpu, num_workers=args.num_workers, seed=args.seed)
    device = torch.device("cuda:0" if args.gpu > 0 and torch.cuda.is_available() else "cpu")
    model = build_model(evaluation_args, device)
    calibration_scores, calibration_labels = _score_one_second_manifest(
        model, args.calibration_manifest, device=device, args=evaluation_args, dataset_path=args.dataset_root
    )
    captured_scores, captured_labels = _score_one_second_manifest(
        model, args.captured_positive_manifest, device=device, args=evaluation_args, dataset_path=args.dataset_root
    )
    _validate_manifest_domain(
        args.captured_positive_manifest,
        captured_scores,
        positive_index=positive_index,
        negative_index=negative_index,
        expected_command="positive",
        observed_labels=captured_labels,
    )
    calibration = select_calibration_threshold(
        scores=calibration_scores,
        labels=calibration_labels,
        manifest_path=args.calibration_manifest,
        target_fah=args.target_fah,
        window_seconds=1.0,
        positive_index=positive_index,
        negative_index=negative_index,
        min_positive_recall=args.min_positive_recall,
    )
    false_wake_records = evaluate_streaming_manifest(
        model,
        args.false_wake_manifest,
        device=device,
        sample_rate=evaluation_args.sample_rate,
        batch_size=args.batch,
        hop_seconds=args.hop_seconds,
        debounce_seconds=args.debounce_seconds,
        threshold=calibration.threshold,
        positive_index=positive_index,
        dataset_path=args.dataset_root,
    )
    long_negative_records = evaluate_streaming_manifest(
        model,
        args.long_negative_manifest,
        device=device,
        sample_rate=evaluation_args.sample_rate,
        batch_size=args.batch,
        hop_seconds=args.hop_seconds,
        debounce_seconds=args.debounce_seconds,
        threshold=calibration.threshold,
        positive_index=positive_index,
        dataset_path=args.dataset_root,
    )
    report = build_deployment_report(
        calibration_scores=calibration_scores,
        calibration_labels=calibration_labels,
        calibration_manifest=args.calibration_manifest,
        captured_positive_scores=captured_scores,
        captured_positive_manifest=args.captured_positive_manifest,
        false_wake_scores=false_wake_records,
        false_wake_manifest=args.false_wake_manifest,
        long_negative_scores=long_negative_records,
        long_negative_manifest=args.long_negative_manifest,
        target_fah=args.target_fah,
        min_positive_recall=args.min_positive_recall,
        window_seconds=1.0,
        hop_seconds=args.hop_seconds,
        debounce_seconds=args.debounce_seconds,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        positive_index=positive_index,
        negative_index=negative_index,
    )
    write_deployment_report(args.output, report)
    print(json.dumps({"output": str(args.output), "threshold": report["threshold"]}, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
