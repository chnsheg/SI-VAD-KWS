"""Deterministic train-only hard-negative selection helpers."""

from __future__ import annotations

import ast
import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as audio_functional

from .build_reclean_v2_pool import iter_jsonl, protected_audio_paths


@dataclass(frozen=True)
class MiningCandidate:
    audio_path: str
    source_split: str
    source_role: str
    start_sample: int
    duration_seconds: float
    positive_score: float
    selection_reason: str = ""


@dataclass(frozen=True)
class ScoringConfig:
    values: Mapping[str, Any]

    @property
    def sample_rate(self) -> int:
        return int(self.values["sample_rate"])

    @property
    def model_size_info(self) -> list[int]:
        return [int(value) for value in self.values["model_size_info"]]


@dataclass(frozen=True)
class HardNegativeMiningResult:
    audit_path: Path
    manifest_path: Path
    report_path: Path
    selected_count: int


SCORING_MODEL_KEYS = (
    "sample_rate",
    "window_stride_ms",
    "dct_coeff",
    "model_size_info",
    "frontend",
    "window_size_ms",
    "bandpass_n_bands",
    "bandpass_f_min",
    "bandpass_f_max",
    "bandpass_spacing",
    "bandpass_kernel_size",
    "bandpass_phase_count",
    "pre_emphasis",
    "pre_emphasis_coeff",
    "mfcc_impl",
    "mel_filter_shape",
    "log_approx_mode",
    "log_pwl_num_segments",
    "log_pwl_strategy",
    "log_pwl_gamma",
    "log_offset",
    "log_input_clamp_min",
)

# Architecture-head options were added after the original checkpoint
# contract.  Keep them optional so historical global-pool checkpoints remain
# fully readable while temporal-head runs can be mined/evaluated faithfully.
OPTIONAL_SCORING_MODEL_DEFAULTS = {
    "model": "dscnn",
    "pooling": "global",
    "temporal_bins": 4,
    "mfcc_scale": "torchaudio_db",
    "mfcc_c0_cmn": False,
    "tcn_channels": 68,
    "tcn_blocks": 4,
    "tcn_kernel_size": 3,
    "tcn_dilations": [1, 1, 2, 2],
    "tcn_temporal_bins": 8,
}


def _parse_argv_value(value: str) -> Any:
    if value in {"True", "False", "None"}:
        return {"True": True, "False": False, "None": None}[value]
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def load_scoring_config(checkpoint_path: Path | str) -> ScoringConfig:
    checkpoint_path = Path(checkpoint_path)
    argv_path = checkpoint_path.parent / ".argv.json"
    try:
        entries = json.loads(argv_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read checkpoint run configuration: {argv_path}") from error
    if not isinstance(entries, list):
        raise ValueError(f"Checkpoint run configuration must be a list: {argv_path}")
    values: dict[str, Any] = {}
    for entry in entries:
        if not isinstance(entry, str) or "=" not in entry:
            raise ValueError(f"Malformed checkpoint run configuration entry: {entry!r}")
        key, value = entry.split("=", maxsplit=1)
        if not key:
            raise ValueError(f"Malformed checkpoint run configuration entry: {entry!r}")
        values[key] = _parse_argv_value(value)
    missing = {"sample_rate", "model_size_info"}.difference(values)
    if missing:
        raise ValueError(f"Checkpoint run configuration is missing: {sorted(missing)}")
    if not isinstance(values["model_size_info"], (list, tuple)):
        raise ValueError("Checkpoint model_size_info must be a list")
    return ScoringConfig(values=values)


def scoring_namespace(values: Mapping[str, Any], checkpoint_path: Path | str) -> argparse.Namespace:
    missing = [key for key in SCORING_MODEL_KEYS if key not in values]
    if missing:
        raise ValueError(f"Checkpoint run configuration is missing model arguments: {', '.join(missing)}")
    namespace_values = {key: values[key] for key in SCORING_MODEL_KEYS}
    for key, default in OPTIONAL_SCORING_MODEL_DEFAULTS.items():
        namespace_values[key] = values.get(key, default)
    namespace_values["ckpt"] = str(checkpoint_path)
    return argparse.Namespace(**namespace_values)


def build_checkpoint_scorer(checkpoint_path: Path | str, device: torch.device):
    """Return a score function that reconstructs the exact trained frontend/model."""
    checkpoint_path = Path(checkpoint_path)
    config = load_scoring_config(checkpoint_path)
    args = scoring_namespace(config.values, checkpoint_path)
    from dscnn_kws.configs import CLASS_ENCODING
    from dscnn_kws.eval_fah_frr import build_model

    model = build_model(args, device)
    positive_index = CLASS_ENCODING["positive"]

    @torch.no_grad()
    def score_candidate(candidate: MiningCandidate) -> float:
        window = _canonical_window(candidate, config.sample_rate)
        waveform = torch.from_numpy(window).view(1, 1, -1).to(device)
        logits = model(waveform)
        return float(torch.softmax(logits.float(), dim=1)[0, positive_index].item())

    return score_candidate


def validate_false_wake_sources(rows: Sequence[Mapping[str, object]]) -> None:
    for row in rows:
        if row.get("source_split") != "train":
            raise ValueError(f"False-wake source must be train split, got {row.get('source_split')!r}")
        value = row.get("audio_path", row.get("audio_filepath"))
        if not isinstance(value, str) or not value.strip():
            raise ValueError("False-wake source must include an audio path")


def select_hard_negatives(candidates: Sequence[MiningCandidate]) -> list[MiningCandidate]:
    raw = [
        row
        for row in candidates
        if row.source_split == "train" and row.source_role == "raw_negative" and math.isfinite(row.positive_score)
    ]
    false_wake = [
        row
        for row in candidates
        if row.source_split == "train" and row.source_role == "false_wake" and math.isfinite(row.positive_score)
    ]
    if not raw:
        raise ValueError("No train-split raw negative candidates")
    ordered_scores = sorted(row.positive_score for row in raw)
    threshold = ordered_scores[math.ceil(0.75 * (len(ordered_scores) - 1))]
    selected = [
        replace(row, selection_reason="raw_negative_upper_quartile")
        for row in raw
        if row.positive_score >= threshold
    ]
    selected.extend(
        replace(row, selection_reason="false_wake_above_raw_q75")
        for row in false_wake
        if row.positive_score >= threshold
    )
    if not selected:
        raise ValueError("Hard-negative selection is empty")
    return selected


def _canonical_window(candidate: MiningCandidate, sample_rate: int) -> np.ndarray:
    samples, source_rate = sf.read(candidate.audio_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(samples.T.copy())
    if source_rate != sample_rate:
        waveform = audio_functional.resample(waveform, source_rate, sample_rate)
        start_sample = round(candidate.start_sample * sample_rate / source_rate)
    else:
        start_sample = candidate.start_sample
    # Match the production streaming-input contract: resample each channel
    # independently, then retain channel 0.  Averaging channels here can
    # introduce phase cancellation and makes mined windows differ from what
    # the deployed evaluator/model actually sees.
    waveform = waveform.narrow(0, 0, 1)
    window_samples = round(candidate.duration_seconds * sample_rate)
    if window_samples <= 0:
        raise ValueError("Hard-negative duration must be positive")
    start_sample = max(0, int(start_sample))
    window = waveform[:, start_sample : start_sample + window_samples]
    if window.shape[1] < window_samples:
        window = torch.nn.functional.pad(window, (0, window_samples - window.shape[1]))
    return window.squeeze(0).clamp(-1.0, 1.0).numpy()


def materialize_selected_window(
    candidate: MiningCandidate,
    output_root: Path | str,
    *,
    sample_rate: int,
) -> dict[str, object]:
    if candidate.source_split != "train":
        raise ValueError(f"Cannot materialize non-train hard negative: {candidate.audio_path}")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(
        f"{candidate.audio_path}|{candidate.start_sample}|{candidate.duration_seconds}|{candidate.selection_reason}".encode("utf-8")
    ).hexdigest()[:20]
    output_path = output_root / f"hard-negative-{digest}.wav"
    if not output_path.exists():
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp.wav", dir=output_root)
        os.close(descriptor)
        try:
            sf.write(temporary_name, _canonical_window(candidate, sample_rate), sample_rate, subtype="PCM_16")
            os.replace(temporary_name, output_path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
    return {
        "audio_filepath": str(output_path.resolve()),
        "command": "negative",
        "online_window_jitter_max_ms": 0,
        "source_role": "hard_negative",
        "source_split": "train",
        "source_audio_filepath": str(Path(candidate.audio_path).resolve()),
        "source_start_sample": int(candidate.start_sample),
        "duration_seconds": float(candidate.duration_seconds),
        "positive_score": float(candidate.positive_score),
        "selection_reason": candidate.selection_reason,
    }


def _resolve_manifest_audio(row: Mapping[str, object], manifest_path: Path) -> str:
    value = row.get("audio_filepath", row.get("audio_path"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing audio path in {manifest_path}")
    path = Path(value.strip())
    if not path.is_absolute():
        path = manifest_path.parent / path
    return str(path.resolve())


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=True, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(value), handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _candidate_key(candidate: MiningCandidate) -> tuple[str, str, int, float]:
    return (candidate.audio_path, candidate.source_role, candidate.start_sample, candidate.duration_seconds)


def _candidate_audit_row(candidate: MiningCandidate, selected: bool) -> dict[str, object]:
    return {
        "audio_path": candidate.audio_path,
        "duration_seconds": candidate.duration_seconds,
        "positive_score": candidate.positive_score,
        "selected": selected,
        "selection_reason": candidate.selection_reason if selected else "",
        "source_role": candidate.source_role,
        "source_split": candidate.source_split,
        "start_sample": candidate.start_sample,
        "command": "negative",
    }


def _false_wake_candidates(rows: Sequence[Mapping[str, object]]) -> list[MiningCandidate]:
    validate_false_wake_sources(rows)
    candidates: list[MiningCandidate] = []
    for row in rows:
        audio_path = str(Path(str(row.get("audio_path", row.get("audio_filepath")))).resolve())
        info = sf.info(audio_path)
        if info.samplerate <= 0 or info.frames < 0:
            raise ValueError(f"Unreadable false-wake audio: {audio_path}")
        window_samples = max(1, int(info.samplerate))
        starts = range(0, max(1, int(info.frames)), window_samples)
        candidates.extend(
            MiningCandidate(
                audio_path=audio_path,
                source_split="train",
                source_role="false_wake",
                start_sample=start,
                duration_seconds=1.0,
                positive_score=float("nan"),
            )
            for start in starts
        )
    return candidates


def mine_hard_negatives(
    *,
    train_manifest: Path | str,
    validation_manifest: Path | str,
    test_manifest: Path | str,
    false_wake_sources: Sequence[Mapping[str, object]],
    output_root: Path | str,
    score_candidate,
    sample_rate: int,
) -> HardNegativeMiningResult:
    """Score eligible negative windows and atomically publish an auditable selected pool."""
    train_manifest = Path(train_manifest)
    protected = protected_audio_paths(Path(validation_manifest), Path(test_manifest))
    candidates: list[MiningCandidate] = []
    for row in iter_jsonl(train_manifest):
        if row.get("command", row.get("label")) != "negative":
            continue
        audio_path = _resolve_manifest_audio(row, train_manifest)
        if audio_path in protected:
            raise ValueError(f"Hard-negative candidate is held-out: {audio_path}")
        candidates.append(
            MiningCandidate(
                audio_path=audio_path,
                source_split="train",
                source_role="raw_negative",
                start_sample=0,
                duration_seconds=1.0,
                positive_score=float("nan"),
            )
        )
    candidates.extend(_false_wake_candidates(false_wake_sources))
    scored = [replace(candidate, positive_score=float(score_candidate(candidate))) for candidate in candidates]
    if not all(math.isfinite(candidate.positive_score) for candidate in scored):
        raise ValueError("Hard-negative scorer returned a non-finite score")
    selected = select_hard_negatives(scored)
    selected_by_key = {_candidate_key(candidate): candidate for candidate in selected}
    output_root = Path(output_root)
    materialized_rows = [
        materialize_selected_window(candidate, output_root / "hard_negative_audio", sample_rate=sample_rate)
        for candidate in selected
    ]
    audit_rows = [
        _candidate_audit_row(
            replace(candidate, selection_reason=selected_by_key[_candidate_key(candidate)].selection_reason)
            if _candidate_key(candidate) in selected_by_key
            else candidate,
            _candidate_key(candidate) in selected_by_key,
        )
        for candidate in scored
    ]
    audit_path = output_root / "hard_negative_audit.jsonl"
    manifest_path = output_root / "hard_negative_manifest.jsonl"
    report_path = output_root / "hard_negative_report.json"
    _atomic_write_jsonl(audit_path, audit_rows)
    _atomic_write_jsonl(manifest_path, materialized_rows)
    _atomic_write_json(
        report_path,
        {
            "candidate_count": len(scored),
            "selected_count": len(materialized_rows),
            "raw_negative_selected_count": sum(row.source_role == "raw_negative" for row in selected),
            "false_wake_selected_count": sum(row.source_role == "false_wake" for row in selected),
            "sample_rate": int(sample_rate),
        },
    )
    return HardNegativeMiningResult(audit_path, manifest_path, report_path, len(materialized_rows))
