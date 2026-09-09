"""Validation-only selection for KWS learning-rate candidates."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CandidateSelectionReport:
    """The selected validation checkpoint and its candidate audit."""

    selected_run: str
    selected_lr: float
    selected_epoch: int
    candidates: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _ValidationEpoch:
    epoch: int
    macro_f1: float
    positive_recall: float
    negative_recall: float


def _skip_json_string(source: str, start: int) -> int | None:
    """Return the first index after one syntactically valid JSON string."""
    if start >= len(source) or source[start] != '"':
        return None
    index = start + 1
    while index < len(source):
        character = source[index]
        if character == '"':
            return index + 1
        if ord(character) < 0x20:
            return None
        if character == "\\":
            if index + 1 >= len(source):
                return None
            escaped = source[index + 1]
            if escaped == "u":
                if index + 5 >= len(source) or any(value not in "0123456789abcdefABCDEF" for value in source[index + 2 : index + 6]):
                    return None
                index += 6
                continue
            if escaped not in '"\\/bfnrt':
                return None
            index += 2
            continue
        index += 1
    return None


def _decode_json_string(source: str, start: int) -> tuple[str, int] | None:
    """Decode one standalone JSON string, never its containing metric row."""
    stop = _skip_json_string(source, start)
    if stop is None:
        return None
    try:
        value = json.loads(source[start:stop])
    except json.JSONDecodeError:
        return None
    return (value, stop) if isinstance(value, str) else None


def _skip_json_value(source: str, start: int) -> int | None:
    """Lexically skip one JSON value without decoding nested fields."""
    if start >= len(source):
        return None
    if source[start] == '"':
        return _skip_json_string(source, start)
    if source[start] not in "[{":
        stop = start
        while stop < len(source) and source[stop] not in ",}]":
            stop += 1
        return stop if stop > start else None

    closers = ["]" if source[start] == "[" else "}"]
    index = start + 1
    while index < len(source):
        character = source[index]
        if character == '"':
            index = _skip_json_string(source, index)
            if index is None:
                return None
            continue
        if character == "[":
            closers.append("]")
        elif character == "{":
            closers.append("}")
        elif character in "]}":
            if character != closers.pop():
                return None
            if not closers:
                return index + 1
        index += 1
    return None


def _skip_whitespace(source: str, start: int) -> int:
    while start < len(source) and source[start].isspace():
        start += 1
    return start


def _top_level_split_value(source: str) -> str | None:
    """Accept exactly one semantically decoded top-level ``split='valid'``."""
    index = _skip_whitespace(source, 0)
    if index >= len(source) or source[index] != "{":
        return None
    index += 1
    split_value: str | None = None
    while True:
        index = _skip_whitespace(source, index)
        if index >= len(source):
            return None
        if source[index] == "}":
            index += 1
            break
        parsed_key = _decode_json_string(source, index)
        if parsed_key is None:
            return None
        key, index = parsed_key
        index = _skip_whitespace(source, index)
        if index >= len(source) or source[index] != ":":
            return None
        index = _skip_whitespace(source, index + 1)
        if key == "split":
            if split_value is not None:
                return None
            parsed_value = _decode_json_string(source, index)
            if parsed_value is None:
                return None
            split_value, index = parsed_value
            if split_value != "valid":
                return None
        else:
            index = _skip_json_value(source, index)
            if index is None:
                return None
        index = _skip_whitespace(source, index)
        if index >= len(source):
            return None
        if source[index] == "}":
            index += 1
            break
        if source[index] != ",":
            return None
        index += 1
    return "valid" if split_value == "valid" and _skip_whitespace(source, index) == len(source) else None


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _read_learning_rate(run_dir: Path) -> float | None:
    argv_path = run_dir / ".argv.json"
    try:
        argv_value = json.loads(argv_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        argv_value = None

    if isinstance(argv_value, Mapping):
        value = argv_value.get("lr")
        if value is not None:
            try:
                lr = float(value)
            except (TypeError, ValueError):
                return None
            return lr if math.isfinite(lr) and lr > 0.0 else None
    elif isinstance(argv_value, list):
        for entry in argv_value:
            if not isinstance(entry, str) or "=" not in entry:
                continue
            key, value = entry.split("=", 1)
            if key.strip().lstrip("-").replace("_", "-") != "lr":
                continue
            try:
                lr = float(value)
            except ValueError:
                return None
            return lr if math.isfinite(lr) and lr > 0.0 else None

    launcher_config_path = run_dir / "launcher_config.json"
    try:
        launcher_config = json.loads(launcher_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(launcher_config, Mapping):
        return None
    accuracy_policy = launcher_config.get("accuracy_policy")
    if not isinstance(accuracy_policy, Mapping):
        return None
    try:
        lr = float(accuracy_policy["lr"])
    except (KeyError, TypeError, ValueError):
        return None
    return lr if math.isfinite(lr) and lr > 0.0 else None


def _read_validation_epochs(run_dir: Path) -> tuple[list[_ValidationEpoch], str | None]:
    metrics_path = run_dir / "kws_metrics.jsonl"
    try:
        lines = metrics_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return [], "missing_validation_metrics"

    validation_epochs: list[_ValidationEpoch] = []
    for line in lines:
        if not line.strip():
            continue
        # Identify the top-level split lexically before parsing.  Locked test
        # rows are consequently never decoded, including malformed rows or
        # rows containing nested fields named ``split``.
        if (_top_level_split_value(line) or "").casefold() != "valid":
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return [], "invalid_validation_metrics"
        if not isinstance(row, Mapping) or str(row.get("split", "")).casefold() != "valid":
            return [], "invalid_validation_metrics"
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            return [], "invalid_validation_metrics"
        try:
            validation_epoch = _ValidationEpoch(
                epoch=int(row["epoch"]),
                macro_f1=float(metrics["macro_f1"]),
                positive_recall=float(metrics["positive_recall"]),
                negative_recall=float(metrics["negative_recall"]),
            )
        except (KeyError, TypeError, ValueError):
            return [], "invalid_validation_metrics"
        if not all(
            math.isfinite(value)
            for value in (
                validation_epoch.macro_f1,
                validation_epoch.positive_recall,
                validation_epoch.negative_recall,
            )
        ):
            return [], "non_finite_validation_metrics"
        if validation_epoch.positive_recall <= 0.0 or validation_epoch.negative_recall <= 0.0:
            return [], "class_collapse"
        validation_epochs.append(validation_epoch)

    if not validation_epochs:
        return [], "missing_validation_metrics"
    return validation_epochs, None


def _candidate_audit(run_dir: Path, min_positive_recall: float) -> dict[str, Any]:
    learning_rate = _read_learning_rate(run_dir)
    validation_epochs, rejection_reason = _read_validation_epochs(run_dir)
    audit: dict[str, Any] = {
        "run_dir": str(run_dir),
        "learning_rate": learning_rate,
        "validation_epoch_count": len(validation_epochs),
        "eligible_epochs": [],
        "peak_validation_macro_f1": None,
        "positive_recall": None,
        "negative_recall": None,
        "selected_epoch": None,
        "rejection_reason": rejection_reason,
    }
    if learning_rate is None and rejection_reason is None:
        audit["rejection_reason"] = "missing_learning_rate"
    if audit["rejection_reason"] is not None:
        return audit

    eligible = [epoch for epoch in validation_epochs if epoch.positive_recall >= min_positive_recall]
    audit["eligible_epochs"] = [epoch.epoch for epoch in eligible]
    if not eligible:
        audit["rejection_reason"] = "positive_recall_below_gate"
        return audit

    selected = min(eligible, key=lambda epoch: (-epoch.macro_f1, -epoch.negative_recall, epoch.epoch))
    audit.update(
        {
            "peak_validation_macro_f1": selected.macro_f1,
            "positive_recall": selected.positive_recall,
            "negative_recall": selected.negative_recall,
            "selected_epoch": selected.epoch,
        }
    )
    return audit


def _default_output_path(run_dirs: Sequence[Path]) -> Path:
    parents = {run_dir.parent for run_dir in run_dirs}
    return (next(iter(parents)) if len(parents) == 1 else Path.cwd()) / "candidate_selection.json"


def select_candidate(
    run_dirs: Sequence[str | Path],
    *,
    min_positive_recall: float,
    output_path: str | Path | None = None,
) -> CandidateSelectionReport:
    """Select the strongest stable candidate from validation metrics only.

    Test metric rows and test manifests are deliberately never read here.  A
    candidate must have finite validation metrics, preserve both classes, and
    meet the requested positive-recall gate before it can compete on macro-F1.
    """
    if not math.isfinite(min_positive_recall) or not 0.0 <= min_positive_recall <= 1.0:
        raise ValueError("min_positive_recall must be finite and between zero and one")
    resolved_run_dirs = [Path(run_dir).expanduser().resolve() for run_dir in run_dirs]
    if not resolved_run_dirs:
        raise ValueError("At least one candidate run is required")
    audits = [_candidate_audit(run_dir, float(min_positive_recall)) for run_dir in resolved_run_dirs]
    eligible = [audit for audit in audits if audit["rejection_reason"] is None]
    selected = (
        min(
            eligible,
            key=lambda audit: (
                -float(audit["peak_validation_macro_f1"]),
                -float(audit["negative_recall"]),
                str(audit["run_dir"]),
            ),
        )
        if eligible
        else None
    )
    output = Path(output_path).expanduser().resolve() if output_path is not None else _default_output_path(resolved_run_dirs)
    payload: dict[str, Any] = {
        "min_positive_recall": float(min_positive_recall),
        "candidates": audits,
        "selected_run": selected["run_dir"] if selected is not None else None,
        "selected_lr": selected["learning_rate"] if selected is not None else None,
        "selected_epoch": selected["selected_epoch"] if selected is not None else None,
    }
    _atomic_write_json(output, payload)
    if selected is None:
        raise RuntimeError(f"No stable candidate met the positive recall gate; see {output}")
    return CandidateSelectionReport(
        selected_run=str(selected["run_dir"]),
        selected_lr=float(selected["learning_rate"]),
        selected_epoch=int(selected["selected_epoch"]),
        candidates=tuple(audits),
    )
