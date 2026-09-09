"""Rank-zero training artifacts consumed by the existing XPS dashboard."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


DASHBOARD_RECO_METRIC = "one_minus_validation_macro_f1"
STATUS_CLAIM_STALE_SECONDS = 60
SENSITIVE_ARG_KEY_PARTS = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "cookie",
    "private_key",
    "access_key",
)


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _normalized_argv_key(key: Any) -> str:
    return str(key).lstrip("-").replace("-", "_")


def _is_sensitive_argv_key(key: str) -> bool:
    normalized_key = key.casefold()
    return any(part in normalized_key for part in SENSITIVE_ARG_KEY_PARTS)


def _stringify_argv_value(value: Any) -> str:
    normalized_value = _json_value(value)
    if isinstance(normalized_value, str):
        return normalized_value
    if isinstance(normalized_value, (dict, list)):
        return json.dumps(normalized_value, ensure_ascii=True, sort_keys=True)
    return str(normalized_value)


def _dashboard_argv_entries(argv: Mapping[str, Any] | None, run_variant: str | None) -> list[str]:
    entries = {}
    for key, value in (argv or {}).items():
        normalized_key = _normalized_argv_key(key)
        if normalized_key and not _is_sensitive_argv_key(normalized_key):
            entries[normalized_key] = _stringify_argv_value(value)
    entries.update(
        {
            "dashboard_reco_metric": DASHBOARD_RECO_METRIC,
            "run_kind": "kws_reclean",
            "selected_best_metric": "validation_macro_f1",
            "variant": _stringify_argv_value(run_variant or "unspecified"),
        }
    )
    return [f"{key}={entries[key]}" for key in sorted(entries)]


def _atomic_write_json(path: Path, value: Any) -> None:
    """Write complete JSON to a sibling temporary file, then replace it."""
    fd, temporary_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(_json_value(value), handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _atomic_write_text(path: Path, text: str) -> None:
    fd, temporary_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _read_complete_lines(path: Path, json_lines: bool = False) -> list[str]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    if raw and not raw.endswith("\n"):
        last_newline = raw.rfind("\n")
        raw = raw[: last_newline + 1] if last_newline >= 0 else ""
    lines = raw.splitlines(keepends=True)
    if not json_lines:
        return lines
    complete_json_lines = []
    for line in lines:
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError:
            continue
        complete_json_lines.append(line)
    return complete_json_lines


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _claim_status_path(path: Path) -> Path | None:
    claim_path = path.with_name(f".{path.name}.claim")
    for _ in range(2):
        try:
            fd = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            try:
                claim_age = time.time() - claim_path.stat().st_mtime
            except OSError:
                continue
            if claim_age < STATUS_CLAIM_STALE_SECONDS:
                return None
            try:
                claim_path.unlink()
            except FileNotFoundError:
                continue
        else:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(f"pid={os.getpid()} timestamp={_timestamp()}\n")
                handle.flush()
                os.fsync(handle.fileno())
            return claim_path
    return None


def _atomic_write_terminal_status(path: Path, value: Any, allow_missing: bool) -> bool:
    """Replace only the active running status with one terminal status."""
    claim_path = _claim_status_path(path)
    if claim_path is None:
        return False
    try:
        if path.exists():
            status = _read_json_object(path)
            if status is None or status.get("status") != "running":
                return False
        elif not allow_missing:
            return False
        _atomic_write_json(path, value)
        return True
    finally:
        try:
            claim_path.unlink()
        except FileNotFoundError:
            pass


def _next_previous_status_path(status_path: Path) -> Path:
    first = status_path.with_name("status.previous.json")
    if not first.exists():
        return first
    index = 1
    while True:
        candidate = status_path.with_name(f"status.previous.{index}.json")
        if not candidate.exists():
            return candidate
        index += 1


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S,000")


class FailureArtifactReporter:
    """Failure-only reporting safe for every distributed rank."""

    def __init__(self, save_dir: str | Path, rank: int):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.rank = int(rank)
        self.status_path = self.save_dir / "status.json"
        self.sentinel_path = self.save_dir / f"failure-rank-{self.rank}.json"
        self.rank_zero_log_path = self.save_dir / "train.log"

    def _rank_zero_log_tail(self) -> str:
        try:
            return self.rank_zero_log_path.read_text(encoding="utf-8", errors="replace")[-4096:]
        except OSError:
            return ""

    def write_terminal_status(self, status: str, payload: Mapping[str, Any] | None = None) -> bool:
        if status not in {"failed", "interrupted", "completed"}:
            raise ValueError(f"Unsupported terminal status: {status}")
        terminal_payload = dict(payload or {})
        terminal_payload["status"] = status
        terminal_payload.setdefault("rank", self.rank)
        terminal_payload.setdefault("timestamp", _timestamp())
        return _atomic_write_terminal_status(
            self.status_path,
            terminal_payload,
            allow_missing=status != "completed",
        )

    def write_failure(self, error: BaseException, traceback_text: str) -> bool:
        payload = {
            "status": "failed",
            "rank": self.rank,
            "timestamp": _timestamp(),
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": traceback_text,
            "rank_zero_train_log_tail": self._rank_zero_log_tail(),
        }
        if self.rank != 0:
            _atomic_write_json(self.sentinel_path, payload)
        return self.write_terminal_status("failed", payload)

    def write_interrupted(self, error: BaseException, traceback_text: str) -> bool:
        payload = {
            "status": "interrupted",
            "rank": self.rank,
            "timestamp": _timestamp(),
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": traceback_text,
            "rank_zero_train_log_tail": self._rank_zero_log_tail(),
        }
        if self.rank != 0:
            _atomic_write_json(self.sentinel_path, payload)
            return False
        return self.write_terminal_status("interrupted", payload)


class TrainingArtifactWriter:
    """Persist dashboard-compatible summaries and complete KWS metrics.

    The dashboard treats ``reco`` as lower-is-better. It is therefore always
    ``1.0 - macro_f1`` here; ``kws_metrics.jsonl`` preserves the KWS metrics.
    """

    def __init__(self, save_dir: str | Path, argv: Mapping[str, Any] | None = None, run_variant: str | None = None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.argv_path = self.save_dir / ".argv.json"
        self.history_path = self.save_dir / "history.json"
        self.metrics_path = self.save_dir / "kws_metrics.jsonl"
        self.mixture_metrics_path = self.save_dir / "mixture_metrics.jsonl"
        self.log_path = self.save_dir / "train.log"
        self.live_metrics_path = self.save_dir / "live_metrics.json"
        self.status_path = self.save_dir / "status.json"
        self._start_run()
        self.history = self._read_history()
        self._log_lines = _read_complete_lines(self.log_path)
        self._metric_lines = _read_complete_lines(self.metrics_path, json_lines=True)
        self._mixture_metric_lines = _read_complete_lines(self.mixture_metrics_path, json_lines=True)
        _atomic_write_json(
            self.argv_path,
            _dashboard_argv_entries(argv, run_variant),
        )

    def _start_run(self) -> None:
        if self.status_path.exists():
            os.replace(self.status_path, _next_previous_status_path(self.status_path))
        _atomic_write_json(
            self.status_path,
            {
                "status": "running",
                "rank": 0,
                "timestamp": _timestamp(),
            },
        )

    def _read_history(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        with self.history_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, list):
            raise ValueError(f"Expected history.json to contain a list: {self.history_path}")
        return value

    @staticmethod
    def dashboard_metrics(metrics: Mapping[str, Any] | Any) -> dict[str, Any]:
        metric_dict = _json_value(metrics)
        if not isinstance(metric_dict, dict):
            raise TypeError("Epoch metrics must serialize to a mapping")
        macro_f1 = float(metric_dict.get("macro_f1", metric_dict.get("f1", 0.0)))
        metric_dict["macro_f1"] = macro_f1
        metric_dict["reco"] = round(1.0 - macro_f1, 12)
        return metric_dict

    def _append_log(self, message: str) -> None:
        lines = [*self._log_lines, f"[{_timestamp()}] [INFO] - {message}\n"]
        _atomic_write_text(self.log_path, "".join(lines))
        self._log_lines = lines

    def _append_metric(self, epoch: int, split: str, metrics: dict[str, Any], metadata: Mapping[str, Any] | None = None) -> None:
        payload = {"epoch": int(epoch), "split": split, "metrics": metrics}
        if metadata:
            payload.update(_json_value(metadata))
        lines = [*self._metric_lines, json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n"]
        _atomic_write_text(self.metrics_path, "".join(lines))
        self._metric_lines = lines

    def write_mixture_metrics(self, epoch: int, role_counts: Mapping[str, int]) -> None:
        payload = {
            "epoch": int(epoch),
            "role_counts": {str(key): int(value) for key, value in role_counts.items()},
        }
        lines = [*self._mixture_metric_lines, json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n"]
        _atomic_write_text(self.mixture_metrics_path, "".join(lines))
        self._mixture_metric_lines = lines

    def _history_row(self, epoch: int) -> dict[str, Any]:
        for row in self.history:
            if int(row.get("epoch", -1)) == int(epoch):
                return row
        row = {"epoch": int(epoch)}
        self.history.append(row)
        self.history.sort(key=lambda item: int(item["epoch"]))
        return row

    def _write_history(self) -> None:
        _atomic_write_json(self.history_path, self.history)

    def replace_history(self, history: list[dict[str, Any]]) -> None:
        self.history = _json_value(history)
        self._write_history()

    def write_train_progress(
        self,
        epoch: int,
        step: int,
        total_steps: int,
        it_per_sec: float,
        batch_loss: float,
        telemetry: Mapping[str, Any] | None = None,
    ) -> None:
        self._append_log(
            f"Train | Epoch {int(epoch)} | {int(step)}/{int(total_steps)} | "
            f"{float(it_per_sec):.2f} it/sec | Loss {float(batch_loss):.4f}"
        )
        if telemetry is not None:
            payload = {
                "epoch": int(epoch),
                "step": int(step),
                "total_steps": int(total_steps),
                "batch_loss": float(batch_loss),
                "it_per_sec": float(it_per_sec),
            }
            payload.update(_json_value(telemetry))
            _atomic_write_json(self.live_metrics_path, payload)

    def write_epoch(
        self,
        epoch: int,
        train_metrics: Mapping[str, Any] | Any,
        valid_metrics: Mapping[str, Any] | Any,
        lr: float | None = None,
        throughput_examples_per_second: float | None = None,
    ) -> None:
        train = self.dashboard_metrics(train_metrics)
        valid = self.dashboard_metrics(valid_metrics)
        epoch_metadata = {}
        if lr is not None:
            epoch_metadata["lr"] = float(lr)
            train["lr"] = float(lr)
        if throughput_examples_per_second is not None:
            epoch_metadata["throughput_examples_per_second"] = float(throughput_examples_per_second)
            train["throughput_examples_per_second"] = float(throughput_examples_per_second)
        row = self._history_row(epoch)
        row["train"] = train
        row["valid"] = valid
        self._write_history()
        self._append_metric(epoch, "train", train, epoch_metadata)
        self._append_metric(epoch, "valid", valid, epoch_metadata)
        self._append_log(f"Train Summary | Epoch {int(epoch)} | Loss={train['loss']:.4f} | Reco={train['reco']:.4f}")
        self._append_log(f"Valid Summary | Epoch {int(epoch)} | Loss={valid['loss']:.4f} | Reco={valid['reco']:.4f}")

    def write_test(self, epoch: int, metrics: Mapping[str, Any] | Any) -> None:
        test = self.dashboard_metrics(metrics)
        row = self._history_row(epoch)
        row["test"] = test
        self._write_history()
        self._append_metric(epoch, "test", test)
        self._append_log(f"Test Summary | Epoch {int(epoch)} | Loss={test['loss']:.4f} | Reco={test['reco']:.4f}")

    def write_terminal_status(self, status: str, payload: Mapping[str, Any] | None = None) -> bool:
        return FailureArtifactReporter(self.save_dir, rank=0).write_terminal_status(status, payload)

    def write_completed(self, last_epoch: int, test_metrics: Mapping[str, Any] | Any) -> bool:
        return self.write_terminal_status(
            "completed",
            {
                "rank": 0,
                "last_epoch": int(last_epoch),
                "test_metrics": self.dashboard_metrics(test_metrics),
            },
        )

    def write_failure(self, error: BaseException, traceback_text: str) -> None:
        FailureArtifactReporter(self.save_dir, rank=0).write_failure(error, traceback_text)
        self._append_log(f"Failure | {type(error).__name__} | {error}")
