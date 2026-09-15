"""Launch the accuracy-first recleaned KWS training run on four GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dscnn_kws.data.build_reclean_training_manifests import (  # noqa: E402
    TrainingManifestPaths,
    build_reclean_training_manifests,
    resolve_current_training_manifests,
)
from dscnn_kws.training_selection import select_candidate  # noqa: E402


DEFAULT_DASHBOARD_XPS_ROOT = Path("/home/chensheng/mss_demucs/demucs/outputs/xps")
MODEL_SIZE_INFO = (5, 64, 10, 4, 2, 2, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1)
PROBE_THROUGHPUT_PATTERN = re.compile(r"\[PROBE\].*?throughput_examples_per_second=([0-9.eE+-]+)")
PROBE_WORLD_SIZE = 4
PROBE_MAX_BATCH_SIZE = 512
PROBE_EVALUATION_ROWS = 1
PACKED_NUM_WORKERS = 1
V2_LR_CANDIDATES = (8e-4, 1.2e-3, 1.6e-3)
V2_BATCH_SIZE_PER_RANK = 4_000
V2_WARMUP_STEPS = 800
V2_MIN_POSITIVE_RECALL = 0.93
V2_CANDIDATE_EPOCHS = 12
V2_FINAL_EPOCHS = 45

Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class RecleanTrainingConfig:
    corpus_root: Path
    run_dir: Path
    run_name: str
    manifests: TrainingManifestPaths
    batch_size: int = 256
    num_workers: int = 4
    amp_backbone: bool = False
    ddp_static_graph: bool = True
    ddp_gradient_as_bucket_view: bool = True
    packed_train_index: Path | None = None
    packed_block_records: int = 16_384
    warmup_steps: int = 0


@dataclass(frozen=True)
class RecleanV2Config:
    """Immutable inputs for one class-balanced V2 four-GPU run."""

    corpus_root: Path
    run_dir: Path
    run_name: str
    train_manifest: Path
    validation_manifest: Path
    base_pack: Path
    raw_anchor_pack: Path
    hard_negative_pack: Path
    steps_per_epoch: int
    lr: float


@dataclass(frozen=True)
class ProbeResult:
    batch_size: int
    num_workers: int
    returncode: int
    elapsed_seconds: float
    throughput_examples_per_second: float | None
    finite: bool
    command: tuple[str, ...]
    stdout_tail: str
    stderr_tail: str


def probe_candidates() -> tuple[tuple[int, int], ...]:
    """Return the quality-safe per-rank batch/worker candidates."""
    return tuple((batch_size, num_workers) for batch_size in (128, 256, 512) for num_workers in (2, 3, 4))


def build_training_command(
    config: RecleanTrainingConfig,
    *,
    max_train_steps: int | None = None,
) -> list[str]:
    """Build a shell-free four-rank command for one fixed training configuration."""
    packed_input = config.packed_train_index is not None
    num_workers = PACKED_NUM_WORKERS if packed_input else config.num_workers
    learning_rate = "0.004" if packed_input else "0.001"
    weight_decay = "1e-05" if packed_input else "1e-06"
    eta_min = "4e-05" if packed_input else "1e-05"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
        "-m",
        "dscnn_kws.train",
        "--distributed",
        "--root",
        str(config.corpus_root.parent),
        "--dataset",
        config.corpus_root.name,
        "--save_dir",
        str(config.run_dir),
        "--run_name",
        config.run_name,
        "--epoch",
        "60",
        "--opt",
        "adamw" if packed_input else "adam",
        "--lr",
        learning_rate,
        "--weight_decay",
        weight_decay,
        "--scheduler",
        "cos",
        "--eta_min",
        eta_min,
        "--batch",
        str(config.batch_size),
        "--num_workers",
        str(num_workers),
        "--prefetch_factor",
        "1" if packed_input else "4",
        "--log_interval",
        "1" if packed_input else "50",
        "--sample_rate",
        "16000",
        "--model_size_info",
        *(str(value) for value in MODEL_SIZE_INFO),
        "--offline_augmented_dataset",
        "--train_manifest",
        str(config.manifests.train_manifest_path),
        "--validation_manifest",
        str(config.manifests.validation_manifest_path),
        "--test_manifest",
        str(config.manifests.test_manifest_path),
        "--no-noise_aug",
        "--no-spec_aug",
        "--label_smoothing",
        "0.0",
    ]
    if packed_input:
        command.extend(
            (
                "--packed_train_index",
                str(config.packed_train_index),
                "--packed_block_records",
                str(config.packed_block_records),
                "--warmup_steps",
                str(config.warmup_steps),
                "--live_telemetry",
            )
        )
    if config.ddp_static_graph:
        command.append("--ddp_static_graph")
    if config.ddp_gradient_as_bucket_view:
        command.append("--ddp_gradient_as_bucket_view")
    if config.amp_backbone:
        command.append("--amp_backbone")
    if max_train_steps is not None:
        if max_train_steps < 1:
            raise ValueError("max_train_steps must be positive")
        command.extend(("--max_train_steps", str(max_train_steps), "--no-verify_sample_rate"))
    return command


def _validate_v2_values(*, steps_per_epoch: int, learning_rate: float, candidate: bool) -> None:
    if isinstance(steps_per_epoch, bool) or not isinstance(steps_per_epoch, int) or steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning rate must be finite and positive")
    if candidate and learning_rate not in V2_LR_CANDIDATES:
        raise ValueError(f"candidate learning rate must be one of {V2_LR_CANDIDATES}")
    epochs = V2_CANDIDATE_EPOCHS if candidate else V2_FINAL_EPOCHS
    minimum_steps = V2_WARMUP_STEPS // epochs + 1
    if steps_per_epoch < minimum_steps:
        run_kind = "candidate" if candidate else "final"
        raise ValueError(
            f"{run_kind} steps_per_epoch must be at least {minimum_steps} so "
            f"{epochs} epochs exceed {V2_WARMUP_STEPS} warmup steps"
        )


def _validate_v2_config(config: RecleanV2Config, *, candidate: bool) -> None:
    _validate_v2_values(
        steps_per_epoch=config.steps_per_epoch,
        learning_rate=config.lr,
        candidate=candidate,
    )


def _build_v2_command(
    config: RecleanV2Config,
    *,
    epochs: int,
    early_stopping: bool,
    max_train_steps: int | None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
        "-m",
        "dscnn_kws.train",
        "--distributed",
        "--root",
        str(config.corpus_root.parent),
        "--dataset",
        config.corpus_root.name,
        "--save_dir",
        str(config.run_dir),
        "--run_name",
        config.run_name,
        "--epoch",
        str(epochs),
        "--opt",
        "adamw",
        "--lr",
        str(config.lr),
        "--weight_decay",
        "1e-5",
        "--scheduler",
        "cos",
        "--eta_min",
        "4e-05",
        "--batch",
        str(V2_BATCH_SIZE_PER_RANK),
        "--num_workers",
        "1",
        "--prefetch_factor",
        "1",
        "--log_interval",
        "1",
        "--sample_rate",
        "16000",
        "--model_size_info",
        *(str(value) for value in MODEL_SIZE_INFO),
        "--offline_augmented_dataset",
        "--train_manifest",
        str(config.train_manifest),
        "--validation_manifest",
        str(config.validation_manifest),
        "--no-noise_aug",
        "--no-spec_aug",
        "--label_smoothing",
        "0.0",
        "--warmup_steps",
        str(V2_WARMUP_STEPS),
        "--min_positive_recall",
        str(V2_MIN_POSITIVE_RECALL),
        "--skip_test",
        "--mixture_base_pack",
        str(config.base_pack),
        "--mixture_raw_anchor_pack",
        str(config.raw_anchor_pack),
        "--mixture_hard_negative_pack",
        str(config.hard_negative_pack),
        "--mixture_steps_per_epoch",
        str(config.steps_per_epoch),
        "--live_telemetry",
        "--ddp_static_graph",
        "--ddp_gradient_as_bucket_view",
    ]
    if early_stopping:
        command.extend(("--early_stopping_min_epoch", "20", "--early_stopping_patience", "8"))
    if max_train_steps is not None:
        if max_train_steps < 1:
            raise ValueError("max_train_steps must be positive")
        command.extend(("--max_train_steps", str(max_train_steps), "--no-verify_sample_rate"))
    return command


def build_v2_candidate_command(
    config: RecleanV2Config,
    *,
    max_train_steps: int | None = None,
) -> list[str]:
    """Build one validation-only candidate command with the fixed V2 policy."""
    _validate_v2_config(config, candidate=True)
    return _build_v2_command(
        config,
        epochs=V2_CANDIDATE_EPOCHS,
        early_stopping=False,
        max_train_steps=max_train_steps,
    )


def build_v2_final_command(
    config: RecleanV2Config,
    *,
    max_train_steps: int | None = None,
) -> list[str]:
    """Build the locked-test final command selected from validation-only trials."""
    _validate_v2_config(config, candidate=False)
    return _build_v2_command(
        config,
        epochs=V2_FINAL_EPOCHS,
        early_stopping=True,
        max_train_steps=max_train_steps,
    )


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _tail(value: str | None, limit: int = 4096) -> str:
    return (value or "")[-limit:]


def _file_tail(path: Path, limit: int = 4096) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - limit), os.SEEK_SET)
            return handle.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _throughput_from_output(stdout: str | None) -> float | None:
    matches = PROBE_THROUGHPUT_PATTERN.findall(stdout or "")
    if not matches:
        return None
    try:
        throughput = float(matches[-1])
    except ValueError:
        return None
    return throughput if math.isfinite(throughput) and throughput > 0.0 else None


def _worker_temporary_dir(run_dir: Path) -> Path:
    """Keep multiprocessing AF_UNIX sockets below their platform path-length limit."""
    resolved_run_dir = Path(run_dir).resolve()
    home_dir = Path.home().resolve()
    if resolved_run_dir.is_relative_to(home_dir):
        temporary_root = home_dir / ".tmp" / "kws"
    else:
        temporary_root = resolved_run_dir.parent / ".tmp" / "kws"
    identifier = hashlib.sha256(str(resolved_run_dir).encode("utf-8")).hexdigest()[:16]
    return temporary_root / identifier


def _run_command(
    command: Sequence[str],
    runner: Runner,
    *,
    run_dir: Path,
    stdout=None,
    stderr=None,
) -> subprocess.CompletedProcess[str]:
    temporary_dir = _worker_temporary_dir(Path(run_dir))
    temporary_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({"TMPDIR": str(temporary_dir), "TMP": str(temporary_dir), "TEMP": str(temporary_dir)})
    kwargs = {
        "cwd": str(PROJECT_ROOT),
        "check": False,
        "text": True,
        "env": environment,
    }
    if stdout is None and stderr is None:
        kwargs["capture_output"] = True
    else:
        kwargs["stdout"] = stdout
        kwargs["stderr"] = stderr
    return runner(list(command), **kwargs)


def _copy_manifest_prefix(source: Path, destination: Path, row_limit: int) -> int:
    if row_limit < 1:
        raise ValueError("row_limit must be positive")
    if not source.is_file():
        raise FileNotFoundError(f"Manifest is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary_path = Path(temporary_name)
    count = 0
    try:
        with source.open("r", encoding="utf-8") as input_handle, os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as output_handle:
            for line in input_handle:
                if not line.strip():
                    continue
                output_handle.write(line)
                count += 1
                if count >= row_limit:
                    break
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if count == 0:
            raise ValueError(f"Manifest has no records: {source}")
        os.replace(temporary_path, destination)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
    return count


def build_probe_manifests(config: RecleanTrainingConfig, *, max_train_steps: int) -> TrainingManifestPaths:
    """Stream deterministic run-local prefixes so probes never parse the full corpus."""
    if max_train_steps < 1:
        raise ValueError("max_train_steps must be positive")
    probe_root = config.run_dir / "probe_manifests"
    paths = TrainingManifestPaths(
        train_manifest_path=probe_root / "train_manifest.json",
        validation_manifest_path=probe_root / "validation_manifest.json",
        test_manifest_path=probe_root / "test_manifest.json",
    )
    train_limit = PROBE_WORLD_SIZE * PROBE_MAX_BATCH_SIZE * max_train_steps
    counts = {
        "train": _copy_manifest_prefix(config.manifests.train_manifest_path, paths.train_manifest_path, train_limit),
        "validation": _copy_manifest_prefix(
            config.manifests.validation_manifest_path,
            paths.validation_manifest_path,
            PROBE_EVALUATION_ROWS,
        ),
        "test": _copy_manifest_prefix(config.manifests.test_manifest_path, paths.test_manifest_path, PROBE_EVALUATION_ROWS),
    }
    _atomic_write_json(
        probe_root / "manifest_counts.json",
        {
            "max_train_steps": max_train_steps,
            "train_row_limit": train_limit,
            "counts": counts,
            "source_manifests": {
                "train": str(config.manifests.train_manifest_path),
                "validation": str(config.manifests.validation_manifest_path),
                "test": str(config.manifests.test_manifest_path),
            },
        },
    )
    return paths


def _probe_result(
    config: RecleanTrainingConfig,
    *,
    max_train_steps: int,
    runner: Runner,
) -> ProbeResult:
    command = build_training_command(config, max_train_steps=max_train_steps)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    log_path = config.run_dir / "launcher.log"
    with log_path.open("a", encoding="utf-8", newline="\n") as log_handle:
        completed = _run_command(
            command,
            runner,
            run_dir=config.run_dir,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    elapsed_seconds = max(time.perf_counter() - started, 0.0)
    log_tail = _file_tail(log_path)
    throughput = _throughput_from_output(log_tail)
    finite = completed.returncode == 0 and throughput is not None
    return ProbeResult(
        batch_size=config.batch_size,
        num_workers=PACKED_NUM_WORKERS if config.packed_train_index is not None else config.num_workers,
        returncode=int(completed.returncode),
        elapsed_seconds=elapsed_seconds,
        throughput_examples_per_second=throughput,
        finite=finite,
        command=tuple(command),
        stdout_tail=log_tail,
        stderr_tail="",
    )


def _probe_result_json(result: ProbeResult) -> dict[str, object]:
    payload = asdict(result)
    payload["command"] = list(result.command)
    return payload


def run_throughput_probe(
    config: RecleanTrainingConfig,
    *,
    max_train_steps: int,
    runner: Runner = subprocess.run,
) -> ProbeResult:
    """Run every bounded candidate and persist the fastest finite selection."""
    if max_train_steps < 1:
        raise ValueError("max_train_steps must be positive")
    config.run_dir.mkdir(parents=True, exist_ok=True)
    probe_manifests = build_probe_manifests(config, max_train_steps=max_train_steps)
    results: list[ProbeResult] = []
    for batch_size, num_workers in probe_candidates():
        candidate = replace(
            config,
            run_dir=config.run_dir / "probes" / f"batch{batch_size}_workers{num_workers}",
            manifests=probe_manifests,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        results.append(_probe_result(candidate, max_train_steps=max_train_steps, runner=runner))

    finite_results = [result for result in results if result.finite]
    payload: dict[str, object] = {
        "max_train_steps": max_train_steps,
        "amp_backbone": config.amp_backbone,
        "results": [_probe_result_json(result) for result in results],
    }
    if not finite_results:
        payload["selected"] = None
        _atomic_write_json(config.run_dir / "probe_results.json", payload)
        raise RuntimeError("No finite four-GPU throughput probe candidate completed successfully")
    selected = max(finite_results, key=lambda result: result.throughput_examples_per_second or float("-inf"))
    payload["selected"] = {"batch_size": selected.batch_size, "num_workers": selected.num_workers}
    _atomic_write_json(config.run_dir / "probe_results.json", payload)
    return selected


def run_amp_smoke(
    config: RecleanTrainingConfig,
    *,
    runner: Runner = subprocess.run,
) -> RecleanTrainingConfig:
    """Keep BF16 only when a four-rank one-step smoke probe is finite."""
    if not config.amp_backbone:
        return config
    probe_manifests = build_probe_manifests(config, max_train_steps=1)
    smoke_config = replace(
        config,
        run_dir=config.run_dir / "amp_smoke",
        manifests=probe_manifests,
        batch_size=128,
        num_workers=PACKED_NUM_WORKERS if config.packed_train_index is not None else 2,
    )
    result = _probe_result(smoke_config, max_train_steps=1, runner=runner)
    _atomic_write_json(
        config.run_dir / "amp_smoke.json",
        {
            "requested_amp_backbone": True,
            "finite": result.finite,
            "result": _probe_result_json(result),
            "selected_amp_backbone": result.finite,
        },
    )
    return config if result.finite else replace(config, amp_backbone=False)


def _reproducibility_payload(config: RecleanTrainingConfig) -> dict[str, object]:
    packed_input = config.packed_train_index is not None
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_root": str(config.corpus_root),
        "run_name": config.run_name,
        "run_dir": str(config.run_dir),
        "manifests": {
            "train": str(config.manifests.train_manifest_path),
            "validation": str(config.manifests.validation_manifest_path),
            "test": str(config.manifests.test_manifest_path),
        },
        "accuracy_policy": {
            "epochs": 60,
            "optimizer": "adamw" if packed_input else "adam",
            "lr": 4e-3 if packed_input else 1e-3,
            "weight_decay": 1e-5 if packed_input else 1e-6,
            "scheduler": "cosine",
            "eta_min": 4e-5 if packed_input else 1e-5,
            "warmup_steps": config.warmup_steps,
            "sample_rate": 16000,
            "model_size_info": list(MODEL_SIZE_INFO),
            "offline_augmented_dataset": True,
            "noise_augmentation": False,
            "spec_augmentation": False,
            "label_smoothing": 0.0,
        },
        "ddp": {
            "world_size": 4,
            "static_graph": config.ddp_static_graph,
            "gradient_as_bucket_view": config.ddp_gradient_as_bucket_view,
            "amp_backbone_requested": config.amp_backbone,
        },
        "packed_input": {
            "index": str(config.packed_train_index) if packed_input else None,
            "block_records": config.packed_block_records if packed_input else None,
            "batch_size_per_rank": config.batch_size if packed_input else None,
            "num_workers_per_rank": PACKED_NUM_WORKERS if packed_input else None,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": str(PROJECT_ROOT),
        },
    }


def _new_run_name() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"kws_reclean_l5_c64_16k_{timestamp}_{uuid.uuid4().hex[:8]}"


def _resolve_run_dir(run_root: Path, run_name: str | None) -> tuple[str, Path]:
    root = Path(run_root).expanduser().resolve()
    if run_name is None:
        resolved_name = _new_run_name()
    else:
        if not isinstance(run_name, str) or not run_name.strip():
            raise ValueError("run name must be a non-empty relative path below run_root")
        raw_name = Path(run_name)
        if raw_name.is_absolute() or PurePosixPath(run_name).is_absolute() or PureWindowsPath(run_name).is_absolute():
            raise ValueError("run name must be a relative path below run_root")
        resolved_name = run_name
    run_dir = (root / resolved_name).resolve()
    if run_dir == root or not run_dir.is_relative_to(root):
        raise ValueError("run name must resolve to a strict descendant of run_root")
    return resolved_name, run_dir


def prepare_run(
    corpus_root: Path,
    run_root: Path,
    *,
    run_name: str | None = None,
    amp_backbone: bool = False,
) -> RecleanTrainingConfig:
    """Build manifests once and reserve one unique local run directory."""
    corpus_root = Path(corpus_root).expanduser().resolve()
    run_root = Path(run_root).expanduser().resolve()
    resolved_name, run_dir = _resolve_run_dir(run_root, run_name)
    manifests = build_reclean_training_manifests(corpus_root)
    run_dir.mkdir(parents=True, exist_ok=False)
    config = RecleanTrainingConfig(
        corpus_root=corpus_root,
        run_dir=run_dir,
        run_name=resolved_name,
        manifests=manifests,
        amp_backbone=amp_backbone,
    )
    _atomic_write_json(run_dir / "launcher_config.json", _reproducibility_payload(config))
    return config


def _v2_reproducibility_payload(config: RecleanV2Config) -> dict[str, object]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_root": str(config.corpus_root),
        "run_name": config.run_name,
        "run_dir": str(config.run_dir),
        "manifests": {
            "train": str(config.train_manifest),
            "validation": str(config.validation_manifest),
        },
        "accuracy_policy": {
            "optimizer": "adamw",
            "lr": config.lr,
            "weight_decay": 1e-5,
            "warmup_steps": V2_WARMUP_STEPS,
            "batch_size_per_rank": V2_BATCH_SIZE_PER_RANK,
            "min_positive_recall": V2_MIN_POSITIVE_RECALL,
            "test_locked": True,
        },
        "mixture": {
            "base_pack": str(config.base_pack),
            "raw_anchor_pack": str(config.raw_anchor_pack),
            "hard_negative_pack": str(config.hard_negative_pack),
            "steps_per_epoch": config.steps_per_epoch,
        },
        "ddp": {"world_size": 4, "static_graph": True, "gradient_as_bucket_view": True},
    }


def prepare_v2_run(
    *,
    corpus_root: Path,
    run_root: Path,
    run_name: str | None,
    train_manifest: Path,
    validation_manifest: Path,
    base_pack: Path,
    raw_anchor_pack: Path,
    hard_negative_pack: Path,
    steps_per_epoch: int,
    lr: float,
) -> RecleanV2Config:
    """Reserve a V2 run without resolving or recording any locked test source."""
    resolved_corpus_root = Path(corpus_root).expanduser().resolve()
    resolved_run_root = Path(run_root).expanduser().resolve()
    resolved_name, run_dir = _resolve_run_dir(resolved_run_root, run_name)
    run_dir.mkdir(parents=True, exist_ok=False)
    config = RecleanV2Config(
        corpus_root=resolved_corpus_root,
        run_dir=run_dir,
        run_name=resolved_name,
        train_manifest=Path(train_manifest).expanduser().resolve(),
        validation_manifest=Path(validation_manifest).expanduser().resolve(),
        base_pack=Path(base_pack).expanduser().resolve(),
        raw_anchor_pack=Path(raw_anchor_pack).expanduser().resolve(),
        hard_negative_pack=Path(hard_negative_pack).expanduser().resolve(),
        steps_per_epoch=int(steps_per_epoch),
        lr=float(lr),
    )
    _atomic_write_json(run_dir / "launcher_config.json", _v2_reproducibility_payload(config))
    return config


def create_dashboard_symlink(run_dir: Path, dashboard_xps_root: Path) -> Path | None:
    """Expose a validated run directory to the Linux dashboard without replacing links."""
    if platform.system() != "Linux":
        return None
    run_dir = Path(run_dir).resolve()
    dashboard_xps_root = Path(dashboard_xps_root).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if not dashboard_xps_root.is_dir():
        raise FileNotFoundError(f"Dashboard XPS root does not exist: {dashboard_xps_root}")
    link_path = dashboard_xps_root / run_dir.name
    if link_path.is_symlink():
        if link_path.resolve() == run_dir:
            return link_path
        raise FileExistsError(f"Dashboard link points at another run: {link_path}")
    if link_path.exists():
        raise FileExistsError(f"Dashboard path already exists: {link_path}")
    link_path.symlink_to(run_dir, target_is_directory=True)
    return link_path


def run_full_training(
    config: RecleanTrainingConfig,
    *,
    runner: Runner = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    """Launch the selected full run with the manifest generation bound at run creation."""
    command = build_training_command(config)
    _atomic_write_json(config.run_dir / "train_command.json", {"command": command})
    log_path = config.run_dir / "launcher.log"
    with log_path.open("a", encoding="utf-8", newline="\n") as log_handle:
        completed = _run_command(
            command,
            runner,
            run_dir=config.run_dir,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    _atomic_write_json(
        config.run_dir / "launcher_status.json",
        {
            "returncode": int(completed.returncode),
            "launcher_log": str(log_path),
            "log_tail": _file_tail(log_path),
        },
    )
    return completed


def run_v2_training(
    config: RecleanV2Config,
    *,
    candidate: bool,
    max_train_steps: int | None = None,
    runner: Runner = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    """Launch a V2 trial or final run while keeping the test split locked."""
    command = (
        build_v2_candidate_command(config, max_train_steps=max_train_steps)
        if candidate
        else build_v2_final_command(config, max_train_steps=max_train_steps)
    )
    _atomic_write_json(config.run_dir / "train_command.json", {"command": command})
    log_path = config.run_dir / "launcher.log"
    with log_path.open("a", encoding="utf-8", newline="\n") as log_handle:
        completed = _run_command(
            command,
            runner,
            run_dir=config.run_dir,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    _atomic_write_json(
        config.run_dir / "launcher_status.json",
        {
            "returncode": int(completed.returncode),
            "launcher_log": str(log_path),
            "log_tail": _file_tail(log_path),
        },
    )
    return completed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Four-GPU accuracy-first recleaned KWS launcher")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("probe", "train", "packed-probe", "train-packed"):
        subparser = subcommands.add_parser(name)
        subparser.add_argument("--corpus-root", type=Path, required=True)
        subparser.add_argument("--run-root", type=Path, required=True)
        subparser.add_argument("--run-name", default=None)
        subparser.add_argument("--probe-steps", type=int, default=20)
        subparser.add_argument("--amp-backbone", action=argparse.BooleanOptionalAction, default=False)
        subparser.add_argument("--dashboard-xps-root", type=Path, default=DEFAULT_DASHBOARD_XPS_ROOT)
        if name in ("packed-probe", "train-packed"):
            subparser.add_argument("--packed-train-index", type=Path, required=True)
            subparser.add_argument("--batch-size", type=int, default=8192)
            subparser.add_argument("--packed-block-records", type=int, default=16_384)
            subparser.add_argument("--warmup-steps", type=int, default=0 if name == "packed-probe" else 200)
        if name == "packed-probe":
            subparser.add_argument("--max-train-steps", type=int, default=1)

    def add_v2_training_arguments(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--corpus-root", type=Path, required=True)
        subparser.add_argument("--run-root", type=Path, required=True)
        subparser.add_argument("--run-name", default=None)
        subparser.add_argument("--train-manifest", type=Path, required=True)
        subparser.add_argument("--validation-manifest", type=Path, required=True)
        subparser.add_argument("--base-pack", type=Path, required=True)
        subparser.add_argument("--raw-anchor-pack", type=Path, required=True)
        subparser.add_argument("--hard-negative-pack", type=Path, required=True)
        subparser.add_argument("--steps-per-epoch", type=int, required=True)
        subparser.add_argument("--max-train-steps", type=int, default=None)
        subparser.add_argument("--dashboard-xps-root", type=Path, default=DEFAULT_DASHBOARD_XPS_ROOT)

    v2_candidate = subcommands.add_parser("v2-candidate")
    add_v2_training_arguments(v2_candidate)
    v2_candidate.add_argument("--lr", type=float, required=True, choices=V2_LR_CANDIDATES)

    v2_final = subcommands.add_parser("v2-final")
    add_v2_training_arguments(v2_final)
    v2_final.add_argument("--selection-report", type=Path, required=True)

    v2_select = subcommands.add_parser("v2-select")
    v2_select.add_argument("--run-dir", type=Path, nargs="+", required=True)
    v2_select.add_argument("--output", type=Path, required=True)
    v2_select.add_argument("--min-positive-recall", type=float, default=V2_MIN_POSITIVE_RECALL)
    return parser


def _validate_packed_arguments(args: argparse.Namespace) -> Path:
    packed_train_index = args.packed_train_index.expanduser().resolve()
    if not packed_train_index.is_file():
        raise FileNotFoundError("--packed-train-index must name an existing file")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.packed_block_records < 1:
        raise ValueError("--packed-block-records must be positive")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    return packed_train_index


def _selected_learning_rate(selection_report: Path) -> float:
    try:
        report = json.loads(Path(selection_report).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read candidate selection report: {selection_report}") from error
    if not isinstance(report, dict):
        raise ValueError("Candidate selection report must be a JSON object")
    try:
        learning_rate = float(report["selected_lr"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Candidate selection report has no selected learning rate") from error
    if not math.isfinite(learning_rate) or learning_rate not in V2_LR_CANDIDATES:
        raise ValueError(f"Candidate selection report selected unsupported learning rate: {learning_rate}")
    return learning_rate


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "v2-select":
        report = select_candidate(
            args.run_dir,
            min_positive_recall=args.min_positive_recall,
            output_path=args.output,
        )
        print(
            json.dumps(
                {
                    "selected_run": report.selected_run,
                    "selected_lr": report.selected_lr,
                    "selected_epoch": report.selected_epoch,
                    "report_path": str(Path(args.output).expanduser().resolve()),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command in {"v2-candidate", "v2-final"}:
        if args.max_train_steps is not None and args.max_train_steps < 1:
            raise ValueError("--max-train-steps must be positive")
        candidate = args.command == "v2-candidate"
        learning_rate = args.lr if candidate else _selected_learning_rate(args.selection_report)
        _validate_v2_values(
            steps_per_epoch=args.steps_per_epoch,
            learning_rate=learning_rate,
            candidate=candidate,
        )
        config = prepare_v2_run(
            corpus_root=args.corpus_root,
            run_root=args.run_root,
            run_name=args.run_name,
            train_manifest=args.train_manifest,
            validation_manifest=args.validation_manifest,
            base_pack=args.base_pack,
            raw_anchor_pack=args.raw_anchor_pack,
            hard_negative_pack=args.hard_negative_pack,
            steps_per_epoch=args.steps_per_epoch,
            lr=learning_rate,
        )
        create_dashboard_symlink(config.run_dir, args.dashboard_xps_root)
        completed = run_v2_training(
            config,
            candidate=candidate,
            max_train_steps=args.max_train_steps,
        )
        return int(completed.returncode)
    if args.probe_steps < 1:
        raise ValueError("--probe-steps must be positive")
    if getattr(args, "max_train_steps", 1) < 1:
        raise ValueError("--max-train-steps must be positive")
    packed_train_index = None
    if args.command in ("packed-probe", "train-packed"):
        packed_train_index = _validate_packed_arguments(args)
    config = prepare_run(
        args.corpus_root,
        args.run_root,
        run_name=args.run_name,
        amp_backbone=args.amp_backbone,
    )
    if args.command in ("packed-probe", "train-packed"):
        config = replace(
            config,
            batch_size=args.batch_size,
            num_workers=PACKED_NUM_WORKERS,
            packed_train_index=packed_train_index,
            packed_block_records=args.packed_block_records,
            warmup_steps=args.warmup_steps,
        )
        _atomic_write_json(config.run_dir / "launcher_config.json", _reproducibility_payload(config))
        config = run_amp_smoke(config)
        if args.command == "packed-probe":
            result = _probe_result(config, max_train_steps=args.max_train_steps, runner=subprocess.run)
            _atomic_write_json(config.run_dir / "packed_probe_result.json", _probe_result_json(result))
            if not result.finite:
                raise RuntimeError("Packed capacity probe did not complete with finite throughput")
            print(
                json.dumps(
                    {
                        "run_dir": str(config.run_dir),
                        "batch_size": result.batch_size,
                        "num_workers": result.num_workers,
                        "throughput_examples_per_second": result.throughput_examples_per_second,
                    }
                )
            )
            return 0

        create_dashboard_symlink(config.run_dir, args.dashboard_xps_root)
        completed = run_full_training(config)
        return int(completed.returncode)

    config = run_amp_smoke(config)
    selected = run_throughput_probe(config, max_train_steps=args.probe_steps)
    selected_config = replace(config, batch_size=selected.batch_size, num_workers=selected.num_workers)
    _atomic_write_json(
        config.run_dir / "selected_config.json",
        {
            "batch_size": selected.batch_size,
            "num_workers": selected.num_workers,
            "throughput_examples_per_second": selected.throughput_examples_per_second,
            "amp_backbone": selected_config.amp_backbone,
        },
    )
    if args.command == "probe":
        print(json.dumps({"run_dir": str(config.run_dir), "batch_size": selected.batch_size, "num_workers": selected.num_workers}))
        return 0

    create_dashboard_symlink(selected_config.run_dir, args.dashboard_xps_root)
    completed = run_full_training(selected_config)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
