"""Resumable, non-interrupting V3 generation-to-training handoff."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Iterable

import soundfile as sf

from dscnn_kws.data.v3_handoff import RoleSpec, probe_v3_role_manifests, publish_role_pack


ERROR_MARKERS = ("Traceback", "Exception", "ERROR")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _metadata_count(paths: Iterable[Path]) -> int:
    count = 0
    for path in paths:
        with path.open("rb") as handle:
            count += sum(1 for _ in handle)
    return count


def renderer_is_live(pid: int) -> bool:
    try:
        if Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8").split()[2] == "Z":
            return False
    except (FileNotFoundError, OSError, IndexError):
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def normalize_false_wake_requests(source: Path, destination: Path) -> tuple[Path, int]:
    """Convert request offsets from native source rates to the 16 kHz render rate."""
    rows: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        info = sf.info(Path(str(row["foreground_path"])))
        start = int(row["foreground_start_sample"])
        start_16k = round(start * 16_000 / info.samplerate)
        length_16k = round(info.frames * 16_000 / info.samplerate)
        if start_16k < 0 or start_16k + 16_000 > length_16k:
            continue
        row["foreground_start_sample"] = start_16k
        row["source_sample_rate"] = int(info.samplerate)
        rows.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    os.replace(temporary, destination)
    return destination, len(rows)


def _terminal_log_is_clean(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    if any(marker in text for marker in ERROR_MARKERS):
        return False
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    try:
        return isinstance(json.loads(lines[-1]), dict)
    except json.JSONDecodeError:
        return False


def renderer_complete(
    metadata_paths: tuple[Path, ...],
    *,
    expected_records: int,
    pids: tuple[int, ...],
    log_paths: tuple[Path, ...],
) -> bool:
    return (
        _metadata_count(metadata_paths) == expected_records
        and not any(renderer_is_live(pid) for pid in pids)
        and len(log_paths) == len(pids)
        and all(_terminal_log_is_clean(path) for path in log_paths)
    )


@dataclass(frozen=True)
class HandoffConfig:
    source_root: Path
    run_root: Path
    base_role_root: Path
    base_expected_records: int
    false_wake_requests: Path
    false_expected_records: int
    validation_manifest: Path
    test_manifest: Path
    train_manifest: Path
    python: str
    poll_seconds: int = 60

    @property
    def handoff_root(self) -> Path:
        return self.run_root / "handoff"

    @property
    def state_path(self) -> Path:
        return self.handoff_root / "state.json"

    @property
    def false_role_root(self) -> Path:
        return self.run_root / "roles" / "false_wake_hard_negative"


def load_config(path: Path) -> HandoffConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return HandoffConfig(
        source_root=Path(payload["source_root"]).resolve(),
        run_root=Path(payload["run_root"]).resolve(),
        base_role_root=Path(payload["base_role_root"]).resolve(),
        base_expected_records=int(payload["base_expected_records"]),
        false_wake_requests=Path(payload["false_wake_requests"]).resolve(),
        false_expected_records=int(payload["false_expected_records"]),
        validation_manifest=Path(payload["validation_manifest"]).resolve(),
        test_manifest=Path(payload["test_manifest"]).resolve(),
        train_manifest=Path(payload["train_manifest"]).resolve(),
        python=str(payload["python"]),
        poll_seconds=int(payload.get("poll_seconds", 60)),
    )


def _read_pids(path: Path) -> tuple[int, ...]:
    return tuple(int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _metadata_paths(root: Path) -> tuple[Path, ...]:
    return tuple(sorted((root / "rendered" / "metadata").glob("*.jsonl")))


def _base_is_complete(config: HandoffConfig) -> bool:
    pids = _read_pids(config.base_role_root / "render_time_placement_20260902.pids")
    logs = tuple(sorted(config.base_role_root.glob("rank-*.time-placement_20260902.stdout.log")))
    return renderer_complete(_metadata_paths(config.base_role_root), expected_records=config.base_expected_records, pids=pids, log_paths=logs)


def _false_is_complete(config: HandoffConfig, state: dict[str, object]) -> bool:
    raw_pids = state.get("false_renderer_pids", [])
    pids = tuple(int(pid) for pid in raw_pids) if isinstance(raw_pids, list) else ()
    logs = tuple(config.false_role_root / f"rank-{rank}.stdout.log" for rank in range(4))
    return renderer_complete(_metadata_paths(config.false_role_root), expected_records=config.false_expected_records, pids=pids, log_paths=logs)


def _start_false_renderer(config: HandoffConfig) -> list[int]:
    role_root = config.false_role_root
    role_root.mkdir(parents=True, exist_ok=True)
    normalized_requests, normalized_count = normalize_false_wake_requests(
        config.false_wake_requests,
        config.handoff_root / "false_wake_hard_negative_16k.requests.jsonl",
    )
    if normalized_count < 1:
        raise RuntimeError("No complete false-wake windows remain after 16 kHz normalization")
    request_link = role_root / "requests.jsonl"
    if request_link.exists() or request_link.is_symlink():
        if request_link.resolve() != normalized_requests:
            raise RuntimeError(f"False-wake request link points to an unexpected file: {request_link}")
    else:
        request_link.symlink_to(normalized_requests)
    renderer = config.run_root / "pilots" / "all_roles_four_gpu_20260831_batched_sync" / "render_pilot.py"
    if not renderer.is_file():
        raise FileNotFoundError(renderer)
    pids: list[int] = []
    for rank in range(4):
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONPATH": str(config.source_root),
                "CUDA_VISIBLE_DEVICES": str(rank),
                "RANK": str(rank),
                "WORLD_SIZE": "4",
                "V3_PILOT_ROOT": str(role_root),
            }
        )
        log_path = role_root / f"rank-{rank}.stdout.log"
        handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            [config.python, str(renderer)],
            cwd=config.source_root,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        handle.close()
        pids.append(process.pid)
    return pids


def _archive_failed_false_attempt(config: HandoffConfig) -> None:
    if not config.false_role_root.exists():
        return
    archive = config.run_root / "roles" / f"false_wake_hard_negative.failed-{int(time.time())}"
    os.replace(config.false_role_root, archive)


def _role_specs(config: HandoffConfig) -> tuple[RoleSpec, ...]:
    role_root = config.run_root / "roles"
    return (
        RoleSpec("base_positive", "positive", _metadata_paths(role_root / "base_positive"), config.run_root / "packed_v3" / "base_positive", 1_047_600),
        RoleSpec("base_negative", "negative", _metadata_paths(role_root / "base_negative"), config.run_root / "packed_v3" / "base_negative", config.base_expected_records),
        RoleSpec("false_wake_hard_negative", "negative", _metadata_paths(config.false_role_root), config.run_root / "packed_v3" / "false_wake_hard_negative", config.false_expected_records),
        RoleSpec("captured_environment_negative", "negative", tuple(sorted((role_root / "captured_environment_negative" / "metadata").glob("*.jsonl"))), config.run_root / "packed_v3" / "captured_environment_negative"),
        RoleSpec("tau_environment_negative", "negative", tuple(sorted((role_root / "tau_environment_negative" / "metadata_repaired").glob("*.jsonl"))), config.run_root / "packed_v3" / "tau_environment_negative"),
    )


def _pack_and_probe(config: HandoffConfig) -> dict[str, str]:
    manifests = {spec.name: str(publish_role_pack(spec).manifest_path) for spec in _role_specs(config)}
    manifests["raw_positive"] = str(config.run_root / "roles" / "raw_positive.reference.json")
    manifests["raw_negative"] = "/home/chensheng/vad_kws_datasets/kws_reclean_aug_v1/v2/raw_anchors_pcm16/manifest.json"
    probe_v3_role_manifests({name: Path(path) for name, path in manifests.items()})
    return manifests


def _training_command(config: HandoffConfig, manifests: dict[str, str]) -> tuple[list[str], Path]:
    training_root = config.run_root / "training" / "v3_packed_mixture"
    command = [
        config.python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
        "-m",
        "dscnn_kws.train",
        "--distributed",
        "--model",
        "dscnn",
        "--epoch",
        "60",
        "--save_dir",
        str(training_root),
        "--run_name",
        "v3_packed_mixture",
        "--root",
        str(config.train_manifest.parent.parent),
        "--dataset",
        config.train_manifest.parent.parent.name,
        "--train_manifest",
        str(config.train_manifest),
        "--validation_manifest",
        str(config.validation_manifest),
        "--test_manifest",
        str(config.test_manifest),
        "--sample_rate",
        "16000",
        "--model_size_info",
        "5",
        "64",
        "10",
        "4",
        "2",
        "2",
        "64",
        "3",
        "3",
        "1",
        "1",
        "64",
        "3",
        "3",
        "1",
        "1",
        "64",
        "3",
        "3",
        "1",
        "1",
        "64",
        "3",
        "3",
        "1",
        "1",
        "64",
        "3",
        "3",
        "1",
        "1",
        "--batch",
        "8200",
        "--num_workers",
        "1",
        "--prefetch_factor",
        "1",
        "--opt",
        "adamw",
        "--lr",
        "0.004",
        "--weight_decay",
        "1e-5",
        "--scheduler",
        "cos",
        "--eta_min",
        "4e-5",
        "--warmup_steps",
        "200",
        "--mixture_steps_per_epoch",
        "64",
        "--mixture-v3-allow-replacement-role",
        "false_wake_hard_negative",
        "--mixture-v3-allow-replacement-role",
        "captured_environment_negative",
        "--offline_augmented_dataset",
        "--no-noise_aug",
        "--no-spec_aug",
        "--ddp_static_graph",
        "--ddp_gradient_as_bucket_view",
        "--live_telemetry",
    ]
    for name in (
        "base_positive",
        "raw_positive",
        "base_negative",
        "raw_negative",
        "false_wake_hard_negative",
        "captured_environment_negative",
        "tau_environment_negative",
    ):
        command.extend(("--mixture-v3-role", f"{name}={manifests[name]}"))
    return command, training_root


def _state(config: HandoffConfig) -> dict[str, object]:
    if config.state_path.is_file():
        return json.loads(config.state_path.read_text(encoding="utf-8"))
    return {"stage": "waiting_for_base", "created_at_utc": datetime.now(timezone.utc).isoformat()}


def _save_state(config: HandoffConfig, state: dict[str, object]) -> dict[str, object]:
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(config.state_path, state)
    return state


def advance(config: HandoffConfig) -> dict[str, object]:
    state = _state(config)
    stage = str(state.get("stage"))
    if stage == "waiting_for_base":
        if not _base_is_complete(config):
            return _save_state(config, state)
        state["stage"] = "rendering_false_wake"
        state["false_renderer_pids"] = _start_false_renderer(config)
        return _save_state(config, state)
    if stage == "rendering_false_wake":
        if not _false_is_complete(config, state):
            raw_pids = state.get("false_renderer_pids", [])
            pids = tuple(int(pid) for pid in raw_pids) if isinstance(raw_pids, list) else ()
            if pids and not any(renderer_is_live(pid) for pid in pids):
                retries = int(state.get("false_retry_count", 0))
                if retries >= 2:
                    raise RuntimeError("False-wake renderer exited before producing the expected records twice")
                _archive_failed_false_attempt(config)
                state["false_retry_count"] = retries + 1
                state["false_renderer_pids"] = _start_false_renderer(config)
            return _save_state(config, state)
        state["stage"] = "packing"
        return _save_state(config, state)
    if stage == "packing":
        manifests = _pack_and_probe(config)
        state["role_manifests"] = manifests
        state["stage"] = "starting_training"
        return _save_state(config, state)
    if stage == "starting_training":
        manifests = state.get("role_manifests")
        if not isinstance(manifests, dict):
            raise RuntimeError("Missing published role manifests")
        command, training_root = _training_command(config, {str(name): str(path) for name, path in manifests.items()})
        training_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(training_root / "train_command.json", {"command": command})
        environment = os.environ.copy()
        environment.update({"PYTHONPATH": str(config.source_root), "CUDA_VISIBLE_DEVICES": "0,1,2,3"})
        handle = (training_root / "train.log").open("a", encoding="utf-8")
        process = subprocess.Popen(command, cwd=config.source_root, env=environment, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        handle.close()
        state.update({"stage": "training", "training_pid": process.pid, "training_dir": str(training_root), "training_command": command})
        return _save_state(config, state)
    return _save_state(config, state)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config.resolve())
    while True:
        try:
            state = advance(config)
        except Exception as error:
            state = _state(config)
            state.update({"stage": "failed", "error": str(error), "traceback": traceback.format_exc()})
            _save_state(config, state)
            return 1
        if args.once or state.get("stage") in {"failed", "succeeded"}:
            return 0 if state.get("stage") != "failed" else 1
        time.sleep(max(5, config.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
