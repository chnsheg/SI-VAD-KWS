from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dscnn_kws.data.build_reclean_training_manifests import TrainingManifestPaths
from dscnn_kws.engine.trainer import Trainer
import dscnn_kws.train as train_module


def _launcher_module():
    path = Path(__file__).resolve().parents[2] / "tools" / "run_kws_reclean_four_gpu.py"
    spec = importlib.util.spec_from_file_location("run_kws_reclean_four_gpu", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(launcher, tmp_path: Path, **overrides):
    manifests = TrainingManifestPaths(
        train_manifest_path=tmp_path / "manifests" / "train.jsonl",
        validation_manifest_path=tmp_path / "manifests" / "validation.jsonl",
        test_manifest_path=tmp_path / "manifests" / "test.jsonl",
    )
    values = {
        "corpus_root": tmp_path / "corpus",
        "run_dir": tmp_path / "runs" / "reclean-run",
        "run_name": "reclean-run",
        "manifests": manifests,
        "batch_size": 256,
        "num_workers": 3,
        "amp_backbone": False,
        "ddp_static_graph": True,
        "ddp_gradient_as_bucket_view": True,
    }
    values.update(overrides)
    return launcher.RecleanTrainingConfig(**values)


def _option(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def _write_manifest(path: Path, count: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"audio_filepath": f"/unused/{path.stem}-{index}.wav", "command": "positive" if index % 2 else "negative"}
        for index in range(count)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_build_training_command_uses_four_ranks_and_accuracy_first_defaults(tmp_path):
    launcher = _launcher_module()

    command = launcher.build_training_command(_config(launcher, tmp_path))

    assert command[:4] == [sys.executable, "-m", "torch.distributed.run", "--standalone"]
    assert "--nproc_per_node=4" in command
    assert _option(command, "--epoch") == "60"
    assert _option(command, "--lr") == "0.001"
    assert _option(command, "--weight_decay") == "1e-06"
    assert _option(command, "--eta_min") == "1e-05"
    assert _option(command, "--sample_rate") == "16000"
    assert _option(command, "--model_size_info") == "5"
    assert _option(command, "--label_smoothing") == "0.0"
    assert "--offline_augmented_dataset" in command
    assert "--no-noise_aug" in command
    assert "--no-spec_aug" in command
    assert "--ddp_static_graph" in command
    assert "--ddp_gradient_as_bucket_view" in command
    assert "--amp_backbone" not in command


def test_probe_candidates_are_the_bounded_batch_worker_cartesian_product():
    launcher = _launcher_module()

    candidates = tuple(launcher.probe_candidates())

    assert candidates == (
        (128, 2),
        (128, 3),
        (128, 4),
        (256, 2),
        (256, 3),
        (256, 4),
        (512, 2),
        (512, 3),
        (512, 4),
    )
    assert max(batch for batch, _ in candidates) == 512


def test_probe_streams_combined_output_to_its_log_and_parses_throughput(tmp_path):
    launcher = _launcher_module()
    config = _config(launcher, tmp_path)
    captured = {}

    def runner(command, **kwargs):
        captured["kwargs"] = kwargs
        kwargs["stdout"].write("[PROBE] finite=true throughput_examples_per_second=321.5\n")
        kwargs["stdout"].flush()
        return subprocess.CompletedProcess(command, 0)

    result = launcher._probe_result(config, max_train_steps=1, runner=runner)

    assert "capture_output" not in captured["kwargs"]
    assert captured["kwargs"]["stderr"] is subprocess.STDOUT
    assert captured["kwargs"]["stdout"].name == str(config.run_dir / "launcher.log")
    assert result.throughput_examples_per_second == pytest.approx(321.5)
    assert result.stdout_tail.endswith("throughput_examples_per_second=321.5\n")
    assert result.stderr_tail == ""


def test_packed_probe_result_records_the_effective_one_worker_configuration(tmp_path):
    launcher = _launcher_module()
    config = _config(
        launcher,
        tmp_path,
        num_workers=4,
        packed_train_index=tmp_path / "packed" / "manifest.json",
    )

    def runner(command, **kwargs):
        kwargs["stdout"].write("[PROBE] finite=true throughput_examples_per_second=321.5\n")
        kwargs["stdout"].flush()
        return subprocess.CompletedProcess(command, 0)

    result = launcher._probe_result(config, max_train_steps=1, runner=runner)

    assert _option(list(result.command), "--num_workers") == "1"
    assert result.num_workers == 1


def test_probe_runner_records_all_candidates_and_selects_fastest_finite_result(tmp_path):
    launcher = _launcher_module()
    config = _config(launcher, tmp_path)
    _write_manifest(config.manifests.train_manifest_path, 12)
    _write_manifest(config.manifests.validation_manifest_path, 3)
    _write_manifest(config.manifests.test_manifest_path, 4)
    launcher.resolve_current_training_manifests = lambda _corpus_root: config.manifests
    commands: list[list[str]] = []

    def runner(command, **kwargs):
        commands.append(command)
        batch = int(_option(command, "--batch"))
        workers = int(_option(command, "--num_workers"))
        throughput = batch * workers
        kwargs["stdout"].write(f"[PROBE] finite=true throughput_examples_per_second={throughput}.0\n")
        kwargs["stdout"].flush()
        return subprocess.CompletedProcess(command, 0)

    selected = launcher.run_throughput_probe(config, max_train_steps=3, runner=runner)

    assert (selected.batch_size, selected.num_workers) == (512, 4)
    assert selected.throughput_examples_per_second == pytest.approx(2048.0)
    assert len(commands) == 9
    assert all(_option(command, "--max_train_steps") == "3" for command in commands)
    results = json.loads((config.run_dir / "probe_results.json").read_text(encoding="utf-8"))
    assert results["selected"] == {"batch_size": 512, "num_workers": 4}
    assert len(results["results"]) == 9


def test_bounded_probe_uses_run_local_manifest_prefixes_and_full_command_uses_current_immutable_paths(tmp_path):
    launcher = _launcher_module()
    config = _config(launcher, tmp_path)
    _write_manifest(config.manifests.train_manifest_path, 12)
    _write_manifest(config.manifests.validation_manifest_path, 3)
    _write_manifest(config.manifests.test_manifest_path, 4)
    launcher.resolve_current_training_manifests = lambda _corpus_root: config.manifests
    commands: list[list[str]] = []

    def runner(command, **kwargs):
        commands.append(command)
        if "--max_train_steps" in command:
            kwargs["stdout"].write("[PROBE] finite=true throughput_examples_per_second=100.0\n")
            kwargs["stdout"].flush()
        return subprocess.CompletedProcess(command, 0)

    launcher.run_throughput_probe(config, max_train_steps=1, runner=runner)
    probe_root = config.run_dir / "probe_manifests"
    probe_train = probe_root / "train_manifest.json"
    probe_validation = probe_root / "validation_manifest.json"
    probe_test = probe_root / "test_manifest.json"

    assert len(probe_train.read_text(encoding="utf-8").splitlines()) == 12
    assert len(probe_validation.read_text(encoding="utf-8").splitlines()) == 1
    assert len(probe_test.read_text(encoding="utf-8").splitlines()) == 1
    assert all(_option(command, "--train_manifest") == str(probe_train) for command in commands)
    assert all(_option(command, "--validation_manifest") == str(probe_validation) for command in commands)
    assert all(_option(command, "--test_manifest") == str(probe_test) for command in commands)

    launcher.run_full_training(config, runner=runner)

    assert _option(commands[-1], "--train_manifest") == str(config.manifests.train_manifest_path)
    assert "--max_train_steps" not in commands[-1]


def test_prepare_run_binds_manifest_generation_returned_by_builder(tmp_path, monkeypatch):
    launcher = _launcher_module()
    corpus_root = tmp_path / "corpus"
    bound_manifests = TrainingManifestPaths(
        train_manifest_path=tmp_path / "bound" / "train.jsonl",
        validation_manifest_path=tmp_path / "bound" / "validation.jsonl",
        test_manifest_path=tmp_path / "bound" / "test.jsonl",
    )
    build_calls = []

    def build(root):
        build_calls.append(root)
        return bound_manifests

    monkeypatch.setattr(launcher, "build_reclean_training_manifests", build)
    monkeypatch.setattr(
        launcher,
        "resolve_current_training_manifests",
        lambda _root: pytest.fail("prepare_run must not re-read the mutable current manifest pointer"),
    )
    config = launcher.prepare_run(corpus_root, tmp_path / "runs", run_name="bound-run")
    commands = []

    def runner(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    launcher.run_full_training(config, runner=runner)

    assert build_calls == [corpus_root.resolve()]
    assert _option(commands[0], "--train_manifest") == str(bound_manifests.train_manifest_path)
    metadata = json.loads((config.run_dir / "launcher_config.json").read_text(encoding="utf-8"))
    assert metadata["manifests"]["train"] == str(bound_manifests.train_manifest_path)


def test_probe_manifests_are_copied_from_the_bound_generation_after_current_pointer_drifts(tmp_path, monkeypatch):
    launcher = _launcher_module()
    config = _config(launcher, tmp_path)
    _write_manifest(config.manifests.train_manifest_path, 12)
    _write_manifest(config.manifests.validation_manifest_path, 3)
    _write_manifest(config.manifests.test_manifest_path, 4)
    drifted_manifests = TrainingManifestPaths(
        train_manifest_path=_write_manifest(tmp_path / "drifted" / "train.jsonl", 2),
        validation_manifest_path=_write_manifest(tmp_path / "drifted" / "validation.jsonl", 2),
        test_manifest_path=_write_manifest(tmp_path / "drifted" / "test.jsonl", 2),
    )
    monkeypatch.setattr(launcher, "resolve_current_training_manifests", lambda _root: drifted_manifests)

    def runner(command, **kwargs):
        kwargs["stdout"].write("[PROBE] finite=true throughput_examples_per_second=100.0\n")
        kwargs["stdout"].flush()
        return subprocess.CompletedProcess(command, 0)

    launcher.run_throughput_probe(config, max_train_steps=1, runner=runner)

    manifest_counts = json.loads((config.run_dir / "probe_manifests" / "manifest_counts.json").read_text(encoding="utf-8"))
    assert manifest_counts["source_manifests"]["train"] == str(config.manifests.train_manifest_path)
    assert manifest_counts["counts"]["train"] == 12


def test_all_launcher_subprocesses_use_created_run_local_temp_environment(tmp_path, monkeypatch):
    launcher = _launcher_module()
    config = _config(launcher, tmp_path)
    _write_manifest(config.manifests.train_manifest_path, 12)
    _write_manifest(config.manifests.validation_manifest_path, 3)
    _write_manifest(config.manifests.test_manifest_path, 4)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    captures = []

    def runner(command, **kwargs):
        captures.append((command, kwargs))
        stdout = "[PROBE] finite=true throughput_examples_per_second=100.0\n" if "--max_train_steps" in command else ""
        kwargs["stdout"].write(stdout)
        kwargs["stdout"].flush()
        return subprocess.CompletedProcess(command, 0)

    launcher.run_throughput_probe(config, max_train_steps=1, runner=runner)
    launcher.run_amp_smoke(launcher.replace(config, amp_backbone=True), runner=runner)
    launcher.run_full_training(config, runner=runner)

    assert len(captures) == 11
    for command, kwargs in captures:
        environment = kwargs["env"]
        process_run_dir = Path(_option(command, "--save_dir"))
        temporary_dir = Path(environment["TMPDIR"])
        assert temporary_dir == launcher._worker_temporary_dir(process_run_dir)
        assert temporary_dir.is_dir()
        assert environment["TMP"] == environment["TEMP"] == environment["TMPDIR"]
        assert environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


def test_launcher_subprocesses_use_short_home_temp_paths_for_worker_sockets(tmp_path, monkeypatch):
    launcher = _launcher_module()
    monkeypatch.setattr(launcher.Path, "home", lambda: tmp_path)
    run_dir = tmp_path / "kws_training_runs" / "kws_reclean_l5_c64_16k_20260829_064319_5bfa42d7" / "amp_smoke"
    captured = {}

    def runner(command, **kwargs):
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0)

    launcher._run_command(["probe"], runner, run_dir=run_dir)

    temporary_dir = Path(captured["kwargs"]["env"]["TMPDIR"])
    assert temporary_dir.is_relative_to(tmp_path / ".tmp" / "kws")
    assert captured["kwargs"]["env"]["TMP"] == captured["kwargs"]["env"]["TEMP"] == str(temporary_dir)

    monkeypatch.setattr(launcher.Path, "home", lambda: Path("/home/chensheng"))
    remote_temporary_dir = launcher._worker_temporary_dir(
        Path("/home/chensheng/kws_training_runs/kws_reclean_l5_c64_16k_20260829_064319_5bfa42d7/amp_smoke")
    )
    assert len(str(remote_temporary_dir / "pymp-01234567" / "listener-0123456789")) < 100


@pytest.mark.parametrize(
    "unsafe_name_factory",
    (
        lambda tmp_path: str(tmp_path / "absolute-run"),
        lambda _tmp_path: "../outside-run",
        lambda _tmp_path: ".",
        lambda _tmp_path: "",
    ),
)
def test_prepare_run_rejects_names_outside_or_equal_to_run_root(tmp_path, monkeypatch, unsafe_name_factory):
    launcher = _launcher_module()
    build_calls = []
    monkeypatch.setattr(launcher, "build_reclean_training_manifests", lambda root: build_calls.append(root))
    monkeypatch.setattr(
        launcher,
        "resolve_current_training_manifests",
        lambda root: TrainingManifestPaths(root / "train.jsonl", root / "validation.jsonl", root / "test.jsonl"),
    )

    with pytest.raises(ValueError, match="run name"):
        launcher.prepare_run(tmp_path / "corpus", tmp_path / "runs", run_name=unsafe_name_factory(tmp_path))

    assert build_calls == []
    assert not (tmp_path / "runs").exists()


def test_prepare_run_keeps_a_simple_name_below_run_root(tmp_path, monkeypatch):
    launcher = _launcher_module()
    manifests = TrainingManifestPaths(
        tmp_path / "corpus" / "generation" / "train.jsonl",
        tmp_path / "corpus" / "generation" / "validation.jsonl",
        tmp_path / "corpus" / "generation" / "test.jsonl",
    )
    monkeypatch.setattr(launcher, "build_reclean_training_manifests", lambda root: manifests)

    config = launcher.prepare_run(tmp_path / "corpus", tmp_path / "runs", run_name="safe-run")

    assert config.run_dir == (tmp_path / "runs" / "safe-run").resolve()
    assert config.run_dir.is_relative_to((tmp_path / "runs").resolve())


def test_full_training_streams_combined_output_to_run_log_and_persists_tail(tmp_path):
    launcher = _launcher_module()
    config = _config(launcher, tmp_path)
    captured = {}

    def runner(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        kwargs["stdout"].write("rank-zero startup\nfinal combined output\n")
        kwargs["stdout"].flush()
        return subprocess.CompletedProcess(command, 7)

    completed = launcher.run_full_training(config, runner=runner)

    assert completed.returncode == 7
    assert "capture_output" not in captured["kwargs"]
    assert captured["kwargs"]["stderr"] is subprocess.STDOUT
    assert captured["kwargs"]["stdout"].name == str(config.run_dir / "launcher.log")
    assert (config.run_dir / "launcher.log").read_text(encoding="utf-8") == "rank-zero startup\nfinal combined output\n"
    status = json.loads((config.run_dir / "launcher_status.json").read_text(encoding="utf-8"))
    assert status["returncode"] == 7
    assert status["log_tail"].endswith("final combined output\n")


def test_packed_probe_cli_runs_exact_bounded_8192_configuration(tmp_path, monkeypatch):
    launcher = _launcher_module()
    config = _config(launcher, tmp_path)
    config.run_dir.mkdir(parents=True)
    packed_index = tmp_path / "packed" / "manifest.json"
    packed_index.parent.mkdir()
    packed_index.write_text("{}\n", encoding="utf-8")
    observed = {}

    monkeypatch.setattr(launcher, "prepare_run", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(launcher, "run_amp_smoke", lambda candidate: candidate)

    def run_probe(candidate, *, max_train_steps, runner=subprocess.run):
        observed["config"] = candidate
        observed["max_train_steps"] = max_train_steps
        return launcher.ProbeResult(
            batch_size=candidate.batch_size,
            num_workers=candidate.num_workers,
            returncode=0,
            elapsed_seconds=1.0,
            throughput_examples_per_second=65_536.0,
            finite=True,
            command=tuple(),
            stdout_tail="",
            stderr_tail="",
        )

    monkeypatch.setattr(launcher, "_probe_result", run_probe)

    result = launcher.main(
        [
            "packed-probe",
            "--corpus-root",
            str(tmp_path / "corpus"),
            "--run-root",
            str(tmp_path / "runs"),
            "--packed-train-index",
            str(packed_index),
            "--max-train-steps",
            "1",
        ]
    )

    assert result == 0
    assert observed["max_train_steps"] == 1
    assert observed["config"].packed_train_index == packed_index
    assert observed["config"].batch_size == 8192
    assert observed["config"].num_workers == 1
    assert observed["config"].warmup_steps == 0


def test_packed_training_enforces_one_worker_in_command_and_metadata(tmp_path):
    launcher = _launcher_module()
    config = _config(
        launcher,
        tmp_path,
        num_workers=4,
        packed_train_index=tmp_path / "packed" / "manifest.json",
    )

    command = launcher.build_training_command(config)
    metadata = launcher._reproducibility_payload(config)

    assert _option(command, "--num_workers") == "1"
    assert metadata["packed_input"]["num_workers_per_rank"] == 1


def test_packed_parser_does_not_expose_num_workers(tmp_path):
    launcher = _launcher_module()

    with pytest.raises(SystemExit):
        launcher.build_parser().parse_args(
            [
                "train-packed",
                "--corpus-root",
                str(tmp_path / "corpus"),
                "--run-root",
                str(tmp_path / "runs"),
                "--packed-train-index",
                str(tmp_path / "packed" / "manifest.json"),
                "--num-workers",
                "1",
            ]
        )


@pytest.mark.parametrize(
    ("index_kind", "option", "value", "exception", "message"),
    (
        ("missing", None, None, FileNotFoundError, "--packed-train-index must name an existing file"),
        ("directory", None, None, FileNotFoundError, "--packed-train-index must name an existing file"),
        ("file", "--batch-size", "0", ValueError, "--batch-size must be positive"),
        ("file", "--packed-block-records", "0", ValueError, "--packed-block-records must be positive"),
        ("file", "--warmup-steps", "-1", ValueError, "--warmup-steps must be non-negative"),
    ),
)
def test_packed_cli_validates_inputs_before_preparing_run(
    tmp_path,
    monkeypatch,
    index_kind,
    option,
    value,
    exception,
    message,
):
    launcher = _launcher_module()
    packed_index = tmp_path / "packed" / "manifest.json"
    packed_index.parent.mkdir()
    if index_kind == "file":
        packed_index.write_text("{}\n", encoding="utf-8")
    elif index_kind == "directory":
        packed_index.mkdir()

    prepare_calls = []
    monkeypatch.setattr(launcher, "prepare_run", lambda *_args, **_kwargs: prepare_calls.append(True))
    argv = [
        "train-packed",
        "--corpus-root",
        str(tmp_path / "corpus"),
        "--run-root",
        str(tmp_path / "runs"),
        "--packed-train-index",
        str(packed_index),
    ]
    if option is not None:
        argv.extend((option, value))

    with pytest.raises(exception, match=message):
        launcher.main(argv)

    assert prepare_calls == []


def test_train_packed_cli_passes_configuration_to_training_and_propagates_return_code(tmp_path, monkeypatch):
    launcher = _launcher_module()
    config = _config(launcher, tmp_path)
    config.run_dir.mkdir(parents=True)
    packed_index = tmp_path / "packed" / "manifest.json"
    packed_index.parent.mkdir()
    packed_index.write_text("{}\n", encoding="utf-8")
    observed = {}
    original_run_full_training = launcher.run_full_training

    monkeypatch.setattr(launcher, "prepare_run", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(launcher, "run_amp_smoke", lambda candidate: candidate)
    monkeypatch.setattr(
        launcher,
        "create_dashboard_symlink",
        lambda run_dir, dashboard_xps_root: observed.setdefault("dashboard", (run_dir, dashboard_xps_root)),
    )

    def run_full_training(candidate):
        observed["config"] = candidate
        return original_run_full_training(
            candidate,
            runner=lambda command, **_kwargs: subprocess.CompletedProcess(command, 23),
        )

    monkeypatch.setattr(launcher, "run_full_training", run_full_training)

    result = launcher.main(
        [
            "train-packed",
            "--corpus-root",
            str(tmp_path / "corpus"),
            "--run-root",
            str(tmp_path / "runs"),
            "--packed-train-index",
            str(packed_index),
            "--dashboard-xps-root",
            str(tmp_path / "dashboard"),
        ]
    )

    assert result == 23
    assert observed["dashboard"] == (config.run_dir, tmp_path / "dashboard")
    assert observed["config"].packed_train_index == packed_index
    assert observed["config"].batch_size == 8192
    assert observed["config"].num_workers == 1
    assert observed["config"].packed_block_records == 16_384
    assert observed["config"].warmup_steps == 200
    metadata = json.loads((config.run_dir / "launcher_config.json").read_text(encoding="utf-8"))
    command = json.loads((config.run_dir / "train_command.json").read_text(encoding="utf-8"))["command"]
    assert metadata["accuracy_policy"]["warmup_steps"] == 200
    assert metadata["packed_input"]["num_workers_per_rank"] == 1
    assert _option(command, "--warmup_steps") == "200"


def test_train_parser_defaults_to_safe_ddp_and_amp_options(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py"])

    args = train_module.parse_args()

    assert args.ddp_static_graph is False
    assert args.ddp_gradient_as_bucket_view is False
    assert args.amp_backbone is False
    assert args.max_train_steps is None


def test_ddp_wrapper_falls_back_when_old_torch_rejects_performance_options(monkeypatch):
    calls = []

    def ddp(model, **kwargs):
        calls.append(kwargs)
        if "static_graph" in kwargs:
            raise TypeError("unexpected keyword argument 'static_graph'")
        return (model, kwargs)

    monkeypatch.setattr(train_module.torch.nn.parallel, "DistributedDataParallel", ddp)
    model = torch.nn.Linear(2, 2)

    wrapped = train_module.wrap_distributed_model(
        model,
        local_rank=2,
        ddp_static_graph=True,
        ddp_gradient_as_bucket_view=True,
    )

    assert calls == [
        {"device_ids": [2], "static_graph": True, "gradient_as_bucket_view": True},
        {"device_ids": [2]},
    ]
    assert wrapped[0] is model


def test_lstm_wrapper_is_not_given_the_dscnn_only_amp_option(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["train.py", "--model", "lstm", "--epoch", "1"])
    args = train_module.parse_args()
    args.save_dir = str(tmp_path)
    args.verify_sample_rate = False
    observed_kwargs = {}

    class CaptureLSTM(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.probe_parameter = torch.nn.Parameter(torch.zeros(()))
            if "amp_backbone" in kwargs:
                raise AssertionError("MFCCLSTM does not implement the DSCNN-only AMP mode")
            observed_kwargs.update(kwargs)

        def forward(self, waveform):
            return torch.zeros((waveform.size(0), 2), dtype=waveform.dtype)

    class CompletedTrainer:
        def __init__(self, **_kwargs):
            pass

        def fit(self):
            return None

    features = torch.zeros((1, 2), dtype=torch.float32)
    labels = torch.zeros(1, dtype=torch.long)
    loader = DataLoader(TensorDataset(features, labels), batch_size=1)
    monkeypatch.setattr(train_module, "parse_args", lambda: args)
    monkeypatch.setattr(
        train_module,
        "init_distributed",
        lambda enabled, local_rank: SimpleNamespace(rank=0, world_size=1, local_rank=0, device=torch.device("cpu")),
    )
    monkeypatch.setattr(train_module, "prepare_device", lambda gpu: (torch.device("cpu"), []))
    monkeypatch.setattr(train_module, "set_random_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module, "build_dataloaders", lambda *args, **kwargs: (loader, loader, loader))
    monkeypatch.setattr(train_module, "MFCCLSTM", CaptureLSTM)
    monkeypatch.setattr(train_module, "Trainer", CompletedTrainer)
    monkeypatch.setattr(train_module, "barrier", lambda: None)
    monkeypatch.setattr(train_module, "destroy_distributed", lambda: None)

    train_module.main()

    assert observed_kwargs["frontend"] == "mfcc"


def test_amp_backbone_keeps_frontend_and_cpu_backbone_in_float32(monkeypatch):
    class CaptureBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_dtype = None

        def forward(self, features):
            self.input_dtype = features.dtype
            return features[:, :2]

    class DummyFeatures(torch.nn.Module):
        def forward(self, waveform):
            return waveform[:, :4].reshape(waveform.size(0), 1, 4)

    backbone = CaptureBackbone()
    model = train_module.MFCCDSCNN(
        backbone=backbone,
        frontend="mfcc",
        sample_rate=8000,
        dct_coeff=1,
        window_size_ms=32,
        window_stride_ms=32,
        bandpass_n_bands=16,
        bandpass_f_min=200.0,
        bandpass_f_max=4000.0,
        bandpass_spacing="log",
        bandpass_kernel_size=63,
        bandpass_phase_count=1,
        pre_emphasis=False,
        pre_emphasis_coeff=0.97,
        spec_aug=False,
        spec_aug_freq_mask_param=1,
        spec_aug_time_mask_param=1,
        spec_aug_num_freq_masks=0,
        spec_aug_num_time_masks=0,
        mfcc_impl="torchaudio",
        mel_filter_shape="triangular",
        log_approx_mode="exact",
        log_pwl_num_segments=6,
        log_pwl_strategy="uniform_logx",
        log_pwl_gamma=1.0,
        log_pwl_breakpoints=None,
        log_pwl_slopes=None,
        log_pwl_intercepts=None,
        log_offset=1e-6,
        log_input_clamp_min=1e-12,
        amp_backbone=True,
    )
    model.feature_extractor = DummyFeatures()
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("CPU must not autocast")))

    output = model(torch.ones(2, 4, dtype=torch.float32))

    assert backbone.input_dtype == torch.float32
    assert output.dtype == torch.float32


def test_max_train_steps_finishes_probe_without_validation_test_or_checkpoints(tmp_path):
    class ProbeTrainer(Trainer):
        def _run_eval(self, loader):
            raise AssertionError("a bounded probe must skip validation and test evaluation")

    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    loader = DataLoader(TensorDataset(features, labels), batch_size=1, shuffle=False)
    model = torch.nn.Linear(2, 2)
    trainer = ProbeTrainer(
        args=SimpleNamespace(epoch=60, max_train_steps=1, label_smoothing=0.0, log_interval=50, resume=None),
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        scheduler=None,
        train_loader=loader,
        valid_loader=loader,
        test_loader=loader,
        device=torch.device("cpu"),
        save_dir=str(tmp_path),
    )

    result = trainer.fit()

    assert sum(sum(row) for row in result.confusion) == 1
    assert not (tmp_path / "best.pt").exists()
    assert not (tmp_path / "last.pt").exists()
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["probe"] is True
    assert status["max_train_steps"] == 1


def test_trainer_passes_float32_logits_to_loss_even_if_model_returns_bfloat16(tmp_path):
    class BFloat16LogitModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, features):
            return self.linear(features).to(torch.bfloat16)

    class CaptureCriterion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_dtype = None

        def forward(self, logits, labels):
            self.input_dtype = logits.dtype
            return torch.nn.functional.cross_entropy(logits, labels)

    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    loader = DataLoader(TensorDataset(features, labels), batch_size=2, shuffle=False)
    model = BFloat16LogitModel()
    trainer = Trainer(
        args=SimpleNamespace(epoch=1, max_train_steps=1, label_smoothing=0.0, log_interval=50, resume=None),
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        scheduler=None,
        train_loader=loader,
        valid_loader=loader,
        test_loader=loader,
        device=torch.device("cpu"),
        save_dir=str(tmp_path),
    )
    criterion = CaptureCriterion()
    trainer.criterion = criterion

    trainer._run_train_epoch(epoch=1)

    assert criterion.input_dtype == torch.float32


def test_bounded_probe_rejects_parameters_that_overflow_after_optimizer_step(tmp_path):
    model = torch.nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    features = torch.full((1, 1), torch.finfo(torch.float32).max / 2, dtype=torch.float32)
    labels = torch.tensor([0], dtype=torch.long)
    loader = DataLoader(TensorDataset(features, labels), batch_size=1, shuffle=False)
    trainer = Trainer(
        args=SimpleNamespace(epoch=1, max_train_steps=1, label_smoothing=0.0, log_interval=50, resume=None),
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=10.0),
        scheduler=None,
        train_loader=loader,
        valid_loader=loader,
        test_loader=loader,
        device=torch.device("cpu"),
        save_dir=str(tmp_path),
    )

    with pytest.raises(FloatingPointError, match="Non-finite parameter"):
        trainer._run_train_epoch(epoch=1)

    assert not bool(torch.isfinite(model.weight).all())
