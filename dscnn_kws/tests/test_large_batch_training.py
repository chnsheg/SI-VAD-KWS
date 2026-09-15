from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from dscnn_kws.engine.trainer import Trainer
from dscnn_kws.train import WarmupCosineScheduler, build_optimizer_scheduler, configure_scheduler_total_steps
from dscnn_kws.utils.training_artifacts import TrainingArtifactWriter


def _launcher_module():
    path = Path(__file__).resolve().parents[2] / "tools" / "run_kws_reclean_four_gpu.py"
    spec = importlib.util.spec_from_file_location("run_kws_reclean_four_gpu_large_batch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _option(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_adamw_warmup_cosine_reaches_peak_then_eta_min():
    model = torch.nn.Linear(2, 2)
    args = SimpleNamespace(
        opt="adamw",
        lr=0.004,
        weight_decay=1e-5,
        scheduler="cos",
        eta_min=0.00004,
        epoch=2,
        warmup_steps=4,
        scheduler_total_steps=12,
    )

    optimizer, scheduler = build_optimizer_scheduler(args, model)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert isinstance(scheduler, WarmupCosineScheduler)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
    for _ in range(4):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.004)
    for _ in range(8):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.00004)


def test_configure_scheduler_total_steps_uses_actual_loader_updates():
    args = SimpleNamespace(epoch=60, warmup_steps=200)

    configure_scheduler_total_steps(args, range(63))

    assert args.scheduler_total_steps == 3780


def test_live_train_progress_keeps_dashboard_syntax_and_writes_atomic_telemetry(tmp_path):
    writer = TrainingArtifactWriter(tmp_path, argv={}, run_variant="packed-b8192")

    writer.write_train_progress(
        epoch=1,
        step=1,
        total_steps=63,
        it_per_sec=2.0,
        batch_loss=0.125,
        telemetry={"data_wait_s": 0.2, "cuda_reserved_bytes": 123, "global_samples_per_second": 65_536.0},
    )

    assert "Train | Epoch 1 | 1/63 | 2.00 it/sec | Loss 0.1250" in (tmp_path / "train.log").read_text(encoding="utf-8")
    live = json.loads((tmp_path / "live_metrics.json").read_text(encoding="utf-8"))
    assert live["data_wait_s"] == pytest.approx(0.2)
    assert live["cuda_reserved_bytes"] == 123
    assert live["global_samples_per_second"] == pytest.approx(65_536.0)


def test_packed_training_command_uses_8192_bounded_prefetch_and_large_batch_policy(tmp_path):
    launcher = _launcher_module()
    manifests = launcher.TrainingManifestPaths(
        train_manifest_path=tmp_path / "manifests" / "train.jsonl",
        validation_manifest_path=tmp_path / "manifests" / "validation.jsonl",
        test_manifest_path=tmp_path / "manifests" / "test.jsonl",
    )
    config = launcher.RecleanTrainingConfig(
        corpus_root=tmp_path / "corpus",
        run_dir=tmp_path / "run",
        run_name="packed-b8192",
        manifests=manifests,
        batch_size=8192,
        num_workers=1,
        packed_train_index=tmp_path / "packed" / "manifest.json",
        packed_block_records=16_384,
        warmup_steps=200,
    )

    command = launcher.build_training_command(config)

    assert _option(command, "--batch") == "8192"
    assert _option(command, "--num_workers") == "1"
    assert _option(command, "--prefetch_factor") == "1"
    assert _option(command, "--opt") == "adamw"
    assert _option(command, "--lr") == "0.004"
    assert _option(command, "--eta_min") == "4e-05"
    assert _option(command, "--warmup_steps") == "200"
    assert _option(command, "--packed_train_index") == str(config.packed_train_index)
    assert _option(command, "--log_interval") == "1"
    assert "--live_telemetry" in command


def test_trainer_scales_packed_pcm16_and_steps_batch_scheduler(tmp_path):
    class PackedBatchDataset(Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return torch.full((1, 4), 32767, dtype=torch.int16), 0, 0

    class CaptureModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 2)
            self.observed = []

        def forward(self, waveform):
            self.observed.append(waveform.detach().clone())
            return self.linear(waveform.reshape(waveform.shape[0], -1))

    class CountingScheduler:
        def __init__(self):
            self.calls = 0

        def step(self):
            self.calls += 1

    model = CaptureModel()
    scheduler = CountingScheduler()
    loader = DataLoader(PackedBatchDataset(), batch_size=2)
    trainer = Trainer(
        args=SimpleNamespace(
            epoch=1,
            max_train_steps=1,
            label_smoothing=0.0,
            log_interval=1,
            resume=None,
            scheduler_step_per_batch=True,
        ),
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        scheduler=scheduler,
        train_loader=loader,
        valid_loader=loader,
        test_loader=loader,
        device=torch.device("cpu"),
        save_dir=str(tmp_path),
        artifact_writer=None,
    )

    trainer._run_train_epoch(epoch=1)

    assert scheduler.calls == 1
    assert model.observed[0].dtype == torch.float32
    assert torch.allclose(model.observed[0], torch.ones_like(model.observed[0]))
