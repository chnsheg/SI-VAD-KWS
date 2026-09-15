import argparse
import json
from types import SimpleNamespace

import pytest
import torch

import dscnn_kws.train as train_module


def test_dataloader_initialization_failure_writes_status_and_only_uses_startup_barrier(tmp_path, monkeypatch):
    args = argparse.Namespace(
        model="dscnn",
        epoch=1,
        save_dir=str(tmp_path),
        run_name="initialization-failure",
        lr=0.001,
        gpu=0,
        distributed=False,
        local_rank=None,
        offline_augmented_dataset=False,
        frontend="mfcc",
        dct_coeff=13,
        bandpass_n_bands=16,
        log_pwl_fit_json=None,
        seed=42,
        non_deterministic=True,
        root=str(tmp_path),
        dataset="synthetic",
        verify_sample_rate=False,
    )
    destroyed = []
    barriers = []

    monkeypatch.setattr(train_module, "parse_args", lambda: args)
    monkeypatch.setattr(
        train_module,
        "init_distributed",
        lambda enabled, local_rank: SimpleNamespace(rank=0, world_size=1, local_rank=0, device=torch.device("cpu")),
    )
    monkeypatch.setattr(train_module, "set_random_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module, "prepare_device", lambda gpu: (torch.device("cpu"), []))
    monkeypatch.setattr(
        train_module,
        "build_dataloaders",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("simulated dataloader initialization failure")),
    )
    monkeypatch.setattr(train_module, "destroy_distributed", lambda: destroyed.append(True))
    monkeypatch.setattr(train_module, "barrier", lambda: barriers.append(True))

    with pytest.raises(RuntimeError, match="simulated dataloader initialization failure"):
        train_module.main()

    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["rank"] == 0
    assert status["exception_message"] == "simulated dataloader initialization failure"
    assert "simulated dataloader initialization failure" in status["traceback"]
    assert barriers == [True]
    assert destroyed == [True]


def test_run_artifact_resolution_failure_writes_bootstrap_status_and_destroys_distributed(tmp_path, monkeypatch):
    args = argparse.Namespace(
        model="dscnn",
        epoch=1,
        save_dir=str(tmp_path),
        run_name=None,
        lr=0.001,
        gpu=0,
        distributed=False,
        local_rank=None,
        offline_augmented_dataset=False,
        frontend="mfcc",
        dct_coeff=13,
        bandpass_n_bands=16,
        log_pwl_fit_json=None,
    )
    destroyed = []
    barriers = []

    monkeypatch.setattr(train_module, "parse_args", lambda: args)
    monkeypatch.setattr(
        train_module,
        "init_distributed",
        lambda enabled, local_rank: SimpleNamespace(rank=0, world_size=1, local_rank=0, device=torch.device("cpu")),
    )
    monkeypatch.setattr(
        train_module,
        "_resolve_run_artifacts",
        lambda ignored_args: (_ for _ in ()).throw(RuntimeError("simulated artifact resolution failure")),
    )
    monkeypatch.setattr(train_module, "destroy_distributed", lambda: destroyed.append(True))
    monkeypatch.setattr(train_module, "barrier", lambda: barriers.append(True))

    with pytest.raises(RuntimeError, match="simulated artifact resolution failure"):
        train_module.main()

    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["rank"] == 0
    assert status["exception_message"] == "simulated artifact resolution failure"
    assert "simulated artifact resolution failure" in status["traceback"]
    assert barriers == []
    assert destroyed == [True]


def test_failure_reporter_construction_failure_after_init_still_destroys_distributed(tmp_path, monkeypatch):
    args = argparse.Namespace(
        model="dscnn",
        epoch=1,
        save_dir=str(tmp_path),
        run_name="reporter-construction-failure",
        lr=0.001,
        gpu=0,
        distributed=False,
        local_rank=None,
        offline_augmented_dataset=False,
        frontend="mfcc",
        dct_coeff=13,
        bandpass_n_bands=16,
        log_pwl_fit_json=None,
    )
    destroyed = []
    barriers = []

    monkeypatch.setattr(train_module, "parse_args", lambda: args)
    monkeypatch.setattr(
        train_module,
        "init_distributed",
        lambda enabled, local_rank: SimpleNamespace(rank=0, world_size=1, local_rank=0, device=torch.device("cpu")),
    )
    monkeypatch.setattr(
        train_module,
        "FailureArtifactReporter",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("simulated reporter construction failure")),
    )
    monkeypatch.setattr(train_module, "destroy_distributed", lambda: destroyed.append(True))
    monkeypatch.setattr(train_module, "barrier", lambda: barriers.append(True))

    with pytest.raises(OSError, match="simulated reporter construction failure"):
        train_module.main()

    assert barriers == []
    assert destroyed == [True]


def test_run_artifact_resolution_keyboard_interrupt_writes_interrupted_status_and_destroys_distributed(tmp_path, monkeypatch):
    args = argparse.Namespace(
        model="dscnn",
        epoch=1,
        save_dir=str(tmp_path),
        run_name=None,
        lr=0.001,
        gpu=0,
        distributed=False,
        local_rank=None,
        offline_augmented_dataset=False,
        frontend="mfcc",
        dct_coeff=13,
        bandpass_n_bands=16,
        log_pwl_fit_json=None,
    )
    destroyed = []
    barriers = []

    monkeypatch.setattr(train_module, "parse_args", lambda: args)
    monkeypatch.setattr(
        train_module,
        "init_distributed",
        lambda enabled, local_rank: SimpleNamespace(rank=0, world_size=1, local_rank=0, device=torch.device("cpu")),
    )
    monkeypatch.setattr(
        train_module,
        "_resolve_run_artifacts",
        lambda ignored_args: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    monkeypatch.setattr(train_module, "destroy_distributed", lambda: destroyed.append(True))
    monkeypatch.setattr(train_module, "barrier", lambda: barriers.append(True))

    with pytest.raises(KeyboardInterrupt):
        train_module.main()

    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "interrupted"
    assert status["rank"] == 0
    assert status["exception_type"] == "KeyboardInterrupt"
    assert barriers == []
    assert destroyed == [True]
