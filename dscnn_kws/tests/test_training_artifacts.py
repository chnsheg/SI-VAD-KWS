import json
import os
import re
import time
from pathlib import Path

import pytest

from dscnn_kws.utils.training_artifacts import FailureArtifactReporter, TrainingArtifactWriter
from dscnn_kws.utils import training_artifacts as artifacts_module


def _metrics(macro_f1: float) -> dict:
    return {
        "loss": 0.25,
        "acc": 0.75,
        "precision": 0.70,
        "recall": 0.65,
        "f1": macro_f1,
        "macro_f1": macro_f1,
        "positive_recall": 0.80,
        "negative_recall": 0.50,
        "confusion": [[8, 2], [1, 1]],
    }


def _dashboard_argv_list_to_map(raw_argv: object) -> dict[str, str]:
    """Mirror the dashboard's `.argv.json` list parsing contract."""
    if not isinstance(raw_argv, list):
        return {}
    argv_map = {}
    for entry in raw_argv:
        if isinstance(entry, str) and "=" in entry:
            key, value = entry.split("=", 1)
            argv_map[key] = value
    return argv_map


def test_writer_emits_dashboard_compatible_and_durable_epoch_artifacts(tmp_path):
    writer = TrainingArtifactWriter(
        tmp_path,
        argv={"model": "dscnn", "epoch": 60, "api_token": "must-not-be-persisted"},
        run_variant="binary-reclean",
    )

    writer.write_train_progress(epoch=1, step=2, total_steps=4, it_per_sec=12.5, batch_loss=0.125)
    writer.write_epoch(epoch=1, train_metrics=_metrics(0.50), valid_metrics=_metrics(0.75))
    writer.write_test(epoch=1, metrics=_metrics(0.80))

    raw_argv = json.loads((tmp_path / ".argv.json").read_text(encoding="utf-8"))
    argv = _dashboard_argv_list_to_map(raw_argv)
    assert isinstance(raw_argv, list)
    assert raw_argv == sorted(raw_argv)
    assert argv["variant"] == "binary-reclean"
    assert argv["dashboard_reco_metric"] == "one_minus_validation_macro_f1"
    assert argv["selected_best_metric"] == "validation_macro_f1"
    assert argv["run_kind"] == "kws_reclean"
    assert argv["epoch"] == "60"
    assert argv["model"] == "dscnn"
    assert "api_token" not in argv
    assert "must-not-be-persisted" not in raw_argv

    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert isinstance(history, list)
    assert history[0]["epoch"] == 1
    assert history[0]["train"]["reco"] == 0.50
    assert history[0]["valid"]["reco"] == 0.25
    assert history[0]["valid"]["macro_f1"] == 0.75
    assert history[0]["test"]["reco"] == 0.20
    assert history[0]["test"]["confusion"] == [[8, 2], [1, 1]]

    log = (tmp_path / "train.log").read_text(encoding="utf-8")
    assert re.search(
        r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\].*\[INFO\] - "
        r"Train \| Epoch 1 \| 2/4 \| 12\.50 it/sec \| Loss 0\.1250$",
        log,
        flags=re.MULTILINE,
    )
    assert "Train Summary | Epoch 1 | Loss=0.2500 | Reco=0.5000" in log
    assert "Valid Summary | Epoch 1 | Loss=0.2500 | Reco=0.2500" in log
    assert "Test Summary | Epoch 1 | Loss=0.2500 | Reco=0.2000" in log

    metric_rows = [json.loads(line) for line in (tmp_path / "kws_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["split"] for row in metric_rows] == ["train", "valid", "test"]
    assert metric_rows[1]["metrics"]["macro_f1"] == 0.75
    assert not list(tmp_path.glob(".*.tmp"))


def test_writer_records_epoch_lr_and_global_throughput_in_history_and_jsonl(tmp_path):
    writer = TrainingArtifactWriter(tmp_path)

    writer.write_epoch(
        epoch=1,
        train_metrics=_metrics(0.60),
        valid_metrics=_metrics(0.70),
        lr=0.001,
        throughput_examples_per_second=321.5,
    )

    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert history[0]["train"]["lr"] == 0.001
    assert history[0]["train"]["throughput_examples_per_second"] == 321.5
    rows = [json.loads(line) for line in (tmp_path / "kws_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(row["lr"] == 0.001 for row in rows)
    assert all(row["throughput_examples_per_second"] == 321.5 for row in rows)


def test_writer_resume_discards_partial_log_and_jsonl_tails(tmp_path):
    (tmp_path / "train.log").write_text("old complete log\npartial log", encoding="utf-8")
    (tmp_path / "kws_metrics.jsonl").write_text('{"legacy": true}\n{"partial":', encoding="utf-8")

    writer = TrainingArtifactWriter(tmp_path)
    writer.write_train_progress(epoch=1, step=1, total_steps=2, it_per_sec=4.0, batch_loss=0.25)
    writer.write_epoch(epoch=1, train_metrics=_metrics(0.6), valid_metrics=_metrics(0.7))

    log = (tmp_path / "train.log").read_text(encoding="utf-8")
    assert log.startswith("old complete log\n")
    assert "partial log" not in log
    assert all(json.loads(line) for line in (tmp_path / "kws_metrics.jsonl").read_text(encoding="utf-8").splitlines())


def test_writer_archives_previous_status_and_marks_new_run_running(tmp_path):
    previous_status = {
        "status": "failed",
        "rank": 2,
        "timestamp": "2026-08-29 10:00:00,000",
        "exception_type": "RuntimeError",
        "exception_message": "previous run failure",
    }
    (tmp_path / "status.json").write_text(json.dumps(previous_status), encoding="utf-8")

    TrainingArtifactWriter(tmp_path, argv={"epoch": 2}, run_variant="resumed-run")

    archived_status = json.loads((tmp_path / "status.previous.json").read_text(encoding="utf-8"))
    active_status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert archived_status == previous_status
    assert active_status["status"] == "running"
    assert active_status["rank"] == 0
    assert active_status["timestamp"]


def test_failure_reporter_replaces_running_status_with_first_failure(tmp_path):
    (tmp_path / "status.json").write_text(
        json.dumps({"status": "running", "rank": 0, "timestamp": "2026-08-29 10:00:00,000"}),
        encoding="utf-8",
    )

    FailureArtifactReporter(tmp_path, rank=3).write_failure(RuntimeError("rank three failure"), "rank three traceback")
    FailureArtifactReporter(tmp_path, rank=2).write_failure(RuntimeError("later failure"), "later traceback")

    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["rank"] == 3
    assert status["exception_message"] == "rank three failure"
    assert (tmp_path / "failure-rank-3.json").exists()
    assert (tmp_path / "failure-rank-2.json").exists()


def test_failure_reporter_reclaims_stale_status_claim_when_status_is_absent(tmp_path):
    claim_path = tmp_path / ".status.json.claim"
    claim_path.write_text("stale claim", encoding="utf-8")
    stale_time = time.time() - 61
    os.utime(claim_path, (stale_time, stale_time))

    FailureArtifactReporter(tmp_path, rank=1).write_failure(RuntimeError("claim recovery"), "claim traceback")

    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["rank"] == 1
    assert not claim_path.exists()


def test_failure_reporter_preserves_unreadable_existing_status(tmp_path):
    status_path = tmp_path / "status.json"
    status_path.write_text("{not valid json", encoding="utf-8")

    FailureArtifactReporter(tmp_path, rank=1).write_failure(RuntimeError("later failure"), "later traceback")

    assert status_path.read_text(encoding="utf-8") == "{not valid json"
    assert (tmp_path / "failure-rank-1.json").exists()


def test_writer_marks_running_status_completed_with_final_epoch_and_test_metrics(tmp_path):
    writer = TrainingArtifactWriter(tmp_path)

    writer.write_completed(last_epoch=3, test_metrics=_metrics(0.80))

    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["rank"] == 0
    assert status["timestamp"]
    assert status["last_epoch"] == 3
    assert status["test_metrics"]["macro_f1"] == 0.80


def test_writer_completion_does_not_overwrite_existing_failed_status(tmp_path):
    writer = TrainingArtifactWriter(tmp_path)
    FailureArtifactReporter(tmp_path, rank=0).write_failure(RuntimeError("first failure"), "first traceback")

    writer.write_completed(last_epoch=3, test_metrics=_metrics(0.80))

    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["exception_message"] == "first failure"


def test_writer_completion_requires_an_active_running_status(tmp_path):
    writer = TrainingArtifactWriter(tmp_path)
    writer.status_path.unlink()

    claimed = writer.write_completed(last_epoch=3, test_metrics=_metrics(0.80))

    assert claimed is False
    assert not writer.status_path.exists()


@pytest.mark.parametrize(
    ("operation", "target_name"),
    [("log", "train.log"), ("metrics", "kws_metrics.jsonl")],
)
def test_atomic_snapshot_rewrite_failure_keeps_previous_log_and_jsonl(tmp_path, monkeypatch, operation, target_name):
    writer = TrainingArtifactWriter(tmp_path)
    writer.write_train_progress(epoch=1, step=1, total_steps=2, it_per_sec=4.0, batch_loss=0.25)
    writer.write_epoch(epoch=1, train_metrics=_metrics(0.6), valid_metrics=_metrics(0.7))
    target_path = tmp_path / target_name
    before = target_path.read_bytes()
    original_replace = artifacts_module.os.replace

    def fail_target_replace(source, destination):
        if Path(destination) == target_path:
            raise OSError(f"simulated {target_name} replacement failure")
        return original_replace(source, destination)

    monkeypatch.setattr(artifacts_module.os, "replace", fail_target_replace)

    with pytest.raises(OSError, match="replacement failure"):
        if operation == "log":
            writer.write_train_progress(epoch=1, step=2, total_steps=2, it_per_sec=4.0, batch_loss=0.20)
        else:
            writer.write_epoch(epoch=2, train_metrics=_metrics(0.65), valid_metrics=_metrics(0.75))

    assert target_path.read_bytes() == before
