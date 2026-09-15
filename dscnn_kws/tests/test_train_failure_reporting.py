import json

import pytest

import dscnn_kws.train as train_module
from dscnn_kws.utils.training_artifacts import FailureArtifactReporter, TrainingArtifactWriter


def test_failed_training_writes_status_and_destroys_without_final_barrier(tmp_path, monkeypatch):
    class FailingTrainer:
        def fit(self):
            raise RuntimeError("simulated training failure")

    writer = TrainingArtifactWriter(tmp_path)
    destroyed = []
    monkeypatch.setattr(train_module, "is_rank_zero", lambda: True)
    monkeypatch.setattr(train_module, "destroy_distributed", lambda: destroyed.append(True))
    monkeypatch.setattr(train_module, "barrier", lambda: pytest.fail("failure teardown must not enter a final barrier"))

    with pytest.raises(RuntimeError, match="simulated training failure"):
        train_module.fit_and_teardown(FailingTrainer(), writer)

    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["exception_type"] == "RuntimeError"
    assert status["exception_message"] == "simulated training failure"
    assert "simulated training failure" in status["traceback"]
    assert destroyed == [True]


def test_nonzero_failure_writes_rank_sentinel_and_first_shared_status_without_barrier(tmp_path, monkeypatch):
    class FailingTrainer:
        def fit(self):
            raise RuntimeError("rank three failure")

    (tmp_path / "train.log").write_text("rank zero previous line\nrank zero log tail\n", encoding="utf-8")
    reporter = FailureArtifactReporter(tmp_path, rank=3)
    destroyed = []
    monkeypatch.setattr(train_module, "is_rank_zero", lambda: False)
    monkeypatch.setattr(train_module, "destroy_distributed", lambda: destroyed.append(True))
    monkeypatch.setattr(train_module, "barrier", lambda: pytest.fail("failure teardown must not enter a final barrier"))

    with pytest.raises(RuntimeError, match="rank three failure"):
        train_module.fit_and_teardown(FailingTrainer(), None, reporter)

    sentinel = json.loads((tmp_path / "failure-rank-3.json").read_text(encoding="utf-8"))
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert sentinel["rank"] == 3
    assert sentinel["exception_message"] == "rank three failure"
    assert sentinel["timestamp"]
    assert "rank zero log tail" in sentinel["rank_zero_train_log_tail"]
    assert status["rank"] == 3
    assert status["exception_message"] == "rank three failure"
    assert destroyed == [True]
    assert not (tmp_path / ".argv.json").exists()
    assert not (tmp_path / "history.json").exists()
    assert not (tmp_path / "kws_metrics.jsonl").exists()

    FailureArtifactReporter(tmp_path, rank=2).write_failure(RuntimeError("later failure"), "later traceback")
    assert json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))["rank"] == 3
    assert (tmp_path / "failure-rank-2.json").exists()


def test_keyboard_interrupt_writes_interrupted_status_and_destroys_without_final_barrier(tmp_path, monkeypatch):
    class InterruptingTrainer:
        def fit(self):
            raise KeyboardInterrupt()

    writer = TrainingArtifactWriter(tmp_path)
    destroyed = []
    monkeypatch.setattr(train_module, "is_rank_zero", lambda: True)
    monkeypatch.setattr(train_module, "destroy_distributed", lambda: destroyed.append(True))
    monkeypatch.setattr(train_module, "barrier", lambda: pytest.fail("interruption teardown must not enter a final barrier"))

    with pytest.raises(KeyboardInterrupt):
        train_module.fit_and_teardown(InterruptingTrainer(), writer)

    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "interrupted"
    assert status["rank"] == 0
    assert status["exception_type"] == "KeyboardInterrupt"
    assert destroyed == [True]


def test_nonzero_keyboard_interrupt_writes_rank_sentinel_without_shared_status(tmp_path, monkeypatch):
    class InterruptingTrainer:
        def fit(self):
            raise KeyboardInterrupt()

    reporter = FailureArtifactReporter(tmp_path, rank=3)
    destroyed = []
    monkeypatch.setattr(train_module, "is_rank_zero", lambda: False)
    monkeypatch.setattr(train_module, "destroy_distributed", lambda: destroyed.append(True))
    monkeypatch.setattr(train_module, "barrier", lambda: pytest.fail("interruption teardown must not enter a final barrier"))

    with pytest.raises(KeyboardInterrupt):
        train_module.fit_and_teardown(InterruptingTrainer(), None, reporter)

    sentinel = json.loads((tmp_path / "failure-rank-3.json").read_text(encoding="utf-8"))
    assert sentinel["status"] == "interrupted"
    assert sentinel["rank"] == 3
    assert sentinel["exception_type"] == "KeyboardInterrupt"
    assert not (tmp_path / "status.json").exists()
    assert destroyed == [True]
