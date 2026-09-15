import json
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dscnn_kws.engine.trainer import EpochMetrics, Trainer, epoch_metrics_from_totals
from dscnn_kws.train import build_optimizer_scheduler
from dscnn_kws.utils.training_artifacts import TrainingArtifactWriter


def _metric(acc: float, macro_f1: float) -> EpochMetrics:
    return EpochMetrics(
        loss=0.25,
        acc=acc,
        precision=macro_f1,
        recall=macro_f1,
        f1=macro_f1,
        macro_f1=macro_f1,
        positive_recall=macro_f1,
        negative_recall=macro_f1,
        confusion=[[1, 0], [0, 1]],
    )


class ScriptedValidationTrainer(Trainer):
    def __init__(self, *args, validation_metrics, **kwargs):
        super().__init__(*args, **kwargs)
        self.validation_metrics = iter(validation_metrics)
        self.trained_epochs = []

    def _run_train_epoch(self, epoch: int) -> EpochMetrics:
        self.trained_epochs.append(epoch)
        return super()._run_train_epoch(epoch)

    def _run_eval(self, loader) -> EpochMetrics:
        return next(self.validation_metrics)


def _loaders():
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    loader = DataLoader(TensorDataset(features, labels), batch_size=2, shuffle=False)
    return loader, loader, loader


def _args(epoch: int, resume=None):
    return SimpleNamespace(epoch=epoch, label_smoothing=0.0, log_interval=50, resume=resume)


def _trainer(tmp_path, args, validation_metrics):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epoch))
    train_loader, valid_loader, test_loader = _loaders()
    return ScriptedValidationTrainer(
        args=args,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        device=torch.device("cpu"),
        save_dir=str(tmp_path),
        artifact_writer=TrainingArtifactWriter(tmp_path, argv={"epoch": args.epoch}),
        validation_metrics=validation_metrics,
    )


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def test_epoch_metrics_include_binary_recalls_and_integer_confusion():
    metrics = epoch_metrics_from_totals(
        loss_sum=torch.tensor(6.0),
        total=torch.tensor(12.0),
        correct=torch.tensor(9.0),
        confusion=torch.tensor([[8.0, 2.0], [1.0, 1.0]]),
    )

    assert metrics.loss == pytest.approx(0.5)
    assert metrics.acc == pytest.approx(0.75)
    assert metrics.precision == pytest.approx((8 / 9 + 1 / 3) / 2)
    assert metrics.recall == pytest.approx((8 / 10 + 1 / 2) / 2)
    assert metrics.macro_f1 == pytest.approx(((2 * (8 / 9) * (8 / 10) / ((8 / 9) + (8 / 10))) + 0.4) / 2)
    assert metrics.f1 == metrics.macro_f1
    assert metrics.positive_recall == pytest.approx(0.8)
    assert metrics.negative_recall == pytest.approx(0.5)
    assert metrics.confusion == [[8, 2], [1, 1]]


def test_trainer_selects_best_by_macro_f1_and_resume_starts_next_epoch(tmp_path):
    first = _trainer(tmp_path, _args(epoch=1), [_metric(0.95, 0.20), _metric(0.0, 0.20)])
    first.fit()

    best_before_resume = _load_checkpoint(tmp_path / "best.pt")
    assert best_before_resume["epoch"] == 1
    assert best_before_resume["best_macro_f1"] == pytest.approx(0.20)

    resumed = _trainer(
        tmp_path,
        _args(epoch=3, resume=str(tmp_path / "last.pt")),
        [_metric(0.99, 0.10), _metric(0.98, 0.15), _metric(0.0, 0.20)],
    )
    resumed.fit()

    assert resumed.trained_epochs == [2, 3]
    best_after_resume = _load_checkpoint(tmp_path / "best.pt")
    last_after_resume = _load_checkpoint(tmp_path / "last.pt")
    assert best_after_resume["epoch"] == 1
    assert best_after_resume["best_macro_f1"] == pytest.approx(0.20)
    assert last_after_resume["epoch"] == 3
    assert last_after_resume["scheduler"]["last_epoch"] == 3
    assert [row["epoch"] for row in last_after_resume["history"]] == [1, 2, 3]


def test_cosine_scheduler_is_single_run_and_accepts_sparse_legacy_args():
    args = SimpleNamespace(opt="adam", lr=0.01, weight_decay=0.0, scheduler="cos", eta_min=None, epoch=4)
    _, scheduler = build_optimizer_scheduler(args, torch.nn.Linear(2, 2))

    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert scheduler.T_max == 4


def test_trainer_records_current_lr_and_global_train_throughput(tmp_path):
    trainer = _trainer(tmp_path, _args(epoch=1), [_metric(0.50, 0.50), _metric(0.50, 0.50)])

    trainer.fit()

    rows = [json.loads(line) for line in (tmp_path / "kws_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    train_row = next(row for row in rows if row["split"] == "train")
    assert train_row["lr"] == pytest.approx(0.01)
    assert train_row["throughput_examples_per_second"] > 0.0


def test_trainer_marks_status_completed_after_writing_final_test_artifacts(tmp_path):
    trainer = _trainer(tmp_path, _args(epoch=1), [_metric(0.50, 0.50), _metric(0.75, 0.75)])

    trainer.fit()

    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["last_epoch"] == 1
    assert status["test_metrics"]["macro_f1"] == pytest.approx(0.75)
