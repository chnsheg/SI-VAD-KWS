from __future__ import annotations

import math
import os
import tempfile
import time
import warnings
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dscnn_kws.utils.distributed import barrier, is_rank_zero, reduce_epoch_totals, reduce_max
from dscnn_kws.utils.training_artifacts import TrainingArtifactWriter


PCM16_SCALE = 32767.0


def _apply_packed_window_jitter(
    waveform: torch.Tensor,
    jitter_ms: torch.Tensor,
    *,
    sample_rate: int,
) -> torch.Tensor:
    """Apply each record's bounded translation after packed PCM reaches its device."""
    if waveform.ndim != 3 or waveform.shape[1] != 1:
        raise ValueError("packed jitter expects waveform shape [batch, 1, samples]")
    jitter_samples = torch.round(jitter_ms.to(device=waveform.device, dtype=torch.float32) * sample_rate / 1000).to(torch.long)
    max_jitter = int(jitter_samples.max().item()) if jitter_samples.numel() else 0
    if max_jitter <= 0:
        return waveform
    random_offsets = torch.floor(
        torch.rand(jitter_samples.shape, device=waveform.device) * (2 * jitter_samples + 1).to(torch.float32)
    ).to(torch.long) - jitter_samples
    padded = F.pad(waveform, (max_jitter, max_jitter))
    positions = torch.arange(waveform.shape[-1], device=waveform.device).unsqueeze(0)
    positions = positions + max_jitter + random_offsets.unsqueeze(1)
    return torch.gather(padded, dim=2, index=positions.unsqueeze(1))


def _host_rss_bytes() -> int | None:
    if os.name != "posix":
        return None
    try:
        pages = int((open("/proc/self/statm", encoding="utf-8").read().split())[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (IndexError, OSError, ValueError):
        return None


@dataclass
class EpochMetrics:
    loss: float
    acc: float
    precision: float
    recall: float
    f1: float
    macro_f1: float | None = None
    positive_recall: float = 0.0
    negative_recall: float = 0.0
    finite_gradients: bool = True
    confusion: list[list[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.macro_f1 is None:
            self.macro_f1 = self.f1


def epoch_metrics_from_totals(
    loss_sum: torch.Tensor,
    total: torch.Tensor,
    correct: torch.Tensor,
    confusion: torch.Tensor,
) -> EpochMetrics:
    total_value = max(float(total.item()), 1.0)
    diagonal = confusion.diag()
    precision_by_class = diagonal / confusion.sum(dim=0).clamp_min(1.0)
    recall_by_class = diagonal / confusion.sum(dim=1).clamp_min(1.0)
    f1_by_class = 2.0 * precision_by_class * recall_by_class / (precision_by_class + recall_by_class).clamp_min(1e-12)
    present_classes = confusion.sum(dim=1) > 0
    if bool(present_classes.any()):
        precision = float(precision_by_class[present_classes].mean().item())
        recall = float(recall_by_class[present_classes].mean().item())
        f1 = float(f1_by_class[present_classes].mean().item())
    else:
        precision = recall = f1 = 0.0
    positive_recall = float(recall_by_class[0].item()) if confusion.shape[0] > 0 else 0.0
    negative_recall = float(recall_by_class[1].item()) if confusion.shape[0] > 1 else 0.0
    return EpochMetrics(
        loss=float(loss_sum.item()) / total_value,
        acc=float(correct.item()) / total_value,
        precision=precision,
        recall=recall,
        f1=f1,
        macro_f1=f1,
        positive_recall=positive_recall,
        negative_recall=negative_recall,
        confusion=confusion.detach().to(device="cpu", dtype=torch.int64).tolist(),
    )


class MarginAnchorLoss(nn.Module):
    """Push negatives below neg_anchor, pull positives above pos_anchor, stop elsewhere."""

    def __init__(self, neg_anchor: float, pos_anchor: float, pos_weight: float, neg_weight: float = 1.0):
        super().__init__()
        self.neg_anchor = float(neg_anchor)
        self.pos_anchor = float(pos_anchor)
        self.pos_weight = float(pos_weight)
        self.neg_weight = float(neg_weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits.float(), dim=1)
        s = probs[:, 0]  # class 0 = positive
        neg_term = self.neg_weight * torch.clamp(s - self.neg_anchor, min=0.0) ** 2
        pos_term = self.pos_weight * torch.clamp(self.pos_anchor - s, min=0.0) ** 2
        is_pos = labels == 0
        return torch.where(is_pos, pos_term, neg_term).mean()


class Trainer:
    def __init__(
        self,
        args,
        model: nn.Module,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
        test_loader,
        device,
        save_dir: str,
        artifact_writer: TrainingArtifactWriter | None = None,
    ):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir
        self.best_macro_f1 = float("-inf")
        self.best_negative_recall = float("-inf")
        self._best_checkpoint_path: str | None = None
        self._best_checkpoint_epoch: int | None = None
        self.early_stopping_bad_epochs = 0
        self.last_completed_epoch = 0
        self._last_train_elapsed_seconds = 0.0
        self.artifact_writer = artifact_writer
        if self.artifact_writer is None and is_rank_zero():
            args_dict = vars(args) if hasattr(args, "__dict__") else {}
            self.artifact_writer = TrainingArtifactWriter(
                save_dir,
                argv=args_dict,
                run_variant=str(getattr(args, "run_name", "unspecified")),
            )
        self.history = list(self.artifact_writer.history) if self.artifact_writer is not None else []
        label_smoothing = float(getattr(self.args, "label_smoothing", 0.0))
        label_smoothing = min(max(label_smoothing, 0.0), 0.2)
        self.daat_lambda = float(getattr(self.args, "daat_lambda", 0.0))
        self.domain_criterion = nn.CrossEntropyLoss() if self.daat_lambda > 0.0 else None
        if str(getattr(self.args, "loss_type", "ce")) == "margin":
            self.criterion = MarginAnchorLoss(
                neg_anchor=float(getattr(self.args, "margin_neg_anchor", 0.5)),
                pos_anchor=float(getattr(self.args, "margin_pos_anchor", 0.9)),
                pos_weight=float(getattr(self.args, "margin_pos_weight", 2.0)),
                neg_weight=float(getattr(self.args, "margin_neg_weight", 1.0)),
            )
            print(
                "[INFO] margin-anchor loss: neg_anchor="
                f"{getattr(self.args, 'margin_neg_anchor', 0.5)} "
                f"pos_anchor={getattr(self.args, 'margin_pos_anchor', 0.9)} "
                f"pos_weight={getattr(self.args, 'margin_pos_weight', 2.0)}",
                flush=True,
            )
        positive_loss_weight = float(getattr(self.args, "positive_loss_weight", 1.0))
        if str(getattr(self.args, "loss_type", "ce")) == "margin":
            pass  # criterion already set to MarginAnchorLoss above
        elif positive_loss_weight > 1.0:
            class_weights = torch.tensor(
                [positive_loss_weight, 1.0], device=self.device, dtype=torch.float32
            )
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
            print(
                f"[INFO] weighted CE: positive_loss_weight={positive_loss_weight} (class 0 = positive)",
                flush=True,
            )
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            print("[INFO] plain CE loss", flush=True)

    def _run_epoch(self, loader, training: bool, epoch: int | None = None) -> EpochMetrics:
        if training:
            self.model.train()
            if epoch is not None:
                for sampler in (getattr(loader, "sampler", None), getattr(loader, "batch_sampler", None)):
                    if hasattr(sampler, "set_epoch"):
                        sampler.set_epoch(epoch)
        else:
            self.model.eval()
        loss_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        total = torch.zeros((), device=self.device, dtype=torch.float64)
        correct = torch.zeros((), device=self.device, dtype=torch.float64)
        confusion: torch.Tensor | None = None
        iterator = tqdm(loader, desc="train", leave=False, disable=not training or not is_rank_zero())
        mixture_role_counts: dict[str, int] = {}
        max_train_steps = self._max_train_steps() if training else None
        total_steps = min(len(loader), max_train_steps) if max_train_steps is not None else len(loader)
        log_interval = max(1, int(getattr(self.args, "log_interval", 50)))
        started_at = time.perf_counter()
        previous_step_finished_at = started_at
        with torch.set_grad_enabled(training):
            for step, batch in enumerate(iterator, start=1):
                data_wait_seconds = max(time.perf_counter() - previous_step_finished_at, 0.0)
                if not isinstance(batch, (tuple, list)) or len(batch) not in (2, 3, 4, 5):
                    raise ValueError("data loader must return waveform, labels, optional packed jitter metadata, optional roles, and optional domains")
                waveform, labels = batch[:2]
                jitter_ms = batch[2] if len(batch) >= 3 else None
                roles = batch[3] if len(batch) >= 4 else None
                domains = batch[4] if len(batch) == 5 else None
                if training and roles is not None:
                    for role in roles:
                        role_name = str(role)
                        mixture_role_counts[role_name] = mixture_role_counts.get(role_name, 0) + 1
                h2d_started_at = time.perf_counter()
                waveform = waveform.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                packed_pcm = waveform.dtype == torch.int16
                if packed_pcm:
                    waveform = waveform.float().div_(PCM16_SCALE)
                    if jitter_ms is not None:
                        waveform = _apply_packed_window_jitter(
                            waveform,
                            jitter_ms,
                            sample_rate=int(getattr(self.args, "sample_rate", 16_000)),
                        )
                if training:
                    gain_db = float(getattr(self.args, "random_gain_db", 0.0))
                    if gain_db > 0.0:
                        g = torch.empty(waveform.shape[0], 1, device=waveform.device, dtype=waveform.dtype)
                        if waveform.dim() == 3:
                            g = g.unsqueeze(-1)
                        g.uniform_(-gain_db, gain_db)
                        waveform = waveform * (10.0 ** (g / 20.0))
                h2d_seconds = max(time.perf_counter() - h2d_started_at, 0.0)
                compute_started_at = time.perf_counter()
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                out = self.model(waveform)
                dom_logits = None
                if isinstance(out, tuple):
                    logits, dom_logits = out
                else:
                    logits = out
                logits_fp32 = logits.float()
                loss = self.criterion(logits_fp32, labels)
                if training and dom_logits is not None and domains is not None:
                    dom_labels = domains.to(self.device, non_blocking=True).long()
                    loss = loss + self.daat_lambda * self.domain_criterion(dom_logits.float(), dom_labels)
                if training:
                    if not bool(torch.isfinite(logits_fp32).all()) or not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Non-finite logits or loss during training")
                    loss.backward()
                    for parameter in self.model.parameters():
                        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                            raise FloatingPointError("Non-finite gradient during training")
                    self.optimizer.step()
                    if self.scheduler is not None and bool(getattr(self.args, "scheduler_step_per_batch", False)):
                        self.scheduler.step()
                    for parameter in self.model.parameters():
                        if not bool(torch.isfinite(parameter).all()):
                            raise FloatingPointError("Non-finite parameter after training optimizer step")
                predictions = torch.argmax(logits_fp32, dim=1)
                class_count = logits_fp32.shape[1]
                if confusion is None:
                    confusion = torch.zeros((class_count, class_count), device=self.device, dtype=torch.float64)
                encoded_pairs = labels * class_count + predictions
                confusion += torch.bincount(encoded_pairs, minlength=class_count * class_count).reshape(class_count, class_count)
                batch_size = labels.size(0)
                loss_sum += loss.detach().to(torch.float64) * batch_size
                total += batch_size
                correct += (predictions == labels).sum().to(torch.float64)
                if (
                    training
                    and self.artifact_writer is not None
                    and (step % log_interval == 0 or step == total_steps)
                ):
                    elapsed = max(time.perf_counter() - started_at, 1e-12)
                    telemetry = None
                    if bool(getattr(self.args, "live_telemetry", False)):
                        if self.device.type == "cuda":
                            torch.cuda.synchronize(self.device)
                            cuda_reserved_bytes = int(torch.cuda.memory_reserved(self.device))
                            cuda_max_reserved_bytes = int(torch.cuda.max_memory_reserved(self.device))
                        else:
                            cuda_reserved_bytes = 0
                            cuda_max_reserved_bytes = 0
                        world_size = max(1, int(getattr(self.args, "world_size", 1)))
                        telemetry = {
                            "data_wait_s": data_wait_seconds,
                            "h2d_enqueue_s": h2d_seconds,
                            "forward_backward_wall_s": max(time.perf_counter() - compute_started_at, 0.0),
                            "cuda_reserved_bytes": cuda_reserved_bytes,
                            "cuda_max_reserved_bytes": cuda_max_reserved_bytes,
                            "host_rss_bytes": _host_rss_bytes(),
                            "global_samples_per_second": float(batch_size * world_size / max(data_wait_seconds + h2d_seconds, 1e-12)),
                        }
                    self.artifact_writer.write_train_progress(
                        epoch=epoch if epoch is not None else 0,
                        step=step,
                        total_steps=total_steps,
                        it_per_sec=step / elapsed,
                        batch_loss=float(loss.detach().item()),
                        telemetry=telemetry,
                    )
                previous_step_finished_at = time.perf_counter()
                if max_train_steps is not None and step >= max_train_steps:
                    break
        if confusion is None:
            raise ValueError("Cannot compute metrics for an empty data loader")
        if training:
            elapsed = max(time.perf_counter() - started_at, 1e-12)
            elapsed_tensor = torch.tensor(elapsed, device=self.device, dtype=torch.float64)
            self._last_train_elapsed_seconds = float(reduce_max(elapsed_tensor).item())
        totals = reduce_epoch_totals(torch.stack((loss_sum, total, correct)))
        reduced_confusion = reduce_epoch_totals(confusion)
        if training and mixture_role_counts:
            role_names = sorted(mixture_role_counts)
            local_role_counts = torch.tensor(
                [mixture_role_counts[name] for name in role_names], device=self.device, dtype=torch.int64
            )
            reduced_role_counts = reduce_epoch_totals(local_role_counts)
            if self.artifact_writer is not None and epoch is not None and is_rank_zero():
                self.artifact_writer.write_mixture_metrics(
                    epoch,
                    {name: int(count) for name, count in zip(role_names, reduced_role_counts.tolist())},
                )
        return epoch_metrics_from_totals(totals[0], totals[1], totals[2], reduced_confusion)

    def _run_eval(self, loader) -> EpochMetrics:
        return self._run_epoch(loader, training=False)

    def _run_train_epoch(self, epoch: int) -> EpochMetrics:
        return self._run_epoch(self.train_loader, training=True, epoch=epoch)

    def _max_train_steps(self) -> int | None:
        value = getattr(self.args, "max_train_steps", None)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("max_train_steps must be a positive integer")
        return value

    @staticmethod
    def _load_torch(path: str, device):
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=device)

    @staticmethod
    def _model_for_state(model: nn.Module) -> nn.Module:
        return model.module if hasattr(model, "module") else model

    @staticmethod
    def _atomic_torch_save(value, path: str) -> None:
        fd, temporary_path = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=os.path.dirname(path))
        os.close(fd)
        try:
            torch.save(value, temporary_path)
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def _checkpoint_state(self, epoch: int) -> dict:
        sampler_epoch = int(epoch)
        for sampler in (getattr(self.train_loader, "sampler", None), getattr(self.train_loader, "batch_sampler", None)):
            if hasattr(sampler, "epoch"):
                sampler_epoch = int(sampler.epoch)
                break
        return {
            "format_version": 4,
            "model": self._model_for_state(self.model).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "epoch": int(epoch),
            "best_macro_f1": float(self.best_macro_f1),
            "best_negative_recall": float(self.best_negative_recall),
            "best_checkpoint_epoch": self._best_checkpoint_epoch,
            "early_stopping_bad_epochs": int(self.early_stopping_bad_epochs),
            "sampler_epoch": sampler_epoch,
            "history": self.history,
            "provenance": {
                "dashboard_reco_metric": "one_minus_validation_macro_f1",
                "run_variant": str(getattr(self.args, "run_name", "unspecified")),
            },
        }

    @staticmethod
    def _merge_history(existing: list[dict], restored: list[dict]) -> list[dict]:
        rows = {int(row["epoch"]): dict(row) for row in restored if "epoch" in row}
        for row in existing:
            if "epoch" not in row:
                continue
            epoch = int(row["epoch"])
            rows[epoch] = {**rows.get(epoch, {}), **row}
        return [rows[epoch] for epoch in sorted(rows)]

    def _restore_state_dict(self, payload) -> bool:
        model_state = None
        if isinstance(payload, dict):
            model_state = payload.get("model") or payload.get("model_state_dict") or payload.get("state_dict")
        if model_state is None and isinstance(payload, dict):
            model_state = payload
        if not isinstance(model_state, dict):
            raise ValueError("Checkpoint does not contain a model state dictionary")
        if int(getattr(self.args, "domain_classes", 0) or 0) > 0:
            self._model_for_state(self.model).load_state_dict(model_state, strict=False)
        else:
            self._model_for_state(self.model).load_state_dict(model_state)
        return (
            isinstance(payload, dict)
            and "epoch" in payload
            and payload.get("optimizer") is not None
            and "scheduler" in payload
        )

    def _resume_if_requested(self) -> int:
        resume_path = getattr(self.args, "resume", None)
        if not resume_path:
            return 1
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        barrier()
        payload = self._load_torch(resume_path, self.device)
        full_checkpoint = self._restore_state_dict(payload)
        if not full_checkpoint:
            warnings.warn(
                "Loaded a legacy model-only checkpoint; optimizer, scheduler, and epoch state are unavailable, "
                "so training restarts at epoch 1.",
                RuntimeWarning,
            )
            barrier()
            return 1
        if payload.get("optimizer") is not None:
            self.optimizer.load_state_dict(payload["optimizer"])
        if self.scheduler is not None and payload.get("scheduler") is not None:
            self.scheduler.load_state_dict(payload["scheduler"])
        self.best_macro_f1 = float(payload.get("best_macro_f1", payload.get("best_acc", float("-inf"))))
        self.best_negative_recall = float(payload.get("best_negative_recall", float("-inf")))
        local_best_path = os.path.join(self.save_dir, "best.pt")
        if os.path.basename(resume_path) == "best.pt":
            self._best_checkpoint_path = resume_path
            self._best_checkpoint_epoch = int(payload.get("best_checkpoint_epoch", payload["epoch"]))
        else:
            matched_best_epoch = self._matching_local_best_epoch(payload, local_best_path)
            if matched_best_epoch is not None:
                self._best_checkpoint_path = local_best_path
                self._best_checkpoint_epoch = matched_best_epoch
        self.early_stopping_bad_epochs = int(payload.get("early_stopping_bad_epochs", 0))
        restored_history = payload.get("history")
        if isinstance(restored_history, list):
            self.history = self._merge_history(self.history, restored_history)
            if self.artifact_writer is not None:
                self.artifact_writer.replace_history(self.history)
        self.last_completed_epoch = int(payload.get("epoch", 0))
        sampler_epoch = int(payload.get("sampler_epoch", self.last_completed_epoch))
        for sampler in (getattr(self.train_loader, "sampler", None), getattr(self.train_loader, "batch_sampler", None)):
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(sampler_epoch)
        barrier()
        return self.last_completed_epoch + 1

    def _matching_local_best_epoch(self, resume_payload: dict, local_best_path: str) -> int | None:
        if not math.isfinite(self.best_macro_f1) or not os.path.isfile(local_best_path):
            return None
        try:
            best_payload = self._load_torch(local_best_path, self.device)
            if not isinstance(best_payload, dict):
                return None
            best_epoch = best_payload.get("epoch")
            if isinstance(best_epoch, bool) or not isinstance(best_epoch, int):
                return None
            if (
                float(best_payload.get("best_macro_f1", float("-inf"))) != self.best_macro_f1
                or float(best_payload.get("best_negative_recall", float("-inf"))) != self.best_negative_recall
            ):
                return None
            resume_provenance = resume_payload.get("provenance")
            best_provenance = best_payload.get("provenance")
            if (
                isinstance(resume_provenance, dict)
                and isinstance(best_provenance, dict)
                and resume_provenance.get("run_variant") != best_provenance.get("run_variant")
            ):
                return None
            format_version = int(resume_payload.get("format_version", 0))
            if format_version >= 4:
                expected_epoch = resume_payload.get("best_checkpoint_epoch")
                if (
                    not isinstance(expected_epoch, bool)
                    and isinstance(expected_epoch, int)
                    and best_epoch == expected_epoch
                    and best_payload.get("best_checkpoint_epoch") == expected_epoch
                ):
                    return expected_epoch
                return None
            if format_version == 3 and self._v3_history_confirms_best_epoch(resume_payload, best_epoch):
                warnings.warn(
                    "Migrating a version 3 checkpoint by validating local best.pt metadata and history.",
                    RuntimeWarning,
                )
                return best_epoch
            return None
        except (OSError, RuntimeError, TypeError, ValueError):
            return None

    def _v3_history_confirms_best_epoch(self, resume_payload: dict, best_epoch: int) -> bool:
        history = resume_payload.get("history")
        if not isinstance(history, list):
            return False
        for row in history:
            if not isinstance(row, dict) or int(row.get("epoch", -1)) != best_epoch:
                continue
            valid_metrics = row.get("valid")
            if not isinstance(valid_metrics, dict):
                continue
            if (
                float(valid_metrics.get("macro_f1", float("-inf"))) == self.best_macro_f1
                and float(valid_metrics.get("negative_recall", float("-inf"))) == self.best_negative_recall
            ):
                return True
        return False

    def _eligible_for_best(self, train_metrics: EpochMetrics, valid_metrics: EpochMetrics) -> bool:
        return bool(train_metrics.finite_gradients) and float(valid_metrics.positive_recall) >= float(
            getattr(self.args, "min_positive_recall", 0.0)
        )

    def _is_better_validation(self, valid_metrics: EpochMetrics) -> bool:
        macro_f1 = float(valid_metrics.macro_f1)
        negative_recall = float(valid_metrics.negative_recall)
        return macro_f1 > self.best_macro_f1 or (
            macro_f1 == self.best_macro_f1 and negative_recall > self.best_negative_recall
        )

    def _update_early_stopping(self, epoch: int, is_best: bool) -> bool:
        min_epoch = int(getattr(self.args, "early_stopping_min_epoch", 0))
        patience = int(getattr(self.args, "early_stopping_patience", 0))
        if patience <= 0 or epoch <= min_epoch:
            return False
        if is_best:
            self.early_stopping_bad_epochs = 0
            return False
        self.early_stopping_bad_epochs += 1
        return self.early_stopping_bad_epochs >= patience

    def _early_stopping_already_reached(self) -> bool:
        min_epoch = int(getattr(self.args, "early_stopping_min_epoch", 0))
        patience = int(getattr(self.args, "early_stopping_patience", 0))
        return (
            patience > 0
            and self.last_completed_epoch > min_epoch
            and self.early_stopping_bad_epochs >= patience
        )

    def _restore_best_model(self) -> None:
        barrier()
        checkpoint_path = self._best_checkpoint_path
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            raise FileNotFoundError("No checkpoint was created before test evaluation")
        self._restore_state_dict(self._load_torch(checkpoint_path, self.device))
        barrier()

    def fit(self):
        if is_rank_zero():
            os.makedirs(self.save_dir, exist_ok=True)
        barrier()
        max_train_steps = self._max_train_steps()
        if max_train_steps is not None:
            if getattr(self.args, "resume", None):
                raise ValueError("--max_train_steps probes cannot resume a checkpoint")
            train_m = self._run_train_epoch(epoch=1)
            self.last_completed_epoch = 1
            global_train_examples = sum(sum(row) for row in train_m.confusion)
            throughput_examples_per_second = global_train_examples / max(self._last_train_elapsed_seconds, 1e-12)
            if self.artifact_writer is not None:
                self.artifact_writer.write_terminal_status(
                    "completed",
                    {
                        "probe": True,
                        "max_train_steps": max_train_steps,
                        "throughput_examples_per_second": throughput_examples_per_second,
                        **(
                            {
                                "last_epoch": 1,
                                "best_macro_f1": None,
                                "training_only": True,
                            }
                            if bool(getattr(self.args, "skip_test", False))
                            else {}
                        ),
                    },
                )
            if is_rank_zero():
                print(
                    f"[PROBE] finite=true max_train_steps={max_train_steps} "
                    f"throughput_examples_per_second={throughput_examples_per_second:.6f}"
                )
            return train_m
        best_path = os.path.join(self.save_dir, "best.pt")
        last_path = os.path.join(self.save_dir, "last.pt")
        start_epoch = self._resume_if_requested()

        last_validation_metrics: EpochMetrics | None = None
        epochs = () if self._early_stopping_already_reached() else range(start_epoch, self.args.epoch + 1)
        for epoch in epochs:
            epoch_lr = float(self.optimizer.param_groups[0]["lr"])
            train_m = self._run_train_epoch(epoch)
            valid_m = self._run_eval(self.valid_loader)
            last_validation_metrics = valid_m
            global_train_examples = sum(sum(row) for row in train_m.confusion)
            throughput_examples_per_second = global_train_examples / max(self._last_train_elapsed_seconds, 1e-12)

            is_best = self._eligible_for_best(train_m, valid_m) and self._is_better_validation(valid_m)
            if is_best:
                self.best_macro_f1 = float(valid_m.macro_f1)
                self.best_negative_recall = float(valid_m.negative_recall)
                self._best_checkpoint_path = best_path
                self._best_checkpoint_epoch = epoch
            self.last_completed_epoch = epoch
            if self.artifact_writer is not None:
                self.artifact_writer.write_epoch(
                    epoch,
                    train_m,
                    valid_m,
                    lr=epoch_lr,
                    throughput_examples_per_second=throughput_examples_per_second,
                )
                self.history = list(self.artifact_writer.history)

            if self.scheduler is not None and not bool(getattr(self.args, "scheduler_step_per_batch", False)):
                self.scheduler.step()

            should_stop = self._update_early_stopping(epoch, is_best)
            checkpoint_state = self._checkpoint_state(epoch)
            if is_rank_zero():
                if is_best:
                    self._atomic_torch_save(checkpoint_state, best_path)
                self._atomic_torch_save(checkpoint_state, last_path)
            barrier()

            if is_rank_zero():
                print(
                    f"Epoch {epoch}/{self.args.epoch} | "
                    f"train_loss {train_m.loss:.4f} acc {train_m.acc:.4f} macro_f1 {train_m.macro_f1:.4f} | "
                    f"valid_loss {valid_m.loss:.4f} acc {valid_m.acc:.4f} macro_f1 {valid_m.macro_f1:.4f} | "
                    f"lr {epoch_lr:.6f}"
                )
            if should_stop:
                if is_rank_zero():
                    print(
                        f"[EARLY_STOP] epoch={epoch} patience={getattr(self.args, 'early_stopping_patience', 0)} "
                        f"bad_epochs={self.early_stopping_bad_epochs}"
                    )
                break

        if bool(getattr(self.args, "skip_test", False)):
            if self.artifact_writer is not None:
                self.artifact_writer.write_terminal_status(
                    "completed",
                    {
                        "last_epoch": int(self.last_completed_epoch),
                        "best_macro_f1": float(self.best_macro_f1) if math.isfinite(self.best_macro_f1) else None,
                        "training_only": True,
                    },
                )
            return last_validation_metrics
        if self._best_checkpoint_path is None:
            raise RuntimeError("No validation checkpoint satisfied the configured selection gates")
        self._restore_best_model()
        test_m = self._run_eval(self.test_loader)

        if self.artifact_writer is not None:
            self.artifact_writer.write_test(self.last_completed_epoch, test_m)
            self.history = list(self.artifact_writer.history)
            self.artifact_writer.write_completed(self.last_completed_epoch, test_m)
        if is_rank_zero():
            print(
                f"[TEST] loss={test_m.loss:.4f} acc={test_m.acc:.4f} "
                f"precision={test_m.precision:.4f} recall={test_m.recall:.4f} macro_f1={test_m.macro_f1:.4f}"
            )
        return test_m
