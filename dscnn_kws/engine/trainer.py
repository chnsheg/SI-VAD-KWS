from __future__ import annotations

import hashlib
import math
import os
import tempfile
import time
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dscnn_kws.engine.confirmation_pair import ConfirmationPairLoss
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
    if waveform.ndim not in (3, 4) or waveform.shape[-2] != 1:
        raise ValueError("packed jitter expects [batch, 1, samples] or [batch, 2, 1, samples]")
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
    index_shape = (waveform.shape[0],) + (1,) * (waveform.ndim - 2) + (waveform.shape[-1],)
    gather_index = positions.reshape(index_shape).expand(waveform.shape)
    return torch.gather(padded, dim=-1, index=gather_index)


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
        pair_train_loader=None,
    ):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.pair_train_loader = pair_train_loader
        self.device = device
        self.save_dir = save_dir
        self.best_macro_f1 = float("-inf")
        self.best_negative_recall = float("-inf")
        self._best_checkpoint_path: str | None = None
        self._best_checkpoint_epoch: int | None = None
        # Set when ``--temporal_init_from_global`` was used to warm-start an
        # order-sensitive head from a legacy global-pooling checkpoint.  A
        # migrated checkpoint is an initialization-only artifact: optimizer,
        # scheduler, and epoch counters must not be restored because their
        # parameter/state shapes belong to the old head.
        self._resume_migrated = False
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
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.pair_criterion: ConfirmationPairLoss | None = None
        if bool(getattr(self.args, "pair_objective", False)):
            self.pair_criterion = ConfirmationPairLoss(
                runtime_threshold=float(getattr(self.args, "pair_runtime_threshold", 0.8)),
                temperature=float(getattr(self.args, "pair_softmin_temperature", 0.25)),
                positive_margin=float(getattr(self.args, "pair_positive_margin", 0.0)),
                negative_margin=float(getattr(self.args, "pair_negative_margin", 0.0)),
                frame_ce_weight=float(getattr(self.args, "pair_frame_ce_weight", 1.0)),
                positive_weight=float(getattr(self.args, "pair_positive_weight", 1.0)),
                positive_hinge_tail_fraction=float(
                    getattr(self.args, "pair_positive_hinge_tail_fraction", 1.0)
                ),
                negative_weight=float(getattr(self.args, "pair_negative_weight", 1.0)),
                negative_cvar_fraction=float(getattr(self.args, "pair_negative_cvar_fraction", 0.1)),
                negative_source_cvar_fraction=float(
                    getattr(self.args, "pair_negative_source_cvar_fraction", 1.0)
                ),
                negative_frame_target_probability=float(
                    getattr(self.args, "pair_negative_frame_target_probability", 0.5)
                ),
                negative_frame_weight=float(getattr(self.args, "pair_negative_frame_weight", 0.0)),
                tail_ranking_weight=float(getattr(self.args, "pair_tail_ranking_weight", 0.0)),
                tail_ranking_margin=float(getattr(self.args, "pair_tail_ranking_margin", 0.0)),
                positive_tail_fraction=float(getattr(self.args, "pair_positive_tail_fraction", 0.1)),
                positive_class=0,
                label_smoothing=label_smoothing,
            )
        if (self.pair_criterion is None) != (self.pair_train_loader is None):
            raise ValueError("pair_objective and pair_train_loader must be enabled together")
        self.pair_objective_weight = float(getattr(self.args, "pair_objective_weight", 1.0))
        if not math.isfinite(self.pair_objective_weight) or self.pair_objective_weight < 0.0:
            raise ValueError("pair_objective_weight must be finite and non-negative")
        pair_interval = getattr(self.args, "pair_interval", 1)
        if isinstance(pair_interval, bool) or not isinstance(pair_interval, int) or pair_interval < 1:
            raise ValueError("pair_interval must be a positive integer")
        self.pair_interval = pair_interval

    @staticmethod
    def _set_loader_epoch(loader, epoch: int) -> None:
        for sampler in (getattr(loader, "sampler", None), getattr(loader, "batch_sampler", None)):
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

    @staticmethod
    def _set_frozen_backbone_eval(
        model: nn.Module,
        *,
        freeze_tail_batch_norm_stats: bool = False,
    ) -> None:
        """Keep frozen DSCNN state fixed while enabling its configured trainable tail."""
        target = Trainer._model_for_state(model)
        backbone = getattr(target, "backbone", None)
        conv_layers = getattr(backbone, "conv_layers", None)
        if backbone is None or conv_layers is None:
            raise ValueError("freeze_backbone requires a DSCNN backbone with conv_layers and a classifier")
        classifier_name = "temporal_fc" if getattr(backbone, "pooling", "global") == "temporal" else "final_fc"
        classifier = getattr(backbone, classifier_name, None)
        if classifier is None:
            raise ValueError(f"freeze_backbone requires a DSCNN backbone.{classifier_name} classifier")

        backbone.eval()
        for layer in conv_layers:
            if any(parameter.requires_grad for parameter in layer.parameters()):
                layer.train()
                if freeze_tail_batch_norm_stats:
                    for module in layer.modules():
                        if isinstance(module, nn.modules.batchnorm._BatchNorm):
                            module.eval()
        classifier.train()

    def _next_pair_batch(self, pair_iterator):
        if self.pair_criterion is None or self.pair_train_loader is None:
            raise RuntimeError("pair batch requested without a configured pair loader")
        try:
            pair_batch = next(pair_iterator)
        except StopIteration:
            pair_iterator = iter(self.pair_train_loader)
            try:
                pair_batch = next(pair_iterator)
            except StopIteration as error:
                raise ValueError("confirmation pair loader is empty") from error
        if not isinstance(pair_batch, Mapping):
            raise ValueError("confirmation pair loader must return a mapping batch")
        required_keys = {"waveform", "labels", "source_ids", "source_splits", "pair_roles", "pair_offsets"}
        missing_keys = sorted(required_keys.difference(pair_batch))
        if missing_keys:
            raise ValueError(f"confirmation pair batch is missing required keys: {missing_keys}")
        pair_waveform = pair_batch["waveform"].to(self.device, non_blocking=True)
        pair_labels = pair_batch["labels"].to(self.device, non_blocking=True)
        if pair_waveform.ndim != 4 or pair_waveform.shape[1] != 2:
            raise ValueError(
                "--pair_objective requires explicit adjacent waveform pairs with shape [batch, 2, channels, samples]"
            )
        if pair_labels.ndim != 1 or pair_labels.shape[0] != pair_waveform.shape[0]:
            raise ValueError("labels must have shape [batch] and align with pair_logits")
        source_splits = pair_batch["source_splits"]
        if isinstance(source_splits, str):
            source_splits = [source_splits]
        else:
            source_splits = list(source_splits)
        if len(source_splits) != pair_waveform.shape[0] or any(split != "train" for split in source_splits):
            raise ValueError("confirmation pair source_splits must contain 'train' for every pair")
        pair_offsets = pair_batch["pair_offsets"]
        if not isinstance(pair_offsets, torch.Tensor) or pair_offsets.shape != (pair_waveform.shape[0], 2):
            raise ValueError("pair_offsets must have shape [batch, 2]")
        expected_hop = round(
            float(getattr(self.args, "sample_rate", 16_000))
            * float(getattr(self.args, "pair_hop_ms", 96.0))
            / 1000.0
        )
        if not bool((pair_offsets[:, 1] - pair_offsets[:, 0] == expected_hop).all()):
            raise ValueError(f"pair_offsets must differ by exactly {expected_hop} samples")

        def normalized_identifiers(identifiers, *, name: str):
            if identifiers is None:
                return None
            if isinstance(identifiers, torch.Tensor):
                if identifiers.ndim != 1 or identifiers.shape[0] != pair_waveform.shape[0]:
                    raise ValueError(f"{name} must align with the full pair batch")
                return identifiers
            if isinstance(identifiers, (str, bytes)):
                raise ValueError(f"{name} must align with the full pair batch")
            values = list(identifiers)
            if len(values) != pair_waveform.shape[0]:
                raise ValueError(f"{name} must align with the full pair batch")
            return values

        source_ids = normalized_identifiers(pair_batch["source_ids"], name="negative_source_ids")
        domain_ids = normalized_identifiers(pair_batch["pair_roles"], name="negative_domain_ids")
        if domain_ids is not None and source_ids is None:
            raise ValueError("negative_source_ids are required when negative_domain_ids are provided")
        return pair_waveform, pair_labels, source_ids, domain_ids, pair_iterator

    def _run_epoch(self, loader, training: bool, epoch: int | None = None) -> EpochMetrics:
        if training:
            self.model.train()
            if bool(getattr(self.args, "freeze_backbone", False)):
                self._set_frozen_backbone_eval(
                    self.model,
                    freeze_tail_batch_norm_stats=bool(
                        getattr(self.args, "freeze_tail_batch_norm_stats", False)
                    ),
                )
            if epoch is not None:
                self._set_loader_epoch(loader, epoch)
                if self.pair_train_loader is not None:
                    self._set_loader_epoch(self.pair_train_loader, epoch)
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
        pair_iterator = iter(self.pair_train_loader) if training and self.pair_train_loader is not None else None
        with torch.set_grad_enabled(training):
            for step, batch in enumerate(iterator, start=1):
                data_wait_seconds = max(time.perf_counter() - previous_step_finished_at, 0.0)
                if isinstance(batch, Mapping):
                    if "waveform" not in batch or "labels" not in batch:
                        raise ValueError("mapping batches must contain waveform and labels")
                    waveform = batch["waveform"]
                    labels = batch["labels"]
                    jitter_ms = batch.get("jitter_ms")
                    roles = batch.get("roles")
                elif isinstance(batch, (tuple, list)) and len(batch) in (2, 3, 4):
                    waveform, labels = batch[:2]
                    jitter_ms = batch[2] if len(batch) >= 3 else None
                    roles = batch[3] if len(batch) == 4 else None
                else:
                    raise ValueError(
                        "data loader must return waveform, labels, optional packed jitter metadata, and optional roles"
                    )
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
                h2d_seconds = max(time.perf_counter() - h2d_started_at, 0.0)
                compute_started_at = time.perf_counter()
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                pair_inputs = None
                if training and pair_iterator is not None and step % self.pair_interval == 0:
                    pair_waveform, pair_labels, pair_source_ids, pair_domain_ids, pair_iterator = (
                        self._next_pair_batch(pair_iterator)
                    )
                    pair_batch_size, pair_frames, pair_channels, pair_samples = pair_waveform.shape
                    flat_pair_waveform = pair_waveform.reshape(
                        pair_batch_size * pair_frames, pair_channels, pair_samples
                    )
                    pair_inputs = (pair_labels, pair_source_ids, pair_domain_ids, pair_batch_size, pair_frames)
                    model_waveform = torch.cat((waveform, flat_pair_waveform), dim=0)
                else:
                    model_waveform = waveform
                joint_logits_fp32 = self.model(model_waveform).float()
                base_batch_size = waveform.shape[0]
                logits_fp32 = joint_logits_fp32[:base_batch_size]
                loss = self.criterion(logits_fp32, labels)
                if pair_inputs is not None:
                    pair_labels, pair_source_ids, pair_domain_ids, pair_batch_size, pair_frames = pair_inputs
                    pair_logits = joint_logits_fp32[base_batch_size:].reshape(pair_batch_size, pair_frames, -1)
                    pair_loss = self.pair_criterion(
                        pair_logits,
                        pair_labels,
                        negative_source_ids=pair_source_ids,
                        negative_domain_ids=pair_domain_ids,
                    )
                    if not bool(torch.isfinite(pair_logits).all()) or not bool(torch.isfinite(pair_loss)):
                        raise FloatingPointError("Non-finite logits or loss during confirmation-pair training")
                    loss = loss + self.pair_objective_weight * pair_loss
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

    @staticmethod
    def _sha256_file(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _pair_provenance(self) -> dict | None:
        if self.pair_criterion is None:
            return None
        pair_dataset = getattr(self.pair_train_loader, "dataset", None)
        manifest_value = getattr(pair_dataset, "manifest_path", None) or getattr(
            self.args, "pair_train_manifest", ""
        )
        manifest_path = os.path.abspath(str(manifest_value))
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Pair manifest disappeared before checkpointing: {manifest_path}")
        sample_rate = int(getattr(self.args, "sample_rate", 16_000))
        hop_ms = float(getattr(self.args, "pair_hop_ms", 96.0))
        effective_pair_batch = getattr(self.pair_train_loader, "batch_size", None)
        if effective_pair_batch is None:
            configured_pair_batch = getattr(self.args, "pair_batch", None)
            effective_pair_batch = (
                int(configured_pair_batch)
                if configured_pair_batch is not None
                else int(getattr(self.args, "batch", 256))
            )
        positive_pairs_per_batch = int(getattr(self.args, "pair_positive_per_batch", 0))
        tail_ranking_weight = float(getattr(self.args, "pair_tail_ranking_weight", 0.0))
        return {
            "manifest_path": manifest_path,
            "manifest_sha256": self._sha256_file(manifest_path),
            "sample_rate": sample_rate,
            "hop_ms": hop_ms,
            "hop_samples": round(sample_rate * hop_ms / 1000.0),
            "pair_batch_size": int(effective_pair_batch),
            **(
                {
                    "positive_pairs_per_rank_batch": positive_pairs_per_batch,
                    "pair_batch_sampling": "distributed_class_role_source_stratified_v1",
                }
                if positive_pairs_per_batch > 0
                else {}
            ),
            "seed": int(getattr(self.args, "seed", 42)),
            "label_smoothing": float(getattr(self.args, "label_smoothing", 0.0)),
            "runtime_threshold": float(getattr(self.args, "pair_runtime_threshold", 0.8)),
            "softmin_temperature": float(getattr(self.args, "pair_softmin_temperature", 0.25)),
            "positive_margin_logit": float(getattr(self.args, "pair_positive_margin", 0.0)),
            "negative_margin_logit": float(getattr(self.args, "pair_negative_margin", 0.0)),
            "frame_ce_weight": float(getattr(self.args, "pair_frame_ce_weight", 1.0)),
            "positive_weight": float(getattr(self.args, "pair_positive_weight", 1.0)),
            "positive_hinge_tail_fraction": float(
                getattr(self.args, "pair_positive_hinge_tail_fraction", 1.0)
            ),
            "negative_weight": float(getattr(self.args, "pair_negative_weight", 1.0)),
            "negative_cvar_fraction": float(getattr(self.args, "pair_negative_cvar_fraction", 0.1)),
            "negative_source_cvar_fraction": float(
                getattr(self.args, "pair_negative_source_cvar_fraction", 1.0)
            ),
            "negative_frame_target_probability": float(
                getattr(self.args, "pair_negative_frame_target_probability", 0.5)
            ),
            "negative_frame_weight": float(getattr(self.args, "pair_negative_frame_weight", 0.0)),
            **(
                {
                    "tail_ranking_weight": tail_ranking_weight,
                    "tail_ranking_margin_logit": float(
                        getattr(self.args, "pair_tail_ranking_margin", 0.0)
                    ),
                    "positive_tail_fraction": float(
                        getattr(self.args, "pair_positive_tail_fraction", 0.1)
                    ),
                }
                if tail_ranking_weight > 0.0
                else {}
            ),
            "negative_balance": "domain_then_source_cvar_v1",
            "objective_weight": self.pair_objective_weight,
            "interval": self.pair_interval,
        }

    @staticmethod
    def _pair_provenance_without_path(value):
        if not isinstance(value, dict):
            return value
        comparable = {key: item for key, item in value.items() if key != "manifest_path"}
        # Checkpoints created before positive hinge CVaR used the exact
        # equivalent of a full-tail fraction. Preserve that resume path while
        # still rejecting any non-default objective change.
        comparable.setdefault("positive_hinge_tail_fraction", 1.0)
        return comparable

    def _validate_pair_resume_provenance(self, payload: dict) -> None:
        provenance = payload.get("provenance")
        saved_has_pair_field = isinstance(provenance, dict) and "confirmation_pair" in provenance
        saved_pair = provenance.get("confirmation_pair") if saved_has_pair_field else None
        current_pair = self._pair_provenance()
        if not saved_has_pair_field:
            if current_pair is not None:
                raise ValueError(
                    "Cannot resume pair-objective training from a full checkpoint without confirmation-pair provenance"
                )
            return
        if saved_pair is None and current_pair is None:
            return
        if not isinstance(saved_pair, dict) or current_pair is None:
            raise ValueError("Checkpoint and current run disagree on whether confirmation-pair training is enabled")

        # A manifest may be relocated, but its content and every objective
        # parameter that changes gradients must remain identical on resume.
        current_comparable = self._pair_provenance_without_path(current_pair)
        saved_comparable = self._pair_provenance_without_path(saved_pair)
        compared_keys = tuple(current_comparable)
        missing_keys = [key for key in compared_keys if key not in saved_comparable]
        if missing_keys:
            raise ValueError(f"Checkpoint confirmation-pair provenance is missing keys: {missing_keys}")
        mismatches = [
            key
            for key in compared_keys
            if saved_comparable.get(key) != current_comparable.get(key)
        ]
        if mismatches:
            raise ValueError(
                "Checkpoint confirmation-pair provenance does not match the current run: "
                + ", ".join(mismatches)
            )

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
                "confirmation_pair": self._pair_provenance(),
            },
        }

    def _write_candidate_snapshot(self, epoch: int) -> None:
        cadence = int(getattr(self.args, "candidate_checkpoint_every", 0))
        if cadence > 0 and epoch % cadence == 0:
            path = os.path.join(self.save_dir, "candidates", f"epoch-{epoch:04d}.pt")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._atomic_torch_save({"model": self._model_for_state(self.model).state_dict()}, path)

    @staticmethod
    def _merge_history(existing: list[dict], restored: list[dict]) -> list[dict]:
        rows = {int(row["epoch"]): dict(row) for row in restored if "epoch" in row}
        for row in existing:
            if "epoch" not in row:
                continue
            epoch = int(row["epoch"])
            rows[epoch] = {**rows.get(epoch, {}), **row}
        return [rows[epoch] for epoch in sorted(rows)]

    @staticmethod
    def _migrate_global_to_temporal_state_dict(model: nn.Module, source_state: dict) -> dict:
        """Expand a legacy global DSCNN classifier into a temporal head.

        Global pooling computes ``W * mean(h_i) + b``.  A temporal head with
        ``B`` bins sees ``[h_1, ..., h_B]``; repeating ``W / B`` for each bin
        therefore gives an exactly equivalent starting function before the
        first gradient update.  All convolution/front-end weights are copied
        unchanged.  The helper deliberately requires a temporal DSCNN and
        matching classifier tensors so malformed checkpoints fail loudly.
        """

        target = Trainer._model_for_state(model)
        backbone = getattr(target, "backbone", target)
        if getattr(backbone, "pooling", None) != "temporal" or not hasattr(backbone, "temporal_fc"):
            raise ValueError("--temporal_init_from_global requires a temporal DSCNN backbone")
        if not isinstance(source_state, dict):
            raise ValueError("Checkpoint model state must be a dictionary")

        # A few legacy DDP runs persisted ``module.*`` keys directly.  Keep
        # migration tolerant of that format while still loading the target
        # model strictly (without leaving prefixed unexpected keys behind).
        if source_state and all(str(key).startswith("module.") for key in source_state):
            source_state = {str(key)[len("module.") :]: value for key, value in source_state.items()}

        current_state = target.state_dict()
        migrated = dict(source_state)
        temporal_bins = int(backbone.temporal_bins)
        copied_temporal = False
        for temporal_key, target_tensor in current_state.items():
            if "temporal_fc." not in temporal_key:
                continue
            global_key = temporal_key.replace("temporal_fc", "final_fc", 1)
            source_tensor = source_state.get(global_key)
            if source_tensor is None:
                raise ValueError(
                    "Legacy global checkpoint is missing the classifier tensor "
                    f"{global_key!r} required for temporal migration"
                )
            if not isinstance(source_tensor, torch.Tensor):
                raise ValueError(f"Checkpoint tensor {global_key!r} is not a torch.Tensor")
            if temporal_key.endswith("weight"):
                if source_tensor.ndim != 2 or target_tensor.ndim != 2:
                    raise ValueError("final_fc/temporal_fc weights must be rank-2 tensors")
                expected_shape = (source_tensor.shape[0], source_tensor.shape[1] * temporal_bins)
                if tuple(target_tensor.shape) != tuple(expected_shape):
                    raise ValueError(
                        f"Temporal classifier shape mismatch: target={tuple(target_tensor.shape)}, "
                        f"expected={expected_shape}"
                    )
                # ``flatten(1)`` on [batch, channels, bins] stores each
                # channel's bins contiguously: [c0b0..c0bB, c1b0..].  Tile
                # along that innermost bin axis, rather than repeating the
                # whole classifier row, to preserve the global-pool function.
                migrated[temporal_key] = (
                    source_tensor.unsqueeze(-1)
                    .expand(-1, -1, temporal_bins)
                    .reshape(source_tensor.shape[0], -1)
                    .div(float(temporal_bins))
                )
            else:
                if tuple(target_tensor.shape) != tuple(source_tensor.shape):
                    raise ValueError(
                        f"Temporal classifier bias shape mismatch: target={tuple(target_tensor.shape)}, "
                        f"source={tuple(source_tensor.shape)}"
                    )
                migrated[temporal_key] = source_tensor.clone()
            copied_temporal = True

        if not copied_temporal:
            raise ValueError("Legacy checkpoint does not contain temporal classifier tensors to migrate")
        return migrated

    def _restore_state_dict(self, payload) -> bool:
        model_state = None
        if isinstance(payload, dict):
            model_state = payload.get("model") or payload.get("model_state_dict") or payload.get("state_dict")
        if model_state is None and isinstance(payload, dict):
            model_state = payload
        if not isinstance(model_state, dict):
            raise ValueError("Checkpoint does not contain a model state dictionary")
        model_for_state = self._model_for_state(self.model)
        try:
            model_for_state.load_state_dict(model_state)
        except RuntimeError:
            if not bool(getattr(self.args, "temporal_init_from_global", False)):
                raise
            migrated_state = self._migrate_global_to_temporal_state_dict(model_for_state, model_state)
            model_for_state.load_state_dict(migrated_state, strict=True)
            self._resume_migrated = True
            warnings.warn(
                "Initialized temporal DSCNN head from a legacy global-pooling checkpoint; "
                "optimizer, scheduler, and epoch state will be restarted.",
                RuntimeWarning,
            )
            return False
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
            if self._resume_migrated:
                barrier()
                return 1
            warnings.warn(
                "Loaded a legacy model-only checkpoint; optimizer, scheduler, and epoch state are unavailable, "
                "so training restarts at epoch 1.",
                RuntimeWarning,
            )
            barrier()
            return 1
        self._validate_pair_resume_provenance(payload)
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
            if isinstance(resume_provenance, dict) and isinstance(best_provenance, dict):
                resume_pair = self._pair_provenance_without_path(resume_provenance.get("confirmation_pair"))
                best_pair = self._pair_provenance_without_path(best_provenance.get("confirmation_pair"))
                if resume_pair != best_pair:
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
                self._write_candidate_snapshot(epoch)
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
