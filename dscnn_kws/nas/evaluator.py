from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn as nn

from dscnn_kws.configs import CLASS_ENCODING, CLASS_LIST
from dscnn_kws.data import build_dataloaders
from dscnn_kws.train import build_optimizer_scheduler
from dscnn_kws.utils import prepare_device, set_random_seed

from .model import NASKWSModel
from .search_space import NASArchitecture


@dataclass
class EvalResult:
    acc: float
    loss: float


def _evaluate(model: nn.Module, loader, device: torch.device, criterion: nn.Module) -> EvalResult:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for waveform, labels in loader:
            waveform = waveform.to(device)
            labels = labels.to(device)
            logits = model(waveform)
            loss = criterion(logits, labels)
            loss_sum += float(loss.item())
            pred = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return EvalResult(acc=correct / max(1, total), loss=loss_sum / max(1, len(loader)))


def train_eval_candidate(
    arch: NASArchitecture,
    args,
    epochs: int = 10,
    seed: int = 42,
) -> dict:
    set_random_seed(seed, deterministic=not getattr(args, "non_deterministic", True))
    device, _ = prepare_device(args.gpu)

    data_path = os.path.join(args.root, args.dataset)
    train_loader, valid_loader, _ = build_dataloaders(data_path, CLASS_LIST, CLASS_ENCODING, args)

    model = NASKWSModel(arch, num_classes=len(CLASS_LIST), sample_rate=args.sample_rate).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(getattr(args, "label_smoothing", 0.0)))
    optimizer, scheduler = build_optimizer_scheduler(args, model)

    for _ in range(epochs):
        model.train()
        for waveform, labels in train_loader:
            waveform = waveform.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(waveform)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

    valid = _evaluate(model, valid_loader, device, criterion)
    return {
        "acc": valid.acc,
        "loss": valid.loss,
    }
