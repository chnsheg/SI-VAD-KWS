from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from dscnn_kws.configs import CLASS_ENCODING, CLASS_LIST
from dscnn_kws.data import build_dataloaders
from dscnn_kws.train import build_optimizer_scheduler
from dscnn_kws.utils import prepare_device, set_random_seed

from .model import NASKWSModel
from .search_space import LayerGene, NASArchitecture


@dataclass
class CandidateTrainResult:
    uid: str
    best_epoch: int
    best_valid_acc: float
    best_valid_loss: float
    test_acc_at_best: float
    test_loss_at_best: float
    total_mults: int | None
    params: int | None
    reason: str | None


def parse_args():
    p = argparse.ArgumentParser("Retrain searched NAS candidates from topk.json")
    p.add_argument("--topk_json", required=True, type=str)
    p.add_argument("--out_dir", required=True, type=str)

    p.add_argument("--root", default="./dataset", type=str)
    p.add_argument("--dataset", default="speech_commands_v0.02_sr8k", type=str)
    p.add_argument("--sample_rate", default=8000, type=int)
    p.add_argument("--batch", default=256, type=int)
    p.add_argument("--gpu", default=1, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--non_deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num_workers", default=8, type=int)
    p.add_argument("--prefetch_factor", default=4, type=int)
    p.add_argument("--noise_aug", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow_online_resample", action="store_true", default=False)
    p.add_argument("--strict_sample_rate", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--epoch", "--epochs", dest="epoch", default=60, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--weight_decay", default=1e-6, type=float)
    p.add_argument("--opt", choices=["adam", "sgd"], default="adam")
    p.add_argument("--momentum", default=0.9, type=float)
    p.add_argument("--scheduler", choices=["cos", "step"], default="cos")
    p.add_argument("--t0", default=10, type=int)
    p.add_argument("--t_mult", default=1, type=int)
    p.add_argument("--eta_min", default=None, type=float)
    p.add_argument("--step_size", default=20, type=int)
    p.add_argument("--gamma", default=0.2, type=float)
    p.add_argument("--label_smoothing", default=0.0, type=float)
    return p.parse_args()


def _evaluate(model: nn.Module, loader, device: torch.device, criterion: nn.Module) -> tuple[float, float]:
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
    return correct / max(1, total), loss_sum / max(1, len(loader))


def _arch_from_dict(d: dict) -> NASArchitecture:
    layers = []
    for l in d["layers"]:
        op = l["op"]
        if op == "eca":
            layers.append(
                LayerGene(
                    op=op,
                    channels=int(l.get("channels", 8)),
                    kernel_t=int(l.get("kernel_t", 3)),
                    kernel_f=int(l.get("kernel_f", 3)),
                    stride_t=int(l.get("stride_t", 1)),
                    stride_f=int(l.get("stride_f", 1)),
                    eca_kernel=int(l.get("eca_kernel", 3)),
                )
            )
        elif op == "dsconv1d":
            layers.append(
                LayerGene(
                    op=op,
                    channels=int(l["channels"]),
                    kernel_t=int(l.get("kernel_t", 1)),
                    kernel_f=int(l.get("kernel_f", 3)),
                    stride_t=int(l.get("stride_t", 1)),
                    stride_f=int(l.get("stride_f", 1)),
                    eca_kernel=int(l.get("eca_kernel", 3)),
                )
            )
        else:
            layers.append(
                LayerGene(
                    op=op,
                    channels=int(l["channels"]),
                    kernel_t=int(l.get("kernel_t", 3)),
                    kernel_f=int(l.get("kernel_f", 3)),
                    stride_t=int(l.get("stride_t", 1)),
                    stride_f=int(l.get("stride_f", 1)),
                    eca_kernel=int(l.get("eca_kernel", 3)),
                )
            )
    return NASArchitecture(
        layers=layers,
        mfcc_window_ms=int(d.get("mfcc_window_ms", 32)),
        mfcc_stride_ms=int(d.get("mfcc_stride_ms", 32)),
        mfcc_n_mfcc=int(d.get("mfcc_n_mfcc", d.get("f_target", 13))),
        t_target=int(d.get("t_target", 32)),
        f_target=int(d.get("f_target", d.get("mfcc_n_mfcc", 13))),
        uid=str(d.get("uid", "")),
    )


def main():
    args = parse_args()
    set_random_seed(args.seed, deterministic=not args.non_deterministic)
    device, _ = prepare_device(args.gpu)

    with open(args.topk_json, "r", encoding="utf-8") as f:
        topk_items = json.load(f)

    data_path = os.path.join(args.root, args.dataset)
    train_loader, valid_loader, test_loader = build_dataloaders(data_path, CLASS_LIST, CLASS_ENCODING, args)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    results: list[CandidateTrainResult] = []

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(topk_items):
        arch = _arch_from_dict(item["arch"])
        uid = item.get("uid", f"cand_{idx:05d}")
        arch.uid = uid

        model = NASKWSModel(arch=arch, num_classes=len(CLASS_LIST), sample_rate=args.sample_rate).to(device)
        optimizer, scheduler = build_optimizer_scheduler(args, model)

        best_valid_acc = -1.0
        best_valid_loss = 1e9
        best_epoch = -1
        best_state = None

        print(f"[RETRAIN] ({idx + 1}/{len(topk_items)}) uid={uid} start")
        for ep in range(args.epoch):
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

            valid_acc, valid_loss = _evaluate(model, valid_loader, device, criterion)
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_valid_loss = valid_loss
                best_epoch = ep + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if (ep + 1) % 10 == 0 or ep == 0 or (ep + 1) == args.epoch:
                print(
                    f"[RETRAIN] uid={uid} epoch={ep + 1}/{args.epoch} "
                    f"valid_acc={valid_acc:.4f} best={best_valid_acc:.4f}"
                )

        if best_state is not None:
            model.load_state_dict(best_state)
        test_acc, test_loss = _evaluate(model, test_loader, device, criterion)

        ckpt_path = out_dir / f"{uid}.pt"
        torch.save(
            {
                "uid": uid,
                "arch": item["arch"],
                "state_dict": model.state_dict(),
                "best_epoch": best_epoch,
                "best_valid_acc": best_valid_acc,
                "test_acc_at_best": test_acc,
            },
            ckpt_path,
        )

        results.append(
            CandidateTrainResult(
                uid=uid,
                best_epoch=best_epoch,
                best_valid_acc=best_valid_acc,
                best_valid_loss=best_valid_loss,
                test_acc_at_best=test_acc,
                test_loss_at_best=test_loss,
                total_mults=item.get("total_mults"),
                params=item.get("params"),
                reason=item.get("reason"),
            )
        )

    results.sort(key=lambda x: x.best_valid_acc, reverse=True)
    with open(out_dir / "retrain_results.json", "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    with open(out_dir / "retrain_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print(f"[RETRAIN] done. results saved to {out_dir / 'retrain_results.json'}")


if __name__ == "__main__":
    main()
