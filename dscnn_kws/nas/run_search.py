from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evolution import run_search


def parse_args():
    p = argparse.ArgumentParser("NAS search for dscnn_kws")
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
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--opt", choices=["adam", "sgd"], default="adam")
    p.add_argument("--weight_decay", default=1e-6, type=float)
    p.add_argument("--momentum", default=0.9, type=float)
    p.add_argument("--scheduler", choices=["cos", "step"], default="cos")
    p.add_argument("--t0", default=10, type=int)
    p.add_argument("--t_mult", default=1, type=int)
    p.add_argument("--eta_min", default=None, type=float)
    p.add_argument("--step_size", default=20, type=int)
    p.add_argument("--gamma", default=0.2, type=float)
    p.add_argument("--label_smoothing", default=0.0, type=float)

    p.add_argument("--nas_num_layers", default=6, type=int)
    p.add_argument("--nas_t_target", default=32, type=int)
    p.add_argument("--nas_f_target", default=16, type=int)
    p.add_argument("--nas_init_samples", default=120, type=int)
    p.add_argument("--nas_total_samples", default=300, type=int)
    p.add_argument("--nas_mutation_prob", default=0.2, type=float)
    p.add_argument("--nas_epochs_per_candidate", default=10, type=int)
    p.add_argument("--nas_mult_limit", default=2_200_000, type=int)
    p.add_argument("--nas_mult_limit_parent", default=3_000_000, type=int)
    p.add_argument("--nas_param_limit", default=120_000, type=int)
    p.add_argument("--nas_sampling_max_tries", default=30, type=int)
    p.add_argument("--nas_topk", default=20, type=int)
    p.add_argument("--nas_log_interval", default=20, type=int)
    p.add_argument("--nas_out_dir", default="dscnn_kws/nas/runs/default", type=str)
    return p.parse_args()


def _arch_to_dict(arch) -> dict:
    def _layer_to_dict(l) -> dict:
        if l.op == "eca":
            return {
                "op": l.op,
                "eca_kernel": l.eca_kernel,
            }
        if l.op == "dsconv1d":
            return {
                "op": l.op,
                "channels": l.channels,
                "kernel_f": l.kernel_f,
                "stride_f": l.stride_f,
            }
        return {
            "op": l.op,
            "channels": l.channels,
            "kernel_t": l.kernel_t,
            "kernel_f": l.kernel_f,
            "stride_t": l.stride_t,
            "stride_f": l.stride_f,
        }

    return {
        "uid": arch.uid,
        "mfcc_window_ms": arch.mfcc_window_ms,
        "mfcc_stride_ms": arch.mfcc_stride_ms,
        "mfcc_n_mfcc": arch.mfcc_n_mfcc,
        "t_target": arch.t_target,
        "f_target": arch.f_target,
        "layers": [_layer_to_dict(l) for l in arch.layers],
    }


def main():
    args = parse_args()
    artifacts = run_search(args)
    out_dir = Path(args.nas_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    topk_path = out_dir / "topk.json"
    with open(topk_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "uid": r.arch.uid,
                    "acc": r.acc,
                    "total_mults": r.total_mults,
                    "frontend_mults": r.frontend_mults,
                    "backbone_mults": r.backbone_mults,
                    "params": r.params,
                    "reason": r.reason,
                    "arch": _arch_to_dict(r.arch),
                }
                for r in artifacts.topk
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    pareto_path = out_dir / "pareto.json"
    with open(pareto_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "uid": r.arch.uid,
                    "acc": r.acc,
                    "total_mults": r.total_mults,
                    "frontend_mults": r.frontend_mults,
                    "backbone_mults": r.backbone_mults,
                    "params": r.params,
                    "reason": r.reason,
                    "arch": _arch_to_dict(r.arch),
                }
                for r in artifacts.pareto
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    best = artifacts.topk[0] if artifacts.topk else None
    best_path = out_dir / "best.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "uid": best.arch.uid if best else None,
                "acc": best.acc if best else None,
                "total_mults": best.total_mults if best else None,
                "frontend_mults": best.frontend_mults if best else None,
                "backbone_mults": best.backbone_mults if best else None,
                "params": best.params if best else None,
                "reason": best.reason if best else "none",
                "arch": _arch_to_dict(best.arch) if best else None,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    summary = {
        "total_candidates": len(artifacts.all_results),
        "valid_candidates": len(artifacts.valid),
        "rejected_candidates": len(artifacts.all_results) - len(artifacts.valid),
        "pareto_count": len(artifacts.pareto),
        "topk_count": len(artifacts.topk),
        "reject_counter": artifacts.reject_counter,
        "best_uid": best.arch.uid if best else None,
        "best_acc": best.acc if best else None,
        "best_mults": best.total_mults if best else None,
        "best_params": best.params if best else None,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[NAS] done. outputs:")
    print(f"  - {out_dir / 'config.json'}")
    print(f"  - {out_dir / 'search_results.jsonl'}")
    print(f"  - {topk_path}")
    print(f"  - {pareto_path}")
    print(f"  - {best_path}")
    print(f"  - {summary_path}")


if __name__ == "__main__":
    main()
