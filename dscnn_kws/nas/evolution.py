from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .constraints import check_arch_constraints
from .evaluator import train_eval_candidate
from .pareto import ParetoItem, pareto_front
from .search_space import NASArchitecture, mutate_architecture, sample_random_architecture


@dataclass
class SearchResult:
    arch: NASArchitecture
    acc: float
    total_mults: int
    frontend_mults: int
    backbone_mults: int
    params: int
    reason: str


@dataclass
class SearchArtifacts:
    topk: list[SearchResult]
    valid: list[SearchResult]
    all_results: list[SearchResult]
    pareto: list[SearchResult]
    reject_counter: dict[str, int]


def _reason_category(reason: str) -> str:
    if reason.startswith("mult_soft_exceed"):
        return "mult_soft_exceed"
    if reason.startswith("mult_exceed"):
        return "mult_exceed"
    if reason.startswith("param_exceed"):
        return "param_exceed"
    if reason.startswith("volume_exceed"):
        return "volume_exceed"
    if reason.startswith("channel_jump"):
        return "channel_jump"
    if reason.startswith("stem_t_out"):
        return "stem_t_out"
    if reason.startswith("stem_f_out"):
        return "stem_f_out"
    if reason.startswith("over_downsample"):
        return "over_downsample"
    if reason.startswith("invalid"):
        return "invalid_shape"
    return "other"


def _arch_to_dict(arch: NASArchitecture) -> dict:
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


def _print_candidate_log(idx: int, total: int, stage: str, result: SearchResult) -> None:
    print(
        f"[NAS][{stage}] [{idx + 1}/{total}] {result.arch.uid} | "
        f"reason={result.reason} | acc={result.acc:.4f} | mults={result.total_mults} "
        f"(frontend={result.frontend_mults}, backbone={result.backbone_mults}) | params={result.params}"
    )


def _print_periodic_summary(results: list[SearchResult], reject_counter: Counter) -> None:
    valid = [r for r in results if r.reason == "ok"]
    if not valid:
        print(f"[NAS][summary] valid=0, rejected={sum(reject_counter.values())}, reject_reasons={dict(reject_counter)}")
        return

    best_acc = max(valid, key=lambda r: r.acc)
    best_mults = min(valid, key=lambda r: r.total_mults)
    best_params = min(valid, key=lambda r: r.params)
    items = [ParetoItem(uid=r.arch.uid, acc=r.acc, mults=r.total_mults, params=r.params, payload={"r": r}) for r in valid]
    front = pareto_front(items)
    print(
        f"[NAS][summary] valid={len(valid)} rejected={sum(reject_counter.values())} pareto={len(front)} | "
        f"best_acc={best_acc.arch.uid}:{best_acc.acc:.4f} | "
        f"best_mults={best_mults.arch.uid}:{best_mults.total_mults} | "
        f"best_params={best_params.arch.uid}:{best_params.params}"
    )
    if reject_counter:
        print(f"[NAS][summary] reject_reasons={dict(reject_counter)}")


def run_search(args) -> SearchArtifacts:
    rng = random.Random(args.seed)
    results: list[SearchResult] = []
    reject_counter: Counter = Counter()
    total = args.nas_total_samples

    def eval_one(arch: NASArchitecture, idx: int) -> SearchResult:
        arch.uid = f"cand_{idx:05d}"
        ok, reason, cost = check_arch_constraints(
            arch,
            mult_limit=args.nas_mult_limit,
            mult_limit_parent=args.nas_mult_limit_parent,
            param_limit=args.nas_param_limit,
            sample_rate=args.sample_rate,
        )
        if not ok:
            reject_counter[_reason_category(reason)] += 1
            return SearchResult(
                arch=arch,
                acc=0.0,
                total_mults=max(0, cost.total_mults),
                frontend_mults=max(0, cost.frontend_mults),
                backbone_mults=max(0, cost.backbone_mults),
                params=max(0, cost.params),
                reason=reason,
            )
        ev = train_eval_candidate(arch=arch, args=args, epochs=args.nas_epochs_per_candidate, seed=args.seed + idx)
        return SearchResult(
            arch=arch,
            acc=float(ev["acc"]),
            total_mults=cost.total_mults,
            frontend_mults=cost.frontend_mults,
            backbone_mults=cost.backbone_mults,
            params=cost.params,
            reason=reason,
        )

    def sample_feasible(num_layers: int) -> NASArchitecture:
        max_tries = max(1, int(getattr(args, "nas_sampling_max_tries", 30)))
        best_arch = None
        best_mults = 10**18
        for _ in range(max_tries):
            arch = sample_random_architecture(
                num_layers=num_layers,
                rng=rng,
                t_target=args.nas_t_target,
                f_target=args.nas_f_target,
                sample_rate=args.sample_rate,
            )
            ok, _, cost = check_arch_constraints(
                arch,
                mult_limit=args.nas_mult_limit,
                mult_limit_parent=args.nas_mult_limit_parent,
                param_limit=args.nas_param_limit,
                sample_rate=args.sample_rate,
            )
            if ok:
                return arch
            if 0 < cost.total_mults < best_mults:
                best_arch = arch
                best_mults = cost.total_mults
        return best_arch if best_arch is not None else sample_random_architecture(
            num_layers=num_layers,
            rng=rng,
            t_target=args.nas_t_target,
            f_target=args.nas_f_target,
            sample_rate=args.sample_rate,
        )

    for i in range(args.nas_init_samples):
        arch = sample_feasible(num_layers=args.nas_num_layers)
        r = eval_one(arch, i)
        results.append(r)
        _print_candidate_log(i, total, "init", r)
        if (i + 1) % max(1, args.nas_log_interval) == 0:
            _print_periodic_summary(results, reject_counter)

    cursor = args.nas_init_samples
    while cursor < args.nas_total_samples:
        valid = [r for r in results if r.reason in {"ok", "mult_soft_exceed"}]
        if not valid:
            parent = sample_feasible(num_layers=args.nas_num_layers)
        else:
            items = [ParetoItem(uid=r.arch.uid, acc=r.acc, mults=r.total_mults, params=r.params, payload={"r": r}) for r in valid]
            front = pareto_front(items)
            picked = rng.choice(front)
            parent = picked.payload["r"].arch

        child = mutate_architecture(parent, rng=rng, mutation_prob=args.nas_mutation_prob)
        r = eval_one(child, cursor)
        results.append(r)
        _print_candidate_log(cursor, total, "evo", r)
        if (cursor + 1) % max(1, args.nas_log_interval) == 0:
            _print_periodic_summary(results, reject_counter)
        cursor += 1

    out_dir = Path(args.nas_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "search_results.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(
                json.dumps(
                    {
                        "uid": r.arch.uid,
                        "acc": r.acc,
                        "total_mults": r.total_mults,
                        "frontend_mults": r.frontend_mults,
                        "backbone_mults": r.backbone_mults,
                        "params": r.params,
                        "reason": r.reason,
                        "arch": _arch_to_dict(r.arch),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    valid = [r for r in results if r.reason == "ok"]
    items = [ParetoItem(uid=r.arch.uid, acc=r.acc, mults=r.total_mults, params=r.params, payload={"r": r}) for r in valid]
    pareto = [x.payload["r"] for x in pareto_front(items)] if items else []
    valid.sort(key=lambda x: x.acc, reverse=True)
    topk = valid[: args.nas_topk]
    return SearchArtifacts(
        topk=topk,
        valid=valid,
        all_results=results,
        pareto=pareto,
        reject_counter=dict(reject_counter),
    )
