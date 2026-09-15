# -*- coding: utf-8 -*-
"""绘制 slidewin_compare.py 结果：浮点链路 vs 新严格整数 INT8 链路的滑窗输出对比。

输入: slidewin_compare.py 产出的 JSON
输出: PNG 图组
  1. overview_curves.png      代表性录音的分数曲线对比（平台 vs 锯齿直观图）
  2. platform_width_hist.png  平台宽度(max_high_run)分布对比
  3. sawtooth_metrics.png     锯齿指标（高分数碎片数 / 一阶差分 p90）对比
  4. recall_summary.png       各阈值标注命中率对比
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THRESHOLDS = ("0.50", "0.65", "0.70", "0.85")
COLOR_FLOAT = "#1f77b4"
COLOR_INT8 = "#d62728"


def _hop_ms(payload: dict) -> float:
    return float(payload["protocol"]["hop_ms"])


def pick_representative(recs: list[dict], threshold: str = "0.50", count: int = 6) -> list[dict]:
    """挑代表性录音: int8 高分碎片多(锯齿)的、平台良好的、以及分数差异最大的。"""

    def sawtooth_score(r: dict) -> float:
        st = r["thresholds"][threshold]["int8"]
        fs = r["thresholds"][threshold]["float"]
        frag_gap = st["high_fragments"] - fs["high_fragments"]
        diff_gap = st["diff_p90"] - fs["diff_p90"]
        return frag_gap + 40.0 * diff_gap

    ranked = sorted(recs, key=sawtooth_score, reverse=True)
    worst = ranked[: count // 2]
    best = ranked[-(count - count // 2) :]
    seen: set[str] = set()
    picked: list[dict] = []
    for r in worst + best:
        if r["recording_id"] not in seen:
            seen.add(r["recording_id"])
            picked.append(r)
    return picked


def plot_curves(payload: dict, recs: list[dict], out: Path, threshold: float) -> None:
    hop = _hop_ms(payload)
    n = len(recs)
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.2 * rows), squeeze=False)
    for idx, r in enumerate(recs):
        ax = axes[idx // cols][idx % cols]
        t = np.asarray(r["starts"]) / 16000.0
        ax.plot(t, r["float_scores"], color=COLOR_FLOAT, lw=1.2, label="float chain")
        ax.plot(t, r["int8_scores"], color=COLOR_INT8, lw=1.2, alpha=0.85, label="int8 strict chain")
        ax.axhline(threshold, color="gray", ls="--", lw=0.8)
        for a in r["annotations"]:
            ax.axvspan(a["start_sample"] / 16000.0, a["end_sample"] / 16000.0, color="green", alpha=0.12)
        st = r["thresholds"][f"{threshold:.2f}"]
        ax.set_title(
            f"{r['recording_id']}  float:run={st['float']['max_high_run']} frag={st['float']['high_fragments']}"
            f"  int8:run={st['int8']['max_high_run']} frag={st['int8']['high_fragments']}",
            fontsize=9,
        )
        ax.set_xlabel("window center time (s)" if idx // cols == rows - 1 else "")
        ax.set_ylabel("wake score")
        ax.set_ylim(-0.02, 1.05)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"Sliding-window scores: float vs strict-int8 (threshold={threshold}, green=keyword span)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_platform_hist(payload: dict, recs: list[dict], out: Path, threshold: str) -> None:
    hop = _hop_ms(payload)
    fr = [r["thresholds"][threshold]["float"]["max_high_run"] for r in recs]
    ir = [r["thresholds"][threshold]["int8"]["max_high_run"] for r in recs]
    bins = np.arange(0, max(max(fr), max(ir)) + 3) - 0.5
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(fr, bins=bins, alpha=0.6, color=COLOR_FLOAT, label="float chain")
    ax.hist(ir, bins=bins, alpha=0.6, color=COLOR_INT8, label="int8 strict chain")
    ax.set_xlabel(f"platform width = max consecutive high frames @th={threshold}  (1 frame = {hop:.0f} ms)")
    ax.set_ylabel("recording count")
    ax.set_title("Platform width distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_sawtooth(payload: dict, recs: list[dict], out: Path, threshold: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ff = [r["thresholds"][threshold]["float"]["high_fragments"] for r in recs]
    fi = [r["thresholds"][threshold]["int8"]["high_fragments"] for r in recs]
    fd = [r["thresholds"][threshold]["float"]["diff_p90"] for r in recs]
    idv = [r["thresholds"][threshold]["int8"]["diff_p90"] for r in recs]
    axes[0].scatter(ff, fi, s=18, alpha=0.7)
    lim = max(max(ff), max(fi), 1)
    axes[0].plot([0, lim], [0, lim], "k--", lw=0.8)
    axes[0].set_xlabel("float chain: high fragments")
    axes[0].set_ylabel("int8 chain: high fragments")
    axes[0].set_title("Sawtooth indicator: fragment count (below diagonal = int8 smoother)")
    axes[1].scatter(fd, idv, s=18, alpha=0.7, color="seagreen")
    lim2 = max(max(fd), max(idv), 1e-6)
    axes[1].plot([0, lim2], [0, lim2], "k--", lw=0.8)
    axes[1].set_xlabel("float chain: |diff| p90")
    axes[1].set_ylabel("int8 chain: |diff| p90")
    axes[1].set_title("Score jitter: first-order diff p90")
    fig.suptitle(f"Sawtooth metrics @th={threshold}")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_recall(payload: dict, out: Path) -> None:
    summary = payload["summary"]
    xs = np.arange(len(THRESHOLDS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    fr = [summary[t]["float_recall"] * 100 for t in THRESHOLDS]
    ir = [summary[t]["int8_recall"] * 100 for t in THRESHOLDS]
    ax.bar(xs - width / 2, fr, width, color=COLOR_FLOAT, label="float chain")
    ax.bar(xs + width / 2, ir, width, color=COLOR_INT8, label="int8 strict chain")
    for x, v in zip(xs, fr):
        ax.text(x - width / 2, v + 1, f"{v:.1f}", ha="center", fontsize=8)
    for x, v in zip(xs, ir):
        ax.text(x + width / 2, v + 1, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"th={t}" for t in THRESHOLDS])
    ax.set_ylabel("annotation hit rate (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Keyword annotation hit rate (one window within midpoint ±0.4s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent / "slidewin_plots")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_curves", type=int, default=6)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    recs = payload["recordings"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    key = f"{args.threshold:.2f}"

    picks = pick_representative(recs, key, args.max_curves)
    plot_curves(payload, picks, args.out_dir / "overview_curves.png", args.threshold)
    plot_platform_hist(payload, recs, args.out_dir / "platform_width_hist.png", key)
    plot_sawtooth(payload, recs, args.out_dir / "sawtooth_metrics.png", key)
    plot_recall(payload, args.out_dir / "recall_summary.png")

    fr_all = np.mean([r["thresholds"][key]["float"]["max_high_run"] for r in recs])
    ir_all = np.mean([r["thresholds"][key]["int8"]["max_high_run"] for r in recs])
    print(f"[DONE] recordings={len(recs)}  mean platform width @th={key}: float={fr_all:.2f} int8={ir_all:.2f}")
    print(f"[DONE] plots -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
