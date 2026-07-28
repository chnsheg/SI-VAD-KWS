from __future__ import annotations

import numpy as np


def binary_frame_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(bool).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float32).reshape(-1)
    length = min(len(y_true), len(y_score))
    y_true = y_true[:length]
    y_score = y_score[:length]
    y_pred = y_score >= float(threshold)

    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(length, 1)
    return {
        "frames": float(length),
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def aggregate_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    tp = sum(item["tp"] for item in metrics)
    tn = sum(item["tn"] for item in metrics)
    fp = sum(item["fp"] for item in metrics)
    fn = sum(item["fn"] for item in metrics)
    frames = sum(item["frames"] for item in metrics)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(frames, 1.0)
    return {
        "files": float(len(metrics)),
        "frames": float(frames),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def choose_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, dict[str, float]]:
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float32).reshape(-1)
    if len(y_true) == 0 or len(y_score) == 0:
        return 0.5, binary_frame_metrics(y_true, y_score, 0.5)
    length = min(len(y_true), len(y_score))
    y_true = y_true[:length]
    y_score = y_score[:length]
    candidates = np.unique(np.concatenate([np.linspace(0.05, 0.95, 91), np.quantile(y_score, np.linspace(0.05, 0.95, 19))]))
    best_threshold = 0.5
    best_metrics = binary_frame_metrics(y_true, y_score, best_threshold)
    for threshold in candidates:
        metrics = binary_frame_metrics(y_true, y_score, float(threshold))
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(bool).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    length = min(len(y_true), len(y_score))
    y_true = y_true[:length]
    y_score = y_score[:length]
    positives = int(np.sum(y_true))
    negatives = int(length - positives)
    if positives == 0 or negatives == 0:
        return 0.0
    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    sorted_true = y_true[order].astype(np.float64)
    _, first_idx, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    avg_ranks = first_idx.astype(np.float64) + (counts.astype(np.float64) + 1.0) / 2.0
    pos_counts = np.add.reduceat(sorted_true, first_idx)
    pos_rank_sum = float(np.sum(pos_counts * avg_ranks))
    return float((pos_rank_sum - positives * (positives + 1) / 2.0) / max(positives * negatives, 1))


def threshold_for_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.315) -> float:
    y_true = np.asarray(y_true).astype(bool).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float32).reshape(-1)
    length = min(len(y_true), len(y_score))
    y_true = y_true[:length]
    y_score = y_score[:length]
    negatives = y_score[~y_true]
    if len(negatives) == 0:
        return 0.5
    target_fpr = float(np.clip(target_fpr, 0.0, 1.0))
    if target_fpr >= 1.0:
        return float(np.min(negatives))
    sorted_neg = np.sort(negatives, kind="mergesort")
    unique_values, first_idx = np.unique(sorted_neg, return_index=True)
    fprs = (len(sorted_neg) - first_idx).astype(np.float64) / float(len(sorted_neg))
    valid = np.flatnonzero(fprs <= target_fpr)
    if len(valid) == 0:
        return float(np.nextafter(sorted_neg[-1], np.inf))
    return float(unique_values[int(valid[0])])


def tpr_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    y_true = np.asarray(y_true).astype(bool).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float32).reshape(-1)
    length = min(len(y_true), len(y_score))
    y_true = y_true[:length]
    y_score = y_score[:length]
    positives = int(np.sum(y_true))
    if positives == 0:
        return 0.0
    return float(np.sum(y_true & (y_score >= threshold)) / positives)


def ava_paper_metrics(
    y_class: np.ndarray,
    y_score: np.ndarray,
    target_fpr: float = 0.315,
    threshold: float | None = None,
) -> dict[str, float]:
    y_class = np.asarray(y_class, dtype=np.uint8).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float32).reshape(-1)
    length = min(len(y_class), len(y_score))
    y_class = y_class[:length]
    y_score = y_score[:length]
    y_binary = y_class > 0
    chosen_threshold = threshold_for_fpr(y_binary, y_score, target_fpr) if threshold is None else float(threshold)
    binary_metrics = binary_frame_metrics(y_binary.astype(np.uint8), y_score, chosen_threshold)
    return {
        "target_fpr": float(target_fpr),
        "threshold_at_fpr": float(chosen_threshold),
        "fpr": float(binary_metrics["fp"] / max(binary_metrics["fp"] + binary_metrics["tn"], 1.0)),
        "tpr_at_target_fpr": float(binary_metrics["recall"]),
        "tpr_at_fpr_0_315": float(binary_metrics["recall"]) if abs(float(target_fpr) - 0.315) < 1e-9 else float("nan"),
        "tpr_all": float(binary_metrics["recall"]),
        "tpr_clean": tpr_at_threshold(y_class == 1, y_score, chosen_threshold),
        "tpr_music": tpr_at_threshold(y_class == 2, y_score, chosen_threshold),
        "tpr_noise": tpr_at_threshold(y_class == 3, y_score, chosen_threshold),
        "auroc_all": roc_auc(y_binary, y_score),
    }
