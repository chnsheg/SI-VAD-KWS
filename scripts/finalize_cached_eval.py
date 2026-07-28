from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from vadbench.cli import (
    _accumulate_source_metrics,
    _check_manifest_lightweight,
    _finalize_source_metrics,
    _load_config,
    _prepare_run_dir,
    _write_json,
    _write_segments,
)
from vadbench.frame_prediction import FramePrediction
from vadbench.features import align_length
from vadbench.manifest import read_manifest
from vadbench.metrics import aggregate_metrics, binary_frame_metrics, choose_best_threshold, roc_auc, threshold_for_fpr


def _load_cached(pred_dir: Path, record_id: str, expected_len: int) -> np.ndarray:
    pred_path = pred_dir / f"{record_id}.npy"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing cached prediction: {pred_path}")
    scores = np.load(pred_path).astype(np.float32)
    return align_length(scores, expected_len, pad_value=0.0).astype(np.float32)


def _load_split_arrays(records, base_dir: Path, pred_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    labels_all = []
    scores_all = []
    for record in records:
        labels = np.load(record.resolve_label(base_dir)).astype(np.uint8)
        scores = _load_cached(pred_dir, record.id, len(labels))
        labels_all.append(labels)
        scores_all.append(scores)
    if not labels_all:
        return np.asarray([], dtype=np.uint8), np.asarray([], dtype=np.float32)
    return np.concatenate(labels_all), np.concatenate(scores_all)


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize eval metrics from cached prediction .npy files.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--write-segments", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = _load_config(config_path)
    manifest_path = Path(config["manifest"])
    base_dir = manifest_path.parent
    _check_manifest_lightweight(manifest_path)
    run_dir = _prepare_run_dir(config, config_path)
    pred_dir = run_dir / "predictions"
    eval_cfg = config.get("eval", {})
    split = eval_cfg.get("split", "test")
    tune_split = eval_cfg.get("tune_split", "val")
    records = read_manifest(manifest_path, split=split)
    tune_records = read_manifest(manifest_path, split=tune_split)

    post = config.get("postprocess", {})
    threshold_cfg = post.get("threshold", None)
    if bool(eval_cfg.get("tune_threshold", False)) or threshold_cfg == "auto":
        y_val, p_val = _load_split_arrays(tune_records, base_dir, pred_dir)
        threshold, _ = choose_best_threshold(y_val, p_val)
    elif threshold_cfg is not None:
        threshold = float(threshold_cfg)
    else:
        threshold = 0.5

    min_speech_ms = float(post.get("min_speech_ms", 60.0))
    min_silence_ms = float(post.get("min_silence_ms", 100.0))
    frame_metrics = []
    per_file = []
    segment_rows = []
    source_counts: dict[str, dict[str, float]] = {}
    labels_all = []
    scores_all = []
    for idx, record in enumerate(records, start=1):
        labels = np.load(record.resolve_label(base_dir)).astype(np.uint8)
        scores = _load_cached(pred_dir, record.id, len(labels))
        metrics = binary_frame_metrics(labels, scores, threshold)
        _accumulate_source_metrics(source_counts, record, labels, scores, threshold)
        frame_metrics.append(metrics)
        per_file.append({"id": record.id, **metrics})
        if args.write_segments:
            segment_rows.append(
                (
                    record.id,
                    FramePrediction(scores=scores, frame_hop_ms=record.frame_hop_ms, source_id=record.id).to_segments(
                        threshold=threshold,
                        min_speech_ms=min_speech_ms,
                        min_silence_ms=min_silence_ms,
                    ),
                )
            )
        labels_all.append(labels)
        scores_all.append(scores)
        if idx == 1 or idx % int(eval_cfg.get("log_every_records", 500) or 500) == 0 or idx == len(records):
            print(f"[finalize] processed {idx}/{len(records)} records", flush=True)

    aggregate = aggregate_metrics(frame_metrics)
    aggregate["threshold"] = float(threshold)
    y_test = np.concatenate(labels_all)
    p_test = np.concatenate(scores_all)
    target_fpr = float(eval_cfg.get("target_fpr", 0.315))
    paper_threshold = threshold_for_fpr(y_test, p_test, target_fpr)
    paper_frame = binary_frame_metrics(y_test, p_test, paper_threshold)
    paper = {
        "target_fpr": target_fpr,
        "threshold_at_fpr": float(paper_threshold),
        "fpr": float(paper_frame["fp"] / max(paper_frame["fp"] + paper_frame["tn"], 1.0)),
        "tpr_at_target_fpr": float(paper_frame["recall"]),
        "tpr_at_fpr_0_315": float(paper_frame["recall"]) if abs(target_fpr - 0.315) < 1e-9 else float("nan"),
        "tpr_all": float(paper_frame["recall"]),
        "auroc_all": roc_auc(y_test, p_test),
    }
    y_val, p_val = _load_split_arrays(tune_records, base_dir, pred_dir)
    val_fpr_threshold = threshold_for_fpr(y_val, p_val, target_fpr)
    val_frame = binary_frame_metrics(y_test, p_test, val_fpr_threshold)
    paper_val_threshold = {
        "target_fpr": target_fpr,
        "threshold_at_fpr": float(val_fpr_threshold),
        "fpr": float(val_frame["fp"] / max(val_frame["fp"] + val_frame["tn"], 1.0)),
        "tpr_at_target_fpr": float(val_frame["recall"]),
        "tpr_at_fpr_0_315": float(val_frame["recall"]) if abs(target_fpr - 0.315) < 1e-9 else float("nan"),
        "tpr_all": float(val_frame["recall"]),
        "auroc_all": paper["auroc_all"],
    }

    result = {
        "aggregate": aggregate,
        "source_metrics": _finalize_source_metrics(source_counts),
        "paper": paper,
        "paper_val_threshold": paper_val_threshold,
    }
    if args.write_segments:
        _write_segments(run_dir / "segments.csv", segment_rows)
    _write_json(run_dir / "per_file_metrics.json", per_file)
    _write_json(run_dir / "paper_metrics.json", paper)
    _write_json(run_dir / "paper_metrics_val_threshold.json", paper_val_threshold)
    _write_json(run_dir / "metrics.json", result)
    print(f"Wrote metrics: {run_dir / 'metrics.json'}", flush=True)
    print(
        {
            "f1": aggregate["f1"],
            "precision": aggregate["precision"],
            "recall": aggregate["recall"],
            "accuracy": aggregate["accuracy"],
            "auroc_all": paper["auroc_all"],
            "tpr_at_fpr_0_315": paper["tpr_at_fpr_0_315"],
        },
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
