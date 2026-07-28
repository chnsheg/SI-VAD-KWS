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
import torch

from vadbench.cli import (
    _cached_prediction,
    _check_manifest_lightweight,
    _load_config,
    _load_or_fit_algorithm,
    _model_stats_with_context,
    _prepare_run_dir,
    _write_json,
)
from vadbench.algorithms.neural import TorchFrameAlgorithm
from vadbench.manifest import read_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Fill missing cached eval predictions in bounded batches.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    torch.set_num_threads(1)
    config_path = Path(args.config)
    config = _load_config(config_path)
    manifest_path = Path(config["manifest"])
    base_dir = manifest_path.parent
    _check_manifest_lightweight(manifest_path)
    run_dir = _prepare_run_dir(config, config_path)
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    records = read_manifest(manifest_path, split=args.split)
    missing = []
    for record in records:
        label_len = len(np.load(record.resolve_label(base_dir), mmap_mode="r"))
        pred_path = pred_dir / f"{record.id}.npy"
        if not pred_path.exists():
            missing.append((record, label_len))
            continue
        try:
            pred_len = len(np.load(pred_path, mmap_mode="r"))
        except Exception:
            missing.append((record, label_len))
            continue
        if pred_len != label_len:
            missing.append((record, label_len))

    if not missing:
        print(f"[fill-pred] {args.split}: all {len(records)} predictions already cached", flush=True)
        return 0

    algorithm = _load_or_fit_algorithm(config, manifest_path, base_dir)
    if isinstance(algorithm, TorchFrameAlgorithm):
        _write_json(run_dir / "model_stats.json", _model_stats_with_context(algorithm.model_stats(frames=1), config))
    target_sr = getattr(algorithm, "sample_rate", None)
    eval_cfg = config.get("eval", {})
    todo = missing[: max(0, int(args.max_records))]
    print(f"[fill-pred] {args.split}: missing={len(missing)} processing={len(todo)} total={len(records)}", flush=True)
    for idx, (record, label_len) in enumerate(todo, start=1):
        _cached_prediction(
            algorithm,
            record,
            base_dir,
            target_sr,
            eval_cfg,
            pred_dir,
            expected_len=label_len,
        )
        if idx == 1 or idx % int(args.log_every) == 0 or idx == len(todo):
            print(f"[fill-pred] {args.split}: processed {idx}/{len(todo)} id={record.id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
