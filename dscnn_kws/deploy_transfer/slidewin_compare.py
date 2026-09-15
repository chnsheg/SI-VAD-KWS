# -*- coding: utf-8 -*-
"""Slide-window comparison on mic243: float chain vs new strict-int8 deploy chain.

Chains
------
A. float  = export_float_onnx.py construction (TorchMFCC float32 exact replica +
   float L5_C64 DSCNN).  Mathematically identical to the delivered
   i22_a65_best_float.onnx (parity was verified at max |delta| ~1e-4).
B. int8   = export_v6_1_strict_int8_onnx.py construction
   (OnnxStrictIntegerMFCCFrontend + QDQ QAT backbone).  Mathematically identical
   to the exported *_v6_1_strict_int8.onnx.

Protocol (batch1_mic243_eval.py)
--------------------------------
1 s windows, 96 ms hop, strict batch-one, softmax positive score.

Metrics per recording: scores for both chains; per threshold: max consecutive
high run (platform width), number of high fragments (sawtooth indicator), peak,
first-order diff stats; annotation hit (one window within midpoint +/- 0.4 s).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)  # int64 算子多线程争抢反而严重劣化（880% CPU 且更慢）

REPO = Path("/home/chensheng/vad_kws_sources/reclean_v3/kws_trainer_20260910_v21bucket")
DEPLOY = Path("/home/chensheng/kws_deploy_transfer")
for p in (REPO, DEPLOY):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from export_float_onnx import (  # noqa: E402
    DSCNN as FloatDSCNN,
    OnnxFloatFullModel,
    OnnxFloatMFCCFrontend,
    calculate_time_steps,
    make_model_size_info,
)
from export_v6_1_strict_int8_onnx import (  # noqa: E402
    OnnxV61FullModel,
    build_qdq_backbone,
)
from onnx_strict_integer_mfcc import OnnxStrictIntegerMFCCFrontend  # noqa: E402

SAMPLE_RATE = 16000
WINDOW = SAMPLE_RATE
HOP = round(SAMPLE_RATE * 96 / 1000)  # 1536
THRESHOLDS = (0.50, 0.65, 0.70, 0.85)
POSITIVE = 1


def load_float_model(ckpt: Path) -> torch.nn.Module:
    dct_coeff = 13
    backbone = FloatDSCNN(
        input_dim=calculate_time_steps(SAMPLE_RATE, 32) * dct_coeff,
        label_count=2,
        model_size_info=make_model_size_info(5, 64),
        dct_coeff=dct_coeff,
    )
    try:
        obj = torch.load(ckpt, map_location="cpu", weights_only=True)
    except Exception:
        obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model" in obj:
        obj = obj["model"]
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    bb = {k.replace("backbone.", "", 1): v for k, v in obj.items() if k.startswith("backbone.")}
    backbone.load_state_dict(bb, strict=True)
    frontend = OnnxFloatMFCCFrontend(
        sample_rate=SAMPLE_RATE, n_mfcc=40, n_fft=512, win_length=512, hop_length=512,
        n_mels=40, f_min=20.0, f_max=8000.0, center=True, mel_filter_shape="triangular",
        log_offset=1e-6, log_input_clamp_min=1e-12, pre_emphasis=True, pre_emphasis_coeff=0.97,
    )
    return OnnxFloatFullModel(frontend, backbone, dct_coeff).cpu().eval()


def load_int8_model(spec: Path, ckpt: Path) -> torch.nn.Module:
    dct_coeff = 13
    frontend = OnnxStrictIntegerMFCCFrontend(spec)
    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
    except Exception:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    qdq, _report = build_qdq_backbone(
        state, num_layers=5, channels=64, label_count=2,
        sample_rate=SAMPLE_RATE, window_stride_ms=32, dct_coeff=dct_coeff,
    )
    return OnnxV61FullModel(frontend, qdq, dct_coeff).cpu().eval()


def slide_scores(model: torch.nn.Module, waveform: np.ndarray) -> tuple[list[int], list[float]]:
    starts = np.arange(0, max(0, len(waveform) - WINDOW) + 1, HOP, dtype=np.int64)
    scores: list[float] = []
    with torch.no_grad():
        for start in starts.tolist():
            window = torch.from_numpy(waveform[start : start + WINDOW]).unsqueeze(0)
            logits = model(window)
            scores.append(float(torch.softmax(logits.float(), dim=1)[0, POSITIVE].item()))
    return starts.tolist(), scores


def curve_stats(scores: list[float], threshold: float) -> dict:
    values = np.asarray(scores, dtype=np.float64)
    high = values >= threshold
    max_run = 0
    fragments = 0
    current = 0
    for flag in high:
        if flag:
            current += 1
            if current == 1:
                fragments += 1
            max_run = max(max_run, current)
        else:
            current = 0
    diffs = np.abs(np.diff(values)) if len(values) > 1 else np.zeros(1)
    return {
        "peak": float(values.max()) if len(values) else 0.0,
        "high_frames": int(high.sum()),
        "max_high_run": int(max_run),
        "high_fragments": int(fragments),
        "diff_mean": float(diffs.mean()),
        "diff_p90": float(np.quantile(diffs, 0.9)),
        "diff_max": float(diffs.max()),
    }


def load_audio(path: Path) -> np.ndarray:
    import torchaudio

    wav, sr = torchaudio.load(str(path))
    if sr != SAMPLE_RATE:
        import torchaudio.functional as AF

        wav = AF.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav[0].numpy().astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--float_ckpt", type=Path, default=Path("/home/chensheng/kws_training_runs/kws_reclean_v2/i22_a65_int8_input/best.pt"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--int8_ckpt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("/home/chensheng/kws_training_runs/kws_reclean_v2/v2_final_s160_locality_20260830/mic243_evaluation_only_manifest_20260830.jsonl"))
    parser.add_argument("--audio_root", type=Path, default=Path("/home/chensheng/VAD-KWS/experiments/microphone_domain_adaptation_20260805/data/microphone_domain_20260806/data/vad"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 = all recordings")
    parser.add_argument("--train_manifest", type=Path, default=None, help="optional positive-train sample comparison")
    parser.add_argument("--train_samples", type=int, default=0)
    args = parser.parse_args()

    print("[INFO] loading float model ...", flush=True)
    float_model = load_float_model(args.float_ckpt)
    print("[INFO] loading int8 model ...", flush=True)
    int8_model = load_int8_model(args.spec, args.int8_ckpt)

    # sanity check on synthetic silence
    silence = torch.zeros(1, WINDOW)
    with torch.no_grad():
        f0 = torch.softmax(float_model(silence).float(), dim=1)[0, POSITIVE].item()
        i0 = torch.softmax(int8_model(silence).float(), dim=1)[0, POSITIVE].item()
    print(f"[SANITY] silence scores: float={f0:.6f} int8={i0:.6f}", flush=True)

    recordings: dict[str, list[dict]] = {}
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        recordings.setdefault(str(item["recording_id"]), []).append(item)
    keys = sorted(recordings.keys())
    if args.limit:
        keys = keys[: args.limit]
    print(f"[INFO] recordings={len(keys)}", flush=True)

    results = []
    started = time.time()
    for idx, rec_id in enumerate(keys):
        audio = load_audio(args.audio_root / f"{rec_id}.wav")
        f_starts, f_scores = slide_scores(float_model, audio)
        i_starts, i_scores = slide_scores(int8_model, audio)
        assert f_starts == i_starts
        annotations = [
            {"id": str(a["id"]), "start_sample": int(a["start_sample"]), "end_sample": int(a["end_sample"])}
            for a in recordings[rec_id]
        ]
        per_threshold = {}
        for threshold in THRESHOLDS:
            fs = curve_stats(f_scores, threshold)
            is_ = curve_stats(i_scores, threshold)
            # annotation hit (one-window, centers within midpoint +/- 0.4 s)
            centers = np.asarray(f_starts, dtype=np.int64) + WINDOW // 2
            f_hit = 0
            i_hit = 0
            f_high = np.asarray(f_scores) >= threshold
            i_high = np.asarray(i_scores) >= threshold
            for a in annotations:
                midpoint = (a["start_sample"] + a["end_sample"]) // 2
                in_range = (centers >= midpoint - 6400) & (centers <= midpoint + 6400)
                f_hit += int(bool(np.any(f_high & in_range)))
                i_hit += int(bool(np.any(i_high & in_range)))
            per_threshold[f"{threshold:.2f}"] = {
                "float": fs, "int8": is_,
                "float_hit": f_hit, "int8_hit": i_hit, "annotations": len(annotations),
            }
        results.append({
            "recording_id": rec_id,
            "duration_s": len(audio) / SAMPLE_RATE,
            "starts": f_starts,
            "float_scores": f_scores,
            "int8_scores": i_scores,
            "annotations": annotations,
            "thresholds": per_threshold,
        })
        print(f"[{idx + 1}/{len(keys)}] {rec_id} elapsed={time.time() - started:.0f}s", flush=True)

    summary = {f"{t:.2f}": {"float_hit": 0, "int8_hit": 0, "annotations": 0} for t in THRESHOLDS}
    for row in results:
        for key, value in row["thresholds"].items():
            summary[key]["float_hit"] += value["float_hit"]
            summary[key]["int8_hit"] += value["int8_hit"]
            summary[key]["annotations"] += value["annotations"]
    for key, value in summary.items():
        value["float_recall"] = value["float_hit"] / max(1, value["annotations"])
        value["int8_recall"] = value["int8_hit"] / max(1, value["annotations"])

    payload = {
        "protocol": {"window_ms": 1000, "hop_ms": 96, "thresholds": list(THRESHOLDS),
                     "float_chain": "TorchMFCC float32 + float DSCNN (= delivered float ONNX)",
                     "int8_chain": "OnnxStrictIntegerMFCC + QDQ QAT backbone (= v6.1 strict int8 ONNX)"},
        "summary": summary,
        "recordings": results,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    print("[DONE]", json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
