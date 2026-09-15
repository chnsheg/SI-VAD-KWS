from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from dscnn_kws.frontend import BitAccurateMFCCFrontend
from dscnn_kws.quantization import qat_bit_accurate_mfcc_accuracy_first as qbf
from dscnn_kws.quantization.qat_int8_noise_snr_scene import (
    choose_backend,
    infer_arch_from_state_dict,
    load_state_dict,
    load_state_dict_to_model,
    quantize_qat_model,
    save_csv,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def unique_model_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (
            row.get("dataset", ""),
            row.get("inference_checkpoint", ""),
            row.get("bit_accurate_mfcc_spec_json", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def scale_spec_path_for_row(row: dict[str, str], scale_spec_dir: Path, suffix: str) -> Path:
    original = Path(row["bit_accurate_mfcc_spec_json"])
    marker = "_bit_accurate_mfcc_spec.json"
    if original.name.endswith(marker):
        name = f"{original.name[: -len(marker)]}{suffix}"
    else:
        name = f"{original.stem}{suffix}"
    return scale_spec_dir / name


def prepared_checkpoint_for_inference(inference_checkpoint: Path) -> Path:
    marker = "_bit_accurate_mfcc_int8_backbone.pt"
    name = inference_checkpoint.name
    if name.endswith(marker):
        return inference_checkpoint.with_name(f"{name[: -len(marker)]}_bit_accurate_qat_prepared_best.pt")
    return inference_checkpoint.with_name(name.replace("mfcc_int8_backbone", "qat_prepared_best"))


def value_from_spec(config: dict[str, Any], key: str, fallback: Any) -> Any:
    return config[key] if key in config and config[key] is not None else fallback


def runtime_args_from_spec(base_args: argparse.Namespace, scale_spec: Path) -> argparse.Namespace:
    args = copy.copy(base_args)
    payload = json.loads(scale_spec.read_text(encoding="utf-8"))
    config = payload.get("config", {})
    sample_rate = int(value_from_spec(config, "sample_rate", args.sample_rate))
    n_fft = int(value_from_spec(config, "n_fft", int(sample_rate * args.window_size_ms / 1000)))
    hop_length = int(value_from_spec(config, "hop_length", int(sample_rate * args.window_stride_ms / 1000)))

    args.sample_rate = sample_rate
    args.window_size_ms = int(round(n_fft * 1000.0 / float(sample_rate)))
    args.window_stride_ms = int(round(hop_length * 1000.0 / float(sample_rate)))
    args.constraint_profile = str(value_from_spec(config, "constraint_profile", args.constraint_profile))
    args.mel_filter_shape = str(value_from_spec(config, "mel_filter_shape", args.mel_filter_shape))
    args.pre_emphasis = bool(value_from_spec(config, "pre_emphasis", args.pre_emphasis))
    args.pre_emphasis_coeff = float(value_from_spec(config, "pre_emphasis_coeff", args.pre_emphasis_coeff))
    args.log_pwl_num_segments = int(value_from_spec(config, "log_pwl_num_segments", args.log_pwl_num_segments))
    args.log_pwl_strategy = str(value_from_spec(config, "log_pwl_strategy", args.log_pwl_strategy))
    args.log_pwl_gamma = float(value_from_spec(config, "log_pwl_gamma", args.log_pwl_gamma))
    args.log_offset = float(value_from_spec(config, "log_offset", args.log_offset))
    args.log_input_clamp_min = float(value_from_spec(config, "log_input_clamp_min", args.log_input_clamp_min))
    args.mfcc_per_channel_scale = bool(
        config.get("stage_quant", {}).get("mfcc", {}).get("per_channel", args.mfcc_per_channel_scale)
    )
    args.mfcc_output_scale = None
    args.stage_bit_overrides_normalized = {}
    return args


def load_scale_quant_metadata(scale_spec: Path) -> dict[str, Any]:
    payload = json.loads(scale_spec.read_text(encoding="utf-8"))
    return payload.get("extra", {}).get("scale_quantization", {})


def build_inference_model(
    args: argparse.Namespace,
    *,
    prepared_checkpoint: Path,
    scale_spec: Path,
) -> tuple[qbf.BitAccurateMFCCQuantizedBackboneModel, int, int, int]:
    prepared_state = load_state_dict(prepared_checkpoint)
    layers, channels, label_count = infer_arch_from_state_dict(prepared_state)
    model = qbf.build_model(args, num_layers=layers, channels=channels, label_count=label_count)
    model = qbf.prepare_model_for_qat(model, args)
    load_state_dict_to_model(model, prepared_state)
    quantized_qat_model = quantize_qat_model(model)
    frontend = BitAccurateMFCCFrontend.from_spec_json(scale_spec).cpu().eval()
    inference_model = qbf.BitAccurateMFCCQuantizedBackboneModel(
        frontend=frontend,
        quantized_backbone=quantized_qat_model.backbone,
        dct_coeff=args.dct_coeff,
    ).cpu()
    inference_model.eval()
    return inference_model, layers, channels, label_count


def run_one_model(
    base_args: argparse.Namespace,
    row: dict[str, str],
    *,
    scale_spec_dir: Path,
    suffix: str,
) -> list[dict[str, Any]]:
    scale_spec = scale_spec_path_for_row(row, scale_spec_dir, suffix)
    if not scale_spec.exists():
        raise FileNotFoundError(f"Scale-quantized spec not found: {scale_spec}")

    inference_checkpoint = Path(row["inference_checkpoint"])
    prepared_checkpoint = prepared_checkpoint_for_inference(inference_checkpoint)
    if not prepared_checkpoint.exists():
        raise FileNotFoundError(
            f"Prepared QAT checkpoint not found: {prepared_checkpoint}. "
            "The first scale-U16 experiment expects the v2 prepared checkpoint to rebuild the INT8 backbone."
        )

    args = runtime_args_from_spec(base_args, scale_spec)
    model, layers, channels, label_count = build_inference_model(
        args,
        prepared_checkpoint=prepared_checkpoint,
        scale_spec=scale_spec,
    )
    dataset = row["dataset"]
    arch = row.get("arch") or f"L{layers}_C{channels}"
    expected_params = int(float(row.get("expected_params") or qbf.expected_params(layers, channels, label_count)))
    scale_meta = load_scale_quant_metadata(scale_spec)
    source_checkpoint = Path(row.get("source_checkpoint") or prepared_checkpoint)

    rows = qbf.run_scene_grid(
        args,
        model=model,
        dataset=dataset,
        arch=arch,
        layers=layers,
        channels=channels,
        expected_param_count=expected_params,
        source_checkpoint=source_checkpoint,
        bit_accurate_spec_json=scale_spec,
        inference_checkpoint=inference_checkpoint,
    )
    for out_row in rows:
        out_row.update(
            {
                "frontend": "bit_accurate_mfcc_scale_u16",
                "constraint_profile": row.get("constraint_profile", ""),
                "baseline_bit_accurate_mfcc_spec_json": row.get("bit_accurate_mfcc_spec_json", ""),
                "scale_u16_bit_accurate_mfcc_spec_json": str(scale_spec),
                "scale_quant_mode": scale_meta.get("mode", "u_shift"),
                "scale_bits": scale_meta.get("scale_bits", base_args.scale_bits),
                "prepared_qat_checkpoint": str(prepared_checkpoint),
                "baseline_inference_checkpoint": str(inference_checkpoint),
                "log_approx_mode": row.get("log_approx_mode", "pwl"),
                "log_pwl_num_segments": row.get("log_pwl_num_segments", args.log_pwl_num_segments),
                "mfcc_per_channel_scale": row.get("mfcc_per_channel_scale", args.mfcc_per_channel_scale),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate scale-quantized bit-accurate MFCC specs with the existing v2 INT8 backbone."
    )
    parser.add_argument(
        "--baseline_grid_csv",
        default="dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_coeff_quant/hardware_coeff_baseline_grid_results.csv",
    )
    parser.add_argument(
        "--scale_spec_dir",
        default="dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_scale_u16/scale_u16_models",
    )
    parser.add_argument(
        "--output_csv",
        default="dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_scale_u16/scale_u16_grid_results.csv",
    )
    parser.add_argument("--scale_spec_suffix", default="_scale_u16_bit_accurate_mfcc_spec.json")
    parser.add_argument("--limit_models", type=int, default=0)

    parser.add_argument("--root", default=qbf.ROOT)
    parser.add_argument("--batch", type=int, default=qbf.BATCH)
    parser.add_argument("--num_workers", type=int, default=qbf.NUM_WORKERS)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=qbf.SEED)
    parser.add_argument("--backend", default=qbf.BACKEND)

    parser.add_argument("--scene_test_root", default=qbf.SCENE_TEST_ROOT)
    parser.add_argument("--scene_names", nargs="+", default=qbf.SCENE_NAMES)
    parser.add_argument("--test_snrs", nargs="+", type=float, default=qbf.TEST_SNRS)

    parser.add_argument("--sample_rate", type=int, default=qbf.SAMPLE_RATE)
    parser.add_argument("--dct_coeff", type=int, default=qbf.DCT_COEFF)
    parser.add_argument("--window_size_ms", type=int, default=qbf.WINDOW_SIZE_MS)
    parser.add_argument("--window_stride_ms", type=int, default=qbf.WINDOW_STRIDE_MS)
    parser.add_argument("--constraint_profile", default=qbf.CONSTRAINT_PROFILE)
    parser.add_argument("--mel_filter_shape", choices=["rectangular"], default=qbf.MEL_FILTER_SHAPE)
    parser.add_argument("--pre_emphasis", action=argparse.BooleanOptionalAction, default=qbf.PRE_EMPHASIS)
    parser.add_argument("--pre_emphasis_coeff", type=float, default=qbf.PRE_EMPHASIS_COEFF)
    parser.add_argument("--log_pwl_num_segments", type=int, default=qbf.LOG_PWL_NUM_SEGMENTS)
    parser.add_argument("--log_pwl_strategy", default=qbf.LOG_PWL_STRATEGY)
    parser.add_argument("--log_pwl_gamma", type=float, default=qbf.LOG_PWL_GAMMA)
    parser.add_argument("--log_offset", type=float, default=qbf.LOG_OFFSET)
    parser.add_argument("--log_input_clamp_min", type=float, default=qbf.LOG_INPUT_CLAMP_MIN)
    parser.add_argument("--mfcc_per_channel_scale", action=argparse.BooleanOptionalAction, default=qbf.MFCC_PER_CHANNEL_SCALE)
    parser.add_argument("--mfcc_output_scale", type=float, default=qbf.MFCC_OUTPUT_SCALE)
    parser.add_argument("--scale_bits", type=int, default=16)
    parser.add_argument("--quantize_frontend", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.backend = choose_backend(args.backend)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    baseline_rows = read_csv(Path(args.baseline_grid_csv))
    model_rows = unique_model_rows(baseline_rows)
    if args.limit_models > 0:
        model_rows = model_rows[: args.limit_models]
    if not model_rows:
        raise FileNotFoundError(f"No model rows found in {args.baseline_grid_csv}")

    print(f"[INFO] models={len(model_rows)}, backend={args.backend}, scale_spec_dir={args.scale_spec_dir}")
    all_rows: list[dict[str, Any]] = []
    for row in model_rows:
        print(f"[MODEL] dataset={row.get('dataset')} inference={row.get('inference_checkpoint')}")
        rows = run_one_model(args, row, scale_spec_dir=Path(args.scale_spec_dir), suffix=args.scale_spec_suffix)
        all_rows.extend(rows)
        save_csv(all_rows, Path(args.output_csv))

    save_csv(all_rows, Path(args.output_csv))
    print(f"[DONE] models={len(model_rows)}, grid_rows={len(all_rows)}")


if __name__ == "__main__":
    main()
