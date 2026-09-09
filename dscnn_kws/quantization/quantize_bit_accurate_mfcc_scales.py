from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DATA_SCALE_STAGES = {
    "pcm",
    "preemphasis",
    "windowed",
    "fft_data",
    "power",
    "mel",
    "pwl_input",
    "log_mel",
    "dct",
    "mfcc",
}

FIXED_COEFF_STAGES = {
    "hann_coeff",
    "twiddle_coeff",
    "dct_coeff",
}


def quantize_scale_u_shift(scale: float, *, scale_bits: int) -> dict[str, float | int]:
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be a positive finite value, got {scale}")

    qmax = (1 << scale_bits) - 1
    shift = math.floor(math.log2(float(qmax) / float(scale)))
    multiplier = int(round(math.ldexp(float(scale), shift)))

    while multiplier > qmax:
        shift -= 1
        multiplier = int(round(math.ldexp(float(scale), shift)))
    while multiplier < 1:
        shift += 1
        multiplier = int(round(math.ldexp(float(scale), shift)))

    multiplier = max(1, min(qmax, multiplier))
    quantized_scale = math.ldexp(float(multiplier), -shift)
    abs_error = quantized_scale - float(scale)
    rel_error = abs_error / float(scale)
    return {
        "scale_bits": int(scale_bits),
        "multiplier": int(multiplier),
        "shift": int(shift),
        "original_scale": float(scale),
        "quantized_scale": float(quantized_scale),
        "abs_error": float(abs_error),
        "rel_error": float(rel_error),
        "abs_rel_error": float(abs(rel_error)),
    }


def quantize_scale_value(
    value: float | list[float],
    *,
    scale_bits: int,
) -> tuple[float | list[float], list[dict[str, float | int | str]]]:
    if isinstance(value, list):
        quantized: list[float] = []
        records: list[dict[str, float | int | str]] = []
        for idx, item in enumerate(value):
            record = quantize_scale_u_shift(float(item), scale_bits=scale_bits)
            record["channel"] = int(idx)
            quantized.append(float(record["quantized_scale"]))
            records.append(record)
        return quantized, records

    record = quantize_scale_u_shift(float(value), scale_bits=scale_bits)
    record["channel"] = ""
    return float(record["quantized_scale"]), [record]


def output_spec_name(path: Path, *, suffix: str) -> str:
    name = path.name
    marker = "_bit_accurate_mfcc_spec.json"
    if name.endswith(marker):
        return f"{name[: -len(marker)]}{suffix}"
    return f"{path.stem}{suffix}"


def quantize_one_spec(
    spec_path: Path,
    *,
    output_dir: Path,
    scale_bits: int,
    stages: set[str],
    suffix: str,
) -> tuple[Path, list[dict[str, Any]]]:
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    stage_quant = payload.get("config", {}).get("stage_quant")
    if not isinstance(stage_quant, dict):
        raise ValueError(f"No config.stage_quant found in {spec_path}")

    output_path = output_dir / output_spec_name(spec_path, suffix=suffix)
    rows: list[dict[str, Any]] = []
    scale_quant_summary: dict[str, Any] = {}

    for stage_name, spec in stage_quant.items():
        if not isinstance(spec, dict):
            continue
        enabled = bool(spec.get("enabled", False))
        scale = spec.get("scale")
        base_row = {
            "source_spec": str(spec_path),
            "output_spec": str(output_path),
            "stage": stage_name,
            "enabled": enabled,
            "scale_bits": scale_bits,
        }

        if stage_name in FIXED_COEFF_STAGES:
            rows.append({**base_row, "channel": "", "status": "skipped_fixed_coeff_scale"})
            continue
        if stage_name not in stages:
            rows.append({**base_row, "channel": "", "status": "skipped_not_selected"})
            continue
        if scale is None:
            rows.append({**base_row, "channel": "", "status": "skipped_no_scale"})
            continue
        if not enabled:
            rows.append({**base_row, "channel": "", "status": "skipped_disabled"})
            continue

        quantized_scale, records = quantize_scale_value(scale, scale_bits=scale_bits)
        spec["scale"] = quantized_scale
        spec["scale_quant"] = {
            "mode": "u_shift",
            "scale_bits": scale_bits,
            "original_scale": scale,
            "quantized_scale": quantized_scale,
            "records": records,
        }
        scale_quant_summary[stage_name] = spec["scale_quant"]
        for record in records:
            rows.append(
                {
                    **base_row,
                    **record,
                    "status": "quantized",
                    "scale_mode": "u_shift",
                }
            )

    extra = payload.setdefault("extra", {})
    extra["scale_quantization"] = {
        "mode": "u_shift",
        "scale_bits": scale_bits,
        "processed_data_scale_stages": sorted(stages),
        "skipped_fixed_coeff_stages": sorted(FIXED_COEFF_STAGES),
        "source_spec": str(spec_path),
        "scale_quantized_spec": str(output_path),
        "stage_scale_quant": scale_quant_summary,
        "note": (
            "Only observer/calibration data scales are quantized. Fixed coefficient tables "
            "and PWL log breakpoints/slopes/intercepts are intentionally unchanged."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path, rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] csv saved to: {path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize bit-accurate MFCC observer data scales to unsigned multiplier + shift specs."
    )
    parser.add_argument(
        "--input_dir",
        default="dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_coeff_quant/hardware_coeff_baseline_models",
    )
    parser.add_argument("--pattern", default="*_bit_accurate_mfcc_spec.json")
    parser.add_argument(
        "--output_dir",
        default="dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_scale_u16/scale_u16_models",
    )
    parser.add_argument("--report_csv", default=None)
    parser.add_argument("--scale_bits", type=int, default=16)
    parser.add_argument("--stages", nargs="*", default=sorted(DATA_SCALE_STAGES))
    parser.add_argument("--suffix", default="_scale_u16_bit_accurate_mfcc_spec.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scale_bits <= 0:
        raise ValueError("--scale_bits must be positive")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    stages = {str(stage) for stage in args.stages}
    unsupported = stages - DATA_SCALE_STAGES
    if unsupported:
        raise ValueError(f"Unsupported data scale stages: {sorted(unsupported)}")

    spec_paths = sorted(input_dir.glob(args.pattern))
    if not spec_paths:
        raise FileNotFoundError(f"No spec json matched {input_dir / args.pattern}")

    all_rows: list[dict[str, Any]] = []
    for spec_path in spec_paths:
        output_path, rows = quantize_one_spec(
            spec_path,
            output_dir=output_dir,
            scale_bits=args.scale_bits,
            stages=stages,
            suffix=args.suffix,
        )
        all_rows.extend(rows)
        print(f"[SCALE-U{args.scale_bits}] {spec_path.name} -> {output_path}")

    report_csv = Path(args.report_csv) if args.report_csv else output_dir / f"scale_u{args.scale_bits}_scale_report.csv"
    write_csv(all_rows, report_csv)
    print(f"[DONE] specs={len(spec_paths)}, rows={len(all_rows)}")


if __name__ == "__main__":
    main()
