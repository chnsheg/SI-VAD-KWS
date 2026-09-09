from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from dscnn_kws.frontend import normalize_stage_bit_name, normalize_stage_bit_overrides


OUTPUT_ROOT = "./dscnn_kws/quantization/bit_accurate_mfcc_experiments_v3_bitwidth_sweep"

STAGE_BIT_SWEEP_VALUES: dict[str, list[int]] = {
    "MFCC_SAMPLE_W": [12, 11, 10, 9, 8],
    "HANN_COEFF_W": [16, 14, 12, 10, 8, 6],
    "FFT_IN_W": [18, 17, 16, 15, 14, 13, 12, 11, 10],
    "TWIDDLE_W": [16, 15, 14, 13, 12, 11, 10, 9, 8],
    "FFT_DATA_W": [20, 19, 18, 17, 16, 15, 14, 13, 12],
    "POWER_W": [41, 38, 36, 34, 32, 30, 28, 26, 24],
    "MEL_ACC_W": [46, 44, 42, 40, 38, 36, 34, 32, 30],
    "PWL_IN_W": [32, 30, 28, 26, 24, 22, 20, 18, 16],
    "LOG_W": [24, 22, 20, 18, 16, 14, 12],
    "DCT_COEFF_W": [8, 7, 6, 5, 4],
    "DCT_ACC_W": [40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20],
}

PAIRWISE_SCREENING_VALUES: dict[tuple[str, str], list[tuple[int, int]]] = {
    ("TWIDDLE_W", "FFT_DATA_W"): [(14, 18), (12, 16), (10, 14)],
    ("FFT_IN_W", "FFT_DATA_W"): [(16, 18), (14, 16), (12, 14)],
    ("POWER_W", "MEL_ACC_W"): [(36, 40), (32, 36), (28, 32)],
    ("PWL_IN_W", "LOG_W"): [(28, 20), (24, 18), (20, 16)],
    ("LOG_W", "DCT_ACC_W"): [(20, 36), (18, 32), (16, 28)],
    ("DCT_COEFF_W", "DCT_ACC_W"): [(7, 36), (6, 32), (5, 28)],
    ("HANN_COEFF_W", "TWIDDLE_W"): [(12, 14), (10, 12), (8, 10)],
}

FIELD_TO_STAGE: dict[str, str] = {
    field: normalize_stage_bit_name(field)
    for field in STAGE_BIT_SWEEP_VALUES
}

STAGE_TO_FIELD: dict[str, str] = {
    stage: field
    for field, stage in FIELD_TO_STAGE.items()
}


def parse_stage_bit_override_items(items: list[str] | tuple[str, ...] | None) -> dict[str, int]:
    overrides: dict[str, int] = {}
    if not items:
        return overrides
    for raw_item in items:
        for item in re.split(r"[,\s]+", str(raw_item).strip()):
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"Invalid stage bit override '{item}', expected NAME=BITS.")
            name, value = item.split("=", 1)
            overrides[name.strip()] = int(value.strip())
    return normalize_stage_bit_overrides(overrides)


def read_stage_bit_overrides_json(path: str | Path | None) -> dict[str, int]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "stage_bit_overrides" in payload:
        payload = payload["stage_bit_overrides"]
    if isinstance(payload, dict) and "stage_bits" in payload:
        payload = payload["stage_bits"]
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid stage bit override JSON: {path}")
    return normalize_stage_bit_overrides({str(k): int(v) for k, v in payload.items()})


def load_stage_bit_overrides(
    *,
    override_items: list[str] | tuple[str, ...] | None = None,
    override_json: str | Path | None = None,
) -> dict[str, int]:
    merged: dict[str, int] = {}
    merged.update(read_stage_bit_overrides_json(override_json))
    merged.update(parse_stage_bit_override_items(override_items))
    return normalize_stage_bit_overrides(merged)


def canonical_stage_bit_overrides_for_display(overrides: Mapping[str, int] | None) -> dict[str, int]:
    normalized = normalize_stage_bit_overrides(overrides)
    return {STAGE_TO_FIELD.get(stage, stage): int(bits) for stage, bits in normalized.items()}


def stage_bit_overrides_json(overrides: Mapping[str, int] | None) -> str:
    return json.dumps(canonical_stage_bit_overrides_for_display(overrides), sort_keys=True)


def safe_override_slug(overrides: Mapping[str, int] | None) -> str:
    display = canonical_stage_bit_overrides_for_display(overrides)
    if not display:
        return "baseline"
    parts = [f"{name}_{bits}" for name, bits in sorted(display.items())]
    slug = "__".join(parts)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", slug)
    return slug[:180]


def build_single_stage_screening_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field, values in STAGE_BIT_SWEEP_VALUES.items():
        baseline_bits = values[0]
        for bits in values[1:]:
            overrides = {field: bits}
            rows.append(
                {
                    "candidate_id": f"single_{field}_{bits}",
                    "candidate_type": "single_stage",
                    "target_fields": field,
                    "target_stages": normalize_stage_bit_name(field),
                    "baseline_bits": baseline_bits,
                    "stage_bit_overrides": stage_bit_overrides_json(overrides),
                    "override_slug": safe_override_slug(overrides),
                    "notes": f"{field}: {baseline_bits} -> {bits}",
                }
            )
    return rows


def build_pairwise_screening_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (field_a, field_b), value_pairs in PAIRWISE_SCREENING_VALUES.items():
        for bits_a, bits_b in value_pairs:
            overrides = {field_a: bits_a, field_b: bits_b}
            fields = f"{field_a}+{field_b}"
            stages = f"{normalize_stage_bit_name(field_a)}+{normalize_stage_bit_name(field_b)}"
            rows.append(
                {
                    "candidate_id": f"pair_{field_a}_{bits_a}__{field_b}_{bits_b}",
                    "candidate_type": "pairwise",
                    "target_fields": fields,
                    "target_stages": stages,
                    "baseline_bits": "",
                    "stage_bit_overrides": stage_bit_overrides_json(overrides),
                    "override_slug": safe_override_slug(overrides),
                    "notes": f"{field_a}={bits_a}, {field_b}={bits_b}",
                }
            )
    return rows


def parse_overrides_json_cell(value: str) -> dict[str, int]:
    if not value:
        return {}
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid stage_bit_overrides cell: {value}")
    return normalize_stage_bit_overrides({str(k): int(v) for k, v in payload.items()})

