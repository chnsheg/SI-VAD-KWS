from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RecleanConfig:
    sample_rate: int
    sample_length: int
    final_rms_alignment: bool
    positive_jitter_max_ms: int
    positive_jitter_mode: str
    speed_factors: tuple[float, ...]
    active_rms_dbfs: tuple[float, ...]
    negative_quotas: dict[str, float]
    snr_probabilities: dict[int, float]


def load_v1_config() -> RecleanConfig:
    config_path = Path(__file__).parents[1] / "reclean_config_v1.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return RecleanConfig(
        sample_rate=int(raw["sample_rate"]),
        sample_length=int(raw["sample_length"]),
        final_rms_alignment=bool(raw["final_rms_alignment"]),
        positive_jitter_max_ms=int(raw["positive_jitter_max_ms"]),
        positive_jitter_mode=str(raw["positive_jitter_mode"]),
        speed_factors=tuple(float(value) for value in raw["speed_factors"]),
        active_rms_dbfs=tuple(float(value) for value in raw["active_rms_dbfs"]),
        negative_quotas={key: float(value) for key, value in raw["negative_quotas"].items()},
        snr_probabilities={int(key): float(value) for key, value in raw["snr_probabilities"].items()},
    )
