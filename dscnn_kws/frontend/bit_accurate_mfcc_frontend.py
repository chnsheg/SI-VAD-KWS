from __future__ import annotations

from pathlib import Path
from typing import Any

from .bit_accurate_mfcc_high_precision import (
    BitAccurateMFCCConfig,
    BitAccurateMFCCHighPrecisionFrontend,
    make_bit_accurate_mfcc_config,
)


class BitAccurateMFCCFrontend(BitAccurateMFCCHighPrecisionFrontend):
    """Hard-quantized inference reference for the bit-accurate MFCC experiment."""

    def __init__(
        self,
        config: BitAccurateMFCCConfig | dict[str, Any] | None = None,
        *,
        constraint_profile: str = "frontend_fakequant",
        observer_enabled: bool = False,
    ):
        if config is None:
            config = make_bit_accurate_mfcc_config(constraint_profile=constraint_profile)
        super().__init__(config=config, observer_enabled=observer_enabled, ste=False)

    @classmethod
    def from_spec_json(cls, path: str | Path, **kwargs) -> "BitAccurateMFCCFrontend":
        base = BitAccurateMFCCHighPrecisionFrontend.from_spec_json(path, **kwargs)
        obj = cls(config=base.config, observer_enabled=kwargs.get("observer_enabled", False))
        obj.observed_absmax.update(base.observed_absmax)
        obj.stage_stats.update(base.stage_stats)
        return obj

