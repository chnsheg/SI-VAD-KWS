"""Fixed microphone-tuning settings selected by two-source offline replay."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from collections.abc import Mapping
from pathlib import Path

from .contracts import (
    CascadeConfig,
    SERIAL_LOG_OBSERVED_CONTROL_CONTRACT,
    ObservedControlContract,
)
from .input_frontend import FrontendProfile


class WakeLifecyclePolicy(StrEnum):
    """Profile-owned action for the outer session after a controller emits wake."""

    HOLD = "hold"
    AUTO_REARM = "auto_rearm"


@dataclass(frozen=True)
class RuntimeProfile:
    frontend_profile: FrontendProfile
    default_target_rms_dbfs: float
    energy_threshold_dbfs: float
    energy_hangover_ms: int
    vad_energy_tail_ms: int
    vad_period_ms: int
    kws_period_ms: int
    vad_threshold: float
    vad_confirmations: int
    kws_threshold: float
    kws_lookback_ms: int = 1500
    kws_confirmations: int = 2
    no_speech_timeout_ms: int = 3_000
    controller_kind: str = "legacy_cascade"
    kws_execution_mode: str = "asynchronous"
    control_contract: ObservedControlContract | None = None
    wake_lifecycle_policy: WakeLifecyclePolicy = WakeLifecyclePolicy.HOLD

    def cascade_config(self, *, kws_positive_index: int, queue_capacity: int) -> CascadeConfig:
        if self.controller_kind != "legacy_cascade":
            raise ValueError("PC deployment control profiles must be used with ObservedControlEngine")
        config = CascadeConfig(
            energy_period_ms=self.vad_period_ms,
            energy_threshold_dbfs=self.energy_threshold_dbfs,
            energy_hangover_ms=self.energy_hangover_ms,
            vad_energy_tail_ms=self.vad_energy_tail_ms,
            vad_period_ms=self.vad_period_ms,
            vad_threshold=self.vad_threshold,
            vad_confirmations=self.vad_confirmations,
            kws_period_ms=self.kws_period_ms,
            kws_positive_index=kws_positive_index,
            kws_threshold=self.kws_threshold,
            kws_lookback_ms=self.kws_lookback_ms,
            kws_confirmations=self.kws_confirmations,
            vad_no_speech_timeout_ms=self.no_speech_timeout_ms,
            queue_capacity=queue_capacity,
        )
        config.validate()
        return config


_RUNTIME_CONFIG_VERSION = 1


def pc_deployment_runtime_config_path() -> Path:
    """Return the per-user settings path shared by the PC service and dashboard."""

    local_app_data = os.environ.get("LOCALAPPDATA")
    root = Path(local_app_data) if local_app_data else Path.home() / ".local"
    return root / "VAD-KWS" / "pc-deployment-runtime-config.json"


def _validated_pc_deployment_runtime_config(
    payload: Mapping[str, object],
) -> dict[str, object]:
    if payload.get("version") != _RUNTIME_CONFIG_VERSION:
        raise ValueError("unsupported runtime config version")

    def probability(name: str) -> float:
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite probability")
        result = float(value)
        if not math.isfinite(result) or not 0.0 <= result <= 1.0:
            raise ValueError(f"{name} must be a finite probability")
        return result

    def boolean(name: str) -> bool:
        value = payload.get(name)
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be boolean")
        return value

    def period(name: str) -> int:
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 10:
            raise ValueError(f"{name} must be an integer >= 10")
        return value

    lookback = payload.get("kws_lookback_ms")
    if isinstance(lookback, bool) or not isinstance(lookback, int) or not 1000 <= lookback <= 2000:
        raise ValueError("kws_lookback_ms must be an integer in [1000, 2000]")
    return {
        "vad_threshold": probability("vad_threshold"),
        "kws_threshold": probability("kws_threshold"),
        "energy_enabled": boolean("energy_enabled"),
        "vad_enabled": boolean("vad_enabled"),
        "vad_period_ms": period("vad_period_ms"),
        "kws_period_ms": period("kws_period_ms"),
        "kws_lookback_ms": lookback,
    }


def load_pc_deployment_runtime_config(path: Path | None = None) -> dict[str, object] | None:
    """Load valid persisted settings, ignoring a missing or corrupt file."""

    source = pc_deployment_runtime_config_path() if path is None else Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("runtime config must be an object")
        return _validated_pc_deployment_runtime_config(payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def save_pc_deployment_runtime_config(
    config: Mapping[str, object], path: Path | None = None
) -> dict[str, object]:
    """Atomically persist the settings that the dashboard successfully applied."""

    values = _validated_pc_deployment_runtime_config(
        {"version": _RUNTIME_CONFIG_VERSION, **dict(config)}
    )
    target = pc_deployment_runtime_config_path() if path is None else Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                {"version": _RUNTIME_CONFIG_VERSION, **values},
                handle,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        temporary_path.replace(target)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return values


LEGACY_RUNTIME_PROFILE = RuntimeProfile(
    frontend_profile=FrontendProfile.RAW,
    default_target_rms_dbfs=-25.0,
    energy_threshold_dbfs=-33.0,
    energy_hangover_ms=1_000,
    vad_energy_tail_ms=1_000,
    vad_period_ms=32,
    kws_period_ms=96,
    vad_threshold=0.80,
    vad_confirmations=3,
    kws_threshold=0.70,
)


# Keep existing callers on the historical PC cascade until they explicitly select parity.
DEFAULT_RUNTIME_PROFILE = LEGACY_RUNTIME_PROFILE


PC_DEPLOYMENT_RUNTIME_PROFILE = RuntimeProfile(
    frontend_profile=FrontendProfile.RAW,
    default_target_rms_dbfs=-25.0,
    energy_threshold_dbfs=-40.0,
    energy_hangover_ms=960,
    vad_energy_tail_ms=0,
    vad_period_ms=32,
    kws_period_ms=96,
    vad_threshold=0.45,
    vad_confirmations=3,
    kws_threshold=0.50,
    kws_confirmations=2,
    no_speech_timeout_ms=3_000,
    controller_kind="observed_control",
    kws_execution_mode="asynchronous",
    control_contract=SERIAL_LOG_OBSERVED_CONTROL_CONTRACT,
    wake_lifecycle_policy=WakeLifecyclePolicy.AUTO_REARM,
)
