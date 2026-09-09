"""Composition boundary between captured channel PCM and mono VAD/KWS PCM."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .features import StreamingChannelConverter
from .input_conditioning import InputConditioningMetrics
from .input_frontend import FrontendProfile, InputFrontend
from .runtime_profile import RuntimeProfile
from .spatial import SpatialMetrics


@dataclass(frozen=True)
class RealtimeInputFrame:
    raw_mono: np.ndarray
    inference_mono: np.ndarray
    conditioned_mono: np.ndarray
    spatial: SpatialMetrics
    conditioning: InputConditioningMetrics


class RealtimeInputPipeline:
    """Convert one selected microphone channel and apply causal conditioning."""

    def __init__(
        self,
        *,
        conditioner_enabled: bool = False,
        target_rms_dbfs: float = -25.0,
        frontend_profile: FrontendProfile | str = FrontendProfile.PEAK_AGC,
    ) -> None:
        self._converter = StreamingChannelConverter()
        self._frontend_profile = FrontendProfile(frontend_profile)
        self._frontend = InputFrontend(self._frontend_profile, target_rms_dbfs=target_rms_dbfs)
        self._conditioner_enabled = _require_bool(conditioner_enabled)

    @classmethod
    def from_profile(
        cls,
        profile: RuntimeProfile,
        *,
        target_rms_dbfs: float | None = None,
        conditioner_enabled: bool | None = None,
    ) -> "RealtimeInputPipeline":
        if not isinstance(profile, RuntimeProfile):
            raise ValueError("profile must be a RuntimeProfile")
        if conditioner_enabled is None:
            conditioner_enabled = profile.frontend_profile is not FrontendProfile.RAW
        return cls(
            conditioner_enabled=conditioner_enabled,
            target_rms_dbfs=(
                profile.default_target_rms_dbfs
                if target_rms_dbfs is None
                else target_rms_dbfs
            ),
            frontend_profile=profile.frontend_profile,
        )

    def process(self, samples: np.ndarray, *, input_sample_rate: int) -> RealtimeInputFrame:
        values = np.asarray(samples)
        if values.dtype != np.float32:
            raise ValueError("input PCM must be float32")
        if values.ndim == 1:
            values = values[:, None]
        channels = self._converter.convert(values, input_sample_rate)
        if channels.size == 0:
            raise ValueError("input conversion did not produce PCM")
        raw_mono = channels[:, 0].copy()
        spatial = SpatialMetrics("mono", 0, 1, None, None, None)
        if self._conditioner_enabled:
            conditioned, conditioning = self._frontend.process(raw_mono)
        else:
            conditioned = raw_mono.copy()
            conditioning = _passthrough_metrics(conditioned)
        return RealtimeInputFrame(
            raw_mono=raw_mono,
            inference_mono=conditioned.copy(),
            conditioned_mono=conditioned,
            spatial=spatial,
            conditioning=conditioning,
        )

    def update_vad_confidence(self, score: float | None) -> None:
        del score

    def reset(self) -> None:
        self._converter.reset()
        self._frontend.reset()

    def set_config(self, *, conditioner_enabled: bool, target_rms_dbfs: float) -> None:
        self._frontend = InputFrontend(self._frontend_profile, target_rms_dbfs=target_rms_dbfs)
        self._conditioner_enabled = _require_bool(conditioner_enabled)
        self.reset()


def _require_bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError("conditioner_enabled must be a boolean")
    return value


def _passthrough_metrics(pcm: np.ndarray) -> InputConditioningMetrics:
    rms = math.sqrt(float(np.mean(np.square(pcm, dtype=np.float64))))
    peak = float(np.max(np.abs(pcm)))
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-8))
    peak_dbfs = 20.0 * math.log10(max(peak, 1e-8))
    return InputConditioningMetrics(
        raw_rms_dbfs=rms_dbfs,
        raw_peak_dbfs=peak_dbfs,
        conditioned_rms_dbfs=rms_dbfs,
        conditioned_peak_dbfs=peak_dbfs,
        applied_gain_db=0.0,
        noise_floor_dbfs=-60.0,
        source_clipped=bool(np.any(np.abs(pcm) >= 0.999)),
    )
