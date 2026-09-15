"""Fail-closed positive/negative augmentation-supervision closure for V3.

The auditor works on generation provenance, before audio is rendered or packed.
An environmental positive can only enter V3 when a non-keyword counterfactual
negative has the same acoustic transformation signature.  This prevents the
classifier from treating a particular noise source, RIR, interferer, speed,
level, or placement as evidence for the wake word.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping


_CLEAN_COUNTERPART_KINDS = frozenset({"clean_speech_negative", "raw_speech_negative"})
_ENVIRONMENT_COUNTERPART_KINDS = frozenset(
    {
        "counterfactual_augmented_negative",
        "counterfactual_augmented_speech_negative",
        "counterfactual_augmented_false_wake",
    }
)


class AugmentationClosureError(ValueError):
    """Raised when a planned V3 positive has no valid negative counterpart."""


@dataclass(frozen=True)
class AugmentationSignature:
    noise_source_sha256: str | None
    noise_scene: str | None
    noise_rms_band: str | None
    rir_applied: bool
    rir_source_sha256: str | None
    interferer_applied: bool
    interferer_source_sha256: str | None
    speed: float
    foreground_rms_band: str
    position_bin: str

    @property
    def has_environmental_transform(self) -> bool:
        return (
            self.noise_source_sha256 is not None
            or self.rir_applied
            or self.interferer_applied
        )

    @property
    def family(self) -> str:
        if not self.has_environmental_transform:
            return "clean_or_speed"
        parts: list[str] = []
        if self.noise_source_sha256 is not None:
            parts.append("noise")
        if self.rir_applied:
            parts.append("rir")
        if self.interferer_applied:
            parts.append("interferer")
        return "+".join(parts)


@dataclass(frozen=True)
class AugmentationClosureReport:
    positive_count: int
    closed_positive_count: int
    negative_counterpart_count: int
    noise_scenes: dict[str, int]
    noise_sources: dict[str, int]
    noise_rms_bands: dict[str, int]
    rir_sources: dict[str, int]
    interferer_sources: dict[str, int]
    speed_factors: dict[str, int]
    foreground_rms_bands: dict[str, int]
    position_bins: dict[str, int]
    transform_families: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _row_id(row: Mapping[str, object], index: int) -> str:
    value = row.get("example_id")
    if not isinstance(value, str) or not value.strip():
        raise AugmentationClosureError(f"row {index} has no example_id")
    return value


def _label(row: Mapping[str, object], example_id: str) -> str:
    value = row.get("label")
    if value in ("positive", 0):
        return "positive"
    if value in ("negative", 1):
        return "negative"
    raise AugmentationClosureError(f"{example_id} has an invalid binary label: {value!r}")


def _optional_text(row: Mapping[str, object], field: str, example_id: str) -> str | None:
    value = row.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise AugmentationClosureError(f"{example_id} has an invalid {field}")
    return value


def _required_text(row: Mapping[str, object], field: str, example_id: str) -> str:
    value = _optional_text(row, field, example_id)
    if value is None:
        raise AugmentationClosureError(f"{example_id} is missing {field}")
    return value


def _required_bool(row: Mapping[str, object], field: str, example_id: str) -> bool:
    value = row.get(field)
    if not isinstance(value, bool):
        raise AugmentationClosureError(f"{example_id} has an invalid {field}")
    return value


def _signature(row: Mapping[str, object], example_id: str) -> AugmentationSignature:
    noise_source_sha256 = _optional_text(row, "noise_source_sha256", example_id)
    noise_scene = _optional_text(row, "noise_scene", example_id)
    noise_rms_band = _optional_text(row, "noise_rms_band", example_id)
    if noise_source_sha256 is None:
        if noise_scene is not None or noise_rms_band is not None:
            raise AugmentationClosureError(
                f"{example_id} declares noise_scene/noise_rms_band without noise_source_sha256"
            )
    else:
        if noise_scene is None or noise_rms_band is None:
            raise AugmentationClosureError(
                f"{example_id} is missing noise_scene or noise_rms_band for noise_source_sha256"
            )

    rir_applied = _required_bool(row, "rir_applied", example_id)
    rir_source_sha256 = _optional_text(row, "rir_source_sha256", example_id)
    if rir_applied != (rir_source_sha256 is not None):
        raise AugmentationClosureError(
            f"{example_id} has inconsistent rir_applied and rir_source_sha256"
        )

    interferer_applied = _required_bool(row, "interferer_applied", example_id)
    interferer_source_sha256 = _optional_text(row, "interferer_source_sha256", example_id)
    if interferer_applied != (interferer_source_sha256 is not None):
        raise AugmentationClosureError(
            f"{example_id} has inconsistent interferer_applied and interferer_source_sha256"
        )

    speed_value = row.get("speed")
    if isinstance(speed_value, bool) or not isinstance(speed_value, (int, float)):
        raise AugmentationClosureError(f"{example_id} has an invalid speed")
    speed = float(speed_value)
    if not math.isfinite(speed) or speed <= 0.0:
        raise AugmentationClosureError(f"{example_id} has an invalid speed")

    return AugmentationSignature(
        noise_source_sha256=noise_source_sha256,
        noise_scene=noise_scene,
        noise_rms_band=noise_rms_band,
        rir_applied=rir_applied,
        rir_source_sha256=rir_source_sha256,
        interferer_applied=interferer_applied,
        interferer_source_sha256=interferer_source_sha256,
        speed=speed,
        foreground_rms_band=_required_text(row, "foreground_rms_band", example_id),
        position_bin=_required_text(row, "position_bin", example_id),
    )


def _counterpart_kind(row: Mapping[str, object], example_id: str) -> str | None:
    value = row.get("counterpart_kind")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise AugmentationClosureError(f"{example_id} has an invalid counterpart_kind")
    return value


def _first_signature_difference(
    expected: AugmentationSignature,
    actual: AugmentationSignature,
) -> str | None:
    for field in (
        "noise_source_sha256",
        "noise_scene",
        "noise_rms_band",
        "rir_applied",
        "rir_source_sha256",
        "interferer_applied",
        "interferer_source_sha256",
        "speed",
        "foreground_rms_band",
        "position_bin",
    ):
        if getattr(expected, field) != getattr(actual, field):
            return field
    return None


def _missing_counterpart_error(
    *,
    example_id: str,
    signature: AugmentationSignature,
    candidates: list[AugmentationSignature],
) -> AugmentationClosureError:
    if candidates:
        difference = _first_signature_difference(signature, candidates[0])
        if difference is not None:
            return AugmentationClosureError(
                f"{example_id} has no negative counterpart with matching {difference}"
            )
    if signature.has_environmental_transform:
        return AugmentationClosureError(
            f"{example_id} has no counterfactual augmented negative counterpart"
        )
    return AugmentationClosureError(
        f"{example_id} has no clean/raw speech negative counterpart"
    )


def audit_augmentation_closure(rows: Iterable[Mapping[str, object]]) -> AugmentationClosureReport:
    """Validate train-only positive/negative augmentation closure before V3 build.

    A matching counterpart has the complete acoustic signature of the positive.
    Environmental counterparts must be an explicitly generated non-keyword
    augmented negative; clean/speed-only positives need a clean/raw speech
    negative.  One matching negative may supervise repeated positives with the
    same signature, but no source or transformation signature may be absent.
    """
    positives: list[tuple[str, AugmentationSignature]] = []
    environmental_negatives: list[AugmentationSignature] = []
    clean_speech_negatives: list[AugmentationSignature] = []
    noise_scenes: Counter[str] = Counter()
    noise_sources: Counter[str] = Counter()
    noise_rms_bands: Counter[str] = Counter()
    rir_sources: Counter[str] = Counter()
    interferer_sources: Counter[str] = Counter()
    speed_factors: Counter[str] = Counter()
    foreground_rms_bands: Counter[str] = Counter()
    position_bins: Counter[str] = Counter()
    transform_families: Counter[str] = Counter()
    negative_counterpart_count = 0

    for index, row in enumerate(rows, start=1):
        example_id = _row_id(row, index)
        if row.get("source_split") != "train":
            raise AugmentationClosureError(
                f"{example_id} is not train-only and cannot be used for augmentation closure"
            )
        label = _label(row, example_id)
        signature = _signature(row, example_id)
        if label == "positive":
            positives.append((example_id, signature))
            transform_families[signature.family] += 1
            speed_factors[format(signature.speed, "g")] += 1
            foreground_rms_bands[signature.foreground_rms_band] += 1
            position_bins[signature.position_bin] += 1
            if signature.noise_source_sha256 is not None:
                assert signature.noise_scene is not None
                assert signature.noise_rms_band is not None
                noise_sources[signature.noise_source_sha256] += 1
                noise_scenes[signature.noise_scene] += 1
                noise_rms_bands[signature.noise_rms_band] += 1
            if signature.rir_source_sha256 is not None:
                rir_sources[signature.rir_source_sha256] += 1
            if signature.interferer_source_sha256 is not None:
                interferer_sources[signature.interferer_source_sha256] += 1
            continue

        kind = _counterpart_kind(row, example_id)
        if signature.has_environmental_transform and kind in _ENVIRONMENT_COUNTERPART_KINDS:
            environmental_negatives.append(signature)
            negative_counterpart_count += 1
        elif not signature.has_environmental_transform and kind in _CLEAN_COUNTERPART_KINDS:
            clean_speech_negatives.append(signature)
            negative_counterpart_count += 1

    for example_id, signature in positives:
        candidates = (
            environmental_negatives
            if signature.has_environmental_transform
            else clean_speech_negatives
        )
        if signature not in candidates:
            raise _missing_counterpart_error(
                example_id=example_id,
                signature=signature,
                candidates=candidates,
            )

    return AugmentationClosureReport(
        positive_count=len(positives),
        closed_positive_count=len(positives),
        negative_counterpart_count=negative_counterpart_count,
        noise_scenes=dict(sorted(noise_scenes.items())),
        noise_sources=dict(sorted(noise_sources.items())),
        noise_rms_bands=dict(sorted(noise_rms_bands.items())),
        rir_sources=dict(sorted(rir_sources.items())),
        interferer_sources=dict(sorted(interferer_sources.items())),
        speed_factors=dict(sorted(speed_factors.items())),
        foreground_rms_bands=dict(sorted(foreground_rms_bands.items())),
        position_bins=dict(sorted(position_bins.items())),
        transform_families=dict(sorted(transform_families.items())),
    )
