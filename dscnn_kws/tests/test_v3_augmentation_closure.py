from __future__ import annotations

import pytest

try:
    from dscnn_kws.data.reclean.v3_augmentation_closure import (
        AugmentationClosureError,
        audit_augmentation_closure,
    )
except ModuleNotFoundError:
    AugmentationClosureError = ValueError
    audit_augmentation_closure = None


def _audit(rows: list[dict[str, object]]):
    assert audit_augmentation_closure is not None, "augmentation-closure gate is not implemented"
    return audit_augmentation_closure(rows)


def _environment_row(
    example_id: str,
    *,
    label: str,
    noise_source_sha256: str = "noise-a",
    noise_scene: str = "road",
    noise_rms_band: str = "-35:-30",
    rir_source_sha256: str | None = "rir-a",
    interferer_source_sha256: str | None = "speech-a",
    speed: float = 0.9,
    foreground_rms_band: str = "-30:-26",
    position_bin: str = "center",
    counterpart_kind: str | None = None,
) -> dict[str, object]:
    return {
        "example_id": example_id,
        "label": label,
        "source_split": "train",
        "noise_source_sha256": noise_source_sha256,
        "noise_scene": noise_scene,
        "noise_rms_band": noise_rms_band,
        "rir_applied": rir_source_sha256 is not None,
        "rir_source_sha256": rir_source_sha256,
        "interferer_applied": interferer_source_sha256 is not None,
        "interferer_source_sha256": interferer_source_sha256,
        "speed": speed,
        "foreground_rms_band": foreground_rms_band,
        "position_bin": position_bin,
        "counterpart_kind": counterpart_kind,
    }


def _clean_row(
    example_id: str,
    *,
    label: str,
    speed: float = 1.25,
    foreground_rms_band: str = "-26:-22",
    position_bin: str = "late",
    counterpart_kind: str | None = None,
) -> dict[str, object]:
    return {
        "example_id": example_id,
        "label": label,
        "source_split": "train",
        "noise_source_sha256": None,
        "noise_scene": None,
        "noise_rms_band": None,
        "rir_applied": False,
        "rir_source_sha256": None,
        "interferer_applied": False,
        "interferer_source_sha256": None,
        "speed": speed,
        "foreground_rms_band": foreground_rms_band,
        "position_bin": position_bin,
        "counterpart_kind": counterpart_kind,
    }


def test_environment_positive_requires_same_source_and_transform_negative() -> None:
    positive = _environment_row("positive", label="positive")
    wrong_negative = _environment_row(
        "negative",
        label="negative",
        noise_source_sha256="noise-b",
        counterpart_kind="counterfactual_augmented_negative",
    )

    with pytest.raises(AugmentationClosureError, match="noise_source_sha256"):
        _audit([positive, wrong_negative])


def test_environment_closure_accepts_same_noise_rir_and_interferer_signature() -> None:
    positive = _environment_row("positive", label="positive")
    negative = _environment_row(
        "negative",
        label="negative",
        counterpart_kind="counterfactual_augmented_negative",
    )

    report = _audit([positive, negative])

    assert report.positive_count == 1
    assert report.closed_positive_count == 1
    assert report.noise_scenes == {"road": 1}
    assert report.noise_sources == {"noise-a": 1}
    assert report.noise_rms_bands == {"-35:-30": 1}
    assert report.rir_sources == {"rir-a": 1}
    assert report.interferer_sources == {"speech-a": 1}
    assert report.speed_factors == {"0.9": 1}
    assert report.foreground_rms_bands == {"-30:-26": 1}
    assert report.position_bins == {"center": 1}
    assert report.transform_families == {"noise+rir+interferer": 1}


def test_environment_closure_rejects_missing_rir_or_interferer_counterpart() -> None:
    positive = _environment_row("positive", label="positive")
    negative = _environment_row(
        "negative",
        label="negative",
        rir_source_sha256=None,
        interferer_source_sha256=None,
        counterpart_kind="counterfactual_augmented_negative",
    )

    with pytest.raises(AugmentationClosureError, match="rir_applied"):
        _audit([positive, negative])


def test_clean_or_speed_positive_requires_clean_speech_counterpart() -> None:
    positive = _clean_row("positive", label="positive")
    pure_noise = _clean_row(
        "negative",
        label="negative",
        counterpart_kind="pure_noise",
    )

    with pytest.raises(AugmentationClosureError, match="clean/raw speech"):
        _audit([positive, pure_noise])

    report = _audit(
        [
            positive,
            _clean_row(
                "speech-negative",
                label="negative",
                counterpart_kind="clean_speech_negative",
            ),
        ]
    )

    assert report.transform_families == {"clean_or_speed": 1}


def test_auditor_rejects_missing_provenance_and_held_out_rows() -> None:
    malformed = _environment_row("positive", label="positive")
    malformed.pop("noise_rms_band")

    with pytest.raises(AugmentationClosureError, match="noise_rms_band"):
        _audit([malformed])

    held_out = _clean_row("held-out", label="positive")
    held_out["source_split"] = "test"
    with pytest.raises(AugmentationClosureError, match="train-only"):
        _audit([held_out])
