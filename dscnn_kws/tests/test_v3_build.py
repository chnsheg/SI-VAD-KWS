from __future__ import annotations

import json
from pathlib import Path

import pytest
import soundfile as sf
import torch

from dscnn_kws.data.reclean.audio import save_pcm16_atomic
import dscnn_kws.data.reclean.generator as generator_module
from dscnn_kws.data.reclean.generator import GenerationRequest, generate_shard, render_request
from dscnn_kws.data.reclean.recipes import Recipe

try:
    from dscnn_kws.data.reclean.v3_build import (
        audit_environment_role,
        audit_v3_train_build,
        validate_v3_provenance,
    )
except ModuleNotFoundError:
    audit_environment_role = None
    audit_v3_train_build = None
    validate_v3_provenance = None

try:
    from dscnn_kws.data.reclean.v3_build import write_train_positive_active_rms_eligible_manifest
except ImportError:
    write_train_positive_active_rms_eligible_manifest = None

try:
    from dscnn_kws.data.reclean.v3_build import write_raw_positive_anchor_requests
except ImportError:
    write_raw_positive_anchor_requests = None

try:
    from dscnn_kws.data.reclean.v3_build import write_false_wake_hard_negative_requests
except ImportError:
    write_false_wake_hard_negative_requests = None

try:
    from dscnn_kws.data.reclean.v3_build import write_ranked_hard_negative_requests
except ImportError:
    write_ranked_hard_negative_requests = None


def _environment_negative(
    example_id: str,
    *,
    source_sha256: str = "source-a",
    waveform_sha256: str | None = None,
    source_start_sample: int = 0,
) -> dict[str, object]:
    return {
        "example_id": example_id,
        "label": "negative",
        "source_split": "train",
        "source_sha256": source_sha256,
        "source_start_sample": source_start_sample,
        "sample_rate": 16000,
        "frames": 16000,
        "output_sha256": waveform_sha256 or f"waveform-{example_id}",
        "noise_source_sha256": None,
        "noise_scene": None,
        "noise_rms_band": None,
        "rir_applied": False,
        "rir_source_sha256": None,
        "interferer_applied": False,
        "interferer_source_sha256": None,
        "speed": 1.0,
        "foreground_rms_band": "-30:-26",
        "position_bin": "center",
        "counterpart_kind": "clean_speech_negative",
    }


def test_environment_role_rejects_duplicate_waveforms_and_insufficient_source_coverage() -> None:
    assert audit_environment_role is not None, "V3 environment-role audit is not implemented"
    rows = [
        _environment_negative("one", waveform_sha256="same", source_start_sample=0),
        _environment_negative("two", waveform_sha256="same", source_start_sample=1536),
    ]

    with pytest.raises(ValueError, match="uniqueness|coverage"):
        audit_environment_role(rows, minimum_unique_ratio=0.9, minimum_coverage_ratio=0.5)


def test_environment_role_defaults_coverage_to_selected_starts_without_a_candidate_total() -> None:
    assert audit_environment_role is not None, "V3 environment-role audit is not implemented"
    rows = [
        _environment_negative("one", source_start_sample=0),
        _environment_negative("two", source_start_sample=1536),
    ]

    report = audit_environment_role(
        rows,
        minimum_unique_ratio=0.9,
        minimum_coverage_ratio=0.5,
    )

    assert report["source_time_coverage"] == {"source-a": 1.0}


def test_active_rms_gate_excludes_quiet_train_positive_sources(tmp_path: Path) -> None:
    assert write_train_positive_active_rms_eligible_manifest is not None, (
        "V3 active-RMS positive-source gate is not implemented"
    )
    loud_path = tmp_path / "loud.wav"
    quiet_path = tmp_path / "quiet.wav"
    sf.write(loud_path, [0.2] * 16_000, 16_000, subtype="PCM_16")
    sf.write(quiet_path, [0.0001] * 16_000, 16_000, subtype="PCM_16")
    source_rows = [
        {
            "prepared_path": str(loud_path),
            "source_sha256": "loud-source",
            "source_kind": "speech",
            "source_label": "positive",
            "source_split": "train",
            "sample_rate": 16_000,
            "active_start": 0,
            "active_end": 16_000,
        },
        {
            "prepared_path": str(quiet_path),
            "source_sha256": "quiet-source",
            "source_kind": "speech",
            "source_label": "positive",
            "source_split": "train",
            "sample_rate": 16_000,
            "active_start": 0,
            "active_end": 16_000,
        },
    ]
    prepared_manifest = tmp_path / "prepared_sources.jsonl"
    prepared_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in source_rows),
        encoding="utf-8",
    )
    destination = tmp_path / "train_positive_active_rms_eligible.jsonl"

    report = write_train_positive_active_rms_eligible_manifest(
        prepared_manifest,
        destination,
        minimum_rms_dbfs=-50.0,
    )

    eligible_rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert report["input_count"] == 2
    assert report["eligible_count"] == 1
    assert report["rejected_count"] == 1
    assert eligible_rows[0]["source_sha256"] == "loud-source"
    assert eligible_rows[0]["active_rms_dbfs"] >= -50.0


def test_raw_positive_requests_require_active_rms_eligible_train_sources(tmp_path: Path) -> None:
    assert write_raw_positive_anchor_requests is not None, "V3 raw-positive request writer is not implemented"
    sources = [
        {
            "prepared_path": str(tmp_path / "one.wav"),
            "source_sha256": "one-source",
            "source_kind": "speech",
            "source_label": "positive",
            "source_split": "train",
            "active_start": 100,
            "active_end": 8_000,
            "active_rms_dbfs": -22.0,
            "active_rms_minimum_dbfs": -50.0,
            "active_rms_gate": "eligible",
        },
        {
            "prepared_path": str(tmp_path / "two.wav"),
            "source_sha256": "two-source",
            "source_kind": "speech",
            "source_label": "positive",
            "source_split": "train",
            "active_start": 100,
            "active_end": 8_000,
            "active_rms_dbfs": -30.0,
            "active_rms_minimum_dbfs": -50.0,
            "active_rms_gate": "eligible",
        },
    ]
    destination = tmp_path / "raw_positive.requests.jsonl"

    report = write_raw_positive_anchor_requests(sources, destination, seed=19)

    requests = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert report["request_count"] == 2
    assert {request["source_role"] for request in requests} == {"raw_positive"}
    assert all(request["augmentation_closure_required"] is False for request in requests)
    assert all(request["recipe"]["speed"] == 1.0 for request in requests)
    assert all(request["recipe"]["noise_scene"] is None for request in requests)


def test_false_wake_hard_negative_requests_use_source_plan_complete_windows(tmp_path: Path) -> None:
    assert write_false_wake_hard_negative_requests is not None, "V3 false-wake request writer is not implemented"
    plan = {
        "schema_version": 1,
        "plans": [
            {
                "source": {
                    "path": "/immutable/false-a.wav",
                    "scene": "false_wake",
                    "domain": "false_wake",
                    "source_sha256": "source-a",
                    "frames": 64_000,
                },
                "blocks": [
                    {
                        "source_sha256": "source-a",
                        "path": "/immutable/false-a.wav",
                        "scene": "false_wake",
                        "domain": "false_wake",
                        "split": "train",
                        "block_index": 0,
                        "block_start": 0,
                        "block_end": 64_000,
                        "allowed_start": 0,
                        "allowed_end": 64_000,
                    }
                ],
            },
            {
                "source": {
                    "path": "/immutable/false-b.wav",
                    "scene": "false_wake",
                    "domain": "false_wake",
                    "source_sha256": "source-b",
                    "frames": 64_000,
                },
                "blocks": [
                    {
                        "source_sha256": "source-b",
                        "path": "/immutable/false-b.wav",
                        "scene": "false_wake",
                        "domain": "false_wake",
                        "split": "train",
                        "block_index": 0,
                        "block_start": 0,
                        "block_end": 64_000,
                        "allowed_start": 0,
                        "allowed_end": 64_000,
                    }
                ],
            },
        ],
    }
    source_plan = tmp_path / "source_plan.json"
    source_plan.write_text(json.dumps(plan), encoding="utf-8")
    destination = tmp_path / "false_wake.requests.jsonl"

    report = write_false_wake_hard_negative_requests(
        source_plan,
        destination,
        target_count=4,
        seed=23,
    )

    requests = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert report["request_count"] == 4
    assert report["source_count"] == 2
    assert {request["source_role"] for request in requests} == {"false_wake_hard_negative"}
    assert {request["recipe"]["source_kind"] for request in requests} == {"false_wake"}
    assert all(request["foreground_start_sample"] + 16_000 <= 64_000 for request in requests)
    assert {request["recipe"]["speed"] for request in requests} == {1.0}
    assert requests[0]["source_sha256"] != requests[1]["source_sha256"]


def test_ranked_hard_negative_requests_retain_false_wakes_and_take_highest_raw_scores(tmp_path: Path) -> None:
    assert write_ranked_hard_negative_requests is not None, "V3 ranked hard-negative writer is not implemented"
    false_wake_requests = [
        {"example_id": "false-a", "label": "negative", "source_role": "false_wake_hard_negative"},
        {"example_id": "false-b", "label": "negative", "source_role": "false_wake_hard_negative"},
    ]
    raw_scores = [
        {
            "audio_path": "/immutable/raw-low.wav",
            "source_sha256": "raw-low",
            "source_role": "raw_negative",
            "start_sample": 0,
            "positive_score": 0.1,
        },
        {
            "audio_path": "/immutable/raw-high.wav",
            "source_sha256": "raw-high",
            "source_role": "raw_negative",
            "start_sample": 0,
            "positive_score": 0.9,
        },
        {
            "audio_path": "/immutable/raw-middle.wav",
            "source_sha256": "raw-middle",
            "source_role": "raw_negative",
            "start_sample": 0,
            "positive_score": 0.5,
        },
    ]
    destination = tmp_path / "hard_negative.requests.jsonl"

    report = write_ranked_hard_negative_requests(
        false_wake_requests,
        raw_scores,
        destination,
        raw_target_count=2,
        seed=29,
    )

    requests = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    raw_requests = [request for request in requests if request["source_role"] == "raw_score_hard_negative"]
    assert report["false_wake_count"] == 2
    assert report["raw_score_count"] == 2
    assert [request["source_sha256"] for request in raw_requests] == ["raw-high", "raw-middle"]
    assert all(request["foreground_start_sample"] == 0 for request in raw_requests)
    assert all(request["recipe"]["source_kind"] == "speech" for request in raw_requests)


def test_false_wake_request_uses_its_explicit_complete_source_window(tmp_path: Path) -> None:
    source_path = tmp_path / "false_wake.wav"
    samples = [0.2] * 4_000 + [0.0] * 24_000 + [0.2] * 4_000
    sf.write(source_path, samples, 16_000, subtype="PCM_16")
    recipe = Recipe(
        seed=0,
        split="train",
        source_id="false-wake-source",
        source_sha256="false-wake-source",
        slot=0,
        label="negative",
        source_kind="false_wake",
        augmentation_group="false_wake_dense",
        speed=1.0,
        jitter_ms=0,
        online_window_jitter_max_ms=0,
        active_rms_dbfs=-20.0,
        noise_scene=None,
        snr_db=None,
        apply_rir=False,
        apply_interferer=False,
        sir_db=None,
    )
    request = GenerationRequest(
        example_id="false-wake-explicit-window",
        recipe=recipe,
        foreground_path=source_path,
        active_span=None,
        foreground_start_sample=16_000,
        source_role="false_wake_hard_negative",
    )

    metadata = render_request(request, tmp_path / "rendered.wav", torch.device("cpu"))
    rendered, _ = sf.read(tmp_path / "rendered.wav", dtype="float32")

    assert metadata["source_start_sample"] == 16_000
    assert abs(rendered[:4_000]).mean() < 1e-4
    assert abs(rendered[-4_000:]).mean() > 1e-2


def test_raw_anchor_render_preserves_augmentation_closure_exemption(tmp_path: Path) -> None:
    source_path = tmp_path / "raw_positive.wav"
    sf.write(source_path, [0.1] * 16_000, 16_000, subtype="PCM_16")
    request = GenerationRequest.from_dict(
        {
            "example_id": "raw-positive-anchor",
            "recipe": {
                "seed": 7,
                "split": "train",
                "source_id": "raw-positive-source",
                "source_sha256": "raw-positive-source",
                "slot": 0,
                "label": "positive",
                "source_kind": "speech",
                "augmentation_group": "raw_anchor",
                "speed": 1.0,
                "jitter_ms": 0,
                "online_window_jitter_max_ms": 200,
                "active_rms_dbfs": -20.0,
                "noise_scene": None,
                "snr_db": None,
                "apply_rir": False,
                "apply_interferer": False,
                "sir_db": None,
            },
            "foreground_path": str(source_path),
            "active_span": [0, 16_000],
            "source_role": "raw_positive",
            "augmentation_closure_required": False,
        }
    )

    metadata = render_request(request, tmp_path / "raw_positive_rendered.wav", torch.device("cpu"))

    assert metadata["augmentation_closure_required"] is False


def test_counterfactual_render_preserves_counterpart_kind(tmp_path: Path) -> None:
    source_path = tmp_path / "nonwake.wav"
    sf.write(source_path, [0.1] * 16_000, 16_000, subtype="PCM_16")
    request = GenerationRequest.from_dict(
        {
            "example_id": "counterfactual-negative",
            "recipe": {
                "seed": 11,
                "split": "train",
                "source_id": "nonwake-source",
                "source_sha256": "nonwake-source",
                "slot": 0,
                "label": "negative",
                "source_kind": "speech",
                "augmentation_group": "environment",
                "speed": 1.0,
                "jitter_ms": 0,
                "online_window_jitter_max_ms": 0,
                "active_rms_dbfs": -20.0,
                "noise_scene": None,
                "snr_db": None,
                "apply_rir": False,
                "apply_interferer": False,
                "sir_db": None,
            },
            "foreground_path": str(source_path),
            "active_span": None,
            "source_role": "base_negative",
            "counterpart_kind": "counterfactual_augmented_speech_negative",
        }
    )

    metadata = render_request(request, tmp_path / "counterfactual.wav", torch.device("cpu"))

    assert metadata["counterpart_kind"] == "counterfactual_augmented_speech_negative"


def test_raw_score_hard_negative_uses_its_explicit_scored_window(tmp_path: Path) -> None:
    source_path = tmp_path / "raw_negative.wav"
    samples = [0.2] * 4_000 + [0.0] * 24_000 + [0.2] * 4_000
    sf.write(source_path, samples, 16_000, subtype="PCM_16")
    recipe = Recipe(
        seed=0,
        split="train",
        source_id="raw-negative-source",
        source_sha256="raw-negative-source",
        slot=0,
        label="negative",
        source_kind="speech",
        augmentation_group="raw_hard_negative",
        speed=1.0,
        jitter_ms=0,
        online_window_jitter_max_ms=0,
        active_rms_dbfs=-20.0,
        noise_scene=None,
        snr_db=None,
        apply_rir=False,
        apply_interferer=False,
        sir_db=None,
    )
    request = GenerationRequest(
        example_id="raw-score-explicit-window",
        recipe=recipe,
        foreground_path=source_path,
        active_span=None,
        foreground_start_sample=16_000,
        source_role="raw_score_hard_negative",
    )

    metadata = render_request(request, tmp_path / "raw_score_rendered.wav", torch.device("cpu"))
    rendered, _ = sf.read(tmp_path / "raw_score_rendered.wav", dtype="float32")

    assert metadata["source_start_sample"] == 16_000
    assert abs(rendered[:4_000]).mean() < 1e-4
    assert abs(rendered[-4_000:]).mean() > 1e-2


def test_false_wake_shard_reuses_decoded_foreground_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path = tmp_path / "false_wake.wav"
    sf.write(source_path, [0.2] * 32_000, 16_000, subtype="PCM_16")
    recipe = Recipe(
        seed=3,
        split="train",
        source_id="false-wake-source",
        source_sha256="false-wake-source",
        slot=0,
        label="negative",
        source_kind="false_wake",
        augmentation_group="false_wake_dense",
        speed=1.0,
        jitter_ms=0,
        online_window_jitter_max_ms=0,
        active_rms_dbfs=-20.0,
        noise_scene=None,
        snr_db=None,
        apply_rir=False,
        apply_interferer=False,
        sir_db=None,
    )
    requests = [
        GenerationRequest(
            example_id=f"false-wake-cache-{start}",
            recipe=recipe,
            foreground_path=source_path,
            active_span=None,
            foreground_start_sample=start,
            source_role="false_wake_hard_negative",
        )
        for start in (0, 1_536)
    ]
    original_load = generator_module._load_16k
    loads = 0

    def count_foreground_load(path: Path, device: torch.device, sample_rate: int = 16_000) -> torch.Tensor:
        nonlocal loads
        loads += 1
        return original_load(path, device, sample_rate)

    monkeypatch.setattr(generator_module, "_load_16k", count_foreground_load)

    generate_shard(requests, tmp_path / "output", rank=0, world_size=1, device=torch.device("cpu"))

    assert loads == 1


def test_generation_shard_batches_completion_metadata_syncs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_path = tmp_path / "false_wake.wav"
    sf.write(source_path, [0.2] * 32_000, 16_000, subtype="PCM_16")
    recipe = Recipe(
        seed=5,
        split="train",
        source_id="false-wake-source",
        source_sha256="false-wake-source",
        slot=0,
        label="negative",
        source_kind="false_wake",
        augmentation_group="false_wake_dense",
        speed=1.0,
        jitter_ms=0,
        online_window_jitter_max_ms=0,
        active_rms_dbfs=-20.0,
        noise_scene=None,
        snr_db=None,
        apply_rir=False,
        apply_interferer=False,
        sir_db=None,
    )
    requests = [
        GenerationRequest(
            example_id=f"false-wake-sync-{start}",
            recipe=recipe,
            foreground_path=source_path,
            active_span=None,
            foreground_start_sample=start,
            source_role="false_wake_hard_negative",
        )
        for start in (0, 1_536)
    ]
    original_fsync = generator_module.os.fsync
    syncs = 0

    def count_completion_sync(descriptor: int) -> None:
        nonlocal syncs
        syncs += 1
        original_fsync(descriptor)

    monkeypatch.setattr(generator_module.os, "fsync", count_completion_sync)

    generate_shard(requests, tmp_path / "output", rank=0, world_size=1, device=torch.device("cpu"))

    assert syncs == 1


def test_v3_provenance_rejects_final_source_and_incomplete_window() -> None:
    assert validate_v3_provenance is not None, "V3 provenance gate is not implemented"
    final = _environment_negative("final", source_sha256="final-source")
    with pytest.raises(ValueError, match="final-test"):
        validate_v3_provenance([final], final_source_hashes={"final-source"})

    incomplete = _environment_negative("incomplete")
    incomplete["frames"] = 15_999
    with pytest.raises(ValueError, match="one-second"):
        validate_v3_provenance([incomplete], final_source_hashes=set())


def test_v3_train_build_gate_calls_augmentation_closure() -> None:
    assert audit_v3_train_build is not None, "V3 build gate is not implemented"
    positive = _environment_negative("positive")
    positive.update(
        {
            "label": "positive",
            "noise_source_sha256": "noise-a",
            "noise_scene": "road",
            "noise_rms_band": "-35:-30",
            "rir_applied": True,
            "rir_source_sha256": "rir-a",
            "interferer_applied": True,
            "interferer_source_sha256": "speech-a",
            "speed": 0.9,
        }
    )

    with pytest.raises(ValueError, match="counterfactual"):
        audit_v3_train_build(
            [positive],
            final_source_hashes=set(),
            environment_roles={},
        )


def test_v3_build_exempts_explicit_unaugmented_raw_anchors_from_closure() -> None:
    assert audit_v3_train_build is not None, "V3 build gate is not implemented"
    raw_positive = _environment_negative("raw-positive")
    raw_positive.update(
        {
            "label": "positive",
            "source_role": "raw_positive",
            "augmentation_closure_required": False,
        }
    )
    raw_negative = _environment_negative("raw-negative")
    raw_negative.update(
        {
            "source_role": "raw_negative",
            "augmentation_closure_required": False,
        }
    )
    for row in (raw_positive, raw_negative):
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
            "counterpart_kind",
        ):
            row.pop(field)

    report = audit_v3_train_build(
        [raw_positive, raw_negative],
        final_source_hashes=set(),
        environment_roles={},
    )

    assert report["augmentation_closure"]["positive_count"] == 0
    assert report["closure_exempt_record_count"] == 2


def test_v3_build_rejects_an_augmented_record_marked_as_closure_exempt() -> None:
    assert audit_v3_train_build is not None, "V3 build gate is not implemented"
    augmented = _environment_negative("bad-exemption")
    augmented.update(
        {
            "label": "positive",
            "source_role": "raw_positive",
            "augmentation_closure_required": False,
            "noise_source_sha256": "noise-a",
            "noise_scene": "road",
            "noise_rms_band": "-35:-30",
        }
    )

    with pytest.raises(ValueError, match="cannot exempt"):
        audit_v3_train_build(
            [augmented, _environment_negative("negative")],
            final_source_hashes=set(),
            environment_roles={},
        )


def _write_source_plan(path: Path, plans: list[dict[str, object]]) -> Path:
    path.write_text(
        json.dumps({"schema_version": 1, "plans": plans}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _source_plan_entry(
    *,
    path: str,
    scene: str,
    domain: str,
    source_sha256: str,
    split: str,
    allowed_end: int,
) -> dict[str, object]:
    return {
        "source": {
            "path": path,
            "scene": scene,
            "domain": domain,
            "source_sha256": source_sha256,
            "frames": allowed_end,
        },
        "blocks": [
            {
                "source_sha256": source_sha256,
                "path": path,
                "scene": scene,
                "domain": domain,
                "split": split,
                "block_index": 0,
                "block_start": 0,
                "block_end": allowed_end,
                "allowed_start": 0,
                "allowed_end": allowed_end,
            }
        ],
    }


def test_environment_request_writer_balances_train_sources_and_locks_provenance(
    tmp_path: Path,
) -> None:
    from dscnn_kws.data.reclean import v3_build

    writer = getattr(v3_build, "write_environment_negative_requests", None)
    assert writer is not None, "V3 environmental request writer is not implemented"
    source_plan = _write_source_plan(
        tmp_path / "source_plan.json",
        [
            _source_plan_entry(
                path="/captured/long.wav",
                scene="road",
                domain="captured",
                source_sha256="captured-long",
                split="train",
                allowed_end=16_000 + 3 * 1536,
            ),
            _source_plan_entry(
                path="/captured/short.wav",
                scene="wind",
                domain="captured",
                source_sha256="captured-short",
                split="train",
                allowed_end=16_000,
            ),
            _source_plan_entry(
                path="/captured/held-out.wav",
                scene="road",
                domain="captured",
                source_sha256="captured-test",
                split="test",
                allowed_end=16_000,
            ),
        ],
    )
    manifest = tmp_path / "captured.jsonl"

    report = writer(
        source_plan,
        manifest,
        role="captured_environment_negative",
        target_count=3,
        seed=23,
    )
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]

    assert report["candidate_count"] == 5
    assert report["selected_count"] == 3
    assert {row["noise_source_sha256"] for row in rows} == {"captured-long", "captured-short"}
    assert all(row["recipe"]["split"] == "train" for row in rows)
    assert all(row["recipe"]["source_kind"] == "pure_noise" for row in rows)
    assert all(row["recipe"]["label"] == "negative" for row in rows)
    assert all(row["noise_scene"] == row["resolved_noise_scene"] for row in rows)
    assert all(row["source_role"] == "captured_environment_negative" for row in rows)
    assert {row["source_candidate_count"] for row in rows if row["noise_source_sha256"] == "captured-long"} == {4}
    assert {row["source_candidate_count"] for row in rows if row["noise_source_sha256"] == "captured-short"} == {1}


def test_environment_request_render_emits_complete_train_provenance(tmp_path: Path) -> None:
    from dscnn_kws.data.reclean import v3_build

    source_audio = tmp_path / "road.wav"
    save_pcm16_atomic(
        source_audio,
        torch.linspace(-0.25, 0.25, 32_000, dtype=torch.float32).view(1, -1),
        sample_rate=16_000,
    )
    source_plan = _write_source_plan(
        tmp_path / "source_plan.json",
        [
            _source_plan_entry(
                path=str(source_audio),
                scene="road",
                domain="captured",
                source_sha256="road-source",
                split="train",
                allowed_end=32_000,
            )
        ],
    )
    request_manifest = tmp_path / "captured.jsonl"
    v3_build.write_environment_negative_requests(
        source_plan,
        request_manifest,
        role="captured_environment_negative",
        target_count=1,
        seed=23,
    )
    request = GenerationRequest.from_dict(
        json.loads(request_manifest.read_text(encoding="utf-8").splitlines()[0])
    )

    metadata = render_request(request, tmp_path / "rendered.wav", torch.device("cpu"))

    assert metadata["source_split"] == "train"
    assert metadata["source_sha256"] == "road-source"
    assert metadata["source_start_sample"] == metadata["noise_start_sample"]
    assert metadata["source_candidate_count"] == 11
    assert metadata["sample_rate"] == 16_000
    assert metadata["frames"] == 16_000
    assert metadata["noise_scene"] == "road"
    assert metadata["resolved_noise_scene"] == "road"
    assert metadata["rir_applied"] is False
    assert metadata["interferer_applied"] is False


def test_environment_shard_reuses_one_decoded_source_for_consecutive_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dscnn_kws.data.reclean import generator, v3_build

    source_audio = tmp_path / "road.wav"
    save_pcm16_atomic(
        source_audio,
        torch.linspace(-0.25, 0.25, 32_000, dtype=torch.float32).view(1, -1),
        sample_rate=16_000,
    )
    source_plan = _write_source_plan(
        tmp_path / "source_plan.json",
        [
            _source_plan_entry(
                path=str(source_audio),
                scene="road",
                domain="captured",
                source_sha256="road-source",
                split="train",
                allowed_end=32_000,
            )
        ],
    )
    request_manifest = tmp_path / "captured.jsonl"
    v3_build.write_environment_negative_requests(
        source_plan,
        request_manifest,
        role="captured_environment_negative",
        target_count=2,
        seed=23,
    )
    requests = [
        GenerationRequest.from_dict(json.loads(line))
        for line in request_manifest.read_text(encoding="utf-8").splitlines()
    ]
    original_loader = generator._load_16k
    load_count = 0

    def counted_loader(path: Path, device: torch.device, sample_rate: int = 16_000) -> torch.Tensor:
        nonlocal load_count
        load_count += 1
        return original_loader(path, device, sample_rate)

    monkeypatch.setattr(generator, "_load_16k", counted_loader)
    telemetry = generate_shard(
        requests,
        tmp_path / "output",
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
    )

    assert telemetry["generated_examples"] == 2
    assert load_count == 1


def test_counterfactual_requests_share_the_full_strong_composition_signature() -> None:
    from dscnn_kws.data.reclean import v3_build

    builder = getattr(v3_build, "iter_counterfactual_augmentation_requests", None)
    assert builder is not None, "V3 counterfactual augmentation planner is not implemented"
    positive = {
        "source_kind": "speech",
        "source_label": "positive",
        "source_split": "train",
        "source_sha256": "positive-source",
        "prepared_path": "/speech/positive.wav",
        "active_start": 1_000,
        "active_end": 12_000,
    }
    negative = {
        "source_kind": "speech",
        "source_label": "negative",
        "source_split": "train",
        "source_sha256": "negative-source",
        "prepared_path": "/speech/negative.wav",
    }
    noise = {
        "noise_path": "/noise/road.wav",
        "noise_start_sample": 1_536,
        "noise_source_sha256": "road-source",
        "noise_scene": "road",
        "resolved_noise_scene": "road",
    }
    rir = {
        "prepared_path": "/rir/room.wav",
        "source_sha256": "rir-source",
    }
    interferer = {
        "prepared_path": "/speech/interferer.wav",
        "source_sha256": "interferer-source",
    }

    requests = list(
        builder(
            positives=[positive],
            nonwake_speech=[negative],
            noise_candidates=[noise],
            rirs=[rir],
            interferers=[interferer],
            variants_per_positive=45,
            seed=20260831,
        )
    )
    pairs: dict[str, list[dict[str, object]]] = {}
    for request in requests:
        pairs.setdefault(str(request["counterpart_id"]), []).append(request)
    strong_pair = next(
        pair for pair in pairs.values()
        if pair[0]["recipe"]["augmentation_group"] == "strong_composition"
    )
    assert len(strong_pair) == 2
    positive_request = next(row for row in strong_pair if row["recipe"]["label"] == "positive")
    negative_request = next(row for row in strong_pair if row["recipe"]["label"] == "negative")

    assert positive_request["source_role"] == "base_positive"
    assert negative_request["source_role"] == "base_negative"
    assert negative_request["counterpart_kind"] == "counterfactual_augmented_speech_negative"
    assert positive_request["noise_path"] == negative_request["noise_path"]
    assert positive_request["noise_start_sample"] == negative_request["noise_start_sample"]
    assert positive_request["noise_source_sha256"] == negative_request["noise_source_sha256"]
    assert positive_request["noise_scene"] == negative_request["noise_scene"]
    assert positive_request["rir_path"] == negative_request["rir_path"]
    assert positive_request["rir_source_sha256"] == negative_request["rir_source_sha256"]
    assert positive_request["interferer_path"] == negative_request["interferer_path"]
    assert positive_request["interferer_source_sha256"] == negative_request["interferer_source_sha256"]
    for field in ("speed", "active_rms_dbfs", "noise_scene", "snr_db", "apply_rir", "apply_interferer", "sir_db"):
        assert positive_request["recipe"][field] == negative_request["recipe"][field]
    assert positive_request["foreground_path"] != negative_request["foreground_path"]
    assert positive_request["active_span"] == [1_000, 12_000]
    assert negative_request["active_span"] is None


def test_counterfactual_request_publisher_atomically_separates_paired_roles(tmp_path: Path) -> None:
    from dscnn_kws.data.reclean import v3_build

    writer = getattr(v3_build, "write_counterfactual_augmentation_requests", None)
    assert writer is not None, "V3 counterfactual request publisher is not implemented"
    positive = {
        "source_kind": "speech",
        "source_label": "positive",
        "source_split": "train",
        "source_sha256": "positive-source",
        "prepared_path": "/speech/positive.wav",
        "active_start": 1_000,
        "active_end": 12_000,
    }
    negative = {
        "source_kind": "speech",
        "source_label": "negative",
        "source_split": "train",
        "source_sha256": "negative-source",
        "prepared_path": "/speech/negative.wav",
    }
    noise = {
        "noise_path": "/noise/road.wav",
        "noise_start_sample": 1_536,
        "noise_source_sha256": "road-source",
        "noise_scene": "road",
        "resolved_noise_scene": "road",
    }
    report = writer(
        destination_root=tmp_path / "requests",
        positives=[positive],
        nonwake_speech=[negative],
        noise_candidates=[noise],
        rirs=[{"prepared_path": "/rir/room.wav", "source_sha256": "rir-source"}],
        interferers=[{"prepared_path": "/speech/interferer.wav", "source_sha256": "interferer-source"}],
        variants_per_positive=1,
        seed=20260831,
    )

    positive_rows = [
        json.loads(line)
        for line in Path(report["role_manifests"]["base_positive"]).read_text(encoding="utf-8").splitlines()
    ]
    negative_rows = [
        json.loads(line)
        for line in Path(report["role_manifests"]["base_negative"]).read_text(encoding="utf-8").splitlines()
    ]
    assert report["counts"] == {"base_positive": 1, "base_negative": 1}
    assert [row["source_role"] for row in positive_rows] == ["base_positive"]
    assert [row["source_role"] for row in negative_rows] == ["base_negative"]
    assert positive_rows[0]["counterpart_id"] == negative_rows[0]["counterpart_id"]


def test_source_balanced_noise_plan_covers_every_scene_source_and_window_before_reuse() -> None:
    from dscnn_kws.data.reclean import v3_build

    planner = getattr(v3_build, "plan_source_balanced_noise_bindings", None)
    assert planner is not None, "V3 source-balanced noise planner is not implemented"
    candidates = [
        {
            "noise_path": f"/{scene}/{source}.wav",
            "noise_start_sample": start,
            "noise_source_sha256": source,
            "noise_scene": scene,
            "resolved_noise_scene": scene,
        }
        for scene in ("road", "tau")
        for source in (f"{scene}-a", f"{scene}-b")
        for start in (0, 1536)
    ]

    planned = planner(candidates, count=8, seed=20260831)

    assert len(planned) == 8
    assert {row["noise_scene"] for row in planned} == {"road", "tau"}
    assert {row["noise_source_sha256"] for row in planned} == {
        "road-a", "road-b", "tau-a", "tau-b"
    }
    assert len({(row["noise_source_sha256"], row["noise_start_sample"]) for row in planned}) == 8
    assert sum(row["noise_scene"] == "road" for row in planned) == 4
    assert sum(row["noise_scene"] == "tau" for row in planned) == 4


def test_noise_binding_audit_rejects_a_collapsed_schedule_with_available_diversity() -> None:
    from dscnn_kws.data.reclean import v3_build

    audit = getattr(v3_build, "audit_source_balanced_noise_bindings", None)
    assert audit is not None, "V3 source-balanced noise audit is not implemented"
    candidates = [
        {
            "noise_path": f"/{scene}/{source}.wav",
            "noise_start_sample": start,
            "noise_source_sha256": source,
            "noise_scene": scene,
            "resolved_noise_scene": scene,
        }
        for scene in ("road", "tau")
        for source in (f"{scene}-a", f"{scene}-b")
        for start in (0, 1536)
    ]

    with pytest.raises(ValueError, match="does not match the source-balanced plan"):
        audit(candidates, [candidates[0]] * 8, seed=20260831)


def test_repair_generation_metadata_publishes_one_identical_completion_per_request(
    tmp_path: Path,
) -> None:
    from dscnn_kws.data.reclean import v3_build

    repair = getattr(v3_build, "repair_generation_metadata", None)
    assert repair is not None, "V3 generation metadata repair is not implemented"
    requests = [
        {
            "example_id": "example-a",
            "recipe": {"seed": 1},
            "label": "negative",
            "noise_path": "/noise/a.wav",
            "noise_start_sample": 0,
            "noise_source_sha256": "noise-a",
            "source_role": "tau_environment_negative",
        },
        {
            "example_id": "example-b",
            "recipe": {"seed": 2},
            "label": "negative",
            "noise_path": "/noise/b.wav",
            "noise_start_sample": 1536,
            "noise_source_sha256": "noise-b",
            "source_role": "tau_environment_negative",
        },
    ]
    request_path = tmp_path / "requests.jsonl"
    request_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in requests),
        encoding="utf-8",
    )
    metadata_root = tmp_path / "metadata"
    metadata_root.mkdir()
    metadata_rows = [
        {
            **request,
            "output_path": str(tmp_path / f"{request['example_id']}.wav"),
            "output_sha256": f"hash-{request['example_id']}",
        }
        for request in requests
    ]
    (metadata_root / "shard-00.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in (metadata_rows[0], metadata_rows[0], metadata_rows[1])
        ),
        encoding="utf-8",
    )

    report = repair(request_path, metadata_root, tmp_path / "metadata_repaired")
    repaired_rows = [
        json.loads(line)
        for line in (tmp_path / "metadata_repaired" / "shard-00.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert report["expected_count"] == 2
    assert report["unique_count"] == 2
    assert report["duplicate_count"] == 1
    assert [row["example_id"] for row in repaired_rows] == ["example-a", "example-b"]


def test_repair_generation_metadata_does_not_publish_a_partial_result(tmp_path: Path) -> None:
    from dscnn_kws.data.reclean import v3_build

    requests = [
        {
            "example_id": "example-a",
            "recipe": {"seed": 1},
            "label": "negative",
            "noise_path": "/noise/a.wav",
            "noise_start_sample": 0,
            "noise_source_sha256": "noise-a",
            "source_role": "tau_environment_negative",
        },
        {
            "example_id": "example-b",
            "recipe": {"seed": 2},
            "label": "negative",
            "noise_path": "/noise/b.wav",
            "noise_start_sample": 1536,
            "noise_source_sha256": "noise-b",
            "source_role": "tau_environment_negative",
        },
    ]
    request_path = tmp_path / "requests.jsonl"
    request_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in requests),
        encoding="utf-8",
    )
    metadata_root = tmp_path / "metadata"
    metadata_root.mkdir()
    (metadata_root / "shard-00.jsonl").write_text(
        json.dumps({**requests[0], "output_path": "a.wav", "output_sha256": "hash-a"}) + "\n",
        encoding="utf-8",
    )
    repaired_root = tmp_path / "metadata_repaired"

    with pytest.raises(ValueError, match="missing 1 expected requests"):
        v3_build.repair_generation_metadata(request_path, metadata_root, repaired_root)

    assert not repaired_root.exists()
