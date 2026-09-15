from __future__ import annotations

from pathlib import Path
import json

import torch

from dscnn_kws.data.reclean import recipes
from dscnn_kws.data.reclean.audio import save_pcm16_atomic
from dscnn_kws.data.reclean.generator import GenerationRequest, render_request
from dscnn_kws.data.reclean import v3_sources
from dscnn_kws.data.reclean.v3_sources import (
    SourceAudio,
    SourceBlock,
    assign_tau_source_splits,
    dense_window_candidates,
    plan_false_wake_recording,
    plan_source_time_blocks,
)
from dscnn_kws.data.reclean.catalog import CAPTURED_SCENES, TAU_SCENES


def _derive_v3_seed(*args: object) -> int:
    function = getattr(recipes, "derive_v3_seed", None)
    assert function is not None, "V3 source-time recipe identity is not implemented"
    return function(*args)


def test_plan_long_source_blocks_excludes_cross_split_collar() -> None:
    plan = plan_source_time_blocks(
        SourceAudio("road.wav", "road", "captured", "road-hash", frames=4_800_000),
        seed=7,
        block_seconds=10,
        collar_seconds=1,
    )

    assert {block.split for block in plan.blocks} == {"train", "validation", "test"}
    by_index = {block.block_index: block for block in plan.blocks}
    for index, block in by_index.items():
        for neighbour in (by_index.get(index - 1), by_index.get(index + 1)):
            if neighbour is not None and neighbour.split != block.split:
                if neighbour.block_index < block.block_index:
                    assert block.allowed_start >= block.block_start + 16_000
                else:
                    assert block.allowed_end <= block.block_end - 16_000


def test_short_livingroom_is_final_test_only() -> None:
    plan = plan_source_time_blocks(
        SourceAudio("livingroom.wav", "livingroom", "captured", "lr-hash", frames=48_000),
        seed=7,
        final_only=True,
    )

    assert len(plan.blocks) == 1
    assert plan.blocks[0].split == "test"
    assert plan.blocks[0].allowed_start == 0
    assert plan.blocks[0].allowed_end == 48_000


def test_dense_candidates_are_complete_windows_at_96ms_hop() -> None:
    block = SourceBlock(
        source_sha256="h",
        path="a.wav",
        scene="road",
        domain="captured",
        split="train",
        block_index=0,
        block_start=0,
        block_end=32_000,
        allowed_start=1_000,
        allowed_end=31_000,
    )

    starts = [row.start_sample for row in dense_window_candidates([block], window_samples=16_000, hop_samples=1536)]

    assert starts == [1_000, 2_536, 4_072, 5_608, 7_144, 8_680, 10_216, 11_752, 13_288, 14_824]


def test_tau_assignment_is_file_disjoint_and_scene_stratified() -> None:
    sources = [
        SourceAudio(f"{scene}-{index}.wav", scene, "tau", f"{scene}-{index}", frames=160_000)
        for scene in ("airport", "bus")
        for index in range(10)
    ]

    assignment = assign_tau_source_splits(sources, seed=13)

    assert set(assignment) == {source.source_sha256 for source in sources}
    assert set(assignment.values()) == {"train", "validation", "test"}
    for scene in ("airport", "bus"):
        scene_splits = {assignment[source.source_sha256] for source in sources if source.scene == scene}
        assert scene_splits == {"train", "validation", "test"}


def test_false_wake_recording_policy_never_exposes_yc_or_yd_to_final_test() -> None:
    yc = SourceAudio("误唤醒_yc_30.wav", "false_wake", "false_wake", "yc", frames=480_000)
    yd = SourceAudio("误唤醒_yd_30.wav", "false_wake", "false_wake", "yd", frames=480_000)

    yc_splits = {block.split for block in plan_false_wake_recording(yc, seed=5).blocks}
    yd_splits = {block.split for block in plan_false_wake_recording(yd, seed=5).blocks}

    assert yc_splits == {"train", "validation"}
    assert yd_splits == {"train", "validation"}


def test_false_wake_zh_is_development_only_and_cs_is_final_only() -> None:
    zh = SourceAudio("误唤醒_zh_30.wav", "false_wake", "false_wake", "zh", frames=480_000)
    cs = SourceAudio("误唤醒_cs_30.wav", "false_wake", "false_wake", "cs", frames=480_000)

    assert {block.split for block in plan_false_wake_recording(zh, seed=5).blocks} == {"validation"}
    assert {block.split for block in plan_false_wake_recording(cs, seed=5).blocks} == {"test"}


def test_v3_seed_is_unique_for_source_time_role_and_variant_identity() -> None:
    assert _derive_v3_seed(42, "train", "noise-sha", 0, "captured_environment", 0) != _derive_v3_seed(
        42, "train", "noise-sha", 1536, "captured_environment", 0
    )
    seeds = {
        _derive_v3_seed(42, "train", "noise-sha", 1536, "captured_environment", variant)
        for variant in range(49)
    }
    assert len(seeds) == 49


def test_v3_noise_request_binds_declared_scene_to_the_resolved_source() -> None:
    function = getattr(recipes, "build_v3_noise_request", None)
    assert function is not None, "V3 exact noise request construction is not implemented"
    road_candidate = {
        "path": "/noise/road.wav",
        "scene": "road",
        "source_sha256": "noise-sha",
        "start_sample": 1536,
    }

    request = function(candidate=road_candidate, role="captured_environment", seed=42)

    assert request["recipe"]["noise_scene"] == "road"
    assert request["noise_path"] == "/noise/road.wav"
    assert request["noise_start_sample"] == 1536
    assert request["noise_source_sha256"] == "noise-sha"
    assert request["resolved_noise_scene"] == "road"


def test_rendered_v3_noise_provenance_preserves_exact_source_start(tmp_path: Path) -> None:
    noise_path = tmp_path / "road.wav"
    save_pcm16_atomic(
        noise_path,
        torch.linspace(-0.25, 0.25, 32_100, dtype=torch.float32).view(1, -1),
        sample_rate=16000,
    )
    raw_request = {
        "example_id": "v3-noise-0001",
        "recipe": {
            "seed": 7,
            "split": "train",
            "source_id": "noise-sha",
            "source_sha256": "noise-sha",
            "slot": 0,
            "label": "negative",
            "source_kind": "pure_noise",
            "augmentation_group": "environment",
            "speed": 1.0,
            "jitter_ms": 0,
            "online_window_jitter_max_ms": 0,
            "active_rms_dbfs": None,
            "noise_scene": "road",
            "snr_db": None,
            "apply_rir": False,
            "apply_interferer": False,
            "sir_db": None,
        },
        "foreground_path": None,
        "active_span": None,
        "noise_path": str(noise_path),
        "noise_start_sample": 1536,
        "noise_source_sha256": "noise-sha",
        "noise_source_scene": "road",
        "resolved_noise_scene": "road",
        "source_role": "captured_environment_negative",
    }

    metadata = render_request(
        GenerationRequest.from_dict(raw_request),
        tmp_path / "rendered.wav",
        torch.device("cpu"),
    )

    assert metadata["noise_start_sample"] == 1536
    assert metadata["noise_source_sha256"] == "noise-sha"
    assert metadata["noise_source_scene"] == "road"
    assert metadata["declared_noise_scene"] == "road"
    assert metadata["resolved_noise_scene"] == "road"


def test_rendered_v3_speech_provenance_preserves_rir_and_interferer_hashes(tmp_path: Path) -> None:
    foreground_path = tmp_path / "foreground.wav"
    rir_path = tmp_path / "rir.wav"
    interferer_path = tmp_path / "interferer.wav"
    save_pcm16_atomic(
        foreground_path,
        torch.linspace(-0.2, 0.2, 32_000, dtype=torch.float32).view(1, -1),
        sample_rate=16_000,
    )
    impulse = torch.zeros(1, 16_000, dtype=torch.float32)
    impulse[0, 0] = 1.0
    save_pcm16_atomic(rir_path, impulse, sample_rate=16_000)
    save_pcm16_atomic(
        interferer_path,
        torch.linspace(0.1, -0.1, 32_000, dtype=torch.float32).view(1, -1),
        sample_rate=16_000,
    )
    raw_request = {
        "example_id": "v3-speech-0001",
        "recipe": {
            "seed": 7,
            "split": "train",
            "source_id": "foreground-sha",
            "source_sha256": "foreground-sha",
            "slot": 0,
            "label": "positive",
            "source_kind": "speech",
            "augmentation_group": "strong_composition",
            "speed": 1.0,
            "jitter_ms": 0,
            "online_window_jitter_max_ms": 200,
            "active_rms_dbfs": -25.0,
            "noise_scene": None,
            "snr_db": None,
            "apply_rir": True,
            "apply_interferer": True,
            "sir_db": 0.0,
        },
        "foreground_path": str(foreground_path),
        "active_span": [4_000, 20_000],
        "rir_path": str(rir_path),
        "rir_source_sha256": "rir-sha",
        "interferer_path": str(interferer_path),
        "interferer_source_sha256": "interferer-sha",
        "source_role": "base_positive",
    }

    metadata = render_request(
        GenerationRequest.from_dict(raw_request),
        tmp_path / "rendered.wav",
        torch.device("cpu"),
    )

    assert metadata["rir_applied"] is True
    assert metadata["rir_source_sha256"] == "rir-sha"
    assert metadata["interferer_applied"] is True
    assert metadata["interferer_source_sha256"] == "interferer-sha"


def test_inventory_source_plan_locks_all_noise_domains_and_special_sources(tmp_path: Path) -> None:
    inventory: list[dict[str, object]] = []
    for scene in TAU_SCENES:
        inventory.extend(
            {
                "role": "noise",
                "path": f"/{scene}-{index}.wav",
                "scene": scene,
                "sha256": f"{scene}-{index}",
                "frames": 480_000,
            }
            for index in range(3)
        )
    for scene in CAPTURED_SCENES:
        inventory.append(
            {
                "role": "noise",
                "path": f"/{scene}.wav",
                "scene": scene,
                "sha256": f"{scene}-hash",
                "frames": 480_000,
            }
        )
    inventory.extend(
        {
            "role": "false_wake",
            "path": f"/误唤醒_{name}_30.wav",
            "sha256": f"{name}-hash",
            "frames": 480_000,
        }
        for name in ("yc", "yd", "zh", "cs")
    )

    planner = getattr(v3_sources, "plan_v3_inventory_sources", None)
    writer = getattr(v3_sources, "write_source_plan", None)
    assert planner is not None, "V3 inventory source planner is not implemented"
    assert writer is not None, "V3 source-plan publisher is not implemented"
    plans = planner(inventory, seed=19)
    written = writer(tmp_path / "source_plan.json", plans)
    payload = json.loads(written.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert len(payload["plans"]) == len(TAU_SCENES) * 3 + len(CAPTURED_SCENES) + 4
    by_path = {plan.source.path: plan for plan in plans}
    assert {block.split for block in by_path["/livingroom.wav"].blocks} == {"test"}
    assert {block.split for block in by_path["/pub.wav"].blocks} == {"test"}
    assert {block.split for block in by_path["/误唤醒_zh_30.wav"].blocks} == {"validation"}
    assert {block.split for block in by_path["/误唤醒_cs_30.wav"].blocks} == {"test"}
    for scene in TAU_SCENES:
        splits = {by_path[f"/{scene}-{index}.wav"].blocks[0].split for index in range(3)}
        assert splits == {"train", "validation", "test"}


def test_inventory_planner_quarantines_environment_audio_shorter_than_one_window() -> None:
    inventory: list[dict[str, object]] = []
    for scene in TAU_SCENES:
        inventory.extend(
            {
                "role": "noise",
                "path": f"/{scene}-{index}.wav",
                "scene": scene,
                "sha256": f"{scene}-{index}",
                "frames": 480_000,
            }
            for index in range(3)
        )
    for scene in CAPTURED_SCENES:
        inventory.append(
            {
                "role": "noise",
                "path": f"/{scene}.wav",
                "scene": scene,
                "sha256": f"{scene}-hash",
                "frames": 480_000,
            }
        )
    inventory.append(
        {
            "role": "noise",
            "path": "/wind-short.wav",
            "scene": "风噪",
            "sha256": "wind-short",
            "frames": 3_636,
        }
    )
    inventory.extend(
        {
            "role": "false_wake",
            "path": f"/误唤醒_{name}_30.wav",
            "sha256": f"{name}-hash",
            "frames": 480_000,
        }
        for name in ("yc", "yd", "zh", "cs")
    )

    planner = getattr(v3_sources, "plan_v3_inventory_sources", None)
    assert planner is not None, "V3 inventory source planner is not implemented"
    plans = planner(inventory, seed=19)

    assert "wind-short" not in {plan.source.source_sha256 for plan in plans}
