from collections import Counter

from dscnn_kws.data.reclean import recipes
from dscnn_kws.data.reclean.recipes import (
    build_generation_requests,
    build_pure_noise_recipe,
    build_recipe,
    build_speech_recipe,
    plan_output_quotas,
)


def test_recipe_seed_is_reproducible_and_composable():
    first = build_recipe(42, "train", "source-x", "hash-x", 17)
    second = build_recipe(42, "train", "source-x", "hash-x", 17)

    assert first == second
    assert first.speed in (0.9, 1.0, 1.25)
    assert first.jitter_ms == 0
    assert first.online_window_jitter_max_ms == 200


def test_label_contract_for_pure_noise_and_noisy_speech():
    assert build_pure_noise_recipe(42, 1).label == "negative"
    assert build_speech_recipe(42, "positive", 1).label == "positive"
    assert build_speech_recipe(42, "negative", 1).label == "negative"


def test_current_quota_arithmetic_is_exact():
    assert plan_output_quotas(21825, 48) == {
        "positive": 1047600,
        "speech_negative": 680940,
        "false_wake_negative": 104760,
        "pure_noise_negative": 261900,
    }


def test_48_slot_matrix_has_the_approved_coverage_groups():
    groups = Counter(build_recipe(42, "train", "source", "hash", slot).augmentation_group for slot in range(48))

    assert groups == {
        "clean_time_placement": 4,
        "environment": 14,
        "speed_jitter_environment": 12,
        "rir_environment": 8,
        "interferer_speech": 6,
        "strong_composition": 4,
    }


def test_clean_slots_use_distinct_deterministic_active_rms():
    first = [build_recipe(20260831, "train", "source", "source-sha", slot) for slot in range(4)]
    second = [build_recipe(20260831, "train", "source", "source-sha", slot) for slot in range(4)]

    assert [recipe.active_rms_dbfs for recipe in first] == [recipe.active_rms_dbfs for recipe in second]
    assert [recipe.jitter_ms for recipe in first] == [recipe.jitter_ms for recipe in second]
    assert len({recipe.active_rms_dbfs for recipe in first}) == 4
    assert len({recipe.jitter_ms for recipe in first}) == 4
    assert all(recipe.jitter_ms != 0 for recipe in first)


def test_interferer_sir_stays_in_the_approved_range():
    sir_values = {
        recipe.sir_db
        for source_index in range(20)
        for slot in range(48)
        if (recipe := build_recipe(42, "train", f"source-{source_index}", "hash", slot)).apply_interferer
    }

    assert sir_values <= {-5, 0, 5, 10}


def test_generation_requests_follow_exact_balanced_quotas_from_prepared_sources():
    positives = [
        {
            "role": "mobvoi_speech",
            "source_kind": "speech",
            "source_label": "positive",
            "source_split": "train",
            "source_sha256": f"positive-{index}",
            "prepared_sha256": f"prepared-positive-{index}",
            "prepared_path": f"positive-{index}.wav",
            "active_start": 0,
            "active_end": 16000,
        }
        for index in range(5)
    ]
    negative = {
        "role": "mobvoi_speech",
        "source_kind": "speech",
        "source_label": "negative",
        "source_split": "train",
        "source_sha256": "negative",
        "prepared_sha256": "prepared-negative",
        "prepared_path": "negative.wav",
    }
    false_wake = {
        "role": "false_wake",
        "source_kind": "false_wake",
        "source_label": "negative",
        "source_split": "train",
        "parent_source_sha256": "false-parent",
        "prepared_sha256": "prepared-false",
        "prepared_path": "false.wav",
    }
    noise = [
        {
            "role": "noise",
            "source_kind": "noise",
            "scene": scene,
            "source_sha256": f"noise-{scene}",
            "prepared_path": f"noise-{scene}.wav",
        }
        for scene in (
            "airport", "bus", "metro", "metro_station", "park", "public_square", "shopping_mall", "street_pedestrian",
            "street_traffic", "tram", "kindgarden", "livingroom", "pub", "road", "风噪",
        )
    ]
    rir = {"role": "rir", "source_kind": "rir", "source_sha256": "rir", "prepared_path": "rir.wav"}

    requests = list(build_generation_requests([*positives, negative, false_wake, *noise, rir], global_seed=42))

    assert len(requests) == 480
    assert sum(request["label"] == "positive" for request in requests) == 240
    assert sum(request["recipe"]["source_kind"] == "false_wake" for request in requests) == 24
    assert sum(request["recipe"]["source_kind"] == "pure_noise" for request in requests) == 60
    assert {
        request["recipe"]["source_id"]
        for request in requests
        if request["recipe"]["source_kind"] == "false_wake"
    } == {"false-parent"}
    assert {
        request["recipe"]["source_sha256"]
        for request in requests
        if request["recipe"]["source_kind"] == "false_wake"
    } == {"prepared-false"}


def test_generation_requests_index_noise_scenes_once(monkeypatch):
    positive = {
        "role": "mobvoi_speech",
        "source_kind": "speech",
        "source_label": "positive",
        "source_split": "train",
        "source_sha256": "positive",
        "prepared_sha256": "prepared-positive",
        "prepared_path": "positive.wav",
        "active_start": 0,
        "active_end": 16000,
    }
    negative = {
        "role": "mobvoi_speech",
        "source_kind": "speech",
        "source_label": "negative",
        "source_split": "train",
        "source_sha256": "negative",
        "prepared_sha256": "prepared-negative",
        "prepared_path": "negative.wav",
    }
    false_wake = {
        "role": "false_wake",
        "source_kind": "false_wake",
        "source_label": "negative",
        "source_split": "train",
        "parent_source_sha256": "false-parent",
        "prepared_sha256": "prepared-false",
        "prepared_path": "false.wav",
    }
    noise = [
        {"role": "noise", "source_kind": "noise", "scene": scene, "prepared_path": f"{scene}.wav"}
        for scene in recipes.ALL_NOISE_SCENES
    ]
    rir = {"role": "rir", "source_kind": "rir", "source_sha256": "rir", "prepared_path": "rir.wav"}
    original_index = recipes._index_noise_by_scene
    calls = 0

    def count_index(rows):
        nonlocal calls
        calls += 1
        return original_index(rows)

    monkeypatch.setattr(recipes, "_index_noise_by_scene", count_index)

    requests = list(
        build_generation_requests([positive, negative, false_wake, *noise, rir], global_seed=42, variants_per_positive=5)
    )

    assert requests
    assert calls == 1
