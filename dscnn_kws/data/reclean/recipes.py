from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from .catalog import ALL_NOISE_SCENES


SPEED_FACTORS = (0.9, 1.0, 1.25)
ACTIVE_RMS_DBFS = (-34.0, -30.0, -26.0, -22.0, -18.0)
CLEAN_STATIC_JITTER_MS = (-150, -50, 50, 150)
SNR_PROBABILITIES = ((-15, 0.30), (-10, 0.25), (-5, 0.15), (0, 0.10), (5, 0.08), (10, 0.05), (15, 0.04), (20, 0.03))
SIR_VALUES_DB = (-5, 0, 5, 10)
_COVERAGE_GROUPS = (
    *(("clean_time_placement",) * 4),
    *(("environment",) * 14),
    *(("speed_jitter_environment",) * 12),
    *(("rir_environment",) * 8),
    *(("interferer_speech",) * 6),
    *(("strong_composition",) * 4),
)


@dataclass(frozen=True)
class Recipe:
    seed: int
    split: str
    source_id: str
    source_sha256: str
    slot: int
    label: str
    source_kind: str
    augmentation_group: str
    speed: float
    jitter_ms: int
    online_window_jitter_max_ms: int
    active_rms_dbfs: float | None
    noise_scene: str | None
    snr_db: int | None
    apply_rir: bool
    apply_interferer: bool
    sir_db: int | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def derive_recipe_seed(global_seed: int, split: str, source_id: str, source_sha256: str, slot: int) -> int:
    payload = f"{global_seed}|{split}|{source_id}|{source_sha256}|{slot}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big", signed=False)


def derive_v3_seed(
    global_seed: int,
    split: str,
    source_sha256: str,
    start_sample: int,
    role: str,
    variant: int,
) -> int:
    """Derive a V3 seed from a complete source-time transformation identity."""
    if not split or not source_sha256 or not role:
        raise ValueError("V3 seed identity requires split, source_sha256, and role")
    if start_sample < 0 or variant < 0:
        raise ValueError("V3 seed start_sample and variant must be non-negative")
    identity = f"{global_seed}|{split}|{source_sha256}|{start_sample}|{role}|{variant}"
    return int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:8], "big")


def _weighted_snr(rng: random.Random) -> int:
    roll = rng.random()
    cumulative = 0.0
    for value, probability in SNR_PROBABILITIES:
        cumulative += probability
        if roll < cumulative:
            return value
    return SNR_PROBABILITIES[-1][0]


def _group_for_slot(slot: int) -> str:
    return _COVERAGE_GROUPS[slot % len(_COVERAGE_GROUPS)]


def _clean_slot_active_rms(global_seed: int, split: str, source_sha256: str, slot: int) -> float:
    """Select distinct source-specific levels for the four clean slots."""
    slot_in_cycle = slot % len(_COVERAGE_GROUPS)
    if not 0 <= slot_in_cycle < 4:
        raise ValueError("clean RMS selection requires a clean slot")
    values = list(ACTIVE_RMS_DBFS)
    seed = derive_v3_seed(global_seed, split, source_sha256, 0, "clean-active-rms", 0)
    random.Random(seed).shuffle(values)
    return float(values[slot_in_cycle])


def _clean_slot_jitter_ms(global_seed: int, split: str, source_sha256: str, slot: int) -> int:
    """Select distinct source-specific static placements for clean slots."""
    slot_in_cycle = slot % len(_COVERAGE_GROUPS)
    if not 0 <= slot_in_cycle < 4:
        raise ValueError("clean jitter selection requires a clean slot")
    values = list(CLEAN_STATIC_JITTER_MS)
    seed = derive_v3_seed(global_seed, split, source_sha256, 0, "clean-static-jitter", 0)
    random.Random(seed).shuffle(values)
    return int(values[slot_in_cycle])



def build_recipe(
    global_seed: int,
    split: str,
    source_id: str,
    source_sha256: str,
    slot: int,
    *,
    label: str = "positive",
    source_kind: str = "speech",
) -> Recipe:
    if label not in {"positive", "negative"}:
        raise ValueError(f"Unsupported label: {label}")
    if slot < 0:
        raise ValueError("slot must be non-negative")

    seed = derive_recipe_seed(global_seed, split, source_id, source_sha256, slot)
    rng = random.Random(seed)
    group = _group_for_slot(slot)
    uses_environment = group != "clean_time_placement"
    apply_rir = group in {"rir_environment", "strong_composition"}
    apply_interferer = group in {"interferer_speech", "strong_composition"}
    speed = rng.choice(SPEED_FACTORS) if group in {"speed_jitter_environment", "strong_composition"} else 1.0
    return Recipe(
        seed=seed,
        split=split,
        source_id=source_id,
        source_sha256=source_sha256,
        slot=slot,
        label=label,
        source_kind=source_kind,
        augmentation_group=group,
        speed=float(speed),
        jitter_ms=(
            _clean_slot_jitter_ms(global_seed, split, source_sha256, slot)
            if group == "clean_time_placement" and source_kind == "speech"
            else 0
        ),
        online_window_jitter_max_ms=200 if label == "positive" else 0,
        active_rms_dbfs=(
            None
            if source_kind == "pure_noise"
            else _clean_slot_active_rms(global_seed, split, source_sha256, slot)
            if group == "clean_time_placement"
            else float(rng.choice(ACTIVE_RMS_DBFS))
        ),
        noise_scene=rng.choice(ALL_NOISE_SCENES) if uses_environment else None,
        snr_db=_weighted_snr(rng) if uses_environment else None,
        apply_rir=apply_rir,
        apply_interferer=apply_interferer,
        sir_db=rng.choice(SIR_VALUES_DB) if apply_interferer else None,
    )


def build_speech_recipe(global_seed: int, label: str, slot: int, split: str = "train") -> Recipe:
    return build_recipe(
        global_seed,
        split,
        source_id=f"synthetic-speech-{label}",
        source_sha256=f"synthetic-speech-{label}",
        slot=slot,
        label=label,
        source_kind="speech",
    )


def build_pure_noise_recipe(global_seed: int, slot: int, split: str = "train") -> Recipe:
    return build_recipe(
        global_seed,
        split,
        source_id="synthetic-pure-noise",
        source_sha256="synthetic-pure-noise",
        slot=slot,
        label="negative",
        source_kind="pure_noise",
    )


def plan_output_quotas(positive_source_count: int, variants_per_positive: int) -> dict[str, int]:
    if positive_source_count <= 0 or variants_per_positive <= 0:
        raise ValueError("positive_source_count and variants_per_positive must be positive")
    positive = positive_source_count * variants_per_positive
    return {
        "positive": positive,
        "speech_negative": positive * 65 // 100,
        "false_wake_negative": positive * 10 // 100,
        "pure_noise_negative": positive * 25 // 100,
    }


def _stable_example_id(kind: str, ordinal: int, recipe: Recipe) -> str:
    digest = hashlib.sha256(f"{kind}|{ordinal}|{recipe.seed}".encode("utf-8")).hexdigest()[:20]
    return f"{kind}-{ordinal:07d}-{digest}"


def _choose_row(rows: list[dict[str, object]], seed: int) -> dict[str, object]:
    if not rows:
        raise ValueError("Cannot choose from an empty source pool")
    return rows[seed % len(rows)]


def _source_sha256(row: dict[str, object]) -> str:
    for field in ("source_sha256", "parent_source_sha256"):
        value = row.get(field)
        if value:
            return str(value)
    raise ValueError("Prepared source has no source SHA-256")


def _index_noise_by_scene(noise_rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    by_scene: dict[str, list[dict[str, object]]] = {scene: [] for scene in ALL_NOISE_SCENES}
    for row in noise_rows:
        scene = str(row.get("scene", ""))
        if scene in by_scene:
            by_scene[scene].append(row)
    missing = [scene for scene, rows in by_scene.items() if not rows]
    if missing:
        raise ValueError(f"Prepared noise is missing scenes: {', '.join(missing)}")
    return by_scene


def _choose_noise_by_scene(noise_by_scene: dict[str, list[dict[str, object]]], seed: int) -> dict[str, object]:
    scene = ALL_NOISE_SCENES[seed % len(ALL_NOISE_SCENES)]
    return _choose_row(noise_by_scene[scene], seed // len(ALL_NOISE_SCENES))


def resolve_noise_source(
    noise_by_scene: Mapping[str, list[dict[str, object]]],
    *,
    scene: str,
    seed: int,
) -> dict[str, object]:
    """Resolve a noise source only from the scene declared by the V3 recipe."""
    rows = noise_by_scene.get(scene, [])
    if not rows:
        raise ValueError(f"Prepared noise is missing declared scene: {scene}")
    row = _choose_row(rows, seed)
    resolved_scene = row.get("scene")
    if resolved_scene != scene:
        raise ValueError(
            f"Resolved noise scene mismatch: expected {scene}, got {resolved_scene!r}"
        )
    return row


def build_v3_noise_request(
    *,
    candidate: Mapping[str, object],
    role: str,
    seed: int,
) -> dict[str, object]:
    """Bind a V3 request to one exact noise source-time candidate.

    The returned binding is merged into a full generation request by the V3
    builder.  It intentionally does not choose a scene from a seed: the
    planned candidate's scene is the declaration and the resolution.
    """
    path = candidate.get("path", candidate.get("prepared_path"))
    scene = candidate.get("scene")
    source_sha256 = candidate.get("source_sha256", candidate.get("parent_source_sha256"))
    start_sample = candidate.get("start_sample")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("V3 noise candidate has no path")
    if not isinstance(scene, str) or not scene.strip():
        raise ValueError("V3 noise candidate has no scene")
    if not isinstance(source_sha256, str) or not source_sha256.strip():
        raise ValueError("V3 noise candidate has no source_sha256")
    if isinstance(start_sample, bool) or not isinstance(start_sample, int) or start_sample < 0:
        raise ValueError("V3 noise candidate has invalid start_sample")
    if not role:
        raise ValueError("V3 noise request role is required")
    return {
        "example_id": f"v3-{role}-{source_sha256[:16]}-{start_sample}-{seed}",
        "role": role,
        "recipe": {"noise_scene": scene},
        "noise_path": path,
        "noise_start_sample": start_sample,
        "noise_source_sha256": source_sha256,
        "noise_source_scene": scene,
        "declared_noise_scene": scene,
        "resolved_noise_scene": scene,
    }


def _request_from_recipe(
    *,
    example_id: str,
    recipe: Recipe,
    foreground: dict[str, object] | None,
    noise_by_scene: dict[str, list[dict[str, object]]],
    rir_rows: list[dict[str, object]],
    interferer_rows: list[dict[str, object]],
) -> dict[str, object]:
    noise = _choose_noise_by_scene(noise_by_scene, recipe.seed) if recipe.noise_scene is not None or recipe.source_kind == "pure_noise" else None
    rir = _choose_row(rir_rows, recipe.seed) if recipe.apply_rir else None
    interferer = _choose_row(interferer_rows, recipe.seed) if recipe.apply_interferer else None
    active_span = None
    if foreground is not None and recipe.label == "positive":
        active_span = [int(foreground["active_start"]), int(foreground["active_end"])]
    return {
        "example_id": example_id,
        "recipe": recipe.as_dict(),
        "label": recipe.label,
        "foreground_path": None if foreground is None else str(foreground["prepared_path"]),
        "active_span": active_span,
        "noise_path": None if noise is None else str(noise["prepared_path"]),
        "rir_path": None if rir is None else str(rir["prepared_path"]),
        "interferer_path": None if interferer is None else str(interferer["prepared_path"]),
    }


def build_generation_requests(
    prepared_rows: list[dict[str, object]],
    global_seed: int,
    variants_per_positive: int = 48,
) -> Iterator[dict[str, object]]:
    """Compile prepared source provenance into the immutable training-generation manifest."""
    train_rows = [row for row in prepared_rows if row.get("source_split") == "train"]
    positives = [row for row in train_rows if row.get("source_kind") == "speech" and row.get("source_label") == "positive"]
    speech_negatives = [row for row in train_rows if row.get("source_kind") == "speech" and row.get("source_label") == "negative"]
    false_wakes = [row for row in train_rows if row.get("source_kind") == "false_wake"]
    noise_rows = [row for row in prepared_rows if row.get("source_kind") == "noise"]
    rir_rows = [row for row in prepared_rows if row.get("source_kind") == "rir"]
    if not positives or not speech_negatives or not false_wakes or not noise_rows or not rir_rows:
        raise ValueError("Prepared sources must include train positive/negative/false-wake, all noise scenes, and at least one RIR")

    quotas = plan_output_quotas(len(positives), variants_per_positive)
    noise_by_scene = _index_noise_by_scene(noise_rows)
    interferer_rows = [*speech_negatives, *false_wakes]
    for positive_index, source in enumerate(positives):
        for slot in range(variants_per_positive):
            recipe = build_recipe(
                global_seed,
                "train",
                str(source["source_sha256"]),
                str(source["prepared_sha256"]),
                slot,
                label="positive",
                source_kind="speech",
            )
            yield _request_from_recipe(
                example_id=_stable_example_id("positive", positive_index * variants_per_positive + slot, recipe),
                recipe=recipe,
                foreground=source,
                noise_by_scene=noise_by_scene,
                rir_rows=rir_rows,
                interferer_rows=interferer_rows,
            )

    for kind, quota, pool, source_kind in (
        ("speech_negative", quotas["speech_negative"], speech_negatives, "speech"),
        ("false_wake_negative", quotas["false_wake_negative"], false_wakes, "false_wake"),
    ):
        for ordinal in range(quota):
            source = _choose_row(pool, derive_recipe_seed(global_seed, "train", kind, kind, ordinal))
            recipe = build_recipe(
                global_seed,
                "train",
                _source_sha256(source),
                str(source["prepared_sha256"]),
                ordinal % variants_per_positive,
                label="negative",
                source_kind=source_kind,
            )
            yield _request_from_recipe(
                example_id=_stable_example_id(kind, ordinal, recipe),
                recipe=recipe,
                foreground=source,
                noise_by_scene=noise_by_scene,
                rir_rows=rir_rows,
                interferer_rows=interferer_rows,
            )

    for ordinal in range(quotas["pure_noise_negative"]):
        recipe = build_recipe(
            global_seed,
            "train",
            "pure-noise",
            "pure-noise",
            ordinal % variants_per_positive,
            label="negative",
            source_kind="pure_noise",
        )
        noise = _choose_noise_by_scene(noise_by_scene, recipe.seed)
        recipe = replace(
            recipe,
            noise_scene=str(noise["scene"]),
            snr_db=None,
            apply_rir=False,
            apply_interferer=False,
            sir_db=None,
        )
        yield _request_from_recipe(
            example_id=_stable_example_id("pure_noise_negative", ordinal, recipe),
            recipe=recipe,
            foreground=None,
            noise_by_scene=noise_by_scene,
            rir_rows=rir_rows,
            interferer_rows=interferer_rows,
        )


def write_generation_requests(requests: Iterable[dict[str, object]], destination: Path) -> int:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for request in requests:
            handle.write(json.dumps(request, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def write_recipe_manifest(recipes: list[Recipe], destination: Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for recipe in recipes:
            handle.write(json.dumps(recipe.as_dict(), ensure_ascii=False, sort_keys=True) + "\n")
