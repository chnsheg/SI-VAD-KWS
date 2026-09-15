"""Deterministic source-disjoint planning for V3 streaming evidence."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal, Mapping, Sequence


Split = Literal["train", "validation", "test"]
_SPLITS: tuple[Split, ...] = ("train", "validation", "test")


@dataclass(frozen=True)
class SourceAudio:
    path: str
    scene: str
    domain: str
    source_sha256: str
    frames: int


@dataclass(frozen=True)
class SourceBlock:
    source_sha256: str
    path: str
    scene: str
    domain: str
    split: Split
    block_index: int
    block_start: int
    block_end: int
    allowed_start: int
    allowed_end: int


@dataclass(frozen=True)
class SourcePlan:
    source: SourceAudio
    blocks: tuple[SourceBlock, ...]


@dataclass(frozen=True)
class WindowCandidate:
    source_sha256: str
    path: str
    scene: str
    domain: str
    split: Split
    start_sample: int
    duration_samples: int
    block_index: int


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big")


def _split_counts(total: int) -> tuple[int, int, int]:
    if total < 3:
        raise ValueError("At least three complete blocks are required for a three-way source-time split")
    counts = [1, 1, 1]
    remaining = total - len(counts)
    weights = (0.80, 0.10, 0.10)
    raw = [remaining * weight for weight in weights]
    floors = [int(value) for value in raw]
    for index, value in enumerate(floors):
        counts[index] += value
    for index in sorted(range(3), key=lambda item: (raw[item] - floors[item], -item), reverse=True)[: remaining - sum(floors)]:
        counts[index] += 1
    return (counts[0], counts[1], counts[2])


def _assigned_splits(count: int, *, seed: int) -> Mapping[int, Split]:
    train_count, validation_count, test_count = _split_counts(count)
    indexes = list(range(count))
    random.Random(seed).shuffle(indexes)
    assignment: dict[int, Split] = {}
    boundaries = (train_count, train_count + validation_count, train_count + validation_count + test_count)
    for index in indexes[: boundaries[0]]:
        assignment[index] = "train"
    for index in indexes[boundaries[0] : boundaries[1]]:
        assignment[index] = "validation"
    for index in indexes[boundaries[1] : boundaries[2]]:
        assignment[index] = "test"
    if len(assignment) != count:
        raise RuntimeError("Split assignment did not cover all source blocks")
    return assignment


def _blocks_from_assignment(
    source: SourceAudio,
    assignment: Mapping[int, Split],
    *,
    block_frames: int,
    collar_frames: int,
) -> tuple[SourceBlock, ...]:
    blocks: list[SourceBlock] = []
    block_count = len(assignment)
    for block_index in range(block_count):
        block_start = block_index * block_frames
        block_end = block_start + block_frames
        split = assignment[block_index]
        allowed_start = block_start
        allowed_end = block_end
        if block_index > 0 and assignment[block_index - 1] != split:
            allowed_start += collar_frames
        if block_index + 1 < block_count and assignment[block_index + 1] != split:
            allowed_end -= collar_frames
        if allowed_end <= allowed_start:
            raise ValueError("Split collar removed an entire source-time block")
        blocks.append(
            SourceBlock(
                source_sha256=source.source_sha256,
                path=source.path,
                scene=source.scene,
                domain=source.domain,
                split=split,
                block_index=block_index,
                block_start=block_start,
                block_end=block_end,
                allowed_start=allowed_start,
                allowed_end=allowed_end,
            )
        )
    return tuple(blocks)


def plan_source_time_blocks(
    source: SourceAudio,
    *,
    seed: int,
    block_seconds: int = 10,
    collar_seconds: int = 1,
    sample_rate: int = 16_000,
    final_only: bool = False,
) -> SourcePlan:
    """Split one continuous source while removing both sides of each split boundary."""
    if source.frames < 1:
        raise ValueError("Source audio must contain at least one frame")
    if not source.path or not source.scene or not source.source_sha256:
        raise ValueError("Source audio path, scene, and SHA-256 are required")
    if block_seconds < 1 or collar_seconds < 0 or sample_rate < 1:
        raise ValueError("block_seconds/sample_rate must be positive and collar_seconds non-negative")
    if final_only:
        return SourcePlan(
            source=source,
            blocks=(
                SourceBlock(
                    source_sha256=source.source_sha256,
                    path=source.path,
                    scene=source.scene,
                    domain=source.domain,
                    split="test",
                    block_index=0,
                    block_start=0,
                    block_end=source.frames,
                    allowed_start=0,
                    allowed_end=source.frames,
                ),
            ),
        )

    block_frames = block_seconds * sample_rate
    collar_frames = collar_seconds * sample_rate
    block_count = source.frames // block_frames
    assignment = _assigned_splits(block_count, seed=_stable_seed(seed, source.source_sha256))
    return SourcePlan(
        source=source,
        blocks=_blocks_from_assignment(
            source,
            assignment,
            block_frames=block_frames,
            collar_frames=collar_frames,
        ),
    )


def dense_window_candidates(
    blocks: Iterable[SourceBlock],
    *,
    window_samples: int = 16_000,
    hop_samples: int = 1536,
) -> Iterator[WindowCandidate]:
    """Yield only complete production windows from a source-time allowlist."""
    if window_samples < 1 or hop_samples < 1:
        raise ValueError("window_samples and hop_samples must be positive")
    for block in blocks:
        stop = block.allowed_end - window_samples + 1
        for start_sample in range(block.allowed_start, max(block.allowed_start, stop), hop_samples):
            if start_sample + window_samples <= block.allowed_end:
                yield WindowCandidate(
                    source_sha256=block.source_sha256,
                    path=block.path,
                    scene=block.scene,
                    domain=block.domain,
                    split=block.split,
                    start_sample=start_sample,
                    duration_samples=window_samples,
                    block_index=block.block_index,
                )


def assign_tau_source_splits(sources: Sequence[SourceAudio], *, seed: int) -> dict[str, Split]:
    """Assign every original TAU source file exactly once, stratified by scene."""
    by_scene: dict[str, list[SourceAudio]] = {}
    for source in sources:
        if source.domain != "tau":
            raise ValueError(f"TAU source split received non-TAU domain: {source.path}")
        by_scene.setdefault(source.scene, []).append(source)
    assignments: dict[str, Split] = {}
    for scene, scene_sources in sorted(by_scene.items()):
        hashes = sorted(source.source_sha256 for source in scene_sources)
        if len(set(hashes)) != len(hashes):
            raise ValueError(f"TAU source hashes must be unique within scene {scene}")
        split_by_index = _assigned_splits(len(hashes), seed=_stable_seed(seed, "tau", scene))
        for index, source_sha256 in enumerate(hashes):
            assignments[source_sha256] = split_by_index[index]
    return assignments


def _fixed_split_plan(source: SourceAudio, split: Split) -> SourcePlan:
    return SourcePlan(
        source=source,
        blocks=(
            SourceBlock(
                source_sha256=source.source_sha256,
                path=source.path,
                scene=source.scene,
                domain=source.domain,
                split=split,
                block_index=0,
                block_start=0,
                block_end=source.frames,
                allowed_start=0,
                allowed_end=source.frames,
            ),
        ),
    )


def plan_false_wake_recording(
    source: SourceAudio,
    *,
    seed: int,
    block_seconds: int = 10,
    collar_seconds: int = 1,
    sample_rate: int = 16_000,
) -> SourcePlan:
    """Apply the locked false-wake recording policy without cross-source leakage."""
    stem = source.path.casefold()
    if "_zh_" in stem:
        return _fixed_split_plan(source, "validation")
    if "_cs_" in stem:
        return _fixed_split_plan(source, "test")
    if "_yc_" not in stem and "_yd_" not in stem:
        raise ValueError(f"Unrecognized false-wake recording: {source.path}")
    block_frames = block_seconds * sample_rate
    collar_frames = collar_seconds * sample_rate
    if block_seconds < 1 or collar_seconds < 0 or sample_rate < 1:
        raise ValueError("block_seconds/sample_rate must be positive and collar_seconds non-negative")
    block_count = source.frames // block_frames
    if block_count < 2:
        raise ValueError("yc/yd false-wake sources need at least two complete 10-second blocks")
    indexes = list(range(block_count))
    random.Random(_stable_seed(seed, "false_wake", source.source_sha256)).shuffle(indexes)
    validation_count = max(1, round(block_count * 0.20))
    validation_indexes = set(indexes[:validation_count])
    assignment: dict[int, Split] = {
        index: "validation" if index in validation_indexes else "train"
        for index in range(block_count)
    }
    return SourcePlan(
        source=source,
        blocks=_blocks_from_assignment(
            source,
            assignment,
            block_frames=block_frames,
            collar_frames=collar_frames,
        ),
    )


def _inventory_source(row: Mapping[str, object], *, domain: str) -> SourceAudio:
    path = row.get("prepared_path", row.get("path"))
    scene = row.get("scene")
    source_sha256 = row.get(
        "source_sha256",
        row.get("parent_source_sha256", row.get("sha256")),
    )
    frames = row.get("frames")
    if not isinstance(path, str) or not path:
        raise ValueError("V3 inventory source has no path")
    if not isinstance(scene, str) or not scene:
        raise ValueError(f"V3 inventory source has no scene: {path}")
    if not isinstance(source_sha256, str) or not source_sha256:
        raise ValueError(f"V3 inventory source has no SHA-256: {path}")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        raise ValueError(f"V3 inventory source has invalid frames: {path}")
    return SourceAudio(
        path=path,
        scene=scene,
        domain=domain,
        source_sha256=source_sha256,
        frames=frames,
    )


def _false_wake_inventory_source(row: Mapping[str, object]) -> SourceAudio:
    normalized = dict(row)
    normalized.setdefault("scene", "false_wake")
    return _inventory_source(normalized, domain="false_wake")


def plan_v3_inventory_sources(
    inventory_rows: Iterable[Mapping[str, object]],
    *,
    seed: int,
) -> tuple[SourcePlan, ...]:
    """Build the locked V3 source-time plan directly from verified inventory rows.

    The planner reads source metadata only.  It never copies, decodes, deletes,
    or mutates V2 data, so the serialized output is a lightweight immutable
    precursor to V3 request rendering.
    """
    from .catalog import ALL_NOISE_SCENES, CAPTURED_SCENES, TAU_SCENES

    noise_rows = [row for row in inventory_rows if row.get("role") == "noise"]
    false_wake_rows = [row for row in inventory_rows if row.get("role") == "false_wake"]
    by_scene: dict[str, list[Mapping[str, object]]] = {}
    for row in noise_rows:
        scene = row.get("scene")
        if not isinstance(scene, str) or not scene:
            raise ValueError("V3 noise inventory row has no scene")
        by_scene.setdefault(scene, []).append(row)
    expected_scenes = set(ALL_NOISE_SCENES)
    observed_scenes = set(by_scene)
    missing = sorted(expected_scenes.difference(observed_scenes))
    unexpected = sorted(observed_scenes.difference(expected_scenes))
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing noise scenes: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected noise scenes: {', '.join(unexpected)}")
        raise ValueError("; ".join(details))

    tau_sources = [
        _inventory_source(row, domain="tau")
        for scene in TAU_SCENES
        for row in by_scene[scene]
    ]
    tau_assignments = assign_tau_source_splits(tau_sources, seed=seed)
    plans: list[SourcePlan] = [
        _fixed_split_plan(source, tau_assignments[source.source_sha256])
        for source in tau_sources
    ]
    for scene in CAPTURED_SCENES:
        for row in by_scene[scene]:
            source = _inventory_source(row, domain="captured")
            if source.frames < 16_000:
                # A production window cannot be padded.  The source remains on
                # disk and is reported by the caller's inventory, but has no
                # legal V3 train/development/test window.
                continue
            final_only = scene in {"livingroom", "pub"} or source.frames < 480_000
            plans.append(
                plan_source_time_blocks(
                    source,
                    seed=seed,
                    final_only=final_only,
                )
            )
    segmented_false_wakes = {
        (str(row["parent_source_path"]), str(row["parent_source_sha256"]))
        for row in false_wake_rows
        if row.get("parent_source_path") and row.get("parent_source_sha256")
    }
    if segmented_false_wakes:
        import soundfile as sf

        false_wake_sources = []
        for parent_path, parent_sha256 in sorted(segmented_false_wakes):
            info = sf.info(parent_path)
            if info.frames < 1:
                raise ValueError(f"V3 false-wake source is empty: {parent_path}")
            false_wake_sources.append(
                SourceAudio(
                    path=parent_path,
                    scene="false_wake",
                    domain="false_wake",
                    source_sha256=parent_sha256,
                    frames=int(info.frames),
                )
            )
    else:
        false_wake_sources = [_false_wake_inventory_source(row) for row in false_wake_rows]
    recognized_false_wake = {source.path.casefold() for source in false_wake_sources}
    required_markers = {"_yc_", "_yd_", "_zh_", "_cs_"}
    if not all(any(marker in path for path in recognized_false_wake) for marker in required_markers):
        raise ValueError("V3 inventory must include yc, yd, zh, and cs false-wake recordings")
    plans.extend(plan_false_wake_recording(source, seed=seed) for source in false_wake_sources)
    return tuple(sorted(plans, key=lambda plan: (plan.source.domain, plan.source.path)))


def write_source_plan(destination: Path | str, plans: Iterable[SourcePlan]) -> Path:
    """Atomically publish source-only V3 split provenance outside V2 inputs."""
    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                {
                    "schema_version": 1,
                    "plans": [asdict(plan) for plan in plans],
                },
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return output
