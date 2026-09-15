"""Fail-closed V3 train-corpus build gates and compact audit summaries."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import tempfile
import hashlib
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import soundfile as sf

from .recipes import Recipe, build_recipe, derive_v3_seed
from .v3_augmentation_closure import audit_augmentation_closure
from .v3_sources import SourceBlock, WindowCandidate, dense_window_candidates


V3_ROLE_QUOTAS_PER_TWENTY = {
    "base_positive": 7,
    "raw_positive": 3,
    "base_negative": 3,
    "raw_negative": 2,
    "false_wake_hard_negative": 2,
    "captured_environment_negative": 2,
    "tau_environment_negative": 1,
}

_RAW_ANCHOR_ROLES = frozenset({"raw_positive", "raw_negative"})
_ENVIRONMENT_ROLE_DOMAINS = {
    "captured_environment_negative": "captured",
    "tau_environment_negative": "tau",
}


def _load_environment_candidates(
    source_plan_path: Path | str,
    *,
    domain: str,
) -> list[WindowCandidate]:
    """Read train-only complete windows from a published V3 source plan."""
    source_plan_path = Path(source_plan_path).expanduser().resolve()
    try:
        payload = json.loads(source_plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read V3 source plan: {source_plan_path}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("V3 source plan has an unsupported schema")
    raw_plans = payload.get("plans")
    if not isinstance(raw_plans, list):
        raise ValueError("V3 source plan has no plans list")

    candidates: list[WindowCandidate] = []
    for raw_plan in raw_plans:
        if not isinstance(raw_plan, dict):
            raise ValueError("V3 source plan has an invalid plan")
        source = raw_plan.get("source")
        blocks = raw_plan.get("blocks")
        if not isinstance(source, dict) or not isinstance(blocks, list):
            raise ValueError("V3 source plan has an invalid source or block list")
        if source.get("domain") != domain:
            continue
        for raw_block in blocks:
            if not isinstance(raw_block, dict):
                raise ValueError("V3 source plan has an invalid block")
            if raw_block.get("split") != "train":
                continue
            if raw_block.get("domain") != domain:
                raise ValueError("V3 source plan block domain differs from its source")
            if raw_block.get("source_sha256") != source.get("source_sha256"):
                raise ValueError("V3 source plan block source hash differs from its source")
            try:
                block = SourceBlock(
                    source_sha256=str(raw_block["source_sha256"]),
                    path=str(raw_block["path"]),
                    scene=str(raw_block["scene"]),
                    domain=str(raw_block["domain"]),
                    split="train",
                    block_index=int(raw_block["block_index"]),
                    block_start=int(raw_block["block_start"]),
                    block_end=int(raw_block["block_end"]),
                    allowed_start=int(raw_block["allowed_start"]),
                    allowed_end=int(raw_block["allowed_end"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("V3 source plan block has invalid provenance") from error
            if not block.path or not block.scene or not block.source_sha256:
                raise ValueError("V3 source plan block has incomplete provenance")
            candidates.extend(dense_window_candidates([block]))
    if not candidates:
        raise ValueError(f"V3 source plan has no train {domain} environment candidates")
    return candidates


def _select_source_balanced_windows(
    candidates: Sequence[WindowCandidate],
    *,
    target_count: int,
    seed: int,
) -> list[WindowCandidate]:
    """Round-robin source-time windows so long recordings cannot dominate."""
    if target_count <= 0:
        raise ValueError("V3 environment target_count must be positive")
    if target_count > len(candidates):
        raise ValueError(
            f"V3 environment selection has {len(candidates)} candidates, needs {target_count}"
        )
    by_source: dict[str, list[WindowCandidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.split != "train":
            raise ValueError("V3 environment candidate is not train-only")
        by_source[candidate.source_sha256].append(candidate)
    source_order = sorted(by_source)
    random.Random(seed).shuffle(source_order)
    for source_sha256 in source_order:
        by_source[source_sha256].sort(key=lambda candidate: (candidate.start_sample, candidate.block_index))

    offsets = {source_sha256: 0 for source_sha256 in source_order}
    selected: list[WindowCandidate] = []
    while len(selected) < target_count:
        selected_this_round = 0
        for source_sha256 in source_order:
            offset = offsets[source_sha256]
            source_candidates = by_source[source_sha256]
            if offset >= len(source_candidates):
                continue
            selected.append(source_candidates[offset])
            offsets[source_sha256] = offset + 1
            selected_this_round += 1
            if len(selected) == target_count:
                break
        if not selected_this_round:
            break
    if len(selected) != target_count:
        raise RuntimeError("V3 environment source-balanced selection stopped early")
    return selected


def _environment_request(
    candidate: WindowCandidate,
    *,
    role: str,
    global_seed: int,
    source_candidate_count: int,
) -> dict[str, object]:
    seed = derive_v3_seed(
        global_seed,
        candidate.split,
        candidate.source_sha256,
        candidate.start_sample,
        role,
        0,
    )
    recipe = Recipe(
        seed=seed,
        split=candidate.split,
        source_id=candidate.source_sha256,
        source_sha256=candidate.source_sha256,
        slot=candidate.block_index,
        label="negative",
        source_kind="pure_noise",
        augmentation_group="environment",
        speed=1.0,
        jitter_ms=0,
        online_window_jitter_max_ms=0,
        active_rms_dbfs=None,
        noise_scene=candidate.scene,
        snr_db=None,
        apply_rir=False,
        apply_interferer=False,
        sir_db=None,
    )
    return {
        "example_id": (
            f"v3-{role}-{candidate.source_sha256[:16]}-"
            f"{candidate.start_sample}-{seed}"
        ),
        "recipe": asdict(recipe),
        "label": "negative",
        "foreground_path": None,
        "active_span": None,
        "noise_path": candidate.path,
        "noise_start_sample": candidate.start_sample,
        "noise_source_sha256": candidate.source_sha256,
        "noise_source_scene": candidate.scene,
        "noise_scene": candidate.scene,
        "declared_noise_scene": candidate.scene,
        "resolved_noise_scene": candidate.scene,
        "source_role": role,
        "source_candidate_count": source_candidate_count,
    }


def _write_jsonl_atomic(destination: Path, rows: Sequence[Mapping[str, object]]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def write_environment_negative_requests(
    source_plan_path: Path | str,
    destination: Path | str,
    *,
    role: str,
    target_count: int | None = None,
    seed: int,
) -> dict[str, object]:
    """Publish an auditable, source-balanced V3 pure-environment request role.

    The source plan is the only authority for domain, split, scene, source
    identity, and exact crop start.  Requests are source-grouped after
    selection so each renderer rank can reuse a decoded source while the
    selected set remains round-robin balanced.
    """
    domain = _ENVIRONMENT_ROLE_DOMAINS.get(role)
    if domain is None:
        raise ValueError(f"Unsupported V3 environment role: {role}")
    candidates = _load_environment_candidates(source_plan_path, domain=domain)
    source_candidate_counts = Counter(candidate.source_sha256 for candidate in candidates)
    requested_count = len(candidates) if target_count is None else target_count
    selected = _select_source_balanced_windows(
        candidates,
        target_count=requested_count,
        seed=seed,
    )
    rows = [
        _environment_request(
            candidate,
            role=role,
            global_seed=seed,
            source_candidate_count=source_candidate_counts[candidate.source_sha256],
        )
        for candidate in selected
    ]
    rows.sort(
        key=lambda row: (
            str(row["noise_path"]),
            int(row["noise_start_sample"]),
            str(row["example_id"]),
        )
    )
    output = Path(destination).expanduser().resolve()
    _write_jsonl_atomic(output, rows)
    return {
        "role": role,
        "domain": domain,
        "source_plan": str(Path(source_plan_path).expanduser().resolve()),
        "destination": str(output),
        "candidate_count": len(candidates),
        "selected_count": len(rows),
        "candidate_source_count": len(source_candidate_counts),
        "selected_source_count": len({candidate.source_sha256 for candidate in selected}),
    }


def _train_speech_row(row: Mapping[str, object], *, label: str) -> None:
    if row.get("source_kind") != "speech" or row.get("source_label") != label:
        raise ValueError(f"V3 counterpart source must be a {label} speech row")
    if row.get("source_split") != "train":
        raise ValueError("V3 counterpart source must be train-only")
    _text(row, "source_sha256")
    _text(row, "prepared_path")


def _active_span_rms_dbfs(row: Mapping[str, object]) -> float:
    """Measure a prepared speech source only over its declared active span."""
    _train_speech_row(row, label="positive")
    sample_rate = _integer(row, "sample_rate")
    active_start = _integer(row, "active_start")
    active_end = _integer(row, "active_end")
    if sample_rate != 16_000:
        raise ValueError("V3 active-RMS source must be 16 kHz")
    if active_start < 0 or active_end <= active_start:
        raise ValueError("V3 active-RMS source has an invalid active span")

    prepared_path = Path(_text(row, "prepared_path"))
    try:
        with sf.SoundFile(prepared_path, "r") as handle:
            if handle.samplerate != sample_rate:
                raise ValueError("V3 active-RMS source sample rate disagrees with its metadata")
            if active_end > handle.frames:
                raise ValueError("V3 active-RMS active span exceeds prepared audio")
            handle.seek(active_start)
            samples = handle.read(active_end - active_start, dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Unable to read V3 active-RMS source: {prepared_path}") from error

    if samples.shape != (active_end - active_start, samples.shape[1]) or samples.size == 0:
        raise ValueError("V3 active-RMS source did not return its complete active span")
    samples64 = samples.astype("float64", copy=False)
    rms = math.sqrt(float((samples64 * samples64).mean()))
    if not math.isfinite(rms):
        raise ValueError("V3 active-RMS source has a non-finite waveform")
    return -math.inf if rms == 0.0 else 20.0 * math.log10(rms)


def write_train_positive_active_rms_eligible_manifest(
    prepared_manifest: Path | str,
    destination: Path | str,
    *,
    minimum_rms_dbfs: float = -50.0,
) -> dict[str, object]:
    """Atomically publish only train positives whose active speech is audible.

    The source manifest remains unchanged.  Each retained row carries the
    measured active-span RMS and its immutable gate threshold so every later
    positive-role request can prove it excluded silent source material.
    """
    if not isinstance(minimum_rms_dbfs, (int, float)) or not math.isfinite(minimum_rms_dbfs):
        raise ValueError("minimum_rms_dbfs must be finite")
    source = Path(prepared_manifest).expanduser().resolve()
    output = Path(destination).expanduser().resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite V3 active-RMS manifest: {output}")

    eligible_rows: list[dict[str, object]] = []
    rms_values: list[float] = []
    rejected_source_hashes: list[str] = []
    input_count = 0
    for row in _iter_jsonl_records(source):
        if not (
            row.get("source_kind") == "speech"
            and row.get("source_label") == "positive"
            and row.get("source_split") == "train"
        ):
            continue
        input_count += 1
        rms_dbfs = _active_span_rms_dbfs(row)
        if rms_dbfs < float(minimum_rms_dbfs):
            rejected_source_hashes.append(_text(row, "source_sha256"))
            continue
        eligible = dict(row)
        eligible["active_rms_dbfs"] = rms_dbfs
        eligible["active_rms_minimum_dbfs"] = float(minimum_rms_dbfs)
        eligible["active_rms_gate"] = "eligible"
        eligible_rows.append(eligible)
        rms_values.append(rms_dbfs)

    if not input_count:
        raise ValueError("V3 active-RMS gate found no train positive sources")
    if not eligible_rows:
        raise ValueError("V3 active-RMS gate rejected every train positive source")
    _write_jsonl_atomic(output, eligible_rows)
    return {
        "prepared_manifest": str(source),
        "destination": str(output),
        "minimum_rms_dbfs": float(minimum_rms_dbfs),
        "input_count": input_count,
        "eligible_count": len(eligible_rows),
        "rejected_count": len(rejected_source_hashes),
        "minimum_eligible_rms_dbfs": min(rms_values),
        "rejected_source_sha256": rejected_source_hashes,
    }


def _raw_positive_anchor_request(
    row: Mapping[str, object],
    *,
    seed: int,
    ordinal: int,
) -> dict[str, object]:
    _train_speech_row(row, label="positive")
    if row.get("active_rms_gate") != "eligible":
        raise ValueError("V3 raw-positive source did not pass the active-RMS gate")
    rms_dbfs = row.get("active_rms_dbfs")
    threshold_dbfs = row.get("active_rms_minimum_dbfs")
    if (
        isinstance(rms_dbfs, bool)
        or not isinstance(rms_dbfs, (int, float))
        or not math.isfinite(rms_dbfs)
        or isinstance(threshold_dbfs, bool)
        or not isinstance(threshold_dbfs, (int, float))
        or not math.isfinite(threshold_dbfs)
        or float(rms_dbfs) < float(threshold_dbfs)
    ):
        raise ValueError("V3 raw-positive source has invalid active-RMS gate provenance")
    active_start = _integer(row, "active_start")
    active_end = _integer(row, "active_end")
    if active_start < 0 or active_end <= active_start:
        raise ValueError("V3 raw-positive source has an invalid active span")
    source_sha256 = _text(row, "source_sha256")
    request_seed = derive_v3_seed(seed, "train", source_sha256, 0, "raw_positive", ordinal)
    recipe = Recipe(
        seed=request_seed,
        split="train",
        source_id=source_sha256,
        source_sha256=source_sha256,
        slot=ordinal,
        label="positive",
        source_kind="speech",
        augmentation_group="raw_anchor",
        speed=1.0,
        jitter_ms=0,
        online_window_jitter_max_ms=200,
        active_rms_dbfs=float(rms_dbfs),
        noise_scene=None,
        snr_db=None,
        apply_rir=False,
        apply_interferer=False,
        sir_db=None,
    )
    return {
        "example_id": f"v3-raw-positive-{source_sha256[:16]}-{request_seed}",
        "recipe": asdict(recipe),
        "label": "positive",
        "foreground_path": _text(row, "prepared_path"),
        "active_span": [active_start, active_end],
        "noise_path": None,
        "noise_start_sample": None,
        "noise_source_sha256": None,
        "noise_source_scene": None,
        "noise_scene": None,
        "declared_noise_scene": None,
        "resolved_noise_scene": None,
        "rir_path": None,
        "rir_source_sha256": None,
        "interferer_path": None,
        "interferer_source_sha256": None,
        "foreground_start_sample": None,
        "source_role": "raw_positive",
        "augmentation_closure_required": False,
    }


def write_raw_positive_anchor_requests(
    positives: Sequence[Mapping[str, object]],
    destination: Path | str,
    *,
    seed: int,
) -> dict[str, object]:
    """Publish V3 raw-positive anchors from the active-RMS eligible manifest."""
    output = Path(destination).expanduser().resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite V3 raw-positive requests: {output}")
    if not positives:
        raise ValueError("V3 raw-positive request writer has no sources")
    requests = [_raw_positive_anchor_request(row, seed=seed, ordinal=index) for index, row in enumerate(positives)]
    source_hashes = [_text(row, "source_sha256") for row in positives]
    if len(set(source_hashes)) != len(source_hashes):
        raise ValueError("V3 raw-positive request writer has duplicate source hashes")
    _write_jsonl_atomic(output, requests)
    return {
        "destination": str(output),
        "request_count": len(requests),
        "source_count": len(source_hashes),
        "active_rms_gate": "eligible",
    }


def _load_train_false_wake_candidates(source_plan_path: Path | str) -> list[WindowCandidate]:
    source_plan_path = Path(source_plan_path).expanduser().resolve()
    try:
        payload = json.loads(source_plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read V3 source plan: {source_plan_path}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("V3 source plan has an unsupported schema")
    raw_plans = payload.get("plans")
    if not isinstance(raw_plans, list):
        raise ValueError("V3 source plan has no plans list")

    candidates: list[WindowCandidate] = []
    for raw_plan in raw_plans:
        if not isinstance(raw_plan, dict):
            raise ValueError("V3 source plan has an invalid plan")
        source = raw_plan.get("source")
        blocks = raw_plan.get("blocks")
        if not isinstance(source, dict) or not isinstance(blocks, list):
            raise ValueError("V3 source plan has an invalid source or block list")
        if source.get("domain") != "false_wake":
            continue
        for raw_block in blocks:
            if not isinstance(raw_block, dict):
                raise ValueError("V3 source plan has an invalid block")
            if raw_block.get("split") != "train":
                continue
            if raw_block.get("domain") != "false_wake":
                raise ValueError("V3 false-wake block domain differs from its source")
            if raw_block.get("source_sha256") != source.get("source_sha256"):
                raise ValueError("V3 false-wake block source hash differs from its source")
            try:
                block = SourceBlock(
                    source_sha256=str(raw_block["source_sha256"]),
                    path=str(raw_block["path"]),
                    scene=str(raw_block["scene"]),
                    domain="false_wake",
                    split="train",
                    block_index=int(raw_block["block_index"]),
                    block_start=int(raw_block["block_start"]),
                    block_end=int(raw_block["block_end"]),
                    allowed_start=int(raw_block["allowed_start"]),
                    allowed_end=int(raw_block["allowed_end"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("V3 false-wake block has invalid provenance") from error
            if not block.path or not block.scene or not block.source_sha256:
                raise ValueError("V3 false-wake block has incomplete provenance")
            candidates.extend(dense_window_candidates([block]))
    if not candidates:
        raise ValueError("V3 source plan has no train false-wake candidates")
    return candidates


def _false_wake_hard_negative_request(
    candidate: WindowCandidate,
    *,
    seed: int,
    source_candidate_count: int,
) -> dict[str, object]:
    request_seed = derive_v3_seed(
        seed,
        "train",
        candidate.source_sha256,
        candidate.start_sample,
        "false_wake_hard_negative",
        0,
    )
    recipe = Recipe(
        seed=request_seed,
        split="train",
        source_id=candidate.source_sha256,
        source_sha256=candidate.source_sha256,
        slot=candidate.block_index,
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
    return {
        "example_id": f"v3-false-wake-{candidate.source_sha256[:16]}-{candidate.start_sample}-{request_seed}",
        "recipe": asdict(recipe),
        "label": "negative",
        "foreground_path": candidate.path,
        "active_span": None,
        "noise_path": None,
        "noise_start_sample": None,
        "noise_source_sha256": None,
        "noise_source_scene": None,
        "noise_scene": None,
        "declared_noise_scene": None,
        "resolved_noise_scene": None,
        "rir_path": None,
        "rir_source_sha256": None,
        "interferer_path": None,
        "interferer_source_sha256": None,
        "foreground_start_sample": candidate.start_sample,
        "source_sha256": candidate.source_sha256,
        "source_role": "false_wake_hard_negative",
        "source_candidate_count": source_candidate_count,
        "counterpart_kind": "false_wake_hard_negative",
        "augmentation_closure_required": True,
    }


def write_false_wake_hard_negative_requests(
    source_plan_path: Path | str,
    destination: Path | str,
    *,
    target_count: int | None = None,
    seed: int,
) -> dict[str, object]:
    """Publish source-balanced, exact-hop train false-wake hard negatives."""
    output = Path(destination).expanduser().resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite V3 false-wake requests: {output}")
    candidates = _load_train_false_wake_candidates(source_plan_path)
    requested_count = len(candidates) if target_count is None else target_count
    selected = _select_source_balanced_windows(candidates, target_count=requested_count, seed=seed)
    candidate_counts = Counter(candidate.source_sha256 for candidate in candidates)
    requests = [
        _false_wake_hard_negative_request(
            candidate,
            seed=seed,
            source_candidate_count=candidate_counts[candidate.source_sha256],
        )
        for candidate in selected
    ]
    _write_jsonl_atomic(output, requests)
    return {
        "source_plan": str(Path(source_plan_path).expanduser().resolve()),
        "destination": str(output),
        "candidate_count": len(candidates),
        "request_count": len(requests),
        "source_count": len(candidate_counts),
        "selected_source_count": len({candidate.source_sha256 for candidate in selected}),
    }


def _ranked_raw_hard_negative_request(
    row: Mapping[str, object],
    *,
    seed: int,
    ordinal: int,
) -> dict[str, object]:
    if row.get("source_role") != "raw_negative":
        raise ValueError("V3 ranked hard-negative row is not a raw negative")
    source_sha256 = _text(row, "source_sha256")
    audio_path = _text(row, "audio_path")
    start_sample = _integer(row, "start_sample")
    score = row.get("positive_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError("V3 ranked hard-negative row has a non-finite score")
    if start_sample < 0:
        raise ValueError("V3 ranked hard-negative row has a negative start")
    request_seed = derive_v3_seed(
        seed,
        "train",
        source_sha256,
        start_sample,
        "raw_score_hard_negative",
        ordinal,
    )
    recipe = Recipe(
        seed=request_seed,
        split="train",
        source_id=source_sha256,
        source_sha256=source_sha256,
        slot=ordinal,
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
    return {
        "example_id": f"v3-raw-hard-{source_sha256[:16]}-{start_sample}-{request_seed}",
        "recipe": asdict(recipe),
        "label": "negative",
        "foreground_path": audio_path,
        "active_span": None,
        "noise_path": None,
        "noise_start_sample": None,
        "noise_source_sha256": None,
        "noise_source_scene": None,
        "noise_scene": None,
        "declared_noise_scene": None,
        "resolved_noise_scene": None,
        "rir_path": None,
        "rir_source_sha256": None,
        "interferer_path": None,
        "interferer_source_sha256": None,
        "foreground_start_sample": start_sample,
        "source_sha256": source_sha256,
        "source_role": "raw_score_hard_negative",
        "source_candidate_count": 1,
        "positive_score": float(score),
        "selection_reason": "v2_best_ranked_raw_negative",
        "counterpart_kind": "raw_speech_negative",
        "augmentation_closure_required": True,
    }


def write_ranked_hard_negative_requests(
    false_wake_requests: Sequence[Mapping[str, object]],
    raw_scores: Sequence[Mapping[str, object]],
    destination: Path | str,
    *,
    raw_target_count: int,
    seed: int,
) -> dict[str, object]:
    """Publish exact hard-negative requests sized for large-batch DDP training."""
    output = Path(destination).expanduser().resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite V3 ranked hard-negative requests: {output}")
    if raw_target_count <= 0:
        raise ValueError("V3 ranked hard-negative raw_target_count must be positive")
    if not false_wake_requests:
        raise ValueError("V3 ranked hard-negative writer has no false-wake requests")
    for request in false_wake_requests:
        if request.get("label") != "negative" or request.get("source_role") != "false_wake_hard_negative":
            raise ValueError("V3 ranked hard-negative writer received an invalid false-wake request")
    false_ids = [_text(request, "example_id") for request in false_wake_requests]
    if len(set(false_ids)) != len(false_ids):
        raise ValueError("V3 ranked hard-negative writer has duplicate false-wake request IDs")

    ranked_raw = [dict(row) for row in raw_scores]
    for row in ranked_raw:
        _ranked_raw_hard_negative_request(row, seed=seed, ordinal=0)
    source_hashes = [_text(row, "source_sha256") for row in ranked_raw]
    if len(set(source_hashes)) != len(source_hashes):
        raise ValueError("V3 ranked hard-negative writer has duplicate raw source hashes")
    ranked_raw.sort(
        key=lambda row: (
            -float(row["positive_score"]),
            str(row["source_sha256"]),
            _integer(row, "start_sample"),
        )
    )
    if raw_target_count > len(ranked_raw):
        raise ValueError(
            f"V3 ranked hard-negative writer has {len(ranked_raw)} raw candidates, needs {raw_target_count}"
        )
    selected_raw = [
        _ranked_raw_hard_negative_request(raw, seed=seed, ordinal=ordinal)
        for ordinal, raw in enumerate(ranked_raw[:raw_target_count])
    ]
    requests = [dict(request) for request in false_wake_requests] + selected_raw
    _write_jsonl_atomic(output, requests)
    return {
        "destination": str(output),
        "false_wake_count": len(false_wake_requests),
        "raw_candidate_count": len(ranked_raw),
        "raw_score_count": len(selected_raw),
        "request_count": len(requests),
        "minimum_selected_raw_score": min(float(row["positive_score"]) for row in selected_raw),
        "maximum_selected_raw_score": max(float(row["positive_score"]) for row in selected_raw),
    }


def _counterpart_source(row: Mapping[str, object]) -> tuple[str, str]:
    return (_text(row, "prepared_path"), _text(row, "source_sha256"))


def _noise_binding(row: Mapping[str, object]) -> tuple[str, int, str, str]:
    path = _text(row, "noise_path")
    source_sha256 = _text(row, "noise_source_sha256")
    scene = _text(row, "noise_scene")
    if row.get("resolved_noise_scene") != scene:
        raise ValueError("V3 counterpart noise has declared/resolved scene disagreement")
    start_sample = _integer(row, "noise_start_sample")
    if start_sample < 0:
        raise ValueError("V3 counterpart noise has a negative crop start")
    return (path, start_sample, source_sha256, scene)


def _source_balanced_cycle(
    candidates: Sequence[Mapping[str, object]],
    *,
    seed: int,
) -> list[Mapping[str, object]]:
    by_source: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for candidate in candidates:
        _, start_sample, source_sha256, _ = _noise_binding(candidate)
        by_source[source_sha256].append(candidate)
    if not by_source:
        raise ValueError("V3 source-balanced noise cycle has no candidates")
    source_order = sorted(by_source)
    random.Random(seed).shuffle(source_order)
    for source_sha256 in source_order:
        by_source[source_sha256].sort(
            key=lambda candidate: (_noise_binding(candidate)[1], str(candidate["noise_path"]))
        )

    offsets = {source_sha256: 0 for source_sha256 in source_order}
    ordered: list[Mapping[str, object]] = []
    while True:
        added = False
        for source_sha256 in source_order:
            offset = offsets[source_sha256]
            source_candidates = by_source[source_sha256]
            if offset >= len(source_candidates):
                continue
            ordered.append(source_candidates[offset])
            offsets[source_sha256] = offset + 1
            added = True
        if not added:
            return ordered


def plan_source_balanced_noise_bindings(
    noise_candidates: Sequence[Mapping[str, object]],
    *,
    count: int,
    seed: int,
) -> list[Mapping[str, object]]:
    """Plan scene-balanced, source-balanced noise bindings before reuse.

    A scene is selected in round-robin order.  Within that scene, each source
    contributes one complete source-time crop before any source contributes a
    second crop.  All source-time crops are then exhausted before reuse.  This
    makes source and waveform coverage deterministic and auditable.
    """
    if count <= 0:
        raise ValueError("V3 source-balanced noise plan count must be positive")
    by_scene: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for candidate in noise_candidates:
        _, _, _, scene = _noise_binding(candidate)
        by_scene[scene].append(candidate)
    if not by_scene:
        raise ValueError("V3 source-balanced noise plan has no candidates")

    scene_order = sorted(by_scene)
    rotation = seed % len(scene_order)
    scene_order = scene_order[rotation:] + scene_order[:rotation]
    cycles = {
        scene: _source_balanced_cycle(rows, seed=derive_v3_seed(seed, "train", scene, 0, "noise-cycle", 0))
        for scene, rows in by_scene.items()
    }
    scene_offsets = {scene: 0 for scene in scene_order}
    planned: list[Mapping[str, object]] = []
    for index in range(count):
        scene = scene_order[index % len(scene_order)]
        cycle = cycles[scene]
        planned.append(cycle[scene_offsets[scene] % len(cycle)])
        scene_offsets[scene] += 1
    return planned


def audit_source_balanced_noise_bindings(
    noise_candidates: Sequence[Mapping[str, object]],
    planned: Sequence[Mapping[str, object]],
    *,
    seed: int,
) -> dict[str, object]:
    """Fail closed unless an emitted noise schedule retains all planned diversity."""
    if not planned:
        raise ValueError("V3 source-balanced noise audit has no planned bindings")
    expected = plan_source_balanced_noise_bindings(
        noise_candidates,
        count=len(planned),
        seed=seed,
    )
    expected_bindings = [_noise_binding(candidate) for candidate in expected]
    planned_bindings = [_noise_binding(candidate) for candidate in planned]
    if planned_bindings != expected_bindings:
        raise ValueError("V3 noise schedule does not match the source-balanced plan")

    eligible_by_scene: dict[str, set[str]] = defaultdict(set)
    selected_by_scene: dict[str, set[str]] = defaultdict(set)
    eligible_windows_by_scene: dict[str, set[tuple[str, int]]] = defaultdict(set)
    selected_windows_by_scene: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for candidate in noise_candidates:
        _, start_sample, source_sha256, scene = _noise_binding(candidate)
        eligible_by_scene[scene].add(source_sha256)
        eligible_windows_by_scene[scene].add((source_sha256, start_sample))
    for path, start_sample, source_sha256, scene in planned_bindings:
        del path
        selected_by_scene[scene].add(source_sha256)
        selected_windows_by_scene[scene].add((source_sha256, start_sample))

    for scene, eligible_sources in eligible_by_scene.items():
        planned_count = sum(binding[3] == scene for binding in planned_bindings)
        if planned_count >= len(eligible_sources) and selected_by_scene[scene] != eligible_sources:
            raise ValueError(f"V3 noise schedule does not cover every source in scene: {scene}")
        eligible_windows = eligible_windows_by_scene[scene]
        if planned_count >= len(eligible_windows) and selected_windows_by_scene[scene] != eligible_windows:
            raise ValueError(f"V3 noise schedule does not cover every window in scene: {scene}")
    return {
        "planned_count": len(planned_bindings),
        "eligible_scene_count": len(eligible_by_scene),
        "selected_scene_count": len(selected_by_scene),
        "eligible_source_count": sum(len(rows) for rows in eligible_by_scene.values()),
        "selected_source_count": sum(len(rows) for rows in selected_by_scene.values()),
        "eligible_window_count": sum(len(rows) for rows in eligible_windows_by_scene.values()),
        "selected_window_count": sum(len(rows) for rows in selected_windows_by_scene.values()),
    }


def _auxiliary_source(row: Mapping[str, object], *, kind: str) -> tuple[str, str]:
    path = _text(row, "prepared_path")
    source_sha256 = _text(row, "source_sha256")
    return (path, source_sha256)


def _counterfactual_request(
    *,
    counterpart_id: str,
    recipe: Recipe,
    foreground: Mapping[str, object],
    positive: bool,
    noise: tuple[str, int, str, str] | None,
    rir: tuple[str, str] | None,
    interferer: tuple[str, str] | None,
) -> dict[str, object]:
    foreground_path, _ = _counterpart_source(foreground)
    active_span: list[int] | None = None
    if positive:
        start = _integer(foreground, "active_start")
        end = _integer(foreground, "active_end")
        if start < 0 or end <= start:
            raise ValueError("V3 positive counterpart source has an invalid active span")
        active_span = [start, end]
    noise_path, noise_start, noise_sha256, noise_scene = (None, None, None, None) if noise is None else noise
    rir_path, rir_sha256 = (None, None) if rir is None else rir
    interferer_path, interferer_sha256 = (None, None) if interferer is None else interferer
    return {
        "example_id": f"{counterpart_id}-{'positive' if positive else 'negative'}",
        "counterpart_id": counterpart_id,
        "recipe": asdict(recipe),
        "label": recipe.label,
        "foreground_path": foreground_path,
        "active_span": active_span,
        "noise_path": noise_path,
        "noise_start_sample": noise_start,
        "noise_source_sha256": noise_sha256,
        "noise_source_scene": noise_scene,
        "noise_scene": noise_scene,
        "declared_noise_scene": noise_scene,
        "resolved_noise_scene": noise_scene,
        "rir_path": rir_path,
        "rir_source_sha256": rir_sha256,
        "interferer_path": interferer_path,
        "interferer_source_sha256": interferer_sha256,
        "foreground_start_sample": 0,
        "source_role": "base_positive" if positive else "base_negative",
        "counterpart_kind": "paired_positive" if positive else (
            "counterfactual_augmented_speech_negative"
            if noise is not None or rir is not None or interferer is not None
            else "clean_speech_negative"
        ),
    }


def iter_counterfactual_augmentation_requests(
    *,
    positives: Sequence[Mapping[str, object]],
    nonwake_speech: Sequence[Mapping[str, object]],
    noise_candidates: Sequence[Mapping[str, object]],
    rirs: Sequence[Mapping[str, object]],
    interferers: Sequence[Mapping[str, object]],
    variants_per_positive: int,
    seed: int,
) -> Iterable[dict[str, object]]:
    """Yield positive/non-wake pairs with an identical acoustic signature.

    This is deliberately a generator: the production manifest contains more
    than two million rows and must not be accumulated in coordinator memory.
    """
    if variants_per_positive <= 0:
        raise ValueError("variants_per_positive must be positive")
    for row in positives:
        _train_speech_row(row, label="positive")
    for row in nonwake_speech:
        _train_speech_row(row, label="negative")
    if not positives or not nonwake_speech:
        raise ValueError("V3 counterparts require train positive and non-wake speech")
    for row in noise_candidates:
        _noise_binding(row)
    for row in rirs:
        _auxiliary_source(row, kind="rir")
    for row in interferers:
        _auxiliary_source(row, kind="interferer")

    noisy_variants_per_positive = sum(
        build_recipe(
            seed,
            "train",
            "noise-count",
            "noise-count",
            variant,
            label="positive",
            source_kind="speech",
        ).noise_scene is not None
        for variant in range(variants_per_positive)
    )
    noise_bindings = (
        plan_source_balanced_noise_bindings(
            noise_candidates,
            count=len(positives) * noisy_variants_per_positive,
            seed=seed,
        )
        if noisy_variants_per_positive
        else []
    )
    if noise_bindings:
        audit_source_balanced_noise_bindings(noise_candidates, noise_bindings, seed=seed)
    noise_schedule = iter(noise_bindings)

    for positive_index, positive_source in enumerate(positives):
        positive_path, positive_sha256 = _counterpart_source(positive_source)
        del positive_path
        for variant in range(variants_per_positive):
            pair_seed = derive_v3_seed(
                seed,
                "train",
                positive_sha256,
                0,
                "counterfactual_augmentation",
                variant,
            )
            base_recipe = build_recipe(
                seed,
                "train",
                positive_sha256,
                positive_sha256,
                variant,
                label="positive",
                source_kind="speech",
            )
            base_recipe = replace(base_recipe, seed=pair_seed)
            needs_noise = base_recipe.noise_scene is not None
            if needs_noise and not noise_candidates:
                raise ValueError("V3 environmental positive has no train environment candidate")
            if base_recipe.apply_rir and not rirs:
                raise ValueError("V3 RIR positive has no RIR source")
            if base_recipe.apply_interferer and not interferers:
                raise ValueError("V3 interferer positive has no interferer source")
            noise = None if not needs_noise else _noise_binding(next(noise_schedule))
            if noise is not None:
                base_recipe = replace(base_recipe, noise_scene=noise[3])
            rir = None if not base_recipe.apply_rir else _auxiliary_source(rirs[pair_seed % len(rirs)], kind="rir")
            interferer = (
                None
                if not base_recipe.apply_interferer
                else _auxiliary_source(interferers[pair_seed % len(interferers)], kind="interferer")
            )
            negative_source = nonwake_speech[pair_seed % len(nonwake_speech)]
            _, negative_sha256 = _counterpart_source(negative_source)
            counterpart_id = f"v3-counterfactual-{positive_index:05d}-{variant:02d}-{pair_seed}"
            positive_recipe = replace(
                base_recipe,
                source_id=positive_sha256,
                source_sha256=positive_sha256,
                label="positive",
                online_window_jitter_max_ms=200,
            )
            negative_recipe = replace(
                base_recipe,
                source_id=negative_sha256,
                source_sha256=negative_sha256,
                label="negative",
                online_window_jitter_max_ms=0,
            )
            yield _counterfactual_request(
                counterpart_id=counterpart_id,
                recipe=positive_recipe,
                foreground=positive_source,
                positive=True,
                noise=noise,
                rir=rir,
                interferer=interferer,
            )
            yield _counterfactual_request(
                counterpart_id=counterpart_id,
                recipe=negative_recipe,
                foreground=negative_source,
                positive=False,
                noise=noise,
                rir=rir,
                interferer=interferer,
            )


def write_counterfactual_augmentation_requests(
    *,
    destination_root: Path | str,
    positives: Sequence[Mapping[str, object]],
    nonwake_speech: Sequence[Mapping[str, object]],
    noise_candidates: Sequence[Mapping[str, object]],
    rirs: Sequence[Mapping[str, object]],
    interferers: Sequence[Mapping[str, object]],
    variants_per_positive: int,
    seed: int,
) -> dict[str, object]:
    """Atomically publish paired V3 base-role request manifests."""
    destination = Path(destination_root).expanduser().resolve()
    if destination.exists():
        raise ValueError(f"Refusing to overwrite V3 counterfactual requests: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    roles = ("base_positive", "base_negative")
    paths = {role: staging / f"{role}.requests.jsonl" for role in roles}
    counts = {role: 0 for role in roles}
    published = False
    try:
        with ExitStack() as stack:
            handles = {
                role: stack.enter_context(paths[role].open("w", encoding="utf-8", newline="\n"))
                for role in roles
            }
            for request in iter_counterfactual_augmentation_requests(
                positives=positives,
                nonwake_speech=nonwake_speech,
                noise_candidates=noise_candidates,
                rirs=rirs,
                interferers=interferers,
                variants_per_positive=variants_per_positive,
                seed=seed,
            ):
                role = request.get("source_role")
                if role not in handles:
                    raise ValueError(f"V3 counterfactual request has an unsupported role: {role!r}")
                handles[role].write(json.dumps(request, ensure_ascii=False, sort_keys=True))
                handles[role].write("\n")
                counts[role] += 1
            for handle in handles.values():
                handle.flush()
                os.fsync(handle.fileno())
        if not all(counts.values()) or counts["base_positive"] != counts["base_negative"]:
            raise ValueError("V3 counterfactual request roles are incomplete or unpaired")
        os.replace(staging, destination)
        published = True
        return {
            "destination_root": str(destination),
            "role_manifests": {role: str(destination / paths[role].name) for role in roles},
            "counts": counts,
        }
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


def _iter_jsonl_records(path: Path) -> Iterable[dict[str, object]]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL record {path}:{line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"Invalid JSONL object {path}:{line_number}")
        yield row


def _completion_matches_request(
    metadata: Mapping[str, object],
    request: Mapping[str, object],
) -> bool:
    return all(
        metadata.get(field) == request.get(field)
        for field in (
            "example_id",
            "recipe",
            "label",
            "noise_path",
            "noise_start_sample",
            "noise_source_sha256",
            "source_role",
        )
    )


def repair_generation_metadata(
    request_manifest: Path | str,
    metadata_root: Path | str,
    repaired_root: Path | str,
) -> dict[str, object]:
    """Publish non-destructive, duplicate-free completions for a V3 role.

    The source metadata directory is never changed.  A repaired directory is
    emitted only when every expected request appears exactly once after
    deduplication and repeated records are byte-for-byte equivalent JSON.
    """
    request_manifest = Path(request_manifest).expanduser().resolve()
    metadata_root = Path(metadata_root).expanduser().resolve()
    repaired_root = Path(repaired_root).expanduser().resolve()
    if repaired_root.exists():
        raise ValueError(f"Refusing to overwrite repaired V3 metadata: {repaired_root}")
    expected: dict[str, dict[str, object]] = {}
    for request in _iter_jsonl_records(request_manifest):
        example_id = _text(request, "example_id")
        if example_id in expected:
            raise ValueError(f"V3 request manifest has a duplicate example_id: {example_id}")
        expected[example_id] = request
    if not expected:
        raise ValueError("V3 request manifest is empty")

    source_paths = sorted(metadata_root.glob("*.jsonl"))
    if not source_paths:
        raise ValueError("V3 metadata repair has no shard manifests")
    repaired_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f".{repaired_root.name}.", dir=repaired_root.parent))
    published = False
    try:
        seen_hashes: dict[str, str] = {}
        duplicate_count = 0
        for source_path in source_paths:
            repaired_rows: list[dict[str, object]] = []
            for metadata in _iter_jsonl_records(source_path):
                example_id = _text(metadata, "example_id")
                request = expected.get(example_id)
                if request is None:
                    raise ValueError(f"V3 completion has an unexpected example_id: {example_id}")
                if not _completion_matches_request(metadata, request):
                    raise ValueError(f"V3 completion does not match its request: {example_id}")
                canonical = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                existing = seen_hashes.get(example_id)
                if existing is not None:
                    if existing != digest:
                        raise ValueError(f"V3 duplicate completion is not identical: {example_id}")
                    duplicate_count += 1
                    continue
                seen_hashes[example_id] = digest
                repaired_rows.append(metadata)
            _write_jsonl_atomic(staging_root / source_path.name, repaired_rows)

        missing = set(expected).difference(seen_hashes)
        if missing:
            raise ValueError(f"V3 metadata repair is missing {len(missing)} expected requests")
        os.replace(staging_root, repaired_root)
        published = True
        return {
            "request_manifest": str(request_manifest),
            "metadata_root": str(metadata_root),
            "repaired_root": str(repaired_root),
            "expected_count": len(expected),
            "unique_count": len(seen_hashes),
            "duplicate_count": duplicate_count,
        }
    finally:
        if not published:
            shutil.rmtree(staging_root, ignore_errors=True)


def _text(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"V3 provenance row has no valid {field}")
    return value


def _integer(row: Mapping[str, object], field: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"V3 provenance row has no valid {field}")
    return value


def _binary_label(row: Mapping[str, object]) -> int:
    value = row.get("label")
    if value in ("positive", 0):
        return 0
    if value in ("negative", 1):
        return 1
    raise ValueError(f"V3 provenance row has invalid binary label: {value!r}")


def validate_v3_provenance(
    rows: Iterable[Mapping[str, object]],
    final_source_hashes: set[str],
) -> None:
    """Reject non-production windows, held-out sources, and scene mismatches."""
    for row in rows:
        example_id = _text(row, "example_id")
        _binary_label(row)
        if row.get("source_split") != "train":
            raise ValueError(f"V3 train record is not train-only: {example_id}")
        source_sha256 = _text(row, "source_sha256")
        if source_sha256 in final_source_hashes:
            raise ValueError(f"V3 train record uses a final-test source: {example_id}")
        if _integer(row, "sample_rate") != 16000:
            raise ValueError(f"V3 train record is not 16 kHz: {example_id}")
        if _integer(row, "frames") != 16000:
            raise ValueError(f"V3 train record is not a complete one-second window: {example_id}")
        if _integer(row, "source_start_sample") < 0:
            raise ValueError(f"V3 train record has negative source_start_sample: {example_id}")
        _text(row, "output_sha256")
        declared_scene = row.get("noise_scene")
        resolved_scene = row.get("resolved_noise_scene")
        if resolved_scene is not None and declared_scene != resolved_scene:
            raise ValueError(f"V3 train record has declared/resolved scene disagreement: {example_id}")


def audit_environment_role(
    rows: Sequence[Mapping[str, object]],
    *,
    minimum_unique_ratio: float,
    minimum_coverage_ratio: float,
    required_source_hashes: set[str] | None = None,
) -> dict[str, object]:
    """Require unique rendered windows and source-balanced time coverage."""
    if not rows:
        raise ValueError("environment role has no rows")
    if not 0.0 < minimum_unique_ratio <= 1.0:
        raise ValueError("minimum_unique_ratio must be in (0, 1]")
    if not 0.0 < minimum_coverage_ratio <= 1.0:
        raise ValueError("minimum_coverage_ratio must be in (0, 1]")

    waveforms: list[str] = []
    starts_by_source: dict[str, set[int]] = defaultdict(set)
    candidate_counts: dict[str, int] = {}
    for row in rows:
        source_sha256 = _text(row, "source_sha256")
        waveforms.append(_text(row, "output_sha256"))
        starts_by_source[source_sha256].add(_integer(row, "source_start_sample"))
        declared_total = row.get("source_candidate_count")
        if declared_total is not None:
            if isinstance(declared_total, bool) or not isinstance(declared_total, int) or declared_total < 1:
                raise ValueError("environment role has invalid source_candidate_count")
            existing = candidate_counts.setdefault(source_sha256, declared_total)
            if existing != declared_total:
                raise ValueError("environment role has inconsistent source_candidate_count")

    unique_ratio = len(set(waveforms)) / len(waveforms)
    if unique_ratio < minimum_unique_ratio:
        raise ValueError(
            f"environment uniqueness ratio {unique_ratio:.6f} is below {minimum_unique_ratio:.6f}"
        )

    source_coverage = {}
    for source_sha256, starts in starts_by_source.items():
        candidate_count = candidate_counts.get(source_sha256, len(starts))
        source_coverage[source_sha256] = len(starts) / candidate_count
    if any(value < minimum_coverage_ratio for value in source_coverage.values()):
        raise ValueError("environment coverage ratio is below the required minimum")
    required = set(starts_by_source) if required_source_hashes is None else set(required_source_hashes)
    missing_sources = required.difference(starts_by_source)
    if missing_sources:
        raise ValueError("environment coverage has no retained row for a required source")

    return {
        "record_count": len(rows),
        "unique_waveform_count": len(set(waveforms)),
        "unique_waveform_ratio": unique_ratio,
        "source_count": len(starts_by_source),
        "source_time_coverage": dict(sorted(source_coverage.items())),
    }


def audit_v3_train_build(
    rows: Sequence[Mapping[str, object]],
    *,
    final_source_hashes: set[str],
    environment_roles: Mapping[str, Sequence[Mapping[str, object]]],
    minimum_unique_ratio: float = 0.9,
    minimum_coverage_ratio: float = 0.5,
) -> dict[str, object]:
    """Run every train-corpus gate before V3 manifests may be published."""
    validate_v3_provenance(rows, final_source_hashes)
    closure_rows: list[Mapping[str, object]] = []
    closure_exempt_count = 0
    for row in rows:
        required = row.get("augmentation_closure_required", True)
        if not isinstance(required, bool):
            raise ValueError("augmentation_closure_required must be boolean when provided")
        if required:
            closure_rows.append(row)
            continue
        source_role = row.get("source_role")
        if source_role not in _RAW_ANCHOR_ROLES:
            raise ValueError("only unaugmented raw anchors can exempt augmentation closure")
        if (
            row.get("noise_source_sha256") is not None
            or row.get("noise_scene") is not None
            or row.get("noise_rms_band") is not None
            or row.get("rir_applied") is True
            or row.get("rir_source_sha256") is not None
            or row.get("interferer_applied") is True
            or row.get("interferer_source_sha256") is not None
            or (
                row.get("speed") is not None
                and (isinstance(row.get("speed"), bool) or float(row["speed"]) != 1.0)
            )
        ):
            raise ValueError("augmented records cannot exempt augmentation closure")
        closure_exempt_count += 1
    closure = audit_augmentation_closure(closure_rows)
    environment_audits = {
        role: audit_environment_role(
            role_rows,
            minimum_unique_ratio=minimum_unique_ratio,
            minimum_coverage_ratio=minimum_coverage_ratio,
        )
        for role, role_rows in sorted(environment_roles.items())
    }
    labels = Counter(_binary_label(row) for row in rows)
    if not labels[0] or not labels[1]:
        raise ValueError("V3 train build must contain both positive and negative labels")
    return {
        "record_count": len(rows),
        "label_counts": {"positive": labels[0], "negative": labels[1]},
        "augmentation_closure": closure.as_dict(),
        "closure_exempt_record_count": closure_exempt_count,
        "environment_roles": environment_audits,
    }
