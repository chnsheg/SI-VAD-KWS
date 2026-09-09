"""Build strict, auditable adjacent-window manifests from v3 source metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import soundfile as sf


PAIR_FORMAT = "kws_confirmation_pair_v1"


@dataclass(frozen=True)
class PairManifestBuildResult:
    output_manifest: Path
    audit_path: Path
    positive_records: int
    negative_records: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(*values: object) -> str:
    encoded = "\0".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _publish_no_replace(temporary_path: Path, output_path: Path) -> None:
    """Atomically publish a same-filesystem temporary file without overwrite."""

    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    try:
        os.link(temporary_path, output_path)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}") from error


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_no_replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _resolve_path(raw_path: object, *, relative_to: Path) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _strict_int(value: object, *, minimum: int = 0) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return None
    return int(value)


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            yield line_number, row


def _source_id(row: dict[str, Any], source_path: Path) -> str:
    for key in ("source_id", "source_sha256", "parent_source_sha256", "prepared_sha256"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    provenance_path = row.get("source_path") or row.get("parent_source_path") or str(source_path)
    return f"pathsha256:{_stable_hash(provenance_path)}"


def _audio_info(
    path: Path,
    cache: dict[Path, tuple[int, int, int]],
) -> tuple[int, int, int] | None:
    if path in cache:
        return cache[path]
    if not path.is_file():
        return None
    try:
        info = sf.info(path)
    except (OSError, RuntimeError, ValueError):
        return None
    value = (int(info.samplerate), int(info.frames), int(info.channels))
    cache[path] = value
    return value


def _select_source_balanced(
    candidates: list[dict[str, Any]],
    *,
    quota: int | None,
    per_source_limit: int | None,
    seed: int,
    kind: str,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        groups[str(candidate["source_id"])].append(candidate)
    for source_id, rows in groups.items():
        rows.sort(key=lambda row: _stable_hash(seed, kind, source_id, row["span_start_sample"], row["pair_id"]))
        if per_source_limit is not None:
            del rows[per_source_limit:]
    source_order = sorted(groups, key=lambda source_id: _stable_hash(seed, kind, source_id))
    selected: list[dict[str, Any]] = []
    depth = 0
    while source_order and (quota is None or len(selected) < quota):
        added = False
        for source_id in source_order:
            rows = groups[source_id]
            if depth < len(rows):
                selected.append(rows[depth])
                added = True
                if quota is not None and len(selected) >= quota:
                    break
        if not added:
            break
        depth += 1
    return selected


def _positive_candidates(
    manifest_path: Path,
    *,
    source_split: str,
    sample_rate: int,
    window_samples: int,
    hop_samples: int,
    audio_cache: dict[Path, tuple[int, int, int]],
    rejected: Counter,
    defaulted: Counter,
) -> tuple[list[dict[str, Any]], int]:
    candidates: list[dict[str, Any]] = []
    seen = 0
    for _line_number, row in _read_jsonl(manifest_path):
        seen += 1
        if row.get("source_split") != source_split:
            rejected["non_requested_split"] += 1
            continue
        if row.get("source_label") != "positive":
            rejected["not_positive"] += 1
            continue
        declared_rate = _strict_int(row.get("sample_rate"), minimum=1)
        if declared_rate != sample_rate:
            rejected["declared_sample_rate_mismatch"] += 1
            continue
        declared_frames = _strict_int(row.get("frames"), minimum=1)
        active_start = _strict_int(row.get("active_start"))
        active_end = _strict_int(row.get("active_end"), minimum=1)
        if declared_frames is None or active_start is None or active_end is None or active_start >= active_end:
            rejected["invalid_frame_or_active_metadata"] += 1
            continue
        prepared_path = _resolve_path(row.get("prepared_path"), relative_to=manifest_path.parent)
        source_path = _resolve_path(row.get("source_path"), relative_to=manifest_path.parent)
        if prepared_path is None or source_path is None:
            rejected["missing_prepared_or_source_path"] += 1
            continue
        info = _audio_info(prepared_path, audio_cache)
        if info is None:
            rejected["missing_or_unreadable_file"] += 1
            continue
        actual_rate, actual_frames, channels = info
        if actual_rate != sample_rate:
            rejected["actual_sample_rate_mismatch"] += 1
            continue
        if actual_frames != declared_frames:
            rejected["frame_count_mismatch"] += 1
            continue
        if channels < 1:
            rejected["no_audio_channel"] += 1
            continue
        if not 0 <= active_start < active_end <= actual_frames:
            rejected["active_span_out_of_bounds"] += 1
            continue

        lower = max(0, active_end - window_samples)
        upper = min(actual_frames - window_samples - hop_samples, active_start - hop_samples)
        if lower > upper:
            rejected["no_feasible_pair_start"] += 1
            continue
        start = (lower + upper) // 2
        source_id = _source_id(row, source_path)
        role = row.get("role")
        if not isinstance(role, str) or not role.strip():
            role = "mobvoi_positive"
            defaulted["role"] += 1
        pair_id = _stable_hash("positive", source_id, start, window_samples, hop_samples)
        candidates.append(
            {
                "format": PAIR_FORMAT,
                "pair_id": pair_id,
                "audio_filepath": str(prepared_path),
                "source_path": str(source_path),
                "command": "positive",
                "role": role.strip(),
                "domain": str(row.get("domain") or row.get("scene") or "speech"),
                "source_split": source_split,
                "source_id": source_id,
                "sample_rate": sample_rate,
                "span_start_sample": start,
                "span_num_samples": window_samples + hop_samples,
                "window_samples": window_samples,
                "hop_samples": hop_samples,
                "active_start_sample": active_start,
                "active_end_sample": active_end,
            }
        )
    return candidates, seen


def _negative_candidates(
    source_plan_path: Path,
    *,
    source_split: str,
    sample_rate: int,
    window_samples: int,
    hop_samples: int,
    stride_samples: int,
    domain_stride_samples: Mapping[str, int] | None,
    audio_cache: dict[Path, tuple[int, int, int]],
    rejected: Counter,
) -> tuple[list[dict[str, Any]], int]:
    try:
        payload = json.loads(source_plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid source plan JSON: {source_plan_path}") from error
    plans = payload.get("plans") if isinstance(payload, dict) else None
    if not isinstance(plans, list):
        raise ValueError("source_plan.json must contain a plans list")
    candidates: list[dict[str, Any]] = []
    seen_blocks = 0
    seen_pairs: set[tuple[str, int]] = set()
    required_samples = window_samples + hop_samples
    for plan in plans:
        if not isinstance(plan, dict) or not isinstance(plan.get("blocks"), list):
            raise ValueError("each source plan entry must contain a blocks list")
        for block in plan["blocks"]:
            seen_blocks += 1
            if not isinstance(block, dict):
                rejected["invalid_block"] += 1
                continue
            if block.get("split") != source_split:
                rejected["non_requested_split"] += 1
                continue
            domain_value = block.get("domain")
            if not isinstance(domain_value, str) or not domain_value.strip():
                rejected["missing_domain"] += 1
                continue
            domain = domain_value.strip()
            audio_path = _resolve_path(block.get("path"), relative_to=source_plan_path.parent)
            if audio_path is None:
                rejected["missing_path"] += 1
                continue
            info = _audio_info(audio_path, audio_cache)
            if info is None:
                rejected["missing_or_unreadable_file"] += 1
                continue
            actual_rate, actual_frames, channels = info
            declared_rate = block.get("sample_rate", plan.get("sample_rate"))
            if declared_rate is not None and _strict_int(declared_rate, minimum=1) != sample_rate:
                rejected["declared_sample_rate_mismatch"] += 1
                continue
            if actual_rate != sample_rate:
                rejected["actual_sample_rate_mismatch"] += 1
                continue
            if channels < 1:
                rejected["no_audio_channel"] += 1
                continue
            allowed_start = _strict_int(block.get("allowed_start"))
            allowed_end = _strict_int(block.get("allowed_end"), minimum=1)
            if (
                allowed_start is None
                or allowed_end is None
                or not 0 <= allowed_start < allowed_end <= actual_frames
            ):
                rejected["invalid_block_bounds"] += 1
                continue
            if allowed_end - allowed_start < required_samples:
                rejected["block_too_short"] += 1
                continue
            merged = {**plan, **block}
            source_id = _source_id(merged, audio_path)
            role_value = block.get("role", plan.get("role", domain))
            role = role_value.strip() if isinstance(role_value, str) else ""
            if not role:
                rejected["missing_role"] += 1
                continue
            block_stride_samples = (
                domain_stride_samples.get(domain, stride_samples)
                if domain_stride_samples is not None
                else stride_samples
            )
            for start in range(allowed_start, allowed_end - required_samples + 1, block_stride_samples):
                dedup_key = (str(audio_path), start)
                if dedup_key in seen_pairs:
                    rejected["duplicate_pair"] += 1
                    continue
                seen_pairs.add(dedup_key)
                pair_id = _stable_hash("negative", source_id, start, window_samples, hop_samples)
                candidates.append(
                    {
                        "format": PAIR_FORMAT,
                        "pair_id": pair_id,
                        "audio_filepath": str(audio_path),
                        "source_path": str(audio_path),
                        "command": "negative",
                        "role": role,
                        "domain": domain,
                        "scene": block.get("scene"),
                        "source_split": source_split,
                        "source_id": source_id,
                        "sample_rate": sample_rate,
                        "span_start_sample": start,
                        "span_num_samples": required_samples,
                        "window_samples": window_samples,
                        "hop_samples": hop_samples,
                    }
                )
    return candidates, seen_blocks


def _validate_limit(name: str, value: int | None, *, allow_zero: bool) -> None:
    if value is None:
        return
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum} or None")


def _normalize_negative_domain_quota(
    quota: Mapping[object, object] | None,
) -> dict[str, int] | None:
    if quota is None:
        return None
    if not isinstance(quota, Mapping) or not quota:
        raise ValueError("negative_domain_quota must be a non-empty mapping or None")
    normalized: dict[str, int] = {}
    for raw_domain, count in quota.items():
        if not isinstance(raw_domain, str) or not raw_domain.strip():
            raise ValueError("negative_domain_quota domains must be non-empty strings")
        domain = raw_domain.strip()
        if domain in normalized:
            raise ValueError(f"duplicate negative domain quota: {domain!r}")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"negative domain quota for {domain!r} must be an integer >= 0")
        normalized[domain] = int(count)
    return dict(sorted(normalized.items()))


def _parse_negative_domain_quota_specs(specs: list[str]) -> dict[str, int] | None:
    if not specs:
        return None
    parsed: dict[str, int] = {}
    for spec in specs:
        if spec.count("=") != 1:
            raise ValueError("negative domain quota must use DOMAIN=COUNT")
        raw_domain, raw_count = spec.split("=", maxsplit=1)
        domain = raw_domain.strip()
        count_text = raw_count.strip()
        if not domain:
            raise ValueError("negative domain quota DOMAIN must not be empty")
        if domain in parsed:
            raise ValueError(f"duplicate negative domain quota: {domain!r}")
        if not count_text or not count_text.isascii() or not count_text.isdigit():
            raise ValueError(f"negative domain quota for {domain!r} must be an integer >= 0")
        parsed[domain] = int(count_text)
    return _normalize_negative_domain_quota(parsed)


def _normalize_negative_domain_stride(
    stride: Mapping[object, object] | None,
) -> dict[str, int] | None:
    if stride is None:
        return None
    if not isinstance(stride, Mapping) or not stride:
        raise ValueError("negative_domain_stride must be a non-empty mapping or None")
    normalized: dict[str, int] = {}
    for raw_domain, samples in stride.items():
        if not isinstance(raw_domain, str) or not raw_domain.strip():
            raise ValueError("negative_domain_stride domains must be non-empty strings")
        domain = raw_domain.strip()
        if domain in normalized:
            raise ValueError(f"duplicate negative domain stride: {domain!r}")
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
            raise ValueError(f"negative domain stride for {domain!r} must be an integer >= 1")
        normalized[domain] = int(samples)
    return dict(sorted(normalized.items()))


def _parse_negative_domain_stride_specs(specs: list[str]) -> dict[str, int] | None:
    if not specs:
        return None
    parsed: dict[str, int] = {}
    for spec in specs:
        if spec.count("=") != 1:
            raise ValueError("negative domain stride must use DOMAIN=SAMPLES")
        raw_domain, raw_samples = spec.split("=", maxsplit=1)
        domain = raw_domain.strip()
        samples_text = raw_samples.strip()
        if not domain:
            raise ValueError("negative domain stride DOMAIN must not be empty")
        if domain in parsed:
            raise ValueError(f"duplicate negative domain stride: {domain!r}")
        if not samples_text or not samples_text.isascii() or not samples_text.isdigit():
            raise ValueError(f"negative domain stride for {domain!r} must be an integer >= 1")
        parsed[domain] = int(samples_text)
    return _normalize_negative_domain_stride(parsed)


def build_confirmation_pair_manifest(
    positive_manifest: str | os.PathLike[str],
    source_plan: str | os.PathLike[str],
    output_manifest: str | os.PathLike[str],
    audit_path: str | os.PathLike[str],
    *,
    source_split: str = "train",
    sample_rate: int = 16_000,
    window_samples: int = 16_000,
    hop_samples: int = 1_536,
    negative_stride_samples: int | None = None,
    negative_domain_stride: Mapping[str, int] | None = None,
    positive_quota: int | None = None,
    negative_quota: int | None = None,
    negative_domain_quota: Mapping[str, int] | None = None,
    max_positive_per_source: int | None = None,
    max_negative_per_source: int | None = None,
    seed: int = 42,
) -> PairManifestBuildResult:
    """Build and atomically publish a strict pair manifest and its audit."""

    if source_split not in {"train", "validation"}:
        raise ValueError("source_split must be 'train' or 'validation'")
    for name, value in (
        ("sample_rate", sample_rate),
        ("window_samples", window_samples),
        ("hop_samples", hop_samples),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    stride_samples = hop_samples if negative_stride_samples is None else negative_stride_samples
    if isinstance(stride_samples, bool) or not isinstance(stride_samples, int) or stride_samples < 1:
        raise ValueError("negative_stride_samples must be a positive integer")
    normalized_domain_stride = _normalize_negative_domain_stride(negative_domain_stride)
    _validate_limit("positive_quota", positive_quota, allow_zero=True)
    _validate_limit("negative_quota", negative_quota, allow_zero=True)
    _validate_limit("max_positive_per_source", max_positive_per_source, allow_zero=False)
    _validate_limit("max_negative_per_source", max_negative_per_source, allow_zero=False)
    normalized_domain_quota = _normalize_negative_domain_quota(negative_domain_quota)
    if (
        normalized_domain_quota is not None
        and negative_quota is not None
        and negative_quota != sum(normalized_domain_quota.values())
    ):
        raise ValueError(
            f"negative_quota ({negative_quota}) must equal the negative_domain_quota total "
            f"({sum(normalized_domain_quota.values())})"
        )

    positive_path = Path(positive_manifest).expanduser().resolve()
    source_plan_path = Path(source_plan).expanduser().resolve()
    output_path = Path(output_manifest).expanduser().resolve()
    audit_output_path = Path(audit_path).expanduser().resolve()
    if not positive_path.is_file():
        raise FileNotFoundError(positive_path)
    if not source_plan_path.is_file():
        raise FileNotFoundError(source_plan_path)
    if output_path == audit_output_path:
        raise ValueError("output_manifest and audit_path must be different files")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output manifest: {output_path}")
    if audit_output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit JSON: {audit_output_path}")

    cache: dict[Path, tuple[int, int, int]] = {}
    positive_rejected: Counter = Counter()
    positive_defaulted: Counter = Counter()
    negative_rejected: Counter = Counter()
    positive_candidates, positive_seen = _positive_candidates(
        positive_path,
        source_split=source_split,
        sample_rate=sample_rate,
        window_samples=window_samples,
        hop_samples=hop_samples,
        audio_cache=cache,
        rejected=positive_rejected,
        defaulted=positive_defaulted,
    )
    negative_candidates, negative_blocks_seen = _negative_candidates(
        source_plan_path,
        source_split=source_split,
        sample_rate=sample_rate,
        window_samples=window_samples,
        hop_samples=hop_samples,
        stride_samples=stride_samples,
        domain_stride_samples=normalized_domain_stride,
        audio_cache=cache,
        rejected=negative_rejected,
    )
    selected_positive = _select_source_balanced(
        positive_candidates,
        quota=positive_quota,
        per_source_limit=max_positive_per_source,
        seed=seed,
        kind="positive",
    )
    quota_errors = []
    if positive_quota is not None and len(selected_positive) < positive_quota:
        quota_errors.append(f"positive quota requested {positive_quota}, available {len(selected_positive)}")
    if normalized_domain_quota is None:
        selected_negative = _select_source_balanced(
            negative_candidates,
            quota=negative_quota,
            per_source_limit=max_negative_per_source,
            seed=seed,
            kind="negative",
        )
        negative_per_domain = Counter(str(row["domain"]) for row in selected_negative)
        if negative_quota is not None and len(selected_negative) < negative_quota:
            quota_errors.append(f"negative quota requested {negative_quota}, available {len(selected_negative)}")
    else:
        candidates_per_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in negative_candidates:
            candidates_per_domain[str(candidate["domain"])].append(candidate)
        selected_negative = []
        negative_per_domain = Counter()
        for domain, domain_quota in normalized_domain_quota.items():
            domain_selected = _select_source_balanced(
                candidates_per_domain.get(domain, []),
                quota=domain_quota,
                per_source_limit=max_negative_per_source,
                seed=seed,
                kind=f"negative:{domain}",
            )
            selected_negative.extend(domain_selected)
            negative_per_domain[domain] = len(domain_selected)
            if len(domain_selected) < domain_quota:
                quota_errors.append(
                    f"negative domain {domain!r} quota requested {domain_quota}, "
                    f"available {len(domain_selected)}"
                )

    rows = selected_positive + selected_negative
    per_source = Counter(str(row["source_id"]) for row in rows)
    per_role = Counter(str(row["role"]) for row in rows)
    per_domain = Counter(str(row["domain"]) for row in rows)
    audit: dict[str, Any] = {
        "schema_version": 1,
        "status": "failed" if quota_errors else "completed",
        "inputs": {
            "positive_manifest": str(positive_path),
            "positive_manifest_sha256": _sha256_file(positive_path),
            "source_plan": str(source_plan_path),
            "source_plan_sha256": _sha256_file(source_plan_path),
        },
        "config": {
            "source_split": source_split,
            "sample_rate": sample_rate,
            "window_samples": window_samples,
            "hop_samples": hop_samples,
            "negative_stride_samples": stride_samples,
            "negative_domain_stride": normalized_domain_stride,
            "positive_quota": positive_quota,
            "negative_quota": negative_quota,
            "negative_domain_quota": normalized_domain_quota,
            "max_positive_per_source": max_positive_per_source,
            "max_negative_per_source": max_negative_per_source,
            "seed": seed,
            "channel_policy": "channel_0",
        },
        "positive": {
            "rows_seen": positive_seen,
            "candidates": len(positive_candidates),
            "emitted": len(selected_positive),
            "rejected": dict(sorted(positive_rejected.items())),
            "defaulted": dict(sorted(positive_defaulted.items())),
        },
        "negative": {
            "blocks_seen": negative_blocks_seen,
            "candidates": len(negative_candidates),
            "emitted": len(selected_negative),
            "rejected": dict(sorted(negative_rejected.items())),
        },
        "emitted": {
            "total": len(rows),
            "per_source": dict(sorted(per_source.items())),
            "per_role": dict(sorted(per_role.items())),
            "per_domain": dict(sorted(per_domain.items())),
            "negative_per_domain": dict(sorted(negative_per_domain.items())),
        },
        "quota_errors": quota_errors,
    }
    if quota_errors:
        _atomic_write_text(audit_output_path, json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        raise ValueError("; ".join(quota_errors))
    if not rows:
        audit["status"] = "failed"
        audit["quota_errors"] = ["no pair records were eligible"]
        _atomic_write_text(audit_output_path, json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        raise ValueError("no pair records were eligible")

    manifest_text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    _atomic_write_text(output_path, manifest_text)
    audit["output_manifest"] = str(output_path)
    audit["output_manifest_sha256"] = _sha256_file(output_path)
    _atomic_write_text(audit_output_path, json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return PairManifestBuildResult(
        output_manifest=output_path,
        audit_path=audit_output_path,
        positive_records=len(selected_positive),
        negative_records=len(selected_negative),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strict adjacent-window KWS pair manifests")
    parser.add_argument("--positive-manifest", required=True)
    parser.add_argument("--source-plan", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--source-split", choices=("train", "validation"), default="train")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--window-samples", type=int, default=16_000)
    parser.add_argument("--hop-samples", type=int, default=1_536)
    parser.add_argument("--negative-stride-samples", type=int, default=None)
    parser.add_argument(
        "--negative-domain-stride",
        action="append",
        default=[],
        metavar="DOMAIN=SAMPLES",
        help="negative candidate stride for one domain; repeat for multiple domains",
    )
    parser.add_argument("--positive-quota", type=int, default=None)
    parser.add_argument("--negative-quota", type=int, default=None)
    parser.add_argument(
        "--negative-domain-quota",
        action="append",
        default=[],
        metavar="DOMAIN=COUNT",
        help="exact negative quota for one domain; repeat for multiple domains",
    )
    parser.add_argument("--max-positive-per-source", type=int, default=None)
    parser.add_argument("--max-negative-per-source", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        args.negative_domain_quota = _parse_negative_domain_quota_specs(args.negative_domain_quota)
        args.negative_domain_stride = _parse_negative_domain_stride_specs(args.negative_domain_stride)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv: list[str] | None = None) -> PairManifestBuildResult:
    args = parse_args(argv)
    return build_confirmation_pair_manifest(
        args.positive_manifest,
        args.source_plan,
        args.output_manifest,
        args.audit_json,
        source_split=args.source_split,
        sample_rate=args.sample_rate,
        window_samples=args.window_samples,
        hop_samples=args.hop_samples,
        negative_stride_samples=args.negative_stride_samples,
        negative_domain_stride=args.negative_domain_stride,
        positive_quota=args.positive_quota,
        negative_quota=args.negative_quota,
        negative_domain_quota=args.negative_domain_quota,
        max_positive_per_source=args.max_positive_per_source,
        max_negative_per_source=args.max_negative_per_source,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
