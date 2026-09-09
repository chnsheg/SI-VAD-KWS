"""Build immutable trainer manifests from a completed recleaned corpus."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


DEFAULT_TRAIN_COUNT = 2_095_200
DEFAULT_VALIDATION_COUNT = 7_360
DEFAULT_TEST_COUNT = 21_282
EXPECTED_POSITIVE_JITTER_MAX_MS = 200
DEFAULT_SAMPLE_RATE = 16_000
_METADATA_SHARD_COUNT = 4


@dataclass(frozen=True)
class TrainingManifestPaths:
    """Locations of one immutable training split."""

    train_manifest_path: Path
    validation_manifest_path: Path
    test_manifest_path: Path


@dataclass(frozen=True)
class TrainingManifestBuildResult(TrainingManifestPaths):
    """Locations and verified row counts for one immutable training split."""

    train_count: int
    validation_count: int
    test_count: int


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _validate_expected_count(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _temporary_file(path: Path) -> tuple[int, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    return descriptor, Path(raw_path)


def _remove_if_present(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _lexical_absolute_path(value: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(value))))


def _training_record(
    metadata_path: Path,
    line_number: int,
    row: object,
    *,
    min_positive_coverage_ratio: float | None = None,
    coverage_aware_positive_jitter: bool = False,
) -> dict[str, object]:
    if not isinstance(row, Mapping):
        raise ValueError(f"Metadata row is not an object: {metadata_path}:{line_number}")

    output_path = row.get("output_path")
    if not isinstance(output_path, str) or not output_path.strip():
        raise ValueError(f"Metadata row has no output_path: {metadata_path}:{line_number}")
    if not Path(output_path).is_absolute():
        raise ValueError(f"Metadata row must have an absolute output_path: {metadata_path}:{line_number}")
    label = row.get("label")
    if label not in {"positive", "negative"}:
        raise ValueError(f"Metadata row has invalid label: {metadata_path}:{line_number}")

    if label == "positive":
        recipe = row.get("recipe")
        if not isinstance(recipe, Mapping):
            raise ValueError(f"Positive metadata row has no recipe: {metadata_path}:{line_number}")
        jitter_max_ms = recipe.get("online_window_jitter_max_ms")
        if isinstance(jitter_max_ms, bool) or not isinstance(jitter_max_ms, int) or jitter_max_ms != EXPECTED_POSITIVE_JITTER_MAX_MS:
            raise ValueError(
                "Positive metadata row has invalid online_window_jitter_max_ms "
                f"at {metadata_path}:{line_number}: {jitter_max_ms!r}"
            )
        # Positive windows may be translated by the online jitter policy.  A
        # very large translation can leave only a small fraction of the
        # detected keyword inside the one-second window while retaining a
        # positive label, which teaches the classifier contradictory examples.
        # Keep the historical permissive behavior by default, but provide an
        # explicit build-time gate for production corpora.
        if min_positive_coverage_ratio is not None:
            coverage = row.get("coverage_ratio")
            if isinstance(coverage, bool) or not isinstance(coverage, (int, float)) or not math.isfinite(float(coverage)):
                raise ValueError(
                    "Positive metadata row has no finite coverage_ratio "
                    f"at {metadata_path}:{line_number}"
                )
            if not min_positive_coverage_ratio <= float(coverage) <= 1.0:
                raise ValueError(
                    "Positive metadata row coverage_ratio is below the configured minimum "
                    f"at {metadata_path}:{line_number}: {coverage!r} < {min_positive_coverage_ratio}"
                )
        if coverage_aware_positive_jitter:
            # The packed trainer applies jitter after loading the one-second
            # record.  If the active span touches an edge, blindly retaining
            # ±200 ms can move a substantial amount of the wake outside the
            # window.  Cap the per-record jitter to the largest symmetric
            # offset that preserves the requested visible-span ratio.
            active_span = row.get("active_span")
            if not isinstance(active_span, (list, tuple)) or len(active_span) != 2:
                raise ValueError(
                    "Positive metadata row has no active_span required for coverage-aware jitter "
                    f"at {metadata_path}:{line_number}"
                )
            try:
                active_start, active_end = int(active_span[0]), int(active_span[1])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Positive metadata row has invalid active_span at {metadata_path}:{line_number}"
                ) from error
            if not 0 <= active_start < active_end <= DEFAULT_SAMPLE_RATE:
                raise ValueError(
                    f"Positive metadata row active_span is outside one-second window at {metadata_path}:{line_number}"
                )
            target_ratio = 0.0 if min_positive_coverage_ratio is None else float(min_positive_coverage_ratio)
            if min_positive_coverage_ratio is None:
                target_ratio = 0.8
            max_requested_samples = round(jitter_max_ms * DEFAULT_SAMPLE_RATE / 1000)
            active_len = active_end - active_start

            def overlap(offset: int) -> int:
                # Positive offset shifts the active span left in the output;
                # negative offset shifts it right.
                return max(0, min(active_end - offset, DEFAULT_SAMPLE_RATE) - max(active_start - offset, 0))

            safe_samples = 0
            for offset in range(max_requested_samples, -1, -1):
                if min(overlap(-offset), overlap(offset)) / active_len >= target_ratio:
                    safe_samples = offset
                    break
            jitter_max_ms = min(jitter_max_ms, int(safe_samples * 1000 // DEFAULT_SAMPLE_RATE))
    else:
        jitter_max_ms = 0

    return {
        "audio_filepath": os.path.normpath(output_path),
        "command": label,
        "online_window_jitter_max_ms": jitter_max_ms,
    }


def _stream_training_manifest(
    metadata_paths: tuple[Path, ...],
    destination: Path,
    *,
    min_positive_coverage_ratio: float | None = None,
    coverage_aware_positive_jitter: bool = False,
) -> tuple[Path, int]:
    descriptor, temporary_path = _temporary_file(destination)
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            for metadata_path in metadata_paths:
                if not metadata_path.is_file():
                    raise FileNotFoundError(f"Metadata shard is missing: {metadata_path}")
                with metadata_path.open("r", encoding="utf-8") as input_handle:
                    for line_number, line in enumerate(input_handle, start=1):
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as error:
                            raise ValueError(f"Invalid metadata JSONL record: {metadata_path}:{line_number}") from error
                        output.write(
                            json.dumps(
                                _training_record(
                                    metadata_path,
                                    line_number,
                                    row,
                                    min_positive_coverage_ratio=min_positive_coverage_ratio,
                                    coverage_aware_positive_jitter=coverage_aware_positive_jitter,
                                ),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        count += 1
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        _remove_if_present(temporary_path)
        raise
    return temporary_path, count


def _stage_manifest_copy(source: Path, destination: Path) -> tuple[Path, int]:
    if not source.is_file():
        raise FileNotFoundError(f"Clean source manifest is missing: {source}")
    descriptor, temporary_path = _temporary_file(destination)
    count = 0
    try:
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
            for line in input_handle:
                if line.strip():
                    count += 1
                output_handle.write(line)
            output_handle.flush()
            os.fsync(output_handle.fileno())
    except BaseException:
        _remove_if_present(temporary_path)
        raise
    return temporary_path, count


def _check_count(name: str, actual: int, expected: int) -> None:
    if actual != expected:
        raise ValueError(f"{name} manifest count is {actual}, expected {expected}")


def _write_current_pointer(manifest_root: Path, generation_name: str) -> None:
    pointer_path = manifest_root / "current.json"
    descriptor, temporary_path = _temporary_file(pointer_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump({"generation": generation_name}, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, pointer_path)
    except BaseException:
        _remove_if_present(temporary_path)
        raise


def _current_generation_name(pointer_path: Path) -> str:
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid current manifest pointer: {pointer_path}") from error
    if not isinstance(pointer, Mapping):
        raise ValueError(f"Invalid current manifest pointer: {pointer_path}")
    generation = pointer.get("generation")
    if not isinstance(generation, str):
        raise ValueError(f"Current manifest pointer must name a relative generation directory: {pointer_path}")
    generation_path = Path(generation)
    if (
        not generation.startswith("generation-")
        or generation_path.is_absolute()
        or len(generation_path.parts) != 1
        or "/" in generation
        or "\\" in generation
    ):
        raise ValueError(f"Current manifest pointer must name a relative generation directory: {pointer_path}")
    return generation


def resolve_current_training_manifests(corpus_root: Path) -> TrainingManifestPaths:
    """Return the complete immutable generation selected by current.json."""
    root = _lexical_absolute_path(Path(corpus_root))
    manifest_root = root / "training_manifests"
    generation_root = manifest_root / _current_generation_name(manifest_root / "current.json")
    paths = TrainingManifestPaths(
        train_manifest_path=generation_root / "train_manifest.json",
        validation_manifest_path=generation_root / "validation_manifest.json",
        test_manifest_path=generation_root / "test_manifest.json",
    )
    missing = [path for path in (paths.train_manifest_path, paths.validation_manifest_path, paths.test_manifest_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Current manifest generation is incomplete: {missing[0]}")
    return paths


def build_reclean_training_manifests(
    corpus_root: Path,
    *,
    expected_train_count: int = DEFAULT_TRAIN_COUNT,
    expected_validation_count: int = DEFAULT_VALIDATION_COUNT,
    expected_test_count: int = DEFAULT_TEST_COUNT,
    min_positive_coverage_ratio: float | None = None,
    coverage_aware_positive_jitter: bool = False,
) -> TrainingManifestBuildResult:
    """Create train/validation/test manifests without reading generated audio."""
    _validate_expected_count("expected_train_count", expected_train_count)
    _validate_expected_count("expected_validation_count", expected_validation_count)
    _validate_expected_count("expected_test_count", expected_test_count)
    if min_positive_coverage_ratio is not None:
        if isinstance(min_positive_coverage_ratio, bool) or not isinstance(min_positive_coverage_ratio, (int, float)):
            raise ValueError("min_positive_coverage_ratio must be a finite number in [0, 1]")
        min_positive_coverage_ratio = float(min_positive_coverage_ratio)
        if not math.isfinite(min_positive_coverage_ratio) or not 0.0 <= min_positive_coverage_ratio <= 1.0:
            raise ValueError("min_positive_coverage_ratio must be a finite number in [0, 1]")

    root = _lexical_absolute_path(Path(corpus_root))
    metadata_paths = tuple(root / "metadata" / f"shard-{index:02d}.jsonl" for index in range(_METADATA_SHARD_COUNT))
    source_root = root / "source_manifests"
    manifest_root = root / "training_manifests"
    generation_id = uuid.uuid4().hex
    staging_root = manifest_root / f".staging-{generation_id}"
    generation_root = manifest_root / f"generation-{generation_id}"
    train_manifest_path = generation_root / "train_manifest.json"
    validation_manifest_path = generation_root / "validation_manifest.json"
    test_manifest_path = generation_root / "test_manifest.json"
    staging_train_path = staging_root / "train_manifest.json"
    staging_validation_path = staging_root / "validation_manifest.json"
    staging_test_path = staging_root / "test_manifest.json"
    manifest_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir()

    temporary_train: Path | None = None
    temporary_validation: Path | None = None
    temporary_test: Path | None = None
    try:
        temporary_train, train_count = _stream_training_manifest(
            metadata_paths,
            staging_train_path,
            min_positive_coverage_ratio=min_positive_coverage_ratio,
            coverage_aware_positive_jitter=coverage_aware_positive_jitter,
        )
        _check_count("Training", train_count, expected_train_count)

        temporary_validation, validation_count = _stage_manifest_copy(
            source_root / "validation_manifest.json", staging_validation_path
        )
        _check_count("Validation", validation_count, expected_validation_count)
        temporary_test, test_count = _stage_manifest_copy(source_root / "test_manifest.json", staging_test_path)
        _check_count("Test", test_count, expected_test_count)

        os.replace(temporary_train, staging_train_path)
        temporary_train = None
        os.replace(temporary_validation, staging_validation_path)
        temporary_validation = None
        os.replace(temporary_test, staging_test_path)
        temporary_test = None
        os.rename(staging_root, generation_root)
        _write_current_pointer(manifest_root, generation_root.name)
    finally:
        _remove_if_present(temporary_train)
        _remove_if_present(temporary_validation)
        _remove_if_present(temporary_test)

    return TrainingManifestBuildResult(
        train_manifest_path=train_manifest_path,
        validation_manifest_path=validation_manifest_path,
        test_manifest_path=test_manifest_path,
        train_count=train_count,
        validation_count=validation_count,
        test_count=test_count,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build immutable manifests for a completed recleaned KWS corpus")
    parser.add_argument("--corpus-root", type=_path, required=True, help="Corpus root containing metadata/ and source_manifests/")
    parser.add_argument("--expected-train-count", type=int, default=DEFAULT_TRAIN_COUNT)
    parser.add_argument("--expected-validation-count", type=int, default=DEFAULT_VALIDATION_COUNT)
    parser.add_argument("--expected-test-count", type=int, default=DEFAULT_TEST_COUNT)
    parser.add_argument(
        "--min-positive-coverage-ratio",
        type=float,
        default=None,
        help=(
            "Optional production gate for generated positive windows. "
            "When set, every positive metadata row must include finite coverage_ratio >= this value."
        ),
    )
    parser.add_argument(
        "--coverage-aware-positive-jitter",
        action="store_true",
        help=(
            "Cap each positive record's packed jitter using its active_span so the visible wake overlap "
            "stays above the minimum coverage ratio (default 0.8 when no ratio is supplied)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_reclean_training_manifests(
        args.corpus_root,
        expected_train_count=args.expected_train_count,
        expected_validation_count=args.expected_validation_count,
        expected_test_count=args.expected_test_count,
        min_positive_coverage_ratio=args.min_positive_coverage_ratio,
        coverage_aware_positive_jitter=args.coverage_aware_positive_jitter,
    )
    print(
        json.dumps(
            {
                "train_manifest": str(result.train_manifest_path),
                "validation_manifest": str(result.validation_manifest_path),
                "test_manifest": str(result.test_manifest_path),
                "counts": {
                    "train": result.train_count,
                    "validation": result.validation_count,
                    "test": result.test_count,
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
