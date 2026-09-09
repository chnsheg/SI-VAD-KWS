"""Build train-only source manifests for the reclean v2 training pool."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class RawAnchorResult:
    path: Path
    counts: dict[str, int]
    protected_source_count: int


def iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield row


def _resolved_audio_path(row: dict[str, object], manifest_path: Path) -> str:
    value = row.get("audio_filepath", row.get("audio_path"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing audio_filepath in {manifest_path}")
    path = Path(value.strip())
    if not path.is_absolute():
        path = manifest_path.parent / path
    return str(path.resolve())


def protected_audio_paths(*manifest_paths: Path) -> set[str]:
    protected: set[str] = set()
    for manifest_path in manifest_paths:
        manifest_path = Path(manifest_path)
        for row in iter_jsonl(manifest_path):
            protected.add(_resolved_audio_path(row, manifest_path))
    return protected


def _atomic_write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def build_raw_anchor_manifest(
    train_manifest: Path,
    validation_manifest: Path,
    test_manifest: Path,
    output_path: Path,
) -> RawAnchorResult:
    """Atomically emit raw train positives and negatives with source-role metadata."""
    train_manifest = Path(train_manifest)
    protected = protected_audio_paths(Path(validation_manifest), Path(test_manifest))
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    counts = {"raw_negative": 0, "raw_positive": 0}

    for source_row in iter_jsonl(train_manifest):
        audio_path = _resolved_audio_path(source_row, train_manifest)
        if audio_path in protected:
            raise ValueError(f"Training manifest includes held-out source: {audio_path}")
        if audio_path in seen:
            raise ValueError(f"Training manifest includes duplicate source: {audio_path}")
        seen.add(audio_path)
        command = source_row.get("command", source_row.get("label"))
        if command not in {"positive", "negative"}:
            raise ValueError(f"Raw anchor label must be positive or negative, got {command!r}")
        source_role = f"raw_{command}"
        rows.append(
            {
                "audio_filepath": audio_path,
                "command": command,
                "online_window_jitter_max_ms": 200 if command == "positive" else 0,
                "source_role": source_role,
                "source_split": "train",
            }
        )
        counts[source_role] += 1

    if not all(counts.values()):
        raise ValueError(f"Raw anchors require both labels, got {counts}")
    rows.sort(key=lambda row: str(row["audio_filepath"]))
    output_path = Path(output_path)
    _atomic_write_jsonl(output_path, rows)
    return RawAnchorResult(output_path, counts, len(protected))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train-only reclean v2 pool artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)
    raw = subparsers.add_parser("raw-anchors", help="write raw train-only Mobvoi anchors")
    raw.add_argument("--train-manifest", type=Path, required=True)
    raw.add_argument("--validation-manifest", type=Path, required=True)
    raw.add_argument("--test-manifest", type=Path, required=True)
    raw.add_argument("--output", type=Path, required=True)
    mine = subparsers.add_parser("hard-negatives", help="score and materialize train-only hard negatives")
    mine.add_argument("--checkpoint", type=Path, required=True)
    mine.add_argument("--train-manifest", type=Path, required=True)
    mine.add_argument("--validation-manifest", type=Path, required=True)
    mine.add_argument("--test-manifest", type=Path, required=True)
    mine.add_argument("--false-wake-train", type=Path, nargs="+", required=True)
    mine.add_argument("--output-root", type=Path, required=True)
    mine.add_argument("--gpu", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "raw-anchors":
        result = build_raw_anchor_manifest(
            train_manifest=args.train_manifest,
            validation_manifest=args.validation_manifest,
            test_manifest=args.test_manifest,
            output_path=args.output,
        )
        print(
            json.dumps(
                {
                    "path": str(result.path),
                    "counts": result.counts,
                    "protected_source_count": result.protected_source_count,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "hard-negatives":
        import torch

        from .hard_negative_mining import build_checkpoint_scorer, mine_hard_negatives

        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        result = mine_hard_negatives(
            train_manifest=args.train_manifest,
            validation_manifest=args.validation_manifest,
            test_manifest=args.test_manifest,
            false_wake_sources=[{"audio_path": str(path), "source_split": "train"} for path in args.false_wake_train],
            output_root=args.output_root,
            score_candidate=build_checkpoint_scorer(args.checkpoint, device),
            sample_rate=load_scoring_sample_rate(args.checkpoint),
        )
        print(
            json.dumps(
                {
                    "audit_path": str(result.audit_path),
                    "manifest_path": str(result.manifest_path),
                    "report_path": str(result.report_path),
                    "selected_count": result.selected_count,
                },
                sort_keys=True,
            )
        )
        return 0
    raise ValueError(f"Unsupported v2 pool command: {args.command}")


def load_scoring_sample_rate(checkpoint: Path) -> int:
    from .hard_negative_mining import load_scoring_config

    return load_scoring_config(checkpoint).sample_rate


if __name__ == "__main__":
    raise SystemExit(main())
