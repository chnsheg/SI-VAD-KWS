"""Stream the completed reclean training manifest into packed PCM16 shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torchaudio

from .packed_pcm import PackedPcmShardWriter


@dataclass(frozen=True)
class PackedCorpusResult:
    index_path: Path
    record_count: int
    output_root: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audio_path(row: dict[str, object], manifest_path: Path) -> Path:
    value = row.get("audio_filepath", row.get("audio_path"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("packed source row is missing audio_filepath")
    path = Path(value.strip())
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _binary_label(row: dict[str, object]) -> int:
    value = row.get("command", row.get("label"))
    if value == "positive":
        return 0
    if value == "negative":
        return 1
    raise ValueError(f"packed source row has a non-binary label: {value!r}")


def _jitter_ms(row: dict[str, object], *, label: int) -> int:
    if label != 0:
        return 0
    value = row.get("online_window_jitter_max_ms", 0)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 65_535:
        raise ValueError("online_window_jitter_max_ms must be an integer in [0, 65535]")
    return value


def pack_training_manifest(
    source_manifest: Path | str,
    output_root: Path | str,
    *,
    expected_count: int,
    shard_records: int = 16_384,
    limit: int | None = None,
    sample_rate: int = 16_000,
) -> PackedCorpusResult:
    """Pack an explicit generated-audio JSONL manifest without retaining its rows."""
    source_path = Path(source_manifest).resolve()
    destination = Path(output_root).resolve()
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if (destination / "manifest.json").exists():
        raise FileExistsError(f"packed corpus already published: {destination / 'manifest.json'}")

    writer = PackedPcmShardWriter(destination, sample_rate=sample_rate, shard_records=shard_records)
    count = 0
    try:
        with source_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                if limit is not None and count >= limit:
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {source_path}:{line_number}") from error
                if not isinstance(row, dict):
                    raise ValueError(f"packed source row is not an object at {source_path}:{line_number}")
                label = _binary_label(row)
                waveform, waveform_sample_rate = torchaudio.load(_audio_path(row, source_path))
                if waveform_sample_rate != sample_rate:
                    raise ValueError(
                        f"sample-rate mismatch at {source_path}:{line_number}: "
                        f"got {waveform_sample_rate}, expected {sample_rate}"
                    )
                writer.add(waveform, label=label, jitter_ms=_jitter_ms(row, label=label))
                count += 1
        if count != expected_count:
            raise ValueError(f"packed record count is {count}, expected {expected_count}")
        index_path = writer.finalize(
            source_manifest=str(source_path),
            metadata={"source_manifest_sha256": _sha256_file(source_path)},
        )
    except Exception:
        writer.close()
        raise
    return PackedCorpusResult(index_path=index_path, record_count=count, output_root=destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack recleaned KWS train WAVs into contiguous PCM16 shards")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--shard-records", type=int, default=16_384)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = pack_training_manifest(
        args.source_manifest,
        args.output_root,
        expected_count=args.expected_count,
        shard_records=args.shard_records,
        limit=args.limit,
        sample_rate=args.sample_rate,
    )
    print(json.dumps({"index_path": str(result.index_path), "record_count": result.record_count}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
