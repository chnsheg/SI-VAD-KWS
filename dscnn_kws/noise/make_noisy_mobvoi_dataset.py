from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF


SPLIT_TO_MANIFEST = {
    "train": "train_manifest.json",
    "validation": "validation_manifest.json",
    "test": "test_manifest.json",
}


def repo_package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_dataset_path(raw: str, data_root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    candidate = data_root / raw
    if candidate.exists() or not Path(raw).exists():
        return candidate
    return Path(raw)


def normalize_manifest_audio_path(raw_path: str) -> str:
    normalized = os.path.normpath(str(raw_path).strip())
    normalized_slash = normalized.replace("\\", "/")
    if normalized_slash.startswith("./"):
        normalized_slash = normalized_slash[2:]
    legacy_prefix = "TorchKWS/dataset/"
    if normalized_slash.startswith(legacy_prefix):
        normalized_slash = normalized_slash[len(legacy_prefix) :]
    return os.path.normpath(normalized_slash)


def resolve_audio_path(dataset_root: Path, manifest_path: Path, rel_or_abs: str) -> Path:
    normalized = normalize_manifest_audio_path(rel_or_abs)
    path = Path(normalized)
    if path.is_absolute():
        return path

    package_root = repo_package_root()
    repo_root = package_root.parent
    dataset_parent = dataset_root.parent
    basename = Path(normalized).name
    candidates = [
        dataset_parent / normalized,
        repo_root / normalized,
        dataset_root / basename,
        dataset_root / normalized,
        manifest_path.parent / normalized,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Failed to resolve audio path '{rel_or_abs}' from {manifest_path}. "
        f"Tried: {[str(p) for p in candidates]}"
    )


def read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def find_noise_files(noise_roots: list[Path]) -> list[Path]:
    noise_files: list[Path] = []
    for root in noise_roots:
        if not root.exists():
            print(f"[WARN] noise root not found, skipped: {root}")
            continue
        for wav in sorted(root.rglob("*.wav")):
            if wav.stat().st_size > 44:
                noise_files.append(wav)
    return noise_files


def load_mono(path: Path, sample_rate: int) -> torch.Tensor:
    waveform, orig_sr = torchaudio.load(str(path))
    if waveform.numel() == 0:
        raise ValueError(f"empty audio: {path}")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if int(orig_sr) != int(sample_rate):
        waveform = AF.resample(waveform, int(orig_sr), int(sample_rate))
    return waveform.to(torch.float32).clamp(-1.0, 1.0)


def fit_to_length(waveform: torch.Tensor, length: int, rng: random.Random, random_crop: bool) -> torch.Tensor:
    if waveform.shape[1] < length:
        waveform = F.pad(waveform, [0, length - waveform.shape[1]])
    if waveform.shape[1] == length:
        return waveform
    max_offset = waveform.shape[1] - length
    if random_crop:
        offset = rng.randint(0, max_offset)
    else:
        offset = max_offset // 2
    return waveform.narrow(1, offset, length)


def noise_segment(noise: torch.Tensor, length: int, rng: random.Random) -> torch.Tensor:
    if noise.shape[1] < length:
        repeat_count = math.ceil(length / max(1, noise.shape[1]))
        noise = noise.repeat(1, repeat_count)
    offset = rng.randint(0, noise.shape[1] - length)
    return noise.narrow(1, offset, length)


def rms(waveform: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(waveform * waveform) + 1e-12)


def mix_at_snr(speech: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    speech_rms = rms(speech)
    noise_rms = rms(noise)
    target_noise_rms = speech_rms / (10.0 ** (snr_db / 20.0))
    scaled_noise = noise * (target_noise_rms / noise_rms)
    mixed = speech + scaled_noise
    peak = mixed.abs().max()
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)
    return mixed.clamp(-1.0, 1.0)


def parse_noise_roots(raw_roots: list[str], package_root: Path) -> list[Path]:
    roots = []
    for raw in raw_roots:
        path = Path(raw)
        if not path.is_absolute():
            path = package_root / raw
        roots.append(path)
    return roots


def sample_records(records: list[dict], limit: int | None, rng: random.Random) -> list[dict]:
    if limit is None or limit <= 0 or limit >= len(records):
        return list(records)
    return rng.sample(records, limit)


def safe_stem(path: Path, fallback_index: int) -> str:
    stem = path.stem.strip()
    if stem:
        return stem
    return f"sample_{fallback_index:08d}"


def build_split(
    split: str,
    records: list[dict],
    source_dataset: Path,
    source_manifest: Path,
    output_dataset: Path,
    noise_files: list[Path],
    sample_rate: int,
    snr_min_db: float,
    snr_max_db: float,
    fixed_snr_db: float | None,
    copies: int,
    include_clean: bool,
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    manifest_rows: list[dict] = []
    metadata_rows: list[dict] = []
    audio_dir = output_dataset / "audio" / split
    audio_dir.mkdir(parents=True, exist_ok=True)
    sample_length = sample_rate

    if include_clean:
        for item in records:
            clean_path = resolve_audio_path(source_dataset, source_manifest, item["audio_filepath"])
            manifest_rows.append(
                {
                    "audio_filepath": str(clean_path.resolve()).replace("\\", "/"),
                    "command": item["command"],
                }
            )

    for record_idx, item in enumerate(records):
        clean_path = resolve_audio_path(source_dataset, source_manifest, item["audio_filepath"])
        try:
            speech = load_mono(clean_path, sample_rate)
        except Exception as exc:
            print(f"[WARN] failed to load speech, skipped: {clean_path} ({exc})")
            continue

        speech = fit_to_length(
            speech,
            sample_length,
            rng=rng,
            random_crop=(split == "train"),
        )

        for copy_idx in range(max(0, copies)):
            noise_path = rng.choice(noise_files)
            try:
                noise = load_mono(noise_path, sample_rate)
            except Exception as exc:
                print(f"[WARN] failed to load noise, skipped: {noise_path} ({exc})")
                continue

            noise = noise_segment(noise, sample_length, rng)
            snr_db = fixed_snr_db if fixed_snr_db is not None else rng.uniform(snr_min_db, snr_max_db)
            mixed = mix_at_snr(speech, noise, snr_db)

            label = str(item["command"])
            stem = safe_stem(clean_path, record_idx)
            out_name = f"{stem}_noisy_{record_idx:08d}_{copy_idx:02d}.wav"
            out_path = audio_dir / label / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(out_path), mixed.cpu(), sample_rate)

            rel_path = os.path.relpath(out_path, output_dataset).replace("\\", "/")
            manifest_rows.append({"audio_filepath": rel_path, "command": label})
            metadata_rows.append(
                {
                    "split": split,
                    "command": label,
                    "audio_filepath": rel_path,
                    "clean_audio_filepath": str(clean_path.resolve()).replace("\\", "/"),
                    "noise_audio_filepath": str(noise_path.resolve()).replace("\\", "/"),
                    "snr_db": round(float(snr_db), 4),
                }
            )

    rng.shuffle(manifest_rows)
    return manifest_rows, metadata_rows


def parse_args() -> argparse.Namespace:
    package_root = repo_package_root()
    parser = argparse.ArgumentParser(
        description="Create an offline noisy Mobvoi-style binary KWS dataset."
    )
    parser.add_argument("--data_root", default=str(package_root / "data"))
    parser.add_argument("--source_dataset", required=True)
    parser.add_argument("--output_dataset", required=True)
    parser.add_argument("--noise_roots", nargs="+", default=[
        "noise/tau",
        "noise/demand",
        "noise/musan/noise",
        "noise/musan/music",
        "noise/musan/speech",
    ])
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--snr_min_db", type=float, default=-5.0)
    parser.add_argument("--snr_max_db", type=float, default=20.0)
    parser.add_argument("--fixed_snr_db", type=float, default=None)
    parser.add_argument("--train_noise_copies", type=int, default=1)
    parser.add_argument("--eval_noise_copies", type=int, default=1)
    parser.add_argument("--include_clean_train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_clean_eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_train", type=int, default=0)
    parser.add_argument("--max_validation", type=int, default=0)
    parser.add_argument("--max_test", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.snr_min_db > args.snr_max_db:
        raise ValueError("--snr_min_db must be <= --snr_max_db")
    return args


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    package_root = repo_package_root()
    data_root = Path(args.data_root)
    source_dataset = resolve_dataset_path(args.source_dataset, data_root)
    output_dataset = resolve_dataset_path(args.output_dataset, data_root)
    output_dataset.mkdir(parents=True, exist_ok=True)

    noise_roots = parse_noise_roots(args.noise_roots, package_root)
    noise_files = find_noise_files(noise_roots)
    if not noise_files:
        raise FileNotFoundError(f"No usable wav noise files found in: {[str(p) for p in noise_roots]}")

    print(f"[INFO] source_dataset={source_dataset}")
    print(f"[INFO] output_dataset={output_dataset}")
    print(f"[INFO] noise_files={len(noise_files)}")
    if args.fixed_snr_db is None:
        print(f"[INFO] random SNR range=[{args.snr_min_db}, {args.snr_max_db}] dB")
    else:
        print(f"[INFO] fixed SNR={args.fixed_snr_db} dB")

    all_metadata: list[dict] = []
    for split, manifest_name in SPLIT_TO_MANIFEST.items():
        source_manifest = source_dataset / manifest_name
        if not source_manifest.exists():
            raise FileNotFoundError(source_manifest)

        records = read_manifest(source_manifest)
        limit = {
            "train": args.max_train,
            "validation": args.max_validation,
            "test": args.max_test,
        }[split]
        records = sample_records(records, limit if limit > 0 else None, rng)

        copies = args.train_noise_copies if split == "train" else args.eval_noise_copies
        include_clean = args.include_clean_train if split == "train" else args.include_clean_eval
        rows, metadata = build_split(
            split=split,
            records=records,
            source_dataset=source_dataset,
            source_manifest=source_manifest,
            output_dataset=output_dataset,
            noise_files=noise_files,
            sample_rate=args.sample_rate,
            snr_min_db=args.snr_min_db,
            snr_max_db=args.snr_max_db,
            fixed_snr_db=args.fixed_snr_db,
            copies=copies,
            include_clean=include_clean,
            rng=rng,
        )

        write_jsonl(output_dataset / manifest_name, rows)
        all_metadata.extend(metadata)
        print(
            f"[INFO] {split}: source={len(records)}, manifest_rows={len(rows)}, "
            f"noisy_rows={len(metadata)} -> {output_dataset / manifest_name}"
        )

    write_jsonl(output_dataset / "noise_metadata.jsonl", all_metadata)
    print(f"[INFO] metadata -> {output_dataset / 'noise_metadata.jsonl'}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
