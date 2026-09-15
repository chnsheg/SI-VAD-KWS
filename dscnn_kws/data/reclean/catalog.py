from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import soundfile as sf
import torch
import torchaudio.functional as AF

from .audio import save_pcm16_atomic, sha256_file


TAU_SCENES = (
    "airport",
    "bus",
    "metro",
    "metro_station",
    "park",
    "public_square",
    "shopping_mall",
    "street_pedestrian",
    "street_traffic",
    "tram",
)
CAPTURED_SCENES = ("kindgarden", "livingroom", "pub", "road", "风噪")
ALL_NOISE_SCENES = (*TAU_SCENES, *CAPTURED_SCENES)
AUDIO_SUFFIXES = frozenset({".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac"})
RIRS_NOISES_LICENSE_URL = "https://www.openslr.org/28/"
RIRS_NOISES_IMPULSE_DIRECTORIES = frozenset({"real_rirs_isotropic_noises", "simulated_rirs"})
MOBVOI_SPLITS = (("train", "train"), ("dev", "validation"), ("test", "test"))


@dataclass(frozen=True)
class SceneChoice:
    scene: str
    path: Path


@dataclass(frozen=True)
class NoiseCatalog:
    files_by_scene: dict[str, tuple[Path, ...]]

    def choose_scene(self, rng: random.Random) -> SceneChoice:
        scene = rng.choice(tuple(sorted(self.files_by_scene)))
        return SceneChoice(scene=scene, path=rng.choice(self.files_by_scene[scene]))


@dataclass(frozen=True)
class FalseWakeSplit:
    source_id: str
    split: str


@dataclass(frozen=True)
class RirItem:
    source_path: Path
    source_sha256: str
    waveform: torch.Tensor
    direct_peak_index: int


@dataclass(frozen=True)
class NormalizedRirItem:
    source_path: Path
    source_sha256: str
    normalized_path: Path
    normalized_sha256: str
    direct_peak_index: int


@dataclass(frozen=True)
class AudioInventoryRow:
    role: str
    path: str
    sha256: str
    frames: int
    sample_rate: int
    channels: int
    format: str
    subtype: str
    duration_seconds: float
    scene: str | None = None
    split: str | None = None
    source_label: str | None = None


def build_noise_catalog(scene_files: Mapping[str, Sequence[Path]]) -> NoiseCatalog:
    normalized = {str(scene): tuple(Path(path) for path in files) for scene, files in scene_files.items()}
    empty_scenes = sorted(scene for scene, files in normalized.items() if not files)
    if empty_scenes:
        raise ValueError(f"Noise scenes without audio files: {', '.join(empty_scenes)}")
    if not normalized:
        raise ValueError("At least one noise scene is required")
    return NoiseCatalog(files_by_scene=normalized)


def validate_noise_scene_roots(scene_roots: Mapping[str, Path]) -> None:
    provided = set(scene_roots)
    expected = set(ALL_NOISE_SCENES)
    missing = sorted(expected - provided)
    unexpected = sorted(provided - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing scenes: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected scenes: {', '.join(unexpected)}")
        raise ValueError("; ".join(details))


def _split_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if total <= 0:
        return (0, 0, 0)
    if total >= 3:
        base = [1, 1, 1]
        remaining = total - 3
    else:
        base = [0, 0, 0]
        remaining = total
    raw = [remaining * ratio for ratio in ratios]
    floors = [int(value) for value in raw]
    for index, value in enumerate(floors):
        base[index] += value
    for index in sorted(range(3), key=lambda item: (raw[item] - floors[item], -item), reverse=True)[: remaining - sum(floors)]:
        base[index] += 1
    return tuple(base)  # type: ignore[return-value]


def assign_false_wake_splits(source_ids: Sequence[str | Path], seed: int) -> list[FalseWakeSplit]:
    """Assign all segments from a continuous source recording to the same split."""
    unique_sources = sorted({str(source) for source in source_ids})
    shuffled = unique_sources[:]
    random.Random(seed).shuffle(shuffled)
    train_count, validation_count, _ = _split_counts(len(shuffled), (0.80, 0.10, 0.10))
    mapping: dict[str, str] = {}
    for source in shuffled[:train_count]:
        mapping[source] = "train"
    for source in shuffled[train_count : train_count + validation_count]:
        mapping[source] = "validation"
    for source in shuffled[train_count + validation_count :]:
        mapping[source] = "test"
    return [FalseWakeSplit(source_id=str(source), split=mapping[str(source)]) for source in source_ids]


def find_audio_files(root: Path) -> list[Path]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Audio root does not exist: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES)


def select_rirs_noises_impulse_paths(paths: Sequence[Path]) -> list[Path]:
    """Keep only true RIR subsets from the mixed RIRS_NOISES release."""
    return [
        Path(path)
        for path in paths
        if any(part.lower() in RIRS_NOISES_IMPULSE_DIRECTORIES for part in Path(path).parts)
    ]


def build_rir_catalog(paths: Sequence[Path], sample_rate: int = 16000) -> list[RirItem]:
    """Read, resample, and direct-peak-normalize non-empty public RIRs."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    catalog: list[RirItem] = []
    for path in paths:
        path = Path(path).expanduser().resolve()
        waveform, source_sample_rate = sf.read(path, dtype="float32", always_2d=True)
        impulse = torch.from_numpy(waveform.T.copy()).mean(dim=0)
        if source_sample_rate != sample_rate:
            impulse = AF.resample(impulse.unsqueeze(0), int(source_sample_rate), sample_rate).squeeze(0)
        if impulse.numel() == 0 or not bool(torch.isfinite(impulse).all()):
            continue
        direct_peak_index = int(impulse.abs().argmax().item())
        direct_peak = float(impulse[direct_peak_index].abs().item())
        if direct_peak <= 1e-8:
            continue
        catalog.append(
            RirItem(
                source_path=path,
                source_sha256=sha256_file(path),
                waveform=(impulse / direct_peak).to(torch.float32),
                direct_peak_index=direct_peak_index,
            )
        )
    return catalog


def normalize_rir_catalog(paths: Sequence[Path], output_root: Path, sample_rate: int = 16000) -> list[NormalizedRirItem]:
    """Persist the normalized RIRs and their provenance for reproducible synthesis."""
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[NormalizedRirItem] = []
    for index, item in enumerate(build_rir_catalog(paths, sample_rate=sample_rate)):
        destination = output_root / f"rir-{index:06d}-{item.source_sha256[:12]}.wav"
        save_pcm16_atomic(destination, item.waveform.unsqueeze(0), sample_rate)
        rows.append(
            NormalizedRirItem(
                source_path=item.source_path,
                source_sha256=item.source_sha256,
                normalized_path=destination,
                normalized_sha256=sha256_file(destination),
                direct_peak_index=item.direct_peak_index,
            )
        )
    catalog_path = output_root / "rir_catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sample_rate": sample_rate,
                "license_url": RIRS_NOISES_LICENSE_URL,
                "items": [
                    {
                        **asdict(row),
                        "source_path": str(row.source_path),
                        "normalized_path": str(row.normalized_path),
                    }
                    for row in rows
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return rows


def download_rir_archive(url: str, archive_path: Path, expected_sha256: str) -> Path:
    """Download an archive atomically and accept it only after SHA-256 verification."""
    expected_sha256 = expected_sha256.lower()
    if len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
        raise ValueError("expected_sha256 must be a 64-character hexadecimal SHA-256")
    archive_path = Path(archive_path).expanduser().resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        actual = sha256_file(archive_path)
        if actual == expected_sha256:
            return archive_path
        raise ValueError(f"Existing archive hash mismatch for {archive_path}: {actual}")

    with tempfile.NamedTemporaryFile(prefix=f".{archive_path.stem}.", suffix=".download", dir=archive_path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        digest = hashlib.sha256()
        with urllib.request.urlopen(url) as response, temporary.open("wb") as destination:
            while chunk := response.read(1024 * 1024):
                destination.write(chunk)
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise ValueError(f"Downloaded archive hash mismatch: expected {expected_sha256}, got {actual}")
        os.replace(temporary, archive_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return archive_path


def _safe_extract_archive(archive_path: Path, extraction_root: Path) -> list[Path]:
    extraction_root.mkdir(parents=True, exist_ok=True)
    root = extraction_root.resolve()
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if any(not (root / name).resolve().is_relative_to(root) for name in names):
                raise ValueError("Archive contains a path outside the extraction root")
            archive.extractall(root)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            members = archive.getmembers()
            if any(not (root / member.name).resolve().is_relative_to(root) for member in members):
                raise ValueError("Archive contains a path outside the extraction root")
            archive.extractall(root, members=members, filter="data")
    else:
        raise ValueError(f"Unsupported RIR archive format: {archive_path}")
    return find_audio_files(root)


def acquire_and_normalize_rirs(
    *,
    url: str,
    expected_sha256: str,
    archive_path: Path,
    extraction_root: Path,
    normalized_root: Path,
) -> list[NormalizedRirItem]:
    archive_path = download_rir_archive(url, archive_path, expected_sha256)
    source_paths = select_rirs_noises_impulse_paths(_safe_extract_archive(archive_path, Path(extraction_root)))
    if not source_paths:
        raise ValueError("No RIR impulses found in the extracted RIRS_NOISES archive")
    rows = normalize_rir_catalog(source_paths, normalized_root)
    manifest = {
        "schema_version": 1,
        "archive_url": url,
        "license_url": RIRS_NOISES_LICENSE_URL,
        "archive_path": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "extraction_root": str(Path(extraction_root).expanduser().resolve()),
        "extracted_audio_files": [str(path) for path in source_paths],
        "normalized_rir_catalog": str(Path(normalized_root).expanduser().resolve() / "rir_catalog.json"),
    }
    Path(normalized_root).expanduser().resolve().joinpath("rir_acquisition.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return rows


def discover_tau_scene_files(tau_root: Path) -> dict[str, list[Path]]:
    files = find_audio_files(tau_root)
    catalog = {scene: [] for scene in TAU_SCENES}
    for path in files:
        location = path.as_posix().lower()
        matched = [scene for scene in TAU_SCENES if scene in location]
        if matched:
            # Long labels, such as metro_station, win over their contained tokens.
            catalog[max(matched, key=len)].append(path)
    return catalog


def _inspect_audio(
    path: Path,
    role: str,
    scene: str | None = None,
    split: str | None = None,
    source_label: str | None = None,
) -> AudioInventoryRow:
    try:
        info = sf.info(path)
    except RuntimeError as error:
        raise ValueError(f"Unreadable audio file: {path}: {error}") from error
    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise ValueError(f"Invalid audio metadata for {path}")
    return AudioInventoryRow(
        role=role,
        path=str(path.resolve()),
        sha256=sha256_file(path),
        frames=int(info.frames),
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        format=str(info.format),
        subtype=str(info.subtype),
        duration_seconds=float(info.frames / info.samplerate),
        scene=scene,
        split=split,
        source_label=source_label,
    )


def _find_manifest_files(root: Path) -> list[Path]:
    manifests = sorted(root.rglob("*_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No *_manifest.json files under {root}")
    return manifests


def _manifest_split(manifest: Path) -> str:
    return "validation" if "validation" in manifest.name else "test" if "test" in manifest.name else "train"


def _iter_mobvoi_manifest_entries(manifest_root: Path):
    for manifest in _find_manifest_files(manifest_root):
        split = _manifest_split(manifest)
        for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                audio_path = (manifest.parent / entry["audio_filepath"]).resolve()
                source_label = str(entry["command"])
            except (KeyError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid manifest record {manifest}:{line_number}") from error
            yield audio_path, split, source_label


def _mobvoi_rows(manifest_root: Path) -> list[AudioInventoryRow]:
    rows: list[AudioInventoryRow] = []
    seen: set[tuple[Path, str]] = set()
    for audio_path, split, source_label in _iter_mobvoi_manifest_entries(manifest_root):
        key = (audio_path, split)
        if key not in seen:
            rows.append(
                _inspect_audio(
                    audio_path,
                    role="mobvoi_speech",
                    split=split,
                    source_label=source_label,
                )
            )
            seen.add(key)
    return rows


def read_mobvoi_manifest_rows(manifest_root: Path) -> list[AudioInventoryRow]:
    return _mobvoi_rows(Path(manifest_root).expanduser().resolve())


def _read_mobvoi_resource_entries(path: Path) -> list[dict[str, object]]:
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid Mobvoi resource JSON: {path}") from error
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise ValueError(f"Mobvoi resource must contain an array of objects: {path}")
    return entries


def _mobvoi_audio_path(audio_root: Path, entry: Mapping[str, object]) -> Path:
    try:
        utterance_id = Path(str(entry["utt_id"])).stem
    except KeyError as error:
        raise ValueError("Mobvoi resource entry has no utt_id") from error
    if not utterance_id:
        raise ValueError("Mobvoi resource entry has an empty utt_id")
    path = (audio_root / f"{utterance_id}.wav").resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Mobvoi WAV is missing: {path}")
    return path


def _keyword_id(entry: Mapping[str, object], resource_path: Path) -> int:
    try:
        return int(entry["keyword_id"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Mobvoi resource entry has an invalid keyword_id: {resource_path}") from error


def _deduplicate_mobvoi_entries(entries: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    unique: dict[str, dict[str, object]] = {}
    for entry in entries:
        utterance_id = Path(str(entry.get("utt_id", ""))).stem
        if not utterance_id:
            raise ValueError("Mobvoi resource entry has an empty utt_id")
        unique.setdefault(utterance_id, entry)
    return [unique[utterance_id] for utterance_id in sorted(unique)]


def build_mobvoi_hard_negative_manifests(
    resource_root: Path,
    audio_root: Path,
    output_root: Path,
    *,
    target_keyword_id: int = 0,
    seed: int = 42,
) -> dict[str, Path]:
    """Build deterministic 1:1 target-keyword manifests from official Mobvoi metadata."""
    resource_root = Path(resource_root).expanduser().resolve()
    audio_root = Path(audio_root).expanduser().resolve()
    if not resource_root.is_dir():
        raise FileNotFoundError(f"Mobvoi resource root does not exist: {resource_root}")
    if not audio_root.is_dir():
        raise FileNotFoundError(f"Mobvoi audio root does not exist: {audio_root}")

    manifest_root = Path(output_root).expanduser().resolve() / "source_manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, Path] = {}
    for resource_split, output_split in MOBVOI_SPLITS:
        positive_resource = resource_root / f"p_{resource_split}.json"
        negative_resource = resource_root / f"n_{resource_split}.json"
        positives = _read_mobvoi_resource_entries(positive_resource)
        negatives = _read_mobvoi_resource_entries(negative_resource)
        target_positives = _deduplicate_mobvoi_entries(
            [entry for entry in positives if _keyword_id(entry, positive_resource) == target_keyword_id]
        )
        hard_negative_candidates = _deduplicate_mobvoi_entries(
            [*negatives, *(entry for entry in positives if _keyword_id(entry, positive_resource) != target_keyword_id)]
        )
        if not target_positives:
            raise ValueError(f"Mobvoi split has no keyword_id={target_keyword_id} positives: {resource_split}")
        if len(hard_negative_candidates) < len(target_positives):
            raise ValueError(
                f"Mobvoi split lacks hard negatives for 1:1 balancing: {resource_split} "
                f"({len(hard_negative_candidates)} available for {len(target_positives)} positives)"
            )

        split_seed = int(
            hashlib.sha256(f"mobvoi-hard-negative-v1:{seed}:{target_keyword_id}:{resource_split}".encode("utf-8")).hexdigest()[:16],
            16,
        )
        rng = random.Random(split_seed)
        selected_negatives = rng.sample(hard_negative_candidates, len(target_positives))
        records = [
            {"audio_filepath": str(_mobvoi_audio_path(audio_root, entry)), "command": "positive"}
            for entry in target_positives
        ]
        records.extend(
            {"audio_filepath": str(_mobvoi_audio_path(audio_root, entry)), "command": "negative"}
            for entry in selected_negatives
        )
        rng.shuffle(records)
        manifest_path = manifest_root / f"{output_split}_manifest.json"
        manifest_path.write_text(
            "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records), encoding="utf-8"
        )
        manifests[output_split] = manifest_path
    return manifests


def _fast_mobvoi_rows(manifest_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[Path, str]] = set()
    for audio_path, split, source_label in _iter_mobvoi_manifest_entries(manifest_root):
        if source_label not in {"positive", "negative"}:
            raise ValueError(f"Unsupported Mobvoi source label: {source_label}")
        if not audio_path.is_file():
            raise FileNotFoundError(f"Mobvoi WAV is missing: {audio_path}")
        key = (audio_path, split)
        if key not in seen:
            rows.append(
                {
                    "role": "mobvoi_speech",
                    "path": str(audio_path),
                    "split": split,
                    "source_label": source_label,
                }
            )
            seen.add(key)
    return rows


def _catalogued_rir_rows(rir_root: Path) -> list[dict[str, object]]:
    rir_root = Path(rir_root).expanduser().resolve()
    catalog_path = rir_root / "rir_catalog.json"
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Normalized RIR catalog does not exist: {catalog_path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid normalized RIR catalog: {catalog_path}") from error
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError(f"Normalized RIR catalog has no items: {catalog_path}")

    rows: list[dict[str, object]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Normalized RIR catalog item {index} is not an object")
        try:
            path = Path(str(item["normalized_path"])).expanduser().resolve()
            normalized_sha256 = str(item["normalized_sha256"]).lower()
        except KeyError as error:
            raise ValueError(f"Normalized RIR catalog item {index} is incomplete") from error
        if not path.is_relative_to(rir_root):
            raise ValueError(f"Normalized RIR path is outside the catalog root: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Normalized RIR is missing: {path}")
        if len(normalized_sha256) != 64 or any(character not in "0123456789abcdef" for character in normalized_sha256):
            raise ValueError(f"Normalized RIR catalog item {index} has an invalid SHA-256")
        rows.append(
            {
                "role": "rir",
                "path": str(path),
                "already_normalized": True,
                "normalized_sha256": normalized_sha256,
            }
        )
    return rows


def _existing_parent(path: Path) -> Path:
    probe = path.expanduser().resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return probe


def _available_space_bytes(path: Path) -> int:
    return int(shutil.disk_usage(_existing_parent(path)).free)


def _write_fast_inventory(output_root: Path, report: dict[str, object]) -> None:
    (output_root / "inventory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = report["files"]
    if not isinstance(rows, list):
        raise ValueError("Fast inventory report has no files list")
    fieldnames = sorted({key for row in rows if isinstance(row, dict) for key in row})
    with (output_root / "inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fast_inventory_sources(
    *,
    mobvoi_resource_root: Path,
    mobvoi_audio_root: Path,
    target_keyword_id: int,
    tau_root: Path,
    captured_scene_roots: Mapping[str, Path],
    false_wake_root: Path,
    rir_root: Path,
    output_root: Path,
    require_free_gib: float = 100.0,
    false_wake_seed: int = 42,
) -> dict[str, object]:
    """Persist a lightweight, reproducible source list without inspecting or hashing source audio."""
    if require_free_gib < 0:
        raise ValueError("require_free_gib must be non-negative")

    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    available_space_bytes = _available_space_bytes(output_root)
    required_space_bytes = int(require_free_gib * 1024**3)
    if available_space_bytes < required_space_bytes:
        raise OSError(
            f"Output volume has {available_space_bytes / 1024**3:.2f} GiB free; requires {require_free_gib:.2f} GiB"
        )

    manifests = build_mobvoi_hard_negative_manifests(
        mobvoi_resource_root,
        mobvoi_audio_root,
        output_root,
        target_keyword_id=target_keyword_id,
        seed=false_wake_seed,
    )
    tau_files = discover_tau_scene_files(tau_root)
    scene_files = {**tau_files, **{scene: find_audio_files(root) for scene, root in captured_scene_roots.items()}}
    validate_noise_scene_roots(scene_files)
    build_noise_catalog(scene_files)

    rows = _fast_mobvoi_rows(output_root / "source_manifests")
    for scene in ALL_NOISE_SCENES:
        rows.extend({"role": "noise", "path": str(path), "scene": scene} for path in scene_files[scene])
    false_wake_files = find_audio_files(false_wake_root)
    false_wake_splits = assign_false_wake_splits(false_wake_files, false_wake_seed)
    rows.extend(
        {"role": "false_wake", "path": str(Path(split.source_id).resolve()), "split": split.split}
        for split in false_wake_splits
    )
    rows.extend(_catalogued_rir_rows(rir_root))

    report: dict[str, object] = {
        "schema_version": 1,
        "output_root": str(output_root),
        "available_space_bytes": available_space_bytes,
        "required_space_bytes": required_space_bytes,
        "mobvoi_target_keyword_id": target_keyword_id,
        "mobvoi_manifests": {split: str(path) for split, path in manifests.items()},
        "noise_scene_file_counts": {scene: len(scene_files[scene]) for scene in ALL_NOISE_SCENES},
        "noise_scene_count": len(ALL_NOISE_SCENES),
        "false_wake_source_splits": [asdict(split) for split in false_wake_splits],
        "files": rows,
    }
    _write_fast_inventory(output_root, report)
    return report


def inventory_sources(
    *,
    mobvoi_manifest_root: Path,
    tau_root: Path,
    captured_scene_roots: Mapping[str, Path],
    false_wake_root: Path,
    rir_root: Path,
    output_root: Path,
    require_free_gib: float = 100.0,
    false_wake_seed: int = 42,
) -> dict[str, object]:
    """Audit all generator inputs and persist hash-addressed source provenance."""
    if require_free_gib < 0:
        raise ValueError("require_free_gib must be non-negative")

    tau_files = discover_tau_scene_files(tau_root)
    scene_files = {**tau_files, **{scene: find_audio_files(root) for scene, root in captured_scene_roots.items()}}
    validate_noise_scene_roots(scene_files)
    build_noise_catalog(scene_files)

    available_space_bytes = _available_space_bytes(output_root)
    required_space_bytes = int(require_free_gib * 1024**3)
    if available_space_bytes < required_space_bytes:
        raise OSError(
            f"Output volume has {available_space_bytes / 1024**3:.2f} GiB free; requires {require_free_gib:.2f} GiB"
        )

    rows = _mobvoi_rows(Path(mobvoi_manifest_root).expanduser().resolve())
    for scene in ALL_NOISE_SCENES:
        rows.extend(_inspect_audio(path, role="noise", scene=scene) for path in scene_files[scene])

    false_wake_files = find_audio_files(false_wake_root)
    false_wake_splits = assign_false_wake_splits(false_wake_files, false_wake_seed)
    rows.extend(
        _inspect_audio(Path(split.source_id), role="false_wake", split=split.split) for split in false_wake_splits
    )
    rows.extend(_inspect_audio(path, role="rir") for path in find_audio_files(rir_root))

    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    by_scene = {scene: len(scene_files[scene]) for scene in ALL_NOISE_SCENES}
    report: dict[str, object] = {
        "schema_version": 1,
        "output_root": str(output_root),
        "available_space_bytes": available_space_bytes,
        "required_space_bytes": required_space_bytes,
        "noise_scene_file_counts": by_scene,
        "noise_scene_count": len(ALL_NOISE_SCENES),
        "false_wake_source_splits": [asdict(split) for split in false_wake_splits],
        "files": [asdict(row) for row in rows],
    }
    (output_root / "inventory.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_root / "inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(AudioInventoryRow.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    return report
