from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import random
import re
from pathlib import Path


TAU_SCENES = [
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
]

DEFAULT_SNRS_DB = [20.0, 10.0, 5.0, 0.0, -5.0]
SPLIT_TO_MANIFEST = {
    "train": "train_manifest.json",
    "validation": "validation_manifest.json",
    "test": "test_manifest.json",
}

torch = None
F = None
torchaudio = None
AF = None


def import_audio_dependencies() -> None:
    global torch
    global F
    global torchaudio
    global AF

    try:
        import torch as torch_mod
        import torch.nn.functional as f_mod
        import torchaudio as torchaudio_mod
        import torchaudio.functional as af_mod
    except ModuleNotFoundError as exc:
        missing = exc.name or "torch/torchaudio"
        raise SystemExit(
            f"Missing dependency '{missing}'. Run this script in the same environment "
            "used for DSCNN training/evaluation, or install torch and torchaudio."
        ) from exc

    torch = torch_mod
    F = f_mod
    torchaudio = torchaudio_mod
    AF = af_mod


def repo_package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(raw: str | Path, roots: list[Path], fallback_root: Path | None = None) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    for root in roots:
        candidate = root / path
        if candidate.exists():
            return candidate
    if fallback_root is not None:
        return fallback_root / path
    return roots[0] / path


def resolve_creatable_path(raw: str | Path, roots: list[Path]) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    for root in roots:
        candidate = root / path
        if candidate.exists():
            return candidate
    return Path.cwd() / path


def resolve_dataset_path(raw: str, data_root: Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return data_root / path


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
    path = Path(normalized).expanduser()
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


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_float(value: float) -> str:
    return f"{float(value):g}"


def snr_token(snr_db: float) -> str:
    sign = "m" if snr_db < 0 else "p"
    magnitude = format_float(abs(snr_db)).replace(".", "p")
    return f"{sign}{magnitude}db"


def safe_name(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw).strip())
    return cleaned.strip("._") or "item"


def rel_to(path: Path, base: Path) -> str:
    return os.path.relpath(path, base).replace("\\", "/")


def is_usable_wav(path: Path) -> bool:
    try:
        return path.is_file() and path.suffix.lower() == ".wav" and path.stat().st_size > 44
    except OSError:
        return False


def infer_scene(path: Path, scenes: list[str]) -> str | None:
    lower_parts = [part.lower() for part in path.parts]
    filename = path.name.lower()
    for scene in sorted(scenes, key=len, reverse=True):
        scene_lower = scene.lower()
        if scene_lower in lower_parts:
            return scene
        if filename.startswith(scene_lower + "-") or filename.startswith(scene_lower + "_"):
            return scene
    return None


def scan_wavs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.wav") if is_usable_wav(path))


def read_noise_list(list_path: Path) -> list[Path]:
    files: list[Path] = []
    base = list_path.parent
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            path = Path(item).expanduser()
            if not path.is_absolute():
                path = base / path
            if is_usable_wav(path):
                files.append(path)
    return sorted(files)


def collect_tau_scene_files(
    tau_root: Path,
    scenes: list[str],
    tau_list: Path | None,
) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {scene: [] for scene in scenes}
    seen: set[Path] = set()

    if tau_list is not None:
        candidates = read_noise_list(tau_list)
    else:
        candidates = []
        for scene in scenes:
            scene_dir = tau_root / scene
            if scene_dir.is_dir():
                candidates.extend(scan_wavs(scene_dir))
        candidates.extend(scan_wavs(tau_root))

    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        scene = infer_scene(resolved, scenes)
        if scene is not None:
            grouped[scene].append(resolved)

    for scene in grouped:
        grouped[scene] = sorted(grouped[scene])
    missing = [scene for scene, files in grouped.items() if not files]
    if missing:
        counts = {scene: len(files) for scene, files in grouped.items()}
        source = str(tau_list) if tau_list is not None else str(tau_root)
        raise FileNotFoundError(
            f"Missing usable TAU wav files for scenes {missing} from {source}. "
            f"Usable counts by scene: {counts}"
        )
    return grouped


def load_mono(path: Path, sample_rate: int, allow_resample: bool) -> torch.Tensor:
    waveform, orig_sr = torchaudio.load(str(path))
    if waveform.numel() == 0:
        raise ValueError(f"empty audio: {path}")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if int(orig_sr) != int(sample_rate):
        if not allow_resample:
            raise ValueError(f"sample-rate mismatch for {path}: got {orig_sr}, expected {sample_rate}")
        waveform = AF.resample(waveform, int(orig_sr), int(sample_rate))
    return waveform.to(torch.float32).clamp(-1.0, 1.0)


def fit_speech_to_length(waveform: torch.Tensor, length: int) -> torch.Tensor:
    if waveform.shape[1] < length:
        waveform = F.pad(waveform, [0, length - waveform.shape[1]])
    if waveform.shape[1] == length:
        return waveform
    offset = (waveform.shape[1] - length) // 2
    return waveform.narrow(1, offset, length)


def noise_segment(noise: torch.Tensor, length: int, rng: random.Random) -> torch.Tensor:
    if noise.shape[1] < length:
        repeat_count = math.ceil(length / max(1, noise.shape[1]))
        noise = noise.repeat(1, repeat_count)
    offset = rng.randint(0, noise.shape[1] - length)
    return noise.narrow(1, offset, length)


def rms(waveform: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(waveform * waveform) + 1e-12)


def mix_at_snr(
    speech: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    speech_rms = rms(speech)
    raw_noise_rms = rms(noise)
    target_noise_rms = speech_rms / (10.0 ** (float(snr_db) / 20.0))
    scaled_noise = noise * (target_noise_rms / raw_noise_rms)
    mixed = speech + scaled_noise
    peak_before = float(mixed.abs().max().item())
    normalization_gain = 1.0
    if peak_before > 0.99:
        normalization_gain = 0.99 / peak_before
        mixed = mixed * normalization_gain
        scaled_noise = scaled_noise * normalization_gain
    return (
        mixed.clamp(-1.0, 1.0),
        scaled_noise.clamp(-1.0, 1.0),
        {
            "speech_rms": float(speech_rms.item()),
            "raw_noise_rms": float(raw_noise_rms.item()),
            "scaled_noise_rms": float(rms(scaled_noise).item()),
            "mixed_peak_before_norm": peak_before,
            "normalization_gain": normalization_gain,
        },
    )


def choose_records_by_label(
    records: list[dict],
    labels: list[str],
    samples_per_label: int,
    rng: random.Random,
) -> dict[str, list[dict]]:
    selected: dict[str, list[dict]] = {}
    command_counts: dict[str, int] = {}
    for record in records:
        command = str(record.get("command", ""))
        command_counts[command] = command_counts.get(command, 0) + 1

    for label in labels:
        label_records = [record for record in records if str(record.get("command", "")) == label]
        if len(label_records) < samples_per_label:
            raise ValueError(
                f"Need {samples_per_label} records for label '{label}', "
                f"but found {len(label_records)}. Manifest command counts: {command_counts}"
            )
        selected[label] = rng.sample(label_records, samples_per_label)
    return selected


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_html_index(path: Path, rows: list[dict], title: str) -> None:
    def audio_cell(rel_path: str) -> str:
        if not rel_path:
            return ""
        escaped = html.escape(rel_path, quote=True)
        return f'<audio controls preload="none" src="{escaped}"></audio>'

    body_rows = []
    for row in rows:
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['label']))}</td>"
            f"<td>{html.escape(str(row['sample_index']))}</td>"
            f"<td>{html.escape(str(row['scene']))}</td>"
            f"<td>{html.escape(str(row['snr_db']))}</td>"
            f"<td>{audio_cell(str(row.get('clean_ref_wav', '')))}</td>"
            f"<td>{audio_cell(str(row['output_wav']))}</td>"
            f"<td>{audio_cell(str(row.get('noise_ref_wav', '')))}</td>"
            "</tr>"
        )

    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<html lang="en">',
                "<head>",
                '<meta charset="utf-8">',
                f"<title>{html.escape(title)}</title>",
                "<style>",
                "body{font-family:Arial,sans-serif;margin:24px;line-height:1.35;}",
                "table{border-collapse:collapse;width:100%;}",
                "th,td{border:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:middle;}",
                "th{background:#f5f5f5;position:sticky;top:0;}",
                "audio{width:220px;max-width:100%;}",
                "</style>",
                "</head>",
                "<body>",
                f"<h1>{html.escape(title)}</h1>",
                f"<p>Total noisy examples: {len(rows)}</p>",
                "<table>",
                "<thead><tr><th>Label</th><th>Sample</th><th>Scene</th><th>SNR dB</th>"
                "<th>Clean</th><th>Noisy mix</th><th>Scaled noise</th></tr></thead>",
                "<tbody>",
                *body_rows,
                "</tbody>",
                "</table>",
                "</body>",
                "</html>",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    package_root = repo_package_root()
    parser = argparse.ArgumentParser(
        description="Generate listenable Mobvoi + TAU noisy preview wav files across labels, scenes, and SNRs."
    )
    parser.add_argument("--data_root", default=str(package_root / "data"))
    parser.add_argument("--dataset", default="mobvoi_hi_xiaowen_binary_hardneg")
    parser.add_argument("--split", choices=sorted(SPLIT_TO_MANIFEST), default="test")
    parser.add_argument("--manifest", default=None, help="Optional manifest path or name; overrides --split.")
    parser.add_argument("--tau_root", default=str(package_root / "noise" / "tau"))
    parser.add_argument("--tau_list", default=None, help="Optional TAU txt list to restrict noise files.")
    parser.add_argument(
        "--out_dir",
        default=str(package_root / "noise" / "listen_examples" / "tau_mobvoi_preview"),
    )
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--labels", nargs="+", default=["positive", "negative"])
    parser.add_argument("--scenes", nargs="+", default=TAU_SCENES)
    parser.add_argument("--snrs_db", nargs="+", type=float, default=DEFAULT_SNRS_DB)
    parser.add_argument("--samples_per_label", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_resample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_clean_refs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_noise_refs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--write_html", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.samples_per_label <= 0:
        raise ValueError("--samples_per_label must be positive")
    if args.sample_rate <= 0:
        raise ValueError("--sample_rate must be positive")
    if not args.snrs_db:
        raise ValueError("--snrs_db must not be empty")
    if not args.scenes:
        raise ValueError("--scenes must not be empty")
    return args


def main() -> None:
    args = parse_args()
    import_audio_dependencies()
    rng = random.Random(args.seed)
    package_root = repo_package_root()
    repo_root = package_root.parent
    data_root = resolve_path(args.data_root, [Path.cwd(), package_root, repo_root], fallback_root=package_root)
    dataset_root = resolve_dataset_path(args.dataset, data_root)
    if args.manifest is None:
        manifest_path = dataset_root / SPLIT_TO_MANIFEST[args.split]
    else:
        manifest_path = resolve_path(args.manifest, [dataset_root, Path.cwd(), package_root, repo_root], dataset_root)
    tau_root = resolve_path(args.tau_root, [Path.cwd(), package_root, repo_root], package_root)
    tau_list = None
    if args.tau_list is not None:
        tau_list = resolve_path(args.tau_list, [Path.cwd(), package_root, repo_root, tau_root], tau_root)
    out_dir = resolve_creatable_path(args.out_dir, [Path.cwd(), package_root, repo_root])

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if tau_list is not None and not tau_list.exists():
        raise FileNotFoundError(f"TAU list not found: {tau_list}")
    if tau_list is None and not tau_root.exists():
        raise FileNotFoundError(f"TAU root not found: {tau_root}")

    records = read_jsonl(manifest_path)
    selected_by_label = choose_records_by_label(records, args.labels, args.samples_per_label, rng)
    scene_noise_files = collect_tau_scene_files(tau_root, args.scenes, tau_list)

    out_dir.mkdir(parents=True, exist_ok=True)
    clean_ref_dir = out_dir / "clean_refs"
    mixed_root = out_dir / "mixed"
    noise_ref_root = out_dir / "scaled_noise_refs"
    if args.include_clean_refs:
        clean_ref_dir.mkdir(parents=True, exist_ok=True)
    if args.save_noise_refs:
        noise_ref_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] dataset_root={dataset_root}")
    print(f"[INFO] manifest={manifest_path}")
    print(f"[INFO] tau_root={tau_root}")
    if tau_list is not None:
        print(f"[INFO] tau_list={tau_list}")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] sample_rate={args.sample_rate}, allow_resample={args.allow_resample}")
    print(f"[INFO] labels={args.labels}")
    print(f"[INFO] scenes={args.scenes}")
    print(f"[INFO] snrs_db={[format_float(snr) for snr in args.snrs_db]}")
    print(f"[INFO] samples_per_label={args.samples_per_label}")
    print(f"[INFO] usable_tau_files_by_scene={ {scene: len(files) for scene, files in scene_noise_files.items()} }")

    metadata_rows: list[dict] = []
    clean_cache: dict[tuple[str, int], tuple[torch.Tensor, Path, Path | None]] = {}
    sample_length = int(args.sample_rate)

    for label in args.labels:
        for sample_index, record in enumerate(selected_by_label[label]):
            clean_path = resolve_audio_path(dataset_root, manifest_path, str(record["audio_filepath"]))
            speech = load_mono(clean_path, args.sample_rate, args.allow_resample)
            speech = fit_speech_to_length(speech, sample_length)

            clean_ref_path = None
            if args.include_clean_refs:
                clean_ref_path = clean_ref_dir / f"{safe_name(label)}_{sample_index:02d}_clean.wav"
                torchaudio.save(str(clean_ref_path), speech.cpu(), args.sample_rate)
            clean_cache[(label, sample_index)] = (speech, clean_path, clean_ref_path)

    item_index = 0
    for label in args.labels:
        for sample_index in range(args.samples_per_label):
            speech, clean_path, clean_ref_path = clean_cache[(label, sample_index)]
            for scene in args.scenes:
                for snr_db in args.snrs_db:
                    noise_path = rng.choice(scene_noise_files[scene])
                    noise = load_mono(noise_path, args.sample_rate, args.allow_resample)
                    noise = noise_segment(noise, sample_length, rng)
                    mixed, scaled_noise, stats = mix_at_snr(speech, noise, snr_db)

                    scene_dir = mixed_root / safe_name(label) / safe_name(scene) / f"snr_{snr_token(snr_db)}"
                    scene_dir.mkdir(parents=True, exist_ok=True)
                    out_name = (
                        f"{safe_name(label)}_{sample_index:02d}_{safe_name(scene)}_"
                        f"snr_{snr_token(snr_db)}.wav"
                    )
                    out_path = scene_dir / out_name
                    torchaudio.save(str(out_path), mixed.cpu(), args.sample_rate)

                    noise_ref_path = None
                    if args.save_noise_refs:
                        noise_ref_dir = noise_ref_root / safe_name(scene) / f"snr_{snr_token(snr_db)}"
                        noise_ref_dir.mkdir(parents=True, exist_ok=True)
                        noise_ref_path = noise_ref_dir / (
                            f"noise_{safe_name(label)}_{sample_index:02d}_{safe_name(scene)}_"
                            f"snr_{snr_token(snr_db)}.wav"
                        )
                        torchaudio.save(str(noise_ref_path), scaled_noise.cpu(), args.sample_rate)

                    metadata_rows.append(
                        {
                            "index": item_index,
                            "label": label,
                            "sample_index": sample_index,
                            "scene": scene,
                            "snr_db": format_float(snr_db),
                            "output_wav": rel_to(out_path, out_dir),
                            "clean_ref_wav": rel_to(clean_ref_path, out_dir) if clean_ref_path is not None else "",
                            "noise_ref_wav": rel_to(noise_ref_path, out_dir) if noise_ref_path is not None else "",
                            "clean_audio_filepath": str(clean_path.resolve()).replace("\\", "/"),
                            "noise_audio_filepath": str(noise_path.resolve()).replace("\\", "/"),
                            "sample_rate": args.sample_rate,
                            "speech_rms": f"{stats['speech_rms']:.8f}",
                            "raw_noise_rms": f"{stats['raw_noise_rms']:.8f}",
                            "scaled_noise_rms": f"{stats['scaled_noise_rms']:.8f}",
                            "mixed_peak_before_norm": f"{stats['mixed_peak_before_norm']:.8f}",
                            "normalization_gain": f"{stats['normalization_gain']:.8f}",
                        }
                    )
                    item_index += 1

    metadata_csv = out_dir / "metadata.csv"
    metadata_jsonl = out_dir / "metadata.jsonl"
    write_csv(metadata_csv, metadata_rows)
    write_jsonl(metadata_jsonl, metadata_rows)

    if args.write_html:
        title = f"Mobvoi TAU preview: {dataset_root.name} {manifest_path.name}"
        html_path = out_dir / "index.html"
        write_html_index(html_path, metadata_rows, title)
        print(f"[INFO] html -> {html_path}")

    print(f"[INFO] metadata_csv -> {metadata_csv}")
    print(f"[INFO] metadata_jsonl -> {metadata_jsonl}")
    print(f"[INFO] noisy_wav_count={len(metadata_rows)}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
