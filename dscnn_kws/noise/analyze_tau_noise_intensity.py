from __future__ import annotations

import argparse
import csv
import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCENES = [
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

EPS = 1e-12


@dataclass
class AudioStats:
    sample_rate: int
    channels: int
    frames: int
    sample_count: int
    sum_squares: float
    peak_abs: float
    duration_sec: float

    @property
    def rms(self) -> float:
        return math.sqrt(self.sum_squares / max(1, self.sample_count))

    @property
    def rms_dbfs(self) -> float:
        return linear_to_dbfs(self.rms)

    @property
    def peak_dbfs(self) -> float:
        return linear_to_dbfs(self.peak_abs)


def linear_to_dbfs(value: float) -> float:
    return 20.0 * math.log10(max(float(value), EPS))


def dbfs_to_linear(dbfs: float) -> float:
    return 10.0 ** (float(dbfs) / 20.0)


def pcm24_to_float32(raw: bytes) -> np.ndarray:
    data = np.frombuffer(raw, dtype=np.uint8)
    if data.size % 3 != 0:
        data = data[: data.size - (data.size % 3)]
    triples = data.reshape(-1, 3).astype(np.int32)
    values = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
    sign_bit = 1 << 23
    values = (values ^ sign_bit) - sign_bit
    return values.astype(np.float32) / float(1 << 23)


def pcm_bytes_to_float32(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        values = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (values - 128.0) / 128.0
    if sample_width == 2:
        values = np.frombuffer(raw, dtype="<i2").astype(np.float32)
        return values / float(1 << 15)
    if sample_width == 3:
        return pcm24_to_float32(raw)
    if sample_width == 4:
        values = np.frombuffer(raw, dtype="<i4").astype(np.float32)
        return values / float(1 << 31)
    raise ValueError(f"unsupported PCM sample width: {sample_width} bytes")


def stats_with_wave(path: Path, chunk_frames: int) -> AudioStats:
    with wave.open(str(path), "rb") as wf:
        sample_rate = int(wf.getframerate())
        channels = int(wf.getnchannels())
        sample_width = int(wf.getsampwidth())
        frames_total = int(wf.getnframes())

        sum_squares = 0.0
        sample_count = 0
        peak_abs = 0.0

        while True:
            raw = wf.readframes(chunk_frames)
            if not raw:
                break
            audio = pcm_bytes_to_float32(raw, sample_width)
            if audio.size == 0:
                continue
            audio64 = audio.astype(np.float64, copy=False)
            sum_squares += float(np.dot(audio64, audio64))
            sample_count += int(audio64.size)
            peak_abs = max(peak_abs, float(np.max(np.abs(audio64))))

    duration_sec = frames_total / sample_rate if sample_rate > 0 else 0.0
    return AudioStats(
        sample_rate=sample_rate,
        channels=channels,
        frames=frames_total,
        sample_count=sample_count,
        sum_squares=sum_squares,
        peak_abs=peak_abs,
        duration_sec=duration_sec,
    )


def stats_with_soundfile(path: Path, blocksize: int) -> AudioStats | None:
    try:
        import soundfile as sf
    except Exception:
        return None

    try:
        info = sf.info(str(path))
        sum_squares = 0.0
        sample_count = 0
        peak_abs = 0.0
        for block in sf.blocks(str(path), blocksize=blocksize, dtype="float64", always_2d=True):
            if block.size == 0:
                continue
            sum_squares += float(np.sum(block * block))
            sample_count += int(block.size)
            peak_abs = max(peak_abs, float(np.max(np.abs(block))))
        return AudioStats(
            sample_rate=int(info.samplerate),
            channels=int(info.channels),
            frames=int(info.frames),
            sample_count=sample_count,
            sum_squares=sum_squares,
            peak_abs=peak_abs,
            duration_sec=float(info.duration),
        )
    except Exception:
        return None


def read_audio_stats(path: Path, chunk_frames: int) -> AudioStats:
    stats = stats_with_soundfile(path, chunk_frames)
    if stats is not None:
        return stats
    return stats_with_wave(path, chunk_frames)


def infer_scene(path: Path, tau_root: Path) -> str | None:
    rel_parts = path.relative_to(tau_root).parts
    for part in rel_parts[:-1]:
        if part in SCENES:
            return part

    stem = path.stem
    for scene in sorted(SCENES, key=len, reverse=True):
        if stem == scene or stem.startswith(scene + "-") or stem.startswith(scene + "_"):
            return scene
    return None


def discover_wavs(tau_root: Path) -> list[tuple[str, Path]]:
    wavs = []
    for path in sorted(tau_root.rglob("*.wav")):
        if not path.is_file():
            continue
        try:
            if path.stat().st_size <= 44:
                continue
        except OSError:
            continue
        scene = infer_scene(path, tau_root)
        if scene is not None:
            wavs.append((scene, path))
    return wavs


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def format_float(value: float, digits: int = 3) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def summarize_by_scene(file_rows: list[dict]) -> list[dict]:
    rows = []
    by_scene = {scene: [] for scene in SCENES}
    for row in file_rows:
        by_scene[row["scene"]].append(row)

    scene_integrated = {}
    for scene, items in by_scene.items():
        if not items:
            scene_integrated[scene] = float("nan")
            continue
        total_sum_squares = sum(float(item["sum_squares"]) for item in items)
        total_sample_count = sum(int(item["sample_count"]) for item in items)
        scene_integrated[scene] = linear_to_dbfs(math.sqrt(total_sum_squares / max(1, total_sample_count)))

    loudest = max((v for v in scene_integrated.values() if not math.isnan(v)), default=float("nan"))

    for scene in SCENES:
        items = by_scene[scene]
        if not items:
            rows.append(
                {
                    "scene": scene,
                    "file_count": 0,
                    "total_duration_sec": 0.0,
                    "integrated_rms_dbfs": float("nan"),
                    "relative_to_loudest_db": float("nan"),
                    "mean_file_rms_dbfs": float("nan"),
                    "median_file_rms_dbfs": float("nan"),
                    "p10_file_rms_dbfs": float("nan"),
                    "p90_file_rms_dbfs": float("nan"),
                    "min_file_rms_dbfs": float("nan"),
                    "max_file_rms_dbfs": float("nan"),
                    "peak_dbfs": float("nan"),
                    "mean_duration_sec": float("nan"),
                }
            )
            continue

        rms_values = [float(item["rms_dbfs"]) for item in items]
        integrated = scene_integrated[scene]
        relative = integrated - loudest if not math.isnan(loudest) else float("nan")
        rows.append(
            {
                "scene": scene,
                "file_count": len(items),
                "total_duration_sec": sum(float(item["duration_sec"]) for item in items),
                "integrated_rms_dbfs": integrated,
                "relative_to_loudest_db": relative,
                "mean_file_rms_dbfs": float(np.mean(rms_values)),
                "median_file_rms_dbfs": percentile(rms_values, 50),
                "p10_file_rms_dbfs": percentile(rms_values, 10),
                "p90_file_rms_dbfs": percentile(rms_values, 90),
                "min_file_rms_dbfs": min(rms_values),
                "max_file_rms_dbfs": max(rms_values),
                "peak_dbfs": max(float(item["peak_dbfs"]) for item in items),
                "mean_duration_sec": float(np.mean([float(item["duration_sec"]) for item in items])),
            }
        )

    rows.sort(
        key=lambda row: (
            math.isnan(float(row["integrated_rms_dbfs"])),
            -float(row["integrated_rms_dbfs"]) if not math.isnan(float(row["integrated_rms_dbfs"])) else 0.0,
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["loudness_rank"] = rank if int(row["file_count"]) > 0 else ""
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path: Path, tau_root: Path, scene_rows: list[dict], file_count: int) -> None:
    lines = [
        "# TAU Noise Intensity Report",
        "",
        f"- TAU root: `{tau_root}`",
        f"- Usable wav files: `{file_count}`",
        "- Unit: `dBFS`, where 0 dBFS is digital full scale. More negative means quieter.",
        "- `integrated_rms_dbfs` is computed from all samples in the scene and is the main scene-level intensity metric.",
        "- This is digital signal level, not physical SPL. It is suitable for relative comparison inside this dataset.",
        "",
        "## Scene Summary",
        "",
        "| rank | scene | files | duration_s | integrated_rms_dbfs | relative_to_loudest_db | median_file_rms_dbfs | p10_file_rms_dbfs | p90_file_rms_dbfs | peak_dbfs |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scene_rows:
        rank = row["loudness_rank"]
        lines.append(
            "| "
            f"{rank} | "
            f"{row['scene']} | "
            f"{row['file_count']} | "
            f"{format_float(float(row['total_duration_sec']), 2)} | "
            f"{format_float(float(row['integrated_rms_dbfs']), 2)} | "
            f"{format_float(float(row['relative_to_loudest_db']), 2)} | "
            f"{format_float(float(row['median_file_rms_dbfs']), 2)} | "
            f"{format_float(float(row['p10_file_rms_dbfs']), 2)} | "
            f"{format_float(float(row['p90_file_rms_dbfs']), 2)} | "
            f"{format_float(float(row['peak_dbfs']), 2)} |"
        )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `tau_noise_intensity_by_scene.csv`: scene-level summary.",
            "- `tau_noise_intensity_by_file.csv`: per-file statistics.",
            "- `tau_noise_intensity_report.md`: this report.",
            "",
            "## Suggested Reading",
            "",
            "- Higher `integrated_rms_dbfs` means the scene is louder in the dataset.",
            "- `relative_to_loudest_db = -6` means the scene RMS is about 6 dB lower than the loudest scene.",
            "- Large `p10~p90` spread means that scene contains files with very different loudness.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Analyze TAU scene noise intensity in dBFS.")
    parser.add_argument("--tau_root", default=str(script_dir / "tau"), help="TAU root directory.")
    parser.add_argument("--out_dir", default=str(script_dir), help="Directory for CSV/Markdown outputs.")
    parser.add_argument("--chunk_frames", type=int, default=48000, help="Frames read per chunk.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tau_root = Path(args.tau_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs = discover_wavs(tau_root)
    print(f"[INFO] tau_root={tau_root}")
    print(f"[INFO] usable wavs={len(wavs)}")

    file_rows = []
    for idx, (scene, path) in enumerate(wavs, start=1):
        try:
            stats = read_audio_stats(path, args.chunk_frames)
        except Exception as exc:
            print(f"[WARN] skipped {path}: {exc}")
            continue
        rel_path = path.relative_to(tau_root).as_posix()
        file_rows.append(
            {
                "scene": scene,
                "path": rel_path,
                "sample_rate": stats.sample_rate,
                "channels": stats.channels,
                "frames": stats.frames,
                "duration_sec": stats.duration_sec,
                "sample_count": stats.sample_count,
                "sum_squares": stats.sum_squares,
                "rms": stats.rms,
                "rms_dbfs": stats.rms_dbfs,
                "peak_abs": stats.peak_abs,
                "peak_dbfs": stats.peak_dbfs,
            }
        )
        if idx % 100 == 0:
            print(f"[INFO] processed {idx}/{len(wavs)} wavs")

    scene_rows = summarize_by_scene(file_rows)

    scene_csv = out_dir / "tau_noise_intensity_by_scene.csv"
    file_csv = out_dir / "tau_noise_intensity_by_file.csv"
    report_md = out_dir / "tau_noise_intensity_report.md"

    scene_fields = [
        "loudness_rank",
        "scene",
        "file_count",
        "total_duration_sec",
        "integrated_rms_dbfs",
        "relative_to_loudest_db",
        "mean_file_rms_dbfs",
        "median_file_rms_dbfs",
        "p10_file_rms_dbfs",
        "p90_file_rms_dbfs",
        "min_file_rms_dbfs",
        "max_file_rms_dbfs",
        "peak_dbfs",
        "mean_duration_sec",
    ]
    file_fields = [
        "scene",
        "path",
        "sample_rate",
        "channels",
        "frames",
        "duration_sec",
        "sample_count",
        "sum_squares",
        "rms",
        "rms_dbfs",
        "peak_abs",
        "peak_dbfs",
    ]
    write_csv(scene_csv, scene_rows, scene_fields)
    write_csv(file_csv, file_rows, file_fields)
    write_report(report_md, tau_root, scene_rows, len(file_rows))

    print(f"[INFO] scene summary -> {scene_csv}")
    print(f"[INFO] file stats    -> {file_csv}")
    print(f"[INFO] report        -> {report_md}")
    print("[INFO] Top scenes by integrated RMS dBFS:")
    for row in scene_rows[:10]:
        print(
            f"  {row['loudness_rank']:>2} {row['scene']:<18} "
            f"files={row['file_count']:<5} "
            f"rms={format_float(float(row['integrated_rms_dbfs']), 2):>8} dBFS "
            f"rel={format_float(float(row['relative_to_loudest_db']), 2):>7} dB"
        )


if __name__ == "__main__":
    main()
