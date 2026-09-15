from __future__ import annotations

import csv
import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def format_float(value: float, digits: int = 3) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def is_usable_wav(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() != ".wav":
        return False
    try:
        return path.stat().st_size > 44
    except OSError:
        return False


def summarize_by_group(file_rows: list[dict], group_order: list[str] | None = None) -> list[dict]:
    groups = {}
    for row in file_rows:
        groups.setdefault(row["group"], []).append(row)
    if group_order:
        for group in group_order:
            groups.setdefault(group, [])

    integrated_by_group = {}
    for group, items in groups.items():
        if not items:
            integrated_by_group[group] = float("nan")
            continue
        total_sum_squares = sum(float(item["sum_squares"]) for item in items)
        total_sample_count = sum(int(item["sample_count"]) for item in items)
        integrated_by_group[group] = linear_to_dbfs(math.sqrt(total_sum_squares / max(1, total_sample_count)))

    loudest = max((v for v in integrated_by_group.values() if not math.isnan(v)), default=float("nan"))
    rows = []
    for group, items in groups.items():
        if not items:
            rows.append(
                {
                    "loudness_rank": "",
                    "group": group,
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
        integrated = integrated_by_group[group]
        rows.append(
            {
                "group": group,
                "file_count": len(items),
                "total_duration_sec": sum(float(item["duration_sec"]) for item in items),
                "integrated_rms_dbfs": integrated,
                "relative_to_loudest_db": integrated - loudest if not math.isnan(loudest) else float("nan"),
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
    rank = 1
    for row in rows:
        if int(row["file_count"]) > 0:
            row["loudness_rank"] = rank
            rank += 1
        else:
            row["loudness_rank"] = ""
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown_report(
    path: Path,
    title: str,
    root: Path,
    group_name: str,
    group_rows: list[dict],
    file_count: int,
    extra_lines: list[str] | None = None,
) -> None:
    lines = [
        f"# {title}",
        "",
        f"- Root: `{root}`",
        f"- Usable wav files: `{file_count}`",
        "- Unit: `dBFS`, where 0 dBFS is digital full scale. More negative means quieter.",
        "- `integrated_rms_dbfs` is computed from all samples in the group and is the main group-level intensity metric.",
        "- This is digital signal level, not physical SPL. It is suitable for relative comparison inside the same dataset.",
    ]
    if extra_lines:
        lines.extend(extra_lines)
    lines.extend(
        [
            "",
            f"## {group_name} Summary",
            "",
            f"| rank | {group_name} | files | duration_s | integrated_rms_dbfs | relative_to_loudest_db | median_file_rms_dbfs | p10_file_rms_dbfs | p90_file_rms_dbfs | peak_dbfs |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in group_rows:
        lines.append(
            "| "
            f"{row['loudness_rank']} | "
            f"{row['group']} | "
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
            "## Suggested Reading",
            "",
            "- Higher `integrated_rms_dbfs` means the group is louder in the dataset.",
            "- `relative_to_loudest_db = -6` means the group RMS is about 6 dB lower than the loudest group.",
            "- Large `p10~p90` spread means the group contains files with very different loudness.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


GROUP_FIELDS = [
    "loudness_rank",
    "group",
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

FILE_FIELDS = [
    "group",
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
