from __future__ import annotations

import argparse
from pathlib import Path

from noise_intensity_common import (
    FILE_FIELDS,
    GROUP_FIELDS,
    format_float,
    is_usable_wav,
    read_audio_stats,
    summarize_by_group,
    write_csv,
    write_markdown_report,
)


TOP_CATEGORIES = ["noise", "music", "speech"]


def parse_musan_group(path: Path, musan_root: Path) -> tuple[str | None, str | None, str | None]:
    rel_parts = path.relative_to(musan_root).parts
    if len(rel_parts) < 2:
        return None, None, None
    category = rel_parts[0]
    if category not in TOP_CATEGORIES:
        return None, None, None
    source = rel_parts[1] if len(rel_parts) >= 3 else "_root"
    return category, source, f"{category}/{source}"


def discover_musan_wavs(musan_root: Path) -> list[dict]:
    rows = []
    for path in sorted(musan_root.rglob("*.wav")):
        if not is_usable_wav(path):
            continue
        category, source, category_source = parse_musan_group(path, musan_root)
        if category is None:
            continue
        rows.append(
            {
                "path": path,
                "category": category,
                "source": source,
                "category_source": category_source,
            }
        )
    return rows


def collect_file_rows(musan_root: Path, chunk_frames: int) -> list[dict]:
    discovered = discover_musan_wavs(musan_root)
    print(f"[INFO] usable wavs={len(discovered)}")

    file_rows = []
    for idx, item in enumerate(discovered, start=1):
        path = item["path"]
        try:
            stats = read_audio_stats(path, chunk_frames)
        except Exception as exc:
            print(f"[WARN] skipped {path}: {exc}")
            continue
        rel_path = path.relative_to(musan_root).as_posix()
        file_rows.append(
            {
                "group": item["category_source"],
                "category": item["category"],
                "source": item["source"],
                "category_source": item["category_source"],
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
            print(f"[INFO] processed {idx}/{len(discovered)} wavs")
    return file_rows


def regroup_file_rows(file_rows: list[dict], key: str) -> list[dict]:
    rows = []
    for row in file_rows:
        new_row = dict(row)
        new_row["group"] = row[key]
        rows.append(new_row)
    return rows


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Analyze MUSAN noise/music/speech intensity in dBFS.")
    parser.add_argument("--musan_root", default=str(script_dir / "musan"))
    parser.add_argument("--out_dir", default=str(script_dir))
    parser.add_argument("--chunk_frames", type=int, default=48000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    musan_root = Path(args.musan_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] musan_root={musan_root}")
    file_rows = collect_file_rows(musan_root, args.chunk_frames)

    category_source_order = sorted({row["category_source"] for row in file_rows})
    by_category_source = summarize_by_group(regroup_file_rows(file_rows, "category_source"), category_source_order)
    by_category = summarize_by_group(regroup_file_rows(file_rows, "category"), TOP_CATEGORIES)

    category_source_csv = out_dir / "musan_noise_intensity_by_category_source.csv"
    category_csv = out_dir / "musan_noise_intensity_by_category.csv"
    file_csv = out_dir / "musan_noise_intensity_by_file.csv"
    report_md = out_dir / "musan_noise_intensity_report.md"

    extended_file_fields = FILE_FIELDS[:1] + ["category", "source", "category_source"] + FILE_FIELDS[1:]
    write_csv(category_source_csv, by_category_source, GROUP_FIELDS)
    write_csv(category_csv, by_category, GROUP_FIELDS)
    write_csv(file_csv, file_rows, extended_file_fields)
    write_markdown_report(
        report_md,
        title="MUSAN Noise Intensity Report",
        root=musan_root,
        group_name="category/source",
        group_rows=by_category_source,
        file_count=len(file_rows),
        extra_lines=[
            "- MUSAN top-level categories are `noise`, `music`, and `speech`.",
            "- `musan_noise_intensity_by_category.csv` merges source subdirectories into the three top-level categories.",
            "- For KWS noise augmentation, `noise` is ordinary background noise; `speech` and `music` are useful harder interference conditions.",
        ],
    )

    print(f"[INFO] category/source summary -> {category_source_csv}")
    print(f"[INFO] category summary        -> {category_csv}")
    print(f"[INFO] file stats              -> {file_csv}")
    print(f"[INFO] report                  -> {report_md}")
    print("[INFO] Top category/source groups by integrated RMS dBFS:")
    for row in by_category_source[:10]:
        print(
            f"  {str(row['loudness_rank']):>2} {row['group']:<24} "
            f"files={row['file_count']:<5} "
            f"rms={format_float(float(row['integrated_rms_dbfs']), 2):>8} dBFS "
            f"rel={format_float(float(row['relative_to_loudest_db']), 2):>7} dB"
        )


if __name__ == "__main__":
    main()
