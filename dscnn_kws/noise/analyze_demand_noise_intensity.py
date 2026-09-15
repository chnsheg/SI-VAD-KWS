from __future__ import annotations

import argparse
import re
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


DEMAND_DOMAIN_BY_PREFIX = {
    "D": "domestic",
    "N": "nature",
    "O": "office",
    "P": "public",
    "S": "street",
    "T": "transport",
}

PREFERRED_ENVIRONMENT_ORDER = [
    "DKITCHEN",
    "DLIVING",
    "DWASHING",
    "NFIELD",
    "NPARK",
    "NRIVER",
    "OHALLWAY",
    "OMEETING",
    "OOFFICE",
    "PCAFETER",
    "PRESTO",
    "PSTATION",
    "SPSQUARE",
    "STRAFFIC",
    "TBUS",
    "TCAR",
    "TMETRO",
]

DEFAULT_SAMPLE_RATE_TAG = "16k"


def parse_demand_dir(path: Path, demand_root: Path) -> tuple[str | None, str | None, str | None]:
    rel_parts = path.relative_to(demand_root).parts
    pattern = re.compile(r"^([A-Z][A-Z0-9]+)(?:_(16k|48k))?$", re.IGNORECASE)
    for part in rel_parts[:-1]:
        match = pattern.match(part)
        if match:
            env = match.group(1).upper()
            sr_tag = (match.group(2) or DEFAULT_SAMPLE_RATE_TAG).lower()
            return env, sr_tag, env
    return None, None, None


def demand_domain(environment: str | None) -> str:
    if not environment:
        return "unknown"
    return DEMAND_DOMAIN_BY_PREFIX.get(environment[0].upper(), "unknown")


def discover_demand_wavs(demand_root: Path) -> list[dict]:
    rows = []
    for path in sorted(demand_root.rglob("*.wav")):
        if not is_usable_wav(path):
            continue
        environment, sample_rate_tag, recording_dir = parse_demand_dir(path, demand_root)
        if recording_dir is None:
            continue
        rows.append(
            {
                "path": path,
                "environment": environment,
                "domain": demand_domain(environment),
                "sample_rate_tag": sample_rate_tag,
                "recording_dir": recording_dir,
            }
        )
    return rows


def collect_file_rows(demand_root: Path, chunk_frames: int) -> list[dict]:
    discovered = discover_demand_wavs(demand_root)
    print(f"[INFO] usable wavs={len(discovered)}")

    file_rows = []
    for idx, item in enumerate(discovered, start=1):
        path = item["path"]
        try:
            stats = read_audio_stats(path, chunk_frames)
        except Exception as exc:
            print(f"[WARN] skipped {path}: {exc}")
            continue
        rel_path = path.relative_to(demand_root).as_posix()
        file_rows.append(
            {
                "group": item["recording_dir"],
                "environment": item["environment"],
                "domain": item["domain"],
                "sample_rate_tag": item["sample_rate_tag"],
                "recording_dir": item["recording_dir"],
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
    parser = argparse.ArgumentParser(description="Analyze DEMAND noise intensity in dBFS.")
    parser.add_argument("--demand_root", default=str(script_dir / "demand"))
    parser.add_argument("--out_dir", default=str(script_dir))
    parser.add_argument("--chunk_frames", type=int, default=48000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demand_root = Path(args.demand_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] demand_root={demand_root}")
    file_rows = collect_file_rows(demand_root, args.chunk_frames)

    discovered_envs = sorted({row["environment"] for row in file_rows})
    environment_order = [
        env for env in PREFERRED_ENVIRONMENT_ORDER if env in discovered_envs
    ] + [
        env for env in discovered_envs if env not in PREFERRED_ENVIRONMENT_ORDER
    ]
    recording_order = environment_order
    domain_order = [
        domain
        for domain in DEMAND_DOMAIN_BY_PREFIX.values()
        if any(row["domain"] == domain for row in file_rows)
    ]
    if any(row["domain"] == "unknown" for row in file_rows):
        domain_order.append("unknown")

    by_recording = summarize_by_group(regroup_file_rows(file_rows, "recording_dir"), recording_order)
    by_environment = summarize_by_group(regroup_file_rows(file_rows, "environment"), environment_order)
    by_domain = summarize_by_group(regroup_file_rows(file_rows, "domain"), domain_order)

    recording_csv = out_dir / "demand_noise_intensity_by_recording_dir.csv"
    environment_csv = out_dir / "demand_noise_intensity_by_environment.csv"
    domain_csv = out_dir / "demand_noise_intensity_by_domain.csv"
    file_csv = out_dir / "demand_noise_intensity_by_file.csv"
    report_md = out_dir / "demand_noise_intensity_report.md"

    extended_file_fields = FILE_FIELDS[:1] + ["domain", "environment", "sample_rate_tag", "recording_dir"] + FILE_FIELDS[1:]
    write_csv(recording_csv, by_recording, GROUP_FIELDS)
    write_csv(environment_csv, by_environment, GROUP_FIELDS)
    write_csv(domain_csv, by_domain, GROUP_FIELDS)
    write_csv(file_csv, file_rows, extended_file_fields)
    write_markdown_report(
        report_md,
        title="DEMAND Noise Intensity Report",
        root=demand_root,
        group_name="recording_dir",
        group_rows=by_recording,
        file_count=len(file_rows),
        extra_lines=[
            "- DEMAND grouping follows directory names such as `DKITCHEN`, `NFIELD`, `OOFFICE`, and `TMETRO`.",
            "- This project uses the 16 kHz DEMAND version only; `sample_rate_tag` is recorded as `16k` for suffix-free directories.",
            "- The parser remains backward-compatible with old names such as `DKITCHEN_16k` if they appear.",
            "- `demand_noise_intensity_by_environment.csv` is equivalent to the environment directory summary for the current suffix-free 16 kHz layout.",
            "- `demand_noise_intensity_by_domain.csv` merges environments by first-letter domain: D=domestic, N=nature, O=office, P=public, S=street, T=transport.",
        ],
    )

    print(f"[INFO] recording summary   -> {recording_csv}")
    print(f"[INFO] environment summary -> {environment_csv}")
    print(f"[INFO] domain summary      -> {domain_csv}")
    print(f"[INFO] file stats          -> {file_csv}")
    print(f"[INFO] report              -> {report_md}")
    print("[INFO] Top recording dirs by integrated RMS dBFS:")
    for row in by_recording[:10]:
        print(
            f"  {str(row['loudness_rank']):>2} {row['group']:<18} "
            f"files={row['file_count']:<5} "
            f"rms={format_float(float(row['integrated_rms_dbfs']), 2):>8} dBFS "
            f"rel={format_float(float(row['relative_to_loudest_db']), 2):>7} dB"
        )


if __name__ == "__main__":
    main()
