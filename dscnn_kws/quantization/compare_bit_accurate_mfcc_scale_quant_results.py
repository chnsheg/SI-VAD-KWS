from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


KEY_COLUMNS = ["dataset", "arch", "scene", "snr_db"]
METRIC_COLUMNS = ["acc", "precision", "recall", "f1"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] csv saved to: {path.resolve()}")


def key_for(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row.get(col, "")) for col in KEY_COLUMNS)


def to_float(value: str | float | int | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def compare_rows(
    baseline_rows: list[dict[str, str]],
    quantized_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    baseline_by_key = {key_for(row): row for row in baseline_rows}
    quantized_by_key = {key_for(row): row for row in quantized_rows}
    common_keys = sorted(set(baseline_by_key) & set(quantized_by_key))
    if not common_keys:
        raise ValueError("No common dataset/arch/scene/snr rows found between baseline and quantized CSVs.")

    rows: list[dict[str, Any]] = []
    for key in common_keys:
        base = baseline_by_key[key]
        quant = quantized_by_key[key]
        out: dict[str, Any] = {
            "dataset": key[0],
            "arch": key[1],
            "scene": key[2],
            "snr_db": key[3],
            "baseline_frontend": base.get("frontend", ""),
            "quantized_frontend": quant.get("frontend", ""),
            "baseline_spec": base.get("bit_accurate_mfcc_spec_json", ""),
            "scale_quantized_spec": quant.get("scale_u16_bit_accurate_mfcc_spec_json")
            or quant.get("bit_accurate_mfcc_spec_json", ""),
            "baseline_inference_checkpoint": base.get("inference_checkpoint", ""),
            "quantized_inference_checkpoint": quant.get("inference_checkpoint", ""),
            "scale_bits": quant.get("scale_bits", ""),
            "scale_quant_mode": quant.get("scale_quant_mode", ""),
            "num_samples": quant.get("num_samples", base.get("num_samples", "")),
        }
        for metric in METRIC_COLUMNS:
            baseline_value = to_float(base.get(metric))
            quantized_value = to_float(quant.get(metric))
            delta = quantized_value - baseline_value
            out[f"baseline_{metric}"] = baseline_value
            out[f"scale_u16_{metric}"] = quantized_value
            out[f"{metric}_delta"] = delta
            out[f"{metric}_delta_pp"] = delta * 100.0
        rows.append(out)
    return rows


def summarize(rows: list[dict[str, Any]], group_cols: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[col] for col in group_cols)
        groups[key].append(row)

    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda item: item[0]):
        summary = {col: value for col, value in zip(group_cols, key)}
        summary["rows"] = len(items)
        for metric in METRIC_COLUMNS:
            deltas = [float(item[f"{metric}_delta_pp"]) for item in items]
            baseline_values = [float(item[f"baseline_{metric}"]) * 100.0 for item in items]
            quantized_values = [float(item[f"scale_u16_{metric}"]) * 100.0 for item in items]
            summary[f"baseline_{metric}_mean_pct"] = statistics.fmean(baseline_values)
            summary[f"scale_u16_{metric}_mean_pct"] = statistics.fmean(quantized_values)
            summary[f"{metric}_delta_mean_pp"] = statistics.fmean(deltas)
            summary[f"{metric}_delta_min_pp"] = min(deltas)
            summary[f"{metric}_delta_max_pp"] = max(deltas)
        out.append(summary)
    return out


def format_pp(value: float) -> str:
    return f"{value:+.4f} pp"


def write_markdown_report(
    *,
    compare_rows_out: list[dict[str, Any]],
    overall: list[dict[str, Any]],
    by_dataset: list[dict[str, Any]],
    by_scene: list[dict[str, Any]],
    by_snr: list[dict[str, Any]],
    path: Path,
) -> None:
    total = overall[0]
    mean_acc_delta = float(total["acc_delta_mean_pp"])
    mean_f1_delta = float(total["f1_delta_mean_pp"])
    if abs(mean_acc_delta) < 0.05 and abs(mean_f1_delta) < 0.05:
        conclusion = "scale U16 表示误差对当前噪声场景精度基本没有可见影响。"
    elif abs(mean_acc_delta) < 0.20 and abs(mean_f1_delta) < 0.20:
        conclusion = "scale U16 带来轻微变化，建议查看具体 dataset/scene/SNR 是否集中掉点。"
    else:
        conclusion = "scale U16 可能带来明显掉点，需要进一步定位 stage scale 误差或饱和率变化。"

    lines = [
        "# Bit-Accurate MFCC Scale U16 Quantization Analysis",
        "",
        "## 结论",
        "",
        conclusion,
        "",
        "## 总体结果",
        "",
        f"- 对齐 grid 行数: {len(compare_rows_out)}",
        f"- baseline mean acc: {float(total['baseline_acc_mean_pct']):.4f}%",
        f"- scale-u16 mean acc: {float(total['scale_u16_acc_mean_pct']):.4f}%",
        f"- mean acc delta: {format_pp(mean_acc_delta)}",
        f"- baseline mean f1: {float(total['baseline_f1_mean_pct']):.4f}%",
        f"- scale-u16 mean f1: {float(total['scale_u16_f1_mean_pct']):.4f}%",
        f"- mean f1 delta: {format_pp(mean_f1_delta)}",
        "",
        "## 按数据集",
        "",
        "| dataset | rows | acc delta | f1 delta |",
        "|---|---:|---:|---:|",
    ]
    for row in by_dataset:
        lines.append(
            f"| {row['dataset']} | {row['rows']} | "
            f"{format_pp(float(row['acc_delta_mean_pp']))} | {format_pp(float(row['f1_delta_mean_pp']))} |"
        )

    lines.extend(
        [
            "",
            "## 按 SNR",
            "",
            "| snr_db | rows | acc delta | f1 delta |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(by_snr, key=lambda item: float(item["snr_db"])):
        lines.append(
            f"| {row['snr_db']} | {row['rows']} | "
            f"{format_pp(float(row['acc_delta_mean_pp']))} | {format_pp(float(row['f1_delta_mean_pp']))} |"
        )

    worst = sorted(compare_rows_out, key=lambda row: float(row["acc_delta_pp"]))[:10]
    lines.extend(
        [
            "",
            "## 最明显掉点的单项",
            "",
            "| dataset | scene | snr_db | acc delta | f1 delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in worst:
        lines.append(
            f"| {row['dataset']} | {row['scene']} | {row['snr_db']} | "
            f"{format_pp(float(row['acc_delta_pp']))} | {format_pp(float(row['f1_delta_pp']))} |"
        )

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 本报告只比较 observer/data scale 从 float 变为 U16 multiplier + shift 后的影响。",
            "- Hann/twiddle/DCT 固定系数和 PWL-log 参数没有在本轮实验中改变。",
            "- 这一版仍使用 Python `q = round(x / scale_q)` 路径，不是完整 RTL `acc * multiplier >> shift` 流水线仿真。",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] markdown saved to: {path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare scale-U16 bit-accurate MFCC grid results against v2 baseline.")
    parser.add_argument(
        "--baseline_csv",
        default="dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_coeff_quant/hardware_coeff_baseline_grid_results.csv",
    )
    parser.add_argument(
        "--quantized_csv",
        default="dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_scale_u16/scale_u16_grid_results.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="dscnn_kws/quantization/bit_accurate_mfcc_experiments_v2_scale_u16",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    baseline_rows = read_csv(Path(args.baseline_csv))
    quantized_rows = read_csv(Path(args.quantized_csv))
    compared = compare_rows(baseline_rows, quantized_rows)

    overall = summarize(compared, [])
    by_dataset = summarize(compared, ["dataset"])
    by_scene = summarize(compared, ["scene"])
    by_snr = summarize(compared, ["snr_db"])

    write_csv(compared, output_dir / "scale_u16_compare_vs_v2_coeff_quant.csv")
    write_csv(overall, output_dir / "scale_u16_summary_overall.csv")
    write_csv(by_dataset, output_dir / "scale_u16_summary_by_dataset.csv")
    write_csv(by_scene, output_dir / "scale_u16_summary_by_scene.csv")
    write_csv(by_snr, output_dir / "scale_u16_summary_by_snr.csv")
    write_markdown_report(
        compare_rows_out=compared,
        overall=overall,
        by_dataset=by_dataset,
        by_scene=by_scene,
        by_snr=by_snr,
        path=output_dir / "scale_u16_analysis_zh.md",
    )
    print(f"[DONE] compared_rows={len(compared)}")


if __name__ == "__main__":
    main()
