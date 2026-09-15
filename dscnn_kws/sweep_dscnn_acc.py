import csv
import os
import re
import shutil
import subprocess
import sys
import zipfile
from html import escape
from pathlib import Path


ROOT = r".\dscnn_kws\data"

DATASETS = [
    "mobvoi_hi_xiaowen_binary_hardneg",
    "mobvoi_nihao_wenwen_binary_hardneg",
]

EPOCH = 30
BATCH = 128
SAMPLE_RATE = 16000
GPU = 0
NUM_WORKERS = 0
DCT_COEFF = 10
WINDOW_SIZE_MS = 32
WINDOW_STRIDE_MS = 32
NUM_CLASSES = 2

OUT_CSV = Path("sweep_dscnn_acc_results.csv")
OUT_XLSX = Path("sweep_dscnn_acc_results.xlsx")
OUT_SHAPES_TXT = Path("sweep_dscnn_shapes.txt")
BEST_MODEL_DIR = Path("dscnn_kws") / "runs" / "sweep_best_models"

# 从大到小 sweep
ARCHS = [
    ("L5_C64", 5, 64),
    ("L5_C48", 5, 48),
    ("L5_C32", 5, 32),
    ("L5_C24", 5, 24),
    ("L5_C16", 5, 16),
    ("L4_C16", 4, 16),
    ("L3_C16", 3, 16),
    ("L5_C12", 5, 12),
    ("L4_C12", 4, 12),
    ("L3_C12", 3, 12),
    ("L5_C8", 5, 8),
    ("L4_C8", 4, 8),
    ("L3_C8", 3, 8),
    ("L2_C16", 2, 16),
    ("L2_C12", 2, 12),
    ("L2_C8", 2, 8),
    ("L5_C6", 5, 6),
    ("L4_C6", 4, 6),
    ("L3_C6", 3, 6),
    ("L2_C6", 2, 6),
    ("L1_C8", 1, 8),
    ("L3_C4", 3, 4),
    ("L2_C4", 2, 4),
    ("L1_C6", 1, 6),
    ("L1_C4", 1, 4),
]


def make_model_size_info(num_layers: int, channels: int):
    info = [num_layers]

    # 第一层普通 Conv2d
    info += [channels, 10, 4, 2, 2]

    # 后续 depthwise separable conv
    for _ in range(num_layers - 1):
        info += [channels, 3, 3, 1, 1]

    return info


def expected_params(num_layers: int, channels: int, num_classes: int = 2):
    """
    当前 DSCNN 二分类参数量近似/实际公式：
    Conv0: 40C + BN 2C
    每个 DS block: C^2 + 9C + BN 4C = C^2 + 13C
    FC: C*num_classes + num_classes
    """
    c = channels
    n = num_layers
    return (n - 1) * c * c + (42 + 13 * (n - 1) + num_classes) * c + num_classes


def calculate_time_steps(sample_rate: int, window_stride_ms: int, audio_duration_ms: int = 1000) -> int:
    stride_samples = int(sample_rate * window_stride_ms / 1000)
    audio_samples = int(sample_rate * audio_duration_ms / 1000)
    if stride_samples <= 0:
        return 1
    return audio_samples // stride_samples + 1


def conv_out_size(size: int, kernel: int, stride: int) -> int:
    padding = kernel // 2
    return (size + 2 * padding - kernel) // stride + 1


def shape_trace(num_layers: int, channels: int):
    time_steps = calculate_time_steps(SAMPLE_RATE, WINDOW_STRIDE_MS)
    rows = [
        ("waveform", f"[B, {SAMPLE_RATE}]"),
        ("mfcc_raw", f"[B, 40, {time_steps}]"),
        ("mfcc_selected", f"[B, {DCT_COEFF}, {time_steps}]"),
        ("flatten_for_backbone", f"[B, {time_steps * DCT_COEFF}]"),
        ("dscnn_input", f"[B, 1, {time_steps}, {DCT_COEFF}]"),
    ]

    t = time_steps
    f = DCT_COEFF
    in_c = 1
    for i in range(num_layers):
        if i == 0:
            kt, kw, st, sw = 10, 4, 2, 2
            op = "conv2d"
        else:
            kt, kw, st, sw = 3, 3, 1, 1
            op = "depthwise_separable_conv2d"
        t = conv_out_size(t, kt, st)
        f = conv_out_size(f, kw, sw)
        rows.append((f"layer{i + 1}_{op}", f"[B, {channels}, {t}, {f}]"))
        in_c = channels

    rows.extend(
        [
            ("adaptive_avg_pool", f"[B, {in_c}, 1, 1]"),
            ("flatten", f"[B, {in_c}]"),
            ("logits", f"[B, {NUM_CLASSES}]"),
        ]
    )
    return rows


def write_shape_report(results):
    lines = [
        "DSCNN sweep shape report",
        f"MFCC selected output: {DCT_COEFF} x {calculate_time_steps(SAMPLE_RATE, WINDOW_STRIDE_MS)}",
        f"sample_rate={SAMPLE_RATE}, window_size_ms={WINDOW_SIZE_MS}, window_stride_ms={WINDOW_STRIDE_MS}",
        "",
    ]

    seen = set()
    for arch_name, num_layers, channels in ARCHS:
        key = (arch_name, num_layers, channels)
        if key in seen:
            continue
        seen.add(key)

        lines.append("=" * 88)
        lines.append(
            f"{arch_name} | layers={num_layers}, channels={channels}, "
            f"params={expected_params(num_layers, channels, NUM_CLASSES)}"
        )
        for stage, shape in shape_trace(num_layers, channels):
            lines.append(f"  {stage:<32} -> {shape}")
        lines.append("")

    if results:
        lines.append("=" * 88)
        lines.append("Saved best model files")
        for r in results:
            if r.get("best_model_saved_as"):
                lines.append(f"  {r['dataset']} | {r['arch']} -> {r['best_model_saved_as']}")
        lines.append("")

    OUT_SHAPES_TXT.write_text("\n".join(lines), encoding="utf-8")


def write_xlsx(rows, fieldnames, out_path: Path):
    def col_name(idx: int) -> str:
        name = ""
        while idx:
            idx, rem = divmod(idx - 1, 26)
            name = chr(65 + rem) + name
        return name

    data = [fieldnames] + [[row.get(field) for field in fieldnames] for row in rows]
    sheet_rows = []
    for row_idx, row in enumerate(data, start=1):
        cells = []
        for col_idx, value in enumerate(row, start=1):
            ref = f"{col_name(col_idx)}{row_idx}"
            if value is None:
                cells.append(f'<c r="{ref}"/>')
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>')
        sheet_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    worksheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheet_rows)}</sheetData>'
        '</worksheet>'
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="sweep_results" sheetId="1" r:id="rId1"/></sheets>'
        '</workbook>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '</Relationships>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '</Types>'
    )

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/worksheets/sheet1.xml", worksheet)


def copy_best_model(output: str, dataset: str, arch_name: str, num_layers: int, channels: int):
    m = re.search(r"\[INFO\]\s+save_dir=(.+)", output)
    if not m:
        return None, None

    save_dir = Path(m.group(1).strip())
    best_path = save_dir / "best.pt"
    if not best_path.exists():
        return str(save_dir), None

    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    params = expected_params(num_layers, channels, NUM_CLASSES)
    safe_dataset = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset)
    out_path = BEST_MODEL_DIR / (
        f"{safe_dataset}_{arch_name}_layers{num_layers}_channels{channels}_"
        f"params{params}_mfcc{DCT_COEFF}x{calculate_time_steps(SAMPLE_RATE, WINDOW_STRIDE_MS)}_best.pt"
    )
    shutil.copy2(best_path, out_path)
    return str(save_dir), str(out_path)


def run_one(dataset: str, arch_name: str, num_layers: int, channels: int):
    msi = make_model_size_info(num_layers, channels)

    cmd = [
        sys.executable,
        "-m",
        "dscnn_kws.train",
        "--root",
        ROOT,
        "--dataset",
        dataset,
        "--sample_rate",
        str(SAMPLE_RATE),
        "--gpu",
        str(GPU),
        "--num_workers",
        str(NUM_WORKERS),
        "--epoch",
        str(EPOCH),
        "--batch",
        str(BATCH),
        "--dct_coeff",
        str(DCT_COEFF),
        "--window_size_ms",
        str(WINDOW_SIZE_MS),
        "--window_stride_ms",
        str(WINDOW_STRIDE_MS),
        "--allow_online_resample",
        "--no-verify_sample_rate",
        "--no-noise_aug",
        "--no-eval_noise_aug",
        "--model_size_info",
        *[str(x) for x in msi],
    ]

    print("\n" + "=" * 100)
    print(f"[RUN] dataset={dataset}, arch={arch_name}, layers={num_layers}, channels={channels}")
    print(f"[RUN] MFCC selected output={DCT_COEFF}x{calculate_time_steps(SAMPLE_RATE, WINDOW_STRIDE_MS)}")
    print("[CMD]", " ".join(cmd))

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )

    output_chunks = []
    assert proc.stdout is not None
    while True:
        chunk = proc.stdout.read(1)
        if chunk == "" and proc.poll() is not None:
            break
        if not chunk:
            continue
        print(chunk, end="", flush=True)
        output_chunks.append(chunk)
    proc.wait()
    output = "".join(output_chunks)

    params = None
    test_loss = None
    test_acc = None
    precision = None
    recall = None
    f1 = None
    best_valid_acc = None

    m = re.search(r"params=(\d+)", output)
    if m:
        params = int(m.group(1))

    valid_accs = [float(x) for x in re.findall(r"valid_loss\s+[0-9.]+\s+acc\s+([0-9.]+)", output)]
    if valid_accs:
        best_valid_acc = max(valid_accs)

    m = re.search(
        r"\[TEST\]\s+loss=([0-9.]+)\s+acc=([0-9.]+)\s+precision=([0-9.]+)\s+recall=([0-9.]+)\s+f1=([0-9.]+)",
        output,
    )
    if m:
        test_loss = float(m.group(1))
        test_acc = float(m.group(2))
        precision = float(m.group(3))
        recall = float(m.group(4))
        f1 = float(m.group(5))

    save_dir, best_model_saved_as = copy_best_model(output, dataset, arch_name, num_layers, channels)
    if best_model_saved_as:
        print(f"[INFO] best model copied to: {best_model_saved_as}")
    else:
        print("[WARN] best model was not copied; save_dir/best.pt was not found")

    return {
        "dataset": dataset,
        "arch": arch_name,
        "layers": num_layers,
        "channels": channels,
        "mfcc_output": f"{DCT_COEFF}x{calculate_time_steps(SAMPLE_RATE, WINDOW_STRIDE_MS)}",
        "expected_params": expected_params(num_layers, channels),
        "printed_params": params,
        "best_valid_acc": best_valid_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "returncode": proc.returncode,
        "train_save_dir": save_dir,
        "best_model_saved_as": best_model_saved_as,
    }


def summarize(results):
    fieldnames = [
        "dataset",
        "arch",
        "layers",
        "channels",
        "mfcc_output",
        "expected_params",
        "printed_params",
        "best_valid_acc",
        "test_loss",
        "test_acc",
        "precision",
        "recall",
        "f1",
        "returncode",
        "train_save_dir",
        "best_model_saved_as",
    ]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    write_xlsx(results, fieldnames, OUT_XLSX)
    write_shape_report(results)

    print("\n" + "=" * 100)
    print(f"[INFO] CSV saved to: {OUT_CSV.resolve()}")
    print(f"[INFO] Excel saved to: {OUT_XLSX.resolve()}")
    print(f"[INFO] Shape report saved to: {OUT_SHAPES_TXT.resolve()}")

    # 单关键词阈值统计
    thresholds = [0.92, 0.95, 0.98]

    for dataset in DATASETS:
        print("\n" + "=" * 100)
        print(f"[SUMMARY] {dataset}")

        ds_results = [
            r for r in results
            if r["dataset"] == dataset and r["test_acc"] is not None
        ]

        ds_results = sorted(ds_results, key=lambda x: x["expected_params"])

        for th in thresholds:
            candidates = [r for r in ds_results if r["test_acc"] >= th]
            if candidates:
                best = candidates[0]
                print(
                    f"acc >= {th:.2%}: "
                    f"{best['arch']} | params={best['expected_params']} | "
                    f"test_acc={best['test_acc']:.4f}"
                )
            else:
                print(f"acc >= {th:.2%}: no candidate found")

    # 两个关键词都满足阈值
    print("\n" + "=" * 100)
    print("[SUMMARY] both keywords must satisfy threshold")

    # 按 arch 汇总
    by_arch = {}
    for r in results:
        if r["test_acc"] is None:
            continue
        by_arch.setdefault(r["arch"], []).append(r)

    for th in thresholds:
        candidates = []
        for arch_name, rows in by_arch.items():
            if len(rows) != len(DATASETS):
                continue

            min_acc = min(r["test_acc"] for r in rows)
            params = rows[0]["expected_params"]

            if min_acc >= th:
                candidates.append((params, arch_name, min_acc, rows))

        candidates.sort(key=lambda x: x[0])

        if candidates:
            params, arch_name, min_acc, rows = candidates[0]
            print(
                f"both acc >= {th:.2%}: "
                f"{arch_name} | params={params} | min_acc={min_acc:.4f}"
            )
            for r in rows:
                print(f"    {r['dataset']}: acc={r['test_acc']:.4f}")
        else:
            print(f"both acc >= {th:.2%}: no candidate found")


def main():
    results = []

    for arch_name, num_layers, channels in ARCHS:
        for dataset in DATASETS:
            result = run_one(dataset, arch_name, num_layers, channels)
            results.append(result)

            # 每跑完一个就保存一次，避免中途停止丢结果
            summarize(results)

    summarize(results)


if __name__ == "__main__":
    main()
