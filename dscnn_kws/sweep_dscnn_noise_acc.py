import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import zipfile
from html import escape
from pathlib import Path


ROOT = r"./dscnn_kws/data"

DATASETS = [
    "mobvoi_hi_xiaowen_binary_hardneg",
    "mobvoi_nihao_wenwen_binary_hardneg",
]

def split_path_env(value: str | None, default: list[str]) -> list[str]:
    if value is None or value.strip() == "":
        return list(default)
    return [item for item in value.split(os.pathsep) if item]


DEFAULT_TRAIN_NOISE_ROOTS = ["./dscnn_kws/noise/lists/tau_train.txt"]
DEFAULT_VALID_NOISE_ROOTS = ["./dscnn_kws/noise/lists/tau_valid.txt"]
DEFAULT_TEST_NOISE_ROOTS = ["./dscnn_kws/noise/lists/tau_test.txt"]

# Override on Linux if needed:
#   TAU_NOISE_ROOTS=./dscnn_kws/noise/tau/airport python dscnn_kws/sweep_dscnn_noise_acc.py
# or split train/valid/test explicitly:
#   TAU_TRAIN_NOISE_ROOTS=... TAU_VALID_NOISE_ROOTS=... TAU_TEST_NOISE_ROOTS=... python dscnn_kws/sweep_dscnn_noise_acc.py
shared_noise_roots = os.environ.get("TAU_NOISE_ROOTS")
TRAIN_NOISE_ROOTS = split_path_env(
    os.environ.get("TAU_TRAIN_NOISE_ROOTS"),
    split_path_env(shared_noise_roots, DEFAULT_TRAIN_NOISE_ROOTS),
)
VALID_NOISE_ROOTS = split_path_env(
    os.environ.get("TAU_VALID_NOISE_ROOTS"),
    split_path_env(shared_noise_roots, DEFAULT_VALID_NOISE_ROOTS),
)
TEST_NOISE_ROOTS = split_path_env(
    os.environ.get("TAU_TEST_NOISE_ROOTS"),
    split_path_env(shared_noise_roots, DEFAULT_TEST_NOISE_ROOTS),
)

EPOCH = 30
BATCH = 128
SAMPLE_RATE = 16000
GPU = 1
NUM_WORKERS = 8
DCT_COEFF = 10
WINDOW_SIZE_MS = 32
WINDOW_STRIDE_MS = 32
NUM_CLASSES = 2

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

TRAIN_NOISE_PROB = 0.8
TRAIN_SNR_MIN_DB = -5.0
TRAIN_SNR_MAX_DB = 20.0
EVAL_NOISE_PROB = 1.0
EVAL_SNR_MIN_DB = 0.0
EVAL_SNR_MAX_DB = 20.0

OUT_CSV = Path("sweep_dscnn_noise_acc_results.csv")
OUT_XLSX = Path("sweep_dscnn_noise_acc_results.xlsx")
OUT_SCENE_CSV = Path("sweep_dscnn_noise_scene_results.csv")
OUT_SCENE_XLSX = Path("sweep_dscnn_noise_scene_results.xlsx")
OUT_SHAPES_TXT = Path("sweep_dscnn_noise_shapes.txt")
BEST_MODEL_DIR = Path("dscnn_kws") / "runs" / "sweep_noise_best_models"

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
    info += [channels, 10, 4, 2, 2]
    for _ in range(num_layers - 1):
        info += [channels, 3, 3, 1, 1]
    return info


def expected_params(num_layers: int, channels: int, num_classes: int = 2):
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


def count_usable_noise_files(noise_roots):
    count = 0
    for raw_root in noise_roots:
        root = Path(raw_root)
        if root.is_file():
            base = root.parent
            for line in root.read_text(encoding="utf-8").splitlines():
                item = line.strip()
                if not item or item.startswith("#"):
                    continue
                path = Path(item)
                if not path.is_absolute():
                    path = base / path
                if path.suffix.lower() == ".wav" and path.exists() and path.stat().st_size > 44:
                    count += 1
        elif root.is_dir():
            count += sum(1 for p in root.rglob("*.wav") if p.stat().st_size > 44)
    return count


def load_state_dict(path, device):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def build_scene_eval_loader(data_path: str, noise_roots: list[str], seed: int):
    from torch.utils.data import DataLoader

    from dscnn_kws.configs import CLASS_ENCODING, CLASS_LIST
    from dscnn_kws.data.dataset import SpeechCommandDataset

    dataset = SpeechCommandDataset(
        dataset_path=data_path,
        json_filename=os.path.join(data_path, "test_manifest.json"),
        is_training=False,
        class_list=CLASS_LIST,
        class_encoding=CLASS_ENCODING,
        sample_rate=SAMPLE_RATE,
        noise_aug=True,
        noise_roots=noise_roots,
        noise_prob=EVAL_NOISE_PROB,
        noise_snr_min_db=EVAL_SNR_MIN_DB,
        noise_snr_max_db=EVAL_SNR_MAX_DB,
        deterministic_noise=True,
        random_seed=seed,
        allow_online_resample=True,
        strict_sample_rate=False,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH,
        shuffle=False,
        drop_last=False,
        num_workers=max(0, NUM_WORKERS // 2),
        pin_memory=GPU > 0,
    )


def build_scene_eval_model(model_size_info: list[int], ckpt: str, device):
    from dscnn_kws.configs import CLASS_LIST
    from dscnn_kws.model import DSCNN
    from dscnn_kws.model.dscnn import calculate_time_steps as model_calculate_time_steps
    from dscnn_kws.train import MFCCDSCNN

    time_steps = model_calculate_time_steps(SAMPLE_RATE, WINDOW_STRIDE_MS)
    input_dim = time_steps * DCT_COEFF
    backbone = DSCNN(
        input_dim=input_dim,
        label_count=len(CLASS_LIST),
        model_size_info=model_size_info,
        dct_coeff=DCT_COEFF,
    )
    model = MFCCDSCNN(
        backbone=backbone,
        frontend="mfcc",
        sample_rate=SAMPLE_RATE,
        dct_coeff=DCT_COEFF,
        window_size_ms=WINDOW_SIZE_MS,
        window_stride_ms=WINDOW_STRIDE_MS,
        bandpass_n_bands=16,
        bandpass_f_min=200.0,
        bandpass_f_max=4000.0,
        bandpass_spacing="log",
        bandpass_kernel_size=63,
        bandpass_phase_count=1,
        pre_emphasis=True,
        pre_emphasis_coeff=0.97,
        spec_aug=False,
        spec_aug_freq_mask_param=1,
        spec_aug_time_mask_param=1,
        spec_aug_num_freq_masks=0,
        spec_aug_num_time_masks=0,
        mfcc_impl="torchaudio",
        mel_filter_shape="triangular",
        log_approx_mode="exact",
        log_pwl_num_segments=6,
        log_pwl_strategy="uniform_logx",
        log_pwl_gamma=1.0,
        log_pwl_breakpoints=None,
        log_pwl_slopes=None,
        log_pwl_intercepts=None,
        log_offset=1e-6,
        log_input_clamp_min=1e-12,
    ).to(device)
    model.load_state_dict(load_state_dict(ckpt, device))
    model.eval()
    return model


def eval_scene_acc(model, loader, device) -> dict:
    import torch
    from sklearn.metrics import f1_score, precision_score, recall_score

    total = 0
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for waveform, labels in loader:
            waveform = waveform.to(device)
            labels = labels.to(device)
            logits = model(waveform)
            preds = torch.argmax(logits, dim=1)
            total += int(labels.size(0))
            correct += int((preds == labels).sum().item())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    return {
        "test_acc": correct / max(1, total),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "num_samples": total,
    }


def run_scene_tests(
    dataset: str,
    arch_name: str,
    num_layers: int,
    channels: int,
    model_size_info: list[int],
    ckpt: str,
    scene_test_root: str,
    scene_names: list[str],
):
    import torch

    device = torch.device("cuda" if GPU > 0 and torch.cuda.is_available() else "cpu")
    data_path = os.path.join(ROOT, dataset)
    model = build_scene_eval_model(model_size_info, ckpt, device)
    rows = []

    for scene_idx, scene in enumerate(scene_names):
        scene_root = os.path.join(scene_test_root, scene)
        noise_roots = [scene_root]
        usable = count_usable_noise_files(noise_roots)
        if usable <= 0:
            print(f"[WARN] scene skipped, no usable wavs: {scene_root}")
            continue
        loader = build_scene_eval_loader(
            data_path=data_path,
            noise_roots=noise_roots,
            seed=300000 + scene_idx * 10007,
        )
        metrics = eval_scene_acc(model, loader, device)
        row = {
            "dataset": dataset,
            "arch": arch_name,
            "layers": num_layers,
            "channels": channels,
            "scene": scene,
            "scene_noise_root": scene_root,
            "usable_noise_files": usable,
            "eval_noise_prob": EVAL_NOISE_PROB,
            "eval_snr_min_db": EVAL_SNR_MIN_DB,
            "eval_snr_max_db": EVAL_SNR_MAX_DB,
            "expected_params": expected_params(num_layers, channels),
            **metrics,
            "ckpt": ckpt,
        }
        rows.append(row)
        print(
            f"[SCENE TEST] {dataset} | {arch_name} | {scene}: "
            f"acc={metrics['test_acc']:.4f}, f1={metrics['f1']:.4f}, noise_files={usable}"
        )
    return rows


def write_shape_report(results):
    lines = [
        "DSCNN noise sweep shape report",
        f"train_noise_roots={TRAIN_NOISE_ROOTS}",
        f"valid_noise_roots={VALID_NOISE_ROOTS}",
        f"test_noise_roots={TEST_NOISE_ROOTS}",
        f"train_noise_prob={TRAIN_NOISE_PROB}, train_snr=[{TRAIN_SNR_MIN_DB}, {TRAIN_SNR_MAX_DB}] dB",
        f"eval_noise_prob={EVAL_NOISE_PROB}, eval_snr=[{EVAL_SNR_MIN_DB}, {EVAL_SNR_MAX_DB}] dB",
        f"MFCC selected output: {DCT_COEFF} x {calculate_time_steps(SAMPLE_RATE, WINDOW_STRIDE_MS)}",
        f"sample_rate={SAMPLE_RATE}, window_size_ms={WINDOW_SIZE_MS}, window_stride_ms={WINDOW_STRIDE_MS}",
        "",
    ]

    for arch_name, num_layers, channels in ARCHS:
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
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships" Target="xl/workbook.xml"/>'
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
        f"{safe_dataset}_{arch_name}_noise_tau_layers{num_layers}_channels{channels}_"
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
        "--noise_aug",
        "--eval_noise_aug",
        "--train_noise_roots",
        *TRAIN_NOISE_ROOTS,
        "--valid_noise_roots",
        *VALID_NOISE_ROOTS,
        "--test_noise_roots",
        *TEST_NOISE_ROOTS,
        "--noise_aug_prob",
        str(TRAIN_NOISE_PROB),
        "--noise_snr_min_db",
        str(TRAIN_SNR_MIN_DB),
        "--noise_snr_max_db",
        str(TRAIN_SNR_MAX_DB),
        "--eval_noise_aug_prob",
        str(EVAL_NOISE_PROB),
        "--eval_noise_snr_min_db",
        str(EVAL_SNR_MIN_DB),
        "--eval_noise_snr_max_db",
        str(EVAL_SNR_MAX_DB),
        "--model_size_info",
        *[str(x) for x in msi],
    ]

    print("\n" + "=" * 100)
    print(f"[RUN] dataset={dataset}, arch={arch_name}, layers={num_layers}, channels={channels}")
    print(f"[RUN] train_noise_roots={TRAIN_NOISE_ROOTS}")
    print(f"[RUN] valid_noise_roots={VALID_NOISE_ROOTS}")
    print(f"[RUN] test_noise_roots={TEST_NOISE_ROOTS}")
    print(
        f"[RUN] train_noise_prob={TRAIN_NOISE_PROB}, train_snr=[{TRAIN_SNR_MIN_DB}, {TRAIN_SNR_MAX_DB}] dB | "
        f"eval_noise_prob={EVAL_NOISE_PROB}, eval_snr=[{EVAL_SNR_MIN_DB}, {EVAL_SNR_MAX_DB}] dB"
    )
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
        "train_noise_roots": ";".join(TRAIN_NOISE_ROOTS),
        "valid_noise_roots": ";".join(VALID_NOISE_ROOTS),
        "test_noise_roots": ";".join(TEST_NOISE_ROOTS),
        "train_noise_prob": TRAIN_NOISE_PROB,
        "train_snr_min_db": TRAIN_SNR_MIN_DB,
        "train_snr_max_db": TRAIN_SNR_MAX_DB,
        "eval_noise_prob": EVAL_NOISE_PROB,
        "eval_snr_min_db": EVAL_SNR_MIN_DB,
        "eval_snr_max_db": EVAL_SNR_MAX_DB,
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
        "train_noise_roots",
        "valid_noise_roots",
        "test_noise_roots",
        "train_noise_prob",
        "train_snr_min_db",
        "train_snr_max_db",
        "eval_noise_prob",
        "eval_snr_min_db",
        "eval_snr_max_db",
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

    print("\n" + "=" * 100)
    print("[SUMMARY] both keywords must satisfy threshold")

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


def summarize_scene_results(scene_results):
    if not scene_results:
        return

    fieldnames = [
        "dataset",
        "arch",
        "layers",
        "channels",
        "scene",
        "scene_noise_root",
        "usable_noise_files",
        "eval_noise_prob",
        "eval_snr_min_db",
        "eval_snr_max_db",
        "expected_params",
        "test_acc",
        "precision",
        "recall",
        "f1",
        "num_samples",
        "ckpt",
    ]

    with open(OUT_SCENE_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scene_results)

    write_xlsx(scene_results, fieldnames, OUT_SCENE_XLSX)
    print(f"[INFO] Scene CSV saved to: {OUT_SCENE_CSV.resolve()}")
    print(f"[INFO] Scene Excel saved to: {OUT_SCENE_XLSX.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep DSCNN architectures with online TAU noise augmentation.")
    parser.add_argument(
        "--noise_roots",
        nargs="+",
        default=None,
        help="Use the same noise root/list for train, validation, and test.",
    )
    parser.add_argument(
        "--train_noise_roots",
        nargs="+",
        default=None,
        help="Noise directories or txt lists for training.",
    )
    parser.add_argument(
        "--valid_noise_roots",
        nargs="+",
        default=None,
        help="Noise directories or txt lists for validation.",
    )
    parser.add_argument(
        "--test_noise_roots",
        nargs="+",
        default=None,
        help="Noise directories or txt lists for testing.",
    )
    parser.add_argument(
        "--per_scene_test",
        action="store_true",
        help="After each training run, evaluate the best checkpoint on each TAU scene separately.",
    )
    parser.add_argument(
        "--scene_test_root",
        default="./dscnn_kws/noise/tau",
        help="Root that contains TAU scene subdirectories such as airport, bus, metro.",
    )
    parser.add_argument(
        "--scene_names",
        nargs="+",
        default=TAU_SCENES,
        help="TAU scene subdirectory names to test when --per_scene_test is enabled.",
    )
    return parser.parse_args()


def apply_cli_overrides(args):
    global TRAIN_NOISE_ROOTS, VALID_NOISE_ROOTS, TEST_NOISE_ROOTS

    if args.noise_roots:
        TRAIN_NOISE_ROOTS = list(args.noise_roots)
        VALID_NOISE_ROOTS = list(args.noise_roots)
        TEST_NOISE_ROOTS = list(args.noise_roots)

    if args.train_noise_roots:
        TRAIN_NOISE_ROOTS = list(args.train_noise_roots)
    if args.valid_noise_roots:
        VALID_NOISE_ROOTS = list(args.valid_noise_roots)
    if args.test_noise_roots:
        TEST_NOISE_ROOTS = list(args.test_noise_roots)


def validate_noise_roots(name: str, roots: list[str]) -> int:
    count = count_usable_noise_files(roots)
    print(f"[INFO] {name}={roots}")
    print(f"[INFO] {name}_usable_noise_files={count}")
    if count <= 0:
        raise FileNotFoundError(
            f"No usable wav files found for {name}={roots}. "
            "Use a TAU directory or a txt list generated by make_tau_split_lists.py."
        )
    return count


def main():
    args = parse_args()
    apply_cli_overrides(args)

    validate_noise_roots("TRAIN_NOISE_ROOTS", TRAIN_NOISE_ROOTS)
    validate_noise_roots("VALID_NOISE_ROOTS", VALID_NOISE_ROOTS)
    validate_noise_roots("TEST_NOISE_ROOTS", TEST_NOISE_ROOTS)

    results = []
    scene_results = []
    for arch_name, num_layers, channels in ARCHS:
        for dataset in DATASETS:
            result = run_one(dataset, arch_name, num_layers, channels)
            results.append(result)
            summarize(results)

            if args.per_scene_test and result.get("best_model_saved_as"):
                msi = make_model_size_info(num_layers, channels)
                scene_rows = run_scene_tests(
                    dataset=dataset,
                    arch_name=arch_name,
                    num_layers=num_layers,
                    channels=channels,
                    model_size_info=msi,
                    ckpt=result["best_model_saved_as"],
                    scene_test_root=args.scene_test_root,
                    scene_names=args.scene_names,
                )
                scene_results.extend(scene_rows)
                summarize_scene_results(scene_results)

    summarize(results)
    summarize_scene_results(scene_results)


if __name__ == "__main__":
    main()
