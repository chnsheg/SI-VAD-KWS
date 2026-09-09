import json
import os
import random
from pathlib import Path

random.seed(42)

SCRIPT_DIR = Path(__file__).resolve().parent

RAW_ROOT = SCRIPT_DIR / "mobvoi_hotwords_raw"

RESOURCES_DIR = RAW_ROOT / "mobvoi_hotword_dataset_resources"

# 你的实际音频路径：
# E:\VSCode\dscnn_kws\dscnn_kws\data\mobvoi_hotwords_raw\mobvoi_hotword_dataset\mobvoi_hotword_dataset
WAV_DIR = RAW_ROOT / "mobvoi_hotword_dataset" / "mobvoi_hotword_dataset"

# 输出给 DSCNN 使用的二分类数据集
OUT_ROOT = SCRIPT_DIR / "mobvoi_hotwords_binary"

# 1.0 表示正负样本 1:1 平衡
# 如果想保留更多负样本，可以改成 2.0 / 3.0 / None
NEG_TO_POS_RATIO = 1.0


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_wav_map(wav_dir: Path):
    print(f"[INFO] Scanning wavs in: {wav_dir}")

    if not wav_dir.exists():
        raise FileNotFoundError(f"WAV_DIR not found: {wav_dir}")

    wav_files = sorted(wav_dir.rglob("*.wav"))
    print(f"[INFO] Found {len(wav_files)} wav files.")

    if len(wav_files) == 0:
        raise FileNotFoundError(f"No wav files found in: {wav_dir}")

    wav_map = {}
    for p in wav_files:
        wav_map[p.stem] = p

    return wav_map


def make_record(wav_map, utt_id, command):
    utt_id = Path(str(utt_id)).stem

    if utt_id not in wav_map:
        print(f"[WARN] Missing wav for utt_id: {utt_id}")
        return None

    wav_path = wav_map[utt_id]

    # 生成相对于 mobvoi_hotwords_binary 的路径
    # 这样不用复制整个 17GB 音频数据集
    rel_path = os.path.relpath(wav_path, OUT_ROOT).replace("\\", "/")

    return {
        "audio_filepath": rel_path,
        "command": command,
    }


def load_split(wav_map, part):
    pos_json = RESOURCES_DIR / f"p_{part}.json"
    neg_json = RESOURCES_DIR / f"n_{part}.json"

    if not pos_json.exists():
        raise FileNotFoundError(pos_json)

    if not neg_json.exists():
        raise FileNotFoundError(neg_json)

    pos_items = read_json(pos_json)
    neg_items = read_json(neg_json)

    positives = []
    negatives = []

    for item in pos_items:
        record = make_record(wav_map, item["utt_id"], "positive")
        if record is not None:
            positives.append(record)

    for item in neg_items:
        record = make_record(wav_map, item["utt_id"], "negative")
        if record is not None:
            negatives.append(record)

    raw_pos = len(positives)
    raw_neg = len(negatives)

    if NEG_TO_POS_RATIO is not None:
        max_neg = int(len(positives) * NEG_TO_POS_RATIO)
        if len(negatives) > max_neg:
            negatives = random.sample(negatives, max_neg)

    records = positives + negatives
    random.shuffle(records)

    return records, raw_pos, raw_neg, len(positives), len(negatives)


def write_manifest(path: Path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    if not RESOURCES_DIR.exists():
        raise FileNotFoundError(f"RESOURCES_DIR not found: {RESOURCES_DIR}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    wav_map = build_wav_map(WAV_DIR)

    split_map = {
        "train": "train",
        "dev": "validation",
        "test": "test",
    }

    for mobvoi_part, out_split in split_map.items():
        records, raw_pos, raw_neg, used_pos, used_neg = load_split(
            wav_map,
            mobvoi_part,
        )

        if out_split == "validation":
            out_path = OUT_ROOT / "validation_manifest.json"
        else:
            out_path = OUT_ROOT / f"{out_split}_manifest.json"

        write_manifest(out_path, records)

        print(
            f"[INFO] {out_split}: "
            f"raw_positive={raw_pos}, raw_negative={raw_neg}, "
            f"used_positive={used_pos}, used_negative={used_neg}, "
            f"total={len(records)} -> {out_path}"
        )

    print(f"[INFO] Done. Output dataset: {OUT_ROOT}")


if __name__ == "__main__":
    main()