import json
import os
import random
from pathlib import Path

random.seed(42)

SCRIPT_DIR = Path(__file__).resolve().parent

RAW_ROOT = SCRIPT_DIR / "mobvoi_hotwords_raw"
RESOURCES_DIR = RAW_ROOT / "mobvoi_hotword_dataset_resources"

# 你的实际 wav 路径：
# data\mobvoi_hotwords_raw\mobvoi_hotword_dataset\mobvoi_hotword_dataset
WAV_DIR = RAW_ROOT / "mobvoi_hotword_dataset" / "mobvoi_hotword_dataset"

KEYWORDS = {
    0: "hi_xiaowen",
    1: "nihao_wenwen",
}

# 训练集负样本比例：1.0 表示 positive : negative = 1 : 1
TRAIN_NEG_TO_POS_RATIO = 1.0

# 如果以后想做更严格任务，可以改成 True：
# hi_xiaowen 任务中，把 nihao_wenwen 也作为 negative；
# nihao_wenwen 任务中，把 hi_xiaowen 也作为 negative。
INCLUDE_OTHER_KEYWORD_AS_NEGATIVE = False


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

    return {p.stem: p for p in wav_files}


def make_record(wav_map, utt_id, command, out_root):
    utt_id = Path(str(utt_id)).stem

    if utt_id not in wav_map:
        print(f"[WARN] Missing wav for utt_id: {utt_id}")
        return None

    wav_path = wav_map[utt_id]

    # 生成相对于当前输出数据集目录的路径，避免复制 17GB 音频
    rel_path = os.path.relpath(wav_path, out_root).replace("\\", "/")

    return {
        "audio_filepath": rel_path,
        "command": command,
    }


def load_split_for_fah(wav_map, part, keyword_id, out_root, is_train):
    pos_json = RESOURCES_DIR / f"p_{part}.json"
    neg_json = RESOURCES_DIR / f"n_{part}.json"

    if not pos_json.exists():
        raise FileNotFoundError(pos_json)

    if not neg_json.exists():
        raise FileNotFoundError(neg_json)

    all_pos_items = read_json(pos_json)
    all_neg_items = read_json(neg_json)

    # 当前关键词 positive
    cur_pos_items = [
        item for item in all_pos_items
        if int(item.get("keyword_id", -1)) == keyword_id
    ]

    # n_*.json 本身就是 non-hotword，全部作为 negative
    neg_items = list(all_neg_items)

    # 可选：把另一个关键词的 positive 也作为 hard negative
    if INCLUDE_OTHER_KEYWORD_AS_NEGATIVE:
        other_pos_items = [
            item for item in all_pos_items
            if int(item.get("keyword_id", -1)) != keyword_id
        ]
        neg_items.extend(other_pos_items)

    positives = []
    negatives = []

    for item in cur_pos_items:
        record = make_record(
            wav_map=wav_map,
            utt_id=item["utt_id"],
            command="positive",
            out_root=out_root,
        )
        if record is not None:
            positives.append(record)

    for item in neg_items:
        record = make_record(
            wav_map=wav_map,
            utt_id=item["utt_id"],
            command="negative",
            out_root=out_root,
        )
        if record is not None:
            negatives.append(record)

    raw_pos = len(positives)
    raw_neg = len(negatives)

    # 只对 train 做负样本下采样，valid/test 保留全部 negative，用于 FAH/FRR 评估
    if is_train and TRAIN_NEG_TO_POS_RATIO is not None:
        max_neg = int(len(positives) * TRAIN_NEG_TO_POS_RATIO)
        if len(negatives) > max_neg:
            negatives = random.sample(negatives, max_neg)

    used_pos = len(positives)
    used_neg = len(negatives)

    records = positives + negatives
    random.shuffle(records)

    return records, raw_pos, raw_neg, used_pos, used_neg


def write_manifest(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    if not RESOURCES_DIR.exists():
        raise FileNotFoundError(f"RESOURCES_DIR not found: {RESOURCES_DIR}")

    wav_map = build_wav_map(WAV_DIR)

    split_map = {
        "train": "train",
        "dev": "validation",
        "test": "test",
    }

    for keyword_id, keyword_name in KEYWORDS.items():
        out_root = SCRIPT_DIR / f"mobvoi_{keyword_name}_binary_fah"
        out_root.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"[INFO] Building FAH/FRR dataset for keyword_id={keyword_id}, keyword={keyword_name}")
        print(f"[INFO] Output root: {out_root}")
        print(f"[INFO] INCLUDE_OTHER_KEYWORD_AS_NEGATIVE={INCLUDE_OTHER_KEYWORD_AS_NEGATIVE}")

        for mobvoi_part, out_split in split_map.items():
            is_train = out_split == "train"

            records, raw_pos, raw_neg, used_pos, used_neg = load_split_for_fah(
                wav_map=wav_map,
                part=mobvoi_part,
                keyword_id=keyword_id,
                out_root=out_root,
                is_train=is_train,
            )

            if out_split == "validation":
                out_path = out_root / "validation_manifest.json"
            else:
                out_path = out_root / f"{out_split}_manifest.json"

            write_manifest(out_path, records)

            hours_neg = used_neg / 3600.0

            print(
                f"[INFO] {keyword_name} | {out_split}: "
                f"raw_positive={raw_pos}, raw_negative={raw_neg}, "
                f"used_positive={used_pos}, used_negative={used_neg}, "
                f"negative_hours≈{hours_neg:.3f}, "
                f"total={len(records)} -> {out_path}"
            )

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()