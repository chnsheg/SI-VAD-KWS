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

# keyword_id 对应关系
KEYWORDS = {
    0: "hi_xiaowen",
    1: "nihao_wenwen",
}

# 1.0 表示 positive : negative = 1 : 1
# 如果想保留全部负样本，改成 None
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


def make_record(wav_map, utt_id, command, out_root):
    utt_id = Path(str(utt_id)).stem

    if utt_id not in wav_map:
        print(f"[WARN] Missing wav for utt_id: {utt_id}")
        return None

    wav_path = wav_map[utt_id]

    # 生成相对于当前输出数据集目录的路径
    # 例如：
    # ../mobvoi_hotwords_raw/mobvoi_hotword_dataset/mobvoi_hotword_dataset/xxx.wav
    rel_path = os.path.relpath(wav_path, out_root).replace("\\", "/")

    return {
        "audio_filepath": rel_path,
        "command": command,
    }


def load_keyword_split(wav_map, part, keyword_id, out_root):
    pos_json = RESOURCES_DIR / f"p_{part}.json"
    neg_json = RESOURCES_DIR / f"n_{part}.json"

    if not pos_json.exists():
        raise FileNotFoundError(pos_json)

    if not neg_json.exists():
        raise FileNotFoundError(neg_json)

    pos_items = read_json(pos_json)
    neg_items = read_json(neg_json)

    # 只筛选 positive 里的 keyword_id
    # keyword_id=0 -> hi_xiaowen
    # keyword_id=1 -> nihao_wenwen
    pos_items = [
        item for item in pos_items
        if int(item.get("keyword_id", -1)) == keyword_id
    ]

    # 注意：negative 不筛选 keyword_id
    # 因为 n_train / n_dev / n_test 里的 keyword_id 全部是 -1
    # 它们本身就是非唤醒词 negative
    positives = []
    negatives = []

    for item in pos_items:
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

    if NEG_TO_POS_RATIO is not None:
        max_neg = int(len(positives) * NEG_TO_POS_RATIO)
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
        out_root = SCRIPT_DIR / f"mobvoi_{keyword_name}_binary"
        out_root.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"[INFO] Building dataset for keyword_id={keyword_id}, keyword={keyword_name}")
        print(f"[INFO] Output root: {out_root}")

        for mobvoi_part, out_split in split_map.items():
            records, raw_pos, raw_neg, used_pos, used_neg = load_keyword_split(
                wav_map=wav_map,
                part=mobvoi_part,
                keyword_id=keyword_id,
                out_root=out_root,
            )

            if out_split == "validation":
                out_path = out_root / "validation_manifest.json"
            else:
                out_path = out_root / f"{out_split}_manifest.json"

            write_manifest(out_path, records)

            print(
                f"[INFO] {keyword_name} | {out_split}: "
                f"raw_positive={raw_pos}, raw_negative={raw_neg}, "
                f"used_positive={used_pos}, used_negative={used_neg}, "
                f"total={len(records)} -> {out_path}"
            )

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()