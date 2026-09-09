import json
import os
import random
from pathlib import Path

random.seed(42)

SCRIPT_DIR = Path(__file__).resolve().parent

RAW_ROOT = SCRIPT_DIR / "mobvoi_hotwords_raw"
RESOURCES_DIR = RAW_ROOT / "mobvoi_hotword_dataset_resources"

WAV_DIR = RAW_ROOT / "mobvoi_hotword_dataset" / "mobvoi_hotword_dataset"

KEYWORDS = {
    0: "hi_xiaowen",
    1: "nihao_wenwen",
}

# 训练集负样本比例。1.0 表示 positive : negative = 1 : 1
TRAIN_NEG_TO_POS_RATIO = 1.0


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

    rel_path = os.path.relpath(wav_path, out_root).replace("\\", "/")

    return {
        "audio_filepath": rel_path,
        "command": command,
    }


def build_records_for_keyword(
    wav_map,
    part,
    keyword_id,
    out_root,
    balance_eval_negative,
):
    """
    balance_eval_negative=True:
        train/valid/test 全部正负 1:1，用于 ACC/F1

    balance_eval_negative=False:
        train 正负 1:1，valid/test 保留全部 negative，用于 FAH/FRR
    """
    pos_json = RESOURCES_DIR / f"p_{part}.json"
    neg_json = RESOURCES_DIR / f"n_{part}.json"

    if not pos_json.exists():
        raise FileNotFoundError(pos_json)

    if not neg_json.exists():
        raise FileNotFoundError(neg_json)

    all_pos_items = read_json(pos_json)
    all_neg_items = read_json(neg_json)

    cur_pos_items = [
        item for item in all_pos_items
        if int(item.get("keyword_id", -1)) == keyword_id
    ]

    other_pos_items = [
        item for item in all_pos_items
        if int(item.get("keyword_id", -1)) != keyword_id
    ]

    # hard negative = non-hotword + 另一个 wake word
    neg_items = list(all_neg_items) + list(other_pos_items)

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

    is_train = part == "train"

    # ACC 版：train/dev/test 都 balance
    # FAH 版：只 balance train，dev/test 保留全部 negative
    should_balance_negative = is_train or balance_eval_negative

    if should_balance_negative and TRAIN_NEG_TO_POS_RATIO is not None:
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


def build_dataset_family(wav_map, suffix, balance_eval_negative):
    split_map = {
        "train": "train",
        "dev": "validation",
        "test": "test",
    }

    for keyword_id, keyword_name in KEYWORDS.items():
        out_root = SCRIPT_DIR / f"mobvoi_{keyword_name}_binary_{suffix}"
        out_root.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"[INFO] Building hard-negative dataset: {out_root}")
        print(f"[INFO] keyword_id={keyword_id}, keyword={keyword_name}")
        print(f"[INFO] balance_eval_negative={balance_eval_negative}")

        for mobvoi_part, out_split in split_map.items():
            records, raw_pos, raw_neg, used_pos, used_neg = build_records_for_keyword(
                wav_map=wav_map,
                part=mobvoi_part,
                keyword_id=keyword_id,
                out_root=out_root,
                balance_eval_negative=balance_eval_negative,
            )

            if out_split == "validation":
                out_path = out_root / "validation_manifest.json"
            else:
                out_path = out_root / f"{out_split}_manifest.json"

            write_manifest(out_path, records)

            neg_hours = used_neg / 3600.0

            print(
                f"[INFO] {keyword_name} | {out_split}: "
                f"raw_positive={raw_pos}, raw_negative={raw_neg}, "
                f"used_positive={used_pos}, used_negative={used_neg}, "
                f"negative_hours≈{neg_hours:.3f}, "
                f"total={len(records)} -> {out_path}"
            )


def main():
    if not RESOURCES_DIR.exists():
        raise FileNotFoundError(f"RESOURCES_DIR not found: {RESOURCES_DIR}")

    wav_map = build_wav_map(WAV_DIR)

    # 1. ACC/F1 用：train/valid/test 都正负 1:1
    build_dataset_family(
        wav_map=wav_map,
        suffix="hardneg",
        balance_eval_negative=True,
    )

    # 2. FAH/FRR 用：train 1:1，valid/test 保留全部 negative
    build_dataset_family(
        wav_map=wav_map,
        suffix="fah_hardneg",
        balance_eval_negative=False,
    )

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()