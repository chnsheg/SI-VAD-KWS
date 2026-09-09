import json
import random
from pathlib import Path

random.seed(42)

# 当前脚本所在目录：
# E:\VSCode\dscnn_kws\dscnn_kws\data
SCRIPT_DIR = Path(__file__).resolve().parent

# 你的数据集目录：
# E:\VSCode\dscnn_kws\dscnn_kws\data\speech_commands_v0.02
DATA_ROOT = SCRIPT_DIR / "speech_commands_v0.02"

TARGET_WORDS = {
    "yes", "no", "up", "down", "left",
    "right", "on", "off", "stop", "go"
}

EXCLUDE_DIRS = {
    "_background_noise_",
    "__pycache__"
}


def read_split_list(filename: str):
    path = DATA_ROOT / filename
    if not path.exists():
        print(f"[WARN] {path} not found.")
        return set()

    with open(path, "r", encoding="utf-8") as f:
        return {
            line.strip().replace("\\", "/")
            for line in f
            if line.strip()
        }


def write_manifest(path: Path, records):
    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_ROOT}")

    print(f"[INFO] DATA_ROOT = {DATA_ROOT}")

    validation_set = read_split_list("validation_list.txt")
    testing_set = read_split_list("testing_list.txt")

    splits = {
        "train": {
            "target": [],
            "unknown": [],
        },
        "validation": {
            "target": [],
            "unknown": [],
        },
        "test": {
            "target": [],
            "unknown": [],
        },
    }

    wav_files = sorted(DATA_ROOT.glob("*/*.wav"))
    print(f"[INFO] Found {len(wav_files)} wav files.")

    for wav_path in wav_files:
        word = wav_path.parent.name

        if word in EXCLUDE_DIRS:
            continue

        rel_path = wav_path.relative_to(DATA_ROOT).as_posix()

        if rel_path in validation_set:
            split = "validation"
        elif rel_path in testing_set:
            split = "test"
        else:
            split = "train"

        if word in TARGET_WORDS:
            command = word
            splits[split]["target"].append({
                "audio_filepath": rel_path,
                "command": command,
            })
        else:
            command = "unknown"
            splits[split]["unknown"].append({
                "audio_filepath": rel_path,
                "command": command,
            })

    bg_files = sorted((DATA_ROOT / "_background_noise_").glob("*.wav"))
    if not bg_files:
        raise FileNotFoundError(
            f"No background noise wav found in: {DATA_ROOT / '_background_noise_'}"
        )

    # silence 样本在 dataset.py 中会直接返回全 0，
    # 但 manifest 里仍然保留一个合法 wav 路径作为占位。
    silence_placeholder = bg_files[0].relative_to(DATA_ROOT).as_posix()

    for split_name, data in splits.items():
        target_records = data["target"]
        unknown_records = data["unknown"]

        # 10 个关键词的平均样本数，用来决定 unknown 和 silence 数量
        avg_target_per_class = max(1, len(target_records) // len(TARGET_WORDS))

        sampled_unknown = random.sample(
            unknown_records,
            k=min(avg_target_per_class, len(unknown_records))
        )

        silence_records = [
            {
                "audio_filepath": silence_placeholder,
                "command": "silence",
            }
            for _ in range(avg_target_per_class)
        ]

        final_records = target_records + sampled_unknown + silence_records
        random.shuffle(final_records)

        if split_name == "train":
            out_path = DATA_ROOT / "train_manifest.json"
        elif split_name == "validation":
            out_path = DATA_ROOT / "validation_manifest.json"
        else:
            out_path = DATA_ROOT / "test_manifest.json"

        write_manifest(out_path, final_records)

        print(
            f"[INFO] {split_name}: "
            f"target={len(target_records)}, "
            f"unknown={len(sampled_unknown)}, "
            f"silence={len(silence_records)}, "
            f"total={len(final_records)} -> {out_path}"
        )

    print("[INFO] Manifest generation done.")


if __name__ == "__main__":
    main()