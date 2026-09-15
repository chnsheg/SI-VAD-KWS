from __future__ import annotations

import argparse
import os
import random
from pathlib import Path


SCENES = [
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


def infer_scene(path: Path, tau_root: Path) -> str | None:
    rel_parts = path.relative_to(tau_root).parts
    for part in rel_parts[:-1]:
        if part in SCENES:
            return part

    stem = path.stem
    for scene in sorted(SCENES, key=len, reverse=True):
        if stem == scene or stem.startswith(scene + "-") or stem.startswith(scene + "_"):
            return scene
    return None


def discover_wavs(tau_root: Path) -> dict[str, list[Path]]:
    by_scene = {scene: [] for scene in SCENES}
    for path in sorted(tau_root.rglob("*.wav")):
        if not path.is_file():
            continue
        try:
            if path.stat().st_size <= 44:
                continue
        except OSError:
            continue
        scene = infer_scene(path, tau_root)
        if scene is not None:
            by_scene[scene].append(path)
    return by_scene


def split_scene_files(
    files: list[Path],
    train_ratio: float,
    valid_ratio: float,
    rng: random.Random,
) -> tuple[list[Path], list[Path], list[Path]]:
    files = list(files)
    rng.shuffle(files)
    n = len(files)
    if n == 0:
        return [], [], []

    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    if n >= 3:
        n_train = max(1, n_train)
        n_valid = max(1, n_valid)
        if n_train + n_valid >= n:
            n_train = max(1, n - 2)
            n_valid = 1
    elif n == 2:
        n_train = 1
        n_valid = 0
    else:
        n_train = 1
        n_valid = 0

    train = files[:n_train]
    valid = files[n_train : n_train + n_valid]
    test = files[n_train + n_valid :]
    return train, valid, test


def rel_to_list(path: Path, list_path: Path) -> str:
    return os.path.relpath(path.resolve(), list_path.parent.resolve()).replace(os.sep, "/")


def write_list(path: Path, files: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [rel_to_list(file_path, path) for file_path in sorted(files)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Create stratified 80/10/10 TAU wav split lists.")
    parser.add_argument("--tau_root", default=str(script_dir / "tau"))
    parser.add_argument("--out_dir", default=str(script_dir / "lists"))
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tau_root = Path(args.tau_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    rng = random.Random(args.seed)

    if args.train_ratio <= 0 or args.valid_ratio < 0 or args.train_ratio + args.valid_ratio >= 1:
        raise ValueError("Require train_ratio > 0, valid_ratio >= 0, and train_ratio + valid_ratio < 1")

    by_scene = discover_wavs(tau_root)
    train_all: list[Path] = []
    valid_all: list[Path] = []
    test_all: list[Path] = []

    print(f"[INFO] tau_root={tau_root}")
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] ratios: train={args.train_ratio}, valid={args.valid_ratio}, test={1 - args.train_ratio - args.valid_ratio:.3f}")

    for scene in SCENES:
        files = by_scene[scene]
        train, valid, test = split_scene_files(files, args.train_ratio, args.valid_ratio, rng)
        train_all.extend(train)
        valid_all.extend(valid)
        test_all.extend(test)
        print(
            f"[SCENE] {scene:<18} total={len(files):<5} "
            f"train={len(train):<5} valid={len(valid):<5} test={len(test):<5}"
        )

    train_path = out_dir / "tau_train.txt"
    valid_path = out_dir / "tau_valid.txt"
    test_path = out_dir / "tau_test.txt"
    write_list(train_path, train_all)
    write_list(valid_path, valid_all)
    write_list(test_path, test_all)

    print("[INFO] totals:")
    print(f"  train={len(train_all)} -> {train_path}")
    print(f"  valid={len(valid_all)} -> {valid_path}")
    print(f"  test ={len(test_all)} -> {test_path}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
