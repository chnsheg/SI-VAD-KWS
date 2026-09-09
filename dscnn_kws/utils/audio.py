from __future__ import annotations

import json
import os
import random
from collections.abc import Mapping

import torch
import torchaudio


def _normalize_manifest_audio_path(raw_path: str) -> str:
    normalized = os.path.normpath(str(raw_path).strip())
    normalized_slash = normalized.replace("\\", "/")
    if normalized_slash.startswith("./"):
        normalized_slash = normalized_slash[2:]
    legacy_prefix = "TorchKWS/dataset/"
    if normalized_slash.startswith(legacy_prefix):
        normalized_slash = normalized_slash[len(legacy_prefix) :]
    return os.path.normpath(normalized_slash)


def apply_pre_emphasis(x: torch.Tensor, coeff: float = 0.97) -> torch.Tensor:
    if coeff <= 0:
        return x
    if x.dim() == 2:
        x = x.unsqueeze(1)
        squeeze_back = True
    else:
        squeeze_back = False
    y = x.clone()
    y[:, :, 1:] = x[:, :, 1:] - coeff * x[:, :, :-1]
    y[:, :, 0] = x[:, :, 0]
    if squeeze_back:
        y = y.squeeze(1)
    return y


def _reservoir_sample_manifest_rows(manifest_path: str | os.PathLike[str], sample_size: int, rng: random.Random):
    if sample_size < 0:
        raise ValueError("sample_per_split must be non-negative")

    sample = []
    item_count = 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item_count < sample_size:
                sample.append(item)
            elif sample_size > 0:
                replacement_index = rng.randrange(item_count + 1)
                if replacement_index < sample_size:
                    sample[replacement_index] = item
            item_count += 1
            item = None
    return sample


def verify_dataset_sample_rate(
    data_path: str,
    expected_sample_rate: int,
    sample_per_split: int = 80,
    random_seed: int = 42,
    manifest_paths: Mapping[str, str | os.PathLike[str]] | None = None,
    include_test: bool = True,
) -> None:
    rng = random.Random(random_seed)
    default_manifests = {
        "train": os.path.join(data_path, "train_manifest.json"),
        "validation": os.path.join(data_path, "validation_manifest.json"),
        "test": os.path.join(data_path, "test_manifest.json"),
    }
    supplied_manifests = manifest_paths or {}
    manifests = {
        "train": supplied_manifests.get("train") or default_manifests["train"],
        "validation": supplied_manifests.get("validation") or supplied_manifests.get("valid") or default_manifests["validation"],
    }
    if include_test:
        manifests["test"] = supplied_manifests.get("test") or default_manifests["test"]

    def _resolve_audio_path(dataset_root: str, manifest_path: str, rel_or_abs: str) -> str:
        normalized = _normalize_manifest_audio_path(rel_or_abs)
        if os.path.isabs(normalized):
            return normalized
        dataset_root_abs = os.path.abspath(dataset_root)
        dataset_parent = os.path.dirname(dataset_root_abs)
        manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
        basename = os.path.basename(normalized)
        candidates = [
            os.path.normpath(os.path.join(dataset_root_abs, normalized)),
            os.path.normpath(os.path.join(dataset_parent, normalized)),
            os.path.normpath(os.path.join(dataset_root_abs, basename)),
            os.path.normpath(os.path.join(manifest_dir, normalized)),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(
            "Failed to resolve audio path from manifest. "
            f"raw='{rel_or_abs}', normalized='{normalized}', candidates={candidates}"
        )

    for split_name, manifest_path in manifests.items():
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"{split_name} manifest not found: {manifest_path}")

        for item in _reservoir_sample_manifest_rows(manifest_path, sample_per_split, rng):
            fp = _resolve_audio_path(data_path, manifest_path, item["audio_filepath"])
            info = torchaudio.info(fp)
            if int(info.sample_rate) != int(expected_sample_rate):
                raise ValueError(
                    f"Sample-rate mismatch in {split_name}: {fp}, "
                    f"got {info.sample_rate}, expected {expected_sample_rate}"
                )
        item = None
