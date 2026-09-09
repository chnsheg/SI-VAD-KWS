from typing import TYPE_CHECKING

from .confirmation_pair import CONFIRMATION_PAIR_FORMAT, ConfirmationPairDataset, extract_adjacent_window_pairs
from .dataset import SpeechCommandDataset, build_confirmation_pair_loader, build_dataloaders
from .online_augmentation import (
    NoiseCatalog,
    OnlineAugmentedDataset,
    OnlineWaveformAugmenter,
    SnrBand,
    SnrDistribution,
    StratifiedRoleBatchSampler,
)
from .packed_mixture import CompositePackedDataset, StratifiedCompositeBatchSampler

if TYPE_CHECKING:
    from .build_reclean_training_manifests import TrainingManifestBuildResult, TrainingManifestPaths


def __getattr__(name: str):
    if name in {"TrainingManifestBuildResult", "TrainingManifestPaths", "build_reclean_training_manifests", "resolve_current_training_manifests"}:
        from .build_reclean_training_manifests import (
            TrainingManifestBuildResult,
            TrainingManifestPaths,
            build_reclean_training_manifests,
            resolve_current_training_manifests,
        )

        globals().update(
            {
                "TrainingManifestBuildResult": TrainingManifestBuildResult,
                "TrainingManifestPaths": TrainingManifestPaths,
                "build_reclean_training_manifests": build_reclean_training_manifests,
                "resolve_current_training_manifests": resolve_current_training_manifests,
            }
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "SpeechCommandDataset",
    "CONFIRMATION_PAIR_FORMAT",
    "ConfirmationPairDataset",
    "extract_adjacent_window_pairs",
    "CompositePackedDataset",
    "StratifiedCompositeBatchSampler",
    "NoiseCatalog",
    "OnlineAugmentedDataset",
    "OnlineWaveformAugmenter",
    "SnrBand",
    "SnrDistribution",
    "StratifiedRoleBatchSampler",
    "TrainingManifestBuildResult",
    "TrainingManifestPaths",
    "build_dataloaders",
    "build_confirmation_pair_loader",
    "build_reclean_training_manifests",
    "resolve_current_training_manifests",
]
