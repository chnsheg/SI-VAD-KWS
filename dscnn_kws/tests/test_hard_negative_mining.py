from __future__ import annotations

from dscnn_kws.data import hard_negative_mining


def _candidate(source_sha256: str, start_sample: int):
    return hard_negative_mining.MiningCandidate(
        audio_path=f"/{source_sha256}.wav",
        source_split="train",
        source_role="false_wake",
        start_sample=start_sample,
        duration_seconds=1.0,
        positive_score=0.0,
        source_sha256=source_sha256,
    )


def test_dense_false_wake_candidates_use_full_96ms_windows_without_tail_padding() -> None:
    function = getattr(hard_negative_mining, "dense_false_wake_candidates", None)
    assert function is not None, "dense V3 false-wake candidate enumeration is not implemented"

    candidates = function(
        [
            {
                "audio_path": "/yc.wav",
                "source_split": "train",
                "source_role": "false_wake",
                "source_sha256": "yc-hash",
                "frames": 32_100,
            }
        ],
        hop_samples=1536,
    )

    assert [candidate.start_sample for candidate in candidates] == list(range(0, 16_101, 1536))
    assert all(candidate.duration_seconds == 1.0 for candidate in candidates)
    assert all(candidate.start_sample + 16_000 <= 32_100 for candidate in candidates)


def test_source_balanced_selection_prevents_a_long_source_monopoly() -> None:
    function = getattr(hard_negative_mining, "select_source_balanced", None)
    assert function is not None, "source-balanced V3 selection is not implemented"
    candidates = [
        *[_candidate("long", start) for start in range(0, 100 * 1536, 1536)],
        *[_candidate("short", start) for start in range(0, 4 * 1536, 1536)],
    ]

    selected = function(candidates, target_count=8, seed=3)

    assert len(selected) == 8
    assert {candidate.source_sha256 for candidate in selected} == {"long", "short"}
    assert all(candidate.source_split == "train" for candidate in selected)
