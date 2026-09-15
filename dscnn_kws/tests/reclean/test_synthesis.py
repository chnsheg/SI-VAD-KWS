from dataclasses import replace

import pytest
import torch

from dscnn_kws.data.reclean.recipes import build_recipe
from dscnn_kws.data.reclean.synthesis import Boundary, crop_positive, peak_guard, set_active_rms


def test_positive_jitter_keeps_label_when_word_exceeds_window():
    recipe = replace(build_recipe(42, "train", "source", "hash", 0), speed=0.9, jitter_ms=100)

    _, metadata = crop_positive(torch.ones(1, 24000), Boundary(0, 24000), recipe)

    assert metadata.label == "positive"
    assert metadata.coverage_ratio < 1.0


def test_peak_guard_preserves_one_second_shape():
    waveform = torch.full((1, 16000), 2.0)

    guarded, gain, _ = peak_guard(waveform)

    assert guarded.shape == (1, 16000)
    assert gain < 1.0
    assert float(guarded.abs().max()) <= 0.99


def test_set_active_rms_uses_only_the_visible_target_interval():
    waveform = torch.cat([torch.zeros(4000), torch.ones(8000) * 0.1, torch.zeros(4000)]).view(1, -1)

    scaled, measured = set_active_rms(waveform, (4000, 12000), -20.0)

    assert measured == pytest.approx(-20.0, abs=1e-4)
    assert torch.allclose(scaled[:, :4000], torch.zeros(1, 4000))
