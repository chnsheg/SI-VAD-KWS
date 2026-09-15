from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from dscnn_kws.data.reclean.catalog import (
    build_rir_catalog,
    normalize_rir_catalog,
    select_rirs_noises_impulse_paths,
)


def test_rir_catalog_rejects_empty_impulse(tmp_path):
    path = tmp_path / "empty.wav"
    sf.write(path, np.zeros(16000), 16000)

    assert build_rir_catalog([path]) == []


def test_rir_catalog_normalizes_direct_peak(tmp_path):
    path = tmp_path / "rir.wav"
    sf.write(path, np.array([0.0, 1.0, 0.5]), 16000)

    item = build_rir_catalog([path])[0]

    assert item.direct_peak_index == 1
    assert float(item.waveform.abs().max()) == pytest.approx(1.0)


def test_normalized_rir_catalog_is_pcm16_mono_16khz(tmp_path):
    source = tmp_path / "rir_8k.wav"
    sf.write(source, np.array([0.0, 1.0, 0.5]), 8000)

    rows = normalize_rir_catalog([source], tmp_path / "normalized")
    info = sf.info(rows[0].normalized_path)

    assert (info.samplerate, info.channels, info.subtype) == (16000, 1, "PCM_16")
    assert (tmp_path / "normalized" / "rir_catalog.json").is_file()


def test_rirs_noises_selection_excludes_pointsource_noise():
    root = Path("/archive/RIRS_NOISES")
    noise = root / "pointsource_noises" / "noise.wav"
    real = root / "real_rirs_isotropic_noises" / "real.wav"
    simulated = root / "simulated_rirs" / "smallroom" / "Room001" / "simulated.wav"

    assert select_rirs_noises_impulse_paths([noise, real, simulated]) == [real, simulated]
