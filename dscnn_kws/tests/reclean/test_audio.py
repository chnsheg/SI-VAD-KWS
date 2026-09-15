import numpy as np
import pytest
import soundfile as sf
import torch

from dscnn_kws.data.reclean.audio import canonicalize_wav, mix_at_active_snr


def test_canonicalize_wav_is_one_second_pcm16_mono_16khz(tmp_path):
    source = tmp_path / "source.wav"
    sf.write(
        source,
        np.stack([np.full(44100, 0.5), np.full(44100, -0.5)], axis=1),
        44100,
        subtype="PCM_24",
    )
    output = tmp_path / "output.wav"

    metadata = canonicalize_wav(source, output, sample_rate=16000, sample_length=16000)

    info = sf.info(output)
    assert (info.samplerate, info.channels, info.subtype, info.frames) == (16000, 1, "PCM_16", 16000)
    assert metadata.source_sha256
    assert metadata.output_sha256
    assert metadata.original_sample_rate == 44100
    assert metadata.original_channels == 2


def test_mix_at_active_snr_ignores_silent_padding():
    speech = torch.cat([torch.zeros(4000), torch.full((8000,), 0.1), torch.zeros(4000)]).view(1, -1)
    noise = torch.full_like(speech, 0.1)

    mixed, metadata = mix_at_active_snr(speech, active_span=(4000, 12000), noise=noise, snr_db=-10.0)

    assert mixed.shape == (1, 16000)
    assert metadata.measured_active_snr_db == pytest.approx(-10.0, abs=0.2)
    assert metadata.peak_guard_gain <= 1.0
