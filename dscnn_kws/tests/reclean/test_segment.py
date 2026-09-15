import torch

from dscnn_kws.data.reclean.segment import detect_positive_boundary, split_false_wake_segments


def test_false_wake_split_uses_300ms_silence_and_120ms_context():
    waveform = torch.cat(
        [
            torch.zeros(1600),
            torch.full((3200,), 0.1),
            torch.zeros(4800),
            torch.full((3200,), 0.1),
        ]
    )

    spans = split_false_wake_segments(waveform, sample_rate=16000, min_silence_ms=300, context_ms=120)

    assert [(span.start, span.end) for span in spans] == [(0, 6720), (7680, 12800)]


def test_positive_boundary_exposes_center_and_active_mask():
    waveform = torch.cat([torch.zeros(2400), torch.full((9600,), 0.1), torch.zeros(4000)])

    boundary = detect_positive_boundary(waveform, sample_rate=16000)

    assert boundary.start < 2600
    assert boundary.end > 11800
    assert boundary.center == (boundary.start + boundary.end) // 2
    assert boundary.active_mask.dtype is torch.bool
    assert boundary.active_mask.sum() > 9000
