from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SegmentSpan:
    """A half-open source interval, measured in samples."""

    start: int
    end: int


@dataclass(frozen=True)
class PositiveBoundary:
    """Detected speech extent and the sample-level detector decision."""

    start: int
    end: int
    center: int
    active_mask: torch.Tensor
    threshold: float
    noise_floor: float
    speech_level: float
    audit_status: str


def _as_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim == 1:
        return waveform.to(torch.float32)
    if waveform.ndim == 2:
        return waveform.to(torch.float32).mean(dim=0)
    raise ValueError(f"Expected [samples] or [channels, samples], got {tuple(waveform.shape)}")


def _frame_rms(samples: torch.Tensor, frame_samples: int, hop_samples: int) -> torch.Tensor:
    if samples.numel() == 0:
        return samples.new_empty(0)
    padded = F.pad(samples.square().view(1, 1, -1), (0, max(0, frame_samples - samples.numel())))
    return torch.sqrt(F.avg_pool1d(padded, frame_samples, hop_samples, ceil_mode=True).squeeze() + 1e-12)


def _bridge_short_gaps(active_mask: torch.Tensor, max_gap_samples: int) -> torch.Tensor:
    """Join intra-phoneme gaps without moving the outer speech boundaries."""
    if max_gap_samples <= 0 or not bool(active_mask.any()):
        return active_mask

    bridged = active_mask.clone()
    active_indices = torch.nonzero(active_mask, as_tuple=False).flatten().tolist()
    for left, right in zip(active_indices, active_indices[1:]):
        if 0 < right - left - 1 <= max_gap_samples:
            bridged[left + 1 : right] = True
    return bridged


def _adaptive_active_mask(waveform: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, float, float, float]:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    samples = _as_mono(waveform)
    if samples.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=samples.device), 0.0, 0.0, 0.0

    magnitude = samples.abs()
    frame_energy = _frame_rms(samples, max(1, round(sample_rate * 0.030)), max(1, round(sample_rate * 0.010)))
    noise_floor = float(torch.quantile(frame_energy, 0.20).item())
    speech_level = float(torch.quantile(frame_energy, 0.90).item())
    peak = float(magnitude.max().item())
    if peak < 1e-5 or speech_level < 1e-5:
        return torch.zeros_like(magnitude, dtype=torch.bool), 0.0, noise_floor, speech_level

    # The sample-level threshold preserves exact cut points.  Frame energy makes the
    # threshold adapt to the recording while the lower off threshold is hysteretic.
    threshold = max(1e-5, speech_level * (0.10 if noise_floor >= speech_level * 0.5 else 0.15))
    release_threshold = threshold * 0.65
    active = torch.zeros_like(magnitude, dtype=torch.bool)
    is_active = False
    for index, value in enumerate(magnitude.tolist()):
        if is_active:
            is_active = value >= release_threshold
        else:
            is_active = value >= threshold
        active[index] = is_active

    return _bridge_short_gaps(active, max(1, round(sample_rate * 0.030))), threshold, noise_floor, speech_level


def _spans_from_mask(active_mask: torch.Tensor) -> list[SegmentSpan]:
    indices = torch.nonzero(active_mask, as_tuple=False).flatten().tolist()
    if not indices:
        return []

    spans: list[SegmentSpan] = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            spans.append(SegmentSpan(start, previous + 1))
            start = index
        previous = index
    spans.append(SegmentSpan(start, previous + 1))
    return spans


def split_false_wake_segments(
    waveform: torch.Tensor,
    sample_rate: int,
    min_silence_ms: int = 300,
    context_ms: int = 120,
) -> list[SegmentSpan]:
    """Cut a continuous false-wake recording at sufficiently long silent gaps."""
    if min_silence_ms < 0 or context_ms < 0:
        raise ValueError("min_silence_ms and context_ms must be non-negative")

    samples = _as_mono(waveform)
    active_mask, _, _, _ = _adaptive_active_mask(samples, sample_rate)
    voiced = _spans_from_mask(active_mask)
    if not voiced:
        return []

    min_silence_samples = round(sample_rate * min_silence_ms / 1000)
    context_samples = round(sample_rate * context_ms / 1000)
    groups: list[SegmentSpan] = []
    group_start = voiced[0].start
    group_end = voiced[0].end
    for span in voiced[1:]:
        if span.start - group_end >= min_silence_samples:
            groups.append(SegmentSpan(group_start, group_end))
            group_start = span.start
        group_end = span.end
    groups.append(SegmentSpan(group_start, group_end))

    return [
        SegmentSpan(max(0, span.start - context_samples), min(samples.numel(), span.end + context_samples))
        for span in groups
    ]


def detect_positive_boundary(waveform: torch.Tensor, sample_rate: int) -> PositiveBoundary:
    """Locate a positive wake-word's active region for later speed/crop remapping."""
    active_mask, threshold, noise_floor, speech_level = _adaptive_active_mask(waveform, sample_rate)
    # A word may contain brief between-syllable silences; retain them in its audit mask.
    active_mask = _bridge_short_gaps(active_mask, max(1, round(sample_rate * 0.100)))
    spans = _spans_from_mask(active_mask)
    if not spans:
        return PositiveBoundary(
            start=0,
            end=0,
            center=0,
            active_mask=active_mask,
            threshold=threshold,
            noise_floor=noise_floor,
            speech_level=speech_level,
            audit_status="no_activity",
        )

    start, end = spans[0].start, spans[-1].end
    return PositiveBoundary(
        start=start,
        end=end,
        center=(start + end) // 2,
        active_mask=active_mask,
        threshold=threshold,
        noise_floor=noise_floor,
        speech_level=speech_level,
        audit_status="detected",
    )
