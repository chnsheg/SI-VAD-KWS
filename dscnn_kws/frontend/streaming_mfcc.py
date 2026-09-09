from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mfcc_torch import apply_piecewise_linear, create_dct_matrix, create_mel_filterbank
from .pwl_fit_utils import fit_piecewise_linear_log_from_samples


@dataclass
class StreamingMFCCState:
    """State for online MFCC frame extraction."""

    history: torch.Tensor
    pending: torch.Tensor
    samples_seen: int = 0


class StreamingMFCC(nn.Module):
    """Causal MFCC frontend with offline and online execution paths.

    Offline forward() uses the same causal hop framing as forward_stream_chunk().
    It does not call torch.stft(center=True), so it does not need future samples.
    Output shape is [B, n_mfcc, frames], matching TorchMFCC.
    """

    def __init__(
        self,
        sample_rate: int,
        n_mfcc: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        f_min: float = 0.0,
        f_max: float | None = None,
        dct_norm: str | None = "ortho",
        mel_filter_shape: str = "triangular",
        log_offset: float = 1e-6,
        log_approx_mode: str = "exact",
        log_pwl_breakpoints: list[float] | None = None,
        log_pwl_slopes: list[float] | None = None,
        log_pwl_intercepts: list[float] | None = None,
        log_pwl_num_segments: int = 6,
        log_pwl_strategy: str = "uniform_logx",
        log_pwl_gamma: float = 1.0,
        log_input_clamp_min: float = 1e-12,
        flush_tail: bool = True,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.n_mfcc = int(n_mfcc)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.f_min = float(f_min)
        self.f_max = float(f_max if f_max is not None else sample_rate / 2)
        self.mel_filter_shape = mel_filter_shape
        self.log_offset = float(log_offset)
        self.log_approx_mode = log_approx_mode
        self.log_input_clamp_min = float(log_input_clamp_min)
        self.flush_tail = bool(flush_tail)

        if self.n_fft < self.win_length:
            raise ValueError("n_fft must be >= win_length")
        if self.win_length <= 0 or self.hop_length <= 0:
            raise ValueError("win_length and hop_length must be positive")
        if self.win_length < self.hop_length:
            raise ValueError("StreamingMFCC expects win_length >= hop_length for causal framing")
        if self.log_approx_mode not in {"exact", "pwl"}:
            raise ValueError(f"Unsupported log_approx_mode: {self.log_approx_mode}")

        self.history_length = self.win_length - self.hop_length
        mel_fb = create_mel_filterbank(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            filter_shape=mel_filter_shape,
        )
        dct_mat = create_dct_matrix(n_mfcc=self.n_mfcc, n_mels=self.n_mels, norm=dct_norm)
        window = torch.hann_window(self.win_length)

        self.register_buffer("mel_fb", mel_fb, persistent=False)
        self.register_buffer("dct_mat", dct_mat, persistent=False)
        self.register_buffer("window", window, persistent=False)

        if self.log_approx_mode == "pwl":
            if log_pwl_breakpoints is None or log_pwl_slopes is None or log_pwl_intercepts is None:
                default_fit = fit_piecewise_linear_log_from_samples(
                    x_samples=torch.logspace(-8, 2, steps=30000),
                    num_segments=log_pwl_num_segments,
                    strategy=log_pwl_strategy,
                    gamma=log_pwl_gamma,
                )
                log_pwl_breakpoints = default_fit["breakpoints"]
                log_pwl_slopes = default_fit["slopes"]
                log_pwl_intercepts = default_fit["intercepts"]

            self.register_buffer(
                "log_pwl_breakpoints",
                torch.tensor(log_pwl_breakpoints, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "log_pwl_slopes",
                torch.tensor(log_pwl_slopes, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                "log_pwl_intercepts",
                torch.tensor(log_pwl_intercepts, dtype=torch.float32),
                persistent=False,
            )

    def _normalize_waveform(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(f"Expected waveform [B, T] or [B, 1, T], got {tuple(x.shape)}")
        return x

    def _empty_mfcc(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(batch_size, self.n_mfcc, 0, device=device, dtype=dtype)

    def _log_transform(self, mel_spec: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(mel_spec + self.log_offset, min=self.log_input_clamp_min)
        if self.log_approx_mode == "exact":
            return torch.log(x)

        bp = self.log_pwl_breakpoints.to(device=x.device, dtype=x.dtype)
        slopes = self.log_pwl_slopes.to(device=x.device, dtype=x.dtype)
        intercepts = self.log_pwl_intercepts.to(device=x.device, dtype=x.dtype)
        return apply_piecewise_linear(x, bp, slopes, intercepts)

    def _mfcc_from_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dim() != 3:
            raise ValueError(f"Expected frames [B, N, win_length], got {tuple(frames.shape)}")
        if frames.size(1) == 0:
            return self._empty_mfcc(frames.size(0), frames.device, frames.dtype)

        window = self.window.to(device=frames.device, dtype=frames.dtype)
        windowed = frames * window.view(1, 1, -1)
        spectrum = torch.fft.rfft(windowed, n=self.n_fft, dim=-1)
        power_spec = spectrum.real.pow(2) + spectrum.imag.pow(2)

        mel_fb = self.mel_fb.to(device=frames.device, dtype=power_spec.dtype)
        mel_spec = torch.matmul(power_spec, mel_fb.transpose(0, 1)).transpose(1, 2)
        log_mel = self._log_transform(mel_spec)

        dct_mat = self.dct_mat.to(device=frames.device, dtype=log_mel.dtype)
        return torch.matmul(dct_mat, log_mel)

    def _frame_offline(self, x: torch.Tensor, flush_tail: bool) -> torch.Tensor:
        batch_size, sample_count = x.shape
        if sample_count <= 0:
            return x.new_empty(batch_size, 0, self.win_length)

        if flush_tail:
            right_pad = (self.hop_length - sample_count % self.hop_length) % self.hop_length
            x = F.pad(x, (self.history_length, right_pad))
        else:
            usable = (sample_count // self.hop_length) * self.hop_length
            if usable <= 0:
                return x.new_empty(batch_size, 0, self.win_length)
            x = F.pad(x[:, :usable], (self.history_length, 0))

        if x.size(1) < self.win_length:
            return x.new_empty(batch_size, 0, self.win_length)
        return x.unfold(dimension=1, size=self.win_length, step=self.hop_length)

    def forward(self, x: torch.Tensor, flush_tail: bool | None = None) -> torch.Tensor:
        x = self._normalize_waveform(x)
        if flush_tail is None:
            flush_tail = self.flush_tail
        frames = self._frame_offline(x, flush_tail=bool(flush_tail))
        return self._mfcc_from_frames(frames)

    def init_stream_state(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> StreamingMFCCState:
        history = torch.zeros(batch_size, self.history_length, device=device, dtype=dtype)
        pending = torch.empty(batch_size, 0, device=device, dtype=dtype)
        return StreamingMFCCState(history=history, pending=pending, samples_seen=0)

    @torch.no_grad()
    def forward_stream_chunk(
        self,
        x: torch.Tensor,
        state: StreamingMFCCState | None = None,
        flush: bool = False,
    ) -> tuple[torch.Tensor, StreamingMFCCState]:
        x = self._normalize_waveform(x)
        batch_size = x.size(0)
        if state is None:
            state = self.init_stream_state(batch_size, device=x.device, dtype=x.dtype)
        if state.history.size(0) != batch_size or state.pending.size(0) != batch_size:
            raise ValueError("StreamingMFCCState batch size does not match the current chunk")

        history = state.history.to(device=x.device, dtype=x.dtype)
        pending = state.pending.to(device=x.device, dtype=x.dtype)
        audio = torch.cat([pending, x], dim=1)

        frames: list[torch.Tensor] = []

        def consume_hop(hop: torch.Tensor, cur_history: torch.Tensor) -> torch.Tensor:
            if self.history_length > 0:
                frame = torch.cat([cur_history, hop], dim=1)
            else:
                frame = hop
            frames.append(frame)
            if self.history_length > 0:
                return frame[:, -self.history_length :].detach()
            return cur_history

        while audio.size(1) >= self.hop_length:
            hop = audio[:, : self.hop_length]
            history = consume_hop(hop, history)
            audio = audio[:, self.hop_length :]

        if flush and audio.size(1) > 0:
            hop = F.pad(audio, (0, self.hop_length - audio.size(1)))
            history = consume_hop(hop, history)
            audio = audio.new_empty(batch_size, 0)

        if frames:
            frame_tensor = torch.stack(frames, dim=1)
            mfcc = self._mfcc_from_frames(frame_tensor)
        else:
            mfcc = self._empty_mfcc(batch_size, x.device, x.dtype)

        next_state = StreamingMFCCState(
            history=history.detach(),
            pending=audio.detach(),
            samples_seen=state.samples_seen + x.size(1),
        )
        return mfcc, next_state

    @torch.no_grad()
    def flush_stream(self, state: StreamingMFCCState) -> tuple[torch.Tensor, StreamingMFCCState]:
        empty = state.pending.new_empty(state.pending.size(0), 0)
        return self.forward_stream_chunk(empty, state=state, flush=True)


def streaming_mfcc_config(frontend: StreamingMFCC) -> dict[str, Any]:
    return {
        "sample_rate": frontend.sample_rate,
        "n_mfcc": frontend.n_mfcc,
        "n_fft": frontend.n_fft,
        "win_length": frontend.win_length,
        "hop_length": frontend.hop_length,
        "n_mels": frontend.n_mels,
        "f_min": frontend.f_min,
        "f_max": frontend.f_max,
        "mel_filter_shape": frontend.mel_filter_shape,
        "log_approx_mode": frontend.log_approx_mode,
        "flush_tail": frontend.flush_tail,
    }

