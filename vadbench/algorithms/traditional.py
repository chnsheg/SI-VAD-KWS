from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from vadbench.algorithms.base import VADAlgorithm
from vadbench.audio import load_audio
from vadbench.features import align_length, frame_signal, log_mel_spectrogram, mfcc_features, rms_zcr, robust_normalize_01
from vadbench.frame_prediction import FramePrediction
from vadbench.manifest import ManifestRecord
from vadbench.metrics import choose_best_threshold


class EnergyAdaptiveVAD(VADAlgorithm):
    name = "energy_adaptive"

    def __init__(self, sample_rate: int = 16000, frame_ms: float = 25.0, hop_ms: float = 10.0, **kwargs: object) -> None:
        super().__init__(frame_hop_ms=hop_ms)
        self.sample_rate = int(sample_rate)
        self.frame_ms = float(frame_ms)
        self.hop_ms = float(hop_ms)

    def fit(
        self,
        train_manifest: Sequence[ManifestRecord],
        val_manifest: Sequence[ManifestRecord],
        base_dir: str | Path,
        **kwargs: object,
    ) -> dict[str, float]:
        labels, scores = _collect_labels_scores(val_manifest, base_dir, self._score_waveform, self.sample_rate)
        self.threshold, metrics = choose_best_threshold(labels, scores)
        return {f"val_{key}": value for key, value in metrics.items()}

    def predict(self, waveform: np.ndarray, sample_rate: int, source_id: str | None = None) -> FramePrediction:
        scores = self._score_waveform(waveform, sample_rate)
        return FramePrediction(scores=scores, frame_hop_ms=self.hop_ms, source_id=source_id)

    def _score_waveform(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        rms, _ = rms_zcr(waveform, sample_rate, self.frame_ms, self.hop_ms)
        log_energy = np.log(rms + 1e-8)
        return robust_normalize_01(log_energy)


class ZCREnergyVAD(EnergyAdaptiveVAD):
    name = "zcr_energy"

    def _score_waveform(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        rms, zcr = rms_zcr(waveform, sample_rate, self.frame_ms, self.hop_ms)
        energy = robust_normalize_01(np.log(rms + 1e-8))
        zcr_norm = robust_normalize_01(zcr)
        score = 0.85 * energy + 0.15 * (1.0 - zcr_norm)
        return np.clip(score, 0.0, 1.0).astype(np.float32)


class SpectralGateVAD(EnergyAdaptiveVAD):
    name = "spectral_gate"

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: float = 25.0,
        hop_ms: float = 10.0,
        n_mels: int = 64,
        **kwargs: object,
    ) -> None:
        super().__init__(sample_rate=sample_rate, frame_ms=frame_ms, hop_ms=hop_ms)
        self.n_mels = int(n_mels)

    def _score_waveform(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        mel = log_mel_spectrogram(
            waveform,
            sample_rate,
            n_mels=self.n_mels,
            frame_ms=self.frame_ms,
            hop_ms=self.hop_ms,
            normalize=False,
        )
        band_energy = mel[:, : max(4, self.n_mels // 2)].mean(axis=1)
        floor = np.percentile(band_energy, 20)
        score = robust_normalize_01(band_energy - floor)
        return score


class MFCCGMMVAD(EnergyAdaptiveVAD):
    name = "mfcc_gmm"
    requires_training = True

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: float = 25.0,
        hop_ms: float = 10.0,
        n_mfcc: int = 64,
        n_mels: int = 64,
        max_fit_frames: int = 60000,
        **kwargs: object,
    ) -> None:
        super().__init__(sample_rate=sample_rate, frame_ms=frame_ms, hop_ms=hop_ms)
        self.n_mfcc = int(n_mfcc)
        self.n_mels = int(n_mels)
        self.max_fit_frames = int(max_fit_frames)
        self.speech_gmm = None
        self.noise_gmm = None

    def fit(
        self,
        train_manifest: Sequence[ManifestRecord],
        val_manifest: Sequence[ManifestRecord],
        base_dir: str | Path,
        **kwargs: object,
    ) -> dict[str, float]:
        from sklearn.mixture import GaussianMixture

        features, labels = _collect_features_labels(
            train_manifest,
            base_dir,
            self._features,
            self.sample_rate,
            self.max_fit_frames,
        )
        if len(features) < 20 or np.sum(labels == 1) < 10 or np.sum(labels == 0) < 10:
            self.speech_gmm = None
            self.noise_gmm = None
        else:
            self.speech_gmm = GaussianMixture(n_components=2, covariance_type="diag", reg_covar=1e-5, random_state=7)
            self.noise_gmm = GaussianMixture(n_components=2, covariance_type="diag", reg_covar=1e-5, random_state=7)
            self.speech_gmm.fit(features[labels == 1])
            self.noise_gmm.fit(features[labels == 0])
        labels_val, scores_val = _collect_labels_scores(val_manifest, base_dir, self._score_waveform, self.sample_rate)
        self.threshold, metrics = choose_best_threshold(labels_val, scores_val)
        return {f"val_{key}": value for key, value in metrics.items()}

    def _features(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        return mfcc_features(
            waveform,
            sample_rate,
            n_mfcc=self.n_mfcc,
            n_mels=self.n_mels,
            frame_ms=self.frame_ms,
            hop_ms=self.hop_ms,
            normalize=True,
        )

    def _score_waveform(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.speech_gmm is None or self.noise_gmm is None:
            return super()._score_waveform(waveform, sample_rate)
        features = self._features(waveform, sample_rate)
        speech_ll = self.speech_gmm.score_samples(features)
        noise_ll = self.noise_gmm.score_samples(features)
        llr = np.clip((speech_ll - noise_ll) / 8.0, -20.0, 20.0)
        return (1.0 / (1.0 + np.exp(-llr))).astype(np.float32)


class KaldiEnergyVAD(EnergyAdaptiveVAD):
    name = "kaldi_energy"

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: float = 25.0,
        hop_ms: float = 10.0,
        energy_threshold: float = 5.5,
        mean_scale: float = 0.5,
        context_frames: int = 2,
        **kwargs: object,
    ) -> None:
        super().__init__(sample_rate=sample_rate, frame_ms=frame_ms, hop_ms=hop_ms)
        self.energy_threshold = float(energy_threshold)
        self.mean_scale = float(mean_scale)
        self.context_frames = int(context_frames)

    def _score_waveform(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        rms, _ = rms_zcr(waveform, sample_rate, self.frame_ms, self.hop_ms)
        log_energy = np.log(np.maximum(rms * rms, 1e-12))
        centered = log_energy - self.mean_scale * float(np.mean(log_energy))
        score = robust_normalize_01(centered - self.energy_threshold)
        if self.context_frames > 0:
            kernel = np.ones(self.context_frames * 2 + 1, dtype=np.float32)
            kernel /= kernel.sum()
            score = np.convolve(score, kernel, mode="same")
        return np.clip(score, 0.0, 1.0).astype(np.float32)


class SpectralFluxLTSDVAD(EnergyAdaptiveVAD):
    name = "spectral_flux_ltsd"

    def __init__(self, sample_rate: int = 16000, frame_ms: float = 25.0, hop_ms: float = 10.0, **kwargs: object) -> None:
        super().__init__(sample_rate=sample_rate, frame_ms=frame_ms, hop_ms=hop_ms)

    def _score_waveform(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        frames = frame_signal(waveform, sample_rate, self.frame_ms, self.hop_ms)
        window = np.hanning(frames.shape[1]).astype(np.float32)
        spec = np.abs(np.fft.rfft(frames * window[None, :], axis=1)).astype(np.float32)
        spec = np.maximum(spec, 1e-8)
        flux = np.zeros(spec.shape[0], dtype=np.float32)
        if len(spec) > 1:
            diff = np.maximum(spec[1:] - spec[:-1], 0.0)
            flux[1:] = np.sqrt(np.mean(diff * diff, axis=1))
        noise = np.percentile(spec, 10, axis=0)
        ltsd = 10.0 * np.log10(np.mean((spec / np.maximum(noise[None, :], 1e-8)) ** 2, axis=1) + 1e-8)
        return robust_normalize_01(0.55 * robust_normalize_01(ltsd) + 0.45 * robust_normalize_01(flux))


class SohnHMMVAD(EnergyAdaptiveVAD):
    name = "sohn_hmm"

    def _score_waveform(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        rms, _ = rms_zcr(waveform, sample_rate, self.frame_ms, self.hop_ms)
        energy = np.log(rms * rms + 1e-12)
        noise_mu = float(np.percentile(energy, 20))
        speech_mu = float(np.percentile(energy, 85))
        noise_var = float(np.var(energy[energy <= np.percentile(energy, 40)]) + 1e-3)
        speech_var = float(np.var(energy[energy >= np.percentile(energy, 60)]) + 1e-3)
        ll_s = -0.5 * ((energy - speech_mu) ** 2 / speech_var + np.log(speech_var))
        ll_n = -0.5 * ((energy - noise_mu) ** 2 / noise_var + np.log(noise_var))
        obs = ll_s - ll_n
        p_stay_speech = 0.97
        p_stay_noise = 0.97
        log_s = np.empty_like(obs)
        log_n = np.empty_like(obs)
        log_s[0] = obs[0] + np.log(1.0 - p_stay_noise)
        log_n[0] = np.log(p_stay_noise)
        for idx in range(1, len(obs)):
            prev_s = max(log_s[idx - 1] + np.log(p_stay_speech), log_n[idx - 1] + np.log(1.0 - p_stay_noise))
            prev_n = max(log_n[idx - 1] + np.log(p_stay_noise), log_s[idx - 1] + np.log(1.0 - p_stay_speech))
            log_s[idx] = obs[idx] + prev_s
            log_n[idx] = prev_n
            m = max(log_s[idx], log_n[idx])
            log_s[idx] -= m
            log_n[idx] -= m
        return (1.0 / (1.0 + np.exp(np.clip(log_n - log_s, -30.0, 30.0)))).astype(np.float32)


class RVADFastVAD(EnergyAdaptiveVAD):
    name = "rvad_fast"

    def _score_waveform(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        rms, zcr = rms_zcr(waveform, sample_rate, self.frame_ms, self.hop_ms)
        log_energy = np.log(rms + 1e-8)
        win = max(5, int(round(0.5 * 1000.0 / self.hop_ms)))
        padded = np.pad(log_energy, (win // 2, win // 2), mode="edge")
        floor = np.array([np.percentile(padded[idx : idx + win], 20) for idx in range(len(log_energy))], dtype=np.float32)
        energy_score = robust_normalize_01(log_energy - floor)
        zcr_gate = 1.0 - robust_normalize_01(zcr)
        smooth = np.convolve(0.8 * energy_score + 0.2 * zcr_gate, np.ones(5, dtype=np.float32) / 5.0, mode="same")
        return np.clip(smooth, 0.0, 1.0).astype(np.float32)


class WebRTCVAD(EnergyAdaptiveVAD):
    name = "webrtc_vad"

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: float = 30.0,
        hop_ms: float = 10.0,
        aggressiveness: int = 2,
        **kwargs: object,
    ) -> None:
        super().__init__(sample_rate=sample_rate, frame_ms=frame_ms, hop_ms=hop_ms)
        self.aggressiveness = int(aggressiveness)

    def _score_waveform(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        try:
            import webrtcvad
        except ImportError as exc:
            raise RuntimeError(
                "webrtcvad is required for webrtc_vad. Install with: "
                "C:\\myApps\\Miniconda\\envs\\eis\\python.exe -m pip install webrtcvad"
            ) from exc
        vad = webrtcvad.Vad(max(0, min(3, self.aggressiveness)))
        frame_ms = 30
        frame_len = int(sample_rate * frame_ms / 1000)
        hop_len = int(sample_rate * self.hop_ms / 1000)
        pcm = np.clip(waveform, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype("<i2")
        count = max(1, int(np.ceil((len(pcm16) - frame_len) / max(hop_len, 1))) + 1)
        scores = np.zeros(count, dtype=np.float32)
        for idx in range(count):
            start = idx * hop_len
            end = start + frame_len
            frame = pcm16[start:end]
            if len(frame) < frame_len:
                frame = np.pad(frame, (0, frame_len - len(frame)))
            scores[idx] = 1.0 if vad.is_speech(frame.tobytes(), sample_rate) else 0.0
        return scores


def _collect_labels_scores(
    records: Sequence[ManifestRecord],
    base_dir: str | Path,
    scorer: Callable[[np.ndarray, int], np.ndarray],
    target_sample_rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    for record in records:
        waveform, sample_rate = load_audio(record.resolve_audio(base_dir), target_sample_rate)
        labels = np.load(record.resolve_label(base_dir)).astype(np.uint8)
        scores = scorer(waveform, sample_rate)
        scores = align_length(scores, len(labels), pad_value=0.0)
        labels_all.append(labels)
        scores_all.append(scores)
    if not labels_all:
        return np.array([], dtype=np.uint8), np.array([], dtype=np.float32)
    return np.concatenate(labels_all), np.concatenate(scores_all)


def _collect_features_labels(
    records: Sequence[ManifestRecord],
    base_dir: str | Path,
    feature_fn: Callable[[np.ndarray, int], np.ndarray],
    target_sample_rate: int,
    max_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    features_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    frames_seen = 0
    rng = np.random.default_rng(7)
    for record in records:
        waveform, sample_rate = load_audio(record.resolve_audio(base_dir), target_sample_rate)
        labels = np.load(record.resolve_label(base_dir)).astype(np.uint8)
        features = feature_fn(waveform, sample_rate)
        length = min(len(features), len(labels))
        features = features[:length]
        labels = labels[:length]
        remaining = max_frames - frames_seen
        if remaining <= 0:
            break
        if length > remaining:
            indices = rng.choice(length, size=remaining, replace=False)
            features = features[indices]
            labels = labels[indices]
            length = remaining
        features_all.append(features)
        labels_all.append(labels)
        frames_seen += length
    if not features_all:
        return np.empty((0, 1), dtype=np.float32), np.empty((0,), dtype=np.uint8)
    return np.concatenate(features_all, axis=0).astype(np.float32), np.concatenate(labels_all).astype(np.uint8)
