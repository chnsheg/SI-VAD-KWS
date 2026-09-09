"""Small injectable ONNX Runtime inference adapters for the cascade."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .contracts import KwsContract, KwsFrameRepairContract, KwsSplitContract, VadContract
from dscnn_kws.vad.state import VadStreamState


class InferenceError(RuntimeError):
    """Raised when a validated ONNX model produces an invalid runtime value."""


class OnnxSession(Protocol):
    def run(self, output_names: list[str], inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Run an ONNX graph and return requested output tensors."""


class VadOnnxRunner:
    def __init__(self, session: OnnxSession, contract: VadContract) -> None:
        self._session = session
        self.contract = contract

    def latest_score(self, features: np.ndarray) -> float:
        values = np.asarray(features)
        if values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != self.contract.n_mels:
            raise InferenceError("VAD features must be float32 [frames, n_mels]")
        if values.shape[0] < 1 or not np.all(np.isfinite(values)):
            raise InferenceError("VAD features must be finite and nonempty")
        output = self._session.run([self.contract.output_name], {self.contract.input_name: values[None]})[0]
        scores = np.asarray(output)
        if (
            scores.dtype != np.float32
            or scores.shape != (1, len(values))
            or not np.all(np.isfinite(scores))
        ):
            raise InferenceError("VAD output must be finite float32 [1, frames]")
        return float(scores[0, -1])


class StatefulVadOnnxRunner:
    """Run the exported causal VAD adapter while carrying its public state."""

    _OUTPUT_NAMES = ["probability_chunk", "cnn_context_out", "gru_hidden_out"]

    def __init__(self, session: OnnxSession, *, n_mels: int = 64) -> None:
        if isinstance(n_mels, bool) or not isinstance(n_mels, int) or n_mels <= 0:
            raise ValueError("n_mels must be a positive integer")
        self._session = session
        self._n_mels = n_mels
        self._state = self._zero_state()

    @property
    def state(self) -> VadStreamState:
        """Return an immutable snapshot rather than exposing mutable runner state."""

        return VadStreamState(self._state.cnn_context.copy(), self._state.gru_hidden.copy())

    def reset(self) -> None:
        self._state = self._zero_state()

    def latest_score(self, features: np.ndarray) -> float:
        return float(self.score_chunk(features)[-1])

    def score_chunk(self, features: np.ndarray) -> np.ndarray:
        """Return every posterior produced by one causal feature chunk."""

        values = np.asarray(features)
        if values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != self._n_mels:
            raise InferenceError("stateful VAD features must be float32 [frames, n_mels]")
        if values.shape[0] < 1 or not np.all(np.isfinite(values)):
            raise InferenceError("stateful VAD features must be finite and nonempty")
        output = self._session.run(
            self._OUTPUT_NAMES,
            {
                "log_mel_chunk": values[None],
                "cnn_context_in": self._state.cnn_context,
                "gru_hidden_in": self._state.gru_hidden,
            },
        )
        if len(output) != 3:
            raise InferenceError("stateful VAD must return probability and both state tensors")
        scores = np.asarray(output[0])
        if (
            scores.dtype != np.float32
            or scores.shape != (1, len(values))
            or not np.all(np.isfinite(scores))
        ):
            raise InferenceError("stateful VAD probability must be finite float32 [1, frames]")
        try:
            next_state = VadStreamState(np.asarray(output[1]), np.asarray(output[2]))
        except ValueError as error:
            raise InferenceError("stateful VAD returned an invalid state tensor") from error
        self._state = next_state
        return scores[0].copy()

    @staticmethod
    def _zero_state() -> VadStreamState:
        return VadStreamState(
            np.zeros((1, 128, 4), dtype=np.float32),
            np.zeros((1, 1, 40), dtype=np.float32),
        )


class KwsOnnxRunner:
    def __init__(self, session: OnnxSession, contract: KwsContract, positive_index: int) -> None:
        _validate_positive_index(positive_index)
        self._session = session
        self.contract = contract
        self._positive_index = positive_index

    def logits(self, waveform: np.ndarray) -> np.ndarray:
        values = _validate_kws_waveform(waveform)
        output = self._session.run(
            [self.contract.output_name], {self.contract.input_name: values.reshape(1, 16000)}
        )[0]
        return _validate_kws_logits(output)

    def score(self, waveform: np.ndarray) -> float:
        return _positive_softmax_score(self.logits(waveform), self._positive_index)


class KwsSplitOnnxRunner:
    """Run the certified UINT8 split without changing the exact code boundary."""

    def __init__(
        self,
        *,
        frontend_session: OnnxSession,
        backbone_session: OnnxSession,
        contract: KwsSplitContract,
        positive_index: int,
    ) -> None:
        _validate_positive_index(positive_index)
        self._frontend_session = frontend_session
        self._backbone_session = backbone_session
        self.contract = contract
        self._positive_index = positive_index

    def logits(self, waveform: np.ndarray) -> np.ndarray:
        values = _validate_kws_waveform(waveform)
        codes = np.asarray(
            self._frontend_session.run(
                [self.contract.frontend_output_name],
                {self.contract.frontend_input_name: values.reshape(1, 16000)},
            )[0]
        )
        if codes.dtype != np.uint8 or codes.shape != (1, 320):
            raise InferenceError("KWS frontend codes must be uint8 [1, 320]")
        output = self._backbone_session.run(
            [self.contract.backbone_output_name],
            {self.contract.backbone_input_name: codes},
        )[0]
        return _validate_kws_logits(output)

    def score(self, waveform: np.ndarray) -> float:
        return _positive_softmax_score(self.logits(waveform), self._positive_index)


class KwsCachedSplitOnnxRunner:
    """Exact split KWS with a 96 ms code cache and certified boundary repair."""

    _WINDOW_SAMPLES = 16_000
    _STEP_SAMPLES = 1_536
    _FRAME_COUNT = 32
    _MFCC_PER_FRAME = 10
    _REUSED_SOURCE = slice(4, 31)
    _REUSED_DESTINATION = slice(1, 28)
    _REPAIR_FRAMES = (0, 28, 29, 30, 31)

    def __init__(
        self,
        *,
        frontend_session: OnnxSession,
        backbone_session: OnnxSession,
        repair_session: OnnxSession,
        contract: KwsSplitContract,
        repair_contract: KwsFrameRepairContract,
        positive_index: int,
    ) -> None:
        _validate_positive_index(positive_index)
        if (
            repair_contract.checkpoint_sha256 != contract.checkpoint_sha256
            or repair_contract.spec_sha256 != contract.spec_sha256
            or repair_contract.source_full_onnx_sha256 != contract.source_full_onnx_sha256
        ):
            raise ValueError("KWS repair artifact provenance does not match the split KWS artifact")
        self._frontend_session = frontend_session
        self._backbone_session = backbone_session
        self._repair_session = repair_session
        self.contract = contract
        self.repair_contract = repair_contract
        self._positive_index = positive_index
        self._previous_waveform: np.ndarray | None = None
        self._previous_codes: np.ndarray | None = None

    def reset(self) -> None:
        self._previous_waveform = None
        self._previous_codes = None

    def logits(self, waveform: np.ndarray) -> np.ndarray:
        values = _validate_kws_waveform(waveform)
        codes = self._codes_for_window(values)
        output = self._backbone_session.run(
            [self.contract.backbone_output_name],
            {self.contract.backbone_input_name: codes},
        )[0]
        logits = _validate_kws_logits(output)
        self._previous_waveform = values.copy()
        self._previous_codes = codes.copy()
        return logits

    def score(self, waveform: np.ndarray) -> float:
        return _positive_softmax_score(self.logits(waveform), self._positive_index)

    def _codes_for_window(self, waveform: np.ndarray) -> np.ndarray:
        if self._can_reuse(waveform):
            return self._repair_cached_codes(waveform)
        output = self._frontend_session.run(
            [self.contract.frontend_output_name],
            {self.contract.frontend_input_name: waveform.reshape(1, self._WINDOW_SAMPLES)},
        )[0]
        return _validate_kws_codes(output, shape=(1, self._FRAME_COUNT * self._MFCC_PER_FRAME))

    def _can_reuse(self, waveform: np.ndarray) -> bool:
        if self._previous_waveform is None or self._previous_codes is None:
            return False
        return np.array_equal(
            self._previous_waveform[self._STEP_SAMPLES :], waveform[: -self._STEP_SAMPLES]
        )

    def _repair_cached_codes(self, waveform: np.ndarray) -> np.ndarray:
        assert self._previous_codes is not None
        output = self._repair_session.run(
            [self.repair_contract.output_name],
            {self.repair_contract.input_name: waveform.reshape(1, self._WINDOW_SAMPLES)},
        )[0]
        repair_codes = _validate_kws_codes(
            output, shape=(1, len(self._REPAIR_FRAMES) * self._MFCC_PER_FRAME)
        )
        prior = self._previous_codes.reshape(1, self._FRAME_COUNT, self._MFCC_PER_FRAME)
        next_codes = np.empty_like(prior)
        next_codes[:, self._REUSED_DESTINATION, :] = prior[:, self._REUSED_SOURCE, :]
        next_codes[:, self._REPAIR_FRAMES, :] = repair_codes.reshape(
            1, len(self._REPAIR_FRAMES), self._MFCC_PER_FRAME
        )
        return next_codes.reshape(1, -1)


def _validate_positive_index(positive_index: int) -> None:
    if isinstance(positive_index, bool) or not isinstance(positive_index, int) or positive_index not in (0, 1):
        raise ValueError("positive_index must be 0 or 1")


def _validate_kws_waveform(waveform: np.ndarray) -> np.ndarray:
    values = np.asarray(waveform)
    if values.dtype != np.float32 or values.shape != (16000,) or not np.all(np.isfinite(values)):
        raise InferenceError("KWS waveform must be finite float32 [16000]")
    return values


def _validate_kws_logits(output: np.ndarray) -> np.ndarray:
    logits = np.asarray(output)
    if logits.dtype != np.float32 or logits.shape != (1, 2) or not np.all(np.isfinite(logits)):
        raise InferenceError("KWS logits must be finite float32 [1, 2]")
    return logits


def _validate_kws_codes(output: np.ndarray, *, shape: tuple[int, int]) -> np.ndarray:
    codes = np.asarray(output)
    if codes.dtype != np.uint8 or codes.shape != shape:
        raise InferenceError(f"KWS frontend codes must be uint8 {list(shape)}")
    return codes


def _positive_softmax_score(logits: np.ndarray, positive_index: int) -> float:
    shifted = logits[0] - np.max(logits[0])
    exp_shifted = np.exp(shifted)
    return float(exp_shifted[positive_index] / exp_shifted.sum())
