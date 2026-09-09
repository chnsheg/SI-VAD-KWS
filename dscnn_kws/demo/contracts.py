"""Validated deployment contracts for the VAD-KWS cascade demo."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import onnxruntime as ort


class DeploymentError(RuntimeError):
    """Raised when an artifact cannot satisfy the demo deployment contract."""


SUPPORTED_VAD_MODEL_IDS = frozenset(
    {"causal-crnn-vad-kws-realneg", "causal-crnn-vad-kws-20h"}
)


class ThresholdComparator(StrEnum):
    """Threshold equality semantics evidenced by a controller profile."""

    GT = "gt"
    GE = "ge"


class LinkGateMode(StrEnum):
    """How UART/link state participates in controller gating."""

    IGNORE = "ignore"


_SERIAL_LOG_OBSERVED_EVIDENCE: tuple[tuple[str, str], ...] = (
    ("kws_probability_comparator", "serial_log_declares_greater_than_or_equal_to_0.85"),
)
_SERIAL_LOG_PC_ASSUMPTIONS: tuple[tuple[str, str], ...] = (
    ("energy_comparator", "GT; equality at -33 dBFS is unproven by the serial log"),
    ("vad_probability_comparator", "GT; equality at 0.80 is unproven by the serial log"),
    ("kws_period_ms", "96 ms is the PC scheduler cadence; sampled logs do not prove controller cadence"),
    ("kws_energy_hangover_endpoint", "PC reducer treats the hangover deadline as inclusive"),
    ("no_speech_timeout_endpoint", "PC reducer ordering at the timeout boundary"),
)
_SERIAL_LOG_CONTROL_PROFILE = (
    "serial-log-observed-v1",
    16_000,
    1_000,
    32,
    96,
    -33.0,
    0.80,
    0.85,
    3,
    2,
    960,
    3_000,
    ThresholdComparator.GT,
    ThresholdComparator.GT,
    ThresholdComparator.GE,
    LinkGateMode.IGNORE,
)


@dataclass(frozen=True)
class ObservedControlContract:
    """Immutable serial-log-derived values plus explicit PC control assumptions."""

    contract_id: str
    sample_rate_hz: int
    window_ms: int
    vad_period_ms: int
    kws_period_ms: int
    energy_threshold_dbfs: int | float
    vad_threshold: float
    kws_threshold: float
    vad_confirmations: int
    kws_confirmations: int
    kws_energy_hangover_ms: int
    no_speech_timeout_ms: int
    energy_comparator: ThresholdComparator
    vad_probability_comparator: ThresholdComparator
    kws_probability_comparator: ThresholdComparator
    link_gate_mode: LinkGateMode
    observed_evidence: tuple[tuple[str, str], ...] = ()
    pc_assumptions: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        """Discard built-in provenance copied onto a changed profile by replace()."""

        if self._is_builtin_serial_log_profile():
            return
        if self.observed_evidence == _SERIAL_LOG_OBSERVED_EVIDENCE:
            object.__setattr__(self, "observed_evidence", ())
        if self.pc_assumptions == _SERIAL_LOG_PC_ASSUMPTIONS:
            object.__setattr__(self, "pc_assumptions", ())

    def validate(self) -> None:
        if not isinstance(self.contract_id, str) or not self.contract_id:
            raise ValueError("contract_id must be nonempty")
        for name in (
            "sample_rate_hz",
            "window_ms",
            "vad_period_ms",
            "kws_period_ms",
            "vad_confirmations",
            "kws_confirmations",
            "kws_energy_hangover_ms",
            "no_speech_timeout_ms",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.sample_rate_hz != 16000:
            raise ValueError("sample_rate_hz must be exactly 16000 for observed control")
        if self.window_ms != 1000:
            raise ValueError("window_ms must be exactly 1000 for observed control")
        if self.kws_period_ms % self.vad_period_ms != 0:
            raise ValueError("kws_period_ms must be an integer multiple of vad_period_ms")
        _require_finite_number("energy_threshold_dbfs", self.energy_threshold_dbfs)
        for name in (
            "energy_comparator",
            "vad_probability_comparator",
            "kws_probability_comparator",
        ):
            if not isinstance(getattr(self, name), ThresholdComparator):
                raise ValueError(f"{name} must be a ThresholdComparator")
        if not isinstance(self.link_gate_mode, LinkGateMode):
            raise ValueError("link_gate_mode must be a LinkGateMode")
        _validate_probability("vad_threshold", self.vad_threshold)
        _validate_probability("kws_threshold", self.kws_threshold)
        _validate_provenance_entries("observed_evidence", self.observed_evidence)
        _validate_provenance_entries("pc_assumptions", self.pc_assumptions)

    def as_dict(self) -> dict[str, object]:
        """Return JSON-compatible values together with evidence and assumption bounds."""

        return {
            "contract_id": self.contract_id,
            "sample_rate_hz": self.sample_rate_hz,
            "window_ms": self.window_ms,
            "vad_period_ms": self.vad_period_ms,
            "kws_period_ms": self.kws_period_ms,
            "energy_threshold_dbfs": self.energy_threshold_dbfs,
            "vad_threshold": self.vad_threshold,
            "kws_threshold": self.kws_threshold,
            "vad_confirmations": self.vad_confirmations,
            "kws_confirmations": self.kws_confirmations,
            "kws_energy_hangover_ms": self.kws_energy_hangover_ms,
            "no_speech_timeout_ms": self.no_speech_timeout_ms,
            "energy_comparator": self.energy_comparator.value,
            "vad_probability_comparator": self.vad_probability_comparator.value,
            "kws_probability_comparator": self.kws_probability_comparator.value,
            "link_gate_mode": self.link_gate_mode.value,
            "observed_evidence": dict(self.observed_evidence),
            "pc_assumptions": dict(self.pc_assumptions),
        }

    def _is_builtin_serial_log_profile(self) -> bool:
        return (
            self.contract_id,
            self.sample_rate_hz,
            self.window_ms,
            self.vad_period_ms,
            self.kws_period_ms,
            self.energy_threshold_dbfs,
            self.vad_threshold,
            self.kws_threshold,
            self.vad_confirmations,
            self.kws_confirmations,
            self.kws_energy_hangover_ms,
            self.no_speech_timeout_ms,
            self.energy_comparator,
            self.vad_probability_comparator,
            self.kws_probability_comparator,
            self.link_gate_mode,
        ) == _SERIAL_LOG_CONTROL_PROFILE


# "OBSERVED" identifies the log source; as_dict() records unproven PC semantics.
SERIAL_LOG_OBSERVED_CONTROL_CONTRACT = ObservedControlContract(
    contract_id="serial-log-observed-v1",
    sample_rate_hz=16_000,
    window_ms=1_000,
    vad_period_ms=32,
    kws_period_ms=96,
    energy_threshold_dbfs=-33.0,
    vad_threshold=0.80,
    kws_threshold=0.85,
    vad_confirmations=3,
    kws_confirmations=2,
    kws_energy_hangover_ms=960,
    no_speech_timeout_ms=3_000,
    energy_comparator=ThresholdComparator.GT,
    vad_probability_comparator=ThresholdComparator.GT,
    kws_probability_comparator=ThresholdComparator.GE,
    link_gate_mode=LinkGateMode.IGNORE,
    observed_evidence=_SERIAL_LOG_OBSERVED_EVIDENCE,
    pc_assumptions=_SERIAL_LOG_PC_ASSUMPTIONS,
)


@dataclass(frozen=True)
class VadContract:
    model_path: Path
    metadata_path: Path
    model_sha256: str
    input_name: str
    output_name: str
    sample_rate: int
    frame_ms: float
    hop_ms: float
    n_mels: int
    f_min: float
    center: bool
    power: float
    log_floor: float
    normalization_floor: float
    threshold: float
    min_speech_ms: float
    min_silence_ms: float


@dataclass(frozen=True)
class KwsContract:
    model_path: Path
    model_sha256: str
    input_name: str
    output_name: str


@dataclass(frozen=True)
class KwsSplitContract:
    """CPU-certified composition contract for the exact KWS ONNX split."""

    frontend_model_path: Path
    frontend_model_sha256: str
    frontend_input_name: str
    frontend_output_name: str
    backbone_model_path: Path
    backbone_model_sha256: str
    backbone_input_name: str
    backbone_output_name: str
    report_path: Path
    checkpoint_sha256: str
    spec_sha256: str
    source_full_onnx_sha256: str
    parity_samples: int


@dataclass(frozen=True)
class KwsFrameRepairContract:
    """Certified ONNX contract for the five strict-MFCC boundary repair frames."""

    model_path: Path
    model_sha256: str
    input_name: str
    output_name: str
    report_path: Path
    checkpoint_sha256: str
    spec_sha256: str
    source_full_onnx_sha256: str
    parity_samples: int


@dataclass(frozen=True)
class TimingSchedule:
    """Validated VAD/KWS controller periods on the shared PCM timeline."""

    vad_period_ms: int
    energy_period_ms: int
    kws_period_ms: int

    @classmethod
    def from_periods(
        cls, vad_period_ms: int, kws_period_ms: int, *, sample_rate_hz: int = 16000
    ) -> "TimingSchedule":
        if isinstance(sample_rate_hz, bool) or not isinstance(sample_rate_hz, int) or sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be a positive integer")
        for name, period_ms in (("vad_period_ms", vad_period_ms), ("kws_period_ms", kws_period_ms)):
            if isinstance(period_ms, bool) or not isinstance(period_ms, int) or period_ms < 10:
                raise ValueError(f"{name} must be an integer of at least 10 ms")
            if (sample_rate_hz * period_ms) % 1000 != 0:
                raise ValueError(f"{name} must resolve to an integer PCM sample count")
        return cls(
            vad_period_ms=vad_period_ms,
            energy_period_ms=vad_period_ms,
            kws_period_ms=kws_period_ms,
        )

    @classmethod
    def from_vad_period(cls, vad_period_ms: int, *, sample_rate_hz: int = 16000) -> "TimingSchedule":
        """Build the default three-times KWS schedule from a VAD period."""

        return cls.from_periods(vad_period_ms, vad_period_ms * 3, sample_rate_hz=sample_rate_hz)


@dataclass(frozen=True)
class CascadeConfig:
    sample_rate_hz: int = 16000
    window_ms: int = 1000
    kws_lookback_ms: int = 1000
    energy_period_ms: int = 32
    energy_threshold_dbfs: float = -33.0
    energy_hangover_ms: int = 1000
    vad_energy_tail_ms: int = 1000
    vad_period_ms: int = 32
    vad_threshold: float = 0.8
    vad_confirmations: int = 3
    kws_period_ms: int = 96
    kws_positive_index: int = 0
    kws_threshold: float = 0.85
    kws_confirmations: int = 2
    vad_no_speech_timeout_ms: int = 3000
    wake_silence_confirmations: int = 3
    queue_capacity: int = 32

    def validate(self) -> None:
        for name in (
            "sample_rate_hz",
            "window_ms",
            "kws_lookback_ms",
            "energy_period_ms",
            "energy_hangover_ms",
            "vad_energy_tail_ms",
            "vad_period_ms",
            "vad_confirmations",
            "kws_period_ms",
            "kws_confirmations",
            "vad_no_speech_timeout_ms",
            "wake_silence_confirmations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.sample_rate_hz != 16000:
            raise ValueError("sample_rate_hz must be exactly 16000 for the VAD-KWS scheduler")
        if self.window_ms != 1000:
            raise ValueError("window_ms must be exactly 1000 for the fixed KWS input")
        if self.kws_lookback_ms < self.window_ms:
            raise ValueError("kws_lookback_ms must be at least window_ms")
        if (self.sample_rate_hz * self.kws_lookback_ms) % 1000 != 0:
            raise ValueError("kws_lookback_ms must resolve to an integer PCM sample count")
        schedule = TimingSchedule.from_periods(
            self.vad_period_ms, self.kws_period_ms, sample_rate_hz=self.sample_rate_hz
        )
        if self.energy_period_ms != schedule.energy_period_ms:
            raise ValueError("energy_period_ms must equal vad_period_ms")
        if (
            isinstance(self.kws_positive_index, bool)
            or not isinstance(self.kws_positive_index, int)
            or self.kws_positive_index < 0
        ):
            raise ValueError("kws_positive_index must be a nonnegative integer")
        if isinstance(self.queue_capacity, bool) or not isinstance(self.queue_capacity, int) or self.queue_capacity < 1:
            raise ValueError("queue_capacity must be at least one")
        if not math.isfinite(float(self.energy_threshold_dbfs)):
            raise ValueError("energy_threshold_dbfs must be finite")
        _validate_probability("vad_threshold", self.vad_threshold)
        _validate_probability("kws_threshold", self.kws_threshold)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_vad_contract(model_path: Path, providers: Sequence[str] = ("CPUExecutionProvider",)) -> VadContract:
    """Load metadata and validate the exact VAD deployment interface."""

    path = _require_onnx_file(model_path, "VAD model")
    metadata_path = path.with_suffix(".metadata.json")
    metadata = _load_metadata(metadata_path)
    if metadata.get("schema_version") != 1 or isinstance(metadata.get("schema_version"), bool):
        raise DeploymentError(f"metadata schema_version must equal 1: {metadata_path}")
    if metadata.get("model_id") not in SUPPORTED_VAD_MODEL_IDS:
        allowed_ids = ", ".join(sorted(SUPPORTED_VAD_MODEL_IDS))
        raise DeploymentError(f"metadata model_id must be one of {allowed_ids}: {metadata_path}")

    artifact = _require_mapping(metadata.get("artifact"), "metadata artifact")
    if artifact.get("filename") != path.name:
        raise DeploymentError(f"metadata artifact filename does not match VAD model: {metadata_path}")
    expected_hash = artifact.get("sha256")
    actual_hash = sha256_file(path)
    if not isinstance(expected_hash, str) or expected_hash.lower() != actual_hash:
        raise DeploymentError(f"metadata artifact sha256 does not match VAD model: {metadata_path}")

    feature = _require_mapping(metadata.get("feature"), "metadata feature")
    export = _require_mapping(metadata.get("export"), "metadata export")
    postprocess = _require_mapping(metadata.get("postprocess"), "metadata postprocess")
    sample_rate = _require_int(feature, "sample_rate", positive=True)
    n_mels = _require_int(feature, "n_mels", positive=True)
    if sample_rate != 16000 or n_mels != 64:
        raise DeploymentError(f"metadata feature must specify 16000 Hz and 64 Mel bins: {metadata_path}")
    frame_ms = _require_finite(feature, "frame_ms", positive=True)
    hop_ms = _require_finite(feature, "hop_ms", positive=True)
    f_min = _require_finite(feature, "f_min", nonnegative=True)
    power = _require_finite(feature, "power", positive=True)
    log_floor = _require_finite(feature, "log_floor", positive=True)
    normalization_floor = _require_finite(feature, "normalization_floor", positive=True)
    center = feature.get("center")
    if center is not False:
        raise DeploymentError(f"metadata feature center must be false: {metadata_path}")
    if frame_ms != 25.0 or hop_ms != 10.0 or f_min != 20.0 or power != 2.0:
        raise DeploymentError(f"metadata feature values do not match the exported VAD contract: {metadata_path}")
    threshold = _require_finite(postprocess, "threshold", nonnegative=True)
    if threshold > 1.0:
        raise DeploymentError(f"metadata postprocess threshold must be in [0, 1]: {metadata_path}")
    min_speech_ms = _require_finite(postprocess, "min_speech_ms", positive=True)
    min_silence_ms = _require_finite(postprocess, "min_silence_ms", positive=True)
    input_name = _require_nonempty_string(export, "input_name")
    output_name = _require_nonempty_string(export, "output_name")
    if export.get("batch_size") != 1 or isinstance(export.get("batch_size"), bool):
        raise DeploymentError(f"metadata export batch_size must equal 1: {metadata_path}")

    session = _cpu_session(path, providers)
    _validate_vad_io(session, input_name, output_name, path)
    return VadContract(
        model_path=path,
        metadata_path=metadata_path,
        model_sha256=actual_hash,
        input_name=input_name,
        output_name=output_name,
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        n_mels=n_mels,
        f_min=f_min,
        center=False,
        power=power,
        log_floor=log_floor,
        normalization_floor=normalization_floor,
        threshold=threshold,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
    )


def validate_kws_model(model_path: Path, providers: Sequence[str] = ("CPUExecutionProvider",)) -> KwsContract:
    """Validate the fixed-window direct-INT8 KWS ONNX interface."""

    path = _require_onnx_file(model_path, "KWS model")
    session = _cpu_session(path, providers)
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise DeploymentError(f"KWS model must have one input and one output: {path}")
    input_meta = inputs[0]
    output_meta = outputs[0]
    if input_meta.type != "tensor(float)" or _shape(input_meta.shape) != (1, 16000):
        raise DeploymentError(f"KWS input must be float32 [1,16000]: {path}")
    if output_meta.type != "tensor(float)" or _shape(output_meta.shape) != (1, 2):
        raise DeploymentError(f"KWS output must be float32 [1,2]: {path}")
    return KwsContract(
        model_path=path,
        model_sha256=sha256_file(path),
        input_name=input_meta.name,
        output_name=output_meta.name,
    )


def validate_kws_split_model(
    frontend_model_path: Path,
    backbone_model_path: Path,
    report_path: Path,
    providers: Sequence[str] = ("CPUExecutionProvider",),
) -> KwsSplitContract:
    """Validate the certified exact KWS frontend/backbone composition on ORT CPU.

    This validates artifact identity and the split report before the two ONNX
    graphs are used.  It intentionally does not claim that another execution
    provider has the same numerical behavior; that requires target-specific
    certification.
    """

    frontend_path = _require_onnx_file(frontend_model_path, "KWS frontend model")
    backbone_path = _require_onnx_file(backbone_model_path, "KWS backbone model")
    if frontend_path == backbone_path:
        raise DeploymentError("KWS frontend and backbone models must be different files")
    resolved_report_path = Path(report_path).resolve()
    report = _load_kws_split_report(resolved_report_path)
    frontend_sha256 = sha256_file(frontend_path)
    backbone_sha256 = sha256_file(backbone_path)
    _validate_kws_split_report(
        report,
        report_path=resolved_report_path,
        frontend_path=frontend_path,
        frontend_sha256=frontend_sha256,
        backbone_path=backbone_path,
        backbone_sha256=backbone_sha256,
    )

    frontend_session = _cpu_session(frontend_path, providers)
    backbone_session = _cpu_session(backbone_path, providers)
    _validate_kws_split_io(frontend_session, backbone_session, frontend_path, backbone_path)
    return KwsSplitContract(
        frontend_model_path=frontend_path,
        frontend_model_sha256=frontend_sha256,
        frontend_input_name="waveform",
        frontend_output_name="strict_mfcc_codes",
        backbone_model_path=backbone_path,
        backbone_model_sha256=backbone_sha256,
        backbone_input_name="strict_mfcc_codes",
        backbone_output_name="logits",
        report_path=resolved_report_path,
        checkpoint_sha256=_require_sha256(report, "checkpoint_sha256", resolved_report_path),
        spec_sha256=_require_sha256(report, "spec_sha256", resolved_report_path),
        source_full_onnx_sha256=_require_sha256(
            report, "source_full_onnx_sha256", resolved_report_path
        ),
        parity_samples=_require_split_parity_samples(report, resolved_report_path),
    )


def validate_kws_frame_repair_model(
    model_path: Path,
    report_path: Path,
    split_contract: KwsSplitContract,
    providers: Sequence[str] = ("CPUExecutionProvider",),
) -> KwsFrameRepairContract:
    """Validate the exact five-frame repair graph against its split-KWS lineage."""

    path = _require_onnx_file(model_path, "KWS frame repair model")
    resolved_report_path = Path(report_path).resolve()
    report = _load_kws_frame_repair_report(resolved_report_path)
    model_sha256 = sha256_file(path)
    _validate_kws_frame_repair_report(
        report,
        report_path=resolved_report_path,
        model_path=path,
        model_sha256=model_sha256,
        split_contract=split_contract,
    )
    session = _cpu_session(path, providers)
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise DeploymentError(f"KWS frame repair model must have one input and one output: {path}")
    input_meta = inputs[0]
    output_meta = outputs[0]
    if (
        input_meta.name != "waveform"
        or input_meta.type != "tensor(float)"
        or _shape(input_meta.shape) != (1, 16000)
    ):
        raise DeploymentError(f"KWS frame repair input must be waveform float32 [1,16000]: {path}")
    if (
        output_meta.name != "strict_mfcc_repair_codes"
        or output_meta.type != "tensor(uint8)"
        or _shape(output_meta.shape) != (1, 50)
    ):
        raise DeploymentError(
            f"KWS frame repair output must be strict_mfcc_repair_codes uint8 [1,50]: {path}"
        )
    return KwsFrameRepairContract(
        model_path=path,
        model_sha256=model_sha256,
        input_name=input_meta.name,
        output_name=output_meta.name,
        report_path=resolved_report_path,
        checkpoint_sha256=split_contract.checkpoint_sha256,
        spec_sha256=split_contract.spec_sha256,
        source_full_onnx_sha256=split_contract.source_full_onnx_sha256,
        parity_samples=_require_split_parity_samples(report, resolved_report_path),
    )


def _require_onnx_file(model_path: Path, label: str) -> Path:
    path = Path(model_path)
    if path.suffix.lower() != ".onnx":
        raise DeploymentError(f"{label} must have a .onnx suffix: {path}")
    if not path.is_file():
        raise DeploymentError(f"{label} file does not exist: {path}")
    return path.resolve()


def _load_metadata(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise DeploymentError(f"VAD metadata file does not exist: {path}")
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DeploymentError(f"could not read VAD metadata: {path}") from error
    return _require_mapping(parsed, "VAD metadata")


def _load_kws_split_report(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise DeploymentError(f"KWS split report file does not exist: {path}")
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DeploymentError(f"could not read KWS split report: {path}") from error
    return _require_mapping(parsed, "KWS split report")


def _load_kws_frame_repair_report(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise DeploymentError(f"KWS frame repair report file does not exist: {path}")
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DeploymentError(f"could not read KWS frame repair report: {path}") from error
    return _require_mapping(parsed, "KWS frame repair report")


def _cpu_session(path: Path, providers: Sequence[str]) -> ort.InferenceSession:
    if "CPUExecutionProvider" not in providers:
        raise DeploymentError("CPUExecutionProvider must be requested for model validation")
    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise DeploymentError("ONNX Runtime CPUExecutionProvider is unavailable")
    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as error:
        raise DeploymentError(f"could not load ONNX model with CPUExecutionProvider: {path}") from error
    if "CPUExecutionProvider" not in session.get_providers():
        raise DeploymentError(f"ONNX Runtime did not activate CPUExecutionProvider: {path}")
    return session


def _validate_vad_io(session: ort.InferenceSession, input_name: str, output_name: str, path: Path) -> None:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise DeploymentError(f"VAD model must have one input and one output: {path}")
    input_meta = inputs[0]
    output_meta = outputs[0]
    if input_meta.name != input_name or output_meta.name != output_name:
        raise DeploymentError(f"VAD metadata input/output names do not match ONNX model: {path}")
    input_shape = input_meta.shape
    output_shape = output_meta.shape
    if input_meta.type != "tensor(float)" or len(input_shape) != 3 or input_shape[0] != 1 or input_shape[2] != 64:
        raise DeploymentError(f"VAD input must be float32 [1,time,64]: {path}")
    if output_meta.type != "tensor(float)" or len(output_shape) != 2 or output_shape[0] != 1:
        raise DeploymentError(f"VAD output must be float32 [1,time]: {path}")


def _validate_kws_split_io(
    frontend_session: ort.InferenceSession,
    backbone_session: ort.InferenceSession,
    frontend_path: Path,
    backbone_path: Path,
) -> None:
    frontend_inputs = frontend_session.get_inputs()
    frontend_outputs = frontend_session.get_outputs()
    if len(frontend_inputs) != 1 or len(frontend_outputs) != 1:
        raise DeploymentError(f"KWS frontend must have one input and one output: {frontend_path}")
    frontend_input = frontend_inputs[0]
    frontend_output = frontend_outputs[0]
    if (
        frontend_input.name != "waveform"
        or frontend_input.type != "tensor(float)"
        or _shape(frontend_input.shape) != (1, 16000)
    ):
        raise DeploymentError(f"KWS frontend input must be waveform float32 [1,16000]: {frontend_path}")
    if (
        frontend_output.name != "strict_mfcc_codes"
        or frontend_output.type != "tensor(uint8)"
        or _shape(frontend_output.shape) != (1, 320)
    ):
        raise DeploymentError(
            f"KWS frontend output must be strict_mfcc_codes uint8 [1,320]: {frontend_path}"
        )

    backbone_inputs = backbone_session.get_inputs()
    backbone_outputs = backbone_session.get_outputs()
    if len(backbone_inputs) != 1 or len(backbone_outputs) != 1:
        raise DeploymentError(f"KWS backbone must have one input and one output: {backbone_path}")
    backbone_input = backbone_inputs[0]
    backbone_output = backbone_outputs[0]
    if (
        backbone_input.name != "strict_mfcc_codes"
        or backbone_input.type != "tensor(uint8)"
        or _shape(backbone_input.shape) != (1, 320)
    ):
        raise DeploymentError(
            f"KWS backbone input must be strict_mfcc_codes uint8 [1,320]: {backbone_path}"
        )
    if (
        backbone_output.name != "logits"
        or backbone_output.type != "tensor(float)"
        or _shape(backbone_output.shape) != (1, 2)
    ):
        raise DeploymentError(f"KWS backbone output must be logits float32 [1,2]: {backbone_path}")


def _validate_kws_split_report(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    frontend_path: Path,
    frontend_sha256: str,
    backbone_path: Path,
    backbone_sha256: str,
) -> None:
    if report.get("status") != "success" or report.get("strict_parity_passed") is not True:
        raise DeploymentError(f"KWS split report is not a certified successful export: {report_path}")
    _require_report_path(report, "frontend_path", frontend_path, report_path)
    _require_report_path(report, "backbone_path", backbone_path, report_path)
    _require_report_sha256(report, "frontend_sha256", frontend_sha256, report_path)
    _require_report_sha256(report, "backbone_sha256", backbone_sha256, report_path)
    if report.get("boundary_name") != "strict_mfcc_codes":
        raise DeploymentError(f"KWS split report boundary_name must be strict_mfcc_codes: {report_path}")
    if report.get("boundary_shape") != [1, 320]:
        raise DeploymentError(f"KWS split report boundary_shape must be [1, 320]: {report_path}")
    if report.get("boundary_dtype") != "uint8":
        raise DeploymentError(f"KWS split report boundary_dtype must be uint8: {report_path}")
    for name in ("frontend_code_mismatch_count", "logit_word_mismatch_count"):
        value = report.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            raise DeploymentError(f"KWS split report {name} must equal zero: {report_path}")
    max_abs_error = report.get("max_abs_error")
    if (
        isinstance(max_abs_error, bool)
        or not isinstance(max_abs_error, (int, float))
        or not math.isfinite(float(max_abs_error))
        or float(max_abs_error) != 0.0
    ):
        raise DeploymentError(f"KWS split report max_abs_error must equal 0.0: {report_path}")
    _require_sha256(report, "checkpoint_sha256", report_path)
    _require_sha256(report, "spec_sha256", report_path)
    _require_sha256(report, "source_full_onnx_sha256", report_path)
    _require_split_parity_samples(report, report_path)


def _validate_kws_frame_repair_report(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    model_path: Path,
    model_sha256: str,
    split_contract: KwsSplitContract,
) -> None:
    if report.get("status") != "success" or report.get("strict_parity_passed") is not True:
        raise DeploymentError(f"KWS frame repair report is not a certified successful export: {report_path}")
    _require_report_path(report, "repair_path", model_path, report_path)
    _require_report_sha256(report, "repair_sha256", model_sha256, report_path)
    for name, expected in (
        ("checkpoint_sha256", split_contract.checkpoint_sha256),
        ("spec_sha256", split_contract.spec_sha256),
        ("source_full_onnx_sha256", split_contract.source_full_onnx_sha256),
    ):
        if _require_sha256(report, name, report_path) != expected:
            raise DeploymentError(
                f"KWS frame repair report {name} does not match the split KWS artifact: {report_path}"
            )
    if report.get("input_shape") != [1, 16000]:
        raise DeploymentError(f"KWS frame repair report input_shape must be [1, 16000]: {report_path}")
    if report.get("output_name") != "strict_mfcc_repair_codes":
        raise DeploymentError(
            f"KWS frame repair report output_name must be strict_mfcc_repair_codes: {report_path}"
        )
    if report.get("output_shape") != [1, 50] or report.get("output_dtype") != "uint8":
        raise DeploymentError(f"KWS frame repair report output must be uint8 [1, 50]: {report_path}")
    if report.get("frame_count") != 32 or report.get("frame_samples") != 512:
        raise DeploymentError(f"KWS frame repair report frame geometry does not match v6.1: {report_path}")
    if report.get("frame_indices") != [0, 28, 29, 30, 31]:
        raise DeploymentError(f"KWS frame repair report frame_indices are invalid: {report_path}")
    code_mismatches = report.get("code_mismatch_count")
    if isinstance(code_mismatches, bool) or not isinstance(code_mismatches, int) or code_mismatches != 0:
        raise DeploymentError(f"KWS frame repair report code_mismatch_count must equal zero: {report_path}")
    _require_split_parity_samples(report, report_path)


def _require_report_path(report: Mapping[str, Any], name: str, expected: Path, report_path: Path) -> None:
    value = report.get(name)
    if not isinstance(value, str) or Path(value).resolve() != expected:
        raise DeploymentError(f"KWS split report {name} does not match the supplied artifact: {report_path}")


def _require_report_sha256(
    report: Mapping[str, Any], name: str, expected: str, report_path: Path
) -> None:
    if _require_sha256(report, name, report_path) != expected:
        raise DeploymentError(f"KWS split report {name} does not match the supplied artifact: {report_path}")


def _require_sha256(report: Mapping[str, Any], name: str, report_path: Path) -> str:
    value = report.get(name)
    if not isinstance(value, str) or len(value) != 64:
        raise DeploymentError(f"KWS split report {name} must be a SHA-256 digest: {report_path}")
    try:
        int(value, 16)
    except ValueError as error:
        raise DeploymentError(f"KWS split report {name} must be a SHA-256 digest: {report_path}") from error
    return value.lower()


def _require_split_parity_samples(report: Mapping[str, Any], report_path: Path) -> int:
    value = report.get("parity_samples")
    if isinstance(value, bool) or not isinstance(value, int) or value < 19:
        raise DeploymentError(f"KWS split report parity_samples must be at least 19: {report_path}")
    return value


def _shape(value: Sequence[object]) -> tuple[int, ...]:
    if len(value) != 2 or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        return ()
    return tuple(value)  # type: ignore[return-value]


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DeploymentError(f"{name} must be a mapping")
    return value


def _require_nonempty_string(mapping: Mapping[str, Any], name: str) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value.strip():
        raise DeploymentError(f"metadata {name} must be a nonempty string")
    return value


def _require_int(mapping: Mapping[str, Any], name: str, *, positive: bool) -> int:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or (positive and value <= 0):
        raise DeploymentError(f"metadata {name} must be a positive integer")
    return value


def _require_finite(
    mapping: Mapping[str, Any], name: str, *, positive: bool = False, nonnegative: bool = False
) -> float:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise DeploymentError(f"metadata {name} must be finite")
    result = float(value)
    if (positive and result <= 0) or (nonnegative and result < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise DeploymentError(f"metadata {name} must be {qualifier}")
    return result


def _require_finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_probability(name: str, value: object) -> None:
    result = _require_finite_number(name, value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _validate_provenance_entries(name: str, entries: object) -> None:
    if not isinstance(entries, tuple):
        raise ValueError(f"{name} must be a tuple of string pairs")
    keys: set[str] = set()
    for entry in entries:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise ValueError(f"{name} must be a tuple of string pairs")
        key, description = entry
        if (
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(description, str)
            or not description.strip()
        ):
            raise ValueError(f"{name} must be a tuple of nonempty string pairs")
        if key in keys:
            raise ValueError(f"{name} must not contain duplicate keys")
        keys.add(key)
