"""Command-line composition for the Windows VAD-KWS cascade demo."""

from __future__ import annotations

import argparse
import gc
import inspect
import math
import platform
import queue
import sys
import threading
import time
import tempfile
import wave
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import onnxruntime as ort

from .audio import (
    AudioChunk,
    AudioSourceError,
    BoundedAudioQueue,
    CaptureMode,
    InputChannelMode,
    MicrophoneSource,
    WavReplaySource,
    list_input_devices,
    resolve_input_device,
)
from dscnn_kws.ONNX.export_vad_stateful import export_stateful_vad
from .cascade import CascadeEngine, CascadeEvent, CascadeState
from .contracts import (
    CascadeConfig,
    DeploymentError,
    KwsContract,
    KwsSplitContract,
    TimingSchedule,
    load_vad_contract,
    sha256_file,
    validate_kws_model,
    validate_kws_split_model,
)
from .diagnostics import (
    RealtimeAnalysisControl,
    RealtimeAnalysisRequest,
    RealtimeCaptureControl,
    RealtimeCaptureModeControl,
    RealtimeDashboardServer,
    RealtimeDeviceControl,
    RealtimeDiagnosticStore,
    RealtimeInputControl,
    RealtimeThresholdControl,
)
from .features import AudioContractError, StreamingChannelConverter, StreamingVadFeatureCache
from .input_frontend import FrontendProfile
from .input_pipeline import RealtimeInputPipeline
from .kws_executor import KwsJob, KwsResult, RealtimeKwsExecutor
from .observed_control import ObservedControlEngine, ObservedControlRuntimeConfig
from .runners import KwsOnnxRunner, KwsSplitOnnxRunner, StatefulVadOnnxRunner
from .runtime_profile import (
    LEGACY_RUNTIME_PROFILE,
    PC_DEPLOYMENT_RUNTIME_PROFILE,
    RuntimeProfile,
    WakeLifecyclePolicy,
    load_pc_deployment_runtime_config,
    save_pc_deployment_runtime_config,
)
from .telemetry import NullSessionLogger, SessionLogger, TelemetryError


class TelemetryLogger(Protocol):
    """The event sink shared by durable and in-memory-only demo runs."""

    def record(self, event: str, **fields: object) -> None: ...

    def close(self, stopped_ns: int | None = None) -> dict[str, object]: ...


@dataclass(frozen=True)
class _ObservedControlRuntimeSettings:
    """Display and capture dimensions derived from the immutable parity contract."""

    sample_rate_hz: int
    window_ms: int
    vad_threshold: float
    kws_threshold: float
    energy_enabled: bool
    vad_enabled: bool
    vad_confirmations: int
    kws_confirmations: int
    vad_period_ms: int
    kws_period_ms: int
    kws_lookback_ms: int
    kws_positive_index: int
    queue_capacity: int


def _controller_runtime_profile(name: str) -> RuntimeProfile:
    if name == "legacy-pc":
        return LEGACY_RUNTIME_PROFILE
    if name == "pc-deployment":
        return PC_DEPLOYMENT_RUNTIME_PROFILE
    raise ValueError(f"unsupported controller profile: {name}")


def _observed_control_runtime_settings(
    profile: RuntimeProfile,
    *,
    kws_positive_index: int,
    queue_capacity: int,
    runtime_config_path: Path | None = None,
) -> _ObservedControlRuntimeSettings:
    contract = profile.control_contract
    if contract is None:
        raise ValueError("PC deployment control profile requires a control contract")
    contract.validate()
    settings = _ObservedControlRuntimeSettings(
        sample_rate_hz=contract.sample_rate_hz,
        window_ms=contract.window_ms,
        vad_threshold=profile.vad_threshold,
        kws_threshold=profile.kws_threshold,
        energy_enabled=True,
        vad_enabled=True,
        vad_confirmations=profile.vad_confirmations,
        kws_confirmations=profile.kws_confirmations,
        vad_period_ms=profile.vad_period_ms,
        kws_period_ms=profile.kws_period_ms,
        kws_lookback_ms=profile.kws_lookback_ms,
        kws_positive_index=kws_positive_index,
        queue_capacity=queue_capacity,
    )
    persisted = load_pc_deployment_runtime_config(runtime_config_path)
    if persisted is None:
        return settings
    return replace(settings, **persisted)


def _convert_first_channel_pcm(
    converter: StreamingChannelConverter,
    samples: np.ndarray,
    *,
    input_sample_rate: int,
) -> np.ndarray:
    """Resample a source without mixing channels, then select channel zero."""

    values = np.asarray(samples)
    if values.ndim == 1:
        values = values[:, None]
    channels = converter.convert(values, input_sample_rate)
    return channels[:, 0].copy()


def _should_apply_realtime_input_frontend(
    *, microphone: MicrophoneSource | None, is_analysis_command: bool, profile: RuntimeProfile
) -> bool:
    """Use the deployed frontend for live, replay, and imported analysis PCM."""

    return (
        microphone is not None
        or is_analysis_command
        or profile.frontend_profile is not FrontendProfile.RAW
    )


def _push_engine_pcm(
    engine: object,
    pcm: np.ndarray,
    captured_ns: int,
    *,
    energy_pcm: np.ndarray,
) -> list[CascadeEvent]:
    """Pass a separate energy reference only to engines that implement it."""

    push_pcm = getattr(engine, "push_pcm")
    try:
        supports_energy_reference = "energy_pcm" in inspect.signature(push_pcm).parameters
    except (TypeError, ValueError):
        supports_energy_reference = False
    if supports_energy_reference:
        return push_pcm(pcm, captured_ns, energy_pcm=energy_pcm)
    return push_pcm(pcm, captured_ns)


class _TimelineClock:
    """Assign a monotonic sample-time axis to retained PCM."""

    def __init__(self) -> None:
        self._end_ns: int | None = None

    def now_ns(self, observed_ns: int) -> int:
        self._end_ns = observed_ns if self._end_ns is None else max(observed_ns, self._end_ns)
        return self._end_ns

    def assign_end_ns(self, *, observed_ns: int, sample_count: int, sample_rate: int) -> int:
        duration_ns = sample_count * 1_000_000_000 // sample_rate
        self._end_ns = observed_ns if self._end_ns is None else self._end_ns + duration_ns
        return self._end_ns

    def append_end_ns(self, *, sample_count: int, sample_rate: int) -> int:
        """Advance an offline analysis timeline by PCM duration only."""

        duration_ns = sample_count * 1_000_000_000 // sample_rate
        self._end_ns = duration_ns if self._end_ns is None else self._end_ns + duration_ns
        return self._end_ns


class _RawTriggerAudioHistory:
    """Retain raw PCM until the KWS worker associates it with a submitted window."""

    def __init__(self, *, sample_rate: int, window_samples: int, history_samples: int) -> None:
        if sample_rate <= 0 or window_samples <= 0 or history_samples < window_samples:
            raise ValueError("raw trigger audio history has invalid dimensions")
        self._sample_rate = sample_rate
        self._window_samples = window_samples
        self._capacity = history_samples
        self._pcm = np.empty(0, dtype=np.float32)
        self._start_ns: int | None = None
        self._end_ns: int | None = None
        self._submitted_windows: dict[int, np.ndarray] = {}

    def append(self, pcm: np.ndarray, *, captured_ns: int) -> None:
        values = np.asarray(pcm)
        if values.dtype != np.float32 or values.ndim != 1 or values.size == 0:
            raise ValueError("raw trigger PCM must be a nonempty float32 mono array")
        if self._end_ns is not None and captured_ns < self._end_ns:
            raise ValueError("raw trigger PCM timestamps must be monotonic")
        self._pcm = np.concatenate((self._pcm, values))[-self._capacity :].copy()
        self._end_ns = captured_ns
        self._start_ns = captured_ns - self._pcm.size * 1_000_000_000 // self._sample_rate

    def capture_kws_window(self, captured_ns: int) -> None:
        waveform = self._window_ending_at(captured_ns)
        if waveform is not None:
            self._submitted_windows[captured_ns] = waveform

    def take_window(self, captured_ns: int) -> np.ndarray | None:
        waveform = self._submitted_windows.pop(captured_ns, None)
        return waveform if waveform is not None else self._window_ending_at(captured_ns)

    def discard_window(self, captured_ns: int) -> None:
        self._submitted_windows.pop(captured_ns, None)

    def discard_submitted_windows(self) -> None:
        """Release snapshots for a superseded KWS session without losing live PCM."""

        self._submitted_windows.clear()

    def reset(self) -> None:
        self._pcm = np.empty(0, dtype=np.float32)
        self._start_ns = None
        self._end_ns = None
        self._submitted_windows.clear()

    def _window_ending_at(self, captured_ns: int) -> np.ndarray | None:
        if (
            self._start_ns is None
            or self._end_ns is None
            or captured_ns < self._start_ns
            or captured_ns > self._end_ns
        ):
            return None
        end_index = (captured_ns - self._start_ns) * self._sample_rate // 1_000_000_000
        end_index = min(self._pcm.size, max(0, int(end_index)))
        start_index = max(0, end_index - self._window_samples)
        waveform = self._pcm[start_index:end_index].copy()
        return waveform if waveform.size else None


class _RawWindowCapturingExecutor:
    """Attach raw PCM snapshots to KWS job times without changing CascadeEngine inputs."""

    def __init__(self, delegate: RealtimeKwsExecutor, raw_history: _RawTriggerAudioHistory) -> None:
        self._delegate = delegate
        self._raw_history = raw_history

    def submit(self, job: KwsJob) -> None:
        self._raw_history.capture_kws_window(
            job.captured_ns if job.window_end_ns is None else job.window_end_ns
        )
        self._delegate.submit(job)

    def collect_ordered(self, *, generation: int) -> list[KwsResult]:
        return self._delegate.collect_ordered(generation=generation)

    def collect_phase_confirmations(self, *, generation: int) -> list[KwsResult]:
        return self._delegate.collect_phase_confirmations(generation=generation)

    def invalidate(self, *, generation: int) -> None:
        self._delegate.invalidate(generation=generation)

    def discard_submitted_windows(self) -> None:
        self._raw_history.discard_submitted_windows()

    def reset_history(self) -> None:
        self._raw_history.reset()

    def wait_for_generation(self, *, generation: int, timeout: float) -> bool:
        return self._delegate.wait_for_generation(generation=generation, timeout=timeout)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Windows VAD-gated v6.1 KWS terminal demo")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("devices", help="list input-capable audio devices")

    listen = subparsers.add_parser("listen", help="listen from a microphone or replay a WAV")
    listen.add_argument("--vad-model", type=Path, required=True)
    listen.add_argument("--kws-model", type=Path, help="certified full exact KWS ONNX model")
    listen.add_argument("--kws-frontend", type=Path, help="certified strict-MFCC frontend ONNX")
    listen.add_argument("--kws-backbone", type=Path, help="certified UINT8-code KWS backbone ONNX")
    listen.add_argument("--kws-split-report", type=Path, help="certification report for the split KWS pair")
    listen.add_argument(
        "--kws-frame-repair",
        type=Path,
        help="legacy 96 ms cached-KWS artifact; unavailable with runtime scheduling",
    )
    listen.add_argument(
        "--kws-frame-repair-report",
        type=Path,
        help="legacy 96 ms cached-KWS report; unavailable with runtime scheduling",
    )
    input_group = listen.add_mutually_exclusive_group()
    input_group.add_argument("--input", choices=["mic"], help="live microphone input (default)")
    input_group.add_argument("--input-wav", type=Path, help="replay WAV/decoded audio input in real time")
    listen.add_argument(
        "--controller-profile",
        choices=["legacy-pc", "pc-deployment"],
        default="legacy-pc",
        help="controller semantics profile",
    )
    listen.add_argument("--device", help="Windows input device index or exact headset/microphone name")
    listen.add_argument("--session-root", type=Path, default=Path("sessions"))
    listen.add_argument("--chunk-ms", type=int, default=100)
    listen.add_argument("--kws-positive-index", type=int, default=0)
    listen.add_argument("--queue-capacity", type=int, default=32)
    listen.add_argument(
        "--record-telemetry",
        action="store_true",
        help="write durable session telemetry and artifacts",
    )
    listen.add_argument("--save-trigger-audio", action="store_true")
    listen.add_argument("--web", action="store_true", help="serve the localhost realtime diagnostic dashboard")
    listen.add_argument("--web-history-seconds", type=int, default=120)
    listen.add_argument("--web-port", type=int, default=19374)
    return parser


def _select_kws_mode(args: argparse.Namespace) -> str:
    """Require either the full exact model or the complete certified split set."""

    split_values = (args.kws_frontend, args.kws_backbone, args.kws_split_report)
    has_split_value = any(value is not None for value in split_values)
    has_complete_split = all(value is not None for value in split_values)
    repair_values = (args.kws_frame_repair, args.kws_frame_repair_report)
    has_repair_value = any(value is not None for value in repair_values)
    if has_repair_value:
        raise ValueError(
            "KWS frame-repair cache is certified only for the prior 96 ms schedule and is unavailable with runtime scheduling"
        )
    if args.kws_model is not None and (has_split_value or has_repair_value):
        raise ValueError("select either --kws-model or the certified KWS split artifacts")
    if args.kws_model is not None:
        return "full"
    if has_split_value:
        if not has_complete_split:
            raise ValueError(
                "KWS split mode requires all three of --kws-frontend, --kws-backbone, and --kws-split-report"
            )
        return "split"
    raise ValueError("provide either --kws-model or all three certified KWS split artifacts")


def _one_thread_session_options() -> ort.SessionOptions:
    """Avoid nested ONNX CPU pools when VAD and KWS run concurrently."""

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return options


def _kws_runner_factory(
    *,
    kws_mode: str,
    kws_contract: KwsContract | None,
    kws_split_contract: KwsSplitContract | None,
    positive_index: int,
) -> Callable[[], KwsOnnxRunner | KwsSplitOnnxRunner]:
    """Create an independent, one-thread KWS runner for one executor worker."""

    if kws_mode == "full":
        if kws_contract is None or kws_split_contract is not None:
            raise ValueError("full KWS factory requires exactly one full KWS contract")

        def create_full() -> KwsOnnxRunner:
            session = ort.InferenceSession(
                str(kws_contract.model_path),
                sess_options=_one_thread_session_options(),
                providers=["CPUExecutionProvider"],
            )
            return KwsOnnxRunner(session, kws_contract, positive_index)

        return create_full

    if kws_mode == "split":
        if kws_split_contract is None or kws_contract is not None:
            raise ValueError("split KWS factory requires exactly one split KWS contract")

        def create_split() -> KwsSplitOnnxRunner:
            frontend_session = ort.InferenceSession(
                str(kws_split_contract.frontend_model_path),
                sess_options=_one_thread_session_options(),
                providers=["CPUExecutionProvider"],
            )
            backbone_session = ort.InferenceSession(
                str(kws_split_contract.backbone_model_path),
                sess_options=_one_thread_session_options(),
                providers=["CPUExecutionProvider"],
            )
            return KwsSplitOnnxRunner(
                frontend_session=frontend_session,
                backbone_session=backbone_session,
                contract=kws_split_contract,
                positive_index=positive_index,
            )

        return create_split

    raise ValueError(f"unsupported KWS mode: {kws_mode}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "devices":
        return _list_devices()
    return _listen(args)


def console_safe_text(value: object, *, encoding: str | None = None) -> str:
    """Render terminal text without failing on legacy Windows code pages."""

    text = str(value)
    target_encoding = encoding or sys.stdout.encoding or "utf-8"
    return text.encode(target_encoding, errors="backslashreplace").decode(target_encoding)


def _list_devices() -> int:
    try:
        devices = list_input_devices()
    except AudioSourceError as error:
        print(console_safe_text(f"audio device error: {error}", encoding=sys.stderr.encoding), file=sys.stderr)
        return 1
    for device in devices:
        print(
            f"{device.index}: {console_safe_text(device.name)} "
            f"({device.max_input_channels} input channel(s), {device.default_sample_rate:.0f} Hz)"
        )
    return 0


def _listen(args: argparse.Namespace) -> int:
    runtime_profile = _controller_runtime_profile(args.controller_profile)
    if runtime_profile.controller_kind == "legacy_cascade":
        config: CascadeConfig | _ObservedControlRuntimeSettings = runtime_profile.cascade_config(
            kws_positive_index=args.kws_positive_index,
            queue_capacity=args.queue_capacity,
        )
    else:
        config = _observed_control_runtime_settings(
            runtime_profile,
            kws_positive_index=args.kws_positive_index,
            queue_capacity=args.queue_capacity,
        )
    logger: SessionLogger | NullSessionLogger | None = None
    session_path: Path | None = None
    microphone: MicrophoneSource | None = None
    dashboard: RealtimeDashboardServer | None = None
    diagnostic_store: RealtimeDiagnosticStore | None = None
    device_control: RealtimeDeviceControl | None = None
    capture_mode_control: RealtimeCaptureModeControl | None = None
    analysis_control: RealtimeAnalysisControl | None = None
    temporary_vad_export: tempfile.TemporaryDirectory[str] | None = None
    stateful_vad: object | None = None
    vad_session: ort.InferenceSession | None = None
    kws_session: ort.InferenceSession | None = None
    frontend_session: ort.InferenceSession | None = None
    backbone_session: ort.InferenceSession | None = None
    kws_runner: KwsOnnxRunner | KwsSplitOnnxRunner | None = None
    kws_executor: RealtimeKwsExecutor | None = None
    engine: CascadeEngine | ObservedControlEngine | None = None
    capture_control = RealtimeCaptureControl()
    threshold_control = RealtimeThresholdControl(
        vad_threshold=config.vad_threshold,
        kws_threshold=config.kws_threshold,
        vad_period_ms=config.vad_period_ms,
        kws_period_ms=config.kws_period_ms,
        kws_lookback_ms=config.kws_lookback_ms,
    )
    input_control = RealtimeInputControl(
        conditioner_enabled=runtime_profile.frontend_profile is not FrontendProfile.RAW,
        target_rms_dbfs=runtime_profile.default_target_rms_dbfs,
    )
    try:
        if isinstance(config, CascadeConfig):
            config.validate()
        if args.save_trigger_audio and not args.record_telemetry:
            raise ValueError("--save-trigger-audio requires --record-telemetry")
        kws_mode = _select_kws_mode(args)
        if args.chunk_ms <= 0:
            raise ValueError("chunk_ms must be positive")
        if args.web and not 1 <= args.web_history_seconds <= 3600:
            raise ValueError("web_history_seconds must be in [1, 3600]")
        if args.web and not 0 <= args.web_port <= 65535:
            raise ValueError("web_port must be in [0, 65535]")
        vad_contract = load_vad_contract(args.vad_model)
        if kws_mode == "full":
            kws_contract = validate_kws_model(args.kws_model)
            kws_split_contract = None
            kws_manifest: dict[str, object] = {
                "mode": "full",
                "path": str(kws_contract.model_path),
                "sha256": kws_contract.model_sha256,
            }
        else:
            kws_contract = None
            kws_split_contract = validate_kws_split_model(
                args.kws_frontend, args.kws_backbone, args.kws_split_report
            )
            kws_manifest = {
                "mode": kws_mode,
                "frontend": {
                    "path": str(kws_split_contract.frontend_model_path),
                    "sha256": kws_split_contract.frontend_model_sha256,
                },
                "backbone": {
                    "path": str(kws_split_contract.backbone_model_path),
                    "sha256": kws_split_contract.backbone_model_sha256,
                },
                "report": str(kws_split_contract.report_path),
                "checkpoint_sha256": kws_split_contract.checkpoint_sha256,
                "spec_sha256": kws_split_contract.spec_sha256,
                "source_full_onnx_sha256": kws_split_contract.source_full_onnx_sha256,
                "parity_samples": kws_split_contract.parity_samples,
            }
            kws_manifest["cache"] = {"mode": "disabled"}
        input_mode = "wav" if args.input_wav is not None else "mic"
        if args.input_wav is not None:
            resolved_input: dict[str, object] = {"path": str(args.input_wav.resolve())}
            source_label = args.input_wav.name
        else:
            selected_device = resolve_input_device(args.device)
            resolved_input = {
                "index": selected_device.index,
                "name": selected_device.name,
                "default_sample_rate": selected_device.default_sample_rate,
                "max_input_channels": selected_device.max_input_channels,
            }
            source_label = f"{selected_device.index}:{selected_device.name}"
            if args.web:
                dashboard_devices = list_input_devices(wasapi_only=True)
                if not any(device.index == selected_device.index for device in dashboard_devices):
                    dashboard_devices.append(selected_device)
                device_control = RealtimeDeviceControl(
                    devices=[
                        {
                            "index": device.index,
                            "name": device.name,
                            "default_sample_rate": device.default_sample_rate,
                            "max_input_channels": device.max_input_channels,
                        }
                        for device in dashboard_devices
                    ],
                    selected_index=selected_device.index,
                )
                capture_mode_control = RealtimeCaptureModeControl(
                    initial_mode=CaptureMode.EXCLUSIVE.value
                )
        controller_manifest = {
            "profile": args.controller_profile,
            "control_contract": (
                runtime_profile.control_contract.as_dict()
                if runtime_profile.control_contract is not None
                else None
            ),
            "wake_lifecycle_policy": runtime_profile.wake_lifecycle_policy.value,
            "wake_lifecycle_source": "configured_pc_session_lifecycle",
            "pcm_continuity_policy": (
                "pc_zero_fill"
                if runtime_profile.controller_kind == "observed_control"
                else "reset_on_discontinuity"
            ),
        }
        if args.record_telemetry:
            logger = SessionLogger.create(
                args.session_root,
                manifest={
                    "providers": ["CPUExecutionProvider"],
                    "vad": {
                        "path": str(vad_contract.model_path),
                        "sha256": vad_contract.model_sha256,
                        "model_id": "causal-crnn-vad-kws-realneg",
                    },
                    "kws": kws_manifest,
                    "configuration": asdict(config),
                    "controller": controller_manifest,
                    "dashboard": {
                        "enabled": args.web,
                        "history_seconds": args.web_history_seconds if args.web else None,
                        "bind_host": "127.0.0.1" if args.web else None,
                        "requested_port": args.web_port if args.web else None,
                    },
                    "input": {"mode": input_mode, "selector": args.device, "resolved": resolved_input},
                    "environment": {
                        "python_version": sys.version,
                        "platform": platform.platform(),
                        "numpy_version": np.__version__,
                        "onnxruntime_version": ort.__version__,
                    },
                },
            )
            session_path = logger.session_path
            stateful_vad_path = session_path / "vad_stateful.onnx"
        else:
            logger = NullSessionLogger()
            temporary_vad_export = tempfile.TemporaryDirectory(prefix="vad-kws-stateful-")
            stateful_vad_path = Path(temporary_vad_export.name) / "vad_stateful.onnx"
        stateful_vad = export_stateful_vad(vad_contract.model_path, stateful_vad_path)
        stateful_vad_sha256 = sha256_file(stateful_vad.output_path) if args.record_telemetry else None
        vad_session = ort.InferenceSession(
            str(stateful_vad.output_path),
            sess_options=_one_thread_session_options(),
            providers=["CPUExecutionProvider"],
        )
        if kws_mode == "full":
            assert kws_contract is not None
            kws_session = ort.InferenceSession(
                str(kws_contract.model_path),
                sess_options=_one_thread_session_options(),
                providers=["CPUExecutionProvider"],
            )
            kws_runner = KwsOnnxRunner(kws_session, kws_contract, config.kws_positive_index)
        else:
            assert kws_split_contract is not None
            frontend_session = ort.InferenceSession(
                str(kws_split_contract.frontend_model_path),
                sess_options=_one_thread_session_options(),
                providers=["CPUExecutionProvider"],
            )
            backbone_session = ort.InferenceSession(
                str(kws_split_contract.backbone_model_path),
                sess_options=_one_thread_session_options(),
                providers=["CPUExecutionProvider"],
            )
            kws_runner = KwsSplitOnnxRunner(
                frontend_session=frontend_session,
                backbone_session=backbone_session,
                contract=kws_split_contract,
                positive_index=config.kws_positive_index,
            )
        raw_trigger_audio: _RawTriggerAudioHistory | None = None
        if args.record_telemetry and args.save_trigger_audio:
            window_samples = config.sample_rate_hz * config.window_ms // 1000
            chunk_samples = max(1, round(config.sample_rate_hz * args.chunk_ms / 1000))
            lookback_ms = getattr(config, "kws_lookback_ms", config.window_ms)
            lookback_samples = config.sample_rate_hz * lookback_ms // 1000
            raw_trigger_audio = _RawTriggerAudioHistory(
                sample_rate=config.sample_rate_hz,
                window_samples=window_samples,
                history_samples=max(lookback_samples, window_samples + chunk_samples + 1),
            )
        if (
            runtime_profile.controller_kind == "observed_control"
            or (args.input_wav is None and runtime_profile.controller_kind == "legacy_cascade")
        ):
            history_windows = (
                math.ceil(
                    max(0, config.kws_lookback_ms - config.window_ms) / config.kws_period_ms
                )
                + 1
            )
            kws_executor = RealtimeKwsExecutor(
                _kws_runner_factory(
                    kws_mode=kws_mode,
                    kws_contract=kws_contract,
                    kws_split_contract=kws_split_contract,
                    positive_index=config.kws_positive_index,
                ),
                worker_count=2,
                queue_capacity=max(8, history_windows),
            )
        engine_kws_executor = (
            _RawWindowCapturingExecutor(kws_executor, raw_trigger_audio)
            if kws_executor is not None and raw_trigger_audio is not None
            else kws_executor
        )
        vad_runner = StatefulVadOnnxRunner(vad_session, n_mels=vad_contract.n_mels)
        vad_feature_cache = StreamingVadFeatureCache(vad_contract)
        if runtime_profile.controller_kind == "observed_control":
            contract = runtime_profile.control_contract
            if contract is None:
                raise ValueError("PC deployment control profile requires a control contract")
            engine = ObservedControlEngine(
                contract=contract,
                vad_contract=vad_contract,
                vad_runner=vad_runner,
                kws_runner=kws_runner,
                vad_feature_cache=vad_feature_cache,
                kws_executor=engine_kws_executor,
                kws_lookback_ms=config.kws_lookback_ms,
                runtime_config=ObservedControlRuntimeConfig.create(
                    contract=contract,
                    vad_threshold=config.vad_threshold,
                    kws_threshold=config.kws_threshold,
                    energy_threshold_dbfs=runtime_profile.energy_threshold_dbfs,
                    energy_enabled=config.energy_enabled,
                    vad_enabled=config.vad_enabled,
                    vad_period_ms=config.vad_period_ms,
                    kws_period_ms=config.kws_period_ms,
                ),
            )
        else:
            assert isinstance(config, CascadeConfig)
            engine = CascadeEngine(
                config=config,
                vad_contract=vad_contract,
                vad_runner=vad_runner,
                kws_runner=kws_runner,
                kws_executor=engine_kws_executor,
                vad_feature_cache=vad_feature_cache,
            )
        logger.record(
            "vad_stateful_export",
            source_path=str(stateful_vad.source_path),
            source_sha256=vad_contract.model_sha256,
            output_path=str(stateful_vad.output_path),
            output_sha256=stateful_vad_sha256,
            cnn_context_shape=list(stateful_vad.layout.context_shape),
            gru_hidden_shape=[1, 1, stateful_vad.layout.gru_hidden_size],
        )
        logger.record("kws_runtime", **kws_manifest)
        controls = _CommandReader()
        controls.start()
        stop_requested = threading.Event()
        last_drop_count = 0
        latest_end_to_end_ms: float | None = None
        expected_sequence: int | None = None
        latest_pcm_end_ns: int | None = None
        initial_input_config = input_control.snapshot()
        realtime_input_pipeline = RealtimeInputPipeline.from_profile(
            runtime_profile,
            target_rms_dbfs=float(initial_input_config["target_rms_dbfs"]),
            conditioner_enabled=bool(initial_input_config["conditioner_enabled"]),
        )
        replay_pcm_converter = StreamingChannelConverter()
        timeline_clock = _TimelineClock()
        runtime_status: dict[str, object] = {
            "queue_depth": 0,
            "drop_count": 0,
            "end_to_end_ms": None,
            "capture_paused": False,
        }
        microphone_queue: BoundedAudioQueue | None = None
        applied_capture_generation = int(capture_control.snapshot()["generation"])
        applied_threshold_generation = int(threshold_control.snapshot()["generation"])
        applied_input_generation = int(input_control.snapshot()["generation"])
        if args.web:
            diagnostic_store = RealtimeDiagnosticStore(
                history_seconds=args.web_history_seconds,
                vad_threshold=config.vad_threshold,
                kws_threshold=config.kws_threshold,
                vad_confirmations=config.vad_confirmations,
                kws_confirmations=config.kws_confirmations,
                vad_period_ms=config.vad_period_ms,
                kws_period_ms=config.kws_period_ms,
                kws_lookback_ms=config.kws_lookback_ms,
                input_profile=runtime_profile,
            )
            set_model_identity = getattr(diagnostic_store, "set_model_identity", None)
            if callable(set_model_identity):
                if kws_contract is not None:
                    set_model_identity(
                        kws_model_path=str(kws_contract.model_path),
                        kws_model_sha256=kws_contract.model_sha256,
                    )
                else:
                    assert kws_split_contract is not None
                    set_model_identity(
                        kws_model_path=(
                            f"split:{kws_split_contract.frontend_model_path}|"
                            f"{kws_split_contract.backbone_model_path}"
                        ),
                        kws_model_sha256=kws_split_contract.source_full_onnx_sha256,
                    )
            analysis_control = RealtimeAnalysisControl(
                maximum_samples=diagnostic_store.capacity_samples,
            )
            dashboard = RealtimeDashboardServer(
                diagnostic_store,
                control=capture_control,
                threshold_control=threshold_control,
                input_control=input_control,
                analysis_control=analysis_control,
                port=args.web_port,
            )
            if runtime_profile.controller_kind == "observed_control":
                configure_runtime_config_persistor = getattr(
                    dashboard, "set_runtime_config_persistor", None
                )
                if callable(configure_runtime_config_persistor):
                    configure_runtime_config_persistor(save_pc_deployment_runtime_config)
            configure_device_control = getattr(dashboard, "set_device_control", None)
            if callable(configure_device_control):
                configure_device_control(device_control)
            configure_capture_mode_control = getattr(dashboard, "set_capture_mode_control", None)
            if callable(configure_capture_mode_control):
                configure_capture_mode_control(capture_mode_control)
            dashboard.start()
            logger.record(
                "dashboard_start",
                url=dashboard.url,
                history_seconds=args.web_history_seconds,
                bind_host="127.0.0.1",
            )
            print(f"dashboard: {dashboard.url}")

        def apply_threshold_control() -> None:
            nonlocal applied_threshold_generation, expected_sequence, last_drop_count
            snapshot = threshold_control.snapshot()
            generation = int(snapshot["generation"])
            if generation == applied_threshold_generation:
                return
            if runtime_profile.controller_kind == "observed_control":
                _apply_observed_runtime_config_transition(
                    snapshot,
                    engine=engine,
                    logger=logger,
                    diagnostic_store=diagnostic_store,
                    runtime_status=runtime_status,
                    captured_ns=timeline_clock.now_ns(time.perf_counter_ns()),
                )
            else:
                _apply_runtime_config_transition(
                    snapshot,
                    engine=engine,
                    logger=logger,
                    diagnostic_store=diagnostic_store,
                    runtime_status=runtime_status,
                    captured_ns=timeline_clock.now_ns(time.perf_counter_ns()),
                )
            expected_sequence = None
            last_drop_count = microphone_queue.drop_count if microphone_queue is not None else 0
            applied_threshold_generation = generation

        def publish_input_status(input_frame: object | None = None) -> None:
            if diagnostic_store is None:
                return
            set_input_status = getattr(diagnostic_store, "set_input_status", None)
            if not callable(set_input_status):
                return
            if runtime_profile.frontend_profile is FrontendProfile.RAW:
                conditioner_enabled = False
                target_rms_dbfs = runtime_profile.default_target_rms_dbfs
            else:
                requested = input_control.snapshot()
                conditioner_enabled = requested["conditioner_enabled"]
                target_rms_dbfs = requested["target_rms_dbfs"]
            status: dict[str, object] = {
                "conditioner_enabled": conditioner_enabled,
                "target_rms_dbfs": target_rms_dbfs,
            }
            if microphone is not None:
                capture_mode = getattr(microphone, "capture_mode", None)
                if capture_mode is not None:
                    status["capture_mode"] = capture_mode
                capability = getattr(microphone, "capability", None)
                if capability is not None:
                    status["capture_format_reason"] = capability.fallback_reason
            if input_frame is None:
                status.update(
                    {
                        "input_gain_db": 0.0,
                    }
                )
            else:
                conditioning = getattr(input_frame, "conditioning")
                status.update(
                    {
                        "input_gain_db": conditioning.applied_gain_db,
                        "input_peak_dbfs": conditioning.conditioned_peak_dbfs,
                        "noise_floor_dbfs": conditioning.noise_floor_dbfs,
                        "input_raw_rms_dbfs": getattr(conditioning, "raw_rms_dbfs", None),
                        "input_conditioned_rms_dbfs": getattr(
                            conditioning, "conditioned_rms_dbfs", None
                        ),
                        "input_control_rms_dbfs": getattr(conditioning, "control_rms_dbfs", None),
                        "input_noise_floor_calibrated": getattr(
                            conditioning, "noise_floor_calibrated", None
                        ),
                        "input_speech_detected": getattr(conditioning, "speech_detected", None),
                    }
                )
            if kws_executor is not None:
                executor_snapshot = getattr(kws_executor, "snapshot", None)
                if callable(executor_snapshot):
                    executor = executor_snapshot()
                    status.update(
                        {
                            "kws_queue": executor["queued"],
                            "kws_overload_count": executor["overload_count"],
                            "kws_stale_count": executor["stale_count"],
                            "kws_inference_ms": executor["last_inference_ms"],
                        }
                    )
            set_input_status(status)

        def apply_input_control() -> None:
            nonlocal applied_input_generation, expected_sequence, last_drop_count, microphone, source_label, latest_end_to_end_ms
            snapshot = input_control.snapshot()
            generation = int(snapshot["generation"])
            if generation == applied_input_generation:
                return
            if args.input_wav is not None:
                logger.record(
                    "input_config_ignored",
                    profile=args.controller_profile,
                    reason="offline_replay_input_is_fixed",
                    conditioner_enabled=bool(snapshot["conditioner_enabled"]),
                    target_rms_dbfs=float(snapshot["target_rms_dbfs"]),
                    generation=generation,
                )
                applied_input_generation = generation
                publish_input_status()
                return
            if runtime_profile.frontend_profile is FrontendProfile.RAW:
                logger.record(
                    "input_config_ignored",
                    profile=args.controller_profile,
                    reason="raw_pcm_contract",
                    conditioner_enabled=bool(snapshot["conditioner_enabled"]),
                    target_rms_dbfs=float(snapshot["target_rms_dbfs"]),
                    generation=generation,
                )
                applied_input_generation = generation
                publish_input_status()
                return
            realtime_input_pipeline.set_config(
                conditioner_enabled=bool(snapshot["conditioner_enabled"]),
                target_rms_dbfs=float(snapshot["target_rms_dbfs"]),
            )
            replay_pcm_converter.reset()
            if raw_trigger_audio is not None:
                raw_trigger_audio.reset()
            captured_ns = timeline_clock.now_ns(time.perf_counter_ns())
            logger.record(
                "input_config_applied",
                conditioner_enabled=snapshot["conditioner_enabled"],
                target_rms_dbfs=snapshot["target_rms_dbfs"],
                    generation=generation,
                    reset=True,
            )
            for event in engine.reset(captured_ns):
                _record_event(logger, event, captured_ns, time.perf_counter_ns())
                _record_diagnostic_event(diagnostic_store, event, engine, runtime_status)
            expected_sequence = None
            last_drop_count = microphone_queue.drop_count if microphone_queue is not None else 0
            applied_input_generation = generation
            publish_input_status()

        def apply_capture_control() -> bool:
            nonlocal applied_capture_generation, expected_sequence, last_drop_count
            snapshot = capture_control.snapshot()
            generation = int(snapshot["generation"])
            if generation == applied_capture_generation:
                return bool(snapshot["paused"])
            paused = bool(snapshot["paused"])
            _apply_capture_transition(
                paused,
                microphone=microphone,
                microphone_queue=microphone_queue,
                pcm_converter=realtime_input_pipeline,
                additional_pcm_converter=replay_pcm_converter,
                engine=engine,
                logger=logger,
                diagnostic_store=diagnostic_store,
                runtime_status=runtime_status,
                captured_ns=timeline_clock.now_ns(time.perf_counter_ns()),
            )
            if not paused and raw_trigger_audio is not None:
                raw_trigger_audio.reset()
            expected_sequence = None
            last_drop_count = microphone_queue.drop_count if microphone_queue is not None else 0
            applied_capture_generation = generation
            return paused

        def apply_input_device_control() -> None:
            nonlocal microphone, source_label, expected_sequence, last_drop_count, latest_end_to_end_ms
            if device_control is None or microphone is None or microphone_queue is None:
                return
            pending = device_control.claim_pending()
            if pending is None:
                return
            requested_index = int(pending["requested_index"])
            generation = int(pending["generation"])
            captured_ns = timeline_clock.now_ns(time.perf_counter_ns())
            try:
                microphone, source_label = _apply_input_device_switch(
                    requested_index,
                    microphone=microphone,
                    microphone_queue=microphone_queue,
                    chunk_ms=args.chunk_ms,
                    capture_mode=CaptureMode(getattr(microphone, "capture_mode", "exclusive")),
                    pcm_converter=realtime_input_pipeline,
                    additional_pcm_converter=replay_pcm_converter,
                    raw_trigger_audio=raw_trigger_audio,
                    engine=engine,
                    logger=logger,
                    diagnostic_store=diagnostic_store,
                    runtime_status=runtime_status,
                    captured_ns=captured_ns,
                    generation=generation,
                )
                if bool(capture_control.snapshot()["paused"]):
                    microphone.pause()
                    microphone_queue.clear()
                    runtime_status["queue_depth"] = 0
                device_control.mark_applied(requested_index, generation=generation)
                expected_sequence = None
                last_drop_count = 0
                latest_end_to_end_ms = None
            except (AudioSourceError, OSError, ValueError, RuntimeError) as error:
                message = str(error) or type(error).__name__
                device_control.mark_failed(requested_index, generation=generation, error=message)
                logger.record(
                    "input_device_switch_error",
                    requested_index=requested_index,
                    generation=generation,
                    message=message,
                )
            if diagnostic_store is not None:
                diagnostic_store.notify_update()

        def apply_capture_mode_control() -> None:
            nonlocal microphone, source_label, expected_sequence, last_drop_count
            nonlocal latest_end_to_end_ms
            if capture_mode_control is None or microphone is None or microphone_queue is None:
                return
            pending = capture_mode_control.claim_pending()
            if pending is None:
                return
            requested_mode = CaptureMode(str(pending["requested_mode"]))
            generation = int(pending["generation"])
            current_device = microphone.device
            if current_device is None:
                capture_mode_control.mark_failed(
                    requested_mode.value,
                    generation=generation,
                    error="microphone device is not open",
                )
                return
            captured_ns = timeline_clock.now_ns(time.perf_counter_ns())
            try:
                microphone, source_label = _apply_capture_mode_switch(
                    requested_mode,
                    microphone=microphone,
                    microphone_queue=microphone_queue,
                    chunk_ms=args.chunk_ms,
                    pcm_converter=realtime_input_pipeline,
                    additional_pcm_converter=replay_pcm_converter,
                    raw_trigger_audio=raw_trigger_audio,
                    engine=engine,
                    logger=logger,
                    diagnostic_store=diagnostic_store,
                    runtime_status=runtime_status,
                    captured_ns=captured_ns,
                    generation=generation,
                )
                if bool(capture_control.snapshot()["paused"]):
                    microphone.pause()
                    microphone_queue.clear()
                capture_mode_control.mark_applied(requested_mode.value, generation=generation)
                expected_sequence = None
                last_drop_count = 0
                latest_end_to_end_ms = None
                publish_input_status()
            except (AudioSourceError, OSError, ValueError, RuntimeError) as error:
                message = str(error) or type(error).__name__
                capture_mode_control.mark_failed(
                    requested_mode.value,
                    generation=generation,
                    error=message,
                )
                logger.record(
                    "capture_mode_switch_error",
                    requested_mode=requested_mode.value,
                    generation=generation,
                    message=message,
                )
            if diagnostic_store is not None:
                diagnostic_store.notify_update()

        def rearm_after_wake(captured_ns: int) -> None:
            """Reset recognition state without breaking the continuous PCM timeline."""

            nonlocal last_drop_count, latest_end_to_end_ms
            retained_microphone_chunks = 0
            drop_count_before = last_drop_count
            drop_count_after = last_drop_count
            if microphone_queue is not None:
                drop_count_before = microphone_queue.drop_count
                retained_microphone_chunks = microphone_queue.qsize()
                drop_count_after = microphone_queue.drop_count
                last_drop_count = drop_count_after
            latest_end_to_end_ms = None
            if raw_trigger_audio is not None:
                raw_trigger_audio.discard_submitted_windows()
            runtime_status["queue_depth"] = (
                microphone_queue.qsize() if microphone_queue is not None else 0
            )
            runtime_status["drop_count"] = last_drop_count
            runtime_status["end_to_end_ms"] = latest_end_to_end_ms
            logger.record(
                "session_rearm",
                policy=runtime_profile.wake_lifecycle_policy.value,
                source="configured_pc_session_lifecycle",
                discarded_microphone_chunks=0,
                retained_microphone_chunks=retained_microphone_chunks,
                discarded_chunks_counted_as_drops=False,
                drop_count_before=drop_count_before,
                drop_count_after=drop_count_after,
            )
            reset_started_ns = time.perf_counter_ns()
            rearm = getattr(engine, "rearm", None)
            reset_events = rearm(captured_ns) if callable(rearm) else engine.reset(captured_ns)
            for reset_event in reset_events:
                _record_event(logger, reset_event, reset_started_ns, time.perf_counter_ns())
                _record_diagnostic_event(diagnostic_store, reset_event, engine, runtime_status)

        def record_engine_events(
            events: list[CascadeEvent],
            *,
            processing_started_ns: int,
            persist_trigger_audio: bool,
        ) -> bool:
            nonlocal latest_end_to_end_ms
            wake_window_ends = {
                int(event.fields.get("window_end_ns", event.captured_ns))
                for event in events
                if event.kind == "wake"
            }
            for event in events:
                latest_end_to_end_ms = _record_event(
                    logger, event, processing_started_ns, time.perf_counter_ns()
                )
                _record_diagnostic_event(diagnostic_store, event, engine, runtime_status)
                _render_event(event, engine, last_drop_count)
                if (
                    persist_trigger_audio
                    and event.kind == "wake"
                    and args.record_telemetry
                    and args.save_trigger_audio
                ):
                    assert session_path is not None
                    window_end_ns = int(event.fields.get("window_end_ns", event.captured_ns))
                    waveform = (
                        raw_trigger_audio.take_window(window_end_ns)
                        if raw_trigger_audio is not None
                        else None
                    )
                    if waveform is None:
                        logger.record(
                            "error",
                            source="trigger_audio",
                            message="raw PCM window unavailable for wake event",
                            capture_ns=event.captured_ns,
                            window_end_ns=window_end_ns,
                        )
                    else:
                        _save_trigger_audio(session_path, window_end_ns, waveform)
            if raw_trigger_audio is not None:
                for event in events:
                    window_end_ns = int(event.fields.get("window_end_ns", event.captured_ns))
                    if event.kind in {"kws_call", "kws_error"} and window_end_ns not in wake_window_ends:
                        raw_trigger_audio.discard_window(window_end_ns)
            if (
                wake_window_ends
                and runtime_profile.wake_lifecycle_policy is WakeLifecyclePolicy.AUTO_REARM
            ):
                if latest_pcm_end_ns is None:
                    raise RuntimeError("wake event arrived before any PCM was processed")
                rearm_after_wake(latest_pcm_end_ns)
                return True
            return False

        def drain_ready_kws_results(
            *,
            processing_started_ns: int,
            persist_trigger_audio: bool,
            wait: bool = False,
            timeout: float = 0.0,
        ) -> bool:
            drain = getattr(engine, "drain_kws_results", None)
            if not callable(drain):
                return False
            return record_engine_events(
                drain(wait=wait, timeout=timeout),
                processing_started_ns=processing_started_ns,
                persist_trigger_audio=persist_trigger_audio,
            )

        def process(
            chunk: AudioChunk,
            *,
            persist_trigger_audio: bool = True,
            apply_runtime_controls: bool = True,
            is_analysis_command: bool = False,
        ) -> None:
            nonlocal expected_sequence, last_drop_count, latest_end_to_end_ms, latest_pcm_end_ns
            session_rearmed = False
            if apply_runtime_controls:
                apply_threshold_control()
                apply_input_control()
                if apply_capture_control():
                    return
            input_frame = None
            if not _should_apply_realtime_input_frontend(
                microphone=microphone,
                is_analysis_command=is_analysis_command,
                profile=runtime_profile,
            ):
                pcm = _convert_first_channel_pcm(
                    replay_pcm_converter,
                    chunk.samples,
                    input_sample_rate=chunk.sample_rate,
                )
                raw_mono = pcm
            else:
                input_frame = realtime_input_pipeline.process(
                    chunk.samples, input_sample_rate=chunk.sample_rate
                )
                pcm = input_frame.inference_mono
                raw_mono = input_frame.raw_mono
            if is_analysis_command:
                captured_ns = timeline_clock.append_end_ns(
                    sample_count=int(pcm.size),
                    sample_rate=16000,
                )
                discontinuity = 0
            else:
                observed_drops = (
                    microphone_queue.drop_count - last_drop_count if microphone_queue is not None else 0
                )
                sequence_gap = (
                    0
                    if expected_sequence is None
                    else max(0, chunk.sequence - expected_sequence)
                )
                discontinuity = max(observed_drops, sequence_gap)
                if (
                    discontinuity
                    and runtime_profile.controller_kind == "observed_control"
                    and microphone is not None
                ):
                    missing_samples = discontinuity * int(pcm.size)
                    if missing_samples:
                        concealment_pcm = np.zeros(missing_samples, dtype=np.float32)
                        concealment_captured_ns = timeline_clock.assign_end_ns(
                            observed_ns=chunk.captured_ns,
                            sample_count=missing_samples,
                            sample_rate=16000,
                        )
                        logger.record(
                            "audio_concealment",
                            policy="pc_zero_fill",
                            dropped_chunks=discontinuity,
                            concealed_samples=missing_samples,
                        )
                        if diagnostic_store is not None:
                            diagnostic_store.append_pcm(
                                concealment_pcm,
                                captured_ns=concealment_captured_ns,
                            )
                        if raw_trigger_audio is not None:
                            raw_trigger_audio.append(
                                concealment_pcm,
                                captured_ns=concealment_captured_ns,
                            )
                        concealment_started_ns = time.perf_counter_ns()
                        latest_pcm_end_ns = concealment_captured_ns
                        session_rearmed = record_engine_events(
                            _push_engine_pcm(
                                engine,
                                concealment_pcm,
                                concealment_captured_ns,
                                energy_pcm=concealment_pcm,
                            ),
                            processing_started_ns=concealment_started_ns,
                            persist_trigger_audio=persist_trigger_audio,
                        )
                        session_rearmed = (
                            drain_ready_kws_results(
                            processing_started_ns=concealment_started_ns,
                            persist_trigger_audio=persist_trigger_audio,
                            )
                            or session_rearmed
                        )
                captured_ns = timeline_clock.assign_end_ns(
                    observed_ns=chunk.captured_ns,
                    sample_count=int(pcm.size),
                    sample_rate=16000,
                )
            if chunk.status:
                logger.record("error", source="audio_callback", message=chunk.status)
            if (
                discontinuity
                and runtime_profile.controller_kind == "legacy_cascade"
                and microphone is not None
            ):
                total_dropped = (
                    microphone_queue.drop_count if microphone_queue is not None else discontinuity
                )
                logger.record(
                    "audio_drop", dropped=discontinuity, total_dropped=total_dropped
                )
                reset_started_ns = time.perf_counter_ns()
                for event in engine.reset(captured_ns):
                    _record_event(logger, event, reset_started_ns, time.perf_counter_ns())
                    _record_diagnostic_event(diagnostic_store, event, engine, runtime_status)
                if input_frame is None:
                    replay_pcm_converter.reset()
                else:
                    realtime_input_pipeline.reset()
                if raw_trigger_audio is not None:
                    raw_trigger_audio.reset()
            if not is_analysis_command:
                if microphone_queue is not None:
                    last_drop_count = microphone_queue.drop_count
                expected_sequence = None if session_rearmed else chunk.sequence + 1
            logger.record(
                "audio_chunk",
                sequence=chunk.sequence,
                input_sample_rate=chunk.sample_rate,
                sample_count=int(pcm.size),
            )
            if pcm.size == 0:
                return
            if diagnostic_store is not None and microphone is not None and not is_analysis_command:
                append_raw_device = getattr(diagnostic_store, "append_raw_device", None)
                if callable(append_raw_device):
                    append_raw_device(
                        chunk.raw_samples if chunk.raw_samples is not None else chunk.samples,
                        sample_rate=chunk.sample_rate,
                        captured_ns=captured_ns,
                    )
            if diagnostic_store is not None:
                diagnostic_store.append_pcm(raw_mono, captured_ns=captured_ns)
            if raw_trigger_audio is not None:
                raw_trigger_audio.append(raw_mono, captured_ns=captured_ns)
            processing_started_ns = time.perf_counter_ns()
            latest_pcm_end_ns = captured_ns
            session_rearmed = (
                record_engine_events(
                _push_engine_pcm(
                    engine,
                    pcm,
                    captured_ns,
                    energy_pcm=(input_frame.raw_mono if input_frame is not None else pcm),
                ),
                processing_started_ns=processing_started_ns,
                persist_trigger_audio=persist_trigger_audio,
                )
                or session_rearmed
            )
            session_rearmed = (
                drain_ready_kws_results(
                processing_started_ns=processing_started_ns,
                persist_trigger_audio=persist_trigger_audio,
                )
                or session_rearmed
            )
            if not is_analysis_command and session_rearmed:
                expected_sequence = chunk.sequence + 1
            if input_frame is not None:
                realtime_input_pipeline.update_vad_confidence(engine.latest_vad_score)
                runtime_status["input_gain_db"] = input_frame.conditioning.applied_gain_db
                runtime_status["input_peak_dbfs"] = input_frame.conditioning.conditioned_peak_dbfs
                runtime_status["noise_floor_dbfs"] = input_frame.conditioning.noise_floor_dbfs
                runtime_status["input_raw_rms_dbfs"] = getattr(
                    input_frame.conditioning, "raw_rms_dbfs", None
                )
                runtime_status["input_conditioned_rms_dbfs"] = (
                    getattr(input_frame.conditioning, "conditioned_rms_dbfs", None)
                )
                runtime_status["input_control_rms_dbfs"] = getattr(
                    input_frame.conditioning, "control_rms_dbfs", None
                )
                runtime_status["input_noise_floor_calibrated"] = (
                    getattr(input_frame.conditioning, "noise_floor_calibrated", None)
                )
                runtime_status["input_speech_detected"] = getattr(
                    input_frame.conditioning, "speech_detected", None
                )
            publish_input_status(input_frame)
            runtime_status["queue_depth"] = microphone_queue.qsize() if microphone_queue is not None else 0
            runtime_status["drop_count"] = last_drop_count
            runtime_status["end_to_end_ms"] = latest_end_to_end_ms
            if apply_runtime_controls:
                _handle_controls(
                    controls,
                    logger,
                    engine,
                    stop_requested,
                    source_label,
                    runtime_status,
                    diagnostic_store,
                    control=capture_control,
                )
                apply_capture_control()
            _render_status(
                engine,
                int(runtime_status["queue_depth"]),
                last_drop_count,
                source_label=source_label,
                end_to_end_ms=latest_end_to_end_ms,
            )

        def process_pending_analysis_request() -> bool:
            nonlocal expected_sequence, last_drop_count
            if analysis_control is None:
                return False
            apply_threshold_control()
            apply_input_control()
            request: RealtimeAnalysisRequest | None = analysis_control.claim_next()
            if request is None:
                return False
            failure_message: str | None = None
            try:
                if microphone is not None:
                    if not bool(capture_control.snapshot()["paused"]):
                        capture_control.pause()
                    apply_capture_control()
                apply_threshold_control()
                expected_sequence = None
                last_drop_count = microphone_queue.drop_count if microphone_queue is not None else 0
                realtime_input_pipeline.reset()
                replay_pcm_converter.reset()
                if raw_trigger_audio is not None:
                    raw_trigger_audio.reset()
                reset_started_ns = timeline_clock.now_ns(time.perf_counter_ns())
                if diagnostic_store is not None:
                    diagnostic_store.clear()
                    set_raw_device_audio = getattr(diagnostic_store, "set_raw_device_audio", None)
                    if callable(set_raw_device_audio):
                        set_raw_device_audio(
                            request.pcm,
                            sample_rate=request.sample_rate,
                            end_ns=reset_started_ns + request.pcm.shape[0] * 1_000_000_000 // request.sample_rate,
                        )
                for event in engine.reset(reset_started_ns):
                    _record_event(logger, event, reset_started_ns, time.perf_counter_ns())
                    _record_diagnostic_event(diagnostic_store, event, engine, runtime_status)
                frame_samples = round(16000 * args.chunk_ms / 1000)
                source_values = request.pcm
                if source_values.ndim == 1:
                    source_values = source_values[:, None]
                for sequence, start in enumerate(range(0, source_values.shape[0], max(1, round(request.sample_rate * args.chunk_ms / 1000)))):
                    source_chunk = source_values[start : start + max(1, round(request.sample_rate * args.chunk_ms / 1000))]
                    process(
                        AudioChunk(
                            sequence=sequence,
                            samples=source_chunk,
                            sample_rate=request.sample_rate,
                            captured_ns=timeline_clock.now_ns(time.perf_counter_ns()),
                        ),
                        persist_trigger_audio=False,
                        apply_runtime_controls=False,
                        is_analysis_command=True,
                    )
                drain_ready_kws_results(
                    processing_started_ns=time.perf_counter_ns(),
                    persist_trigger_audio=False,
                    wait=True,
                    timeout=5.0,
                )
            except Exception as error:
                failure_message = str(error) or type(error).__name__ or "analysis failed"
            finally:
                try:
                    realtime_input_pipeline.reset()
                    replay_pcm_converter.reset()
                    if raw_trigger_audio is not None:
                        raw_trigger_audio.reset()
                    reset_started_ns = timeline_clock.now_ns(time.perf_counter_ns())
                    for event in engine.reset(reset_started_ns):
                        _record_event(logger, event, reset_started_ns, time.perf_counter_ns())
                        _record_diagnostic_event(diagnostic_store, event, engine, runtime_status)
                except Exception as error:
                    if failure_message is None:
                        failure_message = str(error) or type(error).__name__ or "analysis failed"
                expected_sequence = None
                last_drop_count = microphone_queue.drop_count if microphone_queue is not None else 0
                if failure_message is None:
                    analysis_control.complete(request.request_id)
                else:
                    analysis_control.fail(request.request_id, failure_message)
                if diagnostic_store is not None:
                    diagnostic_store.notify_update()
            return True

        if args.input_wav is not None:
            logger.record("device_open", source="wav", path=str(args.input_wav))
            wav_source = iter(WavReplaySource(args.input_wav, chunk_ms=args.chunk_ms))
            while not stop_requested.is_set():
                _handle_controls(
                    controls,
                    logger,
                    engine,
                    stop_requested,
                    source_label,
                    runtime_status,
                    diagnostic_store,
                    control=capture_control,
                )
                apply_threshold_control()
                process_pending_analysis_request()
                apply_input_device_control()
                apply_capture_mode_control()
                if apply_capture_control():
                    capture_control.wait_for_change(applied_capture_generation, timeout=0.1)
                    continue
                try:
                    audio_chunk = next(wav_source)
                except StopIteration:
                    break
                _handle_controls(
                    controls,
                    logger,
                    engine,
                    stop_requested,
                    source_label,
                    runtime_status,
                    diagnostic_store,
                    control=capture_control,
                )
                apply_threshold_control()
                if stop_requested.is_set() or apply_capture_control():
                    continue
                process(audio_chunk)
            drain_ready_kws_results(
                processing_started_ns=time.perf_counter_ns(),
                persist_trigger_audio=True,
                wait=True,
                timeout=5.0,
            )
        else:
            microphone_queue = BoundedAudioQueue(config.queue_capacity)
            microphone = MicrophoneSource(
                device=args.device,
                chunk_ms=args.chunk_ms,
                channel_mode=InputChannelMode.AUTO,
            )
            device = microphone.open(microphone_queue)
            logger.record(
                "device_open",
                source="microphone",
                index=device.index,
                name=device.name,
                sample_rate=microphone.sample_rate,
            )
            while not stop_requested.is_set():
                _handle_controls(
                    controls,
                    logger,
                    engine,
                    stop_requested,
                    source_label,
                    runtime_status,
                    diagnostic_store,
                    control=capture_control,
                )
                apply_threshold_control()
                apply_input_control()
                process_pending_analysis_request()
                apply_input_device_control()
                apply_capture_mode_control()
                if apply_capture_control():
                    capture_control.wait_for_change(applied_capture_generation, timeout=0.1)
                    continue
                try:
                    process(microphone_queue.get(timeout=0.1))
                except queue.Empty:
                    runtime_status["queue_depth"] = microphone_queue.qsize()
                    runtime_status["drop_count"] = microphone_queue.drop_count
                    drain_ready_kws_results(
                        processing_started_ns=time.perf_counter_ns(),
                        persist_trigger_audio=True,
                    )
                    _handle_controls(
                        controls,
                        logger,
                        engine,
                        stop_requested,
                        source_label,
                        runtime_status,
                        diagnostic_store,
                        control=capture_control,
                    )
                    apply_threshold_control()
                    apply_input_control()
                    apply_input_device_control()
                    apply_capture_mode_control()
                    apply_capture_control()

        logger.record("stop", reason="quit" if stop_requested.is_set() else "input_complete")
        logger.close()
        if args.record_telemetry:
            assert session_path is not None
            print(f"\nsession: {session_path}")
        return 0
    except KeyboardInterrupt:
        if logger is not None:
            logger.record("stop", reason="keyboard_interrupt")
            logger.close()
            if args.record_telemetry:
                assert session_path is not None
                print(f"\nsession: {session_path}")
        return 0
    except (
        AudioSourceError,
        AudioContractError,
        DeploymentError,
        TelemetryError,
        OSError,
        ValueError,
        RuntimeError,
    ) as error:
        print(console_safe_text(f"demo error: {error}", encoding=sys.stderr.encoding), file=sys.stderr)
        if logger is not None:
            try:
                logger.record("error", message=str(error))
                logger.record("stop", reason="error")
                logger.close()
                if args.record_telemetry:
                    assert session_path is not None
                    print(
                        console_safe_text(f"session: {session_path}", encoding=sys.stderr.encoding),
                        file=sys.stderr,
                    )
            except TelemetryError:
                pass
        return 1
    finally:
        try:
            if microphone is not None:
                try:
                    microphone.close()
                except AudioSourceError as error:
                    print(console_safe_text(f"audio device close error: {error}", encoding=sys.stderr.encoding), file=sys.stderr)
        finally:
            try:
                if dashboard is not None:
                    try:
                        dashboard.stop()
                    except OSError as error:
                        print(
                            console_safe_text(
                                f"dashboard stop error: {error}", encoding=sys.stderr.encoding
                            ),
                            file=sys.stderr,
                        )
            finally:
                # ONNX Runtime can retain a Windows file handle until its owning graph is released.
                if kws_executor is not None:
                    kws_executor.close()
                engine = None
                kws_runner = None
                kws_session = None
                frontend_session = None
                backbone_session = None
                vad_session = None
                stateful_vad = None
                gc.collect()
                _cleanup_temporary_vad_export(temporary_vad_export)


def _cleanup_temporary_vad_export(
    temporary_vad_export: tempfile.TemporaryDirectory[str] | None,
) -> None:
    """Try to remove the process-only adapter without masking the run result."""

    if temporary_vad_export is None:
        return
    try:
        temporary_vad_export.cleanup()
    except OSError as error:
        print(
            console_safe_text(
                f"temporary VAD adapter cleanup error: {error}", encoding=sys.stderr.encoding
            ),
            file=sys.stderr,
        )


def _record_event(
    logger: TelemetryLogger, event: CascadeEvent, processing_started_ns: int, completed_ns: int
) -> float:
    fields = dict(event.fields)
    fields["capture_ns"] = event.captured_ns
    fields["queue_delay_ms"] = max(
        0.0, (processing_started_ns - event.captured_ns) / 1_000_000.0
    )
    end_to_end_ms = max(0.0, (completed_ns - event.captured_ns) / 1_000_000.0)
    fields["end_to_end_ms"] = end_to_end_ms
    logger.record(event.kind, **fields)
    return end_to_end_ms


def _apply_capture_transition(
    paused: bool,
    *,
    microphone: MicrophoneSource | None,
    microphone_queue: BoundedAudioQueue | None,
    pcm_converter: StreamingChannelConverter | RealtimeInputPipeline,
    additional_pcm_converter: StreamingChannelConverter | RealtimeInputPipeline | None = None,
    engine: CascadeEngine,
    logger: TelemetryLogger,
    diagnostic_store: RealtimeDiagnosticStore | None,
    runtime_status: dict[str, object],
    captured_ns: int,
) -> None:
    """Apply an already-requested pause state at an audio processing boundary."""

    if microphone_queue is not None:
        microphone_queue.clear()
    if paused:
        if microphone is not None:
            microphone.pause()
    else:
        if microphone is not None:
            microphone.resume()
        pcm_converter.reset()
        if additional_pcm_converter is not None:
            additional_pcm_converter.reset()
        if diagnostic_store is not None:
            diagnostic_store.clear()
        for event in engine.reset(captured_ns):
            _record_event(logger, event, captured_ns, time.perf_counter_ns())
            _record_diagnostic_event(diagnostic_store, event, engine, runtime_status)
    runtime_status["capture_paused"] = paused
    capture_event = CascadeEvent("capture_pause" if paused else "capture_resume", captured_ns)
    _record_event(logger, capture_event, captured_ns, time.perf_counter_ns())
    _record_diagnostic_event(diagnostic_store, capture_event, engine, runtime_status)


def _apply_input_device_switch(
    requested_index: int,
    *,
    microphone: MicrophoneSource,
    microphone_queue: BoundedAudioQueue,
    chunk_ms: int,
    capture_mode: CaptureMode = CaptureMode.EXCLUSIVE,
    pcm_converter: StreamingChannelConverter | RealtimeInputPipeline,
    additional_pcm_converter: StreamingChannelConverter | RealtimeInputPipeline | None,
    raw_trigger_audio: _RawTriggerAudioHistory | None,
    engine: CascadeEngine | ObservedControlEngine,
    logger: TelemetryLogger,
    diagnostic_store: RealtimeDiagnosticStore | None,
    runtime_status: dict[str, object],
    captured_ns: int,
    generation: int,
    event_name: str = "input_device_switch",
) -> tuple[MicrophoneSource, str]:
    """Replace a live input stream after its replacement has opened successfully."""

    replacement = MicrophoneSource(
        device=requested_index,
        chunk_ms=chunk_ms,
        channel_mode=InputChannelMode.AUTO,
        capture_mode=capture_mode,
    )
    device = replacement.open(microphone_queue)
    previous_device = getattr(microphone, "device", None)
    try:
        microphone.close()
    except AudioSourceError:
        replacement.close()
        raise
    return _complete_input_stream_transition(
        microphone=replacement,
        device=device,
        previous_device=previous_device,
        microphone_queue=microphone_queue,
        pcm_converter=pcm_converter,
        additional_pcm_converter=additional_pcm_converter,
        raw_trigger_audio=raw_trigger_audio,
        engine=engine,
        logger=logger,
        diagnostic_store=diagnostic_store,
        runtime_status=runtime_status,
        captured_ns=captured_ns,
        generation=generation,
        event_name=event_name,
        capture_mode=capture_mode,
    )


def _apply_capture_mode_switch(
    requested_mode: CaptureMode,
    *,
    microphone: MicrophoneSource,
    microphone_queue: BoundedAudioQueue,
    chunk_ms: int,
    pcm_converter: StreamingChannelConverter | RealtimeInputPipeline,
    additional_pcm_converter: StreamingChannelConverter | RealtimeInputPipeline | None,
    raw_trigger_audio: _RawTriggerAudioHistory | None,
    engine: CascadeEngine | ObservedControlEngine,
    logger: TelemetryLogger,
    diagnostic_store: RealtimeDiagnosticStore | None,
    runtime_status: dict[str, object],
    captured_ns: int,
    generation: int,
) -> tuple[MicrophoneSource, str]:
    """Switch modes on one endpoint by releasing its current WASAPI stream first."""

    previous_device = getattr(microphone, "device", None)
    if previous_device is None:
        raise AudioSourceError("microphone device is not open")
    try:
        previous_mode = CaptureMode(getattr(microphone, "capture_mode", "exclusive"))
    except ValueError as error:
        raise AudioSourceError("microphone has an invalid active capture mode") from error
    microphone.close()
    replacement = MicrophoneSource(
        device=previous_device.index,
        chunk_ms=chunk_ms,
        channel_mode=InputChannelMode.AUTO,
        capture_mode=requested_mode,
    )
    try:
        device = replacement.open(microphone_queue)
    except Exception as switch_error:
        try:
            replacement.close()
        except AudioSourceError:
            pass
        try:
            microphone.open(microphone_queue)
        except Exception as restore_error:
            raise AudioSourceError(
                f"could not open WASAPI {requested_mode.value} input device: {previous_device.name}; "
                f"also failed to restore {previous_mode.value} capture"
            ) from restore_error
        _complete_input_stream_transition(
            microphone=microphone,
            device=previous_device,
            previous_device=previous_device,
            microphone_queue=microphone_queue,
            pcm_converter=pcm_converter,
            additional_pcm_converter=additional_pcm_converter,
            raw_trigger_audio=raw_trigger_audio,
            engine=engine,
            logger=logger,
            diagnostic_store=diagnostic_store,
            runtime_status=runtime_status,
            captured_ns=captured_ns,
            generation=generation,
            event_name="capture_mode_switch_rollback",
            capture_mode=previous_mode,
        )
        raise switch_error
    return _complete_input_stream_transition(
        microphone=replacement,
        device=device,
        previous_device=previous_device,
        microphone_queue=microphone_queue,
        pcm_converter=pcm_converter,
        additional_pcm_converter=additional_pcm_converter,
        raw_trigger_audio=raw_trigger_audio,
        engine=engine,
        logger=logger,
        diagnostic_store=diagnostic_store,
        runtime_status=runtime_status,
        captured_ns=captured_ns,
        generation=generation,
        event_name="capture_mode_switch",
        capture_mode=requested_mode,
    )


def _complete_input_stream_transition(
    *,
    microphone: MicrophoneSource,
    device: object,
    previous_device: object | None,
    microphone_queue: BoundedAudioQueue,
    pcm_converter: StreamingChannelConverter | RealtimeInputPipeline,
    additional_pcm_converter: StreamingChannelConverter | RealtimeInputPipeline | None,
    raw_trigger_audio: _RawTriggerAudioHistory | None,
    engine: CascadeEngine | ObservedControlEngine,
    logger: TelemetryLogger,
    diagnostic_store: RealtimeDiagnosticStore | None,
    runtime_status: dict[str, object],
    captured_ns: int,
    generation: int,
    event_name: str,
    capture_mode: CaptureMode,
) -> tuple[MicrophoneSource, str]:
    """Discard model state and publish the endpoint that now owns capture."""

    microphone_queue.clear()
    pcm_converter.reset()
    if additional_pcm_converter is not None:
        additional_pcm_converter.reset()
    if raw_trigger_audio is not None:
        raw_trigger_audio.reset()
    if diagnostic_store is not None:
        diagnostic_store.clear()
    runtime_status["queue_depth"] = 0
    runtime_status["drop_count"] = 0
    runtime_status["end_to_end_ms"] = None
    logger.record(
        event_name,
        previous_index=None if previous_device is None else previous_device.index,
        previous_name=None if previous_device is None else previous_device.name,
        index=device.index,
        name=device.name,
        sample_rate=getattr(microphone, "sample_rate", None),
        capture_mode=getattr(microphone, "capture_mode", capture_mode.value),
        generation=generation,
        reset=True,
    )
    for event in engine.reset(captured_ns):
        _record_event(logger, event, captured_ns, time.perf_counter_ns())
        _record_diagnostic_event(diagnostic_store, event, engine, runtime_status)
    return microphone, f"{device.index}:{device.name}"


def _apply_threshold_transition(
    thresholds: dict[str, object],
    *,
    engine: CascadeEngine,
    logger: TelemetryLogger,
    diagnostic_store: RealtimeDiagnosticStore | None,
    runtime_status: dict[str, object],
    captured_ns: int,
) -> None:
    """Apply a requested threshold pair at a single audio-safe boundary."""

    vad_threshold = float(thresholds["vad_threshold"])
    kws_threshold = float(thresholds["kws_threshold"])
    generation = int(thresholds["generation"])
    engine.set_runtime_thresholds(vad_threshold=vad_threshold, kws_threshold=kws_threshold)
    if diagnostic_store is not None:
        diagnostic_store.set_thresholds(vad_threshold=vad_threshold, kws_threshold=kws_threshold)
    logger.record(
        "thresholds_applied",
        vad_threshold=vad_threshold,
        kws_threshold=kws_threshold,
        generation=generation,
        reset=True,
    )
    for event in engine.reset(captured_ns):
        _record_event(logger, event, captured_ns, time.perf_counter_ns())
        _record_diagnostic_event(diagnostic_store, event, engine, runtime_status)


def _apply_runtime_config_transition(
    runtime_config: dict[str, object],
    *,
    engine: CascadeEngine,
    logger: TelemetryLogger,
    diagnostic_store: RealtimeDiagnosticStore | None,
    runtime_status: dict[str, object],
    captured_ns: int,
) -> None:
    """Apply one complete runtime configuration at an audio-safe boundary."""

    vad_threshold = float(runtime_config["vad_threshold"])
    kws_threshold = float(runtime_config["kws_threshold"])
    energy_enabled = bool(runtime_config["energy_enabled"])
    vad_enabled = bool(runtime_config["vad_enabled"])
    vad_period_ms = int(runtime_config["vad_period_ms"])
    kws_period_ms = int(runtime_config["kws_period_ms"])
    generation = int(runtime_config["generation"])
    schedule = TimingSchedule.from_periods(vad_period_ms, kws_period_ms)
    engine.set_runtime_thresholds(vad_threshold=vad_threshold, kws_threshold=kws_threshold)
    engine.set_runtime_gates(energy_enabled=energy_enabled, vad_enabled=vad_enabled)
    set_runtime_schedule = getattr(engine, "set_runtime_schedule", None)
    if callable(set_runtime_schedule):
        set_runtime_schedule(vad_period_ms=vad_period_ms, kws_period_ms=kws_period_ms)
    else:
        engine.set_runtime_vad_period(vad_period_ms)
    if diagnostic_store is not None:
        diagnostic_store.set_runtime_config(
            vad_threshold=vad_threshold,
            kws_threshold=kws_threshold,
            energy_enabled=energy_enabled,
            vad_enabled=vad_enabled,
            vad_period_ms=vad_period_ms,
            kws_period_ms=kws_period_ms,
        )
    logger.record(
        "runtime_config_applied",
        vad_threshold=vad_threshold,
        kws_threshold=kws_threshold,
        energy_enabled=energy_enabled,
        vad_enabled=vad_enabled,
        energy_period_ms=schedule.energy_period_ms,
        vad_period_ms=schedule.vad_period_ms,
        kws_period_ms=schedule.kws_period_ms,
        generation=generation,
        reset=True,
    )
    logger.record(
        "runtime_schedule_applied",
        energy_period_ms=schedule.energy_period_ms,
        vad_period_ms=schedule.vad_period_ms,
        kws_period_ms=schedule.kws_period_ms,
        generation=generation,
    )
    for event in engine.reset(captured_ns):
        _record_event(logger, event, captured_ns, time.perf_counter_ns())
        _record_diagnostic_event(diagnostic_store, event, engine, runtime_status)


def _apply_observed_runtime_config_transition(
    runtime_config: dict[str, object],
    *,
    engine: ObservedControlEngine,
    logger: TelemetryLogger,
    diagnostic_store: RealtimeDiagnosticStore | None,
    runtime_status: dict[str, object],
    captured_ns: int,
) -> None:
    """Apply all dashboard controls to the PC controller at one audio boundary."""

    vad_threshold = float(runtime_config["vad_threshold"])
    kws_threshold = float(runtime_config["kws_threshold"])
    energy_enabled = bool(runtime_config["energy_enabled"])
    vad_enabled = bool(runtime_config["vad_enabled"])
    vad_period_ms = int(runtime_config["vad_period_ms"])
    kws_period_ms = int(runtime_config["kws_period_ms"])
    kws_lookback_ms = int(runtime_config["kws_lookback_ms"])
    generation = int(runtime_config["generation"])
    schedule = TimingSchedule.from_periods(vad_period_ms, kws_period_ms)
    reset_events = engine.apply_runtime_config(
        vad_threshold=vad_threshold,
        kws_threshold=kws_threshold,
        energy_enabled=energy_enabled,
        vad_enabled=vad_enabled,
        vad_period_ms=vad_period_ms,
        kws_period_ms=kws_period_ms,
        kws_lookback_ms=kws_lookback_ms,
        captured_ns=captured_ns,
    )
    if diagnostic_store is not None:
        diagnostic_store.set_runtime_config(
            vad_threshold=vad_threshold,
            kws_threshold=kws_threshold,
            energy_enabled=energy_enabled,
            vad_enabled=vad_enabled,
            vad_period_ms=vad_period_ms,
            kws_period_ms=kws_period_ms,
            kws_lookback_ms=kws_lookback_ms,
        )
    logger.record(
        "runtime_config_applied",
        vad_threshold=vad_threshold,
        kws_threshold=kws_threshold,
        energy_enabled=energy_enabled,
        vad_enabled=vad_enabled,
        energy_period_ms=schedule.energy_period_ms,
        vad_period_ms=schedule.vad_period_ms,
        kws_period_ms=schedule.kws_period_ms,
        kws_lookback_ms=kws_lookback_ms,
        generation=generation,
        reset=True,
    )
    logger.record(
        "runtime_schedule_applied",
        energy_period_ms=schedule.energy_period_ms,
        vad_period_ms=schedule.vad_period_ms,
        kws_period_ms=schedule.kws_period_ms,
        generation=generation,
    )
    for event in reset_events:
        _record_event(logger, event, captured_ns, time.perf_counter_ns())
        _record_diagnostic_event(diagnostic_store, event, engine, runtime_status)


def _record_diagnostic_event(
    store: RealtimeDiagnosticStore | None,
    event: CascadeEvent,
    engine: CascadeEngine,
    runtime_status: dict[str, object],
) -> None:
    if store is None:
        return
    missing = object()
    schedule = getattr(engine, "effective_schedule", {})
    runtime: dict[str, object] = {
        "state": engine.state.value,
        "queue_depth": runtime_status["queue_depth"],
        "drop_count": runtime_status["drop_count"],
        "end_to_end_ms": runtime_status["end_to_end_ms"],
        "vad_score": engine.latest_vad_score,
        "kws_score": engine.latest_kws_score,
        "capture_paused": runtime_status["capture_paused"],
    }
    for name in (
        "kws_gate_open",
        "vad_positive_count",
        "vad_silence_count",
        "vad_no_speech_remaining_ms",
    ):
        value = getattr(engine, name, missing)
        if value is not missing:
            runtime[name] = value
    hangover_remaining_ms = getattr(engine, "energy_hangover_remaining_ms", missing)
    if hangover_remaining_ms is missing:
        hangover_remaining_ms = getattr(engine, "kws_hangover_remaining_ms", missing)
    if hangover_remaining_ms is not missing:
        runtime["energy_hangover_remaining_ms"] = hangover_remaining_ms
    vad_confirmed_state = getattr(engine, "vad_confirmed_state", missing)
    if vad_confirmed_state is not missing:
        runtime["vad_confirmed_state"] = getattr(vad_confirmed_state, "value", vad_confirmed_state)
    confirmation_count = getattr(engine, "confirmation_count", missing)
    if confirmation_count is not missing:
        runtime["kws_positive_count"] = confirmation_count
    if isinstance(schedule, dict):
        runtime.update(schedule)
    store.append_event(event, runtime)


def _handle_controls(
    controls: "_CommandReader",
    logger: TelemetryLogger,
    engine: CascadeEngine,
    stop_requested: threading.Event,
    source_label: str,
    runtime_status: dict[str, object],
    diagnostic_store: RealtimeDiagnosticStore | None = None,
    *,
    control: RealtimeCaptureControl | None = None,
) -> None:
    while True:
        try:
            command = controls.commands.get_nowait()
        except queue.Empty:
            return
        normalized = command.strip().casefold()
        if normalized == "status":
            logger.record("control", command="status")
            _render_status(
                engine,
                int(runtime_status["queue_depth"]),
                int(runtime_status["drop_count"]),
                source_label=source_label,
                end_to_end_ms=runtime_status["end_to_end_ms"],
            )
        elif normalized == "devices":
            logger.record("control", command="devices")
            _list_devices()
        elif normalized == "reset":
            logger.record("control", command="reset")
            for event in engine.reset(time.perf_counter_ns()):
                processing_started_ns = time.perf_counter_ns()
                _record_event(logger, event, processing_started_ns, time.perf_counter_ns())
                _record_diagnostic_event(diagnostic_store, event, engine, runtime_status)
        elif normalized == "pause":
            logger.record("control", command="pause")
            if control is None:
                logger.record("error", source="control", message="pause is unavailable")
            else:
                control.pause()
        elif normalized == "resume":
            logger.record("control", command="resume")
            if control is None:
                logger.record("error", source="control", message="resume is unavailable")
            else:
                control.resume()
        elif normalized == "quit":
            logger.record("control", command="quit")
            stop_requested.set()
        elif normalized:
            logger.record("error", source="control", message=f"unknown command: {command}")
            print(console_safe_text(f"unknown command: {command}", encoding=sys.stderr.encoding), file=sys.stderr)


def _render_event(event: CascadeEvent, engine: CascadeEngine, drop_count: int) -> None:
    if event.kind == "wake":
        print(f"\nWAKE score={event.fields.get('score'):.4f} state={engine.state.value} drops={drop_count}")
    elif event.kind == "state_transition":
        print(
            f"\nstate {event.fields['from_state']} -> {event.fields['to_state']} "
            f"({event.fields['reason']})"
        )


def _render_status(
    engine: CascadeEngine,
    queue_depth: int,
    drop_count: int,
    *,
    source_label: str = "-",
    end_to_end_ms: float | None = None,
) -> None:
    print("\r" + format_status_line(engine, queue_depth, drop_count, source_label, end_to_end_ms), end="", flush=True)


def format_status_line(
    engine: CascadeEngine,
    queue_depth: int,
    drop_count: int,
    source_label: str = "-",
    end_to_end_ms: float | None = None,
) -> str:
    vad = "-" if engine.latest_vad_score is None else f"{engine.latest_vad_score:.3f}"
    kws_is_current = bool(getattr(engine, "kws_gate_open", False))
    kws = "-" if not kws_is_current or engine.latest_kws_score is None else f"{engine.latest_kws_score:.3f}"
    vad_onnx_ms = getattr(engine, "latest_vad_onnx_ms", None)
    kws_onnx_ms = getattr(engine, "latest_kws_onnx_ms", None)
    vad_ort = "-" if vad_onnx_ms is None else f"{vad_onnx_ms:.1f}"
    kws_ort = (
        "-"
        if not kws_is_current or kws_onnx_ms is None
        else f"{kws_onnx_ms:.1f}"
    )
    end_to_end = "-" if end_to_end_ms is None else f"{end_to_end_ms:.1f}"
    source = console_safe_text(source_label)
    schedule = getattr(engine, "effective_schedule", {})
    cadence = ""
    if isinstance(schedule, dict):
        cadence = (
            f" cadence=e{schedule.get('energy_period_ms', '-')}/"
            f"v{schedule.get('vad_period_ms', '-')}/k{schedule.get('kws_period_ms', '-')}ms"
        )
    return (
        f"device={source:<24.24} state={engine.state.value:<16} vad={vad:<5} kws={kws:<5} "
        f"vad_ort={vad_ort:<5}ms kws_ort={kws_ort:<5}ms e2e={end_to_end:<5}ms "
        f"queue={queue_depth:<3} drops={drop_count:<4}{cadence}"
    )


def _save_trigger_audio(session_path: Path, captured_ns: int, waveform: np.ndarray) -> Path:
    path = session_path / f"trigger-{captured_ns}.wav"
    pcm16 = np.clip(waveform, -1.0, 1.0)
    pcm16 = np.rint(pcm16 * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(pcm16.tobytes())
    return path


class _CommandReader:
    def __init__(self) -> None:
        self.commands: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._read, daemon=True, name="vad-kws-controls")

    def start(self) -> None:
        self._thread.start()

    def _read(self) -> None:
        while True:
            line = sys.stdin.readline()
            if line == "":
                return
            self.commands.put(line.rstrip("\r\n"))
