"""Audio sources for WAV replay and optional Windows microphone capture."""

from __future__ import annotations

import importlib
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterator

import numpy as np
import torchaudio


class AudioSourceError(RuntimeError):
    """Raised when an audio input cannot be opened or decoded."""


class MissingAudioDependency(AudioSourceError):
    """Raised when live capture is selected without SoundDevice installed."""


@dataclass(frozen=True)
class AudioChunk:
    sequence: int
    samples: np.ndarray
    sample_rate: int
    captured_ns: int
    status: str | None = None
    raw_samples: np.ndarray | None = None
    raw_sample_width: int | None = None


class InputChannelMode(str, Enum):
    """Requested handling mode for channels exposed by one input endpoint."""

    AUTO = "auto"
    MONO = "mono"
    ARRAY = "array"


class CaptureMode(str, Enum):
    """How the Windows endpoint is opened for live capture."""

    EXCLUSIVE = "exclusive"
    SHARED = "shared"


@dataclass(frozen=True)
class InputCapability:
    """The input-channel configuration actually opened for one device."""

    requested_mode: InputChannelMode
    opened_channels: int
    active_mode: str
    fallback_reason: str | None
    capture_mode: str = "shared"


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    default_sample_rate: float
    max_input_channels: int
    host_api: str | None = None


class BoundedAudioQueue:
    """Thread-safe queue that protects the capture callback by dropping oldest PCM."""

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be at least one")
        self._queue: queue.Queue[AudioChunk] = queue.Queue(maxsize=capacity)
        self._put_lock = threading.Lock()
        self.drop_count = 0

    def put(self, item: AudioChunk) -> bool:
        """Put a chunk and return whether an older queued chunk was dropped."""

        with self._put_lock:
            dropped = False
            while True:
                try:
                    self._queue.put_nowait(item)
                    return dropped
                except queue.Full:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        continue
                    self.drop_count += 1
                    dropped = True

    def get(self, timeout: float | None = None) -> AudioChunk:
        return self._queue.get(timeout=timeout)

    def get_nowait(self) -> AudioChunk:
        return self._queue.get_nowait()

    def qsize(self) -> int:
        return self._queue.qsize()

    def clear(self) -> int:
        """Discard pending chunks without treating an operator pause as a drop."""

        cleared = 0
        with self._put_lock:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    return cleared
                cleared += 1


class WavReplaySource:
    """Replay a WAV/other TorchAudio-readable file with its original timing."""

    def __init__(
        self,
        path: Path,
        *,
        chunk_ms: int = 100,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if isinstance(chunk_ms, bool) or not isinstance(chunk_ms, int) or chunk_ms <= 0:
            raise ValueError("chunk_ms must be a positive integer")
        self.path = Path(path)
        self.chunk_ms = chunk_ms
        self._sleep = sleep_fn
        self._clock_ns = clock_ns

    def __iter__(self) -> Iterator[AudioChunk]:
        if not self.path.is_file():
            raise AudioSourceError(f"input WAV file does not exist: {self.path}")
        try:
            waveform, sample_rate = torchaudio.load(str(self.path))
        except Exception as error:
            raise AudioSourceError(f"could not decode input WAV: {self.path}") from error
        if waveform.ndim != 2 or waveform.shape[0] < 1 or waveform.shape[1] < 1:
            raise AudioSourceError(f"input WAV has no samples: {self.path}")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise AudioSourceError(f"input WAV has invalid sample rate: {self.path}")
        samples = waveform.transpose(0, 1).detach().cpu().numpy().astype(np.float32, copy=False)
        if not np.all(np.isfinite(samples)):
            raise AudioSourceError(f"input WAV contains non-finite samples: {self.path}")
        chunk_samples = max(1, round(sample_rate * self.chunk_ms / 1000.0))
        for sequence, start in enumerate(range(0, len(samples), chunk_samples)):
            end = min(start + chunk_samples, len(samples))
            duration_seconds = (end - start) / sample_rate
            self._sleep(duration_seconds)
            yield AudioChunk(
                sequence=sequence,
                samples=samples[start:end].copy(),
                sample_rate=sample_rate,
                captured_ns=self._clock_ns(),
            )


def require_sounddevice() -> ModuleType:
    try:
        return importlib.import_module("sounddevice")
    except ImportError as error:
        raise MissingAudioDependency(
            "Live microphone capture requires sounddevice; install with: "
            "pip install -r dscnn_kws/demo/requirements.txt"
        ) from error


def list_input_devices(
    sounddevice_module: ModuleType | None = None, *, wasapi_only: bool = False
) -> list[AudioDevice]:
    module = sounddevice_module or require_sounddevice()
    try:
        raw_devices = module.query_devices()
    except Exception as error:
        raise AudioSourceError("could not enumerate Windows audio devices") from error
    devices: list[AudioDevice] = []
    for index, raw in enumerate(raw_devices):
        channels = int(raw.get("max_input_channels", 0))
        if channels > 0:
            host_api = _host_api_name(module, raw.get("hostapi"))
            if wasapi_only and not _is_wasapi_host_api(host_api):
                continue
            devices.append(
                AudioDevice(
                    index=index,
                    name=str(raw.get("name", "")),
                    default_sample_rate=float(raw.get("default_samplerate", 0.0)),
                    max_input_channels=channels,
                    host_api=host_api,
                )
            )
    return devices


def resolve_input_device(selector: str | int | None, sounddevice_module: ModuleType | None = None) -> AudioDevice:
    module = sounddevice_module or require_sounddevice()
    devices = list_input_devices(module)
    if not devices:
        raise AudioSourceError("no input-capable audio devices were found")
    wasapi_devices = [device for device in devices if _is_wasapi_host_api(device.host_api)]
    if not wasapi_devices:
        raise AudioSourceError("no Windows WASAPI input device was found for exclusive capture")
    if selector is None:
        try:
            default_index = int(module.default.device[0])
        except Exception as error:
            raise AudioSourceError("could not resolve the default Windows input device") from error
        default_device = next((device for device in devices if device.index == default_index), None)
        if default_device is not None:
            matches = _same_name_devices(default_device.name, wasapi_devices)
            if len(matches) == 1:
                return matches[0]
        wasapi_default = _wasapi_default_input_device(module, wasapi_devices)
        if wasapi_default is not None:
            return wasapi_default
        raise AudioSourceError("could not resolve a default Windows WASAPI input device")
    if isinstance(selector, str) and selector.strip().isdigit():
        selector = int(selector.strip())
    if isinstance(selector, int) and not isinstance(selector, bool):
        for device in wasapi_devices:
            if device.index == selector:
                return device
        requested = next((device for device in devices if device.index == selector), None)
        if requested is not None:
            matches = _same_name_devices(requested.name, wasapi_devices)
            if len(matches) == 1:
                return matches[0]
        raise AudioSourceError(f"input device index is unavailable: {selector}")
    if isinstance(selector, str) and selector.strip():
        matches = _same_name_devices(selector, wasapi_devices)
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise AudioSourceError(f"WASAPI input device name is unavailable: {selector}")
        raise AudioSourceError(f"WASAPI input device name is ambiguous: {selector}")
    raise AudioSourceError("input device must be an index, a nonempty name, or the default")


def _host_api_name(module: ModuleType, host_api_index: object) -> str | None:
    if isinstance(host_api_index, bool) or not isinstance(host_api_index, int):
        return None
    try:
        host_api = module.query_hostapis(host_api_index)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return None
    if not isinstance(host_api, dict):
        return None
    name = host_api.get("name")
    return name if isinstance(name, str) else None


def _is_wasapi_host_api(host_api: str | None) -> bool:
    return isinstance(host_api, str) and host_api.casefold() == "windows wasapi"


def _same_name_devices(name: str, devices: list[AudioDevice]) -> list[AudioDevice]:
    normalized = name.casefold().strip()
    return [device for device in devices if device.name.casefold().strip() == normalized]


def _wasapi_default_input_device(
    module: ModuleType, devices: list[AudioDevice]
) -> AudioDevice | None:
    try:
        host_apis = module.query_hostapis()
    except (AttributeError, TypeError, ValueError):
        return None
    if not isinstance(host_apis, (list, tuple)):
        return None
    for host_api in host_apis:
        if not isinstance(host_api, dict) or not _is_wasapi_host_api(host_api.get("name")):
            continue
        index = host_api.get("default_input_device")
        if isinstance(index, int) and not isinstance(index, bool):
            return next((device for device in devices if device.index == index), None)
    return None


class MicrophoneSource:
    """Optional SoundDevice capture source that performs no inference in its callback."""

    def __init__(
        self,
        *,
        device: str | int | None = None,
        chunk_ms: int = 100,
        channel_mode: InputChannelMode | str = InputChannelMode.AUTO,
        capture_mode: CaptureMode | str = CaptureMode.EXCLUSIVE,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if isinstance(chunk_ms, bool) or not isinstance(chunk_ms, int) or chunk_ms <= 0:
            raise ValueError("chunk_ms must be a positive integer")
        self._selector = device
        self._chunk_ms = chunk_ms
        try:
            self._channel_mode = InputChannelMode(channel_mode)
        except ValueError as error:
            raise ValueError("channel_mode must be auto, mono, or array") from error
        try:
            self._requested_capture_mode = CaptureMode(capture_mode)
        except ValueError as error:
            raise ValueError("capture_mode must be exclusive or shared") from error
        self._clock_ns = clock_ns
        self._stream: object | None = None
        self.device: AudioDevice | None = None
        self.sample_rate: int | None = None
        self.capability: InputCapability | None = None
        self._sequence = 0
        self._paused = False
        self.capture_format = "float32"
        self.capture_mode = self._requested_capture_mode.value

    def open(self, output: BoundedAudioQueue) -> AudioDevice:
        module = require_sounddevice()
        device = resolve_input_device(self._selector, module)
        sample_rate = int(round(device.default_sample_rate))
        if sample_rate <= 0:
            raise AudioSourceError(f"input device has an invalid default sample rate: {device.name}")
        blocksize = max(1, round(sample_rate * self._chunk_ms / 1000.0))

        def callback(indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
            status_text = str(status) if status else None
            raw = np.asarray(indata).copy()
            samples = raw.astype(np.float32, copy=False)
            if np.issubdtype(raw.dtype, np.integer):
                samples = (samples / 2147483648.0).astype(np.float32, copy=False)
            output.put(
                AudioChunk(
                    sequence=self._sequence,
                    samples=samples,
                    sample_rate=sample_rate,
                    captured_ns=self._clock_ns(),
                    status=status_text,
                    raw_samples=raw,
                    raw_sample_width=3,
                )
            )
            self._sequence += 1

        requested_channels = (
            1
            if self._channel_mode is InputChannelMode.MONO
            else min(2, device.max_input_channels)
        )
        stream: object | None = None
        opened_channels: int | None = None
        last_error: Exception | None = None
        wasapi_settings: object | None = None
        if self._requested_capture_mode is CaptureMode.EXCLUSIVE:
            try:
                wasapi_settings = module.WasapiSettings(exclusive=True, auto_convert=False)
            except Exception as error:
                raise AudioSourceError("could not configure WASAPI exclusive capture") from error
        for channels in range(requested_channels, 0, -1):
            candidate: object | None = None
            try:
                kwargs = dict(
                    device=device.index,
                    samplerate=sample_rate,
                    channels=channels,
                    dtype=(
                        "int32"
                        if self._requested_capture_mode is CaptureMode.EXCLUSIVE
                        else "float32"
                    ),
                    blocksize=blocksize,
                    callback=callback,
                )
                if wasapi_settings is not None:
                    kwargs["extra_settings"] = wasapi_settings
                candidate = module.InputStream(**kwargs)
                candidate.start()
            except Exception as error:
                last_error = error
                if candidate is not None:
                    try:
                        candidate.close()
                    except Exception:
                        pass
                continue
            stream = candidate
            opened_channels = channels
            self.capture_mode = self._requested_capture_mode.value
            break
        if stream is None or opened_channels is None:
            raise AudioSourceError(
                f"could not open WASAPI {self._requested_capture_mode.value} input device: {device.name}"
            ) from last_error
        self._stream = stream
        self.device = device
        self.sample_rate = sample_rate
        fallback_reason = (
            None
            if opened_channels == requested_channels
            else f"opened {opened_channels} of requested {requested_channels} channels"
        )
        self.capability = InputCapability(
            requested_mode=self._channel_mode,
            opened_channels=opened_channels,
            active_mode="mono" if opened_channels == 1 else "array_candidate",
            fallback_reason=fallback_reason,
            capture_mode=self.capture_mode,
        )
        self._paused = False
        return device

    def pause(self) -> None:
        """Stop callback delivery while retaining the open microphone device."""

        stream = self._stream
        if stream is None:
            raise AudioSourceError("microphone stream is not open")
        if self._paused:
            return
        try:
            stream.stop()
        except Exception as error:
            raise AudioSourceError("could not pause microphone stream") from error
        self._paused = True

    def resume(self) -> None:
        """Restart callback delivery after a prior pause."""

        stream = self._stream
        if stream is None:
            raise AudioSourceError("microphone stream is not open")
        if not self._paused:
            return
        try:
            stream.start()
        except Exception as error:
            raise AudioSourceError("could not resume microphone stream") from error
        self._paused = False

    def close(self) -> None:
        stream = self._stream
        self._stream = None
        self._paused = False
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception as error:
            raise AudioSourceError("could not close microphone stream") from error
