from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import torch
import torch.nn.functional as F
import torchaudio.functional as AF

from .audio import load_mono_float32, peak_guard, save_pcm16_atomic, sha256_file
from .recipes import Recipe
from .segment import detect_positive_boundary
from .synthesis import (
    Boundary,
    crop_complete_window,
    crop_negative,
    crop_positive,
    fft_convolve_rir,
    remap_boundary_for_speed,
    synthesize_speech_noise,
    time_stretch_preserve_pitch,
)


@dataclass(frozen=True)
class GenerationRequest:
    example_id: str
    recipe: Recipe
    foreground_path: Path | None
    active_span: tuple[int, int] | None
    noise_path: Path | None = None
    rir_path: Path | None = None
    interferer_path: Path | None = None
    rir_source_sha256: str | None = None
    interferer_source_sha256: str | None = None
    foreground_start_sample: int | None = None
    noise_start_sample: int | None = None
    noise_source_sha256: str | None = None
    noise_source_scene: str | None = None
    resolved_noise_scene: str | None = None
    source_role: str | None = None
    source_candidate_count: int | None = None
    augmentation_closure_required: bool = True
    counterpart_kind: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "GenerationRequest":
        recipe_raw = raw.get("recipe", raw)
        if not isinstance(recipe_raw, dict):
            raise ValueError("Generation request recipe must be an object")
        active_span_raw = raw.get("active_span")
        active_span = None if active_span_raw is None else (int(active_span_raw[0]), int(active_span_raw[1]))  # type: ignore[index]
        augmentation_closure_required = raw.get("augmentation_closure_required", True)
        if not isinstance(augmentation_closure_required, bool):
            raise ValueError("Generation request augmentation_closure_required must be boolean")
        counterpart_kind = raw.get("counterpart_kind")
        if counterpart_kind is not None and (not isinstance(counterpart_kind, str) or not counterpart_kind.strip()):
            raise ValueError("Generation request counterpart_kind must be a non-empty string or null")
        return cls(
            example_id=str(raw["example_id"]),
            recipe=Recipe(**recipe_raw),  # type: ignore[arg-type]
            foreground_path=None if raw.get("foreground_path") is None else Path(str(raw["foreground_path"])),
            active_span=active_span,
            noise_path=None if raw.get("noise_path") is None else Path(str(raw["noise_path"])),
            rir_path=None if raw.get("rir_path") is None else Path(str(raw["rir_path"])),
            interferer_path=None if raw.get("interferer_path") is None else Path(str(raw["interferer_path"])),
            rir_source_sha256=None if raw.get("rir_source_sha256") is None else str(raw["rir_source_sha256"]),
            interferer_source_sha256=(
                None if raw.get("interferer_source_sha256") is None else str(raw["interferer_source_sha256"])
            ),
            foreground_start_sample=None if raw.get("foreground_start_sample") is None else int(raw["foreground_start_sample"]),
            noise_start_sample=None if raw.get("noise_start_sample") is None else int(raw["noise_start_sample"]),
            noise_source_sha256=None if raw.get("noise_source_sha256") is None else str(raw["noise_source_sha256"]),
            noise_source_scene=None if raw.get("noise_source_scene") is None else str(raw["noise_source_scene"]),
            resolved_noise_scene=None if raw.get("resolved_noise_scene") is None else str(raw["resolved_noise_scene"]),
            source_role=None if raw.get("source_role") is None else str(raw["source_role"]),
            source_candidate_count=None if raw.get("source_candidate_count") is None else int(raw["source_candidate_count"]),
            augmentation_closure_required=augmentation_closure_required,
            counterpart_kind=counterpart_kind,
        )


def _identifier(value: str | GenerationRequest) -> str:
    return value if isinstance(value, str) else value.example_id


def _stable_shard(identifier: str, world_size: int) -> int:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    return int.from_bytes(hashlib.sha256(identifier.encode("utf-8")).digest()[:8], "big") % world_size


def select_shard(items: Sequence[str] | Sequence[GenerationRequest], rank: int, world_size: int) -> list[str] | list[GenerationRequest]:
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    return [item for item in items if _stable_shard(_identifier(item), world_size) == rank]


def _completed_example_ids(completed_manifest: Path) -> set[str]:
    if not completed_manifest.is_file():
        return set()
    completed: set[str] = set()
    for line_number, line in enumerate(completed_manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if record.get("example_id") and record.get("output_sha256"):
                completed.add(str(record["example_id"]))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid completion record {completed_manifest}:{line_number}") from error
    return completed


def pending_examples(example_ids: Sequence[str], completed_manifest: Path) -> list[str]:
    completed = _completed_example_ids(Path(completed_manifest))
    return [example_id for example_id in example_ids if example_id not in completed]


def _load_16k(path: Path, device: torch.device, sample_rate: int = 16000) -> torch.Tensor:
    waveform, source_sample_rate, _ = load_mono_float32(Path(path))
    if source_sample_rate != sample_rate:
        waveform = AF.resample(waveform, source_sample_rate, sample_rate)
    return waveform.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")


class _LastNoiseSourceCache:
    """Bounded per-rank cache for source-grouped V3 environment requests."""

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._path: Path | None = None
        self._waveform: torch.Tensor | None = None

    def load(self, path: Path) -> torch.Tensor:
        resolved = Path(path).expanduser().resolve()
        if self._path != resolved or self._waveform is None:
            self._path = resolved
            self._waveform = _load_16k(resolved, self._device)
        return self._waveform


class _FalseWakeSourceCache:
    """Keep the two train false-wake recordings decoded per render rank."""

    def __init__(self, device: torch.device, *, capacity: int = 2) -> None:
        self._device = device
        self._capacity = capacity
        self._waveforms: dict[Path, torch.Tensor] = {}

    def load(self, path: Path) -> torch.Tensor:
        resolved = Path(path).expanduser().resolve()
        waveform = self._waveforms.get(resolved)
        if waveform is not None:
            return waveform
        waveform = _load_16k(resolved, self._device)
        if len(self._waveforms) >= self._capacity:
            self._waveforms.pop(next(iter(self._waveforms)))
        self._waveforms[resolved] = waveform
        return waveform


def _validate_example_id(example_id: str) -> None:
    if not example_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for character in example_id):
        raise ValueError(f"Unsafe example_id: {example_id!r}")


def _active_span_for_negative(cropped: torch.Tensor) -> tuple[int, int]:
    detected = detect_positive_boundary(cropped.detach().cpu(), sample_rate=16000)
    if detected.end > detected.start:
        return (detected.start, detected.end)
    return (0, cropped.shape[-1])


def _rms_band(waveform: torch.Tensor) -> str:
    rms = torch.sqrt(torch.mean(waveform.to(dtype=torch.float32).square())).item()
    dbfs = 20.0 * math.log10(max(float(rms), 1e-12))
    lower = int(math.floor(dbfs / 2.0) * 2)
    return f"{lower}:{lower + 2}"


def _validate_v3_noise_provenance(request: GenerationRequest) -> None:
    fields = (
        request.noise_start_sample,
        request.noise_source_sha256,
        request.noise_source_scene,
        request.resolved_noise_scene,
    )
    if not any(value is not None for value in fields):
        return
    if request.noise_path is None:
        raise ValueError("V3 noise provenance requires noise_path")
    if request.noise_start_sample is None or request.noise_start_sample < 0:
        raise ValueError("V3 noise provenance requires a non-negative noise_start_sample")
    if not request.noise_source_sha256 or not request.noise_source_scene or not request.resolved_noise_scene:
        raise ValueError("V3 noise provenance requires source SHA-256 and declared/resolved scenes")
    if request.recipe.noise_scene != request.noise_source_scene:
        raise ValueError("V3 declared recipe scene differs from noise source scene")
    if request.noise_source_scene != request.resolved_noise_scene:
        raise ValueError("V3 declared and resolved noise scenes differ")


def _crop_request_noise(
    request: GenerationRequest,
    device: torch.device,
    *,
    noise_cache: _LastNoiseSourceCache | None = None,
) -> torch.Tensor:
    if request.noise_path is None:
        raise ValueError("Noise crop requires noise_path")
    waveform = _load_16k(request.noise_path, device) if noise_cache is None else noise_cache.load(request.noise_path)
    if request.noise_start_sample is not None:
        return crop_complete_window(waveform, request.noise_start_sample)
    return crop_negative(waveform, request.recipe.seed)


def render_request(
    request: GenerationRequest,
    destination: Path,
    device: torch.device,
    *,
    noise_cache: _LastNoiseSourceCache | None = None,
    false_wake_cache: _FalseWakeSourceCache | None = None,
) -> dict[str, object]:
    """Render exactly one one-second example and return its verified provenance row."""
    _validate_example_id(request.example_id)
    recipe = request.recipe
    _validate_v3_noise_provenance(request)
    destination = Path(destination)
    noise_rms_band: str | None = None
    if recipe.source_kind == "pure_noise":
        if request.noise_path is None:
            raise ValueError("Pure-noise recipe requires noise_path")
        waveform = _crop_request_noise(request, device, noise_cache=noise_cache)
        noise_rms_band = _rms_band(waveform)
        rendered, guard_gain, clipped = peak_guard(waveform)
        active_span = None
        coverage_ratio = 1.0
        synthesis_metadata: dict[str, object] = {
            "active_rms_dbfs": None,
            "measured_snr_db": None,
            "measured_sir_db": None,
            "peak_guard_gain": guard_gain,
            "clipped_before_guard": clipped,
        }
    else:
        if request.foreground_path is None:
            raise ValueError("Speech recipe requires foreground_path")
        foreground = (
            false_wake_cache.load(request.foreground_path)
            if request.source_role == "false_wake_hard_negative" and false_wake_cache is not None
            else _load_16k(request.foreground_path, device)
        )
        boundary: Boundary | None = None
        if recipe.label == "positive" and request.active_span is None:
            detected = detect_positive_boundary(foreground.detach().cpu(), sample_rate=16000)
            if detected.end <= detected.start:
                raise ValueError(f"No active speech found in {request.foreground_path}")
            boundary = Boundary(detected.start, detected.end)
        elif recipe.label == "positive":
            assert request.active_span is not None
            boundary = Boundary(*request.active_span)
        if recipe.speed != 1.0:
            foreground = time_stretch_preserve_pitch(foreground, recipe.speed)
            if boundary is not None:
                boundary = remap_boundary_for_speed(boundary, recipe.speed)
        if recipe.label == "positive":
            assert boundary is not None
            rendered, crop_metadata = crop_positive(foreground, boundary, recipe)
            active_span = crop_metadata.output_active_span
            coverage_ratio = crop_metadata.coverage_ratio
        else:
            if request.source_role in {"false_wake_hard_negative", "raw_score_hard_negative"}:
                if request.foreground_start_sample is None:
                    raise ValueError("V3 exact hard negative requires foreground_start_sample")
                if recipe.speed != 1.0:
                    raise ValueError("V3 exact hard negative cannot apply speed augmentation")
                if request.source_role == "false_wake_hard_negative":
                    rendered = crop_complete_window(foreground, request.foreground_start_sample)
                else:
                    start = request.foreground_start_sample
                    if start < 0 or start >= foreground.shape[-1]:
                        raise ValueError("V3 raw-score hard negative has an invalid foreground_start_sample")
                    rendered = foreground[..., start : start + 16_000]
                    rendered = F.pad(rendered, (0, 16_000 - rendered.shape[-1]))
            else:
                rendered = crop_negative(foreground, recipe.seed)
            active_span = _active_span_for_negative(rendered)
            coverage_ratio = 1.0
        if active_span[1] <= active_span[0]:
            raise ValueError(f"Positive crop excludes all active speech for {request.example_id}")
        if recipe.apply_rir:
            if request.rir_path is None:
                raise ValueError("RIR recipe requires rir_path")
            rir = _load_16k(request.rir_path, device).squeeze(0)
            rendered = fft_convolve_rir(rendered, rir)
        noise = (
            None
            if request.noise_path is None
            else _crop_request_noise(request, device, noise_cache=noise_cache)
        )
        if noise is not None:
            noise_rms_band = _rms_band(noise)
        interferer = None if request.interferer_path is None else crop_negative(_load_16k(request.interferer_path, device), recipe.seed + 1)
        rendered, rendered_metadata = synthesize_speech_noise(rendered, active_span, noise, recipe, interferer=interferer)
        synthesis_metadata = {
            "active_rms_dbfs": rendered_metadata.active_rms_dbfs,
            "measured_snr_db": rendered_metadata.measured_snr_db,
            "measured_sir_db": rendered_metadata.measured_sir_db,
            "peak_guard_gain": rendered_metadata.peak_guard_gain,
            "clipped_before_guard": rendered_metadata.clipped_before_guard,
        }

    save_pcm16_atomic(destination, rendered, sample_rate=16000)
    return {
        "example_id": request.example_id,
        "label": recipe.label,
        "recipe": recipe.as_dict(),
        "foreground_path": None if request.foreground_path is None else str(request.foreground_path.resolve()),
        "noise_path": None if request.noise_path is None else str(request.noise_path.resolve()),
        "noise_start_sample": request.noise_start_sample,
        "noise_source_sha256": request.noise_source_sha256,
        "noise_source_scene": request.noise_source_scene,
        "noise_scene": recipe.noise_scene,
        "declared_noise_scene": recipe.noise_scene,
        "resolved_noise_scene": request.resolved_noise_scene,
        "rir_path": None if request.rir_path is None else str(request.rir_path.resolve()),
        "interferer_path": None if request.interferer_path is None else str(request.interferer_path.resolve()),
        "foreground_start_sample": request.foreground_start_sample,
        "source_role": request.source_role,
        "augmentation_closure_required": request.augmentation_closure_required,
        "source_split": recipe.split,
        "source_sha256": request.noise_source_sha256 or recipe.source_sha256,
        "source_start_sample": (
            request.noise_start_sample
            if request.noise_start_sample is not None
            else (request.foreground_start_sample or 0)
        ),
        "source_candidate_count": request.source_candidate_count,
        "sample_rate": 16000,
        "frames": int(rendered.shape[-1]),
        "noise_rms_band": noise_rms_band,
        "rir_applied": recipe.apply_rir,
        "rir_source_sha256": request.rir_source_sha256,
        "interferer_applied": recipe.apply_interferer,
        "interferer_source_sha256": request.interferer_source_sha256,
        "speed": recipe.speed,
        "foreground_rms_band": "not_applicable" if recipe.source_kind == "pure_noise" else str(recipe.active_rms_dbfs),
        "position_bin": "not_applicable" if recipe.source_kind == "pure_noise" else "rendered",
        "counterpart_kind": (
            request.counterpart_kind
            if request.counterpart_kind is not None
            else ("environment_noise_negative" if recipe.source_kind == "pure_noise" else None)
        ),
        "active_span": active_span,
        "coverage_ratio": coverage_ratio,
        "output_path": str(destination.resolve()),
        "output_sha256": sha256_file(destination),
        **synthesis_metadata,
    }


def iter_generation_requests(manifest_path: Path) -> Iterator[GenerationRequest]:
    for line_number, line in enumerate(Path(manifest_path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            yield GenerationRequest.from_dict(json.loads(line))
        except (json.JSONDecodeError, TypeError, ValueError, KeyError) as error:
            raise ValueError(f"Invalid generation request {manifest_path}:{line_number}") from error


def load_generation_requests(manifest_path: Path) -> list[GenerationRequest]:
    return list(iter_generation_requests(manifest_path))


def _request_category(request: GenerationRequest) -> str:
    if request.recipe.label == "positive":
        return "positive"
    if request.recipe.source_kind == "false_wake":
        return "false_wake_negative"
    if request.recipe.source_kind == "pure_noise":
        return "pure_noise_negative"
    return "speech_negative"


def stratified_pilot_requests(manifest_path: Path, total_examples: int) -> list[GenerationRequest]:
    if total_examples <= 0 or total_examples % 20:
        raise ValueError("Pilot example count must be a positive multiple of 20 for exact class quotas")
    quotas = {
        "positive": total_examples // 2,
        "speech_negative": total_examples * 13 // 40,
        "false_wake_negative": total_examples // 20,
        "pure_noise_negative": total_examples // 8,
    }
    selected: dict[str, list[GenerationRequest]] = {key: [] for key in quotas}
    for request in iter_generation_requests(manifest_path):
        category = _request_category(request)
        if len(selected[category]) < quotas[category]:
            selected[category].append(request)
        if all(len(selected[key]) == quotas[key] for key in quotas):
            break
    missing = {key: quotas[key] - len(rows) for key, rows in selected.items() if len(rows) < quotas[key]}
    if missing:
        raise ValueError(f"Generation manifest cannot satisfy pilot quotas: {missing}")
    return [request for category in quotas for request in selected[category]]


class _CompletionAppender:
    """Batch completion metadata syncs without weakening resumable rendering."""

    def __init__(self, path: Path, *, sync_interval: int = 64) -> None:
        if sync_interval < 1:
            raise ValueError("completion sync_interval must be positive")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")
        self._sync_interval = sync_interval
        self._pending = 0

    def append(self, metadata: dict[str, object]) -> None:
        self._handle.write(json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n")
        self._pending += 1
        if self._pending >= self._sync_interval:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._pending = 0

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._handle.close()


def generate_shard(
    requests: Iterable[GenerationRequest],
    output_root: Path,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    max_examples: int | None = None,
) -> dict[str, object]:
    output_root = Path(output_root).expanduser().resolve()
    shard_name = f"shard-{rank:02d}"
    completed_manifest = output_root / "metadata" / f"{shard_name}.jsonl"
    completed = _completed_example_ids(completed_manifest)
    start_time = time.perf_counter()
    bytes_written = 0
    owned_count = 0
    generated_count = 0
    noise_cache = _LastNoiseSourceCache(device)
    false_wake_cache = _FalseWakeSourceCache(device)
    completions = _CompletionAppender(completed_manifest)
    try:
        for request in requests:
            if _stable_shard(request.example_id, world_size) != rank:
                continue
            owned_count += 1
            if request.example_id in completed:
                continue
            if max_examples is not None and generated_count >= max_examples:
                continue
            destination = output_root / "audio" / request.recipe.split / shard_name / request.recipe.label / f"{request.example_id}.wav"
            metadata = render_request(
                request,
                destination,
                device,
                noise_cache=noise_cache,
                false_wake_cache=false_wake_cache,
            )
            bytes_written += destination.stat().st_size
            completions.append(metadata)
            generated_count += 1
    finally:
        completions.close()
    elapsed_seconds = time.perf_counter() - start_time
    telemetry = {
        "rank": rank,
        "world_size": world_size,
        "device": str(device),
        "owned_examples": owned_count,
        "generated_examples": generated_count,
        "elapsed_seconds": elapsed_seconds,
        "examples_per_second": generated_count / elapsed_seconds if elapsed_seconds else 0.0,
        "bytes_per_second": bytes_written / elapsed_seconds if elapsed_seconds else 0.0,
        "cuda_memory_allocated": int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0,
    }
    telemetry_path = output_root / "telemetry" / f"{shard_name}.json"
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_path.write_text(json.dumps(telemetry, ensure_ascii=False, indent=2), encoding="utf-8")
    return telemetry


def rank_device_from_environment() -> tuple[int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return rank, world_size, torch.device("cuda", local_rank)
    return rank, world_size, torch.device("cpu")
