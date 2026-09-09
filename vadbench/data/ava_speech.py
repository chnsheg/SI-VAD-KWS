from __future__ import annotations

import csv
import json
import math
import sys
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vadbench.audio import load_audio, save_audio
from vadbench.features import frame_count, log_mel_spectrogram, mfcc_features, samples_for_ms
from vadbench.manifest import ManifestRecord, write_manifest


AVA_SPEECH_LABEL_URL = "https://research.google.com/ava/download/ava_speech_labels_v1.csv"
SPEECH_LABELS = {"CLEAN_SPEECH", "SPEECH_WITH_MUSIC", "SPEECH_WITH_NOISE"}
NON_SPEECH_LABELS = {"NO_SPEECH"}
AVA_CLASS_IDS = {
    "NO_SPEECH": 0,
    "CLEAN_SPEECH": 1,
    "SPEECH_WITH_MUSIC": 2,
    "SPEECH_WITH_NOISE": 3,
}


@dataclass(frozen=True)
class AvaSpeechInterval:
    video_id: str
    start_sec: float
    end_sec: float
    label: str

    @property
    def is_speech(self) -> bool:
        return self.label in SPEECH_LABELS


@dataclass
class AvaSpeechConfig:
    cache_root: Path = Path(r"C:\vadbench_cache\ava_speech")
    manifest_out: Path = Path("manifests/ava_speech_manifest.jsonl")
    label_csv: Path | None = None
    download_labels: bool = False
    video_id_file: Path | None = None
    media_root: Path | None = None
    use_yt_dlp: bool = False
    cookies: Path | None = None
    cookies_from_browser: str | None = None
    failed_video_file: Path | None = None
    retry_failed_only: bool = False
    max_videos: int | None = None
    split_seed: int = 7
    sample_rate: int = 16000
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    chunk_sec: float = 30.0
    n_mels: int = 64
    extract_audio: bool = True
    precompute_features: bool = True
    keep_wav: bool = True


@dataclass(frozen=True)
class _AvailableVideo:
    video_id: str
    audio_path: Path
    intervals: list[AvaSpeechInterval]


def prepare_ava_speech(config: AvaSpeechConfig) -> Path:
    config.cache_root.mkdir(parents=True, exist_ok=True)
    _download_audio_with_ytdlp._cookies = config.cookies
    _download_audio_with_ytdlp._cookies_from_browser = config.cookies_from_browser
    labels_csv = _resolve_label_csv(config)
    intervals = parse_ava_speech_csv(labels_csv)
    selected_ids = _select_video_ids(intervals, config)

    available_videos: list[_AvailableVideo] = []
    records: list[ManifestRecord] = []
    failed_videos: list[dict[str, str]] = []
    target_success = config.max_videos
    for video_id in selected_ids:
        video_intervals = sorted([item for item in intervals if item.video_id == video_id], key=lambda item: item.start_sec)
        if not video_intervals:
            continue
        try:
            audio_path = _ensure_audio(video_id, config) if config.extract_audio else _expected_audio_path(video_id, config)
            if not audio_path.exists():
                raise FileNotFoundError(
                    f"Missing audio for {video_id}: {audio_path}. Provide --media-root or install yt-dlp and use --use-yt-dlp."
                )
            available_videos.append(_AvailableVideo(video_id=video_id, audio_path=audio_path, intervals=video_intervals))
            if target_success is not None and len(available_videos) >= target_success:
                break
        except Exception as exc:
            failed_videos.append({"video_id": video_id, "error": str(exc)})
            print(f"warning: skipping AVA video {video_id}: {exc}", file=sys.stderr)

    split_by_video = split_video_ids([item.video_id for item in available_videos], config.split_seed)
    for item in available_videos:
        try:
            waveform, sample_rate = load_audio(item.audio_path, config.sample_rate)
            video_records = _make_video_chunks(
                item.video_id,
                waveform,
                sample_rate,
                item.intervals,
                split_by_video[item.video_id],
                labels_csv,
                config,
            )
            if not video_records:
                raise RuntimeError(f"No chunks produced for {item.video_id}")
            records.extend(video_records)
        except Exception as exc:
            failed_videos.append({"video_id": item.video_id, "error": f"chunking failed: {exc}"})
            print(f"warning: skipping AVA video {item.video_id}: {exc}", file=sys.stderr)

    write_manifest(records, config.manifest_out)
    _write_metadata(config, labels_csv, selected_ids, records, failed_videos)
    return config.manifest_out


def parse_ava_speech_csv(path: str | Path) -> list[AvaSpeechInterval]:
    path = Path(path)
    intervals: list[AvaSpeechInterval] = []
    text = path.read_text(encoding="utf-8")
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\n", "\n")
    lines = text.splitlines()
    has_header = "video" in lines[0].lower() if lines else False
    if has_header:
        reader = csv.DictReader(lines)
        for row in reader:
            intervals.append(_interval_from_mapping(row))
    else:
        reader = csv.reader(lines)
        for row in reader:
            if not row or len(row) < 4:
                continue
            intervals.append(_interval_from_row(row))
    return [item for item in intervals if item.end_sec > item.start_sec]


def split_video_ids(video_ids: list[str], seed: int = 7) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    ids = sorted(set(video_ids))
    if not ids:
        return {}
    order = [str(item) for item in rng.permutation(ids)]
    n = len(order)
    if n == 1:
        return {order[0]: "train"}
    if n == 2:
        return {order[0]: "train", order[1]: "val"}
    if n == 3:
        return {order[0]: "train", order[1]: "val", order[2]: "test"}
    train_end = max(1, int(round(n * 0.8)))
    train_end = min(train_end, n - 2)
    val_end = max(train_end + 1, int(round(n * 0.9)))
    val_end = min(val_end, n - 1)
    result: dict[str, str] = {}
    for idx, video_id in enumerate(order):
        if idx < train_end:
            split = "train"
        elif idx < val_end:
            split = "val"
        else:
            split = "test"
        result[video_id] = split
    return result


def intervals_to_frame_labels(
    intervals: list[AvaSpeechInterval],
    chunk_start_sec: float,
    chunk_end_sec: float,
    sample_rate: int = 16000,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> np.ndarray:
    total_samples = int(round((chunk_end_sec - chunk_start_sec) * sample_rate))
    sample_mask = np.zeros(max(total_samples, 1), dtype=np.uint8)
    for interval in intervals:
        if not interval.is_speech:
            continue
        start = max(interval.start_sec, chunk_start_sec) - chunk_start_sec
        end = min(interval.end_sec, chunk_end_sec) - chunk_start_sec
        if end <= start:
            continue
        start_sample = max(0, int(round(start * sample_rate)))
        end_sample = min(len(sample_mask), int(round(end * sample_rate)))
        sample_mask[start_sample:end_sample] = 1
    n_frames = frame_count(len(sample_mask), sample_rate, frame_ms, hop_ms)
    labels = np.zeros(n_frames, dtype=np.uint8)
    frame_len = samples_for_ms(sample_rate, frame_ms)
    hop_len = samples_for_ms(sample_rate, hop_ms)
    for idx in range(n_frames):
        start = idx * hop_len
        end = min(len(sample_mask), start + frame_len)
        labels[idx] = 1 if end > start and float(np.mean(sample_mask[start:end])) >= 0.2 else 0
    return labels


def intervals_to_frame_class_labels(
    intervals: list[AvaSpeechInterval],
    chunk_start_sec: float,
    chunk_end_sec: float,
    sample_rate: int = 16000,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> np.ndarray:
    total_samples = int(round((chunk_end_sec - chunk_start_sec) * sample_rate))
    class_mask = np.zeros(max(total_samples, 1), dtype=np.uint8)
    for interval in intervals:
        class_id = AVA_CLASS_IDS.get(interval.label, 0)
        start = max(interval.start_sec, chunk_start_sec) - chunk_start_sec
        end = min(interval.end_sec, chunk_end_sec) - chunk_start_sec
        if end <= start:
            continue
        start_sample = max(0, int(round(start * sample_rate)))
        end_sample = min(len(class_mask), int(round(end * sample_rate)))
        if end_sample > start_sample:
            class_mask[start_sample:end_sample] = np.maximum(class_mask[start_sample:end_sample], class_id)
    n_frames = frame_count(len(class_mask), sample_rate, frame_ms, hop_ms)
    labels = np.zeros(n_frames, dtype=np.uint8)
    frame_len = samples_for_ms(sample_rate, frame_ms)
    hop_len = samples_for_ms(sample_rate, hop_ms)
    for idx in range(n_frames):
        start = idx * hop_len
        end = min(len(class_mask), start + frame_len)
        if end > start:
            counts = np.bincount(class_mask[start:end], minlength=4)
            speech_count = int(np.sum(counts[1:]))
            if speech_count / max(end - start, 1) >= 0.2:
                labels[idx] = int(np.argmax(counts[1:]) + 1)
            else:
                labels[idx] = 0
    return labels


def _interval_from_mapping(row: dict[str, str]) -> AvaSpeechInterval:
    lowered = {key.strip().lstrip("\ufeff").lower(): value for key, value in row.items() if key is not None}
    video_id = _first_value(lowered, ["video_id", "videoid", "video", "youtube_id", "id"])
    start = _first_value(lowered, ["start", "start_sec", "start_time", "time_start"])
    end = _first_value(lowered, ["end", "end_sec", "end_time", "time_end"])
    label = _first_value(lowered, ["label", "speech_label", "class"])
    return AvaSpeechInterval(video_id=video_id, start_sec=float(start), end_sec=float(end), label=label.strip())


def _interval_from_row(row: list[str]) -> AvaSpeechInterval:
    return AvaSpeechInterval(video_id=row[0].strip(), start_sec=float(row[1]), end_sec=float(row[2]), label=row[3].strip())


def _first_value(row: dict[str, str], names: list[str]) -> str:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    raise ValueError(f"Missing required AVA CSV column. Expected one of {names}; got {sorted(row)}")


def _resolve_label_csv(config: AvaSpeechConfig) -> Path:
    if config.label_csv is not None:
        return Path(config.label_csv)
    target = config.cache_root / "labels" / "ava_speech_labels_v1.csv"
    if target.exists():
        return target
    if not config.download_labels:
        raise FileNotFoundError("AVA label CSV not found. Pass --label-csv or --download-labels.")
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(AVA_SPEECH_LABEL_URL, target)
    return target


def _select_video_ids(intervals: list[AvaSpeechInterval], config: AvaSpeechConfig) -> list[str]:
    ids = sorted({item.video_id for item in intervals})
    if config.video_id_file is not None:
        wanted = {
            line.strip()
            for line in Path(config.video_id_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        ids = [video_id for video_id in ids if video_id in wanted]
    if config.retry_failed_only and config.failed_video_file is not None:
        wanted = _read_failed_video_ids(config.failed_video_file)
        ids = [video_id for video_id in ids if video_id in wanted]
    return ids


def _read_failed_video_ids(path: str | Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("failed_videos"), list):
            return {str(item.get("video_id")) for item in data["failed_videos"] if item.get("video_id")}
        if isinstance(data, list):
            ids: set[str] = set()
            for item in data:
                if isinstance(item, dict) and item.get("video_id"):
                    ids.add(str(item["video_id"]))
                elif isinstance(item, str):
                    ids.add(item)
            return ids
    return {
        line.strip().split(",", 1)[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def _expected_audio_path(video_id: str, config: AvaSpeechConfig) -> Path:
    return config.cache_root / "audio" / f"{video_id}.wav"


def _ensure_audio(video_id: str, config: AvaSpeechConfig) -> Path:
    target = _expected_audio_path(video_id, config)
    if target.exists():
        return target
    cached_chunks = sorted((config.cache_root / "chunks").glob(f"*/*{video_id}*.wav"))
    if cached_chunks:
        _restore_audio_from_chunks(cached_chunks, target, config.sample_rate)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    source = _find_local_media(video_id, config.media_root)
    if source is not None:
        _extract_audio(source, target, config.sample_rate)
        return target
    if config.use_yt_dlp:
        _download_audio_with_ytdlp(video_id, target, config.sample_rate)
        return target
    raise FileNotFoundError(
        f"No local media found for {video_id}. Put files under --media-root or use --use-yt-dlp after installing yt-dlp."
    )


def _restore_audio_from_chunks(chunks: list[Path], target: Path, sample_rate: int) -> None:
    if not chunks:
        return
    pieces: list[np.ndarray] = []
    for chunk in sorted(chunks):
        waveform, _ = load_audio(chunk, sample_rate)
        pieces.append(waveform)
    if pieces:
        save_audio(target, np.concatenate(pieces).astype(np.float32), sample_rate)


def _find_local_media(video_id: str, media_root: Path | None) -> Path | None:
    if media_root is None:
        return None
    root = Path(media_root)
    if not root.exists():
        return None
    suffixes = [".wav", ".flac", ".mp3", ".m4a", ".mp4", ".mkv", ".webm"]
    for suffix in suffixes:
        candidate = root / f"{video_id}{suffix}"
        if candidate.exists():
            return candidate
    matches = list(root.rglob(f"*{video_id}*"))
    for match in matches:
        if match.is_file() and match.suffix.lower() in suffixes:
            return match
    return None


def _extract_audio(source: Path, target: Path, sample_rate: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to extract AVA audio but was not found on PATH.")
    audio_suffixes = {".wav", ".flac", ".mp3", ".m4a"}
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    if source.suffix.lower() not in audio_suffixes:
        command.extend(["-ss", "900", "-t", "900"])
    command.extend(
        [
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            str(target),
        ]
    )
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        message = _summarize_process_failure(result.stderr or result.stdout)
        raise RuntimeError(f"yt-dlp failed for {video_id}: {message}")


def _download_audio_with_ytdlp(video_id: str, target: Path, sample_rate: int) -> None:
    yt_dlp = _find_ytdlp()
    if yt_dlp is None:
        raise RuntimeError(
            "yt-dlp is required for online AVA downloads but is not installed. "
            "Install it in eis with: C:\\myApps\\Miniconda\\envs\\eis\\python.exe -m pip install yt-dlp"
        )
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for online AVA downloads but was not found on PATH.")
    temp = target.with_suffix(".download.%(ext)s")
    url = f"https://www.youtube.com/watch?v={video_id}"
    command = [
        yt_dlp,
        "-f",
        "bestaudio/best",
        "-o",
        str(temp),
    ]
    cookies = getattr(_download_audio_with_ytdlp, "_cookies", None)
    cookies_from_browser = getattr(_download_audio_with_ytdlp, "_cookies_from_browser", None)
    if cookies:
        command.extend(["--cookies", str(cookies)])
    if cookies_from_browser:
        command.extend(["--cookies-from-browser", str(cookies_from_browser)])
    command.append(url)
    subprocess.run(command, check=True)
    candidates = sorted(target.parent.glob(f"{target.stem}.download.*"))
    if not candidates:
        raise FileNotFoundError(f"yt-dlp did not produce an audio file for {video_id}")
    _extract_audio(candidates[0], target, sample_rate)
    for candidate in candidates:
        candidate.unlink(missing_ok=True)


def _summarize_process_failure(output: str, max_lines: int = 4) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return "no error output"
    important = [line for line in lines if "ERROR:" in line or "WARNING:" in line or "Private video" in line or "Video unavailable" in line]
    selected = important[-max_lines:] if important else lines[-max_lines:]
    return " | ".join(selected)


def _find_ytdlp() -> str | None:
    found = shutil.which("yt-dlp")
    if found is not None:
        return found
    scripts_dir = Path(sys.executable).parent / "Scripts"
    candidates = [scripts_dir / "yt-dlp.exe", scripts_dir / "yt-dlp"]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _make_video_chunks(
    video_id: str,
    waveform: np.ndarray,
    sample_rate: int,
    intervals: list[AvaSpeechInterval],
    split: str,
    label_csv: Path,
    config: AvaSpeechConfig,
) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    label_origin_sec = min((item.start_sec for item in intervals), default=0.0)
    label_end_sec = max((item.end_sec for item in intervals), default=label_origin_sec)
    labeled_duration_sec = max(label_end_sec - label_origin_sec, 0.0)
    waveform = _crop_to_labeled_audio_window(waveform, sample_rate, label_origin_sec, label_end_sec)
    duration_sec = min(len(waveform) / sample_rate, labeled_duration_sec if labeled_duration_sec > 0 else len(waveform) / sample_rate)
    chunk_count = int(math.ceil(duration_sec / config.chunk_sec))
    for idx in range(chunk_count):
        start = idx * config.chunk_sec
        end = min(duration_sec, start + config.chunk_sec)
        if end - start < 1.0:
            continue
        chunk_id = f"{video_id}_{int(start * 1000):09d}_{int(end * 1000):09d}"
        sample_start = int(round(start * sample_rate))
        sample_end = int(round(end * sample_rate))
        chunk_waveform = waveform[sample_start:sample_end]
        label_start = start + label_origin_sec
        label_end = end + label_origin_sec
        chunk_intervals = [item for item in intervals if item.end_sec > label_start and item.start_sec < label_end]
        labels = intervals_to_frame_labels(chunk_intervals, label_start, label_end, sample_rate, config.frame_ms, config.hop_ms)
        class_labels = intervals_to_frame_class_labels(
            chunk_intervals,
            label_start,
            label_end,
            sample_rate,
            config.frame_ms,
            config.hop_ms,
        )

        audio_path = config.cache_root / "chunks" / split / f"{chunk_id}.wav"
        label_path = config.cache_root / "labels_frame" / split / f"{chunk_id}.npy"
        class_label_path = config.cache_root / "labels_class" / split / f"{chunk_id}.npy"
        feature_path = config.cache_root / "features" / split / f"{chunk_id}.logmel64.npy"
        mfcc_feature_path = config.cache_root / "features" / split / f"{chunk_id}.mfcc64.npy"
        if config.keep_wav:
            save_audio(audio_path, chunk_waveform, sample_rate)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(label_path, labels.astype(np.uint8))
        class_label_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(class_label_path, class_labels.astype(np.uint8))
        feature_value = None
        if config.precompute_features:
            features = log_mel_spectrogram(chunk_waveform, sample_rate, config.n_mels, config.frame_ms, config.hop_ms, normalize=True)
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(feature_path, features.astype(np.float32))
            mfcc = mfcc_features(
                chunk_waveform,
                sample_rate,
                n_mfcc=config.n_mels,
                n_mels=config.n_mels,
                frame_ms=config.frame_ms,
                hop_ms=config.hop_ms,
                normalize=True,
            )
            np.save(mfcc_feature_path, mfcc.astype(np.float32))
            feature_value = str(feature_path)

        records.append(
            ManifestRecord(
                id=chunk_id,
                audio_path=str(audio_path),
                label_path=str(label_path),
                split=split,
                sample_rate=sample_rate,
                duration_sec=float(end - start),
                frame_hop_ms=config.hop_ms,
                source="AVA-Speech",
                feature_path=feature_value,
                video_id=video_id,
                chunk_start_sec=float(start),
                chunk_end_sec=float(end),
                label_source=str(label_csv),
                class_label_path=str(class_label_path),
            )
        )
    return records


def _crop_to_labeled_audio_window(
    waveform: np.ndarray,
    sample_rate: int,
    label_origin_sec: float,
    label_end_sec: float,
) -> np.ndarray:
    labeled_duration_sec = max(label_end_sec - label_origin_sec, 0.0)
    if labeled_duration_sec <= 0:
        return waveform
    duration_sec = len(waveform) / sample_rate
    if duration_sec >= label_end_sec - 1.0:
        start_sample = max(0, int(round(label_origin_sec * sample_rate)))
        end_sample = min(len(waveform), int(round(label_end_sec * sample_rate)))
        return waveform[start_sample:end_sample]
    if duration_sec > labeled_duration_sec + 1.0:
        end_sample = int(round(labeled_duration_sec * sample_rate))
        return waveform[:end_sample]
    return waveform


def _write_metadata(
    config: AvaSpeechConfig,
    label_csv: Path,
    selected_ids: list[str],
    records: list[ManifestRecord],
    failed_videos: list[dict[str, str]],
) -> None:
    successful_video_ids = sorted({record.video_id for record in records if record.video_id is not None})
    split_counts: dict[str, int] = {}
    split_frame_totals: dict[str, int] = {}
    split_speech_frames: dict[str, int] = {}
    for record in records:
        split_counts[record.split] = split_counts.get(record.split, 0) + 1
        labels = np.load(record.label_path)
        split_frame_totals[record.split] = split_frame_totals.get(record.split, 0) + int(labels.size)
        split_speech_frames[record.split] = split_speech_frames.get(record.split, 0) + int(labels.sum())
    speech_frame_ratio = {
        split: float(split_speech_frames.get(split, 0) / max(split_frame_totals.get(split, 0), 1))
        for split in sorted(split_counts)
    }
    metadata = {
        "source": "AVA-Speech",
        "label_csv": str(label_csv),
        "cache_root": str(config.cache_root),
        "attemptable_video_count": len(selected_ids),
        "successful_video_count": len(successful_video_ids),
        "successful_video_ids": successful_video_ids,
        "failed_videos": failed_videos,
        "failed_video_count": len(failed_videos),
        "chunk_count": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "speech_frame_ratio": speech_frame_ratio,
        "sample_rate": config.sample_rate,
        "frame_ms": config.frame_ms,
        "hop_ms": config.hop_ms,
        "chunk_sec": config.chunk_sec,
        "n_mels": config.n_mels,
    }
    path = Path(config.manifest_out).with_suffix(".metadata.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
