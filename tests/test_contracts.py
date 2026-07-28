from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from vadbench.algorithms.neural import (
    CausalCRNNVADKWS,
    CausalCRNNVADMicro,
    CausalCRNNVADNano,
    CausalCRNNVADTiny,
    CausalDSCNNGRUVADKWS,
    DSCNNVADKWSMatch,
    DSCNNVADLarge,
    DSCNNVADMedium,
    DSCNNVADSmall,
    DSCNNVADTiny,
    estimate_model_stats,
)
from vadbench.algorithms.registry import create_algorithm, list_algorithms
from vadbench.audio import save_audio
from vadbench.data.aishell4_realneg_vad import (
    Aishell4RealnegVADConfig,
    aishell4_room_type,
    fsd50k_source_split,
    intervals_to_frame_labels as aishell4_intervals_to_frame_labels,
    merge_intervals,
    parse_rttm,
    prepare_aishell4_realneg_vad,
    scan_fsd50k_nonspeech,
    split_aishell4_recordings,
    split_recording_ids,
    SpeechInterval,
)
from vadbench.data.ava_speech import (
    AvaSpeechConfig,
    intervals_to_frame_class_labels,
    intervals_to_frame_labels,
    parse_ava_speech_csv,
    prepare_ava_speech,
    split_video_ids,
)
from vadbench.data.ms_snsd_vad import MS_SNSD_EVENT_CLASSES, MSSNSDVADConfig, prepare_ms_snsd_vad, scan_ms_snsd_sources, split_clean_files
from vadbench.features import frame_count, sample_mask_to_frame_labels
from vadbench.frame_prediction import FramePrediction
from vadbench.manifest import ManifestRecord, read_manifest, validate_manifest, write_manifest
from vadbench.metrics import ava_paper_metrics, roc_auc, threshold_for_fpr
from vadbench.cli import _latency_eval_config, _latency_train_config, _model_stats_with_context, _select_latency_best
from vadbench.cli import _finalize_source_metrics, _accumulate_source_metrics


class ContractTests(unittest.TestCase):
    def test_frame_labels_match_frame_count(self) -> None:
        sample_rate = 16000
        samples = np.zeros(sample_rate, dtype=bool)
        samples[1600:6400] = True
        labels = sample_mask_to_frame_labels(samples, sample_rate)
        self.assertEqual(len(labels), frame_count(len(samples), sample_rate))
        self.assertGreater(labels.sum(), 0)

    def test_segments_merge_short_silence_and_drop_short_speech(self) -> None:
        scores = np.array([0.9, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9], dtype=np.float32)
        prediction = FramePrediction(scores=scores, frame_hop_ms=10.0)
        segments = prediction.to_segments(threshold=0.5, min_speech_ms=20.0, min_silence_ms=30.0)
        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(segments[0]["start_sec"], 0.0)
        self.assertAlmostEqual(segments[0]["end_sec"], 0.07)

    def test_manifest_roundtrip_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = np.zeros(1600, dtype=np.float32)
            labels = np.zeros(10, dtype=np.uint8)
            class_labels = np.zeros(10, dtype=np.uint8)
            save_audio(root / "audio.wav", audio, 16000)
            np.save(root / "labels.npy", labels)
            np.save(root / "classes.npy", class_labels)
            record = ManifestRecord(
                id="x",
                audio_path="audio.wav",
                label_path="labels.npy",
                split="train",
                sample_rate=16000,
                duration_sec=0.1,
                frame_hop_ms=10.0,
                source="unit",
                feature_path="features.npy",
                video_id="video",
                chunk_start_sec=0.0,
                chunk_end_sec=0.1,
                label_source="unit.csv",
                class_label_path="classes.npy",
            )
            np.save(root / "features.npy", np.zeros((10, 64), dtype=np.float32))
            manifest = root / "manifest.jsonl"
            write_manifest([record], manifest)
            loaded = read_manifest(manifest)
            self.assertEqual(loaded[0].id, "x")
            self.assertEqual(loaded[0].feature_path, "features.npy")
            self.assertEqual(loaded[0].video_id, "video")
            self.assertEqual(loaded[0].class_label_path, "classes.npy")
            self.assertEqual(len(validate_manifest(manifest)), 1)

    def test_algorithm_registry_and_prediction_lengths(self) -> None:
        names = list_algorithms()
        self.assertIn("energy_adaptive", names)
        self.assertIn("tiny_mel_cnn", names)
        self.assertIn("marblenet_3x2x64", names)
        self.assertIn("crnn_vad", names)
        self.assertIn("dscnn_vad_tiny", names)
        self.assertIn("dscnn_vad_small", names)
        self.assertIn("dscnn_vad_medium", names)
        self.assertIn("dscnn_vad_large", names)
        self.assertIn("dscnn_vad_kws_match", names)
        self.assertIn("causal_crnn_vad_micro", names)
        self.assertIn("causal_crnn_vad_nano", names)
        self.assertIn("causal_crnn_vad_tiny", names)
        self.assertIn("causal_crnn_vad_kws", names)
        self.assertIn("causal_dscnn_gru_vad_kws", names)
        sample_rate = 16000
        waveform = np.zeros(sample_rate, dtype=np.float32)
        waveform[2000:7000] = 0.2 * np.sin(2 * np.pi * 220 * np.arange(5000) / sample_rate).astype(np.float32)
        expected = frame_count(len(waveform), sample_rate)
        algorithm_names = [
            "energy_adaptive",
            "zcr_energy",
            "spectral_gate",
            "mfcc_gmm",
            "kaldi_energy",
            "spectral_flux_ltsd",
            "sohn_hmm",
            "rvad_fast",
            "tiny_mel_cnn",
            "marblenet_lite",
            "attn_tcn_lite",
            "marblenet_3x2x64",
            "cnn_td_like",
            "crnn_vad",
            "self_attentive_vad",
            "dscnn_vad_tiny",
            "dscnn_vad_small",
            "dscnn_vad_medium",
            "dscnn_vad_large",
            "dscnn_vad_kws_match",
            "causal_crnn_vad_nano",
            "causal_crnn_vad_tiny",
            "causal_crnn_vad_micro",
            "causal_crnn_vad_kws",
            "causal_dscnn_gru_vad_kws",
        ]
        try:
            import webrtcvad  # noqa: F401

            algorithm_names.append("webrtc_vad")
        except ImportError:
            pass
        for name in algorithm_names:
            algorithm = create_algorithm(name, sample_rate=sample_rate, device="cpu")
            prediction = algorithm.predict(waveform, sample_rate)
            self.assertEqual(len(prediction.scores), expected, name)
            self.assertTrue(np.all(prediction.scores >= 0.0), name)
            self.assertTrue(np.all(prediction.scores <= 1.0), name)

    def test_dscnn_vad_output_shape(self) -> None:
        for model_cls in (DSCNNVADTiny, DSCNNVADSmall, DSCNNVADMedium, DSCNNVADLarge, DSCNNVADKWSMatch):
            model = model_cls(n_mels=64)
            model.eval()
            with torch.no_grad():
                for frames in (17, 63, 101):
                    y = model(torch.zeros(2, frames, 64, dtype=torch.float32))
                    self.assertEqual(tuple(y.shape), (2, frames))

    def test_causal_lightweight_models_output_shape_and_causality(self) -> None:
        for model_cls in (CausalCRNNVADNano, CausalCRNNVADTiny, CausalCRNNVADMicro, CausalCRNNVADKWS, CausalDSCNNGRUVADKWS):
            model = model_cls(n_mels=64)
            model.eval()
            with torch.no_grad():
                for frames in (17, 100, 301):
                    y = model(torch.zeros(2, frames, 64, dtype=torch.float32))
                    self.assertEqual(tuple(y.shape), (2, frames))
                x1 = torch.randn(1, 100, 64)
                x2 = x1.clone()
                x2[:, 60:, :] += 5.0
                y1 = model(x1)
                y2 = model(x2)
                self.assertTrue(torch.allclose(y1[:, :50], y2[:, :50], atol=1e-5), model_cls.__name__)

    def test_model_stats_for_kws_scale_causal_models(self) -> None:
        for model_cls in (CausalCRNNVADNano, CausalCRNNVADTiny, CausalCRNNVADMicro, CausalCRNNVADKWS, CausalDSCNNGRUVADKWS):
            stats = estimate_model_stats(model_cls(n_mels=64), n_mels=64, frames=1)
            self.assertGreater(stats["params"], 0.0)
            self.assertGreater(stats["macs_per_frame"], 0.0)
        nano_stats = estimate_model_stats(CausalCRNNVADNano(n_mels=64), n_mels=64, frames=1)
        tiny_stats = estimate_model_stats(CausalCRNNVADTiny(n_mels=64), n_mels=64, frames=1)
        self.assertLess(nano_stats["params"], tiny_stats["params"])
        self.assertLess(tiny_stats["params"], 10000.0)
        kws_stats = estimate_model_stats(CausalCRNNVADKWS(n_mels=64), n_mels=64, frames=1)
        self.assertLessEqual(kws_stats["params"], 25000.0)

    def test_ava_csv_and_frame_label_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "ava.csv"
            csv_path.write_text(
                "video_id,start,end,label\n"
                "vid,0.0,0.5,NO_SPEECH\n"
                "vid,0.5,1.0,CLEAN_SPEECH\n",
                encoding="utf-8",
            )
            intervals = parse_ava_speech_csv(csv_path)
            self.assertEqual(len(intervals), 2)
            labels = intervals_to_frame_labels(intervals, 0.0, 1.0, sample_rate=16000)
            class_labels = intervals_to_frame_class_labels(intervals, 0.0, 1.0, sample_rate=16000)
            self.assertEqual(len(labels), frame_count(16000, 16000))
            self.assertEqual(len(class_labels), len(labels))
            self.assertEqual(int(labels[:40].sum()), 0)
            self.assertEqual(int(class_labels[:40].sum()), 0)
            self.assertGreater(int(labels[55:].sum()), 20)
            self.assertTrue(np.all(class_labels[55:] == 1))

    def test_ava_absolute_label_origin_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "ava.csv"
            csv_path.write_text(
                "video_id,start,end,label\n"
                "vidA,900.0,901.0,NO_SPEECH\n"
                "vidA,901.0,902.0,CLEAN_SPEECH\n"
                "vidA,902.0,903.0,NO_SPEECH\n"
                "vidA,903.0,904.0,SPEECH_WITH_NOISE\n",
                encoding="utf-8",
            )
            intervals = parse_ava_speech_csv(csv_path)
            label_origin = min(item.start_sec for item in intervals)
            labels = intervals_to_frame_labels(intervals, label_origin, label_origin + 4.0, sample_rate=16000)
            class_labels = intervals_to_frame_class_labels(intervals, label_origin, label_origin + 4.0, sample_rate=16000)
            self.assertEqual(len(labels), frame_count(16000 * 4, 16000))
            self.assertEqual(len(class_labels), len(labels))
            self.assertEqual(int(labels[:80].sum()), 0)
            self.assertGreater(int(labels[110:180].sum()), 40)
            self.assertEqual(int(labels[220:280].sum()), 0)
            self.assertGreater(int(labels[310:].sum()), 40)
            self.assertEqual(int(class_labels[:80].sum()), 0)
            self.assertTrue(np.all(class_labels[110:180] == 1))
            self.assertTrue(np.all(class_labels[310:] == 3))

    def test_split_video_ids_keeps_video_in_single_split(self) -> None:
        split = split_video_ids([f"vid{idx:03d}" for idx in range(20)], seed=7)
        self.assertEqual(set(split.values()), {"train", "val", "test"})
        self.assertEqual(len(split), 20)

    def test_ava_paper_metrics(self) -> None:
        y_class = np.array([0, 0, 0, 0, 1, 1, 2, 2, 3, 3], dtype=np.uint8)
        scores = np.array([0.05, 0.10, 0.20, 0.95, 0.60, 0.80, 0.70, 0.90, 0.55, 0.99], dtype=np.float32)
        threshold = threshold_for_fpr(y_class > 0, scores, target_fpr=0.25)
        self.assertAlmostEqual(threshold, 0.95, places=6)
        metrics = ava_paper_metrics(y_class, scores, target_fpr=0.25)
        self.assertAlmostEqual(metrics["threshold_at_fpr"], 0.95)
        self.assertAlmostEqual(metrics["fpr"], 0.25)
        self.assertAlmostEqual(metrics["tpr_all"], 1.0 / 6.0)
        self.assertGreater(roc_auc(y_class > 0, scores), 0.5)

    def test_prepare_ava_speech_with_local_audio_and_cached_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "media"
            media.mkdir()
            sample_rate = 16000
            wav = np.zeros(sample_rate * 4, dtype=np.float32)
            wav[sample_rate : sample_rate * 2] = 0.2 * np.sin(2 * np.pi * 220 * np.arange(sample_rate) / sample_rate)
            wav[sample_rate * 3 :] = 0.2 * np.sin(2 * np.pi * 330 * np.arange(sample_rate) / sample_rate)
            save_audio(media / "vidA.wav", wav, sample_rate)
            csv_path = root / "ava.csv"
            csv_path.write_text(
                "video_id,start,end,label\n"
                "vidA,0.0,1.0,NO_SPEECH\n"
                "vidA,1.0,2.0,CLEAN_SPEECH\n"
                "vidA,2.0,3.0,SPEECH_WITH_NOISE\n"
                "vidA,3.0,4.0,NO_SPEECH\n",
                encoding="utf-8",
            )
            manifest = root / "manifest.jsonl"
            out = prepare_ava_speech(
                AvaSpeechConfig(
                    cache_root=root / "cache",
                    manifest_out=manifest,
                    label_csv=csv_path,
                    media_root=media,
                    chunk_sec=2.0,
                    sample_rate=sample_rate,
                    precompute_features=True,
                )
            )
            records = validate_manifest(out)
            self.assertEqual(len(records), 2)
            self.assertTrue(all(record.feature_path for record in records))
            self.assertTrue(all(record.class_label_path for record in records))
            features = np.load(records[0].resolve_feature(out.parent))
            labels = np.load(records[0].resolve_label(out.parent))
            class_labels = np.load(records[0].resolve_class_label(out.parent))
            self.assertEqual(features.shape[0], labels.shape[0])
            self.assertEqual(len(class_labels), len(labels))
            self.assertTrue(set(np.unique(class_labels)).issubset({0, 1, 3}))

    def test_prepare_ava_speech_crops_full_video_audio_to_labeled_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "media"
            media.mkdir()
            sample_rate = 16000
            wav = np.zeros(sample_rate * 6, dtype=np.float32)
            wav[sample_rate * 3 : sample_rate * 4] = 0.2
            save_audio(media / "vidB.wav", wav, sample_rate)
            csv_path = root / "ava.csv"
            csv_path.write_text(
                "video_id,start,end,label\n"
                "vidB,2.0,3.0,NO_SPEECH\n"
                "vidB,3.0,4.0,CLEAN_SPEECH\n"
                "vidB,4.0,5.0,NO_SPEECH\n",
                encoding="utf-8",
            )
            manifest = root / "manifest.jsonl"
            out = prepare_ava_speech(
                AvaSpeechConfig(
                    cache_root=root / "cache",
                    manifest_out=manifest,
                    label_csv=csv_path,
                    media_root=media,
                    chunk_sec=1.0,
                    sample_rate=sample_rate,
                    precompute_features=True,
                )
            )
            records = validate_manifest(out)
            self.assertEqual(len(records), 3)
            self.assertAlmostEqual(sum(record.duration_sec for record in records), 3.0)
            self.assertGreater(int(np.load(records[1].resolve_label(out.parent)).sum()), 20)

    def test_ms_snsd_source_scan_and_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean = root / "CleanSpeech"
            noise = root / "Noise"
            clean.mkdir()
            noise.mkdir()
            for idx in range(5):
                save_audio(clean / f"clean_{idx}.wav", np.ones(1600, dtype=np.float32) * 0.1, 16000)
            for idx in range(2):
                save_audio(noise / f"noise_{idx}.wav", np.ones(1600, dtype=np.float32) * 0.01, 16000)
            clean_files, noise_files = scan_ms_snsd_sources(root)
            self.assertEqual(len(clean_files), 5)
            self.assertEqual(len(noise_files), 2)
            split_a = split_clean_files(clean_files, seed=7)
            split_b = split_clean_files(clean_files, seed=7)
            self.assertEqual(split_a, split_b)
            self.assertEqual(set(split_a), {"train", "val", "test"})

    def test_prepare_ms_snsd_vad_with_fake_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            clean = raw / "CleanSpeech"
            noise = raw / "Noise"
            clean.mkdir(parents=True)
            noise.mkdir(parents=True)
            sample_rate = 16000
            t = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
            for idx, freq in enumerate((220.0, 330.0, 440.0)):
                wav = 0.25 * np.sin(2 * np.pi * freq * t).astype(np.float32)
                wav[:800] = 0.0
                save_audio(clean / f"clean_{idx}.wav", wav, sample_rate)
            save_audio(noise / "noise_0.wav", np.random.default_rng(1).normal(0.0, 0.02, sample_rate * 3).astype(np.float32), sample_rate)
            save_audio(noise / "Typing_0.wav", np.random.default_rng(2).normal(0.0, 0.02, sample_rate * 3).astype(np.float32), sample_rate)
            save_audio(noise / "Babble_0.wav", np.random.default_rng(3).normal(0.0, 0.02, sample_rate * 3).astype(np.float32), sample_rate)

            manifest = prepare_ms_snsd_vad(
                MSSNSDVADConfig(
                    raw_root=raw,
                    out_dir=root / "ms_snsd_vad",
                    protocol="v2",
                    clip_sec=1.0,
                    total_hours=0.001,
                    snr_levels=[10.0],
                    seed=7,
                    precompute_features=True,
                )
            )
            records = validate_manifest(manifest)
            self.assertEqual(set(record.split for record in records), {"train", "val", "test"})
            self.assertTrue(all(record.feature_path for record in records))
            self.assertTrue(all(record.class_label_path for record in records))
            seen_events: set[int] = set()
            for record in records:
                labels = np.load(record.resolve_label(manifest.parent))
                class_labels = np.load(record.resolve_class_label(manifest.parent))
                features = np.load(record.resolve_feature(manifest.parent))
                self.assertEqual(features.shape[0], labels.shape[0])
                self.assertEqual(features.shape[1], 64)
                self.assertEqual(len(class_labels), len(labels))
                seen_events.update(int(item) for item in np.unique(class_labels))
                self.assertGreater(int(labels.sum()), 0)
                self.assertLess(int(labels.sum()), len(labels))
            metadata = json.loads((manifest.parent / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["source"], "MS-SNSD-derived-VAD")
            self.assertEqual(metadata["protocol"], "v2")
            self.assertGreaterEqual(metadata["speech_frame_ratio"], 0.35)
            self.assertLessEqual(metadata["speech_frame_ratio"], 0.55)
            self.assertGreater(metadata["silence_frame_ratio"], 0.0)
            self.assertGreater(metadata["noise_only_frame_ratio"], 0.0)
            self.assertGreater(metadata["hard_negative_frame_ratio"], 0.0)
            self.assertEqual(metadata["excluded_noise_file_count"], 1)
            self.assertTrue({MS_SNSD_EVENT_CLASSES["clean_speech"], MS_SNSD_EVENT_CLASSES["speech_with_noise"]}.issubset(seen_events))
            self.assertIn(MS_SNSD_EVENT_CLASSES["hard_negative_event"], seen_events)

    def test_ms_snsd_variable_frame_hop_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            clean = raw / "clean_train"
            noise = raw / "noise_train"
            clean.mkdir(parents=True)
            noise.mkdir(parents=True)
            sample_rate = 16000
            t = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
            for idx, freq in enumerate((220.0, 330.0, 440.0)):
                save_audio(clean / f"clean_{idx}.wav", 0.25 * np.sin(2 * np.pi * freq * t).astype(np.float32), sample_rate)
            save_audio(noise / "noise_0.wav", np.random.default_rng(3).normal(0.0, 0.02, sample_rate * 3).astype(np.float32), sample_rate)
            save_audio(noise / "Typing_0.wav", np.random.default_rng(4).normal(0.0, 0.02, sample_rate * 3).astype(np.float32), sample_rate)
            manifest = prepare_ms_snsd_vad(
                MSSNSDVADConfig(
                    raw_root=raw,
                    out_dir=root / "ms_snsd_vad_f20_h20",
                    protocol="v2",
                    frame_ms=20.0,
                    hop_ms=20.0,
                    clip_sec=1.0,
                    total_hours=0.001,
                    snr_levels=[10.0],
                    seed=7,
                    precompute_features=True,
                )
            )
            for record in validate_manifest(manifest):
                labels = np.load(record.resolve_label(manifest.parent))
                features = np.load(record.resolve_feature(manifest.parent))
                self.assertEqual(features.shape[0], labels.shape[0])
                self.assertEqual(record.frame_hop_ms, 20.0)

    def test_aishell4_rttm_union_and_frame_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rttm = root / "sessionA.rttm"
            rttm.write_text(
                "SPEAKER sessionA 1 0.00 0.50 <NA> <NA> spk1 <NA> <NA>\n"
                "SPEAKER sessionA 1 0.25 0.50 <NA> <NA> spk2 <NA> <NA>\n"
                "SPEAKER sessionA 1 1.50 0.25 <NA> <NA> spk1 <NA> <NA>\n",
                encoding="utf-8",
            )
            intervals = parse_rttm(rttm)
            merged = merge_intervals(intervals)
            self.assertEqual(len(merged), 2)
            self.assertAlmostEqual(merged[0].start_sec, 0.0)
            self.assertAlmostEqual(merged[0].end_sec, 0.75)
            config = Aishell4RealnegVADConfig(chunk_sec=2.0)
            labels = aishell4_intervals_to_frame_labels(merged, 0.0, 2.0, config)
            self.assertEqual(len(labels), frame_count(32000, 16000))
            self.assertGreater(int(labels.sum()), 70)
            self.assertLess(int(labels.sum()), 120)

    def test_aishell4_room_and_official_test_split(self) -> None:
        selected = {
            "L001": Path("data/raw/aishell4/train_L/wav/L001.wav"),
            "M001": Path("data/raw/aishell4/train_M/wav/M001.wav"),
            "S001": Path("data/raw/aishell4/train_S/wav/S001.wav"),
            "T001": Path("data/raw/aishell4/test/wav/T001.wav"),
        }
        self.assertEqual(aishell4_room_type(selected["L001"]), "L")
        self.assertEqual(aishell4_room_type(selected["M001"]), "M")
        self.assertEqual(aishell4_room_type(selected["S001"]), "S")
        split = split_aishell4_recordings(selected, seed=7, use_official_test=True)
        self.assertEqual(split["T001"], "test")
        self.assertNotIn("test", {split["L001"], split["M001"], split["S001"]})

    def test_fsd50k_filter_excludes_speech_like_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_dir = root / "FSD50K.dev_audio"
            meta_dir = root / "FSD50K.ground_truth"
            audio_dir.mkdir(parents=True)
            meta_dir.mkdir(parents=True)
            save_audio(audio_dir / "100.wav", np.zeros(16000, dtype=np.float32), 16000)
            save_audio(audio_dir / "101.wav", np.zeros(16000, dtype=np.float32), 16000)
            (meta_dir / "dev.csv").write_text(
                "fname,labels,split\n"
                "100,Door;Knock,train\n"
                "101,Speech;Conversation,train\n",
                encoding="utf-8",
            )
            included, excluded = scan_fsd50k_nonspeech(root, ["speech", "conversation"])
            self.assertEqual([item.item_id for item in included], ["100"])
            self.assertEqual(len(excluded), 1)
            self.assertEqual(fsd50k_source_split(included[0].audio_path), "dev")

    def test_prepare_aishell4_realneg_vad_with_fake_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aishell = root / "aishell4"
            fsd = root / "fsd50k"
            audio_dir = aishell / "wav"
            ann_dir = aishell / "rttm"
            fsd_audio = fsd / "FSD50K.dev_audio"
            fsd_meta = fsd / "FSD50K.ground_truth"
            audio_dir.mkdir(parents=True)
            ann_dir.mkdir(parents=True)
            fsd_audio.mkdir(parents=True)
            fsd_meta.mkdir(parents=True)
            sample_rate = 16000
            t = np.arange(sample_rate * 3, dtype=np.float32) / sample_rate
            for idx, name in enumerate(["meetA", "meetB", "meetC"]):
                wav = 0.2 * np.sin(2 * np.pi * (220 + idx * 40) * t).astype(np.float32)
                save_audio(audio_dir / f"{name}.wav", wav, sample_rate)
                (ann_dir / f"{name}.rttm").write_text(
                    f"SPEAKER {name} 1 0.20 0.80 <NA> <NA> spk1 <NA> <NA>\n"
                    f"SPEAKER {name} 1 1.40 0.60 <NA> <NA> spk2 <NA> <NA>\n",
                    encoding="utf-8",
                )
            save_audio(fsd_audio / "200.wav", np.random.default_rng(1).normal(0.0, 0.02, sample_rate * 2).astype(np.float32), sample_rate)
            save_audio(fsd_audio / "201.wav", np.random.default_rng(2).normal(0.0, 0.02, sample_rate * 2).astype(np.float32), sample_rate)
            (fsd_meta / "dev.csv").write_text(
                "fname,labels,split\n"
                "200,Door;Knock,train\n"
                "201,Speech;Conversation,train\n",
                encoding="utf-8",
            )
            manifest = prepare_aishell4_realneg_vad(
                Aishell4RealnegVADConfig(
                    aishell4_root=aishell,
                    fsd50k_root=fsd,
                    out_dir=root / "aishell4_realneg_vad",
                    chunk_sec=1.0,
                    negative_ratio=0.5,
                    seed=7,
                    precompute_features=True,
                )
            )
            records = validate_manifest(manifest)
            self.assertTrue(any(record.source == "AISHELL4" for record in records))
            self.assertTrue(any(record.source == "FSD50K-hard-negative" for record in records))
            split_by_meeting = {}
            for record in records:
                if record.source == "AISHELL4":
                    previous = split_by_meeting.setdefault(record.video_id, record.split)
                    self.assertEqual(previous, record.split)
                labels = np.load(record.resolve_label(manifest.parent))
                features = np.load(record.resolve_feature(manifest.parent))
                class_labels = np.load(record.resolve_class_label(manifest.parent))
                self.assertEqual(features.shape[0], labels.shape[0])
                self.assertEqual(len(class_labels), len(labels))
                if record.source == "FSD50K-hard-negative":
                    self.assertEqual(int(labels.sum()), 0)
                    self.assertTrue(np.all(class_labels == 2))
            metadata = json.loads((manifest.parent / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["source"], "AISHELL4-realneg-VAD")
            self.assertGreater(metadata["fsd50k_excluded_item_count"], 0)

    def test_prepare_aishell4_realneg_subset_with_fake_lms_and_fsd_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aishell = root / "aishell4"
            fsd = root / "fsd50k"
            sample_rate = 16000
            t = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
            for subset, prefix in [("train_L", "L"), ("train_M", "M"), ("train_S", "S"), ("test", "T")]:
                audio_dir = aishell / subset / "wav"
                ann_dir = aishell / subset / "rttm"
                audio_dir.mkdir(parents=True)
                ann_dir.mkdir(parents=True)
                for idx in range(2):
                    name = f"{prefix}{idx:03d}"
                    wav = 0.2 * np.sin(2 * np.pi * (200 + idx * 20) * t).astype(np.float32)
                    save_audio(audio_dir / f"{name}.wav", wav, sample_rate)
                    (ann_dir / f"{name}.rttm").write_text(
                        f"SPEAKER {name} 1 0.10 0.70 <NA> <NA> spk1 <NA> <NA>\n",
                        encoding="utf-8",
                    )
            dev_audio = fsd / "FSD50K.dev_audio"
            eval_audio = fsd / "FSD50K.eval_audio"
            meta = fsd / "FSD50K.ground_truth"
            dev_audio.mkdir(parents=True)
            eval_audio.mkdir(parents=True)
            meta.mkdir(parents=True)
            for item_id, audio_dir in [("300", dev_audio), ("301", dev_audio), ("400", eval_audio), ("401", eval_audio)]:
                save_audio(audio_dir / f"{item_id}.wav", np.random.default_rng(int(item_id)).normal(0.0, 0.02, sample_rate).astype(np.float32), sample_rate)
            (meta / "all.csv").write_text(
                "fname,labels\n"
                "300,Door;Knock\n"
                "301,Engine\n"
                "400,Clock\n"
                "401,Speech;Conversation\n",
                encoding="utf-8",
            )
            manifest = prepare_aishell4_realneg_vad(
                Aishell4RealnegVADConfig(
                    aishell4_root=aishell,
                    fsd50k_root=fsd,
                    out_dir=root / "aishell4_realneg_vad_20h",
                    chunk_sec=1.0,
                    target_hours=0.003,
                    aishell4_hours=0.002,
                    fsd50k_hours=0.001,
                    aishell4_subsets=["train_L", "train_M", "train_S"],
                    aishell4_room_ratios={"L": 0.25, "M": 0.45, "S": 0.30},
                    use_official_aishell4_test=True,
                    fsd50k_train_source="dev",
                    fsd50k_test_source="eval",
                    seed=7,
                    precompute_features=True,
                )
            )
            records = validate_manifest(manifest)
            split_sources = {(record.split, record.source) for record in records}
            self.assertIn(("train", "AISHELL4"), split_sources)
            self.assertIn(("test", "AISHELL4"), split_sources)
            self.assertTrue(any(record.split in {"train", "val"} and record.source == "FSD50K-hard-negative" for record in records))
            self.assertTrue(any(record.split == "test" and record.source == "FSD50K-hard-negative" for record in records))
            metadata = json.loads((manifest.parent / "metadata.json").read_text(encoding="utf-8"))
            self.assertIn("room_clip_counts", metadata)
            self.assertIn("source_split_clip_counts", metadata)
            self.assertGreater(metadata["fsd50k_hard_negative_frame_ratio"], 0.0)

    def test_source_metrics_for_realneg_vad(self) -> None:
        counts: dict[str, dict[str, float]] = {}
        aishell = ManifestRecord("a", "a.wav", "a.npy", "test", 16000, 1.0, 10.0, "AISHELL4")
        fsd = ManifestRecord("f", "f.wav", "f.npy", "test", 16000, 1.0, 10.0, "FSD50K-hard-negative")
        _accumulate_source_metrics(counts, aishell, np.array([1, 1, 0, 0], dtype=np.uint8), np.array([0.9, 0.2, 0.8, 0.1]), 0.5)
        _accumulate_source_metrics(counts, fsd, np.zeros(4, dtype=np.uint8), np.array([0.1, 0.6, 0.2, 0.7]), 0.5)
        metrics = _finalize_source_metrics(counts)
        self.assertAlmostEqual(metrics["aishell4_speech_recall"], 0.5)
        self.assertAlmostEqual(metrics["fsd50k_hard_negative_false_positive_rate"], 0.5)
        self.assertAlmostEqual(metrics["non_speech_false_positive_rate"], 3.0 / 6.0)

    def test_latency_stats_and_selection_helpers(self) -> None:
        stats = _model_stats_with_context(
            {"params": 6981.0, "macs_per_frame": 6812.0, "macs_total": 6812.0, "profile_frames": 1.0},
            {
                "features": {"frame_ms": 20.0, "hop_ms": 20.0},
                "training": {"augment": {"segment_frames": 13}},
            },
        )
        self.assertEqual(stats["context_frames"], 13.0)
        self.assertEqual(stats["context_audio_ms"], 260.0)
        self.assertEqual(stats["streaming_macs_per_second"], 340600.0)
        self.assertEqual(stats["window_rerun_macs_per_second"], 4427800.0)
        args = type("Args", (), {"sample_rate": 16000, "n_mels": 64, "seed": 7, "epochs": 40, "batch_size": 16})()
        variant = {"frame_ms": 20.0, "hop_ms": 20.0, "context_frames": 13}
        train_cfg = _latency_train_config(args, variant, Path("data/x/manifest.jsonl"), Path("runs/x"))
        eval_cfg = _latency_eval_config(args, variant, Path("data/x/manifest.jsonl"), Path("runs/x"))
        self.assertEqual(train_cfg["training"]["augment"]["segment_frames"], 13)
        self.assertAlmostEqual(eval_cfg["eval"]["sliding_window_sec"], 0.26)
        best = _select_latency_best(
            [
                {"f1": 0.89, "streaming_macs_per_second": 1.0, "context_frames": 1, "auroc_all": 1.0, "tpr_at_fpr_0_315": 1.0},
                {"f1": 0.91, "streaming_macs_per_second": 2.0, "context_frames": 2, "auroc_all": 0.9, "tpr_at_fpr_0_315": 0.9},
                {"f1": 0.92, "streaming_macs_per_second": 2.0, "context_frames": 1, "auroc_all": 0.8, "tpr_at_fpr_0_315": 0.8},
            ],
            accuracy_floor=0.90,
        )
        self.assertEqual(best["context_frames"], 1)


if __name__ == "__main__":
    unittest.main()
