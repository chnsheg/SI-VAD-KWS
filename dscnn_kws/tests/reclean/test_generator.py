from dataclasses import replace
import subprocess
import sys
from pathlib import Path

import soundfile as sf
import torch

from dscnn_kws.data.reclean.generator import GenerationRequest, pending_examples, render_request, select_shard
from dscnn_kws.data.reclean.recipes import build_recipe


def test_four_shards_have_no_overlap():
    ids = [f"example-{index}" for index in range(101)]
    shards = [set(select_shard(ids, rank, 4)) for rank in range(4)]

    assert sum(map(len, shards)) == len(ids)
    assert set.union(*shards) == set(ids)
    assert all(left.isdisjoint(right) for index, left in enumerate(shards) for right in shards[index + 1 :])


def test_completed_examples_are_not_rewritten(tmp_path):
    completed = tmp_path / "completed.jsonl"
    completed.write_text('{"example_id":"x","output_sha256":"h"}\n', encoding="utf-8")

    assert pending_examples(["x", "y"], completed) == ["y"]


def test_render_request_writes_one_second_pcm16_audio_and_metadata(tmp_path):
    foreground = tmp_path / "foreground.wav"
    noise = tmp_path / "noise.wav"
    sf.write(foreground, torch.full((16000,), 0.1).numpy(), 16000)
    sf.write(noise, torch.full((16000,), 0.01).numpy(), 16000)
    recipe = replace(build_recipe(42, "train", "foreground", "hash", 4), jitter_ms=0)
    request = GenerationRequest(
        example_id="example-0",
        recipe=recipe,
        foreground_path=foreground,
        active_span=(0, 16000),
        noise_path=noise,
    )

    metadata = render_request(request, tmp_path / "out.wav", device=torch.device("cpu"))
    info = sf.info(tmp_path / "out.wav")

    assert (info.samplerate, info.channels, info.subtype, info.frames) == (16000, 1, "PCM_16", 16000)
    assert metadata["example_id"] == "example-0"
    assert metadata["output_sha256"]


def test_generator_cli_exposes_pilot_arguments():
    wrapper = Path(__file__).resolve().parents[3] / "run_reclean_augmented_dataset.py"

    result = subprocess.run([sys.executable, str(wrapper), "pilot", "--help"], text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr
    assert "--recipe-manifest" in result.stdout
