from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[2] / "tools" / "run_v3_generation_handoff.py"
    spec = importlib.util.spec_from_file_location("run_v3_generation_handoff", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_renderer_completion_requires_exact_count_and_no_live_pid(tmp_path, monkeypatch):
    module = _module()
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text("".join(json.dumps({"row": index}) + "\n" for index in range(2)), encoding="utf-8")
    terminal_log = tmp_path / "rank-0.log"
    terminal_log.write_text('{"completed": true}\n', encoding="utf-8")

    monkeypatch.setattr(module, "renderer_is_live", lambda _pid: True)

    assert not module.renderer_complete((metadata,), expected_records=2, pids=(123,), log_paths=(terminal_log,))


def test_false_wake_requests_are_normalized_from_source_rate_to_16k(tmp_path):
    module = _module()
    import soundfile as sf

    source = tmp_path / "false_wake.wav"
    sf.write(source, [0.0] * 88_200, 44_100, subtype="PCM_16")
    manifest = tmp_path / "requests.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "example_id": "false-wake-1",
                "foreground_path": str(source),
                "foreground_start_sample": 22_050,
                "recipe": {"label": "negative", "source_kind": "false_wake", "split": "train"},
                "source_role": "false_wake_hard_negative",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    normalized, count = module.normalize_false_wake_requests(manifest, tmp_path / "normalized.jsonl")

    assert count == 1
    assert json.loads(normalized.read_text(encoding="utf-8"))["foreground_start_sample"] == 8_000


def test_training_command_enables_replacement_for_both_small_negative_roles(tmp_path):
    module = _module()
    config = module.HandoffConfig(
        source_root=tmp_path / "source",
        run_root=tmp_path / "run",
        base_role_root=tmp_path / "base",
        base_expected_records=1,
        false_wake_requests=tmp_path / "false.jsonl",
        false_expected_records=1,
        validation_manifest=tmp_path / "validation.json",
        test_manifest=tmp_path / "test.json",
        train_manifest=tmp_path / "train.json",
        python="python",
    )
    manifests = {
        name: str(tmp_path / f"{name}.json")
        for name in (
            "base_positive",
            "raw_positive",
            "base_negative",
            "raw_negative",
            "false_wake_hard_negative",
            "captured_environment_negative",
            "tau_environment_negative",
        )
    }

    command, _ = module._training_command(config, manifests)

    replacement_roles = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--mixture-v3-allow-replacement-role"
    ]
    assert replacement_roles == ["false_wake_hard_negative", "captured_environment_negative"]
