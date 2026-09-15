import json

import numpy as np
import soundfile as sf

from dscnn_kws.data.reclean.prepare import prepare_sources


def test_prepare_sources_canonicalizes_positive_and_records_active_boundary(tmp_path):
    source = tmp_path / "positive.wav"
    sf.write(source, np.concatenate([np.zeros(8000), np.ones(8000) * 0.1]), 16000)
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "role": "mobvoi_speech",
                        "path": str(source),
                        "sha256": "source-hash",
                        "split": "train",
                        "source_label": "positive",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = prepare_sources(inventory, tmp_path / "prepared")

    assert len(rows) == 1
    assert rows[0]["source_label"] == "positive"
    assert rows[0]["active_end"] > rows[0]["active_start"]
    info = sf.info(rows[0]["prepared_path"])
    assert (info.samplerate, info.channels, info.subtype) == (16000, 1, "PCM_16")


def test_prepare_sources_reuses_catalogued_normalized_rir_without_hashing_or_recanonicalizing(tmp_path, monkeypatch):
    rir = tmp_path / "rir-000.wav"
    sf.write(rir, np.zeros(320), 16000, subtype="PCM_16")
    catalog_hash = "b" * 64
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "role": "rir",
                        "path": str(rir),
                        "already_normalized": True,
                        "normalized_sha256": catalog_hash,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def unexpected_reprocessing(*args, **kwargs):
        raise AssertionError("Catalogued normalized RIR must not be reprocessed")

    monkeypatch.setattr("dscnn_kws.data.reclean.prepare.canonicalize_wav", unexpected_reprocessing)
    monkeypatch.setattr("dscnn_kws.data.reclean.prepare.sha256_file", unexpected_reprocessing)

    rows = prepare_sources(inventory, tmp_path / "prepared")

    assert rows == [
        {
            "role": "rir",
            "source_label": None,
            "source_split": None,
            "scene": None,
            "source_path": str(rir.resolve()),
            "source_sha256": catalog_hash,
            "prepared_path": str(rir.resolve()),
            "prepared_sha256": catalog_hash,
            "sample_rate": 16000,
            "frames": 320,
            "source_kind": "rir",
        }
    ]
