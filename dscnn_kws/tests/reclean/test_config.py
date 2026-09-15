from dscnn_kws.data.reclean.types import load_v1_config


def test_v1_config_locks_a_path_and_quotas():
    cfg = load_v1_config()

    assert (cfg.sample_rate, cfg.sample_length, cfg.final_rms_alignment) == (16000, 16000, False)
    assert cfg.positive_jitter_max_ms == 200
    assert cfg.positive_jitter_mode == "online"
    assert cfg.speed_factors == (0.9, 1.0, 1.25)
    assert cfg.active_rms_dbfs == (-34.0, -30.0, -26.0, -22.0, -18.0)
    assert cfg.negative_quotas == {"speech": 0.65, "false_wake": 0.10, "pure_noise": 0.25}
    assert cfg.snr_probabilities == {
        -15: 0.30,
        -10: 0.25,
        -5: 0.15,
        0: 0.10,
        5: 0.08,
        10: 0.05,
        15: 0.04,
        20: 0.03,
    }
