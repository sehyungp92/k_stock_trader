from __future__ import annotations

from backtests.analysis.artifact_hygiene import validate_official_artifact


def _official_summary() -> dict[str, str]:
    return {
        "promotion_status": "official",
        "source_data_fingerprint": "src",
        "score_spec_hash": "score",
        "config_hash": "cfg",
        "strategy_code_hash": "code",
        "live_parity_fill_timing": "next_bar_after_completed_signal",
        "risk_basis": "mark_to_market",
    }


def test_official_artifact_requires_consistent_optimized_config_envelope():
    summary = _official_summary()
    optimized_config = {
        "mutations": {"x": 1},
        "source_data_fingerprint": "src",
        "score_spec_hash": "score",
        "config_hash": "cfg",
        "strategy_code_hash": "code",
        "live_parity_fill_timing": "next_bar_after_completed_signal",
    }

    assert validate_official_artifact(summary, optimized_config).passed


def test_official_artifact_rejects_stale_hashes_and_realized_only_risk():
    summary = _official_summary()
    summary["risk_basis"] = "realized_only"
    optimized_config = {
        "mutations": {},
        "source_data_fingerprint": "stale",
        "score_spec_hash": "score",
        "config_hash": "cfg",
        "strategy_code_hash": "code",
        "live_parity_fill_timing": "same_bar",
    }

    result = validate_official_artifact(summary, optimized_config)

    assert not result.passed
    assert "source_data_fingerprint_mismatch" in result.failures
    assert "fill_timing_mismatch" in result.failures
    assert "risk_basis_not_mtm" in result.failures
