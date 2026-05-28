from __future__ import annotations

from types import SimpleNamespace

from backtests.auto.shared.phase_runner import _apply_phase_metric_contract, _audit_contract_criteria
from backtests.auto.shared.plugin import PhaseAnalysisPolicy, PhaseSpec


def _official_metrics(**overrides):
    metrics = {
        "official_mtm_net_return_pct": 0.012,
        "official_metric_basis": "SimBroker.equity_curve_bar_level_mtm",
        "primary_promotion_metric": "official_mtm_net_return_pct",
        "primary_promotion_basis": "SimBroker.equity_curve_bar_level_mtm",
        "official_replay_pass": True,
        "audit_pass": False,
        "audit_status": "direct_official_replay",
        "same_bar_fill_count": 0.0,
        "forced_replay_close_count": 0.0,
        "rejected_order_count": 0.0,
        "end_open_position_count": 0.0,
        "metric_contract": {
            "primary_promotion_metric": "official_mtm_net_return_pct",
            "primary_promotion_basis": "SimBroker.equity_curve_bar_level_mtm",
            "promotion_requires_audit_pass": False,
            "official_metrics": ["official_mtm_net_return_pct"],
            "execution_contract": {
                "source_fingerprint": "source-a",
                "feature_manifest_hash": "features-a",
                "candidate_snapshot_hash": "candidates-a",
                "capability_level": "real_replay",
            },
        },
    }
    metrics.update(overrides)
    return metrics


def _passed_by_name(metrics):
    return {item.name: item.passed for item in _audit_contract_criteria(metrics)}


def test_direct_official_replay_passes_phase_contract_without_full_audit() -> None:
    passed = _passed_by_name(_official_metrics())

    assert all(passed.values())
    assert "hard_audit_pass" not in passed


def test_strict_promotion_blocks_missing_required_audit() -> None:
    passed = _passed_by_name(_official_metrics(promotion_requires_audit_pass=True))

    assert passed["hard_audit_pass"] is False


def test_strict_promotion_passes_with_official_basis_and_audit() -> None:
    passed = _passed_by_name(
        _official_metrics(
            promotion_requires_audit_pass=True,
            audit_pass=True,
            audit_status="audited_full_bundle_passed",
        )
    )

    assert all(passed.values())


def test_strict_promotion_blocks_synthetic_capability() -> None:
    metrics = _official_metrics(
        promotion_requires_audit_pass=True,
        audit_pass=True,
        audit_status="audited_full_bundle_passed",
    )
    metrics["metric_contract"]["execution_contract"]["capability_level"] = "synthetic"

    passed = _passed_by_name(metrics)

    assert passed["hard_non_synthetic_capability"] is False


def test_contract_blocks_missing_primary_metric() -> None:
    metrics = _official_metrics()
    metrics.pop("official_mtm_net_return_pct")

    passed = _passed_by_name(metrics)

    assert passed["hard_primary_promotion_metric"] is False


def test_contract_blocks_missing_basis() -> None:
    metrics = _official_metrics(official_metric_basis="", primary_promotion_basis="")
    metrics["metric_contract"]["primary_promotion_basis"] = ""

    passed = _passed_by_name(metrics)

    assert passed["hard_official_metric_basis"] is False


def test_contract_blocks_live_parity_hygiene_failures() -> None:
    cases = {
        "hard_same_bar_fill_count": {"same_bar_fill_count": 1.0},
        "hard_forced_replay_close_count": {"forced_replay_close_count": 1.0},
        "hard_rejected_order_count": {"rejected_order_count": 1.0},
        "hard_end_open_position_count": {"end_open_position_count": 1.0},
    }

    for criterion, override in cases.items():
        assert _passed_by_name(_official_metrics(**override))[criterion] is False


def test_contract_blocks_missing_required_hygiene_evidence() -> None:
    metrics = _official_metrics()
    metrics.pop("rejected_order_count")

    passed = _passed_by_name(metrics)

    assert passed["hard_rejected_order_count"] is False


def test_contract_blocks_missing_source_feature_candidate_identity() -> None:
    metrics = _official_metrics()
    metrics["metric_contract"]["execution_contract"] = {
        "source_fingerprint": "source-a",
        "feature_manifest_hash": "",
        "candidate_snapshot_hash": "candidates-a",
    }

    passed = _passed_by_name(metrics)

    assert passed["hard_feature_manifest_hash"] is False


def test_phase_spec_metric_contract_is_applied_before_gate_checks() -> None:
    spec = PhaseSpec(
        "contract",
        [],
        lambda metrics: [],
        {},
        {},
        PhaseAnalysisPolicy(),
        phase_metric_basis="direct_official_replay",
        primary_promotion_metric="official_mtm_net_return_pct",
        official_metric_keys=("official_mtm_net_return_pct",),
        proxy_metric_keys=("portfolio_proxy_net_return_pct",),
        promotion_requires_audit_pass=True,
    )
    plugin = SimpleNamespace(
        name="tiny",
        execution_context={
            "source_fingerprint": "source-a",
            "feature_manifest_hash": "features-a",
            "candidate_snapshot_hash": "candidates-a",
            "capability_level": "real_replay",
        },
        config={},
    )

    metrics = _apply_phase_metric_contract({"official_mtm_net_return_pct": 0.015}, spec, plugin)
    passed = _passed_by_name(metrics)

    assert metrics["primary_promotion_metric"] == "official_mtm_net_return_pct"
    assert metrics["metric_contract"]["phase_metric_basis"] == "direct_official_replay"
    assert metrics["metric_contract"]["official_metrics"] == ["official_mtm_net_return_pct"]
    assert metrics["metric_contract"]["proxy_metrics"] == ["portfolio_proxy_net_return_pct"]
    assert passed["hard_audit_pass"] is False
