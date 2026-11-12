import json
from pathlib import Path

from backtests.auto.shared.phase_runner import _metric_contract_criteria
from backtests.strategies.kalcb.executed_value import digest
from backtests.strategies.portfolio_synergy.current_phase import CurrentPortfolioPlugin


def test_native_phase_reports_real_hygiene_and_honors_registered_incumbent(tmp_path):
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    manifest = tmp_path / "manifest.json"
    save(manifest, {"manifest_hash": "bound-features"})
    study = tmp_path / "study.json"
    save(study, {"code": str(tmp_path), "source_generation_hash": "frozen-source", "manifest": str(manifest),
        "initial_mutations": {"recipe": "active"}, "candidates": {"active": {}, "ablation": {}}})
    for scenario in ("baseline", "combined"):
        output = tmp_path / "cases/active" / scenario
        save(output / "result.json", {"recipe": "active", "scenario": scenario, "return_pct": 10,
            "drawdown_pct": 2, "trades": 4, "active_days": 3, "trades_per_21_sessions": 2,
            "execution_validity_pass": True, "intraday_residual_sessions": 0, "terminal_inventory": {},
            "terminal_pending": 0, "source_generation_hash": "frozen-source"})
        save(output / "execution_audit.json", {"same_bar_fill_count": 0, "forced_replay_close_count": 0,
            "rejected_order_count": 3, "invalid_order_rejection_count": 0,
            "audit_receipt": {"identity": {"qualification_snapshot_hash": "bound-candidates"}}})
        save(output / "request.json", {"study_hash": digest(study)})
        save(output / "completion.json", {"result_sha256": digest(output / "result.json"),
            "audit_sha256": digest(output / "execution_audit.json")})
    plugin = CurrentPortfolioPlugin(study)
    assert plugin.initial_mutations == {"recipe": "active"}
    assert [c.name for c in plugin.get_phase_spec(1, None).candidates] == ["ablation"]
    metrics = plugin.compute_final_metrics({})
    assert metrics["rejected_order_count"] == 6
    assert metrics["invalid_order_rejection_count"] == 0
    assert metrics["feature_manifest_hash"] == "bound-features"
    assert metrics["candidate_snapshot_hash"] == "bound-candidates"
    metrics.update(primary_promotion_metric="combined_return", official_metric_keys=["baseline_return", "combined_return"],
        promotion_requires_audit_pass=False)
    assert all(criterion.passed for criterion in _metric_contract_criteria(metrics))
    metrics["invalid_order_rejection_count"] = 1
    assert not all(criterion.passed for criterion in _metric_contract_criteria(metrics))
