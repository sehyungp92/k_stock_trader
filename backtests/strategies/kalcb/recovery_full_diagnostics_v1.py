"""Versioned report-context repair using the existing KALCB diagnostic engine.

The registered full_diagnostics.py and canonical reports remain unchanged.
Execution evidence is consumed and reauthenticated by the same qualified-artifact
loader. Only research status, period and training descriptions differ.
"""
from pathlib import Path
import json
from .full_diagnostics import (
    analyze_kalcb_result, render_kalcb_diagnostics_report,
    _atomic_write_text, _atomic_write_json, _atomic_write_jsonl,
    _artifact_paths, _index_payload,
)


def recovery_report_context(profile):
    context = profile["recovery_report_context"]
    periods = {"is": {"start": "2025-08-07", "end": "2026-03-31", "sessions": 156},
               "oos": {"start": "2026-04-01", "end": "2026-07-10", "sessions": 69}}
    period, recipe = context["period"], context["recipe"]
    if period not in periods or profile["date_range"] != periods[period]:
        raise ValueError("Recovery report period mismatch")
    if recipe not in ("expanding", "expanding_atr5", "decay60", "decay60_atr5"):
        raise ValueError("Unknown registered recovery recipe")
    if context.get("independent_validation") is not False or context.get("june_july_used_for_research") is not True:
        raise ValueError("Reused research must not claim untouched validation")
    required = {"minimum_training_sessions": 60, "refit_sessions": 20,
                "embargo_sessions": 1, "continuous_original_fitting_clock": True,
                "mature_labels_only": True, "earlier_oof_calibration_only": True}
    if context["training_contract"] != required:
        raise ValueError("Unsupported recovery training clock")
    memory = "Expanding history" if recipe.startswith("expanding") else "Full historical prefixes with exponential observation weights (60-session half-life)"
    return {**context,
        "training": memory + "; minimum 60 completed training sessions; mature labels only; one-session embargo; continuous original 20-session refit clock; calibration uses earlier out-of-fold observations.",
        "validation_scope": "Reused development study, including April-July 2026. OOS denotes the original chronological split only; no independent validation claim. IS and OOS start separate KRW100m accounts while inheriting causal market/model/calibration history.",
        "canonical_replaced": False, "production_enabled": False,
        "qualification_scope": "Source/execution/accounting qualification; no automatic finding of robust positive expectancy or promotion"}


def write_kalcb_recovery_full_diagnostics(bundle, config, mutations, output_dir, *, round_num=1, round_name=""):
    """Render the existing full report from bound native exports, observationally."""
    from .qualified_artifacts import diagnostic_result, load_qualified_profile, read, digest, close
    import shutil
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = bundle["profile"]
    context = recovery_report_context(profile)
    result = diagnostic_result(bundle)
    analysis = analyze_kalcb_result(result, config=config, mutations=mutations)
    analysis.update(round=round_num, round_name=round_name, promotion_status="qualified_research_comparison_not_canonical")
    analysis["gate_result"] = {"status": "source_qualified_research_comparison", "checks": {
        "four_source_audits": True, "frozen_inputs": True, "cash_cost_trade_mtm_reconciliation": True},
        "production_ready": False}
    analysis["live_parity_audit"].update(
        status="shared_core_contract_only_pending_paper_execution",
        observation_watch_count=103, observation_window_minutes=45,
        active_symbol_count_basis="admitted candidate snapshot count; not verified live subscription occupancy",
        paper_promotion_blockers=["Full 103-name prefix and model update delivery latency has not been qualified live.",
            "KIS paper order/fill/restart parity remains required.", "Independent forward validation remains required.",
            "A funded fixed-capital/no-compounding native replay remains required before production candidacy."])
    analysis["diagnostics_summary"]["live_parity_audit"] = analysis["live_parity_audit"]
    scenario_metrics = {s: c["result"] for s, c in bundle["cases"].items()}
    supplemental = {"acceptance_manifest": bundle["manifest"], "acceptance_manifest_hash": bundle["manifest_hash"],
        "scenarios": {s: {k: r[k] for k in ("return_pct", "drawdown_pct", "trades", "active_days", "audit_status")} for s, r in scenario_metrics.items()},
        "scenario_checks": {s: c["validation"] for s, c in bundle["cases"].items()},
        "date_range": profile["date_range"], "initial_equity_krw": profile["initial_equity_krw"],
        "training": context["training"],
        "validation_scope": context["validation_scope"],
        "recovery_context": context,
        "policy": profile["policy"], "audit_manifest": profile["audit_manifest"],
        "forecast_reconstruction": profile["forecast_reconstruction"],
        "drawdown_targets_are_soft": True, "original_trials": profile["all_trials"],
        "new_search_trials": 0, "production_enabled": False}
    # Constant-notional attribution isolates compounding mechanically. It is
    # deliberately not reported as a funded portfolio return or drawdown.
    supplemental["constant_notional_attribution"] = {
        "notional_krw_per_trade": 10_000_000,
        "net_pnl_krw": sum(t.net_pnl * 10_000_000 / (t.entry_price * t.qty) for t in result.trades),
        "scope": "Descriptive rescaling of observed trades; excludes funding/capacity feedback; not a fixed-capital portfolio backtest"}
    close(analysis["final_metrics"]["net_return_pct"] * 100, scenario_metrics["baseline"]["return_pct"], "Diagnostic return drift")
    close(analysis["final_metrics"]["max_drawdown_pct"] * 100, scenario_metrics["baseline"]["drawdown_pct"], "Diagnostic drawdown drift")
    close(analysis["final_metrics"]["profit_factor"], bundle["cases"]["baseline"]["validation"]["net_profit_factor"], "Diagnostic PF drift")
    report = render_kalcb_diagnostics_report(analysis)
    report = report.replace("Full-universe alpha", "Non-selected outcome evidence").replace("Intraday Alpha Curve", "Intraday realized PnL attribution")
    report += "\n\nQUALIFIED NATIVE REPLAY / EXISTING SAFEGUARDS\n" + json.dumps(supplemental, indent=2, default=str) + "\n"
    report += ("\nSortino: exchange-session close-to-close returns, 252 sessions/year, zero downside target.\n"
               "Calmar: annualized session return divided by bar-level MTM drawdown.\n"
               "Shadow comparisons are unavailable without matched outcome-policy evidence. MFE alternatives are hindsight path descriptions.\n"
               "Model capacity and next-bar execution are historical research assumptions; paper/live executability is not certified.\n")
    analysis["diagnostics_summary"]["qualified_native_replay"] = supplemental
    _atomic_write_text(output_dir / "round_final_diagnostics.txt", report)
    _atomic_write_json(analysis["diagnostics_summary"], output_dir / "diagnostics_summary.json")
    _atomic_write_json(analysis["candidate_frontier"], output_dir / "candidate_frontier.json")
    _atomic_write_json(analysis["live_parity_audit"], output_dir / "live_parity_audit.json")
    _atomic_write_json({k: v for k, v in analysis.items() if k not in ("trades", "decisions")}, output_dir / "round_final_full_diagnostics.json")
    _atomic_write_json(supplemental, output_dir / "qualified_replay_diagnostics.json")
    _atomic_write_jsonl(output_dir / "trade_events.jsonl", analysis["trade_events"])
    _atomic_write_jsonl(output_dir / "decision_events.jsonl", [d.to_json_dict() for d in result.decisions])
    _atomic_write_json(profile, output_dir / "selected_research_profile.json")
    shutil.copy2(profile["configuration"], output_dir / "strategy_config.json")
    shutil.copy2(bundle["manifest"], output_dir / "acceptance_manifest.json")
    source_paths = {"qualified_profile": bundle["manifest"], "native_audit": profile["audit_manifest"]}
    artifact_paths = _artifact_paths(output_dir)
    for name in ("round_final_full_diagnostics.json", "qualified_replay_diagnostics.json", "decision_events.jsonl",
                 "selected_research_profile.json", "strategy_config.json", "acceptance_manifest.json", "diagnostic_integrity.json"):
        artifact_paths[name] = str(output_dir / name)
    _atomic_write_json(_index_payload(analysis, artifact_paths, source_paths), output_dir / "full_diagnostics_index.json")
    # Validation after rendering proves the report did not alter native evidence.
    load_qualified_profile(bundle["manifest"])
    _atomic_write_json({"passed": True, "native_evidence_unchanged": True,
        "report_sha256": digest(output_dir / "round_final_diagnostics.txt"),
        "metric_units": "Fractions for returns/drawdowns in canonical metrics; scenario tables explicitly use percent"}, output_dir / "diagnostic_integrity.json")
    return analysis["diagnostics_summary"]


from .qualified_round import KALCBQualifiedRoundPlugin


class KALCBRecoveryQualifiedRoundPlugin(KALCBQualifiedRoundPlugin):
    """Same shared locked-profile lifecycle, with accurate research context."""
    def __init__(self, acceptance_manifest):
        super().__init__(acceptance_manifest)
        recovery_report_context(self.profile)
        self.config["promotion_status"] = "qualified_research_comparison_not_canonical"
        self.execution_context["promotion_status"] = self.config["promotion_status"]

    def write_full_diagnostics(self, state, output_dir, **kwargs):
        return write_kalcb_recovery_full_diagnostics(self.bundle, self.config, state.cumulative_mutations, Path(output_dir), **kwargs)
