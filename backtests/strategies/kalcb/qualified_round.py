"""Locked research acceptance through the shared PhaseRunner/RoundManager.

No search loop or alternative execution model lives here. Historical selection
receipts stay in their original study; this phase accepts exactly that profile.
"""
from dataclasses import asdict
from datetime import time
from pathlib import Path
import json

from backtests.auto.shared.plugin import PhaseSpec, PhaseAnalysisPolicy
from backtests.auto.shared.types import GateCriterion, ScoredCandidate, EndOfRoundArtifacts
from backtests.strategies.common.plugin_base import attach_official_metric_contract
from strategy_kalcb.config import KALCBConfig
from .qualified_artifacts import load_qualified_profile, diagnostic_result, read, digest, require


class KALCBQualifiedRoundPlugin:
    name = "kalcb"
    num_phases = 1
    ultimate_targets = {}
    capability_level = "real_replay"
    requires_full_diagnostics = True

    def __init__(self, acceptance_manifest):
        self.acceptance_manifest = Path(acceptance_manifest).resolve()
        self.bundle = load_qualified_profile(self.acceptance_manifest)
        self.profile = self.bundle["profile"]
        capsule = read(self.acceptance_manifest)
        checks = capsule.get("acceptance_checks") or {}
        require(set(checks) == {"source", "lessons", "tests"}, "Missing final acceptance checks")
        for kind, filename in checks.items():
            require(str(Path(filename).resolve()) in capsule["files"], "Unbound acceptance check")
            require(read(filename).get("passed") is True, "Failed acceptance check: " + kind)
        require(read(checks["source"])["source_generation_hash"] == self.profile["execution_source"]["source_generation_hash"], "Wrong accepted source")
        self.source_fingerprint = self.profile["execution_source"]["source_generation_hash"]
        effective = KALCBConfig.from_mapping(read(self.profile["configuration"]))
        self.initial_mutations = json.loads(json.dumps(asdict(effective), default=lambda x: x.isoformat() if isinstance(x, time) else x.value))
        self.config = {"initial_equity": self.profile["initial_equity_krw"],
            "date_range": self.profile["date_range"],
            "promotion_status": "accepted_optimized_baseline_research_only",
            "artifact_promotion_policy": "research_only_until_independent_validation_and_paper_execution",
            "live_parity_fill_timing": "next_5m_open", "auction_mode": "non_auction_continuous",
            "capability_level": "real_replay", "risk_basis": "native_bar_level_mark_to_market",
            "selected_candidate": self.profile["policy"]["name"],
            "acceptance_manifest": str(self.acceptance_manifest),
            "acceptance_manifest_sha256": digest(self.acceptance_manifest),
            "required_candidate_policy": self.profile["policy"],
            "initial_mutations": self.initial_mutations}
        self.execution_context = {**self.config, "source_fingerprint": self.source_fingerprint,
            "feature_manifest_hash": self.bundle["policy"]["prepared_sha256"],
            "candidate_snapshot_hash": self.profile["policy"]["manifest_sha256"],
            "replay_mode": "audited_native_timed_candidates", "train_start": self.profile["date_range"]["start"],
            "train_end": self.profile["date_range"]["end"], "expected_universe_size": 103,
            "cost_policy": {s: self.bundle["cases"][s]["audit"]["audit_receipt"]["qualification"]["capacity"]["policy"] for s in self.bundle["cases"]}}
        self.result = diagnostic_result(self.bundle)
        from .full_diagnostics import _final_metrics
        final_metrics = _final_metrics(self.result.metrics, self.result.trades, self.config,
            self.result.replay_result.equity_curve, timestamps=self.result.replay_result.timestamps)
        self.metrics = attach_official_metric_contract(final_metrics, requires_audit_pass=True, audit_pass=True,
            execution_contract={**self.execution_context, "metric_units": "return and drawdown fractions; currency KRW",
                "required_candidate_policy": self.profile["policy"], "production_enabled": False,
                "acceptance_manifest_sha256": digest(self.acceptance_manifest)})

    def canonicalize_mutations(self, mutations):
        merged = {**self.initial_mutations, **mutations}
        require(merged == self.initial_mutations, "Locked qualification cannot tune the selected profile")
        return merged

    def get_phase_spec(self, phase, state):
        return PhaseSpec("Integrity acceptance of the locked optimized profile", [],
            lambda m: [GateCriterion("all_scenarios_qualified", 1, int(m["audit_pass"]), bool(m["audit_pass"]))],
            None, {}, PhaseAnalysisPolicy(focus_metrics=["official_mtm_net_return_pct", "max_drawdown_pct"]),
            max_rounds=1, phase_metric_basis="audited_source_replay",
            primary_promotion_metric="official_mtm_net_return_pct",
            official_metric_keys=("official_mtm_net_return_pct", "max_drawdown_pct"), promotion_requires_audit_pass=True)

    def create_evaluate_batch(self, phase, cumulative_mutations, **kwargs):
        def evaluate(candidates, current):
            self.canonicalize_mutations(current)
            return [ScoredCandidate(c.name, self.compute_final_metrics({**current, **c.mutations})["official_mtm_net_return_pct"], metrics=self.metrics) for c in candidates]
        return evaluate

    def compute_final_metrics(self, mutations):
        self.canonicalize_mutations(mutations)
        return dict(self.metrics)

    def run_phase_diagnostics(self, phase, state, metrics, greedy_result):
        return json.dumps({"locked_acceptance": True, "new_search_trials": 0, "previous_study_trials": self.profile["tested_variant_count"],
            "checks": {s: c["validation"] for s, c in self.bundle["cases"].items()},
            "scope": "Reused development history; full source audit and record reconciliation; production remains gated"}, indent=2)

    run_enhanced_diagnostics = run_phase_diagnostics

    def build_end_of_round_artifacts(self, state):
        self.canonicalize_mutations(state.cumulative_mutations)
        # Recheck bytes at finalization; a warm evaluator cannot promote stale exports.
        self.bundle = load_qualified_profile(self.acceptance_manifest)
        report = self.run_phase_diagnostics(1, state, self.metrics, None)
        return EndOfRoundArtifacts(report, {"source_acceptance": report}, "Accepted research round; live and forward qualification remain required")

    def write_full_diagnostics(self, state, output_dir, **kwargs):
        from .full_diagnostics import write_kalcb_qualified_full_diagnostics
        return write_kalcb_qualified_full_diagnostics(self.bundle, self.config, state.cumulative_mutations, Path(output_dir), **kwargs)
