"""Versioned report-only adapter for multi-stage recovery feature lineage.

Native policy manifests, audited outputs and the canonical adapter are immutable.
The legacy prepared_sha256 report slot holds an explicitly bound aggregate feature
lineage digest, never an invented historical single-preparation identifier.
"""
from dataclasses import asdict
from datetime import time
from pathlib import Path
import json

from backtests.strategies.common.plugin_base import attach_official_metric_contract
from strategy_kalcb.config import KALCBConfig
from .qualified_artifacts import load_qualified_profile, diagnostic_result, read, digest, require
from .recovery_full_diagnostics_v1 import KALCBRecoveryQualifiedRoundPlugin as OriginalRecoveryPlugin, recovery_report_context


def adapt_feature_lineage(bundle):
    """Authenticate compatibility metadata without mutating native policy data."""
    profile = bundle["profile"]
    reference = profile["diagnostic_feature_lineage"]
    path = Path(reference["path"]).resolve()
    capsule = read(bundle["manifest"])
    require(capsule["files"].get(str(path)) == reference["sha256"] == digest(path), "Unbound or changed diagnostic feature lineage")
    lineage = read(path)
    require(lineage.get("schema") == "kalcb-recovery-feature-preparation-lineage-v1", "Unsupported feature lineage schema")
    rows = lineage.get("preparations", [])
    require(len(rows) == 4 and len({r["original_path"] for r in rows}) == 4, "Incomplete feature preparation lineage")
    for row in rows:
        archived = str(Path(row["path"]).resolve())
        require(capsule["files"].get(archived) == row["sha256"] == digest(archived), "Unbound or changed prepared feature evidence")
    forecast = lineage["forecast_reconstruction"]
    require(str(Path(forecast["path"]).resolve()) == str(Path(profile["forecast_reconstruction"]).resolve()), "Wrong feature forecast lineage")
    require(capsule["files"].get(str(Path(forecast["path"]).resolve())) == forecast["sha256"] == digest(forecast["path"]), "Changed feature forecast qualification")
    require(read(forecast["path"]).get("passed") is True, "Unqualified feature reconstruction")
    require("prepared_sha256" not in bundle["policy"], "Recovery adapter cannot replace an existing preparation identity")
    return {**bundle, "policy": {**bundle["policy"], "prepared_sha256": reference["sha256"],
        "diagnostic_prepared_hash_basis": "Aggregate bound IS/OOS opening/checkpoint feature-preparation lineage; compatibility metadata only"}}


class KALCBRecoveryQualifiedRoundPlugin(OriginalRecoveryPlugin):
    def __init__(self, acceptance_manifest):
        self.acceptance_manifest = Path(acceptance_manifest).resolve()
        self.bundle = adapt_feature_lineage(load_qualified_profile(self.acceptance_manifest))
        self.profile = self.bundle["profile"]
        recovery_report_context(self.profile)
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
            "promotion_status": "qualified_research_comparison_not_canonical",
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
            "feature_manifest_hash_basis": self.bundle["policy"]["diagnostic_prepared_hash_basis"],
            "feature_preparation_lineage": self.profile["diagnostic_feature_lineage"],
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

    def build_end_of_round_artifacts(self, state):
        artifacts = super().build_end_of_round_artifacts(state)
        self.bundle = adapt_feature_lineage(self.bundle)
        return artifacts
