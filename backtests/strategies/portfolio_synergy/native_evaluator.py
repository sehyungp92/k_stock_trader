"""Historical native policy experiments with the same mandatory audit gate."""
from dataclasses import asdict

from backtests.auto.shared.cache_keys import stable_signature, source_generation_hash
from backtests.auto.shared.types import ScoredCandidate
from backtests.core.input_manifest import active_manifest
from backtests.engine.contracts import execution_valid
from backtests.engine.integrity import audit_broker
from backtests.engine.capacity import capacity_from_config
from .native_policy import native_policy_config
from .source_replay import run_native_risk_shared_replay, summarize_replay


class NativePortfolioPolicyEvaluator:
    def __init__(self, inputs, *, initial_equity, configuration):
        self.inputs = inputs
        self.initial_equity = float(initial_equity)
        self.configuration = dict(configuration)

    def evaluate_mutations(self, mutations):
        policy = native_policy_config(mutations)
        generation = active_manifest()
        result = run_native_risk_shared_replay(self.inputs, initial_equity=self.initial_equity, portfolio_policy=policy,
            capacity_policy=capacity_from_config(self.configuration))
        request = stable_signature({"configuration": self.configuration, "policy": asdict(policy),
            "strategy_configurations": [asdict(self.inputs.kalcb_config), asdict(self.inputs.olr_config)],
            "candidate_hash": getattr(result.broker, "qualification_snapshot_hash", None),
            "source_events": result.broker.source_events.hexdigest(),
            "initial_equity": self.initial_equity, "source": source_generation_hash(),
            "generation": generation.semantic_identity() if generation else None})
        metrics = asdict(summarize_replay(result, initial_equity=self.initial_equity))
        metrics.update(audit_broker(result.broker, identity={"request": request,
            "mutations": stable_signature(dict(mutations)), "input_manifest": generation.receipt() if generation else None,
            "configuration": self.configuration,
            "portfolio_policy": asdict(policy)}))
        metrics.update(evaluation_request=request, evaluation_mutations=stable_signature(dict(mutations)),
            capability_level=self.inputs.lineage.get("capability_level", "unverified"), promotion_requires_audit_pass=True,
            official_mtm_net_return_pct=metrics["net_return_pct"],
            portfolio_policy=asdict(policy),
            admission_count=sum(r["event"] == "portfolio_admission" for r in result.broker.journal),
            changed_admission_count=sum(r["event"] == "portfolio_admission" and r["admitted_qty"] != r["offered_qty"] for r in result.broker.journal))
        metrics["immutable_score"] = metrics["official_mtm_net_return_pct"] if execution_valid(metrics, expected_request=request) else None
        return metrics

    def __call__(self, candidates, current_mutations):
        results = []
        for candidate in candidates:
            metrics = self.evaluate_mutations({**current_mutations, **candidate.mutations})
            valid = execution_valid(metrics)
            results.append(ScoredCandidate(candidate.name, metrics["immutable_score"] if valid else 0,
                rejected=not valid, reject_reason="" if valid else "execution_audit_unverified_or_failed", metrics=metrics))
        return results
