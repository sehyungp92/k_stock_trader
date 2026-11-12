"""Finite current-strategy allocation study using the shared PhaseRunner."""
from pathlib import Path
import json
import os
import subprocess
import sys

from backtests.auto.shared.plugin import PhaseSpec, PhaseAnalysisPolicy
from backtests.auto.shared.types import Experiment, ScoredCandidate, GateCriterion, EndOfRoundArtifacts
from backtests.strategies.kalcb.executed_value import digest, write_json


def score(metrics):
    # Percent points; 8/10 are soft preferences. Frequency counts closed entry
    # lifecycles, never partial fills. Every tradeoff is also kept on a frontier.
    return (100 + .5*metrics["baseline_return"] + metrics["combined_return"]
        - 1.5*(metrics["baseline_dd"] + metrics["combined_dd"])
        + .1*min(metrics["baseline_frequency"], metrics["combined_frequency"])
        + .05*min(metrics["baseline_active"], metrics["combined_active"]))


class CurrentPortfolioPlugin:
    name = "portfolio_synergy_current"
    num_phases = 1
    ultimate_targets = {"baseline_dd": 8., "combined_dd": 10.}
    initial_mutations = {"recipe": "cash_control"}

    def __init__(self, study):
        self.path = Path(study)
        self.study = json.loads(self.path.read_text())
        self.initial_mutations = dict(self.study.get("initial_mutations", {"recipe": "cash_control"}))
        self.root = self.path.parent
        self.code = Path(self.study["code"])
        self.cache = {}
        self.config = {"artifact_promotion_policy": "research_only_pending_full_audit_and_independent_validation"}
        self.execution_context = {"source_fingerprint": self.study["source_generation_hash"],
            "shared_decision_core": "live_shared_core", "live_parity_fill_timing": "next_5m_open",
            "primary_promotion_basis": "native_sim_broker_marked_equity", "capability_level": "real_replay",
            **self.config}

    def canonicalize_mutations(self, mutations):
        if set(mutations) - {"recipe"}:
            raise ValueError("Unregistered portfolio mutation")
        return {**self.initial_mutations, **mutations}

    def get_phase_spec(self, phase, state):
        return PhaseSpec(focus="Current sleeves: funding, concentration and causal priority",
            candidates=[Experiment(name, {"recipe": name}) for name in self.study["candidates"] if name != self.initial_mutations["recipe"]],
            gate_criteria_fn=lambda m: [GateCriterion("physical_execution", 1, float(m["physical_valid"]), m["physical_valid"])],
            scoring_weights=None, hard_rejects={}, analysis_policy=PhaseAnalysisPolicy(
                focus_metrics=["baseline_return", "combined_return", "baseline_frequency", "combined_dd"], min_effective_score_delta_pct=0),
            max_rounds=1, phase_metric_basis="native_research_screening", primary_promotion_metric="combined_return",
            official_metric_keys=("baseline_return", "combined_return"), promotion_requires_audit_pass=False)

    def evaluate(self, mutations):
        name = self.canonicalize_mutations(mutations)["recipe"]
        if name not in self.study["candidates"]:
            raise ValueError("Unregistered recipe")
        if name in self.cache:
            return self.cache[name]
        folder = self.root / "cases" / name
        folder.mkdir(parents=True, exist_ok=True)
        results, audits = {}, {}
        for scenario in ("baseline", "combined"):
            output = folder / scenario
            if not (output / "completion.json").exists():
                if output.exists():
                    raise FileExistsError("Preserving incomplete replay: " + str(output))
                command = [sys.executable, "-B", "-u", str(self.code / "scripts/portfolio_synergy_study.py"), "replay",
                    "--study", str(self.path), "--recipe", name, "--scenario", scenario, "--output", str(output)]
                print(json.dumps({"replay_started": name, "scenario": scenario}), flush=True)
                with (folder / (scenario + ".log")).open("x") as stream:
                    process = subprocess.run(command, cwd=self.code, stdout=stream, stderr=subprocess.STDOUT,
                        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})
                write_json(folder / (scenario + ".receipt.json"), {"command": command, "returncode": process.returncode})
                if process.returncode:
                    raise RuntimeError("Replay failed: " + str(folder / (scenario + ".log")))
            result = json.loads((output / "result.json").read_text())
            completion = json.loads((output / "completion.json").read_text())
            request = json.loads((output / "request.json").read_text())
            if (completion["result_sha256"] != digest(output / "result.json")
                    or completion["audit_sha256"] != digest(output / "execution_audit.json")
                    or request["study_hash"] != digest(self.path)
                    or result["source_generation_hash"] != self.study["source_generation_hash"]
                    or result["recipe"] != name or result["scenario"] != scenario):
                raise ValueError("Replay identity mismatch")
            results[scenario] = result
            audits[scenario] = json.loads((output / "execution_audit.json").read_text())
        valid = all(r["execution_validity_pass"] and not r["intraday_residual_sessions"]
            and not r["terminal_inventory"] and not r["terminal_pending"] for r in results.values())
        metrics = {"physical_valid": valid, "research_only": True, "audit_status": "unverified",
            "official_metric_basis": "SimBroker.equity_curve_bar_level_mtm", "primary_promotion_basis": "SimBroker.equity_curve_bar_level_mtm",
            "source_fingerprint": self.study["source_generation_hash"]}
        snapshots = {a["audit_receipt"]["identity"]["qualification_snapshot_hash"] for a in audits.values()}
        if len(snapshots) != 1 or not next(iter(snapshots)):
            raise ValueError("Scenario candidate identities differ")
        metrics.update(candidate_snapshot_hash=next(iter(snapshots)),
            feature_manifest_hash=json.loads(Path(self.study["manifest"]).read_text())["manifest_hash"],
            end_open_position_count=sum(len(r["terminal_inventory"]) for r in results.values()))
        for field in ("same_bar_fill_count", "forced_replay_close_count", "rejected_order_count", "invalid_order_rejection_count"):
            metrics[field] = sum(a[field] for a in audits.values())
        for scenario, r in results.items():
            for key, source in (("return", "return_pct"), ("dd", "drawdown_pct"), ("trades", "trades"),
                                ("active", "active_days"), ("frequency", "trades_per_21_sessions")):
                metrics[scenario+"_"+key] = r[source]
        record = {"recipe": name, "metrics": metrics, "score": score(metrics),
            "results": {s: str(folder / s / "result.json") for s in results}}
        self.cache[name] = record
        write_json(folder / "case.json", record)
        self.ledger()
        print(json.dumps({"complete": name, **metrics}), flush=True)
        return record

    def create_evaluate_batch(self, phase, cumulative_mutations, **kwargs):
        def evaluate(candidates, current):
            scored = []
            for candidate in candidates:
                record = self.evaluate({**current, **candidate.mutations})
                valid = record["metrics"]["physical_valid"]
                scored.append(ScoredCandidate(candidate.name, record["score"], not valid,
                    "native_execution_failure" if not valid else "", record["metrics"]))
            return scored
        return evaluate

    def compute_final_metrics(self, mutations):
        return self.evaluate(mutations)["metrics"]

    def ledger(self):
        records = [json.loads(p.read_text()) for p in sorted((self.root / "cases").glob("*/case.json"))]
        def objectives(row):
            m = row["metrics"]
            return [m["baseline_return"], m["combined_return"], -m["baseline_dd"], -m["combined_dd"],
                m["baseline_frequency"], m["combined_frequency"], m["baseline_active"], m["combined_active"]]
        valid = [r for r in records if r["metrics"]["physical_valid"]]
        for record in records:
            v = objectives(record)
            record["pareto"] = record in valid and not any(all(a >= b - 1e-9 for a,b in zip(objectives(other),v))
                and any(a > b + 1e-9 for a,b in zip(objectives(other),v)) for other in valid)
        write_json(self.root / "trial_ledger.json", {"scope": "reused_development", "targets_are_preferences": True, "trials": records})

    def run_phase_diagnostics(self, phase, state, metrics, greedy_result):
        self.ledger()
        return json.dumps({"phase": phase, "metrics": metrics, "scope": "screening; full qualification pending"}, indent=2)

    run_enhanced_diagnostics = run_phase_diagnostics

    def build_end_of_round_artifacts(self, state):
        record = self.evaluate(state.cumulative_mutations)
        write_json(self.root / "screening_selection.json", record)
        report = json.dumps(record, indent=2)
        return EndOfRoundArtifacts(report, {"native_economics": report}, "Research screening only")
