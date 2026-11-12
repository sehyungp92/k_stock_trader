"""One registered development comparison, using PhaseRunner and frozen native replays.

No historical holdout is opened, and no canonical round is promoted. The four
missing scenarios are evaluated once; failed gates retain the complete control.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.olr_rule_study import digest, load_study, read, signature, write
from scripts.olr_rule_analysis import load_case, select_finalist, trade_difference, trade_statistics, candidate_difference
from scripts.olr_rule_finish import qualification_decision, published_control_stress


MODES = ("sizing_cap", "participation-stress", "cost-stress", "combined-stress")
PARENT_CASE = "stability__control__low_plus25"


def phase_metrics(metrics, directory):
    """Describe existing native evidence in the shared phase metadata schema."""
    receipt = read(Path(directory) / "completed.json")
    source = metrics["audit_receipt"]["identity"]["input_manifest"]
    binding = {"scheme": "frozen-generation-and-exported-candidate-evidence-v1",
               "source_generation_hash": metrics["audit_receipt"]["source_generation_hash"],
               "input_manifest_hash": source["manifest_hash"],
               "source_snapshot_hash": metrics["source_snapshot_hash"],
               "candidate_evidence_sha256": receipt["outputs"]["candidate_selection.jsonl.gz"]}
    return {**metrics, "source_fingerprint": signature({key: binding[key] for key in
                ("source_generation_hash", "input_manifest_hash", "source_snapshot_hash")}),
            "feature_manifest_hash": binding["candidate_evidence_sha256"],
            "phase_evidence_binding": binding, "capability_level": "full_native_replay"}


def register(parent_path, destination):
    parent = load_study(parent_path)
    destination = Path(destination).resolve()
    if (destination / "study.json").exists():
        spec = load_study(destination / "study.json")
        if spec["parent_study_hash"] != parent["study_hash"]:
            raise ValueError("Different parent study")
        return spec
    challenger = read(Path(parent["output"]) / "registered_neighbours" / (PARENT_CASE + ".json"))
    spec = copy.deepcopy(parent)
    spec.pop("study_hash")
    spec.update(version="olr-low325-bounded-development-v1", output=str(destination),
                parent_study=str(Path(parent_path).resolve()), parent_study_hash=parent["study_hash"],
                registered_at_utc=datetime.now(timezone.utc).isoformat(),
                cases={"control": parent["cases"]["control"], "low325": {"mutations": challenger}},
                additional_modes=list(MODES),
                independent_evaluation={"outcomes_opened": False, "adaptive_reselection": False},
                stability_policy={"scope": "Existing control neighbourhood is supporting development evidence only; no new neighbourhood search or claim of full inherited-study qualification."})
    spec["study_hash"] = signature(spec)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "temp").mkdir(exist_ok=True)
    write(destination / "study.json", spec)
    # Reuse the reviewed memory wrapper, not a copy of the changing workspace engine.
    shutil.copyfile(Path(parent["output"]) / "memory_runtime_v6_manifest.json", destination / "memory_runtime_v6_manifest.json")
    return spec


def run_missing(spec, mode):
    directory = Path(spec["output"]) / "cases/low325" / mode
    if not any((directory / f).exists() for f in ("completed.json", "rejected.json")):
        directory.mkdir(parents=True, exist_ok=True)
        worker = Path(spec["parent_study"]).parent / "harness_v6/olr_rule_memory_worker.py"
        print("START", mode, flush=True)
        with (directory / "worker.log").open("a", encoding="utf-8") as stream:
            subprocess.run([sys.executable, "-B", "-u", str(worker), "worker", "--study",
                            str(Path(spec["output"]) / "study.json"), "--case", "low325", "--mode", mode],
                           cwd=spec["engine_root"], stdout=stream, stderr=subprocess.STDOUT, check=True)
    result = load_case(spec, "low325", mode)
    print("RESOLVED", mode, "rejected" if result.get("qualification_rejected") else
          {k: result["metrics"][k] for k in ("official_mtm_net_return_pct", "total_trades", "max_drawdown_pct")}, flush=True)
    return mode, result


class Low325Qualification:
    name = "olr"
    num_phases = 1
    ultimate_targets = {}

    def __init__(self, spec, workers):
        self.spec, self.workers = spec, workers
        self.parent = load_study(spec["parent_study"])
        self.initial_mutations = copy.deepcopy(spec["cases"]["control"]["mutations"])
        self.decision = None
        self.cases = {}
        self.control_metrics = read(Path(self.parent["output"]) / "cases/control/nominal/metrics.json")
        # Previous stress request/economic equivalence is already established;
        # verify its bound published metrics rather than rerun identical accounts.
        self.control_stress = published_control_stress(spec)

    def qualify(self):
        if self.decision is not None:
            return
        new_cases = {}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for mode, result in pool.map(lambda mode: run_missing(self.spec, mode), MODES):
                new_cases[mode] = result
        # Candidate/trade inventories are needed only after the native workers
        # finish. Avoid retaining the old accounts while rebuilding new ones.
        self.cases = {"control": {mode: load_case(self.parent, "control", mode) for mode in ("nominal", "sizing_cap")},
                      "low325": {"nominal": load_case(self.parent, PARENT_CASE, "nominal"), **new_cases}}
        selected = select_finalist(self.cases, self.spec["selection_policy"],
                                   read(Path(self.parent["output"]) / "support_policy.json"))
        stresses = {mode: self.cases["low325"][mode]["metrics"] for mode in MODES[1:]}
        self.decision = qualification_decision(selected["selected_challenger"], stresses,
            self.control_stress, {}, self.cases["low325"]["nominal"]["metrics"], self.spec)
        self.decision.update(development_selection=selected, study_hash=self.spec["study_hash"],
                             qualified_at_utc=datetime.now(timezone.utc).isoformat(),
                             scope="Bounded nominal, sizing-cap and three-stress development qualification; independent validation pending; canonical round unchanged")
        write(Path(self.spec["output"]) / "qualification_decision.json", self.decision)
        write(Path(self.spec["output"]) / "scenario_summary.json", {
            name: {mode: {"metrics": {k: case["metrics"].get(k) for k in
                ("audit_pass", "official_mtm_net_return_pct", "max_drawdown_pct", "total_trades", "profit_factor")},
                "mean_net_r": case.get("trade_statistics", {}).get("mean_net_r"), "directory": case["directory"]}
            for mode, case in modes.items()} for name, modes in self.cases.items()})
        comparisons = {}
        for mode in ("nominal", "sizing_cap"):
            candidate, control = self.cases["low325"][mode], self.cases["control"][mode]
            if candidate.get("qualification_rejected"):
                continue
            changes = trade_difference(candidate["trades"], control["trades"])
            comparisons[mode] = {"trades": changes,
                "added_trade_statistics": trade_statistics(changes["added"]),
                "removed_trade_statistics": trade_statistics(changes["removed"]),
                "candidate_changes": candidate_difference(candidate["candidates"], control["candidates"]),
                "interpretation": "Development attribution; common-trade PnL also reflects changed equity, sizing and account path. Net trade-count change is not the count of newly admitted trades."}
        write(Path(self.spec["output"]) / "incremental_trade_attribution.json", comparisons)

    def get_phase_spec(self, phase, state):
        return PhaseSpec(focus="One low-band-325 hypothesis; no adaptive search",
            candidates=[Experiment("low325", self.spec["cases"]["low325"]["mutations"])],
            gate_criteria_fn=lambda m: [GateCriterion("qualified_native_account", 1, int(m["audit_pass"]), bool(m["audit_pass"]))],
            scoring_weights=None, hard_rejects={}, analysis_policy=PhaseAnalysisPolicy(), max_rounds=1,
            phase_metric_basis="verified_frozen_native_accounts",
            primary_promotion_metric="official_mtm_net_return_pct",
            official_metric_keys=("official_mtm_net_return_pct",), promotion_requires_audit_pass=True)

    def create_evaluate_batch(self, phase, cumulative_mutations, **kwargs):
        def evaluate(experiments, base):
            results = []
            for exp in experiments:
                if exp.name == "__baseline__":
                    results.append(ScoredCandidate(exp.name, 1.0, metrics=self.compute_final_metrics(base)))
                    continue
                self.qualify()
                rejected = self.decision["selected"] != "low325"
                assessment = self.decision["development_selection"]["assessment"]["low325"]
                gain = assessment.get("minimum_relative_wealth_gain") or 0
                results.append(ScoredCandidate(exp.name, 1 + max(gain, 1e-9), rejected,
                    json.dumps(self.decision["development_selection"] if rejected else {}),
                    self.cases["low325"]["nominal"]["metrics"]))
            return results
        return evaluate

    def compute_final_metrics(self, mutations):
        name = "control" if signature(mutations) == signature(self.initial_mutations) else "low325"
        if not self.cases and name == "control":
            return phase_metrics(self.control_metrics, Path(self.parent["output"]) / "cases/control/nominal")
        case = self.cases[name]["nominal"]
        return phase_metrics(case["metrics"], case["directory"])

    def run_phase_diagnostics(self, *args):
        return json.dumps(self.decision, indent=2)

    run_enhanced_diagnostics = run_phase_diagnostics

    def build_end_of_round_artifacts(self, state):
        self.qualify()
        if state.cumulative_mutations != self.decision["mutations"]:
            raise ValueError("PhaseRunner recipe differs from registered qualification decision")
        write(Path(self.spec["output"]) / "selected_recipe.json", {
            "selected": self.decision["selected"], "mutations": state.cumulative_mutations,
            "mutations_hash": signature(state.cumulative_mutations), "study_hash": self.spec["study_hash"]})
        return EndOfRoundArtifacts(json.dumps(self.decision, indent=2), {}, self.decision["status"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    spec = register(args.parent, args.output)
    # Native receipt validation checks the executing source generation. Run the
    # shared framework from the same frozen engine, never the mutable workspace.
    if "backtests" in sys.modules:
        raise RuntimeError("Run qualification in a fresh process before importing a different backtest engine")
    sys.path.insert(0, spec["engine_root"])
    os.chdir(spec["engine_root"])
    global PhaseAnalysisPolicy, PhaseSpec, EndOfRoundArtifacts, Experiment, GateCriterion, ScoredCandidate
    from backtests.auto.shared.phase_runner import PhaseRunner
    from backtests.auto.shared.plugin import PhaseAnalysisPolicy, PhaseSpec
    from backtests.auto.shared.types import EndOfRoundArtifacts, Experiment, GateCriterion, ScoredCandidate
    plugin = Low325Qualification(spec, args.workers)
    # Retain the earlier adapter attempts, which failed missing shared metadata
    # and execution-source identity. Economic outcomes and thresholds are unchanged.
    PhaseRunner(plugin, Path(spec["output"]) / "phase_v3", round_name="bounded-low325-development",
                max_rounds=1, min_delta=0, max_retries=0, max_diagnostic_retries=0).run_all_phases()


if __name__ == "__main__":
    main()
