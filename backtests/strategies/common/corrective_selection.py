"""A frozen corrective comparison hosted by the shared phased-auto lifecycle."""
from dataclasses import asdict
from datetime import date
import json
import subprocess
import sys
from pathlib import Path

from backtests.auto.shared.cache_keys import stable_signature, source_generation_hash
from backtests.auto.shared.plugin import PhaseSpec, PhaseAnalysisPolicy
from backtests.auto.shared.types import Experiment, ScoredCandidate, GateCriterion, EndOfRoundArtifacts
from backtests.core.input_manifest import InputManifest
from backtests.engine.contracts import execution_valid
from backtests.engine.capacity import CAPACITY_SCENARIOS

VERSION = "bounded-corrective-selection-v1"
MATERIALITY_REVISION = "causal-source-and-liquidation-v2"


def shared_candidate_input_rejection(generation, scope, row, baseline, audit, *, window):
    """Prove a common required input is absent, without inferring performance."""
    from datetime import datetime
    from backtests.engine.qualification_evidence import source_rows, _covered
    from backtests.engine.retained_source import sparse_dispositions
    coverage = audit.get("coverage", {})
    failure = audit.get("qualification", {}).get("replay_completeness", "")
    if (scope.get("strategy") != "kalcb" or coverage.get("input_provenance") != "pass"
            or coverage.get("replay_completeness") != "unverified"
            or not failure.startswith("management_feed_incomplete:")):
        return None
    _, symbol, day = failure.split(":", 2)
    declared = dict(audit)
    digest = declared.pop("receipt_hash", None)
    configuration = audit.get("identity", {}).get("configuration", {})
    if (not digest or stable_signature(declared) != digest
            or audit.get("source_generation_hash") != generation.payload["source_generation_hash"]
            or audit.get("identity", {}).get("input_manifest", {}).get("manifest_hash") != generation.identity
            or configuration.get("date_range") != window or not window["start"] <= day <= window["end"]):
        return None
    if f"{symbol}:{day}" not in audit.get("qualification", {}).get("candidate_sessions", ()):
        return None
    if configuration.get("source_dependency_scope") != scope:
        return None
    current, control = row["mutations"], baseline["mutations"]
    for key in ("_kalcb.pool_source.path", "_kalcb.source.path"):
        if not current.get(key) or not control.get(key):
            return None
        if not generation.digest(current[key]) or generation.digest(current[key]) != generation.digest(control[key]):
            return None
    for key in sorted(set(current) | set(control)):
        if not key.startswith("_") or key in {"_kalcb.pool_source.path", "_kalcb.source.path"}:
            continue
        if current.get(key) != control.get(key):
            return None
    pool_path = current["_kalcb.pool_source.path"]
    rows = [json.loads(line) for line in generation.read(pool_path).decode("utf-8-sig").splitlines()]
    rows = sorted((r for r in rows if r.get("trade_date") == day),
        key=lambda r: (int(float(r.get("pool_rank") or r.get("causal_rank_in_day") or 999)), str(r.get("symbol") or "")))
    active_count = int(current.get("_kalcb.pool_source.active_count") or 16)
    if symbol not in [r.get("symbol") for r in rows[:active_count]]:
        return None
    read = lambda path: json.loads(generation.read(generation.root / path))
    evidence = read(generation.payload["evidence"]["qualification_path"])
    evidence["market_files"] = [r for r in evidence["market_files"] if r["symbol"] == symbol]
    if not evidence["market_files"]:
        return None
    with generation.activate():
        stamps = [datetime.fromisoformat(r["timestamp"]) for r in source_rows(read, generation, evidence)
                  if r["symbol"] == symbol and r["timeframe"] == "5m" and r["timestamp"].startswith(day)]
        if _covered(stamps, date.fromisoformat(day), 5, symbol, sparse_dispositions(read, evidence),
                    before=scope["candidate_intraday_before"]):
            return None  # A replay omission is not proof of absent source data.
    proof = {"reason": "shared_candidate_inputs_unverified", "baseline_receipt_hash": audit["receipt_hash"],
        "manifest_hash": generation.identity, "source_dependency_scope": scope, "required_candidate_session": f"{symbol}:{day}",
        "pool_source_hash": generation.digest(pool_path), "pool_active_count": active_count,
        "fixed_source_hash": generation.digest(current["_kalcb.source.path"]),
        "market_files": {r["path"]: generation.digest(generation.root / r["path"]) for r in evidence["market_files"]}}
    return proof


def recover_recipes(root, strategy):
    root = Path(root)
    from backtests.strategies.olr.phase_candidates import get_phase_candidates as olr_candidates
    from backtests.strategies.kalcb.fixed_trade_plan_phase import get_phase_candidates as kalcb_candidates, _rescue_experiments_for_phase, _normalize_mutations
    from strategy_kalcb.config import KALCBConfig
    from strategy_olr.config import OLRConfig
    from backtests.config import load_yaml_config
    raw = load_yaml_config(root / f"config/optimization/{strategy}.yaml")
    cls = OLRConfig if strategy == "olr" else KALCBConfig
    records, missing, sources = {}, [], {}
    import hashlib
    config_path = root / f"config/optimization/{strategy}.yaml"
    sources[config_path.relative_to(root).as_posix()] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    def read(path):
        import hashlib
        sources[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text(encoding="utf-8-sig"))
    def add(label, mutations, provenance):
        cfg = cls.from_mapping(raw, _normalize_mutations(mutations) if strategy == "kalcb" else mutations)
        effective = {"config": asdict(cfg), "private_inputs": {k:v for k,v in mutations.items() if k.startswith("_")}}
        key = stable_signature(effective)
        alias = {"label": label, "source": provenance}
        if key in records:
            records[key]["aliases"].append(alias)
        else:
            records[key] = {"id": key, "name": label, "mutations": mutations,
                            "effective_config": effective, "aliases": [alias]}
        return key
    saved_round = 5 if strategy == "olr" else 3
    saved_path = root / f"data/backtests/output/{strategy}/round_{saved_round}/optimized_config.json"
    baseline = add(f"saved_round_{saved_round}", read(saved_path)["mutations"], saved_path.relative_to(root).as_posix())
    phase_root = root / f"data/backtests/output/{strategy}/round_2"
    state = read(phase_root / "phase_state.json")
    for phase in range(1, 7):
        path = phase_root / f"phase_{phase}_greedy.json"
        result = read(path)
        parent = dict(state["phase_results"][str(phase)]["base_mutations"])
        candidates = olr_candidates(phase) if strategy == "olr" else [*kalcb_candidates(phase), *_rescue_experiments_for_phase(phase)]
        lookup = {c.name: c.mutations for c in candidates}
        rounds = {r["round_num"]: r for r in result["rounds"]}
        for number in sorted(rounds):
            for row in result["candidate_evaluations"]:
                if row["round_num"] != number:
                    continue
                if row["name"] not in lookup:
                    missing.append({"phase": phase, "name": row["name"], "reason": "recipe_not_recoverable"})
                    continue
                add(f"round2_p{phase}_r{number}_{row['name']}", {**parent, **lookup[row["name"]]}, path.relative_to(root).as_posix())
            winner = rounds[number]
            if winner.get("kept"):
                if winner["best_name"] not in lookup:
                    raise ValueError("Cannot reconstruct the next retained comparison parent")
                parent.update(lookup[winner["best_name"]])
        if stable_signature(parent) != stable_signature(state["phase_results"][str(phase)]["final_mutations"]):
            raise ValueError(f"Retained {strategy} phase {phase} does not match its reconstructed parent")
        add(f"round2_p{phase}_retained_parent", state["phase_results"][str(phase)]["final_mutations"],
            (phase_root / "phase_state.json").relative_to(root).as_posix())
    for number in range(1, saved_round + 1):
        folder = root / f"data/backtests/output/{strategy}/round_{number}"
        path = folder / "optimized_config.json"
        if path.exists():
            add(f"saved_round_{number}", read(path)["mutations"], path.relative_to(root).as_posix())
        path = folder / "evaluations.jsonl"
        if path.exists():
            import hashlib
            sources[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                row = json.loads(line)
                if row.get("mutations"):
                    add(f"round{number}_{row['label']}", row["mutations"], path.relative_to(root).as_posix())
    if strategy == "kalcb":
        missing.append({"source": "tmp/kalcb_round2_oos_high_notional_target_20260527/oos_ablation_results.json",
                        "reason": "missing_original_trial_population; saved winning recipe retained"})
    return {"baseline": baseline, "recipes": list(records.values()), "unrecoverable_lineage": missing, "source_files": sources}


def register_study(root, path):
    from backtests.strategies.olr.phase_scoring import IMMUTABLE_SCORE_COMPONENTS, PHASE_HARD_REJECTS
    from backtests.strategies.kalcb.fixed_trade_plan_phase import SCORE_COMPONENTS, HARD_REJECTS
    study = {"version": VERSION, "scope": "best_eligible_recipe_in_registered_retained_comparison",
        "qualification_revision": MATERIALITY_REVISION,
        "train": {"start": "2025-05-12", "end": "2026-03-31"},
        "regression": {"start": "2026-04-01", "end": "2026-05-12", "usage": "previously_consumed"},
        "accounts": {"olr": {"initial_equity": 10000000, "leverage": 1}, "kalcb": {"initial_equity": 100000000, "leverage": 2}},
        "scenarios": [list(row) for row in CAPACITY_SCENARIOS],
        "finalist_rule": "top_two_eligible_training_recipes_plus_eligible_baseline; freeze_before_stress",
        "stress_rule": "same_registered_economic_limits_each_case; no_reranking_using_regression",
        "shared_prerequisite_rule": "Replay every retained recipe using the declared causal source fallback; no shared coverage flag substitutes for an actual candidate replay.",
        "selection_score": {"olr": dict(IMMUTABLE_SCORE_COMPONENTS), "kalcb": dict(SCORE_COMPONENTS)},
        "limits": {"olr": dict(PHASE_HARD_REJECTS[6]), "kalcb": dict(HARD_REJECTS)},
        "intraday_requirement": "KALCB carry_off must be flat by actual continuous-session end each day",
        "source_dependency_scopes": {s: {"version": "causal-feature-and-management-v2", "strategy": s,
            "feature_intraday_before": feature, "candidate_intraday_before": candidate}
            for s, feature, candidate in (("olr", "14:30", "16:00"), ("kalcb", "09:30", "15:20"))},
        "paper_live_qualified": False,
        "strategies": {s: recover_recipes(root, s) for s in ("olr", "kalcb")}}
    study["limits"]["olr"].pop("min_total_trades", None)
    study["limits"]["kalcb"].pop("min_trades", None)
    study["activity_rule"] = "Position-level trade counts remain diagnostics and existing score components; no binary minimum trade-count qualification. Economic/risk limits otherwise unchanged."
    study["source_fallback"] = "OLR excludes incomplete completed prefixes from current and prior sector features and candidates. KALCB blocks new entries after a missing completed interval and cancels unfilled flat entries. Observed execution bars remain unchanged. Gaps overlapping working orders or positions remain unverified."
    study["kalcb_repair_policies"] = [
        {"kalcb.session.flatten_time": clock, "kalcb.risk.max_participation_30m": participation}
        for clock, participation in (("15:00", .01), ("14:45", .01), ("14:45", .005), ("14:45", .0025))]
    study["kalcb_repair_rule"] = "In listed order, replay the saved control at 1% and 0.5% execution participation. Choose the first policy with zero daily residual inventory, zero terminal positions and a valid execution/source audit in both. Short-circuit a failed policy. Ignore returns when choosing. Freeze that common overlay before comparing all 35 retained recipes. If no control policy passes, compare all recipes under the last declared overlay; no automatic promotion. Existing capital/leverage stay unchanged."
    study["study_hash"] = stable_signature(study)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(study, stream, indent=2, default=str)
    return study


class CorrectiveSelectionPlugin:
    """Fixed recipes, one shared greedy comparison; no adaptive candidate loop."""
    num_phases = 1
    ultimate_targets = {}
    capability_level = "real_replay"
    requires_full_diagnostics = False

    def __init__(self, study, strategy, generation, output):
        from backtests.config import load_yaml_config
        self.study, self.name, self.generation, self.output = study, strategy, generation, Path(output)
        self.rows = {r["id"]: r for r in study["strategies"][strategy]["recipes"]}
        self.baseline = study["strategies"][strategy]["baseline"]
        self.initial_mutations_override = {"_corrective_recipe": self.baseline}
        self.initial_mutations = dict(self.initial_mutations_override)
        with generation.activate():
            self.config = load_yaml_config(f"config/optimization/{strategy}.yaml")
        self.config.update(input_manifest_path=str(generation.path), date_range=dict(study["train"]),
            initial_equity=study["accounts"][strategy]["initial_equity"], initial_mutations=self.rows[self.baseline]["mutations"],
            baseline_integrity_required=False, previous_promotion_guard_required=False,
            validation_gate_enabled=False, holdout_excluded_from_optimization=True,
            promotion_status="corrective_research_only", skip_initial_baseline_eval=True,
            capacity_scenario={"name": "research-baseline", "participation": .01}, workers=1)
        self.config["source_dependency_scope"] = dict(study["source_dependency_scopes"][strategy])
        if study.get("qualification_revision") == MATERIALITY_REVISION:
            self.config["require_complete_feature_prefix"] = strategy == "olr"
            self.config["entry_require_complete_prefix"] = strategy == "kalcb"
        self.config["corrective_study_hash"] = study["study_hash"]
        self.config.pop("initial_mutations_path", None)
        self.source_fingerprint = generation.identity
        self.execution_context = {"strategy": strategy, "capability_level": "real_replay",
            "study_hash": study["study_hash"], "source_fingerprint": generation.identity,
            "date_window": study["train"], "initial_equity": self.config["initial_equity"],
            "artifact_promotion_policy": "bounded_historical_research_only"}
        self.metrics = {}
        self.final_report = None
        self.output.mkdir(parents=True, exist_ok=True)
        self.policy_selection = None
        if study.get("qualification_revision") == MATERIALITY_REVISION:
            overlay = {f"{strategy}.source.require_complete_prefix": True}
            if strategy == "kalcb":
                overlay.update(self._select_liquidation_policy(overlay))
            originals = self.rows
            self.rows = {new["id"]: new for row in originals.values() for new in [self._overlay_recipe(row, overlay)]}
            self.baseline = self._overlay_recipe(originals[self.baseline], overlay)["id"]
            self.initial_mutations_override = {"_corrective_recipe": self.baseline}
            self.initial_mutations = dict(self.initial_mutations_override)
            self.config["initial_mutations"] = self.rows[self.baseline]["mutations"]
            frozen_recipes = {"study_hash": study["study_hash"], "manifest_hash": generation.identity,
                "overlay": overlay, "baseline": self.baseline, "recipes": list(self.rows.values())}
            frozen_recipes["recipe_set_hash"] = stable_signature(frozen_recipes)
            target = self.output / "execution_recipes.json"
            if target.exists() and json.loads(target.read_text(encoding="utf-8")) != frozen_recipes:
                raise ValueError("Frozen execution recipes changed")
            target.write_text(json.dumps(frozen_recipes, indent=2, default=str), encoding="utf-8")
        code_identity = source_generation_hash()
        for path in (self.output / "candidate_receipts").glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            key = payload.get("recipe", {}).get("id")
            if (payload.get("study_hash") == study["study_hash"] and payload.get("source_generation_hash") == code_identity
                    and payload.get("manifest_hash") == generation.identity and payload.get("recipe") == self.rows.get(key)):
                self.metrics[key] = payload["metrics"]

    def _overlay_recipe(self, row, overlay):
        from strategy_kalcb.config import KALCBConfig
        from strategy_olr.config import OLRConfig
        mutations = {**row["mutations"], **overlay}
        cls = OLRConfig if self.name == "olr" else KALCBConfig
        normalized = mutations
        if self.name == "kalcb":
            from backtests.strategies.kalcb.fixed_trade_plan_phase import _normalize_mutations
            normalized = _normalize_mutations(mutations)
        effective = {"config": asdict(cls.from_mapping({**self.config, "initial_mutations": mutations}, normalized)),
            "private_inputs": {k:v for k,v in mutations.items() if k.startswith("_")}}
        effective = json.loads(json.dumps(effective, default=str))
        return {**row, "id": stable_signature(effective), "retained_recipe_id": row["id"],
            "mutations": mutations, "effective_config": effective}

    def _select_liquidation_policy(self, source_overlay):
        results, chosen = [], None
        for policy in self.study["kalcb_repair_policies"]:
            row = self._overlay_recipe(self.rows[self.baseline], {**source_overlay, **policy})
            cases = []
            for participation in (.01, .005):
                print(json.dumps({"event": "liquidation_policy_probe", "policy": policy, "participation": participation}), flush=True)
                metrics = self._case(row, f"policy-{participation}", participation, 0, final=False)
                passed = (execution_valid(metrics) and metrics.get("intraday_residual_session_count") == 0
                    and metrics.get("end_open_position_count") == 0)
                cases.append({"participation": participation, "passed": passed, "metrics": metrics})
                if not passed:
                    break
            passed = len(cases) == 2 and all(case["passed"] for case in cases)
            results.append({"policy": policy, "passed": passed, "cases": cases})
            if passed:
                chosen = policy
                break
        chosen = chosen or self.study["kalcb_repair_policies"][-1]
        self.policy_selection = {"study_hash": self.study["study_hash"], "manifest_hash": self.generation.identity,
            "selected_overlay": chosen, "control_policy_passed": results[-1]["passed"], "results": results}
        self.policy_selection["policy_selection_hash"] = stable_signature(self.policy_selection)
        (self.output / "execution_policy_selection.json").write_text(json.dumps(self.policy_selection, indent=2, default=str), encoding="utf-8")
        return chosen

    def _recipe(self, mutations):
        if "_corrective_recipe" in mutations:
            return self.rows[mutations["_corrective_recipe"]]
        digest = stable_signature(mutations)
        return next(r for r in self.rows.values() if stable_signature(r["mutations"]) == digest)

    def _evaluate(self, row):
        key = row["id"]
        if key in self.metrics:
            return self.metrics[key]
        path = self.output / "candidate_receipts" / f"{key}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (payload["study_hash"] == self.study["study_hash"] and payload["source_generation_hash"] == source_generation_hash()
                    and payload.get("manifest_hash") == self.generation.identity and payload.get("recipe") == row):
                self.metrics[key] = payload["metrics"]
                return self.metrics[key]
        control = self.metrics.get(self.baseline, {})
        audit = control.get("audit_receipt", {})
        coverage = audit.get("coverage", {})
        candidate_proof = (shared_candidate_input_rejection(self.generation, self.config["source_dependency_scope"],
            row, self.rows[self.baseline], audit, window=self.study["train"])
            if self.name == "kalcb" and key != self.baseline and self.study.get("qualification_revision") != MATERIALITY_REVISION else None)
        if (self.study.get("qualification_revision") != MATERIALITY_REVISION and key != self.baseline and coverage.get("input_provenance") == "pass"
                and coverage.get("source_completeness") == "unverified"
                and audit.get("qualification", {}).get("source_completeness")):
            proof = {"baseline_receipt_hash": audit["receipt_hash"], "manifest_hash": self.generation.identity,
                "source_dependency_scope": self.config["source_dependency_scope"], "study_hash": self.study["study_hash"],
                "recipe_id": key, "missing_dependencies": audit["qualification"]["source_completeness"]}
            proof["proof_hash"] = stable_signature(proof)
            metrics = {"audit_pass": False, "audit_status": "unverified", "replay_performed": False,
                "source_precondition_rejection": proof}
        elif candidate_proof:
            proof = {**candidate_proof, "study_hash": self.study["study_hash"], "recipe_id": key}
            proof["proof_hash"] = stable_signature(proof)
            metrics = {"audit_pass": False, "audit_status": "unverified", "replay_performed": False,
                "source_precondition_rejection": proof}
        else:
            metrics = self._case(row, "selection", .01, 0, final=False)
            metrics["replay_performed"] = True
        metrics["corrective_recipe_id"] = key
        metrics["corrective_study_hash"] = self.study["study_hash"]
        self.metrics[key] = metrics
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"study_hash": self.study["study_hash"], "source_generation_hash": source_generation_hash(),
            "manifest_hash": self.generation.identity, "recipe": row, "metrics": metrics}, indent=2, default=str), encoding="utf-8")
        return metrics

    def _score(self, metrics):
        if metrics.get("source_precondition_rejection"):
            return None, metrics["source_precondition_rejection"].get("reason", "shared_feature_inputs_unverified")
        if not execution_valid(metrics):
            return None, "execution_or_source_audit_unverified_or_failed"
        if self.name == "olr":
            from backtests.strategies.olr.phase_scoring import olr_reject_reason, score_olr_phase
            reject = olr_reject_reason(6, metrics, self.study["limits"][self.name])
            score = score_olr_phase(6, metrics, self.study["selection_score"][self.name])
        else:
            from backtests.strategies.kalcb.fixed_trade_plan_phase import reject_reason, score_fixed
            reject = reject_reason(metrics, self.study["limits"][self.name])
            score = score_fixed(metrics, self.study["selection_score"][self.name])
            if self.rows[metrics["corrective_recipe_id"]]["effective_config"]["config"].get("carry_mode") == "off" and metrics.get("intraday_residual_session_count", float("inf")):
                reject = "intraday_inventory_remains_after_continuous_session"
        return (None if reject else score), reject

    def _case(self, row, name, participation, extra_bps, *, final, regression=False):
        folder = self.output / "cases" / row["id"] / name
        folder.mkdir(parents=True, exist_ok=True)
        request = {"strategy": self.name, "config": self.config, "recipe": row,
            "manifest": str(self.generation.path), "manifest_hash": self.generation.identity,
            "source_generation_hash": source_generation_hash(), "study_hash": self.study["study_hash"],
            "participation": participation, "extra_bps": extra_bps, "final": final,
            "regression": dict(self.study["regression"]) if regression else None, "output": str(folder.resolve())}
        request["request_hash"] = stable_signature(request)
        result_path = folder / "result.json"
        if result_path.exists():
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            if saved.get("request_hash") == request["request_hash"]:
                return saved["metrics"]
        request_path = folder / "request.json"
        request_path.write_text(json.dumps(request, indent=2, default=str), encoding="utf-8")
        # Each actual evaluator owns its caches; process exit releases large replay
        # objects before the next recipe. PhaseRunner still owns the study lifecycle.
        with (folder / "worker.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-X", "utf8", "-m", "backtests.auto.cli", "corrective-case",
                "--request", str(request_path.resolve())], stdout=log, stderr=subprocess.STDOUT, check=True)
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        if saved["request_hash"] != request["request_hash"]:
            raise ValueError("Corrective case receipt mismatch")
        return saved["metrics"]

    def _qualify(self, finalists):
        results = {}
        for key in finalists:
            row = self.rows[key]
            cases = {}
            for name, participation, extra_bps in self.study["scenarios"]:
                print(json.dumps({"event": "qualification_start", "strategy": self.name, "recipe": key, "scenario": name}), flush=True)
                metrics = self._case(row, name, participation, extra_bps, final=True)
                metrics["corrective_recipe_id"] = key
                _, reason = self._score(metrics)
                parity = self.name != "kalcb" or metrics.get("fast_full_audit_passed") is True
                cold = True
                if participation == .01 and extra_bps == 0:
                    before = self.metrics[key]
                    cold = all(metrics.get(field) == before.get(field) for field in
                        ("official_mtm_net_return_pct", "final_equity", "total_trades", "source_fingerprint"))
                    cold = cold and metrics["audit_receipt"]["ledger_hash"] == before["audit_receipt"]["ledger_hash"]
                cases[name] = {"passed": not reason and parity and cold, "rejection": reason,
                    "compiled_parity": parity, "cold_parity": cold, "metrics": metrics}
            regression = self._case(row, "consumed-regression", .01, 0, final=True, regression=True)
            results[key] = {"qualified": all(case["passed"] for case in cases.values()), "scenarios": cases,
                "regression": {"usage": "previously_consumed; diagnostic_only; no_reranking", "metrics": regression}}
        return results

    def get_phase_spec(self, phase, state):
        return PhaseSpec("Frozen corrective comparison of retained recipes",
            [Experiment(r["name"], {"_corrective_recipe": key}) for key, r in self.rows.items() if key != self.baseline],
            lambda m: [GateCriterion("registered_eligibility", 1, int(not self._score(m)[1]), not self._score(m)[1])],
            None, {}, PhaseAnalysisPolicy(), max_rounds=1, phase_metric_basis="corrected_training_replay",
            primary_promotion_metric="official_mtm_net_return_pct", promotion_requires_audit_pass=True)

    def create_evaluate_batch(self, phase, cumulative_mutations, **kwargs):
        def evaluate(items, parent):
            result = []
            for item in items:
                row = self._recipe({**parent, **item.mutations})
                print(json.dumps({"event": "candidate_start", "strategy": self.name, "name": row["name"], "id": row["id"]}), flush=True)
                metrics = self._evaluate(row)
                score, reason = self._score(metrics)
                result.append(ScoredCandidate(item.name, score if score is not None else -1e100, bool(reason), reason, metrics))
                print(json.dumps({"event": "candidate_complete", "strategy": self.name, "name": row["name"], "eligible": not bool(reason), "reason": reason}), flush=True)
            return result
        return evaluate

    def compute_final_metrics(self, mutations):
        return self._evaluate(self._recipe(mutations))

    def run_phase_diagnostics(self, phase, state, metrics, greedy_result):
        return json.dumps({"registered_recipes": len(self.rows), "evaluated": len(self.metrics),
            "eligible": sum(not self._score(m)[1] for m in self.metrics.values()),
            "scope": self.study["scope"]}, indent=2)

    run_enhanced_diagnostics = run_phase_diagnostics

    def build_end_of_round_artifacts(self, state):
        eligible = sorted(((self._score(m)[0], key) for key, m in self.metrics.items() if not self._score(m)[1]), reverse=True)
        finalists = [key for _, key in eligible[:2]]
        if any(key == self.baseline for _, key in eligible) and self.baseline not in finalists:
            finalists.append(self.baseline)
        frozen = {"study_hash": self.study["study_hash"], "source_generation_hash": source_generation_hash(),
                  "manifest_hash": self.generation.identity,
                  "finalists": [self.rows[key] for key in finalists]}
        frozen["finalist_hash"] = stable_signature(frozen)
        target = self.output / "frozen_finalists.json"
        if target.exists() and json.loads(target.read_text(encoding="utf-8")) != frozen:
            raise ValueError("Frozen finalist set changed")
        target.write_text(json.dumps(frozen, indent=2, default=str), encoding="utf-8")
        if len(self.metrics) != len(self.rows):
            raise ValueError("Registered comparison is incomplete")
        qualification = self._qualify(finalists)
        qualified = [key for _, key in eligible if key in qualification and qualification[key]["qualified"]]
        selected = qualified[0] if qualified else None
        self.final_report = {"process_status": "completed", "registered_recipes": len(self.rows),
            "dispositioned_recipes": len(self.metrics),
            "replayed_recipes": sum(m.get("replay_performed", True) for m in self.metrics.values()),
            "shared_prerequisite_rejections": sum(bool(m.get("source_precondition_rejection")) for m in self.metrics.values()),
            "eligible_recipes": len(eligible), "finalist_ids": finalists,
            "study_hash": self.study["study_hash"], "manifest_hash": self.generation.identity,
            "historical_qualification": bool(selected), "paper_live_qualified": False,
            "qualification_status": "qualified_under_declared_capacity_model" if selected else "no_qualifying_candidate",
            "selected_recipe": selected, "qualification": qualification,
            "qualification_revision": self.study.get("qualification_revision"),
            "execution_policy_selection": getattr(self, "policy_selection", None),
            "unrecoverable_lineage": self.study["strategies"][self.name]["unrecoverable_lineage"]}
        self.config["promotion_status"] = self.final_report["qualification_status"]
        self.execution_context["historical_qualification"] = bool(selected)
        self.execution_context["selected_recipe"] = selected
        # The saved control remains inspectable when nobody qualifies, but its
        # status explicitly forbids treating it as a promoted fallback.
        state.cumulative_mutations = dict(self.rows[selected or self.baseline]["mutations"])
        self.metrics[selected or self.baseline]["corrective_qualification"] = {
            k:v for k,v in self.final_report.items() if k != "qualification"}
        return EndOfRoundArtifacts(json.dumps(self.final_report, indent=2),
            {"selection": self.run_phase_diagnostics(1, state, {}, None)}, self.final_report["qualification_status"])


def run_study(study_path, manifest_path, strategy, output_root):
    from backtests.auto.shared.phase_runner import PhaseRunner
    from backtests.auto.shared.round_manager import RoundManager
    study = json.loads(Path(study_path).read_text(encoding="utf-8"))
    declared = dict(study)
    digest = declared.pop("study_hash")
    if study["version"] != VERSION or stable_signature(declared) != digest:
        raise ValueError("Registered corrective study identity mismatch")
    generation = InputManifest(manifest_path)
    if generation.payload["source_generation_hash"] != source_generation_hash():
        raise ValueError("Source differs from the frozen study generation")
    import hashlib
    for name, expected in study["strategies"][strategy]["source_files"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError("Retained study lineage changed: " + name)
    manager = RoundManager("stock", strategy, base_dir=Path(output_root))
    number, output = manager.resolve_round(1, for_write=True, expected_phases=1)
    plugin = CorrectiveSelectionPlugin(study, strategy, generation, output)
    PhaseRunner(plugin, output, round_name="bounded_corrective_selection", max_rounds=1,
        min_delta=0, max_retries=0, max_diagnostic_retries=0, round_manager=manager, round_num=number).run_all_phases()
    if generation.payload["source_generation_hash"] != source_generation_hash():
        raise ValueError("Source changed during the registered comparison")
    return plugin.final_report


def evaluate_case(request_path):
    """Run an existing strategy evaluator in a resource-isolated worker."""
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    declared = dict(request)
    digest = declared.pop("request_hash")
    if digest != stable_signature(declared) or request["source_generation_hash"] != source_generation_hash():
        raise ValueError("Case request or source identity changed")
    generation = InputManifest(request["manifest"])
    if generation.identity != request["manifest_hash"]:
        raise ValueError("Case input identity changed")
    strategy = request["strategy"]
    config = dict(request["config"])
    mutations = dict(request["recipe"]["mutations"])
    if request["extra_bps"]:
        mutations[f"{strategy}.robustness.slippage_bps"] = float(request["recipe"]["effective_config"]["config"]["slippage_bps"]) + request["extra_bps"]
        if strategy == "olr":
            mutations["olr.cost.slippage_bps"] = mutations["olr.robustness.slippage_bps"]
    config["capacity_scenario"] = {"name": next(name for name,p,b in CAPACITY_SCENARIOS if p == request["participation"] and b == request["extra_bps"]), "participation": request["participation"]}
    config["initial_mutations"] = mutations
    regression = request.get("regression")
    if regression:
        config.update(date_range={k:regression[k] for k in ("start", "end")}, start=regression["start"], end=regression["end"], use_full_available_window=True)
        if strategy == "kalcb" and "_kalcb.pool_source.path" in mutations:
            mutations["_kalcb.pool_source.path"] = mutations["_kalcb.pool_source.path"].replace("train_guarded_prefilter_pool_rows", "holdout_guarded_prefilter_pool_rows")
    if strategy == "kalcb":
        # Isolated case directories have no preceding round from which the
        # existing evaluator can infer its initial candidate source.
        config["fixed_candidate_source"] = {
            "path": mutations["_kalcb.source.path"],
            "section": mutations.get("_kalcb.source.section", "top_portfolio_proxy"),
            "rank": int(mutations.get("_kalcb.source.rank", 0)),
        }
    with generation.activate():
        if strategy == "olr":
            if regression:
                from backtests.strategies.olr.replay_cache import load_olr_declared_window_bundle
                from backtests.strategies.olr.runner import run_olr_backtest
                bundle = load_olr_declared_window_bundle(config, mutations)
                metrics = dict(run_olr_backtest(config, mutations, replay_bundle=bundle).metrics)
            else:
                from backtests.strategies.olr.plugin import OLROptimizationPlugin
                backend = OLROptimizationPlugin(config, output_dir=Path(request["output"]) / "evaluator", max_workers=1)
                metrics = dict(backend.compute_final_metrics(mutations))
        else:
            from backtests.strategies.kalcb.fixed_trade_plan_phase import KALCBFixedTradePlanOptimizationPlugin
            from backtests.strategies.common.plugin_base import attach_official_metric_contract, build_execution_contract
            backend = KALCBFixedTradePlanOptimizationPlugin(config, output_dir=Path(request["output"]) / "evaluator", max_workers=1)
            if request["final"]:
                metrics = dict(backend.compute_final_metrics(mutations))
            else:
                metrics = dict(backend.evaluate_mutations(mutations))
                metrics = attach_official_metric_contract(metrics, primary_metric="official_mtm_net_return_pct", requires_audit_pass=True,
                    audit_pass=execution_valid(metrics), audit_status=metrics.get("audit_status", "unverified"), official_replay_pass=True,
                    execution_contract=build_execution_contract(backend, metrics))
    if request["source_generation_hash"] != source_generation_hash():
        raise ValueError("Source changed during evaluation")
    Path(request["output"], "result.json").write_text(json.dumps({"request_hash": digest, "metrics": metrics}, indent=2, default=str), encoding="utf-8")
