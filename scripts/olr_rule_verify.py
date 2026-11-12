"""Independent post-run identity checks; no selection or strategy transitions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

try:
    from scripts.olr_rule_study import digest, load_study, read, signature, stability_recipes, write
    from scripts.olr_rule_analysis import load_case, select_finalist
    from scripts.olr_rule_finish import qualification_decision
except ModuleNotFoundError:
    from olr_rule_study import digest, load_study, read, signature, stability_recipes, write
    from olr_rule_analysis import load_case, select_finalist
    from olr_rule_finish import qualification_decision


def verify_qualification_evidence(spec, challenger, evidence, actual_metrics, decision):
    """Require every registered qualification replay, including rejected trials."""
    parents = ["control"] + ([challenger] if challenger else [])
    if set(evidence["stability"]) != set(parents) or set(evidence["stress"]) != set(parents):
        raise ValueError("Qualification parents differ from the frozen finalist")
    for parent in parents:
        expected = stability_recipes(spec["cases"][parent]["mutations"])
        if set(evidence["stability"][parent]) != set(expected):
            raise ValueError("Registered local stability evidence is incomplete")
        for label, metrics in evidence["stability"][parent].items():
            name = "stability__" + parent + "__" + label.replace("+", "plus").replace("-", "minus")
            if actual_metrics.get((name, "nominal")) != metrics:
                raise ValueError("Local stability summary differs from audited replay")
        if set(evidence["stress"][parent]) != set(spec["stress_policy"]["scenarios"]):
            raise ValueError("Registered stress evidence is incomplete")
        for mode, metrics in evidence["stress"][parent].items():
            if actual_metrics.get((parent, mode)) != metrics:
                raise ValueError("Stress summary differs from audited replay")
    computed = qualification_decision(
        challenger, evidence["stress"].get(challenger, {}), evidence["stress"]["control"],
        evidence["stability"].get(challenger, {}), actual_metrics[(challenger or "control", "nominal")], spec)
    if any(decision.get(key) != value for key, value in computed.items()):
        raise ValueError("Saved qualification decision differs from registered gates")


def verify(study, completed_only=False):
    spec = load_study(study)
    root = Path(spec["output"])
    sys.path.insert(0, spec["engine_root"])
    from strategy_olr.config import OLRConfig
    from backtests.engine.capacity import CAPACITY_SCENARIOS
    allowed = {name: row["mutations"] for name, row in spec["cases"].items()}
    for name, case in spec["cases"].items():
        for label, mutations in stability_recipes(case["mutations"]).items():
            allowed["stability__" + name + "__" + label.replace("+", "plus").replace("-", "minus")] = mutations
    verified, actual_metrics, core_cases = [], {}, {}
    receipt_paths = list((root / "cases").glob("*/*/completed.json")) + list((root / "cases").glob("*/*/rejected.json"))
    for receipt_path in sorted(receipt_paths):
        mode, name = receipt_path.parent.name, receipt_path.parent.parent.name
        if name not in allowed:
            raise ValueError("Unregistered evaluated recipe: " + name)
        job = read(receipt_path.parent / "job.json")
        mutations = dict(allowed[name])
        config = dict(spec["config"])
        if mode == "sizing_cap":
            mutations["olr.allocation.equity_cap_krw"] = float(config["initial_equity"])
        elif mode in spec["stress_policy"]["scenarios"]:
            _, participation, bps = next(row for row in CAPACITY_SCENARIOS if row[0] == mode)
            cfg = OLRConfig.from_mapping(config, mutations)
            mutations["olr.cost.slippage_bps"] = cfg.slippage_bps + bps
            mutations["olr.robustness.slippage_bps"] = cfg.slippage_bps + bps
            config["capacity_scenario"] = {"name": mode, "participation": participation}
        elif mode == "regression":
            config["date_range"] = dict(config["regression_window"])
            config["use_full_available_window"] = True
            config.pop("holdout_start", None)
            config.pop("holdout_end", None)
        elif mode != "nominal":
            raise ValueError("Unregistered evaluation mode")
        if mutations != job["mutations"] or config != job["config"]:
            raise ValueError("Job differs from its named, registered intervention")
        loaded = load_case(spec, name, mode)
        m = loaded["metrics"]
        actual_metrics[(name, mode)] = m
        if name in spec["cases"] and mode in ("nominal", "sizing_cap"):
            core_cases.setdefault(name, {})[mode] = loaded
        if name == "control" and mode == "nominal":
            for key in ("official_mtm_net_return_pct", "max_drawdown_pct", "total_trades", "profit_factor"):
                if abs(m[key] - spec["baseline_metrics"][key]) > 1e-9:
                    raise ValueError("Baseline parity differs")
        verified.append({"case": name, "mode": mode, "receipt_sha256": digest(receipt_path),
                         "request": m["evaluation_request"], "trades": m.get("total_trades"),
                         "status": "unqualified_native_result" if loaded.get("qualification_rejected") else "qualified"})
    if not completed_only:
        expected = {(name, mode) for name in spec["cases"] for mode in ("nominal", "sizing_cap")}
        if not expected <= {(row["case"], row["mode"]) for row in verified}:
            raise ValueError("Core study incomplete")
        decision = read(root / "qualification_decision.json")
        selected = read(root / "selected_recipe.json")
        import yaml
        config = yaml.safe_load((root / "selected_config.yaml").read_text(encoding="utf-8"))
        if (decision["mutations"] != allowed[decision["selected"]]
                or selected["mutations"] != decision["mutations"]
                or config["initial_mutations"] != decision["mutations"]
                or selected["mutations_hash"] != signature(decision["mutations"])):
            raise ValueError("Selected saved recipe does not match the qualified decision")
        if decision["live_promotion"] or decision["independent_oos_pass"]:
            raise ValueError("Development results incorrectly promoted")
        frozen = read(root / "frozen_finalist.json")
        support = read(root / "support_policy.json")
        support_hash = support.pop("policy_hash")
        if signature(support) != support_hash or support["study_hash"] != spec["study_hash"]:
            raise ValueError("Support policy identity differs")
        recomputed = select_finalist(core_cases, spec["selection_policy"], support)
        if any(frozen.get(key) != value for key, value in recomputed.items()):
            raise ValueError("Frozen finalist differs from registered selection gates")
        published = read(Path(spec["round_dir"]) / "artifact_manifest.json")
        if (published["source_generation_hash"] != spec["source_generation_hash"]
                or published["input_manifest_hash"] != read(spec["input_manifest"])["manifest_hash"]):
            raise ValueError("Published control stress lineage differs")
        for mode in spec["stress_policy"]["scenarios"]:
            relative = "diagnostics/" + mode + "/metrics.json"
            path = Path(spec["round_dir"]) / relative
            if digest(path) != published["files"][relative]:
                raise ValueError("Published control stress bytes changed")
            actual_metrics[("control", mode)] = read(path)
        verify_qualification_evidence(spec, frozen["selected_challenger"],
                                      read(root / "qualification_results.json"), actual_metrics, decision)
        if decision["qualified_at_utc"] < frozen["frozen_at_utc"]:
            raise ValueError("Finalist was not frozen before qualification")
        for row in verified:
            if row["mode"] == "regression":
                label = "rejected.json" if row["status"] == "unqualified_native_result" else "completed.json"
                timestamp = "resolved_at_utc" if label == "rejected.json" else "completed_at_utc"
                job_time = read(root / "cases" / row["case"] / row["mode"] / label)[timestamp]
                if job_time < decision["qualified_at_utc"]:
                    raise ValueError("Consumed regression preceded the frozen qualification decision")
        if read(root / "analysis/price_path_attribution.json")["status"] != "complete":
            raise ValueError("Price-path attribution has incomplete support")
        contract = read(root / "prospective_validation_contract.json")
        contract_hash = contract.pop("contract_hash")
        if (signature(contract) != contract_hash or contract["selected"] != decision["selected"]
                or contract["mutations"] != decision["mutations"]
                or contract["frozen_at_utc"] < decision["qualified_at_utc"]):
            raise ValueError("Prospective contract differs from the qualified recipe")
        artifacts = read(root / "study_artifacts.json")
        if artifacts["study_hash"] != spec["study_hash"]:
            raise ValueError("Final artifact manifest belongs to another study")
        for relative, expected_hash in artifacts["files"].items():
            if digest(root / relative) != expected_hash:
                raise ValueError("Final artifact bytes changed: " + relative)
    if digest(Path(spec["round_dir"]) / "diagnostics/training/metrics.json") != spec["baseline_metrics_sha256"]:
        raise ValueError("Canonical baseline changed during study")
    output = root / ("partial_identity_verification.json" if completed_only else "final_identity_verification.json")
    write(output, {"status": "passed", "scope": "completed_cases_only" if completed_only else "complete_study",
                   "verified_at_utc": datetime.now(timezone.utc).isoformat(), "verifier_sha256": digest(__file__),
                   "study_hash": spec["study_hash"], "cases": verified})
    print("Identity verification passed", len(verified), "cases", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--completed-only", action="store_true")
    args = parser.parse_args()
    verify(args.study, args.completed_only)


if __name__ == "__main__":
    main()
