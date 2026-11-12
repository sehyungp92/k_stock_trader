"""Finish the frozen OLR study: qualify one finalist and preserve all evidence."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import median
import subprocess
import sys
import time

try:
    from scripts.olr_rule_study import digest, load_study, read, signature, stability_recipes, write
    from scripts.olr_rule_analysis import analyse_core, load_case, prospective_contract
except ModuleNotFoundError:
    from olr_rule_study import digest, load_study, read, signature, stability_recipes, write
    from olr_rule_analysis import analyse_core, load_case, prospective_contract


def qualification_decision(challenger, stress, control_stress, neighbours, parent_metrics, spec):
    """Failure returns the complete baseline recipe, never a partial rollback."""
    reasons = []
    if challenger:
        policy = spec["stress_policy"]
        for mode in policy["scenarios"]:
            m, b = stress[mode], control_stress[mode]
            if m.get("audit_pass") is not True:
                reasons.append(mode + ":audit")
                continue
            if m["official_mtm_net_return_pct"] < policy["minimum_return_ratio_vs_control"] * b["official_mtm_net_return_pct"]:
                reasons.append(mode + ":return")
            if m["total_trades"] < policy["minimum_trade_count_ratio_vs_control"] * b["total_trades"]:
                reasons.append(mode + ":trade_count")
            if m["max_drawdown_pct"] > policy["maximum_drawdown"]:
                reasons.append(mode + ":drawdown")
            if m["profit_factor"] < policy["minimum_profit_factor"]:
                reasons.append(mode + ":profit_factor")
        if neighbours:
            returns = [m["official_mtm_net_return_pct"] for m in neighbours.values() if m.get("audit_pass") is True]
            parent = parent_metrics["official_mtm_net_return_pct"]
            if returns and (median(returns) < .9 * parent or min(returns) < .6 * parent):
                reasons.append("local_stability:return_fragility")
            if any(m.get("audit_pass") is not True or m["max_drawdown_pct"] > .15 for m in neighbours.values()):
                reasons.append("local_stability:drawdown_or_audit")
    selected = challenger if challenger and not reasons else "control"
    return {"selected": selected, "provisional_challenger": challenger, "rejection_reasons": reasons,
            "mutations": copy.deepcopy(spec["cases"][selected]["mutations"]),
            "status": "challenger_development_qualified_independent_validation_pending" if selected != "control"
                      else "control_retained_no_qualified_challenger_independent_validation_pending",
            "challenger_development_qualified": bool(challenger and not reasons),
            "independent_oos_pass": False, "live_promotion": False,
            "scope": "Training selection and registered qualification only; consumed regression cannot alter this decision"}


def run_job(spec, study, name, mode, mutations=None):
    out = Path(spec["output"]) / "cases" / name / mode
    complete = out / "completed.json"
    if complete.exists() or (out / "rejected.json").exists():
        return load_case(spec, name, mode)["metrics"]
    args = [sys.executable, "-B", "-u", spec.get("resolution_worker", spec["harness"]), "worker", "--study", str(study), "--case", name, "--mode", mode]
    if mutations is not None:
        mutation_path = Path(spec["output"]) / "registered_neighbours" / (name + ".json")
        if mutation_path.exists() and read(mutation_path) != mutations:
            raise ValueError("Neighbour definition changed")
        write(mutation_path, mutations)
        args += ["--mutation-file", str(mutation_path)]
    out.mkdir(parents=True, exist_ok=True)
    print("QUALIFICATION START", name, mode, flush=True)
    with (out / "worker.log").open("w", encoding="utf-8") as stream:
        subprocess.run(args, cwd=spec["engine_root"], stdout=stream, stderr=subprocess.STDOUT, check=True)
    m = load_case(spec, name, mode)["metrics"]
    print("QUALIFICATION DONE", name, mode, "rejected" if m.get("qualification_rejected") else (m["official_mtm_net_return_pct"], m["total_trades"]), flush=True)
    return m


def published_control_stress(spec):
    root = Path(spec["round_dir"])
    artifacts = read(root / "artifact_manifest.json")
    if artifacts["source_generation_hash"] != spec["source_generation_hash"]:
        raise ValueError("Published baseline engine differs")
    source = read(spec["input_manifest"])
    if artifacts["input_manifest_hash"] != source["manifest_hash"]:
        raise ValueError("Published baseline inputs differ")
    result, provenance = {}, {}
    for mode in spec["stress_policy"]["scenarios"]:
        relative = "diagnostics/" + mode + "/metrics.json"
        path = root / relative
        if digest(path) != artifacts["files"][relative]:
            raise ValueError("Published stress bytes changed")
        m = read(path)
        if m.get("audit_pass") is not True or m["audit_receipt"]["source_generation_hash"] != spec["source_generation_hash"]:
            raise ValueError("Unqualified published stress")
        result[mode] = m
        provenance[mode] = {"path": str(path), "sha256": digest(path), "source": "previously published same-source same-input qualified control"}
    write(Path(spec["output"]) / "control_stress_provenance.json", provenance)
    return result


def render_report(spec, decision, stability, stresses):
    root = Path(spec["output"])
    core = read(root / "analysis/core_results.json")
    lines = ["OLR frozen rule-ablation study", "", "Selected research recipe: " + decision["selected"],
             "Status: " + decision["status"], "Independent OOS pass: false. Live promotion: false.", "",
             "Core full-account results; return and drawdown in percent:",
             "case / sizing                 return       DD    closed trades    net R/trade"]
    for name, modes in sorted(core["cases"].items()):
        for mode, values in sorted(modes.items()):
            if values.get("status") == "unqualified_native_result":
                lines.append(name + "/" + mode + ": REJECTED; " + ", ".join(values["rejection_reasons"]) + "; economics excluded")
                continue
            lines.append(f"{name}/{mode}: {values['return_pct']:.6f}, {values['drawdown_pct']:.6f}, {values['closed_trades']}, {values['mean_net_r']:.6f}")
    lines += ["", "Selection failures and trade-offs:"]
    for name, result in core["selection"]["assessment"].items():
        lines.append(name + ": " + (", ".join(result["reasons"]) or "eligible"))
    lines += ["", "Qualification rejection reasons: " + (", ".join(decision["rejection_reasons"]) or "none"), "",
              "Local sensitivity: neighbours are diagnostic and cannot become new finalists."]
    for parent, neighbours in stability.items():
        for name, m in neighbours.items():
            if m.get("qualification_rejected"):
                lines.append(parent + "/" + name + ": REJECTED; " + ", ".join(m["rejection_reasons"]))
                continue
            lines.append(f"{parent}/{name}: return {100*m['official_mtm_net_return_pct']:.6f}%; DD {100*m['max_drawdown_pct']:.6f}%; trades {m['total_trades']:.0f}")
    lines += ["", "Capacity and cost qualification:"]
    for name, modes in stresses.items():
        for mode, m in modes.items():
            if m.get("qualification_rejected"):
                lines.append(name + "/" + mode + ": REJECTED; " + ", ".join(m["rejection_reasons"]))
                continue
            lines.append(f"{name}/{mode}: return {100*m['official_mtm_net_return_pct']:.6f}%; DD {100*m['max_drawdown_pct']:.6f}%; trades {m['total_trades']:.0f}")
    lines += ["", "Interpretation:",
              "All core cases rebuild selection through the frozen native engine; no trade-deletion counterfactuals.",
              "Control must reproduce the published round before challengers can be interpreted.",
              "The core selector requires return and closed-trade count no lower than control in both normal and capped sizing, preserves at least 95% of mean net R, and enforces risk and intervention support.",
              "Added trade counts are separated from changed sizing/profits on common trades. Those descendants are not independent opportunities.",
              "Paired time-block intervals are conditional development diagnostics, not selection-adjusted confidence or independent alpha tests.",
              "Regression remains consumed; its results cannot feed back into selection. The independent evaluation contract awaits future data and paper parity.",
              "Static universe/survivorship, capacity assumptions and researcher design hindsight remain limitations.",
              "Canonical round 1 remains the published baseline while independent validation is pending.",
              "", "Reproduction:",
              f"python -B {spec['harness']} core --study {root / 'study.json'}",
              "Qualification, decisions, source identities and file hashes are saved beside this report."]
    (root / "OLR_rule_study_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finish(study, wait, pipeline_manifest=None):
    spec = load_study(study)
    root = Path(spec["output"])
    pipeline = read(pipeline_manifest or root / "analysis_pipeline_manifest.json")
    pipeline_root = root / pipeline.get("harness_directory", "harness")
    if pipeline.get("resolution_worker"):
        spec["resolution_worker"] = str(pipeline_root / pipeline["resolution_worker"])
    for filename, expected_hash in pipeline["files"].items():
        if digest(pipeline_root / filename) != expected_hash:
            raise ValueError("Frozen analysis pipeline changed")
    expected = [root / "cases" / n / mode for n in spec["cases"] for mode in ("nominal", "sizing_cap")]
    resolved = lambda path: any((path / label).exists() for label in ("completed.json", "rejected.json"))
    previous = None
    while not all(resolved(p) for p in expected):
        if not wait:
            raise ValueError("Core replay set is incomplete")
        done = sum(resolved(p) for p in expected)
        if done != previous:
            print("Core progress", done, "/", len(expected), flush=True)
            previous = done
        error_log = root / pipeline.get("core_error_log", "pipeline.err")
        if error_log.exists() and "Traceback" in error_log.read_text(encoding="utf-8"):
            raise RuntimeError("Core supervisor failed; preserve evidence and repair before qualification")
        time.sleep(15)
    analyse_core(spec)
    frozen = read(root / "frozen_finalist.json")
    challenger = frozen["selected_challenger"]
    parents = ["control"] + ([challenger] if challenger else [])
    stability = {}
    for parent in parents:
        stability[parent] = {}
        for label, mutations in stability_recipes(spec["cases"][parent]["mutations"]).items():
            name = "stability__" + parent + "__" + label.replace("+", "plus").replace("-", "minus")
            stability[parent][label] = run_job(spec, study, name, "nominal", mutations)
    stresses = {"control": published_control_stress(spec)}
    if challenger:
        stresses[challenger] = {mode: run_job(spec, study, challenger, mode) for mode in spec["stress_policy"]["scenarios"]}
    parent_metrics = load_case(spec, challenger or "control", "nominal")["metrics"]
    decision = qualification_decision(challenger, stresses.get(challenger, {}), stresses["control"],
                                      stability.get(challenger, {}), parent_metrics, spec)
    decision.update(study_hash=spec["study_hash"], qualifier_sha256=digest(__file__),
                    qualified_at_utc=datetime.now(timezone.utc).isoformat())
    write(root / "qualification_decision.json", decision)
    write(root / "qualification_results.json", {"stability": stability, "stress": stresses})
    selected = decision["selected"]
    write(root / "selected_recipe.json", {"selected": selected, "mutations": decision["mutations"],
                                         "mutations_hash": signature(decision["mutations"]), "study_hash": spec["study_hash"]})
    import yaml
    config = copy.deepcopy(spec["config"])
    config["initial_mutations"] = decision["mutations"]
    (root / "selected_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if yaml.safe_load((root / "selected_config.yaml").read_text())["initial_mutations"] != spec["cases"][selected]["mutations"]:
        raise ValueError("Saved selected config differs from qualified recipe")
    prospective_contract(spec, selected, root / "prospective_validation_contract.json")
    # The decision is frozen before this consumed regression is read or run.
    if selected != "control":
        run_job(spec, study, selected, "regression")
    alpha_observer = pipeline_root / "olr_rule_alpha_attribution.py"
    with (root / "alpha_attribution.log").open("w", encoding="utf-8") as stream:
        subprocess.run([sys.executable, "-B", "-u", str(alpha_observer), "--study", str(study),
                        "--cases", *parents], cwd=spec["engine_root"], stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    render_report(spec, decision, stability, stresses)
    for filename, expected_hash in pipeline["files"].items():
        if digest(pipeline_root / filename) != expected_hash:
            raise ValueError("Analysis pipeline changed during qualification")
    artifact_hashes = {p.relative_to(root).as_posix(): digest(p) for p in root.rglob("*")
                       if p.is_file() and p.suffix not in {".log", ".err", ".pid"}
                       and "temp" not in p.parts and "pytest" not in str(p)
                       and p.name not in {"study_artifacts.json", "task_checkpoint.json"}}
    write(root / "study_artifacts.json", {"study_hash": spec["study_hash"], "files": artifact_hashes})
    print("STUDY COMPLETE", selected, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--wait-core", action="store_true")
    parser.add_argument("--pipeline-manifest", type=Path)
    args = parser.parse_args()
    finish(args.study, args.wait_core, args.pipeline_manifest)


if __name__ == "__main__":
    main()
