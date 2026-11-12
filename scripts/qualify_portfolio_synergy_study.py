"""Qualify the registered screening winner without modifying frozen replay code."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb.executed_value import digest, write_json


SCENARIOS = {
    "baseline": ("baseline", False),
    "combined": ("combined", False),
    "participation": ("participation", False),
    "cost": ("cost", False),
    "combined_no_upward_compounding": ("combined", True),
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_case(output, study_path, recipe, scenario, capped):
    study = read(study_path)
    result, audit, request, completion = [read(output / (name + ".json"))
        for name in ("result", "execution_audit", "request", "completion")]
    arguments = request["arguments"]
    audit_identity = audit["audit_receipt"]["identity"]
    if (not completion.get("complete")
            or completion["result_sha256"] != digest(output / "result.json")
            or completion["audit_sha256"] != digest(output / "execution_audit.json")
            or request["study_hash"] != digest(study_path)
            or result["source_generation_hash"] != study["source_generation_hash"]
            or completion["source_generation_hash"] != study["source_generation_hash"]
            or audit["audit_receipt"]["source_generation_hash"] != study["source_generation_hash"]
            or result["recipe"] != recipe or result["scenario"] != scenario
            or arguments["recipe"] != recipe or arguments["scenario"] != scenario
            or request["recipe"] != study["candidates"][recipe]
            or audit_identity["study_hash"] != digest(study_path)
            or audit_identity["recipe"] != study["candidates"][recipe]
            or audit_identity["scenario"] != scenario
            or audit_identity["no_upward_compounding"] != capped
            or audit_identity["absolute_caps"] or audit_identity["standalone"]
            or not arguments["full_audit"]
            or arguments["no_upward_compounding"] != capped
            or arguments["absolute_caps"] or arguments["standalone"] or arguments["canonical_costs"]):
        raise ValueError("Qualification identity mismatch: " + str(output))
    if (not result["execution_validity_pass"] or result["audit_status"] != "passed"
            or not audit["audit_pass"] or audit["audit_status"] != "passed"
            or result["intraday_residual_sessions"] or result["terminal_inventory"]
            or result["terminal_pending"]):
        raise ValueError("Qualification failed: " + str(output))
    if scenario in ("baseline", "combined") and not capped:
        screening = read(study_path.parent / "cases" / recipe / scenario / "result.json")
        for field in ("return_pct", "drawdown_pct", "trades", "active_days", "trades_per_21_sessions"):
            if abs(result[field] - screening[field]) > 1e-9:
                raise ValueError("Screening economics changed: " + field)
    return {"output": str(output), "result_sha256": completion["result_sha256"],
        "audit_sha256": completion["audit_sha256"], "audit_status": "passed",
        **{key: result[key] for key in ("return_pct", "drawdown_pct", "trades", "active_days",
            "trades_per_21_sessions", "strategy_counts", "unique_blocked_orders", "unique_resized_orders")}}


def qualify(study_path, workers=3):
    study_path = Path(study_path).resolve()
    study = read(study_path)
    selection_path = study_path.parent / "screening_selection.json"
    selection = read(selection_path)
    recipe = selection["recipe"]
    if recipe not in study["candidates"] or not selection["metrics"]["physical_valid"]:
        raise ValueError("A valid registered screening selection is required")
    root = study_path.parent / "qualified" / recipe
    root.mkdir(parents=True, exist_ok=True)
    registration = {"study_hash": digest(study_path), "selection_hash": digest(selection_path),
        "recipe": recipe, "scenarios": SCENARIOS, "runner_hash": digest(Path(__file__)),
        "scope": "reused_development; research_only; independent_validation_required"}
    registration_path = root / "registration.json"
    if registration_path.exists():
        if read(registration_path) != json.loads(json.dumps(registration)):
            raise ValueError("Changed qualification registration; preserve this run and use a new study")
    else:
        write_json(registration_path, registration)

    def run(name, scenario, capped):
        output = root / name
        if not (output / "completion.json").exists():
            if output.exists():
                raise FileExistsError("Preserving incomplete qualification: " + str(output))
            command = [sys.executable, "-B", "-u", str(Path(study["code"]) / "scripts/portfolio_synergy_study.py"),
                "replay", "--study", str(study_path), "--recipe", recipe,
                "--scenario", scenario, "--output", str(output), "--full-audit"]
            if capped:
                command.append("--no-upward-compounding")
            print(json.dumps({"qualification_started": name, "recipe": recipe}), flush=True)
            with (root / (name + ".log")).open("x", encoding="utf-8") as stream:
                process = subprocess.run(command, cwd=study["code"], stdout=stream, stderr=subprocess.STDOUT,
                    env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})
            write_json(root / (name + ".receipt.json"), {"command": command, "returncode": process.returncode})
            if process.returncode:
                raise RuntimeError("Qualification process failed: " + str(root / (name + ".log")))
        result = validate_case(output, study_path, recipe, scenario, capped)
        print(json.dumps({"qualification_completed": name, **result}), flush=True)
        return result

    results, errors = {}, {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(run, name, *case): name for name, case in SCENARIOS.items()}
        for future in as_completed(pending):
            name = pending[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"
                print(json.dumps({"qualification_failed": name, "error": errors[name]}), flush=True)
    receipt = {**registration, "status": "failed" if errors else "passed", "cases": results, "errors": errors}
    write_json(root / "qualification.json", receipt)
    if errors:
        raise RuntimeError("Qualification failed; inspect " + str(root / "qualification.json"))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    qualify(args.study, args.workers)
