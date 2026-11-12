"""Describe every accepted study neighbour without changing the frozen decision."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def describe(root):
    acceptance = read(root / "final_identity_verification.json")
    if acceptance.get("status") != "passed" or acceptance.get("scope") != "complete_study":
        raise ValueError("Complete independent acceptance is required")
    native = read(root / "native_receipt_verification.json")
    if (native.get("status") != "passed"
            or native["structural_acceptance_sha256"] != sha(root / "final_identity_verification.json")
            or len(native["cases"]) != len(acceptance["cases"])):
        raise ValueError("Complete native receipt verification is required")
    artifact_manifest = read(root / "study_artifacts.json")
    pipeline = read(root / "analysis_pipeline_v6_manifest.json")
    harness = root / pipeline["harness_directory"]
    for name, expected in pipeline["files"].items():
        if sha(harness / name) != expected:
            raise ValueError("Frozen scientific helper changed")
    sys.path.insert(0, str(harness))
    import olr_rule_study as frozen_study
    # The frozen observer supports both repository and standalone imports.
    # Bind its repository spelling to the same verified frozen dependency.
    sys.modules["scripts.olr_rule_study"] = frozen_study
    from olr_rule_study import load_study, stability_recipes
    from olr_rule_analysis import load_case, candidate_difference, trade_difference, bootstrap_difference
    dependency = load_case.__globals__["signature"]
    if sha(Path(dependency.__code__.co_filename)) != pipeline["files"]["olr_rule_study.py"]:
        raise ValueError("Analysis imported a study dependency outside the frozen implementation")
    spec = load_study(root / "study.json")
    baseline = load_case(spec, "control", "nominal")
    results = {}
    for label in stability_recipes(spec["cases"]["control"]["mutations"]):
        name = "stability__control__" + label.replace("+", "plus").replace("-", "minus")
        current = load_case(spec, name, "nominal")
        receipt_name = "rejected.json" if current.get("qualification_rejected") else "completed.json"
        receipt = root / "cases" / name / "nominal" / receipt_name
        relative = receipt.relative_to(root).as_posix()
        if sha(receipt) != artifact_manifest["files"][relative]:
            raise ValueError("Accepted neighbour receipt changed")
        if current.get("qualification_rejected"):
            results[label] = {"status": "unqualified_native_result", "performance": "excluded",
                              "rejection_reasons": current["metrics"]["rejection_reasons"]}
            continue
        results[label] = {
            "status": "qualified_native_result", "receipt_sha256": sha(receipt),
            "candidate_changes": candidate_difference(current["candidates"], baseline["candidates"]),
            "trade_changes": trade_difference(current["trades"], baseline["trades"]),
            "trade_statistics": current["trade_statistics"],
            "paired_development_uncertainty": bootstrap_difference(current["daily"], baseline["daily"]),
        }
    output = root / "delivery"
    output.mkdir(exist_ok=True)
    result = {
        "status": "complete", "study_hash": spec["study_hash"],
        "scope": "Post-acceptance descriptive attribution of all registered control neighbours. No selection or qualification changes; no independent OOS claim.",
        "baseline_trade_statistics": baseline["trade_statistics"], "neighbours": results,
        "acceptance_sha256": sha(root / "final_identity_verification.json"),
        "native_receipt_verification_sha256": sha(root / "native_receipt_verification.json"),
        "artifact_manifest_sha256": sha(root / "study_artifacts.json"),
        "observer_sha256": sha(Path(__file__)), "scientific_helpers_sha256": pipeline["files"],
    }
    target = output / "sensitivity_attribution.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Described every registered control neighbour:", len(results), "saved", target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    describe(parser.parse_args().study_root)
