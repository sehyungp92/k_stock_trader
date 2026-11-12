"""Earlier-OOF full-account calibration of executed-value admission thresholds.

The inner simulations use the normal replay CLI. At each outer fit, only their
completed historical equity and liquidation observations may select a threshold.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb import executed_value as ev


def select_portfolio_threshold(cases, first, *, minimum_sessions=20):
    trials = [{"threshold": "cash", "net_pnl_krw": 0.0, "trades": 0, "feasible": True}]
    for threshold, case in cases:
        past = [row for row in case["daily"] if row["day"] < first]
        invalid = [day for day, quantity in case["residual_sessions"].items() if day < first and quantity]
        physical_invalid = [day for day in case.get("invalid_execution_days", []) if day < first]
        feasible = len(past) >= minimum_sessions and not invalid and not physical_invalid and not past[-1]["inventory"] and not past[-1]["pending"]
        trades = sum(t["entry_fill_time"][:10] < first and t["exit_fill_time"][:10] < first for t in case["trades"])
        trials.append({"threshold": threshold, "net_pnl_krw": past[-1]["equity"] - ev.REFERENCE_EQUITY if past else 0,
            "trades": trades, "feasible": bool(feasible), "sessions": len(past),
            "observed_through": past[-1]["day"] if past else None, "deadline_failures": invalid,
            "invalid_execution_days": physical_invalid})
    best = max((t for t in trials if t["feasible"]),
               key=lambda t: (t["net_pnl_krw"], t["trades"] if t["net_pnl_krw"] > 0 else -t["trades"]))
    return best["threshold"], {"selected": best, "trials": trials,
        "method": "earlier_oof_full_account_executed_net_pnl_with_deadline_constraints"}


def verify_files(directory):
    m = json.loads((directory / "policies_manifest.json").read_text())
    for name, expected in m["files"].items():
        if ev.digest(directory / name) != expected:
            raise ValueError("Changed upstream policy input")
    return m


def prepare_inner(plan, prepared, policies, output):
    from scripts.kalcb_executed_value_study import emit
    original = verify_files(policies)
    observations = ev.read_jsonl(policies / "inner_calibration.jsonl")
    ids = {r["opportunity_id"] for r in observations}
    rows = {}
    with (prepared / "opportunities.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["opportunity_id"] in ids:
                rows[row["opportunity_id"]] = row
    if set(rows) != ids:
        raise ValueError("Missing inner opportunity")
    grouped = {}
    for observation in observations:
        grouped.setdefault(observation["trade_date"], []).append(observation)
    provenance = json.loads((prepared / "prepared.json").read_text())
    sessions = [d for d in provenance["sessions"] if min(grouped) <= d <= max(grouped)]
    thresholds = plan["comparison_contract"]["thresholds_equity_bps"]
    names = {}
    for i, threshold in enumerate(thresholds):
        name = f"inner_threshold_{i}"
        names[name] = threshold
        snapshots = []
        with (output / (name + ".admission.jsonl")).open("x", encoding="utf-8") as stream:
            for day in sessions:
                observations_for_day = grouped.get(day, [])
                selected = [(rows[r["opportunity_id"]], r["score"]) for r in observations_for_day
                            if threshold is None or r["score"] > threshold]
                snapshots.append(ev.snapshot_for(day, selected, source=provenance["opportunities_sha256"],
                    forecast={"policy": name, "threshold_equity_bps": threshold, "inner_oof": True}).to_json_dict())
                selected_ids = {r["opportunity_id"] for r, _ in selected}
                for r in observations_for_day:
                    emit(stream, {"opportunity_id": r["opportunity_id"], "trade_date": day, "score": r["score"],
                        "reason": "admitted" if r["opportunity_id"] in selected_ids else "net_value_threshold"})
        ev.write_json(output / (name + ".snapshots.json"), snapshots)
    chronology = json.loads((policies / "chronology.json").read_text())
    ev.write_json(output / "chronology.json", {"folds": [p for p in chronology["folds"] if p["purpose"] == "inner_admission_calibration"],
        "calibration": [], "scope": "earlier OOF predictions; fixed threshold portfolio trajectories"})
    ev.write_json(output / "thresholds.json", names)
    ev.write_json(output / "policies_manifest.json", {**original, "evaluation_from": sessions[0], "evaluation_through": sessions[-1],
        "policies": list(names), "files": {p.name: ev.digest(p) for p in output.iterdir() if p.is_file()},
        "parent_policy_manifest_sha256": ev.digest(policies / "policies_manifest.json")})


def calibrate(plan, policies, inner, replays, output):
    from scripts.kalcb_executed_value_study import emit
    original = verify_files(policies)
    verify_files(inner)
    thresholds = json.loads((inner / "thresholds.json").read_text())
    cases = []
    for name, threshold in thresholds.items():
        directory = replays / name
        result = json.loads((directory / "result.json").read_text())
        request = json.loads((directory / "request.json").read_text())
        completion = json.loads((directory / "completion.json").read_text())
        audit = json.loads((directory / "execution_audit.json").read_text())
        errors = set(audit["audit_receipt"]["errors"])
        if errors - {"shared_bar_volume_exceeded"} or result["policy"] != name or result["scenario"] != "baseline":
            raise ValueError("Invalid inner execution replay")
        failures = ev.physical_fill_failures(json.loads((directory / "fills.json").read_text()))
        if bool(failures) != ("shared_bar_volume_exceeded" in errors):
            raise ValueError("Independent physical-volume check disagrees with auditor")
        if not completion.get("complete") or request["source_generation_hash"] != result["source_generation_hash"] or completion["source_generation_hash"] != result["source_generation_hash"]:
            raise ValueError("Inner source generation changed")
        case = {"daily": json.loads((directory / "daily.json").read_text()),
                "trades": json.loads((directory / "trades.json").read_text()),
                "residual_sessions": result["residual_sessions"],
                "invalid_execution_days": sorted({f["timestamp"][:10] for f in failures})}
        cases.append((threshold, case))
        for artifact in ("result.json", "request.json", "daily.json", "trades.json", "execution_audit.json"):
            shutil.copy2(directory / artifact, output / (name + "." + artifact))
    for name in original["files"]:
        shutil.copy2(policies / name, output / name)
    shutil.copy2(inner / "policies_manifest.json", output / "inner_policies_manifest.json")
    # Preserve calibration source as a bound artifact as well as executing source.
    shutil.copy2(__file__, output / "portfolio_calibration_source.py")
    chronology = json.loads((policies / "chronology.json").read_text())
    folds = [r for r in chronology["folds"] if r["purpose"] == "outer_forecast" and r["target"] == "net"]
    thresholds_by_start, calibration = {}, []
    for fold in folds:
        first = fold["valid_from"]
        threshold, detail = select_portfolio_threshold(cases, first)
        thresholds_by_start[first] = threshold
        through = max((t["observed_through"] for t in detail["trials"] if t.get("observed_through")), default=None)
        calibration.append({"policy": "net_portfolio_calibrated", "valid_from": first,
            "valid_through": fold["valid_through"], "threshold": threshold,
            "latest_label_available_at": through + "T23:59:59+09:00" if through else None, **detail})
    snapshots = json.loads((policies / "net_before.snapshots.json").read_text())
    decisions = ev.read_jsonl(policies / "net_before.admission.jsonl")
    selected_by_day = {}
    for row in decisions:
        threshold = thresholds_by_start[row["valid_from"]]
        admitted = row["reason"] == "admitted"
        row["policy"] = "net_portfolio_calibrated"
        row["threshold_equity_bps"] = threshold
        if threshold == "cash":
            row["reason"] = "cash_policy"
        elif admitted and threshold is not None and row["score"] <= threshold:
            row["reason"] = "net_value_threshold"
        if row["reason"] == "admitted":
            selected_by_day.setdefault(row["trade_date"], set()).add(row["opportunity_id"])
    from strategy_kalcb.models import KALCBDailySnapshot
    converted = []
    for raw in snapshots:
        day = raw["trade_date"]
        fold = next(f for f in folds if f["valid_from"] <= day <= f["valid_through"])
        selected = []
        for candidate in raw["candidates"]:
            if day + "/" + candidate["symbol"] in selected_by_day.get(day, set()):
                selected.append(({"candidate": candidate,
                    "feature_available_at": candidate["metadata"].get("feature_available_at")}, candidate["selection_score"]))
        converted.append(ev.snapshot_for(day, selected, source=raw["source_fingerprint"], forecast={
            "policy": "net_portfolio_calibrated", "valid_from": fold["valid_from"],
            "threshold_equity_bps": thresholds_by_start[fold["valid_from"]]}).to_json_dict())
    name = "net_portfolio_calibrated"
    ev.write_json(output / (name + ".snapshots.json"), converted)
    with (output / (name + ".admission.jsonl")).open("x", encoding="utf-8") as stream:
        for row in decisions:
            emit(stream, row)
    chronology["calibration"].extend(calibration)
    ev.write_json(output / "chronology.json", chronology)
    ev.write_json(output / "policies_manifest.json", {**original, "policies": original["policies"] + [name],
        "files": {p.name: ev.digest(p) for p in output.iterdir() if p.is_file()},
        "counts": {**original["counts"], name: sum(len(s["candidates"]) for s in converted)},
        "parent_policy_manifest_sha256": ev.digest(policies / "policies_manifest.json")})
    print(json.dumps({"portfolio_calibration": calibration}), flush=True)


def main():
    from scripts.kalcb_executed_value_study import load_plan
    from backtests.auto.shared.cache_keys import source_generation_hash
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare-inner", "calibrate"))
    for name in ("plan", "policies", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("prepared", "inner", "replays"):
        p.add_argument("--" + name, type=Path)
    a = p.parse_args()
    plan, source = load_plan(a.plan), source_generation_hash()
    a.output.mkdir(parents=True, exist_ok=False)
    if a.command == "prepare-inner":
        prepare_inner(plan, a.prepared, a.policies, a.output)
    else:
        calibrate(plan, a.policies, a.inner, a.replays, a.output)
    if source_generation_hash() != source:
        raise RuntimeError("Source changed during portfolio calibration")
    ev.write_json(a.output / "completion.json", {"complete": True, "source_generation_hash": source})


if __name__ == "__main__":
    main()
