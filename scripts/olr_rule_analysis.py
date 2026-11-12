"""Analyse registered OLR rule replays without selecting on regression data."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import gzip
import json
from pathlib import Path
import random
from statistics import mean

try:
    from scripts.olr_rule_study import digest, forward_validation_policy, load_study, read, signature, stability_recipes, write
except ModuleNotFoundError:
    from olr_rule_study import digest, forward_validation_policy, load_study, read, signature, stability_recipes, write


def rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def trade_key(trade):
    return trade["symbol"], trade["entry_fill_time"]


def trade_statistics(trades):
    net_r = []
    returns = []
    for trade in trades:
        risk = float(trade["route_metadata"].get("risk_per_share", 0)) * trade["qty"]
        if risk > 0:
            net_r.append(trade["net_pnl"] / risk)
        notional = trade["entry_price"] * trade["qty"]
        if notional > 0:
            returns.append(trade["net_pnl"] / notional)
    groups = defaultdict(list)
    for trade in trades:
        for dimension, label in (("month", trade["entry_fill_time"][:7]), ("symbol", trade["symbol"]),
                                 ("sector", trade["route_metadata"].get("sector", "unknown")),
                                 ("band", trade["route_metadata"].get("afternoon_score_band_rule", "no_band"))):
            groups[(dimension, label)].append(trade)
    cohorts = [{"dimension": d, "label": label, "trades": len(rr),
                "net_pnl_krw": sum(r["net_pnl"] for r in rr), "wins": sum(r["net_pnl"] > 0 for r in rr)}
               for (d, label), rr in sorted(groups.items())]
    return {"mean_net_r": mean(net_r) if net_r else None, "net_r_count": len(net_r),
            "mean_trade_net_return": mean(returns) if returns else None,
            "entry_notional_krw": sum(t["entry_price"] * t["qty"] for t in trades),
            "realized_net_pnl_krw": sum(t["net_pnl"] for t in trades),
            "distinct_entry_dates": len({t["entry_fill_time"][:10] for t in trades}), "cohorts": cohorts}


def bootstrap_difference(left, right, *, samples=2000, block=10, seed=61006):
    """Paired circular time-block resampling; conditional, not selection-adjusted."""
    if [r["date"] for r in left] != [r["date"] for r in right]:
        raise ValueError("Daily return calendars differ")
    delta = [a["net_return"] - b["net_return"] for a, b in zip(left, right)]
    if not delta:
        raise ValueError("No paired observations")
    n = len(delta)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        sample = []
        while len(sample) < n:
            start = rng.randrange(n)
            sample.extend(delta[(start + i) % n] for i in range(min(block, n - len(sample))))
        estimates.append(mean(sample))
    estimates.sort()
    return {"sessions": n, "block_sessions": block, "resamples": samples, "seed": seed,
            "mean_daily_return_difference": mean(delta),
            "conditional_95pct_interval": [estimates[int(samples * .025)], estimates[min(samples - 1, int(samples * .975))]],
            "bootstrap_positive_fraction": sum(v > 0 for v in estimates) / samples,
            "interpretation": "Conditional uncertainty on development data; not an independent test, selection correction, or probability of true alpha"}


def trade_difference(left, right):
    a, b = {trade_key(r): r for r in left}, {trade_key(r): r for r in right}
    if len(a) != len(left) or len(b) != len(right):
        raise ValueError("Expected collapsed entry lifecycles")
    added, removed, common = a.keys() - b.keys(), b.keys() - a.keys(), a.keys() & b.keys()
    return {"added_trades": len(added), "removed_trades": len(removed), "common_trades": len(common),
            "added_net_pnl_krw": sum(a[k]["net_pnl"] for k in added),
            "removed_net_pnl_krw": sum(b[k]["net_pnl"] for k in removed),
            "common_net_pnl_change_krw": sum(a[k]["net_pnl"] - b[k]["net_pnl"] for k in common),
            "common_quantity_changes": sum(a[k]["qty"] != b[k]["qty"] for k in common),
            "added": [a[k] for k in sorted(added)], "removed": [b[k] for k in sorted(removed)]}


def candidate_difference(left, right):
    def index(items):
        return {(d["date"], c["symbol"]): c for d in items for c in d["selected"]}
    a, b = index(left), index(right)
    common = a.keys() & b.keys()
    changed = (a.keys() ^ b.keys()) | {k for k in common if a[k]["rank"] != b[k]["rank"]}
    return {"selected_pairs": len(a), "added_selected_pairs": len(a.keys() - b.keys()),
            "removed_selected_pairs": len(b.keys() - a.keys()),
            "common_rank_changes": sum(a[k]["rank"] != b[k]["rank"] for k in common),
            "common_score_changes": sum(abs(a[k]["score"] - b[k]["score"]) > 1e-9 for k in common),
            "changed_selected_issuer_days": len(changed),
            "changed_calendar_months": sorted({k[0][:7] for k in changed}),
            "rejection_reasons": dict(Counter(reason for d in left for reasons in d["rejected"].values() for reason in reasons))}


def load_case(spec, name, mode, *, verify=True):
    directory = Path(spec["output"]) / "cases" / name / mode
    rejected = (directory / "rejected.json").exists()
    if rejected and (directory / "completed.json").exists():
        raise ValueError("Case has conflicting accepted and rejected receipts")
    receipt = read(directory / ("rejected.json" if rejected else "completed.json"))
    if receipt["study_hash"] != spec["study_hash"]:
        raise ValueError("Unbound study result")
    job = read(directory / "job.json")
    if (receipt["job_hash"] != signature(job) or job["study_hash"] != spec["study_hash"]
            or job["case"] != name or job["mode"] != mode):
        raise ValueError("Case directory and request identity differ")
    if verify:
        for filename, expected in receipt["outputs"].items():
            if digest(directory / filename) != expected:
                raise ValueError("Changed result: " + str(directory / filename))
    metrics = read(directory / ("forensic_metrics.json" if rejected else "metrics.json"))
    if not rejected and (metrics.get("audit_pass") is not True or metrics.get("daily_equity_coverage", {}).get("status") != "complete"):
        raise ValueError("Unqualified case")
    request = read(directory / ("forensic_request.json" if rejected else "request.json"))
    if request["config"] != job["config"] or request["mutations"] != job["mutations"]:
        raise ValueError("Exported evaluation differs from registered job")
    if metrics["audit_receipt"]["source_generation_hash"] != spec["source_generation_hash"]:
        raise ValueError("Case execution source differs")
    if metrics["evaluation_mutations"] != signature(job["mutations"]):
        raise ValueError("Evaluated mutations differ")
    manifest = read(spec["input_manifest"])
    expected_request = signature({"config": job["config"], "mutations": job["mutations"],
                                  "input_generation": manifest["manifest_hash"]})
    if metrics["evaluation_request"] != expected_request or (not rejected and request["evaluation_request"] != expected_request):
        raise ValueError("Evaluation receipt request mismatch")
    if rejected:
        native = dict(metrics["audit_receipt"])
        receipt_hash = native.pop("receipt_hash")
        if (signature(native) != receipt_hash or native["identity"]["request"] != expected_request
                or native["identity"]["mutations"] != signature(job["mutations"])):
            raise ValueError("Rejected native audit identity differs")
        if metrics.get("audit_pass") is True and metrics.get("daily_equity_coverage", {}).get("status") == "complete":
            raise ValueError("A qualified result cannot be disguised as a rejected case")
        if name == "control" and mode in ("nominal", "sizing_cap"):
            raise ValueError("Required control is unqualified; study cannot select")
        reasons = list(native.get("errors", [])) + [key + ":" + value for key, value in native["coverage"].items()
                                                        if value not in {"pass", "not_applicable"}]
        return {"qualification_rejected": True, "metrics": {"audit_pass": False, "qualification_rejected": True,
                "rejection_reasons": reasons, "evaluation_request": expected_request}, "directory": str(directory)}
    trades = rows(directory / "trades.jsonl.gz")
    return {"metrics": metrics, "trade_statistics": trade_statistics(trades), "trades": trades,
            "daily": read(directory / "daily_returns.json"),
            "candidates": rows(directory / "candidate_selection.jsonl.gz"), "directory": str(directory)}


def select_finalist(cases, policy, support_policy=None):
    """Only the two development modes can affect selection, even if others exist."""
    baseline = cases["control"]
    eligible = []
    assessment = {}
    for name, candidate in sorted(cases.items()):
        if name == "control":
            continue
        rejected_modes = [mode for mode in ("nominal", "sizing_cap") if candidate[mode].get("qualification_rejected")]
        if rejected_modes:
            assessment[name] = {"eligible": False, "reasons": [mode + ":unqualified_native_result" for mode in rejected_modes],
                                "minimum_relative_wealth_gain": None, "minimum_added_trades": None}
            continue
        errors, wealth_gains, trade_gains = [], [], []
        strict_gain = False
        for mode in ("nominal", "sizing_cap"):
            current, ref = candidate[mode], baseline[mode]
            m, b = current["metrics"], ref["metrics"]
            ret, prior = m["official_mtm_net_return_pct"], b["official_mtm_net_return_pct"]
            if ret + 1e-10 < prior:
                errors.append(mode + ":lower_net_return")
            if m["total_trades"] < b["total_trades"]:
                errors.append(mode + ":lower_trade_frequency")
            if m["max_drawdown_pct"] > min(policy["maximum_drawdown"], b["max_drawdown_pct"] + policy["maximum_drawdown_increase"]):
                errors.append(mode + ":drawdown")
            if m["profit_factor"] < policy["minimum_profit_factor"]:
                errors.append(mode + ":profit_factor")
            quality, previous = current["trade_statistics"]["mean_net_r"], ref["trade_statistics"]["mean_net_r"]
            if quality is None or previous is None or quality < policy["minimum_mean_net_r_ratio"] * previous:
                errors.append(mode + ":net_trade_quality")
            if m.get("audit_pass") is not True:
                errors.append(mode + ":audit")
            wealth_gains.append((1 + ret) / (1 + prior) - 1)
            trade_gains.append(m["total_trades"] - b["total_trades"])
            strict_gain |= ret > prior + 1e-8 or m["total_trades"] > b["total_trades"]
        if not strict_gain:
            errors.append("no_strict_gain")
        if support_policy:
            support = candidate_difference(candidate["nominal"]["candidates"], baseline["nominal"]["candidates"])
            if support["changed_selected_issuer_days"] < support_policy["minimum_changed_selected_issuer_days"]:
                errors.append("insufficient_changed_issuer_days")
            if len(support["changed_calendar_months"]) < support_policy["minimum_changed_calendar_months"]:
                errors.append("insufficient_changed_months")
        assessment[name] = {"eligible": not errors, "reasons": errors,
                            "minimum_relative_wealth_gain": min(wealth_gains), "minimum_added_trades": min(trade_gains)}
        if not errors:
            eligible.append(name)
    eligible.sort(key=lambda n: (-assessment[n]["minimum_relative_wealth_gain"], -assessment[n]["minimum_added_trades"], n))
    return {"selected_challenger": eligible[0] if eligible else None, "fallback": "control", "assessment": assessment,
            "scope": "training nominal and capped sizing only; stress and independent evaluation still required"}


def analyse_core(spec, *, complete=True):
    root = Path(spec["output"])
    cases = {}
    for name in spec["cases"]:
        cases[name] = {}
        for mode in ("nominal", "sizing_cap"):
            if any((root / "cases" / name / mode / label).exists() for label in ("completed.json", "rejected.json")):
                cases[name][mode] = load_case(spec, name, mode)
            elif complete:
                raise ValueError("Core study incomplete: " + name + "/" + mode)
    summary, comparisons = {}, {}
    for name, modes in cases.items():
        summary[name] = {}
        for mode, case in modes.items():
            m = case["metrics"]
            if case.get("qualification_rejected"):
                summary[name][mode] = {"status": "unqualified_native_result", "rejection_reasons": m["rejection_reasons"],
                                       "performance": "not_reported_or_scored"}
                continue
            summary[name][mode] = {"return_pct": 100 * m["official_mtm_net_return_pct"],
                "drawdown_pct": 100 * m["max_drawdown_pct"], "closed_trades": int(m["total_trades"]),
                "trades_per_session": m["total_trades"] / len(case["daily"]),
                "profit_factor": m["profit_factor"], **case["trade_statistics"]}
            control = cases["control"].get(mode)
            if control and name != "control":
                first = next(({"date": a["date"], "candidate_equity": a["equity"], "control_equity": b["equity"]}
                              for a, b in zip(case["daily"], control["daily"]) if abs(a["equity"] - b["equity"]) > .01), None)
                comparison = {"trades": trade_difference(case["trades"], control["trades"]),
                              "candidates": candidate_difference(case["candidates"], control["candidates"]),
                              "first_session_equity_difference": first,
                              "paired_uncertainty": bootstrap_difference(case["daily"], control["daily"])}
                path = root / "analysis" / f"{name}_{mode}_versus_control.json"
                write(path, comparison)
                comparisons[name + "/" + mode] = str(path)
    result = {"study_hash": spec["study_hash"], "complete": complete, "cases": summary, "comparisons": comparisons}
    if complete:
        support = read(root / "support_policy.json")
        expected = support.pop("policy_hash")
        if signature(support) != expected or support["study_hash"] != spec["study_hash"]:
            raise ValueError("Support policy identity mismatch")
        first_completion = min(read(p)["completed_at_utc"] for p in (root / "cases").glob("*/*/completed.json"))
        if support["registered_at_utc"] >= first_completion:
            raise ValueError("Support criteria were not registered before outcomes")
        selection = select_finalist(cases, spec["selection_policy"], support)
        selection["support_policy_hash"] = expected
        frozen_path = root / "frozen_finalist.json"
        frozen = {**selection, "study_hash": spec["study_hash"], "analysis_sha256": digest(__file__)}
        if frozen_path.exists():
            previous = read(frozen_path)
            previous.pop("frozen_at_utc", None)
            if previous != frozen:
                raise ValueError("A different finalist has already been frozen")
        else:
            write(frozen_path, {**frozen, "frozen_at_utc": datetime.now(timezone.utc).isoformat()})
        result["selection"] = selection
    write(root / "analysis" / ("core_results.json" if complete else "partial_results.json"), result)
    print(json.dumps({"complete": complete, "cases": {
        n: {mode: {k: v for k, v in d.items() if k in {"return_pct", "drawdown_pct", "closed_trades", "mean_net_r"}}
            for mode, d in modes.items()} for n, modes in summary.items()},
        "challenger": result.get("selection", {}).get("selected_challenger")}), flush=True)


def prospective_contract(spec, selected, output):
    """Freeze a recipe for independent evidence without a future-session completion gate."""
    mutations = spec["cases"][selected]["mutations"]
    frozen_at = datetime.now(timezone.utc)
    # No market calendar inference: earliest eligible date is followed by the
    # qualified exchange calendar's first trading session on/after that date.
    earliest = (frozen_at + timedelta(hours=9)).date() + timedelta(days=1)
    payload = {"version": "olr-prospective-validation-v4", "study_hash": spec["study_hash"],
               "selected": selected, "mutations": mutations, "mutations_hash": signature(mutations),
               "frozen_at_utc": frozen_at.isoformat(), "earliest_eligible_kst_date": earliest.isoformat(),
               "start_rule": "First qualified KRX session on/after earliest eligible date",
               **forward_validation_policy(),
               "sample_policy": "Retain every eligible session, report sparse evidence as inconclusive, and disclose all evaluated windows; do not select a reporting endpoint for favourable returns.",
               "status": "protocol_registered_performance_not_evaluated",
               "use_for_selection": False,
               "decision": {"positive_net_expectancy_required": True, "maximum_drawdown": 0.15,
                            "minimum_profit_factor": 1.1,
                            "uncertainty": "Report paired time-block intervals and opportunity support; inconclusive evidence is not a pass"},
               "required_inputs": ["timestamped immutable causal candidate artifacts", "point-in-time universe and sector membership",
                                   "qualified market/reference/cost inputs", "native continuous account replay and complete daily equity"],
               "operational_readiness": "Paper orders, fills, rejects and account reconciliation are separate from offline performance evaluation.",
               "prohibitions": ["No outcomes from the reserved window used to modify this recipe",
                                "No deployment qualification from development-only profits",
                                "No automatic live trading activation"],
               "execution_model": spec["config"]["capacity_scenario"],
               "source_generation_hash": spec["source_generation_hash"]}
    payload["contract_hash"] = signature(payload)
    write(output, payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    analyse_core(load_study(args.study), complete=not args.partial)


if __name__ == "__main__":
    main()
