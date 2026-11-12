"""Read-only, fail-closed bridge from audited native replays to diagnostics.

This consumes complete source-engine replays, never reconstructs portfolio
execution from completed trades, and never invokes a replacement selector.
"""
from collections import Counter, defaultdict
from dataclasses import fields
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path

from backtests.auto.shared.cache_keys import stable_signature

SCENARIOS = ("baseline", "participation", "cost", "combined")
REPLAY_FILES = ("request.json", "completion.json", "result.json", "execution_audit.json",
                "trades.json", "fills.json", "daily.json", "equity.json", "decisions.jsonl.gz", "orders.jsonl.gz")
SOURCE_CHECKS = ("input_provenance", "source_completeness", "replay_completeness",
                 "dated_references", "price_volume_basis", "capacity")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message, tolerance=1e-5):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)) and
            abs(float(actual) - float(expected)) <= tolerance, message)


def freeze_profile(profile_path, destination, *, acceptance_checks=None, additional_evidence=()):
    """Content-bind the already-qualified profile before changing round paths."""
    profile_path = Path(profile_path).resolve()
    profile = read(profile_path)
    paths = [profile_path, Path(profile["configuration"]), Path(profile["audit_manifest"]),
             Path(profile["plan"]), Path(profile["all_trials"])]
    paths += [Path(p) for p in profile["research_protocols"]]
    policy = Path(profile["policy"]["directory"])
    paths += [p for p in policy.iterdir() if p.is_file()]
    for result in profile["result_paths"].values():
        paths += [Path(result).parent / name for name in REPLAY_FILES]
    for key in ("scope_preparation_parity", "forecast_reconstruction", "final_account_verification", "input_verification", "attribution"):
        paths.append(Path(profile[key]))
    paths += [Path(profile["regression_tests"]["log"])]
    paths += [Path(p) for p in (acceptance_checks or {}).values()]
    paths += [Path(p) for p in additional_evidence]
    payload = {"version": "kalcb-qualified-native-exports-v1", "profile": str(profile_path),
               "files": {str(p.resolve()): digest(p) for p in paths},
               "acceptance_checks": {k: str(Path(v).resolve()) for k, v in (acceptance_checks or {}).items()},
               "scope": "Historical research acceptance; no production or independent forward claim"}
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as out:
        json.dump({**payload, "manifest_hash": stable_signature(payload)}, out, indent=2)
    return target


def validate_case(directory, profile, scenario):
    p = Path(directory)
    result, audit, done, request = (read(p / name) for name in
                                   ("result.json", "execution_audit.json", "completion.json", "request.json"))
    require(done.get("complete") is True, "Incomplete native replay")
    require(result.get("audit_status") == audit.get("audit_status") == "passed" and
            result.get("execution_validity_pass") is True and audit.get("audit_pass") is True,
            "A failed or unverified source replay cannot be promoted")
    source = profile["execution_source"]["source_generation_hash"]
    require(all(x.get("source_generation_hash") == source for x in (result, done, request)), "Stale replay generation")
    receipt = dict(audit["audit_receipt"])
    receipt_hash = receipt.pop("receipt_hash", None)
    require(receipt_hash == stable_signature(receipt), "Changed execution audit receipt")
    require(receipt["source_generation_hash"] == source and not receipt["errors"], "Invalid execution receipt")
    require(all(receipt["coverage"].get(k) == "pass" for k in SOURCE_CHECKS), "Incomplete source qualification")
    require(not result["residual_sessions"] and not result["terminal_inventory"] and not result["terminal_pending"], "Unresolved native account")
    require(result["scenario"] == scenario and result["policy"] == profile["policy"]["name"], "Wrong scenario or selector")
    require(receipt["identity"]["candidate_policy_manifest_sha256"] == profile["policy"]["manifest_sha256"], "Wrong candidate evidence")
    require(receipt["identity"]["configuration"]["source_dependency_scope"] == {
        "version": "causal-timed-candidates-v1", "strategy": "kalcb",
        "feature_intraday_before": "09:45", "candidate_intraday_before": "15:20"}, "Shortened source audit scope")
    require(receipt["identity"]["input_manifest"]["manifest_hash"] == read(profile["audit_manifest"])["manifest_hash"], "Wrong immutable source bundle")
    trades, fills, daily, equity = (read(p / name) for name in ("trades.json", "fills.json", "daily.json", "equity.json"))
    initial = float(profile["initial_equity_krw"])
    cash = initial
    positions = defaultdict(int)
    ids = set()
    costs = receipt["qualification"]["capacity"]["policy"]["default_costs"]
    last = None
    for fill in fills:
        q, price, meta = int(fill["qty"]), float(fill["price"]), fill["metadata"]
        at = datetime.fromisoformat(fill["timestamp"])
        require(at.tzinfo is not None and (last is None or at >= last), "Nonchronological executions")
        last = at
        eid = meta["execution_id"]
        require(eid not in ids and q > 0 and price > 0, "Invalid or duplicate execution")
        ids.add(eid)
        buy = fill["side"] == "BUY"
        require(fill["side"] in ("BUY", "SELL"), "Unsupported execution side")
        fee = q * price * (costs["commission_bps"] + (0 if buy else costs["tax_bps_on_sell"])) / 10000
        close(meta["fee"], fee, "Incorrect per-fill costs")
        slip = costs["auction_slippage_bps"] if meta["execution_phase"] == "auction" else costs["slippage_bps"]
        close(price, meta["base_price"] * (1 + (1 if buy else -1) * slip / 10000), "Non-adverse slippage")
        bar = meta["execution_bar"]
        require(bar["low"] - 1e-7 <= meta["base_price"] <= bar["high"] + 1e-7 and bar["volume"] > 0, "Unattainable source price")
        available = meta.get("decision_available_at")
        if available:
            require(at >= datetime.fromisoformat(available), "Execution before decision availability")
        cash += (-1 if buy else 1) * q * price - fee
        positions[fill["symbol"]] += q if buy else -q
        require(positions[fill["symbol"]] >= 0, "Unfunded short or duplicate exit")
    require(all(q == 0 for q in positions.values()), "Terminal position mismatch")
    close(cash, equity[-1][1], "Independent cash reconciliation failed")
    close(sum(t["net_pnl"] for t in trades), cash - initial, "Trades do not reconcile to cash")
    require(len(trades) == result["trades"], "Trade denominator mismatch")
    gains = sum(max(t["net_pnl"], 0) for t in trades)
    losses = -sum(min(t["net_pnl"], 0) for t in trades)
    close(result["metrics"]["profit_factor"], gains / losses, "Gross or R-based headline profit factor")
    peak, dd = initial, 0.0
    for stamp, value in equity:
        require(profile["date_range"]["start"] <= stamp[:10] <= profile["date_range"]["end"], "Out-of-scope marks")
        peak = max(peak, value)
        dd = max(dd, 100 * (1 - value / peak))
    close(result["drawdown_pct"], dd, "Realized-only or stale drawdown")
    close(result["return_pct"], 100 * (cash / initial - 1), "Inflated account return")
    require(len(daily) == result["sessions"] == profile["date_range"]["sessions"], "Omitted cash sessions")
    require(all(not d["inventory"] and not d["pending"] for d in daily), "Residual intraday inventory")
    for key in ("return_pct", "drawdown_pct", "trades", "active_days"):
        close(result[key], profile["metrics"][scenario][key], "Stale selected metrics: " + key)
    return {"result": result, "audit": audit, "trades": trades, "fills": fills, "daily": daily, "equity": equity,
            "validation": {"passed": True, "independent_cash_krw": cash, "fill_count": len(fills),
                           "net_profit_factor": gains / losses, "mtm_drawdown_pct": dd}}


def load_qualified_profile(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = read(manifest_path)
    identity = manifest.pop("manifest_hash", None)
    require(identity == stable_signature(manifest), "Changed acceptance manifest")
    require(manifest.get("version") == "kalcb-qualified-native-exports-v1", "Unsupported acceptance schema")
    for filename, expected in manifest["files"].items():
        require(digest(filename) == expected, "Changed qualified artifact: " + filename)
    def bound(filename):
        require(str(Path(filename).resolve()) in manifest["files"], "Unbound qualified artifact: " + str(filename))
    bound(manifest["profile"])
    profile = read(manifest["profile"])
    for key in ("configuration", "audit_manifest", "plan", "all_trials", "attribution"):
        bound(profile[key])
    for key in ("scope_preparation_parity", "forecast_reconstruction", "final_account_verification", "input_verification"):
        bound(profile[key])
        require(read(profile[key]).get("passed") is True, "Failed qualification evidence: " + key)
    for filename, expected in profile["research_protocols"].items():
        bound(filename)
        require(digest(filename) == expected, "Changed registered research protocol")
    for result_path in profile["result_paths"].values():
        for name in REPLAY_FILES:
            bound(Path(result_path).parent / name)
    for filename in Path(profile["policy"]["directory"]).iterdir():
        if filename.is_file():
            bound(filename)
    require(profile["status"] == "qualified_research_only" and profile["live_enabled"] is False, "Unsupported release status")
    require(digest(profile["configuration"]) == profile["configuration_sha256"], "Changed strategy configuration")
    require(digest(profile["audit_manifest"]) == profile["audit_manifest_sha256"], "Changed audit source manifest")
    from scripts.kalcb_executed_value_replay import verify_policy, timed_policy, validate_watch_universe
    policy, opening = verify_policy(Path(profile["policy"]["directory"]), profile["policy"]["name"])
    updates = timed_policy(Path(profile["policy"]["directory"]), policy, opening)
    import yaml
    plan = read(profile["plan"])
    universe = {str(s).zfill(6) for s in yaml.safe_load(Path(plan["universe"]).read_text())["symbols"]}
    validate_watch_universe(policy["observation_symbols"], universe)
    require(len(universe) == 103 and policy["observation_minutes"] == 45, "Changed observation contract")
    require(policy["evaluation_from"] == profile["date_range"]["start"] and policy["evaluation_through"] == profile["date_range"]["end"], "Changed policy dates")
    cases = {s: validate_case(Path(profile["result_paths"][s]).parent, profile, s) for s in SCENARIOS}
    return {"profile": profile, "manifest": str(manifest_path), "manifest_hash": identity,
            "cases": cases, "opening": opening, "updates": updates, "policy": policy}


def diagnostic_result(bundle, scenario="baseline"):
    from strategy_common.events import TradeOutcome, DecisionEvent
    from strategy_common.actions import action_from_json_dict
    from backtests.engine.replay import ReplayResult
    from .runner import StrategyBacktestResult
    profile, case = bundle["profile"], bundle["cases"][scenario]
    names = {f.name for f in fields(TradeOutcome)}
    trades = []
    for row in case["trades"]:
        row = {k: v for k, v in row.items() if k in names}
        for key in ("entry_decision_time", "entry_fill_time", "exit_fill_time"):
            row[key] = datetime.fromisoformat(row[key]) if row[key] else None
        trades.append(TradeOutcome(**row))
    decisions = []
    p = Path(profile["result_paths"][scenario]).parent
    with gzip.open(p / "decisions.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            row["timestamp"] = datetime.fromisoformat(row["timestamp"])
            row["actions"] = tuple(action_from_json_dict(x) for x in row["actions"])
            decisions.append(DecisionEvent(**row))
    require(dict(Counter(d.reason for d in decisions)) == case["result"]["decision_counts"], "Decision export mismatch")
    source = profile["execution_source"]["source_generation_hash"]
    snapshots = tuple(s for day, s in sorted(bundle["opening"].items()))
    all_snapshots = [s for day, base in sorted(bundle["opening"].items()) for s in (base, *bundle["updates"].get(day, ()))]
    pool_max = max(len(s.candidates) for s in all_snapshots)
    metrics = {**case["result"]["metrics"], **{k: case["audit"][k] for k in ("audit_status", "audit_pass", "same_bar_fill_count", "forced_replay_close_count", "rejected_order_count")},
        "official_mtm_net_return_pct": case["result"]["return_pct"] / 100, "official_replay_pass": True,
        "official_metric_basis": "native_bar_marked_equity", "net_return_pct_basis": "net_cash_change_over_initial_equity",
        "observed_session_days": len(case["daily"]), "candidate_snapshot_count": len(snapshots),
        "universe_size": 103, "data_available_symbol_count": 103, "unavailable_symbol_count": 0,
        "candidate_pool_max": pool_max, "candidate_pool_universe_fraction": pool_max / 103,
        "candidate_pool_data_available_fraction": pool_max / 103,
        "active_symbol_max": pool_max, "selected_universe_fraction": pool_max / 103,
        "shared_decision_core": "strategy_kalcb.core.logic.step_kalcb_core", "live_parity_fill_timing": "next_5m_open",
        "auction_mode": "non_auction_continuous", "replay_mode": "audited_native_timed_candidates",
        "replay_event_count": case["audit"]["audit_receipt"]["source_event_count"],
        "end_open_position_count": 0, "qualification_scope": "historical_research_model"}
    equity = [profile["initial_equity_krw"], *[r[1] for r in case["equity"]]]
    replay = ReplayResult(trades, decisions, equity, [datetime.fromisoformat(r[0]) for r in case["equity"]], None)
    return StrategyBacktestResult("kalcb", metrics, replay, source, "real_replay",
        stable_signature([s.to_json_dict() for s in all_snapshots]), bundle["policy"]["prepared_sha256"])
