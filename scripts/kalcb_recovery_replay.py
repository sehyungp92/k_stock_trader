"""Replay a saved KALCB recipe on a frozen successor generation, serially.

Example (run this script FROM the successor generation's files directory):
  python scripts/kalcb_recovery_replay.py --request OLD/request.json \
    --manifest ../manifest.json --output STUDY/r3 --price-lock

All capacity and next-open fill rules remain in the shared replay engine.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtests.auto.shared.cache_keys import source_generation_hash
from backtests.core.input_manifest import InputManifest
import backtests.strategies.kalcb.fixed_trade_plan_phase as fixed


def write(path, payload):
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def bind_requested_window(config, start=None, end=None):
    """The engine gives nested date_range precedence over top-level dates."""
    result = dict(config)
    window = dict(result.get("date_range") or {})
    for key, override in (("start", start), ("end", end)):
        value = override or window.get(key) or result.get(key)
        if value:
            result[key] = value
            window[key] = str(value)[:10]
    if window.get("start", "") > window.get("end", "9999-12-31"):
        raise ValueError("Requested replay window is empty")
    result["date_range"] = window
    result["use_full_available_window"] = False
    return result


def research_capacity_scenario(participation, slippage_bps):
    # These names are part of the immutable declared execution policy identity.
    # Custom scenarios remain explicitly unregistered and cannot inherit a pass.
    names = {(.01, 3.0): "research-baseline", (.005, 3.0): "participation-stress",
        (.01, 13.0): "cost-stress", (.005, 13.0): "combined-stress"}
    return {"name": names.get((participation, slippage_bps), "unregistered-recovery-research"),
        "participation": participation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--participation", type=float, default=.01)
    parser.add_argument("--slippage-bps", type=float)
    parser.add_argument("--entry-cap", type=float)
    parser.add_argument("--price-lock", action="store_true", help="Explicitly enable completed-print prior-close proxy")
    parser.add_argument("--pool", type=Path)
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()
    if not 0 < args.participation <= 1 or args.entry_cap is not None and not 0 < args.entry_cap <= 1:
        raise ValueError("Participation must be in (0, 1]")
    generation = InputManifest(args.manifest)
    expected = generation.payload["source_generation_hash"]
    if source_generation_hash() != expected:
        raise ValueError("Execute the script from the frozen successor source")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    config = dict(request["config"])
    mutations = dict(request["recipe"]["mutations"])
    config["input_manifest_path"] = str(args.manifest.resolve())
    if args.slippage_bps is not None:
        mutations["kalcb.robustness.slippage_bps"] = args.slippage_bps
    slippage = float(mutations.get("kalcb.robustness.slippage_bps", config.get("slippage_bps", 3.0)))
    config["capacity_scenario"] = research_capacity_scenario(args.participation, slippage)
    if args.entry_cap is not None:
        mutations["kalcb.risk.max_participation_30m"] = args.entry_cap
    if args.price_lock:
        mutations.update({"kalcb.exit.price_lock_bars": 2,
            "kalcb.exit.price_lock_allow_prior_close_proxy": True,
            "kalcb.exit.price_lock_min_prior_close_return": .29})
    if args.pool:
        if not generation.digest(args.pool):
            raise ValueError("Successor pool must be frozen in the input manifest")
        mutations.update({"_kalcb.pool_source.path": str(args.pool.resolve()),
            "_kalcb.pool_source.label": "past-only recovery research", "_kalcb.pool_source.active_count": 16})
    config = bind_requested_window(config, args.start, args.end)
    config["initial_mutations"] = mutations
    config["fixed_candidate_source"] = {"path": mutations["_kalcb.source.path"],
        "section": mutations["_kalcb.source.section"], "rank": mutations["_kalcb.source.rank"]}
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "request.json", {"scope": "exploratory; no promotion or untouched OOS claim",
        "parent_request_sha256": hashlib.sha256(args.request.read_bytes()).hexdigest(),
        "manifest": generation.identity, "source": expected, "config": config, "mutations": mutations})
    actual_run = fixed.run_replay
    def capture(*a, **kw):
        replay = actual_run(*a, **kw)
        write(args.output / "fills.json", [asdict(f) for f in replay.broker.fills])
        write(args.output / "equity.json", list(zip(map(str, replay.timestamps), replay.equity_curve)))
        return replay
    fixed.run_replay = capture
    start = time.monotonic()
    try:
        with generation.activate():
            backend = fixed.KALCBFixedTradePlanOptimizationPlugin(config, output_dir=args.output / "context", max_workers=1)
            context = backend._context_for_mutations(mutations)
            dates = list(context.train_dates)
            requested = config["date_range"]
            if not dates or (requested.get("start") and str(min(dates)) < requested["start"]) or (requested.get("end") and str(max(dates)) > requested["end"]):
                raise ValueError("Prepared replay dates escape the requested window")
            actual_window = {"first_session": str(min(dates)), "last_session": str(max(dates)), "sessions": len(dates)}
            write(args.output / "actual_window.json", actual_window)
            evaluation = backend._evaluate(mutations)
            write(args.output / "trades.json", evaluation.trade_rows)
            write(args.output / "folds.json", evaluation.fold_rows)
            write(args.output / "decisions_summary.json", evaluation.decision_summary)
        if source_generation_hash() != expected:
            raise RuntimeError("Source changed during replay")
        write(args.output / "result.json", {"scope": "exploratory; not promoted", "seconds": time.monotonic() - start,
            "metrics": evaluation.metrics, "actual_window": actual_window, "source_generation_hash": expected, "manifest": generation.identity})
        print(json.dumps({k: evaluation.metrics.get(k) for k in ("official_mtm_net_return_pct", "broker_max_drawdown_pct",
            "broker_trade_count", "audit_status", "intraday_residual_session_count", "worst_fold_net")}), flush=True)
    finally:
        fixed.run_replay = actual_run


if __name__ == "__main__":
    main()
