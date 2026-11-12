"""Frozen-data KALCB executed-value labels, chronological forecasts and replay.

Run --help. Every output directory is new; inputs and executing source are hashed.
No deployment configuration or execution mechanics are changed by this study.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import date, datetime, time, timedelta
from itertools import groupby
import json
import math
from pathlib import Path
import sqlite3
import sys
import time as wallclock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb import executed_value as ev


def load_plan(path):
    plan = json.loads(path.read_text())
    for name, expected in plan["input_sha256"].items():
        if ev.digest(name) != expected:
            raise ValueError(f"Changed study input: {name}")
    return plan


def config_for(plan, scenario="baseline"):
    from strategy_kalcb.config import KALCBConfig
    raw = json.loads(Path(plan["config"]).read_text())
    raw.update(slippage_bps=ev.SCENARIOS[scenario][1], fast_replay_suppress_rejections=False)
    return KALCBConfig.from_mapping(raw)


def connect(plan):
    connection = sqlite3.connect(f"file:{Path(plan['database']).as_posix()}?mode=ro", uri=True)
    connection.execute("PRAGMA cache_size=-8192")
    return connection


def load_day(conn, day, source, symbols=None):
    from strategy_common.market import MarketBar
    query = "SELECT symbol,ts,o,h,l,c,v FROM bars WHERE day=?"
    parameters = [day]
    if symbols is not None:
        if not symbols:
            return []
        query += " AND symbol IN (" + ",".join("?" for _ in symbols) + ")"
        parameters.extend(sorted(symbols))
    query += " ORDER BY ts,symbol"
    return [MarketBar(s, datetime.fromisoformat(t), "5m", *values, source="kis_krx_parquet",
                      source_fingerprint=source) for s, t, *values in conn.execute(query, parameters)]


def emit(stream, row):
    stream.write(json.dumps(row, default=str, sort_keys=True, allow_nan=False) + "\n")


def prepare(plan, output):
    import pandas as pd
    import yaml
    from kis_core.input_manifest import InputManifest
    from backtests.strategies.kalcb import first30_signal_sweep as native
    from backtests.strategies.kalcb.trade_plan_sweep import _candidate_from_context
    from kis_core.trading_calendar import get_trading_calendar
    generation = InputManifest(plan["market_manifest"])
    cfg = config_for(plan)
    universe = tuple(str(s).zfill(6) for s in yaml.safe_load(Path(plan["universe"]).read_text())["symbols"])
    legacy = {(r["trade_date"], r["symbol"]): r for r in ev.read_jsonl(plan["guard_features"])}
    frame = pd.read_csv(plan["pipeline"], dtype={"symbol": str}) if not plan.get('forward_feature_only') else None
    targets = {}
    for r in (frame.loc[frame.window.eq("train")].to_dict("records") if frame is not None else []):
        targets[(r["trade_date"], r["symbol"])] = math.log1p(max(float(r["positive_mfe_r_proxy"]), 0)) + .35 * math.log1p(max(float(r["net_eod_r_proxy"]), 0)) - .25 * math.log1p(max(-float(r["mae_r_proxy"]), 0))
    conn = connect(plan)
    sessions = [r[0] for r in conn.execute("SELECT DISTINCT day FROM bars WHERE day BETWEEN ? AND ? ORDER BY day", (plan["start"], plan["end"]))]
    context_symbols = tuple(r[0] for r in conn.execute("SELECT DISTINCT symbol FROM bars ORDER BY symbol"))
    daily_root = generation.root / "data/krx_daily_parquet"
    if plan.get('forward_feature_only'):
        # A missing frozen table silently makes every sector UNKNOWN and changes
        # both model inputs and the financial-sector admission rule.
        generation.read(daily_root / "tables/sector_map.parquet")
    end = date.fromisoformat(sessions[-1])
    with generation.activate():
        daily = native._load_daily_rows(daily_root, context_symbols, end)
        flow = native._load_flow_rows(daily_root, context_symbols, end)
        foreign = native._load_foreign_flow_rows(daily_root, context_symbols, end)
        institutional = native._load_institutional_flow_rows(daily_root, context_symbols, end)
        index = native._load_index_rows(daily_root, end)
        sectors = native.load_sector_map(daily_root)
    if plan.get('forward_feature_only'):
        missing_sectors = [s for s in universe if str(sectors.get(s, '')).strip().upper() in {'', 'UNKNOWN'}]
        if missing_sectors:
            raise ValueError('Incomplete frozen forward sector map: ' + ','.join(missing_sectors))
    source = plan["input_sha256"][plan["database"]]
    dataset = native.KALCBFirst30Dataset(asdict(cfg), source, generation.identity,
        generation.root / "data/kis_intraday_parquet", daily_root, "5m", context_symbols,
        context_symbols, tuple(daily), (), daily, flow, index, (), {}, sectors, foreign, institutional)
    calendar = get_trading_calendar()
    counts, parity = Counter(), {"comparisons": 0, "mismatches": []}
    forward_context = []
    seed = json.loads(Path(plan["seed"]).read_text())["mutations"]
    q3 = json.loads(Path(plan["q3"]).read_text())["mutations"]
    with (output / "opportunities.jsonl").open("x", encoding="utf-8") as rows_out, (output / "coverage.jsonl").open("x", encoding="utf-8") as coverage:
        for number, day in enumerate(sessions):
            dt = date.fromisoformat(day)
            opening, _ = calendar.session_hours(dt)
            available = datetime.combine(dt, opening, ev.KST) + timedelta(minutes=30)
            # The feature builder receives no later-session prices or volumes.
            bars = defaultdict(list)
            for bar in load_day(conn, day, source):
                if bar.timestamp + timedelta(minutes=5) <= available:
                    bars[(dt, bar.symbol)].append(bar)
            for cache in (native._SORTED_BARS_CACHE, native._FIRST30_BARS_CACHE, native._POST_ENTRY_BARS_CACHE):
                cache.clear()
            one = replace(dataset, trading_dates=(dt,), bars_by_key={k: tuple(v) for k, v in bars.items()})
            contexts = {ctx.symbol: ctx for ctx in native.build_contexts(one, require_post_entry_bars=False).get(dt, ())}
            for symbol in universe:
                ctx = contexts.get(symbol)
                if ctx is None:
                    counts["feature_unavailable"] += 1
                    emit(coverage, {"trade_date": day, "symbol": symbol, "status": "opening_or_prior_context_unavailable"})
                    continue
                candidate = _candidate_from_context(ctx, native.Selection(dt, symbol, 0, "executed_value"), one, cfg, 1)
                meta = dict(candidate.metadata)
                opening_bars = bars[(dt, symbol)]
                last = opening_bars[-1]
                total_volume = sum(b.volume for b in opening_bars)
                meta["first30_signal_bar_cpr"] = (last.close - last.low) / max(last.high - last.low, 1e-9)
                meta["first30_range_close_location"] = ctx.close_location
                # Preserve the historical guard's feature contract: omitted flow/
                # accumulation fields are not silently replaced with new inputs.
                old = legacy.get((day, symbol))
                if old is None and plan.get('forward_guard_schema'):
                    old = {k:meta[k] for k in plan['forward_guard_schema'] if k in meta}
                guard = {k: v for k, v in (old or meta).items() if k.startswith(("daily_", "first30_", "sector_", "stock_sector_", "continuation_"))}
                guard.update({k: meta[k] for k in guard if k in meta})
                for key, value in (old or {}).items():
                    if key in meta and isinstance(value, (int, float)) and key not in {"first30_score", "daily_adv20_krw_log"}:
                        parity["comparisons"] += 1
                        if not math.isclose(value, meta[key], rel_tol=1e-9, abs_tol=1e-8):
                            if len(parity["mismatches"]) < 50:
                                parity["mismatches"].append([day, symbol, key, value, meta[key]])
                features = {k: meta.get(k) for k in ev.FEATURES}
                if plan.get('forward_feature_only'):
                    from scripts.kalcb_positive_r_pipeline_breakdown import context_features
                    forward_context.append({'trade_date':day,'symbol':symbol,'window':'holdout',**context_features(ctx)})
                features.update(first30_rel_volume_log=math.log1p(max(ctx.rel_volume, 0)),
                    daily_atr_pct=ctx.daily.atr14 / ctx.daily.prev_close,
                    daily_adv20_krw_log=math.log1p(ctx.daily.adv20_krw),
                    first30_volume_log=math.log1p(total_volume),
                    last_opening_volume_share=last.volume / max(total_volume, 1),
                    opening_volume_min_share=min(b.volume for b in opening_bars) / max(total_volume, 1),
                    opening_capacity_ratio=candidate.average_30m_volume / max(last.volume, 1))
                row = {"opportunity_id": day + "/" + symbol, "trade_date": day, "symbol": symbol,
                    "sector": ctx.sector, "feature_available_at": available.isoformat(), "features": features,
                    "guard_features": guard, "candidate": candidate.to_json_dict(), "quality_target": targets.get((day, symbol))}
                row["guard_at_rank1"] = ev.eligible(row, seed, q3, rank=1)
                emit(rows_out, row)
                counts["opportunities"] += 1
                counts["guard_eligible_at_rank1"] += row["guard_at_rank1"]
            if number % 20 == 0:
                print(json.dumps({"prepare_session": number + 1, "sessions": len(sessions), **counts}), flush=True)
    conn.close()
    if plan.get('forward_feature_only'):
        # No post-entry rows or outcome proxies enter this feature-only table.
        pd.DataFrame(forward_context).to_csv(output/'causal_context.csv',index=False)
    ev.write_json(output / "prepared.json", {"version": ev.VERSION, "sessions": sessions, "universe": universe,
        "context_symbols": context_symbols, "counts": counts, "native_archived_feature_comparison": parity,
        "guard_cpr_semantics": "actual_last_opening_bar; archived guard used whole_opening_range_location",
        "historical_guard_control": "scripts/kalcb_executed_value_guard_control.py reuses the identical forecasts with the historical formula",
        "causal_contract": "native features built with only bars completed by opening+30m; later outcome coverage never controls membership",
        "opportunities_sha256": ev.digest(output / "opportunities.jsonl"), "plan_sha256": ev.signature(plan)})


def run_episode(row, bars, cfg, scenario, source, conn, sessions):
    from backtests.engine.capacity import BarParticipationScenario
    from backtests.engine.replay import drive_timestamp_batch
    from backtests.engine.sim_broker import SimBroker, BrokerCosts
    from backtests.strategies.kalcb.runner import KALCBReplayAdapter
    from kis_core.trading_calendar import get_trading_calendar
    participation, slippage, name = ev.SCENARIOS[scenario]
    costs = BrokerCosts(cfg.commission_bps, cfg.tax_bps_on_sell, slippage)
    broker = SimBroker(ev.REFERENCE_EQUITY, costs, buying_power_leverage=cfg.intraday_leverage,
                       capacity_policy=BarParticipationScenario(participation, name))
    snapshot = ev.snapshot_for(row["trade_date"], [(row, 0)], source=source)
    adapter = KALCBReplayAdapter(cfg, {snapshot.trade_date: snapshot}, initial_equity=ev.REFERENCE_EQUITY, costs=costs)
    decisions, deadline_inventory = [], {}
    calendar = get_trading_calendar()
    last_available = None
    def step(sequence):
        nonlocal last_available
        for bar in sequence:
            decisions.extend(e.to_json_dict() for e in drive_timestamp_batch(broker, (bar,), (adapter,)))
            last_available = bar.timestamp + timedelta(minutes=5)
            closing = calendar.session_hours(bar.timestamp.date())[1]
            deadline = datetime.combine(bar.timestamp.date(), closing, ev.KST) - timedelta(minutes=10)
            day = bar.timestamp.date().isoformat()
            if last_available >= deadline and day not in deadline_inventory:
                deadline_inventory[day] = sum(p.qty for p in broker.positions.values())
    step(bars)
    first_day_complete = row["trade_date"] in deadline_inventory
    # Ordinary fills only. Carry failures remain in the labels and risk outcomes.
    if broker.positions or broker.orders:
        for day in sessions[sessions.index(row["trade_date"]) + 1:]:
            step(load_day(conn, day, source, {row["symbol"]}))
            if not broker.positions and not broker.orders:
                break
    terminal_qty = sum(p.qty for p in broker.positions.values())
    value = ev.ledger_value(broker.fills, costs, terminal_inventory=terminal_qty, pending=len(broker.orders))
    if abs(value["net_cash_flow_krw"] - (broker.cash - ev.REFERENCE_EQUITY)) > .01:
        raise ValueError("Independent cash ledger differs from broker")
    if not value["unresolved"] and abs(sum(t.net_pnl for t in broker.trades) - value["net_value_krw"]) > .01:
        raise ValueError("Independent fill PnL differs from trades")
    submissions = [r for r in broker.journal if r["event"] == "submit" and r["order"]["side"] == "BUY"]
    physical_failures = ev.physical_fill_failures(broker.fills)
    valid = first_day_complete and not value["unresolved"] and not broker.same_bar_fill_violations and not physical_failures
    status = ("execution_invalid" if physical_failures else "unresolved" if value["unresolved"] else "source_incomplete" if not first_day_complete
              else "filled" if broker.fills else "unfilled" if submissions else "no_order")
    result = {"opportunity_id": row["opportunity_id"], "trade_date": row["trade_date"], "symbol": row["symbol"],
        "scenario": scenario, "status": status, "label_valid": valid,
        "physical_fill_failures": physical_failures,
        "label_available_at": last_available.isoformat() if valid else None,
        # An unresolved action has no mature return, but its failure must not
        # disappear from admission calibration. That failure is observable by
        # the end of its opening session, independently of later liquidation.
        "risk_available_at": (datetime.combine(date.fromisoformat(row["trade_date"]),
            calendar.session_hours(date.fromisoformat(row["trade_date"]))[1], ev.KST)
            + timedelta(minutes=5)).isoformat(),
        "deadline_inventory": deadline_inventory, "deadline_violation": any(deadline_inventory.values()),
        "terminal_inventory": terminal_qty, "terminal_pending": len(broker.orders),
        "entry_orders_submitted": len(submissions), "decision_reason_counts": dict(Counter(d["reason"] for d in decisions)),
        **value}
    if not valid:
        result["net_value_krw"] = result["net_value_equity_bps"] = None
    if physical_failures:
        result["risk_available_at"] = max(datetime.fromisoformat(result["risk_available_at"]),
            *(datetime.fromisoformat(f["timestamp"]) + timedelta(minutes=5) for f in physical_failures)).isoformat()
    detail = {"opportunity_id": row["opportunity_id"], "scenario": scenario,
        "decisions": decisions, "fills": [asdict(f) for f in broker.fills],
        "orders": [r for r in broker.journal if r["event"] in {"submit", "amend", "cancel", "reject", "expire"}],
        "source_events_hash": broker.source_events.hexdigest(), "source_event_count": broker.source_event_count}
    return result, detail


def label(plan, prepared, output, scenario, start_index=0, stop_index=None):
    import gzip
    provenance = json.loads((prepared / "prepared.json").read_text())
    if ev.digest(prepared / "opportunities.jsonl") != provenance["opportunities_sha256"]:
        raise ValueError("Changed opportunity input")
    sessions = provenance["sessions"]
    selected_sessions = sessions[start_index:stop_index]
    wanted = set(selected_sessions)
    by_day = defaultdict(list)
    with (prepared / "opportunities.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["trade_date"] in wanted:
                by_day[row["trade_date"]].append(row)
    cfg, conn = config_for(plan, scenario), connect(plan)
    source = plan["input_sha256"][plan["database"]]
    counts = Counter()
    for number, day in enumerate(selected_sessions):
        began = wallclock.monotonic()
        bars = defaultdict(list)
        for bar in load_day(conn, day, source):
            bars[bar.symbol].append(bar)
        dayfile = output / (day + ".jsonl")
        with dayfile.open("x", encoding="utf-8") as values, gzip.open(output / (day + ".events.jsonl.gz"), "xt", encoding="utf-8", compresslevel=1) as events:
            for row in by_day[day]:
                result, detail = run_episode(row, bars[row["symbol"]], cfg, scenario, source, conn, sessions)
                emit(values, result)
                emit(events, detail)
                counts[result["status"]] += 1
                counts["deadline_violations"] += result["deadline_violation"]
        print(json.dumps({"label_session": number + 1, "of": len(selected_sessions), "day": day,
                          "seconds": round(wallclock.monotonic()-began, 2), **counts}), flush=True)
    conn.close()
    ev.write_json(output / "labels_manifest.json", {"version": ev.VERSION, "scenario": scenario,
        "config": asdict(cfg), "prepared_sha256": ev.digest(prepared / "prepared.json"),
        "request_sha256": ev.digest(output / "request.json"),
        "sessions": selected_sessions, "counts": counts,
        "label_files": {p.name: ev.digest(p) for p in output.glob("*.jsonl")},
        "event_files": {p.name: ev.digest(p) for p in output.glob("*.gz")}})


def main():
    from backtests.auto.shared.cache_keys import source_generation_hash
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "label"))
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--scenario", choices=ev.SCENARIOS, default="baseline")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stop-index", type=int)
    args = parser.parse_args()
    plan, source = load_plan(args.plan), source_generation_hash()
    args.output.mkdir(parents=True, exist_ok=False)
    ev.write_json(args.output / "request.json", {"arguments": vars(args), "plan": plan, "source_generation_hash": source})
    if args.command == "prepare":
        prepare(plan, args.output)
    else:
        label(plan, args.prepared, args.output, args.scenario, args.start_index, args.stop_index)
    if source_generation_hash() != source:
        raise RuntimeError("Executing source changed")
    load_plan(args.plan)
    ev.write_json(args.output / "completion.json", {"source_generation_hash": source, "complete": True})


if __name__ == "__main__":
    main()
