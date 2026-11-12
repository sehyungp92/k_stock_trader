"""Disk-backed research feed, using the existing KALCB adapter and shared broker.

No alternative fill engine or forced terminal liquidation. Before trusting a
new experiment, reproduce a saved audited result with its original code root.
The broker audit remains unchanged: execution validity and formal promotion
qualification are reported separately. Frozen parquet and snapshot hashes are
verified independently of the replay's observations.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import asdict, replace
from datetime import date, datetime, time
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import re
import sqlite3
import sys


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def effective_config(reference):
    """Decode the auditor's dataclass repr without executing Python text."""
    text = reference["metrics"]["audit_receipt"]["identity"]["config"]
    text = re.sub(r"<\w+\.\w+: ('[^']*')>", r"\1", text)
    expression = ast.parse(text, mode="eval").body
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name) or expression.func.id != "KALCBConfig":
        raise ValueError("Unexpected recorded configuration")
    result = {}
    for item in expression.keywords:
        node = item.value
        if isinstance(node, ast.Call):
            if ast.unparse(node.func) != "datetime.time" or node.keywords:
                raise ValueError("Unsupported config value")
            result[item.arg] = time(*(ast.literal_eval(x) for x in node.args)).isoformat()
        else:
            result[item.arg] = ast.literal_eval(node)
    return result


def build_market(manifest, database, start, end):
    import pyarrow.parquet as pq
    from backtests.core.input_manifest import InputManifest
    from strategy_common.clock import ensure_kst
    generation = InputManifest(manifest)
    evidence = json.loads(generation.read(generation.root / generation.payload["evidence"]["qualification_path"]))
    if database.exists():
        raise FileExistsError(database)
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA cache_size=-8192")
    connection.execute("CREATE TABLE bars (symbol TEXT, ts TEXT, day TEXT, o REAL,h REAL,l REAL,c REAL,v REAL, PRIMARY KEY(symbol,ts)) WITHOUT ROWID")
    counts = {}
    sources = {}
    for item in evidence["market_files"]:
        if item["timeframe"] != "5m":
            continue
        if item["parser_version"] != "completed_ohlcv_parquet_v1":
            raise ValueError("Unsupported market parser")
        path = generation.root / item["path"]
        count = 0
        with generation.open_verified(path) as stream:
            parquet = pq.ParquetFile(stream)
            for batch in parquet.iter_batches(batch_size=4096):
                rows = []
                for row in batch.to_pylist():
                    stamp = ensure_kst(row["timestamp"])
                    day = stamp.date().isoformat()
                    if start <= day <= end:
                        if not row.get("is_completed", True):
                            raise ValueError("Incomplete frozen bar")
                        rows.append((item["symbol"], stamp.isoformat(), day,
                            *[float(row[k]) for k in ("open", "high", "low", "close", "volume")]))
                connection.executemany("INSERT INTO bars VALUES (?,?,?,?,?,?,?,?)", rows)
                count += len(rows)
        connection.commit()
        counts[item["symbol"]] = count
        sources[item["path"]] = generation.digest(path)
        print(json.dumps({"symbol": item["symbol"], "bars": count}), flush=True)
    connection.execute("CREATE INDEX by_day ON bars(day,symbol,ts)")
    connection.commit()
    connection.close()
    write(database.with_suffix(".provenance.json"), {"source_manifest": str(manifest.resolve()),
        "manifest_hash": generation.identity, "start": start, "end": end,
        "source_files": sources, "counts": counts, "database_sha256": digest(database)})


def replay(args):
    from backtests.core.input_manifest import InputManifest
    from backtests.auto.shared.cache_keys import source_generation_hash
    from backtests.engine.capacity import BarParticipationScenario
    from backtests.engine.replay import drive_timestamp_batch
    from backtests.engine.sim_broker import SimBroker, BrokerCosts
    from backtests.engine.integrity import audit_broker
    from backtests.analysis.metrics import compute_trade_metrics
    from backtests.strategies.kalcb.runner import KALCBReplayAdapter, _collapse_exit_legs
    from strategy_common.market import MarketBar
    from strategy_kalcb.config import KALCBConfig
    from strategy_kalcb.models import KALCBDailySnapshot
    source = source_generation_hash()
    generation = InputManifest(args.manifest)
    provenance = json.loads(args.database.with_suffix(".provenance.json").read_text())
    if provenance["manifest_hash"] != generation.identity or digest(args.database) != provenance["database_sha256"]:
        raise ValueError("Changed market database or manifest")
    if args.start < provenance["start"] or args.end > provenance["end"]:
        raise ValueError("Requested period escapes the frozen database")
    snapshots = {}
    snapshot_hashes = {}
    for path in sorted(args.snapshots.glob("*.json")):
        raw = json.loads(generation.read(path))
        day = raw["trade_date"]
        if args.start <= day <= args.end:
            snapshot = KALCBDailySnapshot.from_json_dict(raw)
            if snapshot.artifact_hash != raw["artifact_hash"]:
                raise ValueError("Changed snapshot")
            snapshots[snapshot.trade_date] = snapshot
            snapshot_hashes[str(path)] = generation.digest(path)
    if args.snapshot_override:
        snapshots = {date.fromisoformat(r["trade_date"]): KALCBDailySnapshot.from_json_dict(r)
            for r in json.loads(args.snapshot_override.read_text()) if args.start <= r["trade_date"] <= args.end}
        snapshot_hashes = {str(args.snapshot_override): digest(args.snapshot_override)}
    reference = json.loads(args.reference.read_text())
    mapping = effective_config(reference)
    mutations = json.loads(args.mutations.read_text()) if args.mutations else {}
    mapping.update(slippage_bps=args.slippage, fast_replay_suppress_rejections=True)
    config = KALCBConfig.from_mapping(mapping, mutations)
    if config.frontier_shadow_enabled or config.frontier_rotation_enabled:
        raise ValueError("Bounded study requires the saved no-shadow/no-rotation recipe")
    reference_binding = {}
    if args.audit_manifest:
        if args.snapshot_override:
            raise ValueError("Formal audit requires the independently frozen candidate-source pool")
        audit_generation = InputManifest(args.audit_manifest)
        if audit_generation.payload["source_generation_hash"] != source:
            raise ValueError("Audit manifest must bind the executing source generation")
        evidence = json.loads(audit_generation.read(audit_generation.root / audit_generation.payload["evidence"]["qualification_path"]))
        daily_path = audit_generation.root / evidence["references"]["daily_rows"]
        daily = {(r["symbol"],r["date"]):r for r in json.loads(audit_generation.read(daily_path))}
        from kis_core.trading_calendar import get_trading_calendar
        calendar = get_trading_calendar()
        for day, snapshot in tuple(snapshots.items()):
            prior = calendar.previous_trading_day(day).isoformat()
            candidates = []
            for candidate in snapshot.candidates:
                retained = daily[(candidate.symbol,prior)]
                if candidate.metadata.get("prior_day_date") not in (None,prior):
                    raise ValueError("Existing snapshot reference date disagrees with calendar")
                if any(not math.isclose(getattr(candidate,"prior_day_"+field),float(retained[field]),rel_tol=1e-10)
                       for field in ("high","low","close")):
                    raise ValueError("Snapshot prior prices do not match the dated frozen source")
                candidates.append(replace(candidate,metadata={**dict(candidate.metadata),
                    "prior_day_date":prior,"prior_reference_status":"verified_previous_session"}))
            snapshots[day] = replace(snapshot,candidates=tuple(candidates))
        reference_binding = {"basis":"matched_frozen_daily_OHLC_and_previous_exchange_session",
            "source_path":str(daily_path),"source_sha256":audit_generation.digest(daily_path),
            "candidate_count":sum(len(s.candidates) for s in snapshots.values())}
    if args.legacy_lock:
        import strategy_kalcb.core.logic as logic
        ordinary = logic._exit_reason
        def lock_reason(ss, pos, bar, cfg, **kw):
            reason = ordinary(ss, pos, bar, cfg, **kw)
            if reason:
                return reason
            recent = ss.bars_today[-2:]
            if ss.candidate and len(recent) == 2:
                p = recent[-1].close
                if (recent[-1].timestamp - recent[0].timestamp).total_seconds() == 300 and p >= 1.29 * ss.candidate.prior_day_close and all(b.open == b.high == b.low == b.close == p for b in recent):
                    return "exploratory_two_flat_near_upper_limit"
            return ""
        logic._exit_reason = lock_reason
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "request.json", {"scope": "Frozen candidate research; no untouched OOS or promotion claim",
        "source_generation_hash": source, "manifest_hash": generation.identity,
        "database_sha256": provenance["database_sha256"], "snapshots": snapshot_hashes,
        "reference": str(args.reference), "reference_sha256": digest(args.reference),
        "dated_reference_binding": reference_binding,
        "effective_config": asdict(config), "mutations": mutations, "arguments": vars(args)})
    names = {(.01, 3.): "research-baseline", (.005, 3.): "participation-stress",
             (.01, 13.): "cost-stress", (.005, 13.): "combined-stress"}
    policy = BarParticipationScenario(args.participation, names[(args.participation, args.slippage)])
    costs = BrokerCosts(config.commission_bps, config.tax_bps_on_sell, config.slippage_bps)
    broker = SimBroker(100_000_000, costs, buying_power_leverage=config.intraday_leverage, capacity_policy=policy)
    adapter = KALCBReplayAdapter(config, snapshots, initial_equity=100_000_000, costs=costs)
    conn = sqlite3.connect(f"file:{args.database.as_posix()}?mode=ro", uri=True)
    conn.execute("PRAGMA cache_size=-8192")
    dates = [r[0] for r in conn.execute("SELECT DISTINCT day FROM bars WHERE day BETWEEN ? AND ? ORDER BY day", (args.start,args.end))]
    decisions = Counter()
    day_rows = []
    for index, day in enumerate(dates):
        snapshot = snapshots.get(date.fromisoformat(day))
        symbols = {c.symbol for c in snapshot.candidates} if snapshot else set()
        symbols.update(symbol for _, symbol in broker.positions)
        symbols.update(order.symbol for order in broker.orders)
        if symbols:
            placeholders = ",".join("?" for _ in symbols)
            rows = conn.execute(f"SELECT symbol,ts,o,h,l,c,v FROM bars WHERE day=? AND symbol IN ({placeholders}) ORDER BY ts,symbol", (day,*sorted(symbols)))
            for stamp, batch in groupby(rows, key=lambda r:r[1]):
                bars = tuple(MarketBar(r[0], datetime.fromisoformat(r[1]), "5m", *r[2:],
                    source="kis_krx_parquet", source_fingerprint=provenance["database_sha256"]) for r in batch)
                for event in drive_timestamp_batch(broker, bars, (adapter,)):
                    decisions[event.reason] += 1
        equity = broker.equity_curve[-1] if broker.equity_curve else 100_000_000
        day_rows.append({"day":day,"equity":equity,"inventory":{str(k):p.qty for k,p in broker.positions.items()},
                         "pending":len(broker.orders)})
        if index % 20 == 0:
            print(json.dumps({"session":index+1,"of":len(dates),"day":day,"equity":equity}),flush=True)
    conn.close()
    adapter._sync_new_fills(broker)
    trades = _collapse_exit_legs(broker.trades)
    metrics = compute_trade_metrics(trades, broker.equity_curve, timestamps=broker.timestamps,
        mark_coverage=broker.mark_coverage, initial_equity=100_000_000)
    identity = {"scope":"execution-accounting; frozen source verified by bounded feed"}
    if args.audit_manifest:
        if args.snapshot_override:
            raise ValueError("Formal audit requires the independently frozen candidate-source pool")
        audit_generation = InputManifest(args.audit_manifest)
        if audit_generation.payload["source_generation_hash"] != source:
            raise ValueError("Audit manifest must bind the executing source generation")
        for name, value in provenance["source_files"].items():
            if audit_generation.digest(audit_generation.root / name) != value:
                raise ValueError("Audit generation differs from the database's source files")
        pool_name = "data/backtests/output/kalcb/round_1/guarded_q3_full_diagnostics/train_guarded_prefilter_pool_rows.jsonl"
        pool = [json.loads(line) for line in audit_generation.read(audit_generation.root / pool_name).splitlines()]
        expected_keys = {(r["trade_date"], r["symbol"]) for r in pool if args.start <= r["trade_date"] <= args.end}
        actual_keys = {(day.isoformat(), c.symbol) for day, snap in snapshots.items() for c in snap.candidates}
        if expected_keys != actual_keys:
            raise ValueError("Frozen candidate-source keys differ from replay snapshots")
        for path, value in snapshot_hashes.items():
            relative = Path(path).relative_to(generation.root)
            if audit_generation.digest(audit_generation.root / relative) != value:
                raise ValueError("Audit generation differs from frozen replay snapshots")
        broker.qualification_day_dispositions = {"KALCB": {day: "builder_verified_empty"
            for day in dates if date.fromisoformat(day) not in snapshots}}
        configuration = dict(reference["metrics"]["audit_receipt"]["identity"]["configuration"])
        # Preserve the independently declared strategy universe. The shared
        # source archive also contains symbols from other strategies.
        if configuration.get("symbols"):
            symbols = configuration["symbols"]
        else:
            import yaml
            name = Path(configuration["universe"])
            universe_path = next(p for p in (audit_generation.root/name,audit_generation.root/"config"/name)
                                 if audit_generation.digest(p))
            symbols = yaml.safe_load(audit_generation.read(universe_path))["symbols"]
        if not {str(s).zfill(6) for s in symbols} <= set(provenance["counts"]):
            raise ValueError("Declared universe escapes the verified market database")
        configuration.update(date_range={"start":args.start,"end":args.end},start=args.start,end=args.end,
            input_manifest_path=str(args.audit_manifest.resolve()),
            capacity_scenario={"name":policy.name,"participation":args.participation},
            entry_require_complete_prefix=config.entry_require_complete_prefix)
        identity.update(input_manifest=audit_generation.receipt(), configuration=configuration, config=repr(config))
        print(json.dumps({"audit":"standard_provenance_and_coverage_started"}),flush=True)
    audit = audit_broker(broker, identity=identity)
    write(args.output / "execution_audit.json", audit)
    write(args.output / "trades.json", [t.to_json_dict() for t in trades])
    write(args.output / "fills.json", [asdict(f) for f in broker.fills])
    write(args.output / "daily.json", day_rows)
    write(args.output / "decisions.json", dict(decisions))
    write(args.output / "equity.json", list(zip(map(str,broker.timestamps),broker.equity_curve)))
    if source_generation_hash() != source:
        raise RuntimeError("Source changed during research replay")
    residuals = audit["audit_receipt"]["continuous_session_end_inventory"]
    result = {"scope":"Full-period shared-core portfolio research; no independent alpha or promotion claim",
        "source_generation_hash":source,"return_pct":100*(equity/100_000_000-1),
        "drawdown_pct":100*metrics["max_drawdown_pct"],"trades":len(trades),"sessions":len(dates),
        "active_days":len({t.entry_fill_time.date() for t in trades}),
        "residual_sessions":{d:q for d,q in residuals.items() if q},
        "terminal_inventory":day_rows[-1]["inventory"],"terminal_pending":day_rows[-1]["pending"],
        "execution_validity_pass":audit["execution_validity_pass"],
        "formal_qualification_status":audit.get("audit_status"),
        "input_hashes_verified":True,"snapshot_count":len(snapshots),"metrics":metrics}
    write(args.output / "result.json", result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("metrics","residual_sessions")}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["build-market","replay"])
    parser.add_argument("--code-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--audit-manifest",type=Path, help="Optional successor input generation bound to this executing code")
    parser.add_argument("--database",type=Path,required=True)
    parser.add_argument("--start",default="2025-05-12")
    parser.add_argument("--end",default="2026-03-31")
    parser.add_argument("--snapshots",type=Path)
    parser.add_argument("--snapshot-override",type=Path)
    parser.add_argument("--reference",type=Path)
    parser.add_argument("--mutations",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--participation",type=float,default=.01)
    parser.add_argument("--slippage",type=float,default=3.)
    parser.add_argument("--legacy-lock",action="store_true")
    args=parser.parse_args()
    sys.path.insert(0,str(args.code_root.resolve()))
    if args.mode == "build-market":
        build_market(args.manifest,args.database,args.start,args.end)
    else:
        replay(args)


if __name__ == "__main__":
    main()
