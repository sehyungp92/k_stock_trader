"""Full-account replay of authenticated chronological KALCB candidate policies."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import date
from itertools import groupby
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb import executed_value as ev


def equity_records(timestamps, equity_curve):
    """Pair each observed timestamp with its mark, excluding initial capital."""
    if len(equity_curve) != len(timestamps) + 1:
        raise ValueError("Equity must contain initial capital plus one mark per timestamp")
    return [(str(timestamp), value) for timestamp, value in zip(timestamps, equity_curve[1:])]


def verify_policy(directory, name):
    completion = json.loads((directory / "completion.json").read_text())
    if not completion.get("complete"):
        raise ValueError("Uncompleted policy generation")
    manifest = json.loads((directory / "policies_manifest.json").read_text())
    if name not in manifest["policies"]:
        raise ValueError("Undeclared candidate policy")
    for path, expected in manifest["files"].items():
        if ev.digest(directory / path) != expected:
            raise ValueError(f"Changed policy artifact: {path}")
    chronology = json.loads((directory / "chronology.json").read_text())
    from datetime import datetime
    for fold in chronology["folds"]:
        if fold["train_through"] >= fold["valid_from"] or datetime.fromisoformat(fold["label_available_through"]) >= datetime.fromisoformat(fold["fit_at"]):
            raise ValueError("Noncausal fit provenance")
    for calibration in chronology.get("calibration", []):
        latest = calibration.get("latest_label_available_at")
        cutoff = datetime.fromisoformat(calibration["valid_from"] + "T00:00:00+09:00")
        if latest and datetime.fromisoformat(latest) >= cutoff:
            raise ValueError("Noncausal admission calibration")
    from strategy_kalcb.models import KALCBDailySnapshot
    raw = json.loads((directory / (name + ".snapshots.json")).read_text())
    snapshots = {}
    for row in raw:
        snap = KALCBDailySnapshot.from_json_dict(row)
        if snap.artifact_hash != row["artifact_hash"] or snap.trade_date in snapshots:
            raise ValueError("Changed/duplicate snapshot")
        snapshots[snap.trade_date] = snap
    decisions = ev.read_jsonl(directory / (name + ".admission.jsonl"))
    if len({r["opportunity_id"] for r in decisions}) != len(decisions):
        raise ValueError("Duplicate admission decision")
    admitted = {r["opportunity_id"]: r for r in decisions if r["reason"] == "admitted"}
    actual = {f"{day}/{c.symbol}": c for day, s in snapshots.items() for c in s.candidates}
    if actual.keys() != admitted.keys():
        raise ValueError("Snapshot/causal admission mismatch")
    for key, candidate in actual.items():
        if candidate.selection_score != admitted[key]["score"]:
            raise ValueError("Snapshot/forecast score mismatch")
    for snapshot in snapshots.values():
        if len(snapshot.candidates) > manifest["contract"]["budget"]:
            raise ValueError("Candidate budget exceeded")
    return manifest, snapshots


def timed_policy(directory, manifest, snapshots):
    """Validate additional decision-time snapshots without changing the opening pool."""
    from datetime import datetime
    from strategy_kalcb.models import KALCBDailySnapshot
    filename=manifest.get('timed_snapshot_file')
    if not filename:return {}
    if filename not in manifest['files']:raise ValueError('Unbound intraday candidates')
    admissions=manifest.get('timed_admission_file')
    if not admissions or admissions not in manifest['files']:raise ValueError('Unbound intraday decisions')
    decisions=ev.read_jsonl(directory/admissions)
    expected={(r['trade_date'],r['symbol'],r['available_at']):r for r in decisions}
    if len(expected)!=len(decisions):raise ValueError('Duplicate intraday decision')
    used=set(); result={}
    for raw in ev.read_jsonl(directory/filename):
        snap=KALCBDailySnapshot.from_json_dict(raw)
        if snap.artifact_hash!=raw['artifact_hash']:raise ValueError('Changed intraday snapshot')
        if snap.trade_date not in snapshots or snap.generated_at is None:raise ValueError('Undeclared intraday session')
        prior=result.setdefault(snap.trade_date,[])
        if prior and prior[-1].generated_at>=snap.generated_at:raise ValueError('Unordered intraday updates')
        if len(snap.candidates)>manifest['contract']['timed_budget']:raise ValueError('Intraday budget exceeded')
        base=snapshots[snap.trade_date].by_symbol()
        symbols=[c.symbol for c in snap.candidates]
        if len(set(symbols))!=len(symbols) or not set(base).issubset(symbols):
            raise ValueError('Intraday update removes or duplicates opening candidates')
        for c in snap.candidates:
            if c.symbol in base and c.to_json_dict()==base[c.symbol].to_json_dict():continue
            if c.symbol in base:raise ValueError('Intraday update changes locked opening candidate')
            if c.symbol not in manifest.get('observation_symbols',[]):raise ValueError('Undeclared watch symbol')
            key=(str(snap.trade_date),c.symbol,str(c.metadata.get('entry_checkpoint_at')))
            if key not in expected or key in used:raise ValueError('Missing/duplicate timed admission')
            row=expected[key]
            if c.selection_score!=row['score'] or datetime.fromisoformat(key[2])!=snap.generated_at:
                raise ValueError('Intraday score/clock mismatch')
            if datetime.fromisoformat(c.metadata['feature_available_at'])>snap.generated_at:
                raise ValueError('Future features in intraday selection')
            used.add(key)
        prior.append(snap)
    if used!=set(expected):raise ValueError('Unused intraday admission')
    return result


def observation_prefix(bars,day,execution_symbols,minutes):
    """Watch every declared symbol through the fixed clock, then execution names.

    Later admissions cannot alter the observed universe before that clock.
    Unused symbols after the last decision cannot generate orders or inventory.
    """
    from datetime import datetime,timedelta
    from kis_core.trading_calendar import get_trading_calendar
    if minutes not in (45,60):raise ValueError('Undeclared watch horizon')
    opening=get_trading_calendar().session_hours(day)[0]
    cutoff=datetime.combine(day,opening,ev.KST)+timedelta(minutes=minutes)
    return [b for b in bars if b.symbol in execution_symbols or b.timestamp+timedelta(minutes=5)<=cutoff]


def validate_watch_universe(observation_symbols,declared_symbols):
    extra=set(observation_symbols)-set(declared_symbols)
    if extra:raise ValueError('Watch symbols outside declared source universe: '+','.join(sorted(extra)))


def audit_configuration(plan):
    """Use an explicitly bound evaluation context for a new dated window."""
    if plan.get('audit_configuration'):
        path = Path(plan['audit_configuration'])
        expected = plan.get('input_sha256', {}).get(str(path))
        if not expected or ev.digest(path) != expected:
            raise ValueError('Unbound or changed evaluation audit configuration')
        return json.loads(path.read_text(encoding='utf-8'))
    reference = json.loads(Path(plan['historical_reference']).read_text())
    return dict(reference['metrics']['audit_receipt']['identity']['configuration'])


def run(plan, policies, name, scenario, output, audit_manifest=None):
    from backtests.core.input_manifest import InputManifest
    from backtests.auto.shared.cache_keys import source_generation_hash
    from backtests.engine.capacity import BarParticipationScenario
    from backtests.engine.replay import drive_timestamp_batch
    from backtests.engine.sim_broker import SimBroker, BrokerCosts
    from backtests.engine.integrity import audit_broker
    from backtests.analysis.metrics import compute_trade_metrics
    from backtests.strategies.kalcb.runner import KALCBReplayAdapter, _collapse_exit_legs
    from scripts.kalcb_executed_value_study import config_for, connect, load_day, emit
    import gzip
    policy_manifest, snapshots = verify_policy(policies, name)
    import yaml
    declared=tuple(str(s).zfill(6) for s in yaml.safe_load(Path(plan['universe']).read_text())['symbols'])
    validate_watch_universe(policy_manifest.get('observation_symbols',()),declared)
    updates = timed_policy(policies, policy_manifest, snapshots)
    cfg = config_for(plan, scenario)
    if updates and not cfg.entry_checkpoint_enabled:raise ValueError('Timed candidates require checkpoint execution')
    participation, slippage, scenario_name = ev.SCENARIOS[scenario]
    costs = BrokerCosts(cfg.commission_bps, cfg.tax_bps_on_sell, slippage)
    policy = BarParticipationScenario(participation, scenario_name)
    broker = SimBroker(ev.REFERENCE_EQUITY, costs, buying_power_leverage=cfg.intraday_leverage, capacity_policy=policy)
    adapter = KALCBReplayAdapter(cfg, snapshots, initial_equity=ev.REFERENCE_EQUITY, costs=costs,
        timed_snapshots=updates, observation_symbols=tuple(policy_manifest.get('observation_symbols',())))
    conn = connect(plan)
    start, end = policy_manifest["evaluation_from"], policy_manifest["evaluation_through"]
    days = [r[0] for r in conn.execute("SELECT DISTINCT day FROM bars WHERE day BETWEEN ? AND ? ORDER BY day", (start, end))]
    if {d.isoformat() for d in snapshots} != set(days):
        raise ValueError("Candidate policy does not account for every evaluation session")
    source = plan["input_sha256"][plan["database"]]
    daily, reasons, journal_cursor = [], Counter(), 0
    with gzip.open(output / "decisions.jsonl.gz", "xt", encoding="utf-8", compresslevel=1) as decisions, gzip.open(output / "orders.jsonl.gz", "xt", encoding="utf-8", compresslevel=1) as orders:
        for index, day in enumerate(days):
            snapshot = snapshots[date.fromisoformat(day)]
            symbols = {c.symbol for c in snapshot.candidates}
            for update in updates.get(date.fromisoformat(day),()):symbols.update(c.symbol for c in update.candidates)
            symbols.update(s for _, s in broker.positions)
            symbols.update(o.symbol for o in broker.orders)
            if not symbols:
                # Observe a real, declared-universe symbol on cash-only sessions.
                # This maintains the calendar/mark timeline without inventing fills.
                symbols = {"000100"}
            execution_symbols=set(symbols)
            symbols.update(policy_manifest.get('observation_symbols',()))
            bars = load_day(conn, day, source, symbols)
            if policy_manifest.get('observation_minutes'):
                bars=observation_prefix(bars,date.fromisoformat(day),execution_symbols,policy_manifest['observation_minutes'])
            for _, batch in groupby(bars, key=lambda b: b.timestamp):
                for event in drive_timestamp_batch(broker, tuple(batch), (adapter,)):
                    reasons[event.reason] += 1
                    emit(decisions, event.to_json_dict())
            for record in broker.journal[journal_cursor:]:
                if record["event"] in {"submit", "amend", "cancel", "reject", "expire"}:
                    emit(orders, record)
            journal_cursor = len(broker.journal)
            daily.append({"day": day, "equity": broker.equity_curve[-1],
                "inventory": {str(k):p.qty for k,p in broker.positions.items()}, "pending": len(broker.orders)})
            if index % 20 == 0:
                print(json.dumps({"policy": name, "scenario": scenario, "session": index + 1, "equity": broker.equity_curve[-1]}), flush=True)
    conn.close()
    adapter._sync_new_fills(broker)
    trades = _collapse_exit_legs(broker.trades)
    metrics = compute_trade_metrics(trades, broker.equity_curve, timestamps=broker.timestamps,
        mark_coverage=broker.mark_coverage, initial_equity=ev.REFERENCE_EQUITY)
    identity = {"scope": "executed-value chronological development", "candidate_policy": name,
        "candidate_policy_manifest_sha256": ev.digest(policies / "policies_manifest.json"), "config": repr(cfg)}
    if audit_manifest:
        generation = InputManifest(audit_manifest)
        if generation.payload["source_generation_hash"] != source_generation_hash():
            raise ValueError("Audit source generation differs from executing source")
        evidence = generation.payload["evidence"].get("executed_value_policy")
        if not evidence:
            raise ValueError("Missing independently bound derived-policy evidence")
        bound = generation.root / evidence["policies_manifest"]
        if generation.read(bound) != (policies / "policies_manifest.json").read_bytes():
            raise ValueError("Derived policy manifest binding mismatch")
        for relative, expected in policy_manifest["files"].items():
            path = bound.parent / relative
            if generation.digest(path) != expected or ev.digest(path) != expected:
                raise ValueError("Derived policy dependency not bound")
        configuration = audit_configuration(plan)
        if plan.get('audit_configuration'):
            configuration.update(commission_bps=cfg.commission_bps,
                tax_bps_on_sell=cfg.tax_bps_on_sell,slippage_bps=slippage)
        configuration.update(date_range={"start": start, "end": end}, start=start, end=end,
            input_manifest_path=str(audit_manifest.resolve()),
            capacity_scenario={"name": scenario_name, "participation": participation},
            entry_require_complete_prefix=cfg.entry_require_complete_prefix)
        if updates:
            if policy_manifest.get('observation_minutes')!=45:raise ValueError('Unsupported audited candidate decision horizon')
            configuration['source_dependency_scope']={'version':'causal-timed-candidates-v1','strategy':'kalcb',
                'feature_intraday_before':'09:45','candidate_intraday_before':'15:20'}
        identity.update(configuration=configuration, input_manifest=generation.receipt())
        print(json.dumps({"standard_audit": "started", "policy": name}), flush=True)
    audit = audit_broker(broker, identity=identity)
    inventory = audit["audit_receipt"]["continuous_session_end_inventory"]
    result = {"policy": name, "scenario": scenario, "return_pct": 100*(broker.equity_curve[-1]/ev.REFERENCE_EQUITY-1),
        "drawdown_pct": 100*metrics["max_drawdown_pct"], "trades": len(trades), "sessions": len(days),
        "active_days": len({t.entry_fill_time.date() for t in trades}),
        "residual_sessions": {d:q for d,q in inventory.items() if q},
        "terminal_inventory": daily[-1]["inventory"], "terminal_pending": len(broker.orders),
        "execution_validity_pass": audit["execution_validity_pass"], "audit_status": audit["audit_status"],
        "metrics": metrics, "decision_counts": reasons, "source_generation_hash": source_generation_hash(),
        "equity_export_version": "timestamped-marks-v2"}
    ev.write_json(output / "result.json", result)
    ev.write_json(output / "execution_audit.json", audit)
    ev.write_json(output / "trades.json", [t.to_json_dict() for t in trades])
    ev.write_json(output / "fills.json", [asdict(f) for f in broker.fills])
    ev.write_json(output / "daily.json", daily)
    ev.write_json(output / "equity.json", equity_records(broker.timestamps, broker.equity_curve))
    print(json.dumps({k:v for k,v in result.items() if k not in {"metrics", "decision_counts"}}), flush=True)


def main():
    from backtests.auto.shared.cache_keys import source_generation_hash
    from scripts.kalcb_executed_value_study import load_plan
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--policies", required=True, type=Path)
    p.add_argument("--policy", required=True)
    p.add_argument("--scenario", choices=ev.SCENARIOS, default="baseline")
    p.add_argument("--audit-manifest", type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.audit_manifest:
        from backtests.core.input_manifest import InputManifest
        frozen_root = InputManifest(args.audit_manifest).root.resolve()
        if Path(__file__).resolve().parents[1] != frozen_root:
            raise ValueError(f"Run {frozen_root / 'scripts/kalcb_executed_value_replay.py'} so calendar dependencies resolve inside the audited bundle")
    plan, source = load_plan(args.plan), source_generation_hash()
    args.output.mkdir(parents=True, exist_ok=False)
    ev.write_json(args.output / "request.json", {"arguments": vars(args), "plan": plan, "source_generation_hash": source})
    run(plan, args.policies, args.policy, args.scenario, args.output, args.audit_manifest)
    if source_generation_hash() != source:
        raise RuntimeError("Source changed during replay")
    load_plan(args.plan)
    ev.write_json(args.output / "completion.json", {"complete": True, "source_generation_hash": source})


if __name__ == "__main__":
    main()
