"""Prepare causal core actions and compare fixed action-aware KALCB selectors.

Research outputs are append-only. Existing labels and execution controls remain
intact. Full-account replay uses kalcb_executed_value_replay.py unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timedelta
import json
import math
from itertools import groupby
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb import action_value as av, executed_value as ev
from scripts.kalcb_executed_value_study import config_for, connect, load_day, emit, load_plan


def action_features(preview, prefix, scale=1.0):
    last = prefix[-1]
    intent = next(iter(preview["intents"]), {})
    meta = intent.get("metadata", {})
    qty = intent.get("quantity", 0)
    price = float(meta.get("entry_price_ref") or last.close)
    vwap = sum((b.high+b.low+b.close)/3*b.volume for b in prefix)/max(sum(b.volume for b in prefix), 1)
    votes = float(meta.get("entry_quality_votes") or 0)
    low_volume = sorted(b.volume for b in prefix)[min(1, len(prefix)-1)]
    route = str(meta.get("entry_route") or "")
    return {"action_signal_cpr": (last.close-last.low)/max(last.high-last.low, 1e-9),
        "action_avwap_distance": price/float(meta.get("avwap") or vwap)-1 if vwap else 0,
        "action_quality_votes": votes,
        "action_quality_margin": votes-float(meta.get("entry_quality_required_votes") or 0),
        "action_risk_fraction": float(meta.get("risk_per_share") or 0)/max(price, 1),
        "action_notional_equity": qty*price/ev.REFERENCE_EQUITY,
        "action_quantity_last_volume": qty/max(last.volume, 1),
        "action_quantity_opening_volume": qty/max(sum(b.volume for b in prefix), 1),
        "action_low_volume_exit_slices": qty/max(math.floor(low_volume*.005)*.5, 1),
        "action_momentum_score": float(meta.get("momentum_score") or 0),
        "action_is_anchor": float(route == "first30_open_anchor"),
        "action_is_pullback": float("pullback" in route), "action_quantity_scale": scale}


def prepare(plan, prepared, output):
    import gzip
    import pandas as pd
    from scripts import kalcb_r_capture_optimizer as original
    from strategy_kalcb.core.preview import preview_entry
    from strategy_kalcb.core.core_models import KALCBPortfolioView
    provenance = json.loads((prepared / "prepared.json").read_text())
    if ev.digest(prepared / "opportunities.jsonl") != provenance["opportunities_sha256"]:
        raise ValueError("Changed prepared opportunities")
    frame = pd.read_csv(plan["pipeline"], dtype={"symbol": str})
    for key in ("daily_above_sma20", "daily_above_sma60", "flow_available"):
        if key in frame:
            frame[key] = original.bool_to_float(frame[key])
    frame = original.engineer_pipeline_features(frame)
    columns = original.FEATURE_GROUPS["all_context"]
    rich = {(str(r.trade_date), str(r.symbol).zfill(6)): {k:getattr(r,k,None) for k in columns}
            for r in frame.itertuples(index=False)}
    del frame
    cfg, conn = config_for(plan), connect(plan)
    portfolio = KALCBPortfolioView(cash=ev.REFERENCE_EQUITY, equity=ev.REFERENCE_EQUITY)
    counts = Counter()
    with (output / "opportunities.jsonl").open("x", encoding="utf-8") as stream, \
         (prepared / "opportunities.jsonl").open(encoding="utf-8") as source_rows, \
         gzip.open(output / "previews.events.jsonl.gz","xt",encoding="utf-8",compresslevel=1) as events:
        for index, (day, grouped) in enumerate(groupby(map(json.loads,source_rows),key=lambda r:r["trade_date"])):
            items = list(grouped)
            bars = defaultdict(list)
            cutoff = datetime.fromisoformat(items[0]["feature_available_at"])
            for b in load_day(conn, day, plan["input_sha256"][plan["database"]]):
                if b.timestamp+timedelta(minutes=5) <= cutoff:
                    bars[b.symbol].append(b)
            for row in items:
                snapshot = ev.snapshot_for(day, [(row, 0)], source=provenance["opportunities_sha256"])
                preview = preview_entry(bars[row["symbol"]], snapshot, cfg, portfolio, available_at=cutoff)
                extra = rich.get((day, row["symbol"]), {})
                features = dict(row["features"])
                for k in columns:
                    value = extra.get(k)
                    if k not in features:
                        features[k] = float(value) if value is not None and math.isfinite(float(value)) else None
                features.update(action_features(preview, bars[row["symbol"]]))
                emit(events,{"opportunity_id":row["opportunity_id"],**preview})
                row.update(features=features, action={"ready":preview["ready"],
                    "quantity":sum(a["quantity"] for a in preview["intents"]),
                    "routes":[a["metadata"].get("entry_route") for a in preview["intents"]],
                    "reasons":[d["reason"] for d in preview["decisions"]]},
                    opening_bars=[{"symbol":b.symbol, "timestamp":b.timestamp.isoformat(),
                        **{k:getattr(b,k) for k in ("open","high","low","close","volume")}} for b in bars[row["symbol"]]])
                emit(stream, row)
                counts["opportunities"] += 1
                counts["opening_ready"] += preview["ready"]
                counts["guard_and_ready"] += preview["ready"] and row["guard_at_rank1"]
                counts["rich_source_missing"] += not bool(extra)
            if index % 20 == 0:
                print(json.dumps({"prepared_session":index+1, **counts}), flush=True)
    conn.close()
    ev.write_json(output / "prepared.json", {**provenance, "version": av.VERSION,
        "opportunities_sha256":ev.digest(output / "opportunities.jsonl"),
        "parent_prepared":str(prepared), "parent_prepared_sha256":ev.digest(prepared / "prepared.json"),
        "counts":counts, "action_contract":"core preview, rank 1, reference equity; no outcome eligibility",
        "preview_events_sha256":ev.digest(output / "previews.events.jsonl.gz"),
        "rich_source_sha256":ev.digest(plan["pipeline"])})


def recheck_actions(day, selected, cfg, source):
    """Check actual selected ranks/context using only saved completed prefix bars."""
    from strategy_common.market import MarketBar
    from strategy_kalcb.core.preview import preview_entry
    from strategy_kalcb.core.core_models import KALCBPortfolioView
    snapshot = ev.snapshot_for(day, selected, source=source)
    bars = [MarketBar(r["symbol"], datetime.fromisoformat(b["timestamp"]), "5m",
        *[b[k] for k in ("open","high","low","close","volume")]) for r, _ in selected for b in r["opening_bars"]]
    preview = preview_entry(bars, snapshot, cfg,
        KALCBPortfolioView(cash=ev.REFERENCE_EQUITY, equity=ev.REFERENCE_EQUITY),
        available_at=snapshot.generated_at)
    return {a["symbol"] for a in preview["intents"]}


def admit_actions(items, predicate, cfg, source, *, budget=8, threshold=0.0, ready=True, priority_keys=()):
    # A causal admission tier preserves candidate slots, not broker fill priority.
    # Scores remain genuine forecasts; no artificial score bump bypasses guards.
    priority_keys = frozenset(priority_keys)
    ordered = sorted(items, key=lambda pair:(pair[0]["opportunity_id"] not in priority_keys, -pair[1], pair[0]["symbol"]))
    selected, reasons = [], {}
    for row, score in ordered:
        key = row["opportunity_id"]
        if not math.isfinite(score):
            raise ValueError("Invalid forecast")
        if ready and not row["action"]["ready"]:
            reasons[key] = "opening_not_ready"
        elif not predicate(row, len(selected)+1):
            reasons[key] = "guard"
        elif threshold is not None and score <= threshold:
            reasons[key] = "net_value_threshold"
        elif len(selected) >= budget:
            reasons[key] = "admission_budget"
        else:
            proposed = selected + [(row, score)]
            if ready:
                ready_symbols = recheck_actions(row["trade_date"], proposed, cfg, source)
                if any(r["symbol"] not in ready_symbols for r, _ in proposed):
                    reasons[key] = "selected_rank_or_context_not_ready"
                    continue
            selected = proposed
            reasons[key] = "admitted"
    return selected, reasons


def fit(plan, prepared, label_paths, output):
    import joblib
    from threadpoolctl import threadpool_limits
    from scripts.kalcb_executed_value_ranker import load_labels
    provenance = json.loads((prepared / "prepared.json").read_text())
    if ev.digest(prepared / "opportunities.jsonl") != provenance["opportunities_sha256"]:
        raise ValueError("Changed action input")
    with (prepared / "opportunities.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line,object_hook=lambda d:{sys.intern(k):v for k,v in d.items()}) for line in stream]
    labels, manifests = load_labels(label_paths, provenance["parent_prepared_sha256"])
    if set(labels) != {r["opportunity_id"] for r in rows}:
        raise ValueError("Labels do not cover the causal opportunity universe")
    seed = json.loads(Path(plan["seed"]).read_text())["mutations"]
    q3 = json.loads(Path(plan["q3"]).read_text())["mutations"]
    predicate = lambda row, rank: ev.eligible(row, seed, q3, rank=rank)
    sessions, cfg = provenance["sessions"], config_for(plan)
    outer = ev.outer_windows(sessions, min_train=60, block=20)
    names = [s.name for s in av.SPECS] + ["ready_rule", "ready_hgb_unthresholded", "cash"]
    snapshots, decisions = {n:[] for n in names}, {n:[] for n in names}
    proofs = []
    with (output / "forecasts.jsonl").open("x", encoding="utf-8") as forecasts, threadpool_limits(limits=1):
        for fold, (first, through) in enumerate(outer):
            predictions = {}
            for spec in av.SPECS:
                model, scored, proof = av.fit_forecast(rows, labels, sessions, first, through, spec)
                path = output / f"model_{spec.name}_{fold}.joblib"
                joblib.dump(model, path)
                proof.update(model_path=path.name, model_sha256=ev.digest(path))
                proofs.append(proof)
                predictions[spec.name] = defaultdict(list)
                for row, score in scored:
                    predictions[spec.name][row["trade_date"]].append((row,score))
                    emit(forecasts, {"opportunity_id":row["opportunity_id"], "model":spec.name,
                        "score":score, "valid_from":first, "train_through":proof["train_through"]})
            for day in [d for d in sessions if first <= d <= through]:
                daily_rows = [r for r,_ in predictions[av.SPECS[0].name][day]]
                for name in names:
                    if name == "cash":
                        items = [(r,0.0) for r in daily_rows]
                        selected, reasons = [], {r["opportunity_id"]:"cash_policy" for r in daily_rows}
                    else:
                        items = ([(r,av.continuation_score(r)) for r in daily_rows] if name == "ready_rule" else
                            predictions["ready_hgb" if name == "ready_hgb_unthresholded" else name][day])
                        selected, reasons = admit_actions(items,predicate,cfg,provenance["opportunities_sha256"],
                            threshold=None if name == "ready_hgb_unthresholded" else 0.0,
                            ready=name != "unconditional_ridge")
                    proof = {"policy":name,"valid_from":first,"threshold":None if name.endswith("unthresholded") else 0}
                    snapshots[name].append(ev.snapshot_for(day,selected,
                        source=provenance["opportunities_sha256"],forecast=proof).to_json_dict())
                    decisions[name].extend({"opportunity_id":r["opportunity_id"],"trade_date":day,
                        "score":score,"reason":reasons[r["opportunity_id"]],**proof} for r,score in items)
            print(json.dumps({"fold":fold,"from":first,"through":through,
                "admissions":{n:sum(len(s["candidates"]) for s in ss) for n,ss in snapshots.items()}}),flush=True)
    for name in names:
        ev.write_json(output / (name+".snapshots.json"), snapshots[name])
        with (output / (name+".admission.jsonl")).open("x",encoding="utf-8") as stream:
            for row in decisions[name]: emit(stream,row)
    ev.write_json(output / "chronology.json", {"folds":proofs,"scope":"predeclared chronological development; not untouched OOS"})
    files = {p.name:ev.digest(p) for p in output.iterdir() if p.is_file()}
    ev.write_json(output / "policies_manifest.json", {"version":av.VERSION,"files":files,
        "policies":names,"evaluation_from":outer[0][0],"evaluation_through":outer[-1][1],
        "prepared_sha256":ev.digest(prepared / "prepared.json"),"label_manifests":manifests,
        "contract":{"budget":8,"specifications":[asdict(s) for s in av.SPECS],
            "threshold":"fixed zero; additional unthresholded HGB diagnostic",
            "sizing":"unchanged recovered policy", "production_promotion":False},
        "counts":{n:sum(len(s["candidates"]) for s in ss) for n,ss in snapshots.items()}})


def main():
    from backtests.auto.shared.cache_keys import source_generation_hash
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("prepare","fit"))
    for key in ("plan","prepared","output"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--labels",type=Path,action="append",default=[])
    args = p.parse_args()
    plan = load_plan(args.plan)
    source = source_generation_hash()
    args.output.mkdir(parents=True,exist_ok=False)
    ev.write_json(args.output / "request.json", {"arguments":vars(args),"source_generation_hash":source})
    if args.command == "prepare": prepare(plan,args.prepared,args.output)
    else: fit(plan,args.prepared,args.labels,args.output)
    if source_generation_hash() != source:
        raise RuntimeError("Source changed during generation")
    ev.write_json(args.output / "completion.json", {"complete":True,"source_generation_hash":source})


if __name__ == "__main__": main()
