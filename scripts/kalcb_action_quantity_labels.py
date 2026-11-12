"""Replay a declared quantity/scenario grid for causal opening-ready actions.

Quantity outcomes are independently replayed, never scaled from another label.
Only opening information defines membership; future fills cannot admit a row.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime
import gzip
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb import executed_value as ev
from scripts.kalcb_executed_value_study import config_for,connect,load_day,load_plan,run_episode,emit


def scaled_row(row, scale):
    if scale not in (.25,.5,1.0):
        raise ValueError("Undeclared quantity")
    candidate = {**row["candidate"],"metadata":{**row["candidate"]["metadata"],"action_quantity_scale":scale}}
    features = dict(row["features"])
    old_qty = row["action"]["quantity"]
    ratio = int(old_qty*scale)/old_qty if old_qty else 0
    for key in ("action_notional_equity","action_quantity_last_volume","action_quantity_opening_volume","action_low_volume_exit_slices"):
        features[key] *= ratio
    features["action_quantity_scale"] = scale
    return {**row,"candidate":candidate,"features":features,
        "action":{**row["action"],"quantity":int(old_qty*scale)},
        "base_opportunity_id":row["opportunity_id"],
        "opportunity_id":row["opportunity_id"]+f"/q{scale:g}"}


def run(plan, prepared, output, scenario, scales, start_index=0, stop_index=None):
    provenance = json.loads((prepared/"prepared.json").read_text())
    if ev.digest(prepared/"opportunities.jsonl") != provenance["opportunities_sha256"]:
        raise ValueError("Changed causal action population")
    sessions = provenance["sessions"]
    wanted = set(sessions[start_index:stop_index])
    by_day = defaultdict(list)
    with (prepared/"opportunities.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row=json.loads(line)
            if row["trade_date"] in wanted and row["action"]["ready"] and row["guard_at_rank1"]:
                # Retain no-fill attempts and all measured losses.
                by_day[row["trade_date"]].append(row)
    cfg=replace(config_for(plan,scenario),entry_action_quantity_scale_enabled=True)
    conn=connect(plan)
    source=plan["input_sha256"][plan["database"]]
    counts=Counter()
    for index,day in enumerate(sorted(wanted)):
        began=time.monotonic()
        bars=defaultdict(list)
        for bar in load_day(conn,day,source,{r["symbol"] for r in by_day[day]}): bars[bar.symbol].append(bar)
        with (output/(day+".jsonl")).open("x",encoding="utf-8") as values, \
             gzip.open(output/(day+".events.jsonl.gz"),"xt",encoding="utf-8",compresslevel=1) as events:
            for row in by_day[day]:
                for scale in scales:
                    action=scaled_row(row,scale)
                    value,detail=run_episode(action,bars[row["symbol"]],cfg,scenario,source,conn,sessions)
                    value.update(base_opportunity_id=row["opportunity_id"],quantity_scale=scale)
                    emit(values,value);emit(events,detail)
                    counts[value["status"]]+=1
        print(json.dumps({"day":day,"index":index+1,"sessions":len(wanted),
            "scenario":scenario,"seconds":round(time.monotonic()-began,2),**counts}),flush=True)
    conn.close()
    ev.write_json(output/"grid_manifest.json",{"version":"kalcb-action-quantity-v1",
        "scenario":scenario,"scales":scales,"counts":counts,"sessions":sorted(wanted),
        "prepared_sha256":ev.digest(prepared/"prepared.json"),
        "request_sha256":ev.digest(output/"request.json"),
        "files":{p.name:ev.digest(p) for p in output.iterdir() if p.suffix in {".jsonl",".gz"}},
        "membership":"opening ready and original native guard at rank1; future outcomes never filter membership"})


def main():
    from backtests.auto.shared.cache_keys import source_generation_hash
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("plan","prepared","output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--scenario",choices=ev.SCENARIOS,default="baseline")
    p.add_argument("--scales",type=float,nargs="+",default=[.25,.5])
    p.add_argument("--start-index",type=int,default=0)
    p.add_argument("--stop-index",type=int)
    a=p.parse_args()
    if any(q not in (.25,.5,1) for q in a.scales) or len(set(a.scales)) != len(a.scales):
        raise ValueError("Declare unique scales from .25,.5,1")
    plan=load_plan(a.plan);source=source_generation_hash()
    a.output.mkdir(parents=True,exist_ok=False)
    ev.write_json(a.output/"request.json",{"arguments":vars(a),"source_generation_hash":source})
    run(plan,a.prepared,a.output,a.scenario,a.scales,a.start_index,a.stop_index)
    if source_generation_hash()!=source:raise RuntimeError("Source changed during quantity replay")
    ev.write_json(a.output/"completion.json",{"complete":True,"source_generation_hash":source})


if __name__ == "__main__":main()
