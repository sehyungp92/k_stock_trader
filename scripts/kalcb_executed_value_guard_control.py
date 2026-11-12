"""Hold forecasts fixed while reproducing the historical guard's CPR semantics."""
from __future__ import annotations
import argparse
from itertools import groupby
import json
from pathlib import Path
import shutil
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("code-root", "plan", "prepared", "policies", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.code_root))
    from backtests.strategies.kalcb import executed_value as ev
    from backtests.auto.shared.cache_keys import source_generation_hash
    from scripts.kalcb_executed_value_study import load_plan, emit
    plan, source = load_plan(a.plan), source_generation_hash()
    parent = json.loads((a.policies / "policies_manifest.json").read_text())
    for name, expected in parent["files"].items():
        if ev.digest(a.policies / name) != expected:
            raise ValueError("Changed source forecast policy")
    scores = {(r["opportunity_id"], r["target"]): r for r in ev.read_jsonl(a.policies / "forecasts.jsonl")}
    seed = json.loads(Path(plan["seed"]).read_text())["mutations"]
    q3 = json.loads(Path(plan["q3"]).read_text())["mutations"]
    provenance = json.loads((a.prepared / "prepared.json").read_text())
    if ev.digest(a.prepared / "opportunities.jsonl") != provenance["opportunities_sha256"]:
        raise ValueError("Changed opportunity features")
    a.output.mkdir(parents=True, exist_ok=False)
    names = [t + "_" + order + "_legacy_guard" for t in ("quality", "net") for order in ("after", "before")]
    snapshots = {name: [] for name in names}
    streams = {name: (a.output / (name + ".admission.jsonl")).open("x", encoding="utf-8") for name in names}
    def predicate(row, rank):
        # This is the historical formula, rebuilt from decision-time data. The
        # execution engine retains its separate, actual signal-bar CPR feature.
        adapted = {**row, "guard_features": {**row["guard_features"],
            "first30_signal_bar_cpr": row["features"]["first30_close_location"]}}
        return ev.eligible(adapted, seed, q3, rank=rank)
    with (a.prepared / "opportunities.jsonl").open(encoding="utf-8") as stream:
        for day, group in groupby(map(json.loads, stream), key=lambda r: r["trade_date"]):
            rows = list(group)
            if not parent["evaluation_from"] <= day <= parent["evaluation_through"]:
                continue
            for name in names:
                target, construction = name.split("_")[:2]
                predictions = [scores[(r["opportunity_id"], target)]["score"] for r in rows]
                selected, reasons = ev.admit(rows, predictions, predicate, construction=construction,
                    budget=parent["contract"]["budget"], shortlist=parent["contract"]["shortlist_before_guard"])
                first = scores[(rows[0]["opportunity_id"], target)]["forecast_valid_from"]
                proof = {"policy": name, "valid_from": first, "guard_cpr_semantics": "historical_whole_opening_range"}
                snapshots[name].append(ev.snapshot_for(day, selected, source=provenance["opportunities_sha256"], forecast=proof).to_json_dict())
                for row, score in zip(rows, predictions):
                    emit(streams[name], {"opportunity_id": row["opportunity_id"], "trade_date": day,
                        "score": score, "reason": reasons[row["opportunity_id"]], **proof})
    for stream in streams.values():
        stream.close()
    for name, values in snapshots.items():
        ev.write_json(a.output / (name + ".snapshots.json"), values)
    for name in parent["files"]:
        if name in {"forecasts.jsonl", "chronology.json"} or name.endswith(".joblib"):
            shutil.copy2(a.policies / name, a.output / name)
    shutil.copy2(__file__, a.output / "guard_control_builder.py")
    ev.write_json(a.output / "parent_manifest.json", parent)
    ev.write_json(a.output / "policies_manifest.json", {**parent, "policies": names,
        "files": {f.name: ev.digest(f) for f in a.output.iterdir() if f.is_file()},
        "counts": {name: sum(len(s["candidates"]) for s in ss) for name, ss in snapshots.items()},
        "contract": {**parent["contract"], "guard_cpr_semantics": "historical_whole_opening_range"},
        "comparison": "Identical validated-label model forecasts; only guard CPR formula changes; no outcome fitting"})
    if source_generation_hash() != source:
        raise RuntimeError("Frozen runtime changed")
    ev.write_json(a.output / "completion.json", {"complete": True, "source_generation_hash": source,
        "builder_sha256": ev.digest(__file__)})
    print(json.dumps({"guard_controls": names}), flush=True)


if __name__ == "__main__":
    main()
