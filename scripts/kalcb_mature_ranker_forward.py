"""Bind the original champion family at March 31 and score subsequent sessions.

Checks historical pool membership before reporting this as a reconstruction.
The previously consumed April-May interval is a forward diagnostic, not an
untouched holdout. No forward outcome enters fitting or candidate selection.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb import executed_value as ev


def build(plan, prepared, original_pool, output):
    import joblib
    from threadpoolctl import threadpool_limits
    from scripts import kalcb_r_capture_optimizer as opt
    opt.PIPELINE_ROWS, opt.OUT_DIR = Path(plan["pipeline"]), output
    frame = opt.read_pipeline()
    train = frame.loc[frame.window.eq("train")].copy().reset_index(drop=True)
    forward = frame.loc[frame.window.eq("holdout")].copy().reset_index(drop=True)
    if train.trade_date.max() >= forward.trade_date.min():
        raise ValueError("Training and forward partitions overlap")
    columns = tuple(opt.FEATURE_GROUPS["all_context"])
    mask = opt.scope_mask(train, "dataset")
    target = opt.target_values(train.loc[mask], "eod_quality")
    with threadpool_limits(limits=1):
        model = opt.make_model("hgb")
        model.fit(train.loc[mask, list(columns)], target)
        train_predictions = model.predict(train.loc[:, list(columns)])
        predictions = model.predict(forward.loc[:, list(columns)])
    joblib.dump(model, output / "original_champion.joblib")
    fitted_pool = opt.select_by_budget(train, train_predictions, "dataset", "top16", {})
    actual = {(str(r.trade_date), str(r.symbol).zfill(6)) for r in fitted_pool.itertuples()}
    old_rows = ev.read_jsonl(original_pool)
    expected = {(r["trade_date"], r["symbol"]) for r in old_rows}
    parity = {"original_pool_sha256":ev.digest(original_pool),"reconstructed":len(actual),
        "original":len(expected),"exact_membership_match":actual==expected,
        "missing":sorted(expected-actual),"added":sorted(actual-expected)}
    ev.write_json(output / "historical_pool_reconstruction.json", parity)
    if actual != expected:
        raise ValueError("Original pool reconstruction failed; inspect saved differences")
    provenance = json.loads((prepared / "prepared.json").read_text())
    if ev.digest(prepared / "opportunities.jsonl") != provenance["opportunities_sha256"]:
        raise ValueError("Changed forward features")
    rows = ev.read_jsonl(prepared / "opportunities.jsonl")
    by_key = {(r["trade_date"],r["symbol"]):r for r in rows}
    guard_rows = {(r["trade_date"],str(r["symbol"]).zfill(6)):r
        for r in ev.read_jsonl(plan["forward_guard_features"])} if plan.get("forward_guard_features") else {}
    chosen = opt.select_by_budget(forward,predictions,"dataset","top16",{})
    # Archived guard CPR was opening-range location. The core still computes its
    # actual signal-bar CPR. Preserve that distinction in the mature control.
    seed = json.loads(Path(plan["seed"]).read_text())["mutations"]
    q3 = json.loads(Path(plan["q3"]).read_text())["mutations"]
    selected_by_day, scores = defaultdict(list), {}
    missing = []
    for day, group in chosen.groupby("trade_date",sort=True):
        for rank, (_, item) in enumerate(group.sort_values("_score",ascending=False).iterrows(),1):
            key = (str(day),str(item.symbol).zfill(6))
            row = by_key.get(key)
            if row is None:
                missing.append(key)
                continue
            guard = guard_rows.get(key, row["guard_features"])
            row = {**row,"guard_features":{k:v for k,v in guard.items()
                if k.startswith(("daily_","first30_","sector_","stock_sector_","continuation_"))}}
            row["guard_features"]["first30_signal_bar_cpr"] = float(item.first30_close_location)
            # The historical source uses FINANCIAL, whereas the native sector
            # map can use Financials. Preserve the source guard's exact taxonomy.
            guard_row = {**row,"sector":guard.get("sector",row["sector"])}
            if ev.eligible(guard_row,seed,q3,rank=rank):
                selected_by_day[str(day)].append((row,float(item["_score"])))
            scores[row["opportunity_id"]] = float(item["_score"])
    if missing:
        ev.write_json(output / "missing_forward_features.json", missing)
        raise ValueError("Original forward shortlist lacks causal feature coverage")
    if plan.get("forward_reference_pool"):
        # Reference membership is used only as a reconstruction assertion. It
        # never supplies a score, trading outcome, or selection fallback.
        expected_forward = {(r["trade_date"],str(r["symbol"]).zfill(6))
            for r in ev.read_jsonl(plan["forward_reference_pool"])}
        actual_forward = {(day,r["symbol"]) for day,rs in selected_by_day.items() for r,_ in rs}
        ev.write_json(output / "forward_pool_reconstruction.json", {
            "exact_membership_match":actual_forward==expected_forward,
            "expected":len(expected_forward),"actual":len(actual_forward),
            "missing":sorted(expected_forward-actual_forward),"added":sorted(actual_forward-expected_forward)})
        if actual_forward!=expected_forward:raise ValueError("Forward guard reconstruction differs from recorded pool")
    first, through = provenance["sessions"][0], provenance["sessions"][-1]
    snapshots, dispositions = [], []
    for day in provenance["sessions"]:
        selected = selected_by_day[day]
        snapshots.append(ev.snapshot_for(day,selected,source=provenance["opportunities_sha256"],
            forecast={"policy":"mature_original","fit_at":first+"T00:00:00+09:00"}).to_json_dict())
        for row, score in selected:
            dispositions.append({"opportunity_id":row["opportunity_id"],"trade_date":day,
                "score":score,"reason":"admitted"})
    ev.write_json(output / "mature_original.snapshots.json", snapshots)
    from scripts.kalcb_executed_value_study import emit
    with (output / "mature_original.admission.jsonl").open("x",encoding="utf-8") as stream:
        for row in dispositions: emit(stream,row)
    proof = {"valid_from":first,"valid_through":through,"fit_at":first+"T00:00:00+09:00",
        "train_through":str(train.trade_date.max()),
        "label_available_through":str(train.trade_date.max())+"T18:00:00+09:00",
        "label_contract":"original same-session MFE/EOD/MAE proxy, available after session close",
        "feature_columns":columns,"training_rows":int(mask.sum()),
        "training_target_sha256":ev.signature(target.tolist()),
        "model_sha256":ev.digest(output / "original_champion.joblib")}
    ev.write_json(output / "chronology.json", {"folds":[proof],"scope":"consumed forward diagnostic"})
    files = {p.name:ev.digest(p) for p in output.iterdir() if p.is_file()}
    ev.write_json(output / "policies_manifest.json", {"version":"kalcb-mature-original-v1",
        "files":files,"policies":["mature_original"],"evaluation_from":first,"evaluation_through":through,
        "contract":{"budget":16,"training_cutoff":str(train.trade_date.max()),"untouched_oos":False},
        "prepared_sha256":ev.digest(prepared / "prepared.json"),"admissions":len(dispositions)})


def main():
    from scripts.kalcb_executed_value_study import load_plan
    from backtests.auto.shared.cache_keys import source_generation_hash
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("plan","prepared","original-pool","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    plan=load_plan(a.plan)
    source=source_generation_hash()
    a.output.mkdir(parents=True,exist_ok=False)
    ev.write_json(a.output / "request.json",{"arguments":vars(a),"source_generation_hash":source,
        "plan_sha256":ev.digest(a.plan),"input_sha256":plan["input_sha256"]})
    build(plan,a.prepared,a.original_pool,a.output)
    if source_generation_hash()!=source: raise RuntimeError("Source changed during model reconstruction")
    ev.write_json(a.output / "completion.json",{"complete":True,"source_generation_hash":source})


if __name__ == "__main__": main()
