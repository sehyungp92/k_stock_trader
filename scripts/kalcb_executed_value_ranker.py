"""Past-only controlled KALCB value ranking and earlier-OOF admission calibration."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, time
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb import executed_value as ev


def model_for(kind="ridge"):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind != "ridge":
        raise ValueError("Only the predeclared ridge comparison is implemented")
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=12.0))


def fit_forecast(rows, labels, sessions, first, through, *, target, min_sessions=60):
    import numpy as np
    available = [{**row, "label_available_at": labels[row["opportunity_id"]]["label_available_at"]} for row in rows]
    train = ev.mature_training_indices(available, sessions, first, min_sessions=min_sessions, embargo=1)
    train = [i for i in train if labels[rows[i]["opportunity_id"]]["label_valid"] and
             (target != "quality" or rows[i]["quality_target"] is not None)]
    valid = [i for i, row in enumerate(rows) if first <= row["trade_date"] <= through]
    if len({rows[i]["trade_date"] for i in train}) < min_sessions:
        raise ValueError("Insufficient mature training sessions")
    train_rows = [rows[i] for i in train]
    y = np.asarray([rows[i]["quality_target"] if target == "quality" else
                    labels[rows[i]["opportunity_id"]]["net_value_equity_bps"] for i in train], dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Invalid training target")
    model = model_for()
    model.fit(ev.feature_matrix(train_rows), y)
    predictions = model.predict(ev.feature_matrix([rows[i] for i in valid]))
    proof = {"target": target, "valid_from": first, "valid_through": through,
        "fit_at": datetime.combine(date.fromisoformat(first), time.min, ev.KST).isoformat(),
        "train_through": max(r["trade_date"] for r in train_rows),
        "label_available_through": max(labels[r["opportunity_id"]]["label_available_at"] for r in train_rows),
        "train_rows": len(train), "train_sessions": len({r["trade_date"] for r in train_rows}),
        "training_keys_hash": ev.signature([r["opportunity_id"] for r in train_rows]),
        "training_target_hash": ev.signature(y.tolist()), "feature_columns": ev.FEATURES,
        "model": "median_imputer_standard_scaler_ridge_alpha12", "model_selection": "predeclared_no_outer_tuning"}
    return model, [(rows[i], float(score)) for i, score in zip(valid, predictions)], proof


def load_labels(paths, prepared_hash):
    labels, manifests = {}, {}
    for path in paths:
        m = json.loads((path / "labels_manifest.json").read_text())
        completion = json.loads((path / "completion.json").read_text())
        request = json.loads((path / "request.json").read_text())
        if not completion["complete"] or completion["source_generation_hash"] != request["source_generation_hash"] or ev.digest(path / "request.json") != m["request_sha256"]:
            raise ValueError("Uncompleted or changed label generation")
        if m["prepared_sha256"] != prepared_hash or m["scenario"] != "baseline":
            raise ValueError("Label identity/scenario mismatch")
        for name, digest in m["label_files"].items():
            if ev.digest(path / name) != digest:
                raise ValueError("Changed label bytes")
            for row in ev.read_jsonl(path / name):
                key = row["opportunity_id"]
                if key in labels:
                    raise ValueError("Duplicate opportunity label")
                labels[key] = row
        manifests[str(path / "labels_manifest.json")] = ev.digest(path / "labels_manifest.json")
    return labels, manifests


def run(plan, prepared, label_paths, output):
    import joblib
    from threadpoolctl import threadpool_limits
    from scripts.kalcb_executed_value_study import emit
    provenance = json.loads((prepared / "prepared.json").read_text())
    if ev.digest(prepared / "opportunities.jsonl") != provenance["opportunities_sha256"]:
        raise ValueError("Changed opportunities")
    rows = ev.read_jsonl(prepared / "opportunities.jsonl")
    labels, manifests = load_labels(label_paths, ev.digest(prepared / "prepared.json"))
    if set(labels) != {r["opportunity_id"] for r in rows}:
        raise ValueError("Incomplete opportunity labels")
    sessions = provenance["sessions"]
    seed = json.loads(Path(plan["seed"]).read_text())["mutations"]
    q3 = json.loads(Path(plan["q3"]).read_text())["mutations"]
    predicate = lambda row, rank: ev.eligible(row, seed, q3, rank=rank)
    contract = plan["comparison_contract"]
    budget, shortlist = contract["budget"], contract["shortlist_before_guard"]
    outer = ev.outer_windows(sessions, min_train=contract["fit_min_sessions"], block=contract["outer_block_sessions"])
    policies = {name: [] for name in ("quality_after", "quality_before", "net_after", "net_before", "net_calibrated", "simple_before", "cash")}
    decisions = {name: [] for name in policies}
    forecasts, proofs, calibrations, inner = [], [], [], []
    # Inner predictions include early history unavailable to the outer comparison.
    # They use only earlier data. Later fits may consume earlier outer outcomes,
    # exactly as a predeclared rolling retraining algorithm would.
    with threadpool_limits(limits=1):
        for first, through in ev.outer_windows(sessions, min_train=20, block=10):
            if first >= outer[-1][0]:
                break
            model, predictions, proof = fit_forecast(rows, labels, sessions, first, through, target="net", min_sessions=20)
            proof["purpose"] = "inner_admission_calibration"
            proofs.append(proof)
            grouped = defaultdict(list)
            for row, score in predictions:
                grouped[row["trade_date"]].append((row, score))
            for day, items in grouped.items():
                selected, _ = ev.admit([r for r, _ in items], [s for _, s in items], predicate,
                                      construction="before", budget=budget, shortlist=shortlist)
                for row, score in selected:
                    label = labels[row["opportunity_id"]]
                    inner.append({"opportunity_id": row["opportunity_id"], "trade_date": day, "score": score,
                        "net_value_equity_bps": label["net_value_equity_bps"],
                        "deadline_violation": label["deadline_violation"], "label_available_at": label["label_available_at"],
                        "calibration_available_at": label["label_available_at"] or label["risk_available_at"],
                        "forecast_valid_from": first, "forecast_train_through": proof["train_through"]})
        for fold, (first, through) in enumerate(outer):
            cutoff = datetime.combine(date.fromisoformat(first), time.min, ev.KST)
            earlier = [r for r in inner if r["trade_date"] < first and
                       datetime.fromisoformat(r["calibration_available_at"]) < cutoff]
            threshold, calibration = ev.choose_threshold(earlier)
            calibrations.append({"valid_from": first, "valid_through": through, "threshold": threshold,
                                 "latest_label_available_at": max((r["calibration_available_at"] for r in earlier), default=None), **calibration})
            by_target = {}
            for target in ("quality", "net"):
                model, predictions, proof = fit_forecast(rows, labels, sessions, first, through, target=target)
                proof["purpose"] = "outer_forecast"
                proofs.append(proof)
                model_path = output / f"model_{target}_{fold}.joblib"
                joblib.dump(model, model_path)
                proof["model_path"] = model_path.name
                proof["model_sha256"] = ev.digest(model_path)
                grouped = defaultdict(list)
                for row, score in predictions:
                    grouped[row["trade_date"]].append((row, score))
                    forecasts.append({"opportunity_id": row["opportunity_id"], "trade_date": row["trade_date"],
                        "feature_available_at": row["feature_available_at"], "target": target, "score": score,
                        "forecast_valid_from": first, "train_through": proof["train_through"],
                        "label_available_through": proof["label_available_through"]})
                by_target[target] = grouped
            for day in [d for d in sessions if first <= d <= through]:
                for name in policies:
                    target = "quality" if name.startswith("quality") else "net"
                    items = by_target[target].get(day, [])
                    if name == "simple_before":
                        items = [(r, r["features"]["first30_ret"] * r["features"]["first30_rel_volume_log"] +
                                  .25 * r["features"]["stock_sector_daily_ret5_spread"]) for r, _ in items]
                    if name == "cash" or name == "net_calibrated" and threshold == "cash":
                        selected, reasons = [], {r["opportunity_id"]: "cash_policy" for r, _ in items}
                    else:
                        selected, reasons = ev.admit([r for r, _ in items], [s for _, s in items], predicate,
                            construction="after" if name.endswith("after") else "before", budget=budget, shortlist=shortlist,
                            threshold=threshold if name == "net_calibrated" else None)
                    proof = {"policy": name, "valid_from": first, "threshold_equity_bps": threshold if name == "net_calibrated" else None}
                    snap = ev.snapshot_for(day, selected, source=provenance["opportunities_sha256"], forecast=proof)
                    # Empty snapshots explicitly bind cash-only candidate dispositions.
                    policies[name].append(snap.to_json_dict())
                    decisions[name].extend({"opportunity_id": r["opportunity_id"], "trade_date": day,
                        "score": score, "reason": reasons[r["opportunity_id"]], **proof} for r, score in items)
            print(json.dumps({"outer_fold": fold, "from": first, "through": through, "net_admission_threshold": threshold}), flush=True)
    for name, snapshots in policies.items():
        ev.write_json(output / (name + ".snapshots.json"), snapshots)
        with (output / (name + ".admission.jsonl")).open("x", encoding="utf-8") as stream:
            for row in decisions[name]:
                emit(stream, row)
    with (output / "forecasts.jsonl").open("x", encoding="utf-8") as stream:
        for row in forecasts:
            emit(stream, row)
    with (output / "inner_calibration.jsonl").open("x", encoding="utf-8") as stream:
        for row in inner:
            emit(stream, row)
    ev.write_json(output / "chronology.json", {"folds": proofs, "calibration": calibrations,
        "scope": "past-only model/admission fitting; inherited guard and execution historically developed; not untouched OOS"})
    files = {p.name: ev.digest(p) for p in output.iterdir() if p.is_file()}
    ev.write_json(output / "policies_manifest.json", {"version": ev.VERSION, "prepared_path": str(prepared),
        "prepared_sha256": ev.digest(prepared / "prepared.json"), "label_manifests": manifests,
        "evaluation_from": outer[0][0], "evaluation_through": outer[-1][1], "policies": list(policies),
        "contract": contract, "files": files, "counts": {n:sum(len(s["candidates"]) for s in ss) for n,ss in policies.items()}})


def main():
    from backtests.auto.shared.cache_keys import source_generation_hash
    from scripts.kalcb_executed_value_study import load_plan
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--prepared", required=True, type=Path)
    p.add_argument("--labels", required=True, action="append", type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    plan, source = load_plan(args.plan), source_generation_hash()
    args.output.mkdir(parents=True, exist_ok=False)
    run(plan, args.prepared, args.labels, args.output)
    if source_generation_hash() != source:
        raise RuntimeError("Source changed during fitting")
    ev.write_json(args.output / "completion.json", {"source_generation_hash": source, "complete": True})


if __name__ == "__main__":
    main()
