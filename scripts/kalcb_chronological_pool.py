"""Build reproducible past-only KALCB research pools; never promotes a model.

The input CSV is the archived positive-R pipeline. Only its `train` partition
is consumed. Day-end labels become available before the next eligible session.
First-30 predictors are usable only after that session's opening half-hour.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import kalcb_r_capture_optimizer as opt
from backtests.strategies.kalcb.chronological import OUTCOME_FEATURES, past_only_folds, validate_predictor_columns
from backtests.strategies.kalcb.candidate_surfacing_recovery import LEAKAGE_FEATURE_BLOCKLIST
from backtests.auto.shared.cache_keys import source_generation_hash


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_pools(pipeline: Path, features_path: Path, output: Path, *, models=("hgb", "ridge"),
                min_train_sessions=60, validation_sessions=20, embargo_sessions=1, budget=16):
    if budget < 1 or len(set(models)) != len(models):
        raise ValueError("Positive budget and unique models required")
    output.mkdir(parents=True, exist_ok=False)
    source_hash = source_generation_hash()
    input_hashes = {str(p.resolve()): digest(p) for p in (pipeline, features_path)}
    opt.PIPELINE_ROWS, opt.OUT_DIR = pipeline, output
    frame = opt.read_pipeline()
    train = frame.loc[frame.window.eq("train")].copy().reset_index(drop=True)
    del frame  # The holdout partition is never scored or used for selection.
    if train.duplicated(["trade_date", "symbol"]).any():
        raise ValueError("Duplicate training symbol/session")
    folds = past_only_folds(train.trade_date.tolist(), min_train_sessions=min_train_sessions,
        validation_sessions=validation_sessions, embargo_sessions=embargo_sessions)
    columns = list(dict.fromkeys(opt.FEATURE_GROUPS["all_context"]))
    # Failing rather than silently dropping required inputs makes runs comparable.
    validate_predictor_columns(columns)
    if set(columns) & LEAKAGE_FEATURE_BLOCKLIST:
        raise ValueError("Noncausal predictor")
    missing = set(columns) - set(train.columns)
    if missing:
        raise ValueError(f"Missing predictors: {sorted(missing)}")
    feature_rows = {}
    with features_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (str(row["trade_date"])[:10], str(row["symbol"]).zfill(6))
            if key in feature_rows:
                raise ValueError(f"Duplicate feature row: {key}")
            feature_rows[key] = row
    scope = opt.scope_mask(train, "dataset").to_numpy()
    summary = {"scope": "exploratory past-only training-window research; not untouched OOS or promotion",
        "source_generation_hash": source_hash, "input_sha256": input_hashes,
        "target": "eod_quality (potential MFE/EOD, not executed PnL)", "features": columns,
        "min_train_sessions": min_train_sessions, "embargo_sessions": embargo_sessions,
        "validation_sessions": validation_sessions, "budget": budget,
        "evaluation_from": folds[0].valid_from, "evaluation_through": folds[-1].valid_through,
        "folds": [{k: v for k, v in asdict(f).items() if not k.endswith("indices")} for f in folds], "models": {}}
    for kind in models:
        scores = np.full(len(train), np.nan)
        provenance = {}
        with threadpool_limits(limits=1):
            for index, fold in enumerate(folds):
                fit = [i for i in fold.train_indices if scope[i]]
                model = opt.make_model(kind)
                model.fit(train.iloc[fit][columns], opt.target_values(train.iloc[fit], "eod_quality"))
                valid = list(fold.valid_indices)
                scores[valid] = model.predict(train.iloc[valid][columns])
                for i in valid:
                    provenance[str(train.iloc[i].trade_date)[:10]] = fold
                print(json.dumps({"model": kind, "fold": index, "train_through": fold.train_through,
                    "label_through": fold.label_through, "valid_from": fold.valid_from, "train_rows": len(fit)}), flush=True)
        selected = train.loc[np.isfinite(scores) & scope].copy()
        selected["_score"] = scores[np.isfinite(scores) & scope]
        rows = []
        for day, group in selected.groupby("trade_date", sort=True):
            # Stable symbol tie-break, independent of input row ordering.
            chosen = group.sort_values(["_score", "symbol"], ascending=[False, True]).head(budget)
            fold = provenance[str(day)[:10]]
            for rank, (_, item) in enumerate(chosen.iterrows(), 1):
                key = (str(day)[:10], str(item.symbol).zfill(6))
                if key not in feature_rows:
                    raise ValueError(f"Selected candidate missing causal metadata: {key}")
                row = {k: v for k, v in feature_rows[key].items() if k not in OUTCOME_FEATURES
                    and k not in LEAKAGE_FEATURE_BLOCKLIST and not k.startswith(("label_", "optimizer_", "oracle_"))}
                row.update(trade_date=key[0], symbol=key[1], window="train", pool_variant=f"past_only_{kind}",
                    pool_size=budget, pool_rank=rank, pool_active=True, frontier_role_for_replay="initial_active",
                    causal_ranker_score=float(item._score), forecast_train_through=fold.train_through,
                    forecast_label_through=fold.label_through, forecast_valid_from=fold.valid_from)
                rows.append(row)
        path = output / f"pool_{kind}_top{budget}.jsonl"
        with path.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        summary["models"][kind] = {"path": str(path.resolve()), "sha256": digest(path), "rows": len(rows),
            "sessions": len({r["trade_date"] for r in rows}), "unscored_warmup_rows": int(np.isnan(scores).sum())}
    if source_generation_hash() != source_hash or any(digest(Path(p)) != h for p, h in input_hashes.items()):
        raise RuntimeError("Source/input changed during pool construction")
    (output / "provenance.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--models", nargs="+", choices=("hgb", "ridge"), default=["hgb", "ridge"])
    parser.add_argument("--min-train-sessions", type=int, default=60)
    parser.add_argument("--validation-sessions", type=int, default=20)
    parser.add_argument("--embargo-sessions", type=int, default=1)
    parser.add_argument("--budget", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(build_pools(args.pipeline, args.features, args.output, models=args.models,
        min_train_sessions=args.min_train_sessions, validation_sessions=args.validation_sessions,
        embargo_sessions=args.embargo_sessions, budget=args.budget), indent=2))


if __name__ == "__main__":
    main()
