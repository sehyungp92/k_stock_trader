"""Resolve the finite flagged-session inventory and prepare sourced recoveries.

Provider availability, usable archived-basis recovery and historical
qualification are separate outcomes. This script never invents missing bars.
"""
import argparse
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import yaml
from kis_core.trading_calendar import get_trading_calendar
from scripts.recover_flagged_sessions import reconstruct_session
from scripts.reconcile_integrity_sessions import compare_session
from scripts.update_kis_intraday import aggregate_timeframe
from strategy_common.session_coverage import session_bar_coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--availability", type=Path, required=True)
    parser.add_argument("--full-recovery", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    available = json.loads((args.availability / "report.json").read_text(encoding="utf-8"))
    queried = {(r["symbol"], r["date"]): r for r in available["sessions"]}
    older = {(r["symbol"], r["date"]) for r in inventory["sessions"] if r["status"] == "outside_declared_provider_retention"}
    if older != set(queried) or any(r["status"] == "query_failed" for r in queried.values()):
        raise ValueError("Every retention-flagged session needs a successful provider query")
    args.destination.mkdir(parents=True, exist_ok=True)
    calendar = get_trading_calendar()
    calendar_path = ROOT / "kis_core/data/krx_intraday_halts.yaml"
    regimes = yaml.safe_load(calendar_path.read_text(encoding="utf-8"))
    regime_rows = {(r["symbol"], r["date"]): r for r in regimes.get("closing_auctions", [])}
    hashes, cached = {}, {}

    def relative(path):
        return path.resolve().relative_to(ROOT).as_posix()

    def digest(path):
        name = relative(path)
        if name not in hashes:
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return hashes[name]

    outcomes, accepted = [], []
    for item in inventory["sessions"]:
        symbol, day = item["symbol"], date.fromisoformat(item["date"])
        key = symbol, day.isoformat()
        if symbol not in cached:
            cached.clear()
            anchors = list((ROOT / "data/kis_intraday_parquet" / symbol).glob(f"{symbol}_1m_*_20260512.parquet"))
            daily_paths = list((ROOT / "data/krx_daily_parquet/daily_ohlcv" / symbol).glob("*.parquet"))
            if len(anchors) != 1 or len(daily_paths) != 1:
                raise ValueError("Ambiguous archive or daily source")
            cached[symbol] = anchors[0], pd.read_parquet(anchors[0]), daily_paths[0], pd.read_parquet(daily_paths[0])
        anchor_path, anchor, daily_path, daily = cached[symbol]
        old = anchor[anchor.timestamp.dt.date == day].reset_index(drop=True)
        daily_row = daily[daily.date.astype(str).str[:10] == day.isoformat()]
        reference = daily_row.iloc[0].to_dict() if len(daily_row) == 1 else None
        result = {"symbol": symbol, "date": day.isoformat(), "original_status": item["status"],
                  "archive_daily_comparison": compare_session(old, reference), "historical_qualification": False}
        directory = args.inventory.parent / symbol / day.isoformat()
        if key in queried:
            result["provider_availability"] = queried[key]["status"]
            result["availability_receipt"] = relative(args.availability / symbol / day.isoformat() / "receipt.json")
            if queried[key]["status"] == "unavailable_from_current_kis":
                result["disposition"] = ("archived_sparse_daily_reconciled_provider_unavailable"
                    if result["archive_daily_comparison"]["status"] == "exact_daily_ohlcv_match"
                    else "unavailable_from_current_kis_archive_unreconciled")
                outcomes.append(result)
                continue
            directory = args.availability / symbol / day.isoformat()
        if item["status"] == "sourced_periodic_auction_grid":
            result["disposition"] = "sourced_periodic_auction"
            outcomes.append(result)
            continue
        if args.full_recovery and (args.full_recovery / symbol / day.isoformat() / "raw_1m.parquet").exists():
            directory = args.full_recovery / symbol / day.isoformat()
        raw_path = directory / "raw_1m.parquet"
        raw = pd.read_parquet(raw_path)
        result["provider_availability"] = "observations_returned"
        result["raw_source"] = relative(raw_path)
        method = "exact_archived_overlap"
        try:
            combined, proof = reconstruct_session(raw, anchor, day)
        except ValueError as exc:
            if old.empty and compare_session(raw, reference)["status"] == "exact_daily_ohlcv_match" and not any(t.second or t.microsecond for t in raw.timestamp):
                combined, method = raw, "daily_reconciled_empty_session"
                proof = {"added_rows": len(raw), "offset_seconds": 0, "method": method}
            else:
                result.update(disposition="returned_source_incompatible_with_archive", reconstruction_error=str(exc),
                              returned_daily_comparison=compare_session(raw, reference), raw_rows=len(raw))
                outcomes.append(result)
                continue
        result["reconstruction"] = proof
        comparison = compare_session(combined, reference)
        result["recovered_daily_comparison"] = comparison
        closing = calendar.session_hours(day)[1]
        later = combined[combined.timestamp.dt.time > closing]
        if not later.empty:
            if (len(later) != 1 or not (later.open.eq(later.high) & later.high.eq(later.low) & later.low.eq(later.close)).all()
                    or later.timestamp.iloc[0].time().minute >= 40):
                result["disposition"] = "returned_source_requires_other_matching_regime"
                outcomes.append(result)
                continue
            event = {"symbol": symbol, "date": day.isoformat(), "matching_time": later.timestamp.iloc[0].strftime("%H:%M:%S")}
            evidence = {"closing_event": event, "observation": later.to_dict("records")[0],
                "reconstruction": proof, "raw_source": relative(raw_path), "raw_sha256": digest(raw_path),
                "archive_source": relative(anchor_path), "archive_sha256": digest(anchor_path),
                "interpretation": "single-price closing match after scheduled close; cause of extension not inferred"}
            proof_path = args.destination / "closing_events" / f"{symbol}_{day.isoformat()}.json"
            proof_path.parent.mkdir(parents=True, exist_ok=True)
            proof_path.write_text(json.dumps(evidence, indent=2, default=str), encoding="utf-8")
            regimes["evidence_files"][relative(proof_path)] = digest(proof_path)
            regime_rows[key] = {**event, "sources": [relative(proof_path)]}
            from datetime import time
            calendar.closing_auctions[(symbol, day)] = time.fromisoformat(event["matching_time"])
        bars = aggregate_timeframe(combined, "5m")
        complete = session_bar_coverage(bars.timestamp, day, 5, symbol=symbol)["status"] == "pass"
        if proof["added_rows"] and (complete or comparison["status"] == "exact_daily_ohlcv_match"):
            pages = sorted((directory / "provider_responses").rglob("*.json"))
            files = [anchor_path, raw_path, daily_path, *pages]
            row = {"symbol": symbol, "date": day.isoformat(), "method": method,
                "anchor": relative(anchor_path), "raw": relative(raw_path), "daily_source": relative(daily_path),
                "provider_responses": [relative(p) for p in pages], "files": {relative(p): digest(p) for p in files}}
            accepted.append(row)
            result.update(disposition="recovered_for_shared_input", coverage_basis="complete_matching_grid" if complete else "daily_ohlcv_conservation")
        elif comparison["status"] == "exact_daily_ohlcv_match":
            result["disposition"] = "archived_sparse_daily_reconciled"
        else:
            result["disposition"] = "repeat_source_sparse_daily_unreconciled"
        outcomes.append(result)
    regimes["version"] = "krx-intraday-regimes-v3"
    regimes["closing_auctions"] = [regime_rows[key] for key in sorted(regime_rows)]
    calendar_path.write_text(yaml.safe_dump(regimes, sort_keys=False, allow_unicode=True), encoding="utf-8")
    bundle = {"version": "anchored-session-recoveries-v1", "sessions": accepted}
    (args.destination / "bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    report = {"version": "session-recovery-dispositions-v1", "all_original_flags_accounted": len(outcomes) == len(inventory["sessions"]),
        "historical_qualification": False, "counts": dict(Counter(r["disposition"] for r in outcomes)), "sessions": outcomes,
        "source_files": {relative(args.inventory): digest(args.inventory),
                         relative(args.availability / "report.json"): digest(args.availability / "report.json")}}
    for source_root in (args.inventory.parent, args.availability, args.full_recovery):
        if source_root:
            for path in source_root.rglob("*"):
                if path.is_file():
                    report["source_files"][relative(path)] = digest(path)
    (args.destination / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["counts"]), flush=True)


if __name__ == "__main__":
    main()
