"""Recover available flagged observations without rewriting research inputs.

Quotation calls only. Each reconstruction must agree with every overlapping
archived one-minute row and have at least 20 unique exact OHLCV anchors.
Later auction prints retain their true source time; they are never rounded back
to the scheduled close or silently admitted to the continuous execution model.
"""
import argparse
from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from kis_core.trading_calendar import get_trading_calendar
from scripts.recover_integrity_session import align_to_archived_labels
from scripts.reconcile_integrity_sessions import compare_session
from scripts.update_kis_intraday import (
    KST, KoreaInvestAPI, KoreaInvestEnv, _load_dotenv, build_kis_config_from_env,
    fetch_kis_1m_bars, aggregate_timeframe,
)


def reconstruct_session(raw, anchor, day):
    """Return an exact-overlap reconstruction and a compact, reproducible delta."""
    old = anchor[anchor.timestamp.dt.date == day].copy().reset_index(drop=True)
    corrected, proof = align_to_archived_labels(raw, old, day)
    corrected = corrected[corrected.timestamp.dt.date == day].reset_index(drop=True)
    new = corrected[~corrected.timestamp.isin(old.timestamp)]
    combined = pd.concat([old, new], ignore_index=True)
    # Parquet may use pytz while the KIS parser uses ZoneInfo. Pandas otherwise
    # creates an object column even though both are the same exchange timezone.
    combined["timestamp"] = pd.to_datetime(combined.timestamp, utc=True).dt.tz_convert("Asia/Seoul")
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    return combined, {**proof, "old_rows": len(old), "added_rows": len(new),
        "added_timestamps": [t.isoformat() for t in new.timestamp], "revised_rows": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--available-from", type=date.fromisoformat, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reprocess-retained", action="store_true")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    census = json.loads(args.reconciliation.read_text(encoding="utf-8"))
    rows = [r for r in census["observations"] if r["classification"] not in
            {"complete_label_grid", "sourced_suspension"}]
    calendar = get_trading_calendar()
    _load_dotenv(ROOT / ".env")
    from loguru import logger
    logger.remove()
    api = KoreaInvestAPI(KoreaInvestEnv(build_kis_config_from_env()))
    results, cache = [], {}
    fetched = 0
    for item in rows:
        symbol, day = item["symbol"], date.fromisoformat(item["date"])
        directory = args.destination / symbol / day.isoformat()
        record = directory / "receipt.json"
        if record.exists() and not args.reprocess_retained:
            results.append(json.loads(record.read_text(encoding="utf-8")))
            continue
        result = {"symbol": symbol, "date": day.isoformat(), "historical_qualification": False}
        if calendar.periodic_auction_sources(symbol, day):
            result["status"] = "sourced_periodic_auction_grid"
        elif day < args.available_from:
            result["status"] = "outside_declared_provider_retention"
        elif args.limit is not None and fetched >= args.limit:
            result["status"] = "not_attempted_limit"
        else:
            fetched += 1
            directory.mkdir(parents=True, exist_ok=True)
            try:
                if symbol not in cache:
                    paths = sorted((ROOT / "data/kis_intraday_parquet" / symbol).glob(f"{symbol}_1m_*_20260512.parquet"))
                    if len(paths) != 1:
                        raise ValueError("Ambiguous archive")
                    cache.clear()  # Bound memory to one symbol's annual minute file.
                    cache[symbol] = (paths[0], pd.read_parquet(paths[0]), hashlib.sha256(paths[0].read_bytes()).hexdigest())
                anchor_path, anchor, anchor_hash = cache[symbol]
                opening, closing = calendar.session_hours(day)
                start = datetime.combine(day, opening, KST) - timedelta(minutes=2)
                missing = item["coverage"].get("missing_intervals", [])
                if missing:
                    start = max(start, datetime.fromisoformat(min(missing)) - timedelta(minutes=62))
                end = datetime.combine(day, closing, KST) + timedelta(minutes=10)
                if (directory / "raw_1m.parquet").exists():
                    raw = pd.read_parquet(directory / "raw_1m.parquet")
                    pages = len(list((directory / "provider_responses").rglob("*.json")))
                else:
                    raw, pages = fetch_kis_1m_bars(api, symbol, start, end, market_code="J",
                        extra_sleep_sec=.6, max_pages=6, raw_page_root=directory / "provider_responses")
                result.update(raw_rows=len(raw), pages=pages,
                    anchor_path=anchor_path.relative_to(ROOT).as_posix(), anchor_sha256=anchor_hash)
                raw.to_parquet(directory / "raw_1m.parquet", index=False)
                if raw.empty:
                    result["status"] = "provider_returned_no_observations"
                else:
                    combined, proof = reconstruct_session(raw, anchor, day)
                    combined.to_parquet(directory / "reconstructed_1m.parquet", index=False)
                    aggregate_timeframe(combined, "5m").to_parquet(directory / "diagnostic_5m.parquet", index=False)
                    after_close = combined[combined.timestamp.dt.time > closing]
                    comparison = compare_session(combined, item.get("comparison", {}).get("daily"))
                    result.update(status="exact_overlap_reconstruction", reconstruction=proof,
                        daily_comparison=comparison,
                        after_scheduled_close=after_close.to_dict("records"),
                        integration_status="requires_verified_execution_timing_and_basis")
            except Exception as exc:
                import traceback
                location = traceback.extract_tb(exc.__traceback__)[-1]
                result.update(status="reconstruction_unavailable", error_type=type(exc).__name__,
                    error_location={"file": Path(location.filename).name, "line": location.lineno, "function": location.name})
            result["source_files"] = {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in directory.rglob("*") if p.is_file() and p != record}
        # Persist even unavailable cases, making retries explicit and reproducible.
        directory.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        results.append(result)
        if fetched and fetched % 10 == 0:
            print(json.dumps({"processed": len(results), "fetched": fetched,
                              "last": [symbol, day.isoformat(), result["status"]]}), flush=True)
    from collections import Counter
    report = {"version": "flagged-session-recovery-v1", "quotation_only": True,
        "input_reconciliation_sha256": hashlib.sha256(args.reconciliation.read_bytes()).hexdigest(),
        "available_from": args.available_from.isoformat(), "historical_qualification": False,
        "counts": dict(Counter(r["status"] for r in results)), "sessions": results}
    (args.destination / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report["counts"]), flush=True)


if __name__ == "__main__":
    main()
