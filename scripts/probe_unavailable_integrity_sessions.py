"""Query each retention-flagged session using the existing quotation downloader.

An empty successful response is provider unavailability, not proof that the
historical market had no trades. Errors remain unresolved and are retryable.
"""
import argparse
from collections import Counter
from datetime import date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kis_core.trading_calendar import get_trading_calendar
from scripts.update_kis_intraday import (
    KST, KoreaInvestAPI, KoreaInvestEnv, _load_dotenv,
    build_kis_config_from_env, fetch_kis_1m_bars,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--statuses", nargs="+", default=["outside_declared_provider_retention"])
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    args.destination.mkdir(parents=True, exist_ok=True)
    _load_dotenv(ROOT / ".env")
    from loguru import logger
    logger.remove()
    api = KoreaInvestAPI(KoreaInvestEnv(build_kis_config_from_env()))
    results = []
    for row in inventory["sessions"]:
        if row["status"] not in args.statuses:
            continue
        symbol, day = row["symbol"], date.fromisoformat(row["date"])
        directory = args.destination / symbol / day.isoformat()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "receipt.json"
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous["status"] != "query_failed":
                results.append(previous)
                continue
        opening, closing = get_trading_calendar().session_hours(day)
        result = {"symbol": symbol, "date": day.isoformat(),
                  "queried_at": datetime.now(KST).isoformat(), "quotation_only": True,
                  "historical_session_complete": False}
        try:
            frame, pages = fetch_kis_1m_bars(api, symbol,
                datetime.combine(day, opening, KST) - timedelta(minutes=2),
                datetime.combine(day, closing, KST) + timedelta(minutes=10),
                market_code="J", extra_sleep_sec=.6, max_pages=6,
                raw_page_root=directory / "provider_responses")
            frame.to_parquet(directory / "raw_1m.parquet", index=False)
            result.update(rows=len(frame), pages=pages,
                status="unavailable_from_current_kis" if frame.empty else "recovered_requires_reconciliation",
                reason="successful_query_returned_no_target_session_observations" if frame.empty else "target_observations_returned")
        except Exception as exc:
            # Never serialize API exception text: it can contain request secrets.
            result.update(status="query_failed", error_type=type(exc).__name__)
        result["source_files"] = {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in directory.rglob("*") if p.is_file() and p != path}
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(result)
        print(json.dumps({"processed": len(results), "symbol": symbol, "date": day.isoformat(), "status": result["status"]}), flush=True)
    report = {"version": "kis-session-availability-v1", "quotation_only": True,
        "inventory_sha256": hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
        "counts": dict(Counter(r["status"] for r in results)), "sessions": results}
    (args.destination / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["counts"]), flush=True)


if __name__ == "__main__":
    main()
