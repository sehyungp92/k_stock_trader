"""Recover one historical session into a new directory using quotation calls only."""
import argparse
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import sys
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.update_kis_intraday import (KST, KoreaInvestAPI, KoreaInvestEnv, build_kis_config_from_env,
    _load_dotenv, fetch_kis_1m_bars, aggregate_timeframe)
from kis_core.trading_calendar import get_trading_calendar
from strategy_common.session_coverage import session_bar_coverage


def align_to_archived_labels(frame, anchor, day):
    """Require one uniform offset proven by every unambiguous overlapping OHLCV row.

    This is a recorded reconstruction, not an assumed vendor timestamp contract.
    Both raw responses and the original archive remain unchanged.
    """
    fields = ("open", "high", "low", "close", "volume")
    def index(data):
        rows = {}
        duplicates = set()
        for row in data.to_dict("records"):
            if row["timestamp"].date() != day:
                continue
            key = tuple(float(row[f]) for f in fields)
            if key in rows:
                duplicates.add(key)
            rows[key] = row["timestamp"]
        return {k: v for k, v in rows.items() if k not in duplicates}
    raw, old = index(frame), index(anchor)
    shared = raw.keys() & old.keys()
    offsets = {(old[key] - raw[key]).total_seconds() for key in shared}
    if len(shared) < 20 or len(offsets) != 1 or abs(next(iter(offsets))) > 120:
        raise ValueError("No sufficiently supported uniform timestamp reconstruction")
    offset = next(iter(offsets))
    corrected = frame.copy()
    corrected["timestamp"] += pd.Timedelta(seconds=offset)
    if anchor.timestamp.duplicated().any() or corrected.timestamp.duplicated().any():
        raise ValueError("Duplicate timestamp in reconstruction or archive")
    anchored = anchor.set_index("timestamp")
    for row in corrected.to_dict("records"):
        if row["timestamp"] in anchored.index and any(
                float(anchored.loc[row["timestamp"], f]) != float(row[f]) for f in fields):
            raise ValueError("Timestamp reconstruction conflicts with archived OHLCV")
    return corrected, {"offset_seconds": offset, "matching_unique_ohlcv_anchors": len(shared),
        "method": "uniform_offset_from_exact_archived_ohlcv_matches", "vendor_timestamp_contract_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--anchor-parquet", type=Path)
    parser.add_argument("--anchor-root", type=Path,
        help="Archive directory containing SYMBOL/SYMBOL_1m_*_END.parquet")
    parser.add_argument("--anchor-end", default="20260512")
    args = parser.parse_args()
    if args.anchor_parquet and args.anchor_root:
        parser.error("Choose one explicit anchor file or a symbol archive root")
    if args.destination.exists():
        parser.error("Recovery destination must be new; existing evidence is preserved")
    day = date.fromisoformat(args.date)
    calendar = get_trading_calendar()
    if not calendar.has_session_coverage(day) or not calendar.is_trading_day(day):
        parser.error("A verified trading session is required")
    args.destination.mkdir(parents=True)
    _load_dotenv(ROOT / ".env")
    report = {"date": day.isoformat(), "mode": "quotation_only", "symbols": {}}
    try:
        api = KoreaInvestAPI(KoreaInvestEnv(build_kis_config_from_env()))
        opening, closing = calendar.session_hours(day)
        for symbol in args.symbols:
            try:
                frame, pages = fetch_kis_1m_bars(api, symbol, datetime.combine(day, opening, KST) - timedelta(minutes=2),
                    datetime.combine(day, closing, KST) + timedelta(minutes=2), market_code="J", extra_sleep_sec=.6,
                    max_pages=10, raw_page_root=args.destination / "provider_responses")
                reconstruction = None
                anchor = args.anchor_parquet
                if args.anchor_root:
                    paths = sorted((args.anchor_root / symbol).glob(f"{symbol}_1m_*_{args.anchor_end}.parquet"))
                    if len(paths) != 1:
                        raise ValueError("Expected exactly one archived timestamp anchor")
                    anchor = paths[0]
                if anchor is not None:
                    if (args.anchor_parquet and len(args.symbols) != 1) or not anchor.name.startswith(symbol + "_"):
                        raise ValueError("Timestamp anchors must belong to exactly one requested symbol")
                    frame.to_parquet(args.destination / f"{symbol}_raw_timestamp_1m.parquet", index=False)
                    frame, reconstruction = align_to_archived_labels(frame, pd.read_parquet(anchor), day)
                    reconstruction["anchor_sha256"] = hashlib.sha256(anchor.read_bytes()).hexdigest()
                    reconstruction["anchor_path"] = str(anchor.resolve())
                    frame = frame[(frame.timestamp >= datetime.combine(day, opening, KST)) & (frame.timestamp <= datetime.combine(day, closing, KST))]
                coverage = session_bar_coverage(frame.timestamp, day, symbol=symbol)
                report["symbols"][symbol] = {"rows": len(frame), "pages": pages, "coverage": coverage, "timestamp_reconstruction": reconstruction}
                if not frame.empty:
                    for timeframe in ("1m", "5m"):
                        aggregate_timeframe(frame, timeframe).to_parquet(args.destination / f"{symbol}_{timeframe}.parquet", index=False)
            except Exception as exc:
                # API errors can contain account/request details; store the type only.
                report["symbols"][symbol] = {"status": "failed", "error_type": type(exc).__name__}
    except Exception as exc:
        report["initialization_error"] = type(exc).__name__
    report["complete"] = bool(report["symbols"]) and all(row.get("coverage", {}).get("status") == "pass" for row in report["symbols"].values())
    report["verification_scope"] = "session_interval_coverage_and_archived_ohlcv_agreement"
    report["historical_qualification"] = False
    (args.destination / "recovery.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"complete": report["complete"], "initialization_error": report.get("initialization_error"),
        "symbols": {symbol: {"rows": row.get("rows"), "coverage": row.get("coverage", {}).get("status"),
            "missing_intervals": len(row.get("coverage", {}).get("missing_intervals", [])),
            "error_type": row.get("error_type")} for symbol, row in report["symbols"].items()}}, indent=2))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
