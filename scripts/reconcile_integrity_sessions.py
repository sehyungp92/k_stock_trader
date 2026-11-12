"""Compare selected intraday sessions with frozen daily OHLCV without changing inputs.

An exact daily match corroborates an absent interval; it does not prove the
provider's timestamp convention or phase liquidity. Never turns this diagnostic
into a historical qualification receipt.
"""
import argparse
from collections import Counter
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from kis_core.input_manifest import InputManifest
from backtests.core.shared_market_inputs import market_input_contract, selected_intraday_paths
from kis_core.trading_calendar import KRXTradingCalendar
from strategy_common.session_coverage import session_bar_coverage


def compare_session(frame, reference):
    if frame.empty or reference is None:
        return {"status": "missing_intraday" if frame.empty else "missing_daily"}
    actual = {"open": float(frame.open.iloc[0]), "high": float(frame.high.max()),
              "low": float(frame.low.min()), "close": float(frame.close.iloc[-1]),
              "volume": float(frame.volume.sum())}
    expected = {key: float(reference[key]) for key in actual}
    mismatches = [key for key in actual if actual[key] != expected[key]]
    return {"status": "exact_daily_ohlcv_match" if not mismatches else "daily_ohlcv_mismatch",
            "intraday": actual, "daily": expected, "mismatched_fields": mismatches}


def reconcile(manifest_path):
    generation = InputManifest(manifest_path)
    # Use current sourced exceptions explicitly while preserving the old frozen
    # generation and reporting that this is an assessment under a newer calendar.
    current_calendar = KRXTradingCalendar()
    schedule = ROOT / "kis_core/data/krx_sessions.yaml"
    results, gap_counts, all_counts = [], Counter(), Counter()
    with generation.activate():
        generation.calendar = current_calendar
        contract = market_input_contract(generation)
        universe = yaml.safe_load(generation.read(generation.root / contract["universe"]))["symbols"]
        first, last = date.fromisoformat(contract["train_start"]), date.fromisoformat(contract["data_end"])
        days = [first + timedelta(days=i) for i in range((last-first).days+1)
                if current_calendar.is_trading_day(first + timedelta(days=i))]
        for symbol in universe:
            symbol = str(symbol).zfill(6)
            frames = []
            for path in selected_intraday_paths(generation.root / contract["data_root"], symbol, "5m"):
                with generation.open_verified(path) as stream:
                    frames.append(pd.read_parquet(stream))
            frame = pd.concat(frames).sort_values("timestamp")
            if frame.timestamp.duplicated().any():
                raise ValueError(f"Duplicate intraday timestamps: {symbol}")
            groups = dict(tuple(frame.groupby(frame.timestamp.dt.date)))
            daily_paths = sorted((generation.root / contract["daily_data_root"] / "daily_ohlcv" / symbol).glob("*.parquet"))
            if len(daily_paths) != 1:
                raise ValueError(f"Ambiguous daily source: {symbol}")
            with generation.open_verified(daily_paths[0]) as stream:
                daily_frame = pd.read_parquet(stream)
            if daily_frame.date.duplicated().any():
                raise ValueError(f"Duplicate daily rows: {symbol}")
            daily = {str(row["date"])[:10]: row for row in daily_frame.to_dict("records")}
            for day in days:
                bars = groups.get(day, frame.iloc[:0])
                coverage = session_bar_coverage(bars.timestamp, day, 5, symbol=symbol)
                comparison = compare_session(bars, daily.get(day.isoformat()))
                all_counts[comparison["status"]] += 1
                if coverage.get("reason") == "sourced_symbol_suspension":
                    status = "sourced_suspension"
                elif coverage["status"] != "pass":
                    status = comparison["status"]
                else:
                    status = "complete_label_grid"
                if status != "complete_label_grid":
                    gap_counts[status] += 1
                if status != "complete_label_grid" or comparison["status"] != "exact_daily_ohlcv_match":
                    results.append({"symbol": symbol, "date": day.isoformat(), "classification": status,
                        "coverage": coverage, "comparison": comparison,
                        "daily_source": daily_paths[0].relative_to(generation.root).as_posix()})
    return {"version": "session_reconciliation_v1", "manifest_hash": generation.identity,
            "calendar_sha256": hashlib.sha256(schedule.read_bytes()).hexdigest(),
            "calendar_scope": "current_sourced_exceptions_applied_to_preserved_input_generation",
            "symbol_sessions": len(universe)*len(days), "gap_classifications": dict(gap_counts),
            "all_session_daily_comparisons": dict(all_counts), "observations": results,
            "historical_qualification": False,
            "interpretation": "Daily equality corroborates total OHLCV, not every intraday timestamp or executable quantity."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path")
    result = reconcile(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "observations"}, indent=2))


if __name__ == "__main__":
    main()
