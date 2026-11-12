"""Compose recoveries by recomputing their source parsing and exact overlap proof."""
from datetime import date
import hashlib
import json
from pathlib import Path

import pandas as pd

from kis_core.input_manifest import signature
from scripts.recover_flagged_sessions import reconstruct_session
from scripts.update_kis_intraday import _normalize_kis_minute_row, normalize_bar_frame, aggregate_timeframe
from kis_core.trading_calendar import get_trading_calendar
from strategy_common.session_coverage import session_bar_coverage
from scripts.reconcile_integrity_sessions import compare_session


FIELDS = ["timestamp", "open", "high", "low", "close", "volume"]


def verified_reconstruction(root, row):
    root = Path(root).resolve()
    paths = set()

    def bound(name):
        path = (root / name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != row["files"][name]:
            raise ValueError("Recovery source identity mismatch")
        paths.add(path)
        return path

    day, symbol = date.fromisoformat(row["date"]), row["symbol"]
    if row.get("method") == "verified_sparse_closing_auction" and not row.get("daily_source"):
        raise ValueError("Sparse closing recovery requires a daily reference")
    anchor = pd.read_parquet(bound(row["anchor"]))
    raw = pd.read_parquet(bound(row["raw"]))
    if raw.empty or raw.timestamp.duplicated().any():
        raise ValueError("Empty or ambiguous recovered source")
    parsed, provider_rows = [], []
    for name in row["provider_responses"]:
        payload = json.loads(bound(name).read_text(encoding="utf-8"))
        if (payload["symbol"] != symbol or payload["market"] != "J"
                or payload["endpoint"] != "inquire-time-dailychartprice"
                or payload["volume_field"] != "cntg_vol"):
            raise ValueError("Recovery provider contract mismatch")
        parsed.extend(value for value in map(_normalize_kis_minute_row, payload["rows"]) if value is not None)
        provider_rows.extend(payload["rows"])
    parsed = normalize_bar_frame(pd.DataFrame(parsed))
    parsed = parsed[parsed.timestamp.between(raw.timestamp.min(), raw.timestamp.max())]
    duplicates = parsed[parsed.timestamp.duplicated(keep=False)]
    if not duplicates.empty and len(duplicates.drop_duplicates()) != duplicates.timestamp.nunique():
        raise ValueError("Conflicting provider response pages")
    parsed = parsed.drop_duplicates("timestamp").reset_index(drop=True)
    pd.testing.assert_frame_equal(parsed[FIELDS], raw[FIELDS].reset_index(drop=True), check_dtype=False)
    old_minutes = anchor[anchor.timestamp.dt.date == day]
    if row.get("method") == "adjusted_provider_reconstruction":
        from backtests.core.adjusted_recovery import reconstruct_adjusted_session
        daily = pd.read_parquet(bound(row["daily_source"]))
        daily = daily[daily.date.astype(str).str[:10] == day.isoformat()]
        if len(daily) != 1:
            raise ValueError("Daily recovery reference is absent or ambiguous")
        combined, proof = reconstruct_adjusted_session(raw, anchor, day, symbol,
            json.loads(bound(row["original_daily_receipt"]).read_text(encoding="utf-8")),
            json.loads(bound(row["adjusted_daily_receipt"]).read_text(encoding="utf-8")),
            daily.iloc[0].to_dict(), provider_rows)
    elif row.get("method") == "daily_reconciled_empty_session":
        if not old_minutes.empty or any(t.second or t.microsecond for t in raw.timestamp):
            raise ValueError("Empty-session recovery cannot replace existing data or infer a clock shift")
        combined = raw[raw.timestamp.dt.date == day].copy()
        if len(combined) != len(raw):
            raise ValueError("Empty-session recovery contains another session")
        proof = {"method": "raw_provider_rows_and_exact_daily_ohlcv", "old_rows": 0,
                 "added_rows": len(combined), "revised_rows": 0, "offset_seconds": 0,
                 "vendor_timestamp_contract_verified": False}
    else:
        combined, proof = reconstruct_session(raw, anchor, day)
    calendar = get_trading_calendar()
    opening, closing = calendar.session_hours(day)
    match = calendar.closing_match_time(symbol, day)
    if any((not opening <= t.time().replace(tzinfo=None) <= closing and t.time().replace(tzinfo=None) != match)
           or t.second or t.microsecond for t in combined.timestamp):
        raise ValueError("Recovery requires a separate execution-time contract")
    later = combined[combined.timestamp.dt.time > closing]
    if not later.empty and (len(later) != 1 or not (later.open.eq(later.high) & later.high.eq(later.low) & later.low.eq(later.close)).all()):
        raise ValueError("Extended auction must be one sourced single-price print")
    replacement = aggregate_timeframe(combined, "5m")
    complete = session_bar_coverage(replacement.timestamp, day, 5, symbol=symbol)["status"] == "pass"
    daily_match = False
    if row.get("daily_source"):
        daily = pd.read_parquet(bound(row["daily_source"]))
        daily = daily[daily.date.astype(str).str[:10] == day.isoformat()]
        if len(daily) != 1:
            raise ValueError("Daily recovery reference is absent or ambiguous")
        comparison = compare_session(combined, daily.iloc[0].to_dict())
        daily_match = comparison["status"] == "exact_daily_ohlcv_match"
        proof["daily_comparison"] = comparison
    verified_sparse_close = False
    if row.get("method") == "verified_sparse_closing_auction":
        from backtests.core.daily_volume_residual import verify_daily_close_residual
        new = combined[~combined.timestamp.isin(old_minutes.timestamp)]
        if (len(new) != 1 or new.timestamp.iloc[0].time() != match
                or proof.get("offset_seconds") != 0):
            raise ValueError("Sparse closing recovery must add exactly the sourced closing print")
        proof["daily_volume_residual"] = verify_daily_close_residual(combined,
            daily.iloc[0].to_dict(), json.loads(bound(row["original_daily_receipt"]).read_text(encoding="utf-8")),
            provider_rows, day, symbol)
        verified_sparse_close = True
    if not proof["added_rows"] or (not complete and not daily_match and not verified_sparse_close):
        raise ValueError("Recovery has neither complete grid nor daily OHLCV conservation")
    if row.get("method") == "daily_reconciled_empty_session" and not daily_match:
        raise ValueError("Empty-session recovery requires exact independent daily OHLCV")
    proof["coverage_basis"] = ("complete_matching_grid" if complete else
        "verified_closing_print_on_repeated_sparse_source" if verified_sparse_close else "daily_ohlcv_conservation")
    old = aggregate_timeframe(anchor[anchor.timestamp.dt.date == day], "5m")
    return old, replacement, proof, paths


def apply_anchored_recoveries(root, selected, bundle_path):
    root, bundle_path = Path(root).resolve(), Path(bundle_path).resolve()
    if not bundle_path.is_relative_to(root):
        raise ValueError("Recovery bundle is outside research root")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle["version"] != "anchored-session-recoveries-v1":
        raise ValueError("Unsupported recovery bundle")
    result, paths, receipts = dict(selected), {bundle_path}, []
    seen, groups = set(), {}
    for row in bundle["sessions"]:
        key = (row["symbol"], row["date"])
        if key in seen:
            raise ValueError("Ambiguous anchored recovery")
        seen.add(key)
        groups.setdefault(row["symbol"], []).append(row)
    for symbol, rows in groups.items():
        names = selected[symbol]
        frame = pd.concat([pd.read_parquet(root / n) for n in names], ignore_index=True).sort_values("timestamp")
        if frame.timestamp.duplicated().any():
            raise ValueError("Duplicate selected timestamps")
        symbol_receipts = []
        for row in rows:
            day = date.fromisoformat(row["date"])
            old, replacement, proof, sources = verified_reconstruction(root, row)
            selected_day = frame[frame.timestamp.dt.date == day].reset_index(drop=True)
            pd.testing.assert_frame_equal(old[FIELDS], selected_day[FIELDS], check_dtype=False)
            receipt = {"symbol": symbol, "date": row["date"], "reconstruction": proof,
                "source_files": row["files"], "old_session_hash": signature(old.to_dict("records")),
                "new_session_hash": signature(replacement.to_dict("records")),
                "verification_scope": "raw_parser_source_bound_reconstruction_and_session_coverage",
                "historical_qualification": False}
            symbol_receipts.append(receipt)
            paths.update(sources)
            frame = pd.concat([frame[frame.timestamp.dt.date != day], replacement], ignore_index=True)
            frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True).dt.tz_convert("Asia/Seoul")
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        output = bundle_path.parent / "composed" / f"{symbol}_5m_{signature(symbol_receipts)}.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            # Arrow can preserve microseconds or nanoseconds; compare identical
            # timestamps/values without treating representation precision as data.
            pd.testing.assert_frame_equal(pd.read_parquet(output), frame, check_dtype=False)
        else:
            frame.to_parquet(output, index=False)
        result[symbol] = [output.relative_to(root).as_posix()]
        paths.update({output, *(root / n for n in names)})
        receipts.extend(symbol_receipts)
    return result, paths, receipts
