#!/usr/bin/env python3
"""Incrementally update the PyKRX daily LRS store and full Parquet mirror.

The important bit is ordering:
1. Append/upsert only the missing daily rows into SQLite.
2. Re-export the entire strategy-neutral Parquet mirror from SQLite.

Do not window the Parquet export to the incremental dates, or backtests will
see a truncated daily dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backfill_lrs import load_universe, main as backfill_lrs_main
from strategy_common.daily_lrs_parquet import export_lrs_sqlite_to_parquet

KST = ZoneInfo("Asia/Seoul")
DEFAULT_UNIVERSE = "config/olr_kalcb/olr_deployment_universe_103.yaml"
DEFAULT_HISTORY_START = date(2024, 2, 19)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    target_end = date.fromisoformat(args.end) if args.end else _default_completed_krx_day()
    manifest_start, manifest_end = _manifest_daily_range(Path(args.parquet_root))
    history_start = date.fromisoformat(args.history_start) if args.history_start else (
        manifest_start or DEFAULT_HISTORY_START
    )
    db_start, db_end = _sqlite_daily_range(Path(args.db_path))
    start = date.fromisoformat(args.start) if args.start else _choose_update_start(
        history_start=history_start,
        target_end=target_end,
        db_start=db_start,
        db_end=db_end,
        manifest_end=manifest_end,
    )

    print(f"Daily KRX target end: {target_end.isoformat()}")
    print(f"Required history start: {history_start.isoformat()}")
    if manifest_start is not None and manifest_end is not None:
        print(f"Existing daily Parquet date range: {manifest_start.isoformat()} to {manifest_end.isoformat()}")
    else:
        print("Existing daily Parquet manifest not found; backfill_lrs defaults will be used.")
    if db_start is not None and db_end is not None:
        print(f"Existing SQLite daily date range: {db_start.isoformat()} to {db_end.isoformat()}")
    else:
        print("Existing SQLite daily date range: empty or unavailable")

    should_backfill = not args.export_only and (
        args.force or start is None or start <= target_end or not _range_covers(db_start, db_end, history_start, target_end)
    )
    if should_backfill:
        backfill_args = [
            "--db-path",
            args.db_path,
            "--universe-file",
            args.universe_file,
            "--config",
            args.config,
            "--to",
            target_end.isoformat(),
            "--sleep-sec",
            str(args.sleep_sec),
            "--parquet-root",
            args.parquet_root,
            "--skip-parquet-export",
        ]
        if start is not None:
            backfill_args.extend(["--from", start.isoformat()])
        if args.skip_index:
            backfill_args.append("--skip-index")
        if args.skip_kosdaq:
            backfill_args.append("--skip-kosdaq")
        if args.skip_ohlcv:
            backfill_args.append("--skip-ohlcv")
        if args.flow_only_missing:
            backfill_args.append("--flow-only-missing")

        print(
            "Updating SQLite daily store"
            + (f" from {start.isoformat()}" if start is not None else "")
            + f" to {target_end.isoformat()}..."
        )
        rc = backfill_lrs_main(backfill_args)
        if rc:
            return int(rc)
    else:
        reason = "export-only requested" if args.export_only else "daily Parquet already covers target"
        print(f"Skipping SQLite update: {reason}.")

    db_start_after, db_end_after = _sqlite_daily_range(Path(args.db_path))
    if not _range_covers(db_start_after, db_end_after, history_start, target_end):
        print(
            "ERROR: SQLite source does not cover the full requested daily history; "
            "refusing to export a truncated Parquet mirror."
        )
        print(
            json.dumps(
                {
                    "history_start": history_start.isoformat(),
                    "target_end": target_end.isoformat(),
                    "sqlite_min_date": db_start_after.isoformat() if db_start_after else "",
                    "sqlite_max_date": db_end_after.isoformat() if db_end_after else "",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    print("Exporting full daily Parquet mirror from SQLite...")
    universe = load_universe(args.universe_file)
    manifest = export_lrs_sqlite_to_parquet(
        args.db_path,
        args.parquet_root,
        universe=universe,
        source_label=args.source_label,
    )
    print(f"Daily Parquet mirror: {manifest['root']}")
    print(f"Daily Parquet fingerprint: {manifest['source_fingerprint']}")

    validation = _validate_manifest(
        manifest,
        target_end,
        expected_symbols=len(universe),
        history_start=history_start,
    )
    print(json.dumps(validation, indent=2, sort_keys=True))
    if validation["passed"] or args.no_validate_target:
        return 0
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Update daily PyKRX data, then rewrite the full daily Parquet mirror."
    )
    parser.add_argument("--db-path", default=os.environ.get("LRS_DB_PATH", "data/lrs.db"))
    parser.add_argument("--parquet-root", default="data/krx_daily_parquet")
    parser.add_argument("--universe-file", default=DEFAULT_UNIVERSE)
    parser.add_argument("--config", default="config/olr/sector_map.yaml")
    parser.add_argument("--from", dest="start", default=None, help="Override incremental start date, YYYY-MM-DD.")
    parser.add_argument("--to", dest="end", default=None, help="Target end date, YYYY-MM-DD. Defaults to last completed KRX weekday.")
    parser.add_argument(
        "--history-start",
        default=DEFAULT_HISTORY_START.isoformat(),
        help="Earliest date the full Parquet mirror must retain.",
    )
    parser.add_argument("--sleep-sec", type=float, default=0.15)
    parser.add_argument("--source-label", default="scripts/update_krx_daily.py")
    parser.add_argument("--force", action="store_true", help="Run the SQLite update even if the manifest already covers --to.")
    parser.add_argument("--export-only", action="store_true", help="Skip PyKRX calls and only rebuild the full Parquet mirror.")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--skip-kosdaq", action="store_true")
    parser.add_argument("--skip-ohlcv", action="store_true")
    parser.add_argument("--flow-only-missing", action="store_true")
    parser.add_argument("--no-validate-target", action="store_true", help="Do not fail when exported tables do not reach --to.")
    return parser


def _choose_update_start(
    *,
    history_start: date,
    target_end: date,
    db_start: date | None,
    db_end: date | None,
    manifest_end: date | None,
) -> date | None:
    if db_start is None or db_end is None:
        return history_start
    if db_start > history_start:
        return history_start
    if db_end < target_end:
        return db_end + timedelta(days=1)
    if manifest_end is not None and manifest_end < target_end:
        return manifest_end + timedelta(days=1)
    return target_end + timedelta(days=1)


def _range_covers(start: date | None, end: date | None, required_start: date, required_end: date) -> bool:
    return start is not None and end is not None and start <= required_start and end >= required_end


def _default_completed_krx_day() -> date:
    now = datetime.now(KST)
    candidate = now.date()
    if candidate.weekday() >= 5 or now.time() < dt_time(18, 0):
        candidate -= timedelta(days=1)
    return _previous_weekday(candidate)


def _previous_weekday(day: date) -> date:
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def _manifest_daily_range(parquet_root: Path) -> tuple[date | None, date | None]:
    manifest_path = parquet_root / "manifest.json"
    if not manifest_path.exists():
        return None, None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    return (
        _parse_manifest_date(manifest, ("tables", "daily_ohlcv", "min_date")),
        _parse_manifest_date(manifest, ("tables", "daily_ohlcv", "max_date"))
        or _parse_manifest_date(manifest, ("end",)),
    )


def _parse_manifest_date(manifest: dict[str, object], path: tuple[str, ...]) -> date | None:
    value = _nested_get(manifest, path)
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _sqlite_daily_range(db_path: Path) -> tuple[date | None, date | None]:
    if not db_path.exists():
        return None, None
    import sqlite3

    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT MIN(date), MAX(date) FROM daily_ohlcv").fetchone()
    except sqlite3.DatabaseError:
        return None, None
    if not row or not row[0] or not row[1]:
        return None, None
    try:
        return date.fromisoformat(str(row[0])), date.fromisoformat(str(row[1]))
    except ValueError:
        return None, None


def _nested_get(payload: dict[str, object], path: tuple[str, ...]) -> object | None:
    current: object = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _validate_manifest(
    manifest: dict[str, object],
    target_end: date,
    *,
    expected_symbols: int,
    history_start: date,
) -> dict[str, object]:
    tables = manifest.get("tables") if isinstance(manifest.get("tables"), dict) else {}
    failures: list[dict[str, object]] = []
    checked: dict[str, object] = {}
    for table in ("daily_ohlcv", "daily_flow", "daily_foreign_flow", "daily_institutional_flow"):
        summary = tables.get(table, {}) if isinstance(tables, dict) else {}
        min_date = str(summary.get("min_date", "")) if isinstance(summary, dict) else ""
        max_date = str(summary.get("max_date", "")) if isinstance(summary, dict) else ""
        ticker_count = int(summary.get("ticker_count", 0)) if isinstance(summary, dict) else 0
        checked[table] = {"min_date": min_date, "max_date": max_date, "ticker_count": ticker_count}
        if min_date and date.fromisoformat(min_date) > history_start:
            failures.append({"table": table, "error": "truncated_min_date", "min_date": min_date})
        if max_date and date.fromisoformat(max_date) < target_end:
            failures.append({"table": table, "error": "stale_max_date", "max_date": max_date})
        if expected_symbols and ticker_count != expected_symbols:
            failures.append({"table": table, "error": "ticker_count_mismatch", "ticker_count": ticker_count})
    return {
        "passed": not failures,
        "target_end": target_end.isoformat(),
        "expected_symbols": expected_symbols,
        "checked": checked,
        "failures": failures,
    }


if __name__ == "__main__":
    raise SystemExit(main())
