#!/usr/bin/env python3
"""Incrementally refresh KIS intraday CSV and Parquet data.

The historical helper that produced data/kis_intraday and
data/kis_intraday_parquet is not tracked in this repository.  This script
recreates that pipeline:

1. Fetch 1-minute domestic stock bars from KIS.
2. Merge them with the existing raw/parquet history using an overlap window.
3. Write a new rolling CSV window per symbol/timeframe.
4. Convert all changed CSV files to Parquet after the download phase.
5. Validate that the Parquet dataset used by backtests sees the target date.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kis_core import KoreaInvestAPI, KoreaInvestEnv, build_kis_config_from_env
import kis_core.kis_client as kis_client_module


KST = ZoneInfo("Asia/Seoul")
MARKET_OPEN = dt_time(9, 0)
MARKET_CLOSE = dt_time(15, 30)
DATASET_SOURCE = "scripts/update_kis_intraday.py"
DEFAULT_TIMEFRAMES = ("1m", "5m", "15m", "30m", "1h")
DERIVED_TIMEFRAMES = ("5m", "15m", "30m", "1h")
KIS_PAGE_ROWS = 120
KIS_INTRADAY_RETENTION_DAYS = 365


@dataclass(slots=True)
class CsvArtifact:
    timeframe: str
    path: str
    rows: int
    min_timestamp: str
    max_timestamp: str


@dataclass(slots=True)
class SymbolUpdate:
    symbol: str
    name: str
    status: str
    fetch_start: str
    fetch_end: str
    previous_latest_timestamp: str = ""
    previous_raw_path: str = ""
    fetched_rows: int = 0
    fetched_pages: int = 0
    raw_rows: int = 0
    latest_timestamp: str = ""
    raw_1m_path: str = ""
    timeframe_rows: dict[str, int] = field(default_factory=dict)
    timeframe_paths: dict[str, str] = field(default_factory=dict)
    csv_artifacts: list[CsvArtifact] = field(default_factory=list)
    error: str = ""


@dataclass(slots=True)
class ConversionResult:
    csv_path: str
    parquet_path: str
    status: str
    rows: int = 0
    source_bytes: int = 0
    parquet_bytes: int = 0
    message: str = ""


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _load_dotenv(PROJECT_ROOT / ".env")
    configure_kis_request_rate(float(args.request_min_interval_sec))

    symbols = _resolve_symbols(args)
    target_end = date.fromisoformat(args.to)
    timeframes = _normalize_timeframes(args.timeframes)
    raw_root = _resolve_path(args.input_dir)
    parquet_root = _resolve_path(args.parquet_output_dir)
    start_override = date.fromisoformat(args.start) if args.start else None

    api: KoreaInvestAPI | None = None
    if not args.dry_run:
        api = KoreaInvestAPI(KoreaInvestEnv(build_kis_config_from_env()))

    started_at = datetime.now(timezone.utc)
    updates: list[SymbolUpdate] = []
    changed_csvs: list[Path] = []
    failures: list[dict[str, Any]] = []

    updates, changed_csvs, download_failures = download_symbols(
        symbols=symbols,
        api=api,
        raw_root=raw_root,
        parquet_root=parquet_root,
        target_end=target_end,
        start_override=start_override,
        overlap_days=int(args.overlap_days),
        retention_days=int(args.retention_days),
        timeframes=timeframes,
        market_code=args.market_code,
        extra_sleep_sec=float(args.extra_sleep_sec),
        max_pages_per_symbol=args.max_pages_per_symbol,
        dry_run=bool(args.dry_run),
        skip_up_to_date=bool(args.skip_up_to_date),
        download_workers=max(1, int(args.download_workers)),
        fail_fast=bool(args.fail_fast),
    )
    failures.extend(download_failures)

    incremental_manifest = write_incremental_manifest(
        raw_root=raw_root,
        parquet_root=parquet_root,
        updates=updates,
        target_end=target_end,
        started_at=started_at,
        overlap_days=int(args.overlap_days),
        retention_days=int(args.retention_days),
        timeframes=timeframes,
        args=args,
    )

    conversion_results: list[ConversionResult] = []
    if args.convert and changed_csvs and not args.dry_run:
        print(f"Converting {len(changed_csvs)} CSV files to Parquet after download phase", flush=True)
        conversion_results = convert_csvs_to_parquet(
            csv_paths=changed_csvs,
            raw_root=raw_root,
            parquet_root=parquet_root,
            compression=args.compression,
            overwrite=bool(args.overwrite),
            workers=max(1, int(args.workers)),
            min_file_age_sec=max(0.0, float(args.min_file_age_sec)),
        )
        for row in conversion_results:
            if row.status == "error":
                failures.append({"stage": "convert", "path": row.csv_path, "error": row.message})

    validation = (
        validate_parquet_dataset(
            parquet_root=parquet_root,
            symbols=symbols,
            timeframes=timeframes,
            target_end=target_end,
        )
        if not args.dry_run
        else {"passed": True, "skipped": True}
    )

    conversion_manifest = write_conversion_manifest(
        raw_root=raw_root,
        parquet_root=parquet_root,
        symbols=symbols,
        timeframes=timeframes,
        conversion_results=conversion_results,
        incremental_manifest=incremental_manifest,
        validation=validation,
        args=args,
    )

    if not validation.get("passed", False) and not args.dry_run:
        failures.extend(validation.get("failures", []))

    summary = {
        "passed": not failures and bool(validation.get("passed", args.dry_run)),
        "dry_run": bool(args.dry_run),
        "symbols_requested": len(symbols),
        "symbols_updated": sum(1 for row in updates if row.status == "updated"),
        "symbols_skipped_up_to_date": sum(1 for row in updates if row.status == "skipped_up_to_date"),
        "download_failures": sum(1 for row in updates if not _download_status_ok(row.status)),
        "target_end": target_end.isoformat(),
        "incremental_manifest": str(incremental_manifest),
        "conversion_manifest": str(conversion_manifest),
        "validation": validation,
        "failures": failures[:20],
    }
    print(json.dumps(summary, indent=2, sort_keys=True, default=str), flush=True)
    return 0 if summary["passed"] else 1


def update_symbol(
    *,
    symbol: str,
    api: KoreaInvestAPI | None,
    raw_root: Path,
    parquet_root: Path,
    target_end: date,
    start_override: date | None,
    overlap_days: int,
    retention_days: int,
    timeframes: Sequence[str],
    market_code: str,
    extra_sleep_sec: float,
    max_pages_per_symbol: int | None,
    dry_run: bool,
    skip_up_to_date: bool,
) -> SymbolUpdate:
    retention_start = target_end - timedelta(days=retention_days)
    output_start_dt = _combine_kst(retention_start, MARKET_OPEN)
    output_end_dt = _combine_kst(target_end, MARKET_CLOSE)
    previous_latest, previous_path = latest_existing_timestamp(symbol, raw_root, parquet_root)
    fetch_start_date = start_override or (
        (previous_latest.date() - timedelta(days=overlap_days)) if previous_latest is not None else retention_start
    )
    if fetch_start_date < retention_start:
        fetch_start_date = retention_start
    fetch_start_dt = _combine_kst(fetch_start_date, MARKET_OPEN)

    update = SymbolUpdate(
        symbol=symbol,
        name=symbol,
        status="updated",
        fetch_start=fetch_start_dt.isoformat(),
        fetch_end=output_end_dt.isoformat(),
        previous_latest_timestamp=previous_latest.isoformat() if previous_latest is not None else "",
        previous_raw_path=str(previous_path) if previous_path is not None else "",
    )
    if (
        skip_up_to_date
        and previous_latest is not None
        and previous_latest.date() >= target_end
        and parquet_symbol_covers_target(parquet_root, symbol, timeframes, target_end)
        and parquet_symbol_covers_date_window(parquet_root, symbol, timeframes, fetch_start_date, target_end)
    ):
        update.status = "skipped_up_to_date"
        update.latest_timestamp = previous_latest.isoformat()
        return update

    existing = load_existing_1m(symbol, raw_root, parquet_root, output_start_dt, output_end_dt)
    fetched = pd.DataFrame(columns=_bar_columns())
    pages = 0
    if not dry_run:
        if api is None:
            raise RuntimeError("KIS API client is required unless --dry-run is used")
        fetched, pages = fetch_kis_1m_bars(
            api,
            symbol,
            fetch_start_dt,
            output_end_dt,
            market_code=market_code,
            extra_sleep_sec=extra_sleep_sec,
            max_pages=max_pages_per_symbol,
        )
    update.fetched_rows = int(len(fetched))
    update.fetched_pages = int(pages)

    merged = merge_1m_frames(existing, fetched, start_dt=output_start_dt, end_dt=output_end_dt)
    if merged.empty:
        update.status = "empty"
        update.error = "no existing or fetched 1m rows in requested output window"
        return update

    update.raw_rows = int(len(merged))
    update.latest_timestamp = _max_timestamp_string(merged)

    symbol_dir = raw_root / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    start_label = retention_start.strftime("%Y%m%d")
    end_label = target_end.strftime("%Y%m%d")

    frames_by_timeframe: dict[str, pd.DataFrame] = {}
    for timeframe in timeframes:
        frames_by_timeframe[timeframe] = aggregate_timeframe(merged, timeframe)

    artifacts: list[CsvArtifact] = []
    for timeframe, frame in frames_by_timeframe.items():
        path = symbol_dir / f"{symbol}_{timeframe}_{start_label}_{end_label}.csv"
        if not dry_run:
            write_bar_csv(frame, path)
        min_ts = _min_timestamp_string(frame)
        max_ts = _max_timestamp_string(frame)
        artifact = CsvArtifact(
            timeframe=timeframe,
            path=str(path),
            rows=int(len(frame)),
            min_timestamp=min_ts,
            max_timestamp=max_ts,
        )
        artifacts.append(artifact)
        update.timeframe_rows[timeframe] = int(len(frame))
        update.timeframe_paths[timeframe] = str(path)
        if timeframe == "1m":
            update.raw_1m_path = str(path)

    update.csv_artifacts = artifacts
    return update


def download_symbols(
    *,
    symbols: Sequence[str],
    api: KoreaInvestAPI | None,
    raw_root: Path,
    parquet_root: Path,
    target_end: date,
    start_override: date | None,
    overlap_days: int,
    retention_days: int,
    timeframes: Sequence[str],
    market_code: str,
    extra_sleep_sec: float,
    max_pages_per_symbol: int | None,
    dry_run: bool,
    skip_up_to_date: bool,
    download_workers: int,
    fail_fast: bool,
) -> tuple[list[SymbolUpdate], list[Path], list[dict[str, Any]]]:
    total = len(symbols)
    updates_by_symbol: dict[str, SymbolUpdate] = {}
    changed_csvs: list[Path] = []
    failures: list[dict[str, Any]] = []
    target_end_dt = _combine_kst(target_end, MARKET_CLOSE)

    def run_one(index: int, symbol: str) -> SymbolUpdate:
        print(f"[{index:03d}/{total:03d}] updating {symbol}", flush=True)
        return update_symbol(
            symbol=symbol,
            api=api,
            raw_root=raw_root,
            parquet_root=parquet_root,
            target_end=target_end,
            start_override=start_override,
            overlap_days=overlap_days,
            retention_days=retention_days,
            timeframes=timeframes,
            market_code=market_code,
            extra_sleep_sec=extra_sleep_sec,
            max_pages_per_symbol=max_pages_per_symbol,
            dry_run=dry_run,
            skip_up_to_date=skip_up_to_date,
        )

    if download_workers <= 1:
        for index, symbol in enumerate(symbols, start=1):
            try:
                update = run_one(index, symbol)
            except Exception as exc:
                update = _failed_symbol_update(symbol, target_end_dt, exc)
            updates_by_symbol[symbol] = update
            _collect_download_result(update, changed_csvs, failures)
            if fail_fast and not _download_status_ok(update.status):
                break
    else:
        print(f"Download workers: {download_workers} symbols in parallel; KIS requests remain globally rate-limited", flush=True)
        with ThreadPoolExecutor(max_workers=download_workers) as pool:
            future_map = {
                pool.submit(run_one, index, symbol): symbol
                for index, symbol in enumerate(symbols, start=1)
            }
            for completed_count, future in enumerate(as_completed(future_map), start=1):
                symbol = future_map[future]
                try:
                    update = future.result()
                except Exception as exc:
                    update = _failed_symbol_update(symbol, target_end_dt, exc)
                updates_by_symbol[symbol] = update
                _collect_download_result(update, changed_csvs, failures)
                print(
                    f"Completed {completed_count}/{total}: {symbol} status={update.status} "
                    f"rows={update.raw_rows} latest={update.latest_timestamp or '-'}",
                    flush=True,
                )
                if fail_fast and not _download_status_ok(update.status):
                    for pending in future_map:
                        pending.cancel()
                    break

    ordered_updates = [updates_by_symbol[symbol] for symbol in symbols if symbol in updates_by_symbol]
    return ordered_updates, changed_csvs, failures


def _collect_download_result(update: SymbolUpdate, changed_csvs: list[Path], failures: list[dict[str, Any]]) -> None:
    changed_csvs.extend(Path(item.path) for item in update.csv_artifacts if item.path)
    if not _download_status_ok(update.status):
        failures.append({"symbol": update.symbol, "stage": "download", "error": update.error or update.status})


def _download_status_ok(status: str) -> bool:
    return status in {"updated", "skipped_up_to_date"}


def _failed_symbol_update(symbol: str, target_end_dt: datetime, exc: Exception) -> SymbolUpdate:
    error = f"{type(exc).__name__}: {exc}"
    print(f"  ERROR {symbol}: {error}", flush=True)
    return SymbolUpdate(
        symbol=symbol,
        name=symbol,
        status="error",
        fetch_start="",
        fetch_end=target_end_dt.isoformat(),
        error=error,
    )


def fetch_kis_1m_bars(
    api: KoreaInvestAPI,
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    *,
    market_code: str,
    extra_sleep_sec: float,
    max_pages: int | None,
) -> tuple[pd.DataFrame, int]:
    rows: dict[pd.Timestamp, dict[str, Any]] = {}
    cursor = end_dt
    pages = 0
    seen_cursors: set[str] = set()

    while cursor >= start_dt:
        cursor_key = cursor.isoformat()
        if cursor_key in seen_cursors:
            raise RuntimeError(f"KIS pagination did not advance for {symbol} at {cursor_key}")
        seen_cursors.add(cursor_key)
        if max_pages is not None and pages >= int(max_pages):
            break

        page = fetch_kis_1m_page(api, symbol, cursor, market_code=market_code)
        pages += 1
        if extra_sleep_sec > 0:
            time.sleep(extra_sleep_sec)
        if pages % 10 == 0:
            print(
                f"  {symbol} pages={pages} rows={len(rows)} cursor={cursor.astimezone(KST).isoformat()}",
                flush=True,
            )

        normalized = [row for row in (_normalize_kis_minute_row(item) for item in page) if row is not None]
        normalized = [row for row in normalized if start_dt <= row["timestamp"] <= end_dt]
        if not normalized:
            cursor = _previous_day_close(cursor)
            continue

        for row in normalized:
            rows[pd.Timestamp(row["timestamp"])] = row
        oldest = min(row["timestamp"] for row in normalized)
        if oldest <= start_dt:
            break
        next_cursor = oldest - timedelta(minutes=1)
        if next_cursor >= cursor:
            cursor = _previous_day_close(cursor)
        else:
            cursor = next_cursor

    if not rows:
        return pd.DataFrame(columns=_bar_columns()), pages
    frame = pd.DataFrame(list(rows.values()))
    return normalize_bar_frame(frame), pages


def fetch_kis_1m_page(api: KoreaInvestAPI, symbol: str, cursor: datetime, *, market_code: str) -> list[Mapping[str, Any]]:
    endpoint = "/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice"
    params = {
        "FID_COND_MRKT_DIV_CODE": market_code,
        "FID_INPUT_ISCD": symbol,
        "FID_INPUT_DATE_1": cursor.astimezone(KST).strftime("%Y%m%d"),
        "FID_INPUT_HOUR_1": cursor.astimezone(KST).strftime("%H%M%S"),
        "FID_PW_DATA_INCU_YN": "Y",
        "FID_FAKE_TICK_INCU_YN": "",
    }
    response = None
    last_error = ""
    for attempt in range(1, 6):
        response = api._url_fetch(endpoint, "FHKST03010230", params, retry_on_failure=True)
        if response is not None and response.is_ok():
            break
        if response is not None:
            last_error = f"{response.error_code} {response.error_message}"
        else:
            last_error = "no_response"
        if attempt < 5:
            time.sleep(min(30.0, 2.0 * attempt))
    if response is None:
        raise RuntimeError(
            f"KIS returned no response for {symbol} at {params['FID_INPUT_DATE_1']} "
            f"{params['FID_INPUT_HOUR_1']} after retries ({last_error})"
        )
    if not response.is_ok():
        raise RuntimeError(f"KIS error for {symbol}: {response.error_code} {response.error_message}")
    body = response.get_body()
    output = getattr(body, "output2", None)
    if output is None:
        output = getattr(body, "output", None)
    if output is None:
        return []
    if isinstance(output, Mapping):
        return [output]
    return [item for item in output if isinstance(item, Mapping)]


def latest_existing_timestamp(symbol: str, raw_root: Path, parquet_root: Path) -> tuple[datetime | None, Path | None]:
    candidates = [
        *(parquet_root / symbol).glob(f"{symbol}_1m_*.parquet"),
        *(raw_root / symbol).glob(f"{symbol}_1m_*.csv"),
    ]
    best_ts: datetime | None = None
    best_path: Path | None = None
    for path in sorted(candidates):
        try:
            if path.suffix.lower() == ".parquet":
                frame = pd.read_parquet(path, columns=["timestamp"])
            else:
                frame = pd.read_csv(path, usecols=["timestamp"])
            if frame.empty:
                continue
            ts = _coerce_timestamp_series(frame["timestamp"]).max()
            if pd.isna(ts):
                continue
            current = _timestamp_to_datetime(ts)
            if best_ts is None or current > best_ts:
                best_ts = current
                best_path = path
        except Exception:
            continue
    return best_ts, best_path


def parquet_symbol_covers_target(parquet_root: Path, symbol: str, timeframes: Sequence[str], target_end: date) -> bool:
    for timeframe in timeframes:
        files = sorted((parquet_root / symbol).glob(f"{symbol}_{timeframe}_*.parquet"))
        if not files:
            return False
        latest: datetime | None = None
        for path in files:
            try:
                frame = pd.read_parquet(path, columns=["timestamp"])
            except Exception:
                continue
            if frame.empty:
                continue
            ts = _timestamp_to_datetime(_coerce_timestamp_series(frame["timestamp"]).max())
            if latest is None or ts > latest:
                latest = ts
        if latest is None or latest.date() < target_end:
            return False
    return True


def parquet_symbol_covers_date_window(
    parquet_root: Path,
    symbol: str,
    timeframes: Sequence[str],
    start: date,
    end: date,
) -> bool:
    expected_dates = set(expected_trading_dates(start, end))
    if not expected_dates:
        return True
    for timeframe in timeframes:
        files = sorted((parquet_root / symbol).glob(f"{symbol}_{timeframe}_*.parquet"))
        if not files:
            return False
        dates_seen: set[date] = set()
        for path in files:
            try:
                frame = pd.read_parquet(path, columns=["timestamp"])
            except Exception:
                continue
            if frame.empty:
                continue
            ts = _coerce_timestamp_series(frame["timestamp"])
            dates_seen.update(item.date() for item in ts.dropna())
        if not expected_dates.issubset(dates_seen):
            return False
    return True


_EXPECTED_TRADING_DATES_CACHE: dict[tuple[date, date], list[date]] = {}


def expected_trading_dates(start: date, end: date) -> list[date]:
    key = (start, end)
    if key in _EXPECTED_TRADING_DATES_CACHE:
        return list(_EXPECTED_TRADING_DATES_CACHE[key])
    daily_index = PROJECT_ROOT / "data" / "krx_daily_parquet" / "tables" / "index_ohlcv.parquet"
    dates: list[date] = []
    if daily_index.exists():
        try:
            frame = pd.read_parquet(daily_index, columns=["date"])
            parsed = pd.to_datetime(frame["date"]).dt.date
            dates = sorted({item for item in parsed if start <= item <= end})
        except Exception:
            dates = []
    if not dates:
        cursor = start
        while cursor <= end:
            if cursor.weekday() < 5:
                dates.append(cursor)
            cursor += timedelta(days=1)
    _EXPECTED_TRADING_DATES_CACHE[key] = dates
    return list(dates)


def load_existing_1m(symbol: str, raw_root: Path, parquet_root: Path, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    parquet_files = sorted((parquet_root / symbol).glob(f"{symbol}_1m_*.parquet"))
    csv_files = sorted((raw_root / symbol).glob(f"{symbol}_1m_*.csv"))
    for path in parquet_files:
        if not _intraday_file_may_overlap(path, start_dt.date(), end_dt.date()):
            continue
        try:
            frames.append(pd.read_parquet(path))
        except Exception:
            continue
    if not frames:
        for path in csv_files:
            if not _intraday_file_may_overlap(path, start_dt.date(), end_dt.date()):
                continue
            try:
                frames.append(read_bar_csv(path))
            except Exception:
                continue
    if not frames:
        return pd.DataFrame(columns=_bar_columns())
    return merge_1m_frames(*frames, start_dt=start_dt, end_dt=end_dt)


def merge_1m_frames(*frames: pd.DataFrame, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=_bar_columns())
    data = pd.concat(non_empty, ignore_index=True)
    data = normalize_bar_frame(data)
    mask = (data["timestamp"] >= pd.Timestamp(start_dt)) & (data["timestamp"] <= pd.Timestamp(end_dt))
    data = data.loc[mask]
    data = data.drop_duplicates(subset=["timestamp"], keep="last")
    return data.sort_values("timestamp").reset_index(drop=True)


def normalize_bar_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in _bar_columns():
        if column not in data.columns:
            data[column] = 0 if column == "volume" else pd.NA
    data = data[_bar_columns()]
    data["timestamp"] = _coerce_timestamp_series(data["timestamp"])
    data = data.dropna(subset=["timestamp"])
    for column in ("open", "high", "low", "close"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["volume"] = pd.to_numeric(data["volume"], errors="coerce").fillna(0).astype("int64")
    data = data.dropna(subset=["open", "high", "low", "close"])
    return data.sort_values("timestamp").reset_index(drop=True)


def aggregate_timeframe(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    tf = timeframe.lower()
    if tf == "1m":
        return normalize_bar_frame(frame)
    minutes = _timeframe_minutes(tf)
    data = normalize_bar_frame(frame)
    if data.empty:
        return pd.DataFrame(columns=_bar_columns())
    indexed = data.set_index("timestamp").sort_index()
    aggregated = indexed.resample(
        f"{minutes}min",
        label="left",
        closed="left",
        origin="start_day",
    ).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    aggregated = aggregated.dropna(subset=["open", "high", "low", "close"]).reset_index()
    aggregated["volume"] = aggregated["volume"].fillna(0).astype("int64")
    return aggregated[_bar_columns()]


def read_bar_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return normalize_bar_frame(frame)


def write_bar_csv(frame: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    normalize_bar_frame(frame).to_csv(tmp, index=False)
    tmp.replace(target)


def convert_csvs_to_parquet(
    *,
    csv_paths: Iterable[Path],
    raw_root: Path,
    parquet_root: Path,
    compression: str,
    overwrite: bool,
    workers: int,
    min_file_age_sec: float,
) -> list[ConversionResult]:
    unique_paths = sorted({Path(path) for path in csv_paths})
    results: list[ConversionResult] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(
                convert_one_csv_to_parquet,
                path,
                raw_root=raw_root,
                parquet_root=parquet_root,
                compression=compression,
                overwrite=overwrite,
                min_file_age_sec=min_file_age_sec,
            ): path
            for path in unique_paths
        }
        for future in as_completed(future_map):
            results.append(future.result())
    return sorted(results, key=lambda item: item.csv_path)


def convert_one_csv_to_parquet(
    csv_path: Path,
    *,
    raw_root: Path,
    parquet_root: Path,
    compression: str,
    overwrite: bool,
    min_file_age_sec: float,
) -> ConversionResult:
    try:
        rel = csv_path.resolve().relative_to(raw_root.resolve())
        parquet_path = (parquet_root / rel).with_suffix(".parquet")
    except ValueError:
        parquet_path = parquet_root / csv_path.parent.name / csv_path.with_suffix(".parquet").name

    result = ConversionResult(
        csv_path=str(csv_path),
        parquet_path=str(parquet_path),
        status="converted",
        source_bytes=csv_path.stat().st_size if csv_path.exists() else 0,
    )
    try:
        if not csv_path.exists():
            result.status = "error"
            result.message = "csv_missing"
            return result
        age = time.time() - csv_path.stat().st_mtime
        if age < min_file_age_sec:
            result.status = "skipped"
            result.message = f"file_age_below_minimum:{age:.3f}s"
            return result
        if parquet_path.exists() and not overwrite:
            result.status = "skipped"
            result.message = "parquet_exists"
            result.parquet_bytes = parquet_path.stat().st_size
            return result
        frame = read_bar_csv(csv_path)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = parquet_path.with_name(f"{parquet_path.name}.{os.getpid()}.tmp")
        frame.to_parquet(tmp, index=False, compression=compression)
        tmp.replace(parquet_path)
        result.rows = int(len(frame))
        result.parquet_bytes = parquet_path.stat().st_size
        return result
    except Exception as exc:
        result.status = "error"
        result.message = f"{type(exc).__name__}: {exc}"
        return result


def validate_parquet_dataset(
    *,
    parquet_root: Path,
    symbols: Sequence[str],
    timeframes: Sequence[str],
    target_end: date,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    checked = 0
    per_timeframe: dict[str, dict[str, Any]] = {}
    for timeframe in timeframes:
        tf_summary = {"checked": 0, "latest_min": "", "latest_max": "", "missing": 0, "stale": 0}
        latest_values: list[datetime] = []
        for symbol in symbols:
            files = sorted((parquet_root / symbol).glob(f"{symbol}_{timeframe}_*.parquet"))
            if not files:
                failures.append({"stage": "validate", "symbol": symbol, "timeframe": timeframe, "error": "missing_parquet"})
                tf_summary["missing"] += 1
                continue
            latest: datetime | None = None
            for path in files:
                try:
                    frame = pd.read_parquet(path, columns=["timestamp"])
                except Exception as exc:
                    failures.append(
                        {
                            "stage": "validate",
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "path": str(path),
                            "error": f"read_error:{type(exc).__name__}",
                        }
                    )
                    continue
                if frame.empty:
                    continue
                ts = _timestamp_to_datetime(_coerce_timestamp_series(frame["timestamp"]).max())
                if latest is None or ts > latest:
                    latest = ts
            if latest is None:
                failures.append({"stage": "validate", "symbol": symbol, "timeframe": timeframe, "error": "empty_parquet"})
                tf_summary["missing"] += 1
                continue
            checked += 1
            tf_summary["checked"] += 1
            latest_values.append(latest)
            if latest.date() < target_end:
                failures.append(
                    {
                        "stage": "validate",
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "error": "stale_latest_timestamp",
                        "latest": latest.isoformat(),
                        "target_end": target_end.isoformat(),
                    }
                )
                tf_summary["stale"] += 1
        if latest_values:
            tf_summary["latest_min"] = min(latest_values).isoformat()
            tf_summary["latest_max"] = max(latest_values).isoformat()
        per_timeframe[timeframe] = tf_summary
    return {
        "passed": not failures,
        "checked": checked,
        "target_end": target_end.isoformat(),
        "timeframes": per_timeframe,
        "failures": failures[:50],
    }


def write_incremental_manifest(
    *,
    raw_root: Path,
    parquet_root: Path,
    updates: Sequence[SymbolUpdate],
    target_end: date,
    started_at: datetime,
    overlap_days: int,
    retention_days: int,
    timeframes: Sequence[str],
    args: argparse.Namespace,
) -> Path:
    payload = {
        "updated_at": datetime.now(KST).isoformat(),
        "started_at_utc": started_at.isoformat(),
        "run_type": "incremental",
        "source": DATASET_SOURCE,
        "download_workers": int(args.download_workers),
        "kis_limits": {
            "historical_intraday_retention_days": retention_days,
            "historical_intraday_page_rows": KIS_PAGE_ROWS,
            "extra_sleep_sec": float(args.extra_sleep_sec),
            "request_min_interval_sec": float(args.request_min_interval_sec),
            "client_mode": "paper" if os.environ.get("KIS_IS_PAPER", "true").lower() == "true" else "live",
        },
        "date_window": {
            "start": (target_end - timedelta(days=retention_days)).isoformat(),
            "end": target_end.isoformat(),
        },
        "overlap_days": overlap_days,
        "skip_up_to_date": bool(args.skip_up_to_date),
        "timeframes": {tf: _timeframe_minutes(tf) for tf in timeframes if tf != "1m"},
        "raw_timeframe": "1m",
        "excluded_symbols": list(args.exclude_symbols or []),
        "prune_old": bool(args.prune_old),
        "parquet_output_dir": str(parquet_root),
        "symbols": [_symbol_update_to_json(row) for row in updates],
    }
    path = raw_root / "incremental_manifest.json"
    _atomic_write_json(path, payload)
    return path


def write_conversion_manifest(
    *,
    raw_root: Path,
    parquet_root: Path,
    symbols: Sequence[str],
    timeframes: Sequence[str],
    conversion_results: Sequence[ConversionResult],
    incremental_manifest: Path,
    validation: Mapping[str, Any],
    args: argparse.Namespace,
) -> Path:
    counts = {
        "converted": sum(1 for row in conversion_results if row.status == "converted"),
        "skipped": sum(1 for row in conversion_results if row.status == "skipped"),
        "errors": sum(1 for row in conversion_results if row.status == "error"),
        "total": len(conversion_results),
    }
    payload = {
        "created_at": datetime.now().isoformat(),
        "input_dir": str(raw_root),
        "parquet_output_dir": str(parquet_root),
        "symbols": list(symbols),
        "timeframes": list(timeframes),
        "compression": args.compression,
        "workers": int(args.workers),
        "download_workers": int(args.download_workers),
        "request_min_interval_sec": float(args.request_min_interval_sec),
        "skip_up_to_date": bool(args.skip_up_to_date),
        "overwrite": bool(args.overwrite),
        "min_file_age_sec": float(args.min_file_age_sec),
        "source": DATASET_SOURCE,
        "incremental_manifest": str(incremental_manifest),
        "convert_after_download": True,
        "counts": counts,
        "validation": dict(validation),
        "files": [asdict(row) for row in conversion_results],
    }
    path = parquet_root / "conversion_manifest.json"
    _atomic_write_json(path, payload)
    return path


def _symbol_update_to_json(update: SymbolUpdate) -> dict[str, Any]:
    payload = asdict(update)
    payload["csv_artifacts"] = [asdict(item) for item in update.csv_artifacts]
    return payload


def _normalize_kis_minute_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    date_text = str(_first_present(row, "stck_bsop_date", "bsop_date", "xymd", "date") or "")
    time_text = str(_first_present(row, "stck_cntg_hour", "cntg_hour", "xhms", "time") or "")
    if len(date_text) != 8 or len(time_text) < 6:
        return None
    try:
        ts = datetime.strptime(date_text + time_text[:6], "%Y%m%d%H%M%S").replace(tzinfo=KST)
    except ValueError:
        return None
    close = _float_value(_first_present(row, "stck_prpr", "stck_clpr", "close"))
    item = {
        "timestamp": ts,
        "open": _float_value(_first_present(row, "stck_oprc", "open"), close),
        "high": _float_value(_first_present(row, "stck_hgpr", "high"), close),
        "low": _float_value(_first_present(row, "stck_lwpr", "low"), close),
        "close": close,
        "volume": int(_float_value(_first_present(row, "cntg_vol", "acml_vol", "volume"), 0.0)),
    }
    if min(item["open"], item["high"], item["low"], item["close"]) <= 0:
        return None
    return item


def _resolve_symbols(args: argparse.Namespace) -> tuple[str, ...]:
    values = list(args.symbols or [])
    if not values:
        payload = yaml.safe_load(_resolve_path(args.universe_file).read_text(encoding="utf-8")) or {}
        raw = payload.get("symbols") if isinstance(payload, Mapping) else payload
        values = list(raw or [])
    excluded = {str(item).zfill(6) for item in args.exclude_symbols or []}
    symbols = [str(item).zfill(6) for item in values if str(item).strip()]
    symbols = [symbol for symbol in dict.fromkeys(symbols) if symbol not in excluded]
    if args.limit_symbols:
        symbols = symbols[: int(args.limit_symbols)]
    if not symbols:
        raise ValueError("no symbols resolved")
    return tuple(symbols)


def _normalize_timeframes(raw: Sequence[str]) -> tuple[str, ...]:
    values = []
    for item in raw or DEFAULT_TIMEFRAMES:
        value = str(item).strip().lower()
        if not value:
            continue
        _timeframe_minutes(value)
        values.append(value)
    if "1m" not in values:
        values.insert(0, "1m")
    return tuple(dict.fromkeys(values))


def _timeframe_minutes(timeframe: str) -> int:
    value = timeframe.lower().strip()
    if value.endswith("m"):
        minutes = int(value[:-1])
    elif value.endswith("h"):
        minutes = int(value[:-1]) * 60
    else:
        raise ValueError(f"unsupported timeframe {timeframe!r}")
    if minutes <= 0:
        raise ValueError(f"unsupported timeframe {timeframe!r}")
    return minutes


def _coerce_timestamp_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if getattr(parsed.dt, "tz", None) is None:
        return parsed.dt.tz_localize(KST)
    return parsed.dt.tz_convert(KST)


def _timestamp_to_datetime(value: Any) -> datetime:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return value.astimezone(KST) if value.tzinfo else value.replace(tzinfo=KST)
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize(KST)
    else:
        parsed = parsed.tz_convert(KST)
    return parsed.to_pydatetime()


def _intraday_file_may_overlap(path: Path, start: date, end: date) -> bool:
    parts = path.stem.split("_")
    if len(parts) < 4:
        return True
    try:
        file_start = datetime.strptime(parts[-2], "%Y%m%d").date()
        file_end = datetime.strptime(parts[-1], "%Y%m%d").date()
    except ValueError:
        return True
    return file_start <= end and file_end >= start


def _combine_kst(day: date, value: dt_time) -> datetime:
    return datetime.combine(day, value).replace(tzinfo=KST)


def _previous_day_close(cursor: datetime) -> datetime:
    return _combine_kst(cursor.astimezone(KST).date() - timedelta(days=1), MARKET_CLOSE)


def _min_timestamp_string(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    return _timestamp_to_datetime(frame["timestamp"].min()).isoformat()


def _max_timestamp_string(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    return _timestamp_to_datetime(frame["timestamp"].max()).isoformat()


def _bar_columns() -> list[str]:
    return ["timestamp", "open", "high", "low", "close", "volume"]


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).replace(",", "").strip()
        if not text or text == "-":
            return float(default)
        return float(text)
    except (TypeError, ValueError):
        return float(default)


def _first_present(row: Mapping[str, Any], *keys: str) -> Any | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def _resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def configure_kis_request_rate(min_interval_sec: float) -> None:
    if min_interval_sec <= 0:
        return
    limiter = getattr(kis_client_module, "_http_limiter", None)
    if limiter is None:
        return
    if hasattr(limiter, "_min_interval"):
        setattr(limiter, "_min_interval", float(min_interval_sec))
    elif hasattr(limiter, "min_interval"):
        setattr(limiter, "min_interval", float(min_interval_sec))
    print(
        f"KIS request start limiter set to {min_interval_sec:.2f}s "
        f"({1.0 / min_interval_sec:.2f} req/sec max)",
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh KIS intraday CSV/Parquet data.")
    parser.add_argument("--universe-file", default="config/olr_kalcb/olr_deployment_universe_103.yaml")
    parser.add_argument("--symbols", nargs="*", help="Optional symbol subset. Defaults to the universe file.")
    parser.add_argument("--exclude-symbols", nargs="*", default=[])
    parser.add_argument("--limit-symbols", type=int, default=None, help="Debug helper to update only the first N symbols.")
    parser.add_argument("--input-dir", default="data/kis_intraday")
    parser.add_argument("--parquet-output-dir", default="data/kis_intraday_parquet")
    parser.add_argument("--from", dest="start", default=None, help="Override fetch start date, YYYY-MM-DD.")
    parser.add_argument("--to", required=True, help="Target end date, YYYY-MM-DD. Usually the last completed KRX session.")
    parser.add_argument("--overlap-days", type=int, default=7)
    parser.add_argument("--retention-days", type=int, default=KIS_INTRADAY_RETENTION_DAYS)
    parser.add_argument("--timeframes", nargs="+", default=list(DEFAULT_TIMEFRAMES))
    parser.add_argument("--market-code", default="J")
    parser.add_argument("--extra-sleep-sec", type=float, default=0.55)
    parser.add_argument("--max-pages-per-symbol", type=int, default=None)
    parser.add_argument(
        "--download-workers",
        type=int,
        default=4,
        help="Number of symbols to download concurrently. KIS HTTP calls still use the shared rate limiter.",
    )
    parser.add_argument(
        "--request-min-interval-sec",
        type=float,
        default=2.0,
        help="Global minimum interval between KIS HTTP request starts during this update.",
    )
    parser.add_argument("--convert", dest="convert", action="store_true", default=True)
    parser.add_argument("--skip-convert", dest="convert", action="store_false")
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--workers", type=int, default=4, help="CSV-to-Parquet conversion workers.")
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.add_argument("--min-file-age-sec", type=float, default=0.0)
    parser.add_argument("--prune-old", action="store_true", help="Reserved manifest flag; old files are kept by default.")
    parser.add_argument("--skip-up-to-date", dest="skip_up_to_date", action="store_true", default=True)
    parser.add_argument("--no-skip-up-to-date", dest="skip_up_to_date", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
