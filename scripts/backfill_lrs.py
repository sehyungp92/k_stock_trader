#!/usr/bin/env python3
"""Backfill historical Local Research Store data from pykrx.

This loader populates replayable daily OHLCV, investor flow, index history, and
sector mappings. FX is intentionally excluded rather than approximated.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
import xml.etree.ElementTree as et
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
import yaml
from bs4 import BeautifulSoup

from strategy_common.daily_lrs_parquet import export_lrs_sqlite_to_parquet

_PYKRX_FLOW_RETURNED_EMPTY = False
KRX_OPEN = "\uc2dc\uac00"
KRX_HIGH = "\uace0\uac00"
KRX_LOW = "\uc800\uac00"
KRX_CLOSE = "\uc885\uac00"
KRX_VOLUME = "\uac70\ub798\ub7c9"
KRX_NET_BUY = "\uc21c\ub9e4\uc218"
KRX_FOREIGN = "\uc678\uad6d\uc778"
KRX_FOREIGN_TOTAL = "\uc678\uad6d\uc778\ud569\uacc4"
KRX_INSTITUTION = "\uae30\uad00"
KRX_INSTITUTION_TOTAL = "\uae30\uad00\ud569\uacc4"


def ensure_pykrx():
    try:
        from pykrx import stock

        return stock
    except ImportError:
        print("ERROR: pykrx not installed. Run: pip install pykrx")
        return None


def init_db(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS daily_ohlcv (
            ticker TEXT NOT NULL, date TEXT NOT NULL, open REAL, high REAL,
            low REAL, close REAL, volume REAL, PRIMARY KEY (ticker, date)
        );
        CREATE TABLE IF NOT EXISTS daily_flow (
            ticker TEXT NOT NULL, date TEXT NOT NULL, foreign_net REAL,
            inst_net REAL, PRIMARY KEY (ticker, date)
        );
        CREATE TABLE IF NOT EXISTS index_ohlcv (
            index_code TEXT NOT NULL, date TEXT NOT NULL, open REAL, high REAL,
            low REAL, close REAL, volume REAL, PRIMARY KEY (index_code, date)
        );
        CREATE TABLE IF NOT EXISTS fx_rates (
            pair TEXT NOT NULL, date TEXT NOT NULL, close REAL, PRIMARY KEY (pair, date)
        );
        CREATE TABLE IF NOT EXISTS sector_map (
            ticker TEXT PRIMARY KEY, sector TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ohlcv_date ON daily_ohlcv(date);
        CREATE INDEX IF NOT EXISTS idx_flow_date ON daily_flow(date);
        """
    )
    conn.commit()
    return conn


def load_universe(path: str | Path) -> list[str]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    values = (payload.get("symbols") or payload.get("universe") or []) if isinstance(payload, dict) else payload
    return [str(item).zfill(6) for item in values]


def load_sector_map(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw = payload.get("sector_map", {}) if isinstance(payload, dict) else {}
    return {str(symbol).zfill(6): str(sector).strip() for symbol, sector in raw.items() if str(sector).strip()}


def upsert_sector_map(conn: sqlite3.Connection, mapping: dict[str, str]) -> int:
    if not mapping:
        return 0
    conn.executemany(
        "INSERT OR REPLACE INTO sector_map (ticker, sector) VALUES (?, ?)",
        sorted(mapping.items()),
    )
    conn.commit()
    return len(mapping)


def normalize_ohlcv_df(df) -> list[dict[str, Any]]:
    if df is None or getattr(df, "empty", True):
        return []
    rows = []
    for idx, row in df.iterrows():
        rows.append(
            {
                "date": _date_from_index(idx),
                "open": _column_float(row, (KRX_OPEN, "open", "Open"), 0),
                "high": _column_float(row, (KRX_HIGH, "high", "High"), 1),
                "low": _column_float(row, (KRX_LOW, "low", "Low"), 2),
                "close": _column_float(row, (KRX_CLOSE, "close", "Close"), 3),
                "volume": _column_float(row, (KRX_VOLUME, "volume", "Volume"), 4),
            }
        )
    return [row for row in rows if min(row["open"], row["high"], row["low"], row["close"]) > 0.0]


def normalize_flow_df(df) -> list[dict[str, Any]]:
    if df is None or getattr(df, "empty", True):
        return []
    rows = []
    for idx, row in df.iterrows():
        rows.append(
            {
                "date": _date_from_index(idx),
                "foreign_net": _matching_column_float(row, (KRX_FOREIGN_TOTAL, KRX_FOREIGN, "foreign", "foreigner")),
                "inst_net": _matching_column_float(row, (KRX_INSTITUTION_TOTAL, KRX_INSTITUTION, "institution", "inst")),
            }
        )
    return rows


def backfill_stock_daily(conn: sqlite3.Connection, stock, ticker: str, start: date, end: date) -> int:
    df = stock.get_market_ohlcv_by_date(_yyyymmdd(start), _yyyymmdd(end), ticker, adjusted=True)
    rows = normalize_ohlcv_df(df)
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT OR REPLACE INTO daily_ohlcv
            (ticker, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [(ticker, row["date"], row["open"], row["high"], row["low"], row["close"], row["volume"]) for row in rows],
    )
    conn.commit()
    return len(rows)


def backfill_stock_flow(conn: sqlite3.Connection, stock, ticker: str, start: date, end: date) -> int:
    global _PYKRX_FLOW_RETURNED_EMPTY
    rows = []
    if not _PYKRX_FLOW_RETURNED_EMPTY:
        df = stock.get_market_trading_value_by_date(_yyyymmdd(start), _yyyymmdd(end), ticker, on=KRX_NET_BUY, detail=False)
        rows = normalize_flow_df(df)
        if not rows:
            _PYKRX_FLOW_RETURNED_EMPTY = True
    if not rows:
        rows = naver_investor_flow(ticker, start, end)
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT OR REPLACE INTO daily_flow
            (ticker, date, foreign_net, inst_net)
        VALUES (?, ?, ?, ?)
        """,
        [(ticker, row["date"], row["foreign_net"], row["inst_net"]) for row in rows],
    )
    conn.commit()
    return len(rows)


def backfill_index(conn: sqlite3.Connection, stock, index_code: str, pykrx_code: str, start: date, end: date) -> int:
    df = stock.get_index_ohlcv_by_date(_yyyymmdd(start), _yyyymmdd(end), pykrx_code, name_display=False)
    rows = normalize_ohlcv_df(df)
    if not rows:
        rows = naver_index_ohlcv(index_code, start, end)
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT OR REPLACE INTO index_ohlcv
            (index_code, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [(index_code, row["date"], row["open"], row["high"], row["low"], row["close"], row["volume"]) for row in rows],
    )
    conn.commit()
    return len(rows)


def naver_index_ohlcv(index_code: str, start: date, end: date) -> list[dict[str, Any]]:
    symbol = {"KOSPI": "KOSPI", "KOSDAQ": "KOSDAQ"}.get(index_code.upper())
    if not symbol:
        return []
    from pykrx.website.naver import Sise

    days = max((datetime.now().date() - start).days + 5, 30)
    xml = Sise().fetch(symbol, days)
    rows = []
    try:
        root = et.fromstring(xml)
    except et.ParseError:
        return rows
    for node in root.iter(tag="item"):
        raw = str(node.get("data", ""))
        parts = raw.split("|")
        if len(parts) < 6:
            continue
        day = _parse_yyyymmdd(parts[0])
        if day is None or day < start or day > end:
            continue
        rows.append(
            {
                "date": day.isoformat(),
                "open": _to_float(parts[1]),
                "high": _to_float(parts[2]),
                "low": _to_float(parts[3]),
                "close": _to_float(parts[4]),
                "volume": _to_float(parts[5]),
            }
        )
    return sorted(rows, key=lambda row: row["date"])


def naver_investor_flow(ticker: str, start: date, end: date, *, max_pages: int = 80) -> list[dict[str, Any]]:
    rows_by_date: dict[str, dict[str, Any]] = {}
    days_to_cover = max((datetime.now().date() - start).days, 1)
    estimated_pages = int(days_to_cover * 5 / 7 / 20) + 8
    page_limit = max(1, min(max_pages, estimated_pages))
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_naver_investor_page, ticker, page): page for page in range(1, page_limit + 1)}
        for future in as_completed(futures):
            soup = future.result()
            if soup is None:
                continue
            _collect_naver_flow_rows(soup, start, end, rows_by_date)
    return [rows_by_date[key] for key in sorted(rows_by_date)]


def _fetch_naver_investor_page(ticker: str, page: int) -> BeautifulSoup | None:
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://finance.naver.com/item/frgn.naver?code={ticker}&page={page}"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    except requests.RequestException:
        return None


def _collect_naver_flow_rows(soup: BeautifulSoup, start: date, end: date, rows_by_date: dict[str, dict[str, Any]]) -> None:
    table = _naver_daily_investor_table(soup)
    if table is None:
        return
    for tr in table.select("tr"):
        cells = [td.get_text(strip=True).replace(",", "") for td in tr.select("td")]
        if len(cells) < 7 or not cells[0] or "." not in cells[0]:
            continue
        day = _parse_dot_date(cells[0])
        if day is None:
            continue
        if start <= day <= end:
            rows_by_date[day.isoformat()] = {
                "date": day.isoformat(),
                "inst_net": _signed_number(cells[5]),
                "foreign_net": _signed_number(cells[6]),
            }


def _naver_daily_investor_table(soup: BeautifulSoup):
    for table in soup.select("table.type2"):
        for tr in table.select("tr"):
            cells = [td.get_text(strip=True) for td in tr.select("td")]
            if len(cells) >= 7 and _parse_dot_date(cells[0]) is not None:
                return table
        headers = [th.get_text(strip=True) for th in table.select("tr th")]
        if KRX_INSTITUTION in headers and KRX_FOREIGN in headers:
            return table
    return None


def verify(conn: sqlite3.Connection, universe: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for table, code_col, code_val in [
        ("index_ohlcv", "index_code", "KOSPI"),
        ("index_ohlcv", "index_code", "KOSDAQ"),
        ("fx_rates", "pair", "KRWUSD"),
    ]:
        row = conn.execute(
            f"SELECT COUNT(*) AS cnt, MIN(date) AS min_d, MAX(date) AS max_d FROM {table} WHERE {code_col} = ?",
            (code_val,),
        ).fetchone()
        summary[code_val] = {"count": int(row["cnt"] or 0), "min_date": row["min_d"], "max_date": row["max_d"]}
    missing_ohlcv = []
    missing_flow = []
    for ticker in universe:
        ohlcv_count = conn.execute("SELECT COUNT(*) AS cnt FROM daily_ohlcv WHERE ticker = ?", (ticker,)).fetchone()["cnt"]
        flow_count = conn.execute("SELECT COUNT(*) AS cnt FROM daily_flow WHERE ticker = ?", (ticker,)).fetchone()["cnt"]
        if int(ohlcv_count or 0) == 0:
            missing_ohlcv.append(ticker)
        if int(flow_count or 0) == 0:
            missing_flow.append(ticker)
    summary["symbols"] = {
        "requested": len(universe),
        "missing_ohlcv": missing_ohlcv,
        "missing_flow": missing_flow,
    }
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("\n--- LRS Database Summary ---")
    for key in ("KOSPI", "KOSDAQ", "KRWUSD"):
        item = summary.get(key, {})
        print(f"  {key}: {item.get('count', 0)} rows ({item.get('min_date')} to {item.get('max_date')})")
    symbols = summary.get("symbols", {})
    print(f"  Symbols requested: {symbols.get('requested', 0)}")
    print(f"  Missing OHLCV: {len(symbols.get('missing_ohlcv', []))} {symbols.get('missing_ohlcv', [])[:20]}")
    print(f"  Missing flow: {len(symbols.get('missing_flow', []))} {symbols.get('missing_flow', [])[:20]}")


def _column_float(row, names: tuple[str, ...], fallback_index: int) -> float:
    for name in names:
        if name in row:
            return _to_float(row[name])
    if len(row) > fallback_index:
        return _to_float(row.iloc[fallback_index])
    return 0.0


def _matching_column_float(row, names: tuple[str, ...]) -> float:
    for name in names:
        if name in row:
            return _to_float(row[name])
    for column in row.index:
        text = str(column)
        if any(name in text for name in names):
            return _to_float(row[column])
    return 0.0


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _signed_number(value: Any) -> float:
    text = str(value or "").strip().replace(",", "")
    text = text.replace("+", "")
    if not text or text in {"N/A", "-"}:
        return 0.0
    return _to_float(text)


def _parse_yyyymmdd(value: str) -> date | None:
    text = str(value)
    if len(text) != 8 or not text.isdigit():
        return None
    return date(int(text[:4]), int(text[4:6]), int(text[6:8]))


def _parse_dot_date(value: str) -> date | None:
    try:
        year, month, day = str(value).split(".")[:3]
        return date(int(year), int(month), int(day))
    except (TypeError, ValueError):
        return None


def _date_from_index(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value)
    if " " in text:
        text = text.split(" ", 1)[0]
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return date.fromisoformat(text[:10]).isoformat()


def _yyyymmdd(value: date) -> str:
    return value.strftime("%Y%m%d")


def _default_start() -> date:
    return datetime.now().date() - timedelta(days=600)


def _default_end() -> date:
    return datetime.now().date()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill historical Local Research Store data from pykrx")
    parser.add_argument("--db-path", default=os.environ.get("LRS_DB_PATH", "data/lrs.db"))
    parser.add_argument("--universe-file", default="config/universe_103.yaml")
    parser.add_argument("--config", default="config/olr/sector_map.yaml")
    parser.add_argument("--from", dest="start", default=None)
    parser.add_argument("--to", dest="end", default=None)
    parser.add_argument("--days", type=int, default=None, help="Compatibility fallback when --from is omitted")
    parser.add_argument("--sleep-sec", type=float, default=0.15)
    parser.add_argument("--skip-fx", action="store_true", help="Accepted for explicit no-FX runs; FX is never approximated")
    parser.add_argument("--skip-kosdaq", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--skip-ohlcv", action="store_true")
    parser.add_argument("--flow-only-missing", action="store_true")
    parser.add_argument("--parquet-root", default="data/krx_daily_parquet", help="Shared parquet mirror for other strategies")
    parser.add_argument("--skip-parquet-export", action="store_true")
    args = parser.parse_args(argv)

    stock = ensure_pykrx()
    if stock is None:
        return 1
    if args.start:
        start = date.fromisoformat(args.start)
    elif args.days:
        start = _default_end() - timedelta(days=int(args.days))
    else:
        start = _default_start()
    end = date.fromisoformat(args.end) if args.end else _default_end()
    universe = load_universe(args.universe_file)
    sector_map = load_sector_map(args.config)
    conn = init_db(args.db_path)
    print(f"Using LRS database: {args.db_path}")
    print(f"Backfill window: {start.isoformat()} to {end.isoformat()} | symbols={len(universe)} | fx_policy=no_fx_reweighted")
    if not args.skip_fx:
        print("FX backfill is intentionally disabled; no approximate FX data is written.")

    try:
        upsert_sector_map(conn, {symbol: sector_map[symbol] for symbol in universe if symbol in sector_map})
        if not args.skip_index:
            index_count = backfill_index(conn, stock, "KOSPI", "1001", start, end)
            print(f"KOSPI: inserted {index_count} rows")
            if not args.skip_kosdaq:
                kosdaq_count = backfill_index(conn, stock, "KOSDAQ", "2001", start, end)
                print(f"KOSDAQ: inserted {kosdaq_count} rows")
        ohlcv_ok = flow_ok = errors = 0
        for idx, ticker in enumerate(universe, start=1):
            if args.skip_ohlcv:
                daily_count = conn.execute("SELECT COUNT(*) AS cnt FROM daily_ohlcv WHERE ticker = ?", (ticker,)).fetchone()["cnt"]
                if daily_count:
                    ohlcv_ok += 1
            else:
                try:
                    daily_count = backfill_stock_daily(conn, stock, ticker, start, end)
                    if daily_count:
                        ohlcv_ok += 1
                except Exception as exc:
                    errors += 1
                    print(f"WARN: OHLCV {ticker} failed: {exc}")
                time.sleep(max(float(args.sleep_sec), 0.0))
            try:
                existing_flow = conn.execute("SELECT COUNT(*) AS cnt FROM daily_flow WHERE ticker = ?", (ticker,)).fetchone()["cnt"]
                if args.flow_only_missing and existing_flow:
                    flow_count = existing_flow
                else:
                    flow_count = backfill_stock_flow(conn, stock, ticker, start, end)
                if flow_count:
                    flow_ok += 1
            except Exception as exc:
                errors += 1
                print(f"WARN: flow {ticker} failed: {exc}")
            time.sleep(max(float(args.sleep_sec), 0.0))
            if idx % 10 == 0 or idx == len(universe):
                print(f"Progress {idx}/{len(universe)}: ohlcv_symbols={ohlcv_ok}, flow_symbols={flow_ok}, errors={errors}")
        summary = verify(conn, universe)
        print_summary(summary)
        if not args.skip_parquet_export:
            manifest = export_lrs_sqlite_to_parquet(
                args.db_path,
                args.parquet_root,
                start=start,
                end=end,
                universe=universe,
                source_label="nulrimok_lrs_backfill",
            )
            print(f"Parquet mirror: {manifest['root']} | fingerprint={manifest['source_fingerprint']}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
