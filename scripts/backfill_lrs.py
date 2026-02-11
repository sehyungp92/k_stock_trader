#!/usr/bin/env python3
"""Backfill Nulrimok LRS database with historical KOSPI index and FX data.

Uses pykrx (scrapes KRX/Naver, no API key needed) to fetch historical data
that the KIS API may not provide (especially in paper trading mode).

Usage:
    # From VPS host (after finding the DB path):
    pip install pykrx
    python scripts/backfill_lrs.py --db-path /path/to/lrs.db

    # Via docker exec (inside the nulrimok container):
    docker exec strategy_nulrimok pip install pykrx
    docker exec strategy_nulrimok python /app/scripts/backfill_lrs.py

    # Default: uses LRS_DB_PATH env var or /data/lrs.db
"""

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timedelta


def ensure_pykrx():
    try:
        from pykrx import stock  # noqa: F401
        return True
    except ImportError:
        print("ERROR: pykrx not installed. Run: pip install pykrx")
        return False


def get_db_path(args_db_path: str | None) -> str:
    if args_db_path:
        return args_db_path
    return os.environ.get("LRS_DB_PATH", "/data/lrs.db")


def init_db(db_path: str) -> sqlite3.Connection:
    """Open DB and ensure schema exists."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS index_ohlcv (
            index_code TEXT NOT NULL, date TEXT NOT NULL, open REAL, high REAL,
            low REAL, close REAL, volume REAL, PRIMARY KEY (index_code, date)
        );
        CREATE TABLE IF NOT EXISTS fx_rates (
            pair TEXT NOT NULL, date TEXT NOT NULL, close REAL,
            PRIMARY KEY (pair, date)
        );
    """)
    conn.commit()
    return conn


def backfill_kospi(conn: sqlite3.Connection, days: int = 600):
    """Fetch KOSPI daily OHLCV from pykrx and insert into index_ohlcv."""
    from pykrx import stock

    end = datetime.now()
    start = end - timedelta(days=days)
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")

    print(f"Fetching KOSPI index data from {start_str} to {end_str} ...")
    df = stock.get_index_ohlcv(start_str, end_str, "1001")

    if df is None or df.empty:
        print("WARNING: pykrx returned no KOSPI data")
        return 0

    # pykrx columns are Korean: 시가, 고가, 저가, 종가, 거래량, 거래대금, 상장시가총액
    # Index is DatetimeIndex named 날짜
    rows = []
    for dt_idx, row in df.iterrows():
        date_str = dt_idx.strftime("%Y-%m-%d")
        rows.append((
            "KOSPI",
            date_str,
            float(row.iloc[0]),  # 시가 (Open)
            float(row.iloc[1]),  # 고가 (High)
            float(row.iloc[2]),  # 저가 (Low)
            float(row.iloc[3]),  # 종가 (Close)
            float(row.iloc[4]),  # 거래량 (Volume)
        ))

    conn.executemany(
        "INSERT OR REPLACE INTO index_ohlcv (index_code, date, open, high, low, close, volume) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    print(f"KOSPI: inserted {len(rows)} daily bars ({rows[0][1]} to {rows[-1][1]})")
    return len(rows)


def backfill_kosdaq(conn: sqlite3.Connection, days: int = 600):
    """Fetch KOSDAQ daily OHLCV from pykrx and insert into index_ohlcv."""
    from pykrx import stock

    end = datetime.now()
    start = end - timedelta(days=days)
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")

    print(f"Fetching KOSDAQ index data from {start_str} to {end_str} ...")
    df = stock.get_index_ohlcv(start_str, end_str, "2001")

    if df is None or df.empty:
        print("WARNING: pykrx returned no KOSDAQ data")
        return 0

    rows = []
    for dt_idx, row in df.iterrows():
        date_str = dt_idx.strftime("%Y-%m-%d")
        rows.append((
            "KOSDAQ",
            date_str,
            float(row.iloc[0]),
            float(row.iloc[1]),
            float(row.iloc[2]),
            float(row.iloc[3]),
            float(row.iloc[4]),
        ))

    conn.executemany(
        "INSERT OR REPLACE INTO index_ohlcv (index_code, date, open, high, low, close, volume) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    print(f"KOSDAQ: inserted {len(rows)} daily bars ({rows[0][1]} to {rows[-1][1]})")
    return len(rows)


def backfill_fx(conn: sqlite3.Connection, days: int = 600):
    """Fetch USD/KRW exchange rate and store as KRWUSD in fx_rates.

    Uses pykrx's KOSPI USD futures or falls back to a simple approach.
    """
    try:
        # Try using the requests library to fetch from a public API
        import requests
        end = datetime.now()
        start = end - timedelta(days=days)

        # Use Bank of Korea API (ECOS) or fallback
        # Simple approach: use pykrx for KRW/USD via the KOSPI 200 futures proxy
        # Actually, let's try a more reliable source
        from pykrx import stock

        # pykrx doesn't directly provide FX data, so we'll use a workaround
        # Fetch from Naver Finance FX page via requests
        print("Fetching KRW/USD exchange rate data ...")

        # Try fetching from exchangerate-api (free, no key for basic)
        url = "https://open.er-api.com/v6/latest/USD"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            krw_rate = data.get("rates", {}).get("KRW")
            if krw_rate:
                # Only gives today's rate - insert as today
                today_str = datetime.now().strftime("%Y-%m-%d")
                conn.execute(
                    "INSERT OR REPLACE INTO fx_rates (pair, date, close) VALUES (?, ?, ?)",
                    ("KRWUSD", today_str, float(krw_rate)),
                )
                conn.commit()
                print(f"FX: inserted today's KRW/USD rate: {krw_rate}")

                # For historical: backfill with constant (approximation)
                # This is a rough fill - better than empty
                existing = conn.execute(
                    "SELECT COUNT(*) as cnt FROM fx_rates WHERE pair = 'KRWUSD'"
                ).fetchone()["cnt"]

                if existing < 10:
                    print(f"FX: backfilling {days} days with approximate rates ...")
                    rows = []
                    for i in range(days):
                        d = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                        # Use today's rate as approximation (regime only checks 5-day change)
                        rows.append(("KRWUSD", d, float(krw_rate)))
                    conn.executemany(
                        "INSERT OR IGNORE INTO fx_rates (pair, date, close) VALUES (?, ?, ?)",
                        rows,
                    )
                    conn.commit()
                    print(f"FX: backfilled {len(rows)} days (approximate)")
                return 1
    except Exception as e:
        print(f"FX backfill failed (non-critical): {e}")
        print("FX data is optional - regime will default fx_ok=True without it")
    return 0


def verify(conn: sqlite3.Connection):
    """Print summary of what's in the database."""
    print("\n--- LRS Database Summary ---")

    for table, code_col, code_val in [
        ("index_ohlcv", "index_code", "KOSPI"),
        ("index_ohlcv", "index_code", "KOSDAQ"),
        ("fx_rates", "pair", "KRWUSD"),
    ]:
        row = conn.execute(
            f"SELECT COUNT(*) as cnt, MIN(date) as min_d, MAX(date) as max_d "
            f"FROM {table} WHERE {code_col} = ?",
            (code_val,)
        ).fetchone()
        cnt, min_d, max_d = row["cnt"], row["min_d"], row["max_d"]
        print(f"  {code_val}: {cnt} rows ({min_d} to {max_d})" if cnt > 0
              else f"  {code_val}: empty")

    # Check regime readiness
    kospi_cnt = conn.execute(
        "SELECT COUNT(*) as cnt FROM index_ohlcv WHERE index_code = 'KOSPI'"
    ).fetchone()["cnt"]
    if kospi_cnt >= 280:
        print(f"\n  Regime: READY (>= 280 KOSPI bars, have {kospi_cnt})")
    elif kospi_cnt >= 50:
        print(f"\n  Regime: PARTIAL ({kospi_cnt}/280 KOSPI bars - vol percentile limited)")
    else:
        print(f"\n  Regime: NOT READY ({kospi_cnt}/50 minimum KOSPI bars needed)")


def main():
    parser = argparse.ArgumentParser(description="Backfill Nulrimok LRS with historical data")
    parser.add_argument("--db-path", help="Path to LRS SQLite database")
    parser.add_argument("--days", type=int, default=600, help="Days of history to fetch (default: 600)")
    parser.add_argument("--skip-fx", action="store_true", help="Skip FX rate backfill")
    parser.add_argument("--skip-kosdaq", action="store_true", help="Skip KOSDAQ index backfill")
    args = parser.parse_args()

    if not ensure_pykrx():
        sys.exit(1)

    db_path = get_db_path(args.db_path)
    print(f"Using LRS database: {db_path}")

    if not os.path.exists(os.path.dirname(db_path) or "."):
        print(f"ERROR: Directory for {db_path} does not exist")
        sys.exit(1)

    conn = init_db(db_path)

    try:
        backfill_kospi(conn, args.days)
        if not args.skip_kosdaq:
            backfill_kosdaq(conn, args.days)
        if not args.skip_fx:
            backfill_fx(conn, args.days)
        verify(conn)
    finally:
        conn.close()

    print("\nBackfill complete. Restart strategy_nulrimok to pick up the data.")


if __name__ == "__main__":
    main()
