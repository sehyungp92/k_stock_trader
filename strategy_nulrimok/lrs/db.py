"""Local Research Store Database Interface."""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DailyBar:
    ticker: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class DailyFlow:
    ticker: str
    date: date
    foreign_net: float
    inst_net: float

    @property
    def smart_money(self) -> float:
        return self.foreign_net + self.inst_net


class LRSDatabase:
    """Local Research Store database."""

    SCHEMA = """
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
    CREATE TABLE IF NOT EXISTS sector_map (ticker TEXT PRIMARY KEY, sector TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS derived_metrics (
        ticker TEXT NOT NULL, date TEXT NOT NULL, metric_name TEXT NOT NULL,
        value REAL, metadata TEXT, PRIMARY KEY (ticker, date, metric_name)
    );
    CREATE TABLE IF NOT EXISTS watchlist_artifact (date TEXT PRIMARY KEY, artifact_json TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS idx_ohlcv_date ON daily_ohlcv(date);
    CREATE INDEX IF NOT EXISTS idx_flow_date ON daily_flow(date);
    """

    def __init__(self, db_path: str = "lrs.db"):
        self.db_path = Path(db_path)
        with self._conn() as conn:
            conn.executescript(self.SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get_daily_bars(self, ticker: str, start_date: date, end_date: date) -> List[DailyBar]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM daily_ohlcv WHERE ticker = ? AND date >= ? AND date <= ? ORDER BY date",
                (ticker, start_date.isoformat(), end_date.isoformat())
            ).fetchall()
            return [DailyBar(r['ticker'], date.fromisoformat(r['date']), r['open'], r['high'],
                             r['low'], r['close'], r['volume']) for r in rows]

    def get_closes(self, ticker: str, days: int) -> List[float]:
        end = date.today()
        bars = self.get_daily_bars(ticker, end - timedelta(days=days * 2), end)
        return [b.close for b in bars[-days:]]

    def get_daily_flow(self, ticker: str, days: int) -> List[DailyFlow]:
        end = date.today()
        start = end - timedelta(days=days * 2)
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM daily_flow WHERE ticker = ? AND date >= ? AND date <= ? ORDER BY date",
                (ticker, start.isoformat(), end.isoformat())
            ).fetchall()
            return [DailyFlow(r['ticker'], date.fromisoformat(r['date']), r['foreign_net'], r['inst_net'])
                    for r in rows][-days:]

    def get_smart_money_series(self, ticker: str, days: int) -> List[float]:
        return [f.smart_money for f in self.get_daily_flow(ticker, days)]

    def get_index_series(self, index_code: str, days: int) -> List[Dict]:
        end = date.today()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM index_ohlcv WHERE index_code = ? AND date >= ? ORDER BY date",
                (index_code, (end - timedelta(days=days * 2)).isoformat())
            ).fetchall()
            return [dict(r) for r in rows[-days:]]

    def get_fx_series(self, pair: str, days: int) -> List[float]:
        end = date.today()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT close FROM fx_rates WHERE pair = ? AND date >= ? ORDER BY date",
                (pair, (end - timedelta(days=days * 2)).isoformat())
            ).fetchall()
            return [r['close'] for r in rows[-days:]]

    def get_sector(self, ticker: str) -> Optional[str]:
        with self._conn() as conn:
            row = conn.execute("SELECT sector FROM sector_map WHERE ticker = ?", (ticker,)).fetchone()
            return row['sector'] if row else None

    def get_sector_members(self, sector: str) -> List[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT ticker FROM sector_map WHERE sector = ?", (sector,)).fetchall()
            return [r['ticker'] for r in rows]

    def get_all_tickers(self) -> List[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT ticker FROM sector_map").fetchall()
            return [r['ticker'] for r in rows]

    def get_recent_bars(self, ticker: str, days: int) -> List[DailyBar]:
        end = date.today()
        return self.get_daily_bars(ticker, end - timedelta(days=days * 2), end)[-days:]

    def save_artifact(self, artifact_date: date, artifact: dict) -> None:
        with self._conn() as conn:
            conn.execute("INSERT OR REPLACE INTO watchlist_artifact VALUES (?, ?)",
                         (artifact_date.isoformat(), json.dumps(artifact)))

    def load_artifact(self, artifact_date: date) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT artifact_json FROM watchlist_artifact WHERE date = ?",
                               (artifact_date.isoformat(),)).fetchone()
            return json.loads(row['artifact_json']) if row else None

    # =========================================================================
    # WRITE / UPSERT METHODS (used by LRS loader)
    # =========================================================================

    def get_max_date(self, table: str, key_col: str = "ticker", key_val: str = "") -> Optional[str]:
        """Get the maximum date string in a table, optionally filtered by key."""
        allowed_tables = {"daily_ohlcv", "daily_flow", "index_ohlcv", "fx_rates"}
        if table not in allowed_tables:
            return None
        with self._conn() as conn:
            if key_val:
                row = conn.execute(
                    f"SELECT MAX(date) AS md FROM {table} WHERE {key_col} = ?", (key_val,)
                ).fetchone()
            else:
                row = conn.execute(f"SELECT MAX(date) AS md FROM {table}").fetchone()
            return row["md"] if row and row["md"] else None

    def upsert_daily_bars(self, ticker: str, bars: List[Dict]) -> int:
        """Bulk INSERT OR REPLACE into daily_ohlcv. Returns rows written."""
        if not bars:
            return 0
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO daily_ohlcv (ticker, date, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (ticker, b["date"], b["open"], b["high"], b["low"], b["close"], b["volume"])
                    for b in bars
                ],
            )
            return len(bars)

    def upsert_daily_flow(self, ticker: str, flows: List[Dict]) -> int:
        """Bulk INSERT OR REPLACE into daily_flow. Returns rows written."""
        if not flows:
            return 0
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO daily_flow (ticker, date, foreign_net, inst_net) "
                "VALUES (?, ?, ?, ?)",
                [
                    (ticker, f["date"], f["foreign_net"], f["inst_net"])
                    for f in flows
                ],
            )
            return len(flows)

    def upsert_index(self, index_code: str, bars: List[Dict]) -> int:
        """Bulk INSERT OR REPLACE into index_ohlcv. Returns rows written."""
        if not bars:
            return 0
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO index_ohlcv (index_code, date, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (index_code, b["date"], b["open"], b["high"], b["low"], b["close"], b["volume"])
                    for b in bars
                ],
            )
            return len(bars)

    def upsert_fx(self, pair: str, rows: List[Dict]) -> int:
        """Bulk INSERT OR REPLACE into fx_rates. Returns rows written."""
        if not rows:
            return 0
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO fx_rates (pair, date, close) VALUES (?, ?, ?)",
                [(pair, r["date"], r["close"]) for r in rows],
            )
            return len(rows)

    def upsert_sector_map(self, mapping: Dict[str, str]) -> int:
        """Bulk INSERT OR REPLACE into sector_map. Returns rows written."""
        if not mapping:
            return 0
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO sector_map (ticker, sector) VALUES (?, ?)",
                list(mapping.items()),
            )
            return len(mapping)
