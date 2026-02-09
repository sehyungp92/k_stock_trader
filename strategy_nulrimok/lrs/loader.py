"""LRS Loader — populate Local Research Store from KIS API.

Called once at strategy startup before DSE runs.  If today's data already
exists in the database the fetch is skipped (staleness check).
"""

from __future__ import annotations

import time
from datetime import date, datetime
from typing import Dict, List, Optional

from loguru import logger


def _today_iso() -> str:
    """Return today's date as ISO string (YYYY-MM-DD) in KST."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo
    return datetime.now(tz=ZoneInfo("Asia/Seoul")).date().isoformat()


def _bar_df_to_dicts(df) -> List[Dict]:
    """Convert a get_daily_bars() DataFrame to list of dicts for upsert.

    The DataFrame has columns: date (datetime tz-aware), open, high, low, close, volume.
    We convert the date to ISO string (YYYY-MM-DD) for SQLite storage.
    """
    if df is None or df.empty:
        return []
    rows = []
    for _, r in df.iterrows():
        d = r["date"]
        if hasattr(d, "date"):
            d = d.date()
        rows.append({
            "date": d.isoformat() if isinstance(d, date) else str(d)[:10],
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": float(r["volume"]),
        })
    return rows


def _investor_trend_to_dicts(raw: List[Dict]) -> List[Dict]:
    """Convert get_investor_trend() output to list of dicts for upsert.

    Input rows have 'date' as 'YYYYMMDD' string, 'foreign_net', 'inst_net'.
    We convert date to ISO string (YYYY-MM-DD).
    """
    out = []
    for r in raw:
        d = r.get("date", "")
        if len(d) == 8:
            iso = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        else:
            iso = d
        out.append({
            "date": iso,
            "foreign_net": float(r.get("foreign_net", 0)),
            "inst_net": float(r.get("inst_net", 0)),
        })
    return out


def populate_lrs(
    lrs,
    api,
    universe: List[str],
    sector_map: Dict[str, str],
    rate_budget=None,
    *,
    ohlcv_days: int = 60,
    flow_days: int = 20,
    index_days: int = 280,
) -> None:
    """Populate the LRS database from KIS API data.

    Args:
        lrs: LRSDatabase instance.
        api: KoreaInvestAPI instance.
        universe: List of ticker codes to fetch.
        sector_map: Dict mapping ticker -> sector name.
        rate_budget: Optional RateBudget (unused for now; we do our own pacing).
        ohlcv_days: Number of days of daily OHLCV to fetch per ticker.
        flow_days: Number of days of investor flow to fetch per ticker.
        index_days: Number of days of index data to fetch.
    """
    t0 = time.monotonic()
    today = _today_iso()

    # ------------------------------------------------------------------
    # 1. Sector map (instant — from config, no API call)
    # ------------------------------------------------------------------
    if sector_map:
        n = lrs.upsert_sector_map(sector_map)
        logger.info(f"LRS loader: sector_map populated ({n} tickers)")

    # ------------------------------------------------------------------
    # 2. Staleness check — skip if today's OHLCV already present
    # ------------------------------------------------------------------
    max_ohlcv = lrs.get_max_date("daily_ohlcv")
    if max_ohlcv and max_ohlcv >= today:
        elapsed = time.monotonic() - t0
        logger.info(f"LRS already fresh (max_date={max_ohlcv}), skipping fetch ({elapsed:.1f}s)")
        return

    logger.info(f"Populating LRS for {len(universe)} tickers (today={today}) ...")

    # ------------------------------------------------------------------
    # 3. KOSPI index (single call)
    # ------------------------------------------------------------------
    try:
        idx_df = api.get_daily_bars("0001", index_days)
        idx_rows = _bar_df_to_dicts(idx_df)
        if idx_rows:
            lrs.upsert_index("KOSPI", idx_rows)
            logger.info(f"LRS loader: KOSPI index loaded ({len(idx_rows)} bars)")
        else:
            logger.warning("LRS loader: KOSPI index fetch returned no data")
    except Exception as e:
        logger.error(f"LRS loader: KOSPI index fetch failed: {e}")

    # ------------------------------------------------------------------
    # 4. Per-ticker: daily OHLCV + investor flow
    # ------------------------------------------------------------------
    ohlcv_ok = 0
    flow_ok = 0
    errors = 0

    for i, ticker in enumerate(universe, 1):
        # Progress log every 50 tickers
        if i % 50 == 0 or i == len(universe):
            logger.info(f"LRS loader: {i}/{len(universe)} tickers processed "
                        f"(ohlcv={ohlcv_ok}, flow={flow_ok}, errors={errors})")

        # --- Daily OHLCV ---
        try:
            df = api.get_daily_bars(ticker, ohlcv_days)
            rows = _bar_df_to_dicts(df)
            if rows:
                lrs.upsert_daily_bars(ticker, rows)
                ohlcv_ok += 1
        except Exception as e:
            logger.debug(f"LRS loader: OHLCV error {ticker}: {e}")
            errors += 1

        # Rate-limit pacing (~2 calls/sec → 0.5s between calls)
        time.sleep(0.5)

        # --- Investor flow (foreign + inst) ---
        try:
            raw = api.get_investor_trend(ticker, flow_days)
            flow_rows = _investor_trend_to_dicts(raw)
            if flow_rows:
                lrs.upsert_daily_flow(ticker, flow_rows)
                flow_ok += 1
        except Exception as e:
            logger.debug(f"LRS loader: flow error {ticker}: {e}")
            errors += 1

        time.sleep(0.5)

    elapsed = time.monotonic() - t0
    logger.info(
        f"LRS populated: {ohlcv_ok} OHLCV + {flow_ok} flow tickers "
        f"in {elapsed:.1f}s ({errors} errors)"
    )
