"""
Market Snapshot Service — captures point-in-time market state for KRX equities.

Adapted for KIS (Korea Investment & Securities) API:
- get_last_price(ticker) -> float  (current price)
- get_daily_bars(ticker, days=N) -> pd.DataFrame  (columns: date, open, high, low, close, volume)
- get_minute_bars(ticker, minutes=M) -> pd.DataFrame  (columns: timestamp, open, high, low, close, volume)

KRX equity market differences from crypto:
- No bid/ask spread API available on-demand for all symbols (only via WebSocket for subscribed)
- No funding rate (equity market, not derivatives)
- No open interest
- No mark price concept

Snapshots are written to JSONL files in data_dir/snapshots/, one file per day.
"""

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

from loguru import logger


KST = ZoneInfo("Asia/Seoul")


@dataclass
class MarketSnapshot:
    """
    Point-in-time capture of market state for a single KRX symbol.
    Referenced by trade events and missed opportunity events.
    """
    snapshot_id: str              # deterministic: hash(symbol + timestamp)
    symbol: str                   # KRX stock code, e.g. "005930" (Samsung)
    timestamp: str                # KST time, ISO 8601
    bid: Optional[float] = None   # None — not available via KIS REST (use WS for real-time)
    ask: Optional[float] = None   # None — not available via KIS REST (use WS for real-time)
    mid: Optional[float] = None   # same as last_trade_price when bid/ask unavailable
    spread_bps: Optional[float] = None  # None — not computable without bid/ask
    last_trade_price: float = 0.0
    data_source: str = "kis_rest"       # "kis_rest" or "kis_ws" when available
    bid_ask_available: bool = False      # explicitly marks bid/ask data availability
    volume_1m: Optional[float] = None     # last 1 minute volume (from minute bars)
    volume_5m: Optional[float] = None     # last 5 minute volume (from minute bars)
    volume_24h: Optional[float] = None    # daily volume (from daily bars)
    atr_14: Optional[float] = None        # 14-period daily ATR
    funding_rate: Optional[float] = None  # always None (equity market)
    open_interest: Optional[float] = None # always None (equity market)
    mark_price: Optional[float] = None    # always None (equity market)

    def to_dict(self) -> dict:
        return asdict(self)


class MarketSnapshotService:
    """
    Captures and stores market snapshots for KRX equities via KIS API.

    Usage:
        service = MarketSnapshotService(config, kis_api)
        service.start()                       # not blocking — call run_periodic from your loop
        snapshot = service.capture_now("005930")  # on-demand for trade events
    """

    def __init__(self, config: dict, data_provider=None):
        """
        Args:
            config: from instrumentation_config.yaml
            data_provider: optional KoreaInvestAPI instance for market data.
                Supports:
                  - get_last_price(ticker) -> Optional[float]
                  - get_daily_bars(ticker, days=N) -> pd.DataFrame
                  - get_minute_bars(ticker, minutes=M) -> pd.DataFrame
                If None, all snapshots will be degraded (zeros).
        """
        self.bot_id = config.get("bot_id", "k_stock_trader")
        self.data_dir = Path(config.get("data_dir", "instrumentation/data")) / "snapshots"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.interval = config.get("market_snapshots", {}).get("interval_seconds", 300)
        self.symbols: list = config.get("market_snapshots", {}).get("symbols", [])
        self.data_provider = data_provider
        self.data_source_id = "kis_rest"
        self._cache: Dict[str, MarketSnapshot] = {}

    def _compute_snapshot_id(self, symbol: str, timestamp: str) -> str:
        """Deterministic snapshot ID from symbol + timestamp."""
        raw = f"{symbol}|{timestamp}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def capture_now(self, symbol: str) -> MarketSnapshot:
        """
        Capture a snapshot immediately. Call this when a trade or signal occurs.
        Returns the snapshot and caches it.

        Adapted for KIS API:
        - Price from get_last_price(symbol)
        - bid/ask/spread set to 0 (not available on-demand)
        - funding_rate, open_interest, mark_price always None (equity market)
        - ATR computed from daily bars if available
        - volume_24h from daily bars if available
        - volume_1m/5m from minute bars if available
        """
        try:
            last_price = 0.0
            volume_24h = None
            volume_1m = None
            volume_5m = None
            atr_14 = None

            if self.data_provider is not None:
                # --- Get last price ---
                try:
                    price = self.data_provider.get_last_price(symbol)
                    if price is not None and price > 0:
                        last_price = float(price)
                except Exception as e:
                    logger.debug(f"Snapshot: get_last_price failed for {symbol}: {e}")

                # --- Get daily volume and ATR from daily bars ---
                try:
                    daily_bars = self.data_provider.get_daily_bars(symbol, days=20)
                    if daily_bars is not None and len(daily_bars) > 0:
                        # volume_24h: most recent day's volume
                        volume_24h = float(daily_bars.iloc[-1]["volume"])

                        # ATR-14 from daily bars
                        atr_14 = self._compute_atr_from_daily(daily_bars, period=14)
                except Exception as e:
                    logger.debug(f"Snapshot: get_daily_bars failed for {symbol}: {e}")

                # --- Get recent minute volume ---
                try:
                    minute_bars = self.data_provider.get_minute_bars(symbol, minutes=5)
                    if minute_bars is not None and len(minute_bars) > 0:
                        # volume_1m: last bar's volume
                        volume_1m = float(minute_bars.iloc[-1]["volume"])
                        # volume_5m: sum of last 5 bars (or all available)
                        tail = minute_bars.tail(5)
                        volume_5m = float(tail["volume"].sum())
                except Exception as e:
                    logger.debug(f"Snapshot: get_minute_bars failed for {symbol}: {e}")

            now_kst = datetime.now(KST)
            ts_str = now_kst.isoformat()

            snapshot = MarketSnapshot(
                snapshot_id=self._compute_snapshot_id(symbol, ts_str),
                symbol=symbol,
                timestamp=ts_str,
                mid=last_price,
                last_trade_price=last_price,
                data_source="kis_rest",
                bid_ask_available=False,
                volume_1m=volume_1m,
                volume_5m=volume_5m,
                volume_24h=volume_24h,
                atr_14=atr_14,
                funding_rate=None,       # N/A: equity market
                open_interest=None,      # N/A: equity market
                mark_price=None,         # N/A: equity market
            )

            self._cache[symbol] = snapshot
            self._write_snapshot(snapshot)
            return snapshot

        except Exception as e:
            # CRITICAL: snapshot failure must never block trading
            logger.warning(f"Snapshot capture failed for {symbol}, returning degraded: {e}")
            now_kst = datetime.now(KST)
            ts_str = now_kst.isoformat()
            degraded = MarketSnapshot(
                snapshot_id=self._compute_snapshot_id(symbol, ts_str),
                symbol=symbol,
                timestamp=ts_str,
                last_trade_price=0.0,
                data_source="kis_rest",
                bid_ask_available=False,
            )
            try:
                self._write_snapshot(degraded)
            except Exception:
                pass  # even write failure must not crash
            return degraded

    def get_latest(self, symbol: str) -> Optional[MarketSnapshot]:
        """Return the most recent cached snapshot for a symbol."""
        return self._cache.get(symbol)

    def _write_snapshot(self, snapshot: MarketSnapshot):
        """Append snapshot to daily JSONL file."""
        today = datetime.now(KST).strftime("%Y-%m-%d")
        filepath = self.data_dir / f"snapshots_{today}.jsonl"
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(snapshot.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write snapshot to {filepath}: {e}")

    def _compute_atr_from_daily(self, daily_bars, period: int = 14) -> Optional[float]:
        """
        Compute ATR from daily OHLCV DataFrame.

        Args:
            daily_bars: pd.DataFrame with columns ['date', 'open', 'high', 'low', 'close', 'volume']
                        sorted ascending by date.
            period: ATR lookback period (default 14).

        Returns:
            ATR as float, or None if insufficient data.
        """
        if daily_bars is None or len(daily_bars) < period + 1:
            return None

        try:
            trs = []
            for i in range(1, len(daily_bars)):
                high = float(daily_bars.iloc[i]["high"])
                low = float(daily_bars.iloc[i]["low"])
                prev_close = float(daily_bars.iloc[i - 1]["close"])
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                trs.append(tr)

            if len(trs) < period:
                return None

            # Simple moving average ATR over the last `period` true ranges
            return round(sum(trs[-period:]) / period, 2)
        except Exception:
            return None

    def _compute_recent_volume(self, symbol: str):
        """
        Compute 1m and 5m volume from recent minute bars.
        Returns (volume_1m, volume_5m) tuple.
        """
        if self.data_provider is None:
            return None, None

        try:
            minute_bars = self.data_provider.get_minute_bars(symbol, minutes=5)
            if minute_bars is None or len(minute_bars) == 0:
                return None, None

            volume_1m = float(minute_bars.iloc[-1]["volume"])
            tail = minute_bars.tail(5)
            volume_5m = float(tail["volume"].sum()) if len(tail) >= 5 else None
            return volume_1m, volume_5m
        except Exception:
            return None, None

    def run_periodic(self):
        """
        Call this from your bot's main loop or schedule it.
        Captures snapshots for all configured symbols.

        Note: Respects KIS rate limits. With 300s interval (default),
        this is very conservative even for paper trading's 5 req/sec limit.
        """
        for symbol in self.symbols:
            try:
                self.capture_now(symbol)
            except Exception as e:
                logger.warning(f"Periodic snapshot failed for {symbol}: {e}")

    def cleanup_old_files(self, max_age_days: int = 30):
        """Delete snapshot files older than max_age_days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        for filepath in self.data_dir.glob("snapshots_*.jsonl"):
            try:
                date_str = filepath.stem.replace("snapshots_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if file_date < cutoff:
                    filepath.unlink()
                    logger.info(f"Cleaned up old snapshot file: {filepath}")
            except (ValueError, OSError) as e:
                logger.debug(f"Could not clean up {filepath}: {e}")
