"""Order book context logger — market depth at decision points."""
from __future__ import annotations

import json
import hashlib
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("instrumentation.orderbook_logger")


@dataclass
class OrderBookContext:
    """Order book state at a trading decision point."""
    bot_id: str
    pair: str
    timestamp: str
    best_bid: float
    best_ask: float
    spread_bps: float = 0.0         # (ask - bid) / mid * 10000
    bid_depth_10bps: float = 0.0    # volume within 10bps of best bid (0 if unavailable)
    ask_depth_10bps: float = 0.0    # volume within 10bps of best ask (0 if unavailable)
    trade_context: Optional[str] = None  # "entry", "exit", "signal_eval"
    related_trade_id: Optional[str] = None
    event_id: str = ""

    def __post_init__(self):
        if not self.event_id:
            raw = f"{self.bot_id}|{self.timestamp}|orderbook_context|{self.pair}"
            self.event_id = hashlib.sha256(raw.encode()).hexdigest()[:16]
        if self.best_bid > 0 and self.best_ask > 0 and self.spread_bps == 0.0:
            mid = (self.best_bid + self.best_ask) / 2
            self.spread_bps = round((self.best_ask - self.best_bid) / mid * 10000, 2) if mid > 0 else 0.0

    @property
    def imbalance_ratio(self) -> float:
        """Bid/ask depth imbalance. >1 = bid-heavy, <1 = ask-heavy. 0 if no depth data."""
        if self.ask_depth_10bps <= 0:
            return 0.0
        return round(self.bid_depth_10bps / self.ask_depth_10bps, 4)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["imbalance_ratio"] = self.imbalance_ratio
        return d


class OrderBookLogger:
    """Writes order book context to daily JSONL files."""

    def __init__(self, data_dir: str | Path, bot_id: str) -> None:
        self._data_dir = Path(data_dir) / "orderbook"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._bot_id = bot_id

    def log_context(
        self,
        pair: str,
        best_bid: float,
        best_ask: float,
        trade_context: Optional[str] = None,
        related_trade_id: Optional[str] = None,
        bid_depth_10bps: float = 0.0,
        ask_depth_10bps: float = 0.0,
        exchange_timestamp: Optional[datetime] = None,
    ) -> OrderBookContext:
        ts = exchange_timestamp or datetime.now(timezone.utc)
        ts_str = ts.isoformat() if isinstance(ts, datetime) else str(ts)

        ctx = OrderBookContext(
            bot_id=self._bot_id,
            pair=pair,
            timestamp=ts_str,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_depth_10bps=bid_depth_10bps,
            ask_depth_10bps=ask_depth_10bps,
            trade_context=trade_context,
            related_trade_id=related_trade_id,
        )

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filepath = self._data_dir / f"orderbook_{today}.jsonl"
        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(ctx.to_dict(), default=str) + "\n")
        except Exception as e:
            logger.debug("Failed to write orderbook context: %s", e)

        return ctx
