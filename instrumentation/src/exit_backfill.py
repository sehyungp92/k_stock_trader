"""Post-exit price movement backfiller.

After a trade exits, tracks what happened to the price at 1h and 4h intervals.
Similar architecture to MissedOpportunityLogger backfill.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class ExitBackfiller:
    """Tracks post-exit price movement for completed trades."""

    def __init__(self, data_dir: str = "instrumentation/data"):
        self._data_dir = Path(data_dir)
        self._pending: list[dict] = []
        self._lock = threading.Lock()
        (self._data_dir / "exit_movements").mkdir(parents=True, exist_ok=True)

    def queue_exit(self, trade_id: str, symbol: str, side: str,
                   exit_price: float, exit_time: str,
                   stop_price: Optional[float] = None) -> None:
        """Queue a completed trade for post-exit tracking."""
        with self._lock:
            self._pending.append({
                "trade_id": trade_id, "symbol": symbol, "side": side,
                "exit_price": exit_price, "exit_time": exit_time,
                "stop_price": stop_price,
                "queued_at": datetime.now(timezone.utc).isoformat(),
            })

    def run_backfill(self, data_provider) -> int:
        """Process pending exits. Called from periodic_tick background thread."""
        with self._lock:
            items = list(self._pending)

        completed = 0
        for item in items:
            try:
                exit_dt = datetime.fromisoformat(item["exit_time"])
                elapsed = (datetime.now(timezone.utc) - exit_dt.astimezone(timezone.utc)).total_seconds()
                if elapsed < 3600:  # Need at least 1h elapsed
                    continue

                candles = data_provider.get_minute_bars(item["symbol"], minutes=300)
                if candles is None or (hasattr(candles, 'empty') and candles.empty):
                    continue

                records = candles.to_dict("records") if hasattr(candles, 'to_dict') else candles
                movement = self._compute_movement(
                    exit_price=item["exit_price"],
                    exit_time=item["exit_time"],
                    symbol=item["symbol"],
                    candles=records,
                )
                if movement:
                    self._write_movement(item["trade_id"], movement)
                    with self._lock:
                        if item in self._pending:
                            self._pending.remove(item)
                    completed += 1
            except Exception as e:
                logger.debug(f"Exit backfill error for {item.get('trade_id')}: {e}")

        return completed

    def _compute_movement(self, exit_price: float, exit_time: str,
                          symbol: str, candles: list) -> Optional[dict]:
        """Compute price movement at 1h and 4h after exit."""
        if not exit_price or not candles:
            return None

        exit_dt = datetime.fromisoformat(exit_time)
        target_1h = exit_dt + timedelta(hours=1)
        target_4h = exit_dt + timedelta(hours=4)

        price_1h = price_4h = None
        for c in candles:
            c_time = c.get("time") or c.get("date")
            if isinstance(c_time, str):
                c_dt = datetime.fromisoformat(c_time)
            else:
                c_dt = c_time
            c_close = c.get("close", 0)
            if not c_close:
                continue
            if c_dt >= target_1h and price_1h is None:
                price_1h = float(c_close)
            if c_dt >= target_4h and price_4h is None:
                price_4h = float(c_close)

        result: dict[str, Any] = {}
        if price_1h is not None:
            move_1h = (price_1h - exit_price) / exit_price * 100
            result["price_1h"] = price_1h
            result["move_pct_1h"] = round(move_1h, 2)
            result["exit_was_premature_1h"] = price_1h > exit_price  # LONG: price went higher
        if price_4h is not None:
            move_4h = (price_4h - exit_price) / exit_price * 100
            result["price_4h"] = price_4h
            result["move_pct_4h"] = round(move_4h, 2)
            result["exit_was_premature_4h"] = price_4h > exit_price

        return result if result else None

    def _write_movement(self, trade_id: str, movement: dict) -> None:
        """Write post-exit movement to JSONL."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        filepath = self._data_dir / "exit_movements" / f"exit_movements_{today}.jsonl"
        record = {"trade_id": trade_id, **movement, "backfill_status": "complete"}
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
