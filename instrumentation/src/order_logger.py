"""Order-level event logging for tracking order lifecycle."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class OrderEvent:
    """A single order lifecycle event (submit, fill, partial, reject, cancel)."""

    # Identity
    order_id: str                          # broker-assigned order ID
    bot_id: str
    pair: str
    side: str = "LONG"                     # always LONG for k_stock_trader

    # Order details
    order_type: str = ""                   # MARKET | LIMIT | STOP
    status: str = ""                       # SUBMITTED | FILLED | PARTIAL_FILL | REJECTED | CANCELLED
    requested_qty: float = 0.0
    filled_qty: float = 0.0               # 0 for SUBMITTED/REJECTED
    requested_price: Optional[float] = None  # None for MARKET orders
    fill_price: Optional[float] = None     # None until filled
    slippage_bps: Optional[float] = None   # computed on fill

    # Context
    reject_reason: str = ""                # non-empty only for REJECTED
    timestamp: str = ""                    # ISO 8601
    latency_ms: Optional[float] = None     # submission-to-fill latency
    related_trade_id: str = ""             # links to TradeEvent.trade_id

    # Experiment tracking (propagated from strategy)
    experiment_id: str = ""
    experiment_variant: str = ""

    # Standard metadata
    event_metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


class OrderLogger:
    """Writes OrderEvent records to JSONL files."""

    def __init__(self, config: dict):
        self.bot_id = config["bot_id"]
        self.data_dir = Path(config["data_dir"]) / "orders"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._experiment_id = config.get("experiment_id", "")
        self._experiment_variant = config.get("experiment_variant", "")

    def log_order(
        self,
        order_id: str,
        pair: str,
        order_type: str,
        status: str,
        requested_qty: float,
        filled_qty: float = 0.0,
        requested_price: Optional[float] = None,
        fill_price: Optional[float] = None,
        reject_reason: str = "",
        latency_ms: Optional[float] = None,
        related_trade_id: str = "",
        exchange_timestamp: Optional[datetime] = None,
        bar_id: Optional[str] = None,
    ) -> OrderEvent:
        """Record an order lifecycle event."""
        now = exchange_timestamp or datetime.now(timezone.utc)
        ts_str = now.isoformat() if isinstance(now, datetime) else str(now)

        # Compute slippage if both prices available
        slippage_bps = None
        if fill_price is not None and requested_price is not None and requested_price > 0:
            slippage_bps = round(
                abs(fill_price - requested_price) / requested_price * 10_000, 2
            )

        from .event_metadata import create_event_metadata
        meta = create_event_metadata(
            bot_id=self.bot_id,
            event_type="order",
            payload_key=f"{order_id}:{status}",
            exchange_timestamp=now,
            data_source_id="kis_rest",
            bar_id=bar_id,
        )

        event = OrderEvent(
            order_id=order_id,
            bot_id=self.bot_id,
            pair=pair,
            side="LONG",
            order_type=order_type,
            status=status,
            requested_qty=requested_qty,
            filled_qty=filled_qty,
            requested_price=requested_price,
            fill_price=fill_price,
            slippage_bps=slippage_bps,
            reject_reason=reject_reason,
            timestamp=ts_str,
            latency_ms=latency_ms,
            related_trade_id=related_trade_id,
            experiment_id=self._experiment_id,
            experiment_variant=self._experiment_variant,
            event_metadata=meta.to_dict() if hasattr(meta, "to_dict") else meta,
        )

        self._write_event(event)
        return event

    def _write_event(self, event: OrderEvent) -> None:
        try:
            date_str = event.timestamp[:10] if event.timestamp else datetime.now(
                timezone.utc
            ).strftime("%Y-%m-%d")
            filepath = self.data_dir / f"orders_{date_str}.jsonl"
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), default=str) + "\n")
        except Exception:
            logger.exception("Failed to write OrderEvent %s", event.order_id)
