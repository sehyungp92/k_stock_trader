"""
Trade event logger for the K Stock Trader instrumentation layer.

Records structured JSONL trade events (entry and exit) for all four strategies
(KMP, KPR, PCIM, NULRIMOK) routed through the centralized OMS.

All instrumentation is fault-tolerant: failures are caught, logged to the
errors directory, and never propagate back to the trading hot path.
"""

from __future__ import annotations

import json
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .event_metadata import create_event_metadata
from .market_snapshot import MarketSnapshotService


@dataclass
class TradeEvent:
    """Structured representation of a single trade lifecycle event (entry or exit)."""

    # --- Identity ---
    trade_id: str
    event_metadata: Dict[str, Any]

    # --- Snapshots ---
    entry_snapshot: Dict[str, Any]
    exit_snapshot: Optional[Dict[str, Any]] = None

    # --- Core trade fields ---
    pair: str = ""
    side: str = "LONG"
    entry_time: Optional[str] = None
    exit_time: Optional[str] = None
    entry_price: float = 0.0
    exit_price: Optional[float] = None

    # --- Position & PnL (KRW) ---
    position_size: float = 0.0
    position_size_quote: float = 0.0
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    fees_paid: Optional[float] = None

    # --- Signal ---
    entry_signal: str = ""
    entry_signal_id: str = ""
    entry_signal_strength: float = 0.0
    exit_reason: str = ""
    market_regime: str = ""

    # --- Filter tracking ---
    active_filters: List[str] = field(default_factory=list)
    passed_filters: List[str] = field(default_factory=list)
    blocked_by: Optional[str] = None

    # --- Market context at entry ---
    atr_at_entry: Optional[float] = None
    spread_at_entry_bps: Optional[float] = None
    volume_24h_at_entry: Optional[float] = None
    funding_rate_at_entry: Optional[float] = None
    open_interest_at_entry: Optional[float] = None

    # --- Strategy params frozen at entry ---
    strategy_params_at_entry: Optional[Dict[str, Any]] = None

    # --- Slippage ---
    expected_entry_price: Optional[float] = None
    entry_slippage_bps: Optional[float] = None
    expected_exit_price: Optional[float] = None
    exit_slippage_bps: Optional[float] = None

    # --- Latency ---
    entry_latency_ms: Optional[int] = None
    exit_latency_ms: Optional[int] = None

    # --- Lifecycle stage ---
    stage: str = "entry"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TradeLogger:
    """Append-only JSONL trade logger.

    Writes one line per event to ``<data_dir>/trades/trades_YYYY-MM-DD.jsonl``.
    Errors are written to ``<data_dir>/errors/instrumentation_errors_YYYY-MM-DD.jsonl``.
    """

    def __init__(self, config: Dict[str, Any], snapshot_service) -> None:
        self.bot_id: str = config.get("bot_id", "k_stock_trader")
        self.data_dir: Path = Path(config.get("data_dir", "instrumentation/data"))
        self.data_source_id: str = config.get("data_source_id", "kis_rest")
        self.snapshot_service = snapshot_service
        self._open_trades: Dict[str, TradeEvent] = {}

        try:
            (self.data_dir / "trades").mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def log_entry(
        self,
        trade_id: str,
        pair: str,
        side: str,
        entry_price: float,
        position_size: float,
        position_size_quote: float,
        entry_signal: str,
        entry_signal_id: str,
        entry_signal_strength: float,
        active_filters: List[str],
        passed_filters: List[str],
        strategy_params: dict,
        exchange_timestamp: Optional[datetime] = None,
        expected_entry_price: Optional[float] = None,
        entry_latency_ms: Optional[int] = None,
        market_regime: str = "",
        bar_id: Optional[str] = None,
    ) -> TradeEvent:
        """Record a trade entry event. Returns a TradeEvent (possibly degraded on error)."""
        try:
            now = datetime.now(timezone.utc)
            exch_ts = exchange_timestamp or now

            # Capture market snapshot at entry
            entry_snapshot_dict = {}
            atr_14 = None
            spread_bps = None
            volume_24h = None
            try:
                snap = self.snapshot_service.capture_now(pair)
                entry_snapshot_dict = snap.to_dict()
                atr_14 = snap.atr_14
                spread_bps = snap.spread_bps
                volume_24h = snap.volume_24h
            except Exception:
                pass

            # Compute slippage
            entry_slippage_bps = None
            if expected_entry_price and expected_entry_price > 0:
                entry_slippage_bps = round(
                    abs(entry_price - expected_entry_price) / expected_entry_price * 10000, 2
                )

            # Build metadata
            try:
                metadata = create_event_metadata(
                    bot_id=self.bot_id,
                    event_type="trade",
                    payload_key=f"{trade_id}_entry",
                    exchange_timestamp=exch_ts,
                    data_source_id=self.data_source_id,
                    bar_id=bar_id,
                ).to_dict()
            except Exception:
                metadata = {
                    "bot_id": self.bot_id,
                    "event_type": "trade",
                    "timestamp": now.isoformat(),
                }

            trade = TradeEvent(
                trade_id=trade_id,
                event_metadata=metadata,
                entry_snapshot=entry_snapshot_dict,
                pair=pair,
                side=side,
                entry_time=exch_ts.isoformat(),
                entry_price=entry_price,
                position_size=position_size,
                position_size_quote=position_size_quote,
                entry_signal=entry_signal,
                entry_signal_id=entry_signal_id,
                entry_signal_strength=entry_signal_strength,
                market_regime=market_regime,
                active_filters=active_filters,
                passed_filters=passed_filters,
                atr_at_entry=atr_14,
                spread_at_entry_bps=spread_bps,
                volume_24h_at_entry=volume_24h,
                strategy_params_at_entry=strategy_params,
                expected_entry_price=expected_entry_price,
                entry_slippage_bps=entry_slippage_bps,
                entry_latency_ms=entry_latency_ms,
                stage="entry",
            )

            self._open_trades[trade_id] = trade
            self._write_event(trade)
            return trade

        except Exception as e:
            self._write_error("log_entry", trade_id, e)
            return TradeEvent(trade_id=trade_id, event_metadata={}, entry_snapshot={})

    def log_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
        fees_paid: float = 0.0,
        exchange_timestamp: Optional[datetime] = None,
        expected_exit_price: Optional[float] = None,
        exit_latency_ms: Optional[int] = None,
    ) -> Optional[TradeEvent]:
        """Record a trade exit event. Returns updated TradeEvent or None on error."""
        try:
            trade = self._open_trades.pop(trade_id, None)
            if trade is None:
                self._write_error(
                    "log_exit", trade_id,
                    Exception(f"No open trade found for trade_id={trade_id}"),
                )
                return None

            now = datetime.now(timezone.utc)
            exch_ts = exchange_timestamp or now

            # Capture exit snapshot
            exit_snapshot_dict = {}
            try:
                snap = self.snapshot_service.capture_now(trade.pair)
                exit_snapshot_dict = snap.to_dict()
            except Exception:
                pass

            # Compute PnL (always LONG for KRX equity)
            pnl = (exit_price - trade.entry_price) * trade.position_size - fees_paid
            pnl_pct = (
                (exit_price - trade.entry_price) / trade.entry_price * 100
                if trade.entry_price > 0 else 0.0
            )

            # Compute exit slippage
            exit_slippage_bps = None
            if expected_exit_price and expected_exit_price > 0:
                exit_slippage_bps = round(
                    abs(exit_price - expected_exit_price) / expected_exit_price * 10000, 2
                )

            # Update metadata
            try:
                trade.event_metadata = create_event_metadata(
                    bot_id=self.bot_id,
                    event_type="trade",
                    payload_key=f"{trade_id}_exit",
                    exchange_timestamp=exch_ts,
                    data_source_id=self.data_source_id,
                ).to_dict()
            except Exception:
                pass

            trade.exit_snapshot = exit_snapshot_dict
            trade.exit_time = exch_ts.isoformat()
            trade.exit_price = exit_price
            trade.exit_reason = exit_reason
            trade.pnl = round(pnl, 4)
            trade.pnl_pct = round(pnl_pct, 4)
            trade.fees_paid = fees_paid
            trade.expected_exit_price = expected_exit_price
            trade.exit_slippage_bps = exit_slippage_bps
            trade.exit_latency_ms = exit_latency_ms
            trade.stage = "exit"

            self._write_event(trade)
            return trade

        except Exception as e:
            self._write_error("log_exit", trade_id, e)
            return None

    def get_open_trades(self) -> Dict[str, TradeEvent]:
        return dict(self._open_trades)

    def _write_event(self, trade: TradeEvent) -> None:
        """Append trade event to daily JSONL file."""
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            filepath = self.data_dir / "trades" / f"trades_{today}.jsonl"
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(trade.to_dict(), default=str) + "\n")
        except Exception:
            pass

    def _write_error(self, method: str, trade_id: str, error: Exception) -> None:
        """Log instrumentation errors without crashing."""
        try:
            error_dir = self.data_dir / "errors"
            error_dir.mkdir(parents=True, exist_ok=True)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            filepath = error_dir / f"instrumentation_errors_{today}.jsonl"
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "component": "trade_logger",
                "method": method,
                "trade_id": trade_id,
                "error": str(error),
                "error_type": type(error).__name__,
            }
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass
