"""Regression tests for assistant-facing instrumentation payloads."""

from __future__ import annotations

from datetime import datetime, timezone

from instrumentation.src.market_snapshot import MarketSnapshot
from instrumentation.src.missed_opportunity import MissedOpportunityLogger
from instrumentation.src.trade_logger import TradeLogger

from schemas.events import MissedOpportunityEvent as AssistantMissedOpportunityEvent
from schemas.events import TradeEvent as AssistantTradeEvent


class _SnapshotService:
    def capture_now(self, symbol: str) -> MarketSnapshot:
        timestamp = datetime.now(timezone.utc).isoformat()
        return MarketSnapshot(
            snapshot_id="snap-1",
            symbol=symbol,
            timestamp=timestamp,
            mid=70000.0,
            last_trade_price=70000.0,
            volume_1m=1000.0,
            volume_5m=5000.0,
            volume_24h=100000.0,
            atr_14=1200.0,
        )


def test_exit_stage_trade_validates_against_assistant_schema(tmp_path):
    config = {
        "bot_id": "k_stock_trader",
        "data_dir": str(tmp_path),
        "data_source_id": "kis_rest",
    }
    logger = TradeLogger(config, _SnapshotService())
    logger.log_entry(
        trade_id="trade-1",
        pair="005930",
        side="LONG",
        entry_price=70000.0,
        position_size=10,
        position_size_quote=700000.0,
        entry_signal="gap_breakout",
        entry_signal_id="sig-1",
        entry_signal_strength=0.8,
        active_filters=[],
        passed_filters=[],
        strategy_params={},
        bot_id="k_stock_trader",
        strategy_id="ALPHA",
    )

    trade = logger.log_exit(
        trade_id="trade-1",
        exit_price=71000.0,
        exit_reason="SIGNAL",
    )

    assert trade is not None
    AssistantTradeEvent.model_validate(trade.to_dict())


def test_missed_opportunity_validates_against_assistant_schema(tmp_path):
    config = {
        "bot_id": "k_stock_trader",
        "data_dir": str(tmp_path),
        "data_source_id": "kis_rest",
    }
    logger = MissedOpportunityLogger(config, _SnapshotService())

    event = logger.log_missed(
        pair="005930",
        side="LONG",
        signal="gap_breakout",
        signal_id="sig-1",
        signal_strength=0.7,
        blocked_by="risk_cap",
        strategy_type="alpha",
    )

    AssistantMissedOpportunityEvent.model_validate(event.to_dict())
