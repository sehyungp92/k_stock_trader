"""Instrumentation Facade — thin wrapper for strategy integration.

Strategies import InstrumentationKit and call high-level methods:

    kit = InstrumentationKit.create(api, strategy_type="kmp")
    kit.on_entry_fill(...)
    kit.on_exit_fill(...)
    kit.on_signal_blocked(...)
    kit.periodic_tick()
    kit.build_daily_snapshot()
    kit.shutdown()

All methods are sync, fire-and-forget, catch all exceptions internally,
and never crash the strategy.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from .src.market_snapshot import MarketSnapshotService
from .src.trade_logger import TradeLogger
from .src.missed_opportunity import MissedOpportunityLogger
from .src.process_scorer import ProcessScorer
from .src.regime_classifier import RegimeClassifier
from .src.daily_snapshot import DailySnapshotBuilder


class InstrumentationKit:
    """Fire-and-forget instrumentation facade for trading strategies."""

    def __init__(
        self,
        trade_logger: TradeLogger,
        missed_logger: MissedOpportunityLogger,
        snapshot_service: MarketSnapshotService,
        process_scorer: ProcessScorer,
        regime_classifier: RegimeClassifier,
        daily_builder: DailySnapshotBuilder,
        data_provider,
        strategy_type: str,
        data_dir: str,
    ):
        self._trade_logger = trade_logger
        self._missed_logger = missed_logger
        self._snapshot_service = snapshot_service
        self._process_scorer = process_scorer
        self._regime_classifier = regime_classifier
        self._daily_builder = daily_builder
        self._data_provider = data_provider
        self._strategy_type = strategy_type
        self._data_dir = Path(data_dir)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="instr_backfill"
        )

        # Ensure scores directory exists
        try:
            (self._data_dir / "scores").mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    @classmethod
    def create(
        cls,
        data_provider,
        strategy_type: str,
        data_dir: str = "instrumentation/data",
    ) -> "InstrumentationKit":
        """One-line factory for strategy init."""
        config = {
            "bot_id": f"k_stock_trader_{strategy_type}",
            "data_dir": data_dir,
            "data_source_id": "kis_rest",
            "strategy_type": strategy_type,
            "market_snapshots": {"interval_seconds": 300},
        }

        snapshot_service = MarketSnapshotService(config, data_provider=data_provider)
        trade_logger = TradeLogger(config, snapshot_service)
        missed_logger = MissedOpportunityLogger(config, snapshot_service)
        process_scorer = ProcessScorer()
        regime_classifier = RegimeClassifier(data_provider=data_provider)
        daily_builder = DailySnapshotBuilder(config)

        logger.info(
            f"InstrumentationKit created for {strategy_type} "
            f"(data_dir={data_dir})"
        )

        return cls(
            trade_logger=trade_logger,
            missed_logger=missed_logger,
            snapshot_service=snapshot_service,
            process_scorer=process_scorer,
            regime_classifier=regime_classifier,
            daily_builder=daily_builder,
            data_provider=data_provider,
            strategy_type=strategy_type,
            data_dir=data_dir,
        )

    def on_entry_fill(
        self,
        trade_id: str,
        symbol: str,
        entry_price: float,
        qty: int,
        signal: str,
        signal_id: str,
        signal_strength: float = 0.0,
        strategy_params: Optional[Dict[str, Any]] = None,
        signal_factors: Optional[list] = None,
    ) -> None:
        """Record a trade entry. Call after OMS fill confirmed."""
        try:
            regime = self._regime_classifier.current_regime(symbol)
            self._trade_logger.log_entry(
                trade_id=trade_id,
                pair=symbol,
                side="LONG",
                entry_price=entry_price,
                position_size=qty,
                position_size_quote=qty * entry_price,
                entry_signal=signal,
                entry_signal_id=signal_id,
                entry_signal_strength=signal_strength,
                active_filters=[],
                passed_filters=[],
                strategy_params=strategy_params or {},
                market_regime=regime,
                signal_factors=signal_factors or [],
            )
        except Exception as e:
            logger.debug(f"Instrumentation on_entry_fill error: {e}")

    def on_exit_fill(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
    ) -> None:
        """Record a trade exit and compute process score."""
        try:
            trade_event = self._trade_logger.log_exit(
                trade_id=trade_id,
                exit_price=exit_price,
                exit_reason=exit_reason,
            )
            if trade_event:
                # Build scorer-compatible dict
                score_dict = {
                    "trade_id": trade_event.trade_id,
                    "regime": trade_event.market_regime,
                    "signal_strength": trade_event.entry_signal_strength,
                    "entry_latency_ms": trade_event.entry_latency_ms,
                    "entry_slippage_bps": trade_event.entry_slippage_bps,
                    "exit_slippage_bps": trade_event.exit_slippage_bps,
                    "exit_reason": trade_event.exit_reason,
                    "pnl": trade_event.pnl,
                }
                score = self._process_scorer.score_trade(
                    score_dict, self._strategy_type
                )
                self._write_score(score)
        except Exception as e:
            logger.debug(f"Instrumentation on_exit_fill error: {e}")

    def on_signal_blocked(
        self,
        symbol: str,
        signal: str,
        signal_id: str,
        blocked_by: str,
        block_reason: str = "",
        signal_strength: float = 0.0,
        strategy_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a missed opportunity when a gate blocks a signal."""
        try:
            self._missed_logger.log_missed(
                pair=symbol,
                side="LONG",
                signal=signal,
                signal_id=signal_id,
                signal_strength=signal_strength,
                blocked_by=blocked_by,
                block_reason=block_reason,
                strategy_params=strategy_params,
                strategy_type=self._strategy_type,
            )
        except Exception as e:
            logger.debug(f"Instrumentation on_signal_blocked error: {e}")

    def periodic_tick(self) -> None:
        """Submit backfill to background thread. Call from heartbeat loop."""
        try:
            self._executor.submit(
                self._missed_logger.run_backfill, self._data_provider
            )
        except Exception as e:
            logger.debug(f"Instrumentation periodic_tick error: {e}")

    def build_daily_snapshot(self) -> None:
        """Build and save EOD snapshot. Call at shutdown or daily reset."""
        try:
            snapshot = self._daily_builder.build()
            self._daily_builder.save(snapshot)
            logger.info(
                f"Daily snapshot saved: {snapshot.total_trades} trades, "
                f"{snapshot.missed_count} missed"
            )
        except Exception as e:
            logger.debug(f"Instrumentation build_daily_snapshot error: {e}")

    def classify_regime(self, symbol: str) -> str:
        """Classify market regime for symbol. Returns cached result."""
        try:
            return self._regime_classifier.classify(symbol)
        except Exception:
            return "unknown"

    def shutdown(self) -> None:
        """Clean up executor resources."""
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass

    def _write_score(self, score) -> None:
        """Write process score to daily JSONL file."""
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            filepath = self._data_dir / "scores" / f"scores_{today}.jsonl"
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(score), default=str) + "\n")
        except Exception:
            pass
