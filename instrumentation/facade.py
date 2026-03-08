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
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from loguru import logger

from .src.market_snapshot import MarketSnapshotService
from .src.trade_logger import TradeLogger
from .src.missed_opportunity import MissedOpportunityLogger
from .src.process_scorer import ProcessScorer
from .src.regime_classifier import RegimeClassifier
from .src.daily_snapshot import DailySnapshotBuilder
from .src.exit_backfill import ExitBackfiller
from .src.heartbeat import HeartbeatEmitter
from .src.order_logger import OrderLogger
from .src.sidecar import Sidecar
from .src.indicator_logger import IndicatorLogger
from .src.filter_logger import FilterLogger
from .src.orderbook_logger import OrderBookLogger
from .src.config_watcher import ConfigWatcher
from .src.experiment_registry import ExperimentRegistry


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
        exit_backfiller: Optional[ExitBackfiller] = None,
        heartbeat: Optional[HeartbeatEmitter] = None,
        order_logger: Optional[OrderLogger] = None,
        sidecar=None,
        indicator_logger: Optional[IndicatorLogger] = None,
        filter_logger: Optional[FilterLogger] = None,
        orderbook_logger: Optional[OrderBookLogger] = None,
        config_watcher: Optional[ConfigWatcher] = None,
        experiment_registry: Optional[ExperimentRegistry] = None,
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
        self._bot_id = f"k_stock_trader_{strategy_type}"
        self._exit_backfiller = exit_backfiller or ExitBackfiller(data_dir=data_dir)
        self._heartbeat = heartbeat or HeartbeatEmitter(
            bot_id=self._bot_id,
            strategy_type=strategy_type,
            data_dir=data_dir,
        )
        self._order_logger = order_logger or OrderLogger({
            "bot_id": self._bot_id,
            "data_dir": data_dir,
        })
        self._sidecar = sidecar
        self._indicator_logger = indicator_logger or IndicatorLogger(
            data_dir=data_dir, bot_id=self._bot_id,
        )
        self._filter_logger = filter_logger or FilterLogger(
            data_dir=data_dir, bot_id=self._bot_id,
        )
        self._orderbook_logger = orderbook_logger or OrderBookLogger(
            data_dir=data_dir, bot_id=self._bot_id,
        )
        self._config_watcher = config_watcher
        self._experiment_registry = experiment_registry
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
        bot_id = f"k_stock_trader_{strategy_type}"
        config = {
            "bot_id": bot_id,
            "data_dir": data_dir,
            "data_source_id": "kis_rest",
            "strategy_type": strategy_type,
            "market_snapshots": {"interval_seconds": 300},
        }

        # Load sidecar config from YAML, allow env var override for relay_url
        config_path = Path(__file__).parent / "config" / "instrumentation_config.yaml"
        try:
            with open(config_path) as f:
                yaml_config = yaml.safe_load(f) or {}
            config["sidecar"] = yaml_config.get("sidecar", {})
        except (OSError, yaml.YAMLError) as e:
            logger.warning("Failed to load instrumentation config: %s", e)
            config["sidecar"] = {}

        relay_url = os.environ.get("SIDECAR_RELAY_URL", "") or config["sidecar"].get("relay_url", "")
        config["sidecar"]["relay_url"] = relay_url

        snapshot_service = MarketSnapshotService(config, data_provider=data_provider)
        trade_logger = TradeLogger(config, snapshot_service)
        missed_logger = MissedOpportunityLogger(config, snapshot_service)
        process_scorer = ProcessScorer()
        regime_classifier = RegimeClassifier(data_provider=data_provider)

        # Initialize ExperimentRegistry (before DailySnapshotBuilder which depends on it)
        experiment_registry = None
        try:
            exp_path = Path("config/experiments.yaml")
            if exp_path.exists():
                experiment_registry = ExperimentRegistry(exp_path)
        except Exception as e:
            logger.debug("ExperimentRegistry init failed (non-fatal): %s", e)

        daily_builder = DailySnapshotBuilder(config, experiment_registry=experiment_registry)

        # Initialize sidecar when relay_url is configured
        sidecar = None
        if relay_url:
            try:
                sidecar = Sidecar(config)
                sidecar.start()
                logger.info("Sidecar started — forwarding to %s", relay_url)
            except Exception as e:
                logger.warning("Failed to start sidecar: %s", e)
                sidecar = None
        else:
            logger.debug("Sidecar disabled — no SIDECAR_RELAY_URL or relay_url configured")

        # Initialize ConfigWatcher for parameter change detection
        config_watcher = None
        try:
            config_watcher = ConfigWatcher(
                bot_id=bot_id,
                config_modules=[
                    "strategy_kmp.config.constants",
                    "strategy_kpr.config.constants",
                    "strategy_pcim.config.constants",
                    "strategy_nulrimok.config.constants",
                ],
                data_dir=Path(data_dir),
            )
            config_watcher.take_baseline()
        except Exception as e:
            logger.debug("ConfigWatcher init failed (non-fatal): %s", e)
            config_watcher = None

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
            sidecar=sidecar,
            config_watcher=config_watcher,
            experiment_registry=experiment_registry,
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
        filter_decisions: Optional[List[Dict[str, Any]]] = None,
        sizing_context: Optional[Dict[str, Any]] = None,
        portfolio_state: Optional[Dict[str, Any]] = None,
        drawdown_context: Optional[Dict[str, Any]] = None,
        experiment_id: Optional[str] = None,
        experiment_variant: Optional[str] = None,
        param_set_id: Optional[str] = None,
        execution_timeline: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a trade entry. Call after OMS fill confirmed."""
        try:
            regime = self._regime_classifier.current_regime(symbol)
            regime_context = self._regime_classifier.classify_multi_tf(symbol)
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
                regime_context=regime_context,
                signal_factors=signal_factors or [],
                filter_decisions=filter_decisions or [],
                sizing_context=sizing_context,
                portfolio_state=portfolio_state,
                drawdown_context=drawdown_context,
                experiment_id=experiment_id,
                experiment_variant=experiment_variant,
                param_set_id=param_set_id,
                execution_timeline=execution_timeline,
            )
        except Exception as e:
            logger.debug(f"Instrumentation on_entry_fill error: {e}")

    def on_exit_fill(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
        mfe_mae_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a trade exit and compute process score."""
        try:
            trade_event = self._trade_logger.log_exit(
                trade_id=trade_id,
                exit_price=exit_price,
                exit_reason=exit_reason,
                mfe_mae_context=mfe_mae_context,
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
                # Queue for post-exit price tracking
                self._exit_backfiller.queue_exit(
                    trade_id=trade_id,
                    symbol=trade_event.pair,
                    side=trade_event.side,
                    exit_price=exit_price,
                    exit_time=trade_event.exit_time or datetime.now(timezone.utc).isoformat(),
                )
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
        filter_decisions: Optional[List[Dict[str, Any]]] = None,
        blocking_positions: Optional[List[Dict[str, Any]]] = None,
        resource_conflict_type: str = "",
        experiment_id: Optional[str] = None,
        experiment_variant: Optional[str] = None,
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
                filter_decisions=filter_decisions or [],
                blocking_positions=blocking_positions,
                resource_conflict_type=resource_conflict_type,
                experiment_id=experiment_id,
                experiment_variant=experiment_variant,
            )
        except Exception as e:
            logger.debug(f"Instrumentation on_signal_blocked error: {e}")

    def on_order_event(
        self,
        order_id: str,
        pair: str,
        order_type: str,
        status: str,
        requested_qty: float,
        filled_qty: float = 0.0,
        requested_price: float | None = None,
        fill_price: float | None = None,
        reject_reason: str = "",
        latency_ms: float | None = None,
        related_trade_id: str = "",
        exchange_timestamp=None,
        bar_id: str | None = None,
    ) -> None:
        """Record an order lifecycle event. Fire-and-forget."""
        try:
            self._order_logger.log_order(
                order_id=order_id,
                pair=pair,
                order_type=order_type,
                status=status,
                requested_qty=requested_qty,
                filled_qty=filled_qty,
                requested_price=requested_price,
                fill_price=fill_price,
                reject_reason=reject_reason,
                latency_ms=latency_ms,
                related_trade_id=related_trade_id,
                exchange_timestamp=exchange_timestamp,
                bar_id=bar_id,
            )
        except Exception:
            pass  # instrumentation must never affect trading

    def periodic_tick(self) -> None:
        """Submit backfill to background thread. Call from heartbeat loop."""
        try:
            self._executor.submit(
                self._missed_logger.run_backfill, self._data_provider
            )
            self._executor.submit(
                self._exit_backfiller.run_backfill, self._data_provider
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

    def emit_heartbeat(
        self,
        active_positions: int = 0,
        open_orders: int = 0,
        uptime_s: float = 0,
        error_count_1h: int = 0,
        positions: list[dict] | None = None,
        portfolio_exposure: dict | None = None,
    ) -> None:
        """Emit periodic heartbeat. Call every 30s from strategy main loop."""
        try:
            sidecar_diag = None
            if self._sidecar:
                try:
                    sidecar_diag = self._sidecar.get_diagnostics()
                except Exception:
                    pass
            self._heartbeat.emit(
                active_positions=active_positions,
                open_orders=open_orders,
                uptime_s=uptime_s,
                error_count_1h=error_count_1h,
                sidecar_diagnostics=sidecar_diag,
                positions=positions,
                portfolio_exposure=portfolio_exposure,
            )
        except Exception:
            pass

    def emit_error(
        self,
        severity: str,
        error_type: str,
        message: str,
        stack_trace: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit an explicit bot error event for sidecar forwarding.

        Unlike instrumentation_errors (internal failures), these are
        trading-relevant errors: OMS failures, API connectivity, data gaps.
        """
        try:
            now = datetime.now(timezone.utc)
            today = now.strftime("%Y-%m-%d")
            record = {
                "event_type": "bot_error",
                "severity": severity,
                "error_type": error_type,
                "message": message,
                "stack_trace": stack_trace,
                "timestamp": now.isoformat(),
                "strategy_type": self._strategy_type,
                "context": context or {},
            }
            outdir = self._data_dir / "bot_errors"
            outdir.mkdir(parents=True, exist_ok=True)
            filepath = outdir / f"bot_errors_{today}.jsonl"
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass

    def on_indicator_snapshot(
        self,
        pair: str,
        indicators: dict[str, float],
        signal_name: str,
        signal_strength: float,
        decision: str,
        strategy_type: str,
        exchange_timestamp=None,
        bar_id: str | None = None,
        context: dict | None = None,
    ) -> None:
        """Fire-and-forget indicator snapshot at signal evaluation."""
        try:
            self._indicator_logger.log_snapshot(
                pair=pair,
                indicators=indicators,
                signal_name=signal_name,
                signal_strength=signal_strength,
                decision=decision,
                strategy_type=strategy_type,
                exchange_timestamp=exchange_timestamp,
                bar_id=bar_id,
                context=context,
            )
        except Exception:
            pass  # instrumentation must never affect trading

    def on_filter_decision(
        self,
        pair: str,
        filter_name: str,
        passed: bool,
        threshold: float,
        actual_value: float,
        signal_name: str = "",
        signal_strength: float = 0.0,
        strategy_type: str = "",
        exchange_timestamp=None,
        bar_id: str | None = None,
    ) -> None:
        """Fire-and-forget filter decision event."""
        try:
            self._filter_logger.log_decision(
                pair=pair, filter_name=filter_name, passed=passed,
                threshold=threshold, actual_value=actual_value,
                signal_name=signal_name, signal_strength=signal_strength,
                strategy_type=strategy_type, exchange_timestamp=exchange_timestamp,
                bar_id=bar_id,
            )
        except Exception:
            pass

    def on_orderbook_context(
        self,
        pair: str,
        best_bid: float,
        best_ask: float,
        trade_context: str | None = None,
        related_trade_id: str | None = None,
        bid_depth_10bps: float = 0.0,
        ask_depth_10bps: float = 0.0,
        exchange_timestamp=None,
    ) -> None:
        """Fire-and-forget order book context capture."""
        try:
            self._orderbook_logger.log_context(
                pair=pair, best_bid=best_bid, best_ask=best_ask,
                trade_context=trade_context, related_trade_id=related_trade_id,
                bid_depth_10bps=bid_depth_10bps, ask_depth_10bps=ask_depth_10bps,
                exchange_timestamp=exchange_timestamp,
            )
        except Exception:
            pass

    def check_config_changes(self) -> None:
        """Check for parameter changes. Call periodically from main loop."""
        try:
            if self._config_watcher:
                self._config_watcher.check()
        except Exception:
            pass

    @property
    def experiment_registry(self) -> Optional[ExperimentRegistry]:
        """Access experiment registry for variant assignment."""
        return self._experiment_registry

    def shutdown(self) -> None:
        """Clean up executor and sidecar resources."""
        try:
            if self._sidecar:
                self._sidecar.stop()
        except Exception:
            pass
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
