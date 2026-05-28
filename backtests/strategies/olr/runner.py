from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from backtests.analysis.metrics import compute_trade_metrics
from backtests.auto.shared.cache_keys import stable_signature
from backtests.core.replay_bundle import EventReplayBundle
from backtests.core.replay_events import ReplayEvent
from backtests.engine.replay import ReplayResult, run_replay
from backtests.engine.sim_broker import BrokerCosts, SimBroker
from strategy_common.actions import StrategyAction, SubmitEntry, SubmitExit, SubmitPartialExit
from strategy_common.actions import action_to_json_dict
from strategy_common.clock import KST
from strategy_common.events import DecisionEvent
from strategy_common.market import MarketBar, require_completed_bar
from strategy_olr.config import OLRConfig, OLR_CORE_VERSION, STRATEGY_ID
from strategy_olr.core.core_models import OLRExpiredOrderEvent, OLRFillEvent, OLRPortfolioView
from strategy_olr.core.logic import on_olr_fill, on_olr_order_expired, remember_submitted_order, step_olr_core
from strategy_olr.core.state import OLRState
from strategy_olr.execution import normalize_action_prices
from strategy_olr.models import OLRDailyCandidate, OLRDailySnapshot


@dataclass(slots=True)
class StrategyBacktestResult:
    strategy: str
    metrics: dict[str, Any]
    replay_result: ReplayResult
    source_fingerprint: str
    capability_level: str
    candidate_snapshot_hash: str
    feature_bundle_hash: str

    @property
    def trades(self):
        return self.replay_result.trades

    @property
    def decisions(self):
        return self.replay_result.decisions


class OLRReplayAdapter:
    strategy_id = STRATEGY_ID

    def __init__(self, config: OLRConfig, snapshots: dict[date, OLRDailySnapshot]):
        if not snapshots:
            raise ValueError("OLR replay adapter requires at least one candidate snapshot")
        self.config = config
        self.snapshots = dict(sorted(snapshots.items()))
        first_snapshot = next(iter(self.snapshots.values()))
        self.state = OLRState(
            snapshot_hash=_aggregate_snapshot_hash(self.snapshots),
            source_fingerprint=first_snapshot.source_fingerprint,
        )
        self._fill_cursor = 0
        self._expired_cursor = 0
        self.auction_order_count = 0
        self.submitted_order_ids: set[str] = set()
        self.action_payloads: list[dict[str, Any]] = []
        self.exposure_points: list[float] = []
        self._snapshot_symbol_sets = {
            day: frozenset(str(candidate.symbol).zfill(6) for candidate in snapshot.candidates)
            for day, snapshot in self.snapshots.items()
        }

    def on_bar(self, bar: MarketBar, broker: SimBroker) -> list[DecisionEvent]:
        decisions: list[DecisionEvent] = []
        decisions.extend(self._sync_new_fills(broker))
        decisions.extend(self._sync_expired_orders(broker, bar.timestamp))
        snapshot = self.snapshots.get(bar.timestamp.date())
        if not self._should_step_symbol(bar.symbol, snapshot):
            return decisions
        portfolio = self._portfolio_view(broker)
        result = step_olr_core(self.state, bar, self.config, snapshot, portfolio)
        self._submit_actions(result.actions, broker, bar.timestamp)
        decisions.extend(result.decisions)
        return decisions

    def on_timestamp_end(self, timestamp: datetime, bars: tuple[MarketBar, ...], broker: SimBroker) -> list[DecisionEvent]:
        decisions = self._sync_new_fills(broker)
        decisions.extend(self._sync_expired_orders(broker, timestamp))
        return decisions

    def _sync_new_fills(self, broker: SimBroker) -> list[DecisionEvent]:
        decisions: list[DecisionEvent] = []
        new_fills = broker.fills[self._fill_cursor :]
        self._fill_cursor = len(broker.fills)
        for fill in new_fills:
            if fill.strategy_id != self.strategy_id:
                continue
            result = on_olr_fill(
                self.state,
                OLRFillEvent(
                    order_id=fill.order_id,
                    symbol=fill.symbol,
                    side=fill.side,
                    qty=fill.qty,
                    price=fill.price,
                    timestamp=fill.timestamp,
                    reason=fill.reason,
                    metadata=dict(fill.metadata),
                ),
                self.config,
            )
            self._submit_actions(result.actions, broker, fill.timestamp)
            decisions.extend(result.decisions)
        return decisions

    def _sync_expired_orders(self, broker: SimBroker, timestamp: datetime) -> list[DecisionEvent]:
        decisions: list[DecisionEvent] = []
        new_orders = broker.expired_orders[self._expired_cursor :]
        self._expired_cursor = len(broker.expired_orders)
        for order in new_orders:
            if order.strategy_id != self.strategy_id:
                continue
            result = on_olr_order_expired(
                self.state,
                OLRExpiredOrderEvent(
                    order_id=str(order.order_id),
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.order_type,
                    qty=order.qty,
                    timestamp=timestamp,
                    reason=order.reason,
                    metadata=dict(order.metadata),
                ),
                self.config,
            )
            self._submit_actions(result.actions, broker, timestamp)
            decisions.extend(result.decisions)
        return decisions

    def _submit_actions(self, actions: list[StrategyAction], broker: SimBroker, submitted_at: datetime) -> None:
        for action in actions:
            normalized = normalize_action_prices(action)
            self.action_payloads.append(
                {
                    "submitted_at": submitted_at.isoformat(),
                    "action": action_to_json_dict(normalized),
                }
            )
            order_id = broker.submit(normalized, submitted_at)
            if isinstance(normalized, (SubmitEntry, SubmitExit, SubmitPartialExit)) and normalized.order_type == "CLOSE_AUCTION":
                self.auction_order_count += 1
            if order_id:
                self.submitted_order_ids.add(str(order_id))
            remember_submitted_order(self.state, order_id, normalized)

    def _portfolio_view(self, broker: SimBroker) -> OLRPortfolioView:
        positions: dict[str, int] = {}
        invested_notional = 0.0
        for (strategy_id, symbol), position in broker.positions.items():
            if strategy_id != self.strategy_id:
                continue
            positions[symbol] = position.qty
            mark_price = broker.last_prices.get(symbol, position.avg_price)
            invested_notional += float(position.qty) * float(mark_price)
        equity = float(broker.cash) + invested_notional
        gross = invested_notional / equity if equity > 0.0 else 0.0
        self.exposure_points.append(gross)
        return OLRPortfolioView(
            cash=float(broker.cash),
            equity=equity,
            positions=positions,
            open_positions=len(positions),
            open_notional=invested_notional,
            gross_exposure_pct=gross,
        )

    def _should_step_symbol(self, symbol: str, snapshot: OLRDailySnapshot | None) -> bool:
        key = str(symbol).zfill(6)
        symbol_state = self.state.symbols.get(key)
        if symbol_state is not None and (
            symbol_state.position is not None
            or bool(symbol_state.pending_entry_order_id)
            or bool(symbol_state.pending_exit_order_id)
        ):
            return True
        if snapshot is None:
            return False
        return key in self._snapshot_symbol_sets.get(snapshot.trade_date, frozenset())


def run_olr_backtest(
    config: dict[str, Any] | None = None,
    mutations: dict[str, Any] | None = None,
    replay_bundle: EventReplayBundle | None = None,
) -> StrategyBacktestResult:
    raw_config = dict(config or {})
    raw_mutations = dict(mutations or {})
    cfg = OLRConfig.from_mapping(raw_config, raw_mutations)
    capability_level = str(raw_config.get("capability_level", "real_replay")).lower()
    if replay_bundle is None:
        if capability_level in {"synthetic", "dry_run"}:
            replay_bundle = compile_olr_replay_bundle(config=raw_config)
        elif capability_level in {"real", "real_replay", "parquet", "krx_replay"}:
            from .replay_cache import load_olr_real_replay_bundle

            replay_bundle = load_olr_real_replay_bundle(raw_config, raw_mutations)
        else:
            raise ValueError("OLR real/official backtests require an explicit replay_bundle or compiled OLR artifact")

    bars = _bars_from_bundle(replay_bundle)
    snapshots = snapshots_from_bundle(replay_bundle)
    if not snapshots and capability_level in {"synthetic", "dry_run"}:
        synthetic_snapshot = _synthetic_snapshot(bars, replay_bundle.source_fingerprint)
        snapshots = {synthetic_snapshot.trade_date: synthetic_snapshot}
    if not snapshots:
        raise ValueError("OLR replay requires source-fingerprinted candidate snapshots from research.py selection")
    if capability_level not in {"synthetic", "dry_run"}:
        from .replay_cache import _candidate_config_hash

        expected_candidate_config_hash = _candidate_config_hash(cfg, raw_mutations)
        bundle_candidate_config_hash = (replay_bundle.metadata or {}).get("candidate_config_hash")
        if bundle_candidate_config_hash and bundle_candidate_config_hash != expected_candidate_config_hash:
            raise ValueError("OLR replay bundle candidate config hash mismatch")

    costs = BrokerCosts(
        commission_bps=cfg.commission_bps,
        tax_bps_on_sell=cfg.tax_bps_on_sell,
        slippage_bps=cfg.slippage_bps,
        auction_slippage_bps=cfg.auction_adverse_bps,
    )
    initial_equity = float(raw_config.get("initial_equity", 10_000_000.0))
    adapter = OLRReplayAdapter(cfg, snapshots)
    result = run_replay(
        bars,
        adapter,
        initial_equity=initial_equity,
        costs=costs,
        bars_are_ordered=True,
        close_open_positions=False,
    )
    result.decisions.extend(adapter._sync_new_fills(result.broker))
    metrics = compute_trade_metrics(result.trades, result.equity_curve, initial_equity=initial_equity)
    _apply_official_metrics(metrics, result, adapter, initial_equity, costs, cfg)
    candidate_snapshot_hash = _aggregate_snapshot_hash(snapshots)
    feature_bundle_hash = str((replay_bundle.metadata or {}).get("olr_feature_bundle_hash") or replay_bundle.source_fingerprint)
    metrics["strategy_core_version"] = OLR_CORE_VERSION
    metrics["replay_mode"] = str((replay_bundle.metadata or {}).get("replay_mode") or capability_level)
    metrics["replay_event_count"] = float(len(replay_bundle.events))
    metrics["candidate_snapshot_count"] = float(len(snapshots))
    metrics["candidate_snapshot_hash"] = candidate_snapshot_hash
    metrics["trade_entry_plan_name"] = str((cfg.trade_entry_plan or {}).get("name") or cfg.entry_mode)
    metrics["trade_entry_plan_mode"] = str((cfg.trade_entry_plan or {}).get("mode") or cfg.entry_mode)
    metrics["trade_exit_plan_name"] = str((cfg.trade_exit_plan or {}).get("name") or cfg.exit_mode)
    metrics["trade_exit_plan_mode"] = str((cfg.trade_exit_plan or {}).get("mode") or cfg.exit_mode)
    metrics["official_trade_plan_supported"] = 1.0 if _official_exit_plan_supported(cfg.trade_exit_plan) else 0.0
    metrics["official_performance"] = False
    return StrategyBacktestResult(
        "olr",
        metrics,
        result,
        replay_bundle.source_fingerprint,
        capability_level,
        candidate_snapshot_hash,
        feature_bundle_hash,
    )


def _official_exit_plan_supported(payload: dict[str, Any] | None) -> bool:
    data = dict(payload or {})
    mode = str(data.get("mode") or "next_close")
    if mode == "next_close":
        return True
    return mode == "managed"


def compile_olr_replay_bundle(
    bars: Iterable[MarketBar] | None = None,
    snapshots: dict[date, OLRDailySnapshot] | None = None,
    *,
    source_fingerprint: str = "",
    data_root: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> EventReplayBundle:
    raw_bars = _synthetic_bars(config) if bars is None else bars
    ordered_bars = tuple(sorted((require_completed_bar(bar) for bar in raw_bars), key=lambda bar: (bar.timestamp, bar.symbol)))
    snapshot_map = dict(snapshots or {})
    fingerprint = source_fingerprint or stable_signature(
        {
            "bars": [
                (bar.symbol, bar.timestamp.isoformat(), round(float(bar.close), 6), round(float(bar.volume), 2))
                for bar in ordered_bars
            ],
            "snapshots": {day.isoformat(): snapshot.artifact_hash for day, snapshot in sorted(snapshot_map.items())},
        }
    )
    if snapshots is None:
        synthetic_snapshot = _synthetic_snapshot(ordered_bars, fingerprint)
        snapshot_map = {synthetic_snapshot.trade_date: synthetic_snapshot}
    metadata: dict[str, Any] = {
        "strategy": "olr",
        "capability_level": "synthetic" if bars is None else "compiled",
        "replay_mode": "olr_core_simbroker",
        "olr_candidate_snapshots": snapshot_map,
        "official_performance": False,
        "causality_policy": {
            "daily_row_cutoff": "row_date < trade_date",
            "flow_row_cutoff": "row_date < trade_date",
            "intraday_selection_cutoff": "timestamp < 14:30 KST",
        },
    }
    return EventReplayBundle(
        events=tuple(ReplayEvent.from_bar(bar) for bar in ordered_bars),
        source_fingerprint=fingerprint,
        data_root=Path(data_root) if data_root else None,
        metadata=metadata,
    )


def snapshots_from_bundle(bundle: EventReplayBundle) -> dict[date, OLRDailySnapshot]:
    metadata = bundle.metadata or {}
    raw = metadata.get("olr_candidate_snapshots") if isinstance(metadata, dict) else None
    if not raw:
        raw = metadata.get("candidate_snapshots") if isinstance(metadata, dict) else None
    snapshots: dict[date, OLRDailySnapshot] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            snapshot = _coerce_snapshot(value)
            if snapshot is not None:
                snapshots[snapshot.trade_date] = snapshot
            elif isinstance(key, date) and isinstance(value, OLRDailySnapshot):
                snapshots[key] = value
    elif isinstance(raw, (list, tuple)):
        for value in raw:
            snapshot = _coerce_snapshot(value)
            if snapshot is not None:
                snapshots[snapshot.trade_date] = snapshot
    return dict(sorted(snapshots.items()))


def attach_overnight_labels_to_snapshots(
    snapshots: dict[date, OLRDailySnapshot],
    labels_by_key: dict[tuple[date, str], Any],
) -> dict[date, OLRDailySnapshot]:
    if not labels_by_key:
        return snapshots
    out: dict[date, OLRDailySnapshot] = {}
    for day, snapshot in snapshots.items():
        candidates = []
        for candidate in snapshot.candidates:
            label = labels_by_key.get((day, candidate.symbol))
            if label is None:
                candidates.append(candidate)
                continue
            entry = max(float(getattr(label, "entry_close", 0.0) or 0.0), 1e-9)
            next_close = float(getattr(label, "next_close", entry) or entry)
            next_high = float(getattr(label, "next_high", entry) or entry)
            candidates.append(
                replace(
                    candidate,
                    metadata={
                        **dict(candidate.metadata),
                        "close_to_close_label_pct": next_close / entry - 1.0,
                        "next_session_mfe_label_pct": max(0.0, next_high / entry - 1.0),
                    },
                )
            )
        out[day] = replace(snapshot, candidates=tuple(candidates))
    return out


def _coerce_snapshot(value: Any) -> OLRDailySnapshot | None:
    if isinstance(value, OLRDailySnapshot):
        return value
    if isinstance(value, dict):
        try:
            return OLRDailySnapshot.from_json_dict(value)
        except Exception:
            return None
    return None


def _bars_from_bundle(bundle: EventReplayBundle) -> tuple[MarketBar, ...]:
    metadata = bundle.metadata or {}
    cached = metadata.get("_olr_sorted_bars_obj") if isinstance(metadata, dict) else None
    if isinstance(cached, tuple) and all(isinstance(item, MarketBar) for item in cached):
        return cached
    bars = tuple(
        sorted(
            (require_completed_bar(event.bar) for event in bundle.events if event.bar is not None),
            key=lambda bar: (bar.timestamp, bar.symbol),
        )
    )
    if isinstance(metadata, dict):
        metadata["_olr_sorted_bars_obj"] = bars
    return bars


def _apply_official_metrics(
    metrics: dict[str, Any],
    result: ReplayResult,
    adapter: OLRReplayAdapter,
    initial_equity: float,
    costs: BrokerCosts,
    config: OLRConfig,
) -> None:
    final_equity = float(result.equity_curve[-1]) if result.equity_curve else float(initial_equity)
    metrics["official_mtm_net_return_pct"] = (final_equity / float(initial_equity) - 1.0) if initial_equity else 0.0
    metrics["final_equity"] = final_equity
    metrics["official_mtm_max_drawdown_pct"] = float(metrics.get("max_drawdown_pct", 0.0))
    metrics["official_mtm_sharpe"] = float(metrics.get("sharpe", 0.0))
    metrics["gross_exposure_avg_pct"] = float(mean(adapter.exposure_points)) if adapter.exposure_points else 0.0
    metrics["rejected_order_count"] = float(len(result.broker.rejected_orders))
    metrics["same_bar_fill_count"] = float(result.broker.same_bar_fill_violations)
    metrics["auction_order_count"] = float(adapter.auction_order_count)
    metrics["auction_nonfill_count"] = float(result.broker.auction_nonfill_count)
    metrics["open_order_count"] = float(len(result.broker.orders))
    metrics["expired_order_count"] = float(len(result.broker.expired_orders))
    metrics["entry_fill_count"] = float(sum(1 for fill in result.broker.fills if fill.strategy_id == adapter.strategy_id and fill.side == "BUY"))
    metrics["exit_fill_count"] = float(sum(1 for fill in result.broker.fills if fill.strategy_id == adapter.strategy_id and fill.side == "SELL"))
    metrics.update(_entry_level_trade_metrics(result.trades))
    metrics["close_to_close_alpha_capture_pct"] = float(_close_to_close_alpha_capture(result.trades))
    metrics["open_position_count"] = float(len(result.broker.positions))
    metrics["end_open_position_count"] = float(len(result.broker.positions))
    metrics["forced_replay_close_count"] = float(sum(1 for trade in result.trades if trade.exit_reason == "end_of_replay"))
    metrics["net_return_pct_basis"] = "closed_trade_net_pnl_over_initial_equity"
    metrics["legacy_slot_metric_basis"] = "non_official_research_labels"
    metrics["official_metric_basis"] = "SimBroker.equity_curve_bar_level_mtm"
    metrics["cost_policy"] = {
        "commission_bps": float(costs.commission_bps),
        "slippage_bps": float(costs.slippage_bps),
        "tax_bps_on_sell": float(costs.tax_bps_on_sell),
        "auction_adverse_bps": float(config.auction_adverse_bps),
        "auction_slippage_bps": float(costs.auction_slippage_bps),
        "auction_nonfill_rate": float(config.auction_nonfill_rate),
        "auction_limit_offset_bps": float(config.auction_limit_offset_bps),
        "market_entry_price_buffer_bps": float(config.market_entry_price_buffer_bps),
    }
    metrics["live_parity_fill_timing"] = str(config.live_parity_fill_timing)
    metrics["decision_hash"] = stable_signature(_canonical_audit_rows(decision.to_json_dict() for decision in result.decisions))
    metrics["neutral_action_hash"] = stable_signature(_canonical_audit_rows(adapter.action_payloads))
    metrics["fill_hash"] = stable_signature(
        _canonical_audit_rows(_fill_hash_payload(fill) for fill in result.broker.fills if fill.strategy_id == adapter.strategy_id)
    )
    metrics["trade_hash"] = stable_signature(
        _canonical_audit_rows(trade.to_json_dict() for trade in result.trades if trade.strategy_id == adapter.strategy_id)
    )
    metrics["source_snapshot_hash"] = str(adapter.state.snapshot_hash)
    metrics["state_snapshot_hash"] = stable_signature(_state_hash_payload(adapter.state))
    metrics["final_state_hash"] = metrics["state_snapshot_hash"]


def _fill_hash_payload(fill) -> dict[str, Any]:
    return {
        "strategy_id": fill.strategy_id,
        "symbol": fill.symbol,
        "side": fill.side,
        "qty": fill.qty,
        "price": fill.price,
        "timestamp": fill.timestamp.isoformat(),
        "reason": fill.reason,
        "metadata": dict(fill.metadata),
    }


def _canonical_audit_rows(rows: Iterable[Any]) -> list[Any]:
    payloads = [_scrub_audit_payload(row) for row in rows]
    return sorted(payloads, key=stable_signature)


def _scrub_audit_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _scrub_audit_payload(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
            if str(key) not in _NONDETERMINISTIC_AUDIT_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_scrub_audit_payload(item) for item in value]
    return value


_NONDETERMINISTIC_AUDIT_KEYS = frozenset(
    {
        "order_id",
        "entry_order_id",
        "exit_order_id",
        "expired_order_id",
        "pending_entry_order_id",
        "pending_exit_order_id",
    }
)


def _state_hash_payload(state: OLRState) -> dict[str, Any]:
    return {
        "snapshot_hash": state.snapshot_hash,
        "source_fingerprint": state.source_fingerprint,
        "session_date": state.session_date.isoformat() if state.session_date else "",
        "meta": dict(state.meta),
        "order_roles": sorted((dict(role) for role in state.order_roles.values()), key=stable_signature),
        "symbols": [_symbol_state_hash_payload(symbol, symbol_state) for symbol, symbol_state in sorted(state.symbols.items())],
    }


def _symbol_state_hash_payload(symbol: str, symbol_state) -> dict[str, Any]:
    last_bar_ts = symbol_state.session_bars[-1].timestamp.isoformat() if symbol_state.session_bars else ""
    return {
        "symbol": symbol,
        "stage": getattr(symbol_state.stage, "value", str(symbol_state.stage)),
        "session_date": symbol_state.session_date.isoformat() if symbol_state.session_date else "",
        "candidate": symbol_state.candidate.to_json_dict() if symbol_state.candidate is not None else None,
        "pending_entry": bool(symbol_state.pending_entry_order_id),
        "pending_exit": bool(symbol_state.pending_exit_order_id),
        "pending_entry_metadata": dict(symbol_state.pending_entry_metadata),
        "pending_exit_metadata": dict(symbol_state.pending_exit_metadata),
        "session_bar_count": len(symbol_state.session_bars),
        "last_session_bar_ts": last_bar_ts,
        "position": _position_state_hash_payload(symbol_state.position),
        "entry_attempted": bool(symbol_state.entry_attempted),
        "exit_attempted_dates": sorted(day.isoformat() for day in symbol_state.exit_attempted_dates),
        "last_decision_code": symbol_state.last_decision_code,
        "last_decision_details": dict(symbol_state.last_decision_details),
    }


def _position_state_hash_payload(position) -> dict[str, Any] | None:
    if position is None:
        return None
    return {
        "symbol": position.symbol,
        "qty_open": int(position.qty_open),
        "entry_price": float(position.entry_price),
        "entry_time": position.entry_time.isoformat(),
        "candidate_rank": int(position.candidate_rank),
        "candidate_score": float(position.candidate_score),
        "source_artifact_hash": position.source_artifact_hash,
        "sector": position.sector,
        "max_favorable_price": float(position.max_favorable_price),
        "max_adverse_price": float(position.max_adverse_price),
        "metadata": dict(position.metadata),
    }


def _entry_level_trade_metrics(trades) -> dict[str, float]:
    groups: dict[tuple[Any, ...], list[Any]] = {}
    for trade in trades:
        key = (
            getattr(trade, "strategy_id", ""),
            getattr(trade, "symbol", ""),
            getattr(trade, "entry_decision_time", None),
            getattr(trade, "entry_fill_time", None),
            round(float(getattr(trade, "entry_price", 0.0) or 0.0), 8),
            getattr(trade, "source_artifact_hash", ""),
        )
        groups.setdefault(key, []).append(trade)
    entry_net_values: list[float] = []
    entry_r_values: list[float] = []
    exit_leg_count = 0
    for legs in groups.values():
        if not legs:
            continue
        exit_leg_count += len(legs)
        entry_net = sum(float(getattr(leg, "net_pnl", 0.0) or 0.0) for leg in legs)
        entry_net_values.append(entry_net)
        total_qty = sum(max(int(getattr(leg, "qty", 0) or 0), 0) for leg in legs)
        first = legs[0]
        risk = float(getattr(first, "route_metadata", {}).get("risk_per_share", 0.0) or 0.0)
        if risk > 0.0 and total_qty > 0:
            entry_r_values.append(entry_net / (risk * total_qty))
        else:
            weighted_r = 0.0
            for leg in legs:
                qty = max(int(getattr(leg, "qty", 0) or 0), 0)
                weighted_r += float(getattr(leg, "r_multiple", 0.0) or 0.0) * qty
            entry_r_values.append(weighted_r / total_qty if total_qty > 0 else 0.0)
    total_entries = len(groups)
    wins = [value for value in entry_net_values if value > 0.0]
    losses = [value for value in entry_net_values if value < 0.0]
    net_losses = abs(sum(losses))
    expected_total_r = sum(entry_r_values)
    return {
        "entry_level_trade_count": float(total_entries),
        "entry_level_expected_total_r": float(expected_total_r),
        "entry_level_avg_r": float(expected_total_r / total_entries) if total_entries else 0.0,
        "entry_level_win_rate": float(len(wins) / total_entries) if total_entries else 0.0,
        "entry_level_profit_factor": float(sum(wins) / net_losses) if net_losses > 0.0 else (999.0 if wins else 0.0),
        "exit_leg_count": float(exit_leg_count),
        "exit_leg_to_entry_ratio": float(exit_leg_count / total_entries) if total_entries else 0.0,
        "partial_exit_leg_count": float(max(0, exit_leg_count - total_entries)),
    }


def _close_to_close_alpha_capture(trades) -> float:
    captures: list[float] = []
    for trade in trades:
        label = float(trade.route_metadata.get("close_to_close_label_pct", 0.0) or 0.0)
        actual = (float(trade.exit_price or trade.entry_price) / max(float(trade.entry_price), 1e-9)) - 1.0
        if abs(label) > 1e-9:
            captures.append(actual / label)
    return mean(captures) if captures else 0.0


def _synthetic_bars(config: dict[str, Any] | None = None) -> list[MarketBar]:
    symbol = str((config or {}).get("symbol", "005930")).zfill(6)
    trade_date = date(2026, 1, 5)
    next_date = trade_date + timedelta(days=1)
    rows = [
        (trade_date, time(14, 25), 70_000.0, 70_250.0, 69_900.0, 70_200.0, 120_000.0),
        (trade_date, time(14, 30), 70_200.0, 70_450.0, 70_150.0, 70_380.0, 135_000.0),
        (trade_date, time(15, 15), 70_380.0, 70_900.0, 70_300.0, 70_820.0, 180_000.0),
        (trade_date, time(15, 30), 70_820.0, 71_250.0, 70_780.0, 71_100.0, 240_000.0),
        (next_date, time(14, 30), 72_400.0, 73_100.0, 72_250.0, 72_850.0, 175_000.0),
        (next_date, time(15, 15), 72_850.0, 73_400.0, 72_700.0, 73_200.0, 210_000.0),
        (next_date, time(15, 30), 73_200.0, 73_550.0, 73_050.0, 73_450.0, 260_000.0),
    ]
    bars = []
    for day, clock, open_px, high, low, close, volume in rows:
        bars.append(
            MarketBar(
                symbol=symbol,
                timestamp=datetime.combine(day, clock, tzinfo=KST),
                timeframe="5m",
                open=open_px,
                high=high,
                low=low,
                close=close,
                volume=volume,
                source="synthetic_olr",
                source_fingerprint="synthetic_olr_v1",
            )
        )
    return bars


def _synthetic_snapshot(bars: Iterable[MarketBar], source_fingerprint: str) -> OLRDailySnapshot:
    ordered = sorted(bars, key=lambda bar: (bar.timestamp, bar.symbol))
    if not ordered:
        raise ValueError("Cannot build OLR synthetic snapshot without bars")
    trade_date = min(bar.timestamp.date() for bar in ordered)
    symbol = ordered[0].symbol
    first = next(bar for bar in ordered if bar.timestamp.date() == trade_date)
    candidate = OLRDailyCandidate(
        symbol=symbol,
        trade_date=trade_date,
        prior_day_high=first.open * 1.01,
        prior_day_low=first.open * 0.98,
        prior_day_close=first.open * 0.995,
        daily_atr=max(first.open * 0.018, 1.0),
        expected_5m_volume=100_000.0,
        average_30m_volume=700_000.0,
        sector="SYNTH",
        regime_tier="A",
        selection_score=92.0,
        daily_signal_score=88.0,
        rank=1,
        rank_pct=1.0,
        rs_percentile=98.0,
        source_fingerprint=source_fingerprint,
        metadata={"synthetic": True},
    )
    return OLRDailySnapshot(
        trade_date=trade_date,
        candidates=(candidate,),
        source_fingerprint=source_fingerprint,
        generated_at=datetime.combine(trade_date, datetime.min.time(), tzinfo=KST),
        metadata={
            "synthetic": True,
            "daily_row_cutoff": "row_date < trade_date",
            "flow_row_cutoff": "row_date < trade_date",
            "intraday_selection_cutoff": "timestamp < 14:30 KST",
            "official_performance": False,
        },
    )


def _aggregate_snapshot_hash(snapshots: dict[date, OLRDailySnapshot]) -> str:
    if len(snapshots) == 1:
        return next(iter(snapshots.values())).artifact_hash
    return stable_signature({day.isoformat(): snapshot.artifact_hash for day, snapshot in sorted(snapshots.items())})
