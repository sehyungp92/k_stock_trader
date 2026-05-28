from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from typing import Any

from backtests.analysis.metrics import compute_trade_metrics
from backtests.auto.shared.cache_keys import stable_signature
from backtests.core.replay_bundle import EventReplayBundle
from backtests.engine.replay import ReplayResult, run_replay
from backtests.engine.sim_broker import BrokerCosts, SimBroker
from backtests.strategies.common.capabilities import KALCB_OFFICIAL_REQUIREMENTS, require_capabilities
from backtests.strategies.common.synthetic import make_synthetic_replay_bundle
from strategy_common.actions import StrategyAction, SubmitEntry
from strategy_common.events import DecisionEvent, TradeOutcome
from strategy_common.market import MarketBar
from strategy_kalcb.config import KALCBConfig, KALCB_CORE_VERSION, STRATEGY_ID
from strategy_kalcb.core.core_models import KALCBFillEvent, KALCBPortfolioView
from strategy_kalcb.core.logic import on_kalcb_fill, on_kalcb_timer, remember_submitted_order, step_kalcb_core
from strategy_kalcb.core.state import KALCBState
from strategy_kalcb.execution import normalize_action_prices
from strategy_kalcb.models import KALCBDailyCandidate, KALCBDailySnapshot

from .features import build_feature_bundle_hash_for_snapshots, require_kalcb_feature_metadata, snapshot_from_bundle, snapshots_from_bundle
from .replay_cache import _candidate_config_hash, _resolve_sector_map, load_kalcb_real_replay_bundle


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


class KALCBReplayAdapter:
    strategy_id = STRATEGY_ID

    def __init__(
        self,
        config: KALCBConfig,
        snapshots: dict[date, KALCBDailySnapshot],
        *,
        initial_equity: float,
        costs: BrokerCosts,
    ):
        if not snapshots:
            raise ValueError("KALCB replay adapter requires at least one candidate snapshot")
        self.config = config
        self.snapshots = dict(sorted(snapshots.items()))
        first_snapshot = next(iter(self.snapshots.values()))
        self.state = KALCBState(
            snapshot_hash=_aggregate_snapshot_hash(self.snapshots),
            source_fingerprint=first_snapshot.source_fingerprint,
        )
        self._fill_cursor = 0
        self._daily_active_symbols: dict[date, set[str]] = {}
        self.frontier_promotion_log: list[dict[str, Any]] = []
        self.frontier_shadow = (
            KALCBFrontierShadow(config, self.snapshots, initial_equity=initial_equity, costs=costs)
            if config.frontier_enabled and config.frontier_shadow_enabled
            else None
        )

    def on_bar(self, bar: MarketBar, broker: SimBroker) -> list[DecisionEvent]:
        decisions: list[DecisionEvent] = []
        decisions.extend(self._sync_new_fills(broker))
        portfolio = self._portfolio_view(broker)
        full_snapshot = self.snapshots.get(bar.timestamp.date())
        active_snapshot = self._active_snapshot(full_snapshot)
        active_symbols = self._daily_active_symbols.get(bar.timestamp.date(), set())
        execution_symbols = active_symbols
        if self.config.entry_plan_frontier_branch_universe and full_snapshot is not None:
            execution_symbols = {candidate.symbol for candidate in full_snapshot.candidates}
        symbol_state = self.state.symbols.get(bar.symbol)
        has_working_state = bool(
            symbol_state is not None
            and (symbol_state.position is not None or symbol_state.pending_entry_order_id or symbol_state.stage.name == "ENTRY_QUEUED")
        )
        if bar.symbol in execution_symbols or has_working_state:
            result = step_kalcb_core(self.state, bar, self.config, active_snapshot, portfolio)
            self._submit_actions(result.actions, broker, bar.timestamp)
            decisions.extend(result.decisions)
        if self.frontier_shadow is not None and full_snapshot is not None:
            self.frontier_shadow.on_bar(bar, full_snapshot, active_symbols)
        return decisions

    def on_timestamp_end(self, timestamp: datetime, bars: tuple[MarketBar, ...], broker: SimBroker) -> list[DecisionEvent]:
        del bars
        decisions: list[DecisionEvent] = []
        decisions.extend(self._sync_new_fills(broker))
        timer_timestamp = timestamp + _execution_bar_delta(self.config)
        result = on_kalcb_timer(self.state, timer_timestamp, self.config)
        self._submit_actions(result.actions, broker, timestamp)
        decisions.extend(result.decisions)
        return decisions

    def _sync_new_fills(self, broker: SimBroker) -> list[DecisionEvent]:
        decisions: list[DecisionEvent] = []
        new_fills = broker.fills[self._fill_cursor :]
        self._fill_cursor = len(broker.fills)
        for fill in new_fills:
            if fill.strategy_id != self.strategy_id:
                continue
            result = on_kalcb_fill(
                self.state,
                KALCBFillEvent(
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

    def _submit_actions(self, actions: list[StrategyAction], broker: SimBroker, submitted_at: datetime) -> None:
        for action in actions:
            normalized = normalize_action_prices(action)
            order_id = broker.submit(normalized, submitted_at)
            remember_submitted_order(self.state, order_id, normalized)

    def _portfolio_view(self, broker: SimBroker) -> KALCBPortfolioView:
        return _portfolio_view_for_broker(broker, self.strategy_id)

    def _active_snapshot(self, snapshot: KALCBDailySnapshot | None) -> KALCBDailySnapshot | None:
        if snapshot is None:
            return None
        session = snapshot.trade_date
        active = self._daily_active_symbols.get(session)
        if active is None:
            active = self._select_daily_active_symbols(snapshot)
            self._daily_active_symbols[session] = active
        if self.config.entry_plan_frontier_branch_universe:
            return KALCBDailySnapshot(
                trade_date=snapshot.trade_date,
                candidates=tuple(snapshot.candidates),
                source_fingerprint=snapshot.source_fingerprint,
                generated_at=snapshot.generated_at,
                strategy_id=snapshot.strategy_id,
                metadata={
                    **dict(snapshot.metadata),
                    "active_symbols": [candidate.symbol for candidate in snapshot.candidates],
                    "active_symbol_count": len(snapshot.candidates),
                    "frontier_execution_view": True,
                    "frontier_branch_universe": True,
                    "physical_active_symbols": sorted(active),
                    "physical_active_symbol_count": len(active),
                },
            )
        candidates = tuple(candidate for candidate in snapshot.candidates if candidate.symbol in active)
        if len(candidates) == len(snapshot.candidates):
            return snapshot
        return KALCBDailySnapshot(
            trade_date=snapshot.trade_date,
            candidates=candidates,
            source_fingerprint=snapshot.source_fingerprint,
            generated_at=snapshot.generated_at,
            strategy_id=snapshot.strategy_id,
            metadata={
                **dict(snapshot.metadata),
                "active_symbols": [candidate.symbol for candidate in candidates],
                "active_symbol_count": len(candidates),
                "frontier_execution_view": True,
            },
        )

    def _select_daily_active_symbols(self, snapshot: KALCBDailySnapshot) -> set[str]:
        ws_budget = max(1, int(self.config.ws_budget))
        initial = [
            str(symbol)
            for symbol in (snapshot.metadata.get("active_symbols") or [])
            if str(symbol) in snapshot.by_symbol()
        ]
        if not initial:
            initial = [candidate.symbol for candidate in snapshot.candidates[:ws_budget]]
        initial = _dedupe(initial)[:ws_budget]
        promotions: list[dict[str, Any]] = []
        if self.frontier_shadow is not None and self.config.frontier_rotation_enabled and self.config.frontier_rotation_slots > 0:
            promotions = self.frontier_shadow.eligible_promotions(snapshot, exclude=set(initial))[: self.config.frontier_rotation_slots]
        promoted_symbols = [item["symbol"] for item in promotions]
        keep_count = max(0, ws_budget - len(promoted_symbols))
        selected = _dedupe(initial[:keep_count] + promoted_symbols + initial[keep_count:])[:ws_budget]
        if promotions:
            self.frontier_promotion_log.append(
                {
                    "trade_date": snapshot.trade_date.isoformat(),
                    "promoted_symbols": promoted_symbols,
                    "kept_initial_symbols": selected,
                    "promotion_evidence": promotions,
                    "ws_budget": ws_budget,
                }
            )
        return set(selected)

    def finalize_frontier_shadow(self, last_bar: MarketBar | None) -> None:
        if self.frontier_shadow is not None:
            self.frontier_shadow.finalize(last_bar)

    def frontier_metrics(self) -> dict[str, Any]:
        summary = self.frontier_shadow.summary() if self.frontier_shadow is not None else {}
        return {
            "frontier_rotation_enabled": self.config.frontier_rotation_enabled,
            "frontier_rotation_days": float(len(self.frontier_promotion_log)),
            "frontier_rotation_promotion_count": float(sum(len(item.get("promoted_symbols", ())) for item in self.frontier_promotion_log)),
            "frontier_promotion_log": list(self.frontier_promotion_log),
            **summary,
        }


_FRONTIER_SHADOW_CONTEXT_EXPORT_KEYS = (
    "entry_route_priority",
    "entry_route_attempts",
    "entry_route_risk_mult",
    "entry_route_notional_mult",
    "entry_route_participation_mult",
    "entry_route_max_session_trades",
    "entry_route_context_min_keys",
    "entry_route_context_max_keys",
    "entry_route_context_exclude_keys",
    "entry_route_session_count_before",
    "frontier_selection_score",
    "flow_score",
    "accumulation_score",
    "regime_tier",
    "first30_gap",
    "first30_gap_retention_ratio",
    "first30_gap_relvol",
    "first30_low_vs_prev_relvol",
    "first30_open_drawdown",
    "first30_low_vs_prev_close",
    "first30_range_close_location",
    "first30_quality_pct",
    "daily_return_5d",
    "daily_return_20d",
    "daily_return_60d",
    "daily_volume_ratio_20d",
    "daily_close20_loc",
    "daily_acceleration_5v20",
    "daily_momentum_pct",
    "daily_sector_alignment_pct",
    "stock_sector_daily_ret5_spread",
    "stock_sector_daily_ret20_spread",
    "first30_sector_ret_spread",
    "first30_sector_relvol_ratio",
    "first30_sector_leadership_pct",
    "first30_gap_relvol_sector_breadth",
    "first30_gap_retention_sector_breadth",
    "continuation_joint_quality_pct",
    "sector_participation",
    "sector_daily_score_pct",
    "sector_daily_participation",
    "sector_daily_breadth_20d",
    "sector_daily_ret_5d",
    "sector_daily_ret_20d",
    "sector_daily_ret_60d",
    "sector_intraday_score_pct",
    "sector_intraday_ret",
    "sector_intraday_breadth",
    "sector_intraday_participation",
    "sector_intraday_rel_volume",
    "sector_intraday_effective_count",
    "session_sector_intraday_score_pct_mean",
    "session_sector_intraday_positive_share",
    "session_sector_intraday_effective_count_mean",
    "entry_path_anchor_time",
    "entry_path_anchor_price",
    "entry_path_risk_per_share",
    "entry_path_completed_bars",
    "h1_current_r",
    "h1_mfe_r",
    "h1_mae_r",
    "h1_giveback_r",
    "h3_current_r",
    "h3_mfe_r",
    "h3_mae_r",
    "h3_giveback_r",
    "h6_current_r",
    "h6_mfe_r",
    "h6_mae_r",
    "h6_giveback_r",
    "h12_current_r",
    "h12_mfe_r",
    "h12_mae_r",
    "h12_giveback_r",
    "below_or_high_streak",
    "above_vwap_streak",
    "vwap_distance_pct",
)


class KALCBFrontierShadow:
    strategy_id = STRATEGY_ID

    def __init__(
        self,
        config: KALCBConfig,
        snapshots: dict[date, KALCBDailySnapshot],
        *,
        initial_equity: float,
        costs: BrokerCosts,
    ):
        self.config = replace(
            config,
            max_positions=max(config.max_positions, config.frontier_shadow_max_positions),
            max_per_sector=max(config.max_per_sector, config.frontier_shadow_max_positions),
            heat_cap_r=max(config.heat_cap_r, float(config.frontier_shadow_max_positions)),
            frontier_rotation_enabled=False,
        )
        self.snapshots = snapshots
        first_snapshot = next(iter(snapshots.values()))
        self.state = KALCBState(
            snapshot_hash=_aggregate_snapshot_hash(snapshots),
            source_fingerprint=first_snapshot.source_fingerprint,
        )
        self.broker = SimBroker(initial_equity=max(float(initial_equity) * 10.0, float(initial_equity)), costs=costs)
        self._fill_cursor = 0
        self._trade_cursor = 0
        self._pending_trade_legs: dict[tuple[str, str, str, float], list[TradeOutcome]] = {}
        self.symbol_stats: dict[str, dict[str, float]] = {}
        self.trade_rows: list[dict[str, Any]] = []

    def on_bar(self, bar: MarketBar, snapshot: KALCBDailySnapshot, active_symbols: set[str]) -> None:
        self.broker.process_bar(bar)
        self._sync_new_fills()
        portfolio = _portfolio_view_for_broker(self.broker, self.strategy_id)
        result = step_kalcb_core(self.state, bar, self.config, snapshot, portfolio)
        self._submit_actions(result.actions, bar.timestamp, active_symbols)
        self._record_new_trades()

    def finalize(self, last_bar: MarketBar | None) -> None:
        if last_bar is not None:
            self.broker.close_all_at_end(last_bar, reason="frontier_shadow_end_of_replay")
            self._sync_new_fills()
            self._record_new_trades()
        for key, legs in list(self._pending_trade_legs.items()):
            if legs:
                self._record_trade_row(_collapse_trade_legs(legs))
            self._pending_trade_legs.pop(key, None)

    def eligible_promotions(self, snapshot: KALCBDailySnapshot, *, exclude: set[str]) -> list[dict[str, Any]]:
        if not self.frontier_proof_ready():
            return []
        rows = self._eligible_promotion_rows(snapshot, exclude=exclude)
        if not self._eligible_proof_ready(rows):
            return []
        rows.sort(key=lambda item: (item["shadow_avg_r"], item["shadow_total_r"], -item["frontier_rank"]), reverse=True)
        return rows

    def frontier_proof_ready(self) -> bool:
        proof = self._frontier_global_proof()
        if proof["trades"] < self.config.frontier_rotation_min_frontier_trades:
            return False
        return (
            proof["total_r"] >= self.config.frontier_rotation_min_frontier_total_r
            and proof["avg_r"] >= self.config.frontier_rotation_min_frontier_avg_r
        )

    def summary(self) -> dict[str, Any]:
        all_r = [float(row["r"]) for row in self.trade_rows]
        nonselected = [row for row in self.trade_rows if not row["active_at_signal"]]
        nonselected_r = [float(row["r"]) for row in nonselected]
        nonselected_total_r = float(sum(nonselected_r))
        nonselected_avg_r = float(nonselected_total_r / len(nonselected_r)) if nonselected_r else 0.0
        eligible_symbols = [
            {"symbol": symbol, **stats}
            for symbol, stats in sorted(
                self.symbol_stats.items(),
                key=lambda item: (item[1].get("total_r", 0.0), item[1].get("trades", 0.0)),
                reverse=True,
            )
            if int(stats.get("trades", 0.0)) >= self.config.frontier_rotation_min_shadow_trades
            and (float(stats.get("total_r", 0.0)) / max(float(stats.get("trades", 0.0)), 1.0)) >= self.config.frontier_rotation_min_avg_r
            and float(stats.get("total_r", 0.0)) > self.config.frontier_rotation_min_total_r
        ]
        proof = self._promotion_proof()
        global_proof = self._frontier_global_proof()
        return {
            "frontier_shadow_trade_count": float(len(self.trade_rows)),
            "frontier_shadow_expected_total_r": float(sum(all_r)),
            "frontier_shadow_avg_r": float(sum(all_r) / len(all_r)) if all_r else 0.0,
            "frontier_shadow_nonselected_trade_count": float(len(nonselected)),
            "frontier_shadow_nonselected_total_r": nonselected_total_r,
            "frontier_shadow_nonselected_avg_r": nonselected_avg_r,
            "frontier_shadow_eligible_symbol_count": float(len(eligible_symbols)),
            "frontier_shadow_top_symbols": eligible_symbols[:12],
            "frontier_rotation_proof_symbol_count": float(proof["symbols"]),
            "frontier_rotation_proof_trade_count": float(proof["trades"]),
            "frontier_rotation_proof_total_r": float(proof["total_r"]),
            "frontier_rotation_proof_avg_r": float(proof["avg_r"]),
            "frontier_rotation_global_trade_count": float(global_proof["trades"]),
            "frontier_rotation_global_total_r": float(global_proof["total_r"]),
            "frontier_rotation_global_avg_r": float(global_proof["avg_r"]),
            "frontier_rotation_frontier_proof_ready": self.frontier_proof_ready(),
            "frontier_shadow_trade_rows": list(self.trade_rows),
        }

    def _eligible_proof_ready(self, rows: list[dict[str, Any]]) -> bool:
        proof = self._promotion_proof(rows)
        if proof["symbols"] < self.config.frontier_rotation_min_proof_symbols:
            return False
        return (
            proof["trades"] >= self.config.frontier_rotation_min_frontier_trades
            and proof["total_r"] >= self.config.frontier_rotation_min_frontier_total_r
            and proof["avg_r"] >= self.config.frontier_rotation_min_frontier_avg_r
        )

    def _eligible_promotion_rows(self, snapshot: KALCBDailySnapshot, *, exclude: set[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        by_symbol = snapshot.by_symbol()
        for symbol, stats in self.symbol_stats.items():
            if symbol in exclude or symbol not in by_symbol:
                continue
            trades = int(stats.get("trades", 0.0))
            total_r = float(stats.get("total_r", 0.0))
            avg_r = total_r / trades if trades else 0.0
            if trades < self.config.frontier_rotation_min_shadow_trades:
                continue
            if avg_r < self.config.frontier_rotation_min_avg_r:
                continue
            if total_r <= self.config.frontier_rotation_min_total_r:
                continue
            candidate = by_symbol[symbol]
            rows.append(
                {
                    "symbol": symbol,
                    "shadow_trades": trades,
                    "shadow_total_r": total_r,
                    "shadow_avg_r": avg_r,
                    "frontier_rank": int(candidate.metadata.get("frontier_rank") or 0),
                    "frontier_selection_score": float(candidate.metadata.get("frontier_selection_score") or candidate.selection_score or 0.0),
                }
            )
        return rows

    def _promotion_proof(self, rows: list[dict[str, Any]] | None = None) -> dict[str, float]:
        proof_rows = rows
        if proof_rows is None:
            proof_rows = []
            for symbol, stats in self.symbol_stats.items():
                trades = int(stats.get("trades", 0.0))
                total_r = float(stats.get("total_r", 0.0))
                avg_r = total_r / trades if trades else 0.0
                if trades < self.config.frontier_rotation_min_shadow_trades:
                    continue
                if avg_r < self.config.frontier_rotation_min_avg_r:
                    continue
                if total_r <= self.config.frontier_rotation_min_total_r:
                    continue
                proof_rows.append({"symbol": symbol, "shadow_trades": trades, "shadow_total_r": total_r, "shadow_avg_r": avg_r})
        trades = int(sum(int(row.get("shadow_trades", 0) or 0) for row in proof_rows))
        total_r = float(sum(float(row.get("shadow_total_r", 0.0) or 0.0) for row in proof_rows))
        return {
            "symbols": float(len(proof_rows)),
            "trades": float(trades),
            "total_r": total_r,
            "avg_r": float(total_r / trades) if trades else 0.0,
        }

    def _frontier_global_proof(self) -> dict[str, float]:
        nonselected = [row for row in self.trade_rows if not row["active_at_signal"]]
        trades = len(nonselected)
        total_r = float(sum(float(row["r"]) for row in nonselected))
        return {
            "trades": float(trades),
            "total_r": total_r,
            "avg_r": float(total_r / trades) if trades else 0.0,
        }

    def _sync_new_fills(self) -> None:
        new_fills = self.broker.fills[self._fill_cursor :]
        self._fill_cursor = len(self.broker.fills)
        for fill in new_fills:
            if fill.strategy_id != self.strategy_id:
                continue
            result = on_kalcb_fill(
                self.state,
                KALCBFillEvent(
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
            self._submit_actions(result.actions, fill.timestamp, set())

    def _submit_actions(self, actions: list[StrategyAction], submitted_at: datetime, active_symbols: set[str]) -> None:
        for action in actions:
            patched = action
            if isinstance(action, SubmitEntry):
                patched = replace(
                    action,
                    metadata={
                        **dict(action.metadata),
                        "frontier_shadow": True,
                        "frontier_shadow_active_at_signal": action.symbol in active_symbols,
                    },
                )
            normalized = normalize_action_prices(patched)
            order_id = self.broker.submit(normalized, submitted_at)
            remember_submitted_order(self.state, order_id, normalized)

    def _record_new_trades(self) -> None:
        new_trades = self.broker.trades[self._trade_cursor :]
        self._trade_cursor = len(self.broker.trades)
        for trade in new_trades:
            collapsed = self._collapse_incremental_trade(trade)
            if collapsed is None:
                continue
            self._record_trade_row(collapsed)

    def _collapse_incremental_trade(self, trade: TradeOutcome) -> TradeOutcome | None:
        key = _trade_leg_key(trade)
        if _is_partial_trade(trade):
            self._pending_trade_legs.setdefault(key, []).append(trade)
            return None
        pending = self._pending_trade_legs.pop(key, [])
        if not pending:
            return trade
        return _collapse_trade_legs([*pending, trade])

    def _record_trade_row(self, trade: TradeOutcome) -> None:
        route = dict(trade.route_metadata)
        r_value = _trade_net_r(trade)
        risk = max(float(route.get("risk_per_share") or 0.0), 1e-9)
        active_at_signal = bool(route.get("frontier_shadow_active_at_signal", False))
        row = {
            "symbol": trade.symbol,
            "entry_time": trade.entry_fill_time.isoformat(),
            "entry_date": trade.entry_fill_time.date().isoformat(),
            "exit_time": trade.exit_fill_time.isoformat() if trade.exit_fill_time else "",
            "exit_reason": trade.exit_reason,
            "r": r_value,
            "mfe_r": max(float(getattr(trade, "mfe", 0.0) or 0.0), 0.0) / risk,
            "mae_r": min(float(getattr(trade, "mae", 0.0) or 0.0), 0.0) / risk,
            "giveback_r": max(float(getattr(trade, "mfe", 0.0) or 0.0), 0.0) / risk - r_value,
            "active_at_signal": active_at_signal,
            "frontier_rank": int(route.get("frontier_rank") or 0),
            "candidate_rank": int(route.get("candidate_rank") or 0),
            "frontier_role": str(route.get("frontier_role") or ""),
            "sector": str(route.get("sector") or "UNKNOWN"),
            "entry_route": str(route.get("entry_route") or "legacy"),
            "entry_route_mode": str(route.get("entry_route_mode") or route.get("entry_type") or "UNKNOWN"),
            "entry_type": str(route.get("entry_type") or "UNKNOWN"),
            "momentum_score": int(route.get("momentum_score") or 0),
            "bar_rvol": float(route.get("bar_rvol") or 0.0),
            "first30_ret": _optional_float(route.get("first30_ret")),
            "first30_vwap_ret": _optional_float(route.get("first30_vwap_ret")),
            "first30_rel_volume": _optional_float(route.get("first30_rel_volume")),
            "first30_signal_bar_cpr": _optional_float(route.get("first30_signal_bar_cpr", route.get("first30_close_location"))),
            "first30_range_atr": _optional_float(route.get("first30_range_atr")),
        }
        for key in _FRONTIER_SHADOW_CONTEXT_EXPORT_KEYS:
            if key in row:
                continue
            value = route.get(key)
            if value in (None, ""):
                continue
            row[key] = _optional_float(value) if isinstance(value, (int, float)) else value
        self.trade_rows.append(row)
        if active_at_signal:
            return
        stats = self.symbol_stats.setdefault(trade.symbol, {"trades": 0.0, "total_r": 0.0})
        stats["trades"] += 1.0
        stats["total_r"] += r_value


def run_kalcb_backtest(
    config: dict[str, Any] | None = None,
    mutations: dict[str, Any] | None = None,
    replay_bundle: EventReplayBundle | None = None,
) -> StrategyBacktestResult:
    raw_config = dict(config or {})
    raw_mutations = dict(mutations or {})
    cfg = KALCBConfig.from_mapping(raw_config, raw_mutations)
    capability_level = str(raw_config.get("capability_level", "real_replay")).lower()
    if capability_level in {"official", "feature_complete"}:
        official_available = set(
            raw_config.get(
                "available_features",
                (replay_bundle.metadata or {}).get("available_features", ()) if replay_bundle is not None else (),
            )
        )
        require_capabilities("KALCB", capability_level, official_available, KALCB_OFFICIAL_REQUIREMENTS)
    if replay_bundle is None:
        if capability_level == "synthetic":
            replay_bundle = make_synthetic_replay_bundle("kalcb", raw_config)
        elif capability_level in {"real", "real_replay", "parquet", "krx_replay"}:
            replay_bundle = load_kalcb_real_replay_bundle(raw_config, raw_mutations)
        else:
            raise ValueError("KALCB feature-complete and official replays require an explicit replay_bundle")

    bundle_available = set((replay_bundle.metadata or {}).get("available_features", ()))
    default_available = {"synthetic", "completed_5m_signal_bars"} if capability_level == "synthetic" else bundle_available or {"ohlcv"}
    available = set(raw_config.get("available_features", default_available))
    if capability_level not in {"official", "feature_complete"}:
        require_capabilities("KALCB", capability_level, available, KALCB_OFFICIAL_REQUIREMENTS)

    bars = [event.bar for event in replay_bundle.events if event.bar is not None]
    snapshots = snapshots_from_bundle(replay_bundle)
    if not snapshots and capability_level == "synthetic":
        synthetic_snapshot = snapshot_from_bundle(replay_bundle) or _synthetic_snapshot(bars, replay_bundle.source_fingerprint)
        snapshots = {synthetic_snapshot.trade_date: synthetic_snapshot}
    if not snapshots:
        raise ValueError("KALCB real replay requires source-fingerprinted daily candidate snapshots")
    representative_snapshot = next(iter(snapshots.values()))
    if capability_level != "synthetic":
        expected_candidate_config_hash = _candidate_config_hash(cfg, raw_mutations, _resolve_sector_map(raw_config))
        bundle_candidate_config_hash = (replay_bundle.metadata or {}).get("candidate_config_hash")
        if bundle_candidate_config_hash and bundle_candidate_config_hash != expected_candidate_config_hash:
            raise ValueError("KALCB replay bundle candidate config hash mismatch")
        require_kalcb_feature_metadata(replay_bundle, representative_snapshot)

    costs = BrokerCosts(commission_bps=cfg.commission_bps, tax_bps_on_sell=cfg.tax_bps_on_sell, slippage_bps=cfg.slippage_bps)
    initial_equity = float(raw_config.get("initial_equity", 100_000_000.0))
    adapter = KALCBReplayAdapter(cfg, snapshots, initial_equity=initial_equity, costs=costs)
    result = run_replay(
        bars,
        adapter,
        initial_equity=initial_equity,
        costs=costs,
        close_open_positions=(capability_level == "synthetic"),
        buying_power_leverage=cfg.intraday_leverage,
    )
    result.decisions.extend(adapter._sync_new_fills(result.broker))
    adapter.finalize_frontier_shadow(bars[-1] if bars else None)
    result.trades = _collapse_exit_legs(result.trades)
    metrics = compute_trade_metrics(result.trades, result.equity_curve, initial_equity=initial_equity)
    final_equity = float(result.equity_curve[-1]) if result.equity_curve else float(initial_equity)
    metrics["official_mtm_net_return_pct"] = (final_equity / float(initial_equity) - 1.0) if initial_equity else 0.0
    metrics["final_equity"] = final_equity
    metrics["end_open_position_count"] = float(len(result.broker.positions))
    metrics["net_return_pct_basis"] = "closed_trade_net_pnl_over_initial_equity"
    metrics["official_metric_basis"] = "SimBroker.equity_curve_bar_level_mtm"
    _apply_net_r_metrics(metrics, result.trades)
    metrics["decision_count"] = float(len(result.decisions))
    metrics["same_bar_fill_count"] = float(result.broker.same_bar_fill_violations)
    metrics["rejected_order_count"] = float(len(result.broker.rejected_orders))
    metrics["forced_replay_close_count"] = float(sum(1 for trade in result.trades if trade.exit_reason == "end_of_replay"))
    metrics["strategy_core_version"] = KALCB_CORE_VERSION
    metrics["shared_decision_core"] = "live_shared_core"
    metrics["live_parity_fill_timing"] = cfg.live_parity_fill_timing
    metrics["auction_mode"] = cfg.auction_mode
    metrics["replay_mode"] = str((replay_bundle.metadata or {}).get("replay_mode") or capability_level)
    metrics["replay_event_count"] = float(len(replay_bundle.events))
    metrics["candidate_snapshot_count"] = float(len(snapshots))
    metrics["entry_count"] = float(sum(1 for decision in result.decisions if decision.decision_code == "entry"))
    metrics["entry_rejection_count"] = float(sum(1 for decision in result.decisions if decision.decision_code == "entry_rejected"))
    active_counts = [
        int(snapshot.metadata.get("active_symbol_count") or min(len(snapshot.candidates), cfg.ws_budget))
        for snapshot in snapshots.values()
    ]
    frontier_counts = [len(snapshot.candidates) for snapshot in snapshots.values()]
    candidate_pool_counts = [
        int(snapshot.metadata.get("candidate_pool_count") or len(snapshot.candidates))
        for snapshot in snapshots.values()
    ]
    metrics["active_symbol_max"] = float(max(active_counts))
    metrics["frontier_symbol_max"] = float(max(frontier_counts))
    metrics["candidate_pool_max"] = float(max(candidate_pool_counts))
    universe_size = float((replay_bundle.metadata or {}).get("universe_size") or 0.0)
    metrics["universe_size"] = universe_size
    data_available_symbol_count = float((replay_bundle.metadata or {}).get("data_available_symbol_count") or universe_size)
    metrics["data_available_symbol_count"] = data_available_symbol_count
    metrics["unavailable_symbol_count"] = float((replay_bundle.metadata or {}).get("unavailable_symbol_count") or 0.0)
    metrics["unavailable_symbols"] = list((replay_bundle.metadata or {}).get("unavailable_symbols") or [])
    metrics["selected_universe_fraction"] = (
        float(metrics["active_symbol_max"]) / universe_size if universe_size > 0 else 0.0
    )
    metrics["candidate_pool_universe_fraction"] = (
        float(metrics["candidate_pool_max"]) / universe_size if universe_size > 0 else 0.0
    )
    metrics["candidate_pool_data_available_fraction"] = (
        float(metrics["candidate_pool_max"]) / data_available_symbol_count if data_available_symbol_count > 0 else 0.0
    )
    metrics["frontier_universe_fraction"] = (
        float(metrics["frontier_symbol_max"]) / universe_size if universe_size > 0 else 0.0
    )
    metrics["frontier_data_available_fraction"] = (
        float(metrics["frontier_symbol_max"]) / data_available_symbol_count if data_available_symbol_count > 0 else 0.0
    )
    metrics["frontier_enabled"] = cfg.frontier_enabled
    metrics["frontier_size"] = cfg.frontier_size
    metrics["frontier_selection_mode"] = cfg.frontier_selection_mode
    metrics["frontier_active_selection_mode"] = cfg.frontier_active_selection_mode
    metrics.update(adapter.frontier_metrics())
    feature_hash = str((replay_bundle.metadata or {}).get("kalcb_feature_bundle_hash") or "") or build_feature_bundle_hash_for_snapshots(snapshots, replay_bundle.source_fingerprint)
    candidate_snapshot_hash = str((replay_bundle.metadata or {}).get("kalcb_candidate_artifact_hash") or "") or _aggregate_snapshot_hash(snapshots)
    return StrategyBacktestResult(
        "kalcb",
        metrics,
        result,
        replay_bundle.source_fingerprint,
        capability_level,
        candidate_snapshot_hash,
        feature_hash,
    )


def _synthetic_snapshot(bars: list[MarketBar], source_fingerprint: str) -> KALCBDailySnapshot:
    if not bars:
        raise ValueError("Cannot build KALCB synthetic snapshot without bars")
    trade_date = min(bar.timestamp.date() for bar in bars)
    symbols = sorted({bar.symbol for bar in bars})
    candidates: list[KALCBDailyCandidate] = []
    for index, symbol in enumerate(symbols, start=1):
        first = next(bar for bar in bars if bar.symbol == symbol)
        prior_close = first.open * 0.985
        candidates.append(
            KALCBDailyCandidate(
                symbol=symbol,
                trade_date=trade_date,
                prior_day_high=first.open * 1.004,
                prior_day_low=first.open * 0.970,
                prior_day_close=prior_close,
                daily_atr=max(first.open * 0.012, 1.0),
                expected_5m_volume=13_000.0,
                average_30m_volume=60_000.0,
                sector="SYNTH",
                regime_tier="A",
                selection_score=100.0 - index,
                tradable=True,
                source_fingerprint=source_fingerprint,
            )
        )
    return KALCBDailySnapshot(
        trade_date=trade_date,
        candidates=tuple(candidates),
        source_fingerprint=source_fingerprint,
        generated_at=datetime.combine(trade_date, datetime.min.time()),
        metadata={"synthetic": True, "core_version": KALCB_CORE_VERSION, "ws_budget": len(candidates)},
    )


def _aggregate_snapshot_hash(snapshots: dict[date, KALCBDailySnapshot]) -> str:
    if len(snapshots) == 1:
        return next(iter(snapshots.values())).artifact_hash
    return stable_signature({day.isoformat(): snapshot.artifact_hash for day, snapshot in sorted(snapshots.items())})


def _portfolio_view_for_broker(broker: SimBroker, strategy_id: str) -> KALCBPortfolioView:
    positions: dict[str, int] = {}
    sectors: dict[str, int] = {}
    open_risk = 0.0
    open_notional = 0.0
    for (pos_strategy_id, symbol), position in broker.positions.items():
        if pos_strategy_id != strategy_id:
            continue
        positions[symbol] = position.qty
        sector = str(position.route_metadata.get("sector") or "UNKNOWN")
        sectors[sector] = sectors.get(sector, 0) + 1
        open_risk += float(position.route_metadata.get("risk_per_share", 0.0) or 0.0) * position.qty
        open_notional += position.avg_price * position.qty
    equity = float(broker._portfolio_equity())
    return KALCBPortfolioView(
        cash=broker.cash,
        positions=positions,
        open_positions=len(positions),
        sector_counts=sectors,
        open_risk=open_risk,
        open_notional=open_notional,
        equity=equity,
    )


def _execution_bar_delta(config: KALCBConfig) -> timedelta:
    if str(config.execution_timeframe).lower().startswith("5m"):
        return timedelta(minutes=5)
    return timedelta(minutes=5)


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _trade_leg_key(trade: TradeOutcome) -> tuple[str, str, str, float]:
    return (trade.strategy_id, trade.symbol, trade.entry_fill_time.isoformat(), round(float(trade.entry_price), 6))


def _is_partial_trade(trade: TradeOutcome) -> bool:
    cohort = dict(trade.cohort_metadata)
    return str(trade.exit_reason).lower() == "partial_profit" or str(cohort.get("exit_cohort", "")).lower() == "partial"


def _collapse_exit_legs(trades: list[TradeOutcome]) -> list[TradeOutcome]:
    grouped: dict[tuple[str, str, str, float], list[TradeOutcome]] = {}
    order: list[tuple[str, str, str, float]] = []
    singles: list[TradeOutcome] = []
    for trade in trades:
        key = _trade_leg_key(trade)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(trade)
    for key in order:
        legs = grouped[key]
        if len(legs) == 1:
            singles.append(legs[0])
        else:
            singles.append(_collapse_trade_legs(legs))
    return singles


def _collapse_trade_legs(legs: list[TradeOutcome]) -> TradeOutcome:
    ordered = sorted(legs, key=lambda item: (item.exit_fill_time or item.entry_fill_time, 0 if _is_partial_trade(item) else 1))
    first = ordered[0]
    final = ordered[-1]
    total_qty = sum(int(item.qty) for item in ordered)
    if total_qty <= 0:
        return final
    weighted_exit = sum(float(item.exit_price or item.entry_price) * int(item.qty) for item in ordered) / total_qty
    partial_count = sum(1 for item in ordered if _is_partial_trade(item))
    route = dict(first.route_metadata)
    route.update(
        {
            "partial_taken": partial_count > 0,
            "partial_exit_count": partial_count,
            "exit_leg_count": len(ordered),
            "exit_legs": [_trade_leg_metadata(item) for item in ordered],
            "final_exit_reason": final.exit_reason,
        }
    )
    cohort = dict(final.cohort_metadata)
    cohort.update({"partial_taken": partial_count > 0, "partial_exit_count": partial_count, "exit_leg_count": len(ordered)})
    return TradeOutcome(
        strategy_id=first.strategy_id,
        symbol=first.symbol,
        qty=total_qty,
        entry_decision_time=first.entry_decision_time,
        entry_fill_time=first.entry_fill_time,
        entry_price=first.entry_price,
        exit_fill_time=final.exit_fill_time,
        exit_price=weighted_exit,
        gross_pnl=sum(float(item.gross_pnl) for item in ordered),
        commission=sum(float(item.commission) for item in ordered),
        net_pnl=sum(float(item.net_pnl) for item in ordered),
        realized=all(bool(item.realized) for item in ordered),
        exit_reason=final.exit_reason,
        route_metadata=route,
        cohort_metadata=cohort,
        source_artifact_hash=first.source_artifact_hash,
        mfe=max(float(item.mfe) for item in ordered),
        mae=min(float(item.mae) for item in ordered),
    )


def _trade_leg_metadata(trade: TradeOutcome) -> dict[str, Any]:
    return {
        "qty": int(trade.qty),
        "exit_time": trade.exit_fill_time.isoformat() if trade.exit_fill_time else "",
        "exit_price": float(trade.exit_price or trade.entry_price),
        "exit_reason": trade.exit_reason,
        "net_r": _trade_net_r(trade),
    }


def _trade_net_r(trade: Any) -> float:
    risk_per_share = float(trade.route_metadata.get("risk_per_share", 0.0) or 0.0)
    risk_notional = risk_per_share * int(trade.qty)
    if risk_notional <= 0:
        return float(getattr(trade, "r_multiple", 0.0) or 0.0)
    return float(trade.net_pnl) / risk_notional


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _apply_net_r_metrics(metrics: dict[str, Any], trades: list[Any]) -> None:
    gross_total_r = float(metrics.get("expected_total_r", 0.0))
    gross_avg_r = float(metrics.get("avg_r", 0.0))
    net_r_values: list[float] = []
    for trade in trades:
        risk_per_share = float(trade.route_metadata.get("risk_per_share", 0.0) or 0.0)
        risk_notional = risk_per_share * int(trade.qty)
        if risk_notional <= 0:
            continue
        net_r_values.append(float(trade.net_pnl) / risk_notional)
    metrics["gross_expected_total_r"] = gross_total_r
    metrics["gross_avg_r"] = gross_avg_r
    metrics["expected_total_r"] = float(sum(net_r_values))
    metrics["avg_r"] = float(sum(net_r_values) / len(net_r_values)) if net_r_values else 0.0
