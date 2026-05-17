from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Any

from backtests.analysis.metrics import compute_trade_metrics
from backtests.core.replay_bundle import EventReplayBundle
from backtests.engine.replay import ReplayResult, run_replay
from backtests.engine.sim_broker import SimBroker
from backtests.strategies.common.capabilities import KMP_OFFICIAL_REQUIREMENTS, require_capabilities
from backtests.strategies.common.synthetic import make_synthetic_replay_bundle
from strategy_common.actions import SubmitEntry, SubmitExit
from strategy_common.events import DecisionEvent
from strategy_common.market import MarketBar
from strategy_kmp.core.fsm import is_accepted
from strategy_kmp.core.gates import lock_or_and_filter, min_surge_threshold, minutes_since_0916, rvol_ok, spread_ok
from strategy_kmp.core.state import State, SymbolState
from strategy_kmp.core.tick_table import tick_size


@dataclass(slots=True)
class StrategyBacktestResult:
    strategy: str
    metrics: dict[str, float]
    replay_result: ReplayResult
    source_fingerprint: str
    capability_level: str

    @property
    def trades(self):
        return self.replay_result.trades

    @property
    def decisions(self):
        return self.replay_result.decisions


class KMPReplayAdapter:
    strategy_id = "KMP"

    def __init__(self, mutations: dict[str, Any] | None = None):
        self.mutations = mutations or {}
        self.states: dict[str, SymbolState] = {}

    def on_bar(self, bar: MarketBar, broker: SimBroker) -> list[DecisionEvent]:
        state = self.states.setdefault(bar.symbol, SymbolState(code=bar.symbol))
        self._sync_fill_state(state, broker)
        self._update_features(state, bar)
        decisions: list[DecisionEvent] = []

        if state.fsm == State.IN_POSITION:
            action = self._maybe_exit(state, bar, broker)
            if action:
                broker.submit(action, bar.timestamp)
                decisions.append(
                    DecisionEvent(
                        timestamp=bar.timestamp,
                        strategy_id=self.strategy_id,
                        symbol=bar.symbol,
                        decision_code="exit",
                        reason=action.reason,
                        actions=(action,),
                        metadata={"fsm": state.fsm.name},
                    )
                )
            return decisions

        action, reason = self._maybe_enter(state, bar)
        if action:
            broker.submit(action, bar.timestamp)
            decisions.append(
                DecisionEvent(
                    timestamp=bar.timestamp,
                    strategy_id=self.strategy_id,
                    symbol=bar.symbol,
                    decision_code="entry",
                    reason=reason,
                    actions=(action,),
                    metadata={
                        "fsm": state.fsm.name,
                        "surge": state.surge,
                        "or_high": state.or_high,
                        "vwap": state.vwap,
                    },
                )
            )
        return decisions

    def _update_features(self, state: SymbolState, bar: MarketBar) -> None:
        if bar.timestamp.time() < time(9, 15):
            state.or_high = max(state.or_high, bar.high)
            state.or_low = min(state.or_low, bar.low)
            state.value15 += bar.close * bar.volume
        state.update_vwap(bar.close, bar.volume)
        state.curr_1m_vol = bar.volume
        state.avg_1m_vol = max(1.0, (state.avg_1m_vol * 0.9) + (bar.volume * 0.1)) if state.avg_1m_vol else max(1.0, bar.volume / 2.0)
        state.rvol_1m = state.curr_1m_vol / max(state.avg_1m_vol, 1.0)
        spread_bps = float(self.mutations.get("spread_bps", 8.0))
        state.bid = bar.close * (1.0 - spread_bps / 20_000.0)
        state.ask = bar.close * (1.0 + spread_bps / 20_000.0)
        state.update_spread()
        state.surge = max(state.surge, (bar.close / max(state.prev_close or bar.open, 1.0) - 1.0) * 100.0)
        if state.prev_close <= 0:
            state.prev_close = bar.open

    def _maybe_enter(self, state: SymbolState, bar: MarketBar) -> tuple[SubmitEntry | None, str]:
        if (not state.or_locked) and bar.timestamp.time() >= time(9, 15):
            if not lock_or_and_filter(state):
                state.fsm = State.DONE
                return None, "or_lock_fail"
            state.fsm = State.WATCH_BREAK
        if state.fsm in {State.DONE, State.IDLE} or not state.or_locked or bar.timestamp.time() < time(9, 16):
            return None, ""
        if bar.timestamp.time() >= time(14, 30):
            state.fsm = State.DONE
            return None, "entry_cutoff"
        minutes = minutes_since_0916(bar.timestamp)
        surge_threshold = float(self.mutations.get("min_surge_base", min_surge_threshold(minutes)))
        if state.surge < surge_threshold:
            return None, "surge_decay"
        if not spread_ok(state) or (self.mutations.get("enable_rvol_hard_gate") and not rvol_ok(state)):
            return None, "spread_or_rvol"
        tick = tick_size(bar.close)
        if state.fsm == State.WATCH_BREAK:
            if bar.close > state.or_high + tick and bar.close > state.vwap:
                state.break_ts = bar.timestamp.timestamp()
                state.retest_low = bar.close
                state.fsm = State.WAIT_ACCEPTANCE
            return None, ""
        if state.fsm == State.WAIT_ACCEPTANCE:
            state.retest_low = min(state.retest_low, bar.low)
            timeout_min = float(self.mutations.get("accept_timeout_min", 10.0))
            if bar.timestamp.timestamp() - state.break_ts > timeout_min * 60:
                state.fsm = State.DONE
                return None, "acceptance_timeout"
            if not is_accepted(state, bar.close):
                return None, ""
            entry = bar.close
            stop = min(state.retest_low * 0.997, entry * 0.98)
            risk_per_share = max(entry - stop, 1.0)
            qty = int(float(self.mutations.get("fixed_qty", 10)))
            state.structure_stop = stop
            state.hard_stop = stop * 0.995
            state.fsm = State.ARMED
            return (
                SubmitEntry(
                    strategy_id=self.strategy_id,
                    symbol=state.code,
                    qty=max(1, qty),
                    order_type="MARKET",
                    limit_price=None,
                    stop_price=stop,
                    reason="or_break_acceptance",
                    metadata={"risk_per_share": risk_per_share, "signal_bar": bar.timestamp.isoformat()},
                ),
                "or_break_acceptance",
            )
        return None, ""

    def _sync_fill_state(self, state: SymbolState, broker: SimBroker) -> None:
        position = broker.positions.get((self.strategy_id, state.code))
        if position and state.fsm != State.IN_POSITION:
            state.fsm = State.IN_POSITION
            state.entry_px = position.avg_price
            state.entry_ts = position.entry_fill_time.timestamp()
            state.qty = position.qty
            state.max_fav = position.avg_price
            state.min_adverse = position.avg_price
            if not state.structure_stop:
                state.structure_stop = position.avg_price * 0.98
            if not state.hard_stop:
                state.hard_stop = state.structure_stop
        elif not position and state.fsm == State.IN_POSITION:
            state.fsm = State.DONE

    def _maybe_exit(self, state: SymbolState, bar: MarketBar, broker: SimBroker) -> SubmitExit | None:
        state.max_fav = max(state.max_fav, bar.high)
        state.min_adverse = min(state.min_adverse, bar.low)
        target_mult = float(self.mutations.get("target_pct", 0.03))
        if bar.low <= state.hard_stop:
            reason = "hard_stop"
        elif bar.close >= state.entry_px * (1.0 + target_mult):
            reason = "profit_target"
        elif bar.timestamp.time() >= time(14, 30):
            reason = "eod_flatten"
        else:
            return None
        return SubmitExit(
            strategy_id=self.strategy_id,
            symbol=state.code,
            qty=broker.position_qty(self.strategy_id, state.code),
            order_type="MARKET",
            limit_price=None,
            reason=reason,
        )


def run_kmp_backtest(
    config: dict[str, Any] | None = None,
    mutations: dict[str, Any] | None = None,
    replay_bundle: EventReplayBundle | None = None,
) -> StrategyBacktestResult:
    config = dict(config or {})
    mutations = dict(mutations or {})
    capability_level = str(config.get("capability_level", "synthetic"))
    available = set(config.get("available_features", ["ohlcv"] if capability_level != "synthetic" else ["synthetic", "ohlcv"]))
    require_capabilities("KMP", capability_level, available, KMP_OFFICIAL_REQUIREMENTS)
    if capability_level != "synthetic" and replay_bundle is None:
        raise ValueError("KMP feature-complete and official replays require an explicit replay_bundle")
    if replay_bundle:
        bars = [event.bar for event in replay_bundle.events if event.bar is not None]
        source_fingerprint = replay_bundle.source_fingerprint
    else:
        replay_bundle = make_synthetic_replay_bundle("kmp", config)
        bars = [event.bar for event in replay_bundle.events if event.bar is not None]
        source_fingerprint = replay_bundle.source_fingerprint
    initial_equity = float(config.get("initial_equity", 10_000_000.0))
    result = run_replay(bars, KMPReplayAdapter(mutations), initial_equity=initial_equity)
    metrics = compute_trade_metrics(result.trades, result.equity_curve, initial_equity=initial_equity)
    metrics["decision_count"] = float(len(result.decisions))
    return StrategyBacktestResult("kmp", metrics, result, source_fingerprint, capability_level)
