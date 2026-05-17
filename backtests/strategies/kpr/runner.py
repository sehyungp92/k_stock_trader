from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Any

from backtests.analysis.metrics import compute_trade_metrics
from backtests.core.replay_bundle import EventReplayBundle
from backtests.engine.replay import ReplayResult, run_replay
from backtests.engine.sim_broker import SimBroker
from backtests.strategies.common.capabilities import KPR_OFFICIAL_REQUIREMENTS, require_capabilities
from backtests.strategies.common.synthetic import make_synthetic_replay_bundle
from strategy_common.actions import SubmitEntry, SubmitExit
from strategy_common.events import DecisionEvent
from strategy_common.market import MarketBar
from strategy_kpr.core.exits import check_exits
from strategy_kpr.core.fsm import compute_confidence
from strategy_kpr.core.setup_detection import detect_setup
from strategy_kpr.core.state import FSMState, SymbolState
from strategy_kpr.signals.investor import InvestorSignal
from strategy_kpr.signals.micro import MicroSignal
from strategy_kpr.signals.program import ProgramSignal


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


class KPRReplayAdapter:
    strategy_id = "KPR"

    def __init__(self, mutations: dict[str, Any] | None = None):
        self.mutations = mutations or {}
        self.states: dict[str, SymbolState] = {}
        self.vwap_volume: dict[str, float] = {}
        self.vwap_value: dict[str, float] = {}

    def on_bar(self, bar: MarketBar, broker: SimBroker) -> list[DecisionEvent]:
        state = self.states.setdefault(bar.symbol, SymbolState(code=bar.symbol))
        self._sync_fill_state(state, broker)
        vwap = self._update_vwap(bar)
        bar_dict = bar.to_json_dict()
        bar_dict["timestamp"] = bar.timestamp
        decisions: list[DecisionEvent] = []

        if state.fsm == FSMState.IN_POSITION:
            should_exit, reason, qty = check_exits(
                state,
                bar.close,
                bar.timestamp,
                InvestorSignal.STRONG,
                MicroSignal.ACCUMULATE,
            )
            if should_exit and qty > 0:
                action = SubmitExit(self.strategy_id, bar.symbol, qty, "MARKET", None, reason)
                broker.submit(action, bar.timestamp)
                decisions.append(DecisionEvent(bar.timestamp, self.strategy_id, bar.symbol, "exit", reason, actions=(action,)))
            return decisions

        action, reason = self._maybe_enter(state, bar, bar_dict, vwap)
        if action:
            broker.submit(action, bar.timestamp)
            decisions.append(
                DecisionEvent(
                    bar.timestamp,
                    self.strategy_id,
                    bar.symbol,
                    "entry",
                    reason,
                    actions=(action,),
                    metadata={"fsm": state.fsm.name, "setup_type": state.setup_type or "", "vwap": vwap},
                )
            )
        return decisions

    def _update_vwap(self, bar: MarketBar) -> float:
        self.vwap_volume[bar.symbol] = self.vwap_volume.get(bar.symbol, 0.0) + bar.volume
        self.vwap_value[bar.symbol] = self.vwap_value.get(bar.symbol, 0.0) + bar.close * bar.volume
        return self.vwap_value[bar.symbol] / max(self.vwap_volume[bar.symbol], 1.0)

    def _maybe_enter(self, state: SymbolState, bar: MarketBar, bar_dict: dict, vwap: float) -> tuple[SubmitEntry | None, str]:
        if bar.timestamp.time() < time(9, 5) or bar.timestamp.time() > time(14, 45):
            return None, ""
        if bar.high > state.hod:
            state.hod = bar.high
            state.hod_time = bar.timestamp
        state.lod = min(state.lod, bar.low)
        state.vwap = vwap

        if state.fsm == FSMState.IDLE:
            if detect_setup(state, bar_dict, vwap, bar.timestamp):
                state.fsm = FSMState.SETUP_DETECTED
            return None, ""
        if state.fsm == FSMState.SETUP_DETECTED and state.reclaim_level and bar.high >= state.reclaim_level:
            state.fsm = FSMState.ACCEPTING
            state.required_closes = int(self.mutations.get("required_closes", 1))
            return None, ""
        if state.fsm == FSMState.ACCEPTING and state.reclaim_level:
            if bar.close >= state.reclaim_level:
                state.accept_closes += 1
            if state.accept_closes < state.required_closes:
                return None, ""
            investor = InvestorSignal.STRONG
            micro = MicroSignal.ACCUMULATE
            program = ProgramSignal.UNAVAILABLE
            confidence = compute_confidence(investor, micro, program, False, symbol=state.code)
            if confidence == "RED":
                state.fsm = FSMState.INVALIDATED
                return None, "confidence_red"
            stop = state.stop_level or bar.close * 0.98
            qty = int(self.mutations.get("fixed_qty", 10))
            risk_per_share = max(bar.close - stop, 1.0)
            state.fsm = FSMState.PENDING_ENTRY
            return (
                SubmitEntry(
                    self.strategy_id,
                    state.code,
                    max(1, qty),
                    "MARKET",
                    None,
                    stop,
                    "panic_reclaim" if state.setup_type == "panic" else "drift_reclaim",
                    {"risk_per_share": risk_per_share, "confidence": confidence},
                ),
                "reclaim_acceptance",
            )
        return None, ""

    def _sync_fill_state(self, state: SymbolState, broker: SimBroker) -> None:
        position = broker.positions.get((self.strategy_id, state.code))
        if position and state.fsm != FSMState.IN_POSITION:
            state.fsm = FSMState.IN_POSITION
            state.entry_px = position.avg_price
            state.entry_ts = position.entry_fill_time
            state.qty = position.qty
            state.remaining_qty = position.qty
            state.max_price = position.avg_price
            if state.stop_level is None:
                state.stop_level = position.avg_price * 0.98
        elif position and state.fsm == FSMState.IN_POSITION:
            if position.qty < state.qty:
                state.partial_filled = True
            state.remaining_qty = position.qty
        elif not position and state.fsm == FSMState.IN_POSITION:
            state.fsm = FSMState.DONE


def run_kpr_backtest(
    config: dict[str, Any] | None = None,
    mutations: dict[str, Any] | None = None,
    replay_bundle: EventReplayBundle | None = None,
) -> StrategyBacktestResult:
    config = dict(config or {})
    mutations = dict(mutations or {})
    capability_level = str(config.get("capability_level", "synthetic"))
    available = set(config.get("available_features", ["synthetic", "ohlcv"] if capability_level == "synthetic" else ["ohlcv"]))
    require_capabilities("KPR", capability_level, available, KPR_OFFICIAL_REQUIREMENTS)
    if capability_level != "synthetic" and replay_bundle is None:
        raise ValueError("KPR feature-complete and official replays require an explicit replay_bundle")
    if replay_bundle:
        bars = [event.bar for event in replay_bundle.events if event.bar is not None]
        source_fingerprint = replay_bundle.source_fingerprint
    else:
        replay_bundle = make_synthetic_replay_bundle("kpr", config)
        bars = [event.bar for event in replay_bundle.events if event.bar is not None]
        source_fingerprint = replay_bundle.source_fingerprint
    initial_equity = float(config.get("initial_equity", 10_000_000.0))
    result = run_replay(bars, KPRReplayAdapter(mutations), initial_equity=initial_equity)
    metrics = compute_trade_metrics(result.trades, result.equity_curve, initial_equity=initial_equity)
    metrics["decision_count"] = float(len(result.decisions))
    return StrategyBacktestResult("kpr", metrics, result, source_fingerprint, capability_level)
