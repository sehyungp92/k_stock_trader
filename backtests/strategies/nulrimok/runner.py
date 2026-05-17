from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Any

from backtests.analysis.metrics import compute_trade_metrics
from backtests.core.replay_bundle import EventReplayBundle
from backtests.engine.replay import ReplayResult, run_replay
from backtests.engine.sim_broker import SimBroker
from backtests.strategies.common.capabilities import NULRIMOK_OFFICIAL_REQUIREMENTS, require_capabilities
from backtests.strategies.common.synthetic import make_synthetic_replay_bundle
from strategy_common.actions import SubmitEntry, SubmitExit
from strategy_common.events import DecisionEvent
from strategy_common.market import MarketBar
from strategy_nulrimok.dse.artifact import TickerArtifact
from strategy_nulrimok.iepe.entry import EntryState, TickerEntryState, check_confirmation, check_entry_conditions


@dataclass(slots=True)
class StrategyBacktestResult:
    strategy: str
    metrics: dict[str, float]
    replay_result: ReplayResult
    source_fingerprint: str
    capability_level: str
    selection_attribution: dict[str, float]

    @property
    def trades(self):
        return self.replay_result.trades

    @property
    def decisions(self):
        return self.replay_result.decisions


class NulrimokReplayAdapter:
    strategy_id = "NULRIMOK"

    def __init__(self, mutations: dict[str, Any] | None = None):
        self.mutations = mutations or {}
        self.entry_states: dict[str, TickerEntryState] = {}
        self.close_history: dict[str, list[float]] = {}
        self.volume_history: dict[str, list[float]] = {}
        self.artifacts: dict[str, TickerArtifact] = {}

    def on_bar(self, bar: MarketBar, broker: SimBroker) -> list[DecisionEvent]:
        self._ensure_artifact(bar)
        entry_state = self.entry_states.setdefault(bar.symbol, TickerEntryState(ticker=bar.symbol))
        self.close_history.setdefault(bar.symbol, []).append(bar.close)
        self.volume_history.setdefault(bar.symbol, []).append(bar.volume)
        artifact = self.artifacts[bar.symbol]
        decisions: list[DecisionEvent] = []

        if broker.position_qty(self.strategy_id, bar.symbol) > 0:
            action = self._maybe_exit(bar, broker)
            if action:
                broker.submit(action, bar.timestamp)
                decisions.append(DecisionEvent(bar.timestamp, self.strategy_id, bar.symbol, "exit", action.reason, actions=(action,)))
            return decisions

        action, reason = self._maybe_enter(entry_state, artifact, bar)
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
                    metadata={"daily_rank": artifact.daily_rank, "selection_bucket": artifact.setup_type or "synthetic"},
                )
            )
        return decisions

    def _ensure_artifact(self, bar: MarketBar) -> None:
        if bar.symbol in self.artifacts:
            return
        avwap = float(self.mutations.get("avwap_ref", bar.open * 0.995))
        self.artifacts[bar.symbol] = TickerArtifact(
            ticker=bar.symbol,
            regime_tier="A",
            regime_score=0.8,
            flow_score=0.75,
            flow_persistence=0.8,
            flow_pass=True,
            rs_percentile=85.0,
            leader_pass=True,
            trend_pass=True,
            anchor_date=bar.timestamp.date().isoformat(),
            avwap_ref=avwap,
            band_lower=avwap * float(self.mutations.get("band_lower_mult", 0.985)),
            band_upper=avwap * float(self.mutations.get("band_upper_mult", 1.010)),
            acceptance_pass=True,
            avwap_proximity=0.9,
            daily_rank=0.90,
            tradable=True,
            recommended_risk=0.005,
            setup_type="top_rank_synthetic",
            atr30m_est=bar.open * 0.015,
        )

    def _sma5(self, symbol: str) -> float:
        values = self.close_history.get(symbol, [])
        sample = values[-5:]
        return sum(sample) / len(sample) if sample else 0.0

    def _vol_avg(self, symbol: str) -> float:
        values = self.volume_history.get(symbol, [])
        sample = values[-5:]
        return sum(sample) / len(sample) if sample else 1.0

    def _maybe_enter(self, entry_state: TickerEntryState, artifact: TickerArtifact, bar: MarketBar) -> tuple[SubmitEntry | None, str]:
        bar_dict = bar.to_json_dict()
        bar_dict["timestamp"] = bar.timestamp
        sma5 = self._sma5(bar.symbol)
        vol_avg = self._vol_avg(bar.symbol)
        if entry_state.state == EntryState.IDLE:
            if check_entry_conditions(artifact, bar_dict, sma5, vol_avg):
                entry_state.state = EntryState.ARMED
                entry_state.arm_time = bar.timestamp
                entry_state.confirm_bars_remaining = int(self.mutations.get("confirm_bars", 3))
                entry_state.last_30m_low = bar.low
            return None, ""
        if entry_state.state == EntryState.ARMED:
            confirmed, conf_type = check_confirmation(entry_state, artifact, bar_dict)
            if conf_type == "INVALIDATED":
                entry_state.reset()
                return None, "band_invalidation"
            if not confirmed:
                entry_state.confirm_bars_remaining -= 1
                if entry_state.confirm_bars_remaining <= 0:
                    entry_state.reset()
                return None, ""
            stop = min(artifact.avwap_ref * 0.98, bar.close * 0.98)
            qty = int(self.mutations.get("fixed_qty", 10))
            risk_per_share = max(bar.close - stop, 1.0)
            entry_state.state = EntryState.PENDING_FILL
            return (
                SubmitEntry(
                    self.strategy_id,
                    artifact.ticker,
                    max(1, qty),
                    "MARKET",
                    None,
                    stop,
                    f"iepe_{conf_type.lower()}",
                    {"risk_per_share": risk_per_share, "daily_rank": artifact.daily_rank},
                ),
                conf_type.lower(),
            )
        return None, ""

    def _maybe_exit(self, bar: MarketBar, broker: SimBroker) -> SubmitExit | None:
        position = broker.positions.get((self.strategy_id, bar.symbol))
        if not position:
            return None
        target_pct = float(self.mutations.get("target_pct", 0.04))
        stop = position.stop_price or position.avg_price * 0.98
        if bar.low <= stop:
            reason = "setup_stop"
        elif bar.close >= position.avg_price * (1.0 + target_pct):
            reason = "setup_target"
        elif bar.timestamp.time() >= time(15, 0):
            reason = "time_stop_intraday"
        else:
            return None
        return SubmitExit(self.strategy_id, bar.symbol, position.qty, "MARKET", None, reason)


def run_nulrimok_backtest(
    config: dict[str, Any] | None = None,
    mutations: dict[str, Any] | None = None,
    replay_bundle: EventReplayBundle | None = None,
    lrs_bundle: Any | None = None,
) -> StrategyBacktestResult:
    del lrs_bundle
    config = dict(config or {})
    mutations = dict(mutations or {})
    capability_level = str(config.get("capability_level", "synthetic"))
    available = set(config.get("available_features", ["synthetic", "ohlcv"] if capability_level == "synthetic" else ["ohlcv"]))
    require_capabilities("Nulrimok", capability_level, available, NULRIMOK_OFFICIAL_REQUIREMENTS)
    if capability_level != "synthetic" and replay_bundle is None:
        raise ValueError("Nulrimok feature-complete and official replays require an explicit replay_bundle")
    if replay_bundle:
        bars = [event.bar for event in replay_bundle.events if event.bar is not None]
        source_fingerprint = replay_bundle.source_fingerprint
    else:
        replay_bundle = make_synthetic_replay_bundle("nulrimok", config)
        bars = [event.bar for event in replay_bundle.events if event.bar is not None]
        source_fingerprint = replay_bundle.source_fingerprint
    initial_equity = float(config.get("initial_equity", 10_000_000.0))
    result = run_replay(bars, NulrimokReplayAdapter(mutations), initial_equity=initial_equity)
    metrics = compute_trade_metrics(result.trades, result.equity_curve, initial_equity=initial_equity)
    metrics["decision_count"] = float(len(result.decisions))
    selection = {"top_rank_avg_forward_r": metrics.get("expected_total_r", 0.0), "overflow_opportunity_cost": 0.0}
    return StrategyBacktestResult("nulrimok", metrics, result, source_fingerprint, capability_level, selection)
