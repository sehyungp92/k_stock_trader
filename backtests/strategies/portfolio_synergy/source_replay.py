from __future__ import annotations

from kis_core.input_manifest import parquet_input, read_input_bytes
from backtests.core.shared_market_inputs import market_bar_values, selected_intraday_paths

import hashlib
import json
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from backtests.analysis.metrics import compute_trade_metrics
from backtests.auto.shared.cache_keys import fingerprint_paths, stable_signature, source_generation_hash
from backtests.engine.replay import drive_timestamp_batch, ReplayResult, run_replay
from backtests.engine.sim_broker import BrokerCosts, SimBroker, SimOrder
from backtests.engine.contracts import EXECUTION_VERSION, CONTRACT_VERSIONS
from backtests.strategies.kalcb.fixed_trade_plan_phase import _normalize_mutations
from backtests.strategies.kalcb.runner import KALCBReplayAdapter, _collapse_exit_legs
from backtests.strategies.kalcb.trade_plan_sweep import CompiledCoreReplay, _clone_snapshots_for_replay
from backtests.strategies.olr.research_sweep import _frame_rows, _overnight_label_cache
from backtests.strategies.olr.runner import (
    OLRReplayAdapter,
    _aggregate_snapshot_hash,
    attach_overnight_labels_to_snapshots,
)
from strategy_common.market import MarketBar
from strategy_common.daily_lrs_parquet import load_daily_ohlcv
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.models import KALCBDailySnapshot
from strategy_olr.config import OLRConfig
from strategy_olr.models import OLRDailySnapshot
from .native_policy import NativePolicyMixin


@dataclass(frozen=True, slots=True)
class PromotedReplayInputs:
    kalcb_config: KALCBConfig
    kalcb_replay: CompiledCoreReplay
    olr_config: OLRConfig
    olr_snapshots: dict[date, OLRDailySnapshot]
    olr_bars: tuple[MarketBar, ...]
    lineage: dict[str, Any]
    kalcb_timed_snapshots: dict = field(default_factory=dict)
    kalcb_observation_symbols: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReplaySummary:
    initial_equity: float
    final_equity: float
    net_return_pct: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    trades: int
    rejected_orders: int
    open_positions: int
    trade_hash: str
    strategy_trade_counts: dict[str, int]
    strategy_net_pnl: dict[str, float]


def load_promoted_olr_snapshots(
    snapshot_root: Path,
    *,
    daily_data_root: Path,
    training_end: str,
    expected_candidate_snapshot_hash: str,
    expected_candidate_config_hash: str,
    round_generated_at_utc: str,
) -> tuple[dict[date, OLRDailySnapshot], dict[str, Any]]:
    """Recover the exact round-consumed OLR snapshots, including attached labels."""

    paths = sorted(snapshot_root.glob("candidate_snapshot_*.json"))
    if not paths:
        raise FileNotFoundError(f"No OLR candidate snapshots under {snapshot_root}")
    snapshots: dict[date, OLRDailySnapshot] = {}
    raw_artifact_hashes: dict[str, str] = {}
    file_hashes: dict[str, str] = {}
    config_hashes: set[str] = set()
    final_config_hashes: set[str] = set()
    generated_after_round: list[str] = []
    round_generated_at = _parse_datetime(round_generated_at_utc)
    for path in paths:
        payload = json.loads(read_input_bytes(path).decode("utf-8"))
        snapshot = OLRDailySnapshot.from_json_dict(payload)
        snapshots[snapshot.trade_date] = snapshot
        raw_artifact_hashes[snapshot.trade_date.isoformat()] = str(payload.get("artifact_hash") or snapshot.artifact_hash)
        file_hashes[path.name] = _file_hash(path)
        config_hashes.add(str(snapshot.metadata.get("candidate_config_hash") or ""))
        final_config_hashes.add(str(snapshot.metadata.get("final_candidate_config_hash") or ""))
        if round_generated_at is not None:
            snapshot_generated_at = snapshot.generated_at
            if snapshot_generated_at.tzinfo is None:
                snapshot_generated_at = snapshot_generated_at.replace(tzinfo=round_generated_at.tzinfo)
            if snapshot_generated_at.astimezone(round_generated_at.tzinfo) > round_generated_at:
                generated_after_round.append(snapshot.trade_date.isoformat())

    retained_final_config_hashes = {value for value in final_config_hashes if value}
    if config_hashes != {expected_candidate_config_hash} or (
        retained_final_config_hashes
        and retained_final_config_hashes != {expected_candidate_config_hash}
    ):
        raise ValueError(
            "OLR retained snapshot config identity mismatch: "
            f"candidate={sorted(config_hashes)}, final={sorted(final_config_hashes)}, "
            f"expected={expected_candidate_config_hash}"
        )
    if generated_after_round:
        raise ValueError(
            "OLR retained snapshots post-date the promoted round artifact: "
            f"{len(generated_after_round)} snapshots, first={generated_after_round[0]}"
        )
    for day, snapshot in snapshots.items():
        ranks = [int(candidate.rank) for candidate in snapshot.candidates]
        if ranks != sorted(ranks) or len(ranks) != len(set(ranks)) or any(rank <= 0 for rank in ranks):
            raise ValueError(f"OLR retained candidate ordering is invalid on {day}: {ranks}")

    symbols = tuple(sorted({candidate.symbol for snapshot in snapshots.values() for candidate in snapshot.candidates}))
    training_end_date = date.fromisoformat(str(training_end)[:10])
    daily_by_symbol = {
        symbol: _frame_rows(load_daily_ohlcv(daily_data_root, symbol, end=training_end_date))
        for symbol in symbols
    }
    labels = _overnight_label_cache(daily_by_symbol, symbols, tuple(sorted(snapshots)))
    consumed = attach_overnight_labels_to_snapshots(snapshots, labels)
    consumed_hash = _aggregate_snapshot_hash(consumed)
    if consumed_hash != expected_candidate_snapshot_hash:
        raise ValueError(
            "OLR promoted consumed-content identity mismatch: "
            f"expected {expected_candidate_snapshot_hash}, reconstructed {consumed_hash}"
        )
    nonempty_count = sum(bool(snapshot.candidates) for snapshot in consumed.values())
    return consumed, {
        "candidate_snapshot_root": str(snapshot_root),
        "stored_snapshot_count": len(snapshots),
        "nonempty_snapshot_count": nonempty_count,
        "empty_snapshot_count": len(snapshots) - nonempty_count,
        "candidate_count": sum(len(snapshot.candidates) for snapshot in consumed.values()),
        "candidate_symbol_count": len(symbols),
        "candidate_config_hash": expected_candidate_config_hash,
        "raw_candidate_snapshot_hash": stable_signature(raw_artifact_hashes),
        "consumed_candidate_snapshot_hash": consumed_hash,
        "expected_candidate_snapshot_hash": expected_candidate_snapshot_hash,
        "snapshot_file_manifest_hash": stable_signature(file_hashes),
        "overnight_label_count": len(labels),
        "round_generated_at_utc": round_generated_at_utc,
        "snapshots_generated_after_round": 0,
        "candidate_ordering": "strict_ascending_unique_rank",
        "identity_status": "exact_promoted_consumed_content_match",
        "strategy_decay_attribution_allowed": True,
    }


class NativeRiskSharedBroker(NativePolicyMixin, SimBroker):
    """One cash account that preserves each strategy's native cost/leverage contract.

    The old portfolio replay imposed a second, much tighter sizing model on top
    of the strategy cores.  This broker only enforces actual shared buying
    power: each strategy keeps its standalone leverage ceiling and the combined
    book cannot exceed the highest account leverage ceiling.
    """

    def __init__(
        self,
        initial_equity: float,
        *,
        strategy_costs: Mapping[str, BrokerCosts],
        strategy_leverage: Mapping[str, float],
        execution_clock: str = EXECUTION_VERSION,
        capacity_policy=None,
        portfolio_policy=None,
        sectors=None,
        quality_model=None,
        decision_rank=False,
        sizing_equity_cap=0.,
    ) -> None:
        normalized_leverage = {
            str(strategy).upper(): max(float(leverage), 1.0)
            for strategy, leverage in strategy_leverage.items()
        }
        if not normalized_leverage:
            raise ValueError("strategy_leverage is required")
        normalized_costs = {str(strategy).upper(): costs for strategy, costs in strategy_costs.items()}
        missing_costs = sorted(set(normalized_leverage) - set(normalized_costs))
        if missing_costs:
            raise ValueError(f"missing strategy costs for: {', '.join(missing_costs)}")
        default_strategy = sorted(normalized_costs)[0]
        super().__init__(
            initial_equity,
            costs=normalized_costs[default_strategy],
            buying_power_leverage=max(normalized_leverage.values()),
            execution_clock=execution_clock,
            capacity_policy=capacity_policy,
        )
        self.strategy_costs = normalized_costs
        self.strategy_leverage = normalized_leverage
        self.capacity_blocks: list[dict[str, Any]] = []
        self.configure_portfolio_policy(portfolio_policy, sectors)
        self.quality_model = quality_model
        self.decision_rank = bool(decision_rank)
        self.sizing_equity_cap = float(sizing_equity_cap)

    def strategy_sizing_equity(self):
        equity = float(self._portfolio_equity())
        return min(equity, self.sizing_equity_cap) if self.sizing_equity_cap > 0 else equity

    def strategy_risk_cash(self, strategy_id):
        """Standalone-equivalent sleeve risk base; execution uses actual funding.

        KALCB's native core derives risk and buying power from cash minus its
        own positions. Another sleeve's holdings must not be subtracted twice.
        The broker still limits every fill against the entire shared account.
        """
        if self.portfolio_policy and self.portfolio_policy.config.funding_mode == "native_buying_power":
            return self.strategy_sizing_equity() - self._strategy_open_notional(strategy_id)
        return self.cash

    def order_strategy_decisions(self, adapter, bars):
        """Rank known OLR candidates before its core reserves pending cash."""
        if not self.decision_rank or adapter.strategy_id != "OLR":
            return bars
        from datetime import time, timedelta
        from strategy_common.execution_session import session_cutoff
        day = bars[0].timestamp.date()
        if bars[0].timestamp.time().replace(tzinfo=None) < session_cutoff(day, time(14, 30), anchor="open"):
            return bars
        snapshot = adapter.snapshots.get(day)
        if snapshot is None:
            return bars
        candidates = {c.symbol: c for c in snapshot.candidates}
        at = bars[0].timestamp + timedelta(minutes=5)
        def key(bar):
            candidate = candidates.get(bar.symbol)
            if candidate is None:
                return (1, 0., 0, bar.symbol)
            score = self.quality_model.score("OLR", candidate.metadata, at) if self.quality_model else None
            return (0, -score if score is not None else 0., candidate.rank, bar.symbol)
        return sorted(bars, key=key)

    def strategy_buying_power(self, strategy_id: str) -> float:
        """Current marked capacity only; pending exits never supply funding."""
        equity = max(float(self._portfolio_equity()), 0.0)
        return max(0.0, min(
            equity * self.strategy_leverage[strategy_id] - self._strategy_open_notional(strategy_id),
            equity * self.buying_power_leverage - self._open_notional(),
        ))

    def strategy_account_view(self, strategy_id: str) -> tuple[float, float]:
        policy = self.portfolio_policy
        funding = (self.strategy_buying_power(strategy_id)
                   if policy and policy.config.funding_mode == "native_buying_power" else self.cash)
        equity = self.strategy_sizing_equity()
        if self.sizing_equity_cap > 0:
            funding = min(funding, max(equity * self.strategy_leverage[strategy_id] - self._strategy_open_notional(strategy_id), 0))
        return equity, float(funding)

    def _fill_price(self, order: SimOrder, bar: MarketBar) -> float | None:
        return self._with_strategy_costs(order.strategy_id, super()._fill_price, order, bar)

    def _apply_fill(self, order: SimOrder, price: float, timestamp: datetime):
        return self._with_strategy_costs(order.strategy_id, super()._apply_fill, order, price, timestamp)

    def _can_afford(self, order: SimOrder, price: float) -> bool:
        strategy_id = str(order.strategy_id).upper()
        costs = self.strategy_costs[strategy_id]
        qty = int(order.qty or 0)
        if qty <= 0:
            return False
        required = qty * price + qty * price * costs.commission_bps / 10_000.0
        equity = max(float(self._portfolio_equity()), 0.0)
        strategy_open = self._strategy_open_notional(strategy_id)
        total_open = float(self._open_notional())
        strategy_available = max(equity * self.strategy_leverage[strategy_id] - strategy_open, 0.0)
        global_available = max(equity * self.buying_power_leverage - total_open, 0.0)
        accepted = required <= min(strategy_available, global_available) + 1e-9
        if not accepted:
            self.capacity_blocks.append(
                {
                    "strategy_id": strategy_id,
                    "symbol": order.symbol,
                    "submitted_at": order.submitted_at.isoformat(),
                    "required": required,
                    "strategy_available": strategy_available,
                    "global_available": global_available,
                    "equity": equity,
                    "strategy_open_notional": strategy_open,
                    "total_open_notional": total_open,
                }
            )
        return accepted

    def _strategy_open_notional(self, strategy_id: str) -> float:
        total = 0.0
        for (position_strategy, symbol), position in self.positions.items():
            if position_strategy != strategy_id:
                continue
            total += float(position.qty) * float(self.last_prices.get(symbol, position.avg_price))
        return total

    def _with_strategy_costs(self, strategy_id: str, function, *args):
        original = self.costs
        self.costs = self.strategy_costs[str(strategy_id).upper()]
        try:
            return function(*args)
        finally:
            self.costs = original


def load_promoted_kalcb_replay(
    repo_root: Path,
    raw_config: Mapping[str, Any],
    optimized: Mapping[str, Any],
    *,
    cache_dir: Path,
    snapshot_root: Path,
    intraday_root: Path | None = None,
    data_snapshot_end: str = "20260512",
    market_bar_digest_path: Path | None = None,
    snapshot_end_overrides: Mapping[str, str] | None = None,
    allow_digest_mismatch: bool = False,
) -> tuple[KALCBConfig, CompiledCoreReplay, dict[str, Any]]:
    mutations = dict(optimized.get("mutations") or {})
    expected_hash = str((optimized.get("execution_contract") or {}).get("candidate_snapshot_hash") or "")
    source_path = _repo_path(repo_root, mutations.get("_kalcb.source.path"))
    pool_path = _repo_path(repo_root, mutations.get("_kalcb.pool_source.path"))
    if not source_path.is_file() or not pool_path.is_file():
        raise FileNotFoundError("KALCB promoted source and pool artifacts are required")
    if not snapshot_root.is_dir():
        raise FileNotFoundError(f"KALCB promoted candidate snapshots are required: {snapshot_root}")
    source_hash = _file_hash(source_path)
    pool_hash = _file_hash(pool_path)
    resolved_intraday_root = intraday_root or repo_root / "data/kis_intraday_parquet"
    snapshot_paths = sorted(snapshot_root.glob("*.json"))
    bar_paths = sorted(resolved_intraday_root.rglob(f"*_5m_*_{data_snapshot_end}.parquet"))
    for symbol, end_date in (snapshot_end_overrides or {}).items():
        bar_paths.extend((resolved_intraday_root / symbol).glob(f"{symbol}_5m_*_{end_date}.parquet"))
    snapshot_source_fingerprint = fingerprint_paths(snapshot_paths, root=snapshot_root)
    bar_source_fingerprint = fingerprint_paths(bar_paths, root=resolved_intraday_root)
    digest_hash = (
        _file_hash(market_bar_digest_path)
        if market_bar_digest_path is not None and market_bar_digest_path.is_file()
        else ""
    )
    cache_key = stable_signature(
        {
            "version": "promoted_kalcb_replay_v3",
            "contract_versions": CONTRACT_VERSIONS,
            "source_generation": source_generation_hash(),
            "effective_loader_config": dict(raw_config),
            "snapshot_end_overrides": dict(snapshot_end_overrides or {}),
            "source_hash": source_hash,
            "pool_hash": pool_hash,
            "optimized_mutations": mutations,
            "expected_candidate_hash": expected_hash,
            "snapshot_source_fingerprint": snapshot_source_fingerprint,
            "bar_source_fingerprint": bar_source_fingerprint,
            "market_bar_digest_hash": digest_hash,
        }
    )
    cache_path = cache_dir / f"kalcb_promoted_{cache_key[:16]}.pkl"
    cached = _read_pickle(cache_path)
    if isinstance(cached, CompiledCoreReplay) and cached.candidate_artifact_hash == expected_hash:
        compiled = cached
        cache_status = "hit"
    else:
        compiled = _load_exported_kalcb_replay(
            snapshot_root,
            intraday_root=resolved_intraday_root,
            data_snapshot_end=data_snapshot_end,
            start=_config_date(raw_config.get("start")),
            end=_config_date(raw_config.get("end")),
            initial_equity=float(raw_config.get("initial_equity", 100_000_000.0)),
            market_bar_digest_path=market_bar_digest_path,
            snapshot_end_overrides=snapshot_end_overrides,
        )
        if expected_hash and compiled.candidate_artifact_hash != expected_hash:
            raise ValueError(
                "KALCB exported replay identity mismatch: "
                f"expected {expected_hash}, loaded {compiled.candidate_artifact_hash}"
            )
        _write_pickle(cache_path, compiled)
        cache_status = "rebuilt"
    bar_audit = _market_bar_audit(compiled.bars, market_bar_digest_path)
    if market_bar_digest_path is not None and not bar_audit["archived_digest_match"] and not allow_digest_mismatch:
        raise ValueError("KALCB archived market-bar digest mismatch; specify exact input versions or opt into diagnostics")
    base_cfg = KALCBConfig.from_mapping(dict(raw_config), {})
    config = base_cfg.with_mutations(_normalize_mutations(mutations))
    return config, compiled, {
        "round": int(optimized.get("round") or 0),
        "source_path": str(source_path),
        "source_hash": source_hash,
        "pool_path": str(pool_path),
        "pool_hash": pool_hash,
        "candidate_snapshot_root": str(snapshot_root),
        "snapshot_source_fingerprint": snapshot_source_fingerprint,
        "bar_source_fingerprint": bar_source_fingerprint,
        "snapshot_end_overrides": dict(snapshot_end_overrides or {}),
        "promoted_cache_path": str(cache_path),
        "promoted_cache_status": cache_status,
        "candidate_snapshot_hash": compiled.candidate_artifact_hash,
        "bars": len(compiled.bars),
        "snapshots": len(compiled.snapshots),
        "selections": sum(compiled.selection_counts.values()),
        "market_bar_audit": bar_audit,
    }


def _load_exported_kalcb_replay(
    snapshot_root: Path,
    *,
    intraday_root: Path,
    data_snapshot_end: str,
    start: str,
    end: str,
    initial_equity: float,
    market_bar_digest_path: Path | None,
    snapshot_end_overrides: Mapping[str, str] | None = None,
) -> CompiledCoreReplay:
    snapshots: dict[date, KALCBDailySnapshot] = {}
    for path in sorted(snapshot_root.glob("*.json")):
        payload = json.loads(read_input_bytes(path).decode("utf-8"))
        snapshot = KALCBDailySnapshot.from_json_dict(payload)
        snapshots[snapshot.trade_date] = snapshot
    if not snapshots:
        raise FileNotFoundError(f"No exported KALCB snapshots under {snapshot_root}")
    candidate_hash = stable_signature(
        {day.isoformat(): snapshot.artifact_hash for day, snapshot in sorted(snapshots.items())}
    )
    selected_pairs = {
        (day, str(candidate.symbol).zfill(6))
        for day, snapshot in snapshots.items()
        for candidate in snapshot.candidates
    }
    bars = tuple(
        _load_pinned_parquet_bars(
            selected_pairs,
            intraday_root=intraday_root,
            data_snapshot_end=data_snapshot_end,
            snapshot_end_overrides=snapshot_end_overrides,
        )
    )
    digest = {}
    current_bar_hash = ""
    archived_bar_hash_match = True
    if market_bar_digest_path is not None and market_bar_digest_path.is_file():
        digest = json.loads(read_input_bytes(market_bar_digest_path).decode("utf-8"))
        current_bar_hash = _market_bar_hash(bars)
        archived_bar_hash_match = (
            len(bars) == int(digest.get("bar_count", -1))
            and current_bar_hash == str(digest.get("market_bar_hash") or "")
        )
    calendar = _pinned_trading_calendar(
        intraday_root,
        symbol=bars[0].symbol,
        data_snapshot_end=data_snapshot_end,
        start=start,
        end=end,
    )
    selection_counts = {day: 0 for day in calendar}
    for day, snapshot in snapshots.items():
        selection_counts[day] = int(snapshot.metadata.get("active_symbol_count") or len(snapshot.candidates))
    return CompiledCoreReplay(
        bars=bars,
        snapshots=snapshots,
        session_dates=calendar,
        selection_counts=selection_counts,
        initial_equity=initial_equity,
        source_fingerprint=(
            str(digest.get("compiled_replay_fingerprint") or "")
            if archived_bar_hash_match
            else stable_signature([candidate_hash, current_bar_hash])
        ),
        candidate_artifact_hash=candidate_hash,
    )


def run_kalcb_standalone(config: KALCBConfig, replay: CompiledCoreReplay, *, execution_clock: str = EXECUTION_VERSION, capacity_policy=None) -> ReplayResult:
    costs = kalcb_costs(config)
    adapter = KALCBReplayAdapter(
        config,
        _clone_snapshots_for_replay(replay.snapshots),
        initial_equity=replay.initial_equity,
        costs=costs,
    )
    result = run_replay(
        list(replay.bars),
        adapter,
        initial_equity=replay.initial_equity,
        costs=costs,
        close_open_positions=False,
        bars_are_ordered=True,
        buying_power_leverage=max(float(config.intraday_leverage), 1.0),
        execution_clock=execution_clock,
        capacity_policy=capacity_policy,
    )
    result.decisions.extend(adapter._sync_new_fills(result.broker))
    result.trades = _collapse_exit_legs(result.trades)
    return result


def run_olr_standalone(
    config: OLRConfig,
    snapshots: Mapping[date, OLRDailySnapshot],
    bars: Iterable[MarketBar],
    *,
    initial_equity: float,
    execution_clock: str = EXECUTION_VERSION,
    capacity_policy=None,
) -> ReplayResult:
    adapter = OLRReplayAdapter(config, dict(snapshots))
    result = run_replay(
        list(bars),
        adapter,
        initial_equity=initial_equity,
        costs=olr_costs(config),
        close_open_positions=False,
        bars_are_ordered=True,
        execution_clock=execution_clock,
        capacity_policy=capacity_policy,
    )
    result.decisions.extend(adapter._sync_new_fills(result.broker))
    result.trades = _collapse_exit_legs(result.trades)
    return result


def run_native_risk_shared_replay(inputs: PromotedReplayInputs, *, initial_equity: float, execution_clock: str = EXECUTION_VERSION, capacity_policy=None, portfolio_policy=None, quality_model=None, decision_rank=False) -> ReplayResult:
    sectors = {}
    for snapshot in (*inputs.kalcb_replay.snapshots.values(), *inputs.olr_snapshots.values(),
                     *(s for rows in inputs.kalcb_timed_snapshots.values() for s in rows)):
        for candidate in snapshot.candidates:
            sector = str(candidate.sector or "UNKNOWN").strip().upper()
            previous = sectors.setdefault(candidate.symbol, sector)
            if portfolio_policy is not None and previous != sector:
                raise ValueError("Changing sector membership requires dated native policy inputs")
    broker = NativeRiskSharedBroker(
        initial_equity,
        strategy_costs={"KALCB": kalcb_costs(inputs.kalcb_config), "OLR": olr_costs(inputs.olr_config)},
        strategy_leverage={"KALCB": inputs.kalcb_config.intraday_leverage, "OLR": 1.0},
        execution_clock=execution_clock,
        capacity_policy=capacity_policy,
        portfolio_policy=portfolio_policy,
        sectors=sectors,
        quality_model=quality_model,
        decision_rank=decision_rank,
    )
    kalcb = KALCBReplayAdapter(
        inputs.kalcb_config,
        _clone_snapshots_for_replay(inputs.kalcb_replay.snapshots),
        initial_equity=initial_equity,
        costs=kalcb_costs(inputs.kalcb_config),
        timed_snapshots=inputs.kalcb_timed_snapshots,
        observation_symbols=inputs.kalcb_observation_symbols,
    )
    olr = OLRReplayAdapter(inputs.olr_config, dict(inputs.olr_snapshots))
    decisions: list[Any] = []
    events = _merge_strategy_bars(inputs.kalcb_replay.bars, inputs.olr_bars)
    from itertools import groupby
    for timestamp, grouped in groupby(events, key=lambda event: event["bar"].timestamp):
        batch = tuple(grouped)
        membership = {event["bar"].symbol: event for event in batch}
        decisions.extend(drive_timestamp_batch(broker, [event["bar"] for event in batch], (kalcb, olr),
            accepts=lambda adapter, bar: membership[bar.symbol][adapter.strategy_id]))
    decisions.extend(kalcb._sync_new_fills(broker))
    decisions.extend(olr._sync_new_fills(broker))
    trades = _collapse_exit_legs(list(broker.trades))
    trades.sort(key=lambda trade: (trade.exit_fill_time or trade.entry_fill_time, trade.strategy_id, trade.symbol))
    return ReplayResult(
        trades=trades,
        decisions=decisions,
        equity_curve=list(broker.equity_curve),
        timestamps=list(broker.timestamps),
        broker=broker,
    )


def summarize_replay(result: ReplayResult, *, initial_equity: float) -> ReplaySummary:
    final_equity = float(result.equity_curve[-1]) if result.equity_curve else float(initial_equity)
    metrics = compute_trade_metrics(result.trades, result.equity_curve, timestamps=result.timestamps, mark_coverage=result.broker.mark_coverage or None, initial_equity=initial_equity)
    strategy_ids = sorted({str(trade.strategy_id).upper() for trade in result.trades})
    return ReplaySummary(
        initial_equity=float(initial_equity),
        final_equity=final_equity,
        net_return_pct=final_equity / float(initial_equity) - 1.0,
        max_drawdown_pct=float(metrics.get("max_drawdown_pct", 0.0)),
        win_rate=float(metrics.get("win_rate", 0.0)),
        profit_factor=float(metrics.get("profit_factor", 0.0)),
        trades=len(result.trades),
        rejected_orders=len(result.broker.rejected_orders),
        open_positions=len(result.broker.positions),
        trade_hash=_trade_hash(result.trades),
        strategy_trade_counts={sid: sum(1 for trade in result.trades if trade.strategy_id == sid) for sid in strategy_ids},
        strategy_net_pnl={sid: sum(float(trade.net_pnl) for trade in result.trades if trade.strategy_id == sid) for sid in strategy_ids},
    )


def require_parity(
    actual: ReplaySummary,
    expected: Mapping[str, Any],
    *,
    strategy: str,
    return_tolerance: float = 1e-10,
) -> None:
    expected_return = float(expected["net_return_pct"])
    expected_trades = int(expected["trades"])
    if actual.trades != expected_trades or abs(actual.net_return_pct - expected_return) > return_tolerance:
        raise ValueError(
            f"{strategy} standalone parity failed: expected {expected_trades} trades/{expected_return:.12f}, "
            f"got {actual.trades} trades/{actual.net_return_pct:.12f}"
        )
    expected_trade_hash = str(expected.get("trade_hash") or "")
    if expected_trade_hash and actual.trade_hash != expected_trade_hash:
        raise ValueError(
            f"{strategy} standalone trade hash mismatch: expected {expected_trade_hash}, got {actual.trade_hash}"
        )
    if "profit_factor" in expected:
        expected_profit_factor = float(expected["profit_factor"])
        if abs(actual.profit_factor - expected_profit_factor) > return_tolerance:
            raise ValueError(
                f"{strategy} standalone profit factor mismatch: "
                f"expected {expected_profit_factor:.12f}, got {actual.profit_factor:.12f}"
            )


def overlap_attribution(result: ReplayResult) -> dict[str, Any]:
    fills_by_day: dict[str, set[str]] = defaultdict(set)
    symbols_by_day: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for trade in result.trades:
        day = trade.entry_fill_time.date().isoformat()
        sid = str(trade.strategy_id).upper()
        fills_by_day[day].add(sid)
        symbols_by_day[day][sid].add(str(trade.symbol).zfill(6))
    both_days = sorted(day for day, strategies in fills_by_day.items() if {"KALCB", "OLR"}.issubset(strategies))
    same_symbol_days = sorted(
        day
        for day in both_days
        if symbols_by_day[day].get("KALCB", set()) & symbols_by_day[day].get("OLR", set())
    )
    blocks = list(getattr(result.broker, "capacity_blocks", ()))
    return {
        "days_with_entries_by_strategy": dict(Counter(sid for strategies in fills_by_day.values() for sid in strategies)),
        "days_with_both_strategy_entries": len(both_days),
        "days_with_same_symbol_entries": len(same_symbol_days),
        "same_symbol_entry_days": same_symbol_days,
        "shared_buying_power_block_count": len(blocks),
        "shared_buying_power_blocks_by_strategy": dict(Counter(str(item["strategy_id"]) for item in blocks)),
    }


def shared_vs_standalone_attribution(
    shared: ReplaySummary,
    standalone: Mapping[str, ReplaySummary],
) -> dict[str, Any]:
    """Reconcile the shared return to each scale-invariant standalone result."""

    initial_equity = float(shared.initial_equity)
    contributions: dict[str, dict[str, float | int]] = {}
    reference_return = 0.0
    for strategy_id, standalone_summary in sorted(standalone.items()):
        sid = str(strategy_id).upper()
        standalone_return = float(standalone_summary.net_return_pct)
        shared_pnl = float(shared.strategy_net_pnl.get(sid, 0.0))
        shared_return_contribution = shared_pnl / initial_equity
        reference_return += standalone_return
        contributions[sid] = {
            "standalone_return_pct": standalone_return,
            "shared_return_contribution_pct": shared_return_contribution,
            "return_delta_pct": shared_return_contribution - standalone_return,
            "standalone_trades": int(standalone_summary.trades),
            "shared_trades": int(shared.strategy_trade_counts.get(sid, 0)),
            "trade_count_delta": int(shared.strategy_trade_counts.get(sid, 0) - standalone_summary.trades),
        }
    return {
        "by_strategy": contributions,
        "return_sum_reference_pct": reference_return,
        "shared_return_pct": float(shared.net_return_pct),
        "return_delta_vs_sum_pct": float(shared.net_return_pct) - reference_return,
        "contribution_reconciliation_error_pct": (
            sum(float(item["shared_return_contribution_pct"]) for item in contributions.values())
            - float(shared.net_return_pct)
        ),
    }


def kalcb_costs(config: KALCBConfig) -> BrokerCosts:
    return BrokerCosts(
        commission_bps=config.commission_bps,
        tax_bps_on_sell=config.tax_bps_on_sell,
        slippage_bps=config.slippage_bps,
    )


def olr_costs(config: OLRConfig) -> BrokerCosts:
    return BrokerCosts(
        commission_bps=config.commission_bps,
        tax_bps_on_sell=config.tax_bps_on_sell,
        slippage_bps=config.slippage_bps,
        auction_slippage_bps=config.auction_adverse_bps,
    )


def _merge_strategy_bars(
    kalcb_bars: Iterable[MarketBar],
    olr_bars: Iterable[MarketBar],
) -> list[dict[str, Any]]:
    merged: dict[tuple[datetime, str], dict[str, Any]] = {}
    for strategy_id, bars in (("KALCB", kalcb_bars), ("OLR", olr_bars)):
        for bar in bars:
            key = (bar.timestamp, bar.symbol)
            event = merged.setdefault(key, {"bar": bar, "KALCB": False, "OLR": False})
            if market_bar_values(event["bar"]) != market_bar_values(bar):
                raise ValueError(f"Conflicting shared market bar: {key}")
            event[strategy_id] = True
    return [merged[key] for key in sorted(merged)]


def _load_pinned_parquet_bars(
    pairs: set[tuple[date, str]],
    *,
    intraday_root: Path,
    data_snapshot_end: str,
    snapshot_end_overrides: Mapping[str, str] | None = None,
) -> list[MarketBar]:
    import pandas as pd

    from strategy_common.clock import KST

    dates_by_symbol: dict[str, set[date]] = defaultdict(set)
    for day, symbol in pairs:
        dates_by_symbol[str(symbol).zfill(6)].add(day)
    bars: list[MarketBar] = []
    for symbol, dates in sorted(dates_by_symbol.items()):
        selected_end = (snapshot_end_overrides or {}).get(symbol, data_snapshot_end)
        paths = selected_intraday_paths(intraday_root, symbol, "5m")
        if paths is None:
            paths = sorted((intraday_root / symbol).glob(f"{symbol}_5m_*_{selected_end}.parquet"))
        if not paths:
            raise FileNotFoundError(f"No pinned 5m parquet for {symbol}")
        frames = [
            pd.read_parquet(parquet_input(path), columns=["timestamp", "open", "high", "low", "close", "volume"])
            for path in paths
        ]
        frame = pd.concat(frames, ignore_index=True)
        frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        frame = frame[frame["timestamp"].dt.date.isin(dates)]
        for row in frame.itertuples(index=False):
            timestamp = row.timestamp.to_pydatetime() if hasattr(row.timestamp, "to_pydatetime") else row.timestamp
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=KST)
            timestamp = timestamp.astimezone(KST)
            bars.append(
                MarketBar(
                    symbol=symbol,
                    timestamp=timestamp,
                    timeframe="5m",
                    open=float(row.open),
                    high=float(row.high),
                    low=float(row.low),
                    close=float(row.close),
                    volume=float(row.volume),
                    is_completed=True,
                    source="kis_krx_parquet",
                    source_fingerprint=f"pinned_{selected_end}",
                )
            )
    return sorted(bars, key=lambda bar: (bar.timestamp, bar.symbol))


def _pinned_trading_calendar(
    intraday_root: Path,
    *,
    symbol: str,
    data_snapshot_end: str,
    start: str,
    end: str,
) -> tuple[date, ...]:
    import pandas as pd

    paths = selected_intraday_paths(intraday_root, symbol, "5m")
    if paths is None:
        paths = sorted((intraday_root / symbol).glob(f"{symbol}_5m_*_{data_snapshot_end}.parquet"))
    if not paths:
        raise FileNotFoundError(f"No pinned calendar source for {symbol}")
    timestamps = pd.concat(
        [pd.read_parquet(parquet_input(path), columns=["timestamp"]) for path in paths],
        ignore_index=True,
    )["timestamp"].drop_duplicates()
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    return tuple(sorted({item for item in timestamps.dt.date if start_date <= item <= end_date}))


def _trade_hash(trades: Iterable[Any]) -> str:
    return stable_signature([_scrub_execution_ids(trade.to_json_dict()) for trade in trades])


def _scrub_execution_ids(value: Any) -> Any:
    volatile_keys = {
        "order_id",
        "entry_order_id",
        "exit_order_id",
        "expired_order_id",
        "pending_entry_order_id",
        "pending_exit_order_id",
    }
    if isinstance(value, dict):
        return {
            str(key): _scrub_execution_ids(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
            if str(key) not in volatile_keys
        }
    if isinstance(value, (list, tuple)):
        return [_scrub_execution_ids(item) for item in value]
    return value


def _market_bar_hash(bars: Iterable[MarketBar]) -> str:
    return stable_signature(
        [
            {
                "timestamp": bar.timestamp.isoformat(),
                "symbol": bar.symbol,
                "timeframe": bar.timeframe,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
                "is_completed": bool(bar.is_completed),
                "metadata": dict(bar.metadata),
            }
            for bar in bars
        ]
    )


def _market_bar_audit(
    bars: tuple[MarketBar, ...],
    market_bar_digest_path: Path | None,
) -> dict[str, Any]:
    current_hash = _market_bar_hash(bars)
    archived: dict[str, Any] = {}
    if market_bar_digest_path is not None and market_bar_digest_path.is_file():
        archived = json.loads(read_input_bytes(market_bar_digest_path).decode("utf-8"))
    archived_count = int(archived.get("bar_count", -1))
    archived_hash = str(archived.get("market_bar_hash") or "")
    return {
        "current_bar_count": len(bars),
        "current_market_bar_hash": current_hash,
        "archived_bar_count": archived_count if archived else None,
        "archived_market_bar_hash": archived_hash or None,
        "archived_digest_match": bool(
            archived and len(bars) == archived_count and current_hash == archived_hash
        ),
    }


def _file_hash(path: Path) -> str:
    return hashlib.sha256(read_input_bytes(path)).hexdigest()


def _repo_path(repo_root: Path, raw: Any) -> Path:
    path = Path(str(raw or ""))
    path = path if path.is_absolute() else repo_root / path
    from kis_core.input_manifest import active_manifest
    manifest = active_manifest()
    return manifest.resolve(path) if manifest else path


def _config_date(value: Any) -> str:
    return str(value or "")[:10]


def _parse_datetime(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _read_pickle(path: Path) -> Any | None:
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def _write_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
