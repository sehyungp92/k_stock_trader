from __future__ import annotations

from datetime import datetime

import pytest

from backtests.engine.sim_broker import BrokerCosts, SimOrder
from backtests.strategies.portfolio_synergy.source_replay import (
    NativeRiskSharedBroker,
    ReplaySummary,
    _merge_strategy_bars,
    _scrub_execution_ids,
    require_parity,
    shared_vs_standalone_attribution,
)
from strategy_common.clock import KST
from strategy_common.market import MarketBar


def _order(strategy_id: str, qty: int, *, symbol: str = "005930") -> SimOrder:
    return SimOrder(
        order_id=f"{strategy_id}-{qty}",
        strategy_id=strategy_id,
        symbol=symbol,
        side="BUY",
        qty=qty,
        order_type="MARKET",
        submitted_at=datetime(2026, 1, 2, 9, 0, tzinfo=KST),
        reason="test",
    )


def _summary(
    net_return_pct: float,
    trades: int,
    *,
    strategy_trade_counts: dict[str, int] | None = None,
    strategy_net_pnl: dict[str, float] | None = None,
) -> ReplaySummary:
    return ReplaySummary(
        initial_equity=100.0,
        final_equity=100.0 * (1.0 + net_return_pct),
        net_return_pct=net_return_pct,
        max_drawdown_pct=0.0,
        win_rate=0.5,
        trades=trades,
        rejected_orders=0,
        open_positions=0,
        trade_hash="hash",
        strategy_trade_counts=strategy_trade_counts or {},
        strategy_net_pnl=strategy_net_pnl or {},
    )


def test_native_risk_shared_broker_preserves_strategy_and_global_leverage() -> None:
    zero_costs = BrokerCosts(commission_bps=0.0, tax_bps_on_sell=0.0, slippage_bps=0.0)
    broker = NativeRiskSharedBroker(
        100.0,
        strategy_costs={"KALCB": zero_costs, "OLR": zero_costs},
        strategy_leverage={"KALCB": 2.0, "OLR": 1.0},
    )

    assert broker._can_afford(_order("KALCB", 19), 10.0)
    assert not broker._can_afford(_order("OLR", 11), 10.0)

    broker._apply_fill(_order("KALCB", 15), 10.0, datetime(2026, 1, 2, 9, 5, tzinfo=KST))
    assert not broker._can_afford(_order("OLR", 6, symbol="000660"), 10.0)
    assert broker.capacity_blocks[-1]["global_available"] == pytest.approx(50.0)


def test_merge_strategy_bars_uses_one_market_event_for_shared_symbols() -> None:
    timestamp = datetime(2026, 1, 2, 9, 5, tzinfo=KST)
    shared_bar = MarketBar("005930", timestamp, "5m", 10, 11, 9, 10, 100)
    olr_only = MarketBar("000660", timestamp, "5m", 20, 21, 19, 20, 100)

    merged = _merge_strategy_bars([shared_bar], [shared_bar, olr_only])

    assert len(merged) == 2
    assert merged[1]["bar"] == shared_bar
    assert merged[1]["KALCB"] is True
    assert merged[1]["OLR"] is True


def test_standalone_parity_gate_checks_return_trade_count_and_hash() -> None:
    actual = _summary(2.5, 204)
    require_parity(
        actual,
        {"net_return_pct": 2.5, "trades": 204, "trade_hash": "hash"},
        strategy="KALCB",
    )

    with pytest.raises(ValueError, match="standalone parity failed"):
        require_parity(actual, {"net_return_pct": 2.5, "trades": 205}, strategy="KALCB")


def test_trade_identity_ignores_execution_local_order_ids_recursively() -> None:
    left = {"symbol": "005930", "route_metadata": {"entry_order_id": "random-a", "rank": 1}}
    right = {"symbol": "005930", "route_metadata": {"entry_order_id": "random-b", "rank": 1}}

    assert _scrub_execution_ids(left) == _scrub_execution_ids(right)


def test_shared_return_attribution_reconciles_each_strategy_to_the_total() -> None:
    shared = _summary(
        4.1,
        440,
        strategy_trade_counts={"KALCB": 207, "OLR": 233},
        strategy_net_pnl={"KALCB": 237.0, "OLR": 173.0},
    )
    attribution = shared_vs_standalone_attribution(
        shared,
        {"KALCB": _summary(2.58, 204), "OLR": _summary(1.59, 232)},
    )

    assert attribution["return_sum_reference_pct"] == pytest.approx(4.17)
    assert attribution["return_delta_vs_sum_pct"] == pytest.approx(-0.07)
    assert attribution["contribution_reconciliation_error_pct"] == pytest.approx(0.0)
    assert attribution["by_strategy"]["KALCB"]["trade_count_delta"] == 3
    assert attribution["by_strategy"]["OLR"]["return_delta_pct"] == pytest.approx(0.14)
