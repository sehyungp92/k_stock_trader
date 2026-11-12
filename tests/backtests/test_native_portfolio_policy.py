from dataclasses import replace

import pytest

from backtests.engine.integrity import audit_broker
from backtests.engine.replay import drive_timestamp_batch
from backtests.strategies.portfolio_synergy.native_policy import native_policy_config
from backtests.strategies.portfolio_synergy.source_replay import NativeRiskSharedBroker
from tests.backtests.test_strategy_integrity_repairs import ZERO, bar, ts
from strategy_common.actions import SubmitEntry


def simulate(mutations):
    broker = NativeRiskSharedBroker(2000, strategy_costs={"KALCB": ZERO, "OLR": ZERO},
        strategy_leverage={"KALCB": 2, "OLR": 1}, portfolio_policy=native_policy_config(mutations),
        sectors={"000001": "TECH", "000002": "TECH"})
    for strategy, symbol in (("KALCB", "000001"), ("OLR", "000002")):
        broker.submit(SubmitEntry(strategy, symbol, 10, "MARKET", None, None, "entry"), ts())
    drive_timestamp_batch(broker, [bar("000001"), bar("000002")], [])
    return broker


@pytest.mark.parametrize("field", ["max_gross_notional", "max_symbol_notional", "max_sector_notional"])
def test_each_native_cap_changes_real_execution_before_opportunity_expires(field):
    baseline = simulate({})
    limited = simulate({f"portfolio.{field}": 500})
    assert sum(f.qty for f in baseline.fills) == 20
    assert 0 < sum(f.qty for f in limited.fills) < 20
    assert audit_broker(limited)["execution_validity_pass"]
    assert any(row["event"] == "portfolio_admission" for row in limited.journal)


def test_priority_changes_the_actual_funded_strategy():
    first = simulate({"portfolio.max_gross_notional": 1000})
    second = simulate({"portfolio.max_gross_notional": 1000, "portfolio.strategy_priority": ["OLR", "KALCB"]})
    assert [f.strategy_id for f in first.fills] == ["KALCB"]
    assert [f.strategy_id for f in second.fills] == ["OLR"]
    assert audit_broker(second)["execution_validity_pass"]


def test_equity_scaled_caps_and_reference_equity_have_causal_effect():
    fixed = simulate({"portfolio.max_gross_notional": 1000})
    scaled = simulate({"portfolio.max_gross_notional": 1000, "portfolio.notional_cap_mode": "equity_scaled", "portfolio.notional_cap_reference_equity": 4000})
    assert sum(f.qty for f in fixed.fills) == 10
    assert sum(f.qty for f in scaled.fills) == 5
    assert audit_broker(scaled)["execution_validity_pass"]


def test_unrelated_synthetic_mutations_cannot_be_silently_ignored():
    with pytest.raises(ValueError, match="Unsupported"):
        native_policy_config({"portfolio.max_daily_gross": 2})


def test_independent_audit_rejects_false_policy_cash():
    broker = simulate({})
    next(row for row in broker.journal if row["event"] == "portfolio_admission")["cash"] += 100
    assert "portfolio_unavailable_cash_or_equity" in audit_broker(broker)["audit_receipt"]["errors"]
