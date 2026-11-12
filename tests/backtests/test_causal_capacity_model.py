from copy import deepcopy
from dataclasses import replace
from datetime import timedelta, time
from types import SimpleNamespace

import pytest

from backtests.engine.capacity import BarParticipationScenario, CAPACITY_VERSION
from backtests.engine.capacity_audit import audit_capacity
from backtests.engine.integrity import audit_broker
from backtests.engine.replay import drive_timestamp_batch
from backtests.engine.sim_broker import SimBroker
from backtests.engine.qualification_evidence import _capacity, execution_policy
from strategy_common.actions import SubmitEntry, SubmitExit
from tests.backtests.test_strategy_integrity_repairs import ZERO, bar, buy, ts
from tests.backtests.test_qualification_evidence import evidence_fixture
from tests.backtests.test_qualification_scope import inputs


def broker_with_history(volume=1000, participation=.01):
    broker = SimBroker(10000, ZERO, capacity_policy=BarParticipationScenario(participation, "test-model"))
    drive_timestamp_batch(broker, [bar(at=ts(), volume=volume)], [])
    return broker


def test_future_volume_changes_audit_only_not_opening_fills_or_callbacks():
    outcomes = []
    for volume in (1, 1000):
        broker = broker_with_history()
        buy(broker, qty=10)
        callbacks = []
        adapter = SimpleNamespace(on_bar=lambda *_: [], on_fills=lambda b: callbacks.append((b.cash, b.fills[-1].qty)) or [])
        drive_timestamp_batch(broker, [bar(at=ts(5), volume=volume)], [adapter])
        outcomes.append((broker.cash, [(f.qty, f.price) for f in broker.fills], callbacks))
        assert audit_capacity(broker)["status"] == ("failed" if volume == 1 else "model_consistent")
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("next_time", [ts(10), ts(5) + timedelta(days=1)])
def test_gaps_and_new_sessions_do_not_reuse_stale_predictor(next_time):
    broker = broker_with_history()
    buy(broker)
    drive_timestamp_batch(broker, [bar(at=next_time)], [])
    assert not broker.fills
    assert broker.journal[-2]["event"] == "capacity"
    assert broker.journal[-2]["quote"]["reason"] == "missing_prior_continuous_interval"
    drive_timestamp_batch(broker, [bar(at=next_time + timedelta(minutes=5))], [])
    assert len(broker.fills) == 1
    assert audit_capacity(broker)["status"] == "model_consistent"


def test_closing_auction_is_not_an_opening_predictor():
    broker = broker_with_history()
    close = bar(at=ts(30, hour=15), o=100, h=100, l=100, c=100, volume=1_000_000)
    drive_timestamp_batch(broker, [close], [])
    buy(broker, submitted=close.timestamp)
    drive_timestamp_batch(broker, [bar(at=ts() + timedelta(days=1))], [])
    assert not broker.fills


def test_sides_and_strategies_consume_one_budget_and_partial_exit_stays_open():
    broker = broker_with_history(participation=.1)
    buy(broker, qty=20)
    drive_timestamp_batch(broker, [bar(at=ts(5), volume=50)], [])
    broker.submit(SubmitExit("KALCB", "005930", 20, "MARKET", None, "exit"), ts(5))
    broker.submit(SubmitEntry("OLR", "005930", 10, "MARKET", None, None, "entry"), ts(5))
    drive_timestamp_batch(broker, [bar(at=ts(10))], [])
    assert [(f.strategy_id, f.side, f.qty) for f in broker.fills] == [("KALCB", "BUY", 20), ("KALCB", "SELL", 5)]
    assert broker.position_qty("KALCB", "005930") == 15
    assert len(broker.orders) == 2
    assert audit_broker(broker)["execution_validity_pass"]


def test_auction_caps_at_actual_delayed_match_and_expires_remainder(monkeypatch):
    from kis_core.trading_calendar import get_trading_calendar
    calendar = get_trading_calendar()
    monkeypatch.setattr(calendar, "closing_auctions", {("005930", ts().date()): time(15, 32)})
    broker = SimBroker(10000, ZERO, capacity_policy=BarParticipationScenario(.01, "auction"))
    broker.submit(SubmitEntry("OLR", "005930", 15, "CLOSE_AUCTION", None, None, "entry",
        {"expiry_basis": "closing_auction", "expiry_ts": ts(30, hour=15).isoformat()}), ts(30, hour=14))
    drive_timestamp_batch(broker, [bar(at=ts(32, hour=15), o=100, h=100, l=100, c=100)], [])
    assert [(f.qty, f.timestamp) for f in broker.fills] == [(10, ts(32, hour=15))]
    assert not broker.orders and broker.expired_orders[0].qty == 5
    assert audit_broker(broker)["execution_validity_pass"]


def test_non_single_price_auction_never_supplies_verified_capacity():
    broker = SimBroker(10000, ZERO, capacity_policy=BarParticipationScenario(.01, "auction"))
    broker.submit(SubmitEntry("OLR", "005930", 15, "CLOSE_AUCTION", None, None, "entry"), ts(30, hour=14))
    drive_timestamp_batch(broker, [bar(at=ts(30, hour=15))], [])
    assert not broker.fills and not broker.orders
    assert "unsupported_auction_observation" in audit_capacity(broker)["errors"]


def test_auditor_does_not_reuse_the_brokers_quantity_formula(monkeypatch):
    original = BarParticipationScenario.quote
    monkeypatch.setattr(BarParticipationScenario, "quote", lambda self, *args: replace(original(self, *args), budget=1000))
    broker = broker_with_history()
    buy(broker, qty=20)
    drive_timestamp_batch(broker, [bar(at=ts(5))], [])
    assert [f.qty for f in broker.fills] == [20]
    assert {"capacity_quote_mismatch", "shared_capacity_exceeded"}.issubset(audit_capacity(broker)["errors"])


def test_shadow_inherits_policy_but_keeps_separate_liquidity_budget():
    main = SimBroker(10000, ZERO, capacity_policy=BarParticipationScenario(.01, "shared-policy"))
    shadow = SimBroker(100000, ZERO)
    def observe(b, _):
        drive_timestamp_batch(shadow, [b], [])
        return []
    adapter = SimpleNamespace(frontier_shadow=SimpleNamespace(broker=shadow), on_bar=observe)
    drive_timestamp_batch(main, [bar(at=ts())], [adapter])
    buy(main); buy(shadow)
    drive_timestamp_batch(main, [bar(at=ts(5))], [adapter])
    assert main.fills[0].qty == shadow.fills[0].qty == 10
    assert main.capacity_policy == shadow.capacity_policy
    assert audit_broker(main)["execution_validity_pass"]
    shadow.capacity_policy = None
    assert "selection_shadow:capacity_policy_mismatch" in audit_broker(main)["audit_receipt"]["errors"]


def research_declaration(read, proof, broker):
    return {"kind": "declared_bar_participation_model", "model_version": CAPACITY_VERSION,
        "qualification_scope": "historical_research_model", "phase_liquidity_verified": False,
        "market_basis": read(proof["basis"]["source_documents"][0])["normalization"]["target_basis"],
        "registered_execution_policies": [{"policy": execution_policy(broker), "initial_equity": broker.initial_equity,
            "buying_power_leverage": broker.buying_power_leverage}]}


def test_declared_model_has_separate_research_qualification(evidence_fixture):
    read, proof, _, _, _ = inputs(evidence_fixture)
    broker = broker_with_history()
    buy(broker)
    drive_timestamp_batch(broker, [bar(at=ts(5))], [])
    proof["capacity"] = research_declaration(read, proof, broker)
    result = _capacity(read, proof, broker)
    assert result["basis"] == "declared_research_model"
    assert not result["paper_live_qualified"] and not result["phase_liquidity_verified"]
    for defect in ("capital", "cost", "basis", "scope", "version", "tampered_quantity"):
        damaged = deepcopy(proof)
        declaration = damaged["capacity"]
        if defect == "capital":
            declaration["registered_execution_policies"][0]["initial_equity"] *= 10
        elif defect == "cost":
            declaration["registered_execution_policies"][0]["policy"]["default_costs"]["slippage_bps"] += 10
        elif defect == "basis":
            declaration["market_basis"]["volume_unit"] = "lots"
        elif defect == "scope":
            declaration["qualification_scope"] = "live"
        elif defect == "version":
            declaration["model_version"] = "old"
        else:
            next(r for r in broker.journal if r["event"] == "capacity")["allowed_qty"] += 1
        with pytest.raises(ValueError):
            _capacity(read, damaged, broker)


def test_enabled_but_untriggered_stop_is_outside_this_capacity_contract():
    from strategy_common.actions import SubmitProtectiveStop
    broker = broker_with_history()
    broker.submit(SubmitProtectiveStop("KALCB", "005930", 1, 1, "stop"), ts())
    assert "unsupported_capacity_order_type" in audit_capacity(broker)["errors"]


def test_saved_replay_helpers_pass_explicit_capacity_to_execution(monkeypatch):
    from backtests.strategies.portfolio_synergy import source_replay as source
    from strategy_kalcb.config import KALCBConfig
    from strategy_olr.config import OLRConfig
    policy = BarParticipationScenario(.01, "wired")
    class ReachedBroker(Exception):
        pass
    def intercept(*args, **kwargs):
        assert kwargs["capacity_policy"] is policy
        raise ReachedBroker
    monkeypatch.setattr(source, "run_replay", intercept)
    monkeypatch.setattr(source, "KALCBReplayAdapter", lambda *a, **k: None)
    monkeypatch.setattr(source, "OLRReplayAdapter", lambda *a, **k: None)
    compiled = SimpleNamespace(snapshots={}, bars=(), initial_equity=10000)
    with pytest.raises(ReachedBroker):
        source.run_kalcb_standalone(KALCBConfig(), compiled, capacity_policy=policy)
    with pytest.raises(ReachedBroker):
        source.run_olr_standalone(OLRConfig(), {}, (), initial_equity=10000, capacity_policy=policy)


def test_fixed_selection_and_compiled_plan_pass_configured_capacity(monkeypatch):
    from backtests.strategies.kalcb import fixed_trade_plan_phase as fixed
    from backtests.strategies.kalcb import trade_plan_sweep as sweep
    from strategy_kalcb.config import KALCBConfig
    raw = {"name": "research-baseline", "participation": .01}
    compiled = SimpleNamespace(snapshots={}, bars=(), initial_equity=10000, capacity_scenario=raw)
    cfg = KALCBConfig()
    class ReachedBroker(Exception):
        pass
    def intercept(*args, **kwargs):
        assert kwargs["capacity_policy"].contract()["participation"] == .01
        raise ReachedBroker
    monkeypatch.setattr(fixed, "run_replay", intercept)
    monkeypatch.setattr(fixed, "KALCBReplayAdapter", lambda *a, **k: None)
    plugin = fixed.KALCBFixedTradePlanOptimizationPlugin.__new__(fixed.KALCBFixedTradePlanOptimizationPlugin)
    plugin.config = {"capacity_scenario": raw}
    plugin._context_for_mutations = lambda _: SimpleNamespace(compiled_replay=compiled)
    plugin._config_for_mutations = lambda _: cfg
    with pytest.raises(ReachedBroker):
        plugin._evaluate({})
    monkeypatch.setattr(sweep, "run_replay", intercept)
    monkeypatch.setattr(sweep, "KALCBReplayAdapter", lambda *a, **k: None)
    with pytest.raises(ReachedBroker):
        sweep._run_core_replay(sweep.baseline_trade_plan_spec(), compiled, cfg, audit=False)
    assert sweep.score_plan({"capacity_model_valid": False}, (), {}) == (0, "capacity_model_path_failed")


def test_partial_execution_legs_do_not_inflate_trades_or_split_one_loss_into_a_win():
    from backtests.analysis.trade_groups import collapse_exit_legs
    from backtests.analysis.metrics import compute_trade_metrics
    from backtests.engine.sim_broker import SimOrder
    broker = SimBroker(10000, ZERO)
    def order(oid, side, qty):
        return SimOrder(oid, "OLR", "005930", side, qty, "MARKET", ts(), "entry" if side == "BUY" else "exit")
    broker._apply_fill(order("entry", "BUY", 10), 100, ts(5))
    broker._apply_fill(order("exit", "SELL", 5), 110, ts(10))
    broker._apply_fill(order("entry", "BUY", 5), 120, ts(15))
    broker._apply_fill(order("exit", "SELL", 10), 100, ts(20))
    trades = collapse_exit_legs(broker.trades)
    assert len(broker.trades) == 2 and len(trades) == 1
    assert trades[0].qty == 15 and trades[0].net_pnl == -50
    assert trades[0].entry_price == pytest.approx(1600/15)
    assert trades[0].exit_price == pytest.approx(1550/15)
    assert broker.cash == 9950
    metrics = compute_trade_metrics(trades, [10000,9950], initial_equity=10000)
    assert metrics["total_trades"] == 1 and metrics["win_rate"] == metrics["profit_factor"] == 0


@pytest.mark.parametrize("deliver_together", [False, True])
def test_shadow_counts_capacity_fragments_only_when_position_lifetime_closes(deliver_together):
    from backtests.strategies.kalcb.runner import KALCBFrontierShadow
    from backtests.engine.sim_broker import SimOrder
    broker = SimBroker(10000, ZERO, capacity_policy=BarParticipationScenario(.01, "test"))
    shadow = KALCBFrontierShadow.__new__(KALCBFrontierShadow)
    shadow.broker, shadow._trade_cursor = broker, 0
    shadow._pending_trade_legs, shadow._pending_bars = {}, []
    rows = []
    shadow._record_trade_row = rows.append
    def order(oid, side, qty):
        return SimOrder(oid, "KALCB", "005930", side, qty, "MARKET", ts(), "ordinary_exit")
    broker._apply_fill(order("entry", "BUY", 10), 100, ts(5))
    broker._apply_fill(order("exit", "SELL", 5), 110, ts(10))
    if not deliver_together:
        shadow._record_new_trades()
        shadow.finalize(None)
        assert rows == [] and len(shadow._pending_trade_legs) == 1
        assert broker.position_qty("KALCB", "005930") == 5
    broker._apply_fill(order("exit", "SELL", 5), 80, ts(15))
    shadow._record_new_trades()
    assert len(rows) == 1 and rows[0].qty == 10 and rows[0].net_pnl == -50
    assert rows[0].cohort_metadata["position_qty_after"] == 0
    assert not shadow._pending_trade_legs
