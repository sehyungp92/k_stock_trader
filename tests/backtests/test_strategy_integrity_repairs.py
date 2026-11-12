from dataclasses import asdict, replace
from datetime import date, datetime, timedelta
import os
from statistics import mean, pstdev
from types import SimpleNamespace

import pandas as pd
import pytest

from backtests.analysis.metrics import compute_trade_metrics
from backtests.auto.shared.cache_keys import fingerprint_paths
from backtests.auto.shared.phase_runner import _metric_contract_criteria
from backtests.auto.shared.plugin_utils import CachedBatchEvaluator, mutation_signature
from backtests.auto.shared.types import Experiment, ScoredCandidate
from backtests.engine.contracts import LEGACY_EXECUTION_VERSION
from backtests.engine.integrity import audit_broker
from backtests.engine.replay import run_replay
from backtests.engine.sim_broker import BrokerCosts, SimBroker
from strategy_common.actions import SubmitEntry, SubmitExit, SubmitProtectiveStop, ReplaceProtectiveStop
from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.core.core_models import KALCBFillEvent
from strategy_kalcb.core.logic import on_kalcb_fill
from strategy_kalcb.core.state import KALCBState
from strategy_olr.config import OLRConfig
from strategy_olr.core.core_models import OLRFillEvent
from strategy_olr.core.logic import on_olr_fill
from strategy_olr.core.state import OLRState

DAY = date(2026, 1, 5)
ZERO = BrokerCosts(0, 0, 0)


def ts(minute=0, day=DAY, hour=9):
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=KST)


def bar(symbol="005930", at=None, o=100, h=110, l=90, c=100, volume=1000):
    return MarketBar(symbol, at or ts(5), "5m", o, h, l, c, volume)


def buy(broker, symbol="005930", qty=10, submitted=None):
    return broker.submit(SubmitEntry("KALCB", symbol, qty, "MARKET", None, None, "entry"), submitted or ts())


def test_synthetic_entrypoints_require_explicit_fixture():
    from backtests.cli import main
    from backtests.auto.cli import main as optimize
    from backtests.strategies.registry import get_backtest_runner, create_plugin
    from backtests.strategies.portfolio_synergy.replay import load_portfolio_bundle, evaluate_portfolio
    for call in (lambda: main(["run", "--strategy", "portfolio_synergy"]),
                 lambda: optimize(["optimize", "--strategy", "portfolio_synergy", "--dry-run"])):
        with pytest.raises(SystemExit):
            call()
    with pytest.raises(ValueError):
        get_backtest_runner("portfolio_synergy")({"capability_level": "real_replay"})
    with pytest.raises(ValueError):
        create_plugin("portfolio_synergy", {"synthetic_fixture": True})
    with pytest.raises(ValueError):
        load_portfolio_bundle()
    bundle = load_portfolio_bundle(synthetic_fixture=True)
    assert not any(item.source_trade for item in bundle.opportunities)
    result = evaluate_portfolio({}, bundle=bundle, synthetic_fixture=True)
    assert result.metrics["capability_level"] == "synthetic"
    assert result.metrics["execution_contract"]["capability_level"] == "synthetic"
    metrics = {**result.metrics, "promotion_requires_audit_pass": True,
        "allow_synthetic_promotion": True, "fast_full_audit_passed": True}
    gates = {item.name: item.passed for item in _metric_contract_criteria(metrics)}
    assert not gates["hard_audit_pass"] and not gates["hard_non_synthetic_capability"]


@pytest.mark.parametrize("invalid", [bar(volume=0), bar(at=ts(day=date(2026, 1, 10), hour=8)),
    bar(o=float("nan")), bar(c=200), replace(bar(), is_completed=False)])
def test_nonexecutable_bar_keeps_order_until_valid_event(invalid):
    broker = SimBroker(10000, ZERO)
    order_id = buy(broker)
    assert broker.process_bar(invalid) == []
    assert broker.orders[0].order_id == order_id
    assert broker.cash == 10000 and not broker.rejected_orders


def test_ordered_fast_path_validates_order_and_completed_bars():
    adapter = SimpleNamespace(on_bar=lambda *args: [])
    with pytest.raises(ValueError, match="time ordered"):
        run_replay([bar(at=ts(10)), bar()], adapter, bars_are_ordered=True)
    with pytest.raises(ValueError, match="Incomplete"):
        run_replay([replace(bar(), is_completed=False)], adapter, bars_are_ordered=True)


def test_stop_limit_is_rejected_without_awarding_untraded_limit():
    broker = SimBroker(10000, ZERO)
    broker.submit(SubmitEntry("KALCB", "005930", 10, "STOP_LIMIT", 101, 100, "gap"), ts())
    assert broker.process_bar(bar(o=110, h=112, l=108, c=111)) == []
    assert len(broker.rejected_orders) == 1


def test_opening_cash_cannot_use_intrabar_sale_and_is_permutation_invariant():
    def replay(reverse):
        broker = SimBroker(1000, ZERO)
        buy(broker, symbol="000001")
        broker.process_bar(bar(symbol="000001", h=100, l=100))
        broker.submit(SubmitProtectiveStop("KALCB", "000001", 10, 95), ts(5))
        buy(broker, symbol="000002", submitted=ts(5))
        batch = [bar("000001", ts(10), o=100, h=100, l=90), bar("000002", ts(10))]
        broker.process_batch(batch[::-1] if reverse else batch)
        return broker
    first, second = replay(False), replay(True)
    assert first.position_qty("KALCB", "000002") == 0
    assert len(first.rejected_orders) == 1
    assert first.cash == second.cash == 950
    assert [(f.symbol, f.side, f.price) for f in first.fills] == [(f.symbol, f.side, f.price) for f in second.fills]
    assert audit_broker(first)["execution_validity_pass"]


def test_future_close_does_not_expand_opening_leverage():
    broker = SimBroker(1000, ZERO, buying_power_leverage=2)
    buy(broker)
    broker.process_bar(bar(h=100, l=100))
    buy(broker, symbol="000002", qty=15, submitted=ts(5))
    broker.process_batch((bar(at=ts(10), h=200, c=200), bar("000002", ts(10))))
    assert broker.position_qty("KALCB", "000002") == 0
    assert len(broker.rejected_orders) == 1


@pytest.mark.parametrize("enabled", [True, False])
def test_real_core_installs_entry_bar_protection_before_low(enabled):
    from tests.strategy_kalcb.test_core_logic import _or_bars, _snapshot
    from backtests.strategies.kalcb.runner import KALCBReplayAdapter
    cfg = replace(KALCBConfig.from_mapping({"kalcb.entry.entry_score_blocklist": []}), exit_hard_stop_enabled=enabled)
    adapter = KALCBReplayAdapter(cfg, {DAY: _snapshot(DAY)}, initial_equity=1000000, costs=ZERO)
    result = run_replay(_or_bars(DAY) + [bar(at=ts(35), o=102, h=103, l=90, c=102)], adapter,
        initial_equity=1000000, costs=ZERO, close_open_positions=False)
    fills = result.broker.fills
    assert fills[0].side == "BUY" and fills[0].price == 102
    assert sum(fill.side == "SELL" for fill in fills) == int(enabled)
    assert audit_broker(result.broker)["execution_validity_pass"]


def test_amended_stop_never_reuses_the_consumed_interval():
    broker = SimBroker(10000, ZERO)
    buy(broker)
    broker.process_bar(bar(h=100, l=100))
    broker.submit(SubmitProtectiveStop("KALCB", "005930", 10, 90), ts(5))
    broker.process_bar(bar(at=ts(10), o=100, h=120, l=91, c=110))
    broker.submit(ReplaceProtectiveStop("KALCB", "005930", 108, qty=10), ts(10))
    broker.process_bar(bar(at=ts(15), o=95, h=111, l=90, c=110))
    assert broker.fills[-1].price == 95
    assert audit_broker(broker)["execution_validity_pass"]


def test_addon_costs_reconcile_and_terminal_mark_is_not_a_fill():
    broker = SimBroker(10000, BrokerCosts(100, 0, 0))
    buy(broker)
    broker.process_bar(bar(h=100, l=100))
    buy(broker, submitted=ts(5))
    broker.process_bar(bar(at=ts(10), h=100, l=100))
    broker.close_all_at_end(bar("000002", at=ts(30, hour=15)))
    assert len(broker.fills) == 2 and broker.position_qty("KALCB", "005930") == 20
    broker.submit(SubmitExit("KALCB", "005930", 20, "MARKET", None, "exit"), ts(10))
    broker.process_bar(bar(at=ts(15), h=100, l=100))
    assert broker.cash - 10000 == sum(t.net_pnl for t in broker.trades) == -40
    assert audit_broker(broker)["execution_validity_pass"]


@pytest.mark.parametrize("corrupt", ["fee", "cash", "qty", "base", "ledger"])
def test_independent_audit_rejects_tampered_evidence(corrupt):
    broker = SimBroker(10000, ZERO)
    buy(broker)
    broker.process_bar(bar())
    if corrupt == "cash":
        broker.cash += 10
    elif corrupt == "qty":
        broker.positions[("KALCB", "005930")].qty += 1
    elif corrupt == "ledger":
        broker.fills.clear()
    else:
        row = next(row for row in broker.journal if row["event"] == "fill")
        row["fill"]["metadata"]["fee" if corrupt == "fee" else "base_price"] += 1
    metrics = audit_broker(broker)
    assert not metrics["execution_validity_pass"] and not metrics["audit_pass"]


def test_slippage_outside_bar_is_valid_and_volume_capacity_is_disclosed():
    broker = SimBroker(10000, BrokerCosts(0, 0, 10))
    buy(broker)
    broker.process_bar(bar(o=100, h=100, l=100, c=100, volume=1))
    metrics = audit_broker(broker)
    assert metrics["execution_validity_pass"]
    assert metrics["capacity_evidence"]["max_quantity_to_volume"] == 10
    assert not metrics["audit_pass"]  # Capacity and calendar/provenance coverage are unverified.


def test_daily_sharpe_offset_duplicate_timestamps_and_overnight():
    stamps = [ts(30, hour=15), ts(30, hour=15), ts(30, DAY + timedelta(days=1), 15)]
    metrics = compute_trade_metrics([], [100, 105, 110, 99], initial_equity=100, timestamps=stamps)
    returns = [.1, -.1]
    assert metrics["sharpe"] == pytest.approx(mean(returns) / pstdev(returns) * 252 ** .5)
    assert metrics["max_drawdown_pct"] == pytest.approx(.1)
    assert metrics["daily_return_count"] == 2
    with pytest.raises(ValueError, match="initial sample"):
        compute_trade_metrics([], [100, 110], initial_equity=100, timestamps=stamps)


def test_missing_held_session_does_not_fabricate_daily_returns():
    stamps = [ts(30, hour=15), ts(30, DAY + timedelta(days=2), 15)]
    metrics = compute_trade_metrics([], [100, 110, 120], initial_equity=100, timestamps=stamps,
        mark_coverage=[{"cash_only": False, "stale_symbols": []}] * 2)
    assert metrics["daily_return_count"] == 0
    assert metrics["daily_equity_coverage"]["missing_sessions"] == ["2026-01-06"]


def test_cache_identity_detects_same_size_and_mtime_changes(tmp_path):
    path = tmp_path / "data"
    path.write_bytes(b"100")
    stamp = path.stat()
    before = fingerprint_paths([path])
    path.write_bytes(b"200")
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert fingerprint_paths([path]) != before


def test_frame_cache_distinguishes_equal_boundaries_and_updated_files(tmp_path):
    from backtests.strategies.kalcb.replay_cache import _daily_rows_from_frame, _load_symbol_frame
    frame = pd.DataFrame([{"timestamp": ts(), "open": 100., "high": 100., "low": 100., "close": 100., "volume": 1000.}])
    other = frame.copy()
    other[["open", "high", "low", "close"]] = 200.
    assert _daily_rows_from_frame("005930", frame)[0]["close"] == 100
    assert _daily_rows_from_frame("005930", other)[0]["close"] == 200
    folder = tmp_path / "005930"
    folder.mkdir()
    path = folder / "005930_5m_20260105_20260105.parquet"
    frame.to_parquet(path)
    assert _load_symbol_frame(tmp_path, "005930", "5m", DAY).iloc[0]["close"] == 100
    other.to_parquet(path)
    assert _load_symbol_frame(tmp_path, "005930", "5m", DAY).iloc[0]["close"] == 200


def test_seeded_invalid_result_is_reevaluated_and_cannot_score():
    calls = []
    def delegate(candidates, current):
        calls.append(1)
        return [ScoredCandidate("x", 999, metrics={"audit_pass": False, "fast_full_audit_passed": True})]
    evaluator = CachedBatchEvaluator(delegate, seed_results={mutation_signature({}): ScoredCandidate("old", 999, metrics={"audit_pass": True})})
    result = evaluator([Experiment("x", {})], {})[0]
    assert calls and result.rejected and result.score == 0


@pytest.mark.parametrize("strategy", ["kalcb", "olr"])
def test_missing_previous_exchange_session_is_excluded(strategy):
    import importlib
    builder = importlib.import_module(f"strategy_{strategy}.research").build_research_snapshot
    rows = [{"date": "2026-01-02", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000}]
    assert "005930" not in builder({"005930": rows}, date(2026, 1, 6)).symbols
    assert "005930" in builder({"005930": rows}, date(2026, 1, 5)).symbols


@pytest.mark.parametrize("strategy", ["kalcb", "olr"])
def test_fragmented_entries_deduplicate_and_survive_state_restore(strategy):
    import importlib
    state, cfg, handler, event = (KALCBState(), KALCBConfig(), on_kalcb_fill, KALCBFillEvent) if strategy == "kalcb" else (OLRState(), OLRConfig(), on_olr_fill, OLRFillEvent)
    first = event("entry", "005930", "BUY", 4, 100, ts(), metadata={"execution_id": "a", "order_qty": 10, "filled_qty_before": 0, "filled_qty_after": 4, "stop_price": 95})
    handler(state, first, cfg)
    handler(state, first, cfg)
    serializer = importlib.import_module(f"strategy_{strategy}.core.serializers")
    state = serializer.restore_state(serializer.snapshot_state(state))
    handler(state, event("entry", "005930", "BUY", 6, 110, ts(1), metadata={"execution_id": "b", "order_qty": 10, "filled_qty_before": 4, "filled_qty_after": 10}), cfg)
    position = state.symbol_state("005930").position
    assert position.qty_open == 10 and position.entry_price == 106
    assert position.entry_time == ts()
    with pytest.raises(ValueError, match="Unrelated"):
        handler(state, event("unrelated", "005930", "BUY", 1, 100, ts(2)), cfg)


@pytest.mark.parametrize("strategy", ["kalcb", "olr"])
def test_partial_sell_then_entry_fragment_keeps_open_cost_basis(strategy):
    state, cfg, handler, event = (KALCBState(), KALCBConfig(), on_kalcb_fill, KALCBFillEvent) if strategy == "kalcb" else (OLRState(), OLRConfig(), on_olr_fill, OLRFillEvent)
    handler(state, event("entry", "005930", "BUY", 4, 100, ts(), metadata={"execution_id": "a", "stop_price": 95}), cfg)
    sale = event("exit", "005930", "SELL", 3, 120, ts(1), metadata={"execution_id": "sell"})
    result = handler(state, sale, cfg)
    handler(state, sale, cfg)  # Must deduplicate before comparing with one remaining share.
    handler(state, event("entry", "005930", "BUY", 6, 110, ts(2), metadata={"execution_id": "b"}), cfg)
    position = state.symbol_state("005930").position
    assert position.qty_open == 7
    assert position.entry_price == pytest.approx(760 / 7)
    assert position.entry_time == ts()
    if strategy == "kalcb":
        from strategy_common.actions import ReplaceProtectiveStop
        assert any(isinstance(a, ReplaceProtectiveStop) and a.qty == 1 for a in result.actions)


def test_disabled_hard_stop_does_not_install_oms_protection():
    from strategy_common.oms_adapter import action_to_intent
    action = SubmitEntry("KALCB", "005930", 10, "MARKET", None, 95, "entry",
        metadata={"stop_price": 95, "hard_stop_enabled": False})
    intent = action_to_intent(action)
    assert intent.risk_payload.stop_px is None
    assert intent.risk_payload.hard_stop_px is None


@pytest.mark.parametrize("strategy", ["kalcb", "olr"])
def test_distinct_equal_fragments_and_unknown_fees_are_not_inherited(strategy):
    state, cfg, handler, event = (KALCBState(), KALCBConfig(), on_kalcb_fill, KALCBFillEvent) if strategy == "kalcb" else (OLRState(), OLRConfig(), on_olr_fill, OLRFillEvent)
    handler(state, event("entry", "005930", "BUY", 4, 100, ts(), metadata={"execution_id": "a", "fee": 2}), cfg)
    handler(state, event("entry", "005930", "BUY", 4, 100, ts(), metadata={"execution_id": "b"}), cfg)
    position = state.symbol_state("005930").position
    assert position.qty_open == 8
    assert position.metadata["entry_fees"] == 2
    assert position.metadata["entry_fees_complete"] is False


@pytest.mark.parametrize("strategy", ["kalcb", "olr"])
def test_partial_cancel_acknowledgement_then_late_fill_remains_managed(strategy):
    from strategy_kalcb.core.core_models import KALCBOrderUpdateEvent
    from strategy_kalcb.core.logic import on_kalcb_order_update
    from strategy_olr.core.core_models import OLROrderUpdateEvent
    from strategy_olr.core.logic import on_olr_order_update
    state, cfg, handler, event = (KALCBState(), KALCBConfig(), on_kalcb_fill, KALCBFillEvent) if strategy == "kalcb" else (OLRState(), OLRConfig(), on_olr_fill, OLRFillEvent)
    symbol = state.symbol_state("005930")
    symbol.pending_entry_order_id = "entry"
    state.order_roles["entry"] = {"role": "entry", "order_role": "ENTRY"}
    handler(state, event("entry", "005930", "BUY", 4, 100, ts(), metadata={"execution_id": "a",
        "order_qty": 10, "filled_qty_before": 0, "filled_qty_after": 4, "stop_price": 95}), cfg)
    assert symbol.pending_entry_order_id == "entry"
    if strategy == "kalcb":
        on_kalcb_order_update(state, KALCBOrderUpdateEvent("entry", "005930", "CANCELLED", ts(1)))
    else:
        on_olr_order_update(state, OLROrderUpdateEvent("entry", "005930", "CANCELLED", ts(1)), cfg)
    assert symbol.pending_entry_order_id == ""
    handler(state, event("entry", "005930", "BUY", 2, 110, ts(2), metadata={"execution_id": "late",
        "order_qty": 10, "filled_qty_before": 4, "filled_qty_after": 6}), cfg)
    assert symbol.position.qty_open == 6
    assert symbol.position.entry_price == pytest.approx(620 / 6)
    assert symbol.position.entry_time == ts()


def test_olr_target_entitlements_round_on_total_acquired_quantity():
    from strategy_common.actions import SubmitPartialExit
    state = OLRState()
    total = {}
    prices = {}
    for index in range(10):
        metadata = {"execution_id": str(index), "order_qty": 10,
            "filled_qty_before": index, "filled_qty_after": index + 1}
        if index == 0:
            metadata.update({"protective_stop_price": 95, "risk_per_share": 5,
                "trade_exit_plan": {"name": "fragment_test", "mode": "managed", "hard_stop_enabled": True,
                    "partial_trigger_r": 1, "partial_fraction": 0.3, "target_r": 2}})
        result = on_olr_fill(state, OLRFillEvent("entry", "005930", "BUY", 1, 100 + index,
            ts(index), metadata=metadata), OLRConfig())
        for action in result.actions:
            if isinstance(action, (SubmitPartialExit, SubmitExit)):
                role = action.metadata["order_role"]
                total[role] = total.get(role, 0) + action.qty
                prices.setdefault(role, set()).add(action.limit_price)
                assert action.qty > 0
    assert total == {"TARGET": 7, "PARTIAL_TARGET": 3}
    assert prices == {"TARGET": {110}, "PARTIAL_TARGET": {105}}


def test_hydrated_snapshot_rejects_missing_or_stale_reference():
    from dataclasses import replace
    from strategy_common.execution_session import snapshot_reference_issues
    from tests.deployment.olr_kalcb.test_readiness_and_coordinator import _kalcb_snapshot
    snapshot = _kalcb_snapshot(date(2026, 2, 2))
    assert snapshot_reference_issues(snapshot) == {}
    for metadata in ({}, {"prior_day_date": "2026-01-29"}, {"prior_day_date": "2026-02-03"}):
        stale = replace(snapshot, candidates=(replace(snapshot.candidates[0], metadata=metadata),))
        assert snapshot_reference_issues(type(snapshot).from_json_dict(stale.to_json_dict())) == {"005930": "missing_previous_session"}


def test_actual_olr_phase_and_final_metrics_always_require_independent_audit():
    from backtests.strategies.olr.plugin import OLROptimizationPlugin
    plugin = OLROptimizationPlugin.__new__(OLROptimizationPlugin)
    broker = SimBroker(10000, ZERO)
    result = SimpleNamespace(metrics={"audit_pass": True, "fast_full_audit_passed": True},
        replay_result=SimpleNamespace(broker=broker))
    plugin._evaluate_result = lambda _: result
    assert plugin.get_phase_spec(1, None).promotion_requires_audit_pass is True
    metrics = plugin.compute_final_metrics({})
    assert metrics["promotion_requires_audit_pass"] is True
    assert metrics["audit_pass"] is False
    gates = {item.name: item.passed for item in _metric_contract_criteria(metrics)}
    assert gates["hard_audit_pass"] is False


def test_contract_attachment_preserves_explicit_nested_failure():
    from backtests.strategies.common.plugin_base import attach_official_metric_contract
    result = attach_official_metric_contract({"audit_pass": True, "metric_contract": {"audit_pass": False}}, audit_pass=True)
    assert result["audit_pass"] is False


def test_oos_and_allocation_hygiene_cannot_replace_execution_validity():
    from backtests.auto.oos_ablation import _audit_hygiene_pass
    from backtests.strategies.olr.allocation_sweep import _official_replay_pass
    metrics = {"audit_pass": False, "same_bar_fill_count": 0, "forced_replay_close_count": 0,
        "rejected_order_count": 0, "end_open_position_count": 0, "fast_full_audit_passed": True}
    assert not _audit_hygiene_pass(metrics)
    assert not _official_replay_pass(metrics)


def test_auction_entry_has_no_exposure_to_earlier_bar_extremes():
    from strategy_olr.core.logic import step_olr_core
    from strategy_olr.core.core_models import OLRPortfolioView
    broker = SimBroker(10000, ZERO)
    broker.submit(SubmitEntry("OLR", "005930", 10, "CLOSE_AUCTION", None, None, "auction"), ts(25, hour=15))
    auction_bar = bar(at=ts(30, hour=15), h=150, l=50)
    broker.process_bar(auction_bar)
    assert broker.positions[("OLR", "005930")].max_price == 100
    assert broker.positions[("OLR", "005930")].min_price == 100
    state = OLRState()
    on_olr_fill(state, OLRFillEvent("a", "005930", "BUY", 10, 100, auction_bar.timestamp,
        metadata={"execution_phase": "auction"}), OLRConfig())
    step_olr_core(state, auction_bar, OLRConfig(), None, OLRPortfolioView(cash=9000, equity=10000))
    position = state.symbol_state("005930").position
    assert position.max_favorable_price == 100 and position.max_adverse_price == 100


def test_fixed_kalcb_compiler_uses_exact_reference_session():
    from backtests.strategies.kalcb.trade_plan_sweep import _prior_daily_row
    dataset = SimpleNamespace(daily_by_symbol={"005930": [{"date": "2026-01-02", "close": 100}]})
    assert _prior_daily_row(dataset, "005930", date(2026, 1, 5))["close"] == 100
    assert _prior_daily_row(dataset, "005930", date(2026, 1, 6)) is None
