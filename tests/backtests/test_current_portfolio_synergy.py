from dataclasses import replace
from datetime import datetime, timedelta
import math

import pytest

from backtests.engine.integrity import audit_broker
from backtests.engine.replay import drive_timestamp_batch
from backtests.strategies.olr.runner import OLRReplayAdapter
from backtests.strategies.portfolio_synergy.source_replay import NativeRiskSharedBroker
from backtests.strategies.portfolio_synergy.quality import CausalQualityModel
from deployment.olr_kalcb.portfolio import PortfolioPolicyConfig
from strategy_common.actions import SubmitEntry
from strategy_olr.config import OLRConfig
from strategy_olr.models import OLRDailySnapshot
from tests.backtests.test_strategy_integrity_repairs import ZERO, bar, ts


def broker(policy=None, model=None):
    return NativeRiskSharedBroker(2000, strategy_costs={"KALCB": ZERO, "OLR": ZERO},
        strategy_leverage={"KALCB": 2, "OLR": 1}, portfolio_policy=policy or PortfolioPolicyConfig(
            funding_mode="native_buying_power", same_symbol_policy="aggregate"), quality_model=model)


def enter(account, strategy, symbol, qty):
    account.submit(SubmitEntry(strategy, symbol, qty, "MARKET", None, None, "entry"), ts())


def test_current_funding_resizes_to_native_account_limit_with_negative_cash():
    account = broker()
    enter(account, "KALCB", "000001", 30)
    enter(account, "OLR", "000002", 20)
    drive_timestamp_batch(account, [bar("000001"), bar("000002")], [])
    assert [(f.strategy_id, f.qty) for f in account.fills] == [("KALCB", 30), ("OLR", 10)]
    assert account.cash == -2000
    assert account.strategy_buying_power("OLR") == 0
    assert audit_broker(account)["execution_validity_pass"]
    row = next(r for r in account.journal if r["event"] == "portfolio_admission")
    row["available_funding"] += 100
    assert "portfolio_unavailable_funding" in audit_broker(account)["audit_receipt"]["errors"]


def test_shared_olr_nav_includes_other_strategy_and_separate_available_funding():
    account = broker()
    enter(account, "KALCB", "000001", 25)
    drive_timestamp_batch(account, [bar("000001")], [])
    snapshot = OLRDailySnapshot(trade_date=ts().date(), candidates=(), source_fingerprint="test", generated_at=ts())
    adapter = OLRReplayAdapter(OLRConfig(), {snapshot.trade_date: snapshot})
    view = adapter._portfolio_view(account)
    assert view.equity == 2000
    assert view.cash == 1500
    assert view.open_notional == 0


def test_aggregate_symbol_risk_resizes_second_sleeve_without_blocking_it():
    account = broker(PortfolioPolicyConfig(funding_mode="native_buying_power",
        same_symbol_policy="aggregate", max_symbol_notional=1500))
    enter(account, "KALCB", "000001", 10)
    enter(account, "OLR", "000001", 10)
    drive_timestamp_batch(account, [bar("000001")], [])
    assert [(f.strategy_id, f.qty) for f in account.fills] == [("KALCB", 10), ("OLR", 5)]
    assert audit_broker(account)["execution_validity_pass"]


def labels(at):
    return [dict(strategy_id=s, entry_fill_time=(at-timedelta(days=4)).isoformat(),
        exit_fill_time=(at-timedelta(days=3)).isoformat(), qty=10, entry_price=100,
        net_pnl=value, realized=True, route_metadata={})
        for s, value in (("KALCB", 1), ("OLR", 10)) for _ in range(20)]


def test_quality_ignores_future_and_same_day_labels_and_falls_back_as_group():
    at = ts()
    rows = labels(at)
    model = CausalQualityModel(rows)
    prediction = model.score("OLR", {}, at)
    for delta in (0, 10):
        rows.append({**rows[-1], "entry_fill_time": at.isoformat(), "exit_fill_time": (at+timedelta(days=delta)).isoformat(), "net_pnl": 1e9})
    assert CausalQualityModel(rows).score("OLR", {}, at) == prediction
    assert CausalQualityModel(rows[:20]).score("KALCB", {}, at) is None


def test_quality_funds_stronger_same_time_order_and_does_not_reorder_exits():
    account = broker(PortfolioPolicyConfig(max_gross_notional=1000), CausalQualityModel(labels(ts())))
    enter(account, "KALCB", "000001", 10)
    enter(account, "OLR", "000002", 10)
    drive_timestamp_batch(account, [bar("000001"), bar("000002")], [])
    assert [f.strategy_id for f in account.fills] == ["OLR"]
    assert audit_broker(account)["execution_validity_pass"]
    prediction = next(r for r in account.journal if r["event"] == "portfolio_admission")["quality_prediction"]
    prediction["latest_label_exit"] = ts().isoformat()
    assert "portfolio_unbound_or_noncausal_quality" in audit_broker(account)["audit_receipt"]["errors"]


@pytest.mark.parametrize("field,value", [("funding_mode", "unlimited"), ("same_symbol_policy", "ignore")])
def test_unregistered_policy_modes_fail_closed(field, value):
    with pytest.raises(ValueError):
        PortfolioPolicyConfig(**{field: value})


def test_shared_source_qualification_requires_every_strategy_and_preserves_each_cutoff(monkeypatch):
    from types import SimpleNamespace
    import backtests.engine.qualification_evidence as qualification
    account = SimpleNamespace(qualification_snapshot_scopes={"KALCB": ["K"], "OLR": ["O"]},
        journal=[{"event": "submit", "order": {"strategy_id": s}} for s in ("KALCB", "OLR")])
    scopes = {s: {"scope": {"strategy": s.lower(), "feature_intraday_before": cutoff},
        "entry_require_complete_prefix": s == "KALCB"} for s, cutoff in (("KALCB", "09:45"), ("OLR", "14:30"))}
    identity = {"configuration": {"source_dependency_scopes": scopes}}
    calls = []
    def verify(view, child, **kwargs):
        scope = child["configuration"]["source_dependency_scope"]
        strategy = scope["strategy"].upper()
        assert set(view.qualification_snapshot_scopes) == {strategy}
        assert all(r["order"]["strategy_id"] == strategy for r in view.journal)
        calls.append(scope["feature_intraday_before"])
        coverage = dict.fromkeys(qualification.CHECKS, "pass")
        if strategy == "OLR":
            coverage["replay_completeness"] = "unverified"
        return coverage, {}
    monkeypatch.setattr(qualification, "verify_qualification", verify)
    coverage, details = qualification._verify_shared_scopes(account, identity, scopes, {})
    assert calls == ["09:45", "14:30"]
    assert coverage["replay_completeness"] != "pass"
    coverage, _ = qualification._verify_shared_scopes(account, identity, {"KALCB": scopes["KALCB"]}, {})
    assert "pass" not in coverage.values()


def test_candidate_priority_precedes_reservations_and_keeps_all_observations():
    from types import SimpleNamespace
    at = ts(30, hour=14)
    seen = []
    candidates = [SimpleNamespace(symbol=s, rank=r, metadata={}) for s,r in (("000001", 2), ("000002", 1))]
    adapter = SimpleNamespace(strategy_id="OLR", snapshots={at.date(): SimpleNamespace(candidates=candidates)},
        qualification_snapshot_sequence=lambda: (), on_bar=lambda b, _: seen.append(b.symbol) or [])
    account = broker()
    account.decision_rank = True
    drive_timestamp_batch(account, [bar("000001", at), bar("000002", at), bar("000003", at)], [adapter])
    assert seen == ["000002", "000001", "000003"]
    account.order_strategy_decisions = lambda _, bars: bars[:1]
    with pytest.raises(ValueError, match="every market observation"):
        drive_timestamp_batch(account, [bar("000001", at+timedelta(minutes=5)), bar("000002", at+timedelta(minutes=5))], [adapter])


def test_kalcb_native_risk_base_does_not_treat_olr_holdings_as_missing_capital():
    from backtests.strategies.kalcb.runner import _portfolio_view_for_broker
    account = broker()
    enter(account, "OLR", "000001", 15)
    drive_timestamp_batch(account, [bar("000001")], [])
    view = _portfolio_view_for_broker(account, "KALCB")
    assert account.cash == 500
    assert view.cash == view.equity == 2000
    assert account.strategy_buying_power("KALCB") == 2500
    account.sizing_equity_cap = 1500
    view = _portfolio_view_for_broker(account, "KALCB")
    assert view.cash == view.equity == 1500


def test_no_upward_compounding_caps_gains_but_still_scales_down_losses():
    from deployment.olr_kalcb.portfolio import PortfolioArbitrationPolicy
    policy = PortfolioArbitrationPolicy(PortfolioPolicyConfig(notional_cap_mode="equity_capped", notional_cap_reference_equity=1000, max_gross_notional=2000))
    assert policy.effective_notional_caps(1500)["max_gross_notional"] == 2000
    assert policy.effective_notional_caps(500)["max_gross_notional"] == 1000
