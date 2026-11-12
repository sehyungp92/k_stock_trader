from dataclasses import asdict,replace
from datetime import date
import math
import pytest
from backtests.engine.sim_broker import SimBroker,BrokerCosts
from backtests.engine.replay import drive_timestamp_batch
from backtests.strategies.kalcb.runner import KALCBReplayAdapter
from backtests.strategies.kalcb.fixed_allocation_reference_v1 import FixedAllocationReference,allocation_cash
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.core.serializers import snapshot_state
from tests.strategy_kalcb.test_core_logic import _snapshot,_or_bars


def replay(mode=None,initial=1_000_000.,reference=1_000_000.,risk=.001):
    day=date(2026,1,5);snap=_snapshot(day)
    cfg=KALCBConfig(entry_plan_mode='first30_open',rvol_threshold=0,cpr_threshold=0,
        momentum_score_min=0,entry_score_blocklist=(),entry_require_complete_prefix=False,
        frontier_shadow_enabled=False,risk_per_trade_pct=risk,max_position_notional_pct=1.)
    costs=BrokerCosts(2,18,3);broker=SimBroker(initial,costs)
    adapter=KALCBReplayAdapter(cfg,{day:snap},initial_equity=initial,costs=costs)
    benchmark=FixedAllocationReference(reference,enabled=mode is True)
    if mode is not None:benchmark.install()
    try:
        for bar in _or_bars(day):drive_timestamp_batch(broker,(bar,),(adapter,))
        return {'cash':broker.cash,'equity':broker.equity_curve,'fills':[asdict(f) for f in broker.fills],
                'state':snapshot_state(adapter.state),'rejections':[asdict(o) for o in broker.rejected_orders],
                'positions':{k:(p.qty,p.avg_price) for k,p in broker.positions.items()}},benchmark
    finally:
        if mode is not None:benchmark.restore()


def test_identity_overlay_preserves_real_native_economics_and_state():
    plain,_=replay();identity,b= replay(False)
    assert plain==identity
    assert b.rows and all(r['actual_cash']==r['sizing_cash_reference'] for r in b.rows)


def test_fixed_budget_does_not_compound_profitable_cash_or_erase_equity():
    day=date(2026,1,5);cfg=KALCBConfig(frontier_shadow_enabled=False)
    broker=SimBroker(1_000_000.,BrokerCosts())
    adapter=KALCBReplayAdapter(cfg,{day:_snapshot(day)},initial_equity=1_000_000.,costs=BrokerCosts())
    # Two separately funded accounts test the view contract, not trading PnL.
    richer=SimBroker(1_200_000.,BrokerCosts())
    b=FixedAllocationReference(1_000_000.,enabled=True);b.install()
    try:
        a,c=adapter._portfolio_view(broker),adapter._portfolio_view(richer)
        assert a.cash==c.cash==1_000_000.
        assert a.equity==1_000_000. and c.equity==1_200_000.
        assert broker.cash==1_000_000. and richer.cash==1_200_000.
    finally:b.restore()


def test_underfunded_fixed_reference_cannot_create_cash_or_invalid_fills():
    result,b=replay(True,initial=5_000.,reference=1_000_000.,risk=.5)
    cash=5_000.
    for fill in result['fills']:
        fee=fill['metadata']['fee']
        cash+=(-fill['qty']*fill['price'] if fill['side']=='BUY' else fill['qty']*fill['price'])-fee
        assert cash>=-1e-8
    assert result['cash']==pytest.approx(cash)
    assert result['rejections']
    assert b.evidence()['external_capital_flows']==0


def test_view_keeps_invested_cost_out_of_fixed_allocation_budget():
    assert allocation_cash(100.,30.)==70.
    assert allocation_cash(100.,120.)==0.
    for reference,invested in ((0,0),(math.nan,0),(100,-1),(100,math.inf)):
        with pytest.raises(ValueError):allocation_cash(reference,invested)


def test_incompatible_equity_caps_cannot_silently_claim_fixed_sizing():
    from backtests.strategies.kalcb.fixed_allocation_reference_v1 import validate_scope
    with pytest.raises(ValueError,match='Unsupported'):
        validate_scope(KALCBConfig(frontier_shadow_enabled=False,risk_max_single_equity_pct=.5))


def test_actual_drawdown_and_session_history_are_not_frozen_by_the_reference():
    from strategy_kalcb.core.logic import _update_portfolio_context
    day=date(2026,1,5);cfg=KALCBConfig(frontier_shadow_enabled=False)
    broker=SimBroker(900_000.,BrokerCosts())
    adapter=KALCBReplayAdapter(cfg,{day:_snapshot(day)},initial_equity=900_000.,costs=BrokerCosts())
    adapter.state.session_date=day
    adapter.state.meta.update(portfolio_high_water_equity=1_200_000.,portfolio_session_date=str(day),
                              portfolio_session_start_equity=1_000_000.)
    b=FixedAllocationReference(1_000_000.,enabled=True);b.install()
    try:
        view=adapter._portfolio_view(broker)
        _update_portfolio_context(adapter.state,view,cfg)
        assert view.cash==1_000_000. and view.equity==900_000.
        assert view.drawdown_pct==pytest.approx(.25)
        assert view.session_return_pct==pytest.approx(-.1)
        assert adapter.state.meta['portfolio_high_water_equity']==1_200_000.
    finally:b.restore()
