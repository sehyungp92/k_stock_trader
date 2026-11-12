"""Outcome-free contracts using the actual frozen native engine and overlay."""
from dataclasses import replace
from datetime import date, datetime, time, timedelta
import math
import pytest

import fixed_signal_contract as contract
import fixed_signal_runtime as runtime
import fixed_signal_clock as clock
from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_olr.models import OLRDailyCandidate, OLRDailySnapshot
from strategy_olr.config import OLRConfig

DAY = date(2026, 1, 8)
NEXT = date(2026, 1, 9)
PLAN = dict(name='mfe_fade1_g125', mode='managed', hard_stop_enabled=False,
    failed_followthrough_bars=1, failed_followthrough_close_r=-1., failed_followthrough_mfe_r=4.,
    mfe_fade_start_r=1., mfe_fade_gap_r=1.25, mfe_fade_floor_r=0.)
ENTRY = dict(mode='confirm_next_bar', name='confirm_b6_vwap_cap25', max_signal_bars=6,
    max_vwap_extension_pct=.025, min_bar_ret=-.001, min_close_location=.5, min_vwap_ret=0.)


def snapshot(day=DAY):
    candidates = tuple(OLRDailyCandidate(symbol=s, trade_date=day, rank=i+1, rank_pct=i+1,
        prior_day_high=102, prior_day_low=98, prior_day_close=100, daily_atr=2,
        expected_5m_volume=1000, average_30m_volume=6000, selection_score=100-i,
        metadata={'prior_close':100, 'atr20':2}) for i, s in enumerate(('005930', '000660')))
    return OLRDailySnapshot(day, candidates, 'fixture', datetime.combine(day, time(14,30), KST))


def config():
    return OLRConfig(overnight_slot_count=6, min_selected=1, max_position_pct=.65,
        allocation_mode='rank_weighted', rank_decay=1.5, target_gross_exposure=1.2,
        trade_entry_plan=ENTRY, trade_exit_plan=PLAN, capacity_exit_quantile=.5,
        capacity_joint_allocation=True, capacity_redistribute=True,
        capacity_redistribution_evidence='entry', capacity_recycle_liquidity_only=True,
        capacity_loss_atr_multiple=1.25, capacity_loss_position_pct=.04,
        capacity_loss_sector_pct=.06, capacity_loss_total_pct=.08, capacity_exit_priority=True)


def minute(day, stamp, price, volume=1000, symbol='005930'):
    # A flat bar has native close-location zero and cannot confirm this rule.
    # One-share prints are valid observations but give zero 1% execution budget.
    low = price if stamp == time(15,30) else price-.1
    return MarketBar(symbol, datetime.combine(day, stamp, KST), '1m', price, price, low, price,
                     max(1, volume), source='fixture', source_fingerprint='frozen-fixture')


def aggregate(rows):
    groups = {}
    for bar in rows:
        stamp = bar.timestamp.replace(minute=(bar.timestamp.minute//5)*5)
        groups.setdefault(stamp, []).append(bar)
    return [replace(v[0], timeframe='5m', timestamp=k, open=v[0].open, high=max(b.high for b in v),
                    low=min(b.low for b in v), close=v[-1].close, volume=sum(b.volume for b in v))
            for k, v in sorted(groups.items())]


def entry_rows(volume=1000, day=DAY):
    start = datetime.combine(day, time(14,20), KST)
    rows = [minute(day, (start+timedelta(minutes=i)).time(), 100 if i<10 else 100.2, volume) for i in range(60)]
    rows.append(minute(day, time(15,30), 100.2, volume))
    return rows


def replay(budget=2500, scenario='baseline', volume=1000, exit_volume=1000, future=False):
    broker, adapter, driver = runtime.make_lane(config(), snapshot(), '005930', budget, scenario)
    rows = entry_rows(volume)
    if future:
        rows = [replace(b, open=110, high=110, low=110, close=110) if b.timestamp.time()>=time(15) else b for b in rows]
    driver.run_day(rows, aggregate(rows), symbols={'005930'})
    later = [minute(NEXT, time(9,i), 80, exit_volume) for i in range(15)]
    later.append(minute(NEXT, time(15,30), 80, exit_volume))
    driver.run_day(later, aggregate(later), symbols={'005930'})
    return broker, adapter, driver


def test_native_confirmed_entry_partial_fills_costed_exit_and_audits():
    broker, adapter, _ = replay()
    buys = [f for f in broker.fills if f.side=='BUY']
    assert len(buys)>=2 and buys[0].timestamp.time()==time(14,35)
    assert all(f.qty<=10 for f in buys)
    result, audit = runtime.assess(broker, adapter, 2500)
    assert result['status']=='matured'
    assert result['net_pnl']<0 and result['total_fees']>0
    assert not audit['audit_receipt']['errors']
    assert audit['audit_receipt']['minute_clock_audit']['status']=='pass'
    assert audit['capacity_model_audit']['errors']==[]
    assert result['nav']['common_original_drawdown_pct'] is not None
    assert result['nav']['all_event_drawdown_pct']>=result['nav']['common_original_drawdown_pct']


def test_requests_are_fixed_and_rank_does_not_recycle_funding():
    a, _, _ = replay(2500)
    b, _, _ = replay(5000)
    for broker, budget in ((a,2500),(b,5000)):
        orders=[r['order'] for r in broker.journal if r['event']=='submit' and r['order']['side']=='BUY']
        assert len(orders)==1
        m=orders[0]['metadata']
        assert orders[0]['qty']==math.floor(budget/(m['estimated_entry_price']*(1+m['entry_cost_buffer_pct'])))
        assert m['target_notional']==budget


def test_stress_halves_native_minute_capacity_and_increases_costs():
    broker, adapter, _ = replay(scenario='combined')
    assert all(f.qty<=5 for f in broker.fills if f.metadata['execution_phase']=='open')
    result,_=runtime.assess(broker,adapter,2500)
    assert result['status']=='matured'
    for fill in broker.fills:
        base=fill.metadata['base_price']
        assert fill.price==pytest.approx(base*(1+.0015 if fill.side=='BUY' else 1-.0015))


def test_unfilled_is_missing_not_zero_profit():
    broker,adapter,_=replay(volume=0)
    result,_=runtime.assess(broker,adapter,2500)
    assert result['status']=='unfilled' and result['net_pnl'] is None and result['net_bps'] is None


def test_unresolved_inventory_is_censored_not_forced_closed():
    broker,adapter,_=replay(exit_volume=0)
    result,_=runtime.assess(broker,adapter,2500)
    assert result['status']=='right_censored' and result['residual_quantity']>0
    assert result['net_pnl'] is None and not broker.same_day_forced_exit_count


def test_unsupported_source_has_no_payoff_even_after_observed_fills():
    broker,adapter,_=replay()
    result,_=runtime.assess(broker,adapter,2500,unsupported='fixture_gap')
    assert result['status']=='unsupported_source' and result['net_pnl'] is None


def test_missing_prior_minute_cannot_supply_capacity():
    broker,adapter,driver=runtime.make_lane(config(),snapshot(),'005930',2500,'baseline')
    rows=entry_rows()
    # Keep confirmation's complete five-minute prefix; then miss14:35.
    rows=[b for b in rows if b.timestamp.time()!=time(14,35)]
    coarse=aggregate(entry_rows())
    driver.run_day(rows,coarse,symbols={'005930'})
    assert min(f.timestamp.time() for f in broker.fills if f.side=='BUY')==time(14,37)


def test_future_prices_do_not_change_first_request_or_earlier_fills():
    a,_,_=replay()
    b,_,_=replay(future=True)
    early=lambda x:[(f.timestamp,f.qty,f.price) for f in x.fills if f.side=='BUY' and f.timestamp.time()<time(15)]
    assert early(a)==early(b)


def test_non_target_candidate_cannot_enter_lane():
    broker,adapter,driver=runtime.make_lane(config(),snapshot(),'005930',2500,'baseline')
    rows=entry_rows();other=[replace(b,symbol='000660') for b in rows]
    driver.run_day(sorted(rows+other,key=lambda b:(b.timestamp,b.symbol)),
        sorted(aggregate(rows)+aggregate(other),key=lambda b:(b.timestamp,b.symbol)),symbols={'005930','000660'})
    assert {f.symbol for f in broker.fills}=={'005930'}


def test_outside_original_slots_rejected_before_replay():
    with pytest.raises(ValueError,match='in-slot'):
        runtime.make_lane(replace(config(),overnight_slot_count=1),snapshot(),'000660',2500,'baseline')


def test_import_guard_rejects_overlay_mutation_and_foreign_submodule(tmp_path):
    native=tmp_path/'native';foreign=tmp_path/'foreign';over=tmp_path/'overlay.py'
    (native/'sample').mkdir(parents=True);foreign.mkdir()
    (native/'sample/__init__.py').write_text('')
    (foreign/'other.py').write_text('')
    over.write_text('VALUE=1')
    guard=contract.NativeOverlay(native,{'sample/__init__.py':contract.sha(native/'sample/__init__.py')},
        {'sample/child.py':{'path':str(over),'sha256':contract.sha(over)}})
    assert PathLike(guard.find_spec('sample.child').origin)==PathLike(over)
    over.write_text('VALUE=2')
    with pytest.raises(ImportError,match='Changed'):
        guard.find_spec('sample.child')
    with pytest.raises(ImportError,match='Foreign'):
        guard.find_spec('sample.other',[str(foreign)])


def PathLike(value):
    from pathlib import Path
    return Path(value).resolve()


def test_sampling_ignores_original_fill_and_profit_and_groups_dates():
    rows=[]
    for lo,hi in contract.BLOCKS:
        start=date.fromisoformat(lo)
        for i in range(24):
            day=str(start+timedelta(days=i))
            for symbol in ('005930','000660'):
                rows.append(dict(day=day,symbol=symbol,rank=1,native_signal_time=day+' 14:35:00+09:00',filled=i%2,net_pnl=i))
    selected=contract.select(rows)
    assert len(selected)==72
    for r in rows:r.update(filled=False,net_pnl=-999)
    assert contract.select(rows)==selected


def test_close_anchored_completeness_retains_late_open_causality():
    day=date(2026,1,2)
    start=datetime.combine(day,time(10),KST)
    bars=[replace(minute(day,(start+timedelta(minutes=5*i)).time(),100),timeframe='5m') for i in range(54)]
    full={(day,'005930'):bars+[replace(bars[-1],timestamp=datetime.combine(day,time(15,25),KST))]}
    prefix={(day,'005930'):bars}
    assert clock.close_anchored_feature_bars(prefix,time(14,30))[1]==clock.close_anchored_feature_bars(full,time(14,30))[1]==[]


def test_execution_audit_rejects_tampered_capacity_evidence():
    broker,adapter,_=replay()
    row=next(r for r in broker.journal if r['event']=='capacity' and r['quote']['budget'])
    row['quote']['predictor_volume']+=1
    with pytest.raises(ValueError,match='Native execution audit'):
        runtime.assess(broker,adapter,2500)


def test_native_loss_limit_still_caps_large_fixed_request():
    snap=snapshot()
    snap=replace(snap,candidates=(replace(snap.candidates[0],daily_atr=100),*snap.candidates[1:]))
    broker,adapter,driver=runtime.make_lane(config(),snap,'005930',20_000_000,'baseline')
    rows=entry_rows(volume=1_000_000)
    driver.run_day(rows,aggregate(rows),symbols={'005930'})
    order=next(r['order'] for r in broker.journal if r['event']=='submit' and r['order']['side']=='BUY')
    assert order['metadata']['allocation_binding_limit']=='position_loss_budget'
    evidence=order['metadata']['scenario_loss_budget']
    assert evidence['approved_loss']<=4_000_000
    assert evidence['approved_loss']+evidence['unit_loss']>4_000_000
    assert order['qty']<math.floor(20_000_000/order['metadata']['estimated_entry_price'])


def test_overlay_disabled_matches_original_native_entry_function():
    import ast
    import copy
    from pathlib import Path
    from strategy_olr.core import logic
    from strategy_olr.core.state import OLRState
    from strategy_olr.core.core_models import OLRPortfolioView
    from strategy_common.actions import action_to_json_dict
    design=contract.read(contract.ROOT/'design.json')
    ready=contract.read(design['source_ready'])
    source=(Path(ready['files'])/'strategy_olr/core/logic.py').read_text(encoding='utf-8')
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_maybe_submit_entry')
    namespace=dict(logic.__dict__)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<bound-original-entry-function>','exec'),namespace)
    native=namespace['_maybe_submit_entry']
    snap=snapshot();candidate=snap.candidates[0]
    series=aggregate(entry_rows())[:3]
    state=OLRState();state.symbol_state('005930').session_bars.extend(series)
    view=OLRPortfolioView(cash=100_000_000,equity=100_000_000)
    left=logic._maybe_submit_entry(copy.deepcopy(state),series[-1],config(),snap,candidate,view)
    right=native(copy.deepcopy(state),series[-1],config(),snap,candidate,view)
    assert [action_to_json_dict(a) for a in left.actions]==[action_to_json_dict(a) for a in right.actions]
    assert [e.to_json_dict() for e in left.decisions]==[e.to_json_dict() for e in right.decisions]
    assert left.actions, 'Parity fixture must exercise an actual request'


def test_fixed_target_risk_does_not_hide_behind_idle_account_cash():
    broker,adapter,_=replay()
    result,_=runtime.assess(broker,adapter,2500)
    assert result['fixed_target_risk']['max_drawdown_over_target_pct']>10
    assert result['nav']['all_event_drawdown_pct']<.01
