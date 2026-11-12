from dataclasses import replace
from datetime import datetime, time
import pytest

from backtests.strategies.olr.decision_continuation import DecisionContinuationDriver, continuation_state
from backtests.strategies.olr.funded_checkpoint import economic_state, fork, save, load
from backtests.strategies.olr.structural_driver import StructuralDriver
from strategy_common.actions import SubmitExit
from strategy_common.clock import KST
from tests.strategy_olr import test_structural_service as fixtures
from tests.strategy_olr.test_structural_service import broker_type
from tests.strategy_olr.test_minute_exit_v5 import minute, coarse, DAY


def setup_driver(broker_type, tmp_path, monkeypatch, driver_type=DecisionContinuationDriver, mode='control'):
    tmp_path.mkdir(parents=True, exist_ok=True)
    with monkeypatch.context() as m:
        m.setattr(fixtures, 'StructuralDriver', driver_type)
        return fixtures.setup(broker_type, tmp_path, mode)


@pytest.mark.parametrize('mode', ['control', 'failure_clock', 'fade_clock', 'both_clocks', 'structural_partial'])
def test_native_whole_day_is_exactly_preserved(broker_type, tmp_path, monkeypatch, mode):
    b, a, old = setup_driver(broker_type,tmp_path/'old',monkeypatch,StructuralDriver,mode)
    nb, na, new = setup_driver(broker_type,tmp_path/'new',monkeypatch,DecisionContinuationDriver,mode)
    rows=[minute(t=time(9,i),price=p) for i,p in enumerate((120,105,88,100,120,119,115,118,121,120))]
    summaries=[coarse(rows[:5]),coarse(rows[5:])]
    x=old.run_day(rows,summaries);y=new.run_day(rows,summaries)
    assert [d.to_json_dict() for d in x]==[d.to_json_dict() for d in y]
    assert economic_state(old)==economic_state(new)
    assert old.structural_observer.trace==new.structural_observer.trace
    assert old.structural_observer.account.high_water==new.structural_observer.account.high_water
    assert new.checkpoint_receipt['no_action_continuation_equal']


@pytest.mark.parametrize('pause', [1,4,5,9,10])
def test_serialized_intraday_resume_keeps_clocks_fills_and_high_water(broker_type,tmp_path,monkeypatch,pause):
    b,a,d=setup_driver(broker_type,tmp_path,monkeypatch)
    rows=[minute(t=time(9,i),price=p) for i,p in enumerate((100,120,118,110,115,114,113,115,116,117))]
    d.start_day(rows,[coarse(rows[:5]),coarse(rows[5:])])
    for _ in range(pause):d.advance()
    path=tmp_path/'intraday.pkl';receipt=save(d,path);twin=load(path,receipt['sha256'])
    assert continuation_state(d)==continuation_state(twin)
    original=d.finish_day();restored=twin.finish_day()
    assert [x.to_json_dict() for x in original]==[x.to_json_dict() for x in restored]
    assert continuation_state(d)==continuation_state(twin)
    assert len(b.minute_day_scopes)==2
    assert len([x for x in b.minute_clock_schedule if x['kind']=='coarse'])==3
    assert d.structural_observer.account.high_water>=10400


def test_hold_probe_changes_no_native_outcome(broker_type,tmp_path,monkeypatch):
    b,a,d=setup_driver(broker_type,tmp_path,monkeypatch)
    rows=[minute(t=time(9,i)) for i in range(5)]
    d.start_day(rows,[coarse(rows)]);d.advance();twin=fork(d)
    d.request_inventory_target('005930',20,action_id='hold')
    d.finish_day();twin.finish_day()
    assert economic_state(d)==economic_state(twin)
    assert d.action_trace[-1]['status']=='continue_native'


def test_partial_target_uses_real_service_and_residual_inventory(broker_type,tmp_path,monkeypatch):
    b,a,d=setup_driver(broker_type,tmp_path,monkeypatch)
    rows=[minute(t=time(9,i),price=p,volume=v) for i,(p,v) in enumerate([(100,1000),(90,500),(95,1),(110,500),(110,1000)])]
    d.start_day(rows,[coarse(rows)]);d.advance()
    cash_before=b.cash
    d.request_inventory_target('005930',10,action_id='half')
    assert b.cash==cash_before
    d.advance()
    assert a.state.symbols['005930'].position.qty_open==15
    assert b.cash==cash_before+450
    path=tmp_path/'partial.pkl';receipt=save(d,path);twin=load(path,receipt['sha256'])
    d.finish_day();twin.finish_day()
    assert continuation_state(d)==continuation_state(twin)
    assert a.state.symbols['005930'].position.qty_open==10
    assert [(f.timestamp.time(),f.qty) for f in b.fills if f.side=='SELL']==[(time(9,2),5),(time(9,4),5)]
    assert b.cash==cash_before+1000


def test_cancel_acknowledgement_precedes_replacement(broker_type,tmp_path,monkeypatch):
    b,a,d=setup_driver(broker_type,tmp_path,monkeypatch)
    at=datetime.combine(DAY,time(8,59),tzinfo=KST)
    a._submit_actions([SubmitExit('OLR','005930',20,'CLOSE_AUCTION',95,'next_close_exit')],b,at)
    rows=[minute(t=time(9,i),price=100) for i in range(5)]
    d.start_day(rows,[coarse(rows)]);d.advance()
    old_pending=a.state.symbols['005930'].pending_exit_order_id
    with monkeypatch.context() as m:
        m.setattr(a,'_sync_order_updates',lambda *args: [])
        d.request_inventory_target('005930',0,action_id='exit')
    assert a.state.symbols['005930'].pending_exit_order_id==old_pending
    assert not [o for o in b.orders if o.reason=='funded_inventory_target']
    assert '005930' in d.pending_targets
    d.finish_day()
    submitted=[r['order'] for r in b.journal if r['event']=='submit' and r['order']['reason']=='funded_inventory_target']
    assert len(submitted)==1 and submitted[0]['qty']==20
    assert sum(f.qty for f in b.fills if f.side=='SELL')==20
    assert not b.positions and not d.pending_targets


def test_decision_view_and_commitment_do_not_read_future_volume(broker_type,tmp_path,monkeypatch):
    _,_,d=setup_driver(broker_type,tmp_path/'a',monkeypatch)
    _,_,t=setup_driver(broker_type,tmp_path/'b',monkeypatch)
    rows=[minute(t=time(9,i),price=100) for i in range(5)]
    changed=[replace(b,volume=1,high=150,close=150) if i else b for i,b in enumerate(rows)]
    d.start_day(rows,[coarse(rows)]);t.start_day(changed,[coarse(changed)])
    d.advance();t.advance()
    assert d.decision_view()==t.decision_view()
    d.request_inventory_target('005930',0,action_id='same');t.request_inventory_target('005930',0,action_id='same')
    assert d.action_trace==t.action_trace
    d.finish_day();t.finish_day()
    assert not d.broker.positions and t.broker.positions


def test_missing_minutes_and_closing_point_resume_exactly(broker_type,tmp_path,monkeypatch):
    b,a,d=setup_driver(broker_type,tmp_path/'old',monkeypatch,StructuralDriver)
    nb,na,n=setup_driver(broker_type,tmp_path/'new',monkeypatch)
    rows=[minute(t=time(9,i),price=100) for i in (0,3,4)]
    summary=coarse([minute(t=time(9,i),price=100) for i in range(5)])
    auction=minute(t=time(15,30),price=120,volume=10000)
    summaries=[summary,replace(auction,timeframe='5m')]
    d.run_day([*rows,auction],summaries)
    n.start_day([*rows,auction],summaries);n.advance()
    receipt=save(n,tmp_path/'gap.pkl');twin=load(tmp_path/'gap.pkl',receipt['sha256'])
    n.finish_day();twin.finish_day()
    assert economic_state(d)==economic_state(n)
    assert continuation_state(n)==continuation_state(twin)


def test_reject_invalid_or_repeated_targets_and_session_restart(broker_type,tmp_path,monkeypatch):
    b,a,d=setup_driver(broker_type,tmp_path,monkeypatch)
    rows=[minute(t=time(9,i)) for i in range(5)]
    with pytest.raises(ValueError,match='completed interval'):d.decision_view()
    d.start_day(rows,[coarse(rows)])
    with pytest.raises(ValueError,match='Finish'):d.start_day(rows,[coarse(rows)])
    d.advance()
    for q in (-1,21,1.5,True):
        with pytest.raises(ValueError,match='Invalid residual'):d.request_inventory_target('005930',q,action_id='bad')
    d.request_inventory_target('005930',20,action_id='once')
    with pytest.raises(ValueError,match='unused'):d.request_inventory_target('005930',20,action_id='once')
    assert not [o for o in b.orders if o.reason=='funded_inventory_target']
