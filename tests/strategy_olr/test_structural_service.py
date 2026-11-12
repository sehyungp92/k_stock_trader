from dataclasses import replace
from datetime import datetime,timedelta,time
from pathlib import Path
import importlib.util
import sys
import pytest

from scripts.olr_structural_service_patches import patch
from backtests.engine.olr_interval_service import IntervalParticipationScenario,audit_interval_capacity
from backtests.engine.sim_broker import BrokerCosts
from backtests.strategies.olr.structural_driver import StructuralDriver
from backtests.strategies.olr.runner import OLRReplayAdapter
from backtests.strategies.olr.funded_checkpoint import fork,economic_state,save,load
from strategy_common.actions import SubmitEntry,SubmitExit
from strategy_common.clock import KST
from strategy_olr.models import OLRDailySnapshot
from tests.strategy_olr.test_minute_exit_v5 import minute,coarse,PLAN,PREVIOUS,DAY
from tests.strategy_olr.test_capacity_allocation import config


@pytest.fixture(scope='module')
def broker_type():
    name='backtests.engine._structural_sim_fixture'
    path=Path('backtests/engine/sim_broker.py')
    text=path.read_text(encoding='utf-8')
    if 'interval_service = bool' not in text:text=patch('backtests/engine/sim_broker.py',text)
    spec=importlib.util.spec_from_loader(name,loader=None)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    exec(compile(text,str(path),'exec'),module.__dict__)
    yield module.SimBroker
    sys.modules.pop(name,None)


@pytest.fixture(scope='module')
def nav_auditor():
    path=Path('backtests/engine/olr_minute_audit_v5.py')
    text=path.read_text(encoding='utf-8')
    if 'olr-completed-interval-service-v1' not in text:text=patch(path.as_posix(),text)
    spec=importlib.util.spec_from_loader('backtests.engine._structural_nav_fixture',loader=None)
    module=importlib.util.module_from_spec(spec)
    exec(compile(text,str(path),'exec'),module.__dict__)
    return module.minute_nav_evidence


def setup(broker_type,tmp_path,mode='failure_clock',qty=20):
    cfg=config(capacity_joint_allocation=True,capacity_exit_priority=True,trade_exit_plan=PLAN)
    snaps={d:OLRDailySnapshot(d,(),'fixture',datetime.combine(d,time(8),tzinfo=KST)) for d in (PREVIOUS,DAY)}
    a=OLRReplayAdapter(cfg,snaps)
    b=broker_type(10000,costs=BrokerCosts(0,0,0),capacity_policy=IntervalParticipationScenario(.01,'test'))
    d=StructuralDriver(b,a,mode=mode,output=tmp_path)
    at=datetime.combine(PREVIOUS,time(14,24),tzinfo=KST)
    a._submit_actions([SubmitEntry('OLR','005930',qty,'MARKET',None,None,'fixture',
        dict(risk_per_share=10,trade_exit_plan=PLAN,estimated_entry_price=100))],b,at)
    rows=[minute(PREVIOUS,time(14,25+i)) for i in range(5)]
    d.run_day(rows,[coarse(rows)])
    assert sum(f.qty for f in b.fills)==qty
    return b,a,d


def test_service_observes_interval_volume_and_fills_only_at_completion(broker_type,tmp_path):
    b,a,d=setup(broker_type,tmp_path)
    rows=[minute(t=time(9,i),price=p,volume=v) for i,(p,v) in enumerate([(100,1000),(88,1000),(89,1),(90,1000),(100,1000)])]
    d.run_day(rows,[coarse(rows)])
    sells=[f for f in b.fills if f.side=='SELL']
    assert [(f.timestamp.time(),f.qty,f.price) for f in sells]==[(time(9,4),10,90),(time(9,5),10,100)]
    assert audit_interval_capacity(b)['errors']==[]
    assert d.checkpoint_receipt['no_action_continuation_equal']
    assert not b.positions and not b.orders


def test_failure_and_fade_clocks_are_distinct(broker_type,tmp_path):
    b,a,d=setup(broker_type,tmp_path,'failure_clock')
    rows=[minute(t=time(9,i),price=p) for i,p in enumerate((120,105,106,107,120))]
    d.run_day(rows,[coarse(rows)])
    assert not [f for f in b.fills if f.side=='SELL']
    assert not [r for r in b.journal if r['event']=='minute_exit_signal_v5']
    p=tmp_path/'fade';p.mkdir()
    b,a,d=setup(broker_type,p,'fade_clock')
    d.run_day(rows,[coarse(rows)])
    assert sum(f.qty for f in b.fills if f.side=='SELL')==20
    assert all(f.reason=='mfe_fade' for f in b.fills if f.side=='SELL')


def test_source_volume_change_cannot_change_preinterval_order(broker_type,tmp_path):
    b,a,d=setup(broker_type,tmp_path)
    c=fork(d);assert economic_state(c)==economic_state(d)
    rows=[minute(t=time(9,i),price=88) for i in range(5)]
    altered=[replace(x,volume=1) if i>=1 else x for i,x in enumerate(rows)]
    d.run_day(rows,[coarse(rows)]);c.output=str(tmp_path/'twin');Path(c.output).mkdir()
    c.run_day(altered,[coarse(altered)])
    ds=[r['order'] for r in b.journal if r['event']=='submit' and r['order']['side']=='SELL']
    cs=[r['order'] for r in c.broker.journal if r['event']=='submit' and r['order']['side']=='SELL']
    assert ds==cs
    assert not [f for f in c.broker.fills if f.side=='SELL']


def test_missing_minutes_do_not_supply_service(broker_type,tmp_path):
    b,a,d=setup(broker_type,tmp_path)
    rows=[minute(t=time(9,i),price=88) for i in (0,3,4)]
    summary=coarse([minute(t=time(9,i),price=88) for i in range(5)])
    d.run_day(rows,[summary])
    assert [f.timestamp.time() for f in b.fills if f.side=='SELL']==[time(9,4),time(9,5)]


def test_checkpoint_bytes_and_pending_reservations_are_bound(broker_type,tmp_path):
    b,a,d=setup(broker_type,tmp_path)
    at=datetime.combine(DAY,time(8,59),tzinfo=KST)
    a._submit_actions([SubmitExit('OLR','005930',20,'CLOSE_AUCTION',95,'next_close_exit')],b,at)
    p=tmp_path/'manual.pkl';receipt=save(d,p);twin=load(p,receipt['sha256'])
    assert receipt['pending_orders']==1 and economic_state(twin)==economic_state(d)
    p.write_bytes(p.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='Changed'):load(p,receipt['sha256'])


def test_auction_remains_point_service(broker_type,tmp_path,nav_auditor):
    b,a,d=setup(broker_type,tmp_path,'control')
    at=datetime.combine(DAY,time(14,55),tzinfo=KST)
    a._submit_actions([SubmitExit('OLR','005930',20,'CLOSE_AUCTION',90,'next_close_exit')],b,at)
    before=b._portfolio_equity()
    row=minute(t=time(15,30),price=120,volume=10000)
    d.run_day([row],[replace(row,timeframe='5m')])
    sells=[f for f in b.fills if f.side=='SELL']
    assert len(sells)==1 and sells[0].timestamp==row.timestamp
    assert audit_interval_capacity(b)['errors']==[]
    grid=[[datetime.combine(DAY,t,tzinfo=KST).isoformat(),0] for t in (time(15,25),time(15,30))]
    nav=nav_auditor(b,grid)
    assert nav['common_original_grid'][0]['equity']==before
    assert nav['common_original_grid'][1]['equity']==b.cash
    assert nav['final_equity']==b.cash and b.cash==10400


def test_interval_service_precedes_next_open_in_common_grid(broker_type,tmp_path,nav_auditor):
    b,a,d=setup(broker_type,tmp_path,'control')
    rows=[minute(t=time(9,i),price=p) for i,p in enumerate((100,100,100,100,120,90,90,90,90,90))]
    d.run_day(rows,[coarse(rows[:5]),coarse(rows[5:])])
    nav=nav_auditor(b,[[datetime.combine(DAY,time(9),tzinfo=KST).isoformat(),0]])
    assert nav['common_original_grid'][0]['equity']==10400
    assert nav['all_event_drawdown_pct']>=100*600/10400


def test_nav_rejects_actual_time_reversal(nav_auditor):
    from types import SimpleNamespace
    broker=SimpleNamespace(initial_equity=10000,capacity_policy=IntervalParticipationScenario(.01,'test'),
        minute_coarse_nav=[],journal=[
            dict(event='opening_marks',timestamp='2026-04-01T09:01:00+09:00',prices={'005930':100}),
            dict(event='opening_marks',timestamp='2026-04-01T09:00:00+09:00',prices={'005930':100})])
    with pytest.raises(ValueError,match='Noncausal'):nav_auditor(broker)


def test_interval_auditor_rejects_changed_commitment_and_volume(broker_type,tmp_path):
    b,a,d=setup(broker_type,tmp_path)
    f=next(r for r in b.journal if r['event']=='fill')
    f['fill']['qty']=100000
    assert 'interval_shared_capacity_exceeded' in audit_interval_capacity(b)['errors']


def test_independent_native_ledger_and_clock_audits(broker_type,tmp_path,monkeypatch):
    import backtests.engine.capacity_audit as ca
    import backtests.engine.olr_minute_audit_v5 as ma
    import backtests.engine.qualification_evidence as qe
    from scripts.olr_minute_exit_patches_v5 import patch as minute_patch
    def module(name,text,path):
        spec=importlib.util.spec_from_loader(name,loader=None);m=importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules,name,m);exec(compile(text,str(path),'exec'),m.__dict__);return m
    p=Path('backtests/engine/olr_minute_audit_v5.py')
    text=p.read_text(encoding='utf-8')
    if 'olr-completed-interval-service-v1' not in text:text=patch(p.as_posix(),text)
    audited=module('backtests.engine._structural_clock_fixture',text,p)
    monkeypatch.setattr(ma,'audit_minute_clock',audited.audit_minute_clock)
    monkeypatch.setattr(ca,'audit_capacity',audit_interval_capacity)
    # These fixtures verify native ledger/clock contracts. Source-file audits are
    # intentionally left unverified here; the frozen real-data runs must do them.
    monkeypatch.setattr(qe,'verify_qualification',lambda *a,**k: (dict.fromkeys(qe.CHECKS,'unverified'),{}))
    p=Path('backtests/engine/integrity.py');text=p.read_text(encoding='utf-8')
    if 'minute_clock_report=None' not in text:text=minute_patch(p.as_posix(),text)
    if 'interval_service=bool' not in text:text=patch(p.as_posix(),text)
    audited_ledger=module('backtests.engine._structural_ledger_fixture',text,p)
    b,a,d=setup(broker_type,tmp_path)
    rows=[minute(t=time(9,i),price=p) for i,p in enumerate((100,88,89,90,100))]
    d.run_day(rows,[coarse(rows)])
    report=audited_ledger.audit_broker(b)
    assert report['audit_receipt']['errors']==[]
    nav=audited.minute_nav_evidence(b,[[datetime.combine(DAY,time(9),tzinfo=KST).isoformat(),0]])
    assert nav['final_equity']==b.cash
    assert nav['common_original_grid'][0]['equity']==b.cash
    next(r for r in b.journal if r['event']=='fill')['fill']['metadata']['base_price']+=1
    assert 'incorrect_base_price' in audited_ledger.audit_broker(b)['audit_receipt']['errors']
