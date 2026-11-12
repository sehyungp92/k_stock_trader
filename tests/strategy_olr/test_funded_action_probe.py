from copy import deepcopy
from dataclasses import replace
from datetime import time
from pathlib import Path
import importlib.util
import pytest

from backtests.strategies.olr.funded_action_probe import FundedActionProbeDriver, VERSION
from backtests.strategies.olr.decision_continuation import DecisionContinuationDriver
from backtests.engine.olr_interval_service import audit_interval_capacity
from scripts.olr_funded_probe_patches import patch_audit
from scripts.olr_structural_service_patches import patch
from tests.strategy_olr.test_structural_service import broker_type
from tests.strategy_olr.test_decision_continuation import setup_driver
from tests.strategy_olr.test_minute_exit_v5 import minute, coarse


@pytest.fixture(scope='module')
def auditor():
    path=Path('backtests/engine/olr_minute_audit_v5.py')
    text=path.read_text(encoding='utf8')
    if 'olr-completed-interval-service-v1' not in text:text=patch(path.as_posix(),text)
    if 'funded_probe_contract=probe_contract' not in text:text=patch_audit(text)
    spec=importlib.util.spec_from_loader('backtests.engine._funded_probe_audit_fixture',loader=None)
    module=importlib.util.module_from_spec(spec)
    exec(compile(text,str(path),'exec'),module.__dict__)
    return module.audit_minute_clock


def run_case(broker_type,tmp_path,monkeypatch,action,reference=None,future_volume=1000):
    def driver(*args,**kwargs):
        return FundedActionProbeDriver(*args,probe=dict(version=VERSION,action=action,control_reference=reference),**kwargs)
    b,a,d=setup_driver(broker_type,tmp_path,monkeypatch,driver,mode='control')
    rows=[minute(t=time(9,i),price=p,volume=1000 if i<2 else future_volume)
          for i,p in enumerate((120,105,104,110,119))]
    d.run_day(rows,[coarse(rows)])
    return b,a,d


def test_real_checkpoint_hold_reproduces_uninterrupted_native_day(broker_type,tmp_path,monkeypatch,auditor):
    b,a,d=run_case(broker_type,tmp_path/'hold',monkeypatch,'hold')
    nb,na,n=setup_driver(broker_type,tmp_path/'native',monkeypatch,DecisionContinuationDriver,mode='control')
    rows=[minute(t=time(9,i),price=p) for i,p in enumerate((120,105,104,110,119))]
    n.run_day(rows,[coarse(rows)])
    assert b.equity_curve==nb.equity_curve and b.fills==nb.fills
    assert [x for x in b.journal if x['event']!='funded_probe_boundary_v1']==nb.journal
    assert d.checkpoint_receipt['no_action_continuation_equal']
    assert auditor(b)['errors']==[]


@pytest.mark.parametrize('action,remaining',[('half',10),('exit',0)])
def test_native_target_current_quantity_service_and_matched_shadow(broker_type,tmp_path,monkeypatch,auditor,action,remaining):
    _,_,hold=run_case(broker_type,tmp_path/'hold',monkeypatch,'hold')
    b,a,d=run_case(broker_type,tmp_path/action,monkeypatch,action,hold.probe_receipt)
    assert d.probe_receipt['decision']==hold.probe_receipt['decision']
    assert d.probe_receipt['observed_account']==hold.probe_receipt['observed_account']
    assert d.probe_receipt['no_action_session_signature']==hold.probe_receipt['no_action_session_signature']
    p=a.state.symbols['005930'].position
    assert (p.qty_open if p else 0)==remaining
    assert sum(f.qty for f in b.fills if f.side=='SELL')==20-remaining
    assert all(f.timestamp.time()>=time(9,3) for f in b.fills if f.side=='SELL')
    assert audit_interval_capacity(b)['errors']==[]
    assert auditor(b)['errors']==[]


def test_action_selection_independent_of_unseen_future_and_no_invented_service(broker_type,tmp_path,monkeypatch):
    _,_,h1=run_case(broker_type,tmp_path/'h1',monkeypatch,'hold')
    _,_,h2=run_case(broker_type,tmp_path/'h2',monkeypatch,'hold',future_volume=1)
    b,_,x=run_case(broker_type,tmp_path/'x',monkeypatch,'exit',h1.probe_receipt)
    nb,_,y=run_case(broker_type,tmp_path/'y',monkeypatch,'exit',h2.probe_receipt,future_volume=1)
    assert x.probe_selection==y.probe_selection
    assert x.action_trace==y.action_trace
    assert sum(f.qty for f in b.fills if f.side=='SELL')==20
    assert not [f for f in nb.fills if f.side=='SELL']
    assert nb.positions


@pytest.mark.parametrize('field,value',[('quantity',21),('source_prefix','wrong'),('ordinal',2),('high_water',200)])
def test_independent_audit_rejects_changed_boundary(broker_type,tmp_path,monkeypatch,auditor,field,value):
    b,_,d=run_case(broker_type,tmp_path,monkeypatch,'hold')
    row=next(r for r in b.journal if r['event']=='funded_probe_boundary_v1')
    row[field]=value
    assert 'funded_probe_unearned_boundary' in auditor(b)['errors']


def test_independent_audit_rejects_changed_target(broker_type,tmp_path,monkeypatch,auditor):
    _,_,h=run_case(broker_type,tmp_path/'h',monkeypatch,'hold')
    b,_,x=run_case(broker_type,tmp_path/'x',monkeypatch,'half',h.probe_receipt)
    row=next(r for r in b.journal if r['event']=='submit' and r['order']['reason']=='funded_inventory_target')
    row['order']['metadata']['target_inventory']=0
    assert 'funded_probe_invalid_target_submission' in auditor(b)['errors']


def test_wrong_control_shadow_fails_closed(broker_type,tmp_path,monkeypatch):
    _,_,h=run_case(broker_type,tmp_path/'h',monkeypatch,'hold')
    bad=deepcopy(h.probe_receipt);bad['start_signature']='wrong'
    with pytest.raises(ValueError,match='does not match control'):
        run_case(broker_type,tmp_path/'x',monkeypatch,'half',bad)
