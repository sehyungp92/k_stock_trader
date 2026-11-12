from copy import deepcopy
from datetime import time
from pathlib import Path
import importlib.util
import pytest

from backtests.strategies.olr.funded_population import (
    FundedPopulationDriver,VERSION,funding_context,audit_context,decision_evidence)
from backtests.strategies.olr.funded_action_probe import VERSION as PROBE_VERSION
from tests.strategy_olr.test_funded_action_probe import auditor,run_case
from tests.strategy_olr.test_structural_service import broker_type
from tests.strategy_olr.test_decision_continuation import setup_driver
from tests.strategy_olr.test_minute_exit_v5 import minute,coarse
from scripts.olr_funded_population_patches import patch_allocation


def run_population(broker_type,tmp_path,monkeypatch,action='hold',cell=None,selection=None,reference=None,future_volume=1000):
    def factory(*args,**kwargs):
        return FundedPopulationDriver(*args,**kwargs,
            probe=dict(version=PROBE_VERSION,action=action,control_reference=reference),
            population=dict(version=VERSION,cell=cell,catalog_reference=selection))
    b,a,d=setup_driver(broker_type,tmp_path,monkeypatch,factory,mode='control')
    rows=[minute(t=time(9,i),price=p,volume=1000 if i<2 else future_volume)
          for i,p in enumerate((120,105,104,110,119))]
    d.run_day(rows,[coarse(rows)])
    return b,a,d


def test_catalog_has_native_parity_and_preserves_first_cell(broker_type,tmp_path,monkeypatch,auditor):
    b,a,d=run_population(broker_type,tmp_path/'catalog',monkeypatch)
    nb,_,native=run_case(broker_type,tmp_path/'native',monkeypatch,'hold')
    assert b.equity_curve==nb.equity_curve and b.fills==nb.fills and b.journal==nb.journal
    assert d.probe_selection==native.probe_selection
    assert len(d.population_cells)==1
    selected=next(iter(d.population_cells.values()))
    assert selected['available_at']==d.probe_selection['available_at']
    assert selected['bar']['close']==105
    assert selected['funding_context']['proposed_half_sale_notional']>=100
    assert auditor(b)['errors']==[] and audit_context(b)['pass_']


@pytest.mark.parametrize('action,remaining',[('half',10),('exit',0)])
def test_selected_cell_actions_retain_real_service_and_matched_shadow(broker_type,tmp_path,monkeypatch,auditor,action,remaining):
    _,_,cat=run_population(broker_type,tmp_path/'catalog',monkeypatch)
    cell,selection=next(iter(cat.population_cells.items()))
    hb,_,hold=run_population(broker_type,tmp_path/'hold',monkeypatch,cell=cell,selection=selection)
    b,a,d=run_population(broker_type,tmp_path/action,monkeypatch,action,cell,selection,hold.probe_receipt)
    assert decision_evidence(d.probe_selection)==selection
    assert d.probe_receipt['start_signature']==hold.probe_receipt['start_signature']
    assert d.probe_receipt['no_action_session_signature']==hold.probe_receipt['no_action_session_signature']
    p=a.state.symbols['005930'].position
    assert (p.qty_open if p else 0)==remaining
    assert auditor(b)['errors']==[] and audit_context(b)['pass_']
    assert sum(f.qty for f in b.fills if f.side=='SELL')==20-remaining
    assert all(f.timestamp.time()>=time(9,3) for f in b.fills if f.side=='SELL')
    d.export_probe()


def test_population_not_chosen_by_future_service_or_price(broker_type,tmp_path,monkeypatch):
    _,_,a=run_population(broker_type,tmp_path/'a',monkeypatch)
    _,_,b=run_population(broker_type,tmp_path/'b',monkeypatch,future_volume=1)
    assert a.population_cells==b.population_cells


def test_altered_catalog_is_rejected_before_action(broker_type,tmp_path,monkeypatch):
    _,_,cat=run_population(broker_type,tmp_path/'catalog',monkeypatch)
    cell,selection=next(iter(cat.population_cells.items()))
    bad=deepcopy(selection);bad['quantity']+=1
    with pytest.raises(ValueError,match='sealed native catalog'):
        run_population(broker_type,tmp_path/'wrong',monkeypatch,cell=cell,selection=bad)


@pytest.mark.parametrize('field,value',[('ledger_cash',1e9),('pending_buy_reservations',123),('current_equity',1e9),('cell','wrong')])
def test_independent_context_audit_rejects_tampering(broker_type,tmp_path,monkeypatch,field,value):
    _,_,cat=run_population(broker_type,tmp_path/'catalog',monkeypatch)
    cell,selection=next(iter(cat.population_cells.items()))
    b,_,d=run_population(broker_type,tmp_path/'hold',monkeypatch,cell=cell,selection=selection)
    row=next(r for r in b.journal if r['event']=='funded_probe_boundary_v1')
    row['funding_context'][field]=value
    assert not audit_context(b)['pass_']


def test_funding_state_uses_cash_net_of_remaining_reservations(broker_type,tmp_path,monkeypatch):
    b,a,d=run_population(broker_type,tmp_path,monkeypatch)
    row=next(iter(d.population_cells.values()))
    b.cash=3000
    assert funding_context(d,row)['cell'].endswith('cash_rich')
    monkeypatch.setattr('strategy_olr.core.logic.pending_entry_reservation',lambda ss:(1,1500))
    c=funding_context(d,row)
    assert c['deployable_cash']==1500 and c['cell'].endswith('constrained')


def test_token_reduction_is_not_a_population_cell(broker_type,tmp_path,monkeypatch):
    import backtests.strategies.olr.funded_population as module
    original=module.funding_context
    def small(driver,row):
        c=original(driver,row);c['meaningful_notional_floor']=1e8
        return c
    monkeypatch.setattr(module,'funding_context',small)
    _,_,d=run_population(broker_type,tmp_path,monkeypatch)
    assert d.probe_selection is not None  # Unchanged catalog hold remains valid.
    assert not d.population_cells


def test_entry_evidence_separates_real_funding_from_sizing_without_changed_actions():
    from strategy_olr.core.state import OLRState
    from strategy_olr.core.core_models import OLRPortfolioView
    from tests.strategy_olr.test_capacity_allocation import bar,config,snapshot
    source=Path('strategy_olr/core/allocation.py').read_text(encoding='utf8')
    if "state.meta.get('observe_funded_entry_context_v1')" not in source:source=patch_allocation(source)
    spec=importlib.util.spec_from_loader('strategy_olr.core._population_allocation_fixture',loader=None)
    module=importlib.util.module_from_spec(spec);exec(compile(source,'allocation.py','exec'),module.__dict__)
    old=OLRState();new=OLRState();new.meta['observe_funded_entry_context_v1']=True
    args=((bar(),),config(capacity_joint_allocation=True),snapshot(),OLRPortfolioView(1000,10000))
    a=module.step_olr_batch(old,*args);b=module.step_olr_batch(new,*args)
    assert a.actions==b.actions and a.decisions==b.decisions
    row=new.meta['funded_entry_contexts_v1'][0]
    assert row['ledger_cash']==1000 and row['sizing_equity']==10000
    assert row['actual_deployable_cash']==1000 and row['account']['funding_limit']==1000
    assert row['intents'] and sum(d['reserved_notional'] for d in row['decisions'])<=1000
    assert 'funded_entry_contexts_v1' not in old.meta
