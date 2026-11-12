from datetime import datetime,timedelta,time,timezone
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from strategy_olr.core.allocation import step_olr_batch
from strategy_olr.core.state import OLRState
from strategy_olr.core.core_models import OLRPortfolioView
from strategy_olr.core.opportunity import KEY
from strategy_olr.core.joint_decision import Account,Action,ScenarioAxis,Constraints,decide
from backtests.strategies.olr.economic_policy import DatedResponseModel
from tests.strategy_olr.test_capacity_allocation import bar,config,snapshot

H='a'*64

def test_watching_opportunity_reconsidered_then_expires_without_refresh():
    state=OLRState();calls=[]
    policy=SimpleNamespace(size_envelope=lambda at:False,on_book=lambda *args:(args[-1],[]))
    def provider(req):
        calls.append(req);at=req['at']
        return dict(at=at,account_identity=req['account_identity'],actual_high_water=10000,
            scenario_ids=['past'],probabilities=[1.],offsets=[0.,1.],known_at=at,source_sha256=H,
            held_net_paths=[[0.,0.]],meaningful_notional=100,constraints={},actions={i['intent_id']:[
                dict(action_id=i['intent_id']+'zero',quantity=0,reserve_cash=0,net_paths=[[0,0]],
                     known_at=at,source_sha256=H,supported=True,completed_entry_probabilities=[0])]
                for i in req['intents']})
    def step(clock):return step_olr_batch(state,(bar(clock=clock),),config(capacity_joint_allocation=True),snapshot(),
        OLRPortfolioView(10000,10000),decision_context_provider=provider,economic_policy=policy)
    step(time(14,30));first=state.meta[KEY]['005930']['expires']
    assert not state.symbols['005930'].entry_attempted
    step(time(14,35));assert len(calls)==2
    assert state.meta[KEY]['005930']['expires']==first
    step(time(14,55));assert len(calls)==2
    assert state.meta[KEY]['005930']['status']=='expired'
    assert state.symbols['005930'].entry_attempted


def test_management_callback_runs_without_any_new_intent():
    calls=[];policy=SimpleNamespace(size_envelope=lambda at:False,on_book=lambda *args:(calls.append(args) or args[-1],[]))
    step_olr_batch(OLRState(),(bar(clock=time(9)),),config(),snapshot(),OLRPortfolioView(10000,10000),economic_policy=policy)
    assert len(calls)==1


def test_breached_drawdown_keeps_actual_peak_and_selects_least_deterioration():
    at=datetime(2025,9,1,tzinfo=timezone.utc);axis=ScenarioAxis(('past',),(1.,),(0.,1.),at,H)
    zero=Action('cash','g',0,0,((0,0),),axis.ids,at,H,completed_entry_probabilities=(0,))
    buy=Action('buy','g',1,20,((0,-5),),axis.ids,at,H,completed_entry_probabilities=(1,))
    result=decide(Account(88,100,88,0),axis,((0,0),),{'g':(zero,buy)},at,
        Constraints(maximum_tail_drawdown_fraction=.08,restore_feasibility=True))
    assert result.reason=='risk_feasibility_restoration'
    assert result.selected.action_ids==('cash',)
    assert result.selected.tail_drawdown_fraction==pytest.approx(.12)


def test_future_and_same_day_labels_do_not_change_fitted_model():
    at=datetime(2025,9,20,tzinfo=timezone.utc)
    rows=[dict(id=str(i),day=f'2025-09-{i:02}',known_at=f'2025-09-{i+1:02}T00:00:00+00:00',
        kind='entry',scenario='baseline',x=[i/10,1],y=[i/100,0]) for i in range(1,10)]
    fit=DatedResponseModel(rows,minimum_dates=3).fit(at,'entry','baseline')
    future={**rows[0],'id':'future','known_at':at.isoformat(),'y':[1e6,-1e6]}
    other=DatedResponseModel(rows+[future],minimum_dates=3).fit(at,'entry','baseline')
    assert fit['hash']==other['hash'];np.testing.assert_array_equal(fit['beta'],other['beta'])


def test_recovery_budget_prevents_permanent_cash_lock_without_resetting_peak():
    at=datetime(2025,9,1,tzinfo=timezone.utc);axis=ScenarioAxis(('past',),(1.,),(0.,1.,2.),at,H)
    zero=Action('cash','g',0,0,((0,0,0),),axis.ids,at,H,completed_entry_probabilities=(0,))
    buy=Action('buy','g',1,20,((0,-1,10),),axis.ids,at,H,completed_entry_probabilities=(1,))
    risk=Constraints(maximum_tail_drawdown_fraction=.08,restore_feasibility=True,recovery_tail_budget_fraction=.02)
    out=decide(Account(88,100,88,0),axis,((0,0,0),),{'g':(zero,buy)},at,risk)
    assert out.reason=='bounded_risk_recovery'
    assert out.selected.action_ids==('buy',)
    assert out.selected.tail_drawdown_fraction==pytest.approx(.13)
    assert out.selected.tail_drawdown_fraction<=.12+.02*.88


def test_committing_entry_blocks_duplicate_proposals():
    policy=SimpleNamespace(size_envelope=lambda at:False,on_book=lambda *args:(args[-1],[]))
    state=OLRState();args=(config(),snapshot(),OLRPortfolioView(10000,10000))
    first=step_olr_batch(state,(bar(),),*args,economic_policy=policy)
    assert first.actions and state.symbols['005930'].entry_attempted
    second=step_olr_batch(state,(bar(clock=time(14,35)),),*args,economic_policy=policy)
    assert not second.actions


def test_opportunity_wait_deadline_does_not_shorten_native_acquisition():
    policy=SimpleNamespace(size_envelope=lambda at:False,on_book=lambda *args:(args[-1],[]))
    state=OLRState()
    result=step_olr_batch(state,(bar(),),config(),snapshot(),OLRPortfolioView(10000,10000),economic_policy=policy)
    entry=result.actions[0]
    row=state.meta[KEY][entry.symbol]
    assert entry.metadata['expiry_ts']==row['action']['metadata']['expiry_ts']
    assert datetime.fromisoformat(entry.metadata['expiry_ts'])>datetime.fromisoformat(row['expires'])


def test_zero_capacity_signal_waits_without_any_order_or_reserved_cash(monkeypatch):
    from strategy_olr.core import capacity,logic
    def forecast(state,bar,cfg):
        return (0 if bar.timestamp.minute==30 else 10),None,{'exit':{}}
    monkeypatch.setattr(logic,'capacity_forecast',forecast)
    monkeypatch.setattr(capacity,'forecast',forecast)
    policy=SimpleNamespace(size_envelope=lambda at:False,on_book=lambda *args:(args[-1],[]))
    state=OLRState();cfg=config(capacity_entry_bars=3)
    first=step_olr_batch(state,(bar(),),cfg,snapshot(),OLRPortfolioView(10000,10000),economic_policy=policy)
    assert not first.actions
    ss=state.symbols['005930']
    assert not ss.entry_attempted and not ss.pending_entry_order_id
    assert logic.pending_entry_reservation(ss)==(0,0.)
    assert state.meta[KEY]['005930']['status']=='watching'
    second=step_olr_batch(state,(bar(clock=time(14,35)),),cfg,snapshot(),OLRPortfolioView(10000,10000),economic_policy=policy)
    assert len(second.actions)==1 and second.actions[0].qty==10
    assert state.meta[KEY]['005930']['status']=='committed'
