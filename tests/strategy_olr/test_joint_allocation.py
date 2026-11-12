from dataclasses import replace
from datetime import datetime,timedelta,timezone
from types import SimpleNamespace as N
import json
import pytest
from deployment.olr_kalcb.intent_allocator import AllocationAccount, EntryIntent, IntentAllocationConfig, allocate_entry_intents
from strategy_olr.config import OLRConfig
from strategy_olr.core.joint_allocation import KEY,RECEIPT,account_identity,allocate

NOW=datetime(2025,9,1,6,tzinfo=timezone.utc);H='a'*64

def fixture(cash=1000,slots=None):
    account=AllocationAccount(1000,1000,1000,1000,cash,{'OLR':1000},max_positions=slots or {})
    intents=[EntryIntent(x,'OLR',x,'A',60,10,NOW,NOW+timedelta(minutes=5),native_rank=i+1,minimum_qty=1) for i,x in enumerate(('a','b'))]
    native=allocate_entry_intents(intents,account,NOW,IntentAllocationConfig())
    state=N(meta={},symbols={})
    by_id={x.intent_id:N(symbol=x.symbol,metadata={'estimated_entry_price':10.,'risk_per_share':.1,'sector':'A'}) for x in intents}
    ctx=dict(at=NOW.isoformat(),account_identity=account_identity(account,intents,NOW),scenario_ids=['old1','old2'],probabilities=[.5,.5],offsets=[0,1,2],
        known_at=NOW.isoformat(),source_sha256=H,actual_high_water=1000.,held_net_paths=[[0,0,0],[0,0,0]],meaningful_notional=100,
        constraints={'maximum_tail_drawdown_fraction':.1},actions={})
    for x,gain in [('a',10),('b',30)]:
        ctx['actions'][x]=[dict(action_id=x+str(q),quantity=q,reserve_cash=q*10,net_paths=[[0,-q/10,gain*q/60]]*2,
            known_at=NOW.isoformat(),source_sha256=H,supported=True,completed_entry_probabilities=[int(q>0)]*2) for q in [0,40,60]]
    return account,intents,native,state,by_id,ctx

def test_default_returns_identical_native_decision_object():
    a,i,n,s,b,c=fixture()
    assert allocate(i,n,a,NOW,OLRConfig(),s,b) is n
    assert s.meta=={}

def test_joint_choice_reallocates_shared_cash_and_records_consumed_context():
    a,i,n,s,b,c=fixture();s.meta[KEY]=c
    d=allocate(i,n,a,NOW,OLRConfig(),s,b)
    assert {x.intent_id:x.approved_qty for x in d}=={'a':40,'b':60}
    assert sum(x.reserved_notional for x in d)==1000
    assert KEY not in s.meta and s.meta[RECEIPT]['changed']
    assert json.loads(json.dumps(s.meta))[RECEIPT]['selected_quantities']=={'a':40,'b':60}


def test_persisted_solver_receipt_reproduces_and_detects_changed_cash():
    from strategy_olr.core.joint_allocation import replay_receipt
    from backtests.strategies.olr.economic_policy import EconomicPolicy
    a,i,n,s,b,c=fixture();s.meta[KEY]=c
    allocate(i,n,a,NOW,OLRConfig(),s,b)
    receipt=json.loads(json.dumps(s.meta[RECEIPT]))
    result=replay_receipt(receipt)
    assert result.selected.action_ids==('a40','b60')
    assert receipt['solver_inputs']['account']['cash']==1000
    policy=EconomicPolicy([],OLRConfig())
    policy.record_allocation(receipt)
    persisted=json.loads(json.dumps(policy.trace[0]))
    assert replay_receipt(persisted)==result
    receipt['solver_inputs']['account']['cash']=2000
    with pytest.raises(ValueError,match='Changed solver receipt'):
        replay_receipt(receipt)

def test_native_position_slots_are_a_joint_resource():
    a,i,n,s,b,c=fixture(slots={'OLR':1});s.meta[KEY]=c
    d=allocate(i,n,a,NOW,OLRConfig(),s,b)
    assert {x.intent_id:x.approved_qty for x in d}=={'a':0,'b':60}

def test_native_atr_risk_budgets_still_bind():
    a,i,n,s,b,c=fixture();s.meta[KEY]=c
    cfg=OLRConfig(capacity_joint_allocation=True,capacity_loss_position_pct=.01,capacity_loss_sector_pct=.01,capacity_loss_total_pct=.01,capacity_loss_atr_multiple=1.25)
    d=allocate(i,n,a,NOW,cfg,s,b)
    assert sum(x.approved_qty for x in d)<=60

@pytest.mark.parametrize('change',['cash','at','future','unsupported','quantity','source','missing'])
def test_invalid_context_never_reaches_native_orders(change):
    a,i,n,s,b,c=fixture()
    if change=='cash':a=replace(a,funding_limit=999)
    if change=='at':c['at']=(NOW-timedelta(minutes=1)).isoformat()
    if change=='future':c['known_at']=(NOW+timedelta(minutes=1)).isoformat()
    if change=='unsupported':c['actions']['a'][1]['supported']=False
    if change=='quantity':c['actions']['a'][1]['quantity']=100
    if change=='source':c['source_sha256']=''
    if change=='missing':del c['actions']['a']
    s.meta[KEY]=c
    with pytest.raises(ValueError):allocate(i,n,a,NOW,OLRConfig(),s,b)

def test_infeasible_installed_risk_context_never_restores_native_buys():
    a,i,n,s,b,c=fixture();c['held_net_paths']=[[0,-200,-100]]*2;c['constraints']={'maximum_tail_drawdown_fraction':.01};s.meta[KEY]=c
    assert all(d.approved_qty==0 and d.reserved_notional==0 for d in allocate(i,n,a,NOW,OLRConfig(),s,b))
    assert s.meta[RECEIPT]['fallback']=='no_new_risk_infeasible'
    assert s.meta[RECEIPT]['outcome']['selected'] is None


def test_completed_native_cohort_can_supply_context_without_mutating_account():
    from strategy_olr.core.allocation import step_olr_batch
    from strategy_olr.core.state import OLRState
    from strategy_olr.core.core_models import OLRPortfolioView
    from strategy_common.actions import SubmitEntry
    from tests.strategy_olr.test_capacity_allocation import bar,config,snapshot
    state=OLRState();calls=[]
    def provider(req):
        calls.append(req)
        at=req['at'];actions={}
        for item in req['intents']:
            key=item['intent_id'];q=item['desired_qty'];unit=item['unit_cost']
            actions[key]=[dict(action_id=key+str(x),quantity=x,reserve_cash=x*unit,
                net_paths=[[0,0,float(x)]]*2,known_at=at,source_sha256=H,supported=True,completed_entry_probabilities=[int(x>0)]*2) for x in (0,q)]
        context=dict(at=at,account_identity=req['account_identity'],scenario_ids=['past_a','past_b'],probabilities=[.5,.5],offsets=[0,1,2],
            known_at=at,source_sha256=H,actual_high_water=10000,held_net_paths=[[0,0,0]]*2,meaningful_notional=100,
            constraints={'maximum_tail_drawdown_fraction':.1},actions=actions)
        req['account']['funding_limit']=1e99;req['state']['meta']['tamper']=True
        return context
    result=step_olr_batch(state,(bar(),),config(capacity_joint_allocation=True),snapshot(),OLRPortfolioView(10000,10000),decision_context_provider=provider)
    assert len(calls)==1 and 'tamper' not in state.meta
    entries=[a for a in result.actions if isinstance(a,SubmitEntry)]
    assert len(entries)==1 and entries[0].metadata['reserved_notional']<=10000
    assert entries[0].metadata['joint_decision_response']['context_sha256']==state.meta[RECEIPT]['context_sha256']


def test_explicit_unsupported_provider_preserves_whole_native_batch():
    from strategy_olr.core.allocation import step_olr_batch
    from strategy_olr.core.state import OLRState
    from strategy_olr.core.core_models import OLRPortfolioView
    from tests.strategy_olr.test_capacity_allocation import bar,config,snapshot
    args=((bar(),),config(capacity_joint_allocation=True),snapshot(),OLRPortfolioView(10000,10000))
    baseline=step_olr_batch(OLRState(),*args)
    result=step_olr_batch(OLRState(),*args,decision_context_provider=lambda req:None)
    assert result==baseline
