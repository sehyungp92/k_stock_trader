from dataclasses import replace
from datetime import date,datetime
from types import SimpleNamespace
import pytest
from strategy_olr.config import OLRConfig
from strategy_olr.models import OLRDailySnapshot
from strategy_olr.core.state import OLRState,OLRPositionState
from strategy_olr.core.core_models import OLRPortfolioView
from strategy_olr.participation_rescue_v5 import TAG,select,confirmed,constrain,envelope
from strategy_olr.research import final_candidate_config_fingerprint
from strategy_common.actions import SubmitEntry
from strategy_common.clock import KST
from deployment.olr_kalcb.intent_allocator import AllocationAccount,EntryIntent,IntentAllocationConfig,allocate_entry_intents
from tests.strategy_olr.test_core_runner import _candidate
from tests.strategy_olr.test_capacity_allocation import bar


def candidate(symbol='005930',**features):
    c=_candidate(symbol,date(2026,1,5),rank=1)
    f=dict(lagged_flow_5d=1.,sector_intraday_ret=.01,sector_intraday_breadth=.5,sector_intraday_effective_count=3.)
    f.update(features)
    return replace(c,symbol=symbol,metadata={**c.metadata,'afternoon_features':f})


def snapshots(candidates=()):
    at=datetime(2026,1,5,14,30,tzinfo=KST)
    base=OLRDailySnapshot(trade_date=date(2026,1,5),generated_at=at,candidates=tuple(candidates),source_fingerprint='test',metadata={'rejected_symbols':{'005930':['sector_participation_below_floor']}})
    return base,replace(base,candidates=(candidate(),))


def test_nonempty_incumbents_are_identical_and_not_tagged():
    base,alt=snapshots((candidate('000660'),))
    result=select(base,alt,replace(OLRConfig(),research_empty_participation_rescue=True))
    assert result.candidates==base.candidates and not result.candidates[0].metadata.get(TAG)
    assert not result.metadata[TAG]['baseline_empty']


def test_empty_exception_requires_sole_gate_and_preserves_rank_order():
    base,alt=snapshots()
    cfg=replace(OLRConfig(),research_empty_participation_rescue=True)
    result=select(base,alt,cfg)
    assert len(result.candidates)==1 and result.candidates[0].metadata[TAG]
    base.metadata['rejected_symbols']['005930'].append('liquidity_below_floor')
    assert not select(base,alt,cfg).candidates


def test_first_six_precede_filtering_and_do_not_promote_seventh():
    base,alt=snapshots()
    alt=replace(alt,candidates=tuple(candidate(f'{i:06d}') for i in range(6))+(candidate(),))
    assert not select(base,alt,OLRConfig()).candidates


@pytest.mark.parametrize('key,value',[('lagged_flow_5d',0),('sector_intraday_ret',0),('sector_intraday_breadth',.4999),('sector_intraday_effective_count',2.99),('lagged_flow_5d',float('nan')),('sector_intraday_ret',float('inf'))])
def test_confirmation_rejects_boundary_failures_and_nonfinite(key,value):
    assert not confirmed(candidate(**{key:value}))


def test_mismatched_timestamp_fails_closed_and_flag_hash_is_bound():
    base,alt=snapshots()
    with pytest.raises(ValueError): select(base,replace(alt,generated_at=alt.generated_at.replace(minute=31)),OLRConfig())
    assert final_candidate_config_fingerprint(OLRConfig())!=final_candidate_config_fingerprint(replace(OLRConfig(),research_empty_participation_rescue=True))


def setup():
    state=OLRState();portfolio=OLRPortfolioView(10000,10000)
    account=AllocationAccount(10000,10000,10000,10000,10000,{'OLR':10000})
    actions=[SubmitEntry('OLR',s,80,'MARKET',None,None,'signal',
        {TAG:True,'estimated_entry_price':100.,'entry_cost_buffer_pct':0.,'target_notional':8000.}) for s in ('005930','000660','035420')]
    return state,portfolio,account,actions


def test_name_and_shared_budget_cannot_be_bypassed_by_three_simultaneous_orders():
    state,p,a,actions=setup()
    proposals,account=constrain(actions,a,state,(),p)
    assert [x.qty for x in proposals]==[5,5,5] and account.funding_limit==1000
    at=datetime(2026,1,5,14,35,tzinfo=KST)
    intents=[EntryIntent(x.symbol,'OLR',x.symbol,'TECH',x.qty,100,at,at,native_rank=i+1) for i,x in enumerate(proposals)]
    decisions=allocate_entry_intents(intents,account,at,IntentAllocationConfig())
    assert sum(d.reserved_notional for d in decisions)<=1000 and all(d.approved_qty<=5 for d in decisions)


def test_holdings_pending_and_partially_filled_reserve_share_one_envelope():
    state,p,a,actions=setup();item=state.symbol_state('005930');now=datetime(2026,1,5,14,35,tzinfo=KST)
    item.position=OLRPositionState('005930',3,100,now,1,1,entry_order_id='b',metadata={TAG:True,'acquired_qty':3})
    item.session_bars=[replace(bar(),close=110)]
    item.pending_entry_order_id='b';item.pending_entry_metadata={TAG:True,'submitted_qty':5,'entry_filled_qty':3,'reserved_notional':500}
    proof=envelope(state,(),p)
    assert proof['held_marked']==330 and proof['pending_reserved']==200 and proof['remaining']==470
    item.pending_exit_order_id='sell'
    assert envelope(state,(),p)==proof
    _,account=constrain(actions,a,state,(),p)
    assert account.funding_limit==470


def test_cash_limit_stays_stricter_and_unflagged_behavior_identical():
    state,p,a,actions=setup();a=replace(a,funding_limit=50)
    _,account=constrain(actions,a,state,(),p);assert account.funding_limit==50
    ordinary=[replace(x,metadata={k:v for k,v in x.metadata.items() if k!=TAG}) for x in actions]
    assert constrain(ordinary,a,state,(),p)==(ordinary,a)
    with pytest.raises(ValueError): constrain([actions[0],ordinary[1]],a,state,(),p)


def test_zero_nav_or_marked_overlimit_cannot_authorize_risk():
    state,p,a,actions=setup()
    proposals,account=constrain(actions,a,state,(),replace(p,equity=0))
    assert account.funding_limit==0 and not proposals


def test_native_batch_enforces_caps_and_disables_recycled_topups():
    from strategy_olr.core.allocation import step_olr_batch
    from tests.strategy_olr.test_capacity_allocation import config,snapshot
    cfg=config(capacity_joint_allocation=True,capacity_redistribute=True,
        capacity_redistribution_evidence='entry',capacity_loss_position_pct=.04,
        capacity_loss_sector_pct=.06,capacity_loss_total_pct=.08)
    snap=snapshot()
    snap=replace(snap,candidates=tuple(replace(c,metadata={**c.metadata,TAG:True}) for c in snap.candidates))
    result=step_olr_batch(OLRState(),(bar('000660'),bar('005930')),cfg,snap,OLRPortfolioView(10000,10000))
    actions=[a for a in result.actions if isinstance(a,SubmitEntry)]
    assert len(actions)==2
    assert sum(a.metadata['reserved_notional'] for a in actions)<=1000
    assert all(a.metadata['reserved_notional']<=500 and a.metadata['allocation_redistributed_qty']==0 for a in actions)
