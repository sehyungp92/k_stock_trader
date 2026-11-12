from dataclasses import replace
from datetime import date,datetime,timedelta
import copy,json
from types import SimpleNamespace
import pytest
from strategy_olr.config import OLRConfig
from strategy_olr.core.state import OLRState
from strategy_olr.core.serializers import snapshot_state,restore_state
from strategy_olr.core.liquidation_tail_v3 import KEY,advance,empty,scenario
from strategy_olr.core.risk_repair_v4 import quote,joint_stress,largest_feasible
from strategy_olr.core.loss_budget import allocate,unit_loss
from strategy_olr.research import afternoon_selection_from_contexts,final_candidate_config_fingerprint,_afternoon_reject_reasons
from strategy_olr.models import OLRDailySnapshot,OLRAfternoonContext
from strategy_olr.flow_rescue_v4 import eligible
from tests.strategy_olr.test_priority_loss_budget import risk_setup
from tests.strategy_olr.test_core_runner import _candidate
from strategy_common.clock import KST

def seed(state,volume=10000,days=30):
    rows=[];sessions=[]
    for i in range(days):
        day=date(2025,12,1)+timedelta(days=i);prev=day-timedelta(days=1);sessions.append(str(day))
        for symbol,sector,vol in [('005930','TECH',volume),('000660','TECH',10000),('035420','IT',10000),('000001','TECH',100),('000002','TECH',100),('000003','TECH',100)]:
            rows.append(dict(symbol=symbol,sector=sector,day=str(day),previous_day=str(prev),reference_price=100.,
                reference_at=f'{prev}T14:25:00+09:00',available_at=f'{day}T15:35:00+09:00',
                liquidation=[[f'{day}T09:{m:02d}:00+09:00',90.,vol,f'{day}T09:{m-5:02d}:00+09:00'] for m in (5,10,15)]))
    state.meta[KEY]=advance(empty(),rows,sessions[-1],sessions);return state

def setup(mode='quantity'):
    cfg,state,account,at,actions,intents=risk_setup(capacity_loss_position_pct=.04,capacity_loss_sector_pct=.06,
        capacity_loss_total_pct=.08,capacity_liquidation_tail=True,capacity_risk_repair_mode=mode)
    seed(state)
    for key,a in actions.items():
        ev=quote(state,a.symbol,a.metadata['sector'],at,6500,cfg.capacity_participation,mode)
        actions[key]=replace(a,metadata={**a.metadata,'liquidation_tail_evidence':ev})
    return cfg,state,account,at,actions,intents

@pytest.mark.parametrize('mode',['quantity','separated'])
def test_solver_matches_exhaustive_feasible_integer_and_preserves_all_budgets(mode):
    cfg,state,account,at,actions,intents=setup(mode)
    a=actions[intents[0].intent_id]
    values=[]
    for q in range(1,66):
        ev=quote(state,a.symbol,a.metadata['sector'],at,q*100,cfg.capacity_participation,mode)
        risk=q*unit_loss({**a.metadata,'liquidation_tail_evidence':ev},cfg)
        if risk<=400+1e-7:values.append(q)
    decisions=allocate(intents,account,at,cfg,state,actions)
    assert decisions[0].approved_qty==max(values)
    used=[(x,actions[x.intent_id],x.approved_qty*unit_loss(actions[x.intent_id].metadata,cfg)) for x in decisions]
    assert sum(risk for _,_,risk in used)<=800+1e-7
    assert sum(risk for _,a,risk in used if a.metadata['sector']=='TECH')<=600+1e-7
    assert all(risk<=400+1e-7 for _,_,risk in used)
    for decision,a,_ in used:
        if decision.approved_qty:assert a.metadata['liquidation_tail_evidence']['notional_upper']==100*decision.approved_qty

def test_separation_never_inherits_peer_full_notional_impairment_or_dilutes_own():
    cfg,state,account,at,actions,intents=setup()
    old=quote(state,'005930','TECH',at,6500,.01,'quantity')
    new=quote(state,'005930','TECH',at,6500,.01,'separated')
    assert old['peer_incomplete_scenarios']>0
    assert new['peer_names']>=3 and new['peer_dates']==30
    assert new['fraction']==pytest.approx(.1) and new['fraction']<old['fraction']
    seed(state,volume=100)
    illiquid=quote(state,'005930','TECH',at,6500,.01,'separated')
    assert illiquid['fraction']>=illiquid['symbol_tail_mean']>.9

def test_sparse_sector_falls_back_and_uses_unique_dates():
    cfg,state,account,at,actions,intents=setup('separated')
    ev=quote(state,'035420','IT',at,6500,.01,'separated')
    assert ev['pool']=='universe_price_peers' and ev['peer_dates']==30

@pytest.mark.parametrize('mode',['quantity','separated','portfolio'])
def test_prior_history_restart_equality_and_future_rejection(mode):
    cfg,state,account,at,actions,intents=setup(mode)
    restored=restore_state(json.loads(json.dumps(snapshot_state(state))))
    assert quote(restored,'005930','TECH',at,4000,.01,mode)==quote(state,'005930','TECH',at,4000,.01,mode)
    state.meta[KEY]['through']=at.date().isoformat()
    with pytest.raises(ValueError):quote(state,'005930','TECH',at,4000,.01,mode)

def test_topups_requote_total_and_do_not_double_count_or_bypass_base_risk():
    cfg,state,account,at,actions,intents=setup('quantity')
    intent=replace(intents[0],desired_qty=3)
    base=allocate([intent],account,at,cfg,state,actions)
    cash=base[0].reserved_notional
    remaining=replace(account,funding_limit=10000-cash,gross_committed=cash,strategy_committed={'OLR':cash})
    extra=allocate([replace(intent,desired_qty=62)],remaining,at,cfg,state,actions,prior=base)
    total=base[0].approved_qty+extra[0].approved_qty
    m=actions[intent.intent_id].metadata
    assert m['risk_repair_evidence']['total_quote_quantity']==total
    assert m['liquidation_tail_evidence']['notional_upper']==total*100
    assert total*unit_loss(m,cfg)<=400+1e-7
    single_cfg,single_state,single_account,single_at,single_actions,single_intents=setup('quantity')
    single=allocate(single_intents[:1],single_account,single_at,single_cfg,single_state,single_actions)
    assert total==single[0].approved_qty

def test_joint_scenarios_keep_common_dates_and_include_costs():
    cfg,state,account,at,actions,intents=setup('portfolio')
    # Anti-aligned shocks must not be summed as simultaneous maxima.
    for r in state.meta[KEY]['observations']:
        if r['symbol'] in ('005930','000660'):
            bad=(int(r['day'][-2:])%2==0)==(r['symbol']=='005930')
            for b in r['liquidation']:b[1]=80. if bad else 100.
    positions={s:dict(sector='TECH',notional=1000.,cost_fraction=.01) for s in ('005930','000660')}
    out=joint_stress(state,at,positions,.01)
    assert out['common_dates']==30 and out['total_loss']==pytest.approx(220.)
    assert out['sector_losses']['TECH']==pytest.approx(220.)
    state.meta[KEY]['observations']=[r for r in state.meta[KEY]['observations'] if r['symbol']!='000660' or int(r['day'][-2:])<=10]
    assert not joint_stress(state,at,positions,.01)['supported']

def test_joint_overlay_limits_new_risk_and_emits_audit_evidence():
    cfg,state,account,at,actions,intents=setup('portfolio')
    for r in state.meta[KEY]['observations']:
        for x in r['liquidation']:x[1]=50.
    decisions=allocate(intents,account,at,cfg,state,actions)
    assert decisions[0].approved_qty==12 # 6% sector / 50%-price-loss / share
    assert sum(d.approved_qty for d in decisions)==16 # 8% portfolio
    for d in decisions:
        if d.approved_qty:
            ev=actions[d.intent_id].metadata['risk_repair_evidence']['joint_stress']
            assert ev['total_loss']<=800+1e-7 and all(v<=600+1e-7 for v in ev['sector_losses'].values())

def test_disabled_config_identity_and_enabled_feature_identity():
    cfg=OLRConfig();values={k:getattr(cfg,k) for k in cfg.__dataclass_fields__ if k not in ('capacity_risk_repair_mode','afternoon_empty_sector_flow_rescue')}
    legacy=SimpleNamespace(**values,afternoon_empty_sector_flow_rescue=False,__dataclass_fields__=values)
    assert final_candidate_config_fingerprint(cfg)==final_candidate_config_fingerprint(legacy)
    assert final_candidate_config_fingerprint(replace(cfg,afternoon_empty_sector_flow_rescue=True))!=final_candidate_config_fingerprint(cfg)
    with pytest.raises(ValueError):replace(cfg,capacity_risk_repair_mode='quantity')
    with pytest.raises(ValueError):replace(cfg,capacity_risk_repair_mode='unregistered')

def rescue_setup():
    day=date(2026,1,8);candidate=_candidate('005930',day,rank=1)
    ctx=OLRAfternoonContext(day,candidate.symbol,candidate,.02,.01,0.,1.2,.7,-.001,.03,-.001,1.,100.,66,
        lagged_flow_5d=.1,lagged_sector_flow_5d=-.1,intraday_sector_ret=.01,intraday_sector_breadth=.6,intraday_sector_effective_count=5)
    cfg=OLRConfig(afternoon_empty_sector_flow_rescue=True,afternoon_min_sector_flow=0.,afternoon_min_flow_5d=0.)
    snap=OLRDailySnapshot(day,(candidate,),'test',datetime(2026,1,8,8,tzinfo=KST))
    return cfg,snap,{candidate.symbol:ctx}

def test_rescue_is_causal_empty_only_and_runs_downstream_score_gate():
    cfg,snap,contexts=rescue_setup();ctx=contexts['005930']
    assert _afternoon_reject_reasons(ctx,replace(cfg,afternoon_empty_sector_flow_rescue=False))==['lagged_sector_flow_below_floor']
    result=afternoon_selection_from_contexts(snap,contexts,cfg)
    assert len(result.candidates)==1 and result.candidates[0].metadata['empty_sector_flow_rescue_v4']
    rejected=afternoon_selection_from_contexts(snap,contexts,replace(cfg,afternoon_min_score=1e9,afternoon_max_score=1e10))
    assert not rejected.candidates and rejected.metadata['afternoon_rejected_symbols']['005930']==['afternoon_score_below_floor']
    native_contexts={'005930':replace(ctx,lagged_sector_flow_5d=.1)}
    on=afternoon_selection_from_contexts(snap,native_contexts,cfg)
    off=afternoon_selection_from_contexts(snap,native_contexts,replace(cfg,afternoon_empty_sector_flow_rescue=False))
    assert on.candidates==off.candidates and not on.metadata['empty_sector_flow_rescue_v4']['rescued_symbols']

@pytest.mark.parametrize('field,value',[('lagged_flow_5d',0.),('intraday_sector_ret',0.),('intraday_sector_breadth',.49),('intraday_sector_effective_count',2),('intraday_sector_ret',float('nan'))])
def test_rescue_requires_all_registered_confirmation_evidence(field,value):
    cfg,snap,contexts=rescue_setup();ctx=replace(contexts['005930'],**{field:value})
    assert not eligible(ctx,['lagged_sector_flow_below_floor'])
    assert not eligible(contexts['005930'],['lagged_sector_flow_below_floor','vwap_ret_below_floor'])
