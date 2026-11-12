from dataclasses import replace
from datetime import date,datetime,timedelta,time
import pytest
from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.core.state import KALCBState,KALCBPositionState
from backtests.strategies.kalcb.checkpoint_contrast_labels_v1 import checkpoint_actions,paired_episode,REASON


def state():
    s=KALCBState();ss=s.symbol_state('005930')
    ss.position=KALCBPositionState('005930',10,7,100,datetime(2026,1,5,9,30,tzinfo=KST),90,90,10,'KRX_OR_BREAKOUT',5,
        metadata={'entry_route':'first30_open_anchor'})
    return s,ss


@pytest.mark.parametrize('blocked',['no_position','parent_exit','liquidation','other_route'])
def test_no_new_exit_when_ineligible(blocked):
    s,ss=state()
    if blocked=='no_position':ss.position=None
    elif blocked=='parent_exit':ss.position.exit_in_flight=True
    elif blocked=='liquidation':ss.position.metadata['liquidation_reason']='parent'
    else:ss.position.metadata['entry_route']='checkpoint'
    actions,reason=checkpoint_actions(s,'005930')
    assert actions==[] and reason!='eligible'


def test_partial_entry_cancels_preserves_late_fill_role_and_sells_only_owned_quantity():
    s,ss=state();ss.pending_entry_order_id='original-entry'
    actions,reason=checkpoint_actions(s,'005930')
    assert reason=='eligible' and len(actions)==2
    assert actions[1].qty==7 and actions[1].order_type=='MARKET'
    assert ss.pending_entry_metadata['cancel_late_fill_reason']==REASON
    assert s.order_roles['original-entry']['cancel_late_fill_reason']==REASON
    assert checkpoint_actions(s,'005930')[0]==[]


def fixture():
    from tests.strategy_kalcb.test_core_logic import _snapshot,_or_bars
    day=date(2026,1,5);snapshot=_snapshot(day);prefix=_or_bars(day)[:6]
    cfg=KALCBConfig.from_mapping({'kalcb.entry.plan_mode':'first30_open',
        'kalcb.entry.rvol_threshold':0,'kalcb.entry.cpr_threshold':0,'kalcb.entry.momentum_score_min':0,
        'kalcb.entry.entry_score_blocklist':[],'kalcb.exit.hard_stop_enabled':False,
        'kalcb.exit.quick_exit_enabled':False,'kalcb.exit.failure_stop_enabled':False,
        'kalcb.exit.adaptive_trail_enabled':False,'kalcb.exit.mfe_giveback_enabled':False,
        'kalcb.exit.flow_reversal_enabled':False,'kalcb.exit.mfe_conviction_enabled':False})
    cfg=replace(cfg,flatten_time=time(14,45),entry_action_quantity_scale_enabled=True,
        entry_plan_routes=({'name':'first30_open_anchor','mode':'first30_open'},))
    start=prefix[0].timestamp
    bars=[replace(b,volume=100000) for b in prefix]
    for i in range(6,78):
        price=101. if i<9 else 99.
        bars.append(MarketBar('005930',start+timedelta(minutes=5*i),'5m',price,price+.25,price-.25,price,100000))
    candidate=replace(snapshot.candidates[0],expected_5m_volume=100000.).to_json_dict()
    candidate['metadata']={**candidate['metadata'],'action_quantity_scale':.25}
    row={'opportunity_id':str(day)+'/005930/q.25','trade_date':str(day),'symbol':'005930',
        'candidate':candidate,'feature_available_at':(start+timedelta(minutes=30)).isoformat(),
        'features':{'action_quantity_scale':.25},'action':{'quantity':10}}
    from backtests.strategies.kalcb.executed_value import snapshot_for
    from strategy_kalcb.core.preview import preview_entry
    from strategy_kalcb.core.core_models import KALCBPortfolioView
    available=start+timedelta(minutes=30)
    snap=snapshot_for(str(day),[(row,0)],source='fixture')
    row['action']['quantity']=preview_entry(bars[:6],snap,cfg,KALCBPortfolioView(100000000,equity=100000000),available_at=available)['intents'][0]['quantity']
    context={'available_at':(start+timedelta(minutes=45)).isoformat(),'completed_close':101.,'context_features':{},'own_features':{}}
    indexes={str(day):{(b.symbol,b.timestamp.isoformat(),'5m'):(b.open,b.high,b.low,b.close,b.volume,True) for b in bars}}
    return row,bars,cfg,context,indexes


@pytest.mark.parametrize('scenario',['baseline','combined'])
def test_native_same_prestate_next_bar_gap_cost_and_raw_mtm(scenario):
    row,bars,cfg,context,indexes=fixture()
    pair,detail=paired_episode(row,bars,cfg,scenario,None,[row['trade_date']],context,indexes,bridge=True)
    assert pair['paired_label_valid'] and pair['exact_predecision_identity']
    assert pair['original_unhooked_hold_bridge']
    assert pair['liquidate']['actual_changed_exit_quantity']==pair['hold']['checkpoint']['held_quantity']>0
    exits=[f for f in detail['liquidate']['fills'] if f['reason']==REASON]
    assert exits and all(f['timestamp']>=datetime.fromisoformat(context['available_at']) for f in exits)
    assert max(f['price'] for f in exits)<context['completed_close']
    for arm in ('hold','liquidate'):
        assert detail[arm]['independent_account']['raw_close_marks']==78
        assert pair[arm]['full_portfolio_audit_status']=='unverified'
        assert pair[arm]['net_value_krw']==pytest.approx(detail[arm]['terminal_cash']-100000000)


def test_raw_source_poison_fails():
    row,bars,cfg,context,indexes=fixture()
    key=next(iter(indexes[row['trade_date']]))
    indexes[row['trade_date']][key]=(0,0,0,0,0,True)
    with pytest.raises(ValueError,match='non-source'):
        paired_episode(row,bars,cfg,'baseline',None,[row['trade_date']],context,indexes)


def test_absent_checkpoint_never_fabricates_action_or_label():
    row,bars,cfg,context,indexes=fixture()
    context={**context,'available_at':'2026-01-05T09:46:00+09:00'}
    pair,_=paired_episode(row,bars,cfg,'baseline',None,[row['trade_date']],context,indexes)
    assert pair['eligibility']=='missing_native_clock' and not pair['paired_label_valid']
    assert pair['liquidate_minus_hold_bps'] is None
    assert pair['liquidate']['actual_changed_exit_fills']==0


def test_declared_integer_quantity_poison_fails():
    row,bars,cfg,context,indexes=fixture();row['action']['quantity']+=1
    with pytest.raises(ValueError,match='integer entry quantity'):
        paired_episode(row,bars,cfg,'baseline',None,[row['trade_date']],context,indexes)


def test_bootstrap_import_isolated_before_application_imports():
    import ast
    from pathlib import Path
    tree=ast.parse(Path('scripts/kalcb_checkpoint_contrast_labels_v1.py').read_text())
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            names=[node.module] if isinstance(node,ast.ImportFrom) else [n.name for n in node.names]
            assert all(n.split('.')[0] not in ('backtests','scripts','strategy_kalcb','kis_core') for n in names)
