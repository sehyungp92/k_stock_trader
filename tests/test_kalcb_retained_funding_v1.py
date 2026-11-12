from datetime import datetime
from types import SimpleNamespace
from dataclasses import replace
import pytest

from backtests.strategies.kalcb.retained_funding_v1 import RetainedFundingReplay,reserve


def rows():
    return [dict(symbol=s,qty=q,cost=c,score=v,queue_key=('2026-01-05T09:25:00+09:00','KALCB',s,s))
            for s,q,c,v in [('000001',10,60.,1.),('000002',20,70.,5.),('000003',5,30.,0.)]]


def test_components_change_only_priority_and_keep_whole_quantities():
    queue,left,_=reserve(rows(),100,'queue');forecast,fl,order=reserve(rows(),100,'forecast')
    assert [r['qty'] for r in queue.values()]==[10,0,5] and left==10
    assert order==['000002','000001','000003'] and [forecast[s]['qty'] for s in order]==[20,0,5] and fl==0
    assert all(r['qty'] in (0,r['parent_qty']) for r in forecast.values())


def test_full_book_no_added_profitable_threshold_and_ties_keep_native_queue():
    rr=rows()
    for r in rr:r['score']=-10
    book,left,order=reserve(rr,160,'forecast')
    assert order==['000001','000002','000003'] and all(r['qty'] for r in book.values()) and left==0


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1])
def test_invalid_reservation_cost_rejected(bad):
    rr=rows();rr[0]['cost']=bad
    with pytest.raises(ValueError):reserve(rr,100,'forecast')


def native_signal(row,bars):
    return {'day':row['trade_date'],'origin':'2025-01-01','available_at':row['feature_available_at'],
            'action_id':row['opportunity_id'],'quantity_scale':.25,'predictions':{'baseline':10.,'combined':2.}}


def native_fixture():
    from datetime import date,time,timedelta
    from tests.strategy_kalcb.test_core_logic import _snapshot,_or_bars
    from strategy_common.market import MarketBar
    from strategy_kalcb.config import KALCBConfig
    from backtests.strategies.kalcb.executed_value import snapshot_for
    from strategy_kalcb.core.preview import preview_entry
    from strategy_kalcb.core.core_models import KALCBPortfolioView
    day=date(2026,1,5);snapshot=_snapshot(day);prefix=_or_bars(day)[:6]
    cfg=KALCBConfig.from_mapping({'kalcb.entry.plan_mode':'first30_open',
        'kalcb.entry.rvol_threshold':0,'kalcb.entry.cpr_threshold':0,'kalcb.entry.momentum_score_min':0,
        'kalcb.entry.entry_score_blocklist':[],'kalcb.exit.hard_stop_enabled':False,
        'kalcb.exit.quick_exit_enabled':False,'kalcb.exit.failure_stop_enabled':False,
        'kalcb.exit.adaptive_trail_enabled':False,'kalcb.exit.mfe_giveback_enabled':False,
        'kalcb.exit.flow_reversal_enabled':False,'kalcb.exit.mfe_conviction_enabled':False})
    cfg=replace(cfg,flatten_time=time(14,45),entry_action_quantity_scale_enabled=True,frontier_shadow_enabled=False,
        entry_plan_routes=({'name':'first30_open_anchor','mode':'first30_open'},))
    start=prefix[0].timestamp;bars=[replace(b,volume=100000) for b in prefix]
    for i in range(6,78):
        price=101. if i<9 else 99.
        bars.append(MarketBar('005930',start+timedelta(minutes=5*i),'5m',price,price+.25,price-.25,price,100000))
    candidate=replace(snapshot.candidates[0],expected_5m_volume=100000.).to_json_dict()
    candidate['metadata']={**candidate['metadata'],'action_quantity_scale':.25}
    row={'opportunity_id':str(day)+'/005930/q.25','trade_date':str(day),'symbol':'005930',
        'candidate':candidate,'feature_available_at':(start+timedelta(minutes=30)).isoformat(),
        'features':{'action_quantity_scale':.25},'action':{'quantity':10}}
    snap=snapshot_for(str(day),[(row,0)],source='fixture')
    row['action']['quantity']=preview_entry(bars[:6],snap,cfg,KALCBPortfolioView(100000000,equity=100000000),
        available_at=start+timedelta(minutes=30))['intents'][0]['quantity']
    return row,bars,cfg


def frozen_native(function,*args):
    """Exercise the actual retained policy core, not today's canonical checkout."""
    import subprocess,sys,json,os
    from pathlib import Path
    repo=Path(__file__).resolve().parents[1]
    code=Path('E:/CodexResearch/k_stock_trader/kalcb_structural_preparation_20261016/funded_horizon_v1/runtime/is/inputs/files')
    program='''import sys,importlib.util,runpy,json
from pathlib import Path
code,repo,test,function,args=sys.argv[1:]
sys.path[:0]=[code,repo]
import tests
tests.__path__=[str(Path(repo)/'tests')]
name='backtests.strategies.kalcb.retained_funding_v1'
spec=importlib.util.spec_from_file_location(name,Path(repo)/'backtests/strategies/kalcb/retained_funding_v1.py')
module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
import strategy_kalcb.core.logic as logic
assert Path(logic.__file__).resolve().is_relative_to(Path(code).resolve())
assert 'allocate_quantity' in Path(logic.__file__).read_text(encoding='utf-8')
runpy.run_path(test)[function](*json.loads(args))
'''
    result=subprocess.run([sys.executable,'-B','-c',program,str(code),str(repo),str(Path(__file__).resolve()),function,json.dumps(args)],
                          cwd=code,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},capture_output=True,text=True)
    assert result.returncode==0,result.stdout+'\n'+result.stderr


@pytest.mark.parametrize('scenario',['baseline','combined'])
def test_real_native_preview_preserves_affordable_account_and_restores_hooks(scenario):
    frozen_native('native_affordable',scenario)


def native_affordable(scenario):
    from scripts.kalcb_executed_value_study import run_episode
    from backtests.engine import replay
    from strategy_kalcb.core import account_allocation_v1 as allocation
    row,bars,cfg=native_fixture()
    base,detail=run_episode(row,bars,cfg,scenario,'c11-fixture',None,[row['trade_date']])
    c=RetainedFundingReplay('forecast',{row['trade_date']+'/'+row['symbol']:native_signal(row,bars)})
    kernel=replay.drive_timestamp_batch;quantity=allocation.allocate_quantity
    c.install()
    try:actual,ad=run_episode(row,bars,cfg,scenario,'c11-fixture',None,[row['trade_date']])
    finally:c.restore()
    assert actual==base and ad==detail
    assert c.reservation_records[0]['reason']=='affordable_book_parent_preserved'
    assert replay.drive_timestamp_batch is kernel and allocation.allocate_quantity is quantity
    assert not c.attempts and not c.records


def test_real_native_zero_allocation_has_no_orders_and_suppresses_retry():
    frozen_native('native_zero')


def native_zero():
    from scripts.kalcb_executed_value_study import run_episode
    row,bars,cfg=native_fixture()
    class ForcedRoom(RetainedFundingReplay):
        # Integration fixture supplies a constrained room after native intent preview.
        def decision(self,*args):
            record,_=super().decision(*args)
            book,left,priority=reserve(record['intents'],0,'forecast')
            record.update(book=book,remaining_room=left,priority=priority,changed=True)
            return record,book
    c=ForcedRoom('forecast',{row['trade_date']+'/'+row['symbol']:native_signal(row,bars)})
    c.install()
    try:result,detail=run_episode(row,bars,cfg,'baseline','c11-fixture',None,[row['trade_date']])
    finally:c.restore()
    assert result['label_valid'] and not detail['fills'] and not detail['orders']
    assert row['trade_date']+'/'+row['symbol'] in c.suppressed
    c.allocate=lambda s,q,m:q
    assert c.quantity(row['symbol'],12,{'entry_route':'later60'})==12
    assert c.quantity(row['symbol'],12,{'entry_route':'first30_open_anchor'})==0


def test_frozen_multisymbol_reservation_changes_recipient_without_changing_quantity():
    frozen_native('native_multisymbol')


def native_multisymbol():
    from copy import deepcopy
    from backtests.engine import replay
    from backtests.engine.sim_broker import SimBroker,BrokerCosts
    from backtests.engine.capacity import BarParticipationScenario
    from backtests.strategies.kalcb.runner import KALCBReplayAdapter
    from backtests.strategies.kalcb.executed_value import snapshot_for
    row,bars,cfg=native_fixture();other=deepcopy(row);other['symbol']='005931'
    other['candidate']['symbol']=other['symbol'];other['opportunity_id']=other['trade_date']+'/005931/q.25'
    signals={r['trade_date']+'/'+r['symbol']:native_signal(r,bars) for r in (row,other)}
    signals[other['trade_date']+'/'+other['symbol']]['predictions']={'baseline':20.,'combined':10.}
    class ConstrainedFixture(RetainedFundingReplay):
        def decision(self,*args):
            record,_=super().decision(*args)
            assert len(record['intents'])==2
            room=max(x['cost'] for x in record['intents'])
            book,left,priority=reserve(record['intents'],room,self.reservation_mode)
            record.update(reason='supported_overbooked',room=room,book=book,remaining_room=left,priority=priority,changed=True)
            return record,book
    for mode,expected in [('queue','005930'),('forecast','005931')]:
        costs=BrokerCosts(cfg.commission_bps,cfg.tax_bps_on_sell,3.)
        broker=SimBroker(100_000_000,costs,buying_power_leverage=cfg.intraday_leverage,
                         capacity_policy=BarParticipationScenario(.01,'baseline'))
        snapshot=snapshot_for(row['trade_date'],[(row,0),(other,0)],source='fixture')
        adapter=KALCBReplayAdapter(cfg,{snapshot.trade_date:snapshot},initial_equity=100_000_000,costs=costs)
        c=ConstrainedFixture(mode,signals);c.install()
        try:
            for b in bars:replay.drive_timestamp_batch(broker,(b,replace(b,symbol='005931')),(adapter,))
        finally:c.restore()
        orders=[r['order'] for r in broker.journal if r['event']=='submit' and r['order']['side']=='BUY']
        assert len(orders)==1 and orders[0]['symbol']==expected
        planned={r['symbol']:r['qty'] for r in c.reservation_records[0]['intents']}
        assert orders[0]['qty']==planned[expected]
        assert {f.symbol for f in broker.fills}=={expected} and not broker.positions and not broker.orders


def test_missing_forecast_preserves_whole_book_and_future_or_wrong_quantity_rejected():
    at=datetime.fromisoformat('2026-01-05T09:25:00+09:00')
    o=SimpleNamespace(symbol='000001',order_id='a',strategy_id='KALCB',submitted_at=at,side='BUY',qty=10,
        metadata=dict(decision_available_at='2026-01-05T09:30:00+09:00',entry_route_risk_mult=.5,
                      entry_route_notional_mult=.5,entry_price_ref=10.))
    bb=SimpleNamespace(orders=[o],costs=SimpleNamespace(slippage_bps=3.,commission_bps=2.),last_prices={},
                       _buying_power_available=lambda:50.,_portfolio_equity=lambda:100.)
    c=RetainedFundingReplay('forecast',{});c.day='2026-01-05';c.stamp=at;c.preview=lambda *a:(bb,[o])
    candidate=SimpleNamespace(symbol=o.symbol,metadata={'action_quantity_scale':.5})
    adapter=SimpleNamespace(config=SimpleNamespace(entry_action_quantity_scale_enabled=True),
        snapshots={at.date():SimpleNamespace(candidates=[candidate])},timed_snapshots={})
    record,book=c.decision(adapter,None,None);assert book is None and record['reason']=='unsupported_book_parent_preserved'
    c.signals['2026-01-05/000001']=dict(available_at='2026-01-05T09:35:00+09:00',quantity_scale=.5)
    with pytest.raises(ValueError,match='clock'):c.decision(adapter,None,None)
    c.signals['2026-01-05/000001'].update(available_at='2026-01-05T09:30:00+09:00',quantity_scale=.25)
    with pytest.raises(ValueError,match='scale'):c.decision(adapter,None,None)


def test_preview_exception_restores_controller_observations(monkeypatch):
    from backtests.strategies.kalcb import account_history_v1
    c=RetainedFundingReplay('queue',{});c.records=[{'original':True}];c.attempts={'original'};c.finished={'x'}
    monkeypatch.setattr(account_history_v1,'clone_complete_account',lambda a,b:(a,SimpleNamespace(orders=[])))
    def fail(*args):
        c.records.append({'preview':True});c.attempts.add('preview');raise RuntimeError('preview')
    c.kernel=fail
    with pytest.raises(RuntimeError):c.preview(None,None,())
    assert c.records==[{'original':True}] and c.attempts=={'original'} and c.finished=={'x'}
