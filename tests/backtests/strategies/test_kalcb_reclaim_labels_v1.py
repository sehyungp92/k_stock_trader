from copy import deepcopy
from dataclasses import replace
from datetime import date,datetime,timedelta,time
import importlib.util
from pathlib import Path
import pytest
from strategy_common.clock import KST
from strategy_common.market import MarketBar
from backtests.strategies.kalcb.reclaim_labels_v1 import geometry
from backtests.strategies.kalcb.reclaim_source_audit_v1 import verify_rows


def bars():
    from tests.strategy_kalcb.test_core_logic import _pathproof_bars
    day=date(2026,1,5);rows=_pathproof_bars(day)
    values=[(101,102,100,101)]*4+[(104,106,103.5,105.5),(105.5,107,105,106.5)]
    for i,(o,h,l,c) in enumerate(values,12):
        rows.append(MarketBar('005930',datetime(2026,1,5,9,tzinfo=KST)+timedelta(minutes=5*i),'5m',o,h,l,c,5000))
    return [replace(b,open=50+b.open/2,high=50+b.high/2,low=50+b.low/2,close=50+b.close/2) for b in rows]


def test_registered_census_predicate_and_future_independence():
    path=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011/v17_reclaim_investigation/census.py')
    spec=importlib.util.spec_from_file_location('registered_census',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=bars();at=rows[0].timestamp+timedelta(minutes=90)
    serialized=[{'ts':b.timestamp.isoformat(),'o':b.open,'h':b.high,'l':b.low,'c':b.close,'v':b.volume} for b in rows]
    assert geometry(rows,at)[1] is module.classify(serialized,rows[0].timestamp)['qualified'] is True
    poison=replace(rows[-1],timestamp=at,open=1e9,high=1e12,close=1e10,volume=1e15)
    assert geometry(rows+[poison],at)==geometry(rows,at)


@pytest.mark.parametrize('index',[0,6,12,16,17])
def test_missing_prefix_rejected(index):
    rows=bars();at=rows[0].timestamp+timedelta(minutes=90);rows.pop(index)
    with pytest.raises(ValueError,match='Incomplete'):geometry(rows,at)


def fixture():
    rows=bars();day=rows[0].timestamp.date();at=rows[0].timestamp+timedelta(minutes=90)
    original=[{'trade_date':str(day),'symbol':s,'sector':'S'} for s in ('005930','B')]
    indexed={(b.symbol,b.timestamp.isoformat(),'5m'):(b.open,b.high,b.low,b.close,b.volume,True) for b in rows}
    indexed.update({('B',b.timestamp.isoformat(),'5m'):(b.open,b.high,b.low,b.close,b.volume,True) for b in rows[:-1]})
    feature={**original[0],'feature_available_at':at.isoformat(),'guard_at_rank1':True,
        'opening_bars':[dict(symbol=b.symbol,timestamp=b.timestamp.isoformat(),**{k:getattr(b,k) for k in ('open','high','low','close','volume')}) for b in rows],
        'features':{'checkpoint_sector_return':rows[-1].close/rows[0].open-1,'checkpoint_sector_breadth':1.}}
    return original,[feature],indexed,{day},[]


def test_independent_complete_population_and_sector():
    proof=verify_rows(*fixture());assert proof['complete_consumed_prefixes']==1
    assert proof['excluded']==[('B','2026-01-05','missing')]


@pytest.mark.parametrize('mutation',['source_price','sector','geometry','clock','missing_row','invalid_population','future_candidate'])
def test_source_poison_rejection(mutation):
    original,features,indexed,days,snapshots=fixture()
    if mutation=='source_price':features[0]['opening_bars'][-1]['close']+=1
    elif mutation=='sector':features[0]['features']['checkpoint_sector_return']+=.01
    elif mutation=='geometry':features[0]['guard_at_rank1']=False
    elif mutation=='clock':features[0]['feature_available_at']='2026-01-05T10:35:00+09:00'
    elif mutation=='missing_row':features=[]
    elif mutation=='invalid_population':features.append({**deepcopy(features[0]),'symbol':'B'})
    elif mutation=='future_candidate':snapshots=[{'trade_date':'2026-01-05','candidates':[{'symbol':'005930','metadata':{'entry_checkpoint_at':'2026-01-05T10:35:00+09:00'}}]}]
    with pytest.raises(ValueError):verify_rows(original,features,indexed,days,snapshots)


def native_fixture():
    from tests.strategy_kalcb.test_checkpoint_entry import setup
    snap,_,cfg,_=setup();rows=bars();at=rows[0].timestamp+timedelta(minutes=90)
    candidate=replace(snap.candidates[0],metadata={**snap.candidates[0].metadata,'entry_checkpoint_at':at.isoformat(),
        'feature_available_at':at.isoformat(),'action_quantity_scale':.25})
    return replace(snap,candidates=(candidate,),generated_at=at),rows,replace(cfg,entry_action_quantity_scale_enabled=True),at


def test_actual_next_bar_gap_fill_and_no_early_fill():
    from strategy_kalcb.core.preview import preview_entry
    from strategy_kalcb.core.core_models import KALCBPortfolioView
    from backtests.strategies.kalcb.runner import KALCBReplayAdapter
    from backtests.engine.sim_broker import SimBroker,BrokerCosts
    from backtests.engine.replay import drive_timestamp_batch
    snap,rows,cfg,at=native_fixture()
    seen=preview_entry(rows,snap,cfg,KALCBPortfolioView(1000000,equity=1000000),available_at=at)
    assert seen['ready'],seen['decisions']
    base=replace(snap,candidates=(),generated_at=at.replace(hour=0,minute=0));costs=BrokerCosts(1,2,3)
    broker=SimBroker(1000000,costs);adapter=KALCBReplayAdapter(cfg,{snap.trade_date:base},initial_equity=1000000,costs=costs,
        timed_snapshots={snap.trade_date:[snap]},observation_symbols=('005930',))
    for b in rows:drive_timestamp_batch(broker,(b,),(adapter,))
    assert not broker.fills and broker.orders
    nxt=replace(rows[-1],timestamp=at,open=104.5,high=105,low=104.25,close=104.75)
    drive_timestamp_batch(broker,(nxt,),(adapter,))
    fills=[f for f in broker.fills if f.side=='BUY'];assert fills
    assert all(f.timestamp>=at and f.metadata['base_price']==104.5 for f in fills)
    assert fills[0].price==pytest.approx(104.5*1.0003)
    assert not broker.same_bar_fill_violations


def test_isolated_auditor_reuses_native_primitive_and_keeps_full_audit_truthful():
    import inspect
    from backtests.strategies.kalcb.reclaim_labels_v1 import audited_episode,labels
    source=inspect.getsource(audited_episode)
    assert 'run_episode(row,bars,cfg,scenario,source,conn,sessions)' in source
    assert 'audit_broker(broker)' in source and 'full_native_portfolio_qualification=False' in source
    assert "rows=[r for r in group if r['guard_at_rank1']]" in inspect.getsource(labels)


def test_runtime_bootstrap_has_no_prebootstrap_application_import():
    import ast
    path=Path('scripts/kalcb_reclaim_producer_v1.py');tree=ast.parse(path.read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            names=[node.module] if isinstance(node,ast.ImportFrom) else [x.name for x in node.names]
            assert not any(n.split('.')[0] in ('backtests','scripts','strategy_kalcb','kis_core') for n in names)


@pytest.mark.parametrize('scenario',['baseline','combined'])
def test_actual_isolated_observer_identity_and_independent_account(monkeypatch,scenario):
    from backtests.strategies.kalcb import reclaim_labels_v1 as module,executed_value as ev
    from scripts.kalcb_executed_value_study import run_episode
    from scripts.kalcb_action_value_study import action_features
    from strategy_kalcb.core.preview import preview_entry
    from strategy_kalcb.core.core_models import KALCBPortfolioView
    snap,prefix,cfg,at=native_fixture();cfg=replace(cfg,flatten_time=time(14,45),exit_hard_stop_enabled=False);day=str(snap.trade_date)
    preview=preview_entry(prefix,snap,cfg,KALCBPortfolioView(ev.REFERENCE_EQUITY,equity=ev.REFERENCE_EQUITY),available_at=at)
    assert preview['ready']
    row={'opportunity_id':day+'/005930/minute90/q0.25','base_opportunity_id':day+'/005930/minute90',
         'trade_date':day,'symbol':'005930','candidate':snap.candidates[0].to_json_dict(),
         'feature_available_at':at.isoformat(),'features':action_features(preview,prefix,.25),
         'action':{'ready':True,'quantity':sum(a['quantity'] for a in preview['intents'])}}
    replay=prefix+[replace(prefix[-1],timestamp=prefix[0].timestamp+timedelta(minutes=5*i),
        open=104.5,high=105,low=104.25,close=104.75,volume=100000) for i in range(18,78)]
    index={(b.symbol,b.timestamp.isoformat(),'5m'):(b.open,b.high,b.low,b.close,b.volume,True) for b in replay}
    monkeypatch.setattr(module,'independent_day_index',lambda p,d:index)
    expected,events=run_episode(row,replay,cfg,scenario,'test-source',None,[day])
    actual,details=module.audited_episode(row,replay,cfg,scenario,'test-source',None,[day],{})
    assert expected['status']=='filled' and actual['independent_episode_execution_pass'],details['independent_engine_audit']
    assert {k:actual[k] for k in expected}==expected
    assert details['fills']==events['fills']
    assert actual['raw_execution_bars_verified']==78
    assert not actual['full_native_portfolio_qualification']
    assert actual['net_value_planned_notional_bps']==pytest.approx(expected['net_value_krw']/actual['planned_notional_krw']*10000)
    assert details['independent_engine_audit']['audit_status']=='unverified'
    index[(replay[-1].symbol,replay[-1].timestamp.isoformat(),'5m')]=(1,2,1,2,100,True)
    with pytest.raises(ValueError,match='non-source'):module.audited_episode(row,replay,cfg,scenario,'test-source',None,[day],{})
