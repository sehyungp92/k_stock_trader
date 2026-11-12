"""Fine source audit reparses bytes, preserves scope and rejects omissions."""
from dataclasses import replace
from datetime import datetime,time
import hashlib,json,sqlite3
import pandas as pd
import pytest

from backtests.core.input_manifest import InputManifest,signature
from backtests.engine.olr_minute_source_v5 import load_minute_day,verify_minute_sources
from backtests.engine.olr_minute_audit_v5 import minute_nav_evidence
from backtests.engine.olr_minute_v5 import MinuteOLRDriver,MinuteParticipationScenario
from backtests.engine.sim_broker import SimBroker,BrokerCosts
from backtests.strategies.olr.runner import OLRReplayAdapter
from strategy_olr.models import OLRDailySnapshot
from strategy_common.clock import KST
from tests.strategy_olr.test_minute_exit_v5 import DAY,PLAN,minute,coarse,config


def fixture(tmp_path):
    files=tmp_path/'files';files.mkdir()
    rows=[replace(minute(t=time(9,i)),symbol='000100') for i in range(5)]
    frame=pd.DataFrame([dict(timestamp=b.timestamp,open=b.open,high=b.high,low=b.low,close=b.close,volume=b.volume) for b in rows])
    frame.to_parquet(files/'raw.parquet',index=False);digest=hashlib.sha256((files/'raw.parquet').read_bytes()).hexdigest()
    raw_original='C:/minute_fixture/raw.parquet'
    plan=dict(sources={},windows={'train':dict(primary={'000100':dict(path=raw_original,sha256=digest)},whole_session_overrides=[])})
    definition=dict(version='olr-minute-source-definition-v5',source_plan='plan.json',window='train',
        aliases={raw_original:'raw.parquet'},resolution_receipts=[],canonical_database='coarse.sqlite',minute_database='fine.sqlite')
    for name,data in (('plan.json',plan),('definition.json',definition)):(files/name).write_text(json.dumps(data),encoding='utf-8')
    c=sqlite3.connect(files/'coarse.sqlite');c.execute('CREATE TABLE bars(symbol TEXT,day TEXT,ts TEXT,o REAL,h REAL,l REAL,c REAL,v REAL)')
    b=coarse(rows);c.execute('INSERT INTO bars VALUES(?,?,?,?,?,?,?,?)',(b.symbol,str(DAY),b.timestamp.isoformat(),b.open,b.high,b.low,b.close,b.volume));c.commit();c.close()
    c=sqlite3.connect(files/'fine.sqlite');c.execute('CREATE TABLE bars(symbol TEXT,day TEXT,ts TEXT,o REAL,h REAL,l REAL,c REAL,v REAL,source_hash TEXT)')
    c.executemany('INSERT INTO bars VALUES(?,?,?,?,?,?,?,?,?)',[(b.symbol,str(DAY),b.timestamp.isoformat(),b.open,b.high,b.low,b.close,b.volume,digest) for b in rows]);c.commit()
    loaded=load_minute_day(c,DAY,{'000100'});c.close()
    payload=dict(source_root=str(files),source_generation_hash='unit_test_only',files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files.iterdir()},evidence={'shared_market_inputs':{}})
    (tmp_path/'manifest.json').write_text(json.dumps({**payload,'manifest_hash':signature(payload)}),encoding='utf-8')
    generation=InputManifest(tmp_path/'manifest.json')
    adapter=OLRReplayAdapter(config(capacity_joint_allocation=True,capacity_exit_priority=True,trade_exit_plan=PLAN),
        {DAY:OLRDailySnapshot(DAY,(),'source_fixture',datetime.combine(DAY,time(8),tzinfo=KST))})
    broker=SimBroker(initial_equity=10000,costs=BrokerCosts(0,0,0),capacity_policy=MinuteParticipationScenario(.01,'test'))
    driver=MinuteOLRDriver(broker,adapter,fast_exit=True);driver.run_day(loaded,[coarse(rows)],symbols={'000100'})
    identity={'configuration':{'date_range':{'start':str(DAY),'end':str(DAY)},
        'minute_clock':{'version':'olr-minute-clock-v5','source_definition':'definition.json','fast_exit':True}}}
    return broker,generation,{'minute_source_definition':'definition.json'},identity


def test_every_minute_is_compared_with_original_frozen_bytes(tmp_path):
    args=fixture(tmp_path);proof=verify_minute_sources(*args)
    assert proof['passed'] and proof['consumed_minute_rows']==5
    assert proof['required_symbol_sessions']==1 and proof['origins_reparsed']
    nav=minute_nav_evidence(args[0]);assert nav['all_event_drawdown_pct']==0
    assert nav['points']==10 and nav['events']=={'opening_marks':5,'closing_marks':5}


@pytest.mark.parametrize('tamper',['omit','alter','scope','clock','raw'])
def test_source_disposition_or_raw_bytes_cannot_be_changed(tmp_path,tamper):
    broker,generation,evidence,identity=fixture(tmp_path)
    if tamper=='omit':broker.expected_batches.pop(2)
    elif tamper=='alter':broker.expected_batches[2][0]['volume']+=1
    elif tamper=='scope':broker.minute_day_scopes[0]['symbols'].append('005930')
    elif tamper=='clock':identity['configuration']['minute_clock']['fast_exit']=False
    else:
        p=generation.root/'raw.parquet';frame=pd.read_parquet(p);frame.loc[0,'volume']+=1;frame.to_parquet(p,index=False)
    with pytest.raises((AssertionError,ValueError)):verify_minute_sources(broker,generation,evidence,identity)


def nav_broker(journal):
    from types import SimpleNamespace
    return SimpleNamespace(initial_equity=10000,minute_coarse_nav=[],journal=journal)


def mark(kind,stamp,price):
    return dict(event=kind,timestamp=stamp,prices={'000100':price})


def purchase(stamp):
    return dict(event='fill',fill=dict(symbol='000100',side='BUY',qty=10,price=100,
        timestamp=stamp,metadata={'fee':10}))


def test_common_grid_includes_completed_close_but_not_next_open():
    a='2026-06-24T09:00:00+09:00';b='2026-06-24T09:01:00+09:00'
    broker=nav_broker([mark('opening_marks',a,100),purchase(a),mark('closing_marks',a,102),
        mark('opening_marks',b,80),mark('closing_marks',b,80)])
    proof=minute_nav_evidence(broker,[['2026-06-24T08:56:00+09:00',123]])
    assert proof['common_original_grid'][0]['equity']==10010
    assert proof['common_original_drawdown_pct']==0
    assert proof['all_event_drawdown_pct']>2
    assert proof['points']==5  # The reporting sample cannot erase a fine trough.


def test_corrected_auction_never_leaks_into_an_earlier_grid_point():
    a='2026-06-24T15:19:00+09:00';b='2026-06-24T15:32:00+09:00'
    broker=nav_broker([mark('opening_marks',a,100),purchase(a),mark('closing_marks',a,100),
        mark('opening_marks',b,100),mark('closing_marks',b,80)])
    proof=minute_nav_evidence(broker,[['2026-06-24T15:25:00+09:00',0],['2026-06-24T15:30:00+09:00',0]])
    assert [r['equity'] for r in proof['common_original_grid']]==[9990,9790]
    assert proof['common_original_drawdown_pct']==pytest.approx(2.1)


def test_common_grid_and_fine_chronology_cannot_be_reordered():
    a='2026-06-24T09:00:00+09:00';b='2026-06-24T09:01:00+09:00'
    with pytest.raises(ValueError):minute_nav_evidence(nav_broker([]),[[b,0],[a,0]])
    with pytest.raises(ValueError):minute_nav_evidence(nav_broker([mark('closing_marks',b,100),mark('opening_marks',a,100)]))
