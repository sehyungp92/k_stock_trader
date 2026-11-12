"""A clock correction cannot hide a price/volume or continuous-bar change."""
from datetime import datetime
from zoneinfo import ZoneInfo
import importlib.util
from pathlib import Path
import pandas as pd
import pytest

spec=importlib.util.spec_from_file_location('clock_repair',Path(__file__).resolve().parents[2]/'scripts/olr_minute_exit_clock_repair_v5.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def frames():
    zone=ZoneInfo('Asia/Seoul')
    def row(h,m,volume):
        return dict(timestamp=datetime(2026,6,24,h,m,tzinfo=zone),open=100,high=100,low=100,close=100,volume=volume)
    fine=pd.DataFrame([*(row(15,m,10) for m in range(15,20)),row(15,32,1000)])
    original=pd.DataFrame([row(15,15,50),row(15,30,1000)])
    return original,fine


def test_unique_single_price_clock_is_retained_without_value_changes():
    result=module.corrected_session(*frames())
    assert result['from_timestamp']=='2026-06-24T15:30:00+09:00'
    assert result['to_timestamp']=='2026-06-24T15:32:00+09:00'
    assert result['values']['volume']==1000


@pytest.mark.parametrize('mode',['volume','price','continuous','duplicate','late','multiple_prices'])
def test_clock_repair_rejects_other_changes(mode):
    old,fine=frames()
    if mode=='volume':fine.loc[5,'volume']+=1
    elif mode=='price':fine.loc[5,['open','high','low','close']]=101
    elif mode=='continuous':fine.loc[0,'timestamp']-=pd.Timedelta(minutes=5)
    elif mode=='duplicate':fine=pd.concat([fine,fine.iloc[-1:]],ignore_index=True)
    elif mode=='late':fine.loc[5,'timestamp']+=pd.Timedelta(minutes=10)
    else:old.loc[1,'high']=101;fine.loc[5,'high']=101
    with pytest.raises((AssertionError,ValueError)):module.corrected_session(old,fine)


def clock_audit_fixture(tmp_path,monkeypatch,tamper=None):
    import hashlib,json,sqlite3,yaml
    from datetime import date,time
    from backtests.core.input_manifest import InputManifest,signature
    from kis_core.trading_calendar import get_trading_calendar
    old,fine=frames();correction=module.corrected_session(old,fine)
    files=tmp_path/'files';files.mkdir()
    symbol,day='000660','2026-06-24'
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    databases={}
    for name,frame in [('old.sqlite',old),('new.sqlite',old.copy())]:
        if name=='new.sqlite':frame.loc[1,'timestamp']=fine.iloc[-1].timestamp
        if tamper=='price' and name=='new.sqlite':frame.loc[1,'close']=101
        path=files/name;c=sqlite3.connect(path)
        c.execute('CREATE TABLE bars(symbol TEXT,day TEXT,ts TEXT,o REAL,h REAL,l REAL,c REAL,v REAL)')
        c.executemany('INSERT INTO bars VALUES(?,?,?,?,?,?,?,?)',[(symbol,day,r.timestamp.isoformat(),r.open,r.high,r.low,r.close,r.volume) for r in frame.itertuples(index=False)])
        c.commit();c.close();databases[name]=digest(path)
    event=dict(symbol=symbol,date=day,matching_time='15:32');event_name='event.json'
    (files/event_name).write_text(json.dumps(dict(closing_event=event)))
    original=dict(halts=[],closing_auctions=[],evidence_files={})
    current=dict(halts=[],closing_auctions=[{**event,'sources':[event_name]}],evidence_files={event_name:digest(files/event_name)})
    if tamper=='calendar':current['halts']=[{'unexpected':True}]
    (files/'old.yaml').write_text(yaml.safe_dump(original))
    calendar_path=files/'kis_core/data/krx_intraday_halts.yaml';calendar_path.parent.mkdir(parents=True);calendar_path.write_text(yaml.safe_dump(current))
    record=dict(version='olr-minute-closing-clock-correction-v5',window='oos',
        original_database_sha256=databases['old.sqlite'],database_sha256=databases['new.sqlite'],
        corrections=[dict(symbol=symbol,date=day,**correction,primary_source={'sha256':'raw_fixture'},event=event_name,event_sha256=digest(files/event_name))])
    if tamper=='timestamp':record['corrections'][0]['to_timestamp']='2026-06-24T15:33:00+09:00'
    (files/'correction.json').write_text(json.dumps(record))
    payload=dict(source_root=str(files),source_generation_hash='synthetic_fixture_only',
        files={p.relative_to(files).as_posix():digest(p) for p in files.rglob('*') if p.is_file()},evidence={})
    (tmp_path/'manifest.json').write_text(json.dumps({**payload,'manifest_hash':signature(payload)}))
    generation=InputManifest(tmp_path/'manifest.json')
    definition=dict(closing_clock_correction='correction.json',window='oos',original_canonical_database='old.sqlite',canonical_database='new.sqlite',original_calendar='old.yaml')
    calendar=get_trading_calendar()
    monkeypatch.setattr(calendar,'closing_auctions',{**calendar.closing_auctions,(symbol,date.fromisoformat(day)):time(15,32)})
    return generation,definition,lambda s,d:(fine,'raw_fixture',{}),lambda:None


def test_independent_clock_audit_reconstructs_full_session(tmp_path,monkeypatch):
    from backtests.engine.olr_minute_source_v5 import verify_closing_clock_sources
    result=verify_closing_clock_sources(*clock_audit_fixture(tmp_path,monkeypatch))
    assert result['corrected_sessions']==1 and result['all_declared_names_checked']


@pytest.mark.parametrize('mode',['price','calendar','timestamp'])
def test_independent_clock_audit_rejects_tampered_sources(tmp_path,monkeypatch,mode):
    from backtests.engine.olr_minute_source_v5 import verify_closing_clock_sources
    with pytest.raises((AssertionError,ValueError)):
        verify_closing_clock_sources(*clock_audit_fixture(tmp_path,monkeypatch,mode))
