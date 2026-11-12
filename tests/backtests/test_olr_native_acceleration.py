"""Regression checks for the isolated revision19 native implementation."""
from datetime import date, datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import inspect

import pandas as pd
import pytest

from backtests.auto.shared.cache_keys import stable_signature, _default
from kis_core.input_manifest import InputManifest, signature
from backtests.engine.qualification_evidence import source_rows
from scripts.kalcb_executed_value_study import load_day
from scripts.kalcb_executed_value_replay import observation_prefix
from tests.backtests.test_qualification_evidence import evidence_fixture

pytestmark=pytest.mark.skipif('stream' not in inspect.signature(load_day).parameters,
                            reason='Runs against the isolated revision19 replay engine')


@pytest.mark.parametrize('count',[127,128,129,159,160,161,1000])
@pytest.mark.parametrize('sign,default',[(signature,str),(stable_signature,_default)])
def test_batched_hash_exact_boundaries(count,sign,default):
    rows=[{'n':i,'v':[-0.0,1e300,float('nan')],'k':'한글"\n','d':date(2025,8,7)} for i in range(count)]
    assert sign(rows)==hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':'),default=default).encode()).hexdigest()
    assert sign(tuple(rows))==sign(rows)


@pytest.mark.parametrize('sign',[signature,stable_signature])
def test_batched_hash_rejects_cycles_and_invalid_keys(sign):
    rows=[{}]*128;rows[65]={'a':rows}
    with pytest.raises(ValueError,match='Circular reference'):sign(rows)
    with pytest.raises(TypeError):sign([{}]*127+[{(1,2):3}])


def test_manifest_rejects_escape_and_changed_bytes(tmp_path):
    root=tmp_path/'files';root.mkdir();(root/'a').write_bytes(b'original')
    payload={'files':{'a':hashlib.sha256(b'original').hexdigest()},'source_root':str(root)}
    p=tmp_path/'manifest.json'
    def save():p.write_text(json.dumps({**payload,'manifest_hash':signature(payload)}))
    save();m=InputManifest(p)
    (root/'a').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='bytes changed'):
        with m.open_verified(root/'a'):pass
    payload['files']['../escape']='f'*64;save()
    with pytest.raises(ValueError,match='escapes'):InputManifest(p)


def test_non_replayed_duplicate_source_rows_still_fail(evidence_fixture,monkeypatch):
    from tests.backtests.test_qualification_scope import inputs
    from backtests.engine import qualification_evidence as q
    _,_,_,broker,identity=inputs(evidence_fixture)
    original=q.source_rows
    def duplicated(*args):
        rows=list(original(*args))
        extra={**rows[0],'symbol':'999999'}
        yield from rows
        yield extra
        yield extra
    monkeypatch.setattr(q,'source_rows',duplicated)
    coverage,details=q.verify_qualification(broker,identity)
    assert coverage['input_provenance']=='failed'
    assert details['input_provenance']=='duplicate_source_event'


def database():
    c=sqlite3.connect(':memory:');c.execute('CREATE TABLE bars(day,symbol,ts,o,h,l,c,v)')
    rows=[('2025-08-07',s,'2025-08-07T'+t+':00+09:00',100,110,90,100,500)
          for s in ('000100','005930') for t in ('09:00','09:40','09:45','15:20')]
    c.executemany('INSERT INTO bars VALUES(?,?,?,?,?,?,?,?)',rows)
    return c


def test_streamed_watch_feed_exact_and_invalid_discarded_bars_rejected():
    c=database();day=date(2025,8,7)
    expected=observation_prefix(load_day(c,str(day),'s'),day,{'000100'},45)
    actual=observation_prefix(load_day(c,str(day),'s',stream=True),day,{'000100'},45)
    assert [x.to_json_dict() for x in expected]==[x.to_json_dict() for x in actual]
    c.execute("UPDATE bars SET v=-1 WHERE symbol='005930' AND ts LIKE '%15:20%'")
    with pytest.raises(ValueError,match='non-negative'):
        observation_prefix(load_day(c,str(day),'s',stream=True),day,{'000100'},45)


@pytest.mark.parametrize('stamp',['2025-08-07T09:00:00','2025-08-07T09:00:00+09:00',
                               '2025-08-07T00:00:00Z','2025-08-07T09:00:00.123456789+09:00'])
@pytest.mark.parametrize('bad',[False,True])
def test_source_timestamp_precision_and_native_rejection(tmp_path,stamp,bad):
    from strategy_common.market import MarketBar
    from strategy_common.clock import ensure_kst
    path=tmp_path/'bars.parquet'
    frame=pd.DataFrame([dict(timestamp=pd.Timestamp(stamp),open=100,high=110,low=90,close=100,volume=-1 if bad else 50)])
    frame.to_parquet(path)
    class G:
        root=tmp_path
        def open_verified(self,p):return p.open('rb')
    evidence={'market_files':[{'parser_version':'completed_ohlcv_parquet_v1','path':'bars.parquet','symbol':'005930','timeframe':'5m'}]}
    def original():
        row=next(frame.itertuples(index=False))
        return MarketBar('005930',ensure_kst(row.timestamp),'5m',row.open,row.high,row.low,row.close,row.volume).to_json_dict()
    if bad:
        with pytest.raises(ValueError):original()
        with pytest.raises(ValueError):list(source_rows(None,G(),evidence))
    else:assert list(source_rows(None,G(),evidence))==[original()]
