"""Audit exact native minute-to-coarse clocks over the full declared universe.

The earlier inventory's generic resampling concealed some after-close timestamps.
This independent source-only scan preserves actual native parser timestamps.
"""
from pathlib import Path
import hashlib,json,sqlite3,subprocess,sys

REPO=Path(__file__).resolve().parents[1]
OUT=REPO/'data/backtests/analysis/olr_participation_20261011'
ROOT=Path('E:/CodexResearch/k_stock_trader/olr_round1_optimization_20261010/minute_exit_v5')
FIELDS=['timestamp','open','high','low','close','volume']


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main(window):
    import pandas as pd
    path=ROOT/f'native_clock_inventory_{window}.json'
    if path.exists():raise FileExistsError('Preserve previous source inventory')
    ready_path=Path.home()/f'.codex/research/olr_repair_v4/{window}/ready.json'
    ready=json.loads(ready_path.read_text());files=Path(ready['files']);sys.path.insert(0,str(files))
    from scripts.update_kis_intraday import aggregate_timeframe
    from kis_core.trading_calendar import get_trading_calendar
    import scripts.update_kis_intraday as native
    assert Path(native.__file__).resolve().is_relative_to(files.resolve())
    plan_path=OUT/'minute_exit_source_plan.json';plan=json.loads(plan_path.read_text());p=plan['windows'][window]
    db=files/ready['study']['database'];assert sha(db)==ready['study']['database_hash']
    sources={str(ready_path):sha(ready_path),str(plan_path):sha(plan_path),str(db):sha(db),str(Path(__file__).resolve()):sha(__file__),str(Path(native.__file__)):sha(native.__file__)}
    def frame(path,digest):
        assert sha(path)==digest;sources[str(path)]=digest
        f=pd.read_parquet(path)[FIELDS];f.timestamp=pd.to_datetime(f.timestamp,utc=True).dt.tz_convert('Asia/Seoul').dt.as_unit('ns')
        assert not f.timestamp.duplicated().any();return f
    conn=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);differences=[];counts=[]
    for symbol,item in sorted(p['primary'].items()):
        one=frame(item['path'],item['sha256'])
        one=one[one.timestamp.dt.strftime('%Y-%m-%d').between(p['period']['start'],p['period']['end'])]
        overrides=[r for r in p['whole_session_overrides'] if r['symbol']==symbol]
        for r in overrides:
            one=one[one.timestamp.dt.strftime('%Y-%m-%d')!=r['date']]
            one=pd.concat([one,frame(r['path'],r['sha256'])],ignore_index=True)
        one=one.sort_values('timestamp').reset_index(drop=True)
        five=aggregate_timeframe(one,'5m')[FIELDS].set_index('timestamp')
        five.index=pd.to_datetime(five.index,utc=True)
        expected=pd.DataFrame(conn.execute('SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? AND day>=? AND day<=? ORDER BY ts',
            (symbol,p['period']['start'],p['period']['end'])).fetchall(),columns=FIELDS)
        expected.timestamp=pd.to_datetime(expected.timestamp,utc=True);expected=expected.set_index('timestamp')
        dates=sorted(set(five.index.date)|set(expected.index.date));changed=[]
        for day in dates:
            a=five[five.index.date==day];b=expected[expected.index.date==day]
            if a.index.equals(b.index) and (a==b).all().all():continue
            common=a.index.intersection(b.index)
            common_equal=(a.loc[common]==b.loc[common]).all().all()
            added=a.loc[a.index.difference(b.index)];removed=b.loc[b.index.difference(a.index)]
            pairs=[];left=removed.copy()
            for stamp,row in added.iterrows():
                candidates=left[(left==row).all(axis=1)]
                match=(len(candidates)==1 and candidates.index[0].date()==stamp.date() and
                    candidates.index[0].tz_convert('Asia/Seoul').time()==get_trading_calendar().session_hours(day)[1] and
                    stamp.tz_convert('Asia/Seoul').time()>get_trading_calendar().session_hours(day)[1] and len(set(row[['open','high','low','close']]))==1)
                if match:
                    pairs.append(dict(from_timestamp=candidates.index[0].isoformat(),to_timestamp=stamp.isoformat(),values=row.to_dict()))
                    left=left.drop(candidates.index[0])
            is_clock=common_equal and len(pairs)==len(added)==len(removed)
            record=dict(symbol=symbol,date=str(day),kind='identical_single_price_after_close_timestamp' if is_clock else 'other_difference',
                common_rows_equal=bool(common_equal),pairs=pairs,
                source_only=added.reset_index().to_dict('records'),canonical_only=removed.reset_index().to_dict('records'),
                primary_source=item,override_sources=overrides)
            differences.append(record);changed.append(str(day))
        counts.append(dict(symbol=symbol,minute_rows=len(one),coarse_rows=len(five),changed_sessions=changed))
        print(json.dumps(dict(window=window,**counts[-1])),flush=True)
    conn.close()
    for source,digest in sources.items():assert sha(source)==digest
    payload=dict(window=window,sources=sources,symbols=counts,differences=differences,
        exact_parser='Frozen native aggregate_timeframe preserves after-scheduled-close timestamps',
        no_pnl_consumed=True,no_sources_modified=True,repair_not_yet_qualified=True)
    path.write_text(json.dumps(payload,indent=2,default=str)+'\n',encoding='utf-8')
    print(json.dumps(dict(window=window,changed_sessions=len(differences),clock_only=sum(r['kind']=='identical_single_price_after_close_timestamp' for r in differences))),flush=True)


if __name__=='__main__':main(sys.argv[1])
