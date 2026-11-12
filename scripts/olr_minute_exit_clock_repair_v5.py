"""Source-only correction of archived closing-print labels in a NEW generation.

Every correction requires a complete native minute projection whose sole
difference is one unchanged, single-price closing print observed after close.
Never alter original archives, the previous study, prices, shares, or entries.
"""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib,json,shutil,sqlite3,sys

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('E:/CodexResearch/k_stock_trader/olr_round1_optimization_20261010/minute_exit_v5')
FIELDS=['timestamp','open','high','low','close','volume']


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=str)+'\n',encoding='utf-8')


def corrected_session(original,minutes):
    """Derive the correction again; supplied inventory labels are not evidence."""
    import pandas as pd
    from scripts.update_kis_intraday import aggregate_timeframe
    from kis_core.trading_calendar import get_trading_calendar
    def normal(frame):
        value=frame[FIELDS].copy();value.timestamp=pd.to_datetime(value.timestamp,utc=True)
        return value.sort_values('timestamp').reset_index(drop=True)
    old=normal(original);new=normal(aggregate_timeframe(minutes,'5m'))
    if len(old)!=len(new) or old.empty:raise ValueError('Closing repair cannot change row count')
    pd.testing.assert_frame_equal(old[FIELDS[1:]],new[FIELDS[1:]],check_dtype=False,check_exact=True)
    differences=old.timestamp!=new.timestamp
    if int(differences.sum())!=1:raise ValueError('Closing repair requires exactly one changed clock')
    index=old.index[differences][0]
    a,b=old.loc[index],new.loc[index]
    first,last=a.timestamp.tz_convert('Asia/Seoul'),b.timestamp.tz_convert('Asia/Seoul')
    closing=get_trading_calendar().session_hours(first.date())[1]
    if (first.date()!=last.date() or first.time()!=closing or
            not first<last<first+pd.Timedelta(minutes=10) or
            len(set(a[FIELDS[1:5]]))!=1 or a.volume<=0):
        raise ValueError('Not a sourced single-price delayed closing observation')
    points=normal(minutes)
    point=points[points.timestamp==b.timestamp]
    pd.testing.assert_frame_equal(point[FIELDS[1:]].reset_index(drop=True),
        new.loc[[index],FIELDS[1:]].reset_index(drop=True),check_dtype=False,check_exact=True)
    if not (points.timestamp.max()==last):raise ValueError('Closing point must be the final observed print')
    return dict(from_timestamp=first.isoformat(),to_timestamp=last.isoformat(),
        values={k:float(a[k]) for k in FIELDS[1:]})


def main():
    import pandas as pd,yaml
    output=ROOT/'clock_source_correction'
    if output.exists():raise FileExistsError('Preserve prior clock correction attempt')
    output.mkdir()
    ready_path=Path.home()/'.codex/research/olr_repair_v4/oos/ready.json'
    ready=json.loads(ready_path.read_text());files=Path(ready['files'])
    sys.path.insert(0,str(files))
    import scripts.update_kis_intraday as native
    assert Path(native.__file__).resolve().is_relative_to(files.resolve())
    inventory_path=ROOT/'native_clock_inventory_oos.json'
    inventory=json.loads(inventory_path.read_text())
    sources={str(ready_path):sha(ready_path),str(inventory_path):sha(inventory_path),str(Path(__file__).resolve()):sha(__file__)}
    for path,digest in inventory['sources'].items():assert sha(path)==digest;sources[path]=digest
    assert len(inventory['symbols'])==103
    assert all(r['kind']=='identical_single_price_after_close_timestamp' for r in inventory['differences'])
    grouped=defaultdict(list)
    for row in inventory['differences']:grouped[row['symbol']].append(row)
    old_db=files/ready['study']['database'];assert sha(old_db)==ready['study']['database_hash']
    sources[str(old_db)]=sha(old_db)
    db=output/'market_5m_clock.sqlite';shutil.copy2(old_db,db)
    connection=sqlite3.connect(db)
    calendar_path=files/'kis_core/data/krx_intraday_halts.yaml';sources[str(calendar_path)]=sha(calendar_path)
    calendar=yaml.safe_load(calendar_path.read_text(encoding='utf-8'))
    existing={(r['symbol'],r['date']) for r in calendar.get('closing_auctions',[])}
    shared=ready['manifest']['evidence']['shared_market_inputs'];records=[];coarse_files={}
    for symbol,rows in sorted(grouped.items()):
        primary=rows[0]['primary_source'];assert sha(primary['path'])==primary['sha256']
        minute=pd.read_parquet(primary['path'])[FIELDS]
        minute.timestamp=pd.to_datetime(minute.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
        names=shared['intraday_files'][symbol];assert len(names)==1
        name=names[0];source=files/name;assert sha(source)==ready['manifest']['files'][name]
        sources[str(source)]=sha(source);coarse=pd.read_parquet(source)
        for row in rows:
            day=row['date'];assert (symbol,day) not in existing
            assert not any(x['date']==day for x in row['override_sources'])
            assert primary==row['primary_source'] and len(row['pairs'])==1
            old=pd.DataFrame(connection.execute('SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? AND day=? ORDER BY ts',
                (symbol,day)).fetchall(),columns=FIELDS)
            fine=minute[minute.timestamp.dt.strftime('%Y-%m-%d')==day]
            correction=corrected_session(old,fine)
            recorded=row['pairs'][0]
            for k in ('from_timestamp','to_timestamp'):
                assert pd.Timestamp(correction[k])==pd.Timestamp(recorded[k])
            assert correction['values']==recorded['values']
            mask=pd.to_datetime(coarse.timestamp,utc=True)==pd.Timestamp(correction['from_timestamp'])
            assert int(mask.sum())==1
            assert coarse.loc[mask,FIELDS[1:]].iloc[0].to_dict()==correction['values']
            coarse.loc[mask,'timestamp']=pd.Timestamp(correction['to_timestamp'])
            changed=connection.execute('UPDATE bars SET ts=? WHERE symbol=? AND ts=?',
                (correction['to_timestamp'],symbol,correction['from_timestamp'])).rowcount
            assert changed==1
            event=dict(symbol=symbol,date=day,matching_time=pd.Timestamp(correction['to_timestamp']).strftime('%H:%M'))
            event_name=f'research/minute_exit_v5/clock_events/{symbol}_{day}.json'
            receipt=dict(closing_event=event,observed_archive_clock=True,exchange_queue_time_verified=False,
                source=primary,original_coarse_source=dict(path=name,sha256=sha(source)),
                correction=correction,whole_session_values_unchanged=True)
            event_path=output/event_name;write(event_path,receipt)
            calendar['evidence_files'][event_name]=sha(event_path)
            calendar.setdefault('closing_auctions',[]).append({**event,'sources':[event_name]})
            records.append(dict(symbol=symbol,date=day,**correction,primary_source=primary,event=event_name,event_sha256=sha(event_path)))
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True)
        assert coarse.timestamp.is_monotonic_increasing and not coarse.timestamp.duplicated().any()
        coarse.to_parquet(target,index=False)
        coarse_files[name]=dict(path=str(target),sha256=sha(target),original_sha256=sha(source))
        print(json.dumps(dict(symbol=symbol,corrected_clocks=len(rows))),flush=True)
    connection.commit();connection.close()
    calendar_output=output/'kis_core/data/krx_intraday_halts.yaml';calendar_output.parent.mkdir(parents=True,exist_ok=True)
    calendar_output.write_text(yaml.safe_dump(calendar,sort_keys=False,allow_unicode=True),encoding='utf-8')
    for source,digest in sources.items():assert sha(source)==digest
    write(output/'complete.json',dict(version='olr-minute-closing-clock-correction-v5',utc=datetime.now(timezone.utc).isoformat(),
        window='oos',sources=sources,corrections=records,coarse_files=coarse_files,database=str(db),database_sha256=sha(db),
        calendar=str(calendar_output),calendar_sha256=sha(calendar_output),original_database=str(old_db),
        original_database_sha256=sha(old_db),no_pnl_consumed=True,prices_volumes_and_session_membership_unchanged=True,
        inputs_modified=False,qualified_performance=False,
        obligations=['Bind original and corrected coarse sources, minute origins and calendar observations in a new native generation.',
            'Recompute native acceptance and every consumed minute origin; preserve coarse feature and full fine ledger audits.',
            'Separate combined source/execution clock effects from exit treatment with a matched minute control.',
            'Verify unchanged pre-entry features and historical prefix; keep archive clock and phase liquidity limitations explicit.']))
    print(json.dumps(dict(complete=True,corrected_clocks=len(records),symbols=len(coarse_files))),flush=True)


if __name__=='__main__':main()
