"""Resolve raw-archive discrepancies using retained source evidence, never P&L.

Run by absolute script path so window-specific frozen calendar/parser modules
can be selected before any strategy import. Exact aggregation is a prerequisite,
not a replacement for fine-clock execution/source qualification.
"""
from collections import defaultdict
import hashlib,json,sqlite3,sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
OUT=REPO/'data/backtests/analysis/olr_participation_20261011/minute_exit_source_resolution'
FIELDS=['open','high','low','close','volume']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def main(window):
    import pandas as pd
    ready_path=Path.home()/f'.codex/research/olr_repair_v4/{window}/ready.json'
    ready=json.loads(ready_path.read_text());files=Path(ready['files']);manifest=ready['manifest'];study=ready['study']
    sys.path.insert(0,str(files))
    from scripts.update_kis_intraday import aggregate_timeframe
    import scripts.update_kis_intraday as native
    assert Path(native.__file__).resolve().is_relative_to(files.resolve())
    inventory_path=OUT.parent/'minute_exit_source'/f'{window}.json'
    inventory=json.loads(inventory_path.read_text());destination=OUT/window
    destination.mkdir(parents=True,exist_ok=True)
    if (destination/'resolution.json').exists():raise FileExistsError('Preserve prior resolution attempt')
    sources={str(ready_path):sha(ready_path),str(inventory_path):sha(inventory_path),str(Path(__file__).resolve()):sha(__file__)}
    def bound(name):
        path=files/name;digest=manifest['files'][name];assert sha(path)==digest
        sources[str(path)]=digest;return path
    bound('scripts/update_kis_intraday.py')
    bad=defaultdict(set)
    for r in inventory['mismatches']:
        stamp=pd.Timestamp(r['timestamp']).tz_convert('Asia/Seoul');bad[r['symbol']].add(str(stamp.date()))
    db=bound(study['database']);conn=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True)
    def normalized(frame):
        frame=frame.copy();frame['timestamp']=pd.to_datetime(frame.timestamp).dt.tz_convert('Asia/Seoul')
        frame=frame[['timestamp',*FIELDS]].sort_values('timestamp').reset_index(drop=True)
        assert not frame.timestamp.duplicated().any()
        return frame
    results=[]
    for symbol,days in sorted(bad.items()):
        primary=REPO/f'data/kis_intraday_parquet/{symbol}/{symbol}_1m_20250710_20260710.parquet'
        assert sha(primary)==next(x['source_sha256'] for x in inventory['symbols'] if x['symbol']==symbol)
        sources[str(primary)]=sha(primary);archive=normalized(pd.read_parquet(primary))
        for day in sorted(days):
            base=archive[archive.timestamp.dt.strftime('%Y-%m-%d')==day].reset_index(drop=True)
            native_frame=pd.DataFrame(conn.execute('SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? AND day=? ORDER BY ts',(symbol,day)).fetchall(),columns=['timestamp',*FIELDS])
            native_frame=normalized(native_frame);variants=[('primary_archive',base,[])];rejected=[]
            names=[name for name in manifest['files'] if symbol in name and day in name and name.endswith('.parquet') and '1m' in name]
            # Source-derived session candidates are compared without choosing by
            # performance. Disagreement between matching minute paths is explicit.
            for name in names:
                path=bound(name);frame=normalized(pd.read_parquet(path))
                frame=frame[frame.timestamp.dt.strftime('%Y-%m-%d')==day].reset_index(drop=True)
                if frame.empty or (frame.timestamp.dt.second!=0).any():
                    rejected.append(dict(path=name,reason='empty_or_nonminute_clock_requires_provenance_reconstruction'));continue
                variants.append((name,frame,[name]))
                a=base.set_index('timestamp');z=frame.set_index('timestamp');overlap=a.index.intersection(z.index)
                if not len(overlap) or ((a.loc[overlap,FIELDS]-z.loc[overlap,FIELDS]).abs()<=1e-7).all().all():
                    merged=pd.concat([base,frame[~frame.timestamp.isin(base.timestamp)]],ignore_index=True)
                    variants.append(('primary_plus:'+name,normalized(merged),[name]))
            closing_names=[name for name in manifest['files'] if '/closing_events/' in name and name.endswith(f'{symbol}_{day}.json')]
            for name in closing_names:
                event=json.loads(bound(name).read_text());obs=event['observation']
                # Some retained auction receipts encode a single-price match
                # as price/shares, rather than an OHLCV bar. Preserve its actual
                # match timestamp and independently reconcile stated turnover.
                if 'price' in obs and 'shares' in obs:
                    assert obs['price'] > 0 and obs['shares'] >= 0
                    assert obs['price'] * obs['shares'] == obs['turnover_increment']
                    obs=dict(timestamp=obs['timestamp'],open=obs['price'],high=obs['price'],
                        low=obs['price'],close=obs['price'],volume=obs['shares'])
                for source_key,hash_key in [('raw_source','raw_sha256'),('archive_source','archive_sha256')]:
                    if event.get(source_key) in manifest['files']:
                        assert sha(bound(event[source_key]))==event[hash_key]
                point=normalized(pd.DataFrame([obs]));stamp=point.timestamp.iloc[0]
                for label,frame,dependencies in list(variants):
                    if stamp in set(frame.timestamp):continue
                    merged=normalized(pd.concat([frame,point],ignore_index=True))
                    variants.append((label+'+closing_event',merged,[*dependencies,name]))
            matches=[]
            for label,frame,dependencies in variants:
                aggregated=normalized(aggregate_timeframe(frame,'5m'))
                if not aggregated.timestamp.equals(native_frame.timestamp):continue
                if not ((aggregated[FIELDS]-native_frame[FIELDS]).abs()<=1e-7).all().all():continue
                key=hashlib.sha256(frame.to_json(orient='records',date_format='iso',double_precision=15).encode()).hexdigest()
                matches.append((key,label,frame,dependencies))
            unique={x[0] for x in matches};record=dict(symbol=symbol,date=day,native_bins=len(native_frame),
                tested_variants=len(variants),matching_sources=[x[1] for x in matches],distinct_matching_minute_paths=len(unique),
                status='resolved_exact_aggregation' if len(unique)==1 else 'ambiguous_minute_paths' if unique else 'unresolved',rejected_sources=rejected)
            if len(unique)==1:
                key,label,frame,dependencies=matches[0];path=destination/f'{symbol}_{day}_1m.parquet'
                assert not path.exists();frame.to_parquet(path,index=False)
                record.update(output=str(path),output_sha256=sha(path),minute_path_hash=key,selected_source=label,
                    dependencies=dependencies,minute_rows=len(frame),native_parser_sha256=sha(files/'scripts/update_kis_intraday.py'))
            results.append(record)
    conn.close()
    payload=dict(window=window,sources=sources,sessions=results,counts={status:sum(r['status']==status for r in results) for status in ('resolved_exact_aggregation','ambiguous_minute_paths','unresolved')},
        no_pnl_consumed=True,not_yet_fine_execution_qualified=True,
        limitations=['Exact 5m reconstruction does not certify intraminute sequence/vendor timestamps. Retain original repair evidence and native calendar/basis obligations.','Historical repair receipts may predate final qualification; do not treat any supplied pass label as evidence.','Resolved files are a separate derived source plan; no existing archive or frozen study is changed.'])
    (destination/'resolution.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(dict(window=window,counts=payload['counts'],unresolved=[dict(symbol=r['symbol'],date=r['date'],status=r['status']) for r in results if r['status']!='resolved_exact_aggregation'])),flush=True)


if __name__=='__main__':main(sys.argv[1])
