"""Complete unresolved minute sessions from recomputed retained provenance.

No P&L is read. Prior attempts and source files remain unchanged. These outputs
still require the independent fine-clock replay qualification before execution.
"""
from datetime import date
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

REPO = Path(__file__).resolve().parents[1]
OUT = REPO/'data/backtests/analysis/olr_participation_20261011/minute_exit_source_resolution'
FIELDS = ['timestamp','open','high','low','close','volume']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def main(window):
    import pandas as pd
    ready_path=Path.home()/f'.codex/research/olr_repair_v4/{window}/ready.json'
    ready=json.loads(ready_path.read_text());files=Path(ready['files']);manifest=ready['manifest']
    sys.path.insert(0,str(files))
    from scripts.update_kis_intraday import aggregate_timeframe
    from scripts.recover_flagged_sessions import reconstruct_session
    from backtests.core.anchored_recovery import verified_reconstruction
    import backtests.core.anchored_recovery as native
    assert Path(native.__file__).resolve().is_relative_to(files.resolve())
    sources={str(ready_path):sha(ready_path),str(Path(__file__).resolve()):sha(__file__)}
    def read(path):
        sources[str(path)]=sha(path);return json.loads(path.read_text(encoding='utf-8'))
    def bound(name):
        path=files/name;assert sha(path)==manifest['files'][name]
        sources[str(path)]=sha(path);return path
    def norm(frame):
        frame=frame[FIELDS].copy();frame['timestamp']=pd.to_datetime(frame.timestamp,utc=True).dt.tz_convert('Asia/Seoul').dt.as_unit('ns')
        frame=frame.sort_values('timestamp').reset_index(drop=True)
        assert not frame.timestamp.duplicated().any()
        return frame
    first=read(OUT/window/'resolution.json')
    for path,digest in first['sources'].items():assert sha(path)==digest
    destination=OUT/(window+'_completion')
    if destination.exists():raise FileExistsError('Preserve previous completion attempt')
    destination.mkdir()
    db=bound(ready['study']['database']);conn=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True)
    shared=manifest['evidence']['shared_market_inputs'];bundle={}
    if shared.get('anchored_session_recovery'):
        bundle=read(bound(shared['anchored_session_recovery']))
    results=[]
    for record in first['sessions']:
        if record['status']=='resolved_exact_aggregation':continue
        symbol,day=record['symbol'],record['date'];variants=[];errors=[]
        native_frame=norm(pd.DataFrame(conn.execute('SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? AND day=? ORDER BY ts',(symbol,day)).fetchall(),columns=FIELDS))
        matches=[x for x in bundle.get('sessions',[]) if x['symbol']==symbol and x['date']==day]
        assert len(matches)<=1
        if matches:
            row=matches[0]
            # Reparse retained provider JSON, verify all bound inputs and
            # recompute timestamp, daily-control and session-coverage proof.
            _,five,proof,paths=verified_reconstruction(files,row)
            for path in paths:sources[str(path)]=sha(path)
            raw=pd.read_parquet(bound(row['raw']));anchor=pd.read_parquet(bound(row['anchor']))
            if row.get('method')=='adjusted_provider_reconstruction':
                from backtests.core.adjusted_recovery import reconstruct_adjusted_session
                daily=pd.read_parquet(bound(row['daily_source']))
                daily=daily[daily.date.astype(str).str[:10]==day]
                provider_rows=[v for name in row['provider_responses'] for v in read(bound(name))['rows']]
                one,_=reconstruct_adjusted_session(raw,anchor,date.fromisoformat(day),symbol,
                    read(bound(row['original_daily_receipt'])),read(bound(row['adjusted_daily_receipt'])),daily.iloc[0].to_dict(),provider_rows)
            else:
                one,_=reconstruct_session(raw,anchor,date.fromisoformat(day))
            pd.testing.assert_frame_equal(norm(aggregate_timeframe(one,'5m')),norm(five),check_dtype=False)
            variants.append(('recomputed_native_reconstruction',norm(one),dict(method=row.get('method'),proof=proof,files=row['files'])))
        else:
            # Entire retained vintages are compared, with no minute-wise mixing,
            # interpolation or selection based on trade outcomes.
            for path in sorted((REPO/'data/kis_intraday_parquet'/symbol).glob(f'{symbol}_1m_*.parquet')):
                sources[str(path)]=sha(path);frame=norm(pd.read_parquet(path))
                frame=frame[frame.timestamp.dt.strftime('%Y-%m-%d')==day].reset_index(drop=True)
                if not frame.empty:variants.append((str(path),frame,dict(source_sha256=sources[str(path)],newly_bound_archive=True)))
        exact=[]
        for label,frame,proof in variants:
            five=norm(aggregate_timeframe(frame,'5m'))
            try:pd.testing.assert_frame_equal(five,native_frame,check_dtype=False,check_exact=True)
            except AssertionError:
                errors.append(dict(source=label,reason='does_not_reconstruct_complete_native_session'));continue
            key=hashlib.sha256(frame.to_json(orient='records',date_format='iso',double_precision=15).encode()).hexdigest()
            exact.append((key,label,frame,proof))
        unique={x[0] for x in exact}
        result=dict(symbol=symbol,date=day,matching_sources=[x[1] for x in exact],distinct_paths=len(unique),
            status='resolved_exact_aggregation' if len(unique)==1 else 'ambiguous' if unique else 'unresolved',rejections=errors)
        if len(unique)==1:
            key,label,frame,proof=exact[0];path=destination/f'{symbol}_{day}_1m.parquet';frame.to_parquet(path,index=False)
            result.update(output=str(path),output_sha256=sha(path),minute_path_hash=key,source=label,proof=proof)
        results.append(result)
        print(json.dumps({k:result[k] for k in ('symbol','date','status')}),flush=True)
    conn.close()
    payload=dict(window=window,sources=sources,sessions=results,
        unresolved=sum(x['status']!='resolved_exact_aggregation' for x in results),no_pnl_consumed=True,
        all_initial_discrepant_sessions_resolved=all(x['status']=='resolved_exact_aggregation' for x in results),
        not_yet_fine_execution_qualified=True,
        limitations=['Archive timestamps are retained observations/reconstructions, not verified exchange queue timing.',
            'Minute paths require full source/clock/capacity/ledger qualification; 5m equality alone is not execution qualification.',
            'Earlier archive revisions are newly bound inputs and must be frozen before native policy outcomes.'])
    (destination/'resolution.json').write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8')
    print(json.dumps(dict(window=window,resolved=len(results)-payload['unresolved'],unresolved=payload['unresolved'])),flush=True)


if __name__=='__main__':main(sys.argv[1])
