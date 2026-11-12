"""Materialize bound minute inputs and reparse recovery origins, without P&L.

This is source preparation, not a performance qualifier. A later native replay
must independently authenticate the used minute rows, source scope and basis.
"""
from datetime import date,datetime,timezone
from pathlib import Path
import hashlib,json,sqlite3,subprocess,sys,traceback,shutil

REPO=Path(__file__).resolve().parents[1]
OUT=REPO/'data/backtests/analysis/olr_participation_20261011'
ROOT=Path('E:/CodexResearch/k_stock_trader/olr_round1_optimization_20261010/minute_exit_v5')
FIELDS=['timestamp','open','high','low','close','volume']


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=str)+'\n',encoding='utf-8')


def normalize(frame):
    import pandas as pd
    import numpy as np
    frame=frame[FIELDS].copy()
    if not frame.empty and pd.to_datetime(frame.timestamp).dt.tz is None:
        raise ValueError('Source timezone must be explicit')
    frame['timestamp']=pd.to_datetime(frame.timestamp,utc=True).dt.tz_convert('Asia/Seoul').dt.as_unit('ns')
    frame=frame.sort_values('timestamp').reset_index(drop=True)
    if frame.empty:return frame
    assert not frame.timestamp.duplicated().any(),'Duplicate minute observations'
    assert np.isfinite(frame[FIELDS[1:]].to_numpy()).all()
    assert (frame.volume>=0).all() and (frame[FIELDS[1:5]]>0).all().all()
    assert ((frame.low<=frame[['open','close']].min(axis=1))&(frame.high>=frame[['open','close']].max(axis=1))).all()
    return frame


def same(a,b):
    import pandas as pd
    pd.testing.assert_frame_equal(normalize(a),normalize(b),check_dtype=False,check_exact=True)


def window_run(window):
    import pandas as pd
    destination=ROOT/'source_inputs'/window
    if destination.exists():raise FileExistsError('Preserve existing source-preparation attempt')
    destination.mkdir(parents=True)
    ready_path=Path.home()/f'.codex/research/olr_repair_v4/{window}/ready.json'
    ready=json.loads(ready_path.read_text());files=Path(ready['files']);manifest=ready['manifest']
    sys.path.insert(0,str(files))
    from scripts.update_kis_intraday import aggregate_timeframe,_normalize_kis_minute_row
    from scripts.recover_flagged_sessions import reconstruct_session
    from backtests.core.anchored_recovery import verified_reconstruction
    import backtests.core.anchored_recovery as native
    assert Path(native.__file__).resolve().is_relative_to(files.resolve())
    sources={str(ready_path):sha(ready_path),str(Path(__file__).resolve()):sha(__file__)}
    def read(path):
        sources[str(path)]=sha(path);return json.loads(Path(path).read_text(encoding='utf-8'))
    omitted={}
    if window=='oos':
        extra=read(ROOT/'closing_origin_dependencies.json')
        for path,digest in extra['evidence'].items():assert sha(path)==digest;sources[path]=digest
        for name,item in extra['files'].items():
            event=read(files/item['event'])
            assert sha(files/item['event'])==item['event_sha256']
            assert event['source_files'][name]==item['sha256']==sha(item['path'])
            assert not (files/name).exists(),'Never replace an existing frozen origin'
            omitted[name]=item
    def bound(name):
        if name in omitted:
            item=omitted[name];p=Path(item['path']);assert sha(p)==item['sha256']
        else:
            p=files/name;assert sha(p)==manifest['files'][name],name
        sources[str(p)]=sha(p);return p
    plan_path=OUT/'minute_exit_source_plan.json';plan=read(plan_path);binding=plan['windows'][window]
    for path,digest in plan['sources'].items():assert sha(path)==digest,(path,'bound source changed')
    sources.update(plan['sources'])
    for name in ('scripts/update_kis_intraday.py','scripts/recover_flagged_sessions.py',
            'backtests/core/anchored_recovery.py','backtests/core/adjusted_recovery.py','backtests/core/anchored_basis_repair.py'):
        bound(name)
    shared=manifest['evidence']['shared_market_inputs']
    bundle=read(bound(shared['anchored_session_recovery'])) if shared.get('anchored_session_recovery') else {'sessions':[]}
    anchors={(r['symbol'],r['date']):r for r in bundle['sessions']}
    digest_names={}
    for name,digest in manifest['files'].items():digest_names.setdefault(digest,[]).append(name)
    records={}
    for stage in (window,window+'_completion'):
        for record in read(OUT/'minute_exit_source_resolution'/stage/'resolution.json')['sessions']:
            if record['status']=='resolved_exact_aggregation':
                key=(record['symbol'],record['date']);assert key not in records;records[key]=record
    normalized_cache={}
    def frame(path):
        key=str(path)
        if key not in normalized_cache:normalized_cache[key]=normalize(pd.read_parquet(path))
        return normalized_cache[key]
    def dayframe(path,day):
        f=frame(path);return f[f.timestamp.dt.strftime('%Y-%m-%d')==day].reset_index(drop=True)
    def raw_reparse(name,symbol):
        path=bound(name);raw=frame(path)
        names=[n for n in manifest['files'] if n.startswith(str(Path(name).parent).replace('\\','/')+'/provider_responses/') and n.endswith('.json')]
        if not names:raise ValueError('Raw minute capture lacks retained provider responses: '+name)
        parsed=[]
        for n in names:
            payload=read(bound(n))
            assert payload['symbol']==symbol and payload['market']=='J' and payload['endpoint']=='inquire-time-dailychartprice' and payload['volume_field']=='cntg_vol'
            parsed.extend(x for x in map(_normalize_kis_minute_row,payload['rows']) if x is not None)
        parsed=pd.DataFrame(parsed)
        duplicates=parsed[parsed.timestamp.duplicated(keep=False)]
        assert duplicates.empty or len(duplicates.drop_duplicates())==duplicates.timestamp.nunique()
        parsed=normalize(parsed.drop_duplicates('timestamp'))
        parsed=parsed[parsed.timestamp.between(raw.timestamp.min(),raw.timestamp.max())].reset_index(drop=True)
        same(parsed,raw);return raw
    recovery_cache={}
    def recovered(symbol,day):
        key=(symbol,day)
        if key in recovery_cache:return recovery_cache[key]
        row=anchors[key];origin_root=files
        if any(name in omitted for name in row['files']):
            # A new source-only directory supplies the exact old receipt bytes.
            # Never write an omitted origin into the previous frozen generation.
            origin_root=destination/'origin_reconstruction'/symbol/day
            for name,digest in row['files'].items():
                source=bound(name);assert sha(source)==digest
                target=origin_root/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target);assert sha(target)==digest
        _,five,proof,paths=verified_reconstruction(origin_root,row)
        for path in paths:sources[str(path)]=sha(path)
        raw=frame(bound(row['raw']));anchor=frame(bound(row['anchor']))
        if row.get('method')=='adjusted_provider_reconstruction':
            from backtests.core.adjusted_recovery import reconstruct_adjusted_session
            daily=pd.read_parquet(bound(row['daily_source']));daily=daily[daily.date.astype(str).str[:10]==day]
            provider=[x for name in row['provider_responses'] for x in read(bound(name))['rows']]
            one,_=reconstruct_adjusted_session(raw,anchor,date.fromisoformat(day),symbol,
                read(bound(row['original_daily_receipt'])),read(bound(row['adjusted_daily_receipt'])),daily.iloc[0].to_dict(),provider)
        elif row.get('method')=='daily_reconciled_empty_session':one=raw
        else:one,_=reconstruct_session(raw,anchor,date.fromisoformat(day))
        same(aggregate_timeframe(one,'5m'),five)
        recovery_cache[key]=(normalize(one),proof);return recovery_cache[key]
    repair_cache={}
    def basis_repair(symbol,day):
        from backtests.core.anchored_basis_repair import reconstruct_original_session
        from backtests.core.adjusted_recovery import DAILY_FIELDS
        proof=read(bound('research/source_repair/repaired_sessions/proof.json'))
        record=next(r for r in proof['sessions'] if r['symbol']==symbol and r['date']==day)
        extra=read(ROOT/'basis_origin_dependencies.json')
        assert extra['proof_sha256']==sha(bound('research/source_repair/repaired_sessions/proof.json'))
        inputs=[]
        for source in record['inputs']:
            name=next(iter(digest_names.get(source['sha256'],())),None)
            if name is None:
                path=Path(source['path'])
                if extra['sources'].get(str(path))!=source['sha256'] or sha(path)!=source['sha256']:
                    raise ValueError('Unbound basis reconstruction dependency')
                sources[str(path)]=source['sha256'];inputs.append(path)
            else:inputs.append(bound(name))
        original=dayframe(inputs[0],'2026-05-29');adjusted=frame(inputs[1]);raw=frame(inputs[2])
        controls=[]
        for index,label in ((3,False),(4,True)):
            payload=read(inputs[index]);assert payload['adjusted'] is label
            assert payload['params']['FID_ORG_ADJ_PRC']==('0' if label else '1')
            rows=[r for r in payload['rows'] if r['stck_bsop_date']==day.replace('-','')];assert len(rows)==1
            controls.append({**{k:int(rows[0][v]) for k,v in DAILY_FIELDS.items()},'turnover':int(rows[0]['acml_tr_pbmn'])})
        provider=[r for path in inputs[5:] for r in read(path)['rows']]
        decoded,newproof=reconstruct_original_session(raw,original,adjusted,*controls,provider,date.fromisoformat(day),symbol)
        return normalize(decoded),newproof
    def verify_override(symbol,day,override,primary):
        record=records[(symbol,day)]
        assert override['sha256']==record['output_sha256']==sha(override['path'])
        out=frame(override['path']);proof={}
        if record.get('source')=='recomputed_native_reconstruction':candidate,proof=recovered(symbol,day)
        elif 'source' in record:
            path=record['source'];assert sha(path)==record['proof']['source_sha256']
            sources[path]=sha(path);candidate=dayframe(path,day)
        else:
            label=record['selected_source'];closing=label.endswith('+closing_event')
            label=label.removesuffix('+closing_event')
            merge=label.startswith('primary_plus:');name=label.removeprefix('primary_plus:')
            if name=='primary_archive':candidate=primary
            elif 'reconstructed_1m' in name:
                candidate,proof=recovered(symbol,day);same(candidate,frame(bound(name)))
            elif 'source_repair/repaired_sessions/' in name:
                candidate,proof=basis_repair(symbol,day);same(candidate,frame(bound(name)))
            else:candidate=dayframe(bound(name),day);same(candidate,raw_reparse(name,symbol))
            if merge:
                a=primary.set_index('timestamp');b=candidate.set_index('timestamp');overlap=a.index.intersection(b.index)
                same(a.loc[overlap].reset_index(),b.loc[overlap].reset_index())
                candidate=normalize(pd.concat([primary,candidate[~candidate.timestamp.isin(primary.timestamp)]],ignore_index=True))
            if closing:
                names=[n for n in record['dependencies'] if n.endswith('.json') and '/closing_events/' in n]
                assert len(names)==1;event=read(bound(names[0]));obs=event['observation']
                if 'source_files' in event:
                    names=event['source_files']
                    for name,digest in names.items():assert sha(bound(name))==digest
                    def unique(predicate):
                        selected=[n for n in names if predicate(n)];assert len(selected)==1;return selected[0]
                    row=dict(symbol=symbol,date=day,method='adjusted_provider_reconstruction',files=names,
                        raw=unique(lambda n:n.endswith('/raw_1m.parquet')),
                        anchor=unique(lambda n:n.startswith('data/kis_intraday_parquet/')),
                        daily_source=unique(lambda n:n.startswith('data/krx_daily_parquet/')),
                        original_daily_receipt=unique(lambda n:n.endswith('_original.json')),
                        adjusted_daily_receipt=unique(lambda n:n.endswith('_adjusted.json')),
                        provider_responses=[n for n in names if '/provider_responses/' in n])
                    if (symbol,day) not in anchors:anchors[(symbol,day)]=row
                    reconstructed,source_proof=recovered(symbol,day)
                else:
                    raw=raw_reparse(event['raw_source'],symbol);assert sha(bound(event['raw_source']))==event['raw_sha256']
                    anchor=frame(bound(event['archive_source']));assert sha(bound(event['archive_source']))==event['archive_sha256']
                    reconstructed,source_proof=reconstruct_session(raw,anchor,date.fromisoformat(day))
                if 'price' in obs:
                    assert obs['price']*obs['shares']==obs['turnover_increment']
                    obs=dict(timestamp=obs['timestamp'],open=obs['price'],high=obs['price'],low=obs['price'],close=obs['price'],volume=obs['shares'])
                point=normalize(pd.DataFrame([obs]));same(point,reconstructed[reconstructed.timestamp.isin(point.timestamp)])
                assert not candidate.timestamp.isin(point.timestamp).any()
                candidate=normalize(pd.concat([candidate,point],ignore_index=True));proof={**proof,'closing_source_reconstruction':source_proof}
        same(candidate,out)
        assert (out.timestamp.dt.strftime('%Y-%m-%d')==day).all()
        return out,dict(symbol=symbol,date=day,source_recomputed=True,rows=len(out),proof=proof,
            minute_sha256=override['sha256'],receipt_stage=override['receipt_stage'])
    source_db=bound(ready['study']['database']);clock_correction=None
    if window=='oos':
        # Failed attempt 0 exposed after-close timestamps rounded down in the
        # old coarse archive. Preserve those originals and use the separately
        # reconstructed, source-only timestamp correction in this generation.
        clock_path=ROOT/'clock_source_correction/complete.json'
        clock_correction=read(clock_path)
        assert clock_correction['version']=='olr-minute-closing-clock-correction-v5'
        assert clock_correction['original_database_sha256']==sha(source_db)
        for path,digest in clock_correction['sources'].items():assert sha(path)==digest;sources[path]=digest
        source_db=Path(clock_correction['database'])
        assert sha(source_db)==clock_correction['database_sha256'];sources[str(source_db)]=sha(source_db)
    native_db=sqlite3.connect(source_db.as_uri()+'?mode=ro',uri=True)
    database=destination/'market_1m.sqlite';conn=sqlite3.connect(database)
    conn.execute('CREATE TABLE bars(symbol TEXT,day TEXT,ts TEXT,o REAL,h REAL,l REAL,c REAL,v REAL,source_hash TEXT,PRIMARY KEY(symbol,ts)) WITHOUT ROWID')
    proofs=[];symbols=[];overrides={(r['symbol'],r['date']):r for r in binding['whole_session_overrides']}
    for symbol,item in sorted(binding['primary'].items()):
        assert sha(item['path'])==item['sha256'];primary=frame(item['path'])
        primary=primary[primary.timestamp.dt.strftime('%Y-%m-%d').between(binding['period']['start'],binding['period']['end'])].reset_index(drop=True)
        parts=[];day_hashes={}
        for day,group in primary.groupby(primary.timestamp.dt.strftime('%Y-%m-%d')):
            if (symbol,day) not in overrides:parts.append(group);day_hashes[day]=item['sha256']
        for (sy,day),override in overrides.items():
            if sy!=symbol:continue
            group=primary[primary.timestamp.dt.strftime('%Y-%m-%d')==day].reset_index(drop=True)
            revised,proof=verify_override(symbol,day,override,group);parts.append(revised);proofs.append(proof);day_hashes[day]=override['sha256']
        fine=normalize(pd.concat(parts,ignore_index=True));assert (fine.timestamp.dt.second==0).all() and (fine.timestamp.dt.microsecond==0).all()
        five=normalize(aggregate_timeframe(fine,'5m'))
        expected=pd.DataFrame(native_db.execute('SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? AND day>=? AND day<=? ORDER BY ts',
            (symbol,binding['period']['start'],binding['period']['end'])).fetchall(),columns=FIELDS)
        same(five,expected)
        rows=[(symbol,str(r.timestamp.date()),r.timestamp.isoformat(),r.open,r.high,r.low,r.close,r.volume,day_hashes[str(r.timestamp.date())]) for r in fine.itertuples(index=False)]
        conn.executemany('INSERT INTO bars VALUES(?,?,?,?,?,?,?,?,?)',rows);conn.commit()
        symbols.append(dict(symbol=symbol,minute_rows=len(rows),coarse_rows=len(five),sessions=len(day_hashes)))
        normalized_cache.clear()
        print(json.dumps(dict(window=window,**symbols[-1])),flush=True)
    native_db.close();conn.close()
    assert len(proofs)==len(overrides)
    for path,digest in sources.items():assert sha(path)==digest
    write(destination/'complete.json',dict(window=window,utc=datetime.now(timezone.utc).isoformat(),
        database=str(database),database_sha256=sha(database),source_plan_sha256=sha(plan_path),
        canonical_database=str(source_db),canonical_database_sha256=sha(source_db),
        clock_correction_sha256=sha(ROOT/'clock_source_correction/complete.json') if clock_correction else None,
        sources=sources,symbols=symbols,recomputed_recoveries=proofs,native_origin_dependencies=omitted,no_pnl_consumed=True,
        fine_execution_qualified=False,obligations='Native source/basis/scope, clock, capacity and ledger audits remain required in every policy replay.'))


def main():
    if len(sys.argv)==2:return window_run(sys.argv[1])
    ROOT.mkdir(exist_ok=True,parents=True)
    complete=ROOT/'source_preparation_complete.json'
    if complete.exists():raise FileExistsError('Do not rerun completed source materialization')
    reused={}
    for window in ('train','oos'):
        receipt=ROOT/'source_inputs'/window/'complete.json'
        if receipt.exists():
            data=json.loads(receipt.read_text());assert sha(data['database'])==data['database_sha256']
            prior_builder=data['sources'][str(Path(__file__).resolve())]
            for p,digest in data['sources'].items():
                if p==str(Path(__file__).resolve()) and digest!=sha(__file__):
                    # Train completed before the OOS clock failure. Reuse only
                    # its authenticated original receipt and archived builder;
                    # this is an explicit provenance exception, never a rebind.
                    archived=ROOT/'source_failed_attempt_0/materialize_program.py'
                    assert window=='train' and sha(archived)==digest
                    reused[window]=dict(receipt_sha256=sha(receipt),original_builder_sha256=digest,
                        archived_builder=str(archived),reason='Completed exact native IS projection; OOS-only closing clock correction.')
                else:assert sha(p)==digest
            continue
        log=ROOT/(window+'_source_preparation.log')
        if log.exists():raise FileExistsError('Preserve failed source attempt before repair')
        with log.open('w',encoding='utf-8') as stream:
            result=subprocess.run([sys.executable,'-B','-u',str(Path(__file__).resolve()),window],cwd=REPO,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Source preparation failed; inspect '+str(log))
    write(complete,dict(utc=datetime.now(timezone.utc).isoformat(),script_sha256=sha(__file__),
        receipts={w:sha(ROOT/'source_inputs'/w/'complete.json') for w in ('train','oos')},
        authenticated_completed_reuse=reused,no_pnl_consumed=True))


if __name__=='__main__':
    try:main()
    except Exception:
        if len(sys.argv)==1:write(ROOT/'source_preparation_failure.json',dict(utc=datetime.now(timezone.utc).isoformat(),traceback=traceback.format_exc()))
        raise
