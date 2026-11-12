"""Recompute fine observation origins and scope inside a frozen native replay.

Source parsing uses the native vendor/recovery routines. Neither the materializer's
pass label nor the replay's derived SQLite rows establish an audit verdict.
"""
from collections import defaultdict
from datetime import date,datetime,timedelta
from pathlib import Path
import hashlib,json,sqlite3

FIELDS=['timestamp','open','high','low','close','volume']

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


class _SessionFrameCache:
    """Index each verified, normalized archive once per existing cache lifetime.

    Source authentication remains the caller's responsibility on every access.
    Only date-to-row lookup is reused; no reconstructed session or audit verdict
    is shared across accounts or generations.
    """
    def __init__(self, load):
        self._load = load
        self._frames = {}
        self._days = {}

    def frame(self, path):
        key = str(path)
        if key not in self._frames:
            self._frames[key] = normalize(self._load(path))
        return self._frames[key]

    def dayframe(self, path, day):
        key = str(path)
        frame = self.frame(path)
        if key not in self._days:
            # Use exactly the former KST string predicate, once per archive.
            self._days[key] = frame.groupby(
                frame.timestamp.dt.strftime('%Y-%m-%d'), sort=False).indices
        return frame.iloc[self._days[key].get(day, [])].reset_index(drop=True)

    def clear(self):
        self._frames.clear()
        self._days.clear()


def origin_verifier(generation,definition):
    import pandas as pd
    from scripts.update_kis_intraday import aggregate_timeframe,_normalize_kis_minute_row
    from scripts.recover_flagged_sessions import reconstruct_session
    from backtests.core.anchored_recovery import verified_reconstruction
    files=generation.root;manifest=generation.payload;sources={}
    aliases=definition['aliases']
    def resolve(path):
        raw=str(path)
        if raw in aliases:return generation.root/aliases[raw]
        p=Path(path)
        if not p.is_absolute():p=generation.root/p
        p=p.resolve()
        if not p.is_relative_to(generation.root.resolve()) or not generation.digest(p):
            raise ValueError('Minute source is outside its immutable manifest: '+raw)
        return p
    def sha(path):
        p=resolve(path)
        with generation.open_verified(p):pass
        sources[str(p)]=generation.digest(p)
        return generation.digest(p)
    def read(path):
        p=resolve(path);data=generation.read(p);sources[str(p)]=generation.digest(p)
        return json.loads(data)
    def parquet(path):
        p=resolve(path)
        with generation.open_verified(p) as stream:return pd.read_parquet(stream)
    def bound(name):
        p=resolve(name);sha(p);return p
    plan=read(definition['source_plan']);window=definition['window'];binding=plan['windows'][window]
    # Every used source is verified as read; all plan hashes also bind to the
    # new manifest independently of the materializer's receipt.
    for path,digest in plan['sources'].items():
        p=resolve(path)
        if generation.digest(p)!=digest:raise ValueError('Minute source plan hash mismatch')
    shared=manifest['evidence']['shared_market_inputs']
    bundle=read(bound(shared['anchored_session_recovery'])) if shared.get('anchored_session_recovery') else {'sessions':[]}
    anchors={(r['symbol'],r['date']):r for r in bundle['sessions']}
    digest_names={}
    for name,digest in manifest['files'].items():digest_names.setdefault(digest,[]).append(name)
    records={}
    for path in definition['resolution_receipts']:
        for record in read(path)['sessions']:
            if record['status']=='resolved_exact_aggregation':
                key=(record['symbol'],record['date']);assert key not in records;records[key]=record
    session_frames = _SessionFrameCache(parquet)
    frame = session_frames.frame
    dayframe = session_frames.dayframe
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
        row=anchors[key];_,five,proof,paths=verified_reconstruction(files,row)
        for path in paths:sources[str(path)]=sha(path)
        raw=frame(bound(row['raw']));anchor=frame(bound(row['anchor']))
        if row.get('method')=='adjusted_provider_reconstruction':
            from backtests.core.adjusted_recovery import reconstruct_adjusted_session
            daily=parquet(bound(row['daily_source']));daily=daily[daily.date.astype(str).str[:10]==day]
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
        inputs=[]
        for source in record['inputs']:
            name=next(iter(digest_names.get(source['sha256'],())),None)
            if name is None:raise ValueError('Unbound basis reconstruction dependency')
            inputs.append(bound(name))
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
    overrides={(r['symbol'],r['date']):r for r in binding['whole_session_overrides']}
    def session(symbol,day):
        item=binding['primary'][symbol]
        if sha(item['path'])!=item['sha256']:raise ValueError('Minute primary source mismatch')
        primary=dayframe(item['path'],day)
        if (symbol,day) in overrides:
            chosen=overrides[(symbol,day)];value,proof=verify_override(symbol,day,chosen,primary)
            return value,chosen['sha256'],proof
        return primary,item['sha256'],{'symbol':symbol,'date':day,'source_recomputed':True,'kind':'bound_primary_archive'}
    return session,session_frames.clear,sources


def load_minute_day(connection,day,symbols):
    """Load the declared source slice; membership is fixed before today's fills."""
    from strategy_common.market import MarketBar
    names=sorted(symbols)
    placeholders=','.join('?' for _ in names)
    query='SELECT symbol,ts,o,h,l,c,v,source_hash FROM bars WHERE day=? AND symbol IN ('+placeholders+') ORDER BY ts,symbol'
    return [MarketBar(symbol,datetime.fromisoformat(stamp),'1m',o,h,l,c,v,
        source='minute_source_plan_v5',source_fingerprint=digest)
        for symbol,stamp,o,h,l,c,v,digest in connection.execute(query,(str(day),*names))]


def verify_closing_clock_sources(generation,definition,session,clear):
    """Re-derive every declared late-print correction, including untraded names.

    This is independent of the preparation verdict. Original coarse prices and
    volumes must be exact, and every new clock must be present in the retained
    full minute session and independently agree with the native calendar.
    """
    import pandas as pd,yaml
    from scripts.update_kis_intraday import aggregate_timeframe
    from kis_core.trading_calendar import get_trading_calendar
    name=definition.get('closing_clock_correction')
    if not name:return dict(corrected_sessions=0,clock_corrections_verified=True)
    record=json.loads(generation.read(generation.root/name))
    if record['version']!='olr-minute-closing-clock-correction-v5' or record['window']!=definition['window']:
        raise ValueError('Unregistered closing-clock correction')
    old_path=generation.root/definition['original_canonical_database']
    new_path=generation.root/definition['canonical_database']
    for path,digest in ((old_path,record['original_database_sha256']),(new_path,record['database_sha256'])):
        with generation.open_verified(path):pass
        if generation.digest(path)!=digest:raise ValueError('Closing-clock database binding changed')
    original_calendar=yaml.safe_load(generation.read(generation.root/definition['original_calendar']))
    current_calendar=yaml.safe_load(generation.read(generation.root/'kis_core/data/krx_intraday_halts.yaml'))
    before={k:v for k,v in original_calendar.items() if k not in ('closing_auctions','evidence_files')}
    after={k:v for k,v in current_calendar.items() if k not in ('closing_auctions','evidence_files')}
    if before!=after:raise ValueError('Unrelated calendar policy changed')
    old_entries=original_calendar.get('closing_auctions',[])
    if current_calendar.get('closing_auctions',[])[:len(old_entries)]!=old_entries:
        raise ValueError('Original observed auctions changed')
    for path,digest in original_calendar['evidence_files'].items():
        if current_calendar['evidence_files'].get(path)!=digest:raise ValueError('Original calendar evidence changed')
    extra=current_calendar.get('closing_auctions',[])[len(old_entries):]
    if len(extra)!=len(record['corrections']):raise ValueError('Closing clock scope changed')
    old=sqlite3.connect(old_path.as_uri()+'?mode=ro',uri=True)
    new=sqlite3.connect(new_path.as_uri()+'?mode=ro',uri=True)
    calendar=get_trading_calendar();seen=set();last=None
    try:
        for row,calendar_row in zip(record['corrections'],extra):
            symbol,day=row['symbol'],row['date'];key=(symbol,day)
            if key in seen:raise ValueError('Duplicate clock correction')
            seen.add(key)
            if symbol!=last:clear();last=symbol
            fine,digest,_=session(symbol,day)
            if digest!=row['primary_source']['sha256']:raise ValueError('Clock source identity changed')
            query='SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? AND day=? ORDER BY ts'
            prior=normalize(pd.DataFrame(old.execute(query,key).fetchall(),columns=FIELDS))
            current=normalize(pd.DataFrame(new.execute(query,key).fetchall(),columns=FIELDS))
            same(aggregate_timeframe(fine,'5m'),current)
            if len(prior)!=len(current):raise ValueError('Clock correction changed row coverage')
            pd.testing.assert_frame_equal(prior[FIELDS[1:]],current[FIELDS[1:]],check_dtype=False,check_exact=True)
            changed=prior.timestamp!=current.timestamp
            if int(changed.sum())!=1:raise ValueError('Clock correction changed multiple event labels')
            index=prior.index[changed][0];a,b=prior.loc[index],current.loc[index]
            start,end=a.timestamp,b.timestamp
            close=calendar.session_hours(start.date())[1]
            if (start.time()!=close or start.date()!=end.date() or not start<end<start+pd.Timedelta(minutes=10)
                    or len(set(b[FIELDS[1:5]]))!=1 or b.volume<=0 or fine.timestamp.max()!=end):
                raise ValueError('Invalid observed closing print')
            same(fine[fine.timestamp==end],current.iloc[[index]])
            if (start.isoformat()!=row['from_timestamp'] or end.isoformat()!=row['to_timestamp']
                    or b[FIELDS[1:]].to_dict()!=row['values'] or calendar.closing_match_time(symbol,start.date())!=end.time()):
                raise ValueError('Closing timestamp or calendar does not match retained source')
            event_path=generation.root/row['event'];event=json.loads(generation.read(event_path))
            expected=dict(symbol=symbol,date=day,matching_time=end.strftime('%H:%M'))
            if (generation.digest(event_path)!=row['event_sha256'] or event['closing_event']!=expected
                    or calendar_row!={**expected,'sources':[row['event']]}):
                raise ValueError('Closing event receipt differs from independent reconstruction')
    finally:old.close();new.close();clear()
    return dict(corrected_sessions=len(seen),clock_corrections_verified=True,
        original_prices_shares_and_session_membership_exact=True,all_declared_names_checked=True,
        source_clock_not_exchange_queue_time=True)


def verify_minute_sources(broker,generation,evidence,identity):
    """Verify fine source values/origins, complete daily scope and coarse basis."""
    import pandas as pd
    from strategy_common.market import MarketBar
    from scripts.update_kis_intraday import aggregate_timeframe
    from kis_core.trading_calendar import get_trading_calendar
    from backtests.auto.shared.cache_keys import stable_signature
    name=evidence.get('minute_source_definition')
    if not name:raise ValueError('Minute source definition missing')
    definition=json.loads(generation.read(generation.root/name))
    if definition['version']!='olr-minute-source-definition-v5':raise ValueError('Unsupported minute source schema')
    contract=identity['configuration'].get('minute_clock')
    if contract!={'source_definition':name,'fast_exit':broker.minute_fast_exit,'version':'olr-minute-clock-v5'}:
        raise ValueError('Minute clock identity not registered')
    if len(broker.qualification_snapshot_scopes)!=1 or set(broker.qualification_snapshot_scopes)!={'OLR'}:
        raise ValueError('Minute clock supports only standalone OLR')
    candidates={s['trade_date']:{c['symbol'] for c in s['candidates']} for s in broker.qualification_snapshots}
    active={};positions=defaultdict(int);scopes=[]
    for row in broker.journal:
        kind=row['event']
        if kind=='minute_day_scope_v5':
            names=candidates.get(row['day'],set())|{s for s,q in positions.items() if q}|{
                o['symbol'] for o in active.values()}|{'000100'}
            if row['symbols']!=sorted(names):raise ValueError('Minute source day membership is not causal')
            scopes.append(row)
        elif kind in ('submit','amend'):active[row['order']['order_id']]=row['order']
        elif kind in ('cancel','expire','reject'):active.pop(row['order']['order_id'],None)
        elif kind=='fill':
            f=row['fill'];positions[f['symbol']]+=f['qty'] if f['side']=='BUY' else -f['qty']
            if f['qty']==row['order']['qty']:active.pop(f['order_id'],None)
    if scopes!=broker.minute_day_scopes:raise ValueError('Minute scope schedule changed')
    period=identity['configuration']['date_range'];calendar=get_trading_calendar()
    days=[d.date() for d in pd.date_range(period['start'],period['end']) if calendar.is_trading_day(d.date())]
    if [s['day'] for s in scopes]!=list(map(str,days)):raise ValueError('Incomplete or duplicate minute session schedule')
    expected=defaultdict(list)
    for batch in broker.expected_batches:
        for row in batch:expected[(row['symbol'],row['timestamp'][:10])].append(row)
    wanted={(symbol,row['day']) for row in scopes for symbol in row['symbols']}
    if set(expected)-wanted:raise ValueError('Minute replay contains observations outside the causal scope')
    session,clear,sources=origin_verifier(generation,definition)
    closing_clocks=verify_closing_clock_sources(generation,definition,session,clear)
    native_path=generation.root/definition['canonical_database'];fine_path=generation.root/definition['minute_database']
    # The execution loader's database is also immutable, but the audit below
    # independently reparses original source observations instead of trusting it.
    for p in (native_path,fine_path):
        with generation.open_verified(p):pass
    native=sqlite3.connect(native_path.as_uri()+'?mode=ro',uri=True)
    proofs=[];count=0;last_symbol=None;digests=[]
    try:
        for symbol,day in sorted(wanted):
            if symbol!=last_symbol:clear();last_symbol=symbol
            frame,digest,proof=session(symbol,day)
            rows=[MarketBar(symbol,r.timestamp.to_pydatetime(),'1m',r.open,r.high,r.low,r.close,r.volume,
                source='minute_source_plan_v5',source_fingerprint=digest).to_json_dict() for r in frame.itertuples(index=False)]
            if rows!=expected.get((symbol,day),[]):raise ValueError('Fine observations missing, changed or invented: '+symbol+':'+day)
            canonical=pd.DataFrame(native.execute('SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? AND day=? ORDER BY ts',
                (symbol,day)).fetchall(),columns=FIELDS)
            same(aggregate_timeframe(frame,'5m'),canonical)
            count+=len(rows);digests.append((symbol,day,stable_signature(rows)));proofs.append(proof)
    finally:native.close()
    return dict(passed=True,consumed_minute_rows=count,required_symbol_sessions=len(wanted),
        source_days=len(days),observations_hash=stable_signature(digests),source_definition_sha256=generation.digest(generation.root/name),
        recovered_sessions=[p for p in proofs if p.get('receipt_stage')],
        origins_reparsed=True,minute_source_scope_complete=True,coarse_basis_projection_exact=True,
        closing_clock_corrections=closing_clocks,
        source_files_read=sources,phase_liquidity_verified=False,
        limitation='Retained archive/reconstructed minute clock and declared lagged-volume liquidity proxy; not exchange queue timing.')
