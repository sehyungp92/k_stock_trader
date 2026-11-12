"""Read-only native-source panel extraction in an isolated frozen interpreter."""
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
import hashlib, json, math, os, sqlite3, sys

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,sort_keys=True,indent=2);f.write('\n')

def source_universe(primary, database_symbols):
    universe=sorted(primary)
    if len(universe)!=103 or any(len(s)!=6 or not s.isdigit() for s in universe):
        raise ValueError('Expected the frozen 103-name source universe')
    if set(universe)-set(database_symbols):raise ValueError('Missing declared OLR symbol')
    return universe, sorted(set(database_symbols)-set(universe))

def main():
    root,parent,window=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
    if window!='train':raise ValueError('Stage 1 cannot read OOS')
    ready_path=parent/'native_engines'/window/'ready.json';ready=read(ready_path);files=Path(ready['files']).resolve()
    repo=Path(__file__).resolve().parents[1]
    os.chdir(files);sys.path[:]=[str(files)]+[p for p in sys.path if not Path(p or os.getcwd()).resolve().is_relative_to(repo)]
    from backtests.core.input_manifest import InputManifest
    from strategy_common.market import MarketBar
    from strategy_common.clock import KST
    from strategy_common.execution_session import session_cutoff
    from strategy_olr.core.capacity import phase
    from kis_core.trading_calendar import get_trading_calendar
    generation=InputManifest(ready['study']['manifest'])
    bindings={str(ready_path):sha(ready_path),str(ready['study']['manifest']):sha(ready['study']['manifest'])}
    database=files/ready['study']['database'];digest=sha(database)
    if ready['manifest']['files'][ready['study']['database']]!=digest or digest!=ready['study']['database_hash']:
        raise ValueError('Changed coarse database')
    bindings[str(database)]=digest
    # Bind the completed independent original reconstruction and its copied
    # envelope. This extraction is not another source-materialization claim.
    for proof_path in (parent/f'native_source_preflight_{window}.json',parent/'reused_evidence'/window/'original_source_preflight.json'):
        proof=read(proof_path)
        if not proof['all_source_rows_recomputed']:raise ValueError('Missing completed source proof')
        bindings[str(proof_path)]=sha(proof_path)
    snapshots=files/ready['study']['olr_snapshots']
    if sha(snapshots)!=ready['manifest']['files'][ready['study']['olr_snapshots']]:raise ValueError('Changed selection scope')
    bindings[str(snapshots)]=sha(snapshots)
    selected={s['trade_date']:[c['symbol'] for c in s['candidates'] if c['tradable']] for s in read(snapshots)}
    connection=sqlite3.connect(database.as_uri()+'?mode=ro',uri=True)
    rows=[];coverage=defaultdict(int)
    with generation.activate():
        calendar=get_trading_calendar()
        definition=json.loads(generation.read(files/ready['study']['minute_source_definition']))
        source_plan=json.loads(generation.read(files/definition['source_plan']))
        primary=source_plan['windows'][window]['primary']
        universe,other_symbols=source_universe(primary,[r[0] for r in connection.execute('SELECT DISTINCT symbol FROM bars')])
        for name in (ready['study']['minute_source_definition'],definition['source_plan']):
            bindings[str(files/name)]=sha(files/name)
        start=connection.execute('SELECT MIN(day) FROM bars').fetchone()[0]
        cursor=date.fromisoformat(start);sessions=[]
        while cursor<=date(2026,3,31):
            if calendar.is_trading_day(cursor):sessions.append(cursor.isoformat())
            cursor+=timedelta(days=1)
        for day_text in sessions:
            day=date.fromisoformat(day_text);opening,closing=calendar.session_hours(day)
            cutoff=datetime.combine(day,session_cutoff(day,time(14,30)),tzinfo=KST)
            begin=datetime.combine(day,opening,tzinfo=KST)
            grouped=defaultdict(list)
            query='SELECT symbol,ts,o,h,l,c,v FROM bars WHERE day=? AND symbol IN ('+','.join('?' for _ in universe)+') ORDER BY symbol,ts'
            for symbol,stamp,*values in connection.execute(query,(day_text,*universe)):
                grouped[symbol].append((datetime.fromisoformat(stamp),*values))
            for symbol in universe:
                values=grouped[symbol]
                if len(values)!=len({r[0] for r in values}):raise ValueError('Duplicate source timestamp')
                prefix=[r for r in values if begin<=r[0] and r[0]+timedelta(minutes=5)<=cutoff]
                expected=set();stamp=begin
                while stamp+timedelta(minutes=5)<=cutoff:
                    if not any(datetime.combine(day,a,tzinfo=KST)<=stamp and stamp+timedelta(minutes=5)<=datetime.combine(day,z,tzinfo=KST) for a,z,_ in calendar.intraday_halts_for(symbol,day)):
                        expected.add(stamp)
                    stamp+=timedelta(minutes=5)
                periodic=bool(calendar.periodic_auction_sources(symbol,day));suspended=bool(calendar.suspension_source(symbol,day))
                complete=bool(expected) and set(r[0] for r in prefix)==expected and not periodic and not suspended
                for r in prefix:
                    stamp,o,h,l,c,v=r
                    if not all(math.isfinite(x) for x in (o,h,l,c,v)) or not 0<l<=min(o,c)<=max(o,c)<=h or v<0:
                        raise ValueError('Invalid prefix OHLCV')
                prefix_volume=sum(r[-1] for r in prefix) if complete else None
                # Independent SQL sum checks extraction arithmetic; bar identity
                # and provenance remain bound to the authenticated database.
                if complete:
                    sql=connection.execute('SELECT SUM(v) FROM bars WHERE day=? AND symbol=? AND ts>=? AND ts<?',(day_text,symbol,begin.isoformat(),cutoff.isoformat())).fetchone()[0]
                    if prefix_volume!=sql:raise ValueError('Prefix volume aggregation mismatch')
                match=datetime.combine(day,calendar.closing_match_time(symbol,day),tzinfo=KST)
                auctions=[r for r in values if r[0]==match]
                auction_volume=None;disposition='missing_source'
                if periodic or suspended:
                    auction_volume=0.;disposition='native_execution_prohibited'
                elif auctions:
                    r=auctions[0];bar=MarketBar(symbol,r[0],'5m',*r[1:],source_fingerprint=digest)
                    if phase(bar)=='auction':auction_volume=float(bar.volume);disposition='source_qualified_auction'
                    elif bar.volume==0 and bar.open==bar.high==bar.low==bar.close and bar.close>0:
                        auction_volume=0.;disposition='observed_zero_volume'
                    else:disposition='unsupported_source_observation'
                rows.append(dict(day=day_text,symbol=symbol,prefix_volume=prefix_volume,
                    prefix_available_at=cutoff.isoformat(),prefix_rows=len(prefix),prefix_complete=complete,
                    auction_volume=auction_volume,auction_available_at=(match+timedelta(minutes=5)).isoformat(),
                    auction_disposition=disposition,source_hash=digest))
                coverage[disposition]+=1;coverage['complete_prefix' if complete else 'unavailable_prefix']+=1
            print(json.dumps(dict(day=day_text,rows=len(rows))),flush=True)
    connection.close()
    imports={}
    for name,module in list(sys.modules.items()):
        if name.split('.')[0] not in ('backtests','strategy_common','strategy_olr','strategy_kalcb','strategy_pcim','deployment','instrumentation','oms','oms_client','kis_core','scripts') or not getattr(module,'__file__',None):continue
        path=Path(module.__file__).resolve()
        if not path.is_relative_to(files):raise ValueError('Non-native import '+name)
        relative=path.relative_to(files).as_posix();actual=sha(path)
        if ready['manifest']['files'].get(relative)!=actual:raise ValueError('Unbound native import '+name)
        imports[relative]=actual
    # InputManifest records every activated calendar/data access in its own
    # manifest; record all source and code bindings used for this projection.
    for name,digest in ready['manifest']['files'].items():
        if name.startswith('kis_core/data/'):
            path=files/name
            if sha(path)!=digest:raise ValueError('Changed exchange calendar')
            bindings[str(path)]=digest
    panel=dict(version='structural-source-panel-v6',window=window,universe=universe,sessions=sessions,rows=rows,
               selected=selected,coarse_database_sha256=sha(database))
    write(root/'train_panel.json',panel)
    write(root/'train_panel_receipt.json',dict(utc=datetime.now(timezone.utc).isoformat(),
        ready_sha256=sha(ready_path),panel_sha256=sha(root/'train_panel.json'),bindings=bindings,
        native_imports=imports,worker_sha256=sha(__file__),coverage=dict(coverage),oos_read=False,pnl_read=False,
        non_OLR_database_symbols=other_symbols,fixed_OLR_scope=universe,
        source_reconstruction='Authenticated completed original proof reused; this is a new causal feature projection, not a rerun of source preparation.'))

if __name__=='__main__':main()
