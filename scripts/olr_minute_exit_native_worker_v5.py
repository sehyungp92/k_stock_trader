"""Isolated workers: native imports resolve only inside the requested generation."""
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,os,sqlite3,sys


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+'\n',encoding='utf-8')
n=SimpleNamespace(read=read,write=write,sha=sha)
s=SimpleNamespace(utc=lambda:datetime.now(timezone.utc).isoformat())


def isolate(files):
    files=Path(files).resolve();repo=Path(__file__).resolve().parents[1]
    os.chdir(files)
    sys.path[:]=[str(files)]+[p for p in sys.path if not Path(p or os.getcwd()).resolve().is_relative_to(repo)]
    return files


def verify_imports(files,manifest):
    prefixes=('backtests','strategy_common','strategy_kalcb','strategy_olr','strategy_pcim','deployment','instrumentation','oms','oms_client','kis_core','scripts')
    checked={}
    for name,module in list(sys.modules.items()):
        if name.split('.')[0] not in prefixes or not getattr(module,'__file__',None):continue
        path=Path(module.__file__).resolve()
        if not path.is_relative_to(files):raise ValueError('Native worker imported outside its generation: '+name+':'+str(path))
        relative=path.relative_to(files).as_posix()
        if manifest['files'].get(relative)!=sha(path):raise ValueError('Native worker imported unbound code: '+relative)
        checked[relative]=sha(path)
    if not checked:raise ValueError('No native modules audited')
    return checked


def acceptance_worker(path):
    info=n.read(path);files=Path(info['files']);os.chdir(files);sys.path.insert(0,str(files))
    from backtests.engine.retained_source import build_evidence
    manifest=info['manifest'];old=n.read(files/manifest['evidence']['qualification_path'])
    before=n.read(files/old['basis']['acceptance_report'])
    target=files/'research/minute_exit_v5/acceptance'
    qualified=build_evidence(files,manifest['evidence']['shared_market_inputs'],target)
    value=n.read(qualified);after=n.read(files/value['basis']['acceptance_report'])
    assert n.read(files/old['references']['daily_rows'])==n.read(files/value['references']['daily_rows'])
    assert before['basis_failures']==after['basis_failures']
    for k,v in old.items():
        if k not in ('basis','references','version','source_files'):value[k]=v
    replacement=info['coarse_replacements']
    for row in value['market_files']:row['path']=replacement.get(row['path'],row['path'])
    value['minute_source_definition']=info['study']['minute_source_definition']
    value['capacity']['model_version']='causal-minute-participation-olr-v5'
    for row in value['capacity']['registered_execution_policies']:
        capacity=row['policy']['capacity']
        capacity.update(version='causal-minute-participation-olr-v5',opening_basis='previous_contiguous_continuous_1m',
            decision_clock='unchanged_5m_alpha; optional_completed_1m_risk_exit',calibration='same_minute_execution_for_control_and_treatment')
    n.write(qualified,value)
    for p in target.glob('*.json'):manifest['files'][p.relative_to(files).as_posix()]=n.sha(p)
    manifest['evidence']['qualification_path']=qualified.relative_to(files).as_posix()
    manifest['evidence']['minute_retained_acceptance']=dict(old_sha256=n.sha(files/old['basis']['acceptance_report']),
        new_sha256=n.sha(files/value['basis']['acceptance_report']),daily_references_exact=True,basis_dispositions_exact=True,
        scope='Only explicitly sourced closing timestamps change; all old prices/shares remain exact.')
    n.write(path.with_name('acceptance_complete.json'),manifest)


def source_worker(w):
    import pandas as pd
    ready=n.read(t.ENGINES/w/'ready.json');files=Path(ready['files']);os.chdir(files);sys.path.insert(0,str(files))
    from backtests.core.input_manifest import InputManifest
    from backtests.auto.shared.cache_keys import source_generation_hash
    from backtests.engine.olr_minute_source_v5 import origin_verifier,verify_closing_clock_sources,same,FIELDS
    from scripts.update_kis_intraday import aggregate_timeframe
    from kis_core.trading_calendar import get_trading_calendar
    import backtests.engine.olr_minute_source_v5 as audit
    assert Path(audit.__file__).resolve().is_relative_to(files.resolve())
    assert source_generation_hash()==ready['study']['source_generation_hash']
    generation=InputManifest(ready['study']['manifest'])
    definition=json.loads(generation.read(files/ready['study']['minute_source_definition']))
    plan=json.loads(generation.read(files/definition['source_plan']))['windows'][w]
    with generation.activate():
        session,clear,sources=origin_verifier(generation,definition)
        clock=verify_closing_clock_sources(generation,definition,session,clear)
        fine_path=files/definition['minute_database'];coarse_path=files/definition['canonical_database']
        for p in (fine_path,coarse_path):
            with generation.open_verified(p):pass
        fine=sqlite3.connect(fine_path.as_uri()+'?mode=ro',uri=True)
        coarse=sqlite3.connect(coarse_path.as_uri()+'?mode=ro',uri=True)
        days=[str(d.date()) for d in pd.date_range(plan['period']['start'],plan['period']['end']) if get_trading_calendar().is_trading_day(d.date())]
        records=[]
        try:
            assert set(r[0] for r in fine.execute('SELECT DISTINCT symbol FROM bars'))==set(plan['primary'])
            for symbol in sorted(plan['primary']):
                actual=fine.execute('SELECT day,ts,o,h,l,c,v,source_hash FROM bars WHERE symbol=? ORDER BY ts',(symbol,)).fetchall()
                grouped={};count=0
                for row in actual:grouped.setdefault(row[0],[]).append(row[1:])
                if set(grouped)-set(days):raise ValueError('Fine database contains unexpected sessions')
                for day in days:
                    frame,digest,proof=session(symbol,day)
                    expected=[(r.timestamp.isoformat(),r.open,r.high,r.low,r.close,r.volume,digest) for r in frame.itertuples(index=False)]
                    if grouped.get(day,[])!=expected:raise ValueError('Independent fine source mismatch '+symbol+':'+day)
                    native=pd.DataFrame(coarse.execute('SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? AND day=? ORDER BY ts',(symbol,day)).fetchall(),columns=FIELDS)
                    same(aggregate_timeframe(frame,'5m'),native);count+=len(expected)
                records.append(dict(symbol=symbol,minute_rows=count,calendar_sessions=len(days)));clear()
                print(json.dumps(dict(window=w,**records[-1])),flush=True)
        finally:fine.close();coarse.close()
    n.write(t.ROOT/f'native_source_preflight_{w}.json',dict(utc=s.utc(),window=w,records=records,closing_clock=clock,
        source_generation_hash=ready['study']['source_generation_hash'],manifest_sha256=n.sha(ready['study']['manifest']),
        origins_read=sources,all_source_rows_recomputed=True,no_pnl_consumed=True,worker_sha256=n.sha(__file__)))


def main():
    global t
    action=sys.argv[1]
    if action in ('acceptance','probe'):
        path=Path(sys.argv[2]);info=read(path);files=isolate(info['files'])
        if action=='acceptance':acceptance_worker(path)
        else:
            import backtests.engine.retained_source
            import backtests.engine.olr_minute_source_v5
            import backtests.auto.shared.cache_keys
            import scripts.update_kis_intraday
        imports=verify_imports(files,info['manifest'])
        write(path.with_name(action+'_imports.json'),dict(worker_sha256=sha(__file__),native_imports=imports,passed=True))
    elif action=='sources':
        root=Path(sys.argv[2]);w=sys.argv[3];assert w in ('train','oos')
        t=SimpleNamespace(ROOT=root,ENGINES=root/'native_engines')
        info=read(t.ENGINES/w/'ready.json');files=isolate(info['files']);source_worker(w)
        imports=verify_imports(files,info['manifest'])
        write(root/f'native_source_imports_{w}.json',dict(worker_sha256=sha(__file__),native_imports=imports,passed=True))
    else:raise ValueError('Unknown native worker action')


if __name__=='__main__':main()
