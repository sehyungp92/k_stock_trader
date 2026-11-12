"""Source-bound funded native OLR experiment. Never modifies a live profile."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,hashlib,json,os,shutil,subprocess,sys

REPO=Path(__file__).resolve().parents[1]
BASE=Path('E:/CodexResearch/k_stock_trader/olr_round1_optimization_20261010/structural_v6/drawdown_reserve_v1')
CHANGED=('strategy_olr/core/logic.py','strategy_olr/core/core_models.py','strategy_olr/core/allocation.py',
    'strategy_olr/core/opportunity.py','strategy_olr/core/joint_allocation.py','strategy_olr/core/joint_decision.py',
    'backtests/strategies/olr/runner.py','backtests/strategies/olr/decision_account.py',
    'backtests/strategies/olr/economic_policy.py')

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sig(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,default=str)
def source_hash(root):
    dirs=('backtests','strategy_common','strategy_kalcb','strategy_olr','strategy_pcim','deployment','instrumentation','oms','oms_client','kis_core','scripts')
    paths=[p for d in dirs for p in (root/d).rglob('*.py')];paths.extend((root/'kis_core/data').glob('*.yaml'))
    return sig([dict(path=p.relative_to(root).as_posix(),sha256=sha(p)) for p in sorted(paths)])

def retain_full_mark_journal(text):
    """Future generations retain the phase marks needed for DD attribution."""
    old="('minute_day_scope_v5','minute_exit_signal_v5','fill','submit','cancel','expire','reject')"
    new="('minute_day_scope_v5','minute_exit_signal_v5','opening_marks','closing_marks','fill','submit','cancel','expire','reject')"
    if text.count(new)==1 and old not in text:return text
    if text.count(old)!=1:raise ValueError('Unrecognized native minute journal export contract')
    return text.replace(old,new)

def prepare(root,window):
    old=read(BASE/'native_engines'/window/'ready.json');source=Path(old['files'])
    dest=root/'native_engines'/window/'files';dest.mkdir(parents=True,exist_ok=False)
    manifest=copy.deepcopy(old['manifest']);manifest.pop('manifest_hash',None)
    for i,(name,h) in enumerate(manifest['files'].items()):
        src=source/name;dst=dest/name;dst.parent.mkdir(parents=True,exist_ok=True)
        if sha(src)!=h:raise ValueError('Parent source changed: '+name)
        if src.suffix.lower() in ('.parquet','.sqlite','.db','.gz','.pickle','.pkl'):os.link(src,dst)
        else:shutil.copy2(src,dst)
        if i%1000==0:print(json.dumps(dict(phase='copy',files=i)),flush=True)
    for name in CHANGED:
        p=dest/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,p);manifest['files'][name]=sha(p)
    p=dest/'scripts/portfolio_synergy_study.py';text=p.read_text(encoding='utf-8')
    anchor="        minute_adapter=next(a for a in adapters if a.strategy_id=='OLR')\n"
    assert text.count(anchor)==1
    text=text.replace(anchor,anchor+'''        if study.get('economic_policy_settings'):
            from backtests.strategies.olr.economic_policy import EconomicPolicy,EconomicSettings
            settings=EconomicSettings(**bound('economic_policy_settings'))
            policy=EconomicPolicy(bound('economic_policy_labels'),ocfg,args.scenario,settings)
            minute_adapter.economic_policy=policy
            minute_adapter.decision_context_provider=policy.entry_context
''')
    anchor="    if olr_adapter is not None:\n        write_json(args.output/'capacity_terminal_state.json'"
    assert text.count(anchor)==1
    text=text.replace(anchor,"""    if olr_adapter is not None and getattr(olr_adapter,'economic_policy',None) is not None:
        write_json(args.output/'economic_policy_trace.json',olr_adapter.economic_policy.trace)
        write_json(args.output/'economic_policy_fits.json',olr_adapter.economic_policy.model.fits)
        write_json(args.output/'economic_opportunities.json',olr_adapter.state.meta.get('economic_opportunities_v1',{}))
    if olr_adapter is not None:
        write_json(args.output/'capacity_terminal_state.json'""")
    text=retain_full_mark_journal(text)
    compile(text,str(p),'exec');p.write_text(text,encoding='utf-8');manifest['files']['scripts/portfolio_synergy_study.py']=sha(p)
    labels=dest/'research/economic_policy/labels.json';labels.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(root/'labels.json',labels);manifest['files'][labels.relative_to(dest).as_posix()]=sha(labels)
    study=copy.deepcopy(old['study']);study['code']=str(dest);study['source_generation_hash']=source_hash(dest)
    manifest['source_generation_hash']=study['source_generation_hash']
    write(root/'native_engines'/window/'ready.json',dict(files=str(dest),study=study,manifest=manifest))
    print(json.dumps(dict(prepared=window,source=study['source_generation_hash'])),flush=True)

def fork_engine(root,parent,window):
    """Immutable shared inputs; changed implementation gets private new files."""
    old=read(parent/'native_engines'/window/'ready.json');source=Path(old['files'])
    dest=root/'native_engines'/window/'files';dest.mkdir(parents=True,exist_ok=False)
    manifest=copy.deepcopy(old['manifest'])
    for name,h in manifest['files'].items():
        src=source/name;dst=dest/name;dst.parent.mkdir(parents=True,exist_ok=True)
        if name=='scripts/portfolio_synergy_study.py':
            text=retain_full_mark_journal(src.read_text(encoding='utf-8'))
            compile(text,str(dst),'exec');dst.write_text(text,encoding='utf-8');manifest['files'][name]=sha(dst)
        elif name in CHANGED:
            shutil.copy2(REPO/name,dst);manifest['files'][name]=sha(dst)
        else:os.link(src,dst)
    study=copy.deepcopy(old['study']);study['code']=str(dest);study['source_generation_hash']=source_hash(dest)
    manifest['source_generation_hash']=study['source_generation_hash']
    write(root/'native_engines'/window/'ready.json',dict(files=str(dest),study=study,manifest=manifest))
    write(root/'fork_receipt.json',dict(parent_ready_sha256=sha(parent/'native_engines'/window/'ready.json'),
        changed={name:h for name,h in manifest['files'].items() if h!=old['manifest']['files'].get(name)},
        identical_labels=True,prior_results_rewritten=False))
    print(json.dumps(dict(forked=window,source=study['source_generation_hash'])),flush=True)

def run(root,window,mode,scenario,capped,attempt=0):
    r=read(root/'native_engines'/window/'ready.json');files=Path(r['files']);s=copy.deepcopy(r['study']);m=copy.deepcopy(r['manifest'])
    tag=mode+'_'+scenario+('_capped' if capped else '')+('_attempt'+str(attempt) if attempt else '');out=root/'cases'/window/tag
    if out.exists():raise FileExistsError('Do not overwrite case: '+str(out))
    prefix='research/economic_policy/'+window+'/'+tag+'/'
    if mode!='baseline':
        settings=dict(allocation=mode in ('allocation','combined'),management=mode in ('management','combined'))
        name=prefix+'settings.json';write(files/name,settings);m['files'][name]=sha(files/name);s['economic_policy_settings']=name
        s['economic_policy_labels']='research/economic_policy/labels.json'
        # Preserve native warm-up. The core bypasses heuristic redistribution
        # only when both economic models have sufficient matured support.
    s['manifest']=str(files.parent/(tag+'.manifest.json'));s['configuration']['input_manifest_path']=s['manifest']
    s['spec_path']=prefix+'spec.json'
    write(files/s['spec_path'],{k:v for k,v in s.items() if k not in ('code','manifest','spec_path','source_generation_hash')})
    m['files'][s['spec_path']]=sha(files/s['spec_path']);write(s['manifest'],{**m,'manifest_hash':sig(m)})
    sp=root/'runtime'/window/tag/'study.json';write(sp,s)
    cmd=[sys.executable,'-B','-u',str(files/'scripts/portfolio_synergy_study.py'),'replay','--study',str(sp),
         '--recipe','candidate','--scenario',scenario,'--standalone','OLR','--output',str(out),'--full-audit']
    if capped:cmd.append('--no-upward-compounding')
    env=os.environ.copy();tmp=root/'runtime_tmp'/tag;tmp.mkdir(parents=True,exist_ok=True)
    env.update(TEMP=str(tmp),TMP=str(tmp),PYTHONIOENCODING='utf-8',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    log=root/'logs'/window/(tag+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x',encoding='utf-8') as f:
        p=subprocess.Popen(cmd,cwd=files,stdout=f,stderr=subprocess.STDOUT,env=env,creationflags=0x08000000)
        write(sp.parent/'process.json',dict(pid=p.pid,command=cmd,utc=datetime.now(timezone.utc).isoformat()))
        print(json.dumps(dict(pid=p.pid,log=str(log))),flush=True);rc=p.wait()
    if rc:raise RuntimeError('Native replay failed; preserve output: '+str(log))
    print(json.dumps(dict(case=str(out),result=read(out/'result.json').get('return_pct'))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('prepare','run','fork'));p.add_argument('--root',type=Path,required=True)
    p.add_argument('--parent',type=Path)
    p.add_argument('--window',default='train');p.add_argument('--mode',choices=('baseline','lifecycle','allocation','management','combined'),default='baseline')
    p.add_argument('--scenario',default='baseline');p.add_argument('--capped',action='store_true');p.add_argument('--attempt',type=int,default=0);a=p.parse_args()
    if a.phase=='prepare':prepare(a.root,a.window)
    elif a.phase=='fork':fork_engine(a.root,a.parent,a.window)
    else:run(a.root,a.window,a.mode,a.scenario,a.capped,a.attempt)
