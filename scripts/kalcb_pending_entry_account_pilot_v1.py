"""Immutable finite native account-branch pilot and serial background lifecycle."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

BASE=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
REPO=Path(__file__).resolve().parents[1]
PREFIXES=('scripts','backtests','strategy_kalcb','strategy_common','kis_core')
EXTRA=('strategy_kalcb/core/pending_entry_revalidation_v1.py',
       'backtests/strategies/kalcb/pending_entry_account_pilot_v1.py',
       'backtests/strategies/kalcb/checkpoint_contrast_labels_v1.py')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def now():return datetime.now(timezone.utc).isoformat()
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bound(p):return {'path':str(Path(p).resolve()),'sha256':digest(p)}
def verify(bs):
    for b in bs:
        if digest(b['path'])!=b['sha256']:raise ValueError('Changed bound source '+b['path'])
def write(p,v,exclusive=True):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x' if exclusive else 'w',encoding='utf-8') as f:json.dump(v,f,indent=2,default=str,allow_nan=False)
def identity():
    import psutil
    me=psutil.Process();return {'pid':me.pid,'process_created':me.create_time(),'command':me.cmdline(),'at_utc':now()}
def absent(receipt):
    import psutil
    try:
        obj=psutil.Process(receipt['pid'])
        if abs(obj.create_time()-receipt['process_created'])<.01 and obj.cmdline()==receipt['command']:
            raise ValueError('Exact process is still active')
    except psutil.NoSuchProcess:pass


def register(root,tests):
    t=read(tests)
    if not t['passed'] or t['tests']<21:raise ValueError('Lifecycle and native branch tests required')
    verify(t['programs']+[t['log']])
    probe_path=BASE/'v51_pending_entry_revalidation_preflight/frozen_account_probe_v1.json'
    probe=read(probe_path)
    if not probe['passed'] or len(probe['rows'])!=2 or probe['historical_cases']!=0:raise ValueError('Frozen native smoke required')
    verify(list(probe['origins'].values()))
    profile_path=BASE/'v25_pending_funding_screen/cases/control/is/input_profile.json';profile=read(profile_path)
    verify([profile[k] for k in ('plan','policy','manifest')])
    manifest=read(profile['manifest']['path']);code=Path(profile['manifest']['path']).parent/'files'
    root.mkdir(exist_ok=False)
    for name,sha in manifest['files'].items():
        if digest(code/name)!=sha:raise ValueError('Changed parent publication '+name)
    overrides={};programs=[bound(REPO/'scripts/kalcb_pending_entry_account_pilot_v1.py'),
        bound(REPO/'scripts/kalcb_pending_entry_account_review_v1.py'),
        bound(REPO/'tests/test_kalcb_pending_entry_account_pilot_v1.py')]
    for name in EXTRA:
        source=REPO/name;target=root/'runtime_overlay'/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(source.read_bytes())
        overrides[name[:-3].replace('/','.')]=bound(target);programs.append(bound(source))
    for name in ('kalcb_pending_entry_account_pilot_v1.py','kalcb_pending_entry_account_review_v1.py'):
        target=root/'runners'/name;target.parent.mkdir(exist_ok=True)
        with target.open('xb') as f:f.write((REPO/'scripts'/name).read_bytes())
        programs.append(bound(target))
    write(root/'runtime_manifest.json',{'parent':profile['manifest'],'files':manifest['files'],'overrides':overrides,
        'scope':'Original source reused in place with three exact read-only research import overlays; no frozen hardlink edited.'})
    old=read(BASE/'v24_pending_funding_labels/protocol.json');bindings=[bound(tests),bound(probe_path),bound(profile_path),
        profile['plan'],profile['policy'],profile['manifest'],bound(BASE/'v50_pending_entry_materiality/economic_decision.json')]
    accounts={s:str(BASE/'v25_pending_funding_screen/cases/control/is'/s) for s in ('baseline','combined')}
    for folder in accounts.values():
        bindings.extend(bound(Path(folder)/n) for n in ('fills.json','daily.json','equity.json','execution_audit.json','completion.json'))
    bindings.extend(bound(x) for x in old['databases'])
    plan=read(profile['plan']['path']);bindings.extend({'path':k,'sha256':v} for k,v in plan['input_sha256'].items())
    protocol={'registered_at_utc':now(),'version':'pending-entry-actual-account-mechanics-v1','programs':programs,
        'bindings':bindings,'profile':bound(profile_path),'runtime':str(code),'runtime_manifest':bound(root/'runtime_manifest.json'),
        'databases':old['databases'],'parent_accounts':accounts,'scenarios':['baseline','combined'],'states_per_scenario':4,
        'population':'First four chronological IS account states per scenario, starting at original August7 initial capital and continuous parent history, with at least one actual pending original opening BUY satisfying the frozen V51 cancellation predicate at09:35 after parent bar/timer. All eligible same-clock orders in one account intervention. No future fill/PnL, issuer/date or forecast selection. All attempted-state reasons retained.',
        'finite_cases':'Two unchanged parent prefixes, ending at fourth eligible state session; eight same-account pairs (sixteen EOD branch suffixes). No new full-period strategy or forecast fit. No automatic extension if insufficient states.',
        'action':'Original V51 helper; preserve all actual cash/equity/inventory/pending/risk/management/rank/clock state. Cancel unfilled weak-confirmation opening remainder, keep filled stake. Original quantity, routing, management, costs, funding and native next-bar semantics unchanged.',
        'comparability':'Every parent prefix must exactly match authenticated V25 fills, daily balances and timestamped MTM before branch outcomes are recorded; disabled branch must match every broker/adapter state attribute against concurrent untouched parent. Whole account session suffix includes descendants; no summation as strategy return.',
        'gates':'Independent raw attainable price, fee, clock, integer fill, cash/inventory/MTM checks; all frozen origin hashes; exact prestate identity; no candidate later fills on ACKed cancel. Full portfolio qualification remains UNVERIFIED absent new whole-policy qualification manifest.',
        'scope':'Native mechanics and action-matched conditional account labels. No efficacy conclusion, model fitting, live change, parent replacement or portfolio promotion.',
        'runtime_estimate':'10-25 minutes for source authentication, two parent prefixes and sixteen short continuations, then independent read-only review. One native process, sequential scenarios. Not an optimization-success ETA.',
        'study_complete':False}
    write(root/'protocol.json',protocol);write(root/'study.json',{'native_parent_prefixes':2,'paired_states':8,'branch_suffixes':16,'study_complete':False})
    print(json.dumps({'registered':str(root),'files':len(manifest['files']),'states':8}),flush=True)


def worker(root,scenario):
    import ctypes
    import psutil
    p=read(root/'protocol.json');verify(p['programs']+p['bindings']+[p['runtime_manifest']])
    if psutil.virtual_memory().available<384*1024**2:raise ValueError('Physical reserve unavailable')
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in
            ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','extended')]
    memory=Memory();memory.length=ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)) or memory.avail_page<1280*1024**2:
        raise ValueError('Windows commit reserve unavailable')
    write(root/f'{scenario}_start.json',identity())
    if any(n.split('.')[0] in PREFIXES for n in sys.modules):raise ValueError('Application imported before isolation')
    code=Path(p['runtime']).resolve();sys.path[:]=[str(code)]+[x for x in sys.path if x and not Path(x).resolve().is_relative_to(REPO)]
    import importlib.abc
    import importlib.util
    manifest=read(p['runtime_manifest']['path']);overrides=manifest['overrides'];verify(list(overrides.values()))
    class Overrides(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname in overrides:return importlib.util.spec_from_file_location(fullname,overrides[fullname]['path'])
    sys.meta_path.insert(0,Overrides())
    module=importlib.import_module('backtests.strategies.kalcb.pending_entry_account_pilot_v1')
    def origins():
        result={}
        for name,obj in list(sys.modules.items()):
            path=getattr(obj,'__file__',None)
            if not path or name.split('.')[0] not in PREFIXES:continue
            path=Path(path).resolve()
            if name in overrides:
                if path!=Path(overrides[name]['path']).resolve() or digest(path)!=overrides[name]['sha256']:raise ValueError('Changed overlay '+name)
            elif not path.is_relative_to(code) or manifest['files'].get(path.relative_to(code).as_posix())!=digest(path):
                raise ValueError('Application origin outside exact frozen source '+name)
            result[name]=bound(path)
        return result
    write(root/f'{scenario}_imports_before.json',origins());module.run(root,scenario)
    write(root/f'{scenario}_imports_after.json',origins());verify(p['programs']+p['bindings'])
    write(root/f'{scenario}_completion.json',{'complete':True,'parent':bound(root/'parents'/scenario/'completion.json'),'study_complete':False})


def pipeline(root):
    import psutil
    p=read(root/'protocol.json');verify(p['programs']+p['bindings'])
    for proc in psutil.process_iter(['pid','name','cmdline']):
        args=proc.info['cmdline'] or []
        if proc.pid!=os.getpid() and 'python' in (proc.info['name'] or '').lower() and any('kalcb_' in x for x in args) and any(x in ('run','produce','smoke','worker','pipeline') for x in args):
            raise ValueError('Another KALCB producer/pipeline is active')
    write(root/'supervisor_start.json',identity())
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8',
         **{k:'1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
         'TEMP':str(root/'temp'),'TMP':str(root/'temp')};(root/'temp').mkdir(exist_ok=True)
    stages=[(s,[sys.executable,'-I','-B',str(root/'runners/kalcb_pending_entry_account_pilot_v1.py'),'worker','--root',str(root),'--scenario',s]) for s in p['scenarios']]
    stages.append(('review',[sys.executable,'-I','-B',str(root/'runners/kalcb_pending_entry_account_review_v1.py'),'--root',str(root)]))
    for stage,command in stages:
        write(root/'status.json',{'stage':stage,'status':'running','at_utc':now(),'study_complete':False},False)
        with (root/(stage+'.log')).open('xb') as out,(root/(stage+'.err.log')).open('xb') as err:
            child=subprocess.Popen(command,cwd=root,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
            process=psutil.Process(child.pid)
            receipt={'pid':child.pid,'process_created':process.create_time(),'command':process.cmdline(),'at_utc':now()}
            if receipt['command']!=command:raise ValueError('Child command mismatch')
            write(root/(stage+'_launch.json'),receipt)
            code=child.wait()
        absent(receipt);write(root/(stage+'_exit.json'),{**receipt,'exit_code':code,'verified_absent_at_utc':now()})
        if code:raise RuntimeError('Stage failed '+stage)
    write(root/'completion.json',{'complete':True,'review':bound(root/'review_v1/completion.json'),'study_complete':False,'at_utc':now()})
    write(root/'status.json',{'status':'complete','study_complete':False,'at_utc':now()},False)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('register','worker','pipeline'));ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--tests',type=Path);ap.add_argument('--scenario',choices=('baseline','combined'));a=ap.parse_args()
    try:
        if a.mode=='register':register(a.root,a.tests)
        elif a.mode=='worker':worker(a.root,a.scenario)
        else:pipeline(a.root)
    except Exception as exc:
        if a.root.exists():write(a.root/f'failure_{a.mode}_{a.scenario or "all"}.json',{'error':repr(exc),'at_utc':now(),'study_complete':False})
        raise
