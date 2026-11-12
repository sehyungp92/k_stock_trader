"""Registered 18-episode staged-entry mechanics pilot, existing data only."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
BASE=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
CORE='strategy_kalcb/core/logic.py'
HELPER='strategy_kalcb/core/staged_commitment_v1.py'
PILOT='backtests/strategies/kalcb/staged_commitment_pilot_v1.py'
EXTRA=[HELPER,PILOT,'backtests/strategies/kalcb/pending_funding_labels_v1.py',
       'backtests/strategies/kalcb/checkpoint_contrast_labels_v1.py']
PREFIXES=('backtests','scripts','kis_core','strategy_kalcb','strategy_common')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def now():return datetime.now(timezone.utc).isoformat()
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bound(p):return {'path':str(Path(p).resolve()),'sha256':digest(p)}
def verify(bs):
    for b in bs:
        if digest(b['path'])!=b['sha256']:raise ValueError('Registered source changed '+b['path'])
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False,default=str)

def register(root,tests):
    t=read(tests)
    if not t['passed'] or t['tests']<16:raise ValueError('Native lifecycle tests required')
    verify(t['programs']+[t['log']])
    source=BASE/'v24_pending_funding_labels/protocol.json';old=read(source)
    verify(old['programs']+old['bindings']+[old['runtime_manifest']])
    ap=read(BASE/'v29_exit_latency_audit/protocol.json');verify(ap['programs']+ap['bindings'])
    root.mkdir(exist_ok=False);code=Path(old['runtime']).resolve()
    overlay=root/'runtime_overlay';overlay.mkdir()
    manifest=read(old['runtime_manifest']['path']);files=dict(manifest['files'])
    for name,sha in files.items():
        if digest(code/name)!=sha:raise ValueError('Changed original publication '+name)
    tree=ast.parse((REPO/HELPER).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='derive_logic')
    ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),HELPER,'exec'),ns)
    target=overlay/CORE;target.parent.mkdir(parents=True)
    target.write_text(ns['derive_logic']((code/CORE).read_text(encoding='utf-8')),encoding='utf-8')
    overrides={'strategy_kalcb.core.logic':bound(target)}
    for name in EXTRA:
        target=overlay/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write((REPO/name).read_bytes())
        overrides[name[:-3].replace('/','.')]=bound(target)
    write(root/'runtime/inputs/manifest.json',{'files':files,'parent':old['runtime_manifest'],
        'overrides':overrides,'core_override':{'original_sha256':files[CORE],
            'derived_sha256':overrides['strategy_kalcb.core.logic']['sha256']},
        'publication':'Authenticated shared immutable parent plus5exact import overrides; avoids exhausted NTFS hardlink counts, no source/data mutation or duplicate copies.'})
    profile_path=BASE/'v25_pending_funding_screen/cases/control/is/input_profile.json';profile=read(profile_path)
    snapshot=Path(profile['policy']['path']).parent/'interaction_cost_margin.snapshots.json'
    bindings=[bound(source),old['runtime_manifest'],bound(profile_path),profile['plan'],profile['policy'],bound(snapshot),bound(tests),
              bound(BASE/'v46_risk_contract_materiality/review_v1/completion.json'),bound(BASE/'v46_risk_contract_materiality/economic_decision.json')]
    bindings += [profile['manifest'], bound(read(profile['plan']['path'])['config'])]
    specs=[]
    for scenario in ('baseline','combined'):
        path=BASE/f'v25_pending_funding_screen/cases/control/is/{scenario}/observer_evidence.json'
        bindings.append(bound(path));seen=set();selected=[]
        for e in read(path)['entry_exposures']:
            if e['action']['metadata'].get('entry_route')!='first30_open_anchor':continue
            at=datetime.fromisoformat(e['decision_available_at']);day=at.date().isoformat()
            if day in seen:continue
            seen.add(day);selected.append(e)
            if len(selected)==4:break
        if len(selected)!=4:raise ValueError('Missing chronological pilot population')
        for i,e in enumerate(selected):
            at=datetime.fromisoformat(e['decision_available_at']);a=e['action'];acct=e['account_before']
            if acct['positions'] or acct['pending_orders'] or acct['actual_cash']!=acct['actual_equity']:
                raise ValueError('Pilot requires first flat-account action, without outcome filtering')
            specs.append({'id':at.date().isoformat()+'_'+a['symbol']+'_'+scenario,
                'day':at.date().isoformat(),'symbol':a['symbol'],'scenario':scenario,'profile':profile,
                'snapshot':bound(snapshot),'initial_equity':acct['actual_equity'],'parent_qty':a['qty'],
                'reference_price':a['metadata']['entry_price_ref'],'available_at':at.isoformat(),
                'parent_order_id':e['order_id'],'source_exposure':bound(path),'bridge':i==0})
    write(root/'episodes.json',specs)
    programs=[bound(__file__),*[bound(REPO/n) for n in EXTRA],bound(REPO/'tests/test_kalcb_staged_commitment_v1.py')]
    protocol={'registered_at_utc':now(),'version':'staged-commitment-mechanics-pilot-v2-packaging-only',
        'programs':programs,'bindings':bindings+[bound(root/'episodes.json')],
        'runtime':str(code),'runtime_manifest':bound(root/'runtime/inputs/manifest.json'),
        'databases':old['databases'],'episodes':bound(root/'episodes.json'),'native_episodes':18,'pairs':8,
        'population':'First chronological opening SubmitEntry on each of first4 IS sessions, separately in prior100 baseline and combined; no fill/PnL/issuer filter. Same days/scenarios dependent. No OOS efficacy evaluation in mechanics pilot.',
        'mechanism':'Original parent selected opening quantity Q. Initial floor(Q/2), Q=1 unchanged. Exactly one completion opportunity after7th consecutive completed positive-volume5mbar (minute35), after parent bar+timer management. Close>=original completed entry reference AND close>=own contemporary sessionVWAP. Only initial fully filled, no pending initial, no parent exit/sale. Additional dose <= original Q minus cumulative buys AND remaining original reference notional divided by new completed close. Native heat gate, next-OPEN/gaps/cost/funding/capacity retained. Completion expires after one further5m interval; terminal/exit late fills liquidated through native orders. No R/age/excursion/lifetime reset; no extra successful trade counted.',
        'scope':'Mechanics and exact isolated full-cost labels only. New sequential action: old quarter/half/full labels are NOT its targets. No payoff fit, new portfolio policy or whole-portfolio qualification. No live enablement.',
        'gates':'All native raw-source/fill/fee/cash/inventory/timestampedMTM checks required. Two disabled-hook exact bridges against original native entry points. No economic choice based on this small pilot; must retain every failure/loss/no-fill and perform independent review. No automatic larger census or parameter grid.',
        'portfolio_limit':'Actual simultaneous pending/account funding at completion needs full native portfolio verification later; isolated flat starting capital is not evidence of funded availability.',
        'runtime_estimate':'5-10 minutes publication+18short native episodes plus independent review. One worker. No future replay authorized by this registration.',
        'study_complete':False}
    write(root/'protocol.json',protocol);write(root/'study.json',{'data_only':True,'native_episodes':18,'study_complete':False})
    print(json.dumps({'registered':str(root),'runtime_files':len(files),'native_episodes':18}),flush=True)

def run(root):
    import ctypes
    import psutil
    p=read(root/'protocol.json');verify(p['programs']+p['bindings']+[p['runtime_manifest']])
    if (root/'supervisor_start.json').exists():raise ValueError('Never restart a registered attempt')
    mem=psutil.virtual_memory()
    if mem.available<384*1024**2:raise ValueError('Physical memory reserve unavailable')
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in
            ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','extended')]
    memory=Memory();memory.length=ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)) or memory.avail_page<1280*1024**2:
        raise ValueError('Measured Windows commit reserve unavailable')
    for proc in psutil.process_iter(['pid','name','cmdline']):
        args=proc.info['cmdline'] or []
        if proc.pid!=os.getpid() and 'python' in (proc.info['name'] or '').lower() and any('kalcb_' in x and ('worker' in x or 'study' in x or 'pilot' in x) for x in args) and any(x in ('run','produce','smoke') for x in args):
            raise ValueError('Another KALCB worker is active')
    me=psutil.Process();write(root/'supervisor_start.json',{'pid':me.pid,'process_created':me.create_time(),'command':me.cmdline(),'at_utc':now()})
    if any(n.split('.')[0] in PREFIXES for n in sys.modules):raise ValueError('Application loaded before isolation')
    code=Path(p['runtime']).resolve();sys.path[:]=[str(code)]+[x for x in sys.path if x and not Path(x).resolve().is_relative_to(REPO)]
    for key in ('TEMP','TMP'):os.environ[key]=str(root/'temp')
    (root/'temp').mkdir(exist_ok=True)
    import importlib.abc
    import importlib.util
    manifest=read(p['runtime_manifest']['path']);overrides=manifest['overrides'];verify(list(overrides.values()))
    class FrozenOverrides(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in overrides:
                return importlib.util.spec_from_file_location(fullname,overrides[fullname]['path'])
            return None
    sys.meta_path.insert(0,FrozenOverrides())
    module=importlib.import_module('backtests.strategies.kalcb.staged_commitment_pilot_v1')
    def origins():
        result={}
        for name,obj in list(sys.modules.items()):
            path=getattr(obj,'__file__',None)
            if not path or name.split('.')[0] not in PREFIXES:continue
            path=Path(path).resolve()
            if name in overrides:
                if path!=Path(overrides[name]['path']).resolve() or digest(path)!=overrides[name]['sha256']:
                    raise ValueError('Changed exact override '+name)
                result[name]=bound(path);continue
            if not path.is_relative_to(code):raise ValueError('Escaped frozen import '+name)
            sha=manifest['files'].get(path.relative_to(code).as_posix())
            if sha is None or digest(path)!=sha:raise ValueError('Changed frozen import '+name)
            result[name]=bound(path)
        return result
    write(root/'imports_before.json',origins());module.run(root);write(root/'imports_after.json',origins())
    verify(p['programs']+p['bindings']);done=read(root/'labels/completion.json')
    if done['episodes']!=18 or done['pairs']!=8:raise ValueError('Incomplete fixed pilot')
    write(root/'completion.json',{'complete':True,'study_complete':False,'at_utc':now(),'labels':bound(root/'labels/completion.json')})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('register','run'));ap.add_argument('--root',type=Path,required=True);ap.add_argument('--tests',type=Path)
    a=ap.parse_args()
    try:register(a.root,a.tests) if a.mode=='register' else run(a.root)
    except Exception as e:
        if a.root.exists() and not (a.root/'failure.json').exists():write(a.root/'failure.json',{'error':repr(e),'at_utc':now()})
        raise
