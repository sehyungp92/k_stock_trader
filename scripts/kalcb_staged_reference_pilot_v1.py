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
    import psutil
    t=read(tests)
    if not t['passed'] or t['tests']<20:raise ValueError('Reference lifecycle tests required')
    verify(t['programs']+[t['log']])
    parent=BASE/'v47_staged_commitment_pilot_v2';old=read(parent/'protocol.json')
    for rel in ('supervisor_start.json','review_v2/supervisor_start.json'):
        identity=read(parent/rel)
        try:
            proc=psutil.Process(identity['pid'])
            if proc.cmdline()==identity['command'] and abs(proc.create_time()-identity['process_created'])<.01:raise ValueError('Parent still active')
        except psutil.NoSuchProcess:pass
    if not read(parent/'review_v2/completion.json')['passed']:raise ValueError('Parent not reviewed')
    verify(old['programs']+old['bindings']+[old['runtime_manifest']])
    done=read(parent/'labels/completion.json');verify([b for row in done['outputs'] for b in row.values()])
    root.mkdir(exist_ok=False);overlay=root/'runtime_overlay';overlay.mkdir()
    manifest=read(old['runtime_manifest']['path']);overrides=dict(manifest['overrides']);verify(list(overrides.values()))
    core=overlay/'strategy_kalcb/core/logic.py';core.parent.mkdir(parents=True)
    core.write_text(Path(overrides['strategy_kalcb.core.logic']['path']).read_text()+
        '\nfrom .staged_management_reference_v1 import install_reference as _install_management_reference\n_install_management_reference()\n',encoding='utf-8')
    overrides['strategy_kalcb.core.logic']=bound(core)
    additions=['strategy_kalcb/core/staged_management_reference_v1.py','backtests/strategies/kalcb/staged_reference_pilot_v1.py']
    for name in additions:
        target=overlay/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write((REPO/name).read_bytes())
        overrides[name[:-3].replace('/','.')]=bound(target)
    write(root/'runtime/inputs/manifest.json',{**manifest,'overrides':overrides,'parent_pilot_manifest':old['runtime_manifest']})
    specs=read(old['episodes']['path']);write(root/'episodes.json',specs)
    programs=[bound(__file__),*[bound(REPO/n) for n in additions],
        bound(REPO/'tests/test_kalcb_staged_management_reference_v1.py'),
        bound(REPO/'scripts/kalcb_staged_reference_review_v1.py'),
        bound(REPO/'scripts/kalcb_staged_reference_pipeline_v1.py'),
        bound(REPO/'backtests/strategies/kalcb/research_process_v1.py')]
    bindings=old['bindings']+[bound(parent/'protocol.json'),bound(parent/'completion.json'),bound(parent/'labels/completion.json'),
        bound(parent/'review_v2/completion.json'),bound(parent/'economic_decision.json'),bound(tests),bound(root/'episodes.json')]
    p={'registered_at_utc':now(),'version':'staged-initial-management-reference-v1',
        'programs':programs,'bindings':bindings,'runtime':old['runtime'],'runtime_manifest':bound(root/'runtime/inputs/manifest.json'),
        'databases':old['databases'],'episodes':bound(root/'episodes.json'),'parent_pilot':str(parent),
        'native_episodes':10,'authenticated_reused_controls':8,'pairs':8,
        'mechanism':'Exact V47 timing, half-initial quantity, completed35minute close/VWAP confirmation, total-share/notional cap and native execution preserved. Only contrast: when completing initial commitment, freeze initial fully-filled native average price as explicit reference for R-valued management and future observed excursions. Actual entry_price remains cumulative VWAP; broker cash/fees/PnL/funding and monetary cost guards remain actual-cost based. No R risk-unit, age or observed-history reset. This is a DIFFERENT management rule, not an accounting repair.',
        'population':'Same8known V47 IS issuer-day/scenario pairs; no date/issuer/threshold changes. Dependent conditional mechanics investigation; NOT independent or efficacy confirmation.',
        'sequence':'Two fresh disabled-reference exact bridges against V47 staged episodes BEFORE eight fresh reference episodes. Authenticate/reuse8V47 original controls.10fresh native episodes total. Independent review after exact worker exit, then economic decision. No automatic census/model/portfolio run.',
        'scope':'DATA ONLY. New matching-management outcomes; no old labels reused as new action outcomes. Full portfolio audit remains unverified, original parent100/125 unchanged, no live promotion.',
        'why':'V47 first completion rebased average entry used by R-management and converted an original winning hold into an early quick_exit. Test control-reference separation across ALL original pilot pairs, including losers; not an exemption for the diagnosed issuer.',
        'runtime_estimate':'3-8minutes10short native episodes plus independent review; single native worker,384MiB physical/1280MiB Windowscommit reserve; background pipeline.',
        'study_complete':False}
    write(root/'protocol.json',p);write(root/'study.json',{'data_only':True,'fresh_native_episodes':10,'study_complete':False})
    print(json.dumps({'registered':str(root),'fresh_native_episodes':10,'reused_controls':8}),flush=True)

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
    module=importlib.import_module('backtests.strategies.kalcb.staged_reference_pilot_v1')
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
