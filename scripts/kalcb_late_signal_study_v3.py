"""Finite signal follow-up owned by shared PhaseRunner; immutable research only."""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify,digest,link,env,PERIODS

PROGRAMS=('scripts/kalcb_late_signal_study_v3.py',
 'backtests/strategies/kalcb/late_artifact_validation_v2.py',
 'backtests/strategies/kalcb/late_signal_reuse_v1.py',
 'tests/backtests/strategies/test_kalcb_late_artifact_validation_v2.py',
 'tests/backtests/strategies/test_kalcb_late_signal_reuse_v1.py')


def recipes():
    result={n:{'prior':p,'quantity':0,'flat_only':False,'parent':p,'config':{},'signal':'none'}
            for n,p in [('control',100),('control_prior125',125)]}
    for prior in (100,125):
        for kind in ('raw_affine','compact','compact_affine'):
            result[f'prior{prior}_{kind}']={'prior':prior,'quantity':.25,'flat_only':False,
                'parent':prior,'config':{},'signal':kind}
    return result


def register(root,tests):
    from backtests.strategies.kalcb.late_signal_reuse_v1 import authenticate,publish,seed_verified_cache
    predecessor=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011/v9_signal')
    old=read(predecessor/'protocol.json');test=read(tests)
    if test['exit_code']!=0:raise ValueError('Tests must pass')
    verify(test['programs']+[test['log']])
    proof=authenticate(predecessor)
    if recipes()!=old['recipes']:raise ValueError('Economic recipes must remain unchanged')
    root.mkdir(parents=True,exist_ok=False)
    support={'research/late_signal_validator_v3/'+Path(s).name:bound(REPO/s) for s in PROGRAMS}
    protocol={**old,'version':'kalcb-late-signal-v3-strict-validator-recovery',
        'registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'programs':old['programs']+[bound(REPO/s) for s in PROGRAMS],
        'runtime_root':'C:/Users/sehyu/.codex/research/kalcb_late_entry_20261011/v10_signal',
        'overrides':{period:{**old['overrides'][period],**support} for period in PERIODS},
        'tests':test,'signal_completed_reuse':proof,
        'native_cases':32,'fresh_native_cases':31,'authenticated_screening_cases':1,
        'repair_v10':'Strict signal-policy compatibility only; complete independent accounting unchanged. No producer rerun; original V9 signal model and policy lineage retained.',
        'publication':'Authenticate inherited content blobs and hardlink into fresh cache; no strategy, data or execution change.'}
    write(root/'protocol.json',protocol)
    write(root/'study.json',{'recipes':protocol['recipes'],'code':str(REPO),
        'source_generation_hash':digest(root/'protocol.json'),'periods':{p:{'sessions':v[2]} for p,v in PERIODS.items()},
        'workers':1,'policy_name':'interaction_cost_margin','bound_inputs':[bound(root/'protocol.json')],
        'source_programs':protocol['programs']})
    publish(root,proof)
    write(root/'assessment_inheritance.json',{'contract':bound(predecessor.parent/'v5/assessment_protocol_v1.json'),
        'signal_assessment_inheritance':bound(predecessor/'assessment_inheritance.json'),
        'unchanged_units_and_uncertainty':True,'signal_recipes_and_models_unchanged':True,
        'producer_lineage_retains_v9':True,'quarter_only':True})
    write(root/'cache_reuse.json',seed_verified_cache(Path(old['runtime_root'])/'content_cache',Path(protocol['runtime_root'])/'content_cache'))
    print(json.dumps({'registered':str(root),'fresh_native_cases':31,'authenticated_screening_cases':1,'control_bridge':proof['control']['bridge']}),flush=True)


def commit_available():
    import ctypes
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','extended')]
    m=Memory();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise OSError('Cannot read commit capacity')
    return m.avail_page


def execute(root,command,label,cwd,state):
    import psutil
    while psutil.virtual_memory().available<384*1024**2 or commit_available()<1280*1024**2:
        write(root/'status.json',{**state,'stage':'waiting_for_measured_headroom','next':label,'updated_at_utc':datetime.now(timezone.utc).isoformat()});time.sleep(15)
    log=root/(label+'.log')
    with log.open('x',encoding='utf-8') as stream:
        child=subprocess.Popen(command,cwd=cwd,env=env(cwd),stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        identity={'pid':child.pid,'process_created':psutil.Process(child.pid).create_time(),'command':command}
        write(root/'status.json',{**state,'stage':label,'child':identity,'log':str(log),'updated_at_utc':datetime.now(timezone.utc).isoformat()})
        while child.poll() is None:
            try:
                p=psutil.Process(child.pid)
                with (root/'resource_observations.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps({'at':datetime.now(timezone.utc).isoformat(),'stage':label,'identity':identity,
                        'memory':p.memory_info()._asdict(),'io':p.io_counters()._asdict(),'cpu':p.cpu_times()._asdict()})+'\n')
            except psutil.NoSuchProcess:pass
            time.sleep(5)
    if child.returncode:
        write(root/'failed_stage.json',{'stage':label,'returncode':child.returncode,'log':bound(log),'preserve_evidence':True})
        raise RuntimeError('Registered stage failed: '+label)
    return bound(log)


def native(root,name,period,scenario,authenticated,state):
    from backtests.strategies.kalcb.late_entry_io_v4 import freeze
    from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_native
    from backtests.strategies.kalcb.population_research_io_v1 import bridge_accounts
    p=read(root/'protocol.json');recipe=p['recipes'][name];old=p['profiles'][str(recipe['prior'])][period]
    if recipe['quantity']:
        progress=read(root/'progress.json')
        if sum(r['is_identity'] for r in progress)!=8:raise ValueError('All eight unchanged parent bridges must pass first')
    item=freeze(root,name,period,old,root/'policies'/name/period,p['overrides'][period],authenticated)
    code=Path(item['folder'])/'inputs/files';out=root/'cases'/name/period/scenario
    if (name,period,scenario)==('control','is','baseline'):
        verify(p['signal_completed_reuse']['bindings'])
        metrics=validate_native(out,item,scenario)
        write(out/'identity_bridge.json',bridge_accounts(out,Path(old['references'][scenario]).parent))
        row={'name':name,'period':period,'scenario':scenario,'metrics':metrics,'result':bound(out/'result.json'),
            'authenticated_reuse':True,'is_identity':True,'reuse_registration':bound(root/'protocol.json')}
        if (root/'progress.json').exists():raise ValueError('Refuse duplicate reused control')
        write(root/'progress.json',[row]);print(json.dumps(row),flush=True)
        return read(out/'result.json')
    command=[sys.executable,'-B','-u','-m','scripts.kalcb_executed_value_replay','--plan',item['plan']['path'],
        '--policies',str(code/'research/kalcb_executed_value/policies'),'--policy','interaction_cost_margin',
        '--audit-manifest',item['manifest']['path'],'--scenario',scenario,'--output',str(out)]
    log=execute(root,command,'native_'+name+'_'+period+'_'+scenario,code,state)
    metrics=validate_native(out,item,scenario)
    if recipe['quantity']==0:write(out/'identity_bridge.json',bridge_accounts(out,Path(old['references'][scenario]).parent))
    progress=read(root/'progress.json') if (root/'progress.json').exists() else []
    row={'name':name,'period':period,'scenario':scenario,'metrics':metrics,'result':bound(out/'result.json'),
        'log':log,'authenticated_reuse':False,'is_identity':recipe['quantity']==0}
    progress.append(row);write(root/'progress.json',progress);print(json.dumps(row),flush=True)
    return read(out/'result.json')


def run(root):
    import psutil,msvcrt
    from backtests.strategies.kalcb.late_signal_plugin_v1 import LateSignalPlugin
    from backtests.auto.shared.phase_runner import PhaseRunner
    from backtests.auto.shared.round_manager import RoundManager
    p=read(root/'protocol.json');verify(p['programs']+p['input_bindings']+p['data_bindings'])
    for proc in psutil.process_iter(['pid','cmdline']):
        args=proc.info['cmdline'] or []
        if proc.pid==os.getpid():continue
        if any('kalcb_late_' in str(a) and ('study' in str(a) or 'launcher' in str(a)) for a in args) and ('run' in args or 'preflight' in args):raise ValueError('Existing study supervisor')
        if 'scripts.kalcb_executed_value_replay' in args:raise ValueError('Existing native worker')
    lock=(root/'run.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if (root/'supervisor_start.json').exists():raise ValueError('Preserve earlier attempt; use versioned successor')
    state={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'command':psutil.Process().cmdline(),'mode':'run'}
    write(root/'supervisor_start.json',state)
    try:
        verify(p['signal_completed_reuse']['bindings'])
        if not read(root/'policies/completion.json')['complete']:raise ValueError('Unfinished reused policies')
        authenticated=set()
        def case(name,period,scenario):return native(root,name,period,scenario,authenticated,state)
        plugin=LateSignalPlugin(root/'study.json',case);manager=RoundManager('stock',plugin.name,base_dir=root/'framework')
        PhaseRunner(plugin,manager.get_round_dir(1),round_name='First-hour signal reliability and compact representation',round_manager=manager,
            round_num=1,max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()
        if len(read(root/'progress.json'))!=32:raise ValueError('Incomplete registered cases')
        write(root/'stage_completion.json',{'complete':True,'native_cases':32,'fresh_native_cases':31,'authenticated_screening_cases':1,'study_complete':False,'next':p['remaining']})
        write(root/'status.json',{**state,'stage':'awaiting_economic_assessment_and_finalist_qualification','child':None})
    except Exception as e:
        write(root/'status.json',{**state,'stage':'attention_required','error':repr(e)});raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=('register','run'))
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--tests',type=Path);ap.add_argument('--code',type=Path)
    args=ap.parse_args();root=args.root.resolve()
    if args.stage=='register':register(root,args.tests)
    elif args.stage=='run':run(root)
    else:raise ValueError('Unsupported stage')
