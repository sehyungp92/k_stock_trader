"""Registered first-hour additions on the two qualified expanding parents.

Preparation stages are data producers. The shared PhaseRunner alone owns the
finite native experiment lifecycle; no promotion is performed by this runner.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify,env,PERIODS,digest
from backtests.strategies.kalcb.late_entry_io_v4 import make_overrides,freeze
from backtests.strategies.kalcb.late_artifact_validation_v1 import validate_native

BASE=Path('E:/CodexResearch/k_stock_trader/kalcb_selective_recovery_20261010')
PROGRAMS=('scripts/kalcb_late_entry_study_v5.py','backtests/strategies/kalcb/late_entry_v1.py',
 'backtests/strategies/kalcb/late_entry_io_v4.py','backtests/strategies/kalcb/late_entry_plugin_v1.py',
 'backtests/strategies/kalcb/late_entry_plugin_v2.py','backtests/strategies/kalcb/late_policy_v2.py',
 'scripts/kalcb_late_policy_builder_v5.py','backtests/strategies/kalcb/retention_research_io_v1.py',
 'backtests/strategies/kalcb/selective_recovery_plugin_v1.py','backtests/strategies/kalcb/oos_ablation.py',
 'backtests/strategies/kalcb/population_research_io_v1.py','backtests/strategies/kalcb/qualified_artifacts.py','backtests/strategies/kalcb/late_source_audit_v2.py')


def recipes():
    result={}
    for prior in (100,125):
        result['control' if prior==100 else 'control_prior125']={'prior':prior,'quantity':0,'flat_only':False,'parent':prior,'config':{}}
        for suffix,q,flat in [('quarter',.25,False),('quarter_flat',.25,True),('half_flat',.5,True)]:
            result[f'prior{prior}_{suffix}']={'prior':prior,'quantity':q,'flat_only':flat,'parent':prior,'config':{}}
    return result


def register(root,tests):
    from backtests.strategies.kalcb.late_completed_inputs_v1 import authenticate,publish
    test=read(tests)
    if test['exit_code']!=0:raise ValueError('Tests did not pass')
    verify(test['programs']+[test['log']])
    predecessor=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011/v6')
    proof=authenticate(predecessor);old=read(predecessor/'protocol.json')
    root.mkdir(parents=True,exist_ok=False)
    additions=['scripts/kalcb_late_entry_study_v7.py',
      'backtests/strategies/kalcb/late_artifact_validation_v1.py',
      'backtests/strategies/kalcb/late_completed_inputs_v1.py']
    support={'research/first_hour_v7/'+Path(p).name:bound(REPO/p) for p in additions}
    protocol={**old,'version':'kalcb-first-hour-v7-policy-bound-account-validator','registered_at_utc':datetime.now(timezone.utc).isoformat(),
      'runtime_root':'C:/Users/sehyu/.codex/research/kalcb_late_entry_20261011/v7',
      'programs':old['programs']+[bound(REPO/p) for p in additions],
      'bindings':old['bindings']+proof['bindings'],'completed_input_reuse':proof,'tests':test,
      'overrides':{period:{**old['overrides'][period],**support} for period in PERIODS},
      'repair_v7':'Post-replay independent validator used a fixed45-minute schema despite a passed source audit under the stronger first-hour-v2 schema. Preserve accounting body exactly, bind exact scope to policy, require first-hour population proof and immutable feature evidence. Authenticated reuse of completed data producers and revalidated unchanged control; no economic change or audit bypass.',
      'native_cases':36,'fresh_native_cases':31,'authenticated_preflight_cases':4,'authenticated_screening_cases':1}
    write(root/'protocol.json',protocol)
    write(root/'study.json',{'recipes':protocol['recipes'],'code':str(REPO),'source_generation_hash':digest(root/'protocol.json'),
      'periods':{p:{'sessions':v[2]} for p,v in PERIODS.items()},'workers':1,'policy_name':'interaction_cost_margin',
      'bound_inputs':[bound(root/'protocol.json')],'source_programs':protocol['programs']})
    write(root/'preflight/completion.json',read(predecessor/'preflight/completion.json'))
    write(root/'progress.json',read(predecessor/'progress.json'))
    publish(root,proof)
    write(root/'assessment_inheritance.json',{'contract':bound(predecessor.parent/'v5/assessment_protocol_v1.json'),
      'unchanged':True,'scope':'Pre-result assessment requirements unchanged across validator-only repair'})
    print(json.dumps({'registered':str(root),'fresh_native_cases':31,'reused_parent_identities':4,'revalidated_screening_control':proof['control']['bridge']}),flush=True)


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
    log=root/(label.replace('/','_')+'.log')
    with log.open('x',encoding='utf-8') as stream:
        child=subprocess.Popen(command,cwd=cwd,env=env(cwd),stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        identity={'pid':child.pid,'process_created':psutil.Process(child.pid).create_time(),'command':command}
        write(root/'status.json',{**state,'stage':label,'child':identity,'log':str(log),'updated_at_utc':datetime.now(timezone.utc).isoformat()})
        while child.poll() is None:
            try:
                p=psutil.Process(child.pid);sample={'at':datetime.now(timezone.utc).isoformat(),'stage':label,'identity':identity,'memory':p.memory_info()._asdict(),'io':p.io_counters()._asdict(),'cpu':p.cpu_times()._asdict()}
                with (root/'resource_observations.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(sample)+'\n')
            except psutil.NoSuchProcess:pass
            time.sleep(5)
    if child.returncode:
        write(root/'failed_stage.json',{'stage':label,'returncode':child.returncode,'log':bound(log),'preserve_evidence':True})
        raise RuntimeError('Registered stage failed: '+label)
    return bound(log)


def native(root,name,period,scenario,policy,authenticated,state):
    protocol=read(root/'protocol.json');prior=protocol['recipes'].get(name,{}).get('prior',int(name.split('_')[1])) if name.startswith('identity_') else protocol['recipes'][name]['prior']
    old=protocol['profiles'][str(prior)][period]
    item=freeze(root,name,period,old,policy,protocol['overrides'][period],authenticated);code=Path(item['folder'])/'inputs/files';out=root/'cases'/name/period/scenario
    reused=(name,period,scenario)==('control','is','baseline')
    if reused:
        verify(protocol['completed_input_reuse']['bindings'])
        log=bound(Path(protocol['completed_input_reuse']['root'])/'native_control_is_baseline.log')
    else:
        import_command=[sys.executable,'-B','-c',
            'import backtests.strategies.kalcb.late_entry_v1; import backtests.strategies.kalcb.late_entry_io_v4; import backtests.strategies.kalcb.late_entry_plugin_v2; import backtests.strategies.kalcb.late_policy_v2; import scripts.kalcb_checkpoint_prepare; import scripts.kalcb_action_quantity_labels; import scripts.kalcb_management_refit; print("frozen import preflight passed")']
        execute(root,import_command,'imports_'+name+'_'+period+'_'+scenario,code,state)
        command=[sys.executable,'-B','-u','-m','scripts.kalcb_executed_value_replay','--plan',item['plan']['path'],
            '--policies',str(code/'research/kalcb_executed_value/policies'),'--policy','interaction_cost_margin','--audit-manifest',item['manifest']['path'],
            '--scenario',scenario,'--output',str(out)]
        log=execute(root,command,'native_'+name+'_'+period+'_'+scenario,code,state)
    metrics=validate_native(out,item,scenario)
    if name.startswith('identity_') or protocol['recipes'].get(name,{}).get('quantity')==0:
        from backtests.strategies.kalcb.population_research_io_v1 import bridge_accounts
        write(out/'identity_bridge.json',bridge_accounts(out,Path(old['references'][scenario]).parent))
    row={'name':name,'period':period,'scenario':scenario,'metrics':metrics,'result':bound(out/'result.json'),'log':log,'authenticated_reuse':reused}
    progress=read(root/'progress.json') if (root/'progress.json').exists() else []
    progress.append(row);write(root/'progress.json',progress);print(json.dumps(row),flush=True)
    return read(out/'result.json')



def run(root):
    import psutil,msvcrt
    from backtests.strategies.kalcb.late_entry_plugin_v2 import LateEntryPluginV2
    from backtests.auto.shared.phase_runner import PhaseRunner
    from backtests.auto.shared.round_manager import RoundManager
    protocol=read(root/'protocol.json');verify(protocol['programs']+protocol['bindings']+protocol['data_bindings'])
    if not read(root/'preflight/completion.json')['passed']:raise ValueError('Native preflight incomplete')
    verify(protocol['completed_input_reuse']['bindings'])
    for p in psutil.process_iter(['pid','cmdline']):
        args=p.info['cmdline'] or []
        if p.pid!=os.getpid() and any('kalcb_late_entry_study_v' in str(a) for a in args) and ('run' in args or 'preflight' in args):raise ValueError('Existing supervisor')
        if 'scripts.kalcb_executed_value_replay' in args:raise ValueError('Existing native worker')
    lock=(root/'run.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if (root/'supervisor_start.json').exists():raise ValueError('Preserve prior run in versioned successor')
    state={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'command':sys.argv,'mode':'run'};write(root/'supervisor_start.json',state)
    code=Path(protocol['producer_runtime']['folder'])/'inputs/files';authenticated=set()
    try:
        verify(protocol['programs']+list(protocol['overrides']['is'].values())+list(protocol['overrides']['oos'].values()))
        def case(name,period,scenario):
            return native(root,name,period,scenario,root/'policies'/name/period,authenticated,state)
        plugin=LateEntryPluginV2(root/'study.json',case);manager=RoundManager('stock',plugin.name,base_dir=root/'framework')
        PhaseRunner(plugin,manager.get_round_dir(1),round_name='Causal first-hour additions on expanding100 and125',round_manager=manager,
            round_num=1,max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()
        if len(read(root/'progress.json'))!=36:raise ValueError('Missing registered native cases')
        write(root/'stage_completion.json',{'complete':True,'native_cases':36,'study_complete':False,'next':protocol['remaining']})
        write(root/'status.json',{**state,'stage':'awaiting_economic_assessment_and_finalist_qualification','child':None})
    except Exception as e:
        write(root/'status.json',{**state,'stage':'attention_required','error':repr(e)});raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage');p.add_argument('--root',type=Path,required=True);p.add_argument('--tests',type=Path)
    a=p.parse_args();root=a.root.resolve()
    if a.stage=='register':register(root,a.tests)
    elif a.stage=='run':run(root)
    else:raise ValueError('V7 only registers and runs the native consumer; never restart completed producers')
