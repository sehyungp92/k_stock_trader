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
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify,env,validate_native,PERIODS,digest
from backtests.strategies.kalcb.late_entry_io_v2 import make_overrides,freeze

BASE=Path('E:/CodexResearch/k_stock_trader/kalcb_selective_recovery_20261010')
PROGRAMS=('scripts/kalcb_late_entry_study_v3.py','backtests/strategies/kalcb/late_entry_v1.py',
 'backtests/strategies/kalcb/late_entry_io_v2.py','backtests/strategies/kalcb/late_entry_plugin_v1.py',
 'backtests/strategies/kalcb/late_entry_plugin_v2.py','backtests/strategies/kalcb/late_policy_v2.py',
 'scripts/kalcb_late_policy_builder_v3.py','backtests/strategies/kalcb/retention_research_io_v1.py',
 'backtests/strategies/kalcb/selective_recovery_plugin_v1.py','backtests/strategies/kalcb/oos_ablation.py',
 'backtests/strategies/kalcb/population_research_io_v1.py','backtests/strategies/kalcb/qualified_artifacts.py')


def recipes():
    result={}
    for prior in (100,125):
        result['control' if prior==100 else 'control_prior125']={'prior':prior,'quantity':0,'flat_only':False,'parent':prior,'config':{}}
        for suffix,q,flat in [('quarter',.25,False),('quarter_flat',.25,True),('half_flat',.5,True)]:
            result[f'prior{prior}_{suffix}']={'prior':prior,'quantity':q,'flat_only':flat,'parent':prior,'config':{}}
    return result


def register(root,tests):
    import yaml
    test=read(tests)
    if test['exit_code']!=0:raise ValueError('Tests did not pass')
    verify(test['programs']+[test['log']])
    if not read(BASE/'v8_population_reports/final_review_v1/completion.json')['study_complete']:raise ValueError('Previous study still active')
    root.mkdir(parents=True,exist_ok=False);profiles={};bindings=[]
    for q in (100,125):
        profiles[str(q)]={}
        for period in PERIODS:
            profile=BASE/f'v8_population_reports/artifacts/profiles/expanding_removal_prior{q}/{period}/research_profile.json'
            p=read(profile)
            item={'name':p['primary_profile'],'period':period,'folder':str(Path(p['plan']).parent),'plan':bound(p['plan']),
                'manifest':bound(p['audit_manifest']),'policy':bound(Path(p['policy']['directory'])/'policies_manifest.json'),
                'source_generation_hash':p['execution_source']['source_generation_hash'],'references':p['result_paths']}
            profiles[str(q)][period]=item;bindings+=[bound(profile),bound(p['configuration'])]+[item[k] for k in ('plan','manifest','policy')]
            for scenario,path in p['result_paths'].items():
                from backtests.strategies.kalcb.qualified_artifacts import REPLAY_FILES
                folder=Path(path).parent;validate_native(folder,item,scenario,save=False)
                bindings += [bound(folder/f) for f in (*REPLAY_FILES,'independent_account_verification.json')]
    opening=['E:/CodexResearch/k_stock_trader/kalcb_continuation_20261006/prepared_v1',
        'C:/Users/sehyu/.codex/research/olr_round1_oos_20261009/forward_features/continuation']
    plans=[read(profiles['100'][p]['plan']['path']) for p in ('is','oos')]
    databases=[p['database'] for p in plans]
    data=[bound(p) for p in databases]+[bound(Path(p)/f) for p in opening for f in ('prepared.json','opportunities.jsonl')]
    for p in opening:
        if read(Path(p)/'prepared.json')['opportunities_sha256']!=digest(Path(p)/'opportunities.jsonl'):raise ValueError('Changed opening context')
    programs={p:bound(REPO/p) for p in PROGRAMS}
    overrides={}
    for period in PERIODS:
        old=profiles['100'][period];overrides[period]=make_overrides(Path(old['folder'])/'inputs/files',root/'overrides'/period,programs)
    universe=tuple(str(s).zfill(6) for s in yaml.safe_load(Path(plans[0]['universe']).read_text())['symbols'])
    protocol={'version':'kalcb-first-hour-v3-complete-dependencies-and-local-runtime','registered_at_utc':datetime.now(timezone.utc).isoformat(),'recipes':recipes(),
        'runtime_root':'C:/Users/sehyu/.codex/research/kalcb_late_entry_20261011/v3','profiles':profiles,'programs':list(programs.values()),'overrides':overrides,'bindings':bindings,'data_bindings':data,
        'opening_prepared':opening,'databases':databases,'watch_symbols':universe,'tests':test,'periods':PERIODS,
        'repair':'V1 native replay failed before scoring because its inherited source omitted retention_research_io_v1. V3 binds the complete helper dependency set and checks frozen imports before replay; v1 and launch_v1 are preserved. Native inputs use an authenticated content-addressed cache on C for efficient repeated publication; all economic outputs remain on E. Final rank/context preview and reporting-contract repairs from v2 are included.',
        'checkpoint_route_budget':'Preserve the original shared checkpoint_continuation session limit: minute60 is another opportunity to use unconsumed checkpoint capacity, not a second successful checkpoint trade after a filled45-minute entry. Flat-only additionally rejects exposure from opening entries or pending orders. Count actual support before economic selection.',
        'design':'Two locked expanding priors; one new issuer at minute60; quarter quantity unrestricted versus flat-only, with half-flat nearby size. Opening and minute45 incumbents unchanged. No exit changes.',
        'hypothesis':'A later completed confirmation can admit newly executable net-positive continuation while flat-only allocation limits interference with the valuable incumbent right tail. Quarter versus half measures quantity/cost sensitivity. Lower DD is an objective to test, never an asserted result.',
        'confirmation':'Complete first twelve5m bars; final two closes at/above their own contemporaneous session VWAP and their own opens. Existing shared-core checkpoint guard still required. Earlier VWAP pullback recorded for attribution, not outcome-selected membership.',
        'training':'Fresh isolated full-cost baseline/combined labels at each quantity under unchanged management; keep failures and no-fills. One expanding interaction learner per scenario; original global20-session fold clock, minimum60 mature sessions, one full-session embargo. A fold lacking mature data admits no late actions; never shift the clock.',
        'selection':'Physical failure probability <0.001, completed-volume capacity gate, minimum of baseline/combined net forecasts exceeds48bps*planned notional/reference equity. One issuer per day, absent from all incumbent snapshots. No date/symbol blacklist.',
        'native_cases':36,'preflight_cases':4,'screening_cases':32,
        'scope':'IS2025-08-07 through2026-03-31; OOS2026-04-01 through2026-07-10 reused development. Continuous217-session prior intraday history on April1. No independent-validation claim or automatic promotion.',
        'remaining':'After32 screening cells inspect actual minute60 submits/fills, parent funding descendants, right tails and costs, IS sacrifice bands, monthly/issuer/MTM attribution. Qualify credible candidates with all four stresses, finite nearby/backout support, independent accounts and existing-engine full reports. No forced success.'}
    write(root/'protocol.json',protocol)
    write(root/'study.json',{'recipes':recipes(),'code':str(REPO),'source_generation_hash':digest(root/'protocol.json'),
        'periods':{p:{'sessions':v[2]} for p,v in PERIODS.items()},'workers':1,'policy_name':'interaction_cost_margin',
        'bound_inputs':[bound(root/'protocol.json')],'source_programs':list(programs.values())})
    print(json.dumps({'registered':str(root),'new_variants':6,'native_cases':36}),flush=True)


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
    import_command=[sys.executable,'-B','-c',
        'import backtests.strategies.kalcb.late_entry_v1; import backtests.strategies.kalcb.late_entry_io_v2; import backtests.strategies.kalcb.late_entry_plugin_v2; import backtests.strategies.kalcb.late_policy_v2; import scripts.kalcb_checkpoint_prepare; import scripts.kalcb_action_quantity_labels; import scripts.kalcb_management_refit; print("frozen import preflight passed")']
    execute(root,import_command,'imports_'+name+'_'+period+'_'+scenario,code,state)
    command=[sys.executable,'-B','-u','-m','scripts.kalcb_executed_value_replay','--plan',item['plan']['path'],
        '--policies',str(code/'research/kalcb_executed_value/policies'),'--policy','interaction_cost_margin','--audit-manifest',item['manifest']['path'],
        '--scenario',scenario,'--output',str(out)]
    log=execute(root,command,'native_'+name+'_'+period+'_'+scenario,code,state)
    metrics=validate_native(out,item,scenario)
    if name.startswith('identity_') or protocol['recipes'].get(name,{}).get('quantity')==0:
        from backtests.strategies.kalcb.population_research_io_v1 import bridge_accounts
        write(out/'identity_bridge.json',bridge_accounts(out,Path(old['references'][scenario]).parent))
    row={'name':name,'period':period,'scenario':scenario,'metrics':metrics,'result':bound(out/'result.json'),'log':log}
    progress=read(root/'progress.json') if (root/'progress.json').exists() else []
    progress.append(row);write(root/'progress.json',progress);print(json.dumps(row),flush=True)
    return read(out/'result.json')


def preflight(root):
    import psutil
    protocol=read(root/'protocol.json');verify(protocol['programs']+protocol['bindings']+protocol['data_bindings'])
    state={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'command':sys.argv,'mode':'preflight'};authenticated=set()
    for prior in (100,125):
        for period in PERIODS:
            old=protocol['profiles'][str(prior)][period]
            from backtests.strategies.kalcb.retention_research_io_v1 import link
            source=Path(old['policy']['path']).parent;policy=root/'preflight/policies'/str(prior)/period
            policy.mkdir(parents=True,exist_ok=False);m=read(source/'policies_manifest.json')
            for f in m['files']:link(source/f,policy/f)
            m['observation_minutes']=60
            write(policy/'policies_manifest.json',m);write(policy/'completion.json',{'complete':True})
            native(root,f'identity_{prior}',period,'baseline',policy,authenticated,state)
    code=Path(read(root/'cases/identity_100/is/input_profile.json')['folder'])/'inputs/files'
    for stage in ('prepare_smoke','labels_smoke_baseline','labels_smoke_combined'):
        execute(root,[sys.executable,'-B','-u',str(code/PROGRAMS[0]),stage,'--root',str(root)],stage,code,state)
    counts={s:read(root/'preflight/labels'/s/'completion.json')['counts'] for s in ('baseline','combined')}
    if any(c.get('filled',0)<1 for c in counts.values()):raise ValueError('No actual filled smoke action')
    write(root/'preflight/completion.json',{'passed':True,'native_identity_bridges':4,'fresh_filled_labels':counts,'source_audits_and_independent_accounts':True})


def run(root):
    import psutil,msvcrt
    from backtests.strategies.kalcb.late_entry_plugin_v2 import LateEntryPluginV2
    from backtests.auto.shared.phase_runner import PhaseRunner
    from backtests.auto.shared.round_manager import RoundManager
    protocol=read(root/'protocol.json');verify(protocol['programs']+protocol['bindings']+protocol['data_bindings'])
    if not read(root/'preflight/completion.json')['passed']:raise ValueError('Native preflight incomplete')
    for p in psutil.process_iter(['pid','cmdline']):
        args=p.info['cmdline'] or []
        if p.pid!=os.getpid() and any('kalcb_late_entry_study_v' in str(a) for a in args) and ('run' in args or 'preflight' in args):raise ValueError('Existing supervisor')
        if 'scripts.kalcb_executed_value_replay' in args:raise ValueError('Existing native worker')
    lock=(root/'run.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if (root/'prepared').exists():raise ValueError('Preserve prior run in versioned successor')
    state={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'command':sys.argv,'mode':'run'};write(root/'supervisor_start.json',state)
    code=Path(read(root/'cases/identity_100/is/input_profile.json')['folder'])/'inputs/files';authenticated=set()
    try:
        for stage in ('prepare','labels_baseline','labels_combined','fit','policies'):
            verify(protocol['programs']+list(protocol['overrides']['is'].values())+list(protocol['overrides']['oos'].values()))
            command=[sys.executable,'-B','-u',str(code/PROGRAMS[0]),stage,'--root',str(root)]
            if stage=='policies':command=[sys.executable,'-B','-u',str(code/'scripts/kalcb_late_policy_builder_v3.py'),'--root',str(root),'--code',str(code),'--module',str(code/'backtests/strategies/kalcb/late_policy_v2.py')]
            execute(root,command,stage,code,state)
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
    elif a.stage=='preflight':preflight(root)
    elif a.stage=='run':run(root)
    else:
        from backtests.strategies.kalcb.late_entry_v1 import prepare,labels,fit,build_policies
        if a.stage=='prepare':prepare(root)
        elif a.stage=='prepare_smoke':prepare(root,limit=3)
        elif a.stage.startswith('labels_'):labels(root,a.stage.split('_')[-1],smoke='smoke' in a.stage)
        elif a.stage=='fit':fit(root)
        elif a.stage=='policies':build_policies(root)
        else:raise ValueError('Unknown stage')
