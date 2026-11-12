"""Storage-only recovery; reuse completed production and exactly two audited controls."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import os,sys,shutil
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,digest,verify,PERIODS
from backtests.strategies.kalcb.research_process_v1 import execute
from backtests.strategies.kalcb.incremental_storage_reuse_v1 import authenticate,publish,seed_cache

BASE=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011')
OLD=BASE/'v13_incremental'
PROGRAMS=('scripts/kalcb_late_incremental_study_v2.py',
 'backtests/strategies/kalcb/incremental_storage_reuse_v1.py',
 'backtests/strategies/kalcb/late_entry_io_v5.py',
 'backtests/strategies/kalcb/research_process_v1.py',
 'tests/backtests/strategies/test_kalcb_storage_successor_v1.py')


def native_processes():
    import psutil
    result=[]
    for p in psutil.process_iter(['pid','cmdline','create_time']):
        if p.pid==os.getpid():continue
        args=p.info['cmdline'] or []
        if any(a=='scripts.kalcb_executed_value_replay' or
            (a.startswith('scripts.kalcb_') and '_worker_v' in a and ' ' not in a) or
            (Path(a).name.startswith('kalcb_') and '_worker_v' in Path(a).name and Path(a).suffix=='.py') for a in args):
            result.append(dict(pid=p.pid,command=args,created=p.info['create_time']))
    return result


def register(root,tests):
    test=read(tests)
    if test['exit_code']!=0:raise ValueError('Tests must pass')
    verify(test['programs']+[test['log']])
    proof=authenticate(OLD);p=read(OLD/'protocol.json')
    if root.exists():raise ValueError('Preserve earlier successor registration')
    if shutil.disk_usage(root.parent).free<10*1024**3:raise ValueError('Insufficient external storage')
    root.mkdir();write(root/'completed_input_reuse.json',proof)
    transport=dict(old_root=str(OLD),runtime_root=str(root/'runtime'),producer_restarted=False,
        reason='Closed V13 exhausted C volume during OOS publication. Preserve its partial artifacts and move new publications/cache/temp to E. Same economic design, formulas, policy/model bytes and frozen runtime sources.',
        reused_native_cases=2,fresh_native_cases=30)
    programs=p['programs']+[bound(REPO/s) for s in PROGRAMS]
    p.update(version='kalcb-incremental-selection-storage-v2',registered_at_utc=datetime.now(timezone.utc).isoformat(),
        programs=programs,input_bindings=p['input_bindings']+proof['bindings']+[bound(root/'completed_input_reuse.json'),bound(tests)],
        runtime_root=transport['runtime_root'],tests=test,storage_repair=transport,
        native_cases=32,fresh_native_cases=30,authenticated_screening_cases=2,
        migration=bound(BASE/'storage_repair_20261012/runtime_migration_completion.json'))
    write(root/'protocol.json',p)
    write(root/'study.json',dict(recipes=p['recipes'],code=str(REPO),source_generation_hash=digest(root/'protocol.json'),
        periods={k:dict(sessions=v[2]) for k,v in PERIODS.items()},workers=1,policy_name='interaction_cost_margin',
        bound_inputs=[bound(root/'protocol.json')],source_programs=programs))
    publish(root,proof)
    cache=Path(read(OLD/'protocol.json')['runtime_root'])/'content_cache'
    write(root/'cache_reuse.json',seed_cache(cache,root/'runtime/content_cache'))
    print(dict(registered=str(root),fresh=30,reused=2),flush=True)


def run(root):
    import psutil,msvcrt
    from backtests.strategies.kalcb.late_entry_io_v5 import freeze
    from backtests.strategies.kalcb.late_incremental_validation_v1 import validate_native
    from backtests.strategies.kalcb.population_research_io_v1 import bridge_accounts
    from backtests.strategies.kalcb.late_incremental_plugin_v1 import IncrementalPlugin
    from backtests.auto.shared.phase_runner import PhaseRunner
    from backtests.auto.shared.round_manager import RoundManager
    p=read(root/'protocol.json');verify(p['programs']+p['input_bindings']+p['data_bindings']+[p['migration']])
    if native_processes():raise ValueError('Another KALCB native worker exists')
    temp=root/'temporary';temp.mkdir(exist_ok=True);os.environ.update(TEMP=str(temp),TMP=str(temp))
    lock=(root/'run.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if (root/'supervisor_start.json').exists():raise ValueError('Preserve previous attempt')
    state=dict(pid=os.getpid(),process_created=psutil.Process().create_time(),command=psutil.Process().cmdline())
    write(root/'supervisor_start.json',state);progress=[];authenticated=set()
    proof=read(root/'completed_input_reuse.json');reused={(x['row']['name'],x['row']['period'],x['row']['scenario']):x for x in proof['controls']}
    def status(stage,**kw):write(root/'status.json',{**state,'stage':stage,'completed':len(progress),'child':None,**kw})
    def native(name,period,scenario):
        if native_processes():raise ValueError('Another KALCB native worker exists')
        recipe=p['recipes'][name];old=p['profiles'][str(recipe['prior'])][period]
        if recipe['quantity'] and sum(r['is_identity'] for r in progress)!=8:raise ValueError('All eight exact controls required')
        key=(name,period,scenario);out=root/'cases'/name/period/scenario
        if key in reused:
            item=read(root/'cases'/name/period/'input_profile.json');log=reused[key]['row']['log'];status('revalidating_completed_control',case=key)
        else:
            if out.exists():raise ValueError('Preserve partial native result')
            if shutil.disk_usage(root).free<2*1024**3:raise ValueError('External storage reserve exhausted')
            status('publishing_external_runtime',case=key)
            item=freeze(root,name,period,old,root/'policies'/name/period,p['overrides'][period],authenticated)
            code=Path(item['manifest']['path']).parent/'files'
            command=[sys.executable,'-B','-u','-m','scripts.kalcb_executed_value_replay','--plan',item['plan']['path'],
                '--policies',str(Path(item['policy']['path']).parent),'--policy','interaction_cost_margin',
                '--audit-manifest',item['manifest']['path'],'--scenario',scenario,'--output',str(out)]
            log=execute(root,command,'native_'+name+'_'+period+'_'+scenario,code,state)
        metrics=validate_native(out,item,scenario)
        if not recipe['quantity']:
            bridge=bridge_accounts(out,Path(old['references'][scenario]).parent)
            if not bridge['passed']:raise ValueError('Parent path mismatch')
            # Original bridge is an immutable hardlink on reused cases.
            if key in reused:
                if not read(out/'identity_bridge.json')['passed']:raise ValueError('Changed inherited bridge')
            else:write(out/'identity_bridge.json',bridge)
        row=dict(name=name,period=period,scenario=scenario,metrics=metrics,result=bound(out/'result.json'),
            log=log,authenticated_reuse=key in reused,is_identity=recipe['quantity']==0)
        progress.append(row);write(root/'progress.json',progress);status('case_audited',case=key);print(row,flush=True)
        return read(out/'result.json')
    try:
        plugin=IncrementalPlugin(root/'study.json',native);manager=RoundManager('stock',plugin.name,base_dir=root/'framework')
        PhaseRunner(plugin,manager.get_round_dir(1),round_name='Unchanged incremental design with external storage',
            round_manager=manager,round_num=1,max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()
        if len(progress)!=32 or sum(r['authenticated_reuse'] for r in progress)!=2:raise ValueError('Incomplete registered matrix')
        verify(p['programs']+p['input_bindings']+p['data_bindings'])
        write(root/'stage_completion.json',dict(complete=True,native_cases=32,fresh_native_cases=30,authenticated_screening_cases=2,
            study_complete=False,next=p['remaining']))
        status('awaiting_economic_assessment_and_full_reports')
    except Exception as e:status('attention_required',error=repr(e));raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=('register','run'));ap.add_argument('--root',type=Path,required=True);ap.add_argument('--tests',type=Path)
    args=ap.parse_args()
    if args.stage=='register':register(args.root.resolve(),args.tests)
    else:run(args.root.resolve())
