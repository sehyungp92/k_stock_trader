"""Registered four-parent population calibration with native shared-framework replay."""
import argparse
import ctypes
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.population_research_io_v1 import (
    read,write,bound,digest,verify,link,env,freeze_case,validate_native,bridge_accounts,native_support,PERIODS,POLICY)
from backtests.strategies.kalcb.population_policy_v1 import BASE,QUAL,ALIGNMENT,PARENTS,RECIPES

DEST=BASE/'v5_population'
CONTROLS={'control':'expanding',**{'control_'+p:p for p in PARENTS[1:]}}
WORKER=REPO/'scripts/kalcb_population_worker_v1.py'


def selection_dependencies():
    """Authenticate external configuration read while reconstructing policies."""
    result=[];registration=read(ALIGNMENT/'protocol.json')
    registered={str(Path(b['path']).resolve()):b['sha256'] for b in registration['bound_inputs']}
    is_plans={c['kind']:read(c['parent_plan']) for c in registration['cells'] if c['period']=='is'}
    for cell in registration['cells']:
        plan=read(cell['parent_plan'])
        for key in ('seed','q3','config'):
            path=plan[key];expected=plan['input_sha256'].get(path) or registered.get(str(Path(path).resolve()))
            if not expected and cell['period']=='oos' and key in ('seed','q3'):
                # OOS feature-contract copies must be byte-identical to the bound IS inputs.
                original=is_plans[cell['kind']]
                expected=original['input_sha256'].get(original[key])
            if not expected:raise ValueError('Unbound selection input '+key)
            item={'path':path,'sha256':expected};verify([item]);result.append(item)
    return result


def commit_available():
    class Info(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in
            ('CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable','SystemCache','KernelTotal','KernelPaged','KernelNonpaged','PageSize')]+[(k,ctypes.c_ulong) for k in ('HandleCount','ProcessCount','ThreadCount')]
    v=Info();v.cb=ctypes.sizeof(v)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(v),v.cb):raise OSError('Commit inventory unavailable')
    return (v.CommitLimit-v.CommitTotal)*v.PageSize


def register(out,build,tests):
    done=read(build/'completion.json');test=read(tests)
    if not done['complete'] or test['exit_code']!=0:raise ValueError('Prelaunch incomplete')
    verify(done['policies']+[done[k] for k in ('inputs','control_bridge','boundary','calibration','decisions')])
    verify(read(build/'input_bindings.json'));verify([test['log']])
    if not read(build/'control_bridges.json')['passed'] or not read(build/'boundary_continuity.json')['passed']:
        raise ValueError('Control/boundary gate failed')
    profiles=read(QUAL/'v3/preparation.json')['profiles']
    prior=read(BASE/'v4_retention/protocol.json');verify(prior['programs'])
    programs={b['path']:b for b in prior['programs']}
    for p in [Path(__file__),WORKER,*REPO.glob('backtests/strategies/kalcb/population*_v1.py'),
              *REPO.glob('tests/backtests/strategies/test_kalcb_population*_v1.py')]:
        programs[str(p.resolve())]=bound(p)
    inputs=[bound(build/'completion.json'),bound(build/'design.json'),bound(tests),test['log'],
        bound(BASE/'v4_retention/retention_screening_assessment.json'),bound(QUAL/'v3/preparation.json'),
        bound(BASE/'v4_retention/observational_review_20261010_1833/followup_design.json')]
    inputs+=done['policies']+[done[k] for k in ('inputs','control_bridge','boundary','calibration','decisions')]
    inputs+=read(build/'input_bindings.json')
    inputs+=selection_dependencies()
    for p in profiles:
        inputs += [p[k] for k in ('manifest','plan','policy')]
    verify(inputs);out.mkdir(parents=True,exist_ok=False)
    recipes={n:s for n,s in RECIPES.items() if n not in done['aliases']}
    write(out/'protocol.json',{'registered_at_utc':datetime.now(timezone.utc).isoformat(),'recipes':recipes,
        'all_declared_recipes':RECIPES,'aliases':done['aliases'],'build':str(build),'parent_profiles':profiles,
        'programs':list(programs.values()),'bindings':inputs,'tests':test,'periods':PERIODS,
        'objective':'Improve net IS/OOS returns and meaningful activity with lower MTM DD. Strong global IS73.1303145472/47.8121465627; report2/5/10pp sacrifice bands, not cutoffs. DD8/10 preferences.',
        'hypothesis':'Correct selection-conditioned population transfer using historical eligible exact-quantity matching-origin/sign/rank labels. Full/additions-only/removals-only backouts on four original parents. Original management unchanged.',
        'calibration':'Five-session continuous residual clock,60 prior completed sessions,one-session embargo,actual label maturity; prior100; pool20issuer-days/8sessions,local8/5,unsupported identity. No cross-origin/sign/quantity borrowing.',
        'boundary':'All217 prior intraday sessions retained at April1; March10 base and March31 residual fit inherited; next base/residual fit April7. No reporting-boundary reset. Pre-August outcomes inform base models.',
        'execution':'Original frozen economic runtime/config. Completed-prefix signals, actual funding/partial fills/next-bar/gaps/taxes and physical gate remain. Checkpoint can readmit an opening removal: no day-long veto.',
        'native_gate':'Two fresh unchanged baseline bridges (IS and full OOS) must reproduce every mark/balance/physical fill before variants; every scored case requires source and independent account audit.',
        'support':'Actual filled additions and paired parent/variant issuer-day changes reported. Removed parent fills are not same-state causal removal proof; direct broker backout evidence required before finalist qualification.',
        'resources':'One worker,384MiB physical headroom and1280MiB available Windows commit, measured peaks; no blanket2GiB free RAM guard.',
        'scope':'Reused April-July development, not independent validation. No canonical promotion or live changes. Screening not complete study; qualify credible finalists in four stresses, concentration, parameter stability, paired uncertainty and funded constant-capital tests.'})
    write(out/'study.json',{'recipes':{**{n:{'parent':p,'config':{}} for n,p in CONTROLS.items()},**recipes},
        'controls':CONTROLS,'periods':{p:{'sessions':v[2]} for p,v in PERIODS.items()},'workers':1,'code':str(REPO),
        'policy_name':POLICY,'source_generation_hash':digest(out/'protocol.json'),
        'bound_inputs':[bound(out/'protocol.json')],'source_programs':list(programs.values())})
    print(json.dumps({'registered':str(out),'variant_cases':4*len(recipes),'native_bridges':2,'aliases':done['aliases']}),flush=True)


def run(out):
    import msvcrt,psutil
    from backtests.strategies.kalcb.population_research_plugin_v1 import PopulationResearchPlugin
    from backtests.strategies.kalcb.qualified_artifacts import REPLAY_FILES
    from backtests.auto.shared.phase_runner import PhaseRunner
    from backtests.auto.shared.round_manager import RoundManager
    protocol=read(out/'protocol.json');verify(protocol['bindings']+protocol['programs'])
    for p in psutil.process_iter(['pid','cmdline','create_time']):
        args=p.info['cmdline'] or []
        if p.pid==os.getpid():continue
        if 'scripts.kalcb_executed_value_replay' in args or any(str(a).endswith(('kalcb_retention_worker_v1.py','kalcb_population_worker_v1.py')) for a in args):
            raise ValueError('Another native KALCB worker is active')
        if any(str(a).endswith(('kalcb_selective_recovery_v3.py','kalcb_retention_research_v1.py','kalcb_population_research_v1.py')) for a in args) and 'run' in args:
            raise ValueError('Another KALCB supervisor is active')
    lock=(out/'run.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if (out/'progress.json').exists():raise ValueError('Preserve existing run; versioned successor required')
    state={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'command':sys.argv,'child':None}
    def status(stage,**kwargs):
        state.update(stage=stage,updated_at_utc=datetime.now(timezone.utc).isoformat(),**kwargs)
        write(out/'status.tmp',state);os.replace(out/'status.tmp',out/'status.json')
    profiles={(x['name'],x['period']):x for x in protocol['parent_profiles']}
    progress=[];authenticated=set();bridges={}
    def native_case(name,period,scenario):
        spec=read(out/'study.json')['recipes'][name];old=profiles[(spec['parent'],period)]
        reference=QUAL/'v4/profiles'/spec['parent']/period/scenario;target=out/'cases'/name/period/scenario
        bridge=name=='control' and scenario=='baseline'
        if name in CONTROLS and not bridge:
            validate_native(reference,old,scenario,save=False)
            target.mkdir(parents=True,exist_ok=False)
            for f in (*REPLAY_FILES,'independent_account_verification.json'):link(reference/f,target/f)
            write(target/'reuse.json',{'reference':str(reference),'authenticated':True})
            return read(target/'result.json')
        if name not in CONTROLS and set(bridges)!=set(PERIODS):raise ValueError('Both continuous-history native bridges required')
        verify(protocol['bindings']+protocol['programs'])
        status('freezing_case',case=name+'/'+period+'/'+scenario,child=None,completed=len(progress))
        policy=Path(old['policy']['path']).parent if bridge else Path(protocol['build'])/'policies'/name/period
        item=freeze_case(out,name,period,spec,old,policy,authenticated)
        while psutil.virtual_memory().available<384*1024**2 or commit_available()<1280*1024**2:
            status('waiting_for_measured_headroom',physical_available=psutil.virtual_memory().available,commit_available=commit_available())
            time.sleep(15)
        if target.exists():raise ValueError('Preserve existing native output')
        code=Path(item['folder'])/'inputs/files'
        command=[sys.executable,'-B','-u',str(WORKER),'--code',str(code),'--plan',item['plan']['path'],
            '--policies',str(Path(item['policy']['path']).parent),'--audit-manifest',item['manifest']['path'],
            '--output',str(target),'--scenario',scenario]
        logfile=target.with_suffix('.log');began=time.monotonic()
        with logfile.open('x',encoding='utf-8') as stream:
            child=subprocess.Popen(command,cwd=code,env=env(code),stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            identity={'pid':child.pid,'process_created':psutil.Process(child.pid).create_time(),'command':command}
            status('native_replay',child=identity)
            if name not in CONTROLS and not (out/'launch_receipt.json').exists():
                write(out/'launch_receipt.json',{'verified':True,'at_utc':datetime.now(timezone.utc).isoformat(),
                    'supervisor':state,'bridges':bridges,'first_variant':identity,'tests':protocol['tests'],'study_complete':False})
            while child.poll() is None:
                try:
                    sample={'at_utc':datetime.now(timezone.utc).isoformat(),'identity':identity,
                        'memory':psutil.Process(child.pid).memory_info()._asdict(),'physical_available':psutil.virtual_memory().available,
                        'commit_available':commit_available()}
                    with (out/'resource_observations.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(sample)+'\n')
                except psutil.NoSuchProcess:pass
                time.sleep(10)
        row={'name':name,'period':period,'scenario':scenario,'returncode':child.returncode,'seconds':time.monotonic()-began,'log':bound(logfile)}
        if child.returncode:
            write(out/'failed_case.json',row);raise RuntimeError('Native worker failed; preserve evidence')
        row['metrics']=validate_native(target,item,scenario);row['result']=bound(target/'result.json')
        if bridge:
            result=bridge_accounts(target,reference);write(target/'native_bridge.json',result);bridges[period]=bound(target/'native_bridge.json')
        else:
            support=native_support(target,reference,Path(item['policy']['path']).parent,Path(old['policy']['path']).parent)
            write(target/'intervention_support.json',support);row['support']=bound(target/'intervention_support.json')
        progress.append(row);write(out/'progress.json',progress);status('case_completed',child=None,completed=len(progress))
        print(json.dumps(row),flush=True);return read(target/'result.json')
    try:
        plugin=PopulationResearchPlugin(out/'study.json',native_case)
        manager=RoundManager('stock',plugin.name,base_dir=out/'framework')
        PhaseRunner(plugin,manager.get_round_dir(1),round_name='Four-parent population calibration',round_manager=manager,
            round_num=1,max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()
        verify(protocol['bindings']+protocol['programs'])
        write(out/'screening_completion.json',{'complete':True,'native_cases':len(progress),'study_complete':False,
            'next':'Assess all full/additions/removals evidence and global IS preservation. Qualify credible finalists; no automatic promotion.'})
        status('awaiting_population_review',child=None)
    except Exception as error:
        status('attention_required',error=repr(error));raise
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('register','run'))
    parser.add_argument('--root',type=Path,default=DEST);parser.add_argument('--build',type=Path);parser.add_argument('--tests',type=Path)
    args=parser.parse_args()
    if args.stage=='register':register(args.root,args.build,args.tests)
    else:run(args.root)
