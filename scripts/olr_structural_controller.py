"""Bounded mechanism study; authenticated completion precedes advancement.

Run in a hidden process only after source/tests/protocol are frozen. This is a
research controller, not live trading. Unqualified results never activate OOS.
"""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import argparse,json,os,sys,time,traceback
import psutil

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts.olr_structural_study import launch,prepare
from scripts.olr_economic_study import read,write,sha,source_hash
from backtests.strategies.olr.economic_evidence import completed_case,compare
from scripts.olr_economic_assessment import contrast


def now():return datetime.now(timezone.utc).isoformat()


def status(root,phase,**values):
    value=dict(at=now(),phase=phase,**values);p=root/'controller_status.json';tmp=root/'controller_status.tmp'
    tmp.write_text(json.dumps(value,indent=2,default=str),encoding='utf-8');os.replace(tmp,p)
    print(json.dumps(value,default=str),flush=True)


def verify_identity(identity):
    try:
        p=psutil.Process(identity['pid'])
        if abs(p.create_time()-identity['created_at'])>.01:return False
        actual=p.cmdline();expected=identity['command']
        if len(actual)!=len(expected) or any(os.path.normcase(a)!=os.path.normcase(b) for a,b in zip(actual,expected)):return False
        return p.is_running()
    except (psutil.NoSuchProcess,psutil.AccessDenied):return False


def case(root,window,mode,scenario='baseline',capped=False,legacy=False):
    tag=('legacy_' if legacy else '')+mode+'_'+scenario+('_capped' if capped else '')
    output=root/'cases'/window/tag;identity_path=root/'runtime'/window/tag/'process.json'
    if (output/'completion.json').exists():
        pass  # Completed cases must still pass qualification and checkpoint checks below.
    elif identity_path.exists():
        identity=read(identity_path)
        if not verify_identity(identity):raise RuntimeError('Incomplete dead case; preserve it and repair in a new generation: '+tag)
        while verify_identity(identity):time.sleep(10)
    else:
        if output.exists():raise RuntimeError('Orphan case must not be overwritten: '+tag)
        process,identity=launch(root,window,mode,scenario,capped,legacy)
        code=process.wait()
        if code:raise RuntimeError('Native process failed: '+str(identity['log']))
    result=completed_case(output)
    if not result['valid']:raise RuntimeError('Execution/source qualification failed: '+str(output))
    checkpoint=read(output/'funded_checkpoint_receipt.json')
    if not checkpoint['no_action_continuation_equal'] or sha(output/'native_account_checkpoint.pkl')!=checkpoint['sha256']:
        raise RuntimeError('Funded continuation evidence failed')
    return result


def group(root,tasks):
    # Bounded concurrency also keeps source audits from exhausting host memory.
    # Keep only running work submitted. A failed shared engineering contract
    # must not drain an already queued list of further doomed native replays.
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending=iter(tasks.items());results={}
        while True:
            batch=[row for row in (next(pending,None),next(pending,None)) if row is not None]
            if not batch:break
            active={pool.submit(case,root,**arguments):name for name,arguments in batch}
            # Qualify the whole running batch before submitting another pair.
            while active:
                done,_=wait(active,return_when=FIRST_COMPLETED)
                try:
                    for future in done:
                        name=active.pop(future);results[name]=future.result()
                except BaseException as exc:
                    status(root,'engineering_failure_draining_active_cases',error=repr(exc),
                        remaining_cases=list(active.values()),economic_objective_achieved=False)
                    raise
        return results


def comparison(root,window,mode,scenario='baseline',capped=False):
    suffix=scenario+('_capped' if capped else '')
    a=root/'cases'/window/('control_'+suffix);b=root/'cases'/window/(mode+'_'+suffix)
    av,bv=completed_case(a),completed_case(b)
    c=compare(av,bv,window=window,scenario=suffix)
    c['funded_accounting']=contrast(read(a/'trades.json'),read(b/'trades.json'))
    # Risk-component status has no artificial frequency reward or quota. It is
    # separately labelled and never promoted as the user's complete objective.
    c['risk_component_pass']=bool(av['valid'] and bv['valid'] and
        bv['return_pct']>=av['return_pct']-(1.5 if window=='train' else 0) and
        bv['full_dd']<av['full_dd'] and bv['common_dd']<av['common_dd'] and
        bv['trades']>=av['trades'] and bv['days']>=av['days'])
    return c


def run(root):
    protocol=read(root/'protocol.json');ready=read(root/'native_engines/train/ready.json')
    if sha(root/'protocol.json')!=read(root/'preflight.json')['protocol_sha256']:
        raise ValueError('Protocol changed after preflight')
    if source_hash(Path(ready['files']))!=ready['study']['source_generation_hash']:raise ValueError('Frozen source changed')
    preflight=read(root/'preflight.json')
    if not preflight['tests_passed'] or sha(Path(preflight['tests']))!=preflight['tests_sha256']:
        raise ValueError('Test receipt changed')
    status(root,'legacy_parity_running',economic_objective_achieved=False)
    legacy=case(root,'train','control',legacy=True)
    old=Path(protocol['legacy_control']);control=completed_case(old)
    lp=Path(legacy['path'])
    parity=bool(legacy['valid'] and control['valid'] and read(lp/'equity.json')==read(old/'equity.json')
        and legacy['trades']==control['trades'] and legacy['days']==control['days']
        and abs(legacy['full_dd']-control['full_dd'])<1e-10)
    write(root/'legacy_parity.json',dict(pass_=parity,old=control['receipt_sha256'],new=legacy['receipt_sha256'],
        full_nav_path_equal=read(lp/'equity.json')==read(old/'equity.json')))
    if not parity:raise ValueError('Native legacy parity failed before policy study')
    status(root,'nominal_mechanisms_running',modes=protocol['modes'])
    group(root,{m:dict(window='train',mode=m) for m in protocol['modes']})
    nominal={m:comparison(root,'train',m) for m in protocol['modes'] if m!='control'}
    # Preserve the retained nominal floor as well as the newly paired interval
    # floor; a worse execution control cannot silently lower the objective.
    for mode,c in nominal.items():
        v=completed_case(root/'cases/train'/(mode+'_baseline'))
        c['retained_nominal_return_pass']=v['return_pct']>=control['return_pct']-1.5
    write(root/'nominal_assessment.json',nominal)
    eligible=[m for m,c in nominal.items() if c['risk_component_pass'] and c['retained_nominal_return_pass']]
    if eligible:
        status(root,'stress_capped_running',eligible=eligible)
        tasks={}
        for mode in ['control',*eligible]:
            for scenario,capped in [('combined',False),('baseline',True)]:
                tasks[mode+'_'+scenario+str(capped)]=dict(window='train',mode=mode,scenario=scenario,capped=capped)
        group(root,tasks)
        checks={m:[comparison(root,'train',m,'combined'),comparison(root,'train',m,'baseline',True)] for m in eligible}
        write(root/'stress_capped_assessment.json',checks)
        # All retained numerical checks are evaluated by the continuation agent
        # before any OOS source preparation. This controller cannot substitute a
        # partial new comparison for the older reference predicates.
        status(root,'awaiting_retained_gate_review',eligible=eligible,
            robust_component_candidates=[m for m,rows in checks.items() if all(x['risk_component_pass'] for x in rows)],
            economic_objective_achieved=False,automatic_oos_authority=False)
    else:
        status(root,'nominal_mechanisms_complete_no_qualified_component',economic_objective_achieved=False,
            next='Assess exact funded failure/giveback effects and the recorded entry episodes; continue the authorized structural programme. No rescue parameter sweep or unqualified OOS.')


def progress(root):
    observations=[]
    for path in sorted((root/'runtime').glob('*/*/process.json')):
        identity=read(path);alive=verify_identity(identity);item=dict(identity,alive=alive)
        if alive:
            p=psutil.Process(identity['pid']);item.update(cpu_seconds=sum(p.cpu_times()[:2]),rss=p.memory_info().rss,io=p.io_counters()._asdict())
        log=Path(identity['log'])
        if log.exists():
            item.update(log_bytes=log.stat().st_size,log_modified=log.stat().st_mtime)
            with log.open('rb') as f:
                f.seek(max(0,log.stat().st_size-2500));item['log_tail']=f.read().decode('utf-8',errors='replace')
        output=Path(identity['case'])
        if (output/'completion.json').exists():
            try:
                c=completed_case(output);c.pop('acquired');item['completion']=c
            except Exception as exc:item['completion_error']=repr(exc)
        observations.append(item)
    controller_process=None
    identity_path=root/'controller_process.json'
    if identity_path.exists():
        identity=read(identity_path);controller_process=dict(identity,alive=verify_identity(identity))
        if controller_process['alive']:
            p=psutil.Process(identity['pid'])
            controller_process.update(cpu_seconds=sum(p.cpu_times()[:2]),rss=p.memory_info().rss)
    value=dict(at=now(),controller=read(root/'controller_status.json') if (root/'controller_status.json').exists() else None,
        controller_process=controller_process,
        processes=observations,economic_objective_achieved=False)
    name='progress_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'
    write(root/'progress'/name,value);print(json.dumps(value,indent=2),flush=True)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('run','progress'));p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    if a.phase=='progress':progress(a.root)
    else:
        try:run(a.root)
        except BaseException as exc:
            status(a.root,'needs_engineering_review',error=repr(exc),economic_objective_achieved=False)
            traceback.print_exc();raise
