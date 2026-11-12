"""Queue read-only final assessments/reports after the registered native job exits."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import os
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify,env
from scripts.kalcb_population_final_artifacts_v1 import REPORT_PROGRAMS


def dependency_ready(root,identity,process_lookup):
    state=read(root/'status.json')
    if state['stage']=='attention_required':raise ValueError('Native prerequisite needs attention: '+str(state.get('error')))
    process=process_lookup(identity['pid'])
    if process is not None:
        if abs(process.create_time()-identity['process_created'])>.001 or process.cmdline()!=identity['command']:
            raise ValueError('Prerequisite process identity changed')
        return False
    if not (root/'stage_completion.json').exists():raise ValueError('Native prerequisite ended without complete evidence')
    done=read(root/'stage_completion.json')
    if not done['complete'] or done['native_cases']!=36:raise ValueError('Native prerequisite incomplete')
    return True


def register(root,out,tests):
    import psutil
    if out.exists():raise ValueError('Preserve existing report pipeline')
    p=read(root/'protocol.json');verify(p['programs']+p['bindings'])
    test=read(tests);verify(test['programs']+[test['log']])
    if test['exit_code']!=0:raise ValueError('Failed report tests')
    s=read(root/'status.json');process=psutil.Process(s['pid'])
    if abs(process.create_time()-s['process_created'])>.001 or process.cmdline()[3:]!=s['command']:
        raise ValueError('Native supervisor identity mismatch')
    programs=[bound(REPO/s) for s in (*REPORT_PROGRAMS,'scripts/kalcb_population_report_supervisor_v1.py')]
    out.mkdir(parents=True)
    write(out/'protocol.json',{'registered_at_utc':datetime.now(timezone.utc).isoformat(),'native_root':str(root),
        'dependency':{'pid':process.pid,'process_created':process.create_time(),'command':process.cmdline()},
        'programs':programs,'bindings':p['programs']+p['bindings']+[bound(root/'protocol.json'),bound(tests),test['log']],
        'tests':str(tests),'scope':'Wait without touching healthy native work; after all36 native cases and supervisor exit, run read-only all-case assessment, immutable acceptance packaging and eight shared-PhaseRunner full reports. No new trading trials or canonical changes. Final economic judgment remains pending.'})
    print(str(out/'protocol.json'),flush=True)


def run(out):
    import psutil,msvcrt
    protocol=read(out/'protocol.json');verify(protocol['programs']+protocol['bindings'])
    root=Path(protocol['native_root']);identity=protocol['dependency']
    lock=(out/'run.lock').open('a+b');lock.seek(0)
    if not lock.read(1):lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if (out/'status.json').exists():raise ValueError('Preserve prior supervisor evidence; use versioned successor')
    state={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'command':sys.argv,'child':None}
    def status(stage,**extras):
        state.update(stage=stage,updated_at_utc=datetime.now(timezone.utc).isoformat(),**extras);write(out/'status.json',state)
    def lookup(pid):
        try:return psutil.Process(pid)
        except psutil.NoSuchProcess:return None
    def child(stage,command,log):
        verify(protocol['programs']+protocol['bindings'])
        with log.open('x',encoding='utf-8') as stream:
            p=subprocess.Popen(command,cwd=REPO,env=env(REPO),stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            status(stage,child={'pid':p.pid,'process_created':psutil.Process(p.pid).create_time(),'command':command},log=str(log))
            while p.poll() is None:time.sleep(10)
        if p.returncode:raise ValueError('Report stage failed; preserve '+str(log))
        status(stage+'_complete',child=None)
    try:
        status('waiting_for_native_completion')
        while not dependency_ready(root,identity,lookup):time.sleep(30)
        prefix=[sys.executable,'-B','-u']
        assessment=out/'assessment_v1/assessment.json';artifacts=out/'artifacts'
        child('assessing',prefix+['scripts/kalcb_funded_qualification_assessment_v1.py','--root',str(root),
            '--output',str(assessment.parent)],out/'assessment.log')
        child('preparing_reports',prefix+['scripts/kalcb_population_final_artifacts_v1.py','prepare','--root',str(root),
            '--output',str(artifacts),'--assessment',str(assessment),'--tests',protocol['tests']],out/'prepare.log')
        reports=[]
        for r in read(artifacts/'preparation.json')['profiles']:
            log=out/(r['name']+'_'+r['period']+'.log')
            child('rendering_'+r['name']+'_'+r['period'],prefix+['scripts/kalcb_population_final_artifacts_v1.py','render',
                '--output',str(artifacts),'--name',r['name'],'--period',r['period']],log)
            done=artifacts/'profiles'/r['name']/r['period']/'diagnostics_completion.json'
            if not read(done)['complete']:raise ValueError('Uncompleted report')
            reports.append(bound(done))
        verify(protocol['programs']+protocol['bindings'])
        write(out/'completion.json',{'complete':True,'assessment':bound(assessment),'reports':reports,
            'study_complete':False,'final_economic_assessment_required':True,'canonical_replaced':False,'live_enabled':False})
        status('awaiting_final_economic_review',child=None)
    except Exception as e:status('attention_required',error=repr(e));raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('register','run'))
    for k in ('root','output','tests'):p.add_argument('--'+k,type=Path,required=k=='output')
    a=p.parse_args()
    if a.stage=='register':register(a.root,a.output,a.tests)
    else:run(a.output)
