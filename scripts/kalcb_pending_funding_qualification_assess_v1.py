"""Register and assess completed V28 once; read-only, no next experiment."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys,os
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify
from backtests.strategies.kalcb.pending_funding_qualification_assessment_v1 import completion_gate,assess,require


def register(root,output,tests):
    rows=completion_gate(root);p=read(root/'protocol.json');verify(p['programs']+p['bindings'])
    t=read(tests);require(t['exit_code']==0,'Tests failed');verify(t['programs']+[t['log']])
    bindings=[bound(root/f) for f in ('protocol.json','study.json','progress.json','supervisor_start.json','stage_completion.json')]+[bound(tests)]
    for r in rows:
        folder=Path(r['folder'])
        bindings += [bound(folder/f) for f in ('completion.json','result.json','observer_evidence.json','independent_exposure_mtm.json')]
    programs=p['programs']+[bound(REPO/f) for f in ('scripts/kalcb_pending_funding_qualification_assess_v1.py',
        'backtests/strategies/kalcb/pending_funding_qualification_assessment_v1.py','tests/backtests/strategies/test_kalcb_pending_funding_qualification_assessment_v1.py')]
    output.mkdir(parents=True,exist_ok=False)
    write(output/'protocol.json',{'version':'pending-funding-four-stress-assessment-v1','registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'root':str(root),'programs':programs,'bindings':bindings+p['bindings'],'tests':bound(tests),
        'activity':'Unique profitable position lifetimes and profitable notional/PnL; one-share separately, disclosed 0.1/1/5%-equity size sensitivities, no policy thresholds.',
        'scope':'Read-only prior100 four-stress sixteen-account comparison (eight fresh, eight authenticated V25 reuse). Complete all existing source/account checks and attribution. No economic experiments or promotion.'})


def run(root,output):
    import psutil,msvcrt
    completion_gate(root);require(not (output/'supervisor_start.json').exists(),'Preserve previous attempt')
    lock=(output/'run.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    temp=output/'temp';temp.mkdir(exist_ok=True);os.environ['TEMP']=os.environ['TMP']=str(temp)
    identity={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'command':psutil.Process().cmdline()}
    write(output/'supervisor_start.json',identity)
    try:
        write(output/'status.json',{**identity,'stage':'read_only_assessment','child':None});assess(root,output)
        write(output/'status.json',{**identity,'stage':'complete','child':None,'study_complete':False})
    except Exception as e:
        write(output/'status.json',{**identity,'stage':'attention_required','error':repr(e)});raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=('register','run'))
    for name in ('root','output','tests'):ap.add_argument('--'+name,type=Path,required=name!='tests')
    a=ap.parse_args();register(a.root,a.output,a.tests) if a.stage=='register' else run(a.root,a.output)
