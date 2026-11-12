"""Wait for exact V28 exit; register assessment and eight new/eight reused full reports once."""
import argparse,os,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,verify
from backtests.strategies.kalcb.pending_funding_qualification_assessment_v1 import completion_gate,require


def alive(identity):
    import psutil
    try:p=psutil.Process(identity['pid'])
    except psutil.NoSuchProcess:return False
    if abs(p.create_time()-identity['process_created'])>=.01:return False
    require(p.cmdline()==identity['command'],'Same-created process command mismatch')
    return True


def ready(root,identity,check_alive=alive):
    status=read(root/'status.json')
    require(status['stage']!='attention_required','Native batch failed; preserve and reassess')
    if check_alive(identity):return False
    completion_gate(root,alive=check_alive)
    return True


def run(output):
    import psutil,msvcrt
    from backtests.strategies.kalcb.research_process_v1 import execute
    p=read(output/'protocol.json');verify(p['programs']+p['bindings']);root=Path(p['root'])
    require(not (output/'supervisor_start.json').exists(),'Preserve prior launcher')
    lock=(output/'run.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    own={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'command':psutil.Process().cmdline()}
    write(output/'supervisor_start.json',own)
    temp=output/'temp';temp.mkdir(exist_ok=True);os.environ['TEMP']=os.environ['TMP']=str(temp)
    try:
        while not ready(root,p['native_supervisor']):
            write(output/'status.json',{**own,'stage':'waiting_exact_native_completion','child':None,
                'audited':len(read(root/'progress.json'))});time.sleep(15)
        require(not (root/'assessment_v1').exists() and not (root/'full_reports_v1').exists(),'Existing assessment/report registration; do not duplicate')
        verify(p['programs']+p['bindings'])
        commands=[('register_assessment','scripts/kalcb_pending_funding_qualification_assess_v1.py',
            ['register','--root',str(root),'--output',str(root/'assessment_v1'),'--tests',p['tests']]),
            ('assess','scripts/kalcb_pending_funding_qualification_assess_v1.py',['run','--root',str(root),'--output',str(root/'assessment_v1')]),
            ('register_reports','scripts/kalcb_pending_funding_qualification_reports_register_v1.py',
            ['--root',str(root),'--output',str(root/'full_reports_v1'),'--tests',p['tests']]),
            ('reports','scripts/kalcb_pending_funding_qualification_reports_v1.py',['run','--output',str(root/'full_reports_v1')]),
            ('verify_reports','scripts/kalcb_pending_funding_qualification_report_verify_v1.py',
            ['--root',str(root),'--output',str(root/'full_reports_v1/verification_v1')])]
        for label,script,args in commands:
            execute(output,[sys.executable,'-B','-u',str(REPO/script),*args],label,REPO,own)
        verify(p['programs']+p['bindings'])
        write(output/'completion.json',{'complete':True,'study_complete':False,'automatic_next_experiment':False,
            'scope':'Sixteen accounted native cells (eight fresh/eight reuse), assessment and eight new/eight reused full reports verified. Joint economic judgment remains required; no automatic next batch.'})
        write(output/'status.json',{**own,'stage':'awaiting_joint_economic_judgment','child':None})
    except Exception as error:
        write(output/'status.json',{**own,'stage':'attention_required','error':repr(error)});raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);run(ap.parse_args().output)
