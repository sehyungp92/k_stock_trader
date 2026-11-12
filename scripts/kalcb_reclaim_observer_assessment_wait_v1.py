"""Wait for the exact native V21 exit, then register/run its read-only assessment once."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify
from backtests.strategies.kalcb.reclaim_observer_assessment_v1 import completion_gate,exact_process,require


def dependency_ready(root,identity,alive=exact_process):
    require(read(root/'supervisor_start.json')==identity,'Native supervisor identity changed')
    state=read(root/'status.json')
    require(state['stage']!='attention_required','Native dependency failed; preserve evidence')
    if alive(identity):return False
    completion_gate(root,alive)
    return True


def main(directory):
    import psutil,msvcrt
    p=read(directory/'protocol.json');verify(p['programs']+p['bindings'])
    root=Path(p['native_root']);output=Path(p['assessment_output']);tests=Path(p['assessment_tests'])
    require(not output.exists(),'Assessment output already exists; never duplicate or resume it')
    require(not (directory/'supervisor_start.json').exists(),'Preserve previous wait attempt')
    lock=(directory/'run.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    me={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'command':psutil.Process().cmdline()}
    write(directory/'supervisor_start.json',me)
    def state(stage,**kw):write(directory/'status.json',{**me,'stage':stage,'updated_at_utc':datetime.now(timezone.utc).isoformat(),**kw})
    def child(stage,command):
        log=directory/(stage+'.log')
        with log.open('w',encoding='utf-8') as stream:
            job=subprocess.Popen(command,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW,env={**os.environ,'TEMP':str(directory/'temp'),'TMP':str(directory/'temp'),
                'PYTHONDONTWRITEBYTECODE':'1','PYTHONUTF8':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
            process=psutil.Process(job.pid)
            identity={'pid':job.pid,'process_created':process.create_time(),'command':process.cmdline()}
            require(identity['command']==command,'Wrong assessment child command')
            write(directory/(stage+'_child.json'),identity);state(stage,child=identity,log=str(log))
            code=job.wait()
        require(code==0,'Read-only '+stage+' failed; preserve registered attempt: '+str(log))
        return bound(log)
    try:
        state('waiting_for_native_exit',child=None,native=p['native_supervisor'])
        while not dependency_ready(root,p['native_supervisor']):time.sleep(30)
        verify(p['programs']+p['bindings'])
        require(not output.exists(),'Another assessment was registered; do not duplicate')
        write(directory/'dependency_completion.json',{'passed':True,'native_completion':bound(root/'stage_completion.json'),
            'native_progress':bound(root/'progress.json'),'exact_native_supervisor_exited':True})
        script=str(REPO/'scripts/kalcb_reclaim_observer_assessment_v1.py')
        command=[sys.executable,'-B','-u',script]
        registration=child('register',command+['register','--root',str(root),'--output',str(output),'--tests',str(tests)])
        assessment=child('assess',command+['run','--root',str(root),'--output',str(output)])
        done=read(output/'completion.json');require(done['complete'] is True and done['native_cases']==done['exact_bridges']==8,'Incomplete assessment')
        verify(p['programs']+p['bindings'])
        write(directory/'completion.json',{'complete':True,'assessment':bound(output/'assessment.json'),
            'logs':[registration,assessment],'study_complete':False})
        state('complete',child=None,study_complete=False)
    except Exception as error:
        state('attention_required',error=repr(error));raise
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    main(parser.parse_args().directory.resolve())
