"""Mechanical dependency handoff; never owns strategy selection or optimization."""
import argparse,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,verify,bound,env


def same_process(identity,process):
    return abs(process.create_time()-identity['process_created'])<.01 and process.cmdline()==identity['command']


def ready(preflight):
    receipt=preflight/'preflight/completion.json'
    if not receipt.exists():return False
    value=read(receipt)
    if not value.get('passed') or value.get('native_identity_bridges')!=4:raise ValueError('Incomplete identity gates')
    if any(value['fresh_filled_labels'][s].get('filled',0)<1 for s in ('baseline','combined')):raise ValueError('Unfilled real-data smoke gate')
    return True


def run(root):
    import psutil,msvcrt
    request=read(root/'request.json');verify(request['programs']+request['bindings'])
    lock=(root/'run.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    self_id={'pid':os.getpid(),'process_created':psutil.Process().create_time(),'command':psutil.Process().cmdline()}
    write(root/'supervisor_start.json',self_id)
    preflight=Path(request['preflight_root']);identity=request['preflight_process']
    try:
        while True:
            complete=ready(preflight)
            try:p=psutil.Process(identity['pid']);active=same_process(identity,p)
            except psutil.NoSuchProcess:active=False
            if not active:
                if not complete:raise RuntimeError('The registered preflight exited without all gates passing')
                break
            if (preflight/'failed_stage.json').exists():raise RuntimeError('Preflight reported a failure; preserve evidence')
            write(root/'status.json',{'stage':'waiting_for_existing_preflight','supervisor':self_id,'preflight':identity,
                'gates_complete':complete,'updated_at_utc':datetime.now(timezone.utc).isoformat()})
            time.sleep(15)
        code=Path(request['repository']);study=Path(request['study_root']);script=code/'scripts/kalcb_late_entry_study_v3.py'
        commands=[('run',[sys.executable,'-B','-u',str(script),'run','--root',str(study)])]
        for stage,command in commands:
            verify(request['programs']+request['bindings'])
            with (root/(stage+'.log')).open('x',encoding='utf-8') as log:
                child=subprocess.Popen(command,cwd=code,env=env(code),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                child_id={'pid':child.pid,'process_created':psutil.Process(child.pid).create_time(),'command':command}
                write(root/'status.json',{'stage':stage,'supervisor':self_id,'child':child_id,'updated_at_utc':datetime.now(timezone.utc).isoformat()})
                while child.poll() is None:time.sleep(15)
            if child.returncode:raise RuntimeError('Versioned study stage failed: '+stage)
        write(root/'completion.json',{'pipeline_complete':True,'study_complete':False,'next':'Economic review, credible finalist qualification and existing-engine full reports via hourly follow-up.'})
        write(root/'status.json',{'stage':'awaiting_qualification','supervisor':self_id,'child':None})
    except Exception as e:
        write(root/'status.json',{'stage':'attention_required','supervisor':self_id,'error':repr(e)});raise
    finally:lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
