"""Run the registered read-only producer and then review in the background."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]


def write(p,value):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)


def run(root):
    import psutil
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    for b in protocol['programs']+protocol['bindings']:
        with Path(b['path']).open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest!=b['sha256']:raise ValueError('Changed registered source')
    me=psutil.Process();identity={'pid':me.pid,'process_created':me.create_time(),'command':me.cmdline()}
    write(root/'pipeline_start.json',{**identity,'at_utc':datetime.now(timezone.utc).isoformat()})
    env=os.environ.copy();env.update(TEMP=str(root/'temp'),TMP=str(root/'temp'),PYTHONDONTWRITEBYTECODE='1')
    (root/'temp').mkdir(exist_ok=True)
    try:
        for stage,script,args in [('support','kalcb_pending_entry_materiality_v1.py',['run']),('review','kalcb_pending_entry_materiality_review_v1.py',[])]:
            command=[sys.executable,'-I','-B','-u',str(REPO/'scripts'/script),*args,'--root',str(root)]
            with (root/(stage+'.log')).open('xb') as log:
                child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                process=psutil.Process(child.pid)
                observed={'pid':child.pid,'process_created':process.create_time(),'command':process.cmdline()}
                if observed['command']!=command:raise ValueError('Child full command differs')
                write(root/(stage+'_launch_receipt.json'),observed)
                (root/'status.json').write_text(json.dumps({**identity,'stage':stage,'child':observed}),encoding='utf-8')
                while child.poll() is None:time.sleep(1)
                if child.returncode:raise ValueError(stage+' exited '+str(child.returncode))
                write(root/(stage+'_exit_receipt.json'),{'identity':observed,'exit_code':child.returncode})
        if not json.loads((root/'review_v1/completion.json').read_text())['passed']:raise ValueError('Review failed')
        write(root/'pipeline_completion.json',{'complete':True,'study_complete':False,'at_utc':datetime.now(timezone.utc).isoformat()})
    except Exception as error:
        write(root/'pipeline_failure.json',{'error':repr(error),'preserve_evidence':True});raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();run(a.root)
