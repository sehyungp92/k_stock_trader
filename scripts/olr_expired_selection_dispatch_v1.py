"""Run only the predefined0039 causal review after four accepted accounts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil
from olr_selective_half_continuation_v1 import read,write,sha,now,env_for


def update(root,phase,**fields):
    value=dict(at=now(),phase=phase,**fields)
    p=root/'review_dispatch_status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf8');tmp.replace(p)
    print(json.dumps(value),flush=True)


def run(root):
    binding=read(root/'review_dispatch_binding.json')
    review=Path(__file__).with_name('olr_expired_selection_review_v1.py')
    assert sha(__file__)==binding['dispatcher_sha256'] and sha(review)==binding['review_sha256']
    assert sha(root/'protocol.json')==binding['campaign_protocol_sha256']
    previous=None
    while not (root/'assessment.json').exists():
        state=read(root/'status.json') if (root/'status.json').exists() else {}
        if state.get('phase') in ('needs_repair','preserved_preparation_failure'):
            update(root,'campaign_requires_repair',campaign_state=state);return
        receipt=read(root/'controller_process.json')
        try:
            p=psutil.Process(receipt['pid'])
            live=abs(p.create_time()-receipt['created_at'])<.001 and p.cmdline()==receipt['command']
        except psutil.NoSuchProcess:live=False
        if not live:
            if (root/'assessment.json').exists():break
            update(root,'campaign_not_live_without_assessment',campaign_state=state);return
        phase=(state.get('phase'),str(state.get('case')),state.get('window'))
        if phase!=previous:update(root,'waiting_for_fixed_campaign',campaign_state=state);previous=phase
        time.sleep(30)
    assert all((root/'accepted'/(w+'_'+p+'.json')).exists() for w in ('train','oos') for p in ('control','access'))
    if (root/'causal_review_v1').exists():
        update(root,'existing_review_requires_inspection');return
    assert sha(review)==binding['review_sha256']
    cmd=[sys.executable,'-X','utf8','-B','-u',str(review),str(root)]
    with (root/'causal_review.log').open('x',encoding='utf8') as stream:
        p=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env_for(root),creationflags=0x08000000)
        write(root/'review_worker_process.json',dict(at=now(),pid=p.pid,created_at=psutil.Process(p.pid).create_time(),command=cmd))
        update(root,'review_running',pid=p.pid)
        code=p.wait()
    update(root,'review_complete_awaiting_economic_decision' if code==0 else 'review_failed_preserve_and_repair',returncode=code)


def start(root):
    script=Path(__file__).resolve();review=script.with_name('olr_expired_selection_review_v1.py')
    write(root/'review_dispatch_binding.json',dict(at=now(),dispatcher_sha256=sha(script),review_sha256=sha(review),campaign_protocol_sha256=sha(root/'protocol.json')))
    cmd=[sys.executable,'-X','utf8','-B','-u',str(script),'run',str(root)]
    with (root/'review_dispatch.log').open('x',encoding='utf8') as stream:
        p=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env_for(root),creationflags=0x08000000)
    value=dict(at=now(),pid=p.pid,created_at=psutil.Process(p.pid).create_time(),command=cmd)
    write(root/'review_dispatch_process.json',value);print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['start','run']);p.add_argument('root',type=Path);a=p.parse_args()
    try:globals()[a.mode](a.root)
    except Exception as exc:
        update(a.root,'dispatcher_failed',error=repr(exc));raise
