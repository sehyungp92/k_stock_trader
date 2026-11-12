"""Finite background native/reference review pipeline; no automatic next grid."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,default=str)
def run(root):
    import psutil
    sys.path.insert(0,str(REPO))
    from backtests.strategies.kalcb.research_process_v1 import execute
    p=read(root/'protocol.json')
    for binding in p['programs']+p['bindings']:
        with Path(binding['path']).open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
        if sha!=binding['sha256']:raise ValueError('Changed registered source')
    me=psutil.Process();identity={'pid':me.pid,'process_created':me.create_time(),'command':me.cmdline()}
    write(root/'pipeline_start.json',{**identity,'at_utc':datetime.now(timezone.utc).isoformat()})
    try:
        execute(root,[sys.executable,'-I','-B','-u',str(REPO/'scripts/kalcb_staged_reference_pilot_v1.py'),
            'run','--root',str(root)],'native',REPO,identity)
        execute(root,[sys.executable,'-I','-B','-u',str(REPO/'scripts/kalcb_staged_reference_review_v1.py'),
            '--root',str(root)],'review',REPO,identity)
        done=read(root/'review_v1/completion.json')
        if not done['passed']:raise ValueError('Independent review failed')
        write(root/'pipeline_completion.json',{'complete':True,'native_and_review_complete':True,
            'study_complete':False,'at_utc':datetime.now(timezone.utc).isoformat()})
    except Exception as e:
        write(root/'pipeline_failure.json',{'error':repr(e),'preserve_evidence':True});raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();run(a.root)
