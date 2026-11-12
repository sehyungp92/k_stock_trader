"""Extend original opening/checkpoint native targets into the corrected OOS."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_full_oos_recovery import ROOT,FULL,read,write,bound,env

def run(root):
    labels=root/'labels';labels.mkdir(exist_ok=True);jobs=[]
    oldroot=Path('E:/CodexResearch/k_stock_trader')
    originals={'opening':oldroot/'kalcb_action_value_20261006/quantity_plan_v2.json',
        'timed':oldroot/'kalcb_intraday_advantage_20261007/checkpoint/plan.json'}
    for kind,stage,planname in [('opening','continuation','plan.json'),('timed','checkpoint','checkpoint_plan.json')]:
        plan=FULL/'forward_features'/planname;prepared=FULL/'forward_features'/stage
        before=read(originals[kind]);after=read(plan)
        if read(before['config'])!=read(after['config']):raise ValueError('Changed target management')
        for scenario in ('baseline','combined'):
            name=f'{kind}_{scenario}';start=1
            if name=='opening_baseline':name+='_rest';start=2
            jobs.append({'kind':kind,'scenario':scenario,'plan':str(plan),'prepared':str(prepared),
                'output':str(labels/name),'start_index':start,'original_target_plan':bound(originals[kind]),
                'prepared_input':bound(prepared/'prepared.json')})
    protocol={'registered_at_utc':datetime.now(timezone.utc).isoformat(),'program':bound(__file__),'jobs':jobs,
        'scales':[.25,.5,1.],'membership':'Causal ready native guard population; retain losses, no-fills and physical failures.',
        'purpose':'Unchanged target extension for causal scheduled refits; not selected-management target alignment.'}
    if not (labels/'protocol.json').exists():write(labels/'protocol.json',protocol)
    elif read(labels/'protocol.json')['jobs']!=jobs:raise ValueError('Changed registered label jobs')
    code=root/'inputs/files'
    for job in jobs:
        out=Path(job['output'])
        if (out/'completion.json').exists():continue
        if out.exists():raise ValueError('Preserve partial grid '+str(out))
        command=[sys.executable,'-B','-u','-m','scripts.kalcb_action_quantity_labels','--plan',job['plan'],
            '--prepared',job['prepared'],'--output',str(out),'--scenario',job['scenario'],'--scales','.25','.5','1',
            '--start-index',str(job['start_index'])]
        log=labels/(out.name+'.log')
        with log.open('x',encoding='utf-8') as f:rc=subprocess.run(command,cwd=code,env=env(code),stdout=f,stderr=subprocess.STDOUT).returncode
        write(labels/(out.name+'.receipt.json'),{'command':command,'returncode':rc,'log':str(log)})
        if rc:raise ValueError('Label extension failed: '+str(log))
        print(json.dumps({'completed':out.name,'counts':read(out/'grid_manifest.json')['counts']}),flush=True)
    write(labels/'completion.json',{'complete':True,'protocol':bound(labels/'protocol.json'),
        'grids':[bound(p/'grid_manifest.json') for p in labels.iterdir() if p.is_dir() and (p/'completion.json').exists()]})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);run(p.parse_args().root)
