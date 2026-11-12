"""Resource-bounded full-account scenario comparison; one replay at a time."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('code','plan','policies','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--names',nargs='+')
    p.add_argument('--scenarios',nargs='+',default=['baseline','combined'])
    p.add_argument('--audit-manifest',type=Path)
    a=p.parse_args()
    manifest=json.loads((a.policies/'policies_manifest.json').read_text())
    names=a.names or manifest['policies']
    if set(names)-set(manifest['policies']):raise ValueError('Undeclared policy')
    if set(a.scenarios)-{'baseline','participation','cost','combined'}:raise ValueError('Undeclared scenario')
    a.output.mkdir(parents=True,exist_ok=True)
    records=[]
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
    for name in names:
        for scenario in a.scenarios:
            target=a.output/(name+'_'+scenario)
            if not (target/'completion.json').exists():
                if target.exists():raise FileExistsError(f'Preserved incomplete attempt: {target}')
                command=[sys.executable,'-u',str(a.code/'scripts/kalcb_executed_value_replay.py'),
                    '--plan',str(a.plan),'--policies',str(a.policies),'--policy',name,
                    '--scenario',scenario,'--output',str(target)]
                if a.audit_manifest:command.extend(['--audit-manifest',str(a.audit_manifest)])
                print(json.dumps({'start':name,'scenario':scenario}),flush=True)
                with (a.output/(name+'_'+scenario+'.log')).open('x',encoding='utf-8') as log:
                    result=subprocess.run(command,cwd=a.code,env=env,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f'Replay failed ({result.returncode}); see {name}_{scenario}.log')
            result=json.loads((target/'result.json').read_text())
            request=json.loads((target/'request.json').read_text())
            if Path(request['arguments']['plan']).resolve()!=a.plan.resolve() or Path(request['arguments']['policies']).resolve()!=a.policies.resolve():
                raise ValueError('Existing replay is from another input generation')
            compact={k:result[k] for k in ('policy','scenario','return_pct','drawdown_pct','trades',
                'active_days','residual_sessions','terminal_inventory','execution_validity_pass','audit_status')}
            compact['result_path']=str(target/'result.json')
            records.append(compact)
            (a.output/'comparison.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
            print(json.dumps(compact),flush=True)


if __name__=='__main__':main()
