"""Bounded native preflight only. Shared PhaseRunner will own economic search."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import psutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb.retention_research_io_v1 import read,bound,digest,env,PERIODS
from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_case
from scripts.kalcb_round3_freeze_v1 import write

ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_round3_20261017/attempt_2')
OUT=ROOT/'bridge_v1'
SOURCE=Path('E:/CodexResearch/k_stock_trader/kalcb_opportunity_lifecycle_20261016/funded_interaction_v1')


def execute(period,scenario,hydrate=False):
    item=read(OUT/'preparation.json')['profiles'][period]
    name=period+'_'+scenario+('_hydration' if hydrate else '_observer')
    folder=OUT/'accounts'/name;code=Path(item['manifest']['path']).parent/'files'
    for k in ('manifest','plan','policy'):
        if digest(item[k]['path'])!=item[k]['sha256']:raise ValueError('Changed source capsule')
    cmd=[sys.executable,'-B','-u','-m','scripts.kalcb_round3_native_worker_v1',
         '--plan',item['plan']['path'],'--policies',str(Path(item['policy']['path']).parent),
         '--audit-manifest',item['manifest']['path'],'--output',str(folder),
         '--scenario',scenario,'--observe','--bridge',str(SOURCE/'cases'/f'combined_{period}_{scenario}')]
    if hydrate:cmd.append('--hydrate')
    start=time.monotonic();peak=0;guard=None
    with (OUT/(name+'.log')).open('x',encoding='utf-8') as log:
        child=subprocess.Popen(cmd,cwd=code,env=env(code),stdout=log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        proc=psutil.Process(child.pid)
        write(OUT/(name+'_launch.json'),dict(pid=child.pid,created=proc.create_time(),argv=cmd))
        while child.poll() is None:
            try:peak=max(peak,proc.memory_info().rss)
            except psutil.NoSuchProcess:break
            if peak>2*1024**3 or time.monotonic()-start>1200:
                guard='memory' if peak>2*1024**3 else 'time';child.terminate();break
            time.sleep(2)
        rc=child.wait()
    receipt=dict(period=period,scenario=scenario,hydration=hydrate,exit_code=rc,
        seconds=time.monotonic()-start,peak_rss=peak,guard=guard)
    write(OUT/(name+'_process.json'),receipt)
    if rc or guard:raise RuntimeError('Failed bridge '+name+'; see preserved log')
    result=read(folder/'result.json');startday,endday,sessions=PERIODS[period]
    profile=dict(execution_source=dict(source_generation_hash=item['source_generation_hash']),
        audit_manifest=item['manifest']['path'],policy=dict(name='interaction_cost_margin',
        manifest_sha256=item['policy']['sha256'],directory=str(Path(item['policy']['path']).parent)),
        initial_equity_krw=1e8,date_range=dict(start=startday,end=endday,sessions=sessions),metrics={scenario:result})
    checked=validate_case(folder,profile,scenario)
    if not read(folder/'full_path_bridge.json')['passed'] or not read(folder/'independent_journal.json')['passed']:
        raise ValueError('Native acceptance incomplete')
    write(folder/'independent_account_verification.json',checked['validation'])
    write(OUT/(name+'_acceptance.json'),dict(passed=True,process=receipt,
        profile=item,files={n:bound(folder/n) for n in ('completion.json','result.json','execution_audit.json',
        'full_path_bridge.json','independent_journal.json','independent_account_verification.json')}))
    print(name+' accepted',flush=True);return receipt


def main():
    write(OUT/'runner_started.json',dict(program=bound(__file__),pid=os.getpid(),max_workers=2,
        started_utc=datetime.now(timezone.utc).isoformat()))
    receipts=[]
    for hydrated in (False,True):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(execute,p,s,hydrated) for p,s in [('is','baseline'),('oos','combined')]]
            for future in as_completed(futures):receipts.append(future.result())
    write(OUT/'completion.json',dict(passed=True,accounts=receipts,max_workers=2,full_economic_path_parity=True,
        scope='Native preflight; treatment round not launched'))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
