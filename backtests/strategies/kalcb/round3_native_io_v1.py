"""Bounded native evaluation adapter for the shared KALCB PhaseRunner plugin."""
from datetime import datetime,timezone
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import psutil
from .retention_research_io_v1 import read,bound,digest,env,PERIODS
from .late_artifact_validation_v2 import validate_case
from .round3_contract_v1 import verify_acceptance

ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_round3_20261017/attempt_2')
NATIVE=ROOT/'native_v4'
SOURCE=Path('E:/CodexResearch/k_stock_trader/kalcb_opportunity_lifecycle_20261016/funded_interaction_v1')


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,default=str,allow_nan=False)


def profile_for(item,folder,scenario):
    start,end,n=PERIODS[item['period']]
    return dict(execution_source={'source_generation_hash':item['source_generation_hash']},
        audit_manifest=item['manifest']['path'],configuration=read(item['plan']['path'])['config'],
        policy=dict(name='interaction_cost_margin',manifest_sha256=item['policy']['sha256'],
                    directory=str(Path(item['policy']['path']).parent)),
        initial_equity_krw=item['initial'],date_range=dict(start=start,end=end,sessions=n),
        result_paths={scenario:str(Path(folder)/'result.json')},metrics={scenario:read(Path(folder)/'result.json')})


def execute(item,scenario,folder,*,smoke=False,hydrate=False,bridge=None,delay=0):
    """One worker, no nested evaluator/pool, failed evidence never returned valid."""
    folder=Path(folder);accept=folder.parent/(folder.name+'_acceptance.json')
    if accept.exists():
        receipt=read(accept);verify_acceptance(receipt);return receipt
    if folder.exists():raise ValueError('Preserve interrupted/failed native attempt: '+str(folder))
    folder.parent.mkdir(parents=True,exist_ok=True)
    code=Path(item['manifest']['path']).parent/'files'
    for k in ('manifest','plan','policy'):
        if digest(item[k]['path'])!=item[k]['sha256']:raise ValueError('Changed native input '+k)
    cmd=[sys.executable,'-B','-u','-m','scripts.kalcb_round3_native_worker_v3','--plan',item['plan']['path'],
        '--policies',str(Path(item['policy']['path']).parent),'--audit-manifest',item['manifest']['path'],
        '--output',str(folder),'--scenario',scenario,'--observe']
    if hydrate:cmd.append('--hydrate')
    if bridge:cmd+=['--bridge',str(bridge)]
    if delay:cmd+=['--delay-minutes',str(delay)]
    if smoke:cmd+=['--smoke-through',read(item['plan']['path'])['round3_execution']['smoke_through']]
    started=time.monotonic();peak=0;guard=None
    with folder.with_suffix('.log').open('x',encoding='utf-8') as log:
        child=subprocess.Popen(cmd,cwd=code,env={**env(code),'ARROW_IO_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        proc=psutil.Process(child.pid)
        write(folder.parent/(folder.name+'_launch.json'),dict(pid=child.pid,created=proc.create_time(),argv=cmd,
            at_utc=datetime.now(timezone.utc).isoformat(),nested_native_workers=0))
        while child.poll() is None:
            try:peak=max(peak,proc.memory_info().rss)
            except psutil.NoSuchProcess:break
            if peak>2*1024**3 or time.monotonic()-started>1200:
                guard='memory_2GiB' if peak>2*1024**3 else 'time_20min';child.terminate();break
            time.sleep(2)
        rc=child.wait()
    process=dict(seconds=time.monotonic()-started,peak_rss=peak,exit_code=rc,guard=guard,
        arm=item['arm'],period=item['period'],scenario=scenario,capital=item['initial'],smoke=smoke,delay=delay)
    write(folder.parent/(folder.name+'_process.json'),process)
    if rc or guard:raise RuntimeError('Native process failed; preserved '+str(folder.with_suffix('.log')))
    if smoke:
        receipt=read(folder/'smoke.json')
        if not receipt['native_lifecycle_completed'] or receipt['eligible_for_scoring'] or receipt['inventory'] or receipt['pending']:
            raise ValueError('Incomplete/invalid smoke boundary')
        receipt=dict(passed=True,smoke_only=True,eligible_for_scoring=False,process=process,item=item,folder=str(folder),
            files={n:bound(folder/n) for n in ('smoke.json','smoke_source.json','smoke_fills.json','decisions.jsonl.gz','orders.jsonl.gz')})
    else:
        profile=profile_for(item,folder,scenario)
        try:
            checked=validate_case(folder,profile,scenario)
            account=read(folder/'independent_journal.json')
            if not account['passed'] or account['terminal_inventory'] or account['pending']:raise ValueError('Independent account failed')
            if bridge and not read(folder/'full_path_bridge.json')['passed']:raise ValueError('Full physical parity failed')
            write(folder/'independent_account_verification.json',checked['validation'])
        except Exception as exc:
            write(folder/'qualification_failure.json',dict(passed=False,error=repr(exc),eligible_for_scoring=False))
            return dict(passed=False,eligible_for_scoring=False,folder=str(folder),process=process,error=repr(exc))
        receipt=dict(passed=True,eligible_for_scoring=True,process=process,item=item,profile=profile,folder=str(folder),
            files={n:bound(folder/n) for n in ('request.json','completion.json','result.json','execution_audit.json',
                'fills.json','trades.json','daily.json','equity.json','decisions.jsonl.gz','orders.jsonl.gz',
                'broker_journal.jsonl.gz','account_states.jsonl.gz','horizon_actions.json',
                'independent_journal.json','independent_account_verification.json')})
    write(accept,receipt);return receipt


def bundle(receipt,scenario):
    verify_acceptance(receipt)
    if not receipt.get('eligible_for_scoring') or receipt.get('smoke_only'):raise ValueError('Smoke cannot become a scored account')
    from scripts.kalcb_executed_value_replay import verify_policy
    from .late_entry_v1 import cumulative_timed_policy
    profile=receipt['profile'];policy_dir=Path(profile['policy']['directory'])
    policy,opening=verify_policy(policy_dir,'interaction_cost_margin')
    updates=cumulative_timed_policy(policy_dir,policy,opening)
    case=validate_case(receipt['folder'],profile,scenario)
    return dict(profile=profile,cases={scenario:case},opening=opening,updates=updates,
        policy={**policy,'prepared_sha256':receipt['item']['manifest']['sha256']},
        manifest=receipt['item']['manifest']['path'],manifest_hash=receipt['item']['manifest']['sha256'])
