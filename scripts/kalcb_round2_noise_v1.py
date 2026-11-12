"""C16 bounded deterministic repeats and registered +/-5% capital sensitivity."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import psutil

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,bound,digest,env,source_hash
from backtests.auto.shared.cache_keys import stable_signature
from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_case

BASE=Path('E:/CodexResearch/k_stock_trader')
SOURCE=BASE/'kalcb_opportunity_lifecycle_20261016/funded_interaction_v1'
ROOT=BASE/'kalcb_round2_assessment_20261017/v1'
OUT=ROOT/'noise'
SCENARIOS=('baseline','participation','cost','combined')


def write(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,default=str,allow_nan=False)


def freeze(old,period,initial):
    folder=OUT/'runtime'/f'{period}_{initial}';folder.mkdir(parents=True,exist_ok=False)
    mp=Path(old['manifest']['path']);parent=read(mp);sig=parent.pop('manifest_hash');assert stable_signature(parent)==sig
    plan=read(old['plan']['path']);ctx=read(plan['audit_configuration']);ctx['initial_equity']=initial;ctx['output_root']=str(folder)
    write(folder/'audit_configuration.json',ctx)
    plan['input_sha256'].pop(plan['audit_configuration']);plan['audit_configuration']=str(folder/'audit_configuration.json')
    plan['input_sha256'][plan['audit_configuration']]=digest(plan['audit_configuration'])
    plan['assessment_capital']={'initial_equity':initial,'protocol':bound(OUT/'registration.json'),'unchanged_cap':2.0}
    write(folder/'plan.json',plan)
    rel='scripts/kalcb_opportunity_interaction_worker_v1.py';worker=(mp.parent/'files'/rel).read_text()
    anchor="    a.output.mkdir(parents=True,exist_ok=False)"
    assert worker.count(anchor)==1
    replacement="    from backtests.strategies.kalcb import executed_value as ev\n    initial=plan['assessment_capital']['initial_equity']\n    if initial not in (95000000,100000000,105000000) or read(plan['audit_configuration'])['initial_equity']!=initial:\n        raise ValueError('Unregistered initial capital')\n    if ev.REFERENCE_EQUITY!=100000000:raise ValueError('Unexpected initial constant')\n    ev.REFERENCE_EQUITY=initial\n"+anchor
    worker=worker.replace(anchor,replacement).replace('    finally:controller.restore()', '    finally:\n        controller.restore();ev.REFERENCE_EQUITY=100000000.')
    (folder/'worker.py').write_text(worker,encoding='utf-8')
    replacements={rel:folder/'worker.py','research/kalcb_executed_value/plan.json':folder/'plan.json',
        'research/holding_account/audit_configuration.json':folder/'audit_configuration.json',
        'research/c16/registration.json':OUT/'registration.json'}
    code=folder/'inputs/files';files={}
    for relative,sha in parent['files'].items():
        if relative in replacements:continue
        src=mp.parent/'files'/relative;dst=code/relative;dst.parent.mkdir(parents=True,exist_ok=True)
        if digest(src)!=sha:raise ValueError('Changed parent input '+relative)
        os.link(src,dst);files[relative]=sha
    for relative,src in replacements.items():
        dst=code/relative;dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst);files[relative]=digest(src)
    payload=copy.deepcopy(parent);payload['files']=files;payload['source_generation_hash']=source_hash(code)
    payload['evidence']['c16_capital']={'parent_manifest':old['manifest'],'initial_equity':initial,'registration':bound(OUT/'registration.json')}
    write(folder/'inputs/manifest.json',{**payload,'manifest_hash':stable_signature(payload)})
    policy=code/'research/kalcb_executed_value/policies/policies_manifest.json';assert digest(policy)==old['policy']['sha256']
    # Every native economic code file other than the explicit adapter is unchanged.
    changed=[r for r in parent['files'] if r.endswith('.py') and files.get(r)!=parent['files'][r]]
    assert changed==[rel],changed
    item=dict(period=period,initial=initial,folder=str(folder),manifest=bound(folder/'inputs/manifest.json'),plan=bound(folder/'plan.json'),
        policy=bound(policy),source_generation_hash=payload['source_generation_hash'],changed_python=changed)
    write(folder/'profile.json',item);return item


def execute(item,period,scenario,name,initial,bridge=None):
    folder=OUT/'cases'/name;code=Path(item['manifest']['path']).parent/'files'
    cmd=[sys.executable,'-B','-u','-m','scripts.kalcb_opportunity_interaction_worker_v1','--plan',item['plan']['path'],
        '--policies',str(Path(item['policy']['path']).parent),'--audit-manifest',item['manifest']['path'],
        '--output',str(folder),'--scenario',scenario,'--mode','veto']
    if bridge:cmd+=['--bridge',str(bridge)]
    for k in ('manifest','plan','policy'):assert digest(item[k]['path'])==item[k]['sha256']
    began=time.monotonic();peak=0;stop=None
    with (OUT/(name+'.log')).open('x',encoding='utf-8') as log:
        p=subprocess.Popen(cmd,cwd=code,env=env(code),stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        proc=psutil.Process(p.pid);write(OUT/(name+'_launch.json'),dict(pid=p.pid,created=proc.create_time(),argv=cmd))
        while p.poll() is None:
            try:peak=max(peak,proc.memory_info().rss)
            except psutil.NoSuchProcess:break
            if peak>2*1024**3 or time.monotonic()-began>1200:
                stop='memory_guard' if peak>2*1024**3 else 'time_guard';p.terminate();break
            time.sleep(2)
        codeexit=p.wait()
    receipt=dict(name=name,period=period,scenario=scenario,initial=initial,seconds=time.monotonic()-began,peak_rss=peak,exit_code=codeexit,guard=stop)
    write(OUT/(name+'_process.json'),receipt)
    if codeexit or stop:raise RuntimeError('Native failed '+name)
    result=read(folder/'result.json');start,end,n=('2025-08-07','2026-03-31',156) if period=='is' else ('2026-04-01','2026-07-10',69)
    profile=dict(execution_source={'source_generation_hash':item['source_generation_hash']},audit_manifest=item['manifest']['path'],
        policy={'name':'interaction_cost_margin','manifest_sha256':item['policy']['sha256'],'directory':str(Path(item['policy']['path']).parent)},
        initial_equity_krw=initial,date_range={'start':start,'end':end,'sessions':n},metrics={scenario:result})
    if result['residual_sessions'] or result['terminal_inventory'] or result['terminal_pending']:
        receipt['qualified']=False;receipt['reason']='intraday_or_terminal_residual'
    else:
        checked=validate_case(folder,profile,scenario);write(folder/'independent_account_verification.json',checked['validation']);receipt['qualified']=True
    write(folder/'input_profile.json',item)
    receipt['result']=bound(folder/'result.json');receipt['profile']=bound(folder/'input_profile.json')
    write(OUT/(name+'_verdict.json'),receipt)
    print(json.dumps(receipt,default=str),flush=True)
    return receipt


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    write(OUT/'registration.json',dict(program=bound(__file__),design=bound(REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0041.md'),
        binding=bound(REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0041-capital-binding.md'),
        initial_equities=[95000000,100000000,105000000],max_accounts_including_external_pilot=26,no_fits=True,no_treatment=True))
    old=read(SOURCE/'registration.json')['profiles'];items={};receipts=[]
    for period in ('is','oos'):
        # Adapter bridge first: future capital cells may not run after a bridge failure.
        item=freeze(old[period+'_cap2'],period,100000000);items[period,100000000]=item
        sc='baseline' if period=='is' else 'combined'
        receipts.append(execute(item,period,sc,f'bridge_{period}',100000000,SOURCE/'cases'/f'combined_{period}_{sc}'))
    for period in ('is','oos'):
        for initial in (95000000,105000000):items[period,initial]=freeze(old[period+'_cap2'],period,initial)
    for period in ('is','oos'):
        for sc in SCENARIOS:
            if (period,sc)!=('is','baseline'):
                receipts.append(execute(old[period+'_cap2'],period,sc,f'exact_{period}_{sc}',100000000,SOURCE/'cases'/f'combined_{period}_{sc}'))
            for initial in (95000000,105000000):
                receipts.append(execute(items[period,initial],period,sc,f'capital_{period}_{sc}_{initial}',initial))
    write(OUT/'completion.json',dict(complete=True,cases=receipts,accounts=len(receipts)+1,extra_pilot=str(ROOT/'is_baseline_exact')))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():write(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
