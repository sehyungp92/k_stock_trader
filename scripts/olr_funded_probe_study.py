"""Frozen three-arm, one-decision complete-account causal diagnostic.

This is not a rescue sweep or a new policy qualification. All later native
funding and observation effects remain endogenous in every full-history arm.
"""
import argparse
import copy
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import psutil

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.olr_economic_study import read,write,sha,source_hash,sig
from scripts.olr_structural_study import private_write
from scripts.olr_structural_controller import status,progress,verify_identity
from scripts.olr_funded_probe_patches import patch_runner,patch_audit
from scripts.olr_economic_assessment import contrast
from backtests.strategies.olr.economic_evidence import completed_case
from backtests.strategies.olr.funded_action_probe import VERSION

PARENT=Path('E:/CodexResearch/k_stock_trader/olr_structural_execution_20261013/generation_v4')
UPDATED=('backtests/strategies/olr/funded_action_probe.py','scripts/olr_funded_probe_patches.py',
    'scripts/olr_funded_probe_study.py','tests/strategy_olr/test_funded_action_probe.py')


def prepare(root):
    ready=read(PARENT/'native_engines/train/ready.json');source=Path(ready['files'])
    if source_hash(source)!=read(PARENT/'preflight.json')['source']:
        raise ValueError('Changed parent source')
    parity=read(PARENT/'native_parity.json')
    parent=completed_case(PARENT/'cases/train/control_baseline')
    if not parity['pass_'] or not parent['valid'] or parent['receipt_sha256']!=parity['new_completion']:
        raise ValueError('Parent native parity is not authenticated')
    destination=root/'native_engines/train/files';destination.mkdir(parents=True,exist_ok=False)
    manifest=copy.deepcopy(ready['manifest']);manifest.pop('manifest_hash',None)
    for name in manifest['files']:
        p=destination/name;p.parent.mkdir(parents=True,exist_ok=True);os.link(source/name,p)
    for name in UPDATED:
        private_write(destination/name,(REPO/name).read_text(encoding='utf8'))
        manifest['files'][name]=sha(destination/name)
    for name,patcher in [('scripts/portfolio_synergy_study.py',patch_runner),('backtests/engine/olr_minute_audit_v5.py',patch_audit)]:
        text=patcher((source/name).read_text(encoding='utf8'));compile(text,name,'exec')
        private_write(destination/name,text);manifest['files'][name]=sha(destination/name)
    generation=source_hash(destination);manifest['source_generation_hash']=generation
    study=copy.deepcopy(ready['study']);study.update(code=str(destination),source_generation_hash=generation)
    write(root/'native_engines/train/ready.json',dict(files=str(destination),study=study,
        manifest=manifest,interval_qualification=ready['interval_qualification']))
    write(root/'native_engines/train/freeze_receipt.json',dict(parent=str(PARENT),source=generation,
        parent_source=parent['source'],parent_completion=parent['receipt_sha256'],
        parent_ready_sha256=sha(PARENT/'native_engines/train/ready.json'),
        private_files={n:h for n,h in manifest['files'].items() if ready['manifest']['files'].get(n)!=h}))
    write(root/'protocol.json',dict(version='olr-funded-action-diagnostic-v5',
        inherited_protocol=str(PARENT/'protocol.json'),inherited_protocol_sha256=sha(PARENT/'protocol.json'),
        inherited_gates=read(PARENT/'protocol.json'),parent_completion=parent['receipt_sha256'],
        selector=VERSION,selector_definition='First completed own-symbol minute after acquisition day satisfying the unchanged native mfe_fade predicate; quantity >=2; no pending acquisition or conflicting market exit. Ties sort by symbol. No date, symbol or future-outcome selection.',
        actions=['hold','half','exit'],residual_targets='Current quantity, floor(current quantity/2), zero. One intervention only; native management resumes thereafter.',
        horizon='Entire original IS native account horizon, including later source scope, reservations, costs, acquisitions, quantities and market history.',
        required_parity='Hold has exact V4 equity/daily/trades/full NAV; real intraday restored suffix equals uninterrupted hold. Both action shadows and initial complete-account states equal hold.',
        assessment='Authenticate execution/source/current-quantity target; evaluate full-event/common-grid DD, marked peak, direct and downstream funding, acquisitions, notional and winners. One decision is diagnostic, not general policy support.',
        numerical_gates_relaxed=False,oos_authorized=False,economic_objective_achieved=False,
        continuation='After authenticated comparisons, assess full conditional economic effects and supported next population/action design. Continue meaningful hard-eligible entry implementation; do not stop at this engineering diagnostic or promote a single favorable branch.'))
    print(json.dumps(dict(root=str(root),source=generation)),flush=True)


def launch(root,action):
    ready=read(root/'native_engines/train/ready.json');files=Path(ready['files'])
    s=copy.deepcopy(ready['study']);m=copy.deepcopy(ready['manifest'])
    tag=action+'_baseline';out=root/'cases/train'/tag
    if out.exists():raise FileExistsError('Preserve existing output '+str(out))
    reference=None
    if action!='hold':
        hold=root/'cases/train/hold_baseline';c=completed_case(hold)
        if not c['valid']:raise ValueError('Unqualified hold reference')
        reference=read(hold/'funded_probe_receipt.json')
    prefix='research/funded_probe_v1/'+tag+'/'
    for key,name,value in [('structural_settings','structural.json',dict(mode='control',interval_service=True)),
            ('funded_probe_settings','probe.json',dict(version=VERSION,action=action,control_reference=reference))]:
        s[key]=prefix+name;write(files/s[key],value);m['files'][s[key]]=sha(files/s[key])
    s.pop('economic_policy_settings',None);s.pop('drawdown_reserve_contract',None)
    m['evidence']['qualification_path']=ready['interval_qualification']
    s['manifest']=str(files.parent/(tag+'.manifest.json'));s['configuration']['input_manifest_path']=s['manifest']
    s['spec_path']=prefix+'spec.json'
    write(files/s['spec_path'],{k:v for k,v in s.items() if k not in ('code','manifest','spec_path','source_generation_hash')})
    m['files'][s['spec_path']]=sha(files/s['spec_path']);write(s['manifest'],{**m,'manifest_hash':sig(m)})
    sp=root/'runtime/train'/tag/'study.json';write(sp,s)
    command=[sys.executable,'-B','-u',str(files/'scripts/portfolio_synergy_study.py'),'replay','--study',str(sp),
        '--recipe','candidate','--scenario','baseline','--standalone','OLR','--output',str(out),'--full-audit']
    tmp=root/'tmp'/tag;tmp.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(TEMP=str(tmp),TMP=str(tmp),PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    log=root/'logs/train'/(tag+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x',encoding='utf8') as f:
        child=subprocess.Popen(command,cwd=files,env=env,stdout=f,stderr=subprocess.STDOUT,creationflags=0x08000000)
    identity=dict(pid=child.pid,created_at=psutil.Process(child.pid).create_time(),command=command,
        source=s['source_generation_hash'],settings=dict(action=action),log=str(log),case=str(out),started_at=datetime.now(timezone.utc).isoformat())
    write(sp.parent/'process.json',identity)
    return child


def case(root,action):
    output=root/'cases/train'/(action+'_baseline');identity=root/'runtime/train'/(action+'_baseline')/'process.json'
    if not (output/'completion.json').exists():
        if identity.exists():
            import time
            process=read(identity)
            if not verify_identity(process):raise RuntimeError('Dead incomplete case; preserve and repair separately')
            while verify_identity(process):time.sleep(10)
        else:
            child=launch(root,action)
            if child.wait():raise RuntimeError('Native funded action failed: '+action)
    result=completed_case(output);cp=read(output/'funded_checkpoint_receipt.json')
    if not result['valid'] or not cp['no_action_continuation_equal'] or sha(output/'native_account_checkpoint.pkl')!=cp['sha256']:
        raise ValueError('Unqualified native action: '+action)
    return result


def run(root):
    pre=read(root/'preflight.json')
    if (not pre['tests_passed'] or sha(Path(pre['tests']))!=pre['tests_sha256'] or
            sha(root/'protocol.json')!=pre['protocol_sha256'] or source_hash(root/'native_engines/train/files')!=pre['source']):
        raise ValueError('Frozen preflight changed')
    status(root,'funded_hold_parity_running',economic_objective_achieved=False)
    hold=case(root,'hold');a=PARENT/'cases/train/control_baseline';b=Path(hold['path'])
    checks={n:read(a/n)==read(b/n) for n in ('equity.json','daily.json','trades.json','minute_nav_evidence.json')}
    write(root/'hold_parity.json',dict(checks=checks,pass_=all(checks.values()),
        parent_completion=completed_case(a)['receipt_sha256'],hold_completion=hold['receipt_sha256']))
    if not all(checks.values()):raise ValueError('No-action probe changed native path')
    status(root,'funded_one_shot_alternatives_running',economic_objective_achieved=False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={action:pool.submit(case,root,action) for action in ('half','exit')}
        cases={action:future.result() for action,future in futures.items()}
    comparison={}
    for action,c in cases.items():
        comparison[action]=dict(completion=c['receipt_sha256'],
            return_delta_pp=c['return_pct']-hold['return_pct'],full_dd_delta_pp=c['full_dd']-hold['full_dd'],
            common_dd_delta_pp=c['common_dd']-hold['common_dd'],acquisition_delta=c['trades']-hold['trades'],
            entry_day_delta=c['days']-hold['days'],
            funded_accounting=contrast(read(b/'trades.json'),read(Path(c['path'])/'trades.json')))
    write(root/'funded_action_comparisons.json',dict(control_completion=hold['receipt_sha256'],comparisons=comparison,
        single_decision_diagnostic=True,general_policy_qualified=False,economic_objective_achieved=False))
    status(root,'awaiting_funded_response_review',economic_objective_achieved=False,automatic_oos_authority=False,
        next='Authenticate entire DD/funding paths and actual action service; continue supported structural holding and meaningful entry work. One decision cannot establish a general policy.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('prepare','run','progress'));p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    if a.phase=='prepare':prepare(a.root)
    elif a.phase=='progress':progress(a.root)
    else:
        try:run(a.root)
        except BaseException as exc:
            status(a.root,'needs_engineering_review',error=repr(exc),economic_objective_achieved=False)
            raise
