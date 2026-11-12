"""Finite native OLR capacity interventions on an immutable historical generation.

Uses existing portfolio replay/auditor; no data, signal or execution-capacity
change and no live activation. Only the explicit implementation allowlist is
copied; unrelated workspace work cannot silently enter the experiment.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.executed_value import digest, signature, write_json

FILES=('strategy_olr/config.py','strategy_olr/research.py','strategy_olr/core/logic.py','strategy_olr/core/capacity.py',
       'strategy_olr/core/allocation.py','strategy_olr/core/core_models.py','strategy_olr/engine.py',
       'backtests/strategies/olr/runner.py','backtests/engine/replay.py',
       'deployment/olr_kalcb/intent_allocator.py','scripts/portfolio_synergy_study.py',
       'deployment/olr_kalcb/runtime.py','deployment/olr_kalcb/session_driver.py',
       'deployment/olr_kalcb/offline_replay.py',
       'backtests/strategies/olr/capacity_qualification.py','scripts/qualify_olr_capacity.py',
       'scripts/olr_capacity_study.py','scripts/olr_capacity_observe.py')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))

def profiles():
    joint={'capacity_joint_allocation':True}
    entry={**joint,'capacity_entry_bars':6}
    return {'control':{}, 'joint':joint,
        'entry3':{**joint,'capacity_entry_bars':3}, 'entry6':entry,
        'exit_cap':{**joint,'capacity_exit_quantile':.5},
        'exit_guard':{**joint,'capacity_exit_guard':True},
        'entry_exit_cap':{**entry,'capacity_exit_quantile':.5},
        'entry_guard':{**entry,'capacity_exit_guard':True},
        'entry_guard_min':{**entry,'capacity_exit_guard':True,'capacity_min_position_pct':.0025},
        'entry6_defer':{**entry,'capacity_defer_minutes':5},
        'entry_guard_defer':{**entry,'capacity_exit_guard':True,'capacity_defer_minutes':5},
        'exit_guard_defer':{**joint,'capacity_exit_guard':True,'capacity_defer_minutes':5},
        'exit_cap_history':{**joint,'capacity_exit_quantile':.5,'capacity_broad_auction_history':True},
        'entry_exit_cap_history':{**entry,'capacity_exit_quantile':.5,'capacity_broad_auction_history':True},
        'entry_guard_history':{**entry,'capacity_exit_guard':True,'capacity_broad_auction_history':True},
        'entry_guard_min_history':{**entry,'capacity_exit_guard':True,'capacity_min_position_pct':.0025,'capacity_broad_auction_history':True},
        'entry_guard_defer_history':{**entry,'capacity_exit_guard':True,'capacity_defer_minutes':5,'capacity_broad_auction_history':True},
        'slice_history':{**joint,'capacity_exit_guard':True,'capacity_exit_guard_mode':'excess_only','capacity_broad_auction_history':True},
        'entry_slice_history':{**entry,'capacity_exit_guard':True,'capacity_exit_guard_mode':'excess_only','capacity_broad_auction_history':True},
        'entry_slice_defer_history':{**entry,'capacity_exit_guard':True,'capacity_exit_guard_mode':'excess_only','capacity_broad_auction_history':True,'capacity_defer_minutes':5}}

def prepare(parent_path, output):
    parent=read(parent_path);manifest=read(parent['manifest']);expected=manifest.pop('manifest_hash')
    if signature(manifest)!=expected:raise ValueError('Changed parent manifest')
    output.mkdir(parents=True,exist_ok=False);root=output/'inputs/files';root.mkdir(parents=True)
    hashes={};changed={}
    for name,sha in manifest['files'].items():
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
        if name in FILES:
            shutil.copy2(REPO/name,target);hashes[name]=digest(target)
            changed[name]={'parent':sha,'new':hashes[name]}
        else:
            os.link(Path(parent['code'])/name,target);hashes[name]=sha
    for name in FILES:
        if name in hashes:continue
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,target)
        hashes[name]=digest(target);changed[name]={'parent':None,'new':hashes[name]}
    base=read(Path(parent['code'])/parent['spec_path'])
    native=copy.deepcopy(base['candidates'].get('cash_control',base['candidates'].get('control')))
    if native is None:raise ValueError('Missing registered control')
    base['candidates']={name:{**copy.deepcopy(native),'quality':False,'decision_rank':False,
                            'olr_capacity':settings} for name,settings in profiles().items()}
    base['initial_mutations']={'recipe':'control'}
    specs={}
    for capital in (10_000_000,20_000_000,50_000_000,90_000_000,100_000_000,110_000_000):
        spec=copy.deepcopy(base);spec['initial_equity']=capital
        spec['configuration']['initial_equity']=capital
        spec['registration_revision']='olr-native-capacity-v4; sourced auction history; bounded deferral; sequential exit slices'
        name=f'research/olr_capacity/study_{capital}_{signature(spec)[:16]}.json'
        if (root/name).exists():
            if read(root/name)!=spec:raise ValueError('Conflicting frozen specification')
        else:write_json(root/name,spec)
        hashes[name]=digest(root/name);specs[capital]=(name,spec)
    program='from kis_core.trading_calendar import get_trading_calendar;get_trading_calendar();from backtests.auto.shared.cache_keys import source_generation_hash;print(source_generation_hash())'
    source=subprocess.check_output([sys.executable,'-B','-c',program],cwd=root,text=True).strip()
    payload={**manifest,'files':hashes,'source_generation_hash':source,
        'evidence':{**manifest['evidence'],'olr_capacity_parent':expected}}
    write_json(output/'inputs/manifest.json',{**payload,'manifest_hash':signature(payload)})
    for capital,(name,spec) in specs.items():
        write_json(output/f'study_{capital}.json',{**parent,**spec,'manifest':str((output/'inputs/manifest.json').resolve()),
            'code':str(root.resolve()),'spec_path':name,'source_generation_hash':source})
    registration={'registered_at_utc':datetime.now(timezone.utc).isoformat(),'source_generation_hash':source,
        'parent':str(parent_path),'changed_files':changed,'profiles':profiles(),
        'research_lineage':'Initial nine ablations retained; follow-up broad history and bounded deferral; final three sequential-slice interventions. All are development trials.',
        'scope':'historical_development_only','objective':'net return at 100m with executable capacity, risk and meaningful frequency',
        'selection':{'baseline_dd':'no worse than contemporaneous control',
            'combined_dd':'no worse than contemporaneous control',
            'return':'beat control in both baseline and combined stress',
            'frequency':'at least 95% of control completed trades in each scenario',
            'support':'changed feasible quantities or exit orders on at least 10 distinct issuer-days',
            'tie_break':'highest combined-stress return, then baseline return',
            'qualification':'10m/20m/50m/90m/110m paired control; full source/execution audit at 100m baseline and stress',
            'failure':'retain incumbent; no automatic relaxation or promotion'},
        'mechanism_metrics':['entry_request_shortfall','late_entry_notional_fraction','turnover',
                             'tiny_ticket_count','cash_clipping','native_quantity_changes','exit_interventions'],
        'qualification_capitals':list(specs),'future_session_completion_gate':None}
    write_json(output/'registration.json',registration)
    print(json.dumps({'prepared':str(output),'source':source}),flush=True)

def run_one(root,capital,recipe,scenario,full=False):
    study=root/f'study_{capital}.json';spec=read(study)
    name=f'{recipe}_{capital}_{scenario}';target=root/('qualified' if full else 'cases')/name
    if (target/'completion.json').exists():return read(target/'result.json')
    if target.exists():raise FileExistsError('Preserving incomplete case: '+str(target))
    log=root/'logs'/((('qualified_' if full else '')+name)+'.log');log.parent.mkdir(exist_ok=True)
    command=[sys.executable,'-B','-u',str(Path(spec['code'])/'scripts/olr_capacity_observe.py'),
        '--study',str(study),'--output',str(target),'--recipe',recipe,'--scenario',scenario]
    if full:command+=['--full-audit']
    print(json.dumps({'started':name,'full_audit':full}),flush=True)
    with log.open('x',encoding='utf-8') as stream:
        run=subprocess.run(command,cwd=spec['code'],stdout=stream,stderr=subprocess.STDOUT,
            env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'})
    write_json(root/'receipts'/((('qualified_' if full else '')+name)+'.json'),{'command':command,'returncode':run.returncode})
    if run.returncode:raise RuntimeError(str(log))
    result=read(target/'result.json');done=read(target/'completion.json')
    assert digest(target/'result.json')==done['result_sha256']
    assert digest(target/'execution_audit.json')==done['audit_sha256']
    assert result['execution_validity_pass'] and not result['terminal_inventory'] and not result['terminal_pending']
    print(json.dumps({'completed':name,**{k:result[k] for k in ('return_pct','drawdown_pct','trades')}}),flush=True)
    return result

def run(root,capital,names,scenarios,workers,full):
    registered=read(root/'registration.json')
    if capital not in registered['qualification_capitals']:raise ValueError('Unregistered capital')
    if any(n not in registered['profiles'] for n in names):raise ValueError('Unregistered profile')
    jobs=[(n,s) for n in names for s in scenarios]
    failures={}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(run_one,root,capital,n,s,full):(n,s) for n,s in jobs}
        for f in as_completed(futures):
            try:f.result()
            except Exception as e:
                failures[str(futures[f])]=str(e);print(json.dumps({'failed':str(futures[f]),'error':str(e)}),flush=True)
    if failures:raise RuntimeError(failures)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','run'))
    p.add_argument('--parent',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--capital',type=int,default=100_000_000);p.add_argument('--profiles',nargs='+',default=list(profiles()))
    p.add_argument('--scenarios',nargs='+',default=['baseline','combined']);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--full-audit',action='store_true');a=p.parse_args()
    if a.command=='prepare':prepare(a.parent,a.output)
    else:run(a.output,a.capital,a.profiles,a.scenarios,a.workers,a.full_audit)
