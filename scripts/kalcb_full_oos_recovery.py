"""Registered KALCB recovery on the complete April 1--July 10 OOS window.

Preparation preserves the qualified economic core and source facts. Native
replays, supervised targets, and scheduled forecasts remain separate stages.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
ROOT=Path.home()/'.codex/research/kalcb_full_oos_recovery_20261009/v3'
PRIOR=Path.home()/'.codex/research/kalcb_round1_oos_20261009/parser_bound'
FULL=Path.home()/'.codex/research/olr_round1_oos_20261009'
DIAG=Path.home()/'.codex/research/kalcb_round1_oos_diagnosis_20261009'
POLICY='interaction_cost_margin'
START='2026-04-01'
END='2026-07-10'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8')
def bound(path):return {'path':str(Path(path).resolve()),'sha256':digest(path)}
def jsonl(path,rows):
    with Path(path).open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,sort_keys=True,default=str,allow_nan=False)+'\n')
def rows(path):
    with Path(path).open(encoding='utf-8') as f:return [json.loads(line) for line in f]
def link(source,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    try:os.link(source,target)
    except OSError:shutil.copy2(source,target)
def env(code):return {**os.environ,'PYTHONPATH':str(code),'PYTHONDONTWRITEBYTECODE':'1','PYTHONUTF8':'1',
    'PYTHONIOENCODING':'utf-8','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
    'NUMEXPR_NUM_THREADS':'1','NUMEXPR_MAX_THREADS':'1'}
def source_hash(code):
    return subprocess.check_output([sys.executable,'-B','-c',
        'from backtests.auto.shared.cache_keys import source_generation_hash;print(source_generation_hash())'],
        cwd=code,env=env(code),text=True).strip()
def frozen_imports():
    code=PRIOR/'inputs/files'
    sys.path.insert(0,str(code))
    import scripts
    scripts.__path__.append(str(REPO/'scripts'))
    return code

def prepare(root):
    code=frozen_imports()
    from scripts.portfolio_frozen_inference import infer
    from scripts.kalcb_executed_value_replay import verify_policy,timed_policy
    from scripts.kalcb_executed_value_freeze import freeze
    from backtests.strategies.kalcb.executed_value import signature
    from backtests.core.input_manifest import InputManifest
    from backtests.engine.retained_source import verify_acceptance
    root.mkdir(parents=True,exist_ok=False)
    selected=read(REPO/'config/research/kalcb_intraday_advantage_20261007/selected_research.json')
    features=FULL/'forward_features';completion=read(features/'completion.json')
    for item in [completion['database'],completion['market_manifest'],*completion['stages'].values()]:
        if digest(item['path'])!=item['sha256']:raise ValueError('Changed complete-window feature input')
    sessions=[d for d in read(features/'checkpoint/prepared.json')['sessions'] if START<=d<=END]
    if len(sessions)!=69 or sessions[0]!=START or sessions[-1]!=END:raise ValueError('Expected all 69 proper OOS sessions')
    if read(FULL/'source_preflight.json')['basis_failures']:raise ValueError('Unaccepted full-period source basis')
    write(root/'protocol.json',{'version':'kalcb-full-oos-recovery-v1','registered_utc':datetime.now(timezone.utc).isoformat(),
        'is':{'start':'2025-08-07','end':'2026-03-31','sessions':156},
        'oos':{'start':START,'end':END,'sessions':69},'capital_krw':100_000_000,
        'canonical_profile':bound(REPO/'config/research/kalcb_intraday_advantage_20261007/selected_research.json'),
        'canonical_config':bound(selected['configuration']),'economic_source':bound(PRIOR/'inputs/manifest.json'),
        'full_feature_completion':bound(features/'completion.json'),'full_source_preflight':bound(FULL/'source_preflight.json'),
        'user_period_correction':'OOS starts April 1, not June 1. All sessions including July 9 and 10 permit entries.',
        'independence':'Consumed development observations; no renewed independent OOS claim.',
        'soft_drawdown_preferences':{'baseline':8,'combined':10,'hard_reject':False},
        'sequence':['immutable full-period control','paired removal and local perturbation',
            'terminal and scheduled causal model/calibration update controls','target-aligned and targeted interaction experiments','full finalist qualification'],
        'excluded_mutations':{'hard_stop_restored':'Previous native physical failures; no accepted economics until separately repaired and qualified.'},
        'canonical_promotion':'Only a fully qualified research improvement may be published; no live enablement.',
        'program':bound(__file__)})
    policies=root/'policies/frozen';policies.mkdir(parents=True)
    parent_dir=code/'research/kalcb_executed_value/policies'
    for p in parent_dir.iterdir():
        if p.is_file():shutil.copy2(p,policies/p.name)
    # The prior forward bundle retained forecasts but omitted a feasibility
    # model. Recover only the exact digest recorded in its training proof.
    for proof in read(FULL/'frozen_inference_bridge.json')['models']:
        p=Path(proof['source']['path'])
        if digest(p)!=proof['source']['sha256']:raise ValueError('Changed model evidence')
        name=('opening_' if proof['kind']=='opening' else '')+proof['training_proof']['model_path']
        shutil.copy2(p,policies/name)
    opening=features/'continuation/opportunities.jsonl';timed=features/'checkpoint/opportunities.jsonl'
    fit_directory=Path(read(FULL/'frozen_inference_bridge.json')['models'][0]['source']['path']).parent
    result=infer(fit_directory,rows(opening),rows(timed),read(features/'plan.json'),read(features/'checkpoint_plan.json'),
        sessions,digest(opening),tail=(),timed_source=digest(timed))
    prior_manifest,prior_base=verify_policy(parent_dir,POLICY);prior_timed=timed_policy(parent_dir,prior_manifest,prior_base)
    def choices(s):return {c.symbol:(c.metadata.get('action_quantity_scale',1),c.selection_score) for c in s.candidates}
    comparisons=0
    for day,old in prior_base.items():
        for a,b in [(old,result['snapshots'][day]),(prior_timed.get(day,[old])[-1],result['updates'].get(day,[result['snapshots'][day]])[-1])]:
            ca,cb=choices(a),choices(b)
            if ca.keys()!=cb.keys() or any(ca[s][0]!=cb[s][0] or abs(ca[s][1]-cb[s][1])>1e-8 for s in ca):
                raise ValueError('June-July frozen selection drift')
            comparisons+=1
    write(policies/(POLICY+'.snapshots.json'),[s.to_json_dict() for s in result['snapshots'].values()])
    jsonl(policies/(POLICY+'.admission.jsonl'),result['admissions'])
    jsonl(policies/'timed.snapshots.jsonl',(s.to_json_dict() for values in result['updates'].values() for s in values))
    jsonl(policies/'timed.admission.jsonl',result['timed_admissions'])
    write(policies/'forward_lineage.json',{'scope':'Full April-July, frozen model reconstruction, all days eligible.',
        'features':bound(features/'completion.json'),'protocol':bound(root/'protocol.json'),
        'models':result['model_proofs'],'calibrations':result['calibrations'],'june_july_day_phase_comparisons':comparisons})
    manifest={**prior_manifest,'evaluation_from':START,'evaluation_through':END,
        'scope':'full_april_july_research_control_all_69_sessions'}
    manifest['files']={p.name:digest(p) for p in policies.iterdir() if p.is_file() and p.name not in ('policies_manifest.json','completion.json')}
    write(policies/'policies_manifest.json',manifest);write(policies/'completion.json',{'complete':True})
    m,s=verify_policy(policies,POLICY);timed_policy(policies,m,s)
    plan=read(PRIOR/'plan.json');old_db=plan['database'];old_audit=plan['audit_configuration']
    context=read(old_audit)
    context.update(start=START,end=END,holdout_start=START,data_snapshot_end=END,date_range={'start':START,'end':END},
        initial_mutations=read(selected['configuration']),baseline={'name':'full_oos_frozen_control','evaluation_start':START,'evaluation_end':END})
    write(root/'audit_configuration.json',context)
    plan.update(start=START,end=END,database=completion['database']['path'],audit_configuration=str(root/'audit_configuration.json'))
    inputs=set(plan['input_sha256'])-{old_db,old_audit}
    inputs.update((plan['database'],plan['audit_configuration']))
    plan['input_sha256']={p:digest(p) for p in sorted(inputs)}
    write(root/'plan.json',plan)
    parent_path=FULL/'forward_inputs/manifest.json';parent=read(parent_path);identity=parent.pop('manifest_hash')
    if signature(parent)!=identity:raise ValueError('Changed full-period market generation')
    target=root/'market_parent/files';files={}
    for name,h in parent['files'].items():
        if (name.endswith('.py') and not name.startswith('research/')) or name.startswith(('tests/','research/kalcb_executed_value/')):continue
        p=parent_path.parent/'files'/name
        if digest(p)!=h:raise ValueError('Changed full-period input '+name)
        link(p,target/name);files[name]=h
    filtered={**parent,'files':files}
    write(root/'market_parent/manifest.json',{**filtered,'manifest_hash':signature(filtered)})
    freeze(root/'market_parent/manifest.json',code,policies,features/'checkpoint',[],root/'plan.json',root/'inputs')
    generation=InputManifest(root/'inputs/manifest.json')
    qualification=read(generation.root/generation.payload['evidence']['qualification_path'])
    acceptance=verify_acceptance(lambda n:read(generation.root/n),qualification,generation=generation)
    if acceptance['basis_failures']:raise ValueError('Source acceptance failure')
    # Validate economic parity file by file; unrelated source packages do not
    # imply a change to the trading or fill algorithms.
    economic=read(PRIOR/'economic_code_parity.json')
    economic_files=[n for n in read(PRIOR/'inputs/manifest.json')['files'] if n.endswith('.py') and n.startswith(('strategy_kalcb/','backtests/engine/','strategy_common/'))]
    for n in economic_files:
        if digest(code/n)!=digest(generation.root/n):raise ValueError('Changed economic source '+n)
    write(root/'preparation.json',{'passed':True,'sessions':sessions,'source_manifest':bound(root/'inputs/manifest.json'),
        'source_generation_hash':generation.payload['source_generation_hash'],'economic_file_comparisons':len(economic_files),
        'june_july_day_phase_comparisons':comparisons,'opening_admissions':len(result['admissions']),
        'checkpoint_admissions':len(result['timed_admissions']),'feature_completion':bound(features/'completion.json'),
        'source_acceptance_passed':True,'is_reference':bound(selected['result_paths']['baseline']),
        'protocol':bound(root/'protocol.json'),'predecessor_economic_parity':bound(PRIOR/'economic_code_parity.json')})
    print(json.dumps({'prepared':str(root),'sessions':len(sessions),'opening':len(result['admissions']),
        'checkpoint':len(result['timed_admissions']),'source':generation.payload['source_generation_hash']}),flush=True)

def control(root,scenarios):
    from concurrent.futures import ThreadPoolExecutor
    if not read(root/'preparation.json')['passed']:raise ValueError('Incomplete preparation')
    code=root/'inputs/files';source=read(root/'preparation.json')['source_generation_hash']
    def run(s):
        out=root/'control'/s;log=root/'logs'/('control_'+s+'.log');log.parent.mkdir(exist_ok=True)
        if not (out/'completion.json').exists():
            if out.exists():raise ValueError('Preserve incomplete native case')
            cmd=[sys.executable,'-B','-u','-m','scripts.kalcb_executed_value_replay','--plan',str(root/'plan.json'),
                '--policies',str(code/'research/kalcb_executed_value/policies'),'--policy',POLICY,
                '--scenario',s,'--audit-manifest',str(root/'inputs/manifest.json'),'--output',str(out)]
            with log.open('x',encoding='utf-8') as f:rc=subprocess.run(cmd,cwd=code,env=env(code),stdout=f,stderr=subprocess.STDOUT).returncode
            write(root/'control'/(s+'.receipt.json'),{'command':cmd,'returncode':rc,'log':str(log)})
            if rc:raise ValueError('Native control failed; '+str(log))
        r=read(out/'result.json')
        if r['source_generation_hash']!=source or r['audit_status']!='passed' or not r['execution_validity_pass']:raise ValueError('Unqualified full-period control '+s)
        daily=read(out/'daily.json');trades=read(out/'trades.json')
        if len(daily)!=69 or r['terminal_inventory'] or r['terminal_pending'] or r['residual_sessions']:raise ValueError('Control calendar or inventory failure')
        if abs(sum(t['net_pnl'] for t in trades)-(daily[-1]['equity']-100_000_000))>.01:raise ValueError('Control cash mismatch')
        print(json.dumps({'scenario':s,**{k:r[k] for k in ['return_pct','drawdown_pct','trades','active_days','audit_status']}}),flush=True)
        return r
    with ThreadPoolExecutor(max_workers=2) as pool:results=dict(zip(scenarios,pool.map(run,scenarios)))
    write(root/'control_summary.json',{'scope':'Continuous 69-session full OOS account, no stitching.',
        'results':{s:{k:r[k] for k in ['return_pct','drawdown_pct','trades','active_days','audit_status']} for s,r in results.items()}})

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','control'])
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--scenarios',nargs='+',default=['baseline','participation','cost','combined'])
    a=p.parse_args()
    if a.stage=='prepare':prepare(a.root)
    elif a.stage=='control':control(a.root,a.scenarios)

if __name__=='__main__':main()
