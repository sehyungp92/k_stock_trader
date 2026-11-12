"""Fixed-policy April--July OLR validation; no search, fitting or live activation."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

for _thread_variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_thread_variable]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'

REPO = Path(__file__).resolve().parents[1]
ROOT = Path.home()/'.codex/research/olr_round1_oos_20261009'
PARENT = Path.home()/'.codex/research/portfolio_source_repair_20261009/oos'
ROUND = REPO/'data/backtests/output/olr/round_1'
WINDOW = {'start':'2026-04-01','end':'2026-07-10'}

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p):
    with Path(p).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
def signature(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,sort_keys=True,default=str)+'\n',encoding='utf-8')
def bound(p): return {'path':str(Path(p).resolve()),'sha256':digest(p)}
def link(a,b):
    b.parent.mkdir(parents=True,exist_ok=True)
    try: os.link(a,b)
    except OSError: shutil.copyfile(a,b)
def source_hash(code):
    program='from kis_core.trading_calendar import get_trading_calendar;get_trading_calendar();from backtests.auto.shared.cache_keys import source_generation_hash;print(source_generation_hash())'
    return subprocess.check_output([sys.executable,'-B','-c',program],cwd=code,text=True).strip()

def freeze():
    """Freeze repaired retained market bytes, original OLR core and fixed dates."""
    ROOT.mkdir(parents=True,exist_ok=False)
    prior=read(PARENT/'validation/study.json');parent=read(prior['manifest']);base=Path(prior['code'])
    assert signature({k:v for k,v in parent.items() if k!='manifest_hash'})==parent['manifest_hash']
    train=read(ROUND/'native_study.json');tm=read(train['manifest']);tc=Path(train['code'])
    code=ROOT/'forward_inputs/files';code.mkdir(parents=True)
    files={};changes=[]
    # Keep the exact selected OLR core. Later portfolio-only optional OLR rules
    # are not part of this evaluation. Execution/audit repairs remain explicit.
    core={n for n in tm['files'] if n.startswith('strategy_olr/') and n.endswith('.py')}
    for n,h in parent['files'].items():
        if '__pycache__' in Path(n).parts or n.endswith('.pyc'): continue
        origin=tc/n if n in core else base/n
        expected=tm['files'][n] if n in core else h
        if digest(origin)!=expected: raise ValueError('Changed frozen dependency: '+n)
        link(origin,code/n);files[n]=expected
        if expected!=h: changes.append({'file':n,'parent':h,'selected_round':expected})
    # The July archive has 34 revisions on May 12. Retain the original,
    # already-audited historical bars through May 12 rather than mix vintages.
    import pandas as pd
    contract=copy.deepcopy(parent['evidence']['shared_market_inputs'])
    tcontract=tm['evidence']['shared_market_inputs'];market_changes=[]
    for symbol,names in contract['intraday_files'].items():
        if len(names)!=1: raise ValueError('Expected single repaired July archive')
        name=names[0];new=pd.read_parquet(code/name)
        historical=pd.concat([pd.read_parquet(tc/n) for n in tcontract['intraday_files'][symbol]],ignore_index=True)
        old=historical[historical.timestamp.dt.strftime('%Y-%m-%d')<='2026-05-12']
        new=pd.concat([old,new[new.timestamp.dt.strftime('%Y-%m-%d')>'2026-05-12']],ignore_index=True)
        # Bound the source evaluator and feature DB to one prior session plus
        # the complete requested window; daily feature history stays intact.
        new=new[new.timestamp.dt.strftime('%Y-%m-%d').between('2026-03-31',WINDOW['end'])]
        new=new.sort_values('timestamp',kind='stable').reset_index(drop=True)
        if new.timestamp.duplicated().any(): raise ValueError('Duplicate source bars')
        (code/name).unlink();new.to_parquet(code/name,index=False);files[name]=digest(code/name)
        market_changes.append({'symbol':symbol,'path':name,'source_parent':parent['files'][name],
            'result':files[name],'sessions':new.timestamp.dt.date.nunique(),'bars':len(new),
            'first':str(new.timestamp.min()),'last':str(new.timestamp.max())})
    # Reuse the reviewed preparation algorithm, changing its declared dates
    # only. The replay, strategy, model and fill algorithms are not retuned.
    n='scripts/portfolio_forward_validation.py';text=(code/n).read_text(encoding='utf-8')
    text=text.replace("'2026-05-29'","'2026-03-31'").replace("'2026-06-01'","'2026-04-01'")
    (code/n).unlink();(code/n).write_text(text,encoding='utf-8');files[n]=digest(code/n)
    n='research/portfolio_current/olr_request.json'
    (code/n).unlink();shutil.copyfile(tc/train['olr_request'],code/n);files[n]=digest(code/n)
    contract.update(train_start='2026-03-31',holdout_start=WINDOW['start'])
    contract.pop('market_input_hash')
    for n in contract['files']:
        if n not in files: raise ValueError('Unbound market input '+n)
        contract['files'][n]=files[n]
    contract['market_input_hash']=signature(contract)
    evidence=copy.deepcopy(parent['evidence']);evidence['shared_market_inputs']=contract
    source=source_hash(code)
    payload={**{k:v for k,v in parent.items() if k!='manifest_hash'},'files':files,'evidence':evidence,'source_generation_hash':source}
    write(ROOT/'forward_inputs/manifest.json',{**payload,'manifest_hash':signature(payload)})
    write(ROOT/'development/study.json',train)
    for n in ['frozen_inference_bridge.json','olr_inference_bridge.json','forward_feature_bridge/comparison.json']:
        dst=ROOT/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PARENT/n,dst)
    write(ROOT/'registration.json',{'registered_utc':datetime.now(timezone.utc).isoformat(),
        'window':WINDOW,'initial_equity':100_000_000,'scenarios':['baseline','combined'],
        'source_round':bound(ROUND/'optimized_config.json'),'selected_config':bound(ROUND/'selected_config.yaml'),
        'training_study':bound(ROUND/'native_study.json'),'repaired_market_parent':bound(prior['manifest']),
        'recipe':train['candidates']['native_entry_recycle2'],'olr_core_restored':changes,
        'market_changes':market_changes,'feature_lookback_session':'2026-03-31',
        'liquidation_only_sessions':['2026-07-09','2026-07-10'],
        'account_start':'flat cash; capacity state starts empty as in saved IS',
        'model_fit_cutoff':'2026-03-10; observation schedule inference only, no fitting',
        'source_generation_hash':source,'outcomes_used_for_selection':False,
        'independent_validation_claim':False,'selection_or_optimization_permitted':False,
        'note':'April-May and June-July are consumed research observations. One continuous evaluation, no segment equity stitching.'})
    print('Frozen complete requested window and exact selected OLR core.',flush=True)

def source_evidence():
    code=ROOT/'forward_inputs/files';sys.path.insert(0,str(code));os.chdir(code)
    from backtests.engine.retained_source import build_evidence
    manifest=read(ROOT/'forward_inputs/manifest.json');contract=manifest['evidence']['shared_market_inputs']
    qpath=build_evidence(code,contract,code/'research/olr_april_july_acceptance')
    q=read(qpath);prior=read(code/manifest['evidence']['qualification_path'])
    q['capacity']=prior['capacity'];q['market_files']=prior['market_files']
    q['session_resolution_report']=prior['session_resolution_report']
    write(qpath,q)
    acceptance=read(qpath.parent/'acceptance.json')
    write(ROOT/'source_preflight.json',acceptance)
    if acceptance['basis_failures']: raise ValueError('Unresolved source basis failures; see source_preflight.json')
    manifest.pop('manifest_hash');manifest['evidence']['qualification_path']=qpath.relative_to(code).as_posix()
    for p in qpath.parent.iterdir(): manifest['files'][p.relative_to(code).as_posix()]=digest(p)
    manifest['manifest_hash']=signature(manifest);write(ROOT/'forward_inputs/manifest.json',manifest)
    print('Full-window price/share basis evidence built.',flush=True)

def prepare_features():
    code=ROOT/'forward_inputs/files'
    subprocess.run([sys.executable,'-B','-u',str(code/'scripts/portfolio_forward_validation.py'),
        'prepare-features','--root',str(ROOT)],cwd=code,check=True)

def predict():
    code=ROOT/'forward_inputs/files'
    subprocess.run([sys.executable,'-B','-u',str(code/'scripts/portfolio_forward_validation.py'),
        'predict','--root',str(ROOT)],cwd=code,check=True)

def freeze_replay():
    """Bind new dated snapshots and one unchanged recipe to the native evaluator."""
    code=ROOT/'forward_inputs/files';sys.path.insert(0,str(code));os.chdir(code)
    from dataclasses import asdict, replace
    from datetime import date
    from strategy_olr.config import OLRConfig
    from scripts.kalcb_executed_value_study import connect, load_day
    from scripts.kalcb_executed_value_replay import verify_policy, timed_policy
    from strategy_common.source_quality import complete_feature_bars, POLICY
    from strategy_common.sector_intraday import AFTERNOON_CUTOFF
    from kis_core.trading_calendar import get_trading_calendar
    import yaml
    prior=read(ROOT/'forward_inputs/manifest.json');train=read(ROOT/'development/study.json')
    parent=read(PARENT/'validation/study.json')
    pm,ks=verify_policy(ROOT/'forward_policies',train['kalcb_policy']);timed_policy(ROOT/'forward_policies',pm,ks)
    days=sorted(map(str,ks));calendar=get_trading_calendar()
    from datetime import timedelta
    day=date.fromisoformat(WINDOW['start']);expected=[]
    while day<=date.fromisoformat(WINDOW['end']):
        if calendar.is_trading_day(day):expected.append(str(day))
        day+=timedelta(days=1)
    if days!=expected:raise ValueError('Replay does not cover every requested exchange session')
    snapshots=read(ROOT/'forward_olr_snapshots.json')
    if [s['trade_date'] for s in snapshots]!=days[:-2]:raise ValueError('Unexpected OLR entry/tail horizon')
    symbols=tuple(str(s).zfill(6) for s in yaml.safe_load((code/'config/universe_103.yaml').read_text())['symbols'])
    db=ROOT/'forward_features/market.sqlite';dbhash=digest(db);connection=connect({'database':str(db)})
    for snap in snapshots:
        day=date.fromisoformat(snap['trade_date']);rows={(day,s):[] for s in symbols}
        for bar in load_day(connection,str(day),dbhash,set(symbols)):rows[day,bar.symbol].append(bar)
        _,exclusions=complete_feature_bars(rows,AFTERNOON_CUTOFF)
        excluded=[s.split(':')[0] for s in exclusions]
        if set(excluded)&{c['symbol'] for c in snap['candidates']}:raise ValueError('Incomplete feature prefix was admitted')
        for key,value in [('feature_prefix_policy',POLICY),('feature_prefix_exclusions',excluded)]:
            if key in snap['metadata'] and snap['metadata'][key]!=value:raise ValueError('Incorrect feature prefix annotation')
            snap['metadata'][key]=value
    connection.close()
    write(ROOT/'annotated_olr_snapshots.json',snapshots)
    target=ROOT/'validation/inputs/files';target.mkdir(parents=True,exist_ok=False);files={}
    def bind(origin,name):
        link(Path(origin),target/name);files[name]=digest(origin)
    for name,h in prior['files'].items():
        link(code/name,target/name);files[name]=h
    pp='research/olr_april_july/policies'
    for p in (ROOT/'forward_policies').iterdir():
        if p.is_file():bind(p,pp+'/'+p.name)
    fp='research/olr_april_july/features'
    for p in (ROOT/'forward_features').rglob('*'):
        if p.is_file():bind(p,fp+'/'+p.relative_to(ROOT/'forward_features').as_posix())
    sp='research/olr_april_july/olr_snapshots.json';bind(ROOT/'annotated_olr_snapshots.json',sp)
    bind(Path(__file__),'research/olr_april_july/evaluation_program.py')
    bind(ROOT/'registration.json','research/olr_april_july/registration.json')
    spec=copy.deepcopy(read(Path(train['code'])/train['spec_path']))
    recipe=copy.deepcopy(train['candidates']['native_entry_recycle2'])
    spec.update(initial_equity=100_000_000,candidates={'native_entry_recycle2':recipe},
        initial_mutations={'recipe':'native_entry_recycle2'},kalcb_policies=pp,olr_snapshots=sp,
        database=fp+'/market.sqlite',database_hash=dbhash,corporate_actions=parent['corporate_actions'],
        scope='fixed_current_round1_continuous_april_july_evaluation',
        limitations=['Consumed historical validation; no independent-performance claim.','Saved modeled execution costs and liquidity assumptions retained.'],
        forward_validation={'window':WINDOW,'model_fit_reused':'2026-03-10','no_entry_tail':days[-2:],
            'registration':bound(ROOT/'registration.json'),'new_fitting':False,'new_rule_selection':False})
    out=ROOT/'validation';configuration=spec['configuration']
    configuration.update(start=WINDOW['start'],end=WINDOW['end'],date_range=WINDOW,
        initial_equity=100_000_000,data_snapshot_end=WINDOW['end'],input_manifest_path=str(out/'inputs/manifest.json'),
        market_input_hash=prior['evidence']['shared_market_inputs']['market_input_hash'])
    request=read(target/spec['olr_request'])
    resolved=asdict(replace(OLRConfig.from_mapping(request['config'],request['mutations']),slippage_bps=5,
        tax_bps_on_sell=20,**recipe['olr_capacity']))
    if json.loads(json.dumps(resolved,default=str))!=read(ROUND/'resolved_strategy_config.json'):
        raise ValueError('Selected OLR configuration changed')
    sn='research/olr_april_july/study.json';write(target/sn,spec);files[sn]=digest(target/sn)
    evidence=copy.deepcopy(prior['evidence']);evidence['portfolio_current_study']=sn
    evidence['executed_value_policy']['policies_manifest']=pp+'/policies_manifest.json'
    evidence['executed_value_policy']['prepared']=fp+'/checkpoint/prepared.json'
    source=source_hash(target)
    payload={**{k:v for k,v in prior.items() if k!='manifest_hash'},'files':files,'evidence':evidence,'source_generation_hash':source}
    write(out/'inputs/manifest.json',{**payload,'manifest_hash':signature(payload)})
    write(out/'study.json',{**spec,'code':str(target),'manifest':str(out/'inputs/manifest.json'),
        'source_generation_hash':source,'spec_path':sn})
    write(ROOT/'replay_preflight.json',{'pass':True,'window':WINDOW,'sessions':len(days),
        'entry_sessions':len(snapshots),'tail':days[-2:],'selected_config_matches':True,
        'study':bound(out/'study.json'),'manifest':bound(out/'inputs/manifest.json')})
    print(json.dumps({'replay_ready':True,'sessions':len(days),'entry_sessions':len(snapshots)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','source-evidence','prepare-features','predict','freeze-replay'])
    args=p.parse_args();{'freeze':freeze,'source-evidence':source_evidence,'prepare-features':prepare_features,'predict':predict,'freeze-replay':freeze_replay}[args.stage]()
