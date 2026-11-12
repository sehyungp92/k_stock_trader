"""Frozen-model portfolio validation on reserved dates, with no OOS fitting."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.portfolio_targeted_validation import ROOT,read,bound
from backtests.strategies.kalcb.executed_value import digest,signature,write_json


def freeze_market(root):
    """Freeze source facts before model inference or portfolio outcomes."""
    from scripts.prepare_olr_kalcb_integrity_inputs import shared_market_contract
    from backtests.engine.retained_source import build_evidence
    development=read(root/'development/study.json');parent=read(development['manifest'])
    residual=read(root/'source_residuals/resolution.json')
    if not residual['all_residuals_resolved']:raise ValueError('Unresolved OOS source residuals')
    protocol=read(root/'protocol.json');window=protocol['reserved_validation']
    spec={'market_inputs':{'data_root':'data/kis_intraday_parquet','daily_data_root':'data/krx_daily_parquet',
        'snapshot_end':'20260710','daily_snapshot_end':'20260710','timeframe':'5m',
        'train_start':'2026-05-29','holdout_start':window['start'],'data_end':window['end'],
        'universe':'config/universe_103.yaml','sector_map_path':'config/olr/sector_map.yaml'}}
    contract,paths=shared_market_contract(REPO,spec)
    destination=root/'forward_inputs';destination.mkdir(parents=True,exist_ok=False)
    target=destination/'files';target.mkdir();files={}
    overlays=('scripts/portfolio_forward_validation.py','scripts/portfolio_targeted_validation.py',
              'scripts/kalcb_executed_value_study.py','backtests/engine/retained_source.py')
    def bind(source,name,copy_file=False):
        out=target/name;out.parent.mkdir(parents=True,exist_ok=True)
        if out.exists():raise FileExistsError('Duplicate immutable binding '+name)
        if copy_file:shutil.copyfile(source,out)
        else:
            try:os.link(source,out)
            except OSError:shutil.copyfile(source,out)
        files[name]=digest(out)
    replacement_names={p.relative_to(REPO).as_posix() for p in paths}|set(overlays)
    for name in parent['files']:
        if name in replacement_names:continue
        if (name.startswith(('data/kis_intraday_parquet/','data/krx_daily_parquet/'))
                and name != 'data/krx_daily_parquet/tables/sector_map.parquet'):continue
        bind(Path(development['code'])/name,name)
    for p in paths:bind(p,p.relative_to(REPO).as_posix(),True)
    sector_table = 'data/krx_daily_parquet/tables/sector_map.parquet'
    if sector_table not in files:raise ValueError('Frozen KALCB sector table is required')
    contract.pop('market_input_hash')
    contract['files'][sector_table] = files[sector_table]
    contract['market_input_hash'] = signature(contract)
    for name in overlays:bind(REPO/name,name,True)
    # Every sparse disposition binds the archived observations and fresh raw
    # provider receipts. No missing interval is filled or daily volume assigned.
    resolution=read(root/'source_resolution/resolution.json');dispositions=[];source_files={}
    for folder in ('source_resolution','source_residuals'):
        for p in (root/folder).rglob('*'):
            if p.is_file():
                name='research/forward_sources/'+str(p.relative_to(root)).replace('\\','/')
                bind(p,name,True);source_files[name]=files[name]
    for row in resolution['sessions']:
        dispositions.append({'symbol':row['symbol'],'date':row['date'],
            'disposition':'archived_sparse_daily_reconciled' if row['disposition']=='sparse_archive_exact_daily_ohlcv_conservation'
                          else 'archived_sparse_daily_close_residual_reconciled',
            'basis':'Exact archived daily OHLCV, or existing exact original-price turnover residual test; fresh provider confirms absent minutes.',
            'residual_added_to_bars':False})
    resolution_name='research/forward_sources/session_resolution.json'
    write_json(target/resolution_name,{'version':'session-recovery-dispositions-v1','all_original_flags_accounted':True,
        'source_files':source_files,'sessions':dispositions,'independent_intraday_completeness_verified':False})
    files[resolution_name]=digest(target/resolution_name)
    # Include resolution inside the market identity before acceptance is built.
    contract.pop('market_input_hash');contract['session_resolution_report']=resolution_name
    contract['files'].update({resolution_name:files[resolution_name],**source_files})
    contract['market_input_hash']=signature(contract)
    qp=build_evidence(target,contract,target/'research/forward_acceptance')
    qualified=read(qp);oldq=read(Path(development['code'])/parent['evidence']['qualification_path'])
    qualified['market_files']=[{'symbol':s,'path':n,'timeframe':'5m','parser_version':'completed_ohlcv_parquet_v1'}
                              for s,names in contract['intraday_files'].items() for n in names]
    qualified['capacity']=oldq['capacity'];qualified['session_resolution_report']=resolution_name
    write_json(qp,qualified)
    for p in qp.parent.iterdir():files[p.relative_to(target).as_posix()]=digest(p)
    evidence={**parent['evidence'],'shared_market_inputs':contract,'qualification_path':qp.relative_to(target).as_posix(),
        'reserved_validation':{'protocol':bound(root/'protocol.json'),'start':window['start'],'end':window['end'],
            'source_resolution':bound(root/'source_residuals/resolution.json'),'performance_opened':False}}
    command='from kis_core.trading_calendar import get_trading_calendar;get_trading_calendar();from backtests.auto.shared.cache_keys import source_generation_hash;print(source_generation_hash())'
    generation=subprocess.check_output([sys.executable,'-B','-c',command],cwd=target,text=True).strip()
    payload={'version':parent['version'],'source_root':str(REPO),'source_generation_hash':generation,'files':files,'evidence':evidence}
    write_json(destination/'manifest.json',{**payload,'manifest_hash':signature(payload)})
    write_json(root/'forward_market.json',{'manifest':bound(destination/'manifest.json'),'source_generation_hash':generation,
        'source_only':True,'strategy_outcomes_opened':False,'session_resolution_count':len(dispositions)})
    print(json.dumps({'forward_market':str(destination/'manifest.json'),'source_generation_hash':generation}),flush=True)


def prepare_features(root):
    """Causal feature preparation; no OOS labels or target proxy is produced."""
    import pandas as pd
    from backtests.core.input_manifest import InputManifest
    from scripts import kalcb_executed_value_study as initial
    from scripts import kalcb_action_value_study as action
    from scripts import kalcb_continuation_study as continuation
    from scripts import kalcb_checkpoint_prepare as checkpoint
    generation=InputManifest(root/'forward_inputs/manifest.json')
    contract=generation.payload['evidence']['shared_market_inputs']
    destination=root/'forward_features';destination.mkdir(parents=True,exist_ok=False)
    database=destination/'market.sqlite';conn=sqlite3.connect(database)
    conn.execute('CREATE TABLE bars (day TEXT,symbol TEXT,ts TEXT,o REAL,h REAL,l REAL,c REAL,v REAL)')
    for symbol,names in contract['intraday_files'].items():
        for name in names:
            with generation.open_verified(generation.root/name) as f:frame=pd.read_parquet(f)
            stamps=pd.to_datetime(frame.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
            frame=frame.assign(timestamp=stamps,day=stamps.dt.strftime('%Y-%m-%d'))
            frame=frame[frame.day.between('2026-05-29','2026-07-10')]
            conn.executemany('INSERT INTO bars VALUES (?,?,?,?,?,?,?,?)',
                ((r.day,symbol,r.timestamp.isoformat(),r.open,r.high,r.low,r.close,r.volume) for r in frame.itertuples()))
    conn.execute('CREATE UNIQUE INDEX bars_identity ON bars(day,symbol,ts)');conn.commit();conn.close()
    lineage=generation.root/'research/kalcb_executed_value/policies'
    original_path=Path(read(lineage/'opening_core_request.json')['arguments']['plan'])
    original=read(original_path)
    contracts=destination/'feature_contract';contracts.mkdir()
    shutil.copyfile(original_path,contracts/'opening_plan.json')
    def input_for(name):
        path=original[name];sha=original['input_sha256'][path]
        if digest(path)!=sha:raise ValueError('Changed original feature input '+name)
        candidate=contracts/(name+'_'+Path(path).name);shutil.copyfile(path,candidate)
        return str(candidate)
    prepared=generation.root/'research/kalcb_executed_value/additional_evidence_9'
    example=json.loads((prepared/'opportunities.jsonl').open(encoding='utf-8').readline())
    plan={**original,'start':'2026-05-29','end':'2026-07-10','database':str(database),
          'market_manifest':str(generation.path),'forward_feature_only':True,
          'forward_guard_schema':list(example['guard_features']),
          **{n:input_for(n) for n in ('guard_features','seed','q3','config')},
          'universe':str(generation.root/contract['universe'])}
    plan['input_sha256']={str(database):digest(database)}
    def stage(name,function,*args):
        folder=destination/name;folder.mkdir()
        function(*args,folder)
        write_json(folder/'completion.json',{'complete':True,'feature_only':True})
        return folder
    base=stage('opening',initial.prepare,plan)
    plan['pipeline']=str(base/'causal_context.csv')
    # Action feature preparation requires the original feature-snapshot sizing
    # at 100m, independently of the physical portfolio's 25m capital.
    actions=stage('actions',action.prepare,plan,base)
    cont=stage('continuation',continuation.prepare,plan,actions)
    cp_plan_path=Path(read(lineage/'request.json')['arguments']['plan']);cp_plan=read(cp_plan_path)
    if digest(cp_plan['config'])!=cp_plan['input_sha256'][cp_plan['config']]:raise ValueError('Changed checkpoint feature config')
    shutil.copyfile(cp_plan_path,contracts/'checkpoint_plan.json')
    source_cfg=read(cp_plan['config'])
    cp=destination/'checkpoint_config.json';write_json(cp,source_cfg)
    checkpoint_plan={**plan,'config':str(cp)}
    timed=stage('checkpoint',checkpoint.prepare,checkpoint_plan,actions)
    write_json(destination/'plan.json',plan);write_json(destination/'checkpoint_plan.json',checkpoint_plan)
    write_json(destination/'completion.json',{'complete':True,'database':bound(database),'policy_outcomes_opened':False,
        'stages':{n:bound(destination/n/'prepared.json') for n in ('opening','actions','continuation','checkpoint')},
        'market_manifest':bound(generation.path),'program':bound(Path(__file__))})
    print(json.dumps({'forward_features_complete':True}),flush=True)


def feature_bridge(root):
    """Reproduce past rich features using the same target-free preparation path."""
    from scripts import kalcb_executed_value_study as initial
    from scripts import kalcb_action_value_study as action
    from scripts import kalcb_continuation_study as continuation
    from backtests.strategies.kalcb.continuation_value import COLUMNS
    from backtests.strategies.kalcb import executed_value as ev
    import math
    study=read(root/'development/study.json');policies=Path(study['code'])/study['kalcb_policies']
    original=read(Path(read(policies/'opening_core_request.json')['arguments']['plan']))
    original_prepared=policies.parent/'additional_evidence_9'
    example=json.loads((original_prepared/'opportunities.jsonl').open(encoding='utf-8').readline())
    plan={**original,'start':'2026-03-09','end':'2026-03-11','forward_feature_only':True,
          'forward_guard_schema':list(example['guard_features'])}
    destination=root/'forward_feature_bridge';destination.mkdir(parents=True,exist_ok=False)
    base=destination/'opening';base.mkdir();initial.prepare(plan,base)
    plan['pipeline']=str(base/'causal_context.csv')
    actions=destination/'actions';actions.mkdir();action.prepare(plan,base,actions)
    cont=destination/'continuation';cont.mkdir();continuation.prepare(plan,actions,cont)
    old={r['opportunity_id']:r for r in ev.read_jsonl(original_prepared/'opportunities.jsonl')
         if '2026-03-10'<=r['trade_date']<='2026-03-11'}
    new={r['opportunity_id']:r for r in ev.read_jsonl(cont/'opportunities.jsonl')
         if '2026-03-10'<=r['trade_date']<='2026-03-11'}
    errors=[];count=0
    if old.keys()!=new.keys():errors.append({'population_missing':sorted(old.keys()-new.keys()),'population_added':sorted(new.keys()-old.keys())})
    for key in old.keys()&new.keys():
        for field in ('sector', 'guard_at_rank1'):
            if old[key].get(field) != new[key].get(field):
                errors.append({'identity':key,'field':field,'saved':old[key].get(field),'rebuilt':new[key].get(field)})
        for feature in COLUMNS:
            a,b=old[key]['features'].get(feature),new[key]['features'].get(feature);count+=1
            if a is None or b is None:
                same=a is None and b is None
            else:same=math.isclose(float(a),float(b),rel_tol=1e-8,abs_tol=1e-8)
            if not same:errors.append({'identity':key,'feature':feature,'saved':a,'rebuilt':b})
        if old[key]['action']['quantity']!=new[key]['action']['quantity']:
            errors.append({'identity':key,'action_quantity_changed':True})
    result={'pass':not errors,'comparisons':count,'opportunities':len(old),'errors':errors,
            'categorical_and_admission_comparisons':2*len(old),
            'scope':'Target-free rich feature preparation on March 10–11, with March 9 prior-session warm-up.',
            'program':bound(Path(__file__)),'old_prepared':bound(original_prepared/'prepared.json')}
    write_json(destination/'comparison.json',result)
    print(json.dumps({'feature_bridge_pass':not errors,'comparisons':count,'errors':len(errors)}),flush=True)
    if errors:raise ValueError('Target-free feature generation differs from the saved model inputs')


def predict(root):
    from scripts.portfolio_frozen_inference import infer,FIT
    from scripts.portfolio_forward_olr import build as olr_build
    from scripts.kalcb_executed_value_replay import verify_policy,timed_policy
    from backtests.strategies.kalcb import executed_value as ev
    from datetime import date
    for p in ('frozen_inference_bridge.json','olr_inference_bridge.json','forward_feature_bridge/comparison.json'):
        if not read(root/p)['pass']:raise ValueError('Required reconstruction bridge failed: '+p)
    completion=read(root/'forward_features/completion.json')
    for n,source in completion['stages'].items():
        if digest(source['path'])!=source['sha256']:raise ValueError('Changed prepared features')
    study=read(root/'development/study.json');original=Path(study['code'])/study['kalcb_policies']
    features=root/'forward_features';cont=features/'continuation';cp=features/'checkpoint'
    meta=read(cp/'prepared.json');sessions=[d for d in meta['sessions'] if d>='2026-06-01']
    result=infer(original,ev.read_jsonl(cont/'opportunities.jsonl'),ev.read_jsonl(cp/'opportunities.jsonl'),
        read(features/'plan.json'),read(features/'checkpoint_plan.json'),sessions,
        digest(cont/'opportunities.jsonl'),tail=sessions[-2:],timed_source=digest(cp/'opportunities.jsonl'))
    output=root/'forward_policies';output.mkdir(parents=True,exist_ok=False)
    for p in original.iterdir():
        if p.is_file():shutil.copyfile(p,output/p.name)
    name=study['kalcb_policy']
    write_json(output/(name+'.snapshots.json'),[s.to_json_dict() for s in result['snapshots'].values()])
    def jsonl(path,rows,append=False):
        with path.open('a' if append else 'w',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps(row,sort_keys=True,default=str,allow_nan=False)+'\n')
    jsonl(output/(name+'.admission.jsonl'),result['admissions'])
    jsonl(output/'timed.snapshots.jsonl',(s.to_json_dict() for daily in result['updates'].values() for s in daily))
    jsonl(output/'timed.admission.jsonl',result['timed_admissions'])
    for prefix,key,model in [('opening_core_','core','cv_interaction_calibrated_offset_all'),
                             ('opening_','new','cv_interaction_calibrated_support_offset')]:
        jsonl(output/(prefix+'calibrated_forecasts.jsonl'),({**v,'model':model} for v in result[key].values()),True)
        jsonl(output/(prefix+'forecasts.jsonl'),({**v,'model':'cv_interaction'} for v in result['raw'].values()),True)
        chronology=read(output/(prefix+'chronology.json'))
        for c in chronology['calibration']:
            if c['valid_from']==FIT:c.update(valid_through=sessions[-1],validation_reuse='Frozen March calibration; no April–July labels fitted')
        write_json(output/(prefix+'chronology.json'),chronology)
    proofs=[]
    for i,proof in enumerate(result['model_proofs']):
        source=Path(proof['source']['path']);filename='forward_model_'+str(i)+'.joblib';shutil.copyfile(source,output/filename)
        proofs.append({**proof['training_proof'],'model_path':filename,'valid_from':sessions[0],
                       'valid_through':sessions[-1],'inference_fit_reused':FIT,'model_sha256':digest(output/filename)})
    write_json(output/'chronology.json',{'folds':proofs,'scope':'Frozen last-development fit; reserved-window inference only, no fitting.'})
    write_json(output/'forward_lineage.json',{'features':bound(features/'completion.json'),
        'inference_bridge':bound(root/'frozen_inference_bridge.json'),'feature_bridge':bound(root/'forward_feature_bridge/comparison.json'),
        'models':result['model_proofs'],'program':bound(REPO/'scripts/portfolio_frozen_inference.py'),
        'no_entry_tail':sessions[-2:],'outcomes_used':False})
    pm=read(original/'policies_manifest.json')
    write_json(output/'completion.json',{'complete':True,'frozen_inference':True})
    files={p.name:digest(p) for p in output.iterdir() if p.is_file() and p.name not in ('policies_manifest.json','completion.json')}
    write_json(output/'policies_manifest.json',{**pm,'evaluation_from':sessions[0],'evaluation_through':sessions[-1],
        'prepared_sha256':digest(cp/'prepared.json'),'files':files,'scope':'reserved_validation_fixed_inference',
        'frozen_model_cutoff':FIT,'parent_policy_manifest_sha256':digest(original/'policies_manifest.json')})
    m,s=verify_policy(output,name);timed_policy(output,m,s)
    osnaps=olr_build(root/'forward_inputs/manifest.json',features/'market.sqlite',
        read(Path(study['code'])/study['olr_request']),[date.fromisoformat(d) for d in meta['sessions']])
    write_json(root/'forward_olr_snapshots.json',[s.to_json_dict() for d,s in osnaps.items() if sessions[0]<=str(d)<=sessions[-3]])
    write_json(root/'forward_prediction_completion.json',{'complete':True,'kalcb_policy':bound(output/'policies_manifest.json'),
        'olr_snapshots':bound(root/'forward_olr_snapshots.json'),'feature_completion':bound(features/'completion.json'),
        'performance_evaluated':False,'model_fitting_performed':False,'program':bound(Path(__file__))})
    print(json.dumps({'frozen_forward_prediction_complete':True,'sessions':len(sessions),'outcomes_evaluated':False}),flush=True)


def audit_split(root, output=None):
    """Verify actual forward model/feature artifacts against the reserved split."""
    from datetime import datetime
    from scripts.kalcb_executed_value_replay import verify_policy,timed_policy
    from backtests.strategies.kalcb import executed_value as ev
    protocol=read(root/'protocol.json');study=read(root/'development/study.json')
    start,end=protocol['reserved_validation']['start'],protocol['reserved_validation']['end']
    if not protocol['development']['end']<protocol['consumed_validation']['start']<=protocol['consumed_validation']['end']<start:
        raise ValueError('Overlapping development/consumed/reserved windows')
    policy=root/'forward_policies';manifest,snapshots=verify_policy(policy,study['kalcb_policy'])
    updates=timed_policy(policy,manifest,snapshots)
    sessions=sorted(str(d) for d in snapshots)
    if len(sessions)!=29 or (sessions[0],sessions[-1])!=(start,end):raise ValueError('Wrong validation session set')
    if any(snapshots[d].candidates for d in sorted(snapshots)[-2:]):raise ValueError('Opening entries in liquidation tail')
    if any(str(d) in sessions[-2:] for d in updates):raise ValueError('Timed entries in liquidation tail')
    olr=read(root/'forward_olr_snapshots.json')
    if sorted(r['trade_date'] for r in olr)!=sessions[:-2]:raise ValueError('OLR horizon does not match reserved sessions')
    models=[]
    for proof in read(policy/'chronology.json')['folds']:
        if (proof['fit_at'][:10]!='2026-03-10' or proof['train_through']>='2026-03-10'
                or proof['label_available_through']>=proof['fit_at']
                or proof['valid_from']!=start or proof['valid_through']!=end
                or digest(policy/proof['model_path'])!=proof['model_sha256']):
            raise ValueError('Unbound or noncausal forward model')
        models.append({k:proof[k] for k in ('name','fit_at','train_from','train_through','label_available_through','model_path','model_sha256')})
    features={}
    for stage in ('opening','actions','continuation','checkpoint'):
        path=root/'forward_features'/stage/'opportunities.jsonl';rows=ev.read_jsonl(path)
        if any(r.get('quality_target') is not None for r in rows):raise ValueError('Forward outcome target found')
        for row in rows:
            available=datetime.fromisoformat(row['feature_available_at'])
            if available.date().isoformat()!=row['trade_date']:raise ValueError('Wrong feature availability date')
            for bar in row.get('opening_bars',[]):
                from datetime import timedelta
                if datetime.fromisoformat(bar['timestamp'])+timedelta(minutes=5)>available:
                    raise ValueError('Feature contains a later bar')
        features[stage]={'rows':len(rows),'targets_present':False,'artifact':bound(path)}
    program=root/'program_snapshots'/('portfolio_forward_validation_'+digest(Path(__file__))+'.py')
    program.parent.mkdir(exist_ok=True)
    if not program.exists():shutil.copyfile(Path(__file__),program)
    if digest(program)!=digest(Path(__file__)):raise ValueError('Changed split audit program snapshot')
    result={'pass':True,'protocol':bound(root/'protocol.json'),'development':protocol['development'],
        'consumed_validation':protocol['consumed_validation'],'reserved_validation':{'start':start,'end':end,'sessions':29,'entry_sessions':27},
        'frozen_models':models,'forward_features':features,'policy_manifest':bound(policy/'policies_manifest.json'),
        'no_entry_tail':sessions[-2:],'split_record_sources':bound(root/'source_audit.json'),
        'program_dependency_review':bound(root/'forward_program_dependency_review.json'),
        'human_statement':'The user expects June–July not to have been used and requested split verification.',
        'historical_independence':'No prior use found in retained records; missing original search logs prevent categorical certification.',
        'validation_performance_files_present':list(map(str,(root/'validation/qualified').glob('*/result.json'))),
        'program':bound(program)}
    write_json(output or root/'split_audit_verified.json',result)
    print(json.dumps({'split_audit_pass':True,'frozen_models':len(models),'reserved_sessions':29,'oos_fitting':False}),flush=True)


def freeze_replay(root):
    """Only a sealed development finalist and two predeclared controls may run."""
    basis=read(root/'provider_basis_audit/audit.json')
    if basis.get('source_gate_pass') is not True:
        raise ValueError('Reserved input basis is unqualified; preserve the holdout without performance replay')
    acceptance=read(root/'forward_inputs/files/research/forward_acceptance/acceptance.json')
    if acceptance.get('basis_failures'):
        raise ValueError('Unresolved warm-up or lookback basis failure; preserve the holdout')
    review=read(root/'source_basis_review.json')
    if (review.get('source_gate_pass') is not True
            or review['provider_audit']!=bound(root/'provider_basis_audit/audit.json')
            or review['dependencies']['frozen_manifest']!=bound(root/'forward_inputs/manifest.json')):
        raise ValueError('Source qualification is not bound to these reserved inputs')
    for item in basis['original_and_adjusted_controls'].values():
        if digest(item['path'])!=item['sha256']:raise ValueError('Changed original/adjusted provider control')
    selection=read(root/'development_selection.json');protocol=read(root/'protocol.json')
    if selection['protocol']['sha256']!=digest(root/'protocol.json'):raise ValueError('Changed sealed protocol')
    if selection['screen']['sha256']!=digest(root/'screen.json'):raise ValueError('Changed selection evidence')
    prediction=read(root/'forward_prediction_completion.json')
    split=read(root/'split_audit_verified.json')
    if not split['pass'] or digest(split['program']['path'])!=split['program']['sha256']:
        raise ValueError('IS/OOS artifact audit failed')
    if not read(root/'forward_program_dependency_review.json')['inference_source_parity_pass']:
        raise ValueError('Forward inference used unqualified strategy source changes')
    for key in ('kalcb_policy','olr_snapshots','feature_completion'):
        if digest(prediction[key]['path'])!=prediction[key]['sha256']:raise ValueError('Changed inference artifact '+key)
    development=read(root/'development/study.json');spec=read(Path(development['code'])/development['spec_path'])
    parent=read(root/'forward_inputs/manifest.json');base=root/'forward_inputs/files'
    for name,expected in parent['files'].items():
        if digest(base/name)!=expected:raise ValueError('Changed reserved source generation before replay: '+name)
    out=root/'validation';out.mkdir(parents=True,exist_ok=False);target=out/'inputs/files';target.mkdir(parents=True)
    files={};spec_name='research/portfolio_forward/study.json'
    programs=('scripts/portfolio_forward_validation.py','scripts/portfolio_frozen_inference.py','scripts/portfolio_forward_olr.py')
    def bind(source,name,copy_file=False):
        path=target/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise FileExistsError('Duplicate frozen binding '+name)
        if copy_file:shutil.copyfile(source,path)
        else:
            try:os.link(source,path)
            except OSError:shutil.copyfile(source,path)
        files[name]=digest(path)
    for name in parent['files']:
        if name in programs:continue
        if name == parent['evidence']['qualification_path']:
            qualified = read(base/name)
            basis = read(base/'research/forward_acceptance/market_contract.json')['normalization']['target_basis']
            qualified['capacity']['market_basis'] = basis
            path = target/name;path.parent.mkdir(parents=True,exist_ok=True)
            write_json(path,qualified);files[name]=digest(path)
        else:bind(base/name,name)
    for name in programs:bind(REPO/name,name,True)
    policy_prefix='research/kalcb_executed_value/forward_policies'
    for p in (root/'forward_policies').iterdir():
        if p.is_file():bind(p,policy_prefix+'/'+p.name)
    for p in (root/'forward_features').rglob('*'):
        if p.is_file():bind(p,'research/portfolio_forward_features/'+p.relative_to(root/'forward_features').as_posix())
    for name in ('protocol.json','screen.json','development_selection.json','forward_prediction_completion.json',
                 'frozen_inference_bridge.json','olr_inference_bridge.json','forward_program_dependency_review.json','split_audit_verified.json'):
        bind(root/name,'research/portfolio_forward/'+name)
    bind(root/'forward_olr_snapshots.json','research/portfolio_forward/olr_snapshots.json')
    names=list(dict.fromkeys(('control','incumbent',selection['finalist'])))
    spec['candidates']={n:development['candidates'][n] for n in names}
    if spec['candidates'][selection['finalist']]!=selection['recipe']:raise ValueError('Finalist recipe changed')
    if parent['evidence'].get('corporate_actions'):
        spec['corporate_actions'] = parent['evidence']['corporate_actions']
    spec.update(kalcb_policies=policy_prefix,olr_snapshots='research/portfolio_forward/olr_snapshots.json',
        database='research/portfolio_forward_features/market.sqlite',database_hash=digest(root/'forward_features/market.sqlite'),
        scope='reserved_validation_frozen_development_selection',
        forward_validation={'selection':bound(root/'development_selection.json'),'protocol':bound(root/'protocol.json'),
            'candidates':names,'fit_reused':'2026-03-10','no_entry_tail':['2026-07-09','2026-07-10']})
    cfg=spec['configuration'];cfg.update(start='2026-06-01',end='2026-07-10',date_range={'start':'2026-06-01','end':'2026-07-10'},
        data_snapshot_end='2026-07-10',input_manifest_path=str(out/'inputs/manifest.json'),
        market_input_hash=parent['evidence']['shared_market_inputs']['market_input_hash'])
    spec['limitations']=['Short reserved validation; missing original logs limit independence certainty.',
        'Declared liquidity/slippage scenarios, not actual execution quotes; no financing charge.',
        'Last March model/calibration held fixed; no OOS fitting or candidate selection.']
    spec['native_budget_registration']={**spec['native_budget_registration'],'data_and_feed_unchanged':False,
        'scope':'New dated source generation; identical observation/subscription algorithm and fixed model inference.'}
    write_json(target/spec_name,spec);files[spec_name]=digest(target/spec_name)
    evidence=copy.deepcopy(parent['evidence']);evidence['portfolio_current_study']=spec_name
    evidence['executed_value_policy']['policies_manifest']=policy_prefix+'/policies_manifest.json'
    evidence['executed_value_policy']['prepared']='research/portfolio_forward_features/checkpoint/prepared.json'
    command='from kis_core.trading_calendar import get_trading_calendar;get_trading_calendar();from backtests.auto.shared.cache_keys import source_generation_hash;print(source_generation_hash())'
    source=subprocess.check_output([sys.executable,'-B','-c',command],cwd=target,text=True).strip()
    m={**{k:v for k,v in parent.items() if k!='manifest_hash'},'files':files,'evidence':evidence,'source_generation_hash':source}
    write_json(out/'inputs/manifest.json',{**m,'manifest_hash':signature(m)})
    write_json(out/'study.json',{**spec,'code':str(target.resolve()),'manifest':str((out/'inputs/manifest.json').resolve()),
        'source_generation_hash':source,'spec_path':spec_name})
    jobs=[{'recipe':n,'candidate':n,'scenario':s,'group':'validation','battery':s} for n in names for s in ('baseline','combined')]
    write_json(out/'registration.json',{'jobs':jobs,'protocol':bound(root/'protocol.json'),
        'development_selection':bound(root/'development_selection.json'),'source_generation_hash':source,
        'research_only':True,'oos_ranking_permitted':False})
    print(json.dumps({'validation_registered':names,'jobs':len(jobs)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['freeze-market','prepare-features','feature-bridge','predict','audit-split','freeze-replay'])
    p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args()
    {'freeze-market':freeze_market,'prepare-features':prepare_features,'feature-bridge':feature_bridge,
     'predict':predict,'audit-split':audit_split,'freeze-replay':freeze_replay}[a.command](a.root)
