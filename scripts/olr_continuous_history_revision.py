"""Freeze a complete IS-to-OOS intraday feature history for revision 10.

No strategy parameters, scoring dates, or fill/audit rules are changed.
Historical bars are copied from the authenticated IS database by column name;
the OOS observations remain byte-for-byte equal at the row-value level.
"""
from __future__ import annotations
import argparse
import copy
from datetime import date
import os
from pathlib import Path
import shutil
import sqlite3
import sys

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import olr_round1_comprehensive_study as native
from scripts import olr_round1_mechanism_study as mechanism

PARENT=native.DEFAULT_ROOT/'full_oos_20260401_20260710'
ROOT=PARENT/'continuous_history_v10'


def copy_file(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    try:os.link(source,target)
    except OSError:shutil.copy2(source,target)


def assemble():
    if (ROOT/'source_assembly.json').exists():
        raise ValueError('Assembly already exists; use the qualified next stage')
    ROOT.mkdir(parents=True,exist_ok=True)
    protocol=copy.deepcopy(native.read(PARENT/'protocol.json'))
    infos={w:native.read(native.generation_folder(PARENT,w)/'ready.json') for w in ('train','oos')}
    old=infos['oos'];source=Path(old['files'])
    generation=ROOT/'generations_v10/oos';files=generation/'inputs/files'
    manifest=copy.deepcopy(old['manifest']);manifest.pop('manifest_hash',None)
    for index,(name,digest) in enumerate(manifest['files'].items()):
        target=files/name
        if not target.exists():copy_file(source/name,target)
        if native.sha(target)!=digest:raise ValueError('Changed parent input: '+name)
        if index%1000==0:native.event(ROOT,'copying_continuous_inputs',files=index,total=len(manifest['files']))
    study=copy.deepcopy(old['study'])
    train=Path(infos['train']['files'])/infos['train']['study']['database']
    original=source/study['database']
    dbname='research/olr_continuous_history/market.sqlite';database=files/dbname
    database.parent.mkdir(parents=True,exist_ok=True)
    if database.exists():raise ValueError('Partial database requires preserved versioned repair')
    # Copy a standalone DB: no writes through immutable hardlinks.
    shutil.copy2(train,database)
    with sqlite3.connect(str(database),uri=True) as c:
        c.execute('PRAGMA cache_size=-131072')
        c.execute('ATTACH DATABASE ? AS forward',('file:'+original.as_posix()+'?mode=ro',))
        overlap=c.execute('SELECT COUNT(*) FROM bars a JOIN forward.bars b ON a.symbol=b.symbol AND a.ts=b.ts').fetchone()[0]
        mismatch=c.execute('SELECT COUNT(*) FROM bars a JOIN forward.bars b ON a.symbol=b.symbol AND a.ts=b.ts WHERE a.day!=b.day OR a.o!=b.o OR a.h!=b.h OR a.l!=b.l OR a.c!=b.c OR a.v!=b.v').fetchone()[0]
        if mismatch:raise ValueError('IS/OOS overlapping market observations differ')
        if c.execute("SELECT COUNT(*) FROM bars WHERE day>='2026-04-01'").fetchone()[0]:
            raise ValueError('IS DB contains OOS observations')
        c.execute('INSERT INTO bars (symbol,ts,day,o,h,l,c,v) SELECT symbol,ts,day,o,h,l,c,v FROM forward.bars WHERE day>=?',('2026-04-01',))
        c.execute('CREATE INDEX IF NOT EXISTS continuous_history_day ON bars(day,symbol,ts)')
        if c.execute('SELECT COUNT(*) FROM bars WHERE day!=substr(ts,1,10) OR length(symbol)!=6').fetchone()[0]:
            raise ValueError('Changed identity during database assembly')
        c.commit()
        prior=[r[0] for r in c.execute("SELECT DISTINCT day FROM bars WHERE day<'2026-04-01' ORDER BY day")]
        # Authenticate every original OOS DB row after the merge, including warm-up March31.
        missing=c.execute('SELECT COUNT(*) FROM forward.bars a LEFT JOIN bars b ON a.symbol=b.symbol AND a.ts=b.ts WHERE b.ts IS NULL OR a.day!=b.day OR a.o!=b.o OR a.h!=b.h OR a.l!=b.l OR a.c!=b.c OR a.v!=b.v').fetchone()[0]
        if missing:raise ValueError('Original forward observations changed')
    native.event(ROOT,'continuous_database_assembled',prior_sessions=len(prior),first=prior[0],overlap=overlap)
    import pandas as pd
    contract=manifest['evidence']['shared_market_inputs'];changes=[]
    with sqlite3.connect('file:'+train.as_posix()+'?mode=ro',uri=True) as c:
        for symbol,names in sorted(contract['intraday_files'].items()):
            if len(names)!=1:raise ValueError('Expected one retained OOS file per symbol')
            name=names[0];target=files/name
            original_frame=pd.read_parquet(target)
            first_day=original_frame.timestamp.dt.strftime('%Y-%m-%d').min()
            previous=pd.read_sql_query('SELECT ts AS timestamp,o AS open,h AS high,l AS low,c AS close,v AS volume FROM bars WHERE symbol=? AND day<? ORDER BY ts',c,params=(symbol,first_day))
            previous['timestamp']=pd.to_datetime(previous.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
            frame=pd.concat([previous,original_frame],ignore_index=True).sort_values('timestamp',kind='stable').reset_index(drop=True)
            if frame.timestamp.duplicated().any():raise ValueError('Duplicated continuous source bar')
            target.unlink()  # Remove this link only; never edit shared frozen bytes.
            frame.to_parquet(target,index=False)
            manifest['files'][name]=native.sha(target)
            changes.append({'symbol':symbol,'path':name,'prior_bars_added':len(previous),
                'original_hash':old['manifest']['files'][name],'continuous_hash':manifest['files'][name]})
    receipt={'version':10,'created_utc':mechanism.utc(),'scoring_windows':protocol['windows'],
        'feature_history':'All available authenticated IS intraday observations through March31, followed continuously by OOS. Scoring boundary never resets feature state.',
        'prior_sessions':len(prior),'prior_first':prior[0],'prior_last':prior[-1],
        'train_database':str(train),'train_sha256':native.sha(train),
        'oos_database':str(original),'oos_sha256':native.sha(original),
        'database':dbname,'database_sha256':native.sha(database),
        'overlap_rows':overlap,'overlap_mismatches':mismatch,'changed_oos_rows':missing,
        'source_changes':changes,'strategy_parameters_changed':False,
        'account_scoring':'Separate declared KRW100m account windows; feature history is continuous.',
        'qualification':'pending full source evidence and native audits; not performance evidence'}
    proof_name='research/olr_continuous_history/history_receipt.json'
    native.write(files/proof_name,receipt);manifest['files'][proof_name]=native.sha(files/proof_name)
    manifest['files'][dbname]=native.sha(database)
    contract.update(train_start=prior[0],history_context_extension=proof_name)
    contract.pop('market_input_hash',None)
    for name in contract['files']:contract['files'][name]=manifest['files'][name]
    contract['files'][proof_name]=manifest['files'][proof_name]
    contract['market_input_hash']=native.sig(contract)
    study.update(code=str(files),database=dbname,database_hash=native.sha(database))
    study['configuration']['market_input_hash']=contract['market_input_hash']
    native.write(generation/'assembly_manifest.json',manifest)
    native.write(generation/'assembly_study.json',study)
    native.write(ROOT/'source_assembly.json',receipt)
    native.event(ROOT,'continuous_source_assembly_complete',prior_sessions=len(prior))


def qualify():
    generation=ROOT/'generations_v10/oos';files=generation/'inputs/files'
    manifest=native.read(generation/'assembly_manifest.json')
    sys.path.insert(0,str(files));os.chdir(files)
    from backtests.engine.retained_source import build_evidence
    previous=native.read(files/manifest['evidence']['qualification_path'])
    destination=files/'research/olr_continuous_history/acceptance'
    qpath=build_evidence(files,manifest['evidence']['shared_market_inputs'],destination)
    qualification=native.read(qpath)
    for key in ('capacity','market_files','session_resolution_report'):
        qualification[key]=previous[key]
    native.write(qpath,qualification)
    acceptance=native.read(destination/'acceptance.json')
    native.write(ROOT/'source_preflight.json',acceptance)
    if acceptance['basis_failures']:
        raise ValueError('Expanded history has explicit source-basis dispositions requiring review; no automatic skipping or audit weakening')
    manifest['evidence']['qualification_path']=qpath.relative_to(files).as_posix()
    for p in destination.glob('*.json'):manifest['files'][p.relative_to(files).as_posix()]=native.sha(p)
    mp=generation/'inputs/manifest.json'
    native.write(mp,{**manifest,'manifest_hash':native.sig(manifest)})
    study=native.read(generation/'assembly_study.json');study['manifest']=str(mp)
    study['configuration']['input_manifest_path']=str(mp)
    native.write(generation/'qualified_study.json',study)
    native.verify_market_bindings(manifest)
    native.event(ROOT,'continuous_source_qualified',checked_bars=acceptance['checked_bars'])


def register():
    """Rebuild the unchanged control, then register the preserved experiment set."""
    generation=ROOT/'generations_v10/oos';files=generation/'inputs/files'
    study=native.read(generation/'qualified_study.json')
    manifest=native.read(study['manifest']);manifest.pop('manifest_hash',None)
    prior_parser_receipt=manifest['evidence'].pop('recomputed_retained_acceptance',None)
    manifest['evidence']['continuous_history_qualification']={
        'source_assembly_sha256':native.sha(ROOT/'source_assembly.json'),
        'source_preflight_sha256':native.sha(ROOT/'source_preflight.json'),
        'scope':'Expanded historical observations requalified under unchanged source rules.',
        'parent_parser_requalification_receipt':prior_parser_receipt}
    if native.read(ROOT/'source_preflight.json')['basis_failures']:
        raise ValueError('Unresolved continuous-history source qualification')
    if (ROOT/'mechanism_plan.json').exists():
        raise ValueError('Revision already registered; resume its native supervisor')
    protocol=copy.deepcopy(native.read(PARENT/'protocol.json'))
    protocol.update(harness_revision=10,harness_sha256=native.sha(native.__file__),
        orchestration_sha256=native.sha(mechanism.__file__),
        generation_directories={'train':str(native.DEFAULT_ROOT/'generations_v4'),'oos':'generations_v10'},
        revision={'at_utc':mechanism.utc(),'previous_root':str(PARENT),
            'reason':'Carry all available IS intraday feature history across April1. Strategy and scoring dates unchanged; prior short-context evidence retained.'},
        history_policy='continuous_is_to_oos_no_feature_reset',
        promotion='research only; no automatic canonical or live changes')
    native.write(ROOT/'protocol.json',protocol)
    shutil.copy2(PARENT/'candidate_plan.json',ROOT/'candidate_plan.json')
    original=native.read(ROOT/'candidate_plan.json')
    baseline=next(c for c in original['candidates'] if c['id']==original['baseline_id'])
    # This temporary descriptor uses the old snapshots solely for requested dates.
    # No native evaluation can start until the final registered plan is written.
    info={'study':study,'manifest':manifest,'files':str(files)}
    native.write(generation/'assembly_ready.json',info)
    native.write(generation/'ready.json',info)
    candidate_path=ROOT/'continuous_control_candidate.json'
    native.write(candidate_path,baseline)
    snapshot_name='research/olr_continuous_history/olr_snapshots.json'
    snapshot_path=files/snapshot_name
    rc=native.run_child([sys.executable,'-B','-u',str(Path(native.__file__).resolve()),'snapshots',
        '--root',str(ROOT),'--window','oos','--candidate',str(candidate_path),'--output',str(snapshot_path)],
        files,ROOT/'continuous_control_builder.log')
    if rc:raise ValueError('Continuous control reconstruction failed; retain its evidence')
    receipt=native.read(snapshot_path.with_suffix('.receipt.json'))
    assembly=native.read(ROOT/'source_assembly.json')
    if len(receipt['context_only_dates'])!=assembly['prior_sessions']:
        raise ValueError('Reconstruction discarded part of the IS feature history')
    study['olr_snapshots']=snapshot_name
    study['spec_path']='research/olr_continuous_history/parent_spec.json'
    study['scope']='continuous_is_feature_history_april_july_scoring'
    study['configuration']['input_manifest_path']=str(generation/'inputs/manifest.json')
    spec={k:v for k,v in study.items() if k not in ('manifest','code','spec_path','source_generation_hash')}
    native.write(files/study['spec_path'],spec)
    for name in (snapshot_name,study['spec_path']):manifest['files'][name]=native.sha(files/name)
    manifest['files'][snapshot_path.with_suffix('.receipt.json').relative_to(files).as_posix()]=native.sha(snapshot_path.with_suffix('.receipt.json'))
    native.write(generation/'inputs/manifest.json',{**manifest,'manifest_hash':native.sig(manifest)})
    native.write(generation/'parent_study.json',study)
    native.write(generation/'ready.json',{'study':study,'manifest':manifest,'files':str(files)})
    verifier=Path(__file__).with_name('olr_continuous_history_verification.py')
    rc=native.run_child([sys.executable,'-B','-u',str(verifier.resolve()),str(ROOT)],
        files,ROOT/'continuous_history_verification.log')
    if rc or native.read(ROOT/'continuous_history_verification.json')['status']!='passed':
        raise ValueError('Independent continuous-history feature verification failed')
    protocol['parent_studies']['oos']={'path':str(generation/'parent_study.json'),
        'sha256':native.sha(generation/'parent_study.json')}
    for name in ('historical_leaf_coverage_audit.json','lineage_attribution.json','adaptive_review_requirements.json','superseded_run.json'):
        shutil.copy2(PARENT/name,ROOT/name)
    for name in ('june_july_diagnostic','lineage','builder_bridge/train',
                 'revisions/revision_8_supported_exit_contract/failed_intrabar_stop_oos'):
        shutil.copytree(PARENT/name,ROOT/name)
    (ROOT/'adaptive_plans').mkdir(exist_ok=True)
    shutil.copy2(PARENT/'adaptive_plans/early_interactions.json',ROOT/'adaptive_plans/early_interactions.json')
    for old in (PARENT/'cases').glob('*/train/100000000_baseline'):
        if not (old/'assessment.json').exists() or not native.read(old/'assessment.json').get('qualified'):
            continue
        cid=old.parent.parent.name
        target=ROOT/'cases'/cid/'train/100000000_baseline'
        shutil.copytree(old,target)
        shutil.copytree(PARENT/'registrations/train'/(cid+'_100000000'),ROOT/'registrations/train'/(cid+'_100000000'))
        assessment=native.read(target/'assessment.json');assessment['output']=str(target)
        native.write(target/'assessment.json',assessment)
        mechanism.validate_cached(ROOT,native.read(ROOT/'registrations/train'/(cid+'_100000000')/'candidate.json'),'train')
    plan=copy.deepcopy(native.read(PARENT/'mechanism_plan.json'))
    plan.update(version=10,runner_sha256=native.sha(mechanism.__file__),native_evaluator_sha256=native.sha(native.__file__))
    early=native.read(ROOT/'adaptive_plans/early_interactions.json')
    plan['phases']=mechanism.historical_first_phases(plan,[c['id'] for c in early['candidates']])
    plan['adaptive_rules'].update(max_components_per_family=5,
        complementary_screen='Positive return or increased frequency in either window, subject to component floors; reserve each objective separately. Final paired acceptance gates unchanged.')
    protocol['phases']=[p['name'] for p in plan['phases']]+['paired_robustness','support_and_uncertainty']
    plan['continuous_history_revision']={'prior_sessions':assembly['prior_sessions'],
        'first':assembly['prior_first'],'last':assembly['prior_last'],
        'user_requirement':'Use IS as intraday warm-up so the scoring boundary is continuous.',
        'all_original_definitions_retained':True,'warmup_chosen_by_returns':False}
    plan['source_evidence']['source_assembly.json']=native.sha(ROOT/'source_assembly.json')
    plan['source_evidence']['source_preflight.json']=native.sha(ROOT/'source_preflight.json')
    plan['source_evidence']['continuous_history_verification.json']=native.sha(ROOT/'continuous_history_verification.json')
    native.write(ROOT/'mechanism_plan.json',plan)
    protocol['mechanism_plan_sha256']=native.sha(ROOT/'mechanism_plan.json')
    native.write(ROOT/'protocol.json',protocol)
    native.write(ROOT/'revision10_registration.json',{'revision':10,'registered_utc':mechanism.utc(),
        'scoring_windows':protocol['windows'],'prior_intraday_sessions':assembly['prior_sessions'],
        'all_history_reconstruction_receipt':str(snapshot_path.with_suffix('.receipt.json')),
        'reused_oos_results':0,'full_native_control_audit':'pending',
        'harness_validation':native.read(ROOT/'implementation_validation.json')})
    pointer={'root':str(ROOT),'revision':10,'windows':protocol['windows']}
    native.write(native.DEFAULT_ROOT/'active_study.json',pointer)
    native.write(PARENT/'superseded_mechanism.json',pointer)
    native.event(ROOT,'continuous_history_revision_registered',prior_sessions=assembly['prior_sessions'])


def preflight():
    from collections import Counter
    for window in ('oos','train'):
        rows=[]
        for candidate in mechanism.catalog(ROOT).values():
            try:
                config=native.candidate_preflight(ROOT,window,candidate)
                unsupported=mechanism.unsupported_exit_orders(config)
                rows.append({'id':candidate['id'],'status':'unsupported_execution_contract' if unsupported else 'supported',
                    'unsupported':unsupported})
            except ValueError as exc:
                rows.append({'id':candidate['id'],'status':'invalid_configuration','error':str(exc)})
        counts=dict(Counter(r['status'] for r in rows))
        native.write(ROOT/f'revision10_preflight_{window}.json',{'window':window,'counts':counts,'candidates':rows})
        if counts!={'supported':487,'invalid_configuration':9,'unsupported_execution_contract':22}:
            raise ValueError('Configuration support changed; review before broad dispatch')
    native.event(ROOT,'continuous_history_preflight_complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['assemble','qualify','register','preflight'])
    args=parser.parse_args()
    {'assemble':assemble,'qualify':qualify,'register':register,'preflight':preflight}[args.command]()
