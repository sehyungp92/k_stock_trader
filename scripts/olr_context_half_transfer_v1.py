"""0042: one exact contextual-half OOS diagnostic; no automatic successor."""
from collections import defaultdict
from pathlib import Path
import argparse
import copy
import json
import os
import sqlite3
import subprocess
import sys
import traceback
import psutil
import olr_selective_half_continuation_v1 as shared
import olr_relative_context_qualification_v1 as context

read,write,sha,sig,now,status=shared.read,shared.write,shared.sha,shared.sig,shared.now,shared.status
REPO=shared.REPO
BASE=Path('E:/CodexResearch/k_stock_trader/olr_continuing_research_20261016')
PARENT=BASE/'relative_qualification_v2'
IS=BASE/'relative_context_v2'
SOURCE='bf2148d4a3727ba3b03c6a6d1c0be8013298eb4d6a8dc4954031a9db81e9fa9b'
HELPERS=['olr_selective_half_continuation_v1.py','olr_relative_context_qualification_v1.py',
         'olr_relative_context_assessment_v1.py','olr_context_half_transfer_review_v1.py']


def verify(root):
    p=read(root/'protocol.json')
    assert sha(__file__)==p['runner_sha256'] and sha(root/'design_at_freeze.md')==p['design_sha256']
    for name,h in p['helpers'].items():assert sha(REPO/'scripts'/name)==h
    for path,h in p['inputs'].items():assert sha(path)==h,path
    return p


def prepare(root):
    verify(root);status(root,'authenticating_reused_sources_and_accounts',queued=['half_oos_nominal'])
    for parent in [IS,PARENT,BASE/'half_access_interaction_v1']:
        for path in [parent/'controller_process.json',*parent.glob('runtime/*/*/process.json')]:
            if not path.exists():continue
            r=read(path)
            try:
                p=psutil.Process(r['pid']);assert abs(p.create_time()-r['created_at'])>.001 or p.cmdline()!=r['command'],'Authenticated parent still active'
            except psutil.NoSuchProcess:pass
    ready=copy.deepcopy(read(PARENT/'native_engines/oos/ready.json'));origin=Path(ready['files'])
    files=root/'native_engines/oos/files';files.mkdir(parents=True)
    for n,h in ready['manifest']['files'].items():
        dest=files/n;dest.parent.mkdir(parents=True,exist_ok=True);os.link(origin/n,dest)
        if n.endswith('.py') or n.startswith('kis_core/data/'):
            assert sha(dest)==h,n
    sys.path.insert(0,str(files))
    from scripts.olr_economic_study import source_hash
    from backtests.strategies.olr.economic_evidence import completed_case
    from scripts.olr_economic_assessment import contrast
    assert source_hash(files)==source_hash(origin)==SOURCE
    pre=read(PARENT/'preflight.json');assert pre['passed'] and pre['tests']==114
    assert sha(PARENT/'focused_tests.xml')==pre['tests_sha256']
    assert sha(PARENT/'calendar_preflight.json')==pre['calendar_preflight_sha256']
    assert read(PARENT/'calendar_preflight.json')['pass_']
    study=ready['study'];manifest=ready['manifest'];manifest.pop('manifest_hash',None)
    definition=read(files/study['minute_source_definition'])
    critical=[study['database'],study['minute_source_definition'],definition['minute_database'],
        study['relative_context_features'],ready['interval_qualification'],'config/universe_103.yaml']
    for name in critical:assert sha(files/name)==manifest['files'][name],name
    features=read(files/study['relative_context_features']);assert len(features['evaluation_days'])==69
    original=read(IS/'assessment.json');prior=Path(original['summaries']['half']['case'])
    halfstudy=read(IS/'runtime/train/half/study.json')
    assert read(Path(halfstudy['code'])/halfstudy['relative_context_settings'])['fraction']==.5
    accepted={}
    summaries=read(PARENT/'assessment.json')['summaries']
    for name,summary in [('is_half',original['summaries']['half']),('control',summaries['oos_nominal_control']),('full',summaries['oos_nominal_full'])]:
        case=Path(summary['case']);assert sha(case/'completion.json')==summary['completion_sha256']
        assert completed_case(case)['valid']
        checks={n:sha(case/n)==h for n,h in read(case/'completion.json')['artifacts'].items()};assert all(checks.values())
        accepted[name]=dict(case=str(case),completion_sha256=summary['completion_sha256'],artifacts=checks)
    con=sqlite3.connect((files/study['database']).as_uri()+'?mode=ro',uri=True);con.execute('PRAGMA query_only=ON');count=0
    for symbol,history in read(prior/'capacity_terminal_state.json')['capacity_history_v1'].items():
        for kind in ['auction','continuous']:
            for stamp,volume in history[kind]:
                assert stamp[:10]<'2026-04-01'
                assert con.execute('SELECT v FROM bars WHERE symbol=? AND ts=?',(symbol,stamp)).fetchall()==[(volume,)]
                count+=1
    con.close()
    ready['files']=str(files);study.update(code=str(files),source_generation_hash=SOURCE)
    write(root/'native_engines/oos/ready.json',ready)
    # Actual receipt serialization is exercised before a worker is admitted.
    write(root/'acceptance_roundtrip.json',original['summaries']['half'])
    assert read(root/'acceptance_roundtrip.json')==original['summaries']['half']
    write(root/'preflight.json',dict(at=now(),passed=True,source_by_window=dict(oos=SOURCE,train=context.SOURCE),
        unchanged_python_and_test_bytes=True,reused_tests=114,tests_sha256=pre['tests_sha256'],
        calendar_preflight_sha256=pre['calendar_preflight_sha256'],own_half_history_samples=count,
        evaluation_sessions=69,reused=accepted,protocol_sha256=sha(root/'protocol.json')))
    return source_hash,completed_case,contrast


def launch(root):
    ready=read(root/'native_engines/oos/ready.json');files=Path(ready['files'])
    study=copy.deepcopy(ready['study']);manifest=copy.deepcopy(ready['manifest'])
    prefix='research/context_half_transfer/';case=root/'cases/oos/half_nominal';assert not case.exists()
    for key,name,value in [('structural_settings','structural.json',dict(mode='control',interval_service=True)),
        ('funded_probe_settings','probe.json',dict(version='first-native-fade-funded-probe-v1',action='hold',control_reference=None)),
        ('funded_population_settings','population.json',dict(version='monthly-native-funded-population-v1',cell=None,catalog_reference=None)),
        ('relative_context_settings','context.json',dict(fraction=.5,version='relative-context-v1'))]:
        study[key]=prefix+name;write(files/study[key],value);manifest['files'][study[key]]=sha(files/study[key])
    prior=Path(read(IS/'assessment.json')['summaries']['half']['case'])
    for key,name in [('capacity_checkpoint','capacity_terminal_state.json'),('capacity_checkpoint_completion','completion.json'),('capacity_checkpoint_result','result.json')]:
        study[key]=prefix+'parent_'+name;context.install(prior/name,files/study[key],sha(prior/name));manifest['files'][study[key]]=sha(files/study[key])
    manifest['evidence']['qualification_path']=ready['interval_qualification']
    study['manifest']=str(files.parent/'half_nominal.manifest.json');study['configuration']['input_manifest_path']=study['manifest'];study['spec_path']=prefix+'spec.json'
    write(files/study['spec_path'],{k:v for k,v in study.items() if k not in ['code','manifest','source_generation_hash','spec_path']})
    manifest['files'][study['spec_path']]=sha(files/study['spec_path']);write(study['manifest'],dict(**manifest,manifest_hash=sig(manifest)))
    runtime=root/'runtime/oos/half_nominal';write(runtime/'study.json',study)
    cmd=[sys.executable,'-B','-u',str(files/'scripts/portfolio_synergy_study.py'),'replay','--study',str(runtime/'study.json'),'--recipe','candidate','--scenario','baseline','--standalone','OLR','--output',str(case),'--full-audit']
    log=root/'half_nominal.log'
    with log.open('x',encoding='utf8') as stream:p=subprocess.Popen(cmd,cwd=files,env=shared.env_for(root),stdout=stream,stderr=subprocess.STDOUT,creationflags=0x08000000)
    write(runtime/'process.json',dict(at=now(),pid=p.pid,created_at=psutil.Process(p.pid).create_time(),command=cmd,case=str(case),log=str(log)))
    return p,case


def execute(root):
    source_hash,completed_case,contrast=prepare(root);verify(root)
    assert source_hash(root/'native_engines/oos/files')==SOURCE
    status(root,'running',case='half_oos_nominal',queued=[])
    p,case=launch(root);assert p.wait()==0,'Native half diagnostic failed; preserve generation'
    assert completed_case(case)['valid']
    assert read(case/'funded_checkpoint_receipt.json')['no_action_continuation_equal'] and read(case/'funded_population_receipt.json')['audit']['pass_']
    restart=read(case/'capacity_restart_receipt.json');assert restart['same_configuration_and_scenario'] and restart['all_samples_verified_against_oos_retained_ohlcv']
    half=context.summarize(case);assert half['context_decisions']
    for row in half['context_decisions']:assert row['metadata']['relative_context_evidence']['fraction']==.5
    write(root/'accepted/half_nominal.json',half)
    old=read(PARENT/'assessment.json')['summaries'];summaries=dict(control=old['oos_nominal_control'],half=half,full=old['oos_nominal_full']);comparisons={}
    for left,right in [('control','half'),('control','full'),('half','full')]:
        a,b=summaries[left],summaries[right];accounting=contrast(read(Path(a['case'])/'trades.json'),read(Path(b['case'])/'trades.json'))
        dates=defaultdict(float)
        for row in accounting['per_opportunity']:dates[row['day']]+=row['delta_pnl']
        comparisons[left+'_to_'+right]=dict(delta={k:b[k]-a[k] for k in ['return_pct','full_dd_pct','common_dd_pct','maximum_currency_decline','meaningful','meaningful_days','buy_notional','winning_wealth','losing_wealth','longest_underwater_calendar_days']},accounting=accounting,entry_date_attribution=dict(dates))
    assessment=dict(at=now(),summaries=summaries,comparisons=comparisons,economic_objective_achieved=False,next='Substantive partial recovery/funding and useful-participation disposition; no automatic qualification or successor.')
    write(root/'assessment.json',assessment)
    pair=root/'pairs/oos_nominal';write(pair/'assessment.json',{**assessment,'summaries':{k:v for k,v in summaries.items() if k!='full'},'comparisons':{'control_to_half':comparisons['control_to_half']}})
    previous=read(IS/'assessment.json');write(root/'pairs/train_nominal/assessment.json',dict(summaries={k:previous['summaries'][k] for k in ['control','half']},comparisons={'control_to_half':previous['comparisons']['control_to_half']}))
    status(root,'reviewing_accepted_account',accepted=['half_oos_nominal'],queued=[])
    for label,command in [('causal',[sys.executable,'-B',str(REPO/'scripts/olr_relative_context_assessment_v1.py'),str(pair)]),
                          ('transfer',[sys.executable,'-B',str(REPO/'scripts/olr_context_half_transfer_review_v1.py'),str(root),'nominal'])]:
        with (root/(label+'_review.log')).open('x',encoding='utf8') as stream:
            worker=subprocess.Popen(command,cwd=REPO,env=shared.env_for(root),stdout=stream,stderr=subprocess.STDOUT,creationflags=0x08000000)
            write(root/(label+'_review_process.json'),dict(at=now(),pid=worker.pid,created_at=psutil.Process(worker.pid).create_time(),command=command))
            assert worker.wait()==0,'Preserve analysis failure; do not rerun accepted account: '+label
    status(root,'awaiting_substantive_disposition',accepted=['half_oos_nominal'],queued=[])


def start(root):
    root.mkdir(exist_ok=False)
    design=REPO/'docs/optimisation/olr/experiments/olr-exp-0042.md';(root/'design_at_freeze.md').write_bytes(design.read_bytes())
    inputs=[PARENT/'preflight.json',PARENT/'assessment.json',PARENT/'native_engines/oos/ready.json',IS/'assessment.json',IS/'preflight.json']
    write(root/'protocol.json',dict(at=now(),experiment='0042',runner_sha256=sha(__file__),design_sha256=sha(design),
        helpers={n:sha(REPO/'scripts'/n) for n in HELPERS},inputs={str(p):sha(p) for p in inputs},new_accounts=1,policy='exact0035 contextual half',scenario='nominal OOS',automatic_successor=False))
    for name in [Path(__file__).name,*HELPERS]:
        target=root/'launch_sources'/name;target.parent.mkdir(exist_ok=True);target.write_bytes((REPO/'scripts'/name).read_bytes())
    cmd=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'execute',str(root)]
    with (root/'controller.log').open('x',encoding='utf8') as stream:p=subprocess.Popen(cmd,cwd=REPO,env=shared.env_for(root),stdout=stream,stderr=subprocess.STDOUT,creationflags=0x08000000)
    receipt=dict(at=now(),pid=p.pid,created_at=psutil.Process(p.pid).create_time(),command=cmd);write(root/'controller_process.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['start','execute']);p.add_argument('root',type=Path);args=p.parse_args()
    try:globals()[args.phase](args.root)
    except Exception as exc:
        if args.root.exists():status(args.root,'needs_repair',error=repr(exc),traceback=traceback.format_exc())
        raise
