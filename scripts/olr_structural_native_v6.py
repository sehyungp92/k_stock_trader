"""Fresh native generation and bounded train-first structural migration test."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,copy,difflib,json,os,shutil,subprocess,sys,tempfile,traceback
from scripts import olr_structural_v6 as project
from scripts import olr_structural_patches_v6 as patches
from scripts.olr_minute_exit_prepare_v5 import install
from scripts import olr_minute_exit_study_v5 as t
from scripts import olr_opening_risk_qualify_v2 as engine
from scripts import olr_targeted_risk_results_v1 as metrics
n,s,b=t.n,t.s,t.b
ROOT=project.ROOT/'native_controlled'
OUT=project.OUT/'native_controlled'
PARENT=project.PARENT
PARENT_OUT=project.REPO/'data/backtests/analysis/olr_minute_exit_20261011/clockfix_generation_1'
GROUPS=(('baseline',False,'windows'),('combined',False,'stress'),('baseline',True,'capped'))

def configure():
    project.temp();t.ROOT,t.OUT,t.ENGINES=ROOT,OUT,ROOT/'native_engines';t.activate();engine.t=t

def write(path,value):project.write(path,value)

def register():
    configure();cal=project.read(project.OUT/'forecast_calibration.json');verified=project.read(project.OUT/'forecast_verification.json')
    assert verified['calibration_sha256']==n.sha(project.OUT/'forecast_calibration.json')
    assert verified['passed'] and not cal['normalized_model_advances']
    parent_plan=n.read(PARENT/'candidate_plan.json');base=copy.deepcopy(parent_plan['baseline']);catalog=[]
    for mode in ('off','broad_median'):
        c=dict(id='c'+n.sig({'study':'structural-v6','mode':mode,'mutations':base})[:16],label='legacy_clock_control' if mode=='off' else 'action_independent_median',
               history_policy='literal_carry',mutations=copy.deepcopy(base),mutations_hash=n.sig(base),structural_mode=mode,
               aliases=[],kind='structural_v6',paths=[])
        catalog.append(c)
    plan=dict(baseline=base,candidates=catalog)
    protocol=n.read(PARENT/'protocol.json');protocol['generation_directories']={w:str(t.ENGINES) for w in ('train','oos')}
    mechanism=n.read(PARENT/'mechanism_plan.json');mechanism['candidates']=catalog
    mechanism['snapshot_acceleration']['cache_root']=str(ROOT/'snapshot_cache')
    registration=dict(version='structural-native-v6',utc=s.utc(),parent_registration_sha256=n.sha(project.ROOT/'registration.json'),
        calibration_sha256=n.sha(project.OUT/'forecast_calibration.json'),verification_sha256=n.sha(project.OUT/'forecast_verification.json'),
        excluded='Normalized forecast failed the frozen chronological forecast gates; no normalized native replay.',
        controls='Identical entry artifacts, native minute execution, sizing/entry/exit thresholds and original risk budgets. Only source observation/median sparse-history behavior changes.',
        modes=['off','broad_median'],max_cases=12,
        execution_order='Six IS cases first: both modes under nominal, combined stress, capped. OOS only if all IS return-preservation gates and IS DD gates pass. No retrospective replacement candidate.',
        prior_gates=n.read(PARENT/'registration.json')['gates'],
        training_stop='Treatment must preserve original/guard/retained scenario return floors, matched fine-control IS within 1.5pp, and guard/common DD within original 2pp allowance. Capped uses retained capped return floor. Failure ends native advancement, not permission to weaken gates.',
        full_objective='Every old gate plus meaningful new frequency, matched return preservation and strict DD improvement on both grids remains required. No promotion in this foundation study.',
        retained_id='c4f49be01ed09b8dc',independent_validation=False,promotion=False,
        entry_artifacts='Exact parent artifacts and unchanged entry-building/configuration modules; capacity is a request-time model. Native source-feature qualification runs on every case.',
        restart='Actual terminal observed full-panel prefix, plus actual routed continuous/capacity history, authenticated against independent OOS sources for matching candidate/scenario/cap.',
        report='Report original/guard/retained and new control/migration, both DD grids, scenario counts, actual notional, all admission constraints, concentration and effect accounting. No invented opportunity expansion.')
    for name,value in [('candidate_plan.json',plan),('protocol.json',protocol),('mechanism_plan.json',mechanism),('registration.json',registration)]:write(ROOT/name,value)
    write(OUT/'registration.json',registration)
    # Freeze exact research sources before preparation; generation and final
    # protocol/test receipts are additionally sealed before the first replay.
    source_names=['scripts/olr_structural_v6.py','scripts/olr_structural_source_v6.py','scripts/olr_structural_verify_v6.py',
        'scripts/olr_structural_native_v6.py','scripts/olr_structural_native_test_v6.py','scripts/olr_structural_patches_v6.py',*patches.NEW,
        'tests/strategy_olr/test_liquidity_v6.py','tests/strategy_olr/test_liquidity_runtime_v6.py']
    write(ROOT/'launch_harness.json',dict(files={p:n.sha(project.REPO/p) for p in source_names},
         artifacts={str(project.OUT/f):n.sha(project.OUT/f) for f in ('forecast_calibration.json','forecast_verification.json')},
         registration_sha256=n.sha(ROOT/'registration.json'),utc=s.utc()))

def verify_launch():
    value=n.read(ROOT/'launch_harness.json')
    for name,digest in value['files'].items():assert n.sha(project.REPO/name)==digest,name
    for name,digest in value['artifacts'].items():assert n.sha(name)==digest,name
    assert n.sha(ROOT/'registration.json')==value['registration_sha256']
    return value

def child(command,log,identity,cwd=None):
    Path(log).parent.mkdir(parents=True,exist_ok=True)
    with Path(log).open('x',encoding='utf-8') as stream:
        p=subprocess.Popen(command,cwd=cwd or project.REPO,stdout=stream,stderr=subprocess.STDOUT)
        query="Get-CimInstance Win32_Process -Filter 'ProcessId="+str(p.pid)+"' | Select-Object CommandLine,@{n='creation_time_utc';e={$_.CreationDate.ToUniversalTime().ToString('o')}} | ConvertTo-Json -Compress"
        actual=json.loads(subprocess.check_output(['powershell.exe','-NoProfile','-Command',query],text=True))
        write(identity,dict(pid=p.pid,**actual,command=command,launch_utc=s.utc()))
        print(dict(command=command,pid=p.pid),flush=True);rc=p.wait()
    if rc:raise RuntimeError('Preserve failed child before repair: '+str(log))

def prepare():
    configure();verify_launch()
    if (ROOT/'preparation_complete.json').exists():raise RuntimeError('Do not rerun completed preparation')
    # OOS source projection is permitted only after the train-only model choice
    # is fixed. Same extractor; exactly these period/output-name substitutions.
    worker=ROOT/'oos_source_worker.py';original=(project.REPO/'scripts/olr_structural_source_v6.py').read_text(encoding='utf-8')
    changes=[("if window!='train':raise ValueError('Stage 1 cannot read OOS')","if window!='oos':raise ValueError('Only frozen stage-2 OOS projection')"),
             ('date(2026,3,31)','date(2026,7,10)'),("root/'train_panel.json'","root/'oos_panel.json'"),
             ("root/'train_panel_receipt.json'","root/'oos_panel_receipt.json'"),('oos_read=False','oos_read=True')]
    transformed=original
    for old,new in changes:
        if old not in transformed:raise ValueError('OOS extractor template mismatch')
        transformed=transformed.replace(old,new)
    worker.write_text(transformed,encoding='utf-8',newline='\n')
    write(ROOT/'source_projection_contract.json',dict(template_sha256=n.sha(project.REPO/'scripts/olr_structural_source_v6.py'),
        worker_sha256=n.sha(worker),exact_substitutions=changes,pnl_read=False,selection_preceded_oos=True))
    child([sys.executable,'-B','-u',str(worker),str(ROOT),str(PARENT),'oos'],ROOT/'oos_source.log',ROOT/'oos_source_process.json')
    for w in ('train','oos'):
        parent=n.read(PARENT/'native_engines'/w/'ready.json');old=Path(parent['files']);directory=t.ENGINES/w;files=directory/'inputs/files'
        if (directory/'ready.json').exists():raise RuntimeError('Do not rebind completed generation')
        manifest=copy.deepcopy(parent['manifest']);study=copy.deepcopy(parent['study']);diff=[]
        for index,(name,digest) in enumerate(manifest['files'].items()):
            install(old/name,files/name,digest)
            if index%250==0:print(dict(window=w,phase='copy_and_authenticate',files=index),flush=True)
        for name in patches.CHANGED:
            before=(old/name).read_text(encoding='utf-8');after=patches.patch(name,before);compile(after,name,'exec')
            (files/name).write_text(after,encoding='utf-8',newline='\n');manifest['files'][name]=n.sha(files/name)
            diff.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+w+'/'+name,tofile='b/'+w+'/'+name))
        for name in patches.NEW:
            install(project.REPO/name,files/name,n.sha(project.REPO/name));manifest['files'][name]=n.sha(files/name)
            diff.extend(difflib.unified_diff([],(files/name).read_text(encoding='utf-8').splitlines(True),fromfile='/dev/null',tofile='b/'+w+'/'+name))
        imports=n.read(PARENT/f'native_source_imports_{w}.json')
        for name,digest in imports['native_imports'].items():
            assert name not in patches.CHANGED and manifest['files'][name]==digest==n.sha(files/name)
        # All old source and feature modules are byte-identical, except the
        # specifically enumerated request/driver integrations above.
        for name,digest in parent['manifest']['files'].items():
            if name not in patches.CHANGED:assert manifest['files'][name]==digest
        source_panel=project.ROOT/'train_panel.json' if w=='train' else ROOT/'oos_panel.json'
        panel_name='research/structural_v6/source_panel.json';install(source_panel,files/panel_name,n.sha(source_panel));manifest['files'][panel_name]=n.sha(files/panel_name)
        source=n.source_hash(files);manifest['source_generation_hash']=source
        study.update(code=str(files),manifest=str(directory/'inputs/manifest.json'),source_generation_hash=source,spec_path='research/structural_v6/control_spec.json')
        study['configuration']['input_manifest_path']=study['manifest']
        n.write(files/study['spec_path'],{k:v for k,v in study.items() if k not in ('code','manifest','source_generation_hash','spec_path')})
        manifest['files'][study['spec_path']]=n.sha(files/study['spec_path'])
        n.write(study['manifest'],{**manifest,'manifest_hash':n.sig(manifest)})
        write(directory/'ready.json',dict(files=str(files),study=study,manifest=manifest,parent_ready_sha256=n.sha(PARENT/'native_engines'/w/'ready.json')))
        bundle=OUT/'review_bundle'/w;bundle.mkdir(parents=True,exist_ok=True);(bundle/'implementation.patch').write_text(''.join(diff),encoding='utf-8')
        write(ROOT/f'source_reuse_{w}.json',dict(parent_ready_sha256=n.sha(PARENT/'native_engines'/w/'ready.json'),
            original_source_proof_sha256=n.sha(PARENT/f'native_source_preflight_{w}.json'),source_parser_imports=imports['native_imports'],
            source_files_exact=True,unchanged_entry_snapshot_sha256=manifest['files'][study['olr_snapshots']],
            exact_changes=list(patches.CHANGED),new_files=list(patches.NEW),source_generation_hash=source))
    train=n.read(project.ROOT/'train_panel.json');oos=n.read(ROOT/'oos_panel.json')
    semantic=lambda row:{k:v for k,v in row.items() if k!='source_hash'}
    assert [semantic(r) for r in train['rows']]==[semantic(r) for r in oos['rows'] if r['day']<='2026-03-31']
    write(ROOT/'preparation_complete.json',dict(utc=s.utc(),sources={w:n.read(t.ENGINES/w/'ready.json')['study']['source_generation_hash'] for w in ('train','oos')},
        complete_pre_april_source_prefix_identical=True,per_case_source_projection_recompute_required=True,no_performance_executed=True))

def test_and_freeze():
    configure();launch=verify_launch();assert (ROOT/'preparation_complete.json').exists()
    command=[sys.executable,'-B',str(project.REPO/'scripts/olr_structural_native_test_v6.py'),str(ROOT)]
    child(command,ROOT/'tests.log',ROOT/'tests_process.json',cwd=t.ENGINES/'train/inputs/files')
    import xml.etree.ElementTree as ET
    suites=list(ET.parse(ROOT/'tests.xml').getroot().iter('testsuite'))
    counts={k:sum(int(v.get(k,'0')) for v in suites) for k in ('tests','failures','errors','skipped')}
    assert counts['tests']>=40 and not any(counts[k] for k in ('failures','errors','skipped'))
    artifacts={str(ROOT/nm):n.sha(ROOT/nm) for nm in ('registration.json','candidate_plan.json','protocol.json','mechanism_plan.json','preparation_complete.json','source_reuse_train.json','source_reuse_oos.json','tests.xml','tests.log','source_projection_contract.json','oos_source_worker.py')}
    for w in ('train','oos'):
        ready=t.ENGINES/w/'ready.json';artifacts[str(ready)]=n.sha(ready)
        value=n.read(ready);assert n.source_hash(Path(value['files']))==value['study']['source_generation_hash']
        for name,digest in value['manifest']['files'].items():assert n.sha(Path(value['files'])/name)==digest,name
    write(ROOT/'frozen_harness.json',dict(utc=s.utc(),files=launch['files'],artifacts=artifacts,tests=counts,before_performance=True))

def verify_frozen():
    configure();value=n.read(ROOT/'frozen_harness.json')
    for name,digest in value['files'].items():assert n.sha(project.REPO/name)==digest,name
    for name,digest in value['artifacts'].items():assert n.sha(name)==digest,name
    return value

def evaluate(c,w,scenario,capped):
    verify_frozen();out=b.output(c,w,scenario,capped)
    if (out/'completion.json').exists():raise RuntimeError('Authenticated resume must be explicit; never duplicate a completed replay')
    registered=n.prepare_case(ROOT,w,c,100_000_000);study=copy.deepcopy(n.read(registered));files=Path(study['code'])
    manifest=n.read(study['manifest']);manifest.pop('manifest_hash');tag=scenario+('_capped' if capped else '')
    prefix='research/structural_v6/runs/'+c['id']+'/'+w+'/'+tag+'/'
    contract=prefix+'liquidity_contract.json';n.write(files/contract,dict(mode=c['structural_mode'],panel='research/structural_v6/source_panel.json',version='structural-liquidity-v6'))
    manifest['files'][contract]=n.sha(files/contract);study['structural_liquidity_contract']=contract
    if w=='oos':
        prior=b.output(c,'train',scenario,capped);s.forensic.authenticate(prior)
        for key,name in (('capacity_checkpoint','capacity_terminal_state.json'),('capacity_checkpoint_completion','completion.json'),('capacity_checkpoint_result','result.json')):
            target=files/(prefix+name);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(prior/name,target)
            study[key]=target.relative_to(files).as_posix();manifest['files'][study[key]]=n.sha(target)
    runtime=ROOT/'runtime'/c['id']/w/tag;runtime.mkdir(parents=True,exist_ok=True)
    study.update(manifest=str(files.parent/('structural_'+c['id']+'_'+w+'_'+tag+'.manifest.json')),spec_path=prefix+'spec.json')
    study['configuration']['input_manifest_path']=study['manifest']
    n.write(files/study['spec_path'],{k:v for k,v in study.items() if k not in ('code','manifest','spec_path','source_generation_hash')})
    manifest['files'][study['spec_path']]=n.sha(files/study['spec_path']);n.write(study['manifest'],{**manifest,'manifest_hash':n.sig(manifest)})
    study_path=runtime/'study.json';n.write(study_path,study)
    command=[sys.executable,'-B','-u',str(files/'scripts/portfolio_synergy_study.py'),'replay','--study',str(study_path),'--recipe','candidate','--scenario',scenario,'--standalone','OLR','--output',str(out),'--full-audit']
    if capped:command.append('--no-upward-compounding')
    child(command,out.with_suffix('.log'),runtime/'process.json',cwd=files)
    s.forensic.authenticate(out);result=n.read(out/'result.json');audit=n.read(out/'execution_audit.json')
    assert result['execution_validity_pass'] and audit['audit_pass'] and result['audit_status']=='passed' and not result['terminal_inventory'] and not result['terminal_pending']
    assert n.read(out/'structural_projection_audit.json')['passed'] and n.read(out/'structural_request_audit.json')['passed']
    clock=audit['audit_receipt']['minute_clock_audit'];assert clock['status']=='pass' and not clock['errors'] and clock['control_or_treatment']=='clock_control'
    n.verify_window_contract(ROOT,w,study,out)
    result.update(qualified=True,candidate_id=c['id'],label=c['label'],window=w,output=str(out),history_policy='literal_carry')
    write(out/'assessment.json',result)
    if c['structural_mode']=='off':
        old_id=n.read(PARENT/'candidate_plan.json')['candidates'][0]['id'];old=PARENT/'cases'/old_id/w/('100000000_'+tag)
        s.forensic.authenticate(old)
        for file in ('daily.json','equity.json'):assert n.read(out/file)==n.read(old/file),'Control parity: '+file
        fields=('symbol','qty','entry_decision_time','entry_fill_time','entry_price','exit_fill_time','exit_price','gross_pnl','commission','net_pnl','exit_reason','mfe','mae','r_multiple')
        econ=lambda f:[{k:t.get(k) for k in fields} for t in n.read(f/'trades.json')]
        assert econ(out)==econ(old),'Control trade economic parity'
        write(out/'control_parity.json',dict(passed=True,original=str(old),every_intraday_and_daily_mark_equal=True,trade_economics_equal=True))
    print(dict(candidate=c['label'],window=w,scenario=scenario,capped=capped,return_pct=result['return_pct']),flush=True)

def describe(c,w,scenario,capped):
    folder=b.output(c,w,scenario,capped);row=metrics.describe(folder);nav=n.read(folder/'minute_nav_evidence.json')
    row.update(drawdown_pct=nav['common_original_drawdown_pct'],complete_fine_drawdown_pct=nav['all_event_drawdown_pct'],
        source_audit=n.read(folder/'structural_projection_audit.json'),request_audit=n.read(folder/'structural_request_audit.json'))
    return row

def train_checks(catalog):
    parent=n.read(PARENT_OUT/'results.json');old=parent['references']['retained'];refs=parent['references'];g=n.read(ROOT/'registration.json')['prior_gates'];rows={};checks={}
    for scenario,capped,group in GROUPS:
        ca=describe(catalog[0],'train',scenario,capped);a=describe(catalog[1],'train',scenario,capped)
        rows[group]={'control':ca,'migration':a}
        tests=dict(matched_is_return=a['return_pct']>=ca['return_pct']-1.5,
                   is_selected_return=a['return_pct']>=old[group]['train']['return_pct']-1.5)
        if group!='capped':
            original=refs['original_stress' if group=='stress' else 'original']['train'];guard=refs['guard'][group]['train']
            tests.update(is_original_return=a['return_pct']>=original['return_pct']-g['is_return_loss_max_pp_original'],
                is_guard_return=a['return_pct']>=guard['return_pct']-g['is_return_loss_max_pp_guard'],
                is_guard_drawdown=a['drawdown_pct']<=guard['drawdown_pct']+g['is_dd_increase_max_pp_guard'])
        checks[group]=tests
    result=dict(utc=s.utc(),rows=rows,checks=checks,oos_eligible=all(all(x.values()) for x in checks.values()),
                registration_sha256=n.sha(ROOT/'registration.json'),full_objective_qualified=False,promotion=False)
    write(OUT/'training_decision.json',result);return result

def qualify():
    configure();verify_frozen();catalog=n.read(ROOT/'candidate_plan.json')['candidates']
    for scenario,capped,_ in GROUPS:
        for c in catalog:evaluate(c,'train',scenario,capped)
    decision=train_checks(catalog)
    if decision['oos_eligible']:
        for scenario,capped,_ in GROUPS:
            for c in catalog:evaluate(c,'oos',scenario,capped)
    write(ROOT/'qualification_complete.json',dict(utc=s.utc(),cases=12 if decision['oos_eligible'] else 6,
        training_decision_sha256=n.sha(OUT/'training_decision.json'),oos_evaluated=decision['oos_eligible'],report_review_remaining=True))

def run():
    configure();verify_launch()
    for phase in ('prepare','freeze','qualify'):
        child([sys.executable,'-B','-u','-m','scripts.olr_structural_native_v6',phase],ROOT/(phase+'.controller.log'),ROOT/(phase+'_process.json'))
    write(ROOT/'controller_complete.json',dict(utc=s.utc(),qualification_complete_sha256=n.sha(ROOT/'qualification_complete.json'),report_review_remaining=True))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['register','prepare','freeze','qualify','run']);args=ap.parse_args();configure()
    try:{'register':register,'prepare':prepare,'freeze':test_and_freeze,'qualify':qualify,'run':run}[args.phase]()
    except Exception:
        path=ROOT/(args.phase+'_failure.json')
        if not path.exists():write(path,dict(utc=s.utc(),traceback=traceback.format_exc()))
        raise
