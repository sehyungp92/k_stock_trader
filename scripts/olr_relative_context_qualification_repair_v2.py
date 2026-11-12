"""0036 calendar-only assembly repair; reuse six accepted IS accounts, finish six OOS."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET

import psutil
import olr_relative_context_qualification_v1 as base

shared = base.shared
read, sha, sig, write, status, now = base.read, base.sha, base.sig, base.write, base.status, base.now
REPO = base.REPO
OLD = base.ROOT
ROOT = OLD.with_name('relative_qualification_v2')
CASES = [x for x in base.CASES if x[0] == 'oos']
CALENDARS = ['kis_core/data/krx_intraday_halts.yaml','kis_core/data/krx_sessions.yaml']
HELPERS = base.HELPERS + ['olr_relative_context_qualification_v1.py','olr_qualification_calendar_preflight_v2.py']


def verify(root):
    p = read(root/'protocol.json')
    assert sha(__file__) == p['runner_sha256']
    assert sha(root/'design_at_freeze.md') == p['design_sha256']
    assert sha(root/'repair_design_at_freeze.md') == p['repair_design_sha256']
    assert sha(OLD/'protocol.json') == p['parent_protocol_sha256']
    assert sha(OLD/'preflight.json') == p['parent_preflight_sha256']
    for name,h in p['helpers'].items():
        assert sha(REPO/'scripts'/name) == h
    return p


def run_preflight_process(root, command, cwd, name):
    log = root/(name+'.log')
    with log.open('x',encoding='utf8') as stream:
        p = subprocess.Popen(command,cwd=cwd,env=shared.env_for(root),stdout=stream,
            stderr=subprocess.STDOUT,creationflags=0x08000000)
        write(root/(name+'_process.json'),dict(at=now(),pid=p.pid,
            created_at=psutil.Process(p.pid).create_time(),command=command,log=str(log)))
        assert p.wait() == 0, 'Preserve failed preparation: '+str(log)


def prepare(root):
    protocol = verify(root)
    status(root,'preparing_calendar_repair',queued=CASES)
    assert read(OLD/'status.json')['phase'] == 'needs_repair'
    for path in [OLD/'controller_process.json',*OLD.glob('runtime/*/*/process.json')]:
        receipt = read(path)
        try:
            p = psutil.Process(receipt['pid'])
            assert abs(p.create_time()-receipt['created_at'])>.001 or p.cmdline()!=receipt['command'], 'Parent worker is still active'
        except psutil.NoSuchProcess:
            pass
    original = read(base.OOS/'native_engines/oos/ready.json')
    assert sha(base.OOS/'native_engines/oos/ready.json') == protocol['oos_ready_sha256']
    sources, changes = {}, []
    for window in ['train','oos']:
        ready = copy.deepcopy(read(OLD/'native_engines'/window/'ready.json'))
        origin, files = Path(ready['files']), root/'native_engines'/window/'files'
        files.mkdir(parents=True,exist_ok=False)
        manifest = ready['manifest']; manifest.pop('manifest_hash',None)
        for name,h in manifest['files'].items():
            base.install(origin/name,files/name,h,authenticate=False)
        if window == 'train':
            source_hash, completed_case, _ = shared.frozen_imports(root)
            assert source_hash(files) == source_hash(origin) == base.SOURCE
        else:
            for name in CALENDARS:
                before = manifest['files'][name]; expected = original['manifest']['files'][name]
                assert before != expected
                base.install(Path(original['files'])/name,files/name,expected)
                manifest['files'][name] = expected
                changes.append(dict(path=name,invalid_v1_sha256=before,corrected_sha256=expected))
            # Prove the repair did not mutate any strategy/engine Python byte.
            for name,h in manifest['files'].items():
                if name.endswith('.py'):
                    assert h == ready['manifest']['files'][name] == sha(files/name) == sha(origin/name)
            original_manifest = read(OLD/'native_engines/oos/ready.json')['manifest']['files']
            differences = [n for n,h in manifest['files'].items() if h != original_manifest.get(n)]
            assert sorted(differences) == sorted(CALENDARS)
        sources[window] = source_hash(files)
        ready['files'] = str(files)
        ready['study'].update(code=str(files),source_generation_hash=sources[window])
        manifest['source_generation_hash'] = sources[window]
        write(root/'native_engines'/window/'ready.json',ready)
        if window == 'oos':
            write(root/'native_engines/oos/calendar_preflight.manifest.json',dict(**manifest,manifest_hash=sig(manifest)))
    assert sources['train'] != sources['oos']
    # Reuse accepted IS files byte-for-byte; no new IS account is launched.
    for scenario in ['stress','capped']:
        for policy in ['control','full']:
            oldcase = OLD/'cases/train'/(policy+'_'+scenario)
            assert completed_case(oldcase)['valid']
            case = root/'cases/train'/(policy+'_'+scenario)
            case.mkdir(parents=True)
            for p in oldcase.iterdir():
                if p.is_file(): base.install(p,case/p.name,sha(p))
            entry = read(OLD/'accepted'/('train_'+scenario+'_'+policy+'.json'))
            assert sha(oldcase/'completion.json') == entry['completion_sha256']
            write(root/'accepted'/('train_'+scenario+'_'+policy+'.json'),entry)
    for scenario in ['nominal','stress','capped']:
        for p in (OLD/'pairs'/('train_'+scenario)).rglob('*'):
            if p.is_file():
                base.install(p,root/'pairs'/('train_'+scenario)/p.relative_to(OLD/'pairs'/('train_'+scenario)),sha(p))
    write(root/'repair_receipt.json',dict(at=now(),source_by_window=sources,changes=changes,
        python_unchanged=True,policy_unchanged=True,criteria_unchanged=True,accepted_is_reused=True,
        previous_preflight_sha256=sha(OLD/'preflight.json'),oos_input_receipt_sha256=sha(OLD/'oos_input_receipt.json')))
    files = root/'native_engines/oos/files'
    status(root,'verifying_oos_calendar_sources',queued=CASES)
    program = "import runpy,sys;sys.argv=['calendar',"+repr(str(root))+"];runpy.run_path("+repr(str(REPO/'scripts/olr_qualification_calendar_preflight_v2.py'))+",run_name='__main__')"
    run_preflight_process(root,[sys.executable,'-B','-u','-c',program],files,'calendar_preflight')
    tests=['tests/strategy_olr/test_relative_context.py','tests/strategy_olr/test_precision_v3.py',
        'tests/strategy_olr/test_opening_risk_v2.py','tests/strategy_olr/test_structural_service.py',
        'tests/strategy_olr/test_economic_evidence.py','tests/strategy_olr/test_minute_exit_v5.py',
        'tests/strategy_olr/test_decision_continuation.py']
    status(root,'testing_corrected_calendar_environment',queued=CASES)
    report=root/'focused_tests.xml'
    run_preflight_process(root,[sys.executable,'-B','-m','pytest',*tests,'-q','-p','no:cacheprovider',
        '--basetemp='+str(root/'tmp/pytest'),'--junitxml='+str(report)],files,'focused_tests')
    suite=ET.parse(report).getroot().find('testsuite')
    assert suite is not None and all(suite.get(k)=='0' for k in ['failures','errors','skipped'])
    assert read(root/'calendar_preflight.json')['pass_']
    assert all(source_hash(root/'native_engines'/w/'files')==s for w,s in sources.items())
    write(root/'preflight.json',dict(at=now(),passed=True,tests=int(suite.get('tests')),
        tests_sha256=sha(report),calendar_preflight_sha256=sha(root/'calendar_preflight.json'),
        repair_receipt_sha256=sha(root/'repair_receipt.json'),source_by_window=sources,
        protocol_sha256=sha(root/'protocol.json'),remaining_accounts=CASES))
    return sources


def execute(root):
    sources=prepare(root)
    source_hash, completed_case, contrast=shared.frozen_imports(root)
    summaries,comparisons={},{}
    for scenario in ['nominal','stress','capped']:
        pair=read(root/'pairs'/('train_'+scenario)/'assessment.json')
        comparisons['train_'+scenario]=pair['comparisons']['control_to_full']
        for policy in ['control','full']:
            summary=pair['summaries'][policy]
            assert completed_case(Path(summary['case']))['valid']
            assert sha(Path(summary['case'])/'completion.json')==summary['completion_sha256']
            summaries['train_'+scenario+'_'+policy]=summary
    for index,(window,scenario,policy) in enumerate(CASES):
        verify(root)
        assert source_hash(root/'native_engines'/window/'files')==sources[window]
        key=window+'_'+scenario+'_'+policy
        status(root,'running',case=key,accepted=list(summaries),queued=CASES[index+1:])
        p,case=base.launch(root,window,scenario,policy)
        assert p.wait()==0,'Native replay failed: '+str(case)
        assert completed_case(case)['valid']
        assert read(case/'funded_checkpoint_receipt.json')['no_action_continuation_equal']
        assert read(case/'funded_population_receipt.json')['audit']['pass_']
        restart=read(case/'capacity_restart_receipt.json')
        assert restart['same_configuration_and_scenario'] and restart['all_samples_verified_against_oos_retained_ohlcv']
        summaries[key]=base.summarize(case)
        if policy=='control': assert not summaries[key]['context_decisions']
        write(root/'accepted'/(key+'.json'),summaries[key])
        if policy=='full': comparisons[window+'_'+scenario]=base.assess_pair(root,window,scenario,summaries,contrast)
        status(root,'case_accepted',case=key,accepted=list(summaries),queued=CASES[index+1:])
    activity={w:base.participation_value(root/'pairs'/(w+'_nominal')) for w in ['train','oos']}
    checks={w+'_nominal_material_return':comparisons[w+'_nominal']['delta']['return_pct']>=.5 for w in ['train','oos']}
    for scenario in ['stress','capped']:
        values=[comparisons[w+'_'+scenario]['delta']['return_pct'] for w in ['train','oos']]
        checks[scenario+'_return_support']=sum(values)>0 and min(values)>=-.5
    changes=[comparisons[w+'_nominal']['delta']['meaningful'] for w in ['train','oos']]
    checks['meaningful_participation']=sum(changes)>=2 and changes[1]>=-1
    checks['positive_changed_participation_value']=sum(v['net_pnl_attribution'] for v in activity.values())>0
    write(root/'assessment.json',dict(at=now(),summaries=summaries,comparisons=comparisons,activity=activity,
        necessary_checks=checks,necessary_checks_pass=all(checks.values()),economic_objective_achieved=False,
        next='Apply original 0036 substantive risk/recovery and economic disposition; no automatic redesign.'))
    status(root,'awaiting_qualification_disposition',accepted=list(summaries),queued=[],necessary_checks=checks)


def start(root):
    root.mkdir(parents=True,exist_ok=False)
    original=read(OLD/'protocol.json')
    assert sha(REPO/'scripts/olr_relative_context_qualification_v1.py')==original['runner_sha256']
    (root/'design_at_freeze.md').write_bytes((OLD/'design_at_freeze.md').read_bytes())
    design=REPO/'docs/optimisation/olr/experiments/olr-exp-0036.md'
    (root/'repair_design_at_freeze.md').write_bytes(design.read_bytes())
    write(root/'protocol.json',dict(at=now(),experiment='olr-exp-0036',repair='OOS calendar assembly only',
        design_sha256=sha(root/'design_at_freeze.md'),repair_design_sha256=sha(design),runner_sha256=sha(__file__),
        helpers={n:sha(REPO/'scripts'/n) for n in HELPERS},parent_protocol_sha256=sha(OLD/'protocol.json'),
        parent_preflight_sha256=sha(OLD/'preflight.json'),oos_ready_sha256=sha(base.OOS/'native_engines/oos/ready.json'),
        cases=CASES,reused_is_accounts=6,concurrency=1,original_planned_wall_hours=6,new_data=False,
        source_calendar_files=CALENDARS))
    for name in [Path(__file__).name,*HELPERS]:
        target=root/'launch_sources'/name;target.parent.mkdir(exist_ok=True)
        target.write_bytes((REPO/'scripts'/name).read_bytes())
    command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'execute','--root',str(root)]
    with (root/'controller.log').open('x',encoding='utf8') as stream:
        p=subprocess.Popen(command,cwd=REPO,env=shared.env_for(root),stdout=stream,stderr=subprocess.STDOUT,creationflags=0x08000000)
    identity=dict(at=now(),pid=p.pid,created_at=psutil.Process(p.pid).create_time(),command=command,log=str(root/'controller.log'))
    write(root/'controller_process.json',identity);print(json.dumps(identity),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['start','execute'])
    parser.add_argument('--root',type=Path,default=ROOT);args=parser.parse_args()
    try: globals()[args.phase](args.root)
    except Exception as exc:
        if args.root.exists():status(args.root,'needs_repair',error=repr(exc),traceback=traceback.format_exc())
        raise
