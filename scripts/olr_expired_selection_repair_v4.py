"""0039 acceptance-receipt repair; reuse valid V3 control and unchanged policy."""
import argparse
from collections import defaultdict
import copy
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
import psutil
import olr_expired_selection_campaign_v1 as base

shared=base.shared
read,write,sha,status,now=base.read,base.write,base.sha,base.status,base.now
OLD=base.BASE/'expired_selection_v3'
HELPERS=base.HELPERS+['olr_expired_selection_campaign_v1.py']
REMAINING=base.CASES[1:]


def verify(root):
    p=read(root/'protocol.json')
    assert sha(__file__)==p['runner_sha256']
    for name,h in p['helpers'].items():assert sha(base.REPO/'scripts'/name)==h,name
    for name,h in p['parent_bindings'].items():assert sha(OLD/name)==h,name
    assert sha(root/'design_at_freeze.md')==p['design_sha256']
    assert sha(root/'repair_design_at_freeze.md')==p['repair_design_sha256']
    base.verify(OLD)
    return p


def accept(root,window,policy,case,completed_case):
    valid=completed_case(case);assert valid['valid'],valid
    completion=read(case/'completion.json')
    for name,h in completion['artifacts'].items():assert sha(case/name)==h,name
    assert read(case/'funded_checkpoint_receipt.json')['no_action_continuation_equal']
    assert read(case/'funded_population_receipt.json')['audit']['pass_']
    if window=='oos':
        restart=read(case/'capacity_restart_receipt.json')
        assert restart['same_configuration_and_scenario']
        assert restart['all_samples_verified_against_oos_retained_ohlcv']
    if policy=='control':
        reference=Path(read(base.PARENT/'assessment.json')['summaries'][window+'_nominal_control']['case'])
        checks={n:read(case/n)==read(reference/n) for n in ('equity.json','daily.json','trades.json','minute_nav_evidence.json')}
        assert all(checks.values()),checks
        write(root/(window+'_disabled_parity.json'),dict(at=now(),checks=checks,passed=True,reference=str(reference)))
    value=shared.economic_summary(case)
    # The shared summary owns this field. Verify it, then serialize exactly once.
    assert value['completion_sha256']==sha(case/'completion.json')
    value['selection_decisions']=[]
    with gzip.open(case/'native_decisions.jsonl.gz','rt',encoding='utf8') as stream:
        for line in stream:
            row=json.loads(line)
            if row.get('metadata',{}).get('expired_selection_v1'):value['selection_decisions'].append(row)
    if policy=='control':assert not value['selection_decisions']
    if (window,policy)==('train','access'):assert value['selection_decisions'],'No treatment activation'
    path=root/'accepted'/(window+'_'+policy+'.json')
    write(path,value)
    assert read(path)==value and sha(case/'completion.json')==read(path)['completion_sha256']
    return value


def prepare(root):
    verify(root)
    assert read(OLD/'status.json')['phase']=='needs_repair'
    identities=[]
    for path in [OLD/'controller_process.json',OLD/'review_dispatch_process.json',*OLD.glob('runtime/*/*/process.json')]:
        receipt=read(path)
        try:
            p=psutil.Process(receipt['pid'])
            live=abs(p.create_time()-receipt['created_at'])<.001 and p.cmdline()==receipt['command']
        except psutil.NoSuchProcess:live=False
        assert not live,('Parent process still active',str(path))
        identities.append(dict(receipt=str(path),authenticated_live=False))
    oldpre=read(OLD/'preflight.json')
    assert oldpre['passed'] and oldpre['tests']==81
    assert sha(OLD/'protocol.json')==oldpre['protocol_sha256']
    assert sha(OLD/'focused_tests.xml')==oldpre['test_report_sha256']
    assert sha(OLD/'reach.json')==oldpre['reach_sha256']
    suite=ET.parse(OLD/'focused_tests.xml').getroot().find('testsuite')
    assert int(suite.get('tests'))==81 and all(suite.get(k)=='0' for k in ('failures','errors','skipped'))
    sources={}
    for window in ('train','oos'):
        status(root,'authenticating_unchanged_source',window=window,queued=REMAINING)
        ready=copy.deepcopy(read(OLD/'native_engines'/window/'ready.json'))
        origin=Path(ready['files']);files=root/'native_engines'/window/'files'
        files.mkdir(parents=True,exist_ok=False)
        for name,h in ready['manifest']['files'].items():
            dst=files/name;dst.parent.mkdir(parents=True,exist_ok=True);os.link(origin/name,dst)
            if name.endswith('.py'):assert sha(dst)==h,name
        if window=='train':source_hash,completed_case,_=shared.frozen_imports(root)
        source=source_hash(files)
        assert source==source_hash(origin)==ready['study']['source_generation_hash']==oldpre['sources'][window]['source']
        for key,h in oldpre['sources'][window]['input_hashes'].items():
            assert sha(files/ready['study'][key])==h,key
        ready['files']=str(files);ready['study']['code']=str(files)
        write(root/'native_engines'/window/'ready.json',ready)
        sources[window]=source
    # Copy no mutable runtime: only completed output files, retained byte-for-byte.
    oldcase=OLD/'cases/train/control';case=root/'cases/train/control'
    assert completed_case(oldcase)['valid']
    case.mkdir(parents=True,exist_ok=False)
    for path in oldcase.iterdir():
        if path.is_file():os.link(path,case/path.name)
    assert sha(case/'completion.json')==sha(oldcase/'completion.json')
    value=accept(root,'train','control',case,completed_case)
    for name in ('reach.json','focused_tests.xml','focused_tests.log'):
        (root/name).write_bytes((OLD/name).read_bytes())
    write(root/'repair_receipt.json',dict(at=now(),repair='Acceptance serialization only; no duplicate completion_sha256 keyword',
        prior_status_sha256=sha(OLD/'status.json'),prior_process_checks=identities,
        source_by_window=sources,trading_code_unchanged=True,criteria_unchanged=True,
        reused_control=str(oldcase),control_completion_sha256=value['completion_sha256'],
        reverified_control_artifacts=len(read(case/'completion.json')['artifacts']),
        actual_acceptance_roundtrip_passed=True,remaining_accounts=REMAINING))
    write(root/'preflight.json',dict(at=now(),passed=True,sources=sources,tests_reused=81,
        reuse_basis='All private Python bytes, aggregate source, test report and critical data hashes authenticated; no trading/source change.',
        test_report_sha256=sha(root/'focused_tests.xml'),reach_sha256=sha(root/'reach.json'),
        repair_receipt_sha256=sha(root/'repair_receipt.json'),protocol_sha256=sha(root/'protocol.json'),
        control_parity_sha256=sha(root/'train_disabled_parity.json'),acceptance_roundtrip_passed=True))
    return {'train_control':value}


def run(root):
    summaries=prepare(root)
    source_hash,completed_case,contrast=shared.frozen_imports(root)
    for i,(window,policy) in enumerate(REMAINING):
        verify(root)
        ready=read(root/'native_engines'/window/'ready.json')
        assert source_hash(Path(ready['files']))==ready['study']['source_generation_hash']
        status(root,'running',case=[window,policy],accepted=list(summaries),queued=REMAINING[i+1:])
        p,case=base.launch(root,window,policy)
        assert p.wait()==0,'Preserve failed financial worker: '+str(case)
        summaries[window+'_'+policy]=accept(root,window,policy,case,completed_case)
        status(root,'case_accepted',case=[window,policy],accepted=list(summaries),queued=REMAINING[i+1:])
    comparisons={}
    for window in ('train','oos'):
        a,b=[summaries[window+'_'+p] for p in ('control','access')]
        account=contrast(read(Path(a['case'])/'trades.json'),read(Path(b['case'])/'trades.json'))
        dates=defaultdict(float)
        for r in account['per_opportunity']:dates[r['day']]+=r['delta_pnl']
        comparisons[window]=dict(delta={k:b[k]-a[k] for k in ('return_pct','full_dd_pct','common_dd_pct','maximum_currency_decline',
            'meaningful','meaningful_days','buy_notional','winning_wealth','losing_wealth','longest_underwater_calendar_days','mean_economic_event_exposure')},
            accounting=account,entry_date_attribution=dict(dates),excluding_best_date=sum(dates.values())-max(dates.values(),default=0))
    write(root/'assessment.json',dict(at=now(),summaries=summaries,comparisons=comparisons,economic_objective_achieved=False,
        next='Complete source/decision/order/lifecycle and funding attribution, then assess principal readiness. No automatic new account or variant.'))
    status(root,'awaiting_economic_interpretation',accepted=list(summaries),queued=[])


def start(root):
    base.verify(OLD)
    root.mkdir(parents=True,exist_ok=False)
    design=base.REPO/'docs/optimisation/olr/experiments/olr-exp-0039.md'
    (root/'design_at_freeze.md').write_bytes((OLD/'design_at_freeze.md').read_bytes())
    (root/'repair_design_at_freeze.md').write_bytes(design.read_bytes())
    bindings=['protocol.json','preflight.json','focused_tests.xml','reach.json','status.json','train_disabled_parity.json',
        'native_engines/train/ready.json','native_engines/oos/ready.json','cases/train/control/completion.json']
    write(root/'protocol.json',dict(at=now(),experiment='0039',repair='Receipt serialization; reuse valid IS control',
        runner_sha256=sha(__file__),helpers={n:sha(base.REPO/'scripts'/n) for n in HELPERS},
        parent_bindings={n:sha(OLD/n) for n in bindings},design_sha256=sha(root/'design_at_freeze.md'),
        repair_design_sha256=sha(design),cases=base.CASES,remaining_accounts=REMAINING,reused_accounts=1,
        concurrency=1,new_data=False,policy_unchanged=True,criteria_unchanged=True,remaining_planned_minutes=90))
    for name in [Path(__file__).name,*HELPERS]:
        target=root/'launch_sources'/name;target.parent.mkdir(exist_ok=True)
        target.write_bytes((base.REPO/'scripts'/name).read_bytes())
    cmd=[sys.executable,'-X','utf8','-B','-u',str(Path(__file__).resolve()),'run',str(root)]
    with (root/'controller.log').open('x',encoding='utf8') as stream:
        p=subprocess.Popen(cmd,cwd=base.REPO,env=shared.env_for(root),stdout=stream,stderr=subprocess.STDOUT,creationflags=0x08000000)
    identity=dict(at=now(),pid=p.pid,created_at=psutil.Process(p.pid).create_time(),command=cmd)
    write(root/'controller_process.json',identity);print(json.dumps(identity))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['start','run']);p.add_argument('root',type=Path);a=p.parse_args()
    try:globals()[a.mode](a.root)
    except Exception:
        status(a.root,'needs_repair',error=traceback.format_exc());raise
