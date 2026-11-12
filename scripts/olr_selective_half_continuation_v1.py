"""O3 prospective robustness extension; immutable research engine, no live edits."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from collections import defaultdict
import psutil

REPO = Path(__file__).resolve().parents[1]
BASE = Path('E:/CodexResearch/k_stock_trader/olr_structural_execution_20261013')
PARENT = BASE / 'generation_v12'
ROOT = Path('E:/CodexResearch/k_stock_trader/olr_continuing_research_20261016/selective_half_v1')
CASES = [('partial_nominal', True, 'baseline', False),
         ('control_stress', False, 'combined', False), ('partial_stress', True, 'combined', False),
         ('control_capped', False, 'baseline', True), ('partial_capped', True, 'baseline', True)]


def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def sig(v): return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',',':'), default=str).encode()).hexdigest()
def write(p, v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f: json.dump(v,f,indent=2,default=str,allow_nan=False)
def status(root, phase, **kw):
    value=dict(at=now(),phase=phase,economic_objective_achieved=False,**kw)
    tmp=root/'status.tmp';tmp.write_text(json.dumps(value,indent=2),encoding='utf8');os.replace(tmp,root/'status.json')
    print(json.dumps(value),flush=True)
def env_for(root):
    tmp=root/'tmp';tmp.mkdir(exist_ok=True)
    return dict(os.environ,TEMP=str(tmp),TMP=str(tmp),PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf8',
                OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
def frozen_imports(root):
    files=root/'native_engines/train/files'
    sys.path.insert(0,str(files))
    from scripts.olr_economic_study import source_hash
    from backtests.strategies.olr.economic_evidence import completed_case
    from scripts.olr_economic_assessment import contrast
    assert Path(sys.modules['scripts.olr_economic_study'].__file__).resolve().is_relative_to(files)
    return source_hash, completed_case, contrast


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    source=read(PARENT/'native_engines/train/ready.json');src=Path(source['files'])
    dest=root/'native_engines/train/files';dest.mkdir(parents=True)
    for number,name in enumerate(source['manifest']['files']):
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);os.link(src/name,target)
        if number%2000==0: print(json.dumps(dict(phase='linking_frozen_inputs',files=number)),flush=True)
    manifest=copy.deepcopy(source['manifest']);manifest.pop('manifest_hash',None)
    study=copy.deepcopy(source['study']);study['code']=str(dest)
    ready=dict(files=str(dest),study=study,manifest=manifest,interval_qualification=source['interval_qualification'])
    write(root/'native_engines/train/ready.json',ready)
    source_hash,completed_case,_=frozen_imports(root)
    generation=source_hash(dest)
    assert generation==source_hash(src)==study['source_generation_hash']==read(PARENT/'preflight.json')['source']
    reference=BASE/'generation_v3/cases/train/structural_partial_baseline'
    for p in [reference,PARENT/'cases/train/native_control']:
        assert completed_case(p)['valid']
    spec=REPO/'docs/optimisation/olr/experiments/olr-exp-0032.md'
    protocol=dict(at=now(),experiment='olr-exp-0032',objective='O3',criteria='C5',source=generation,
        design=str(spec),design_sha256=sha(spec),runner=str(Path(__file__)),runner_sha256=sha(__file__),
        nominal_reference=str(reference),nominal_completion_sha256=sha(reference/'completion.json'),
        cases=CASES,concurrency=1,planned_minutes=120,new_market_data=False,live_promotion=False,
        advancement='Both new paired return deltas >0, at least one >=0.5pp, followed by joint risk/recovery review. No automatic OOS or promotion.')
    write(root/'protocol.json',protocol)
    write(root/'native_engines/train/freeze_receipt.json',dict(at=now(),parent=str(PARENT),source=generation,
        parent_ready_sha256=sha(PARENT/'native_engines/train/ready.json'),private_source_changes=[],
        strategy_source_unchanged=True,files=len(manifest['files'])))
    # Frozen tests exercise the actual half-rule, expiry/priority, source service,
    # full account checkpoint and evidence acceptance. No workspace imports.
    tests=['tests/strategy_olr/test_precision_v3.py','tests/strategy_olr/test_opening_risk_v2.py',
           'tests/strategy_olr/test_structural_service.py','tests/strategy_olr/test_economic_evidence.py']
    report=root/'focused_tests.xml'
    args=tests+['-q','-p','no:cacheprovider','--basetemp='+str(root/'tmp/pytest'),'--junitxml='+str(report)]
    code='import pathlib,pytest,strategy_olr.core.weak_partial as m;assert pathlib.Path(m.__file__).resolve().is_relative_to(pathlib.Path.cwd());raise SystemExit(pytest.main('+repr(args)+'))'
    command=[sys.executable,'-B','-c',code]
    result=subprocess.run(command,cwd=dest,env=env_for(root),capture_output=True,text=True)
    (root/'focused_tests.log').write_text(result.stdout+result.stderr,encoding='utf8')
    if result.returncode: raise RuntimeError('Frozen readiness tests failed; preserve generation. '+str(root/'focused_tests.log'))
    import xml.etree.ElementTree as ET
    suite=ET.parse(report).getroot().find('testsuite')
    assert suite is not None and all(suite.get(k)=='0' for k in ('failures','errors','skipped'))
    assert source_hash(dest)==generation
    write(root/'preflight.json',dict(at=now(),source=generation,tests_passed=True,tests=int(suite.get('tests')),
        test_report=str(report),test_report_sha256=sha(report),protocol_sha256=sha(root/'protocol.json'),
        runner_sha256=sha(__file__),source_unchanged=True))
    status(root,'ready_for_nominal_parity',tests=int(suite.get('tests')),queued=[r[0] for r in CASES])


def launch_case(root, tag, partial, scenario, capped):
    ready=read(root/'native_engines/train/ready.json');files=Path(ready['files'])
    s=copy.deepcopy(ready['study']);m=copy.deepcopy(ready['manifest']);out=root/'cases/train'/tag
    if out.exists():raise FileExistsError('Never overwrite case '+str(out))
    prefix='research/olr_o3_selective_half/'+tag+'/'
    values=[('structural_settings','structural.json',dict(mode='control',interval_service=True)),
            ('funded_probe_settings','probe.json',dict(version='first-native-fade-funded-probe-v1',action='hold',control_reference=None)),
            ('funded_population_settings','population.json',dict(version='monthly-native-funded-population-v1',cell=None,catalog_reference=None))]
    for key,name,value in values:
        s[key]=prefix+name;write(files/s[key],value);m['files'][s[key]]=sha(files/s[key])
    s.pop('economic_policy_settings',None);s.pop('drawdown_reserve_contract',None)
    s['candidates']['candidate']['olr_capacity'].update(capacity_structural_partial=partial,capacity_weak_partial_fraction=.5 if partial else 0.)
    m['evidence']['qualification_path']=ready['interval_qualification']
    s['manifest']=str(files.parent/(tag+'.manifest.json'));s['configuration']['input_manifest_path']=s['manifest']
    s['spec_path']=prefix+'spec.json'
    write(files/s['spec_path'],{k:v for k,v in s.items() if k not in ('code','manifest','spec_path','source_generation_hash')})
    m['files'][s['spec_path']]=sha(files/s['spec_path']);write(s['manifest'],{**m,'manifest_hash':sig(m)})
    sp=root/'runtime/train'/tag/'study.json';write(sp,s)
    cmd=[sys.executable,'-B','-u',str(files/'scripts/portfolio_synergy_study.py'),'replay','--study',str(sp),
         '--recipe','candidate','--scenario',scenario,'--standalone','OLR','--output',str(out),'--full-audit']
    if capped:cmd.append('--no-upward-compounding')
    log=root/'logs'/(tag+'.log');log.parent.mkdir(exist_ok=True)
    with log.open('x',encoding='utf8') as f:
        process=subprocess.Popen(cmd,cwd=files,env=env_for(root),stdout=f,stderr=subprocess.STDOUT,creationflags=0x08000000)
    identity=dict(pid=process.pid,created_at=psutil.Process(process.pid).create_time(),command=cmd,
                  log=str(log),case=str(out),started_at=now(),source=s['source_generation_hash'])
    write(sp.parent/'process.json',identity)
    return process,identity


def economic_summary(path):
    cash=high=100_000_000.;qty=defaultdict(int);marks={};points=[];full=currency=0.;pair=cpair=None
    exposure_sum=0.;observations=0;peak_at=None;spells=[];spell=None
    buy_notional=sell_notional=costs=0.;partial_orders=set();partial_fills=0;max_concentration=0.
    with gzip.open(path/'minute_events.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            row=json.loads(line);event=row['event']
            if event in ('opening_marks','interval_marks','closing_marks'):
                marks.update(row['prices']);stamp=row['timestamp']
            elif event=='fill':
                fill=row['fill'];stamp=fill['timestamp'];q=fill['qty'];price=fill['price'];fee=fill['metadata']['fee']
                assert math.isclose(cash,row['cash_before'],abs_tol=1e-5)
                sign=1 if fill['side']=='BUY' else -1
                cash-=sign*q*price+fee;qty[fill['symbol']]+=sign*q
                assert qty[fill['symbol']]>=0 and math.isclose(cash,row['cash_after'],abs_tol=1e-5)
                if sign==1:buy_notional+=q*price
                else:sell_notional+=q*price
                costs+=fee
                if fill['reason']=='weak_followthrough_partial':partial_orders.add(fill['order_id']);partial_fills+=q
            else:continue
            at=datetime.fromisoformat(stamp)+timedelta(minutes=event in ('interval_marks','closing_marks'))
            values=[q*marks[s] for s,q in qty.items() if q];gross=sum(values);nav=cash+gross
            points.append((at,nav));exposure_sum+=gross/nav;observations+=1
            max_concentration=max(max_concentration,max(values,default=0)/nav)
            if nav>=high:
                if spell is not None:
                    spells.append({**spell,'recovery_at':at.isoformat(),'calendar_days':(at-datetime.fromisoformat(spell['peak_at'])).total_seconds()/86400})
                    spell=None
                high=nav;peak_at=at.isoformat()
            elif spell is None:spell=dict(peak_at=peak_at,peak_equity=high)
            loss=high-nav;dd=100*loss/high
            value=dict(peak_at=peak_at,trough_at=at.isoformat(),peak_equity=high,trough_equity=nav,loss=loss,dd_pct=dd)
            if dd>full:full=dd;pair=value
            if loss>currency:currency=loss;cpair=value
    assert not any(qty.values())
    for episode in (pair,cpair):
        recovery=next((at for at,nav in points if at>datetime.fromisoformat(episode['trough_at']) and nav>=episode['peak_equity']),None)
        episode['recovery_at']=recovery.isoformat() if recovery else None
        episode['peak_to_recovery_calendar_days']=(recovery-datetime.fromisoformat(episode['peak_at'])).total_seconds()/86400 if recovery else None
    if spell:spells.append({**spell,'recovery_at':None,'calendar_days':(points[-1][0]-datetime.fromisoformat(spell['peak_at'])).total_seconds()/86400})
    trades=read(path/'trades.json');result=read(path/'result.json');nav=read(path/'minute_nav_evidence.json')
    assert math.isclose(cash,nav['final_equity'],abs_tol=1e-5) and math.isclose(full,nav['all_event_drawdown_pct'],abs_tol=1e-8)
    assert math.isclose(sum(t['net_pnl'] for t in trades),cash-1e8,abs_tol=.01)
    meaningful=[t for t in trades if t['qty']*t['entry_price']>=1e6]
    return dict(case=str(path),completion_sha256=sha(path/'completion.json'),return_pct=result['return_pct'],
        full_dd_pct=full,common_dd_pct=nav['common_original_drawdown_pct'],maximum_currency_decline=currency,
        percentage_episode=pair,currency_episode=cpair,longest_underwater_calendar_days=max(s['calendar_days'] for s in spells),
        underwater_spell_count=len(spells),unrecovered_at_end=bool(spell),
        trades=len(trades),days=result['active_days'],meaningful=len(meaningful),meaningful_days=len({t['entry_decision_time'][:10] for t in meaningful}),
        winning_wealth=sum(max(t['net_pnl'],0) for t in trades),losing_wealth=-sum(min(t['net_pnl'],0) for t in trades),
        buy_notional=buy_notional,sell_notional=sell_notional,fees_tax=costs,partial_orders=len(partial_orders),partial_shares=partial_fills,
        mean_economic_event_exposure=exposure_sum/observations,max_single_name_nav_share=max_concentration,
        exposure_note='Event-weighted descriptive exposure, not time-weighted occupancy. Full order/holding journals retained.')


def run(root):
    source_hash,completed_case,contrast=frozen_imports(root)
    pre=read(root/'preflight.json');protocol=read(root/'protocol.json')
    assert sha(__file__)==pre['runner_sha256']==protocol['runner_sha256']
    assert sha(root/'protocol.json')==pre['protocol_sha256']
    assert sha(pre['test_report'])==pre['test_report_sha256'] and pre['tests_passed']
    assert source_hash(root/'native_engines/train/files')==pre['source']
    summaries={}
    for i,args in enumerate(CASES):
        tag=args[0]
        status(root,'running',case=tag,accepted=list(summaries),queued=[x[0] for x in CASES[i+1:]])
        process,identity=launch_case(root,*args)
        if process.wait():raise RuntimeError('Native case failed: '+identity['log'])
        path=Path(identity['case']);valid=completed_case(path)
        checkpoint=read(path/'funded_checkpoint_receipt.json')
        assert valid['valid'] and checkpoint['no_action_continuation_equal']
        assert sha(path/'native_account_checkpoint.pkl')==checkpoint['sha256']
        assert read(path/'funded_population_receipt.json')['audit']['pass_']
        if i==0:
            parent=Path(protocol['nominal_reference'])
            fields=('equity.json','daily.json','trades.json','minute_nav_evidence.json')
            checks={name:read(path/name)==read(parent/name) for name in fields}
            write(root/'nominal_parity.json',dict(at=now(),checks=checks,pass_=all(checks.values()),
                new_completion=valid['receipt_sha256'],old_completion=sha(parent/'completion.json')))
            if not all(checks.values()):raise ValueError('Nominal partial reproduction mismatch; do not advance')
        summaries[tag]=economic_summary(path)
        write(root/'assessments'/(tag+'.json'),summaries[tag])
        status(root,'case_accepted',case=tag,accepted=list(summaries),queued=[x[0] for x in CASES[i+1:]])
    comparisons={}
    for scenario in ('stress','capped'):
        a,b=(summaries[k+'_'+scenario] for k in ('control','partial'))
        accounting=contrast(read(Path(a['case'])/'trades.json'),read(Path(b['case'])/'trades.json'))
        month=defaultdict(float);date=defaultdict(float)
        for row in accounting['per_opportunity']:
            month[row['day'][:7]]+=row['delta_pnl'];date[row['day']]+=row['delta_pnl']
        comparisons[scenario]=dict(delta={k:b[k]-a[k] for k in ('return_pct','full_dd_pct','common_dd_pct','maximum_currency_decline','meaningful','meaningful_days','buy_notional','winning_wealth','losing_wealth','longest_underwater_calendar_days')},
            accounting=accounting,entry_month_attribution=dict(month),excluding_best_entry_date=sum(date.values())-max(date.values()))
    gains=[c['delta']['return_pct'] for c in comparisons.values()]
    support=all(x>0 for x in gains) and max(gains)>=.5
    write(root/'assessment.json',dict(at=now(),summaries=summaries,comparisons=comparisons,
        return_advancement_screen_pass=support,economic_objective_achieved=False,
        next='Assess the registered campaign, attribution, monetary drawdown/recovery and meaningful activity. No automatic new experiment. Substantive redesign requires a failed assumption, new evidence and revised prospective design. Close an empty monitor.'))
    status(root,'awaiting_substantive_next_decision',accepted=list(summaries),queued=[],return_screen_pass=support,
        next='Evaluate assessment.json against the registered campaign. Report economic success separately from experiment completion. No automatic new research cycle.')


def start(root):
    if (root/'controller_process.json').exists():raise FileExistsError('Existing controller identity; inspect instead of duplicating')
    pre=read(root/'preflight.json');assert pre['tests_passed'] and sha(__file__)==pre['runner_sha256']
    log=root/'controller.log';command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'run','--root',str(root)]
    with log.open('x',encoding='utf8') as f:
        p=subprocess.Popen(command,cwd=REPO,env=env_for(root),stdout=f,stderr=subprocess.STDOUT,creationflags=0x08000000)
    value=dict(pid=p.pid,created_at=psutil.Process(p.pid).create_time(),command=command,log=str(log),at=now())
    write(root/'controller_process.json',value);print(json.dumps(value),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('prepare','start','run'));parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    if args.phase in ('start','run') and (args.root/'deferred_by_readiness_review.json').exists():
        parser.error('Campaign deferred by economic readiness review. Read deferred_by_readiness_review.json and olr-exp-0032.md; do not launch this frozen preparation.')
    try:globals()[args.phase](args.root)
    except BaseException as error:
        if args.root.exists():status(args.root,'needs_repair',error=repr(error),traceback=traceback.format_exc())
        raise
