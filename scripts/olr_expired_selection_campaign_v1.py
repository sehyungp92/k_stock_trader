"""0039 four-account nominal access diagnostic; immutable private generations."""
from __future__ import annotations
import argparse
from collections import defaultdict
import copy
import gzip
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
import psutil
import olr_selective_half_continuation_v1 as shared
from olr_relative_context_campaign_v1 import private_text

REPO=shared.REPO
BASE=Path('E:/CodexResearch/k_stock_trader/olr_continuing_research_20261016')
PARENT=BASE/'relative_qualification_v2'
read,sha,write,status,now,sig=shared.read,shared.sha,shared.write,shared.status,shared.now,shared.sig
CASES=[('train','control'),('train','access'),('oos','control'),('oos','access')]
HELPERS=['olr_selective_half_continuation_v1.py','olr_relative_context_campaign_v1.py',
    'olr_expired_selection_rule_v1.py','olr_expired_selection_tests_v1.py','olr_relative_context_assessment_v1.py']


def patch(files,manifest):
    for target,source in [('strategy_olr/core/expired_selection.py','olr_expired_selection_rule_v1.py'),
                          ('tests/strategy_olr/test_expired_selection.py','olr_expired_selection_tests_v1.py')]:
        private_text(files,target,(REPO/'scripts'/source).read_text(encoding='utf8'),manifest)
    def change(name,before,after):
        text=(files/name).read_text(encoding='utf8');assert text.count(before)==1,(name,before)
        private_text(files,name,text.replace(before,after),manifest)
    change('strategy_olr/core/allocation.py','    for bar in sorted(bars, key=lambda b:b.symbol):\n',
        '    from .expired_selection import prepare as prepare_expired_selection\n    prepare_expired_selection(state,bars,config,snapshot)\n    for bar in sorted(bars, key=lambda b:b.symbol):\n')
    change('strategy_olr/core/allocation.py','    while True:\n',
        "    # Preserve the complete original allocation, including native top-ups.\n"
        "    expired_intents = [i for i in intents if by_id[i.intent_id].metadata.get('expired_selection_v1')]\n"
        "    intents = [i for i in intents if i not in expired_intents]\n"
        "    while True:\n")
    change('strategy_olr/core/allocation.py','    pre_joint_quantities=None\n',
        "    if expired_intents:\n"
        "        original_used = sum(d.reserved_notional for d in decisions)\n"
        "        residual = replace(account, funding_limit=max(0,account.funding_limit-original_used),\n"
        "            gross_committed=account.gross_committed+original_used,\n"
        "            strategy_committed={'OLR':account.strategy_committed['OLR']+original_used})\n"
        "        expired_decisions = allocate_with_loss_budget(expired_intents,residual,at,config,state,by_id,prior=decisions)\n"
        "        native_approved.update({d.intent_id:d.approved_qty for d in expired_decisions})\n"
        "        intents += expired_intents\n"
        "        decisions += expired_decisions\n"
        "    pre_joint_quantities=None\n")
    change('strategy_olr/core/logic.py','    if structural_proposal is None:\n        if len(selected)',
        "    from .expired_selection import proposal as expired_selection_proposal\n    extra = expired_selection_proposal(state,bar,snapshot,candidate) if structural_proposal is None else None\n    if extra is not None:\n        weight, extra_signal, extra_evidence = extra\n    elif structural_proposal is None:\n        if len(selected)")
    change('strategy_olr/core/logic.py','signal = structural_signal if structural_proposal is not None else find_olr_entry_signal',
        'signal = extra_signal if extra is not None else structural_signal if structural_proposal is not None else find_olr_entry_signal')
    change('strategy_olr/core/logic.py','        **capacity_metadata,\n',
        '        **capacity_metadata,\n        **({"expired_selection_v1":extra_evidence} if extra is not None else {}),\n')
    anchor='        result = step_olr_batch(self.state, selected, self.config, snapshot, self._portfolio_view(broker),\n'
    change('backtests/strategies/olr/runner.py',anchor,
        "        if getattr(self,'expired_selection_enabled_v1',False):\n            self.state.meta['expired_selection_enabled_v1']=True\n"+anchor)
    anchor='    if args.standalone:\n        adapters = tuple(a for a in adapters if a.strategy_id == args.standalone)\n'
    change('scripts/portfolio_synergy_study.py',anchor,anchor+
        "    if study.get('expired_selection_settings'):\n        settings=bound('expired_selection_settings')\n        if settings.get('version')!='expired-selection-v1' or type(settings.get('enabled')) is not bool:\n            raise ValueError('Unregistered expired selection settings')\n        next(a for a in adapters if a.strategy_id=='OLR').expired_selection_enabled_v1=settings['enabled']\n")


def reach(root):
    # No future payoff. Source predicates and contemporaneous acknowledged ledger only.
    rows=read(BASE/'opportunity_continuity_v2/train_rows.json')
    ready=read(PARENT/'native_engines/train/ready.json')
    snapshots=read(Path(ready['files'])/ready['study']['olr_snapshots'])
    candidates={(s['trade_date'],c['symbol']):c for s in snapshots for c in s['candidates']}
    days=defaultdict(list)
    for r in rows:days[r['day']].append(r)
    output=[]
    for day,group in days.items():
        original=sorted((r for r in group if r['rank']<=6),key=lambda r:r['rank'])
        if not original:continue
        expired=[]
        for r in original:
            z=r['timeline'][-1];b=z['book']
            if (z['source_complete'] and not r['first_confirmation'] and not r['proposals']
                and not b['quantities'].get(r['symbol']) and not any(o['symbol']==r['symbol'] and o['side']=='BUY' for o in b['pending'])):
                expired.append(r['symbol'])
        extras=[]
        for r in sorted(group,key=lambda r:r['rank']):
            z=r['timeline'][-1];b=z['book']
            if (r['rank']>6 and z['source_complete'] and z['passes'] and not b['quantities'].get(r['symbol'])
                and not any(o['symbol']==r['symbol'] and o['side']=='BUY' for o in b['pending'])):
                extras.append(r)
        for r in extras[:len(expired)]:
            z=r['timeline'][-1];b=z['book'];c=candidates[(day,r['symbol'])]
            weight=1.2*r['rank']**-1.5/sum(o['rank']**-1.5 for o in original)
            price=z['close']*1.015;unit=price*1.0007;risk=max(c['daily_atr'],z['close']*.005,1.)
            loss=1.25*risk+.0034*price
            limits=dict(weight=math.floor(b['equity']*weight/unit),cash=math.floor(max(0,b['deployable'])/unit),
                gross=math.floor(max(0,b['equity']-b['gross']-b['reserve'])/unit),
                sector=math.floor(max(0,.06*b['equity']-b['sector_risk'].get(c['sector'],0))/loss),
                account=math.floor(max(0,.08*b['equity']-sum(b['risk'].values()))/loss),
                name=math.floor(.04*b['equity']/loss))
            q=min(limits.values())
            output.append(dict(day=day,symbol=r['symbol'],at=z['at'],expired=expired,rank=r['rank'],
                original_weight=weight,qty_upper_bound=q,notional_upper_bound=q*price,limits=limits))
    receipt=dict(at=now(),rows=output,dates=len(set(r['day'] for r in output)),
        meaningful_upper_bounds=sum(r['notional_upper_bound']>=1e6 for r in output),
        source_sha256=sha(BASE/'opportunity_continuity_v2/train_rows.json'),
        scope='Incumbent after-cohort cash/loss upper bounds; extra-to-extra competition, fresh native capacity and physical fill remain decisive in funded replay. No payoff inspected.')
    write(root/'reach.json',receipt)
    assert receipt['dates']>=2 and receipt['meaningful_upper_bounds']>=1,'Insufficient predeclared diagnostic reach'


def verify(root):
    p=read(root/'protocol.json');assert sha(__file__)==p['runner_sha256']
    assert sha(root/'design_at_freeze.md')==p['design_sha256']
    for name,h in p['helpers'].items():assert sha(REPO/'scripts'/name)==h,name
    return p


def prepare(root):
    verify(root);reach(root)
    sources={}
    for window in ('train','oos'):
        status(root,'preparing_private_source',window=window)
        old=read(PARENT/'native_engines'/window/'ready.json');origin=Path(old['files'])
        files=root/'native_engines'/window/'files';files.mkdir(parents=True,exist_ok=False)
        m=copy.deepcopy(old['manifest']);m.pop('manifest_hash',None)
        # Inputs are the accepted immutable generation. Hash the large market databases once
        # here; prior accepted audits bind the unchanged remainder, also checked by native replay.
        bindings={key:sha(origin/old['study'][key]) for key in ('database','olr_snapshots','olr_request','minute_source_definition')}
        for key,h in bindings.items():assert h==m['files'][old['study'][key]],key
        for name,h in m['files'].items():
            dst=files/name;dst.parent.mkdir(parents=True,exist_ok=True);os.link(origin/name,dst)
        if window=='train':source_hash,completed_case,_=shared.frozen_imports(root)
        assert source_hash(files)==source_hash(origin)==old['study']['source_generation_hash']
        patch(files,m)
        source=source_hash(files);s=copy.deepcopy(old['study']);s.update(code=str(files),source_generation_hash=source)
        for key in ('economic_policy_settings','drawdown_reserve_contract','structural_entry_settings'):s.pop(key,None)
        m['source_generation_hash']=source
        write(root/'native_engines'/window/'ready.json',dict(files=str(files),study=s,manifest=m,interval_qualification=old['interval_qualification']))
        sources[window]=dict(source=source,parent=old['study']['source_generation_hash'],input_hashes=bindings,
            changes={n:h for n,h in m['files'].items() if h!=old['manifest']['files'].get(n)})
        assert source_hash(origin)==old['study']['source_generation_hash']
    tests=['tests/strategy_olr/test_expired_selection.py','tests/strategy_olr/test_capacity_allocation.py',
        'tests/strategy_olr/test_precision_v3.py','tests/strategy_olr/test_opening_risk_v2.py',
        'tests/strategy_olr/test_structural_service.py','tests/strategy_olr/test_economic_evidence.py']
    report=root/'focused_tests.xml';files=Path(read(root/'native_engines/train/ready.json')['files'])
    cmd=[sys.executable,'-B','-m','pytest',*tests,'-q','-p','no:cacheprovider','--basetemp='+str(root/'tmp/pytest'),'--junitxml='+str(report)]
    result=subprocess.run(cmd,cwd=files,env=shared.env_for(root),capture_output=True,text=True,creationflags=0x08000000)
    (root/'focused_tests.log').write_text(result.stdout+result.stderr,encoding='utf8')
    assert result.returncode==0,'Preserve failed preflight: '+str(root/'focused_tests.log')
    suite=ET.parse(report).getroot().find('testsuite');assert all(suite.get(k)=='0' for k in ('failures','errors','skipped'))
    write(root/'preflight.json',dict(at=now(),passed=True,sources=sources,tests=int(suite.get('tests')),
        test_report_sha256=sha(report),reach_sha256=sha(root/'reach.json'),protocol_sha256=sha(root/'protocol.json'),
        source_scope='Unchanged corrected window calendars/native source definitions. Tests cover altered strategy path; full original source audit remains mandatory per account.'))


def launch(root,window,policy):
    ready=read(root/'native_engines'/window/'ready.json');files=Path(ready['files'])
    s,m=copy.deepcopy(ready['study']),copy.deepcopy(ready['manifest']);out=root/'cases'/window/policy
    prefix=f'research/expired_selection/{window}/{policy}/'
    values=[('structural_settings','structural.json',dict(mode='control',interval_service=True)),
        ('funded_probe_settings','probe.json',dict(version='first-native-fade-funded-probe-v1',action='hold',control_reference=None)),
        ('funded_population_settings','population.json',dict(version='monthly-native-funded-population-v1',cell=None,catalog_reference=None)),
        ('relative_context_settings','context.json',dict(fraction=0.,version='relative-context-v1')),
        ('expired_selection_settings','selection.json',dict(enabled=policy=='access',version='expired-selection-v1'))]
    for key,name,value in values:
        s[key]=prefix+name;write(files/s[key],value);m['files'][s[key]]=sha(files/s[key])
    if window=='oos':
        prior=root/'cases/train'/policy
        for key,name in [('capacity_checkpoint','capacity_terminal_state.json'),('capacity_checkpoint_completion','completion.json'),('capacity_checkpoint_result','result.json')]:
            s[key]=prefix+'parent_'+name;(files/s[key]).write_bytes((prior/name).read_bytes());m['files'][s[key]]=sha(files/s[key])
    s['candidates']['candidate']['olr_capacity'].update(capacity_structural_partial=False,capacity_weak_partial_fraction=0.)
    m['evidence']['qualification_path']=ready['interval_qualification']
    s['manifest']=str(files.parent/(policy+'.manifest.json'));s['configuration']['input_manifest_path']=s['manifest']
    s['spec_path']=prefix+'spec.json';write(files/s['spec_path'],{k:v for k,v in s.items() if k not in ('code','manifest','spec_path','source_generation_hash')})
    m['files'][s['spec_path']]=sha(files/s['spec_path']);write(s['manifest'],dict(**m,manifest_hash=sig(m)))
    sp=root/'runtime'/window/policy/'study.json';write(sp,s)
    cmd=[sys.executable,'-B','-u',str(files/'scripts/portfolio_synergy_study.py'),'replay','--study',str(sp),'--recipe','candidate',
        '--scenario','baseline','--standalone','OLR','--output',str(out),'--full-audit']
    log=root/'logs'/window/(policy+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x',encoding='utf8') as f:
        p=subprocess.Popen(cmd,cwd=files,env=shared.env_for(root),stdout=f,stderr=subprocess.STDOUT,creationflags=0x08000000)
    write(sp.parent/'process.json',dict(at=now(),pid=p.pid,created_at=psutil.Process(p.pid).create_time(),command=cmd,log=str(log),case=str(out)))
    return p,out


def run(root):
    verify(root);prepare(root)
    source_hash,completed_case,contrast=shared.frozen_imports(root)
    summaries={};parent=read(PARENT/'assessment.json')['summaries']
    for i,(window,policy) in enumerate(CASES):
        verify(root)
        ready=read(root/'native_engines'/window/'ready.json')
        assert source_hash(Path(ready['files']))==ready['study']['source_generation_hash']
        status(root,'running',case=[window,policy],accepted=list(summaries),queued=CASES[i+1:])
        p,case=launch(root,window,policy);assert p.wait()==0,'Preserve failed financial worker: '+str(case)
        valid=completed_case(case);assert valid['valid'],valid
        assert read(case/'funded_checkpoint_receipt.json')['no_action_continuation_equal']
        assert read(case/'funded_population_receipt.json')['audit']['pass_']
        if policy=='control':
            reference=Path(parent[window+'_nominal_control']['case'])
            checks={n:read(case/n)==read(reference/n) for n in ('equity.json','daily.json','trades.json','minute_nav_evidence.json')}
            write(root/(window+'_disabled_parity.json'),dict(at=now(),checks=checks,passed=all(checks.values())))
            assert all(checks.values()),'Disabled economic parity failed'
        value=shared.economic_summary(case)
        value['selection_decisions']=[]
        with gzip.open(case/'native_decisions.jsonl.gz','rt',encoding='utf8') as stream:
            for line in stream:
                row=json.loads(line)
                if row.get('metadata',{}).get('expired_selection_v1'):value['selection_decisions'].append(row)
        if (window,policy)==('train','access'):assert value['selection_decisions'],'No treatment activation'
        summaries[window+'_'+policy]=value
        write(root/'accepted'/(window+'_'+policy+'.json'),dict(**value,completion_sha256=sha(case/'completion.json')))
        status(root,'case_accepted',case=[window,policy],accepted=list(summaries),queued=CASES[i+1:])
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
    root.mkdir(parents=True,exist_ok=False)
    design=REPO/'docs/optimisation/olr/experiments/olr-exp-0039.md'
    (root/'design_at_freeze.md').write_bytes(design.read_bytes())
    (root/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    write(root/'protocol.json',dict(at=now(),experiment='0039',runner_sha256=sha(__file__),design_sha256=sha(design),
        helpers={n:sha(REPO/'scripts'/n) for n in HELPERS},cases=CASES,parent_assessment_sha256=sha(PARENT/'assessment.json'),
        parent_ready={w:sha(PARENT/'native_engines'/w/'ready.json') for w in ('train','oos')}))
    cmd=[sys.executable,'-X','utf8','-B','-u',str(Path(__file__).resolve()),'run',str(root)]
    with (root/'controller.log').open('x',encoding='utf8') as f:
        p=subprocess.Popen(cmd,cwd=REPO,env=shared.env_for(root),stdout=f,stderr=subprocess.STDOUT,creationflags=0x08000000)
    value=dict(at=now(),pid=p.pid,created_at=psutil.Process(p.pid).create_time(),command=cmd)
    write(root/'controller_process.json',value);print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['start','run']);p.add_argument('root',type=Path);a=p.parse_args()
    try:globals()[a.mode](a.root)
    except Exception:
        status(a.root,'needs_repair',error=traceback.format_exc());raise
