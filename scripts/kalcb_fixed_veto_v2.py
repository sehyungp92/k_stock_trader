"""C13 R1 historical profile resolution; trading policy and matrix unchanged."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import traceback
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify,digest,source_hash,env
from backtests.strategies.kalcb.late_entry_io_v5 import cached_link
from backtests.auto.shared.cache_keys import stable_signature
from backtests.strategies.kalcb.research_process_v1 import execute
from backtests.strategies.kalcb import capital_sensitivity_diagnostics_v1 as diag
from scripts.kalcb_capital_sensitivity_v2 import classify_case
from scripts.kalcb_opening_veto_qualification_v1 import useful_days
from scripts.kalcb_opportunity_interaction_v1 import episode_traces
from scripts.kalcb_retained_funding_v1 import status

BASE=Path('E:/CodexResearch/k_stock_trader')
C6=BASE/'kalcb_capital_sensitivity_20261016/v2'
C10=BASE/'kalcb_opening_veto_qualification_20261016/v3'
C12=BASE/'kalcb_opportunity_lifecycle_20261016'
DESIGN=REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0037.md'
WORKER='scripts/kalcb_fixed_veto_worker_v1.py'
SCENARIOS=('baseline','participation','cost','combined')


def no_producer():
    import psutil
    for process in psutil.process_iter(['pid','name','cmdline']):
        if process.pid==os.getpid() or 'python' not in (process.info['name'] or '').lower():continue
        if any('kalcb' in a.lower() and ('worker' in a.lower() or 'kalcb_fixed_veto_v' in a.lower() or 'kalcb_opportunity_interaction_v1' in a.lower()) for a in process.info['cmdline'] or []):
            raise ValueError('Another producer '+str(process.pid))


def cases(capitals,controls):
    result=[dict(id='bridge_fixed',scenario='baseline',initial_equity=capitals['baseline'],mode='fixed',horizon_mode='disabled',kind='bridge',bridge=controls['fixed']['baseline']['folder']),
            dict(id='bridge_veto',scenario='baseline',initial_equity=capitals['baseline'],mode='compound',horizon_mode='veto',kind='bridge',bridge=controls['veto']['baseline']['folder'])]
    result += [dict(id='joint_'+s,scenario=s,initial_equity=capitals[s],mode='fixed',horizon_mode='veto',kind='treatment') for s in SCENARIOS]
    return result


def plan_for(oldplan,oldcontext,case,root,dest):
    p=deepcopy(oldplan);context=deepcopy(oldcontext)
    for path in (p['audit_configuration'],p['capital_sensitivity']['protocol']['path']):p['input_sha256'].pop(path,None)
    context.update(initial_equity=case['initial_equity'],output_root=str(root/'cases'/case['id']),promotion_status='c13_diagnostic_only')
    write(dest/'audit_configuration.json',context)
    p['audit_configuration']=str(dest/'audit_configuration.json')
    p.pop('fixed_allocation_reference',None)
    p['capital_sensitivity']={**case,'reference':100000000.,'protocol':bound(root/'protocol.json')}
    for path in (dest/'audit_configuration.json',root/'protocol.json'):p['input_sha256'][str(path)]=digest(path)
    write(dest/'plan.json',p)
    return p


def freeze(root,p):
    old=p['parent'];verify([old[k] for k in ('manifest','plan','policy')])
    mp=Path(old['manifest']['path']);parent=read(mp);sig=parent.pop('manifest_hash')
    if stable_signature(parent)!=sig:raise ValueError('Parent manifest mismatch')
    folder=root/'runtime';code=folder/'inputs/files';files={}
    for n,(rel,sha) in enumerate(parent['files'].items()):
        cached_link(mp.parent/'files'/rel,code/rel,sha,root/'content_cache');files[rel]=sha
        if n%500==0:status(root,'source_authentication',checked=n+1,total=len(parent['files']))
    if WORKER in files:raise ValueError('Do not overwrite a frozen worker')
    cached_link(REPO/WORKER,code/WORKER,digest(REPO/WORKER),root/'content_cache');files[WORKER]=digest(REPO/WORKER)
    oldplan=read(old['plan']['path']);oldcontext=read(oldplan['audit_configuration']);profiles={}
    for case in p['cases']:
        dest=folder/'plans'/case['id'];plan_for(oldplan,oldcontext,case,root,dest)
        for name in ('plan.json','audit_configuration.json'):
            rel='research/fixed_veto/'+case['id']+'/'+name
            cached_link(dest/name,code/rel,digest(dest/name),root/'content_cache');files[rel]=digest(dest/name)
        profiles[case['id']]=dict(name=case['id'],period='oos',folder=str(folder),plan=bound(dest/'plan.json'),policy=bound(code/'research/kalcb_executed_value/policies/policies_manifest.json'))
    rel='research/fixed_veto/protocol.json';cached_link(root/'protocol.json',code/rel,digest(root/'protocol.json'),root/'content_cache');files[rel]=digest(root/'protocol.json')
    payload=deepcopy(parent);payload['files']=files;payload['source_generation_hash']=source_hash(code)
    payload['evidence']['capital_sensitivity']=dict(protocol=bound(root/'protocol.json'),parent_manifest=old['manifest'],full_continuous_state_claim=False,core_engine_policy_changes=False)
    write(folder/'inputs/manifest.json',{**payload,'manifest_hash':stable_signature(payload)})
    for item in profiles.values():
        item.update(manifest=bound(folder/'inputs/manifest.json'),source_generation_hash=payload['source_generation_hash'])
        if item['policy']['sha256']!=old['policy']['sha256']:raise ValueError('Changed inherited policy')
    return profiles


def command(item,out):
    return [sys.executable,'-B','-u','-m',WORKER[:-3].replace('/','.'),'--plan',item['plan']['path'],'--policies',str(Path(item['policy']['path']).parent),'--audit-manifest',item['manifest']['path'],'--output',str(out)]


PROBE='''import json
from scripts import kalcb_executed_value_replay as replay
from scripts import kalcb_fixed_veto_worker_v1 as worker
from backtests.strategies.kalcb.runner import KALCBReplayAdapter
from strategy_kalcb.core import account_allocation_v1 as allocation
from backtests.strategies.kalcb import executed_value as ev
class ReachedReplay(Exception):pass
def sentinel(*a,**kw):raise ReachedReplay()
original=KALCBReplayAdapter._portfolio_view;quantity=allocation.allocate_quantity
replay.run=sentinel
try:worker.main()
except ReachedReplay:pass
else:raise AssertionError('Real replay boundary not reached')
assert KALCBReplayAdapter._portfolio_view is original and allocation.allocate_quantity is quantity and ev.REFERENCE_EQUITY==100000000
print(json.dumps(dict(reached_replay_boundary=True,financial_replay_executed=False,hooks_restored=True)))
'''


def preflight(root,profiles):
    results={}
    for name,item in profiles.items():
        code=Path(item['manifest']['path']).parent/'files'
        cmd=[sys.executable,'-B','-c',PROBE]+command(item,root/'preflight'/name)[5:]
        r=subprocess.run(cmd,cwd=code,env=env(code),capture_output=True,text=True)
        write(root/'preflight_logs'/(name+'.json'),dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
        if r.returncode:raise RuntimeError('Frozen full-worker preflight failed '+name)
        result=__import__('json').loads(r.stdout.strip().splitlines()[-1])
        if not result['reached_replay_boundary'] or result['financial_replay_executed'] or not result['hooks_restored']:raise ValueError('Invalid probe')
        if (root/'preflight'/name/'completion.json').exists():raise ValueError('Probe ran an account')
        results[name]=result
    write(root/'runtime_preflight.json',results)


def registered_profile(arm,scenario,folder):
    folder=Path(folder)
    if arm in ('prior','fixed'):
        name=('compound_' if arm=='prior' else 'fixed_')+scenario
        item=read(C6.parent/'v1/registration.json')['profiles'][name]
    elif arm=='veto':
        item=read(C10/'registration.json')['capital_profiles']['capital_'+scenario]
    else:raise ValueError('Unknown control arm')
    verify([item[k] for k in ('manifest','plan','policy')])
    if (folder/'input_profile.json').exists() and read(folder/'input_profile.json')!=item:
        raise ValueError('Per-case profile contradicts original registration')
    return item


def register(root,tests):
    no_producer();test=read(tests)
    if test['exit_code']:raise ValueError('Readiness tests failed')
    verify(test['bindings']);root.mkdir(parents=True,exist_ok=False);(root/'temp').mkdir()
    os.environ['TEMP']=os.environ['TMP']=str(root/'temp');status(root,'authenticating_controls')
    a=read(C6/'assessment.json');b=read(C10/'assessment.json');c10p=read(C10/'protocol.json');reg=read(C10/'registration.json')
    inputs=[bound(DESIGN),bound(tests),bound(C6.parent/'v1/registration.json'),bound(C10/'registration.json'),bound(REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0037-repair-r1.md'),bound(BASE/'kalcb_fixed_veto_20261017/v1/failure.json')]+test['bindings']
    for base in (C6,C10,C12/'funded_interaction_v1',C12/'allocation_intersection_v1'):
        done=read(base/'completion.json');verify([done['assessment']]);inputs += [bound(base/'completion.json'),done['assessment']]
    controls={arm:{} for arm in ('prior','fixed','veto')}
    for s in SCENARIOS:
        for arm,name in [('prior','compound_'+s),('fixed','fixed_'+s)]:
            row=a['case_ledger'][name];folder=Path(row['folder']);controls[arm][s]=dict(folder=str(folder),qualified=row['qualified'])
        folder=C10/'cases'/('capital_'+s);controls['veto'][s]=dict(folder=str(folder),qualified=b['capital'][s]['verdict']['qualified'])
        for arm in controls:
            row=controls[arm][s];folder=Path(row['folder']);item=registered_profile(arm,s,folder)
            verdict=classify_case(folder,item,save=False)
            if verdict['qualified']!=row['qualified']:raise ValueError('Changed predecessor verdict')
            row['verdict']=verdict
            if verdict['qualified'] and not diag.funding_review(folder,c10p['capitals'][s])['verified']:raise ValueError('Control funding rejected')
            inputs.extend(bound(path) for path in folder.iterdir() if path.is_file())
    programs=[bound(REPO/f) for f in (WORKER,'scripts/kalcb_fixed_veto_v2.py','tests/test_kalcb_fixed_veto_v1.py','tests/test_kalcb_fixed_veto_v2.py',
        'backtests/strategies/kalcb/fixed_allocation_reference_v1.py','backtests/strategies/kalcb/funded_horizon_v1.py',
        'backtests/strategies/kalcb/capital_sensitivity_diagnostics_v1.py','backtests/strategies/kalcb/late_artifact_validation_v2.py',
        'scripts/kalcb_capital_sensitivity_v2.py','scripts/kalcb_opportunity_interaction_v1.py')]
    p=dict(experiment='kalcb-exp-0037',criteria='C13',inputs=inputs,programs=programs,controls=controls,
        capitals=c10p['capitals'],parent=reg['capital_profiles']['bridge_veto'],databases=c10p['databases'],
        signal_map=c10p['signal_map'],cases=cases(c10p['capitals'],controls),native_accounts=6,fits=0,principal_qualification=False)
    write(root/'protocol.json',p);write(root/'completed_cases.json',[])
    verify(inputs+programs+p['databases']);profiles=freeze(root,p);status(root,'full_worker_preflight');preflight(root,profiles)
    verify(inputs+programs+p['databases'])
    write(root/'registration.json',dict(protocol=bound(root/'protocol.json'),profiles=profiles,preflight=bound(root/'runtime_preflight.json')))


def compare(folder,control,initial):
    folder,control=Path(folder),Path(control)
    a,b=diag.account(folder,initial),diag.account(control,initial)
    aa,bb=diag.opportunities(folder,initial),diag.opportunities(control,initial)
    return dict(account=a,control=b,metrics=diag.metrics(folder,initial),control_metrics=diag.metrics(control,initial),
        activity=aa,control_activity=bb,path=diag.paired_attribution(a,b),activity_delta=diag.compare(aa,bb),
        useful_days=useful_days(aa),control_useful_days=useful_days(bb),net_delta_krw=a['net_pnl_krw']-b['net_pnl_krw'])


def assess(root):
    p=read(root/'protocol.json');ledger=read(root/'completed_cases.json');cells={}
    for row in ledger:
        c=row['case'];s=c['scenario']
        if c['kind']!='treatment':continue
        cell=dict(verdict=row['verdict'],initial_equity=c['initial_equity'],comparisons={},unqualified_controls={})
        if row['verdict']['qualified']:
            folder=root/'cases'/c['id']
            for arm,items in p['controls'].items():
                if not items[s]['qualified']:cell['unqualified_controls'][arm]=items[s]['verdict'];continue
                cell['comparisons'][arm]=compare(folder,items[s]['folder'],c['initial_equity'])
                write(root/'traces'/(s+'_versus_'+arm+'.json'),episode_traces(folder,items[s]['folder']))
            if set(cell['comparisons'])=={'prior','fixed','veto'}:
                component=compare(p['controls']['fixed'][s]['folder'],p['controls']['prior'][s]['folder'],c['initial_equity'])
                cell['interaction_net_krw']=cell['comparisons']['veto']['net_delta_krw']-component['net_delta_krw']
        cells[s]=cell
    fixed=[c['comparisons']['fixed'] for c in cells.values() if 'fixed' in c['comparisons']]
    prior=[c['comparisons']['prior'] for c in cells.values() if 'prior' in c['comparisons']]
    criteria=dict(all_joint_accounts_qualified=len(fixed)==4,principal_qualification=False,economic_objective_achieved=False)
    if len(fixed)==4:criteria['median_net_vs_fixed']=median(c['net_delta_krw'] for c in fixed)
    if len(prior)==3:
        criteria['median_net_vs_qualified_prior']=median(c['net_delta_krw'] for c in prior)
        criteria['median_useful_vs_qualified_prior']=median(c['useful_days']-c['control_useful_days'] for c in prior)
    write(root/'assessment.json',dict(cells=cells,criteria=criteria,human_economic_assessment_required=True,
        scope='C13 diagnostic; flat restart, reused OOS, no matched comparison against unqualified controls. No automatic successor.'))


def run(root):
    p=read(root/'protocol.json');r=read(root/'registration.json');verify([r['protocol'],r['preflight']]+p['inputs']+p['programs']+p['databases'])
    for i,c in enumerate(p['cases']):
        if c['kind']=='treatment':
            for name in ('bridge_fixed','bridge_veto'):
                if not read(root/'cases'/name/'identity_bridge.json')['passed']:raise ValueError('Both bridges required')
                if not any(x['case']['id']==name and x['verdict']['qualified'] for x in read(root/'completed_cases.json')):raise ValueError('Unaccepted bridge')
        item=r['profiles'][c['id']];folder=root/'cases'/c['id'];code=Path(item['manifest']['path']).parent/'files'
        execute(root,command(item,folder),c['id'],code,dict(case=c,cases_complete=i,cases_total=6))
        write(folder/'input_profile.json',item);verdict=classify_case(folder,item,save=True)
        if verdict['qualified']:write(folder/'funding_review.json',diag.funding_review(folder,c['initial_equity']))
        else:write(folder/'qualification_rejection.json',verdict)
        actions=read(folder/'horizon_actions.json');allocation=read(folder/'allocation_reference.json')
        if (c['horizon_mode']=='disabled' and actions) or (c['horizon_mode']=='veto' and not actions):raise ValueError('VETO activation mismatch')
        if allocation['enabled']!=(c['mode']=='fixed') or allocation['ledger_cash_mutations'] or allocation['external_capital_flows']:raise ValueError('Allocation activation mismatch')
        write(folder/'activation.json',dict(passed=True,mode=c['mode'],horizon_mode=c['horizon_mode'],veto_actions=len(actions),allocation_observations=len(allocation['rows'])))
        ledger=read(root/'completed_cases.json');ledger.append(dict(case=c,verdict=verdict,completion=bound(folder/'completion.json')));write(root/'completed_cases.json',ledger)
    status(root,'independent_assessment');assess(root);verify(p['inputs']+p['programs']+p['databases'])
    write(root/'completion.json',dict(complete=True,registration=bound(root/'registration.json'),assessment=bound(root/'assessment.json'),native_accounts=6,economic_objective_achieved=False))
    status(root,'diagnostic_assessed_disposition_pending')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--tests',type=Path,required=True);args=parser.parse_args()
    try:register(args.output,args.tests);run(args.output)
    except Exception as error:
        if args.output.exists():write(args.output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        raise
