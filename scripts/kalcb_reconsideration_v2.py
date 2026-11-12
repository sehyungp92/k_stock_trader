"""C15 R1: identical six accounts with bounded audit-only source storage."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import gzip,json,os,subprocess,sys,traceback
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify,env
from backtests.strategies.kalcb.reconsideration_io_v1 import freeze
from backtests.strategies.kalcb.qualification_storage_v1 import patched_verifier
from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_native
from backtests.strategies.kalcb.reconsideration_process_v2 import execute
from backtests.strategies.kalcb import capital_sensitivity_diagnostics_v1 as diag
from scripts.kalcb_opening_veto_qualification_v1 import classify_standard
from scripts.kalcb_retained_funding_v1 import status,comparison
from scripts.kalcb_opportunity_interaction_v1 import episode_traces
BASE=Path('E:/CodexResearch/k_stock_trader')
C=BASE/'kalcb_structural_preparation_20261016/funded_horizon_v1'
C14=BASE/'kalcb_rejection_confirmation_20261017/readiness_v2'
DESIGN=REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0039.md'
WORKER='scripts/kalcb_reconsideration_worker_v1.py'
CONTROLLER='backtests/strategies/kalcb/reconsideration_v1.py'
STORAGE='backtests/strategies/kalcb/qualification_storage_v1.py'
REPAIR=REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0039-repair-r1.md'


def no_producer():
    import psutil
    for p in psutil.process_iter(['pid','name','cmdline']):
        if p.pid==os.getpid() or 'python' not in (p.info['name'] or '').lower():continue
        if any('kalcb' in a.lower() and ('worker' in a.lower() or 'kalcb_reconsideration_v' in a.lower()) for a in p.info['cmdline'] or []):
            raise ValueError('Another producer '+str(p.pid))


def cases(controls):
    result=[dict(id='bridge_'+p+'_'+s,period=p,scenario=s,mode='disabled',bridge=controls[p+'/'+s])
            for p,s in [('is','baseline'),('oos','combined')]]
    return result+[dict(id='reconsider_'+p+'_'+s,period=p,scenario=s,mode='reconsider')
                   for p in ('is','oos') for s in ('baseline','combined')]


def command(item,case,out):
    cmd=[sys.executable,'-B','-u','-m',WORKER[:-3].replace('/','.'),'--plan',item['plan']['path'],
         '--policies',str(Path(item['policy']['path']).parent),'--audit-manifest',item['manifest']['path'],
         '--output',str(out),'--mode',case['mode'],'--scenario',case['scenario']]
    if case.get('bridge'):cmd+=['--bridge',case['bridge']]
    return cmd


PROBE='''import json
from scripts import kalcb_executed_value_replay as replay
from scripts import kalcb_reconsideration_worker_v1 as worker
from strategy_kalcb.core import account_allocation_v1 as allocation,logic
from backtests.engine import replay as engine
class ReachedReplay(Exception):pass
def sentinel(*a,**kw):raise ReachedReplay()
replay.run=sentinel
original=allocation.allocate_quantity;driver=engine.drive_timestamp_batch;evaluate=logic._evaluate_entry
try:worker.main()
except ReachedReplay:pass
else:raise AssertionError('No real replay boundary')
assert allocation.allocate_quantity is original and engine.drive_timestamp_batch is driver and logic._evaluate_entry is evaluate
print(json.dumps(dict(reached_replay_boundary=True,financial_replay_executed=False,hooks_restored=True)))
'''


def preflight(root,profiles,matrix):
    results={}
    for case in matrix:
        item=profiles[case['period']];code=Path(item['manifest']['path']).parent/'files';name=case['id']
        cmd=[sys.executable,'-B','-c',PROBE]+command(item,case,root/'preflight'/name)[5:]
        result=subprocess.run(cmd,cwd=code,env=env(code),capture_output=True,text=True)
        write(root/'preflight_logs'/(name+'.json'),dict(command=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode:raise RuntimeError('Real worker probe failed '+name)
        proof=json.loads(result.stdout.strip().splitlines()[-1])
        if not proof['reached_replay_boundary'] or proof['financial_replay_executed'] or not proof['hooks_restored']:raise ValueError('Invalid dry probe')
        if (root/'preflight'/name/'completion.json').exists():raise ValueError('Probe ran an account')
        results[name]=proof
    write(root/'runtime_preflight.json',results)


def register(root,tests):
    no_producer();receipt=read(tests)
    if receipt['exit_code']:raise ValueError('Readiness tests failed')
    verify(receipt['bindings']);root.mkdir(parents=True,exist_ok=False);(root/'temp').mkdir()
    os.environ['TEMP']=os.environ['TMP']=str(root/'temp')
    status(root,'authenticating_predecessors')
    cp=read(C/'protocol.json');cd=read(C/'completion.json');verify([cd['registration'],cd['assessment']])
    parents={p:read(C/'runtime'/p/'profile.json') for p in ('is','oos')}
    controls={};prior_controls={}
    inputs=[bound(DESIGN),bound(REPAIR),bound(tests),bound(C/'completion.json'),bound(C/'protocol.json'),bound(C/'horizon_map.json'),
            bound(C14/'registration.json'),bound(C14/'assessment.json'),bound(C14/'completion.json')]+receipt['bindings']
    for period,item in parents.items():
        verify([item[k] for k in ('manifest','plan','policy')]);inputs.extend(item[k] for k in ('manifest','plan','policy'))
        for scenario in ('baseline','combined'):
            folder=C/'cases/veto'/period/scenario
            validate_native(folder,item,scenario,save=False)
            if not diag.funding_review(folder)['verified']:raise ValueError('Unfunded control')
            controls[period+'/'+scenario]=str(folder)
            inputs.extend(bound(path) for path in folder.iterdir() if path.is_file())
            prior=Path(cp['references'][period][scenario]);prior_item=read(prior/'input_profile.json')
            validate_native(prior,prior_item,scenario,save=False)
            prior_controls[period+'/'+scenario]=str(prior)
            inputs.extend(bound(path) for path in prior.iterdir() if path.is_file())
    overlays={}
    for rel in (WORKER,CONTROLLER,STORAGE):
        dest=root/'overlays'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((REPO/rel).read_bytes());overlays[rel]=bound(dest)
    audit_rel='backtests/engine/qualification_evidence.py'
    originals=[Path(parent['manifest']['path']).parent/'files'/audit_rel for parent in parents.values()]
    original_bindings=[bound(path) for path in originals]
    if len({b['sha256'] for b in original_bindings})!=1:raise ValueError('Different historical audit implementations')
    for parent,binding in zip(parents.values(),original_bindings):
        if read(parent['manifest']['path'])['files'][audit_rel]!=binding['sha256']:raise ValueError('Unbound audit source')
    inputs.extend(original_bindings)
    dest=root/'overlays'/audit_rel;dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(patched_verifier(originals[0].read_text(encoding='utf-8')),encoding='utf-8')
    overlays[audit_rel]=bound(dest)
    programs=[bound(REPO/f) for f in (WORKER,CONTROLLER,STORAGE,'scripts/kalcb_reconsideration_v2.py',
       'tests/test_kalcb_reconsideration_v1.py','scripts/kalcb_retained_funding_v1.py',
       'scripts/kalcb_opening_veto_qualification_v1.py','backtests/strategies/kalcb/reconsideration_io_v1.py',
       'backtests/strategies/kalcb/capital_sensitivity_diagnostics_v1.py','backtests/strategies/kalcb/late_artifact_validation_v2.py',
       'backtests/strategies/kalcb/reconsideration_process_v2.py','scripts/kalcb_opportunity_interaction_v1.py')]
    matrix=cases(controls)
    p=dict(operational_repair='C15 R1 disk-backed audit source index; unchanged trading policy and criteria',at_utc=datetime.now(timezone.utc).isoformat(),experiment='kalcb-exp-0039',criteria='C15 diagnostic',
           inputs=inputs,programs=programs,parents=parents,controls=controls,prior_controls=prior_controls,overlays=overlays,cases=matrix,
           signal_map=cp['signal_map'],database_bindings=cp['database_bindings'],native_accounts=6,principal_qualification=False)
    write(root/'protocol.json',p);write(root/'completed_cases.json',[]);verify(inputs+programs+p['database_bindings'])
    profiles={}
    for period,parent in parents.items():
        status(root,'freezing_'+period);profiles[period]=freeze(root,period,parent)
    status(root,'real_worker_preflight');preflight(root,profiles,matrix)
    verify(inputs+programs+p['database_bindings'])
    write(root/'registration.json',dict(protocol=bound(root/'protocol.json'),profiles=profiles,preflight=bound(root/'runtime_preflight.json')))


def activation(folder,mode):
    actions=read(folder/'reconsideration_actions.json');vetoes=read(folder/'horizon_actions.json')
    orders=[]
    with gzip.open(folder/'orders.jsonl.gz','rt',encoding='utf8') as stream:
        for line in stream:
            row=json.loads(line);o=row.get('order',{})
            if row['event']=='submit' and o.get('side')=='BUY':orders.append(o)
    renewed=[o for o in orders if o.get('metadata',{}).get('research_reconsideration')]
    if mode=='disabled' and (actions or renewed):raise ValueError('Disabled reconsideration acted')
    if not vetoes:raise ValueError('Original VETO inactive')
    for o in renewed:
        meta=o['metadata'];tag=meta['research_reconsideration'];at=meta['decision_available_at'];key=at[:10]+'/'+o['symbol']
        if at!=tag['evidence_available_at'] or at[11:16]!='09:35':raise ValueError('Wrong renewed action time')
        matches=[r for r in actions if r['kind']=='renewal_considered' and r['key']==key and r.get('submitted_actions')==1]
        if len(matches)!=1 or matches[0]['requested_quantity']!=o['qty'] or not matches[0]['confirmed']:raise ValueError('Native renewal activation mismatch')
        if meta['entry_price_ref']!=matches[0]['close']:raise ValueError('Inherited stale entry price')
        if len([x for x in renewed if x['symbol']==o['symbol'] and x['metadata']['decision_available_at'][:10]==at[:10]])!=1:raise ValueError('Repeated renewal')
    expected=sum(r.get('submitted_actions',0) for r in actions if r['kind']=='renewal_considered')
    if expected!=len(renewed):raise ValueError('Unsubmitted renewal action')
    fills=read(folder/'fills.json');order_ids={o['order_id'] for o in renewed}
    result=dict(passed=True,mode=mode,counts=dict(Counter(r['kind'] for r in actions)),
        vetoed_distinct=len({r['key'] for r in vetoes}),confirmed_distinct=len({r['key'] for r in actions if r.get('confirmed')}),
        renewed_submissions=len(renewed),renewed_filled_orders=len({f['order_id'] for f in fills if f['side']=='BUY' and f['order_id'] in order_ids}),
        renewed_orders=renewed,scope='Actual actions/fills; distinct opportunities, not repeated model or scenario rows.')
    write(folder/'activation.json',result);return result


def assess(root):
    p=read(root/'protocol.json');ledger=read(root/'completed_cases.json');cells={};prior={};rejected={}
    for row in ledger:
        c=row['case'];name=c['id']
        if c['mode']=='disabled':continue
        if not row['verdict']['qualified']:rejected[name]=row['verdict'];continue
        folder=root/'cases'/name;key=c['period']+'/'+c['scenario']
        cells[name]=comparison(folder,p['controls'][key]);prior[name]=comparison(folder,p['prior_controls'][key])
        write(root/'traces'/(name+'.json'),episode_traces(folder,p['controls'][key]))
    write(root/'assessment.json',dict(cells=cells,versus_prior100=prior,rejected=rejected,
        human_causal_economic_assessment_required=True,economic_objective_achieved=False,principal_qualification=False,
        scope='C15 native new-event package; adaptively reused development; no automatic financial successor.'))


def run(root):
    p=read(root/'protocol.json');reg=read(root/'registration.json')
    verify([reg['protocol'],reg['preflight']]+p['inputs']+p['programs']+p['database_bindings'])
    for i,c in enumerate(p['cases']):
        if c['mode']!='disabled':
            for name in ('bridge_is_baseline','bridge_oos_combined'):
                if not read(root/'cases'/name/'identity_bridge.json')['passed']:raise ValueError('Missing bridge')
                if not any(r['case']['id']==name and r['verdict']['qualified'] for r in read(root/'completed_cases.json')):raise ValueError('Unaccepted bridge')
        item=reg['profiles'][c['period']];out=root/'cases'/c['id'];code=Path(item['manifest']['path']).parent/'files'
        execute(root,command(item,c,out),c['id'],code,dict(case=c,cases_complete=i,cases_total=6))
        write(out/'input_profile.json',item);verdict=classify_standard(out,item,c['scenario'])
        if verdict['qualified']:write(out/'funding_review.json',diag.funding_review(out))
        else:write(out/'qualification_rejection.json',verdict)
        act=activation(out,c['mode']);ledger=read(root/'completed_cases.json')
        ledger.append(dict(case=c,verdict=verdict,activation=act,completion=bound(out/'completion.json')));write(root/'completed_cases.json',ledger)
    status(root,'independent_assessment');assess(root);verify(p['inputs']+p['programs']+p['database_bindings'])
    write(root/'completion.json',dict(complete=True,at_utc=datetime.now(timezone.utc).isoformat(),native_accounts=6,
          registration=bound(root/'registration.json'),assessment=bound(root/'assessment.json'),economic_objective_achieved=False))
    status(root,'diagnostic_assessed_disposition_pending')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--tests',type=Path,required=True)
    a=parser.parse_args()
    try:register(a.output,a.tests);run(a.output)
    except Exception as error:
        if a.output.exists():write(a.output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        raise
