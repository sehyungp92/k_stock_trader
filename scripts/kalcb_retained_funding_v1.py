"""C11 finite source-bound retained-book funding diagnostic and assessment."""
import argparse
from datetime import datetime,timezone
import gzip,json,os,shutil,subprocess,sys,traceback
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify,env
from backtests.strategies.kalcb.holding_account_io_v1 import freeze
from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_native
from backtests.strategies.kalcb.research_process_v1 import execute
from backtests.strategies.kalcb import capital_sensitivity_diagnostics_v1 as diag
from backtests.strategies.kalcb.retained_funding_v1 import reserve
from scripts.kalcb_opening_veto_qualification_v1 import useful_days,classify_standard
BASE=Path('E:/CodexResearch/k_stock_trader')
C=BASE/'kalcb_structural_preparation_20261016/funded_horizon_v1'
PREVIOUS=BASE/'kalcb_opening_veto_qualification_20261016'
DESIGN=REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0035.md'
WORKER='scripts/kalcb_retained_funding_worker_v1.py'
CONTROLLER='backtests/strategies/kalcb/retained_funding_v1.py'


def status(root,phase,**kw):
    import psutil
    write(root/'status.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),phase=phase,pid=os.getpid(),
          created_epoch=psutil.Process().create_time(),argv=sys.argv,**kw))


def no_producer():
    import psutil
    for p in psutil.process_iter(['pid','name','cmdline']):
        if p.pid==os.getpid() or 'python' not in (p.info['name'] or '').lower():continue
        if any('kalcb' in a.lower() and ('worker' in a.lower() or 'kalcb_retained_funding_v1.py' in a.lower()) for a in p.info['cmdline'] or []):
            raise ValueError('Another producer '+str(p.pid))


PROBE='''import json
from scripts import kalcb_executed_value_replay as replay
from scripts import kalcb_retained_funding_worker_v1 as worker
from strategy_kalcb.core import account_allocation_v1 as allocation
from backtests.engine import replay as engine
class ReachedReplay(Exception):pass
def sentinel(*a,**kw):raise ReachedReplay()
replay.run=sentinel
original=allocation.allocate_quantity;driver=engine.drive_timestamp_batch
try:worker.main()
except ReachedReplay:pass
else:raise AssertionError('No real replay boundary')
assert allocation.allocate_quantity is original and engine.drive_timestamp_batch is driver
print(json.dumps(dict(reached_replay_boundary=True,financial_replay_executed=False,hooks_restored=True)))
'''


def command(item,mode,scenario,out,bridge=None):
    cmd=[sys.executable,'-B','-u','-m',WORKER[:-3].replace('/','.'),'--plan',item['plan']['path'],
         '--policies',str(Path(item['policy']['path']).parent),'--audit-manifest',item['manifest']['path'],
         '--output',str(out),'--mode',mode,'--scenario',scenario]
    if bridge:cmd+=['--bridge',str(bridge)]
    return cmd


def preflight(root,profiles):
    results={}
    for period,item in profiles.items():
        code=Path(item['manifest']['path']).parent/'files'
        for mode in ('disabled','queue','forecast'):
            for scenario in ('baseline','combined'):
                key=period+'_'+mode+'_'+scenario
                cmd=command(item,mode,scenario,root/'preflight'/key)
                cmd=[sys.executable,'-B','-c',PROBE]+cmd[5:]
                result=subprocess.run(cmd,cwd=code,env=env(code),capture_output=True,text=True)
                write(root/'preflight_logs'/(key+'.json'),dict(command=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
                if result.returncode:raise RuntimeError('Real worker probe failed '+key)
                proof=json.loads(result.stdout.strip().splitlines()[-1])
                if not proof['reached_replay_boundary'] or proof['financial_replay_executed'] or not proof['hooks_restored']:raise ValueError('Invalid dry probe')
                if (root/'preflight'/key/'completion.json').exists():raise ValueError('Probe became financial account')
                results[key]=proof
    write(root/'runtime_preflight.json',results)


def register(root,tests):
    no_producer();receipt=read(tests)
    if receipt['exit_code']:raise ValueError('Tests failed')
    verify(receipt['bindings']);root.mkdir(parents=True,exist_ok=False);(root/'temp').mkdir()
    status(root,'authenticating_predecessors')
    done=read(PREVIOUS/'v3/completion.json');verify([done['registration'],done['assessment']])
    cp=read(C/'protocol.json');cd=read(C/'completion.json');verify([cd['registration'],cd['assessment']])
    parents={p:read(C/'runtime'/p/'profile.json') for p in ('is','oos')}
    controls={};prior_controls={};inputs=[bound(DESIGN),bound(tests),bound(PREVIOUS/'v3/completion.json'),bound(PREVIOUS/'v3/assessment.json'),
        bound(PREVIOUS/'closeout_review_v1.json'),bound(PREVIOUS/'disposition_v1/intersection.json'),
        bound(PREVIOUS/'disposition_v1/information_scope.json'),bound(PREVIOUS/'disposition_v1/funding_audit.json'),
        bound(C/'protocol.json'),bound(C/'horizon_map.json')]+receipt['bindings']
    for period,item in parents.items():
        verify([item[k] for k in ('manifest','plan','policy')]);inputs.extend(item[k] for k in ('manifest','plan','policy'))
        for scenario in ('baseline','combined'):
            folder=C/'cases/veto'/period/scenario;validate_native(folder,item,scenario,save=False)
            controls[period+'/'+scenario]=str(folder)
            inputs.extend(bound(path) for path in folder.iterdir() if path.is_file())
            prior=Path(cp['references'][period][scenario]);prior_item=read(prior/'input_profile.json')
            validate_native(prior,prior_item,scenario,save=False)
            prior_controls[period+'/'+scenario]=str(prior)
            inputs.extend(bound(path) for path in prior.iterdir() if path.is_file())
    helper=PREVIOUS/'v3/capital/inputs/files/backtests/strategies/kalcb/fixed_allocation_reference_v1.py'
    hm=read(PREVIOUS/'v3/capital/inputs/manifest.json');hb=bound(helper)
    if hb['sha256']!=hm['files']['backtests/strategies/kalcb/fixed_allocation_reference_v1.py']:raise ValueError('Historical helper changed')
    overlays={}
    for rel,source in [(WORKER,REPO/WORKER),(CONTROLLER,REPO/CONTROLLER),('backtests/strategies/kalcb/fixed_allocation_reference_v1.py',helper)]:
        dest=root/'overlays'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);overlays[rel]=bound(dest)
    cases=[dict(id='bridge_'+p+'_'+s,period=p,scenario=s,mode='disabled',bridge=controls[p+'/'+s]) for p,s in [('is','baseline'),('oos','combined')]]
    cases += [dict(id=m+'_'+p+'_'+s,period=p,scenario=s,mode=m) for m in ('queue','forecast') for p in ('is','oos') for s in ('baseline','combined')]
    programs=[bound(REPO/f) for f in (WORKER,CONTROLLER,'scripts/kalcb_retained_funding_v1.py',
       'tests/test_kalcb_retained_funding_v1.py','tests/test_kalcb_retained_funding_assessment_v1.py',
       'scripts/kalcb_opening_veto_qualification_v1.py','backtests/strategies/kalcb/holding_account_io_v1.py',
       'backtests/strategies/kalcb/capital_sensitivity_diagnostics_v1.py','backtests/strategies/kalcb/late_artifact_validation_v2.py',
       'backtests/strategies/kalcb/research_process_v1.py')]
    p=dict(at_utc=datetime.now(timezone.utc).isoformat(),experiment='kalcb-exp-0035',criteria='C11 diagnostic',
           inputs=inputs,programs=programs,parents=parents,controls=controls,prior_controls=prior_controls,overlays=overlays,cases=cases,
           signal_map=cp['signal_map'],database_bindings=cp['database_bindings'],native_accounts=10,principal_qualification=False)
    write(root/'protocol.json',p);write(root/'completed_cases.json',[]);verify(inputs+programs+p['database_bindings'])
    profiles={}
    for period,parent in parents.items():
        status(root,'freezing_'+period);profiles[period]=freeze(root,period,parent)
    status(root,'real_worker_preflight');preflight(root,profiles)
    verify(inputs+programs+p['database_bindings'])
    write(root/'registration.json',dict(protocol=bound(root/'protocol.json'),profiles=profiles,preflight=bound(root/'runtime_preflight.json')))


def activation(folder,mode):
    records=read(folder/'reservation_actions.json');submits={}
    with gzip.open(folder/'orders.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);o=row.get('order',{});meta=o.get('metadata',{})
            if row['event']=='submit' and o.get('side')=='BUY' and meta.get('entry_route')=='first30_open_anchor':
                submits.setdefault((meta['decision_available_at'],o['symbol']),[]).append(o['qty'])
    if mode=='disabled' and records:raise ValueError('Disabled reservation acted')
    changed=[]
    for r in records:
        if r['reason']!='supported_overbooked':continue
        book,left,priority=reserve(r['intents'],r['room'],mode)
        if priority!=r['priority'] or abs(left-r['remaining_room'])>1e-7:raise ValueError('Reservation decision drift')
        for symbol,row in book.items():
            if row['qty']!=r['book'][symbol]['qty']:raise ValueError('Allocation drift')
            actual=submits.get((r['available_at'],symbol),[])
            if actual!=([row['qty']] if row['qty'] else []):raise ValueError('Native allocation activation mismatch '+symbol)
            if not row['qty'] and any(s==symbol and at[:10]==r['day'] and at>=r['available_at']
                                     for at,s in submits):raise ValueError('Omitted first30 retried '+symbol)
        if r['changed']:changed.append(r['day'])
    result=dict(passed=True,preview_batches=len(records),changed_batches=len(changed),changed_sessions=sorted(set(changed)),
                omitted_requests=sum(x['qty']==0 for r in records for x in r.get('book',{}).values()))
    write(folder/'activation.json',result);return result


def comparison(folder,control):
    folder=Path(folder);control=Path(control)
    a,b=diag.account(folder),diag.account(control);aa,bb=diag.opportunities(folder),diag.opportunities(control)
    return dict(account=a,control=b,metrics=diag.metrics(folder),control_metrics=diag.metrics(control),
        activity=aa,control_activity=bb,activity_delta=diag.compare(aa,bb),path=diag.paired_attribution(a,b),
        useful_days=useful_days(aa),control_useful_days=useful_days(bb),net_delta_krw=a['net_pnl_krw']-b['net_pnl_krw'])


def assess(root):
    p=read(root/'protocol.json');ledger=read(root/'completed_cases.json');cells={};prior_cells={};interaction={};rejected={}
    accepted={r['case']['id']:r for r in ledger if r['verdict']['qualified']}
    for row in ledger:
        c=row['case'];name=c['id']
        if c['mode']=='disabled':continue
        if not row['verdict']['qualified']:rejected[name]=row['verdict'];continue
        cells[name]=comparison(root/'cases'/name,p['controls'][c['period']+'/'+c['scenario']])
        prior_cells[name]=comparison(root/'cases'/name,p['prior_controls'][c['period']+'/'+c['scenario']])
    for period in ('is','oos'):
        for scenario in ('baseline','combined'):
            key=period+'_'+scenario;f='forecast_'+key;q='queue_'+key
            if f in accepted and q in accepted:interaction[key]=comparison(root/'cases'/f,root/'cases'/q)
    write(root/'assessment.json',dict(cells=cells,versus_prior100=prior_cells,forecast_minus_queue=interaction,rejected=rejected,
        human_causal_economic_assessment_required=True,economic_objective_achieved=False,principal_qualification=False,
        scope='C11 full funded diagnostic; reused development; no automatic successor or economic completion.'))


def run(root):
    p=read(root/'protocol.json');reg=read(root/'registration.json')
    verify([reg['protocol'],reg['preflight']]+p['inputs']+p['programs']+p['database_bindings'])
    for i,c in enumerate(p['cases']):
        if c['mode']!='disabled':
            for name in ('bridge_is_baseline','bridge_oos_combined'):
                if not read(root/'cases'/name/'identity_bridge.json')['passed']:raise ValueError('Missing mandatory bridge')
        item=reg['profiles'][c['period']];out=root/'cases'/c['id'];code=Path(item['manifest']['path']).parent/'files'
        execute(root,command(item,c['mode'],c['scenario'],out,c.get('bridge')),c['id'],code,dict(case=c,cases_complete=i,cases_total=10))
        write(out/'input_profile.json',item);verdict=classify_standard(out,item,c['scenario'])
        if verdict['qualified']:write(out/'funding_review.json',diag.funding_review(out))
        else:write(out/'qualification_rejection.json',verdict)
        act=activation(out,c['mode']);ledger=read(root/'completed_cases.json')
        ledger.append(dict(case=c,verdict=verdict,activation=act,completion=bound(out/'completion.json')));write(root/'completed_cases.json',ledger)
    status(root,'independent_assessment');assess(root);verify(p['inputs']+p['programs']+p['database_bindings'])
    write(root/'completion.json',dict(complete=True,at_utc=datetime.now(timezone.utc).isoformat(),native_accounts=10,
          registration=bound(root/'registration.json'),assessment=bound(root/'assessment.json'),economic_objective_achieved=False))
    status(root,'diagnostic_assessed_disposition_pending')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--tests',type=Path,required=True);a=p.parse_args()
    try:register(a.output,a.tests);run(a.output)
    except Exception as e:
        if a.output.exists():write(a.output/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
        raise
