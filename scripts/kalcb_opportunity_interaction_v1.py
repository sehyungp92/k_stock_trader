"""C12 registered VETO/pending-cap funded interaction, no policy search."""
import argparse,gzip,json,os,shutil,subprocess,sys,traceback
from datetime import datetime
from pathlib import Path
from statistics import median
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,verify,env
from backtests.strategies.kalcb.opportunity_interaction_io_v1 import freeze
from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_native
from backtests.strategies.kalcb.research_process_v1 import execute
from backtests.strategies.kalcb import capital_sensitivity_diagnostics_v1 as diag
from backtests.strategies.kalcb.funded_horizon_v1 import enabled_signal
from scripts.kalcb_opening_veto_qualification_v1 import classify_standard
from scripts.kalcb_retained_funding_v1 import status,comparison
BASE=Path('E:/CodexResearch/k_stock_trader')
C11=BASE/'kalcb_retained_funding_20261016/v1'
C10=BASE/'kalcb_opening_veto_qualification_20261016/v3'
V28=BASE/'kalcb_late_entry_20261011/v28_pending_funding_qualification'
DESIGN=REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0036-funded-design.md'
WORKER='scripts/kalcb_opportunity_interaction_worker_v1.py'
SCENARIOS=('baseline','participation','cost','combined')
PERIODS=('is','oos')


def no_producer():
    import psutil
    for p in psutil.process_iter(['pid','name','cmdline']):
        if p.pid==os.getpid() or 'python' not in (p.info['name'] or '').lower():continue
        if any('kalcb' in a.lower() and any(t in a.lower() for t in ('worker','opportunity_interaction_v1.py','retained_funding_v1.py')) for a in p.info['cmdline'] or []):
            raise ValueError('Another producer '+str(p.pid))


def declared_cases(controls):
    rows=[]
    for arm,cap,mode,cells in [('veto',0,'veto',[('is','baseline'),('oos','combined')]),
                             ('cap',2,'disabled',[('is','baseline'),('oos','cost')])]:
        for period,scenario in cells:
            rows.append(dict(id=f'bridge_{arm}_{period}_{scenario}',period=period,scenario=scenario,cap=cap,mode=mode,
                             bridge=controls[arm][period+'/'+scenario],kind='bridge'))
    rows.extend(dict(id=f'combined_{p}_{s}',period=p,scenario=s,cap=2,mode='veto',kind='treatment') for p in PERIODS for s in SCENARIOS)
    return rows


def command(item,c,out):
    cmd=[sys.executable,'-B','-u','-m',WORKER[:-3].replace('/','.'),'--plan',item['plan']['path'],
        '--policies',str(Path(item['policy']['path']).parent),'--audit-manifest',item['manifest']['path'],
        '--output',str(out),'--mode',c['mode'],'--scenario',c['scenario']]
    if c.get('bridge'):cmd+=['--bridge',c['bridge']]
    return cmd


PROBE='''import json
from scripts import kalcb_executed_value_replay as replay
from scripts import kalcb_opportunity_interaction_worker_v1 as worker
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


def preflight(root,profiles,cases):
    results={}
    for c in cases:
        item=profiles[c['period']+'_cap'+str(c['cap'])];code=Path(item['manifest']['path']).parent/'files'
        cmd=[sys.executable,'-B','-c',PROBE]+command(item,c,root/'preflight'/c['id'])[5:]
        r=subprocess.run(cmd,cwd=code,env=env(code),capture_output=True,text=True)
        write(root/'preflight_logs'/(c['id']+'.json'),dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
        if r.returncode:raise RuntimeError('Full worker dry probe failed '+c['id'])
        proof=json.loads(r.stdout.strip().splitlines()[-1])
        if not proof['reached_replay_boundary'] or proof['financial_replay_executed'] or not proof['hooks_restored']:raise ValueError('Invalid dry probe')
        if (root/'preflight'/c['id']/'completion.json').exists():raise ValueError('Dry probe executed account')
        results[c['id']]=proof
    write(root/'runtime_preflight.json',results)


def register(root,tests):
    no_producer();t=read(tests)
    if t['exit_code']:raise ValueError('Readiness tests failed')
    verify(t['bindings']);root.mkdir(parents=True,exist_ok=False);(root/'temp').mkdir()
    os.environ['TEMP']=os.environ['TMP']=str(root/'temp')
    status(root,'authenticating_predecessors')
    c11=read(C11/'protocol.json');c10=read(C10/'protocol.json')
    for r in (C11,C10):
        done=read(r/'completion.json');verify([done['registration'],done['assessment']])
    cp=read(BASE/'kalcb_structural_preparation_20261016/funded_horizon_v1/protocol.json')
    caprows={r['period']+'/'+r['scenario']:r for r in read(V28/'progress.json') if r['name']=='pending_100'}
    if len(caprows)!=8 or not read(V28/'stage_completion.json')['complete']:raise ValueError('Incomplete cap predecessor')
    controls={k:{} for k in ('prior','veto','cap')};parents={p:read(C11/'runtime'/p/'profile.json') for p in PERIODS}
    inputs=[bound(DESIGN),bound(REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0036-results-a.md'),bound(tests),
        bound(V28/'progress.json'),bound(V28/'stage_completion.json'),bound(C11/'completion.json'),bound(C10/'completion.json'),
        bound(C11/'protocol.json'),bound(C10/'protocol.json'),bound(BASE/'kalcb_opportunity_lifecycle_20261016/census_v2/completion.json')]+t['bindings']
    reviews={}
    for period in PERIODS:
        parent=parents[period];verify([parent[k] for k in ('manifest','plan','policy')]);inputs.extend(parent[k] for k in ('manifest','plan','policy'))
        for scenario in SCENARIOS:
            key=period+'/'+scenario
            folders={'prior':Path(c10['controls'][key]),'veto':Path(c10['reused'][key]),'cap':Path(caprows[key]['folder'])}
            for arm,folder in folders.items():
                if arm=='cap':
                    verify([caprows[key]['result'],caprows[key]['input_profile'],caprows[key]['exposure_mtm']])
                    item=read(caprows[key]['input_profile']['path'])
                else:
                    item=read(folder/'input_profile.json') if (folder/'input_profile.json').exists() else c10['profiles'][period]
                metrics=validate_native(folder,item,scenario,save=False)
                if arm=='cap' and metrics!=caprows[key]['metrics']:raise ValueError('Changed historical cap metrics')
                review=diag.funding_review(folder)
                if not review['verified']:raise ValueError('Invalid predecessor funding')
                controls[arm][key]=str(folder);reviews[arm+'/'+key]=dict(metrics=metrics,funding=review,profile=item)
                inputs.extend(item[k] for k in ('manifest','plan','policy'))
                inputs.extend(bound(f) for f in folder.iterdir() if f.is_file())
    overlays={WORKER:bound(REPO/WORKER)}
    dest=root/'overlays'/WORKER;dest.parent.mkdir(parents=True);shutil.copyfile(REPO/WORKER,dest);overlays[WORKER]=bound(dest)
    program_paths=(WORKER,'scripts/kalcb_opportunity_interaction_v1.py','backtests/strategies/kalcb/opportunity_interaction_io_v1.py',
        'tests/test_kalcb_opportunity_interaction_v1.py','scripts/kalcb_retained_funding_v1.py','scripts/kalcb_opening_veto_qualification_v1.py',
        'backtests/strategies/kalcb/capital_sensitivity_diagnostics_v1.py','backtests/strategies/kalcb/late_artifact_validation_v2.py',
        'backtests/strategies/kalcb/research_process_v1.py','backtests/strategies/kalcb/funded_horizon_v1.py')
    programs=[bound(REPO/f) for f in program_paths];cases=declared_cases(controls)
    p=dict(experiment='kalcb-exp-0036',criteria='C12-B diagnostic',inputs=inputs,programs=programs,parents=parents,controls=controls,
        overlays=overlays,cases=cases,signal_map=c11['signal_map'],database_bindings=c11['database_bindings'],native_accounts=12,
        fits=0,principal_qualification=False,unchanged_model_map_policy=True)
    write(root/'protocol.json',p);write(root/'completed_cases.json',[]);write(root/'predecessor_reviews.json',reviews)
    verify(inputs+programs+p['database_bindings'])
    profiles={}
    for period,parent in parents.items():
        for cap in (0,2):
            key=f'{period}_cap{cap}';status(root,'freezing_'+key);profiles[key]=freeze(root,period,cap,parent)
    status(root,'full_worker_preflight');preflight(root,profiles,cases)
    verify(inputs+programs+p['database_bindings'])
    write(root/'registration.json',dict(protocol=bound(root/'protocol.json'),profiles=profiles,
        preflight=bound(root/'runtime_preflight.json'),predecessors=bound(root/'predecessor_reviews.json')))


def lines(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for l in f:yield json.loads(l)


def activation(folder,c,signals):
    observed=[];submits=[];planned=[];vetoed=[]
    for d in lines(folder/'decisions.jsonl.gz'):
        m=d.get('metadata',{});cap=m.get('pending_funding_reservation')
        if cap:
            if c['cap']!=2 or m.get('entry_route')!='first30_open_anchor' or cap['limit']!=2:raise ValueError('Wrong cap scope')
            expected=min(cap['original_qty'],int(max(0.,2*cap['equity']-cap['reserved_gross'])/cap['completed_close']))
            if expected!=cap['capped_qty'] or not 0<=expected<=cap['original_qty']:raise ValueError('Cap arithmetic mismatch')
            observed.append(dict(symbol=d['symbol'],at=d['timestamp'],decision=d['decision_code'],reason=d['reason'],cap=cap))
        for a in d.get('actions',[]):
            if a['action_type']=='SubmitEntry':planned.append((d['timestamp'],a['symbol'],a['qty']))
    for row in lines(folder/'orders.jsonl.gz'):
        o=row.get('order',{});m=o.get('metadata',{})
        if row['event']!='submit' or o.get('side')!='BUY':continue
        submits.append((m['signal_bar'],o['symbol'],o['qty']))
        if m.get('entry_route')=='first30_open_anchor':
            if c['cap'] and (not m.get('pending_funding_reservation') or o['qty']!=m['pending_funding_reservation']['capped_qty']):
                raise ValueError('Submitted opening cap mismatch')
            signal=signals.get(m['signal_bar'][:10]+'/'+o['symbol'])
            if c['mode']=='veto' and enabled_signal(signal,datetime.fromisoformat(m['decision_available_at'])):raise ValueError('Negative opening admitted')
        elif m.get('pending_funding_reservation'):raise ValueError('Later route capped')
    if sorted(planned)!=sorted(submits):raise ValueError('Core proposals do not reconcile to submitted orders')
    for x in read(folder/'horizon_actions.json'):
        if c['mode']!='veto' or x['kind']!='veto':raise ValueError('Unexpected horizon action')
        if not enabled_signal(x['signal'],datetime.fromisoformat(x['available_at'])):raise ValueError('Invalid VETO time/predicate')
        vetoed.append(x['key'])
    if c['cap']==0 and observed:raise ValueError('Disabled cap acted')
    result=dict(passed=True,cap_decisions=len(observed),trimmed=sum(x['cap']['capped_qty']<x['cap']['original_qty'] for x in observed),
        zero_cap=sum(x['cap']['capped_qty']==0 for x in observed),distinct_cap_dates=sorted({x['at'][:10] for x in observed if x['cap']['capped_qty']<x['cap']['original_qty']}),
        veto_issuer_days=sorted(set(vetoed)),observations=observed)
    write(folder/'activation.json',result);return {k:v for k,v in result.items() if k!='observations'}


def bridge_gate(root,cases,ledger):
    for c in cases:
        if c['kind']!='bridge':continue
        matches=[r for r in ledger if r['case']['id']==c['id']]
        if len(matches)!=1 or not matches[0]['verdict']['qualified'] or not read(root/'cases'/c['id']/'identity_bridge.json')['passed']:
            raise ValueError('All four accepted exact bridges must gate treatment')


def episode_traces(folder,control):
    # Select from complete compact economic/action indexes, then stream only the
    # predeclared trace episodes. Avoid retaining every repeated decision object.
    def index(f):
        values={};orders={}
        for t in read(f/'trades.json'):
            k=t['entry_decision_time'][:10]+'/'+t['symbol'];values[k]=values.get(k,0.)+t['net_pnl']
        for r in lines(f/'orders.jsonl.gz'):
            o=r.get('order',{})
            if o.get('side')!='BUY':continue
            at=r.get('timestamp') or o['submitted_at'];k=at[:10]+'/'+o['symbol']
            values.setdefault(k,0.);orders.setdefault(k,[]).append((at,r['event'],o['qty'],o['submitted_at'],r.get('reason')))
        return values,orders
    folder,control=Path(folder),Path(control);a,ao=index(folder);b,bo=index(control)
    deltas={k:a.get(k,0.)-b.get(k,0.) for k in a.keys()|b.keys()}
    changed=sorted(k for k,v in deltas.items() if abs(v)>.001)
    keys=[]
    if changed:
        order=sorted(changed,key=lambda k:(deltas[k],k));keys=[order[0],order[-1],changed[0],order[len(order)//2]]
    keys+=sorted(k for k in deltas if k[:10] in ('2026-04-23','2026-05-29'))
    divergence=[]
    for k in ao.keys()|bo.keys():
        left,right=ao.get(k,[]),bo.get(k,[])
        if left==right:continue
        at=next((min(left[n][0],right[n][0]) for n in range(min(len(left),len(right))) if left[n]!=right[n]),None)
        if at is None:at=(left if len(left)>len(right) else right)[min(len(left),len(right))][0]
        divergence.append((at,k))
    first=min(divergence)[1] if divergence else None
    if first:keys.append(first)
    keys=list(dict.fromkeys(keys))
    def selected(f):
        out={k:dict(decisions=[],orders=[],fills=[],trades=[],net=0.) for k in keys}
        for d in lines(f/'decisions.jsonl.gz'):
            k=d['timestamp'][:10]+'/'+d['symbol']
            if k in out:out[k]['decisions'].append(d)
        for d in lines(f/'orders.jsonl.gz'):
            o=d.get('order',{})
            if o.get('symbol'):
                k=(d.get('timestamp') or o['submitted_at'])[:10]+'/'+o['symbol']
                if k in out:out[k]['orders'].append(d)
        for d in read(f/'fills.json'):
            k=d['timestamp'][:10]+'/'+d['symbol']
            if k in out:out[k]['fills'].append(d)
        for d in read(f/'trades.json'):
            k=d['entry_decision_time'][:10]+'/'+d['symbol']
            if k in out:out[k]['trades'].append(d);out[k]['net']+=d['net_pnl']
        return out
    traces_a,traces_b=selected(folder),selected(control)
    return dict(first_order_divergence=first,changed_issuer_days=len(changed),changed_dates=sorted({k[:10] for k in changed}),
        deltas=deltas,traces={k:dict(delta=deltas[k],actual=traces_a[k],control=traces_b[k]) for k in keys})


def assess(root):
    p=read(root/'protocol.json');ledger=read(root/'completed_cases.json');cells={};rejected={}
    for row in ledger:
        c=row['case'];key=c['period']+'/'+c['scenario']
        if c['kind']=='bridge':continue
        if not row['verdict']['qualified']:rejected[key]=row['verdict'];continue
        folder=root/'cases'/c['id'];v={arm:comparison(folder,p['controls'][arm][key]) for arm in ('prior','veto','cap')}
        component=comparison(p['controls']['cap'][key],p['controls']['prior'][key])
        cells[key]=dict(comparisons=v,cap_minus_prior=component,
            interaction_net_krw=v['veto']['net_delta_krw']-component['net_delta_krw'],
            interaction_useful_days=(v['veto']['useful_days']-v['veto']['control_useful_days'])-(component['useful_days']-component['control_useful_days']))
        for arm in ('veto','cap'):
            write(root/'traces'/(c['id']+'_versus_'+arm+'.json'),episode_traces(folder,p['controls'][arm][key]))
    o=[v for k,v in cells.items() if k.startswith('oos/')];i=[v for k,v in cells.items() if k.startswith('is/')]
    criteria=dict(all_cells_qualified=len(cells)==8 and not rejected,principal_qualification=False,economic_objective_achieved=False)
    if len(o)==len(i)==4:
        criteria.update(median_oos_vs_veto_krw=median(v['comparisons']['veto']['net_delta_krw'] for v in o),
            median_oos_useful_vs_prior=median(v['comparisons']['prior']['useful_days']-v['comparisons']['prior']['control_useful_days'] for v in o),
            median_is_vs_prior_krw=median(v['comparisons']['prior']['net_delta_krw'] for v in i))
    write(root/'assessment.json',dict(cells=cells,rejected=rejected,criteria=criteria,human_causal_economic_assessment_required=True,
        scope='C12-B bounded funded interaction; dependent reused history; no automatic variant, promotion or successful delivery.'))


def run(root):
    p=read(root/'protocol.json');reg=read(root/'registration.json')
    verify([reg['protocol'],reg['preflight'],reg['predecessors']]+p['inputs']+p['programs']+p['database_bindings'])
    signals=read(p['signal_map']['path'])
    for i,c in enumerate(p['cases']):
        ledger=read(root/'completed_cases.json')
        if c['kind']=='treatment':bridge_gate(root,p['cases'],ledger)
        item=reg['profiles'][c['period']+'_cap'+str(c['cap'])];out=root/'cases'/c['id'];code=Path(item['manifest']['path']).parent/'files'
        execute(root,command(item,c,out),c['id'],code,dict(case=c,cases_complete=i,cases_total=len(p['cases'])))
        write(out/'input_profile.json',item);verdict=classify_standard(out,item,c['scenario'])
        if verdict['qualified']:write(out/'funding_review.json',diag.funding_review(out))
        else:write(out/'qualification_rejection.json',verdict)
        act=activation(out,c,signals)
        ledger.append(dict(case=c,verdict=verdict,activation=act,completion=bound(out/'completion.json')));write(root/'completed_cases.json',ledger)
    status(root,'independent_assessment');assess(root);verify(p['inputs']+p['programs']+p['database_bindings'])
    write(root/'completion.json',dict(complete=True,native_accounts=len(p['cases']),registration=bound(root/'registration.json'),
        assessment=bound(root/'assessment.json'),economic_objective_achieved=False))
    status(root,'diagnostic_assessed_disposition_pending')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--tests',type=Path,required=True);a=ap.parse_args()
    try:register(a.output,a.tests);run(a.output)
    except Exception as e:
        if a.output.exists():write(a.output/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
        raise
