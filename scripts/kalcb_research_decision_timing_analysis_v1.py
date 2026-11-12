"""Independent C18 selective-delay accounting and factorial attribution."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import json,csv,hashlib,importlib.util,time
import numpy as np
ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_research_decision_20261018/v1')
OLD=Path('E:/CodexResearch/k_stock_trader/kalcb_round3_20261017/attempt_2/round_run_v2')
REPO=Path(__file__).resolve().parents[1]
SCENARIOS=['baseline','participation','cost','combined']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def bound(p):
    with Path(p).open('rb') as f:return dict(path=str(p),sha256=hashlib.file_digest(f,'sha256').hexdigest())
def save(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,default=str,allow_nan=False)
def table(p,x):
    if not x:return
    with Path(p).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(x[0]));w.writeheader();w.writerows(x)
def cell(r):return r['item']['period']+'/'+r['process']['scenario']
def verify(r):
    assert r['passed']
    for b in r['files'].values():assert bound(b['path'])['sha256']==b['sha256']
def main():
    start=time.monotonic();run=ROOT/'timing_run_v2';assert read(run/'timing_complete.json')['passed']
    out=ROOT/'timing_analysis_v1';out.mkdir(exist_ok=False)
    source=REPO/'scripts/kalcb_research_decision_audit_v1.py'
    spec=importlib.util.spec_from_file_location('c18_independent_accounting',source);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    collections={'control':{cell(r):r for r in read(OLD/'evaluations/v6_h0.seal.json')['receipts']},
        'all_delay':{cell(r):r for r in read(OLD/'baseline_delay.json')['receipts'].values()}}
    for arm in ['entry','exit']:collections[arm]={cell(r):r for r in read(run/('evaluations/'+arm+'.json'))['receipts']}
    save(out/'registration.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),program=bound(__file__),accounting_program=bound(source),
        design=bound(REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0043.md'),timing=bound(run/'registration.json'),
        estimands='Separate actual entry-only, exit-only and all-delay account effects, plus all-entry-exit interaction; price attribution descriptive',
        bootstrap='4000 paired session blocks5/10/20; fixed original seed; no scenario pseudo-samples',native_evaluations=0))
    accounts=[];trades={};vectors={};dates=[];lives=[];effects=[];inputs=[]
    for arm,records in collections.items():
        assert len(records)==8
        for key,r in sorted(records.items()):
            verify(r);inputs.extend(r['files'].values());a,t=audit.account(r['folder']);accounts.append(dict(arm=arm,cell=key,**a));trades[arm,key]=t
            daily=read(Path(r['folder'])/'daily.json');vectors[arm,key]=np.diff([1e8]+[d['equity'] for d in daily])/1e6
            dates.extend(dict(arm=arm,cell=key,day=d['day'],pnl_pp=float(v)) for d,v in zip(daily,vectors[arm,key]))
            print('Independently reconstructed',arm,key,flush=True)
    for arm in ['entry','exit','all_delay']:
        for key in sorted(collections['control']):
            groups=[]
            for name in ['control',arm]:
                g=defaultdict(list)
                for n,t in enumerate(trades[name,key],1):g[(t['symbol'],t['entry_decision_time'],t['route_metadata'].get('entry_route'))].append((n,t))
                groups.append(g)
            bg,cg=groups;matched=0.;sums=defaultdict(float);counts=defaultdict(int)
            for k in sorted(bg.keys()&cg.keys()):
                if len(bg[k])!=1 or len(cg[k])!=1:continue
                bn,b=bg[k][0];cn,c=cg[k][0];qb=b['qty'];dq=c['qty']-qb
                parts=dict(quantity=dq*(b['exit_price']-b['entry_price']),entry_price=-qb*(c['entry_price']-b['entry_price']),
                    exit_price=qb*(c['exit_price']-b['exit_price']),interaction=dq*((c['exit_price']-b['exit_price'])-(c['entry_price']-b['entry_price'])),cost=b['commission']-c['commission'])
                delta=c['net_pnl']-b['net_pnl'];assert abs(delta-sum(parts.values()))<.02
                matched+=delta
                for field,value in parts.items():sums[field]+=value
                counts['entry_clock_changed']+=b['entry_fill_time']!=c['entry_fill_time'];counts['exit_clock_changed']+=b['exit_fill_time']!=c['exit_fill_time']
                counts['quantity_changed']+=b['qty']!=c['qty'];counts['exit_reason_changed']+=b['exit_reason']!=c['exit_reason'];counts['matched']+=1
                lives.append(dict(arm=arm,cell=key,symbol=k[0],decision=k[1],route=k[2],baseline_row=bn,diagnostic_row=cn,
                    baseline_net=b['net_pnl'],diagnostic_net=c['net_pnl'],delta=delta,**parts,
                    baseline_quantity=qb,diagnostic_quantity=c['qty'],baseline_entry=b['entry_fill_time'],diagnostic_entry=c['entry_fill_time'],
                    baseline_exit=b['exit_fill_time'],diagnostic_exit=c['exit_fill_time'],baseline_reason=b['exit_reason'],diagnostic_reason=c['exit_reason'],
                    baseline_folder=collections['control'][key]['folder'],diagnostic_folder=collections[arm][key]['folder']))
            added=sum(t['net_pnl'] for k,ts in cg.items() if k not in bg for _,t in ts)
            displaced=-sum(t['net_pnl'] for k,ts in bg.items() if k not in cg for _,t in ts)
            delta=sum(t['net_pnl'] for t in trades[arm,key])-sum(t['net_pnl'] for t in trades['control',key])
            effects.append(dict(arm=arm,cell=key,delta_krw=delta,matched_krw=matched,added_krw=added,displaced_krw=displaced,
                ambiguous_residual_krw=delta-matched-added-displaced,**{k:sums[k] for k in ['quantity','entry_price','exit_price','interaction','cost']},
                **{k:counts[k] for k in ['matched','entry_clock_changed','exit_clock_changed','quantity_changed','exit_reason_changed']}))
    ci=[];factorial=[]
    for arm in ['entry','exit','all_delay','interaction']:
        delta={}
        for key in sorted(collections['control']):
            delta[key]=(vectors['all_delay',key]-vectors['entry',key]-vectors['exit',key]+vectors['control',key]) if arm=='interaction' else vectors[arm,key]-vectors['control',key]
            factorial.append(dict(contrast=arm,cell=key,delta_pp=float(sum(delta[key]))))
        for scenario in SCENARIOS+['equal_weight_aggregate']:
            vs={p:np.mean([delta[p+'/'+s] for s in (SCENARIOS if scenario=='equal_weight_aggregate' else [scenario])],axis=0) for p in ['is','oos']}
            ci.extend(dict(contrast=arm,scenario=scenario,**x) for x in audit.uncertainty(vs))
    for name,data in [('accounting',accounts),('daily_contributions',dates),('matched_lifetimes',lives),('descriptive_attribution',effects),('factorial_contrasts',factorial),('paired_uncertainty',ci)]:table(out/(name+'.csv'),data)
    save(out/'completion.json',dict(passed=True,seconds=time.monotonic()-start,accounts=len(accounts),source_artifacts=inputs,
        accounting_independent_of_native_engine=True,independent_market_validation=False,holdout_accesses=0,
        limitations=['all225sessions consumed development','price/quantity attribution is algebraic, not independently feasible','actual selective-delay accounts provide causal model interventions, not real broker latency measurements',
            'no quote or queue evidence','chronological periods/scenarios are not independent market replications','earlier selection not corrected by these CIs'],
        files={p.name:bound(p) for p in out.glob('*.csv')}))
if __name__=='__main__':main()
