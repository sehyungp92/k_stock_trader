"""Read-only C13 receipt, attribution and prospectively declared trace closeout."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from scripts.kalcb_opportunity_closeout_v1 import bound, read, write, verify, journal, first_submission

REPO = Path(__file__).resolve().parents[1]
SOURCE = Path('E:/CodexResearch/k_stock_trader/kalcb_fixed_veto_20261017/v2')
DESIGN = REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0037-verification.md'
FORCED = ['2026-04-08/047040','2026-04-15/047040','2026-04-23/403870',
          '2026-05-29/035720','2026-05-29/066570','2026-06-04/004170','2026-05-12/950160']


def main(output):
    output.mkdir(exist_ok=False)
    done=read(SOURCE/'completion.json')
    assert done['complete']
    verify(done['registration']); verify(done['assessment'])
    p=read(SOURCE/'protocol.json'); a=read(SOURCE/'assessment.json')
    reg=read(SOURCE/'registration.json'); verify(reg['protocol']); verify(reg['preflight'])
    probes=read(SOURCE/'runtime_preflight.json')
    assert len(probes)==6 and all(x['reached_replay_boundary'] and x['hooks_restored'] and not x['financial_replay_executed'] for x in probes.values())
    sources=[bound(DESIGN),bound(Path(__file__)),bound(REPO/'scripts/kalcb_opportunity_closeout_v1.py')]
    sources += [bound(SOURCE/n) for n in ('completion.json','registration.json','protocol.json','assessment.json','completed_cases.json','runtime_preflight.json')]
    receipts=[]
    for row in read(SOURCE/'completed_cases.json'):
        verify(row['completion']); c=row['case']; f=SOURCE/'cases'/c['id']
        for name,flag in [('activation.json','passed'),('independent_account_verification.json','passed'),('funding_review.json','verified')]:
            if not row['verdict']['qualified'] and name!='activation.json': continue
            data=read(f/name); assert data[flag]
            for binding in data.get('bindings',[]): verify(binding)
        if c['kind']=='bridge': assert read(f/'identity_bridge.json')['passed']
        sources += [bound(x) for x in f.iterdir() if x.is_file()]
        receipts.append(dict(case=c['id'],qualified=row['verdict']['qualified'],bridge=c['kind']=='bridge'))
    folders=set(); selections={}
    for s,c in a['cells'].items():
        for arm,v in c['comparisons'].items():
            tr=SOURCE/'traces'/(s+'_versus_'+arm+'.json'); t=read(tr); sources.append(bound(tr))
            changed=sorted(((k,x) for k,x in t['deltas'].items() if abs(x)>0.01),key=lambda z:(z[1],z[0]))
            chosen=list(FORCED)
            if changed: chosen += [changed[0][0],changed[-1][0],changed[len(changed)//2][0]]
            actual=SOURCE/'cases'/('joint_'+s); control=Path(p['controls'][arm][s]['folder'])
            folders.update([actual,control]); selections[(s,arm)]=(actual,control,sorted(set(chosen)),t)
    targets=set(FORCED)
    for actual,control,chosen,t in selections.values(): targets.update(chosen)
    for f in folders:
        for name in ('orders.jsonl.gz','decisions.jsonl.gz','fills.json','trades.json','daily.json','equity.json'):
            sources.append(bound(f/name))
    sources=list({x['path']:x for x in sources}.values())
    write(output/'registration.json',dict(design=bound(DESIGN),bindings=sources,financial_replays=0,fits=0,forced=FORCED))
    cache={}
    for f in sorted(folders):
        write(output/'status.json',dict(phase='journals',completed=len(cache),total=len(folders),folder=str(f)))
        cache[f]=journal(f,targets)
    results={}
    for (s,arm),(actual,control,chosen,t) in selections.items():
        v=a['cells'][s]['comparisons'][arm]; x=cache[actual]; y=cache[control]
        assert abs(x['net']-v['activity']['summary']['net_pnl_krw'])<.01
        assert abs(y['net']-v['control_activity']['summary']['net_pnl_krw'])<.01
        d=v['activity_delta']; reconciliation=d['shared_quantity_at_control_unit_pnl_krw']+d['shared_unit_outcome_at_treatment_qty_krw']+d['added_pnl_krw']-d['removed_control_pnl_krw']
        assert abs(reconciliation-v['net_delta_krw'])<.01
        days=defaultdict(float)
        for k,z in t['deltas'].items(): days[k[:10]]+=z
        days=sorted(days.items(),key=lambda z:(z[1],z[0]))
        first=first_submission(x['submits'],y['submits'])
        selected={k:dict(actual=x['affected'][k],control=y['affected'][k],net_delta=x['affected'][k]['net']-y['affected'][k]['net']) for k in chosen}
        dates={k[:10] for k in chosen}
        if first:
            for o in (first['actual'],first['control']):
                if o: dates.add(o['episode'][:10])
        books={date:{label:[o for o in j['submits'] if o['episode'][:10]==date] for label,j in [('actual',x),('control',y)]} for date in sorted(dates)}
        report=dict(net_delta=v['net_delta_krw'],metrics=v['metrics'],control_metrics=v['control_metrics'],
            useful_days=v['useful_days'],control_useful_days=v['control_useful_days'],activity=v['activity']['summary'],control_activity=v['control_activity']['summary'],
            attribution=d,monthly=v['path']['monthly'],first_submission_divergence=first,
            historical_first_order_divergence=t['first_order_divergence'],days=dict(days),
            best_date=days[-1] if days else None,worst_date=days[0] if days else None,
            net_excluding_best_date=v['net_delta_krw']-(days[-1][1] if days else 0),
            selected=selected,books=books)
        write(output/'comparisons'/(s+'_versus_'+arm+'.json'),report)
        results[s+'/'+arm]={k:z for k,z in report.items() if k not in ('selected','books')}
    for b in sources: verify(b)
    write(output/'review.json',dict(valid=True,criteria=a['criteria'],receipts=receipts,comparisons=results,bindings_verified=len(sources),
        scope='Observed paths; reused OOS development; no counterfactual sums or participation comparisons against unqualified accounts.',economic_objective_achieved=False))
    write(output/'completion.json',dict(complete=True,review=bound(output/'review.json'),registration=bound(output/'registration.json'),financial_replays=0,fits=0))
    write(output/'status.json',dict(phase='complete',at=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);main(ap.parse_args().output)
