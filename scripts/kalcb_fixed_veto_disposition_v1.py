"""Finite read-only C13 participation/funding explanation; no fitted policy."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from scripts.kalcb_opportunity_closeout_v1 import bound, read, write, verify, journal, lines

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_fixed_veto_20261017')
PANEL=Path('E:/CodexResearch/k_stock_trader/kalcb_structural_preparation_20261016/entry_lifecycle_v2')
DESIGN=REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0037-disposition-design.md'


def days(activity):
    groups=defaultdict(list)
    for k,r in sorted(activity['opportunities'].items()): groups[k[:17]].append(dict(key=k,**r))
    result={}
    for k,rr in groups.items():
        material=any(r['material'] for r in rr); net=sum(r['net_pnl_krw'] for r in rr); equity=rr[0]['decision_equity_krw']
        result[k[7:17]+'/'+k[:6]]=dict(net=net,material=material,reference_profit=.0001*equity,
            useful=material and net>=.0001*equity,opportunities=rr)
    return result


def main(out):
    out.mkdir(exist_ok=False)
    source=ROOT/'v2'; completed=read(source/'completion.json');verify(completed['assessment'])
    pdone=read(PANEL/'completion.json'); verify(pdone['panel']);verify(pdone['registration'])
    a=read(source/'assessment.json');p=read(source/'protocol.json')
    sources=[bound(DESIGN),bound(Path(__file__)),bound(REPO/'scripts/kalcb_opportunity_closeout_v1.py'),
        bound(source/'assessment.json'),bound(source/'protocol.json'),bound(PANEL/'completion.json'),pdone['panel'],pdone['registration']]
    for s,c in a['cells'].items():
        f=source/'cases'/('joint_'+s)
        sources += [bound(f/n) for n in ['orders.jsonl.gz','decisions.jsonl.gz','fills.json','trades.json','funding_review.json']]
    write(out/'registration.json',dict(bindings=sources,design=bound(DESIGN),fits=0,native_replays=0))
    comparisons={};targets=set();funding={}
    for s,c in a['cells'].items():
        for arm,v in c['comparisons'].items():
            aa,bb=days(v['activity']),days(v['control_activity'])
            assert sum(x['useful'] for x in aa.values())==v['useful_days']
            assert sum(x['useful'] for x in bb.values())==v['control_useful_days']
            gained=sorted(k for k,x in aa.items() if x['useful'] and not bb.get(k,{}).get('useful',False))
            lost=sorted(k for k,x in bb.items() if x['useful'] and not aa.get(k,{}).get('useful',False))
            changes={k:dict(actual=aa.get(k),control=bb.get(k)) for k in gained+lost}
            comparisons[s+'/'+arm]=dict(gained=gained,lost=lost,changes=changes)
            targets.update(changes)
            if arm=='fixed':
                for k in v['activity_delta']['removed_keys']:targets.add(k[7:17]+'/'+k[:6])
        f=source/'cases'/('joint_'+s); review=read(f/'funding_review.json')
        rejected={x['decision_available_at'][:10]+'/'+x['symbol'] for x in review['rejected_buy_events']}
        targets.update(rejected); funding[s]=dict(review=review,rejected=sorted(rejected))
    # Same original labels: no newly simulated target or quantity interpolation.
    panel={k:[] for k in targets}
    for row in lines(PANEL/'panel.jsonl.gz'):
        if row['base_id'] in panel:
            panel[row['base_id']].append({k:row[k] for k in ['action_id','scenario','quantity_scale','planned_qty','planned_notional',
                'available_at30','selected_prior100','actual_prior100','physical_capacity_pass','forecast_core','label_valid','deadline_violation','unresolved',
                'native_status','prefix_supported','states','net_reconstructed_krw','marked_net_at35_krw','remaining_net_after35_krw','pending_qty_at35',
                'buy_qty_before35','original_exit_intent_by35','features30']})
    for s,v in funding.items():
        j=journal(source/'cases'/('joint_'+s),targets)
        v['episodes']={k:j['affected'][k] for k in v['rejected']}
        v['books']={d:[o for o in j['submits'] if o['episode'][:10]==d] for d in sorted({k[:10] for k in v['rejected']})}
        v['partial_requests']=[o for o in j['submits'] if 0<o['filled_qty']<o['qty']]
        v['unfilled_requests']=[o for o in j['submits'] if not o['filled_qty']]
    for b in sources:verify(b)
    write(out/'assessment.json',dict(comparisons=comparisons,funding=funding,panel=panel,distinct_target_issuer_days=len(targets),
        panel_covered=sum(bool(x) for x in panel.values()),bindings_verified=len(sources),
        scope='Observed funded sets and original isolated labels. No interpolation, new fit, executable discriminator or hypothetical funded PnL.'))
    write(out/'completion.json',dict(complete=True,assessment=bound(out/'assessment.json'),registration=bound(out/'registration.json'),fits=0,native_replays=0,at=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);main(ap.parse_args().output)
