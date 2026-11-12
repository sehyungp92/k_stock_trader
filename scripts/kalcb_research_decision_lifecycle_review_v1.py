"""Read-only classification of authenticated incumbent native lifecycles.

No replay, counterfactual target, predictive fit or stopped omission operation.
Repeated rejects stay observations of a route, not additional opportunities.
"""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import csv,gzip,json,hashlib,time,argparse

ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_research_decision_20261018/v1')
OLD=ROOT.parent.parent/'kalcb_round3_20261017/attempt_2/round_run_v2'
OUT=ROOT/'lifecycle_review_v1'
CATEGORIES={
    'entry_signal_window_expired':'route_time_window_expired',
    'entry_route_session_limit':'route_session_counter_exhausted',
    'max_positions':'position_or_pending_occupancy',
    'heat_cap_exceeded':'risk_capacity',
    'qty_zero':'sizing_zero_cause_not_further_inferred',
    'entry_min_bar_ret':'current_setup_condition',
    'entry_min_vwap_ret':'current_setup_condition',
    'entry_context_min:first30_rel_volume':'current_setup_condition',
    'entry_quality_votes':'current_setup_condition',
    'pullback_acceptance_missing':'current_setup_condition',
}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def bound(p):
    with Path(p).open('rb') as f:return dict(path=str(p),sha256=hashlib.file_digest(f,'sha256').hexdigest())
def save(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,default=str,allow_nan=False)
def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:
        for n,line in enumerate(f,1):yield n,json.loads(line)
def csvsave(p,items):
    if not items:return
    with Path(p).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
def clock(value):return datetime.fromisoformat(str(value).replace(' ','T'))

def register():
    OUT.mkdir(exist_ok=False)
    audit=read(ROOT/'saved_evidence_v1/registration.json')
    # Existing complete-account bindings are already independently authenticated.
    seal=read(OLD/'evaluations/v6_h0.seal.json')
    controls=[]
    for receipt in seal['receipts']:
        folder=Path(receipt['folder'])
        controls.append(dict(cell=receipt['item']['period']+'/'+receipt['process']['scenario'],folder=str(folder),
            files=[bound(folder/n) for n in ['decisions.jsonl.gz','orders.jsonl.gz','fills.json','trades.json','horizon_actions.json']]))
    save(OUT/'registration.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),program=bound(__file__),
        controls=controls,categories=CATEGORIES,financial_evaluations=0,maximum_seconds=1200,
        study_start_epoch=audit.get('study_start_epoch'),
        definitions=[
            'Entry lifetimes are submitted BUY order IDs, with all partial fills and final observed order events retained.',
            'Rejection states group by issuer-day and native route; timestamps/rows remain separate observations, not independent opportunities.',
            'A later actual BUY fill on the same day/route is observed subsequent reach; it does not prove the same setup stayed valid or identify genuinely new information.',
            'Only explicit broker buying_power rejection is classified as unfunded; qty_zero is ambiguous unless its original metadata explains it.',
            'Signal-window expiry and route-counter exhaustion are native state limits, not new adverse market forecasts.',
            'Only gates actually evaluated before short circuit are observed; absent downstream gates remain unavailable.',
            'Pending-cap quantities are counted once at actual initial submit; copied reservation metadata in amendments/updates is not a new decision.',
            'Original checkpoint snapshots are separate information clocks; repeated guards alone are not a new forecast.',
            'This is read-only cohort attribution; no omission account forks, VETO re-entry or new economic arms.'],
        decision_use='Describe original candidate reach, rejection, cancellation and funding coverage; no treatment selection.'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--register',action='store_true');args=ap.parse_args()
    if args.register:register();return
    start=time.monotonic();reg=read(OUT/'registration.json');assert bound(__file__)==reg['program']
    for item in reg['controls']:
        for b in item['files']:assert bound(b['path'])==b
    lifetimes=[];states=[];caps=[];vetoes=[];totals=[]
    for item in reg['controls']:
        folder=Path(item['folder']);cell=item['cell'];orders={};order_events=defaultdict(list)
        filled=defaultdict(list);submitted=defaultdict(list);decisions=defaultdict(list);counts=Counter()
        for n,fill in enumerate(read(folder/'fills.json'),1):
            if fill['side']=='BUY':filled[fill['order_id']].append((n,fill))
        for n,r in rows(folder/'orders.jsonl.gz'):
            order=r['order']
            if order['side']!='BUY':continue
            oid=order['order_id'];meta=order.get('metadata',{});order_events[oid].append((n,r))
            if r['event']=='submit':
                assert oid not in orders;orders[oid]=(n,order)
                key=(str(order['submitted_at'])[:10],order['symbol'],meta.get('entry_route','unavailable'))
                submitted[key].append(oid)
                cap=meta.get('pending_funding_reservation')
                if cap:
                    caps.append(dict(cell=cell,day=key[0],symbol=key[1],route=key[2],order_id=oid,source_row=n,
                        original_qty=cap['original_qty'],capped_qty=cap['capped_qty'],reduced=cap['capped_qty']<cap['original_qty'],
                        source=str(folder/'orders.jsonl.gz')))
        for oid,(n,order) in orders.items():
            fs=filled.get(oid,[]);events=order_events[oid]
            terminal=[(row,r) for row,r in events if r['event'] in ('reject','cancel','expire')]
            lifetimes.append(dict(cell=cell,day=str(order['submitted_at'])[:10],symbol=order['symbol'],
                route=order.get('metadata',{}).get('entry_route','unavailable'),order_id=oid,submit_row=n,
                submitted_quantity=order['qty'],filled_quantity=sum(f['qty'] for _,f in fs),fill_fragments=len(fs),
                first_fill=min((f['timestamp'] for _,f in fs),default=None),last_fill=max((f['timestamp'] for _,f in fs),default=None),
                terminal_events='|'.join(r['event']+':'+str(r.get('reason')) for _,r in terminal),
                terminal_rows='|'.join(str(row) for row,_ in terminal),
                explicit_unfunded=any(r['event']=='reject' and r.get('reason')=='buying_power' for _,r in terminal),
                source=str(folder/'orders.jsonl.gz'),fills_source=str(folder/'fills.json')))
        for n,r in rows(folder/'decisions.jsonl.gz'):
            counts['all_decisions']+=1
            if r['decision_code']!='entry_rejected':continue
            meta=r.get('metadata',{});key=(r['timestamp'][:10],r['symbol'],meta.get('entry_route','unavailable'))
            decisions[key].append((n,r));counts['rejected_observations']+=1
        for (day,symbol,route),seq in sorted(decisions.items()):
            later=[(oid,f) for oid in submitted.get((day,symbol,route),[]) for _,f in filled.get(oid,[])]
            by_reason=defaultdict(list)
            for n,r in seq:by_reason[r['reason']].append((n,r))
            for reason,observations in by_reason.items():
                first=min(clock(r['timestamp']) for _,r in observations);last=max(clock(r['timestamp']) for _,r in observations)
                after=[(oid,f) for oid,f in later if clock(f['timestamp'])>first]
                evaluated=sorted({str(g.get('filter_name')) for _,r in observations for g in r.get('metadata',{}).get('filter_decisions',[])})
                states.append(dict(cell=cell,day=day,symbol=symbol,route=route,reason=reason,
                    category=CATEGORIES.get(reason,'unclassified_native_reason'),observations=len(observations),
                    first_observed=first.isoformat(),last_observed=last.isoformat(),
                    same_route_later_filled_orders=len({oid for oid,_ in after}),
                    first_subsequent_fill=min((str(f['timestamp']) for _,f in after),default=None),
                    evaluated_gate_names='|'.join(evaluated),downstream_unevaluated_gates='unknown',
                    source_rows='|'.join(str(n) for n,_ in observations),source=str(folder/'decisions.jsonl.gz')))
        for n,r in enumerate(read(folder/'horizon_actions.json'),1):
            if r['kind']=='veto':vetoes.append(dict(cell=cell,issuer_day=r['key'],available_at=r['available_at'],
                requested_qty=r['requested_qty'],chosen_qty=r['chosen_qty'],source_row=n,source=str(folder/'horizon_actions.json')))
        own=[r for r in lifetimes if r['cell']==cell];cs=[r for r in caps if r['cell']==cell]
        totals.append(dict(cell=cell,**counts,distinct_rejection_route_issuer_days=len(decisions),submitted_entry_orders=len(own),
            filled_entry_orders=sum(bool(r['filled_quantity']) for r in own),unfilled_entry_orders=sum(not r['filled_quantity'] for r in own),
            explicit_unfunded_entry_orders=sum(r['explicit_unfunded'] for r in own),
            actual_pending_cap_records=len(cs),quantity_reduced_cap_records=sum(r['reduced'] for r in cs),
            veto_records=sum(r['cell']==cell for r in vetoes)))
        assert time.monotonic()-start<reg['maximum_seconds']
    csvsave(OUT/'entry_lifetimes.csv',lifetimes);csvsave(OUT/'rejection_states.csv',states)
    csvsave(OUT/'pending_cap_decisions.csv',caps);csvsave(OUT/'veto_decisions.csv',vetoes);csvsave(OUT/'summary.csv',totals)
    expected={r['cell']:int(r['decision_rows']) for r in csv.DictReader((ROOT/'saved_evidence_v1/cohort_census.csv').open())}
    assert all(r['all_decisions']==expected[r['cell']] for r in totals)
    save(OUT/'completion.json',dict(passed=True,seconds=time.monotonic()-start,financial_evaluations=0,holdout_accesses=0,
        registration=bound(OUT/'registration.json'),files={p.name:bound(p) for p in OUT.glob('*.csv')},
        caveat='Source-linked descriptive reach, not local counterfactual value, causal renewal evidence or independent market validation.'))
    print(json.dumps(dict(passed=True,seconds=time.monotonic()-start,cells=len(totals))))

if __name__=='__main__':main()
