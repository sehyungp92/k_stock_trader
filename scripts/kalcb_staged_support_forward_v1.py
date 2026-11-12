"""Fixed observational time partition; future native fills are not a policy."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def bound(p):
    p=Path(p).resolve()
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(p),'sha256':h}
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)


def partition(fills,at,close,expected_net):
    """Next-bar fills timestamped exactly at the decision are still future."""
    clock=datetime.fromisoformat(at);cash=0.;qty=0;before_fee=0.;after_cash=0.
    for f in fills:
        sign=1 if f['side']=='BUY' else -1
        flow=-sign*f['qty']*f['price']-f['metadata']['fee']
        if datetime.fromisoformat(f['timestamp'])<clock:
            cash+=flow;qty+=sign*f['qty'];before_fee+=f['metadata']['fee']
        else:after_cash+=flow
    if qty<0:raise ValueError('Negative inventory at decision')
    marked=cash+qty*close;remaining=after_cash-qty*close
    if abs(marked+remaining-expected_net)>.01:raise ValueError('Time partition fails cashflow identity')
    return {'quantity_at_decision':qty,'paid_fee_before_decision_krw':before_fee,
        'observed_net_marked_before35_krw':marked,'observed_remaining_net_after35_krw':remaining}


def summarize(rows):
    net=[r['observed_remaining_net_after35_krw'] for r in rows]
    normalized=[v/r['parent_entry_cost_krw']*10000 for r,v in zip(rows,net)]
    return {'unique_positions':len(rows),'sessions':len({r['day'] for r in rows}),
        'before35_marked_krw':sum(r['observed_net_marked_before35_krw'] for r in rows),
        'after35_remaining_krw':sum(net),'after35_positive_count':sum(v>0 for v in net),
        'after35_mean_original_entry_notional_bps':sum(normalized)/len(normalized) if normalized else None,
        'remaining_positive_krw':sum(max(0,v) for v in net),'remaining_negative_krw':sum(min(0,v) for v in net)}


def register(root):
    parent=root.parent;p=read(parent/'protocol.json');done=read(parent/'completion.json')
    tests=parent/'forward_preflight/tests_receipt.json';receipt=read(tests)
    if not receipt['passed'] or receipt['tests']!=3:raise ValueError('Time boundary tests missing')
    for b in receipt['programs']+[receipt['log']]:
        if bound(b['path'])['sha256']!=b['sha256']:raise ValueError('Changed time partition tests')
    paths=[parent/'protocol.json',parent/'completion.json',parent/'review_v1/completion.json']
    paths.append(tests)
    paths += [Path(b['path']) for b in done['outputs']]
    paths += [Path(c['folder'])/'fills.json' for c in p['cases']]
    root.mkdir(exist_ok=False)
    write(root/'protocol.json',{'registered_at_utc':datetime.now(timezone.utc).isoformat(),
        'formula':'For each original unique opening lifetime: net marked PnL at completed09:35=actual cashflows with physical fill timestamps STRICTLY<09:35 + heldqty*completedclose35. Remaining original native net=all later cashflows - heldqty*close35. Exact additive identity with original net; later original partial entries/fees included. Sameclock nextOPEN is future. NOT an achievable exit, return label for newaction or candidatePnL.',
        'population':'Every filled original lifetime in completed V49, separately all12accounts. Fixed priceconfirmed vsnotconfirmed partition. Preserve partial/notfullyfilled states separately; no day/issuer outcome filters or new threshold.',
        'outputs':'All positions, aggregate/month/issuer cashflow splits and same fullparent filled-no-exit-at35 subset. Existing fourday pilot separately identified, never counted as fresh confirmation.',
        'limits':'The classification already includes one observed postentry bar, so fulltrade PnL contrast is partly mechanical. This split measures remaining original path only. New execution prices, capital allocation and changed management still require matched native outcomes. No posthoc half-scaling.',
        'program':bound(__file__),'bindings':[bound(x) for x in dict.fromkeys(paths)],'study_complete':False})


def run(root):
    p=read(root/'protocol.json')
    for b in [p['program']]+p['bindings']:
        assert bound(b['path'])['sha256']==b['sha256'],b['path']
    import psutil
    me=psutil.Process();write(root/'supervisor_start.json',{'pid':me.pid,'process_created':me.create_time(),'command':me.cmdline()})
    parent=root.parent;spec=read(parent/'protocol.json');outputs=[];summary={}
    for case in spec['cases']:
        source=read(parent/'accounts'/Path(case['key']).with_suffix('.json'));grouped={};active={};qty=defaultdict(int)
        for f in read(Path(case['folder'])/'fills.json'):
            s=f['symbol']
            if f['side']=='BUY' and not qty[s]:
                key=s+'/'+datetime.fromisoformat(f['timestamp']).isoformat();active[s]=key;grouped[key]=[]
            grouped[active[s]].append(f);qty[s]+=f['qty']*(1 if f['side']=='BUY' else -1)
            if qty[s]<0:raise ValueError('Oversold source')
            if not qty[s]:del active[s]
        if active:raise ValueError('Unclosed source')
        rows=[]
        for row in source['rows']:
            if not row['position_id']:continue
            if 'completed_close35' not in row:raise ValueError('Missing supported prefix for filled original')
            rows.append({k:row[k] for k in ['position_id','symbol','day','category','parent_net_pnl_krw',
                         'parent_entry_cost_krw','parent_full_quantity_filled_before35','parent_sell_intent_at_or_before35','parent_sale_before35']}
                |partition(grouped[row['position_id']],row['completion_at'],row['completed_close35'],row['parent_net_pnl_krw']))
        groups={k:summarize([r for r in rows if r['category']==k]) for k in sorted({r['category'] for r in rows})}
        eligible=[r for r in rows if r['parent_full_quantity_filled_before35'] and not r['parent_sell_intent_at_or_before35'] and not r['parent_sale_before35']]
        safe={k:summarize([r for r in eligible if r['category']==k]) for k in groups}
        result={'account':case['key'],'rows':rows,'groups':groups,'fullyfilled_noexit_original_subset':safe,
            'months':{month:{k:summarize([r for r in rows if r['day'][:7]==month and r['category']==k]) for k in groups} for month in sorted({r['day'][:7] for r in rows})},
            'issuers':{s:{k:summarize([r for r in rows if r['symbol']==s and r['category']==k]) for k in groups} for s in sorted({r['symbol'] for r in rows})},
            'scope':'Original observed temporal cashflow partition, no counterfactual gains or newaction labels.'}
        file=root/'accounts'/Path(case['key']).with_suffix('.json');write(file,result);outputs.append(bound(file))
        summary[case['key']]={'groups':groups,'fullyfilled_noexit_original_subset':safe}
        print(json.dumps({'account':case['key'],'positions':len(rows),'groups':groups}),flush=True)
    write(root/'assessment.json',{'accounts':summary,'outputs':outputs,'study_complete':False})
    write(root/'completion.json',{'complete':True,'assessment':bound(root/'assessment.json'),'outputs':outputs,'study_complete':False})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['register','run']);ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    register(a.root) if a.mode=='register' else run(a.root)
