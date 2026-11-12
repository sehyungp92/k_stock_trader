"""Independent arithmetic/clock review using original fills; no producer import."""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def near(x,y):
    if abs(x-y)>max(1e-6,abs(y)*1e-10):raise ValueError(f'Mismatch {x} vs {y}')
def stamp(x):return datetime.fromisoformat(x)


def run(root):
    import psutil
    identity=read(root/'supervisor_start.json')
    try:
        process=psutil.Process(identity['pid'])
        if abs(process.create_time()-identity['process_created'])<.01 and process.cmdline()==identity['command']:
            raise ValueError('Exact producer still active')
    except psutil.NoSuchProcess:pass
    p=read(root/'protocol.json');done=read(root/'completion.json')
    bindings=p['programs']+p['bindings']+done['outputs']+[done['assessment']]
    for b in bindings:
        if digest(b['path'])!=b['sha256']:raise ValueError('Changed binding '+b['path'])
    out=root/'review_v1';out.mkdir(exist_ok=False);me=psutil.Process()
    write(out/'supervisor_start.json',{'pid':me.pid,'process_created':me.create_time(),'command':me.cmdline()})
    checks=[]
    for c in p['cases']:
        previous=read(Path(p['source'])/'accounts'/Path(c['key']).with_suffix('.json'))
        old={r['order_id']:r for r in previous['rows']}
        result=read(root/'accounts'/Path(c['key']).with_suffix('.json'))
        assert len(result['rows'])==len(old) and {r['order_id'] for r in result['rows']}==set(old)
        allfills=read(Path(c['folder'])/'fills.json');orders=defaultdict(list);seen=set()
        for f in allfills:
            eid=f['metadata']['execution_id'];assert eid not in seen;seen.add(eid)
            if f['side']=='BUY':orders[f['order_id']].append(f)
        attached=set()
        for r in result['rows']:
            original=old[r['order_id']];at=stamp(original['completion_at']);fs=orders[r['order_id']]
            a=sum(f['qty'] for f in fs if stamp(f['timestamp'])<at)
            later=[f for f in fs if stamp(f['timestamp'])>=at];b=sum(f['qty'] for f in later)
            assert r['before_qty']==a==original['parent_qty_bought_before35'] and r['later_qty']==b
            assert a+b<=original['planned_qty'] and r['requested_unfilled_at35']==original['planned_qty']-a
            supported=original['category'] in {'price_confirmed','price_not_confirmed','single_share_unchanged'}
            if supported:
                raw=original['raw_prefix'];vwap=sum((x[2]+x[3]+x[4])*x[5] for x in raw)/(3*sum(x[5] for x in raw))
                weak=not(raw[6][4]>=raw[5][4] and raw[6][4]>=vwap)
                assert r['weak_price_at35']==weak
            else:weak=False
            blocked=original['parent_sell_intent_at_or_before35'] or original['parent_sale_before35']
            assert r['potential_remainder_observation']==bool(supported and weak and original['planned_qty']>a and not blocked)
            assert r['later_execution_ids']==[f['metadata']['execution_id'] for f in later]
            near(r['later_purchase_notional_krw'],sum(f['qty']*f['price'] for f in later))
            near(r['later_purchase_fee_krw'],sum(f['metadata']['fee'] for f in later))
            if supported:near(r['later_price_minus_original_reference_krw'],sum(f['qty']*(f['price']-original['original_reference']) for f in later))
            assert r['position_id']==original['position_id'] and r['candidate_pnl'] is None and not r['pending_status_at35_qualified']
            near(r['original_whole_lifetime_net_krw'],original['parent_net_pnl_krw'])
            near(r['original_entry_cost_krw'],original['parent_entry_cost_krw'])
            if r['position_id']:
                assert r['position_id'] not in attached;attached.add(r['position_id'])
        groups={'all':result['rows'],
            'potential_remainders':[r for r in result['rows'] if r['potential_remainder_observation']],
            'potential_with_later_fills':[r for r in result['rows'] if r['potential_remainder_observation'] and r['later_qty']>0],
            'confirmed_with_later_fills':[r for r in result['rows'] if r['prefix_supported'] and not r['weak_price_at35'] and r['later_qty']>0]}
        def check(g,rs):
            assert g['orders']==len(rs) and g['sessions']==len({r['day'] for r in rs})
            assert g['issuer_days']==len({(r['symbol'],r['day']) for r in rs})
            assert g['original_winners']==sum(r['original_whole_lifetime_net_krw']>0 for r in rs)
            assert g['later_filled_orders']==sum(r['later_qty']>0 for r in rs)
            for field in ['before_qty','later_qty','later_purchase_notional_krw','later_purchase_fee_krw',
                          'original_entry_cost_krw','original_whole_lifetime_net_krw','later_price_minus_original_reference_krw']:
                near(g[field],sum(r[field] or 0 for r in rs))
            near(g['original_positive_whole_lifetime_net_krw'],sum(max(0,r['original_whole_lifetime_net_krw']) for r in rs))
            near(g['original_negative_whole_lifetime_net_krw'],sum(min(0,r['original_whole_lifetime_net_krw']) for r in rs))
        for k,rs in groups.items():check(result['groups'][k],rs)
        for m,gs in result['months'].items():
            for k,g in gs.items():check(g,[r for r in groups[k] if r['day'][:7]==m])
        for symbol,g in result['issuers'].items():check(g,[r for r in groups['potential_with_later_fills'] if r['symbol']==symbol])
        checks.append({'account':c['key'],'orders':len(old),'unique_lifetimes':len(attached),'passed':True})
        print(json.dumps(checks[-1]),flush=True)
    for b in bindings:
        if digest(b['path'])!=b['sha256']:raise ValueError('Post-review drift')
    write(out/'review.json',{'passed':True,'accounts':checks,'source_bindings':len(bindings),
        'scope':'Independent original fill grouping, strict same-clock boundary, reviewed raw-prefix predicate, once-only parent attribution and all month/issuer sums. Historical full-account/MTM/rawsource reviews reused. No cancellation policy results or exact pending-state qualification.', 'study_complete':False})
    write(out/'completion.json',{'complete':True,'passed':True,'review':{'path':str(out/'review.json'),'sha256':digest(out/'review.json')},'study_complete':False})


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);args=a.parse_args();run(args.root)
