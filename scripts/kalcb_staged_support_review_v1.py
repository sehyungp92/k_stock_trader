"""Independent raw prefix, position cashflow and partition review; no fitting."""
import argparse
from collections import defaultdict
from datetime import datetime,timedelta,timezone
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def stamp(x):return datetime.fromisoformat(x)
def near(a,b):
    if abs(a-b)>max(.000001,abs(b)*1e-10):raise ValueError(f'Independent mismatch {a} vs {b}')


def run(root):
    import psutil
    identity=read(root/'supervisor_start.json')
    try:
        alive=psutil.Process(identity['pid'])
        if abs(alive.create_time()-identity['process_created'])<.01 and alive.cmdline()==identity['command']:
            raise ValueError('Exact producer still active')
    except psutil.NoSuchProcess:pass
    p=read(root/'protocol.json');done=read(root/'completion.json')
    bindings=p['programs']+p['bindings']+done['outputs']+[done['assessment']]
    for b in bindings:
        if digest(b['path'])!=b['sha256']:raise ValueError('Changed binding '+b['path'])
    out=root/'review_v1';out.mkdir(exist_ok=False);me=psutil.Process()
    write(out/'supervisor_start.json',{'pid':me.pid,'process_created':me.create_time(),'command':me.cmdline()})
    connections=[sqlite3.connect('file:'+Path(db).as_posix()+'?mode=ro',uri=True) for db in p['databases']]
    cache={};checks=[];totalrows=0;totalprefix=0
    try:
        for case in p['cases']:
            folder=Path(case['folder']);result=read(root/'accounts'/Path(case['key']).with_suffix('.json'))
            obs=read(folder/'observer_evidence.json')['entry_exposures']
            exposures={e['order_id']:e for e in obs if e['action']['metadata'].get('entry_route')=='first30_open_anchor'}
            if len(exposures)!=sum(e['action']['metadata'].get('entry_route')=='first30_open_anchor' for e in obs):
                raise ValueError('Duplicate source submitted intent')
            active={};qty=defaultdict(int);lifetimes={};seen=set()
            for fill in read(folder/'fills.json'):
                if fill['metadata']['execution_id'] in seen:raise ValueError('Duplicate physical execution')
                seen.add(fill['metadata']['execution_id']);s=fill['symbol']
                if fill['side']=='BUY' and not qty[s]:
                    key=s+'/'+stamp(fill['timestamp']).isoformat();active[s]=key;lifetimes[key]=[]
                lifetimes[active[s]].append(fill);qty[s]+=fill['qty']*(1 if fill['side']=='BUY' else -1)
                if qty[s]<0:raise ValueError('Oversold source inventory')
                if not qty[s]:del active[s]
            if active:raise ValueError('Unclosed source inventory')
            sells=defaultdict(list)
            with gzip.open(folder/'orders.jsonl.gz','rt',encoding='utf-8') as f:
                for line in f:
                    d=json.loads(line)
                    if d['event']=='submit' and d['order']['side']=='SELL':sells[d['order']['symbol']].append(d['order'])
            if len(result['rows'])!=len(exposures) or {r['order_id'] for r in result['rows']}!=set(exposures):
                raise ValueError('Incomplete selected-parent exposure population')
            attached=set();groupvalues=defaultdict(list)
            for row in result['rows']:
                e=exposures[row['order_id']];action=e['action'];at=stamp(e['decision_available_at'])+timedelta(minutes=5)
                assert at==stamp(row['completion_at']) and row['planned_qty']==action['qty']
                key=action['symbol'],at.date().isoformat()
                if key not in cache:
                    raw=[]
                    for conn in connections:
                        raw.extend(conn.execute('SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? AND day=? ORDER BY ts',key))
                    cache[key]=[list(x) for x in sorted(raw) if stamp(x[0])+timedelta(minutes=5)<=at]
                raw=cache[key]
                if (at.hour,at.minute)!=(9,35):expected='unsupported_opening_clock'
                elif len(raw)!=7 or any(stamp(b[0])!=at.replace(hour=9,minute=0)+timedelta(minutes=5*i) for i,b in enumerate(raw)) or any(min(b[1:])<=0 for b in raw):expected='invalid_prefix'
                else:
                    near(raw[5][4],action['metadata']['entry_price_ref']);assert raw==row['raw_prefix'];totalprefix+=7
                    weighted=sum((float(b[2])+float(b[3])+float(b[4]))*float(b[5]) for b in raw)/(3*sum(float(b[5]) for b in raw))
                    near(weighted,row['completed_vwap35']);near(raw[6][4],row['completed_close35'])
                    near((raw[6][4]/raw[5][4]-1)*10000,row['completed_quote_change_bps'])
                    passed=raw[6][4]>=raw[5][4] and raw[6][4]>=weighted
                    assert row['price_confirmed']==passed
                    expected='single_share_unchanged' if action['qty']<2 else 'price_confirmed' if passed else 'price_not_confirmed'
                assert row['category']==expected
                pos=row['position_id'];fs=[]
                if pos:
                    if pos in attached:raise ValueError('Duplicated lifetime PnL')
                    attached.add(pos);fs=lifetimes[pos]
                    assert fs[0]['order_id']==row['order_id'] and fs[0]['side']=='BUY'
                    gross=sum(f['qty']*f['price']*(1 if f['side']=='SELL' else -1) for f in fs)
                    fees=sum(f['metadata']['fee'] for f in fs)
                    near(gross-fees,row['parent_net_pnl_krw']);near(fees,row['parent_fees_krw'])
                    near(sum(f['qty']*f['price'] for f in fs if f['side']=='BUY'),row['parent_entry_cost_krw'])
                else:
                    assert not any(f['order_id']==row['order_id'] for fs0 in lifetimes.values() for f in fs0)
                    assert row['parent_net_pnl_krw']==row['parent_entry_cost_krw']==0
                bought=sum(f['qty'] for f in fs if f['side']=='BUY' and stamp(f['timestamp'])<at)
                assert bought==row['parent_qty_bought_before35']
                assert (bought==action['qty'])==row['parent_full_quantity_filled_before35']
                assert any(f['side']=='SELL' and stamp(f['timestamp'])<at for f in fs)==row['parent_sale_before35']
                blocked=any(stamp(fs[0]['timestamp'])<=stamp(o['metadata']['decision_available_at'])<=min(at,stamp(fs[-1]['timestamp'])) for o in sells[action['symbol']]) if fs else False
                assert blocked==row['parent_sell_intent_at_or_before35']
                before=e['account_before'];assert bool(before['positions'] or before['pending_orders'])==row['initial_nonflat_or_pending']
                groupvalues[expected].append(row)
            for category,rs in groupvalues.items():
                summary=result['groups'][category]
                assert len(rs)==summary['submitted_orders']
                near(sum(r['parent_net_pnl_krw'] for r in rs),summary['parent_net_krw'])
                near(sum(max(0,r['parent_net_pnl_krw']) for r in rs),summary['parent_positive_net_krw'])
                near(sum(min(0,r['parent_net_pnl_krw']) for r in rs),summary['parent_negative_net_krw'])
                assert sum(r['parent_net_pnl_krw']>0 for r in rs)==summary['parent_winner_count']
            totalrows+=len(result['rows']);checks.append({'account':case['key'],'orders':len(result['rows']),'unique_lifetimes':len(attached),'passed':True})
            print(json.dumps(checks[-1]),flush=True)
            del obs,exposures,lifetimes
    finally:
        for conn in connections:conn.close()
    report={'passed':True,'accounts':checks,'submitted_orders':totalrows,'raw_prefix_rows_with_dependent_repeats':totalprefix,
        'scope':'Independent rawprefix/decision boundaries/position cashflows/group partitions. Scenarios and priors dependent; no staged outcomes, portfolio qualification or independent validation.','study_complete':False}
    write(out/'review.json',report)
    write(out/'completion.json',{'passed':True,'complete':True,'review':{'path':str(out/'review.json'),'sha256':digest(out/'review.json')},'study_complete':False})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();run(a.root)
