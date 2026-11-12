"""Independent recorded source, execution and account review; no simulation."""
import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
from math import floor
from pathlib import Path
import sqlite3
import sys

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bound(p):return {'path':str(Path(p).resolve()),'sha256':digest(p)}
def verify(bs):
    for b in bs:
        if digest(b['path'])!=b['sha256']:raise ValueError('Changed binding '+b['path'])
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,default=str,allow_nan=False)
def dt(s):return datetime.fromisoformat(s)
def near(a,b,tol=.011):
    if abs(a-b)>tol:raise ValueError(f'Independent mismatch {a} != {b}')

def review(root):
    import psutil
    p=read(root/'protocol.json');identity=read(root/'supervisor_start.json')
    try:
        live=psutil.Process(identity['pid'])
        if abs(live.create_time()-identity['process_created'])<.01 and live.cmdline()==identity['command']:
            raise ValueError('Producer still active')
    except psutil.NoSuchProcess:pass
    done=read(root/'completion.json');verify([done['labels']]+p['programs']+p['bindings']+[p['runtime_manifest']])
    labels=read(done['labels']['path']);verify([v for r in labels['outputs'] for v in r.values()])
    manifest=read(p['runtime_manifest']['path']);verify(list(manifest['overrides'].values()))
    for origin_file in ('imports_before.json','imports_after.json'):verify(list(read(root/origin_file).values()))
    reg=root/'review_registration_v1.json'
    write(reg,{'program':bound(__file__),'inputs':[bound(root/'completion.json'),bound(root/'labels/completion.json')],
               'registered_at_utc':datetime.now(timezone.utc).isoformat(),'read_only':True})
    out=root/'review_v1';out.mkdir(exist_ok=False)
    me=psutil.Process();write(out/'supervisor_start.json',{'pid':me.pid,'process_created':me.create_time(),'command':me.cmdline()})
    cache={}
    def bars(symbol,day):
        if (symbol,day) not in cache:
            db=p['databases'][int(day>='2026-04-01')]
            with sqlite3.connect('file:'+Path(db).as_posix()+'?mode=ro',uri=True) as c:
                cache[symbol,day]={dt(ts):(o,h,l,cl,v) for ts,o,h,l,cl,v in c.execute('SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? AND day=? ORDER BY ts',(symbol,day))}
        return cache[symbol,day]
    rows=[];details={};totalmarks=0
    for output in labels['outputs']:
        value=read(output['label']['path'])
        with gzip.open(output['events']['path'],'rt',encoding='utf-8') as f:d=json.load(f)
        spec=d['spec'];symbol=spec['symbol'];orders={r['order']['order_id']:r['order'] for r in d['journal'] if r['event']=='submit'}
        cash=spec['initial_equity'];qty=0;bytime=defaultdict(list);executions=set();fees=0.;buys=defaultdict(int)
        for f in d['fills']:
            m=f['metadata'];key=(f['order_id'],m['execution_id'])
            if key in executions:raise ValueError('Duplicate physical execution')
            executions.add(key);order=orders[f['order_id']];stamp=dt(f['timestamp'])
            if f['symbol']!=symbol or f['qty']<=0:raise ValueError('Invalid physical fill')
            raw=bars(symbol,stamp.date().isoformat())[stamp];eb=m['execution_bar']
            if tuple(eb[k] for k in ('open','high','low','close','volume'))!=raw:raise ValueError('Execution source mismatch')
            if stamp<dt(m['decision_available_at']):raise ValueError('Fill before decision observation')
            typ=order['order_type'];side=f['side'];phase=m['execution_phase']
            if phase=='open':
                base=raw[0]
                if typ=='LIMIT':base=min(base,order['limit_price']) if side=='BUY' else max(base,order['limit_price'])
                elif typ=='STOP':base=max(base,order['stop_price']) if side=='BUY' else min(base,order['stop_price'])
                elif typ!='MARKET':raise ValueError('Undeclared OPEN order type')
            elif phase=='intrabar':
                base=order['limit_price'] if typ=='LIMIT' else order['stop_price'] if typ=='STOP' else None
                if base is None or not raw[2]<=base<=raw[1]:raise ValueError('Unattainable intrabar fill')
            elif phase in ('close_auction','auction'):base=raw[3]
            else:raise ValueError('Unknown execution phase '+phase)
            near(base,m['base_price'])
            price=base*(1+(1 if side=='BUY' else -1)*d['costs']['slippage_bps']/10000)
            if typ=='LIMIT':price=min(price,order['limit_price']) if side=='BUY' else max(price,order['limit_price'])
            near(price,f['price'],1e-7)
            fee=f['qty']*f['price']*(d['costs']['commission_bps']+(d['costs']['tax_bps_on_sell'] if side=='SELL' else 0))/10000
            near(fee,m['fee'],1e-7);fees+=fee;bytime[stamp].append((f,fee))
            if side=='BUY':buys[f['order_id']]+=f['qty']
        ts=[dt(t) for t in d['timestamps']];eq=d['equity_curve']
        if len(eq)==len(ts)+1:
            near(eq[0],spec['initial_equity']);eq=eq[1:]
        if len(ts)!=len(eq):raise ValueError('MTM length differs')
        for stamp,equity in zip(ts,eq):
            for f,fee in bytime.pop(stamp,[]):
                signed=f['qty'] if f['side']=='BUY' else -f['qty'];qty+=signed;cash-=signed*f['price']+fee
                if qty<0:raise ValueError('Negative independent inventory')
            price=bars(symbol,stamp.date().isoformat())[stamp][3]
            near(cash+qty*price,equity);totalmarks+=1
        if bytime or qty:raise ValueError('Unreconciled terminal fills/inventory')
        near(cash,d['terminal_cash']);near(cash-spec['initial_equity'],value['net_value_krw'])
        if value['full_portfolio_audit_status']!='unverified':raise ValueError('Isolated portfolio qualification overclaimed')
        if not value['label_valid'] or value['deadline_violation'] or not value['independent_execution_pass']:raise ValueError('Failed pilot episode')
        commitments=d['staged_commitments']
        if d['arm']=='staged':
            if len(commitments)!=1:raise ValueError('Missing unique declared commitment')
            c=next(iter(commitments.values()))
            if orders[c['initial_id']]['qty']!=spec['parent_qty']//2 or sum(buys.values())>spec['parent_qty']:
                raise ValueError('Initial or cumulative commitment mismatch')
            if c['completion_id']:
                at=dt(c['completion_at']);prefix=[(t,b) for t,b in bars(symbol,spec['day']).items() if t+timedelta(minutes=5)<=at]
                if len(prefix)!=7 or any(b[4]<=0 for t,b in prefix):raise ValueError('Invalid causal confirmation prefix')
                vwap=sum((b[1]+b[2]+b[3])/3*b[4] for t,b in prefix)/sum(b[4] for t,b in prefix)
                close=prefix[-1][1][3]
                if close<spec['reference_price'] or close<vwap:raise ValueError('Invalid confirmation admission')
                initial=[f for f in d['fills'] if f['order_id']==c['initial_id'] and dt(f['timestamp'])<at]
                if sum(f['qty'] for f in initial)!=spec['parent_qty']//2:raise ValueError('Completion before initial fully filled')
                remaining=min(spec['parent_qty']-sum(f['qty'] for f in initial),floor(max(0.,spec['parent_qty']*spec['reference_price']-sum(f['qty']*f['price'] for f in initial))/close))
                if remaining!=orders[c['completion_id']]['qty']:raise ValueError('Completion dose mismatch')
                if any(dt(f['timestamp'])<at for f in d['fills'] if f['order_id']==c['completion_id']):raise ValueError('Early child fill')
        row={'id':spec['id'],'arm':Path(output['label']['path']).parent.name,'net_krw':cash-spec['initial_equity'],
             'fee_krw':fees,'marks':len(ts),'physical_fills':len(executions),'cumulative_bought':sum(buys.values())}
        rows.append(row);details[spec['id'],row['arm']]=d
    bridges=0
    for (key,arm),d in details.items():
        if arm!='original_bridge':continue
        other=details[key,'control']
        for k in ('fills','orders','journal','equity_curve','timestamps','terminal_cash'):
            if d[k]!=other[k]:raise ValueError('Original/disabled parity mismatch')
        bridges+=1
    if len(rows)!=18 or bridges!=2:raise ValueError('Incomplete pilot coverage')
    result={'passed':True,'episodes':len(rows),'bridges':bridges,'raw_mtm_marks':totalmarks,'rows':rows,
        'verified':'Source OHLCV, attainable price/slippage, fees, integer quantities, native clock, cash, inventory and every raw-close MTM; frozen origins/hashes; causal confirmation and dose; exact disabled bridge.',
        'scope':'Isolated mechanics pilot; not economic efficacy, physical-risk certification or full portfolio qualification.','study_complete':False}
    write(out/'review.json',result);write(out/'completion.json',{'complete':True,'passed':True,'review':bound(out/'review.json'),'study_complete':False})
    print(json.dumps({k:result[k] for k in ('passed','episodes','bridges','raw_mtm_marks')}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();review(a.root)
