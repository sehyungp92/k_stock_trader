"""Read-only, independently reconstructed review of actual-account continuations."""
import argparse
from collections import defaultdict
from datetime import datetime,timedelta,timezone
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def gz(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return json.load(f)
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bound(p):return {'path':str(Path(p).resolve()),'sha256':digest(p)}
def verify(bs):
    for b in bs:
        if digest(b['path'])!=b['sha256']:raise ValueError('Changed evidence '+b['path'])
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False,default=str)
def dt(v):return datetime.fromisoformat(v)
def near(a,b,tol=.011):
    if abs(a-b)>tol:raise ValueError(f'Independent mismatch {a} != {b}')


def validate(d,bars):
    orders={r['order']['order_id']:r['order'] for r in d['journal'] if r['event']=='submit'}
    executions=set();bytime=defaultdict(list);fees=0.
    for f in d['fills']:
        m=f['metadata'];key=(f['order_id'],m['execution_id']);q=f['qty']
        if key in executions or not isinstance(q,int) or isinstance(q,bool) or q<=0:raise ValueError('Invalid/duplicate physical fill')
        executions.add(key);order=orders[f['order_id']];stamp=dt(f['timestamp'])
        raw=bars(f['symbol'],str(stamp.date()))[stamp];eb=m['execution_bar']
        if tuple(eb[k] for k in ('open','high','low','close','volume'))!=raw or raw[4]<=0:raise ValueError('Execution raw source mismatch')
        if stamp<dt(m['decision_available_at']):raise ValueError('Fill precedes decision')
        typ=order['order_type'];side=f['side'];phase=m['execution_phase']
        if phase=='open':
            base=raw[0]
            if typ=='LIMIT':base=min(base,order['limit_price']) if side=='BUY' else max(base,order['limit_price'])
            elif typ=='STOP':base=max(base,order['stop_price']) if side=='BUY' else min(base,order['stop_price'])
            elif typ!='MARKET':raise ValueError('Unknown opening order type')
        elif phase=='intrabar':
            base=order['limit_price'] if typ=='LIMIT' else order['stop_price'] if typ=='STOP' else None
            if base is None or not raw[2]<=base<=raw[1]:raise ValueError('Unattainable intrabar fill')
        elif phase in ('auction','close_auction'):base=raw[3]
        else:raise ValueError('Unknown execution phase')
        near(base,m['base_price'],1e-7)
        bps=d['costs']['slippage_bps']
        if phase=='auction':bps=max(0.,bps+float(order['metadata'].get('auction_adverse_bps',d['costs'].get('auction_slippage_bps',0)) or 0))
        price=base*(1+(1 if side=='BUY' else -1)*bps/10000)
        if typ=='LIMIT':price=min(price,order['limit_price']) if side=='BUY' else max(price,order['limit_price'])
        near(price,f['price'],1e-7)
        fee=q*price*(d['costs']['commission_bps']+(d['costs']['tax_bps_on_sell'] if side=='SELL' else 0))/10000
        near(fee,m['fee'],1e-7);fees+=fee;bytime[stamp].append((f,fee))
    cash=100000000.;qty=defaultdict(int);marks=0
    for timestamp,equity in d['equity']:
        stamp=dt(timestamp)
        for f,fee in bytime.pop(stamp,[]):
            signed=f['qty'] if f['side']=='BUY' else -f['qty'];qty[f['symbol']]+=signed;cash-=signed*f['price']+fee
            if qty[f['symbol']]<0:raise ValueError('Negative reconstructed inventory')
        marked=cash+sum(n*bars(s,str(stamp.date()))[stamp][3] for s,n in qty.items() if n)
        near(marked,equity);marks+=1
    if bytime or any(qty.values()) or d['pending'] or d['positions']:raise ValueError('Unresolved whole-account terminal state')
    near(cash,d['cash'])
    if not d['engine_audit']['execution_validity_pass'] or d['engine_audit']['audit_status']!='unverified':
        raise ValueError('Failed execution or overstated whole-portfolio qualification')
    return {'cash':cash,'fees':fees,'marks':marks,'fills':len(executions)}


def review(root):
    import psutil
    p=read(root/'protocol.json');verify(p['programs']+p['bindings']+[p['runtime_manifest']])
    manifest=read(p['runtime_manifest']['path']);verify(list(manifest['overrides'].values()))
    for scenario in p['scenarios']:
        receipt=read(root/(scenario+'_start.json'))
        try:
            proc=psutil.Process(receipt['pid'])
            if abs(proc.create_time()-receipt['process_created'])<.01 and proc.cmdline()==receipt['command']:
                raise ValueError('Exact native worker still active')
        except psutil.NoSuchProcess:pass
        exited=read(root/(scenario+'_exit.json'))
        if exited['exit_code']!=0:raise ValueError('Failed native child')
        verify([read(root/(scenario+'_completion.json'))['parent']])
        for stage in ('before','after'):verify(list(read(root/f'{scenario}_imports_{stage}.json').values()))
    out=root/'review_v1';out.mkdir(exist_ok=False);me=psutil.Process()
    write(out/'supervisor_start.json',{'pid':me.pid,'process_created':me.create_time(),'command':me.cmdline()})
    write(root/'review_registration_v1.json',{'program':bound(__file__),'protocol':bound(root/'protocol.json'),'at_utc':datetime.now(timezone.utc).isoformat(),'read_only':True})
    cache={}
    def bars(symbol,day):
        if (symbol,day) not in cache:
            db=p['databases'][int(day>='2026-04-01')]
            with sqlite3.connect('file:'+Path(db).as_posix()+'?mode=ro',uri=True) as c:
                cache[symbol,day]={dt(t):(o,h,l,cl,v) for t,o,h,l,cl,v in c.execute('SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? AND day=? ORDER BY ts',(symbol,day))}
        return cache[symbol,day]
    rows=[];marks=0
    for scenario in p['scenarios']:
        done=read(root/'parents'/scenario/'completion.json');verify(done['pairs_bound'])
        census=read(root/'parents'/scenario/'census.json')
        first=[x['day'] for x in census if any(r['reason']=='cancel_requested' for r in x['observations'])]
        if len(first)!=4:raise ValueError('Changed finite causal selection')
        original=Path(p['parent_accounts'][scenario]);of=read(original/'fills.json');oe=read(original/'equity.json')
        parent=read(root/'parents'/scenario/'account.json');last=parent['equity'][-1][0][:10]
        if parent['fills']!=[f for f in of if f['timestamp'][:10]<=last] or parent['equity']!=[e for e in oe if e[0][:10]<=last]:
            raise ValueError('Independent original prefix bridge mismatch')
        marks+=validate(parent,bars)['marks']
        for ordinal,binding in enumerate(done['pairs_bound']):
            pair=read(binding['path']);folder=Path(binding['path']).parent;verify(pair['bindings'])
            if pair['day']!=first[ordinal] or not pair['exact_disabled_continuation']:raise ValueError('Pair selection/bridge mismatch')
            pre=gz(folder/'prestate.json.gz')
            encoded=json.dumps(pre['state'],sort_keys=True,separators=(',',':'),allow_nan=False)
            if hashlib.sha256(encoded.encode()).hexdigest()!=pair['prestate_sha256']:raise ValueError('Prestate hash mismatch')
            hold=gz(folder/'hold.json.gz');cancel=gz(folder/'cancel.json.gz');clock=dt(pair['clock'])
            if hold['fills']!=[f for f in of if f['timestamp'][:10]<=pair['day']] or hold['equity']!=[e for e in oe if e[0][:10]<=pair['day']]:
                raise ValueError('Disabled continuation differs from original source account')
            if [f for f in hold['fills'] if dt(f['timestamp'])<clock]!=[f for f in cancel['fills'] if dt(f['timestamp'])<clock]:
                raise ValueError('Candidate changed predecision fills')
            if [e for e in hold['equity'] if dt(e[0])<clock]!=[e for e in cancel['equity'] if dt(e[0])<clock]:
                raise ValueError('Candidate changed predecision MTM')
            native=dict(pre['state']['broker']);pending={o['order_id']:o for o in native['orders']}
            for oid in pair['cancelled_orders']:
                order=pending[oid];meta=dict(order['metadata']);symbol=order['symbol']
                if order['side']!='BUY' or order['order_type']!='MARKET' or order['qty']<=0:raise ValueError('Not a live entry remainder')
                if len([o for o in pending.values() if o['symbol']==symbol])!=1:raise ValueError('Unsafe symbol-scoped cancellation')
                prefix=[(t,b) for t,b in bars(symbol,pair['day']).items() if t+timedelta(minutes=5)<=clock]
                if len(prefix)!=7 or any(b[4]<=0 for t,b in prefix):raise ValueError('Invalid seven-bar raw prefix')
                ref=meta['entry_price_ref'];near(ref,prefix[5][1][3],1e-7)
                vwap=sum((b[1]+b[2]+b[3])*b[4] for t,b in prefix)/(3*sum(b[4] for t,b in prefix))
                if prefix[-1][1][3]>=ref and prefix[-1][1][3]>=vwap:raise ValueError('Cancelled confirmed entry')
                acks=[r for r in cancel['journal'] if r['event']=='cancel' and r['order']['order_id']==oid and r.get('timestamp')==str(clock-timedelta(minutes=5)).replace(' ','T')]
                if len(acks)!=1 or acks[0]['order']['qty']!=order['qty']:raise ValueError('Missing exact cancellation acknowledgement')
                if any(f['order_id']==oid and f['side']=='BUY' and dt(f['timestamp'])>=clock for f in cancel['fills']):raise ValueError('Cancelled order received a future fill')
            hc=validate(hold,bars);cc=validate(cancel,bars);marks+=hc['marks']+cc['marks']
            near(cc['cash']-hc['cash'],pair['account_cash_difference_krw'])
            rows.append({k:pair[k] for k in ('day','scenario','cancelled_orders','original_later_buy_qty','original_later_buy_notional','account_cash_difference_krw')})
    if len(rows)!=8:raise ValueError('Incomplete paired states')
    result={'passed':True,'pairs':8,'original_parent_prefixes':2,'raw_mtm_marks':marks,'rows':rows,
        'scope':'Eight dependent same-account session contrasts; no sequential portfolio return or robust efficacy finding. Preserves failures/losses and sacrificed winners. Entire study remains incomplete.',
        'study_complete':False}
    write(out/'review.json',result);write(out/'completion.json',{'complete':True,'passed':True,'review':bound(out/'review.json'),'study_complete':False})
    print(json.dumps({'passed':True,'pairs':8,'raw_mtm_marks':marks}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();review(a.root)
