"""Prepared read-only review. Must register only after exact producer exit."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone,timedelta
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3

REASON='research_completed45_liquidation_contrast_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def stamp(t):return datetime.fromisoformat(str(t)).isoformat()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),default=str,allow_nan=False)
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bound(p):return {'path':str(Path(p).resolve()),'sha256':digest(p)}
def verify(items):
    for b in items:
        if digest(b['path'])!=b['sha256']:raise ValueError('Changed evidence '+b['path'])
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def close(a,b):
    if abs(a-b)>.011:raise ValueError('Independent numeric mismatch')


def account(detail,value,raw):
    """Rebuild from raw closes and physical executions, not producer MTM totals."""
    fills=detail['fills'];cost=detail['costs'];cash=100000000.;qty=defaultdict(int);fees=0.;pending=0
    curve=detail['equity_curve'];times=detail['timestamps'];marks=0
    if len(curve)!=len(times)+1 or curve[0]!=cash:raise ValueError('Invalid initial equity schema')
    executions=set()
    for ts,equity in zip(times,curve[1:]):
        ts=stamp(ts)
        while pending<len(fills) and stamp(fills[pending]['timestamp'])<=ts:
            f=fills[pending];t=stamp(f['timestamp']);symbol=f['symbol'];n=f['qty'];price=f['price']
            if n<=0 or price<=0 or f['side'] not in ('BUY','SELL'):raise ValueError('Invalid fill')
            eid=f['metadata']['execution_id']
            if eid in executions:raise ValueError('Duplicate physical execution')
            executions.add(eid)
            source=raw(t[:10])[symbol,t,'5m'];b=f['metadata']['execution_bar']
            if tuple(b[k] for k in ('open','high','low','close','volume','is_completed'))!=source:
                raise ValueError('Fill raw-source mismatch')
            if n>source[4]:raise ValueError('Physically impossible fill volume')
            fee=n*price*(cost['commission_bps']+(cost['tax_bps_on_sell'] if f['side']=='SELL' else 0))/10000
            fees+=fee;cash+=(-1 if f['side']=='BUY' else 1)*n*price-fee
            qty[symbol]+=n if f['side']=='BUY' else -n
            if qty[symbol]<0:raise ValueError('Sold unowned inventory')
            pending+=1
        marked=cash+sum(n*raw(ts[:10])[s,ts,'5m'][3] for s,n in qty.items() if n)
        close(marked,equity);marks+=1
        checkpoint=detail.get('checkpoint',{})
        if checkpoint.get('state') and datetime.fromisoformat(ts)+timedelta(minutes=5)==datetime.fromisoformat(checkpoint['state']['clock']):
            close(cash,checkpoint['state']['cash'])
            held={s:n for s,n in qty.items() if n}
            captured={p['symbol']:p['qty'] for p in checkpoint['state']['positions']}
            if held!=captured:raise ValueError('Prestate inventory differs from physical fills')
    if pending!=len(fills):raise ValueError('Unmatched fills')
    close(cash,detail['terminal_cash']);close(fees,value['fees_and_tax_krw'])
    if sum(qty.values())!=value['terminal_inventory']:raise ValueError('Terminal inventory mismatch')
    if value['net_value_krw'] is not None:close(cash-100000000,value['net_value_krw'])
    if detail['independent_engine_audit']['audit_status']!='unverified' or value['full_portfolio_audit_status']!='unverified':
        raise ValueError('Isolated labels must not claim portfolio qualification')
    return {'marks':marks,'fills':len(fills),'fees':fees,'terminal_cash':cash}


def pair_review(value,events,raw):
    if events['pair']['opportunity_id']!=value['opportunity_id'] or events['pair']['scenario']!=value['scenario']:
        raise ValueError('Pair identity mismatch')
    h,l=events['hold'],events['liquidate'];clock=stamp(value['checkpoint_at'])
    for arm in (h,l):
        c=arm['checkpoint']
        if c['sha256'] is not None and hashlib.sha256(canonical(c['state']).encode()).hexdigest()!=c['sha256']:
            raise ValueError('Prestate checksum mismatch')
    if h['checkpoint']['sha256']!=l['checkpoint']['sha256']:raise ValueError('Different pre-action accounts')
    audits={arm:account(events[arm],value[arm],raw) for arm in ('hold','liquidate')}
    exit_orders=[r['order'] for r in l['journal'] if r['event']=='submit' and r['order']['reason']==REASON]
    if value['eligibility']=='eligible':
        if h['checkpoint']['parent_complete_exit'] or len(exit_orders)!=1:raise ValueError('Not a genuinely additional exit')
        order=exit_orders[0]
        if order['qty']!=h['checkpoint']['held_quantity']:raise ValueError('Not actual held quantity')
        if stamp(order['metadata']['decision_available_at'])!=clock:raise ValueError('Wrong exit availability clock')
        for f in l['fills']:
            if f['order_id']==order['order_id']:
                if stamp(f['timestamp'])<clock:raise ValueError('Premature exit fill')
                # The new action is MARKET; ordinary adverse slippage applies to
                # actual next executable raw open, never the checkpoint close.
                source=raw(stamp(f['timestamp'])[:10])[f['symbol'],stamp(f['timestamp']),'5m']
                close(f['price'],source[0]*(1-l['costs']['slippage_bps']/10000))
        before=h['checkpoint']['state']
        if any(stamp(f['timestamp'])>=clock for f in before['fills']):raise ValueError('Future fill in prestate')
    else:
        if exit_orders or any(canonical(h[k])!=canonical(l[k]) for k in ('fills','orders','equity_curve')):
            raise ValueError('Unsupported branch changed economics')
    if value['paired_label_valid']:
        if value['eligibility']!='eligible' or value['held_notional_krw']<=0:raise ValueError('Invalid mature target')
        delta=value['liquidate']['net_value_krw']-value['hold']['net_value_krw']
        close(delta,value['liquidate_minus_hold_krw']);close(delta/value['held_notional_krw']*10000,value['liquidate_minus_hold_bps'])
        if value['label_available_at']!=max(value[a]['label_available_at'] for a in ('hold','liquidate')):
            raise ValueError('Immature paired target')
    elif value['liquidate_minus_hold_bps'] is not None:raise ValueError('Invalid outcome admitted')
    return audits


def exact_exit(root):
    import psutil
    for name in ('launch_receipt.json','supervisor_start.json','status.json'):
        v=read(root/name)
        for item in (v,v.get('child')):
            if not item:continue
            try:
                p=psutil.Process(item['pid'])
                if p.cmdline()==item['command'] and abs(p.create_time()-item['process_created'])<.01:
                    raise ValueError('Registered native process still active')
            except psutil.NoSuchProcess:pass
    if read(root/'status.json')['stage']!='awaiting_independent_pair_review_before_any_fit':raise ValueError('Producer did not complete normally')
    done=read(root/'completion.json');labels=read(root/'labels/completion.json')
    if not done['complete'] or not labels['complete'] or labels['pairs']!=read(root/'protocol.json')['planned_pairs']:
        raise ValueError('Incomplete finite label population')


def run(root,output,tests):
    exact_exit(root);p=read(root/'protocol.json');done=read(root/'labels/completion.json');test=read(tests)
    if test['exit_code']!=0:raise ValueError('Review tests failed')
    verify(test['programs']+[test['log']]);verify(p['programs']+p['bindings']+done['artifacts'])
    origins=[]
    for stage in ('smoke','labels'):
        for suffix in ('before','after'):
            file=root/(stage+'_imports_'+suffix+'.json');o=read(file)
            verify(list(o['origins'].values()));origins.append(bound(file))
    output.mkdir(exist_ok=False)
    write(output/'protocol.json',{'registered_at_utc':datetime.now(timezone.utc).isoformat(),'programs':test['programs'],
        'tests':bound(tests),'bindings':[bound(root/'protocol.json'),bound(root/'completion.json'),bound(root/'labels/completion.json'),*origins],
        'read_only':True,'no_policy_or_fit':True,'study_complete':False})
    from functools import lru_cache
    @lru_cache(maxsize=3)
    def raw(day):
        with sqlite3.connect('file:'+Path(p['databases'][int(day>='2026-04-01')]).as_posix()+'?mode=ro',uri=True) as c:
            return {(s,stamp(t),'5m'):(o,h,l,cl,v,True) for s,t,o,h,l,cl,v in c.execute('SELECT symbol,ts,o,h,l,c,v FROM bars WHERE day=?',(day,))}
    seen=set();counts=Counter();support=defaultdict(set);maturity=[];rows=[]
    files=sorted(b['path'] for b in done['artifacts'] if '.events.' not in b['path'])
    for file in files:
        path=Path(file);ef=path.with_name(path.name.replace('.jsonl.gz','.events.jsonl.gz'))
        with gzip.open(path,'rt',encoding='utf-8') as vf,gzip.open(ef,'rt',encoding='utf-8') as ev:
            for line,event in itertools.zip_longest(vf,ev):
                if line is None or event is None:raise ValueError('Pair/event count mismatch')
                v=json.loads(line);e=json.loads(event);key=(v['opportunity_id'],v['scenario'])
                if key in seen:raise ValueError('Duplicate dependent pair')
                seen.add(key);audit=pair_review(v,e,raw)
                counts['pairs']+=1;counts[v['eligibility']]+=1;counts['raw_close_marks']+=sum(a['marks'] for a in audit.values())
                period='pre_is' if v['day']<'2025-08-07' else 'is' if v['day']<'2026-04-01' else 'oos'
                if v['paired_label_valid']:
                    support[period].add((v['day'],v['symbol']));maturity.append({'day':v['day'],'symbol':v['symbol'],'label_available_at':v['label_available_at']})
                rows.append({k:v[k] for k in ('opportunity_id','day','symbol','scenario','quantity_scale','eligibility','paired_label_valid','held_notional_krw','liquidate_minus_hold_bps','label_available_at')})
        print(json.dumps({'reviewed':path.name,'pairs':counts['pairs']}),flush=True)
    if len(seen)!=p['planned_pairs']:raise ValueError('Finite count mismatch')
    with gzip.open(output/'reviewed_targets.jsonl.gz','xt',encoding='utf-8') as f:
        for row in rows:f.write(canonical(row)+'\n')
    verify(p['programs']+p['bindings']+done['artifacts'])
    write(output/'review.json',{'passed':True,'counts':dict(counts),'support':{k:{'unique_issuer_days':len(v),'unique_sessions':len({d for d,s in v})} for k,v in support.items()},
        'targets':bound(output/'reviewed_targets.jsonl.gz'),'full_portfolio_audit_status':'unverified',
        'exchangeability_proven':False,'mature_chronological_fit_support_still_requires_review':True,'study_complete':False})
    write(output/'completion.json',{'complete':True,'study_complete':False,'review':bound(output/'review.json')})


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True);a.add_argument('--tests',type=Path,required=True)
    v=a.parse_args();run(v.root.resolve(),v.output.resolve(),v.tests.resolve())
