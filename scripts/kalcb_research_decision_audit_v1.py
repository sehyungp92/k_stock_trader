"""C18 read-only evidence authentication, full event census and delay attribution.

No strategy/core imports; no financial replay, fit, trading policy or stopped fork.
"""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv, gzip, hashlib, json, math, sys, time
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUN = Path('E:/CodexResearch/k_stock_trader/kalcb_round3_20261017/attempt_2/round_run_v2')
OUT = Path('E:/CodexResearch/k_stock_trader/kalcb_research_decision_20261018/v1')
SCENARIOS = ['baseline', 'participation', 'cost', 'combined']

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def bound(p): return dict(path=str(p), sha256=sha(p))
def save(p, x):
    with Path(p).open('x', encoding='utf-8') as f: json.dump(x, f, indent=2, allow_nan=False, default=str)
def rows(p):
    op = gzip.open if str(p).endswith('.gz') else open
    with op(p, 'rt', encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            if line.strip(): yield n, json.loads(line)
def table(p, data):
    if not data: save(str(p)+'.empty.json', dict(rows=0)); return
    with Path(p).open('x', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
def bindings(x):
    if isinstance(x, dict):
        if isinstance(x.get('path'), str) and isinstance(x.get('sha256'), str): yield x
        for v in x.values(): yield from bindings(v)
    elif isinstance(x, list):
        for v in x: yield from bindings(v)
def key(r): return r['item']['period']+'/'+r['process']['scenario']

def account(folder):
    """Independent streamed accounting, not the existing strategy audit helper."""
    folder=Path(folder); cash=1e8; inv=Counter(); marks={}; orders={}; peak=1e8; dd=0.; fee=0.; nfill=0
    equity=[]; lastrow=0; errors=[]; opening_capacity_failures=[]
    for n, wrapper in rows(folder/'broker_journal.jsonl.gz'):
        assert wrapper['row']==n
        r=wrapper['record']; kind=r['event']; o=r.get('order', {}); oid=o.get('order_id')
        if kind in ('submit','amend'): orders[oid]=o
        elif kind in ('cancel','expire','reject'): orders.pop(oid,None)
        elif kind in ('opening_marks','closing_marks'):
            marks.update(r['prices'])
            if kind=='closing_marks':
                nav=cash+sum(q*marks[s] for s,q in inv.items()); peak=max(peak,nav);dd=max(dd,100*(peak-nav)/peak)
                equity.append([r['timestamp'],nav])
        elif kind=='fill':
            f=r['fill']; q=f['qty']; px=f['price']; sign=1 if f['side']=='BUY' else -1; c=r['costs']
            assert type(q) is int and q>0 and px>0
            cost=q*px*(c['commission_bps']+(c['tax_bps_on_sell'] if sign<0 else 0))/1e4
            assert abs(cash-r['cash_before'])<.02 and abs(cost-f['metadata']['fee'])<.02
            if sign>0:
                nav=cash+sum(v*marks[s] for s,v in inv.items())
                available=cash+nav # native leverage2: cash + (leverage-1)*equity
                if q*px+cost>available+.02: opening_capacity_failures.append(n)
            cash-=sign*q*px+cost; inv[f['symbol']]+=sign*q; fee+=cost; nfill+=1
            assert inv[f['symbol']]>=0 and abs(cash-r['cash_after'])<.02
            assert datetime.fromisoformat(f['timestamp'])>=datetime.fromisoformat(o['metadata']['decision_available_at'])
            orders.pop(oid,None)
        lastrow=n
    native=read(folder/'equity.json')
    assert len(native)==len(equity)
    err=max(abs(a[1]-b[1]) for a,b in zip(native,equity))
    assert err<.02 and all(datetime.fromisoformat(a[0])==datetime.fromisoformat(b[0]) for a,b in zip(native,equity))
    assert not any(inv.values()) and not orders and not opening_capacity_failures
    result=read(folder/'result.json'); trades=read(folder/'trades.json'); fills=read(folder/'fills.json')
    assert abs((cash/1e8-1)*100-result['return_pct'])<1e-8 and abs(dd-result['drawdown_pct'])<1e-8
    assert abs(sum(t['net_pnl'] for t in trades)-(cash-1e8))<.02
    return dict(folder=str(folder),passed=True,journal_rows=lastrow,fills=nfill,terminal_cash=cash,
        fees=fee,return_pp=(cash/1e8-1)*100,dd_pp=dd,marks=len(equity),max_mtm_error=err,
        accounting_scope='cash/inventory/fees/MTM/native2x buying power; no separate collateral or financing model'),trades

def uncertainty(vectors):
    out=[]
    for length in [5,10,20]:
        samples=np.zeros(4000)
        for period,v in vectors.items():
            v=np.asarray(v); n=len(v); rng=np.random.default_rng(43018+length+n)
            starts=rng.integers(0,n,(4000,math.ceil(n/length)))
            ix=((starts[:,:,None]+np.arange(length))%n).reshape(4000,-1)[:,:n]
            samples+=.5*v[ix].sum(axis=1)
        lo,hi=np.quantile(samples,[.025,.975]);se=float(np.std(samples,ddof=1))
        out.append(dict(block_sessions=length,delta_pp=.5*sum(sum(v) for v in vectors.values()),
            lower95=float(lo),upper95=float(hi),SE=se,planning_MDE_pp=(1.96+.8416)*se,
            ESS='unidentified',scope='conditional reused-history diagnostic; no full search correction'))
    return out

def main():
    start=time.monotonic(); OUT.mkdir(parents=True,exist_ok=True); dest=OUT/'saved_evidence_v1';dest.mkdir(exist_ok=False)
    design=REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0043.md'
    save(dest/'registration.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),design=bound(design),program=bound(__file__),
        scope='Authenticate and census all accepted comparator/delay evidence; no new native results or fits',
        stopped_operations_run=False,prior_source=bound(RUN/'registration.json')))
    assert sha(RUN/'registration.json')=='6161249e390ab0ebfc9e1316d42a560869f82430a9b0b56e15a80b21ea642bac'
    base=read(RUN/'evaluations/v6_h0.seal.json'); delay=read(RUN/'baseline_delay.json'); supplement=read(RUN/'final_review_v1/supplement.json')
    control={key(r):r for r in base['receipts']}; lag={key(r):r for r in delay['receipts'].values()}
    assert len(control)==len(lag)==8 and base['complete'] and delay['native_valid']
    bs={}
    for payload in [base,read(RUN/'baseline_delay.seal.json'),supplement['full_ordered_reproduction']]:
        for b in bindings(payload):
            if b['path'] in bs: assert bs[b['path']]==b['sha256']
            bs[b['path']]=b['sha256']
    for r in list(control.values())+list(lag.values()):
        assert r['passed'] and r['eligible_for_scoring']
        mp=Path(r['item']['manifest']['path']); m=read(mp)
        for rel,digest in m['files'].items():bs[str(mp.parent/'files'/rel)]=digest
        plan=read(r['item']['plan']['path'])
        bs.update(plan['input_sha256'])
        for b in bindings(read(Path(r['folder'])/'completion.json')):bs[b['path']]=b['sha256']
    totalbytes=0
    for path,digest in bs.items():
        assert sha(path)==digest, 'Evidence changed: '+path
        totalbytes+=Path(path).stat().st_size
    save(dest/'authentication.json',dict(passed=True,files=len(bs),bytes_hashed=totalbytes,seconds=time.monotonic()-start,bindings=bs))
    print('Authenticated',len(bs),'files',flush=True)
    reproduction=[]
    for r in supplement['full_ordered_reproduction']:
        a,b=r['baseline']['path'],r['reproduced']['path']
        x=[v for _,v in rows(a)] if a.endswith('.gz') else read(a)
        y=[v for _,v in rows(b)] if b.endswith('.gz') else read(b)
        assert x==y
        reproduction.append(dict(cell=r['cell'],artifact=r['artifact'],records=len(x),ordered_equal=True,baseline=bound(a),reproduced=bound(b)))
    save(dest/'reproduction.json',dict(passed=True,checks=reproduction))
    account_rows=[]; cache={}
    for name,collection in [('control',control),('all_delay',lag)]:
        for cell,r in sorted(collection.items()):
            ar,tr=account(r['folder']);cache[name,cell]=tr;account_rows.append(dict(policy=name,cell=cell,**ar))
            print('Accounting',name,cell,flush=True)
    table(dest/'accounting.csv',account_rows)
    census=[];cohort=[];eventcounts=[];policyrows=[]
    for cell,r in sorted(control.items()):
        folder=Path(r['folder']);episodes={};counts=Counter();fillby=defaultdict(list);orderby=defaultdict(list)
        for n,f in enumerate(read(folder/'fills.json'),1):fillby[f['order_id']].append((n,f))
        for n,o in rows(folder/'orders.jsonl.gz'):
            order=o.get('order',{});orderby[order.get('order_id')].append((n,o));counts['order:'+o['event']]+=1
        for n,d in rows(folder/'decisions.jsonl.gz'):
            counts['decision:'+d['decision_code']+':'+d['reason']]+=1
            if d['decision_code'] not in ['entry_rejected','entry_signal','entry_order_submitted'] and not any(a.get('action_type')=='SubmitEntry' for a in d['actions']):continue
            meta=d.get('metadata',{});route=meta.get('entry_route','unavailable_short_circuit')
            acts=[a for a in d['actions'] if a.get('action_type')=='SubmitEntry']
            if acts:route=acts[0].get('metadata',{}).get('entry_route',route)
            ek=(d['timestamp'][:10],d['symbol'],d['timestamp'],route)
            row=dict(cell=cell,day=ek[0],symbol=ek[1],decision_timestamp=ek[2],route=route,reason=d['reason'],
                decision_code=d['decision_code'],actions=len(acts),requested_quantity=sum(a.get('qty',0) for a in acts),
                source=str(folder/'decisions.jsonl.gz'),source_row=n,episode_id='|'.join(ek),features=json.dumps(meta,sort_keys=True),
                action_records=json.dumps(acts,sort_keys=True))
            cohort.append(row);episodes[ek]=row
        for typ,count in counts.items():eventcounts.append(dict(cell=cell,event=typ,count=count))
        pp=Path(r['item']['policy']['path']).parent
        if cell.endswith('/baseline'):
            for filename in ['interaction_cost_margin.admission.jsonl','timed.admission.jsonl','later60.admission.jsonl']:
                path=pp/filename
                if not path.exists():continue
                for n,a in rows(path):
                    policyrows.append(dict(period=r['item']['period'],file=filename,row=n,source=str(path),record=json.dumps(a,sort_keys=True)))
        census.append(dict(cell=cell,decision_rows=sum(v for k,v in counts.items() if k.startswith('decision:')),
            entry_episode_rows=sum(x['cell']==cell for x in cohort),distinct_entry_episode_keys=len(episodes),
            distinct_entry_issuer_days=len({k[:2] for k in episodes}),order_ids=len(orderby),filled_order_ids=len(fillby),
            submitted_unfilled_order_ids=sum(oid not in fillby and any(v['event']=='submit' for _,v in rec) for oid,rec in orderby.items()),
            journal=str(folder/'decisions.jsonl.gz')))
    table(dest/'entry_events.csv',cohort);table(dest/'event_counts.csv',eventcounts);table(dest/'cohort_census.csv',census);table(dest/'policy_admissions.csv',policyrows)
    contrast=[];paired=[];dailyvectors={}; dates=[]
    for cell in sorted(control):
        groups=[]
        for name in ['control','all_delay']:
            g=defaultdict(list)
            for n,t in enumerate(cache[name,cell],1):g[(t['symbol'],t['entry_decision_time'],t['route_metadata'].get('entry_route'))].append((n,t))
            groups.append(g)
        bg,cg=groups;delta=sum(t['net_pnl'] for t in cache['all_delay',cell])-sum(t['net_pnl'] for t in cache['control',cell])
        matched=0.;quantity=entry=exitp=interaction=cost=0.
        for k in sorted(bg.keys()&cg.keys()):
            if len(bg[k])!=1 or len(cg[k])!=1:continue
            bn,b=bg[k][0];cn,c=cg[k][0];qb=b['qty'];qc=c['qty'];dq=qc-qb
            qeff=dq*(b['exit_price']-b['entry_price']);eeff=-qb*(c['entry_price']-b['entry_price']);xeff=qb*(c['exit_price']-b['exit_price'])
            ieff=dq*((c['exit_price']-b['exit_price'])-(c['entry_price']-b['entry_price']));ceff=b['commission']-c['commission']
            dn=c['net_pnl']-b['net_pnl'];assert abs(dn-qeff-eeff-xeff-ieff-ceff)<.02
            matched+=dn;quantity+=qeff;entry+=eeff;exitp+=xeff;interaction+=ieff;cost+=ceff
            paired.append(dict(cell=cell,symbol=k[0],decision=k[1],route=k[2],baseline_row=bn,delayed_row=cn,baseline_net=b['net_pnl'],delayed_net=c['net_pnl'],
                delta=dn,quantity_effect=qeff,entry_price_effect=eeff,exit_price_effect=xeff,interaction=ieff,cost_effect=ceff,
                baseline_qty=qb,delayed_qty=qc,baseline_entry=b['entry_fill_time'],delayed_entry=c['entry_fill_time'],
                baseline_exit=b['exit_fill_time'],delayed_exit=c['exit_fill_time'],baseline_reason=b['exit_reason'],delayed_reason=c['exit_reason'],
                baseline_folder=control[cell]['folder'],delay_folder=lag[cell]['folder']))
        added=sum(t['net_pnl'] for k,ts in cg.items() if k not in bg for _,t in ts)
        removed=-sum(t['net_pnl'] for k,ts in bg.items() if k not in cg for _,t in ts)
        residual=delta-matched-added-removed
        contrast.append(dict(cell=cell,delta_krw=delta,matched_krw=matched,added_krw=added,displaced_krw=removed,ambiguous_residual=residual,
            quantity=quantity,entry_price=entry,exit_price=exitp,interaction=interaction,cost=cost,matched_lifetimes=sum(x['cell']==cell for x in paired)))
        b=read(Path(control[cell]['folder'])/'daily.json');c=read(Path(lag[cell]['folder'])/'daily.json');assert [x['day'] for x in b]==[x['day'] for x in c]
        v=(np.diff([1e8]+[x['equity'] for x in c])-np.diff([1e8]+[x['equity'] for x in b]))/1e6;dailyvectors[cell]=v
        for i,(x,y) in enumerate(zip(b,c)):dates.append(dict(cell=cell,day=x['day'],daily_delta_pp=float(v[i]),baseline_equity=x['equity'],delayed_equity=y['equity']))
    table(dest/'all_delay_lifetimes.csv',paired);table(dest/'all_delay_attribution.csv',contrast);table(dest/'all_delay_dates.csv',dates)
    v={p:np.mean([dailyvectors[p+'/'+s] for s in SCENARIOS],axis=0) for p in ['is','oos']}
    table(dest/'all_delay_uncertainty.csv',uncertainty(v))
    save(dest/'completion.json',dict(passed=True,at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start,
        native_replays=0,stopped_operations=0,authenticated_files=len(bs),account_count=len(account_rows),reproduction_checks=len(reproduction),
        scope='Full saved event census and descriptive funded delay attribution; matched price effects are not independently executable interventions; entry-only/exit-only diagnosis remains required'))
    print('SAVED EVIDENCE COMPLETE',flush=True)

if __name__=='__main__':main()
