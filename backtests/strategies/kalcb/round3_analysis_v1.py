"""C16 account-level contrasts and observational finer marks, no execution model."""
from collections import Counter,defaultdict
from datetime import datetime,timedelta
from pathlib import Path
import hashlib
import json
import numpy as np
from .round3_native_io_v1 import read,bound,write,ROOT
from .round3_integrity_v1 import rows
from .round3_contract_v1 import verify_acceptance,paired_statistics,scalar


def account(receipt):
    verify_acceptance(receipt)
    if not receipt.get('eligible_for_scoring') or receipt.get('smoke_only'):raise ValueError('Unqualified account')
    folder=Path(receipt['folder']);r=read(folder/'result.json');trades=read(folder/'trades.json');daily=read(folder/'daily.json')
    initial=receipt['item']['initial'];campaigns=defaultdict(list)
    for t in trades:campaigns[t['entry_fill_time'][:10],t['symbol']].append(t)
    useful=[]
    for key,ts in campaigns.items():
        ts=sorted(ts,key=lambda t:t['entry_decision_time']);eq=ts[0]['route_metadata']['portfolio_equity']
        if any(t['qty']*t['entry_price']>=.01*t['route_metadata']['portfolio_equity'] for t in ts) and sum(t['net_pnl'] for t in ts)>=.0001*eq:
            useful.append(key)
    increments=(100*np.diff([initial]+[d['equity'] for d in daily])/initial).tolist()
    path=read(folder/'fills.json')
    physical=[{k:f[k] for k in ('order_id','strategy_id','symbol','side','qty','price','timestamp','reason')}|{'fee':f['metadata']['fee']} for f in path]
    identity=hashlib.sha256(json.dumps(dict(fills=physical,equity=read(folder/'equity.json')),sort_keys=True).encode()).hexdigest()
    return dict(return_pp=r['return_pct'],mtm_dd_pp=r['drawdown_pct'],native_valid=True,
        daily_increment_pp=increments,calendar=[d['day'] for d in daily],lifetimes=len(trades),
        useful_issuer_days=useful,issuer_days=list(campaigns),economic_path_hash=identity,
        net_pnl_krw=sum(t['net_pnl'] for t in trades),source=str(folder))


def compare(base,candidate):
    delta={}
    for p in ('is','oos'):
        delta[p]=[]
        for s in ('baseline','participation','cost','combined'):
            key=p+'/'+s;b=base[key];c=candidate[key]
            if c['calendar']!=b['calendar']:raise ValueError('Nonpaired calendars')
            delta[p].append((np.asarray(c['daily_increment_pp'])-b['daily_increment_pp']).tolist())
    return paired_statistics(delta)


def funded_decomposition(control,candidate):
    """Exact issuer-day arithmetic; shared differences are interacting effects."""
    def grouped(folder):
        result=defaultdict(float)
        for t in read(Path(folder)/'trades.json'):result[t['entry_fill_time'][:10],t['symbol']]+=t['net_pnl']
        return result
    b,c=grouped(control),grouped(candidate);shared=set(b)&set(c);added=set(c)-set(b);removed=set(b)-set(c)
    a=sum(c[k] for k in added);d=-sum(b[k] for k in removed);s=sum(c[k]-b[k] for k in shared)
    delta=sum(c.values())-sum(b.values())
    if abs(delta-a-d-s)>.02:raise ValueError('Funded decomposition residual')
    return dict(delta_krw=delta,added_issuer_days=sorted(added),displaced_issuer_days=sorted(removed),
        added_net_krw=a,displaced_contribution_delta_krw=d,shared_lifecycle_delta_krw=s,residual_krw=delta-a-d-s,
        sacrificed_profitable_issuer_days=[dict(day=k[0],symbol=k[1],baseline_net=b[k]) for k in sorted(removed) if b[k]>0],
        scope='Actual funded account contrast. Shared changes combine quantities, management, timing, costs and downstream capital; not additive causal attribution.')


def funding_chronology(receipt):
    """Independently check every filled buy against the native 2x funding rule."""
    verify_acceptance(receipt);cash=float(receipt['item']['initial']);q=Counter();prices={};buy_checks=0;min_headroom=None
    for item in rows(Path(receipt['folder'])/'broker_journal.jsonl.gz'):
        r=item['record'];event=r['event']
        if event in ('opening_marks','closing_marks'):prices.update(r['prices'])
        elif event=='fill':
            f=r['fill'];buy=f['side']=='BUY';fee=f['metadata']['fee'];notional=f['qty']*f['price']
            if buy:
                gross=sum(n*prices[s] for s,n in q.items() if n);equity=cash+gross
                available=max(2*equity-gross,0);headroom=available-notional-fee
                if headroom < -.02:raise ValueError('Filled buy exceeds native funding capacity')
                min_headroom=headroom if min_headroom is None else min(min_headroom,headroom);buy_checks+=1
            cash+=(-notional if buy else notional)-fee;q[f['symbol']]+=(1 if buy else -1)*f['qty']
            if q[f['symbol']]<0 or abs(cash-r['cash_after'])>.02:raise ValueError('Funding/position chronology mismatch')
    return dict(passed=True,buy_checks=buy_checks,minimum_fill_headroom_krw=min_headroom,
        terminal_cash=cash,initial_capital=receipt['item']['initial'],leverage=2,
        scope='Broker execution affordability; submission reservation equality separately checked against complete pending-order journals.')


def native_reach(folder,baseline,*,smoke=False):
    folder,baseline=Path(folder),Path(baseline)
    decisions=list(rows(folder/'decisions.jsonl.gz'));control=list(rows(baseline/'decisions.jsonl.gz'))
    fields=('symbol','qty','side','price','timestamp','reason')
    filename='smoke_fills.json' if smoke else 'fills.json'
    fills=read(folder/filename);oldfills=read(baseline/filename)
    def actions(ds):
        return Counter(json.dumps(dict(timestamp=d['timestamp'],symbol=d['symbol'],actions=[
            {k:a.get(k) for k in ('action_type','symbol','qty','order_type','reason')} for a in d['actions']]),sort_keys=True)
            for d in ds if d['actions'])
    def physical(fs):return Counter(json.dumps({k:f.get(k) for k in fields},sort_keys=True) for f in fs)
    a,b=actions(decisions),actions(control);f,g=physical(fills),physical(oldfills)
    typed=lambda ds:Counter((d['timestamp'],d['symbol'],d['decision_code'],d['reason']) for d in ds)
    changed=(typed(decisions)-typed(control))+(typed(control)-typed(decisions))
    actiondiff=(a-b)+(b-a);filldiff=(f-g)+(g-f)
    newentry={ (f['timestamp'][:10],f['symbol']) for f in fills if f['side']=='BUY'}
    oldentry={ (f['timestamp'][:10],f['symbol']) for f in oldfills if f['side']=='BUY'}
    exits=[d for d in decisions if d['reason']=='opening_thesis_failure' and d['actions']]
    return dict(candidate_events=sum(d['decision_code'] in ('entry','entry_rejected') for d in decisions),
        candidate_issuer_days=len({(d['timestamp'][:10],d['symbol']) for d in decisions if d['decision_code'] in ('entry','entry_rejected')}),
        changed_decision_events=sum(changed.values()),changed_action_records=sum(actiondiff.values()),
        changed_physical_fills=sum(filldiff.values()),added_funded_issuer_days=sorted(newentry-oldentry),
        removed_funded_issuer_days=sorted(oldentry-newentry),opening_failure_exit_decisions=len(exits),
        opening_failure_episodes=sorted({(d['timestamp'][:10],d['symbol']) for d in exits}),
        reached_native_action=bool(actiondiff),changed_economic_fill=bool(filldiff),
        caveat='Changed-fill counts include downstream quantity and timing effects; issuer-day additions are not automatically useful.')


def finer_marks(receipt,output):
    """Same retained minute convention as C16; source gaps remain explicit."""
    verify_acceptance(receipt)
    import pandas as pd
    import pyarrow
    pyarrow.set_cpu_count(1);pyarrow.set_io_thread_count(1)
    folder=Path(receipt['folder']);fills=read(folder/'fills.json');eq=read(folder/'equity.json');daily=read(folder/'daily.json')
    if any(f['metadata']['execution_phase']!='open' for f in fills):raise ValueError('Unsupported finer-fill chronology')
    regpath=ROOT.parent.parent/'kalcb_round2_assessment_20261017/v1/capital_minute_registration.json'
    reg=read(regpath);wanted=defaultdict(set)
    for f in fills:wanted[f['symbol']].add(f['timestamp'][:10])
    quotes=defaultdict(lambda:defaultdict(dict));sources={}
    for sym,dates in wanted.items():
        binding=reg['sources'].get(sym)
        if binding is None:raise ValueError('Unregistered finer instrument '+sym)
        if bound(binding['path'])['sha256']!=binding['sha256']:raise ValueError('Finer source changed')
        sources[sym]=binding;frame=pd.read_parquet(binding['path']);frame=frame[frame.timestamp.dt.strftime('%Y-%m-%d').isin(dates)]
        if frame.timestamp.duplicated().any():raise ValueError('Duplicate finer marks')
        for r in frame.itertuples():quotes[r.timestamp.strftime('%Y-%m-%d')][r.timestamp.to_pydatetime()][sym]=(float(r.open),float(r.close),int(r.Index))
    events=defaultdict(list)
    for f in fills:events[datetime.fromisoformat(f['timestamp'])].append(f)
    native={datetime.fromisoformat(t)+timedelta(minutes=5):v for t,v in eq}
    cash=float(receipt['item']['initial']);q=Counter();marks={};observed={};indices={};fields={};peak=cash;worst=0.;peak_point=None;critical=None;errors=[];stale=0.;seen=0
    def capture(at,phase,nav):
        return dict(at=str(at),phase=phase,nav=nav,cash=cash,holdings=[dict(symbol=s,qty=n,price=marks[s],
            source=sources[s],parquet_row=indices[s],field=fields[s],available_at=str(observed[s]),
            age_minutes=(at-observed[s]).total_seconds()/60) for s,n in q.items() if n])
    def mark(at,phase):
        nonlocal peak,worst,critical,peak_point,stale
        nav=cash+sum(n*marks[s] for s,n in q.items() if n)
        stale=max([stale]+[(at-observed[s]).total_seconds()/60 for s,n in q.items() if n])
        if nav>=peak:peak=nav;peak_point=capture(at,phase,nav)
        dd=100*(1-nav/peak)
        if dd>worst:worst=dd;critical=dict(peak=peak_point,trough=capture(at,phase,nav))
        return nav
    for day in daily:
        d=day['day'];at=datetime.fromisoformat(d+'T09:00:00+09:00');end=datetime.fromisoformat(d+'T16:40:00+09:00')
        while at<=end:
            here=quotes[d].get(at,{})
            for s,(o,c,i) in here.items():marks[s]=o;observed[s]=at;indices[s]=i;fields[s]='open'
            for f in events.get(at,[]):
                sign=1 if f['side']=='BUY' else -1;cash-=sign*f['qty']*f['price']+f['metadata']['fee'];q[f['symbol']]+=sign*f['qty'];seen+=1
                if q[f['symbol']]<0 or (sign>0 and f['symbol'] not in marks):raise ValueError('Finer inventory/entry mark')
            mark(at,'open_after_fill');nextat=at+timedelta(minutes=1)
            for s,(o,c,i) in here.items():marks[s]=c;observed[s]=nextat;indices[s]=i;fields[s]='close'
            nav=mark(nextat,'close_before_next_open')
            if nextat in native:errors.append(abs(nav-native[nextat]))
            at=nextat
        if any(q.values()) or abs(cash-day['equity'])>.02:raise ValueError('Finer daily account mismatch')
    if seen!=len(fills) or len(errors)!=len(eq) or max(errors,default=0)>.02:raise ValueError('Finer/native full curve mismatch')
    result=dict(passed=True,retained_minute_dd_pct=worst,source_registration=bound(regpath),critical_points=critical,
        source_gap_limitation=True,strict_continuous_time_coverage=False,maximum_held_staleness_minutes=stale,
        critical_max_staleness_minutes=max([h['age_minutes'] for p in (critical or {}).values() for h in p['holdings']],default=0),
        checked_native_marks=len(errors),max_native_nav_error_krw=max(errors,default=0))
    write(output,result);return result
