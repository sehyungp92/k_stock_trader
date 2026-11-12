"""0037: read-only opportunity/ledger continuity census. No policy replay."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import gzip
import math
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace
import traceback
import psutil
import yaml
from olr_selective_half_continuation_v1 import read, write, sha, now, status

PARENT=Path('E:/CodexResearch/k_stock_trader/olr_continuing_research_20261016/relative_qualification_v2')
REPO=Path(__file__).resolve().parents[1]


def ledger(case, points):
    """Preserve journal order; retain both pre-cohort and end-boundary state."""
    cash=1e8;qty=defaultdict(int);marks={};position_meta={};pending={};orders={}
    snapshots={};pre_buy={};index=0;last=None;clock_clamps=0
    events=defaultdict(list)
    def snap():
        reserve=sum(o['remaining']*o['metadata']['estimated_entry_price']*(1+o['metadata']['entry_cost_buffer_pct'])
                    for o in pending.values() if o['side']=='BUY')
        risks=defaultdict(float);sector_risks=defaultdict(float)
        def add(symbol,q,m):
            if q<=0:return
            unit=m['risk_per_share']*1.25+m['estimated_entry_price']*.0034
            risks[symbol]+=q*unit;sector_risks[m.get('sector','UNKNOWN')]+=q*unit
        for s,q in qty.items():
            if q:add(s,q,position_meta[s])
        for o in pending.values():
            if o['side']=='BUY':add(o['symbol'],o['remaining'],o['metadata'])
        gross=sum(q*marks[s] for s,q in qty.items() if q)
        return dict(cash=cash,reserve=reserve,deployable=cash-reserve,equity=cash+gross,gross=gross,
                    quantities={s:q for s,q in qty.items() if q},risk=dict(risks),sector_risk=dict(sector_risks),
                    pending=[{k:o[k] for k in ('order_id','symbol','side','remaining','reason')} for o in pending.values()])
    with gzip.open(case/'minute_events.jsonl.gz','rt',encoding='utf8') as stream:
        for line in stream:
            e=__import__('json').loads(line);kind=e['event']
            if kind in ('opening_marks','interval_marks','closing_marks'):
                at=datetime.fromisoformat(e['timestamp'])+timedelta(minutes=kind!='opening_marks')
            elif kind=='fill':at=datetime.fromisoformat(e['fill']['timestamp'])
            elif kind=='submit':at=datetime.fromisoformat(e['order']['metadata'].get('decision_available_at',e['order']['submitted_at']))
            elif kind=='amend' and e.get('reason')=='partial_remainder':
                assert last is not None
                at=last
            elif kind in ('cancel','expire','reject','amend'):
                at=datetime.fromisoformat(e.get('timestamp',e['order']['submitted_at']))
            else:continue
            if last and at<last:at=last;clock_clamps+=1
            last=at
            while index<len(points) and points[index]<at:
                snapshots[points[index].isoformat()]=snap();index+=1
            if kind=='submit':
                o=e['order'];ident=o['order_id']
                if o['side']=='BUY' and at in points and at.isoformat() not in pre_buy:
                    pre_buy[at.isoformat()]=snap()
                pending[ident]={**o,'remaining':o['qty']};orders[ident]={**o,'events':[]}
            elif kind=='fill':
                f=e['fill'];s=f['symbol'];sign=1 if f['side']=='BUY' else -1
                assert math.isclose(cash,e['cash_before'],abs_tol=.001)
                cash-=sign*f['qty']*f['price']+f['metadata']['fee'];qty[s]+=sign*f['qty']
                assert qty[s]>=0 and math.isclose(cash,e['cash_after'],abs_tol=.001)
                if f['side']=='BUY':position_meta[s]=e['order']['metadata']
                ident=f['order_id'];pending[ident]['remaining']-=f['qty']
                assert pending[ident]['remaining']>=0
                if not pending[ident]['remaining']:pending.pop(ident)
            elif kind in ('cancel','expire','reject'):pending.pop(e['order']['order_id'],None)
            elif kind=='amend':
                # Native broker reports its remaining order after each fill.
                # The fill already reduced our reservation; do not reduce twice.
                o=e['order']
                assert e.get('reason')=='partial_remainder',e
                assert pending[o['order_id']]['remaining']==o['qty'],e
            else:marks.update(e['prices'])
            if kind in ('submit','fill','cancel','expire','reject','amend'):
                o=e['order'];key=(o['symbol'],str(o['submitted_at'])[:10])
                record=dict(kind=kind,at=at.isoformat(),order_id=o['order_id'],side=o['side'],reason=o['reason'],qty=o['qty'])
                if kind=='fill':record.update(fill_qty=e['fill']['qty'],price=e['fill']['price'],fee=e['fill']['metadata']['fee'])
                events[key].append(record)
                if o['order_id'] in orders:orders[o['order_id']]['events'].append(record)
    while index<len(points):snapshots[points[index].isoformat()]=snap();index+=1
    assert not any(qty.values()) and not pending
    return dict(snapshots=snapshots,pre_buy=pre_buy,events=events,orders=orders,terminal_cash=cash,clock_clamps=clock_clamps)


def window(root,window):
    ready=read(PARENT/'native_engines'/window/'ready.json');files=Path(ready['files']);study=ready['study']
    sys.path.insert(0,str(files))
    from strategy_olr.execution import OLREntryPlan,find_olr_entry_signal,_common_entry_bar_passes,_running_vwap,_close_location,DECISION_CUTOFF
    from strategy_common.execution_session import session_cutoff
    from strategy_common.market import MarketBar
    summary=read(PARENT/'assessment.json')['summaries'][window+'_nominal_control'];case=Path(summary['case'])
    completion=read(case/'completion.json')
    assert sha(case/'completion.json')==summary['completion_sha256']
    checks={n:sha(case/n)==h for n,h in completion['artifacts'].items()};assert all(checks.values())
    snapshots=read(files/study['olr_snapshots']);dates=study['configuration']['date_range']
    snapshots=[s for s in snapshots if dates['start']<=s['trade_date']<=dates['end']]
    assert len({s['trade_date'] for s in snapshots})==len(snapshots)
    request=read(files/study['olr_request']);plan=OLREntryPlan(**request['mutations']['olr.trade_plan.entry'])
    assert plan.mode=='confirm_next_bar' and plan.max_signal_bars==6
    calendar=yaml.safe_load((files/'kis_core/data/krx_sessions.yaml').read_text(encoding='utf8'))
    points=sorted(datetime.combine(datetime.fromisoformat(s['trade_date']).date(),session_cutoff(datetime.fromisoformat(s['trade_date']).date(),DECISION_CUTOFF)).replace(tzinfo=datetime.fromisoformat(s['generated_at']).tzinfo)+timedelta(minutes=5*(i+1)) for s in snapshots for i in range(6))
    status(root,'ledger',window=window)
    book=ledger(case,points)
    assert math.isclose(book['terminal_cash'],1e8+summary['return_pct']*1e6,abs_tol=.01)
    population=read(case/'funded_population_receipt.json')['entry_contexts'];proposals=defaultdict(list)
    reconciliation=[]
    for p in population:
        b=book['pre_buy'].get(p['at'],book['snapshots'][p['at']])
        assert math.isclose(b['cash'],p['ledger_cash'],abs_tol=.01),(p['at'],b['cash'],p['ledger_cash'])
        if p['at'] in book['pre_buy']:
            assert math.isclose(b['reserve'],p['pending_buy_reservations'],abs_tol=.01)
        reconciliation.append(dict(at=p['at'],cash_equal=True,reservation_equal=p['at'] in book['pre_buy']))
        for decision in p['decisions']:
            proposal=p['proposals'][decision['intent_id']]
            proposals[(proposal['symbol'],p['at'][:10])].append(dict(at=p['at'],proposal=proposal,decision=decision,account=p['account'],cash=p['actual_deployable_cash']))
    trades={(t['symbol'],t['entry_decision_time'][:10]):t for t in read(case/'trades.json')}
    con=sqlite3.connect((files/study['database']).as_uri()+'?mode=ro',uri=True);con.execute('PRAGMA query_only=ON')
    rows=[];unmatched=set(proposals);source_checks=0
    for number,snapshot in enumerate(snapshots):
        day=snapshot['trade_date'];cutoff=session_cutoff(datetime.fromisoformat(day).date(),DECISION_CUTOFF)
        open_clock=calendar.get('exceptions',{}).get(day,{}).get('open',calendar['regular']['open'])
        for order,c in enumerate(snapshot['candidates']):
            symbol=c['symbol'];key=(symbol,day);bars=[]
            for stamp,o,h,l,close,v in con.execute('SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? AND day=? ORDER BY ts',(symbol,day)):
                bars.append(MarketBar(symbol=symbol,timestamp=datetime.fromisoformat(stamp),open=o,high=h,low=l,close=close,volume=v,timeframe='5m',is_completed=True))
            timeline=[];candidate=SimpleNamespace(**c)
            for i,bar in enumerate(bars):
                if bar.timestamp.time().replace(tzinfo=None)<cutoff:continue
                if len(timeline)>=6:break
                prefix=bars[:i+1];prior=[b for b in prefix if b.timestamp.time().replace(tzinfo=None)<cutoff]
                opening=datetime.fromisoformat(day+'T'+open_clock+':00+09:00')
                expected=[opening+timedelta(minutes=5*j) for j in range(int((bar.timestamp-opening).total_seconds()/300)+1)]
                complete=[b.timestamp for b in prefix]==expected and bool(prior)
                at=(bar.timestamp+timedelta(minutes=5)).isoformat();vwap=_running_vwap(prefix)
                valid=complete and _common_entry_bar_passes(bar,vwap,_close_location(bar),prior[-1].close,plan)
                signal=find_olr_entry_signal(prefix,candidate,plan,require_fill_bar=False) if complete else None
                b=book['snapshots'][at]
                timeline.append(dict(at=at,source_label=bar.timestamp.isoformat(),source_complete=complete,close=bar.close,
                    vwap=vwap,bar_return=bar.close/bar.open-1,passes=bool(valid),
                    first_native_signal=(prefix[signal.signal_index].timestamp+timedelta(minutes=5)).isoformat() if signal else None,
                    book={k:b[k] for k in ('cash','reserve','deployable','equity','gross','risk','sector_risk','quantities','pending')}))
            ps=proposals.get(key,[]);t=trades.get(key)
            if ps:
                unmatched.discard(key)
                for p in ps:
                    actual=next(z for z in timeline if z['at']==p['at'])
                    assert actual['first_native_signal']==p['at'] and actual['passes'],key
                    assert actual['close']==p['proposal']['metadata']['entry_submission_close'];source_checks+=1
            first=next((z for z in timeline if z['passes']),None)
            if t:disposition='acquired'
            elif ps:disposition='submitted_unfilled' if any(p['decision']['approved_qty'] for p in ps) else 'funding_rejected'
            elif order>=6:disposition='outside_slots'
            elif not c.get('tradable',True):disposition='untradable'
            elif not first:disposition='unconfirmed_or_unsupported'
            elif first['book']['quantities'].get(symbol,0):disposition='already_held'
            else:disposition='unexplained_no_proposal'
            retry=[]
            if disposition=='funding_rejected':
                p=ps[0];m=p['proposal']['metadata'];base=next(z for z in timeline if z['at']==p['at'])
                for z in timeline:
                    if z['at']<=p['at'] or not z['passes'] or z['book']['quantities'].get(symbol):continue
                    b=z['book'];price=z['close']*1.015;unit=price*1.0007;risk=max(c['daily_atr'],z['close']*.005,1.)
                    unit_loss=1.25*risk+.0034*price
                    limits={'original_request':p['proposal']['qty'],
                        'cash':math.floor(max(0,b['deployable'])/unit),
                        'gross':math.floor(max(0,b['equity']-b['gross']-b['reserve'])/unit),
                        'position_loss':math.floor(max(0,.04*b['equity']-b['risk'].get(symbol,0))/unit_loss),
                        'sector_loss':math.floor(max(0,.06*b['equity']-b['sector_risk'].get(c['sector'],0))/unit_loss),
                        'total_loss':math.floor(max(0,.08*b['equity']-sum(b['risk'].values()))/unit_loss)}
                    if m['capacity_forecast']['exit']['quantity'] is not None:limits['original_exit_forecast']=m['capacity_forecast']['exit']['quantity']
                    q=min(limits.values())
                    retry.append(dict(at=z['at'],qty_upper_bound=q,notional_upper_bound=q*price,limits=limits,
                        cash_change=b['deployable']-p['cash'],gross_change=b['gross']-p['account']['gross_committed'],
                        fresh_recovery=any(not x['passes'] for x in timeline if p['at']<x['at']<z['at']),
                        scope='After incumbent cohort; no new entry-capacity/service forecast or competing retries. Upper bound, not funded trade.'))
            row=dict(window=window,day=day,symbol=symbol,rank=c['rank'],score=c['selection_score'],sector=c['sector'],
                disposition=disposition,first_confirmation=first['at'] if first else None,timeline=timeline,proposals=ps,reconsideration=retry,
                orders=[o for o in book['orders'].values() if o['symbol']==symbol and str(o['submitted_at'])[:10]==day],
                pnl=t['net_pnl'] if t else None,acquired_notional=t['qty']*t['entry_price'] if t else 0,
                exit_reason=t['exit_reason'] if t else None)
            rows.append(row)
        if number%20==0:status(root,'source_census',window=window,sessions=number+1,rows=len(rows))
    assert not unmatched,unmatched
    assert sum(r['disposition']=='acquired' for r in rows)==len(trades)
    con.close()
    summary_out=dict(rows=len(rows),dates=len(snapshots),dispositions=dict(Counter(r['disposition'] for r in rows)),
        source_proposal_checks=source_checks,ledger_cash_checks=len(reconciliation),terminal_cash=book['terminal_cash'],
        temporary_rejection_reach=[dict(day=r['day'],symbol=r['symbol'],attempt=r['proposals'][0]['at'],reason=r['proposals'][0]['decision']['reason'],
            later=r['reconsideration']) for r in rows if r['disposition']=='funding_rejected'],
        pending_buy_cancel_events=[dict(symbol=o['symbol'],submitted_at=o['submitted_at'],events=o['events']) for o in book['orders'].values()
            if o['side']=='BUY' and any(e['kind'] in ('cancel','expire','reject') for e in o['events'])],
        clock_clamps=book['clock_clamps'])
    write(root/(window+'_rows.json'),rows);write(root/(window+'_assessment.json'),summary_out)
    print(__import__('json').dumps({k:v for k,v in summary_out.items() if k not in ('temporary_rejection_reach','pending_buy_cancel_events')}),flush=True)


def run(root):
    root.mkdir(parents=True,exist_ok=False)
    design=REPO/'docs/optimisation/olr/experiments/olr-exp-0037.md'
    (root/'design_at_freeze.md').write_bytes(design.read_bytes());(root/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    sources={str(PARENT/'assessment.json'):sha(PARENT/'assessment.json')}
    for w in ('train','oos'):
        p=PARENT/'native_engines'/w/'ready.json';r=read(p);sources[str(p)]=sha(p)
        for key in ('database','olr_snapshots','olr_request'):
            p=Path(r['files'])/r['study'][key];sources[str(p)]=sha(p)
    write(root/'protocol.json',dict(at=now(),experiment='0037',script_sha256=sha(__file__),design_sha256=sha(design),sources=sources,
        windows=['train','oos'],new_financial_accounts=0,scope='Read-only complete opportunity continuity; no treatment outcomes.'))
    for w in ('train','oos'):
        cmd=[sys.executable,'-X','utf8','-B','-u',str(Path(__file__).resolve()),str(root),'--window',w]
        with (root/(w+'.log')).open('x',encoding='utf8') as log:
            p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            write(root/(w+'_process.json'),dict(at=now(),pid=p.pid,created_at=psutil.Process(p.pid).create_time(),command=cmd))
            code=p.wait()
        if code:raise RuntimeError(f'{w} extraction failed: {code}')
    write(root/'assessment.json',dict(at=now(),windows={w:read(root/(w+'_assessment.json')) for w in ('train','oos')},
        economic_objective_achieved=False,scope='Diagnostic census; new opportunity value and funded-policy results unestablished.'))
    status(root,'awaiting_economic_interpretation',prepared_next='Assess reachable state changes and next named lifecycle dependency; no automatic financial successor')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--window',choices=['train','oos']);args=p.parse_args()
    try:
        if args.window:window(args.root,args.window)
        else:run(args.root)
    except Exception:
        traceback.print_exc()
        if args.root.exists():status(args.root,'failed',window=args.window,traceback=traceback.format_exc())
        raise
