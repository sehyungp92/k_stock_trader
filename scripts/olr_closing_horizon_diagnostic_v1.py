"""0038 planned auction-price/capacity attribution; no financial replay."""
from collections import Counter,defaultdict
from datetime import datetime,timedelta
import argparse,copy,gzip,json,math,sqlite3,subprocess,sys,traceback
from pathlib import Path
from types import SimpleNamespace
import psutil
from olr_selective_half_continuation_v1 import read,write,sha,now,status

REPO=Path(__file__).resolve().parents[1]
PARENT=Path('E:/CodexResearch/k_stock_trader/olr_continuing_research_20261016/relative_qualification_v2')


def window(root,w):
    ready=read(PARENT/'native_engines'/w/'ready.json');files=Path(ready['files']);study=ready['study']
    sys.path.insert(0,str(files))
    from backtests.engine.sim_broker import SimBroker,_auction_target
    from strategy_common.market import MarketBar
    summary=read(PARENT/'assessment.json')['summaries'][w+'_nominal_control'];case=Path(summary['case'])
    assert sha(case/'completion.json')==summary['completion_sha256']
    completion=read(case/'completion.json');checks={n:sha(case/n)==h for n,h in completion['artifacts'].items()};assert all(checks.values())
    definition=read(files/study['minute_source_definition'])
    db=files/definition['canonical_database'];con=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);con.execute('PRAGMA query_only=ON')
    orders={};fills=[];last=None
    with gzip.open(case/'minute_events.jsonl.gz','rt',encoding='utf8') as stream:
        for line in stream:
            e=json.loads(line);kind=e['event']
            if kind in ('opening_marks','interval_marks','closing_marks'):
                t=datetime.fromisoformat(e['timestamp'])+timedelta(minutes=kind!='opening_marks')
                last=max(last,t) if last else t
            elif kind=='submit':
                o=e['order'];at=datetime.fromisoformat(o['metadata'].get('decision_available_at',o['submitted_at']))
                last=max(last,at) if last else at
                if o['reason']=='next_close_exit':
                    assert o['side']=='SELL' and o['order_type']=='CLOSE_AUCTION'
                    orders[o['order_id']]={**o,'available_at':at.isoformat(),'events':[]}
            elif kind=='fill':
                f=e['fill'];t=datetime.fromisoformat(f['timestamp']);last=max(last,t) if last else t
                fills.append(f)
                if f['order_id'] in orders:orders[f['order_id']]['events'].append(dict(kind=kind,at=t.isoformat(),qty=f['qty'],price=f['price'],fee=f['metadata']['fee']))
            elif kind in ('cancel','expire','reject'):
                o=e['order'];at=datetime.fromisoformat(e.get('timestamp',last.isoformat()));last=max(last,at)
                if o['order_id'] in orders:orders[o['order_id']]['events'].append(dict(kind=kind,at=last.isoformat(),original_timestamp=e.get('timestamp'),remaining=o['qty'],reason=e.get('reason')))
    trades=read(case/'trades.json');population=read(case/'funded_population_receipt.json')['entry_contexts']
    rows=[]
    costs=SimpleNamespace(slippage_bps=5.,auction_slippage_bps=0.)
    broker=SimpleNamespace(costs=costs)
    for ident,order in orders.items():
        o=SimpleNamespace(**{k:v for k,v in order.items() if k not in ('events','available_at')});o.submitted_at=datetime.fromisoformat(o.submitted_at)
        clock=_auction_target(o);at=o.submitted_at.replace(hour=clock.hour,minute=clock.minute,second=0,microsecond=0)
        ts=at.isoformat();day=ts[:10];symbol=o.symbol
        t=[t for t in trades if t['symbol']==symbol and datetime.fromisoformat(t['entry_fill_time'])<=o.submitted_at<=datetime.fromisoformat(t['exit_fill_time'])]
        assert len(t)==1,(symbol,day,t);t=t[0]
        base=dict(window=w,symbol=symbol,entry_day=t['entry_decision_time'][:10],order_day=day,order_id=ident,
                  submitted_at=order['available_at'],target_at=ts,limit=o.limit_price,submitted_qty=o.qty,events=order['events'],
                  actual_trade_pnl=t['net_pnl'],actual_exit_at=t['exit_fill_time'],actual_exit_reason=t['exit_reason'])
        cancelled=[x for x in order['events'] if x['kind'] in ('cancel','reject') and x['at'][:10]<=day]
        if cancelled:
            rows.append(dict(**base,disposition='priority_or_other_cancel',local_value=0,additional_shares=0));continue
        point=con.execute('SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? AND ts=?',(symbol,ts)).fetchone()
        if point is None:
            rows.append(dict(**base,disposition='missing_auction_source',local_value=None,additional_shares=None));continue
        stamp,op,hi,lo,cl,vol=point
        assert op==hi==lo==cl and vol>=0,(symbol,point)
        bar=MarketBar(symbol=symbol,timestamp=at,open=op,high=hi,low=lo,close=cl,volume=vol,timeframe='1m',is_completed=True)
        quote=SimBroker._close_auction_fill_price(broker,o,bar)
        unlimited=copy.copy(o);unlimited.limit_price=None
        market_quote=SimBroker._close_auction_fill_price(broker,unlimited,bar)
        assert market_quote is not None and o.metadata.get('auction_nonfill_rate',0)==0
        actual=[f for f in fills if f['order_id']==ident]
        q=sum(f['qty'] for f in actual)
        for f in actual:
            assert f['metadata']['execution_phase']=='auction' and f['metadata']['base_price']==cl
            assert f['metadata']['execution_bar']['volume']==vol and math.isclose(f['price'],quote,abs_tol=.0001)
        shared=[f for f in fills if f['symbol']==symbol and datetime.fromisoformat(f['timestamp'])==at and f['metadata']['execution_phase']=='auction' and f['order_id']!=ident]
        allowance=max(0,math.floor(vol*.01)-sum(f['qty'] for f in shared))
        possible=min(o.qty,allowance)
        if quote is not None:assert q==possible,(symbol,ts,q,possible)
        else:assert q==0
        more=possible-q
        future=[f for f in fills if f['symbol']==symbol and f['side']=='SELL' and at<datetime.fromisoformat(f['timestamp'])<=datetime.fromisoformat(t['exit_fill_time'])]
        remaining=sum(f['qty'] for f in future)
        assert q+remaining==o.qty,(symbol,ts,q,remaining,o.qty)
        later_proceeds=sum(f['qty']*f['price']-f['metadata']['fee'] for f in future)
        marginal=more*(market_quote*(1-.0022)-later_proceeds/remaining) if more else 0.
        carried=remaining*cl
        affected=[]
        if more:
            for p in population:
                pa=datetime.fromisoformat(p['at'])
                if not at<pa<datetime.fromisoformat(t['exit_fill_time']):continue
                residual=sum(f['qty'] for f in future if datetime.fromisoformat(f['timestamp'])>pa)
                if not residual:continue
                for d in p['decisions']:
                    prop=p['proposals'][d['intent_id']]
                    if d['approved_qty']<prop['qty']:
                        affected.append(dict(at=p['at'],symbol=prop['symbol'],desired=prop['qty'],approved=d['approved_qty'],
                            binding=d['binding_limit'],actual_cash=p['actual_deployable_cash'],held_residual=residual))
        disposition=('full_completion' if q==o.qty else 'price_and_capacity' if quote is None and possible<o.qty else 'price' if quote is None else 'capacity')
        rows.append(dict(**base,disposition=disposition,source=list(point),quote=quote,market_quote=market_quote,actual_auction_shares=q,
            physical_allowance=allowance,other_shared_shares=sum(f['qty'] for f in shared),additional_shares=more,remaining_shares=remaining,
            actual_remaining_proceeds=later_proceeds,local_value=marginal,local_value_per_extra_notional_bps=1e4*marginal/(more*cl) if more else None,
            carried_notional=carried,carry_calendar_hours=(datetime.fromisoformat(t['exit_fill_time'])-at).total_seconds()/3600 if remaining else 0,
            subsequent_constrained_proposals=affected,
            scope='Fixed-parent quantity/proceeds assay; proportional residual allocation, actual original auction shared-volume constraint; not funded policy value.'))
    con.close()
    changed=[r for r in rows if r.get('additional_shares')]
    days=defaultdict(float)
    for r in changed:days[r['order_day']]+=r['local_value']
    result=dict(at=now(),window=w,rows=rows,counts=dict(Counter(r['disposition'] for r in rows)),
        source_checks=len([r for r in rows if 'source' in r]),changed_orders=len(changed),changed_dates=len(days),local_value=sum(days.values()),
        positive=sum(r['local_value']>0 for r in changed),negative=sum(r['local_value']<0 for r in changed),date_values=dict(days),
        constraint_contacts=sum(len(r['subsequent_constrained_proposals']) for r in changed),
        case_completion_sha256=sha(case/'completion.json'),source_database_sha256=sha(db),
        limits=['Orders cancelled by original risk priority remain cancelled.',
                'Fixed residual average future proceeds are attribution only; an early partial sale can change subsequent management and capital.',
                'Native interval/capacity source model, not exchange queue or untouched qualification.'])
    write(root/(w+'_assessment.json'),result)
    print(json.dumps({k:result[k] for k in ['window','counts','source_checks','changed_orders','changed_dates','local_value','positive','negative','constraint_contacts']}),flush=True)


def run(root):
    root.mkdir(exist_ok=False,parents=True);design=REPO/'docs/optimisation/olr/experiments/olr-exp-0038.md'
    (root/'design_at_freeze.md').write_bytes(design.read_bytes());(root/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    sources={str(PARENT/'assessment.json'):sha(PARENT/'assessment.json')}
    for w in ('train','oos'):
        rp=PARENT/'native_engines'/w/'ready.json';r=read(rp);sources[str(rp)]=sha(rp)
        p=Path(r['files'])/r['study']['minute_source_definition'];sources[str(p)]=sha(p)
    write(root/'protocol.json',dict(at=now(),experiment='0038',sources=sources,script_sha256=sha(__file__),design_sha256=sha(design),new_financial_accounts=0))
    for w in ('train','oos'):
        status(root,'extracting',window=w)
        cmd=[sys.executable,'-X','utf8','-B','-u',str(Path(__file__).resolve()),str(root),'--window',w]
        with (root/(w+'.log')).open('x',encoding='utf8') as log:
            p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            write(root/(w+'_process.json'),dict(at=now(),pid=p.pid,created_at=psutil.Process(p.pid).create_time(),command=cmd))
            code=p.wait()
        if code:raise RuntimeError(f'{w} extraction failed: {code}')
    write(root/'assessment.json',dict(at=now(),windows={w:read(root/(w+'_assessment.json')) for w in ('train','oos')},economic_objective_achieved=False))
    status(root,'awaiting_economic_interpretation',prepared_next='Assess fixed-horizon payoff, recovery and next-access dependencies; no automatic financial successor')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--window',choices=['train','oos']);args=p.parse_args()
    try:
        if args.window:window(args.root,args.window)
        else:run(args.root)
    except Exception:
        traceback.print_exc()
        if args.root.exists():status(args.root,'failed',window=args.window,traceback=traceback.format_exc())
        raise
