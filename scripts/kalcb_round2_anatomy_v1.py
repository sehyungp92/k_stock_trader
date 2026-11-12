"""C16 gate census and independently reconciled lifetime drawdown contributions."""
from collections import Counter,defaultdict
from pathlib import Path
from datetime import datetime
import csv,gzip,hashlib,json,sqlite3

ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_round2_assessment_20261017/v1')
SOURCE=Path('E:/CodexResearch/k_stock_trader/kalcb_opportunity_lifecycle_20261016/funded_interaction_v1')
OUT=ROOT/'anatomy'
def read(p):return json.loads(p.read_text())
def stamp(t):return datetime.fromisoformat(t).isoformat()
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2)
def csvwrite(p,rows):
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['none']);w.writeheader();w.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    write(OUT/'binding.json',dict(program=__file__,sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        design='kalcb-exp-0041 / C16 trade and concurrent-excursion attribution; no policy intervention'))
    summary=[]
    for period in ('is','oos'):
        plan=read(SOURCE/'runtime'/f'{period}_cap2/plan.json')
        db=sqlite3.connect('file:'+Path(plan['database']).as_posix()+'?mode=ro',uri=True)
        for sc in ('baseline','participation','cost','combined'):
            cell=period+'_'+sc;folder=SOURCE/'cases'/('combined_'+cell);a=read(ROOT/'analysis'/(cell+'_account.json'));dd=a['dd']
            fills=read(folder/'fills.json');trades=read(folder/'trades.json');episodes=[];live={};q=Counter()
            for i,f in enumerate(fills,1):
                sym=f['symbol'];buy=f['side']=='BUY'
                if buy and not q[sym]:live[sym]=dict(symbol=sym,entry=stamp(f['timestamp']),fills=[])
                assert sym in live
                live[sym]['fills'].append((i,f));q[sym]+=f['qty']*(1 if buy else -1)
                assert q[sym]>=0
                if not q[sym]:
                    e=live.pop(sym);e['exit']=stamp(f['timestamp']);episodes.append(e)
            assert not live and len(episodes)==len(trades)
            def nav_at(ep,ts):
                cash=0.;qty=0
                for i,f in ep['fills']:
                    if stamp(f['timestamp'])<=ts:
                        sign=1 if f['side']=='BUY' else -1;cash+=-sign*f['qty']*f['price']-f['metadata']['fee'];qty+=sign*f['qty']
                price=0.
                if qty:
                    row=db.execute('SELECT c FROM bars WHERE symbol=? AND day=? AND ts<=? ORDER BY ts DESC LIMIT 1',(ep['symbol'],ts[:10],ts)).fetchone()
                    assert row is not None;price=row[0]
                return cash+qty*price,qty
            rows=[]
            for ep in episodes:
                before,qbefore=nav_at(ep,dd['peak_at']);after,qafter=nav_at(ep,dd['trough_at'])
                change=after-before
                if abs(change)<1e-8:continue
                trow=next(i for i,t in enumerate(trades,1) if t['symbol']==ep['symbol'] and stamp(t['entry_fill_time'])==ep['entry'])
                net=sum((-1 if f['side']=='BUY' else 1)*f['qty']*f['price']-f['metadata']['fee'] for _,f in ep['fills'])
                assert abs(net-trades[trow-1]['net_pnl'])<.02
                rows.append(dict(symbol=ep['symbol'],entry=ep['entry'],exit=ep['exit'],contribution_krw=change,
                    qty_at_peak=qbefore,qty_at_trough=qafter,net_lifetime_krw=net,subsequent_recovery_krw=net-after,
                    trade_source=str(folder/'trades.json'),trade_row=trow,first_fill_row=ep['fills'][0][0],last_fill_row=ep['fills'][-1][0]))
            rows.sort(key=lambda r:r['contribution_krw'])
            error=abs(sum(r['contribution_krw'] for r in rows)+dd['decline_krw']);assert error<.02
            csvwrite(OUT/(cell+'_drawdown_lifetimes.csv'),rows)
            gates=defaultdict(lambda:{'pass_rows':0,'fail_rows':0,'pass_days':set(),'fail_days':set(),'fail_issuer_days':set()})
            reasons=defaultdict(lambda:{'rows':0,'days':set(),'issuer_days':set()})
            with gzip.open(folder/'decisions.jsonl.gz','rt',encoding='utf-8') as f:
                for idx,line in enumerate(f,1):
                    d=json.loads(line);m=d.get('metadata',{});day=d['timestamp'][:10];sym=d['symbol'];reasons[d['reason']]['rows']+=1
                    reasons[d['reason']]['days'].add(day);reasons[d['reason']]['issuer_days'].add((day,sym))
                    for g in m.get('filter_decisions',[]):
                        if not g.get('applicable',True):continue
                        target=gates[g['filter_name']];p=bool(g['passed']);target['pass_rows' if p else 'fail_rows']+=1
                        target['pass_days' if p else 'fail_days'].add(day)
                        if not p:target['fail_issuer_days'].add((day,sym))
            gs=[dict(gate=k,**{f:len(v) if isinstance(v,set) else v for f,v in val.items()}) for k,val in gates.items()]
            rs=[dict(reason=k,**{f:len(v) if isinstance(v,set) else v for f,v in val.items()}) for k,val in reasons.items()]
            csvwrite(OUT/(cell+'_gates.csv'),gs);csvwrite(OUT/(cell+'_reasons.csv'),rs)
            summary.append(dict(period=period,scenario=sc,affected_lifetimes=len(rows),affected_entry_dates=len({r['entry'][:10] for r in rows}),
                contribution_sum_krw=sum(r['contribution_krw'] for r in rows),reconciliation_error_krw=error,
                peak_open_lifetimes=sum(r['qty_at_peak']>0 for r in rows),trough_open_lifetimes=sum(r['qty_at_trough']>0 for r in rows),
                at_trough_remaining_to_actual_exit_krw=sum(r['subsequent_recovery_krw'] for r in rows if r['qty_at_trough']>0)))
        db.close()
    csvwrite(OUT/'summary.csv',summary);write(OUT/'completion.json',dict(complete=True,cells=8))


if __name__=='__main__':main()
