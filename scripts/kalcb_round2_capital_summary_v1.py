"""C16 capital-pair summaries and independent actual-capital cash/MTM checks."""
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import sqlite3
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_round2_assessment_v1 import read,write,bound,csvwrite,bootstrap,SOURCE,SCENARIOS,stamp

ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_round2_assessment_20261017/v1')
RUN=ROOT/'noise_r2b'
OUT=ROOT/'capital_summary'


def reconstruct(folder,initial,conn):
    fills=read(folder/'fills.json');eq=read(folder/'equity.json');trades=read(folder/'trades.json')
    byts=defaultdict(list)
    for f in fills:byts[stamp(f['timestamp'])].append(f)
    cash=initial;qty=defaultdict(int);prices={};day=None;error=0;fees=0;slip=0;seen=0
    for ts,nav in eq:
        ts=stamp(ts)
        if ts[:10]!=day:
            day=ts[:10];bars=defaultdict(dict)
            for sym,t,c in conn.execute('SELECT symbol,ts,c FROM bars WHERE day=? ORDER BY ts,symbol',(day,)):
                bars[stamp(t)][sym]=float(c)
        for f in byts[ts]:
            sign=1 if f['side']=='BUY' else -1
            fee=f['qty']*f['price']*(.0002 if sign==1 else .002)
            assert abs(fee-f['metadata']['fee'])<.02
            cash-=sign*f['qty']*f['price']+fee;qty[f['symbol']]+=sign*f['qty']
            assert qty[f['symbol']]>=0 and int(f['qty'])==f['qty']
            fees+=fee;slip+=sign*(f['price']-f['metadata']['base_price'])*f['qty'];seen+=1
        prices.update(bars.get(ts,{}));actual=cash+sum(q*prices[s] for s,q in qty.items() if q)
        error=max(error,abs(actual-nav))
    assert seen==len(fills) and error<.02
    assert not any(qty.values()) and abs(cash-initial-sum(t['net_pnl'] for t in trades))<.02
    campaigns=defaultdict(list)
    for t in trades:campaigns[t['entry_fill_time'][:10],t['symbol']].append(t)
    useful=0
    for ts in campaigns.values():
        useful+=int(any(t['qty']*t['entry_price']>=.01*t['route_metadata']['portfolio_equity'] for t in ts)
            and sum(t['net_pnl'] for t in ts)>=.0001*ts[0]['route_metadata']['portfolio_equity'])
    return dict(max_nav_error_krw=error,final_cash=cash,explicit_fees_krw=fees,embedded_slippage_krw=slip,
        lifetimes=len(trades),issuer_days=len(campaigns),useful_issuer_days=useful,fill_count=len(fills)),campaigns


def main():
    done=read(RUN/'completion.json');assert done['complete']
    OUT.mkdir(exist_ok=False)
    write(OUT/'binding.json',dict(program=bound(__file__),run=bound(RUN/'completion.json'),protocol=bound(ROOT/'noise/registration.json'),
        estimand='Own-capital native path. Capital paired differences use daily PnL divided by respective initial capital, in percentage points. Never independent market samples.'))
    profiles=read(SOURCE/'registration.json')['profiles'];rows=[];pairs=[];aggregate={};sources=[];unqualified=[]
    for p in ('is','oos'):
        plan=read(profiles[p+'_cap2']['plan']['path']);conn=sqlite3.connect('file:'+Path(plan['database']).as_posix()+'?mode=ro',uri=True)
        incs={95000000:[],105000000:[]}
        for sc in SCENARIOS:
            original=SOURCE/'cases'/f'combined_{p}_{sc}';base=read(ROOT/'analysis'/f'{p}_{sc}_account.json')
            baseline=100*np.diff([1e8]+[r['equity'] for r in read(original/'daily.json')])/1e8
            oldtrades=read(original/'trades.json');oldkeys={(t['entry_fill_time'][:10],t['symbol']) for t in oldtrades}
            for initial in (95000000,100000000,105000000):
                folder=original if initial==100000000 else RUN/'cases'/f'capital_{p}_{sc}_{initial}'
                result=read(folder/'result.json');native=read(folder/'execution_audit.json')
                if initial!=100000000:
                    verdict=read(RUN/f'capital_{p}_{sc}_{initial}_verdict.json')
                    if not verdict['qualified']:
                        bad=dict(period=p,scenario=sc,initial_krw=initial,return_pct=result['return_pct'],dd_pct=result['drawdown_pct'],
                            native_qualified=False,hard_dd_pass=False,source=str(folder),reason=verdict['reason'])
                        rows.append(bad);unqualified.append(bad)
                        continue
                assert result['audit_status']=='passed' and native['audit_pass']
                rec,campaigns=reconstruct(folder,initial,conn);daily=read(folder/'daily.json');assert [r['day'] for r in daily]==base['calendar']
                returns=100*np.diff([initial]+[r['equity'] for r in daily])/initial
                assert abs(returns.sum()-result['return_pct'])<1e-8
                row=dict(period=p,scenario=sc,initial_krw=initial,return_pct=result['return_pct'],dd_pct=result['drawdown_pct'],
                    net_krw=rec['final_cash']-initial,return_delta_pp=result['return_pct']-base['return_pct'],dd_delta_pp=result['drawdown_pct']-base['dd_pct'],
                    useful_delta=rec['useful_issuer_days']-base['useful_issuer_days'],added_issuer_days=len(set(campaigns)-oldkeys),
                    removed_issuer_days=len(oldkeys-set(campaigns)),native_qualified=True,hard_dd_pass=result['drawdown_pct']<=(12 if sc=='baseline' else 15),
                    **rec,source=str(folder))
                rows.append(row);sources.extend(bound(folder/f) for f in ('result.json','fills.json','equity.json','daily.json','trades.json','execution_audit.json'))
                if initial!=100000000:
                    diff=returns-baseline;uncertainty=bootstrap(diff);incs[initial].append(diff)
                    pairs.append(dict(period=p,scenario=sc,initial_krw=initial,statistics=uncertainty))
                    write(OUT/f'{p}_{sc}_{initial}_daily_pair.json',dict(calendar=base['calendar'],paired_delta_pp=diff.tolist(),statistics=uncertainty))
        aggregate[p]={str(cap):bootstrap(np.mean(v,axis=0)) if len(v)==4 else {'unavailable':'A capital scenario is unqualified; no partial aggregate'} for cap,v in incs.items()};conn.close()
    csvwrite(OUT/'cells.csv',rows);write(OUT/'paired_cells.json',pairs);write(OUT/'paired_aggregate.json',aggregate)
    write(OUT/'sources.json',sources)
    bridges=[read(ROOT/'is_baseline_exact/identity_bridge.json')]
    for p in ('is','oos'):
        bridges.append(read(ROOT/'noise/cases'/f'bridge_{p}'/'identity_bridge.json'))
        for sc in SCENARIOS:
            if (p,sc)!=('is','baseline'):bridges.append(read(RUN/'cases'/f'exact_{p}_{sc}'/'identity_bridge.json'))
    assert all(b['passed'] for b in bridges)
    write(OUT/'repeatability.json',dict(complete=True,exact_original_accounts=8,capital_adapter_bridges=2,all_physical_paths_exact=True,
        observed_implementation_dispersion=0,bridges=bridges,additional_repaired_95m_identity=read(RUN/'cases/capital_is_baseline_95000000/identity_bridge.json')))
    write(OUT/'completion.json',dict(complete=True,capital_cells=16,source_reference_cells=8,all_qualified_native_mtm_independently_reconciled=True,unqualified_cells=unqualified,
        total_native_executions=done['total_executions_including_original_pilot_and_preserved_failure'],new_treatments=0))


if __name__=='__main__':main()
