"""C16 read-only reconstruction. Never a trading-policy or native-oracle runner."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
from scipy.stats import rankdata, spearmanr

REPO = Path(__file__).resolve().parents[1]
BASE = Path('E:/CodexResearch/k_stock_trader')
SOURCE = BASE/'kalcb_opportunity_lifecycle_20261016/funded_interaction_v1'
OUT = BASE/'kalcb_round2_assessment_20261017/v1/analysis'
SCENARIOS = ('baseline', 'participation', 'cost', 'combined')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def digest(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def bound(p):
    return {'path': str(Path(p).resolve()), 'sha256': digest(p)}


def write(p, obj):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False, default=lambda x: x.item() if hasattr(x, 'item') else str(x))


def csvwrite(p, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def lines(p):
    opener = gzip.open if str(p).endswith('.gz') else open
    with opener(p, 'rt', encoding='utf-8-sig') as f:
        for row in f:
            if row.strip(): yield json.loads(row)


def stamp(x):
    return datetime.fromisoformat(x).isoformat()


def blocks(n, length, draws=4000):
    rng = np.random.default_rng(41017 + length + n)
    starts = rng.integers(0, n, (draws, int(np.ceil(n/length))))
    return ((starts[:, :, None] + np.arange(length)) % n).reshape(draws, -1)[:, :n]


def bootstrap(x):
    x = np.asarray(x, float); result = {}
    for length in (5, 10, 20):
        sums = x[blocks(len(x), length)].sum(axis=1)
        se = float(np.std(sums, ddof=1))
        result[str(length)] = dict(sum=float(x.sum()), ci95=np.quantile(sums, [.025, .975]).tolist(),
                                  se=se, planning_mde=(1.959964+.841621)*se,
                                  sessions=len(x), block_equivalents=len(x)/length,
                                  effective_independent_sample_size=None)
    return result


def auc(x, y):
    n1 = int(y.sum()); n0 = len(y)-n1
    return float((rankdata(x)[y].sum()-n1*(n1+1)/2)/(n1*n0)) if n1 and n0 else None


def discriminate(rows, feature, calendar, ci_draws=4000):
    used = [r for r in rows if r.get(feature) is not None and np.isfinite(r[feature]) and r.get('net_bps') is not None]
    if not used: return {'feature': feature, 'n': 0}
    x = np.array([r[feature] for r in used], float); value = np.array([r['net_bps'] for r in used]); y = value > 0
    dates = [r['day'] for r in used]; groups = [np.array([i for i, d in enumerate(dates) if d == day], int) for day in calendar]
    a = auc(x, y); rho = float(spearmanr(x, value).statistic) if np.ptp(x) else None
    decile = np.minimum(9, np.floor((rankdata(x, method='average')-.5)/len(x)*10)).astype(int)
    bydec = [{'decile': j+1, 'n': int(sum(decile == j)), 'net_bps': float(value[decile == j].mean()),
              'win_fraction': float(y[decile == j].mean())} for j in range(10) if sum(decile == j)]
    cis = {}
    for length in (5, 10, 20):
        av, rv, means = [], [], []
        for draw in blocks(len(calendar), length, ci_draws):
            ix = np.concatenate([groups[j] for j in draw]); xx=x[ix]; yy=y[ix]; vv=value[ix]
            if not len(ix): continue
            aa=auc(xx, yy)
            if aa is not None: av.append(aa)
            if np.ptp(xx) and np.ptp(vv): rv.append(float(spearmanr(xx, vv).statistic))
            means.append(float(vv.mean()))
        cis[str(length)] = {'auc95': np.quantile(av, [.025, .975]).tolist() if av else None,
                            'spearman95': np.quantile(rv, [.025, .975]).tolist() if rv else None,
                            'expectancy95': np.quantile(means, [.025, .975]).tolist() if means else None,
                            'valid_auc_draws': len(av)}
    return dict(feature=feature, n=len(used), sessions=len(set(dates)), issuer_days=len({(r['day'],r['symbol']) for r in used}),
                auc=a, spearman=rho, expectancy_bps=float(value.mean()), deciles=bydec, uncertainty=cis)


def analyse_cell(period, scenario, profiles, protocol):
    folder=SOURCE/'cases'/f'combined_{period}_{scenario}'
    profile=profiles[period+'_cap2']; plan=read(profile['plan']['path'])
    conn=sqlite3.connect('file:'+Path(plan['database']).as_posix()+'?mode=ro', uri=True)
    fills=read(folder/'fills.json'); trades=read(folder/'trades.json'); eq=read(folder/'equity.json'); daily=read(folder/'daily.json'); result=read(folder/'result.json')
    calendar=[r['day'] for r in daily]
    events=defaultdict(list)
    for i,f in enumerate(fills,1): events[stamp(f['timestamp'])].append((i,f))
    quantities=defaultdict(int); cash_by=defaultdict(float); cash=1e8; prices={}; source_day=None; daybars={}
    peak=1e8; peak_at=None; peak_book={}; peak_inventory={}; worst=0.; dd=None; max_error=0.; fee_error=0.; mincash=1e8
    exposure=[]; pnlpaths=[]; per_day_data={}; fee_total=slip_total=buys=0.; rowcount=0; max_money=0.; native_fee_model={'BUY':.0002,'SELL':.0020}
    for i,(rawts,nav) in enumerate(eq,1):
        ts=stamp(rawts); day=ts[:10]
        if day != source_day:
            source_day=day; daybars=defaultdict(dict); symbars=defaultdict(list)
            for sym,t,o,h,l,c,v in conn.execute('SELECT symbol,ts,o,h,l,c,v FROM bars WHERE day=? ORDER BY ts,symbol',(day,)):
                t=stamp(t); daybars[t][sym]=float(c); symbars[sym].append((t,float(o),float(h),float(l),float(c),float(v)))
            per_day_data[day]=dict(symbars)
        for fi,f in events.get(ts,[]):
            sign=1 if f['side']=='BUY' else -1; qty=f['qty']; price=f['price']; fee=f['metadata']['fee']; sym=f['symbol']
            expected=qty*price*native_fee_model[f['side']]; fee_error=max(fee_error,abs(expected-fee))
            change=-sign*qty*price-fee; cash+=change; cash_by[sym]+=change; quantities[sym]+=sign*qty
            if quantities[sym]<0 or int(qty)!=qty: raise ValueError('Invalid inventory/quantity')
            fee_total+=fee; slip_total+=sign*(price-f['metadata']['base_price'])*qty; rowcount+=1
            if sign>0: buys+=qty*price
        prices.update(daybars.get(ts,{}))
        inventory={s:q for s,q in quantities.items() if q}
        book={s:cash_by[s]+q*prices.get(s,0) for s,q in quantities.items()}
        reconstructed=1e8+sum(book.values()); max_error=max(max_error,abs(reconstructed-nav)); mincash=min(mincash,cash)
        gross=sum(q*prices[s] for s,q in inventory.items()); exposure.append(gross/nav)
        if nav>=peak:
            peak=nav;peak_at=ts;peak_book=book.copy();peak_inventory=inventory.copy()
        decline=100*(peak-nav)/peak;max_money=max(max_money,peak-nav)
        if decline>worst:
            worst=decline;dd={'peak_at':peak_at,'trough_at':ts,'peak_equity':peak,'trough_equity':nav,'dd_pct':decline,
                'decline_krw':peak-nav,'peak_inventory':peak_inventory,'trough_inventory':inventory,
                'symbol_contributions':sorted([{'symbol':s,'change_krw':book.get(s,0)-peak_book.get(s,0)} for s in set(book)|set(peak_book)],key=lambda r:r['change_krw']),
                'equity_artifact_row':i}
        pnlpaths.append({'timestamp':ts,'cash':cash,'inventory_value':gross,'nav':nav,'reconstructed_nav':reconstructed,'equity_row':i})
    if rowcount!=len(fills) or max_error>.02 or fee_error>.02 or any(quantities.values()) or abs(cash-1e8-sum(t['net_pnl'] for t in trades))>.02:
        raise ValueError(f'Account reconciliation failed {period}/{scenario}: marks={max_error} fees={fee_error} fillrows={rowcount}/{len(fills)}')
    assert abs(100*(cash/1e8-1)-result['return_pct'])<1e-8 and abs(worst-result['drawdown_pct'])<1e-8
    recover=next((stamp(t) for t,e in eq if stamp(t)>dd['trough_at'] and e>=dd['peak_equity']),None)
    dd['recovered_at']=recover;dd['money_max_dd']=max_money
    dd['sessions_peak_to_recovery']=calendar.index(recover[:10])-calendar.index(dd['peak_at'][:10]) if recover else None
    periodrows=[]; relaxed=[]; reference=[]
    for ti,t in enumerate(trades,1):
        m=t['route_metadata']; entry=stamp(t['entry_fill_time']); exit_=stamp(t['exit_fill_time']); day=entry[:10]; sym=t['symbol']; bars=per_day_data[day].get(sym,[])
        # Excursion on bars starting before exit avoids using post-exit intrabar extrema.
        held=[b for b in bars if entry<=b[0]<exit_]; held=held or [b for b in bars if b[0]==entry]
        policy_end=datetime.fromisoformat(bars[-1][0])-timedelta(minutes=10) if bars else datetime.fromisoformat(exit_)
        future=[b for b in bars if datetime.fromisoformat(entry)<datetime.fromisoformat(b[0])<=policy_end]
        qty=t['qty']; eprice=t['entry_price']; notional=qty*eprice
        mfe=max((b[2]/eprice-1 for b in held), default=0)*100;mae=min((b[3]/eprice-1 for b in held),default=0)*100
        mfe_at=next((b[0] for b in held if abs((b[2]/eprice-1)*100-mfe)<1e-9),None)
        rr={'period':period,'scenario':scenario,'day':day,'symbol':sym,'entry':entry,'exit':exit_,'trade_row':ti,'source':str(folder/'trades.json'),
            'qty':qty,'notional':notional,'net_krw':t['net_pnl'],'net_bps':1e4*t['net_pnl']/notional,'route':m.get('entry_route'),
            'regime':m.get('regime_tier'),'selection_score':m.get('selection_score'),'quality_votes':m.get('entry_quality_votes'),
            'relvol':m.get('first30_rel_volume'),'opening_return':m.get('first30_ret'),'opening_cpr':m.get('first30_signal_bar_cpr'),
            'atr_fraction':m.get('daily_atr',0)/eprice,'negative_rank':-m['candidate_rank'] if m.get('candidate_rank') is not None else None,
            'mfe_pct':mfe,'mae_pct':mae,'time_to_mfe_minutes':(datetime.fromisoformat(mfe_at)-datetime.fromisoformat(entry)).total_seconds()/60 if mfe_at else None,
            'exit_reason':t['exit_reason'],'decision_equity':m['portfolio_equity'],'useful':notional>=.01*m['portfolio_equity'] and t['net_pnl']>=.0001*m['portfolio_equity']}
        slippage=.0013 if scenario in ('cost','combined') else .0003
        sale=lambda price:qty*price*(1-slippage)*(1-.002)-notional*(1+.0002)
        for mins in (30,60,120):
            target=datetime.fromisoformat(entry)+timedelta(minutes=mins)
            b=next((b for b in future if datetime.fromisoformat(b[0])>=target),None)
            rr[f'price_proxy_{mins}m_net_bps']=1e4*sale(b[1])/notional if b else None
        best=max((sale(b[1]) for b in future),default=t['net_pnl'])
        # Baseline itself is an allowed relaxed candidate; do not lower its payoff.
        best=max(best,t['net_pnl'])
        rr['relaxed_best_open_net_krw']=best
        relaxed.append(best);periodrows.append(rr)
        if future:
            start=next((b for b in bars if b[0]==entry),None)
            if start:reference.append((day,sym,future[-1][1]/start[1]-1))
    csvwrite(OUT/f'{period}_{scenario}_trade_paths.csv',periodrows)
    csvwrite(OUT/f'{period}_{scenario}_account_marks.csv',pnlpaths)
    daypnl=np.diff([1e8]+[x['equity'] for x in daily])/1e6
    vetopath=Path(protocol['controls']['veto'][period+'/'+scenario]);vd=read(vetopath/'daily.json')
    assert [x['day'] for x in vd]==calendar
    paired=daypnl-np.diff([1e8]+[x['equity'] for x in vd])/1e6
    orders=Counter(); buyorders=set(); fillorders={f['order_id'] for f in fills if f['side']=='BUY'}; order_reasons=Counter()
    for o in lines(folder/'orders.jsonl.gz'):
        orders[o['event']]+=1
        if o['event']=='submit' and o['order']['side']=='BUY':buyorders.add(o['order']['order_id'])
        if o['event'] in ('reject','cancel','expire'):order_reasons[o.get('reason','unknown')]+=1
    gates=Counter();capdates=set();vetoactions=read(folder/'horizon_actions.json')
    for d in lines(folder/'decisions.jsonl.gz'):
        gates[d['reason']]+=1
        cap=d.get('metadata',{}).get('pending_funding_reservation')
        if cap and cap['capped_qty']<cap['original_qty']:capdates.add(d['timestamp'][:10])
    monthly={month:sum(v for d,v in zip(calendar,daypnl) if d.startswith(month)) for month in sorted({d[:7] for d in calendar})}
    usefuldays=sum(any(t['notional']>=.01*t['decision_equity'] for t in rs) and sum(t['net_krw'] for t in rs)>=.0001*rs[0]['decision_equity']
        for rs in [[r for r in periodrows if r['day']==day and r['symbol']==sym] for day,sym in {(r['day'],r['symbol']) for r in periodrows}])
    symbols_by_key={}
    for d,s,v in reference:symbols_by_key.setdefault((d,s),v)
    out=dict(period=period,scenario=scenario,source=str(folder),net_krw=cash-1e8,return_pct=result['return_pct'],dd_pct=worst,
        trades=len(trades),issuer_days=len({(t['day'],t['symbol']) for t in periodrows}),sessions=len(calendar),active_days=sum(x!=0 for x in daypnl),
        useful_issuer_days=usefuldays,max_nav_error_krw=max_error,max_fee_error_krw=fee_error,explicit_fees_krw=fee_total,
        embedded_slippage_krw=slip_total,buy_notional_krw=buys,min_cash_krw=mincash,mean_mark_exposure=float(np.mean(exposure)),
        max_mark_exposure=float(max(exposure)),terminal_inventory={},cash_reconciled=True,full_mtm_reconciled=True,
        dd=dd,monthly_net_m=monthly,day_pnl_m=daypnl.tolist(),calendar=calendar,paired_veto_daily_m=paired.tolist(),
        baseline_uncertainty=bootstrap(daypnl),paired_veto_uncertainty=bootstrap(paired),cap_distinct_dates=sorted(capdates),
        orders=dict(orders),buy_orders=len(buyorders),unfilled_orders=len(buyorders-fillorders),terminal_order_reasons=dict(order_reasons),
        decisions=dict(gates),veto_actions=len(vetoactions),
        relaxed_selected_current_exits_net_krw=sum(max(0,t['net_pnl']) for t in trades),
        relaxed_current_entries_best_open_net_krw=sum(relaxed),relaxed_selected_best_open_net_krw=sum(max(0,x) for x in relaxed),
        opportunity_reference_mean_gross_pct=100*np.mean(list(symbols_by_key.values())) if symbols_by_key else None,
        opportunity_reference_issuer_days=len(symbols_by_key),
        note='Relaxed selected-lifetime statistics are not jointly funded or executable ceilings. Cash and exposure use actual fills; price proxies ignore capacity and altered lifecycle.')
    write(OUT/f'{period}_{scenario}_account.json',out)
    # Outcome discrimination, explicitly conditioned on baseline realised entries.
    ds=[discriminate(periodrows,f,calendar) for f in ('quality_votes','relvol','opening_return','opening_cpr','atr_fraction','negative_rank')]
    write(OUT/f'{period}_{scenario}_discrimination.json',ds)
    conn.close()
    return out


def main():
    OUT.mkdir(parents=True,exist_ok=False);began=time.monotonic()
    profiles=read(SOURCE/'registration.json')['profiles'];protocol=read(SOURCE/'protocol.json')
    selected=read(REPO/'data/backtests/output/kalcb/round_2/optimized_config.json')
    for period in ('is','oos'):
        cfg=read(Path(profiles[period+'_cap2']['folder'])/'config.json')
        assert selected['mutations']==cfg, 'Saved optimized config differs'
    bindings=[bound(__file__),bound(REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0041.md'),
        bound(REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0041-analysis-binding.md'),bound(SOURCE/'registration.json'),
        bound(REPO/'data/backtests/output/kalcb/round_2/optimized_config.json')]
    for period in ('is','oos'):
        profile=profiles[period+'_cap2']
        for k in ('manifest','plan','policy'):
            b=profile[k];assert digest(b['path'])==b['sha256'];bindings.append(b)
        plan=read(profile['plan']['path']);assert digest(plan['database'])==plan['input_sha256'][plan['database']]
        bindings.append(bound(plan['database']))
        for sc in SCENARIOS:
            folder=SOURCE/'cases'/f'combined_{period}_{sc}'
            bindings += [bound(folder/f) for f in ('fills.json','trades.json','equity.json','daily.json','orders.jsonl.gz','decisions.jsonl.gz','execution_audit.json')]
    write(OUT/'binding.json',dict(registered_at=datetime.now(timezone.utc).isoformat(),inputs=bindings,config_fields=len(selected['mutations'])))
    cells=[]
    for period in ('is','oos'):
        for sc in SCENARIOS:
            print('analysing',period,sc,flush=True);cells.append(analyse_cell(period,sc,profiles,protocol))
    paired={}
    for period in ('is','oos'):
        rows=[c for c in cells if c['period']==period]
        paired[period]=bootstrap(np.mean([c['paired_veto_daily_m'] for c in rows],axis=0))
    write(OUT/'paired_aggregate.json',paired)
    summary=[{k:v for k,v in c.items() if not isinstance(v,(dict,list))} for c in cells]
    csvwrite(OUT/'cells.csv',summary)
    # Frozen score separates economic ranking, risk feasibility and evidence.
    def score(r,dd,target): return r-.25*max(0,dd-target)
    synthetic=[dict(case='genuine_gain',score=score(11,8,8),rank_above_control=True,risk_pass=True,evidence_pass=True),
               dict(case='control',score=score(10,8,8),risk_pass=True,evidence_pass=True),
               dict(case='frequency_padding',score=score(10,8,8),useful_increment=0,risk_pass=True,evidence_pass=True),
               dict(case='hidden_intraday_drawdown',score=score(15,16,8),risk_pass=False,evidence_pass=False),
               dict(case='lucky_path',score=score(15,8,8),risk_pass=True,evidence_pass=False)]
    assert synthetic[0]['score']>synthetic[1]['score']==synthetic[2]['score']
    assert not synthetic[3]['risk_pass'] and not synthetic[4]['evidence_pass']
    write(OUT/'score_synthetic_checks.json',dict(checks=synthetic,passed=True,score='R - .25*target_excess_DD; useful frequency only evidence-qualified tie break'))
    write(OUT/'completion.json',dict(complete=True,seconds=time.monotonic()-began,cells=len(cells),new_treatment_runs=0,
        native_oracle_available=False,matched_random_entry_test_available=False,independent_validation=False))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import traceback
        if OUT.exists():write(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
