"""Capital-parameterized research diagnostics; original helpers remain immutable.

Only initial-capital denominators differ from the named predecessor functions.
The native policy-bound independent validator is called unchanged.
"""
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
from backtests.strategies.kalcb.retention_research_io_v1 import read, write, bound, verify, digest
from scripts.kalcb_population_assessment_v1 import compound, paired_attribution
from scripts.kalcb_frontier_activity_attribution_v1 import compare


def validate_capital_native(folder, item, *, save=True):
    """Preserve the original account verifier; bind its capital argument explicitly."""
    from backtests.auto.shared.cache_keys import stable_signature
    from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_case
    folder=Path(folder);verify([item[k] for k in ('manifest','plan','policy')])
    plan=read(item['plan']['path']);contract=plan['capital_sensitivity']
    request=read(folder/'request.json');result=read(folder/'result.json')
    manifest=read(item['manifest']['path']);signature=manifest.pop('manifest_hash')
    if stable_signature(manifest)!=signature or manifest['source_generation_hash']!=item['source_generation_hash']:
        raise ValueError('Changed frozen manifest')
    if request['plan']!=plan or request['capital_sensitivity']!=contract:
        raise ValueError('Capital plan mismatch')
    if Path(request['arguments']['audit_manifest']).resolve()!=Path(item['manifest']['path']).resolve():
        raise ValueError('Wrong capital source manifest')
    policy=Path(request['arguments']['policies'])
    if digest(policy/'policies_manifest.json')!=item['policy']['sha256']:
        raise ValueError('Wrong policy')
    for relative,sha in read(policy/'policies_manifest.json')['files'].items():
        if Path(relative).name!=relative or digest(policy/relative)!=sha:raise ValueError('Changed policy member')
    initial=contract['initial_equity'];audit=read(folder/'execution_audit.json')['audit_receipt']
    if audit['identity']['configuration']['initial_equity']!=initial:
        raise ValueError('Audit capital mismatch')
    if audit['qualification']['capacity']['initial_equity']!=initial or audit['qualification']['capacity']['capital_observed']['initial_equity']!=initial:
        raise ValueError('Broker capacity capital mismatch')
    if read(folder/'completion.json')['initial_equity']!=initial:raise ValueError('Worker capital mismatch')
    profile=dict(execution_source=dict(source_generation_hash=item['source_generation_hash']),
        audit_manifest=item['manifest']['path'],policy=dict(name='interaction_cost_margin',
        manifest_sha256=item['policy']['sha256'],directory=str(policy)),initial_equity_krw=initial,
        date_range=dict(start='2026-04-01',end='2026-07-10',sessions=69),metrics={contract['scenario']:result})
    checked=validate_case(folder,profile,contract['scenario'])
    if checked['daily'][0]['day']!='2026-04-01' or checked['daily'][-1]['day']!='2026-07-10':
        raise ValueError('Calendar truncated')
    allocation=read(folder/'allocation_reference.json')
    if allocation['enabled']!=(contract['mode']=='fixed') or allocation['reference']!=100_000_000:
        raise ValueError('Wrong effective allocation mode')
    if allocation['ledger_cash_mutations'] or allocation['external_capital_flows']:
        raise ValueError('Account replenished')
    if not allocation['rows'] or allocation['rows'][0]['actual_equity']!=initial:
        raise ValueError('Adapter capital mismatch')
    if save:write(folder/'independent_account_verification.json',checked['validation'])
    return checked['validation']

# Parameterized from backtests/strategies/kalcb/fixed_allocation_funding_v1.py
def funding_review(folder, initial_equity=100_000_000.):
    folder=Path(folder)
    if not read(folder/'independent_account_verification.json')['passed']:
        raise ValueError('Unverified funded account')
    submitted={};rejects=[];events=Counter()
    with gzip.open(folder/'orders.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);events[row['event']]+=1
            order=row.get('order',{})
            if order.get('side')!='BUY':continue
            if row['event']=='submit':submitted[order['order_id']]=order
            if row['event']=='reject':
                rejects.append({'order_id':order['order_id'],'symbol':order['symbol'],
                    'reason':row['reason'],'timestamp':row['timestamp'],
                    'decision_available_at':order.get('metadata',{}).get('decision_available_at'),
                    'actual_cash':row.get('cash'),'attempted_qty':row.get('attempted_qty')})
    cash=float(initial_equity);minimum=cash;executions=set();filled=set()
    for fill in read(folder/'fills.json'):
        eid=fill['metadata']['execution_id']
        if eid in executions:raise ValueError('Duplicate physical execution')
        executions.add(eid)
        buy=fill['side']=='BUY';cash+=(-1 if buy else 1)*fill['qty']*fill['price']-fill['metadata']['fee']
        minimum=min(minimum,cash)
        if buy:
            if fill['order_id'] not in submitted:raise ValueError('Unsubmitted physical buy')
            filled.add(fill['order_id'])
    if abs(cash-read(folder/'daily.json')[-1]['equity'])>.01:raise ValueError('Funding cash mismatch')
    return {'verified':True,'submitted_buy_orders':len(submitted),'physically_filled_buy_orders':len(filled),
        'submitted_buy_orders_without_fill':len(set(submitted)-filled),'events':dict(events),
        'rejected_buy_events':rejects,'unique_rejected_buy_orders':len({r['order_id'] for r in rejects}),
        'rejection_reasons':dict(Counter(r['reason'] for r in rejects)),
        'minimum_funded_cash_krw':minimum,'terminal_cash_krw':cash,
        'scope':'Actual broker journals and physical fills only. Borrowed cash is allowed within original audited leverage; negative cash alone is not an invalid fill. Retries/partial fills are dependent. These observations neither isolate gate PnL nor prove a hypothetical order was fundable.',
        'bindings':[bound(folder/f) for f in ('orders.jsonl.gz','fills.json','daily.json','independent_account_verification.json')]}


# Parameterized from docs/optimisation/kalcb/evidence/reconstruct_frontier.py
def metrics(folder, initial_equity=100_000_000.):
    result=read(folder/'result.json'); eq=read(folder/'equity.json'); fills=read(folder/'fills.json')
    days=sorted({s[:10] for s,_ in eq}); di={d:i for i,d in enumerate(days)}
    peak=float(initial_equity); pi=0; worst=0.; money=0.; wi=0; wpi=0; wp=peak
    for i,(s,v) in enumerate(eq):
        if v>=peak: peak=v;pi=i
        dd=1-v/peak; money=max(money,peak-v)
        if dd>worst: worst=dd;wi=i;wpi=pi;wp=peak
    recovery=next((i for i in range(wi+1,len(eq)) if eq[i][1]>=wp),None)
    ret=100*(eq[-1][1]/float(initial_equity)-1)
    assert abs(ret-result['return_pct'])<1e-8
    assert abs(100*worst-result['drawdown_pct'])<1e-8
    orders={};cash=float(initial_equity);mincash=cash; fees=0.
    for f in fills:
        fee=f['metadata']['fee'];fees+=fee
        cash+=(-1 if f['side']=='BUY' else 1)*f['qty']*f['price']-fee;mincash=min(mincash,cash)
        if f['side']=='BUY':
            o=orders.setdefault(f['order_id'],dict(notional=0.,equity=f['metadata']['portfolio_equity']))
            o['notional']+=f['qty']*f['price']
    assert abs(cash-eq[-1][1])<.01
    mat=sum(o['notional']>=.01*o['equity'] for o in orders.values())
    return dict(return_pct=ret,dd_pct=100*worst,max_money_dd_krw=money,
                money_loss_at_percentage_trough_krw=wp-eq[wi][1],
                dd_peak=eq[wpi][0],dd_trough=eq[wi][0],dd_recovery=eq[recovery][0] if recovery is not None else None,
                recovery_or_end_sessions=di[eq[recovery if recovery is not None else -1][0][:10]]-di[eq[wpi][0][:10]],
                end_underwater_pct=100*(1-eq[-1][1]/peak),end_underwater_sessions=di[eq[-1][0][:10]]-di[eq[pi][0][:10]],
                trades=result['trades'],profitable_trades=round(result['metrics']['win_rate']*result['trades']),
                funded_orders=len(orders),material_funded_entries=mat,buy_notional_krw=sum(o['notional'] for o in orders.values()),
                costs_excluding_embedded_slippage_krw=fees,min_cash_krw=mincash,profit_factor=result['metrics']['profit_factor'],
                sessions=len(days),expectancy_krw=result['metrics']['expectancy'],source_generation_hash=result['source_generation_hash'])


# Parameterized from scripts/kalcb_population_assessment_v1.py
def account(folder, initial_equity=100_000_000.):
    folder=Path(folder);daily=read(folder/'daily.json');trades=read(folder/'trades.json');result=read(folder/'result.json')
    if not read(folder/'independent_account_verification.json')['passed']:raise ValueError('Unverified account')
    dates=[d['day'] for d in daily]
    if dates!=sorted(set(dates)):raise ValueError('Duplicate/nonchronological session')
    balances=[float(initial_equity)]+[float(d['equity']) for d in daily]
    daily_returns=[balances[i+1]/balances[i]-1 for i in range(len(daily))]
    trade_net=sum(t['net_pnl'] for t in trades)
    if abs(trade_net-(balances[-1]-balances[0]))>.01:raise ValueError('Trade cash mismatch')
    if abs(compound(daily_returns)*100-result['return_pct'])>1e-8:raise ValueError('Daily return mismatch')
    monthly={};issuer=defaultdict(lambda:{'net_pnl_krw':0.,'trades':0});issuer_days=defaultdict(float)
    for i,day in enumerate(dates):
        month=day[:7]
        row=monthly.setdefault(month,{'start_equity_krw':balances[i],'end_equity_krw':balances[i],'sessions':0,'trades':0,'trade_net_pnl_krw':0.})
        row['end_equity_krw']=balances[i+1];row['sessions']+=1
    for t in trades:
        day=t['entry_decision_time'][:10]
        if day!=t['exit_fill_time'][:10]:raise ValueError('Overnight position requires different attribution')
        monthly[day[:7]]['trades']+=1;monthly[day[:7]]['trade_net_pnl_krw']+=t['net_pnl']
        issuer[t['symbol']]['net_pnl_krw']+=t['net_pnl'];issuer[t['symbol']]['trades']+=1
        issuer_days[day+'/'+t['symbol']]+=t['net_pnl']
    for row in monthly.values():
        row['return_pct']=100*(row['end_equity_krw']/row['start_equity_krw']-1)
        row['cash_change_krw']=row['end_equity_krw']-row['start_equity_krw']
        if abs(row['cash_change_krw']-row['trade_net_pnl_krw'])>.01:raise ValueError('Monthly cash mismatch')
    gains=sorted((v['net_pnl_krw'] for v in issuer.values() if v['net_pnl_krw']>0),reverse=True)
    issuer_day_values=sorted(issuer_days.values(),reverse=True)
    top_count=max(1,math.ceil(len(issuer_day_values)*.05))
    gross=sum(t['gross_pnl'] for t in trades);costs=sum(t['commission'] for t in trades)
    if abs(gross-costs-trade_net)>.01:raise ValueError('Cost reconciliation failed')
    metrics={k:result[k] for k in ('return_pct','drawdown_pct','trades','active_days','audit_status')}
    return {'metrics':metrics,'dates':dates,'daily_returns':daily_returns,'monthly':monthly,
            'issuers':dict(issuer),'issuer_days':dict(issuer_days),'net_pnl_krw':trade_net,
            'gross_pnl_krw':gross,'explicit_costs_krw':costs,
            'explicit_cost_note':'Fees/taxes recorded in commission; slippage is already in physical prices and gross PnL.',
            'positive_issuer_profit_top1_share':gains[0]/sum(gains) if gains else None,
            'positive_issuer_profit_top5_share':sum(gains[:5])/sum(gains) if gains else None,
            'top_5pct_issuer_days_count':top_count,'top_5pct_issuer_days_net_pnl_krw':sum(issuer_day_values[:top_count]),
            'traded_issuer_days':len(issuer_days),
            'bindings':[bound(folder/name) for name in ('result.json','daily.json','trades.json','equity.json','fills.json','independent_account_verification.json')]}


# Parameterized from scripts/kalcb_frontier_activity_attribution_v1.py
def opportunities(folder, initial_equity=100_000_000.):
    folder = Path(folder)
    trades = read(folder/'trades.json')
    fills = read(folder/'fills.json')
    if not read(folder/'independent_account_verification.json')['passed']:
        raise ValueError('Unverified account')
    buys = defaultdict(list)
    for fill in fills:
        if fill['side'] == 'BUY':
            buys[fill['symbol']+'/'+fill['metadata']['signal_bar']].append(fill)
    rows = {}
    for trade in trades:
        key = trade['symbol']+'/'+trade['entry_decision_time']
        if key in rows:
            raise ValueError('Repeated decision clock needs explicit grouping')
        actual = buys[key]
        if sum(f['qty'] for f in actual) != trade['qty']:
            raise ValueError('Trade quantity does not reconcile to native entry fills')
        notional = sum(f['qty']*f['price'] for f in actual)
        if abs(notional-trade['qty']*trade['entry_price']) > .01:
            raise ValueError('Trade entry notional mismatch')
        equity = actual[0]['metadata']['portfolio_equity']
        rows[key] = dict(qty=trade['qty'], net_pnl_krw=trade['net_pnl'],
                         net_pnl_per_share=trade['net_pnl']/trade['qty'],
                         entry_notional_krw=notional, decision_equity_krw=equity,
                         material=notional >= .01*equity,
                         entry_orders=len({f['order_id'] for f in actual}),
                         entry_fill=trade['entry_fill_time'], exit_fill=trade['exit_fill_time'],
                         entry_price=trade['entry_price'], exit_price=trade['exit_price'],
                         exit_reason=trade['exit_reason'])
    if set(rows) != set(buys):
        raise ValueError('Unmatched filled opportunity')
    pnl = sum(x['net_pnl_krw'] for x in rows.values())
    result = read(folder/'result.json')
    if abs(pnl-result['return_pct']*float(initial_equity)/100) > .01:
        raise ValueError('Opportunity account PnL mismatch')
    issuer_days = defaultdict(float)
    for key, row in rows.items():
        issuer_days[key[:17]] += row['net_pnl_krw']  # six-digit symbol + slash + ISO day
    material = [r for r in rows.values() if r['material']]
    gains = sorted((r['net_pnl_krw'] for r in rows.values() if r['net_pnl_krw'] > 0), reverse=True)
    losses = [r['net_pnl_krw'] for r in rows.values() if r['net_pnl_krw'] < 0]
    summary = dict(opportunities=len(rows), issuer_days=len(issuer_days),
                   profitable_issuer_days=sum(v > 0 for v in issuer_days.values()),
                   material_opportunities=len(material),
                   profitable_material_opportunities=sum(r['net_pnl_krw'] > 0 for r in material),
                   material_opportunities_net_above_1bp_equity=sum(r['net_pnl_krw'] >= .0001*r['decision_equity_krw'] for r in material),
                   net_pnl_krw=pnl, gross_positive_trade_pnl_krw=sum(gains), gross_negative_trade_pnl_krw=sum(losses),
                   top1_positive_trade_share=gains[0]/sum(gains) if gains else None,
                   top5_positive_trade_share=sum(gains[:5])/sum(gains) if gains else None,
                   mean_win_krw=sum(gains)/len(gains) if gains else None,
                   mean_loss_krw=sum(losses)/len(losses) if losses else None)
    return dict(summary=summary, opportunities=rows,
                bindings=[bound(folder/f) for f in ('trades.json','fills.json','result.json','independent_account_verification.json')])
