"""Descriptive funded opportunity attribution; no selection or trading policy.

Matches actual symbol/decision clocks and reconciles every PnL delta. Quantity
and per-share terms are path arithmetic, not isolated causal counterfactuals.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb.retention_research_io_v1 import read, write, bound, verify


def opportunities(folder):
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
    if abs(pnl-result['return_pct']*1_000_000) > .01:
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


def compare(treatment, control):
    a=treatment['opportunities']; b=control['opportunities']
    shared=set(a)&set(b); added=set(a)-set(b); removed=set(b)-set(a)
    quantity=sum((a[k]['qty']-b[k]['qty'])*b[k]['net_pnl_per_share'] for k in shared)
    unit=sum(a[k]['qty']*(a[k]['net_pnl_per_share']-b[k]['net_pnl_per_share']) for k in shared)
    added_pnl=sum(a[k]['net_pnl_krw'] for k in added)
    removed_pnl=sum(b[k]['net_pnl_krw'] for k in removed)
    delta=treatment['summary']['net_pnl_krw']-control['summary']['net_pnl_krw']
    if abs(quantity+unit+added_pnl-removed_pnl-delta) > .01:
        raise ValueError('Funded path difference does not reconcile')
    fields=('opportunities','issuer_days','profitable_issuer_days','material_opportunities',
            'profitable_material_opportunities','material_opportunities_net_above_1bp_equity')
    return dict(summary_delta={k:treatment['summary'][k]-control['summary'][k] for k in fields},
                shared=len(shared), added=len(added), removed=len(removed), net_pnl_delta_krw=delta,
                shared_quantity_at_control_unit_pnl_krw=quantity, shared_unit_outcome_at_treatment_qty_krw=unit,
                added_pnl_krw=added_pnl, removed_control_pnl_krw=removed_pnl,
                shared_quantity_changed=sum(a[k]['qty']!=b[k]['qty'] for k in shared),
                shared_entry_fill_changed=sum(a[k]['entry_fill']!=b[k]['entry_fill'] for k in shared),
                shared_exit_fill_changed=sum(a[k]['exit_fill']!=b[k]['exit_fill'] for k in shared),
                shared_exit_reason_changed=sum(a[k]['exit_reason']!=b[k]['exit_reason'] for k in shared),
                shared_entry_price_changed=sum(abs(a[k]['entry_price']-b[k]['entry_price'])>1e-8 for k in shared),
                added_keys=sorted(added), removed_keys=sorted(removed))


def main(root):
    output=root/'activity_attribution_v1.json'
    if output.exists():raise ValueError('Preserve existing attribution')
    if not read(root/'completion.json')['complete']:raise ValueError('Assessment incomplete')
    index=read(root/'account_index.json');accounts={k:opportunities(v) for k,v in index.items()}
    comparisons={}
    for period in ('is','oos'):
        for scenario in ('baseline','participation','cost','combined'):
            suffix='/'+period+'/'+scenario
            for a,b in (('fixed_prior100','compound_prior100'),('fixed_prior100_hold8','compound_prior100_hold8'),
                        ('fixed_prior100_hold8','fixed_prior100'),('compound_prior100_hold8','compound_prior100')):
                comparisons[a+'_vs_'+b+suffix]=compare(accounts[a+suffix],accounts[b+suffix])
    bindings=[bound(root/'completion.json'),bound(root/'account_index.json'),bound(__file__)]
    for a in accounts.values():bindings.extend(a['bindings'])
    verify(bindings)
    write(output,dict(at_utc=datetime.now(timezone.utc).isoformat(),accounts=accounts,comparisons=comparisons,bindings=bindings,
        scope='Descriptive implementation of C5 useful-frequency and funded path attribution. Symbol/decision-clock opportunities reconcile to all entry fills and terminal trade/account PnL. Material means >=1% decision-equity filled notional; positive and >=1bp decision-equity net profit are reported separately, not selection gates. Shared quantity and unit-outcome terms are exact arithmetic on different funded policies, not isolated alpha or causal effects. All history is reused.'))
    print(json.dumps({k:v['summary'] for k,v in accounts.items()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
