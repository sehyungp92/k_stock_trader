"""Authenticate completed nominal cases and explain path-dependent sizing.

Diagnostic only: no new policy, parameter selection or modification of the
running registered qualification. Accounting attribution is not causal proof.
"""
import hashlib
import json
from html import escape
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('E:/CodexResearch/k_stock_trader/olr_round1_optimization_20261010/participation_v5')
OUT = REPO / 'data/backtests/analysis/olr_participation_20261011'
IDS = ('caa25a36d10de3dc4', 'cb0901ba5660e7a21')
TAG = 'empty_participation_rescue_v5'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def drawdown(equity):
    high, peak, worst = 1e8, 'initial_account', dict(pct=0.)
    for timestamp, nav in equity:
        if nav > high:
            high, peak = nav, timestamp
        dd = 100 * (1 - nav / high)
        if dd > worst['pct']:
            worst = dict(pct=dd, peak_time=peak, peak_nav=high,
                         trough_time=timestamp, trough_nav=nav, loss=high-nav)
    return worst


def run():
    sources = {str(Path(__file__).resolve()): sha(__file__)}
    def read(path):
        sources[str(path)] = sha(path)
        return json.loads(path.read_text(encoding='utf-8'))
    def case(candidate, window):
        folder = ROOT / 'cases' / candidate / window / '100000000_baseline'
        done = read(folder / 'completion.json')
        for name in ('trades.json', 'equity.json', 'execution_audit.json'):
            assert sha(folder / name) == done['artifacts'][name]
        assert read(folder / 'execution_audit.json')['audit_pass']
        trades, equity = read(folder / 'trades.json'), read(folder / 'equity.json')
        return trades, dict(return_pct=100*(equity[-1][1]/1e8-1),
            entries=len(trades), entry_days=len({x['entry_decision_time'][:10] for x in trades}),
            drawdown=drawdown(equity))
    windows = {}
    key = lambda x: (x['symbol'], x['entry_decision_time'])
    for window in ('train', 'oos'):
        old, old_metrics = case(IDS[0], window)
        new, new_metrics = case(IDS[1], window)
        a, b = {key(x): x for x in old}, {key(x): x for x in new}
        matched, added, removed = a.keys() & b.keys(), b.keys() - a.keys(), a.keys() - b.keys()
        effects = []
        for k in sorted(matched):
            x, y = a[k], b[k]
            xm, ym = x['route_metadata'], y['route_metadata']
            xf, yf = xm['capacity_forecast'], ym['capacity_forecast']
            # Same convention as the existing qualified report: reprice the
            # original quantity first, then apply quantity change at new P&L/share.
            quantity = (y['qty']-x['qty'])*y['net_pnl']/y['qty']
            execution = x['qty']*(y['net_pnl']/y['qty']-x['net_pnl']/x['qty'])
            delta = y['net_pnl']-x['net_pnl']
            assert abs(quantity+execution-delta) < 1e-7
            effects.append(dict(symbol=k[0], entry=k[1], old_qty=x['qty'], new_qty=y['qty'],
                old_pnl=x['net_pnl'], new_pnl=y['net_pnl'], pnl_delta=delta,
                quantity_effect=quantity, per_unit_execution_effect=execution,
                old_exit_forecast=xf['exit'], new_exit_forecast=yf['exit'],
                old_entry_forecast=xf['entry'], new_entry_forecast=yf['entry'],
                forecast_history_changed=xf['exit']['samples'] != yf['exit']['samples'],
                forecast_status_changed=xf['exit']['status'] != yf['exit']['status'],
                old_binding=xm['allocation_binding_limit'], new_binding=ym['allocation_binding_limit'],
                old_available_cash=xm['available_cash_before_entry'], new_available_cash=ym['available_cash_before_entry']))
        pnl_delta = sum(x['net_pnl'] for x in new)-sum(x['net_pnl'] for x in old)
        quantity = sum(x['quantity_effect'] for x in effects)
        execution = sum(x['per_unit_execution_effect'] for x in effects)
        membership = sum(b[k]['net_pnl'] for k in added)-sum(a[k]['net_pnl'] for k in removed)
        assert abs(pnl_delta-quantity-execution-membership) < 1e-6
        winners = {k for k in a if a[k]['net_pnl'] > 0}
        pilot = [x for x in new if x['route_metadata'].get(TAG)]
        windows[window] = dict(control=old_metrics, treatment=new_metrics,
            accounting=dict(net_pnl_difference=pnl_delta, quantity_effect=quantity,
                per_unit_execution_effect=execution, added_minus_removed_effect=membership),
            added=len(added), removed=len(removed), matched=len(matched),
            removed_trades=[dict(symbol=k[0],entry=k[1],qty=a[k]['qty'],net_pnl=a[k]['net_pnl']) for k in sorted(removed)],
            pilot=dict(entries=len(pilot), notional=sum(x['qty']*x['entry_price'] for x in pilot),
                pnl=sum(x['net_pnl'] for x in pilot)),
            all_original_winner_retention_pct=100*sum(b.get(k,{}).get('net_pnl',0) for k in winners)/sum(a[k]['net_pnl'] for k in winners),
            winner_to_loser=sum(b[k]['net_pnl'] <= 0 for k in winners & b.keys()),
            changed_exit_history_count=sum(x['forecast_history_changed'] for x in effects),
            changed_exit_status_count=sum(x['forecast_status_changed'] for x in effects),
            matched_effects=sorted(effects, key=lambda x: (x['pnl_delta'], x['entry'], x['symbol'])))
    result = dict(sources=sources, windows=windows, scope='Authenticated nominal cases only; stress and capped comparisons still governed by the registered controller.',
        limitations=['Accounting decomposition is not a randomized causal estimate.',
            'Added observations are valid information; suppressing them solely to recover historical winners would be unjustified.',
            'Sparse-history status changes, allocator substitution and later compounding interact; component effects are not independent.',
            'No parameter or acceptance gate is changed; no successor promoted by this diagnostic.'])
    (OUT/'nominal_interactions.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    def table(headers, rows):
        return '<table><tr>'+''.join('<th>'+escape(h)+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'
    body = '<h1>OLR entry pilot: interim nominal results and sizing interactions</h1>'
    body += '<p>The treatment improves all three OOS objectives in the nominal replay, but materially reduces IS return. It therefore cannot qualify for promotion. Stress and capped comparisons continue unchanged. Faster-exit engineering remains a separate unfinished component.</p>'
    body += table(['Window','Policy','Return %','Intraday DD %','Entries','Entry days'],[[w,label,f"{m['return_pct']:.3f}",f"{m['drawdown']['pct']:.3f}",m['entries'],m['entry_days']] for w,r in windows.items() for label,m in [('Control',r['control']),('Treatment',r['treatment'])]])
    body += '<h2>Where the P&amp;L changed</h2>'+table(['Window','Net change KRW m','Quantity effect','Per-unit execution effect','Added minus removed','Pilot entries','Pilot P&amp;L KRW m'],[[w,*[f'{r["accounting"][k]/1e6:.4f}' for k in ('net_pnl_difference','quantity_effect','per_unit_execution_effect','added_minus_removed_effect')],r['pilot']['entries'],f'{r["pilot"]["pnl"]/1e6:.4f}'] for w,r in windows.items()])
    body += '<p>Additional candidates change routed auction-history coverage. When a symbol moves from insufficient history to an available forecast, the exit-capacity cap can activate abruptly. That changes later incumbent sizes and the cash left for other trades, even though incumbent selection is unchanged.</p>'
    body += '<p>The 24 added OOS pilot entries collectively lost KRW0.953m. The nominal OOS gain depends principally on removal of the May27 329180 entry, which lost KRW2.013m in the control. This is evidence of changed portfolio allocation, not established positive alpha from the newly admitted family.</p>'
    body += table(['Window','Date','Symbol','Old quantity','New quantity','P&amp;L change KRW m','Exit forecast history changed','Exit forecast status changed'],[[w,x['entry'][:10],x['symbol'],x['old_qty'],x['new_qty'],f'{x["pnl_delta"]/1e6:.4f}',x['forecast_history_changed'],x['forecast_status_changed']] for w,r in windows.items() for x in r['matched_effects'][:8]])
    body += '<h2>Drawdown levels</h2>'+table(['Window','Policy','Peak KRW m','Trough KRW m','Peak-to-trough loss KRW m'],[[w,label,*[f'{m["drawdown"][k]/1e6:.4f}' for k in ('peak_nav','trough_nav','loss')]] for w,r in windows.items() for label,m in [('Control',r['control']),('Treatment',r['treatment'])]])
    body += '<p>The next structural diagnosis is to distinguish the sparse-history transition from ordinary cash competition and compounding. Valid new observations must not be hidden to preserve a known winner. Any alternative observation or allocation policy requires a separate registration, control comparison and all original gates.</p>'
    body += '<p>April–July is repeatedly inspected development data. These findings establish historical nominal behaviour, not future expected returns or independent validation. No live policy was changed.</p>'
    body += '<p><a href="nominal_interactions.json">Authenticated detailed evidence</a> · <a href="background_handoff.txt">Background continuation</a> · <a href="review_bundle/implementation.patch">Implemented research patch</a></p>'
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OLR interim sizing diagnosis</title><style>body{font:16px/1.6 system-ui;max-width:1450px;margin:40px auto;padding:0 24px;color:#17243a}table{border-collapse:collapse;width:100%;margin:20px 0;font-size:13px}td,th{padding:8px;border:1px solid #ccd5df;text-align:left}th{background:#eaf0f7}h1,h2{line-height:1.3}</style>'+body+'</html>'
    (OUT/'nominal_interactions.html').write_text(html,encoding='utf-8')
    print(json.dumps({w:{k:v for k,v in r.items() if k!='matched_effects'} for w,r in windows.items()},indent=2))


if __name__ == '__main__':
    run()
