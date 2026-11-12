"""C16 lifetime accounting of the already observed finer-MTM drawdown episodes."""
from collections import Counter
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_round2_assessment_v1 import read, write, bound, csvwrite, SOURCE, SCENARIOS

ROOT = Path('E:/CodexResearch/k_stock_trader/kalcb_round2_assessment_20261017/v1')
OUT = ROOT / 'minute_anatomy'


def stamp(value):
    return datetime.fromisoformat(value).isoformat()


def main():
    OUT.mkdir(exist_ok=False)
    points = read(ROOT / 'minute_marks/critical_mark_review.json')['points']
    sources = [bound(SOURCE / 'cases' / f'combined_{p}_{sc}' / name)
               for p in ('is', 'oos') for sc in SCENARIOS for name in ('fills.json', 'trades.json')]
    write(OUT / 'binding.json', dict(program=bound(__file__),
        protocol=bound(Path(__file__).resolve().parents[1] / 'docs/optimisation/kalcb/experiments/kalcb-exp-0041-minute-anatomy.md'),
        critical_points=bound(ROOT / 'minute_marks/critical_mark_review.json'), sources=sources))
    summaries = []
    for period in ('is', 'oos'):
        for sc in SCENARIOS:
            folder = SOURCE / 'cases' / f'combined_{period}_{sc}'
            fills = read(folder / 'fills.json'); trades = read(folder / 'trades.json')
            chosen = {p['kind']: p for p in points if p['period'] == period and p['scenario'] == sc}
            episodes = []; live = {}; qty = Counter()
            for i, f in enumerate(fills, 1):
                sym = f['symbol']; sign = 1 if f['side'] == 'BUY' else -1
                if sign == 1 and not qty[sym]:
                    live[sym] = dict(symbol=sym, entry=stamp(f['timestamp']), fills=[])
                assert sym in live
                live[sym]['fills'].append((i, f)); qty[sym] += sign * f['qty']; assert qty[sym] >= 0
                if not qty[sym]:
                    ep = live.pop(sym); ep['exit'] = stamp(f['timestamp']); episodes.append(ep)
            assert not live and len(episodes) == len(trades)

            def value_at(ep, point):
                cash = 0.; quantity = 0; at = stamp(point['at']); inclusive = point['phase'] == 'open_after_fill'
                for _, f in ep['fills']:
                    ts = stamp(f['timestamp'])
                    if ts < at or (ts == at and inclusive):
                        sign = 1 if f['side'] == 'BUY' else -1
                        cash -= sign * f['qty'] * f['price'] + f['metadata']['fee']; quantity += sign * f['qty']
                if quantity:
                    source = next(s for s in point['source_marks'] if s['symbol'] == ep['symbol'])
                    assert quantity == source['qty'] and source['age_minutes'] == 0
                    cash += quantity * source['price']
                return cash, quantity

            rows = []
            for ep in episodes:
                before, q0 = value_at(ep, chosen['peak']); after, q1 = value_at(ep, chosen['trough'])
                delta = after - before
                if abs(delta) < 1e-8: continue
                tr = next(i for i, t in enumerate(trades, 1) if t['symbol'] == ep['symbol'] and stamp(t['entry_fill_time']) == ep['entry'])
                net = sum((-1 if f['side'] == 'BUY' else 1) * f['qty'] * f['price'] - f['metadata']['fee'] for _, f in ep['fills'])
                assert abs(net - trades[tr - 1]['net_pnl']) < .02
                rows.append(dict(symbol=ep['symbol'], entry=ep['entry'], exit=ep['exit'],
                    contribution_krw=delta, value_at_peak_krw=before, value_at_trough_krw=after,
                    qty_at_peak=q0, qty_at_trough=q1, net_lifetime_krw=net,
                    subsequent_mark_to_final_krw=net-after, trade_source=str(folder / 'trades.json'),
                    trade_row=tr, first_fill_row=ep['fills'][0][0], last_fill_row=ep['fills'][-1][0]))
            rows.sort(key=lambda r: r['contribution_krw'])
            expected = chosen['trough']['nav'] - chosen['peak']['nav']
            error = abs(sum(r['contribution_krw'] for r in rows) - expected); assert error < .02
            csvwrite(OUT / f'{period}_{sc}_lifetimes.csv', rows)
            summaries.append(dict(period=period, scenario=sc, peak_at=chosen['peak']['at'], trough_at=chosen['trough']['at'],
                peak_phase=chosen['peak']['phase'], trough_phase=chosen['trough']['phase'],
                dd_pct=100 * (1-chosen['trough']['nav']/chosen['peak']['nav']), decline_krw=-expected,
                affected_lifetimes=len(rows), affected_entry_dates=len({r['entry'][:10] for r in rows}),
                peak_open_lifetimes=sum(r['qty_at_peak'] > 0 for r in rows),
                trough_open_lifetimes=sum(r['qty_at_trough'] > 0 for r in rows),
                at_trough_remaining_to_actual_exit_krw=sum(r['subsequent_mark_to_final_krw'] for r in rows if r['qty_at_trough'] > 0),
                reconciliation_error_krw=error))
    csvwrite(OUT / 'summary.csv', summaries)
    write(OUT / 'completion.json', dict(complete=True, cells=8, new_accounts=0, max_reconciliation_error_krw=max(r['reconciliation_error_krw'] for r in summaries)))
    print(summaries)


if __name__ == '__main__': main()
