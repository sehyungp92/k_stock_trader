"""Additional read-only funding and data-support checks for v4 failure analysis."""
import gzip
import json
from pathlib import Path
import sqlite3
from statistics import median

from scripts.olr_repair_failure_diagnosis_v5 import REPO, ROOT, OUT, SOURCES, read, bind, sha


def main():
    micro = read(OUT / 'one_minute_feasibility.json')
    result = {'one_minute_support': {}, 'entry_support': {}, 'peer_tail_activation': {}}
    for w in ['train', 'oos']:
        selected = read(ROOT / 'repair_v4/cases/c4f49be01ed09b8dc' / w / '100000000_baseline/trades.json')
        final_days = {(r['symbol'],r['entry_fill_time']):r['exit_fill_time'][:10] for r in selected}
        window = micro['windows'][w]
        bad_days = {(r['symbol'], r['timestamp'][:10]) for r in window['mismatch_examples']}
        excluded = [r for r in window['rows'] if any(s == r['symbol'] and r['entry'][:10] <= d <= final_days[r['symbol'],r['entry']] for s,d in bad_days)]
        clean = [r for r in window['rows'] if r not in excluded]
        result['one_minute_support'][w] = {'complete_lifecycle_trades': len(clean), 'excluded_trade_keys': [[r['symbol'],r['entry']] for r in excluded],
            'gross_exit_value_change_supported_only': sum(r['isolated_gross_exit_value_change'] for r in clean),
            'winner_gross_exit_value_change_supported_only': sum(r['isolated_gross_exit_value_change'] for r in clean if r['native_net_pnl']>0),
            'supported_earlier_fills': sum(r['early_qty']>0 for r in clean),
            'supported_winners_hurt': sum(r['isolated_gross_exit_value_change'] < -1 and r['native_net_pnl']>0 for r in clean),
            'comparison': 'Fixed-quantity gross exit-value screen; NOT strategy P&L, portfolio return or DD.'}
        frequency = read(OUT / f'frequency_feasibility_{w}.json')
        rows = [dict(r) for r in frequency['rows'] if r['six_bar_signal']]
        by_time = {}
        for r in rows:
            by_time.setdefault(r['signal_time'], []).append(r)
        trace = ROOT / 'remaining_drawdown_v2' / w
        comp = read(trace / 'completion.json')
        path = bind(trace / 'native_journal.jsonl.gz')
        assert sha(path) == comp['artifacts']['native_journal.jsonl.gz']
        old = read(trace / 'trades.json')
        economic = lambda ts: [(t['symbol'], t['entry_fill_time'], t['qty'], round(t['net_pnl'],5)) for t in ts]
        assert economic(old) == economic(selected), 'Trace must economically match retained control'
        cash = 100_000_000.
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            for line in f:
                event = json.loads(line)
                if event['event'] == 'fill':
                    assert abs(event['cash_before'] - cash) < 1e-5
                    cash = event['cash_after']
                if event['event'] == 'strategy_state' and event['timestamp'] in by_time:
                    for r in by_time[event['timestamp']]:
                        r['control_cash_at_signal'] = cash
                        r['already_held_in_control'] = r['symbol'] in event['positions']
                        r['pending_entry_in_control'] = r['symbol'] in event['pending_entries']
        study = read(ROOT / 'repair_v4/runtime/c4f49be01ed09b8dc' / w / 'baseline/study.json')
        db = bind(Path(study['code']) / study['database'])
        con = sqlite3.connect(db.as_uri() + '?mode=ro', uri=True)
        for r in rows:
            assert 'control_cash_at_signal' in r
            bar = con.execute('select c,v from bars where symbol=? and day=? and ts=?', (r['symbol'],r['date'],r['signal_time'])).fetchone()
            if bar is None:
                # Frozen DB timestamps sometimes use a space; no time-shift inference.
                bar = con.execute('select c,v from bars where symbol=? and day=? and replace(ts,\' \',\'T\')=?', (r['symbol'],r['date'],r['signal_time'])).fetchone()
            assert bar is not None
            r['one_bar_capacity_notional_at_signal_close'] = int(bar[1] * .01) * bar[0]
        con.close()
        supported = [r for r in rows if not r['already_held_in_control'] and not r['pending_entry_in_control'] and r['control_cash_at_signal']>=1e6 and r['one_bar_capacity_notional_at_signal_close']>=1e6]
        result['entry_support'][w] = {'signal_count':len(rows), 'signal_days':len({r['date'] for r in rows}),
            'already_held_count':sum(r['already_held_in_control'] for r in rows),
            'cash_ge_1m_count':sum(r['control_cash_at_signal']>=1e6 for r in rows),
            'bar_capacity_ge_1m_count':sum(r['one_bar_capacity_notional_at_signal_close']>=1e6 for r in rows),
            'no_incumbent_and_both_ge_1m_count':len(supported), 'no_incumbent_and_both_ge_1m_days':len({r['date'] for r in supported}),
            'median_control_cash':median(r['control_cash_at_signal'] for r in rows), 'rows':rows,
            'limitations':'Local support only, not independent funding for every candidate or a fill forecast: shared cash, rank allocation, position/sector/account risk, historical exit capacity, pending order reservation and later displacement still need native replay.'}
        peer_folder = ROOT / 'repair_v4/cases/c1da7e504a33bf2ad' / w / '100000000_baseline'
        peer_comp = read(peer_folder / 'completion.json')
        assert sha(peer_folder / 'trades.json') == peer_comp['artifacts']['trades.json']
        peer_trades = read(peer_folder / 'trades.json')
        tail_rows = []
        original = {(x['symbol'],x['entry_fill_time']):x for x in selected}
        for tr in peer_trades:
            meta = tr['route_metadata']; tail = meta['liquidation_tail_evidence']
            atr_fraction = 1.25 * meta['daily_atr'] / tr['entry_price']
            prior = original.get((tr['symbol'],tr['entry_fill_time']))
            tail_rows.append({'symbol':tr['symbol'], 'entry':tr['entry_fill_time'], 'own_tail':tail['symbol_tail_mean'],
                'peer_tail':tail['peer_tail_mean'], 'combined_tail':tail['fraction'], 'atr_1p25_fraction':atr_fraction,
                'actual_qty':tr['qty'], 'control_qty':prior['qty'] if prior else None,
                'combined_exceeds_atr':tail['fraction']>atr_fraction,
                'peer_alone_causes_exceedance':tail['fraction']>atr_fraction>=tail['symbol_tail_mean']})
        result['peer_tail_activation'][w] = {'trades':len(tail_rows), 'combined_exceeds_atr':sum(x['combined_exceeds_atr'] for x in tail_rows),
            'peer_alone_causes_exceedance':sum(x['peer_alone_causes_exceedance'] for x in tail_rows), 'rows':tail_rows,
            'interpretation':'ATR component normalized by actual entry price, before costs; diagnostic threshold comparison, not an allocator feasibility rerun. Actual quantity is authoritative, not a top-up clipped flag.'}
    bind(Path(__file__))
    result['source_sha256'] = SOURCES
    (OUT / 'support_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:{w:{x:y for x,y in v.items() if x not in ['rows','excluded_trade_keys']} for w,v in d.items()} for k,d in result.items() if k!='source_sha256'},indent=2))


if __name__=='__main__':
    main()
