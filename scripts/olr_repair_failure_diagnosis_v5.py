"""Read-only diagnostic screens; never a replacement for native qualification.

Uses unchanged exit thresholds on authentic one-minute observations after entry
day. Existing entries, quantities and exit VWAPs remain fixed. No parameter
search, profile mutation, portfolio counterfactual, or new qualified claim.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('E:/CodexResearch/k_stock_trader/olr_round1_optimization_20261010')
OUT = REPO / 'data/backtests/analysis/olr_repair_failure_20261011'
SOURCES = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def bind(path):
    SOURCES[str(path)] = sha(path)
    return path


def read(path):
    return json.loads(bind(Path(path)).read_text(encoding='utf-8'))


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    parent = REPO / 'data/backtests/analysis/olr_repair_20261011'
    assert sha(parent / 'results.json') == '85d1a4cd53a1b7b3466093bc9eb4f235fd0b838a9b5811ba8d9be79cefd9cc83'
    assert sha(parent / 'selected_research_policy.json') == '49a88f0b5c015cc1e5c262a38641c80402977d8f2bf7f1957125754a7de778eb'
    read(parent / 'results.json')
    read(parent / 'selected_research_policy.json')
    prior = read(REPO / 'data/backtests/analysis/olr_remaining_drawdown_20261010/paths_and_screens.json')
    output = {'purpose': __doc__, 'limitations': [
        'Observation/isolated-price screen only; not a native portfolio backtest or a return/DD estimate.',
        'Keeps actual entries and total quantity fixed; compares early sales to original aggregate exit VWAP.',
        'Allows one percent of immediately preceding contiguous minute volume, at next observed open less 5bp.',
        'Does not fill across missing-minute gaps using stale volume; does not identify the cause of missing minutes.',
        'Ignores queue competition, tax differences, future cash/risk redeployment, cancellations and residual execution changes.',
        'Starts after entry day with actual entry-day high-water state; only sells before first native sale.',
        'All completed-minute high/close information becomes actionable at minute end, never its opening timestamp.',
        'No unseen validation: all IS/OOS results and thresholds are already part of the research record.',
    ], 'windows': {}}
    for w in ['train', 'oos']:
        folder = ROOT / 'repair_v4/cases/c4f49be01ed09b8dc' / w / '100000000_baseline'
        completion = read(folder / 'completion.json')
        for name in ['trades.json', 'execution_audit.json']:
            assert sha(folder / name) == completion['artifacts'][name]
        assert read(folder / 'execution_audit.json')['audit_pass']
        trades = read(folder / 'trades.json')
        study = read(ROOT / 'repair_v4/runtime/c4f49be01ed09b8dc' / w / 'baseline/study.json')
        db = bind(Path(study['code']) / study['database'])
        con = sqlite3.connect(db.as_uri() + '?mode=ro', uri=True)
        old = {(r['symbol'], r['entry']): r for r in prior[w]['trades']}
        data = {}; reconciled = set(); comparisons = 0; mismatches = []; dupes = 0
        rows = []
        for t in trades:
            key = (t['symbol'], t['entry_fill_time'])
            p = old[key]
            assert p['qty'] == t['qty'] and abs(p['net_pnl'] - t['net_pnl']) < 1e-6
            s = t['symbol']
            if s not in data:
                path = bind(REPO / f'data/kis_intraday_parquet/{s}/{s}_1m_20250710_20260710.parquet')
                df = pd.read_parquet(path).set_index('timestamp').sort_index()
                dupes += int(df.index.duplicated().sum())
                assert not df.index.duplicated().any()
                assert ((df.low <= df[['open', 'close']].min(axis=1)) & (df.high >= df[['open', 'close']].max(axis=1)) & (df.volume >= 0)).all()
                data[s] = df
            df = data[s]
            start = pd.Timestamp(t['entry_fill_time'])
            end = pd.Timestamp(t['exit_fill_time'])
            held = df[(df.index.date >= start.date()) & (df.index.date <= end.date())]
            days = sorted(set(held.index.date))
            for day in days:
                if (s, day) in reconciled:
                    continue
                reconciled.add((s, day))
                one = df[df.index.date == day]
                five = one.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
                native = pd.DataFrame(con.execute('select ts,o,h,l,c,v from bars where symbol=? and day=? order by ts', (s, str(day))).fetchall(), columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                native['timestamp'] = pd.to_datetime(native['timestamp'])
                native = native.set_index('timestamp')
                union = five.index.union(native.index)
                a = five.reindex(union); b = native.reindex(union)
                comparisons += len(union)
                bad = ((a-b).abs() > 1e-7).any(axis=1) | a.isna().any(axis=1) | b.isna().any(axis=1)
                for stamp in union[bad]:
                    clean = lambda row: {k: (None if pd.isna(v) else v) for k, v in row.items()}
                    mismatches.append({'symbol': s, 'timestamp': str(stamp), 'one_resampled': clean(a.loc[stamp].to_dict()), 'native': clean(b.loc[stamp].to_dict())})
            first_sale = min(pd.Timestamp(x['bar']['timestamp']) for x in p['sells'])
            future = held[(held.index.date > start.date()) & (held.index < first_sale)]
            high = p['entry_day_last']['mfe_price'] if p['entry_day_last'] else t['entry_price']
            entry = t['entry_price']; risk = p['risk']
            plan = t['route_metadata']['trade_exit_plan']
            assert plan['failed_followthrough_bars'] == 1 and plan['failed_followthrough_close_r'] == -1
            assert plan['failed_followthrough_mfe_r'] == 4 and plan['mfe_fade_start_r'] == 1 and plan['mfe_fade_gap_r'] == 1.25
            trigger = None; remaining = t['qty']; fills = []; prev = None
            for stamp, bar in future.iterrows():
                # A decision from the previous completed minute may execute now.
                if trigger and prev is not None and stamp - prev[0] == pd.Timedelta(minutes=1):
                    qty = min(remaining, int(float(prev[1]['volume']) * .01))
                    if qty:
                        price = float(bar['open']) * (1 - .0005)
                        fills.append({'time': str(stamp), 'qty': qty, 'price': price, 'lag_volume': int(prev[1]['volume'])})
                        remaining -= qty
                        if remaining == 0:
                            break
                high = max(high, float(bar['high']))
                cr = (float(bar['close']) - entry) / risk
                mr = (high - entry) / risk
                reason = 'failed_followthrough' if mr < 4 and cr <= -1 else 'mfe_fade' if mr >= 1 and cr <= max(0, mr - 1.25) else ''
                if trigger is None and reason:
                    trigger = {'bar_start': str(stamp), 'decision_available': str(stamp + pd.Timedelta(minutes=1)), 'reason': reason, 'close_r': cr, 'mfe_r': mr}
                prev = (stamp, bar)
            fill_qty = sum(f['qty'] for f in fills)
            delta = sum(f['qty'] * (f['price'] - t['exit_price']) for f in fills)
            rows.append({'symbol': s, 'entry': t['entry_fill_time'], 'native_first_sale': str(first_sale), 'native_exit_vwap': t['exit_price'],
                         'qty': t['qty'], 'native_net_pnl': t['net_pnl'], 'trigger': trigger, 'early_qty': fill_qty,
                         'isolated_gross_exit_value_change': delta, 'fills': fills})
        con.close()
        groups = {}
        for name, subset in [('all', rows), ('native_winners', [x for x in rows if x['native_net_pnl'] > 0]), ('native_losers', [x for x in rows if x['native_net_pnl'] <= 0])]:
            groups[name] = {'trades': len(subset), 'earlier_trigger': sum(x['trigger'] is not None for x in subset), 'earlier_fill': sum(x['early_qty'] > 0 for x in subset),
                            'positive_changes': sum(x['isolated_gross_exit_value_change'] > 1 for x in subset), 'negative_changes': sum(x['isolated_gross_exit_value_change'] < -1 for x in subset),
                            'gross_exit_value_change': sum(x['isolated_gross_exit_value_change'] for x in subset)}
        output['windows'][w] = {'trade_count': len(rows), 'symbols': len(data), 'held_symbol_sessions': len(reconciled), 'reconciled_5m_bars': comparisons,
                                'mismatch_count': len(mismatches), 'mismatch_examples': mismatches, 'duplicate_1m_timestamps': dupes, 'groups': groups, 'rows': rows}
        print(w, json.dumps({k:v for k,v in output['windows'][w].items() if k not in ['rows', 'mismatch_examples']}), flush=True)
    bind(Path(__file__))
    output['source_sha256'] = SOURCES
    (OUT / 'one_minute_feasibility.json').write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    run()
