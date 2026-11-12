"""Read-only support for an unchanged entry predicate on held positions.

No future returns are read or compared. Snapshot absence is never a signal.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

REPO = Path(__file__).resolve().parents[1]
BASE = Path('E:/CodexResearch/k_stock_trader/olr_structural_execution_20261013')
PARENT = BASE / 'generation_v12'
OUT = Path('E:/CodexResearch/k_stock_trader/olr_continuing_research_20261016/daily_thesis_support_v1')


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, data):
    with path.open('x', encoding='utf8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def run():
    OUT.mkdir(exist_ok=False, parents=True)
    study_path = PARENT / 'runtime/train/native_control/study.json'
    study = read(study_path)
    files = Path(study['code'])
    manifest_path = Path(study['manifest'])
    manifest = read(manifest_path)['files']
    case = PARENT / 'cases/train/native_control'
    census_path = BASE / 'generation_v11/existing_opening_book_evidence_v1.json'
    census = read(census_path)
    daily_rel = 'data/backtests/integrity/corrective-selection-20261005-v20/source/daily_references.json'
    rels = [daily_rel, daily_rel.replace('daily_references', 'daily_contract'),
            study['olr_snapshots'], study['database'], 'strategy_olr/research.py']
    dependencies = {}
    for rel in rels:
        path = files / rel
        dependencies[str(path)] = sha(path)
        assert dependencies[str(path)] == manifest[rel], rel
    for path in (Path(__file__), study_path, manifest_path, census_path,
                 REPO / 'docs/optimisation/olr/experiments/olr-exp-0030.md', case/'completion.json'):
        dependencies[str(path)] = sha(path)
    completion = read(case/'completion.json')
    for name in ('marked_book_observations.json', 'daily.json', 'minute_nav_evidence.json'):
        path = case/name
        dependencies[str(path)] = sha(path)
        assert dependencies[str(path)] == completion['artifacts'][name]
    # Reuse the verified census only after authenticating its own dependencies
    # and economic NAV/daily parity to the later valid N0 account.
    for path, expected in census['dependencies'].items():
        assert sha(Path(path)) == expected, path
        dependencies[path] = expected
    for name in ('daily.json', 'minute_nav_evidence.json'):
        assert sha(case/name) == sha(BASE/'generation_v11/cases/train/native_control'/name)
    save(OUT/'protocol.json', dict(at=datetime.now(timezone.utc).isoformat(),
        experiment='olr-exp-0030', stage='daily_structural_support_only',
        inputs=dependencies, predicate='prior_close < 0.96 * SMA60(prior_completed_daily_closes)',
        decision='first actual completed own 5m observation of held session',
        conditional_future_returns_read=False, native_replays=0,
        information='prior dated daily rows only; no afternoon rank/absence'))
    raw = read(files/daily_rel)
    daily = defaultdict(list)
    for row in raw:
        daily[row['symbol']].append(row)
    for symbol, rows in daily.items():
        rows.sort(key=lambda row: row['date'])
        assert len(rows) == len({r['date'] for r in rows}), symbol
    dates = {s: [r['date'] for r in rows] for s, rows in daily.items()}
    market_days = sorted({r['date'] for r in raw})

    def feature(symbol, day):
        stop = bisect_left(dates.get(symbol, []), day)
        rows = daily[symbol][max(0, stop-60):stop]
        previous_index = bisect_left(market_days, day)-1
        expected = market_days[previous_index] if previous_index >= 0 else None
        if len(rows) < 60 or rows[-1]['date'] != expected:
            return dict(available=False, n=len(rows), last_date=rows[-1]['date'] if rows else None,
                        expected_last_date=expected)
        sma = sum(r['close'] for r in rows)/60
        close = rows[-1]['close']
        assert all(r['date'] < day for r in rows) and close > 0 and sma > 0
        return dict(available=True, last_date=rows[-1]['date'], prior_close=close,
                    sma60=sma, threshold=0.96*sma, broken=close < 0.96*sma)

    snapshots = {s['trade_date']: s for s in read(files/study['olr_snapshots'])}
    reconciled, contradictions, missing = 0, [], []
    # Audit the exact pure-daily predicate against saved source evidence.
    # Other rejection reasons may short-circuit scoring; no negative inference.
    for session in census['sessions']:
        day = session['day']
        snap = snapshots.get(day)
        if snap is None:
            missing.append(day)
            continue
        for candidate in snap['candidates']:
            components = candidate['metadata'].get('frontier_score_components', {})
            if 'trend_broken' not in components:
                continue
            f = feature(candidate['symbol'], day)
            if f['available']:
                observed = bool(components['trend_broken'])
                reconciled += 1
                if f['broken'] != observed:
                    contradictions.append(dict(day=day, symbol=candidate['symbol'], feature=f, observed=observed))
        for symbol, reasons in snap['metadata'].get('rejected_symbols', {}).items():
            if 'trend_broken' in reasons:
                f = feature(symbol, day)
                if f['available']:
                    reconciled += 1
                    if not f['broken']:
                        contradictions.append(dict(day=day, symbol=symbol, feature=f, observed=True))
    save(OUT/'feature_reconciliation.json', dict(checked=reconciled, contradictions=contradictions,
                                               missing_snapshot_days=missing))
    assert not contradictions, 'Daily classification mismatch; no dependent support conclusion'
    books = {b['observed_at']: b for b in read(case/'marked_book_observations.json')}
    con = sqlite3.connect((files/study['database']).as_uri()+'?mode=ro', uri=True)
    con.execute('PRAGMA query_only=ON')
    rows = []
    for session in census['sessions']:
        day = session['day']
        next_day = (datetime.fromisoformat(day)+timedelta(days=1)).date().isoformat()
        for symbol, holding in session['open']['holdings'].items():
            f = feature(symbol, day)
            source = con.execute('SELECT ts,c,v FROM bars WHERE symbol=? AND ts>=? AND ts<? ORDER BY ts LIMIT 1',
                                 (symbol, day, next_day)).fetchone()
            row = dict(day=day, symbol=symbol, opening_quantity=holding['quantity'],
                       opening_notional=holding['notional'], daily=f)
            if source:
                at = (datetime.fromisoformat(source[0])+timedelta(minutes=5)).isoformat()
                book = books.get(at)
                position = next((p for p in book['positions'] if p['symbol']==symbol), None) if book else None
                row.update(decision_available_at=at, own_bar_label=source[0], own_close=source[1],
                           own_observed_volume=source[2], position_at_decision=position,
                           book_available=book is not None)
                row['supported'] = bool(f['available'] and f['broken'] and position
                                        and position['quantity'] > 0 and not position.get('pending_exit'))
            else:
                row.update(book_available=False, supported=False)
            rows.append(row)
    con.close()
    assert len(rows) == 233 and len({r['day'] for r in rows}) == 98
    save(OUT/'support_rows.json', rows)
    broken = [r for r in rows if r['daily'].get('broken')]
    supported = [r for r in rows if r['supported']]
    summary = dict(at=datetime.now(timezone.utc).isoformat(), held_sessions=len(rows),
        distinct_days=len({r['day'] for r in rows}), feature_checks=reconciled,
        missing_daily=sum(not r['daily']['available'] for r in rows),
        broken_hold_sessions=len(broken), broken_dates=len({r['day'] for r in broken}),
        supported_hold_sessions=len(supported), supported_dates=len({r['day'] for r in supported}),
        supported_opening_notional=sum(r['opening_notional'] for r in supported),
        supported_symbols=dict(Counter(r['symbol'] for r in supported)),
        conditional_future_returns_read=False, economic_improvement=False)
    save(OUT/'assessment.json', summary)
    assert all(sha(Path(p)) == h for p, h in dependencies.items())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    run()
