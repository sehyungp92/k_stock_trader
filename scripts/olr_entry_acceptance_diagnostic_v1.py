"""Read-only OLR entry/holding diagnostic; never a counterfactual portfolio."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from statistics import mean, median

PARENT = Path('E:/CodexResearch/k_stock_trader/olr_structural_execution_20261013/generation_v12')
ROOT = Path('E:/CodexResearch/k_stock_trader/olr_continuing_research_20261016')
REPO = Path(__file__).resolve().parents[1]


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(p, value):
    with Path(p).open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def dt(s):
    return datetime.fromisoformat(s)


def vwap(bars):
    volume = sum(max(b['v'], 0) for b in bars)
    return sum((b['h'] + b['l'] + b['c']) / 3 * max(b['v'], 0) for b in bars) / volume if volume else bars[-1]['c']


def classify(prefix, prior_high):
    """Only an already observed prefix can be passed to this function."""
    assert len(prefix) >= 2
    last, previous = prefix[-1], prefix[-2]
    level = max(prior_high, max(b['h'] for b in prefix[:-1]))
    if previous['c'] <= level < last['c']:
        return 'fresh_breakout', level
    if previous['c'] <= vwap(prefix[:-1]) and last['c'] > vwap(prefix):
        return 'fresh_reclaim', vwap(prefix)
    return 'generic_acceptance', vwap(prefix)


def first_loss(bars, level, first_fill, final_exit):
    prior = None
    for b in bars:
        available = dt(b['ts']) + timedelta(minutes=5)
        # A source interval must begin after acquisition. No pre-fill low/close.
        if dt(b['ts']) < first_fill or available >= final_exit:
            continue
        if (prior is not None and dt(b['ts']) - dt(prior['ts']) == timedelta(minutes=5)
                and prior['c'] < level and b['c'] < level):
            return b, available
        prior = b
    return None


def summary(rows):
    if not rows:
        return {'n': 0}
    by_date = defaultdict(list)
    for r in rows:
        by_date[r['day']].append(r['net_bps'])
    amounts = [r['net_pnl'] for r in rows]
    return dict(n=len(rows), dates=len(by_date), mean_net_bps=mean(r['net_bps'] for r in rows),
                median_net_bps=median(r['net_bps'] for r in rows),
                date_equal_mean_bps=mean(mean(x) for x in by_date.values()),
                net_pnl=sum(amounts), excluding_best_pnl=sum(amounts)-max(amounts),
                winning_wealth=sum(max(x, 0) for x in amounts), losing_wealth=-sum(min(x, 0) for x in amounts),
                meaningful=sum(r['notional'] >= 1_000_000 for r in rows))


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    case = PARENT / 'cases/train/native_control'
    study_path = PARENT / 'runtime/train/native_control/study.json'
    study = read(study_path)
    files = Path(study['code'])
    manifest = read(study['manifest'])
    completion = read(case / 'completion.json')
    dependencies = [study_path, Path(study['manifest']), case / 'completion.json', Path(__file__),
                    REPO / 'docs/optimisation/olr/experiments/olr-exp-0031.md']
    for name in ('trades.json', 'minute_events.jsonl.gz', 'result.json'):
        assert completion['artifacts'][name] == sha(case / name), name
        dependencies.append(case / name)
    for name in (study['database'], study['olr_snapshots'], study['olr_request']):
        assert manifest['files'][name] == sha(files / name), name
        dependencies.append(files / name)
    protocol = dict(at=datetime.now(timezone.utc).isoformat(), experiment='olr-exp-0031',
                    source=study['source_generation_hash'], inputs={str(p): sha(p) for p in dependencies},
                    economic_claim=False, new_portfolio_replays=0, source_access='read_only',
                    chronological_thirds=['2025-08-07..2025-10-31', '2025-11-01..2026-01-15', '2026-01-16..2026-03-31'])
    write(out / 'protocol.json', protocol)
    print(json.dumps({'phase': 'registered', 'protocol': str(out / 'protocol.json')}), flush=True)
    assert read(case / 'result.json')['execution_validity_pass']
    trades = read(case / 'trades.json')
    snapshots = {s['trade_date']: s for s in read(files / study['olr_snapshots'])}
    database = files / study['database']
    con = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
    con.execute('PRAGMA query_only=ON')
    rows = []
    for t in trades:
        symbol = t['symbol']; signal = dt(t['entry_decision_time']); day = signal.date().isoformat()
        available = dt(t['route_metadata']['decision_available_at'])
        assert available == signal + timedelta(minutes=5)
        candidate = next(x for x in snapshots[day]['candidates'] if x['symbol'] == symbol)
        end_day = t['exit_fill_time'][:10]
        data = con.execute('SELECT ts,o,h,l,c,v FROM bars WHERE symbol=? AND ts>=? AND ts<? ORDER BY ts',
                           (symbol, day, (dt(end_day) + timedelta(days=1)).date().isoformat())).fetchall()
        bars = [dict(zip(('ts','o','h','l','c','v'), b)) for b in data]
        assert len({b['ts'] for b in bars}) == len(bars)
        prefix = [b for b in bars if dt(b['ts']).date() == signal.date() and dt(b['ts']) <= signal]
        assert prefix and dt(prefix[-1]['ts']) == signal
        b = prefix[-1]; plan = t['route_metadata']['trade_entry_plan']; anchor = vwap(prefix)
        assert math.isclose(b['c'], t['route_metadata']['entry_submission_close'])
        assert b['c']/b['o']-1 >= plan['min_bar_ret']-1e-12
        assert b['c']/anchor-1 >= plan['min_vwap_ret']-1e-12
        assert b['c']/anchor-1 <= plan['max_vwap_extension_pct']+1e-12
        assert (b['c']-b['l'])/max(b['h']-b['l'],1e-9) >= plan['min_close_location']-1e-12
        kind, level = classify(prefix, candidate['prior_day_high'])
        first, final = dt(t['entry_fill_time']), dt(t['exit_fill_time'])
        loss = first_loss(bars, anchor, first, final)
        row = dict(symbol=symbol, day=day, class_=kind, level=level, entry_vwap=anchor,
                   signal_label=signal.isoformat(), available_at=available.isoformat(),
                   notional=t['qty']*t['entry_price'], net_pnl=t['net_pnl'],
                   net_bps=t['net_pnl']/(t['qty']*t['entry_price'])*10000,
                   sector=t['route_metadata']['sector'], qty=t['qty'], final_exit=t['exit_fill_time'],
                   first_fill=t['entry_fill_time'],
                   entry_order_id=None,
                   third=0 if day <= '2025-10-31' else 1 if day <= '2026-01-15' else 2)
        if loss:
            bar, at = loss
            later = [x for x in bars if at < dt(x['ts']) + timedelta(minutes=5) <= final]
            row['loss'] = dict(available_at=at.isoformat(), label=bar['ts'], close=bar['c'],
                               phase='entry_session' if at.date()==first.date() else 'overnight',
                               recovered_before_native_exit=any(x['c']>=anchor for x in later),
                               later_native_buys=0., later_native_sales=0., residual_qty=0)
        rows.append(row)
    con.close()
    index = {(r['symbol'], r['day']): r for r in rows}
    assert len(index) == len(trades) == 188
    quantities = defaultdict(int); orders = {}; events = []
    # Decode once and keep only economically relevant records.
    with gzip.open(case / 'minute_events.jsonl.gz', 'rt', encoding='utf8') as f:
        for line in f:
            e = json.loads(line)
            if e['event'] in ('submit','amend','cancel','expire','reject','fill'):
                events.append(e)
    buy_to_key = {}
    for e in events:
        if e['event'] == 'submit' and e['order']['side'] == 'BUY':
            o = e['order']; key = (o['symbol'], str(o['metadata']['decision_available_at'])[:10])
            if key in index:
                buy_to_key[o['order_id']] = key
                index[key]['entry_order_id'] = o['order_id']
    for r in rows:
        assert r['entry_order_id']
        if 'loss' not in r:
            continue
        loss = r['loss']; boundary = dt(loss['available_at']); final = dt(r['final_exit'])
        q = 0; active = {}; purchases = sales = 0.
        for e in events:
            o = e['order']
            if o['symbol'] != r['symbol']:
                continue
            kind = e['event']
            if kind == 'fill':
                f = e['fill']; at = dt(f['timestamp'])
                if at < dt(r['first_fill']) or at > final:
                    continue
                sign = 1 if f['side']=='BUY' else -1
                if at <= boundary:
                    q += sign*f['qty']
                    remainder = o['qty']-f['qty']
                    if remainder:
                        active[o['order_id']] = {**o,'qty':remainder}
                    else:
                        active.pop(o['order_id'], None)
                else:
                    amount = f['qty']*f['price']
                    if sign == 1:
                        purchases += amount + f['metadata']['fee']
                    else:
                        sales += amount - f['metadata']['fee']
            else:
                stamp = (o['metadata'].get('decision_available_at') if kind in ('submit','amend') else e.get('timestamp')) or o['submitted_at']
                at = dt(stamp)
                if at < dt(r['available_at']) or at > boundary:
                    continue
                if kind in ('submit','amend'):
                    active[o['order_id']] = o
                else:
                    active.pop(o['order_id'], None)
        loss.update(residual_qty=q, marked_notional=q*loss['close'],
                    pending=[dict(id=x['order_id'],side=x['side'],qty=x['qty'],reason=x['reason']) for x in active.values()],
                    later_native_buys=purchases,later_native_sales=sales,
                    continuation_wealth=sales-purchases-q*loss['close'])
        loss['preempted'] = any(x['side']=='SELL' and x['reason'] not in ('next_close_exit',) for x in active.values())
        assert q >= 0
    groups = {}
    for kind in ('fresh_breakout','fresh_reclaim','generic_acceptance'):
        selected = [r for r in rows if r['class_']==kind]
        groups[kind] = dict(all=summary(selected), thirds=[summary([r for r in selected if r['third']==i]) for i in range(3)])
    loss_groups = {}
    for phase in ('entry_session','overnight'):
        selected = [r for r in rows if r.get('loss',{}).get('phase')==phase and r['loss']['residual_qty']>0]
        free = [r for r in selected if not r['loss']['preempted']]
        vals = [r['loss']['continuation_wealth'] for r in free]
        loss_groups[phase] = dict(n=len(selected), actionable_without_native_preemption=len(free),
            pending_buys=sum(any(o['side']=='BUY' for o in r['loss']['pending']) for r in free),
            recovered=sum(r['loss']['recovered_before_native_exit'] for r in free),
            native_continuation_wealth=sum(vals), positive_continuations=sum(x>0 for x in vals),
            thirds=[dict(n=sum(r['third']==i for r in free),continuation_wealth=sum(r['loss']['continuation_wealth'] for r in free if r['third']==i)) for i in range(3)],
            excluding_largest_absolute=sum(vals)-max(vals,key=abs) if vals else 0)
    write(out/'rows.json', rows)
    assessment = dict(at=datetime.now(timezone.utc).isoformat(), lifecycle_count=len(rows),
                      reconciled_net_pnl=sum(r['net_pnl'] for r in rows), entry_groups=groups,
                      acceptance_loss=loss_groups, rows_sha256=sha(out/'rows.json'),
                      limitations=['Descriptive realised incumbent paths, not causal treatment effects.',
                                   'Continuation wealth uses current coarse mark, not executable hypothetical liquidation.',
                                   'Later acquisition cash flows included; a management policy must handle pending orders.',
                                   'All data reused for development. No independent validation or economic improvement.'])
    assert math.isclose(assessment['reconciled_net_pnl'], read(case/'result.json')['metrics']['net_profit'], abs_tol=1e-5)
    write(out/'assessment.json', assessment)
    print(json.dumps(assessment, indent=2), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'entry_acceptance_v1')
    run(parser.parse_args().output)
