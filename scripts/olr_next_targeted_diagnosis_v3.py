"""Read-only forensic accounting of the completed OLR opening-risk round.

This does not run a strategy, change a profile, or estimate a counterfactual
portfolio. Writes are restricted to a new analysis directory.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('E:/CodexResearch/k_stock_trader/olr_round1_optimization_20261010')
OUT = REPO / 'data/backtests/analysis/olr_next_targeted_20261011'
SOURCES = {}


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    p = Path(p)
    SOURCES[str(p)] = digest(p)
    return json.loads(p.read_text(encoding='utf-8'))


def records(p):
    p = Path(p)
    SOURCES[str(p)] = digest(p)
    with gzip.open(p, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def authenticate(p):
    comp = read(p / 'completion.json')
    assert comp['complete']
    for name, expected in comp['artifacts'].items():
        assert digest(p / name) == expected, (p, name)
    assert read(p / 'execution_audit.json')['audit_pass']
    return {'completion_sha256': digest(p / 'completion.json'),
            'authenticated_artifacts': len(comp['artifacts'])}


def summary(rows):
    return {'count': len(rows), 'winners': sum(r['net_pnl'] > 0 for r in rows),
            'pnl': sum(r['net_pnl'] for r in rows)}


def window(w):
    folder = ROOT / 'remaining_drawdown_v2' / w
    proof = authenticate(folder)
    trades = read(folder / 'trades.json')
    study = read(ROOT / f'opening_risk_v2/runtime/c6718aff2b6b3bbc1/{w}/baseline/study.json')
    period = study['configuration']['date_range']
    snapshots = [r for r in read(Path(study['code']) / study['olr_snapshots'])
                 if period['start'] <= r['trade_date'] <= period['end']]
    entries = {(r['entry_fill_time'][:10], r['symbol']): r for r in trades}
    assert len(entries) == len(trades), 'Lifecycle key must be unique'
    daily = []
    morning, morning_only, afternoon, afternoon_only = (Counter() for _ in range(4))
    nonentries = []
    for s in snapshots:
        m = s['metadata']; cs = s['candidates']; d = s['trade_date']
        mr = m.get('rejected_symbols', {}); ar = m.get('afternoon_rejected_symbols', {})
        for counter, single, mapping in [(morning, morning_only, mr), (afternoon, afternoon_only, ar)]:
            for reasons in mapping.values():
                counter.update(set(reasons))
                if len(set(reasons)) == 1:
                    single.update(set(reasons))
        pool = m['candidate_pool_count']
        # Metadata pool includes any daily top-20 overflow; retain separately.
        daily.append({'date': d, 'daily_pool': pool, 'daily_overflow': m.get('overflow_symbol_count', 0),
                      'afternoon_pass_before_top8': m['afternoon_candidate_pool_count'],
                      'afternoon_rejected': len(ar), 'final_candidates': len(cs),
                      'first_six': len(cs[:6]), 'entries': sum(day == d for day, _ in entries),
                      'empty_stage': ('daily' if pool == 0 else 'afternoon') if not cs else None,
                      'afternoon_rejection_reasons': dict(Counter(x for rs in ar.values() for x in set(rs)))})
        for c in cs:
            if (d, c['symbol']) not in entries:
                nonentries.append({'date': d, 'symbol': c['symbol'], 'rank': c['rank'],
                                   'score': c['selection_score'] if 'selection_score' in c else c.get('score'),
                                   'band': c['metadata'].get('afternoon_score_band_rule')})
    assert sum(r['entries'] for r in daily) == len(trades)
    submits = {}; account_states = {}
    for r in records(folder / 'native_journal.jsonl.gz'):
        if r['event'] == 'submit' and r['order']['side'] == 'BUY':
            o = r['order']; submits[(o['submitted_at'][:10], o['symbol'])] = o
        if r['event'] == 'strategy_state':
            account_states[r['timestamp']] = r
    for r in nonentries:
        o = submits.get((r['date'], r['symbol']))
        r['buy_submitted'] = bool(o)
        r['submitted_qty'] = o['qty'] if o else None
    # Load only the frozen generation. Both windows' entry code is identical.
    # This reconstructs signals; it does not assume executable fills or returns.
    code = Path(study['code'])
    sys.path.insert(0, str(code))
    from dataclasses import replace
    from strategy_olr import execution
    from strategy_olr.models import OLRDailyCandidate
    from strategy_common.market import MarketBar
    assert digest(execution.__file__) == digest(code / 'strategy_olr/execution.py')
    SOURCES[str(code / 'strategy_olr/execution.py')] = digest(execution.__file__)
    db = code / study['database']
    con = sqlite3.connect(db.as_uri() + '?mode=ro', uri=True)
    plan = execution.OLREntryPlan(**trades[0]['route_metadata']['trade_entry_plan'])
    signal_counts = Counter(); signal_rows = []
    for s in snapshots:
        for c in s['candidates'][:6]:
            candidate = OLRDailyCandidate.from_json_dict(c)
            bars = [MarketBar(symbol=c['symbol'], timestamp=datetime.fromisoformat(ts), timeframe='5m',
                              open=o, high=h, low=l, close=close, volume=v, is_completed=True)
                    for ts, o, h, l, close, v in con.execute(
                        'select ts,o,h,l,c,v from bars where symbol=? and day=? order by ts',
                        (c['symbol'], s['trade_date']))]
            sig = execution.find_olr_entry_signal(bars, candidate, plan, require_fill_bar=False)
            ent = entries.get((s['trade_date'], c['symbol']))
            status = 'entered' if ent else 'signal_but_not_submitted' if sig else 'no_six_bar_confirmation'
            if ent:
                assert sig is not None
                assert bars[sig.signal_index].timestamp == datetime.fromisoformat(ent['entry_decision_time']), (w, c['symbol'])
            later = None
            if not sig:
                later = execution.find_olr_entry_signal(bars, candidate, replace(plan, max_signal_bars=9), require_fill_bar=False)
            signal_counts[status] += 1
            signal_counts['additional_nine_bar_signals'] += later is not None
            state_at_signal = account_states.get(bars[sig.signal_index].timestamp.isoformat(), {}) if sig else {}
            signal_rows.append({'date': s['trade_date'], 'symbol': c['symbol'], 'rank': c['rank'], 'status': status,
                                'signal_time': bars[sig.signal_index].timestamp.isoformat() if sig else None,
                                'already_held_at_signal': c['symbol'] in state_at_signal.get('positions', {}),
                                'nine_bar_signal_time': bars[later.signal_index].timestamp.isoformat() if later else None})
    con.close()
    rows = []
    for t in trades:
        m = t['route_metadata']
        base = m.get('allocation_before_redistribution', t['qty'])
        extra = m.get('allocation_redistributed_qty', 0)
        # Base-first allocation of aggregated fills is a descriptive proxy only.
        # Same average entry/exit prices are applied to both slices. Queue, price,
        # capacity and endogenous reinvestment effects need a separate native run.
        actual_extra_proxy = min(extra, max(0, t['qty'] - base))
        rows.append({'symbol': t['symbol'], 'entry': t['entry_fill_time'], 'qty': t['qty'],
                     'net_pnl': t['net_pnl'], 'rank': m['candidate_rank'], 'sector': m.get('sector'),
                     'band': m.get('afternoon_score_band_rule'), 'base_approved_qty': base,
                     'extra_approved_qty': extra, 'filled_extra_base_first_proxy': actual_extra_proxy,
                     'extra_pnl_same_price_proxy': actual_extra_proxy / t['qty'] * t['net_pnl'],
                     'allocation_limit': m.get('allocation_binding_limit'),
                     'risk_pct': 100 * m['risk_per_share'] / t['entry_price'],
                     'entry_price': t['entry_price'], 'scenario': m.get('scenario_loss_budget')})
    f = {'snapshot_days': len(daily), 'empty_days': sum(x['final_candidates'] == 0 for x in daily),
         'empty_daily_stage': sum(x['empty_stage'] == 'daily' for x in daily),
         'empty_afternoon_stage': sum(x['empty_stage'] == 'afternoon' for x in daily),
         'daily_pool_total': sum(x['daily_pool'] for x in daily),
         'daily_overflow_total': sum(x['daily_overflow'] for x in daily),
         'afternoon_rejected_total': sum(x['afternoon_rejected'] for x in daily),
         'final_candidates': sum(x['final_candidates'] for x in daily),
         'first_six_candidates': sum(x['first_six'] for x in daily),
         'trades': len(trades), 'entry_days': len({x['entry_fill_time'][:10] for x in trades}),
         'daily_reasons_multi_count': dict(morning.most_common()),
         'daily_sole_reasons': dict(morning_only.most_common()),
         'afternoon_reasons_multi_count': dict(afternoon.most_common()),
         'afternoon_sole_reasons': dict(afternoon_only.most_common()),
         'nonentries_first6': sum(r['rank'] <= 6 for r in nonentries),
         'nonentries_first6_with_buy_submitted': sum(r['rank'] <= 6 and r['buy_submitted'] for r in nonentries),
         'nonentries_rank7plus': sum(r['rank'] > 6 for r in nonentries)}
    cohorts = {field: {str(v): summary([r for r in rows if r[field] == v])
                       for v in sorted({r[field] for r in rows}, key=str)} for field in ['rank', 'band', 'sector']}
    return {'authentication': proof, 'funnel': f, 'daily_funnel': daily,
            'signal_reconstruction': dict(signal_counts), 'signal_rows': signal_rows,
            'nonentries': nonentries, 'trade_rows': rows, 'cohorts_descriptive': cohorts,
            'topups': {'approved_count': sum(r['extra_approved_qty'] > 0 for r in rows),
                       'filled_base_first_proxy_count': sum(r['filled_extra_base_first_proxy'] > 0 for r in rows),
                       'same_price_proxy_pnl': sum(r['extra_pnl_same_price_proxy'] for r in rows),
                       'winner_proxy_pnl': sum(r['extra_pnl_same_price_proxy'] for r in rows if r['net_pnl'] > 0),
                       'loser_proxy_pnl': sum(r['extra_pnl_same_price_proxy'] for r in rows if r['net_pnl'] < 0)}}


def decompose(candidate, w):
    folder = ROOT / f'opening_risk_v2/cases/{candidate}/{w}/100000000_baseline'
    proof = authenticate(folder)
    control = read(ROOT / f'remaining_drawdown_v2/{w}/trades.json')
    changed = read(folder / 'trades.json')
    def key(r):
        return r['entry_fill_time'][:10], r['symbol']
    a, b = ({key(r): r for r in rs} for rs in [control, changed])
    assert len(a) == len(control) and len(b) == len(changed)
    details = []
    for k in sorted(a.keys() & b.keys()):
        old, new = a[k], b[k]
        # Exact Oaxaca-style identity, not causal isolation.
        unit = old['qty'] * (new['net_pnl']/new['qty'] - old['net_pnl']/old['qty'])
        size = (new['qty'] - old['qty']) * new['net_pnl']/new['qty']
        delta = new['net_pnl'] - old['net_pnl']
        assert abs(unit + size - delta) < 1e-6
        details.append({'date': k[0], 'symbol': k[1], 'old_qty': old['qty'], 'new_qty': new['qty'],
                        'old_pnl': old['net_pnl'], 'new_pnl': new['net_pnl'], 'delta': delta,
                        'per_unit_execution_effect': unit, 'quantity_effect': size})
    added = [b[k] for k in b.keys()-a.keys()]; removed = [a[k] for k in a.keys()-b.keys()]
    entry_delta = sum(x['net_pnl'] for x in added) - sum(x['net_pnl'] for x in removed)
    actual_delta = sum(x['net_pnl'] for x in changed) - sum(x['net_pnl'] for x in control)
    assert abs(sum(x['delta'] for x in details)+entry_delta-actual_delta) < 1e-6
    return {'authentication': proof, 'matched': len(details), 'added': len(added), 'removed': len(removed),
            'total_delta': actual_delta, 'per_unit_execution_effect': sum(x['per_unit_execution_effect'] for x in details),
            'quantity_effect': sum(x['quantity_effect'] for x in details), 'entry_set_effect': entry_delta,
            'winner_to_nonpositive': sum(x['old_pnl'] > 0 and x['new_pnl'] <= 0 for x in details),
            'old_winner_delta': sum(x['delta'] for x in details if x['old_pnl'] > 0),
            'old_loser_delta': sum(x['delta'] for x in details if x['old_pnl'] < 0),
            'details': sorted(details, key=lambda x: x['delta'])}


def structural_screen(w):
    """One descriptive falsification screen, no simulated exits or threshold search.

    Require the old weak-partial trigger plus two contiguous completed closes
    below the entry-session low and current-session cumulative typical-price VWAP.
    Search only before the incumbent native condition; all levels are causal.
    """
    folder = ROOT / 'remaining_drawdown_v2' / w
    previous = read(REPO / 'data/backtests/analysis/olr_remaining_drawdown_20261010/paths_and_screens.json')[w]
    candidates = { (r['symbol'], r['entry']): r for r in previous['screens']['next_day_weak_no_followthrough']['rows'] if r['strictly_earlier'] }
    study = read(ROOT / f'opening_risk_v2/runtime/c6718aff2b6b3bbc1/{w}/baseline/study.json')
    db = Path(study['code']) / study['database']
    con = sqlite3.connect(db.as_uri() + '?mode=ro', uri=True)
    import yaml
    halt_days = {str(r['date']) for r in yaml.safe_load((Path(study['code']) / 'kis_core/data/krx_intraday_halts.yaml').read_text(encoding='utf-8'))['halts']}
    paths = read(REPO / f'data/backtests/analysis/olr_remaining_drawdown_20261010/{w}_drawdown.json')['trade_execution_paths']
    observations = defaultdict(list)
    for r in records(folder / 'managed_state.jsonl.gz'):
        observations[(r['symbol'], r['entry_time'])].append(r)
    output = []
    for key, r in candidates.items():
        assert len({v['timestamp'] for v in observations[key]}) == len(observations[key])
        low = con.execute('select min(l) from bars where symbol=? and day=?', (r['symbol'], r['entry'][:10])).fetchone()[0]
        stamps = [v for v in observations[key] if v['timestamp'][:10] > r['entry'][:10]
                  and v['timestamp'] < r['native_condition_time']]
        prior_ts = None; streak = 0; count = Counter(); hit = None
        for v in stamps:
            day = v['timestamp'][:10]; count[day] += 1
            typical, volume = con.execute('select sum((h+l+c)/3*v),sum(v) from bars where symbol=? and day=? and ts<=?',
                                          (r['symbol'], day, v['timestamp'])).fetchone()
            if not volume:
                continue
            vwap = typical / volume
            weak = (v['close_r'] <= -.4 and v['mfe_r'] < .5 and v['bar']['close'] < low and v['bar']['close'] < vwap)
            now = datetime.fromisoformat(v['timestamp'])
            contiguous = prior_ts is not None and (now-prior_ts).total_seconds() == 300
            streak = (streak+1 if contiguous else 1) if weak else 0
            prior_ts = now
            if streak >= 2 and count[day] >= 3:
                hit = v['timestamp']; break
        item = {'symbol': r['symbol'], 'entry': r['entry'], 'net_pnl': r['net_pnl'],
                'old_trigger': r['hit_time'], 'structural_trigger': hit}
        if hit:
            path = next(p for p in paths if p['symbol'] == r['symbol'] and datetime.fromisoformat(p['buys'][0]['timestamp']) == datetime.fromisoformat(r['entry']))
            at = datetime.fromisoformat(hit); nxt = at + timedelta(minutes=5)
            qty = r['qty'] // 2
            capacity = int(v['bar']['volume'] * .01)
            nextbar = con.execute('select o from bars where symbol=? and ts=?', (r['symbol'], nxt.isoformat())).fetchone()
            reasons = []
            if any(datetime.fromisoformat(x['timestamp']) > at for x in path['buys']): reasons.append('entry_incomplete')
            if any(datetime.fromisoformat(x['timestamp']) <= at for x in path['sells']): reasons.append('already_sold')
            if hit[:10] in halt_days: reasons.append('halt_date')
            if not '09:05' <= nxt.strftime('%H:%M') <= '15:15': reasons.append('outside_continuous')
            if not nextbar: reasons.append('no_next_bar')
            if qty < 1: reasons.append('no_half_share')
            if capacity < qty: reasons.append('insufficient_one_bar_capacity')
            item.update(half_qty=qty, capacity=capacity, arithmetic_exclusions=reasons)
            if not reasons:
                avg_actual_net_sale = sum(x['qty'] * x['price'] - x['fee'] for x in path['sells']) / r['qty']
                item['isolated_half_exit_delta'] = qty * (float(nextbar[0]) * (1-.0005) * (1-.0022) - avg_actual_net_sale)
        output.append(item)
    con.close()
    return {'definition': structural_screen.__doc__, 'original_earlier_hits': summary(list(candidates.values())),
            'structural_hits': summary([r for r in output if r['structural_trigger']]),
            'supported_half_exit_arithmetic_count': sum('isolated_half_exit_delta' in r for r in output),
            'supported_half_exit_arithmetic_delta': sum(r.get('isolated_half_exit_delta', 0) for r in output),
            'supported_baseline_losers_worsened': sum(r.get('isolated_half_exit_delta', 0) < 0 for r in output),
            'arithmetic_caveat': 'Fixed original quantity and average sale price; 1% known contiguous prior-bar capacity, 5bp slippage and 22bp sell costs. No reinvestment, cancellation delay or compounding. Unsupported cases are unknown, not zero. Not native strategy return or DD.',
            'rows': output}


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    previous = REPO / 'data/backtests/analysis/olr_opening_risk_20261010'
    profile = read(previous / 'selected_research_policy.json')
    assert profile['results_sha256'] == digest(previous / 'results.json')
    result = {'purpose': __doc__, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'previous_profile_binding_verified': True,
              'warnings': ['No new strategy performance is claimed.',
                           'Cohort P&L and marginal top-up slices are descriptive, not causal estimates.',
                           'April-July is repeatedly inspected development data.',
                           'Rejection reasons may overlap; sole reasons are counted separately.',
                           'Sole recorded rejection is stage-local; later gates were not necessarily evaluated.'],
              'windows': {w: window(w) for w in ['train', 'oos']},
              'structural_screen': {w: structural_screen(w) for w in ['train', 'oos']},
              'decompositions': {name: {w: decompose(cid, w) for w in ['train', 'oos']}
                                  for name, cid in [('concentration55', 'ca0978ca0a31f0340'),
                                                    ('weak_partial', 'c2074991df76dbe41')]}}
    result['source_sha256'] = SOURCES
    result['script_sha256'] = digest(__file__)
    (OUT / 'diagnosis.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({w: {k: v for k, v in r.items() if k in ['signal_reconstruction', 'topups']}
                      for w, r in result['windows'].items()}, indent=2))


if __name__ == '__main__':
    main()
