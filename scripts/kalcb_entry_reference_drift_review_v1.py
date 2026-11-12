"""Independent Decimal arithmetic over original fills; no producer import."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def stamp(x): return datetime.fromisoformat(x)
def dec(x): return Decimal(str(x))
def near(x, y):
    if abs(dec(x) - dec(y)) > max(Decimal('.0001'), abs(dec(y)) * Decimal('1e-10')):
        raise ValueError(f'Decomposition mismatch {x} vs {y}')
def digest(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f: json.dump(value, f, indent=2, allow_nan=False)


def run(root):
    import psutil
    identity = read(root / 'supervisor_start.json')
    try:
        p = psutil.Process(identity['pid'])
        if abs(p.create_time() - identity['create_time']) < .01 and p.cmdline() == identity['cmdline']:
            raise RuntimeError('Exact producer still active')
    except psutil.NoSuchProcess: pass
    protocol = read(root / 'protocol.json'); complete = read(root / 'completion.json')
    bindings = protocol['programs'] + protocol['bindings'] + protocol['origins'] + complete['outputs']
    for b in bindings:
        if digest(b['path']) != b['sha256']: raise ValueError('Changed input ' + b['path'])
    out = root / 'review_v1'; out.mkdir(exist_ok=False)
    me = psutil.Process(); write(out / 'supervisor_start.json', {'pid': me.pid, 'create_time': me.create_time(), 'cmdline': me.cmdline()})
    checks = []; sums = ['first_qty', 'later_qty', 'reference_notional_krw', 'actual_notional_krw',
        'initial_market_drift_krw', 'later_market_drift_krw', 'initial_slippage_krw', 'later_slippage_krw',
        'entry_fees_krw', 'parent_net_pnl_krw', 'parent_entry_cost_krw']
    for case in protocol['cases']:
        raw = read(Path(case['folder']) / 'fills.json'); by_order = defaultdict(list)
        ids = [f['metadata']['execution_id'] for f in raw]
        assert len(ids) == len(set(ids))
        for f in raw:
            if f['side'] == 'BUY': by_order[f['order_id']].append(f)
        old = read(Path(protocol['source']) / 'accounts' / Path(case['key']).with_suffix('.json'))
        old = {r['order_id']: r for r in old['rows']}
        actual = read(root / 'accounts' / Path(case['key']).with_suffix('.json'))
        assert len(actual['rows']) == len(old) and {r['order_id'] for r in actual['rows']} == set(old)
        positions = set(); slip = Decimal('0.0013') if case['key'].split('/')[-1] in ['cost', 'combined'] else Decimal('0.0003')
        for row in actual['rows']:
            source = old[row['order_id']]; fills = by_order[row['order_id']]
            for key in ['symbol', 'day', 'position_id', 'planned_qty', 'decision_available_at', 'parent_net_pnl_krw', 'parent_entry_cost_krw']:
                assert row[key] == source[key]
            assert row['candidate_pnl'] is None
            if not fills:
                assert row['cohort'] == 'unfilled' and not row['position_id']
                assert all(row[k] == 0 for k in sums)
                continue
            assert row['position_id'] not in positions; positions.add(row['position_id'])
            at = min(stamp(f['timestamp']) for f in fills)
            ref = dec(fills[0]['metadata']['entry_price_ref'])
            if source.get('original_reference'): near(ref, source['original_reference'])
            assert at >= stamp(source['decision_available_at']) and stamp(row['first_fill_at']) == at
            calculated = {k: Decimal(0) for k in sums[:-2]}
            initial_base = Decimal(0); initial_price = Decimal(0)
            for f in fills:
                m = f['metadata']; q = dec(f['qty']); base = dec(m['base_price']); price = dec(f['price'])
                assert isinstance(f['qty'], int) and f['qty'] > 0
                assert f['symbol'] == source['symbol'] and dec(m['entry_price_ref']) == ref
                assert m['entry_route'] == 'first30_open_anchor' and m['execution_phase'] == 'open'
                assert base == dec(m['execution_bar']['open']) and m['execution_bar']['volume'] > 0
                assert stamp(m['execution_bar']['timestamp']) == stamp(f['timestamp'])
                near(price, base * (1 + slip))
                first = stamp(f['timestamp']) == at; label = 'initial' if first else 'later'
                calculated['first_qty' if first else 'later_qty'] += q
                calculated[label + '_market_drift_krw'] += q * (base - ref)
                calculated[label + '_slippage_krw'] += q * (price - base)
                calculated['reference_notional_krw'] += q * ref
                calculated['actual_notional_krw'] += q * price
                calculated['entry_fees_krw'] += dec(m['fee'])
                if first: initial_base += q * base; initial_price += q * price
            for k, v in calculated.items(): near(row[k], v)
            assert calculated['first_qty'] + calculated['later_qty'] <= row['planned_qty']
            near(calculated['actual_notional_krw'], source['parent_entry_cost_krw'])
            near(row['first_base_gap_bps'], (initial_base / calculated['first_qty'] / ref - 1) * 10000)
            near(row['first_fill_gap_bps'], (initial_price / calculated['first_qty'] / ref - 1) * 10000)
            gap = calculated['initial_market_drift_krw']
            assert row['cohort'] == ('adverse_initial_gap' if gap > Decimal('.000001') else 'favorable_initial_gap' if gap < Decimal('-.000001') else 'flat_initial_gap')
            assert set(row['execution_ids']) == {f['metadata']['execution_id'] for f in fills}
        def check(g, rows):
            assert g['orders'] == len(rows) and g['filled_lifetimes'] == sum(bool(r['position_id']) for r in rows)
            assert g['sessions'] == len({r['day'] for r in rows}) and g['issuer_days'] == len({(r['symbol'], r['day']) for r in rows})
            for k in sums: near(g[k], sum((dec(r[k]) for r in rows), Decimal(0)))
            assert g['winners'] == sum(r['parent_net_pnl_krw'] > 0 for r in rows)
            near(g['positive_parent_net_krw'], sum((max(Decimal(0), dec(r['parent_net_pnl_krw'])) for r in rows), Decimal(0)))
            near(g['negative_parent_net_krw'], sum((min(Decimal(0), dec(r['parent_net_pnl_krw'])) for r in rows), Decimal(0)))
            if g['reference_notional_krw']:
                near(g['initial_market_drift_per_all_filled_notional_bps'], dec(g['initial_market_drift_krw']) / dec(g['reference_notional_krw']) * 10000)
            else: assert g['initial_market_drift_per_all_filled_notional_bps'] is None
        for k, g in actual['groups'].items(): check(g, [r for r in actual['rows'] if k == 'all' or r['cohort'] == k])
        for field, grouping in [('months', lambda r: r['day'][:7]), ('issuers', lambda r: r['symbol'])]:
            assert set(actual[field]) == {grouping(r) for r in actual['rows']}
            for value, groups in actual[field].items():
                for k, g in groups.items(): check(g, [r for r in actual['rows'] if grouping(r) == value and (k == 'all' or r['cohort'] == k)])
        checks.append({'account': case['key'], 'orders': len(old), 'unique_lifetimes': len(positions)})
        print(json.dumps(checks[-1]), flush=True)
    for b in bindings:
        if digest(b['path']) != b['sha256']: raise ValueError('Post-review changed binding')
    write(out / 'review.json', {'passed': True, 'accounts': checks, 'bindings': len(bindings),
          'scope': 'Independent Decimal first/later original fills, clocks, reference/base/slippage decomposition, once-only lifetime and complete month/issuer arithmetic. Historical raw-source/account/MTM audit receipts reused, not rerun. No attainable reference-price saving or alternate-policy economics.', 'study_complete': False})
    write(out / 'completion.json', {'complete': True, 'passed': True, 'at_utc': datetime.now(timezone.utc).isoformat(),
          'review': {'path': str(out / 'review.json'), 'sha256': digest(out / 'review.json')}, 'study_complete': False})


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('--root', required=True, type=Path); run(a.parse_args().root)
