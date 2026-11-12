"""Read-only C12-C: authenticate C12-B and inspect temporal access states."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import traceback

REPO = Path(__file__).resolve().parents[1]
SOURCE = Path('E:/CodexResearch/k_stock_trader/kalcb_opportunity_lifecycle_20261016/funded_interaction_v1')
DESIGN = REPO / 'docs/optimisation/kalcb/experiments/kalcb-exp-0036-closeout-design.md'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, value):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def bound(p):
    p = Path(p)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return dict(path=str(p), sha256=h.hexdigest())


def verify(binding):
    assert bound(binding['path'])['sha256'] == binding['sha256'], binding['path']


def lines(p):
    with gzip.open(p, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def stamp(s):
    return datetime.fromisoformat(s).isoformat() if s else None


def key(day, symbol):
    return day[:10] + '/' + symbol


def journal(folder, targets):
    """Only observed states: no alternative-price/funding simulation."""
    orders = {}
    submits = []
    for n, row in enumerate(lines(folder / 'orders.jsonl.gz')):
        o = row.get('order', {})
        if o.get('side') != 'BUY':
            continue
        oid = o['order_id']
        m = o.get('metadata', {})
        if row['event'] == 'submit':
            s = dict(order_id=oid, symbol=o['symbol'], qty=o['qty'], sequence=n,
                     at=stamp(m.get('decision_available_at') or o['submitted_at']),
                     signal=stamp(m.get('signal_bar') or o['submitted_at']),
                     route=m.get('entry_route'), reference=m.get('entry_price_ref'),
                     stop=o.get('stop_price'), risk=m.get('risk_per_share'),
                     cap=m.get('pending_funding_reservation'),
                     rank=m.get('candidate_rank'), entry_quality=m.get('entry_quality_votes'),
                     filled_qty=0, buy_fills=[], terminal_events=[])
            orders[oid] = s
            submits.append(s)
        elif row['event'] in ('reject', 'cancel', 'expire') and oid in orders:
            orders[oid]['terminal_events'].append(dict(event=row['event'], at=stamp(row.get('timestamp')),
                                                      reason=row.get('reason'), qty=o['qty']))
    fills = read(folder / 'fills.json')
    exits = defaultdict(list)
    for f in fills:
        if f['side'] == 'BUY':
            assert f['order_id'] in orders
            o = orders[f['order_id']]
            o['filled_qty'] += f['qty']
            o['buy_fills'].append(dict(at=stamp(f['timestamp']), qty=f['qty'], price=f['price']))
        else:
            exits[key(f['timestamp'], f['symbol'])].append(dict(at=stamp(f['timestamp']), qty=f['qty'], price=f['price']))
    for o in submits:
        assert o['filled_qty'] <= o['qty']
        terminals = [x['at'] for x in o['terminal_events'] if x['at']]
        if o['filled_qty'] == o['qty'] and o['buy_fills']:
            terminals.append(o['buy_fills'][-1]['at'])
        o['observed_terminal_at'] = min(terminals) if terminals else None
        o['episode'] = key(o['signal'], o['symbol'])
    trades = read(folder / 'trades.json')
    episodes = defaultdict(lambda: dict(net=0., trades=[]))
    for t in trades:
        k = key(t['entry_decision_time'], t['symbol'])
        episodes[k]['net'] += t['net_pnl']
        episodes[k]['trades'].append({x: t.get(x) for x in ('entry_decision_time', 'entry_fill_time', 'exit_fill_time', 'qty', 'net_pnl', 'exit_reason')})
    decisions = {k: [] for k in targets}
    for d in lines(folder / 'decisions.jsonl.gz'):
        k = key(d['timestamp'], d['symbol'])
        if k not in decisions:
            continue
        m = d.get('metadata', {})
        decisions[k].append(dict(at=stamp(d['timestamp']), code=d['decision_code'], reason=d['reason'],
            route=m.get('entry_route'), route_attempts=m.get('entry_route_attempts'),
            cap=m.get('pending_funding_reservation'),
            actions=[dict(type=a['action_type'], qty=a.get('qty'), symbol=a.get('symbol')) for a in d.get('actions', [])]))
    rows = {}
    for k in targets:
        rows[k] = dict(**episodes[k], orders=[o for o in submits if o['episode'] == k],
                       exits=exits[k], decisions=decisions[k])
    return dict(submits=submits, affected=rows, net=sum(t['net_pnl'] for t in trades))


def first_submission(a, b):
    # File order is execution ordering within a timestamp batch. Never time an
    # amendment using the order's original submitted_at.
    def signature(o):
        return tuple(o.get(k) for k in ('at', 'symbol', 'qty', 'route', 'reference', 'stop', 'risk'))
    for n in range(max(len(a), len(b))):
        left = a[n] if n < len(a) else None
        right = b[n] if n < len(b) else None
        if left is None or right is None or signature(left) != signature(right):
            return dict(index=n, actual=left, control=right)
    return None


def main(output):
    output.mkdir(exist_ok=False)
    p = read(SOURCE / 'protocol.json')
    done = read(SOURCE / 'completion.json')
    verify(done['registration']); verify(done['assessment'])
    ledger = read(SOURCE / 'completed_cases.json')
    assert len(ledger) == 12 and done['complete']
    receipts = []
    sources = [bound(DESIGN), bound(__file__), bound(SOURCE / 'protocol.json'),
               bound(SOURCE / 'registration.json'), bound(SOURCE / 'assessment.json'),
               bound(SOURCE / 'completed_cases.json'), bound(SOURCE / 'completion.json')]
    folders = []
    for row in ledger:
        c = row['case']; folder = SOURCE / 'cases' / c['id']
        verify(row['completion'])
        assert row['verdict']['qualified']
        for name, field in [('activation.json', 'passed'), ('independent_account_verification.json', 'passed'), ('funding_review.json', 'verified')]:
            data = read(folder / name)
            assert data[field], (folder, name)
            sources.append(bound(folder / name))
            for binding in data.get('bindings', []):
                verify(binding)
        bridge = c['kind'] == 'bridge'
        if bridge:
            assert read(folder / 'identity_bridge.json')['passed']
            sources.append(bound(folder / 'identity_bridge.json'))
        receipts.append(dict(case=c['id'], qualified=True, exact_bridge=bridge))
        if not bridge:
            folders += [folder, Path(p['controls']['veto'][c['period'] + '/' + c['scenario']])]
            sources.append(bound(SOURCE / 'traces' / (c['id'] + '_versus_veto.json')))
    for folder in dict.fromkeys(folders):
        for name in ('orders.jsonl.gz', 'decisions.jsonl.gz', 'trades.json', 'fills.json'):
            sources.append(bound(folder / name))
    write(output / 'registration.json', dict(design=bound(DESIGN), bindings=sources,
        scope='Read-only C12-C; after B outcomes; no new fit or financial policy.', native_replays=0, fits=0))
    assessment = read(SOURCE / 'assessment.json')
    results = {}
    for cell, value in assessment['cells'].items():
        period, scenario = cell.split('/')
        write(output / 'status.json', dict(phase='extracting', cell=cell, completed=len(results), total=8))
        folder = SOURCE / 'cases' / ('combined_' + period + '_' + scenario)
        control = Path(p['controls']['veto'][cell])
        observations = read(folder / 'activation.json')['observations']
        trimmed = {key(o['at'], o['symbol']): o for o in observations if o['cap']['capped_qty'] < o['cap']['original_qty']}
        actual = journal(folder, trimmed); parent = journal(control, trimmed)
        v = value['comparisons']['veto']; prior = value['comparisons']['prior']
        assert abs(actual['net'] - v['activity']['summary']['net_pnl_krw']) < .01
        assert abs(parent['net'] - v['control_activity']['summary']['net_pnl_krw']) < .01
        d = v['activity_delta']
        decomposition = d['shared_quantity_at_control_unit_pnl_krw'] + d['shared_unit_outcome_at_treatment_qty_krw'] + d['added_pnl_krw'] - d['removed_control_pnl_krw']
        assert abs(decomposition - v['net_delta_krw']) < .01
        dates = sorted({k[:10] for k in trimmed})
        affected = {}
        for k, o in trimmed.items():
            a, b = actual['affected'][k], parent['affected'][k]
            affected[k] = dict(cap=o['cap'], actual=a, control=b, net_delta=a['net'] - b['net'])
        books = {day: {arm: [s for s in j['submits'] if s['episode'][:10] == day] for arm, j in [('actual', actual), ('control', parent)]} for day in dates}
        first = first_submission(actual['submits'], parent['submits'])
        saved_trace = read(SOURCE / 'traces' / ('combined_' + period + '_' + scenario + '_versus_veto.json'))
        episode_deltas = saved_trace['deltas']
        day_delta = defaultdict(float)
        for k, amount in episode_deltas.items(): day_delta[k[:10]] += amount
        sorted_dates = sorted(day_delta.items(), key=lambda x: (x[1], x[0]))
        report = dict(
            metrics=v['metrics'], control_metrics=v['control_metrics'],
            net_vs_veto=v['net_delta_krw'], net_vs_prior=prior['net_delta_krw'],
            net_vs_cap=value['comparisons']['cap']['net_delta_krw'],
            interaction_net_krw=value['interaction_net_krw'],
            useful=dict(actual=v['useful_days'], veto=v['control_useful_days'], prior=prior['control_useful_days']),
            activity=v['activity']['summary'], control_activity=v['control_activity']['summary'],
            attribution=d, monthly=v['path']['monthly'],
            distinct_trimmed_dates=dates, trimmed_issuer_days=len(trimmed),
            zero_cap=sum(x['cap']['capped_qty'] == 0 for x in trimmed.values()),
            first_submission_divergence=first, historical_first_order_divergence=saved_trace['first_order_divergence'],
            all_day_deltas=dict(sorted_dates),
            best_day=sorted_dates[-1], worst_day=sorted_dates[0],
            net_excluding_best_date=v['net_delta_krw'] - sorted_dates[-1][1],
            affected=affected, opening_books=books)
        write(output / 'cells' / (period + '_' + scenario + '.json'), report)
        results[cell] = {k: value for k, value in report.items() if k not in ('affected', 'opening_books')}
    # Registration precedes extraction and all consumed sources remain unchanged.
    for binding in sources: verify(binding)
    write(output / 'review.json', dict(cells=results, receipts=receipts, criteria=assessment['criteria'],
        consumed_bindings=sources, valid=True, economic_objective_achieved=False,
        interpretation='Descriptive complete funded accounts and observed paths; no alternative retry/add values. '
                       'Submission divergence corrects amendment-time ambiguity; original trace untouched.'))
    write(output / 'completion.json', dict(complete=True, review=bound(output / 'review.json'),
        registration=bound(output / 'registration.json'), native_replays=0, fits=0))
    write(output / 'status.json', dict(phase='complete', at=datetime.now(timezone.utc).isoformat(), economic_objective_achieved=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try: main(args.output)
    except Exception as error:
        if args.output.exists(): write(args.output / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
        raise
