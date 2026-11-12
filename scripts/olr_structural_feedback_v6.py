"""Read-only admission/funding accounting; no counterfactual alpha labels."""
from collections import Counter
from pathlib import Path
import gzip
import json
from scripts import olr_structural_native_v6 as e
from scripts.olr_structural_v6 import read, sha, write


def compact(event):
    meta = event.get('metadata', {})
    forecast = meta.get('capacity_forecast', {})
    exit_ = forecast.get('exit', {})
    loss = meta.get('scenario_loss_budget', {})
    return dict(timestamp=event['timestamp'], symbol=event['symbol'],
        code=event['decision_code'], reason=event['reason'],
        rank=meta.get('candidate_rank'), score=meta.get('candidate_score'),
        forecast_asof=forecast.get('asof'), auction_samples=exit_.get('samples'),
        exit_cap=exit_.get('quantity'), exit_status=exit_.get('status'),
        entry_cap=forecast.get('entry', {}).get('quantity'),
        initial_qty=meta.get('capacity_native_qty'), after_capacity_qty=meta.get('capacity_approved_qty'),
        allocator_input_qty=meta.get('allocation_native_qty'), approved_qty=meta.get('qty'),
        binding=meta.get('allocation_binding_limit'), target_notional=meta.get('target_notional'),
        reserved_notional=meta.get('reserved_notional'), decision_cash=meta.get('available_cash_before_entry'),
        equity=loss.get('equity'), approved_loss=loss.get('approved_loss'),
        before_redistribution=meta.get('allocation_before_redistribution'),
        redistributed_qty=meta.get('allocation_redistributed_qty'),
        recycling=meta.get('capacity_recycling'))


def inspect(folder):
    e.s.forensic.authenticate(folder)
    records, event_counts = [], Counter()
    with gzip.open(folder / 'native_decisions.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            event = json.loads(line)
            if event['decision_code'].startswith('ENTRY_'):
                event_counts[event['decision_code'] + ':' + event['reason']] += 1
            # Count funding decisions once. Partial fills and expired copies
            # contain the same forecast but are execution events, not signals.
            if event['decision_code'] in ('ENTRY_SUBMITTED', 'ENTRY_SKIPPED', 'ENTRY_DEFERRED'):
                records.append(compact(event))
    keys = [(r['timestamp'], r['symbol']) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError('Ambiguous repeated admission key: retain event sequence before comparing')
    indexed = dict(zip(keys, records))
    trades = read(folder / 'trades.json')
    trade_index = {(r['entry_decision_time'], r['symbol']): r for r in trades}
    if len(trades) != len(trade_index):
        raise ValueError('Non-unique trade lifecycle')
    for key, trade in trade_index.items():
        decision = indexed[key]
        assert decision['code'] == 'ENTRY_SUBMITTED', key
        decision.update(filled_qty=trade['qty'], entry_notional=trade['qty'] * trade['entry_price'],
                        net_pnl=trade['net_pnl'], entry_fill=trade['entry_fill_time'], exit_fill=trade['exit_fill_time'])
    counts = Counter(r['code'] + ':' + r['reason'] for r in records)
    bindings = Counter(r['binding'] for r in records if r['binding'] is not None)
    proposed = [r for r in records if r['after_capacity_qty'] is not None]
    return dict(folder=str(folder), completion_sha256=sha(folder / 'completion.json'),
        native_decisions_sha256=sha(folder / 'native_decisions.jsonl.gz'), decisions=records,
        event_counts=dict(event_counts), admission_counts=dict(counts), funding_bindings=dict(bindings),
        admissions=len(records), filled_lifecycles=len(trades),
        capacity_clipped=sum(r['initial_qty'] is not None and r['after_capacity_qty'] < r['initial_qty'] for r in proposed),
        allocator_clipped=sum(r['approved_qty'] is not None and r['allocator_input_qty'] is not None
                              and r['approved_qty'] < r['allocator_input_qty'] for r in records),
        submitted_without_fill=sum(r['code']=='ENTRY_SUBMITTED' and 'filled_qty' not in r for r in records),
        filled_notional=sum(t['qty'] * t['entry_price'] for t in trades),
        known_filled_net_pnl=sum(t['net_pnl'] for t in trades))


def compare(a, z):
    key = lambda r: (r['timestamp'], r['symbol'])
    old, new = {key(r): r for r in a['decisions']}, {key(r): r for r in z['decisions']}
    matched = []
    for k in sorted(old.keys() & new.keys()):
        x, y = old[k], new[k]
        matched.append(dict(timestamp=k[0], symbol=k[1], control=x, migration=y,
            history_changed=x['auction_samples'] != y['auction_samples'],
            cap_changed=x['exit_cap'] != y['exit_cap'],
            allocation_changed=x['approved_qty'] != y['approved_qty'],
            fill_changed=x.get('filled_qty', 0) != y.get('filled_qty', 0),
            cash_change=None if x['decision_cash'] is None or y['decision_cash'] is None else y['decision_cash']-x['decision_cash'],
            net_pnl_change=y.get('net_pnl', 0)-x.get('net_pnl', 0)))
    added = [new[k] for k in sorted(new.keys()-old.keys())]
    removed = [old[k] for k in sorted(old.keys()-new.keys())]
    accounted = sum(r['net_pnl_change'] for r in matched) + sum(r.get('net_pnl', 0) for r in added) - sum(r.get('net_pnl', 0) for r in removed)
    delta = z['known_filled_net_pnl'] - a['known_filled_net_pnl']
    assert abs(accounted-delta) < 1e-5
    return dict(matched=matched, added_admissions=added, removed_admissions=removed,
        matched_count=len(matched), changed_histories=sum(r['history_changed'] for r in matched),
        changed_caps=sum(r['cap_changed'] for r in matched), changed_allocations=sum(r['allocation_changed'] for r in matched),
        changed_fills=sum(r['fill_changed'] for r in matched), accounted_pnl_delta=accounted,
        first_changed_allocation=next((r for r in matched if r['allocation_changed']), None),
        largest_absolute_changes=sorted(matched, key=lambda r: abs(r['net_pnl_change']), reverse=True)[:12],
        interpretation='Decision-path accounting, not isolated causal effects. An absent or unfilled opportunity has no observed alpha label. Cash is recorded before the batch, not independently available cash for each order.')


def collect():
    e.configure()
    data = read(e.OUT / 'results.json')
    rows = {}
    for _, _, group in e.GROUPS:
        rows[group] = {}
        for window in data['candidates'][0][group]:
            a, z = [inspect(Path(c[group][window]['folder'])) for c in data['candidates']]
            rows[group][window] = dict(control=a, migration=z, comparison=compare(a, z))
    return dict(version='structural-admission-feedback-v6', results_sha256=sha(e.OUT / 'results.json'),
        scenarios=rows, counterfactual_unfilled_returns=False, excludes_new_alpha_claims=True)


if __name__ == '__main__':
    write(e.OUT / 'admission_feedback.json', collect())
