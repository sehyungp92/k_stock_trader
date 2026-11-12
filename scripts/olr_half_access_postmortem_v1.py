"""0040 read-only failed-interaction postmortem; never launches financial work."""
from collections import defaultdict
from pathlib import Path
import argparse
import json
import math
from olr_selective_half_continuation_v1 import read, write, sha, now


def run(root):
    out = root/'postmortem_v1'
    out.mkdir(exist_ok=False)
    inputs = [root/'assessment.json', root/'causal_review_v1/assessment.json',
              root/'causal_review_v1/joint_extra_traces.json', root/'half_disabled_parity.json']
    write(out/'protocol.json', dict(at=now(), script_sha256=sha(__file__),
        inputs={str(p):sha(p) for p in inputs},
        selection='All four arms, all seven extras, all changed meaningful identities and immediately preceding same-symbol lifecycle; first changed allocation and five largest signed contributions already selected by the prospective review. Inspect the largest adverse 29 October funding pair across all arms.',
        question='Does valid joint activation fail through direct rescue coverage, altered history/quantity, or source/accounting error? Preserve partial gains; no fitted rule, new replay or qualification claim.',
        resource='Read-only sequential extraction from accepted files, planned under five minutes; no user compute limit.'))
    a, c = read(inputs[0]), read(inputs[1])
    trades, contexts, checks = {}, {}, []
    for policy, summary in a['summaries'].items():
        case = Path(summary['case'])
        assert sha(case/'completion.json') == summary['completion_sha256']
        for name, digest in read(case/'completion.json')['artifacts'].items():
            assert sha(case/name) == digest, (policy, name)
            checks.append(dict(policy=policy, file=name, sha256=digest))
        trades[policy] = {(t['symbol'], t['entry_decision_time'][:10]):t for t in read(case/'trades.json')}
        contexts[policy] = []
        for batch in read(case/'funded_population_receipt.json')['entry_contexts']:
            for d in batch['decisions']:
                p = batch['proposals'][d['intent_id']]
                contexts[policy].append(dict(at=batch['at'], symbol=p['symbol'], proposal=p,
                    decision=d, account=batch['account'], cash=batch['actual_deployable_cash']))
    lineage = {}
    for name in ['control_to_joint', 'half_to_joint']:
        pair = a['comparisons'][name]
        left, right = pair['control'], pair['treatment']
        lookup = {(r['symbol'],r['day']):r for r in pair['accounting']['per_opportunity']}
        changed = {(r['symbol'],r['day']) for r in c['comparisons'][name]['changed_meaningful']}
        parents, rows = set(), []
        for key in sorted(changed):
            preceding = {}
            for policy in [left,right]:
                candidates = [k for k in trades[policy] if k[0] == key[0] and k[1] < key[1]]
                parent = max(candidates, key=lambda k:k[1]) if candidates else None
                preceding[policy] = None if parent is None else trades[policy][parent]
                if parent: parents.add(parent)
            rows.append(dict(symbol=key[0], day=key[1], contribution=lookup[key], prior_lifecycles=preceding))
        # Conservative descriptive lineage sensitivity, not an identified renewal effect.
        union = changed|parents
        lineage[name] = dict(rows=rows, prior_keys=[list(k) for k in sorted(parents)],
            base_changed_meaningful_pnl=sum(lookup[k]['delta_pnl'] for k in changed),
            union_pnl_without_double_count=sum(lookup[k]['delta_pnl'] for k in union),
            scope='Immediate previous same-symbol lifecycles in both arms, counted once. This broad attribution sensitivity does not imply every predecessor caused renewal.')
    pair_rows = {p:[r for r in rows if r['at'].startswith('2025-10-29') and r['symbol'] in ['010120','051910']]
                 for p,rows in contexts.items()}
    write(out/'october29_native_contexts.json',pair_rows)
    half = next(r for r in pair_rows['half'] if r['symbol']=='010120')
    joint = next(r for r in pair_rows['joint'] if r['symbol']=='010120')
    hf, jf = [r['proposal']['metadata']['capacity_forecast'] for r in [half,joint]]
    assert hf['entry'] == jf['entry']
    hs, js = set(map(tuple,hf['exit']['samples'])), set(map(tuple,jf['exit']['samples']))
    assert js-hs == {('2025-10-27T15:30:00+09:00',46590.0)} and not hs-js
    assert [hf['exit']['quantity'],jf['exit']['quantity']] == [675,465]
    assert [half['decision']['approved_qty'],joint['decision']['approved_qty']] == [627,465]
    successor = {p:next(r for r in pair_rows[p] if r['symbol']=='051910') for p in ['half','joint']}
    assert [successor[p]['decision']['approved_qty'] for p in ['half','joint']] == [0,25]
    assert successor['half']['proposal']['metadata']['capacity_forecast'] == successor['joint']['proposal']['metadata']['capacity_forecast']
    per = a['comparisons']['half_to_joint']['accounting']['per_opportunity']
    pair_delta = sum(r['delta_pnl'] for r in per if r['day']=='2025-10-29' and r['symbol'] in ['010120','051910'])
    grouped = {}
    for name, pair in a['comparisons'].items():
        by_month = defaultdict(float)
        for row in pair['accounting']['per_opportunity']: by_month[row['day'][:7]] += row['delta_pnl']
        assert math.isclose(sum(by_month.values()),pair['delta']['return_pct']*1e6,abs_tol=.01)
        grouped[name] = dict(month_attribution=dict(by_month),
            changed_meaningful=c['comparisons'][name]['changed_meaningful_pnl'],
            targeted=c['comparisons'][name]['targeted_pnl'], other=c['comparisons'][name]['other_pnl'],
            accounting=pair['accounting']['additive_pnl_accounting'])
    extras = []
    for trace in read(inputs[2]):
        key = trace['symbol'],trace['day']
        row = dict(symbol=key[0],day=key[1], policies={})
        for p in ['access','joint']:
            t = trades[p][key]
            row['policies'][p] = {k:t[k] for k in ['qty','net_pnl','entry_fill_time','exit_fill_time','exit_reason']}
        extras.append(row)
    ready = read(root/'native_engines/train/ready.json')
    code = Path(ready['files'])/'strategy_olr/core/capacity.py'
    write(out/'assessment.json',dict(at=now(), completion_checks=checks, lineage=lineage,
        extras=extras, grouped=grouped, native_contexts_sha256=sha(out/'october29_native_contexts.json'),
        history_pair=dict(extra_sample=list(js-hs)[0],exit_caps=[675,465],actual_quantities=[627,465],
            successor_quantities=[0,25],combined_pnl_delta=pair_delta,
            fraction_of_joint_minus_half_loss=pair_delta/(a['comparisons']['half_to_joint']['delta']['return_pct']*1e6),
            source_code=str(code),source_sha256=sha(code),
            caveat='Observed native history membership and funded consequences; deleting this sample is not a valid alternative funded policy. Legacy feed routing can retain the day after an early exit; do not mislabel history as current holding at the closing auction.'),
        readiness=c['readiness_references'], economic_objective_achieved=False,
        verdict='Exact HA fails principal readiness. Preserve nominal +0.403019pp versus N0 and +0.442816pp versus A, but additional meaningful contribution is negative and HA loses 1.761384pp against H. No qualification or automatic policy successor.',
        remaining_operation='Reconcile the materially different dominant early OOS failure mechanism against existing clock predecessors before deciding whether its specific unresolved timing/execution distinction earns a diagnostic.'))
    print(json.dumps(dict(output=str(out),artifacts=len(checks),pair_delta=pair_delta,
        lineage={k:v['union_pnl_without_double_count'] for k,v in lineage.items()})))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
