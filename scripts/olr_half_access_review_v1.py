"""0040 exhaustive access/partial traces and four-arm funded attribution."""
from collections import defaultdict
from datetime import datetime, timedelta
import argparse
import gzip
import json
import math
from pathlib import Path
import sys
from olr_selective_half_continuation_v1 import read, write, sha, now
from olr_relative_context_assessment_v1 import occupancy
from olr_opportunity_continuity_v1 import ledger


def meaningful(trade):
    return bool(trade and trade['qty']*trade['entry_price'] >= 1e6)


def run(root, out):
    out.mkdir(parents=True, exist_ok=False)
    assessment = read(root/'assessment.json'); summaries = assessment['summaries']
    ready = read(root/'native_engines/train/ready.json'); sys.path.insert(0, ready['files'])
    from strategy_common.execution_session import session_cutoff
    from strategy_olr.execution import DECISION_CUTOFF
    census_path = root.parent/'opportunity_continuity_v2/train_rows.json'
    census = {(r['symbol'], r['day']):r for r in read(census_path)}
    outside = {key:r for key,r in census.items() if r['disposition'] == 'outside_slots'}
    assert len(outside) == 24
    write(out/'protocol.json', dict(at=now(), runner_sha256=sha(__file__),
        assessment_sha256=sha(root/'assessment.json'), census_sha256=sha(census_path),
        method='All 24 outside-slot identities, all actual extras and partial actions, all changed meaningful participation; first and largest signed downstream effects.'))
    trades = {}; contexts = {}; decisions = {}; checks = []; occupancies = {}
    for policy in ('control', 'access', 'half', 'joint'):
        print(json.dumps(dict(at=now(), phase='artifacts_and_occupancy', policy=policy)), flush=True)
        case = Path(summaries[policy]['case'])
        assert sha(case/'completion.json') == read(root/'accepted'/('train_'+policy+'.json'))['completion_sha256']
        for name, digest in read(case/'completion.json')['artifacts'].items():
            assert sha(case/name) == digest, (policy, name)
            checks.append(dict(policy=policy, artifact=name, sha256=digest))
        raw = read(case/'trades.json')
        trades[policy] = {(t['symbol'], t['entry_decision_time'][:10]):t for t in raw}
        assert len(raw) == len(trades[policy]), 'Nonunique native opportunity identity'
        contexts[policy] = {}
        for batch in read(case/'funded_population_receipt.json')['entry_contexts']:
            for decision in batch['decisions']:
                proposal = batch['proposals'][decision['intent_id']]
                key = (proposal['symbol'], batch['at'])
                row = dict(at=batch['at'], proposal=proposal, decision=decision, account=batch['account'], cash=batch['actual_deployable_cash'])
                if key in contexts[policy]: assert contexts[policy][key] == row
                contexts[policy][key] = row
        decisions[policy] = []
        with gzip.open(case/'native_decisions.jsonl.gz', 'rt', encoding='utf8') as stream:
            for line in stream:
                row = json.loads(line)
                if row['decision_code'].startswith(('ENTRY', 'EXIT')): decisions[policy].append(row)
        occupancies[policy] = occupancy(case)
    extras = {p:[r for r in contexts[p].values() if r['proposal']['metadata'].get('expired_selection_v1')] for p in contexts}
    halves = {p:[r for r in decisions[p] if r['decision_code'] == 'EXIT_SUBMITTED' and r['reason'] == 'weak_followthrough_partial'] for p in decisions}
    assert not extras['control'] and not extras['half'] and extras['access'] and extras['joint']
    assert not halves['control'] and not halves['access'] and halves['half'] and halves['joint']
    points = sorted({datetime.fromisoformat(r['timeline'][-1]['at']) for r in outside.values()} |
        {datetime.fromisoformat(r['metadata']['structural_partial_evidence']['decision_available_at']) for rows in halves.values() for r in rows})
    books = {}
    for policy in contexts:
        print(json.dumps(dict(at=now(), phase='funded_ledger', policy=policy)), flush=True)
        books[policy] = ledger(Path(summaries[policy]['case']), points)
        assert math.isclose(books[policy]['terminal_cash'], 1e8+summaries[policy]['return_pct']*1e6, abs_tol=.01)

    def lifecycle(policy, key):
        trade = trades[policy].get(key)
        orders = []
        if trade:
            start = datetime.fromisoformat(trade['entry_decision_time'])
            end = datetime.fromisoformat(trade['exit_fill_time'])
            for order in books[policy]['orders'].values():
                if order['symbol'] == key[0] and start-timedelta(minutes=5) <= datetime.fromisoformat(order['submitted_at']) <= end:
                    orders.append(order)
            buys = [o for o in orders if o['side'] == 'BUY']
            assert len(buys) == 1, (policy, key, len(buys))
            quantities = {side:sum(e['fill_qty'] for o in orders if o['side'] == side for e in o['events'] if e['kind'] == 'fill') for side in ('BUY','SELL')}
            assert quantities['BUY'] == quantities['SELL'] == trade['qty'], (policy, key, quantities, trade['qty'])
        return dict(trade=trade, orders=orders,
            proposals=[r for (symbol,at),r in contexts[policy].items() if symbol == key[0] and at[:10] == key[1]],
            entry_decisions=[r for r in decisions[policy] if r['symbol'] == key[0] and r['timestamp'][:10] == key[1] and r['decision_code'].startswith('ENTRY')])

    extra_traces = {}
    for policy in ('access', 'joint'):
        traces = []
        for row in extras[policy]:
            proposal = row['proposal']; metadata = proposal['metadata']; evidence = metadata['expired_selection_v1']
            at = datetime.fromisoformat(row['at']); key = (proposal['symbol'], at.date().isoformat())
            cutoff = session_cutoff(at.date(), DECISION_CUTOFF)
            assert at == datetime.combine(at.date(), cutoff, at.tzinfo)+timedelta(minutes=30)
            assert evidence['available_at'] == row['at'] and datetime.fromisoformat(evidence['source_label'])+timedelta(minutes=5) == at
            assert key in outside and metadata['candidate_rank'] > 6 and key[0] in evidence['candidates']
            assert len(evidence['candidates']) <= len(evidence['expired']) and not set(evidence['expired']) & set(evidence['reserved'])
            assert metadata['allocation_weight'] == evidence['candidates'][key[0]]['weight']
            source = next(z for z in outside[key]['timeline'] if z['at'] == row['at'])
            assert source['source_complete'] and source['passes'] and source['close'] == metadata['entry_submission_close']
            assert all(not census[(s,key[1])]['first_confirmation'] for s in evidence['expired'])
            assert not books[policy]['snapshots'][row['at']]['quantities'].get(key[0])
            if policy == 'joint': assert metadata['structural_entry_day_low'] > 0
            trade = trades[policy].get(key)
            if trade: assert trade['route_metadata']['risk_per_share'] == metadata['risk_per_share']
            traces.append(dict(symbol=key[0], day=key[1], independent_source=source,
                decision=row, lifecycle=lifecycle(policy,key),
                before_after={p:lifecycle(p,key) for p in ('control','half')},
                boundary_books={p:books[p]['snapshots'][row['at']] for p in books}))
        extra_traces[policy] = traces
        write(out/(policy+'_extra_traces.json'), traces)
    partial_traces = {}
    for policy in ('half','joint'):
        rows = []
        for row in halves[policy]:
            metadata = row['metadata']; evidence = metadata['weak_partial_evidence']; structural = metadata['structural_partial_evidence']
            assert evidence['fraction'] == .5 and metadata['qty'] == evidence['remaining_before']//2
            assert evidence['close_r'] <= -.4 and evidence['mfe_r'] < .5 and evidence['session_bars'] >= 3
            assert datetime.fromisoformat(row['timestamp'])+timedelta(minutes=5) == datetime.fromisoformat(structural['decision_available_at'])
            qualifying = [t for key,t in trades[policy].items() if key[0] == row['symbol'] and
                datetime.fromisoformat(t['entry_fill_time']) <= datetime.fromisoformat(row['timestamp']) <= datetime.fromisoformat(t['exit_fill_time'])]
            assert len(qualifying) == 1, (policy, row)
            trade = qualifying[0]; key = (row['symbol'],trade['entry_decision_time'][:10])
            trace = lifecycle(policy,key)
            orders = [o for o in trace['orders'] if o['reason'] == 'weak_followthrough_partial']
            assert len(orders) == 1 and orders[0]['qty'] == metadata['qty']
            rows.append(dict(decision=row, symbol=key[0], day=key[1],
                lifecycle=trace, comparisons={p:lifecycle(p,key) for p in ('control','access','half','joint') if p != policy}))
        partial_traces[policy] = rows
        write(out/(policy+'_partial_traces.json'), rows)
    outside_rows = []
    for key, source in sorted(outside.items(), key=lambda item:(item[0][1],item[0][0])):
        at = source['timeline'][-1]['at']
        outside_rows.append(dict(symbol=key[0], day=key[1], source=source['timeline'][-1],
            policies={p:dict(proposals=[r for (s,t),r in contexts[p].items() if s == key[0] and t[:10] == key[1]],
                trade=trades[p].get(key), boundary_book=books[p]['snapshots'][at]) for p in contexts}))
    write(out/'all_outside_slot_states.json', outside_rows)
    pair_results = {}
    for name,pair in assessment['comparisons'].items():
        left,right=pair['control'],pair['treatment']; keys=trades[left].keys()|trades[right].keys()
        per=pair['accounting']['per_opportunity']; lookup={(r['symbol'],r['day']):r for r in per}
        changed={k for k in keys if meaningful(trades[left].get(k)) != meaningful(trades[right].get(k))}
        groups=defaultdict(float); participation=[]
        for key in sorted(changed):
            a,b=trades[left].get(key),trades[right].get(key)
            kind='new_meaningful' if not a else 'removed_meaningful' if not b else 'size_threshold_crossing'
            row=dict(**lookup[key], change_kind=kind, before_meaningful=meaningful(a), after_meaningful=meaningful(b))
            participation.append(row);groups[kind]+=row['delta_pnl']
        targeted={k for p in (left,right) for k,t in trades[p].items() if t['route_metadata'].get('expired_selection_v1')}
        targeted_pnl=sum(r['delta_pnl'] for r in per if (r['symbol'],r['day']) in targeted)
        assert math.isclose(sum(r['delta_pnl'] for r in per), pair['delta']['return_pct']*1e6, abs_tol=.01)
        allocations=[]
        for key in sorted(contexts[left].keys()|contexts[right].keys(), key=lambda k:(k[1],k[0])):
            a,b=contexts[left].get(key),contexts[right].get(key)
            if a is None or b is None or a['decision']['approved_qty'] != b['decision']['approved_qty']:
                allocations.append(dict(symbol=key[0],at=key[1],before=a,after=b))
        dates=defaultdict(float)
        for row in per:dates[row['day']]+=row['delta_pnl']
        # Method fixed before outcomes: all changed participation plus five largest
        # positive and negative downstream contributions, with ties sorted by identity.
        ranked=sorted(per,key=lambda r:(r['delta_pnl'],r['day'],r['symbol']))
        selected=changed|{(r['symbol'],r['day']) for r in ranked[:5]+ranked[-5:]}
        traces=[dict(symbol=k[0],day=k[1],attribution=lookup[k],before=lifecycle(left,k),after=lifecycle(right,k)) for k in sorted(selected)]
        write(out/(name+'_traces.json'),traces)
        pair_results[name]=dict(changed_meaningful=participation, changed_meaningful_pnl=sum(groups.values()),
            contribution_by_participation_kind=dict(groups), targeted_pnl=targeted_pnl,
            other_pnl=sum(r['delta_pnl'] for r in per)-targeted_pnl,
            changed_allocations=allocations, first_changed_allocation=allocations[0] if allocations else None,
            date_attribution=dict(dates), excluding_best_date=sum(dates.values())-max(dates.values(),default=0),
            selected_trace_file=name+'_traces.json')
    refs=dict(joint_return_vs_control=assessment['comparisons']['control_to_joint']['delta']['return_pct']>=.5,
        joint_return_vs_half=assessment['comparisons']['half_to_joint']['delta']['return_pct']>=-.5,
        joint_meaningful_vs_control=assessment['comparisons']['control_to_joint']['delta']['meaningful']>=2,
        positive_changed_meaningful_vs_control=pair_results['control_to_joint']['changed_meaningful_pnl']>0,
        positive_changed_meaningful_vs_half=pair_results['half_to_joint']['changed_meaningful_pnl']>0,
        newly_acquired_meaningful=any(r['change_kind']=='new_meaningful' for r in pair_results['half_to_joint']['changed_meaningful']))
    write(out/'assessment.json',dict(at=now(),completion_checks=checks, comparisons=pair_results,
        occupancy=occupancies, extra_counts={p:dict(contacts=len(rows),dates=len({r['at'][:10] for r in rows})) for p,rows in extras.items()},
        half_action_counts={p:len(rows) for p,rows in halves.items()}, readiness_references=refs,
        economic_objective_achieved=False,
        next='Substantive joint risk/concentration/participation assessment and exact-policy readiness decision. No automatic financial successor.',
        scope='Reused IS development, no independent validation. Factorial and attribution are whole-policy contrasts; accounting components are not independent causal effects.'))
    print(json.dumps(dict(at=now(),phase='review_complete',references=refs)),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--out',type=Path)
    args=parser.parse_args();run(args.root,args.out or args.root/'causal_review_v1')
