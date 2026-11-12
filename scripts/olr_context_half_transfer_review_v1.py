"""Post-outcome extraction only: authenticate a finished OOS pair and inspect its decisions.

Does not launch financial replay, select a policy, or alter qualification criteria.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def run(root, scenario):
    pair_dir = root / 'pairs' / ('oos_' + scenario)
    pair = read(pair_dir / 'assessment.json')
    causal = read(pair_dir / 'causal_review_v1/assessment.json')
    parent = read(root / 'pairs' / ('train_' + scenario) / 'assessment.json')
    out = pair_dir / 'transfer_review_v3'
    out.mkdir(exist_ok=False)
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    checks, trades, populations = {}, {}, {}
    for arm, s in pair['summaries'].items():
        case = Path(s['case'])
        completion = read(case / 'completion.json')
        artifacts = {k: sha(case / k) == v for k, v in completion['artifacts'].items()}
        assert all(artifacts.values())
        assert sha(case / 'completion.json') == s['completion_sha256']
        assert completion['complete']
        assert completion['source_generation_hash'] == read(root / 'preflight.json')['source_by_window']['oos']
        audit = read(case / 'execution_audit.json')
        capacity = audit['capacity_model_audit']
        checkpoint = read(case / 'funded_checkpoint_receipt.json')
        population = read(case / 'funded_population_receipt.json')
        restart = read(case / 'capacity_restart_receipt.json')
        train_case = Path(parent['summaries'][arm]['case'])
        assert sha(train_case / 'completion.json') == restart['parent_completion_sha256']
        assert sha(train_case / 'capacity_terminal_state.json') == restart['parent_checkpoint_sha256']
        assert restart['same_configuration_and_scenario']
        assert restart['all_samples_verified_against_oos_retained_ohlcv']
        assert audit['audit_pass'] and audit['execution_validity_pass']
        assert checkpoint['no_action_continuation_equal'] and population['audit']['pass_']
        assert capacity['errors'] == []
        assert capacity['terminal_open_shares'] == capacity['terminal_pending_shares'] == 0
        assert capacity['capital']['initial_equity'] == 100000000
        checks[arm] = dict(artifacts=artifacts, completion_sha256=s['completion_sha256'],
            source=completion['source_generation_hash'], audit_pass=True, execution_validity_pass=True,
            checkpoint_parity=True, population_pass=True, flat_initial_equity=100000000,
            terminal_open_shares=0, terminal_pending_shares=0,
            restart={k:v for k,v in restart.items() if k != 'restored_state'},
            phase_liquidity_verified=capacity['phase_liquidity_verified'])
        trades[arm] = {(t['symbol'],t['entry_decision_time'][:10]):t for t in read(case / 'trades.json')}
        populations[arm] = population['entry_contexts']
    per = {(x['symbol'],x['day']):x for x in pair['comparisons']['control_to_half']['accounting']['per_opportunity']}
    context = []
    for action in pair['summaries']['half']['context_decisions']:
        if action['decision_code'] != 'EXIT_SUBMITTED':
            continue
        m = action['metadata']; e = m['relative_context_evidence']
        order = next(x for x in causal['actions_and_funding']['half']['actual_orders']
                     if x['symbol'] == action['symbol'] and x['action_day'] == e['available_at'][:10])
        key = (order['symbol'],order['entry_day'])
        t = trades['half'][key]; r = t['route_metadata']['risk_per_share']
        assert r == e['original_risk'] and e['own_return'] < 0 and e['market_median'] >= 0
        assert e['peer_count'] >= 68 and e['observed_volume'] > 0
        assert e['available_at'][11:19] == '09:30:00' and e['source_last_label'][11:19] == '09:25:00'
        context.append(dict(symbol=key[0],entry_day=key[1],action_day=order['action_day'],
            evidence=e, entry_rank=t['route_metadata']['candidate_rank'],
            entry_score=t['route_metadata']['candidate_score'],
            early_close_minus_entry_in_original_R=(e['close']-t['entry_price'])/r,
            actual_net_pnl_delta=per[key]['delta_pnl'],
            control={k:trades['control'][key][k] for k in ['qty','entry_price','exit_price','exit_fill_time','net_pnl','exit_reason']},
            half={k:t[k] for k in ['qty','entry_price','exit_price','exit_fill_time','net_pnl','exit_reason']}))
    assert len(context) == causal['actions_and_funding']['half']['actions']
    assert len({(x['symbol'],x['entry_day']) for x in context}) == len(context)
    action_days = {x['action_day'] for x in context}
    funding = {}
    for arm, entries in populations.items():
        funding[arm] = []
        for e in entries:
            if e['at'][:10] not in action_days:
                continue
            proposals = {}
            for key, proposal in e['proposals'].items():
                m = proposal['metadata']
                proposals[key] = dict(symbol=proposal['symbol'], qty=proposal['qty'],
                    contemporaneous={k:m.get(k) for k in ['candidate_rank','candidate_score','source_artifact_hash',
                        'capacity_forecast','capacity_native_qty','capacity_approved_qty','risk_per_share','target_notional']})
            funding[arm].append(dict(at=e['at'],actual_deployable_cash=e['actual_deployable_cash'],
                sizing_equity=e['sizing_equity'],gross_committed=e['account']['gross_committed'],
                proposals=proposals, decisions=e['decisions']))
    result = dict(at=datetime.now(timezone.utc).isoformat(),
        scope='Post-outcome causal extraction of a fixed accepted partial-context diagnostic pair. No new policy or financial replay.',
        pair_sha256=sha(pair_dir/'assessment.json'),causal_sha256=sha(pair_dir/'causal_review_v1/assessment.json'),
        script_sha256=sha(__file__),integrity_checks=checks,
        paired_results=pair['comparisons']['control_to_half'],
        target_states=context,action_day_funding_states=funding,
        limits=['Additive accounting and realised outcomes do not identify a new executable discriminator.',
                'Decision states include only information available at the recorded time; later exits are explicitly outcomes.',
                'Historical interval execution is checked, not exchange queues or untouched forward returns.'])
    (out/'assessment.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(output=str(out),artifact_checks=sum(len(x['artifacts']) for x in checks.values()),
        target_states=[{k:x[k] for k in ['symbol','action_day','early_close_minus_entry_in_original_R','actual_net_pnl_delta']} for x in context])))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('scenario',choices=['nominal','stress','capped'])
    args=p.parse_args();run(args.root,args.scenario)
