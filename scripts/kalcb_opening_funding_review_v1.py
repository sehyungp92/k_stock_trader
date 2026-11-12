"""Read-only decision checkpoint: observed opening demand and actual funding.

No alternative fill/PnL is fabricated here. IS/OOS dates label attribution;
they are never a rule or a selected replay population.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from backtests.strategies.kalcb.retention_research_io_v1 import read, write, bound, verify


def opening_batches(evidence, account):
    groups = defaultdict(list)
    seen = set()
    for row in evidence['entry_exposures']:
        if row['order_id'] in seen:
            raise ValueError('Duplicate physical order exposure')
        seen.add(row['order_id'])
        if row['action']['metadata']['entry_route'] == 'first30_open_anchor':
            groups[row['decision_available_at']].append(row)
    rejected = {r['order_id'] for r in account['funding']['rejected_buy_events']
                if r['reason'] == 'buying_power'}
    positions = account['positions']
    filled = {order for p in positions for order in p['entry_order_ids']}
    output = []
    for stamp, rows in sorted(groups.items()):
        first = rows[0]['account_before']
        equity = first['actual_equity']
        leverage = first['buying_power_leverage']
        if equity <= 0 or leverage != 2:
            raise ValueError('Unexpected parent funding contract')
        # All observed opening submissions occur before any opening fill. This
        # identity is checked, not silently assumed for future accounts.
        if any(r['account_before']['positions'] or
               abs(r['account_before']['actual_equity']-equity) > .01 or
               abs(r['account_before']['actual_cash']-equity) > .01 for r in rows):
            raise ValueError('Opening batch no longer flat/equal-capital')
        demand = sum(r['action']['qty']*r['action']['metadata']['entry_price_ref'] for r in rows)
        orders = {r['order_id'] for r in rows}
        lifetimes = [p for p in positions if orders.intersection(p['entry_order_ids'])]
        if len({p['position_id'] for p in lifetimes}) != len(lifetimes):
            raise ValueError('Duplicate lifetime attribution')
        output.append({'decision_available_at': stamp, 'equity_krw': equity,
            'planned_notional_krw': demand, 'planned_notional_over_equity': demand/equity,
            'funding_limit_multiple': leverage, 'planned_exceeds_funding': demand > leverage*equity,
            'orders': len(rows), 'order_ids': sorted(orders),
            'has_pending_buy_at_submission': any(any(o['side']=='BUY' for o in r['account_before']['pending_orders']) for r in rows),
            'unique_buying_power_rejections': len(orders & rejected),
            'orders_with_no_fill': len(orders-filled),
            'filled_position_lifetimes': len(lifetimes),
            'actual_lifetime_net_krw_once': sum(p['net_pnl_krw'] for p in lifetimes),
            'attribution_only_not_saved_pnl': True})
    return {'batches': output, 'summary': {
        'opening_batches': len(output),
        'batches_with_pending_buys': sum(r['has_pending_buy_at_submission'] for r in output),
        'planned_over_2x_batches': sum(r['planned_exceeds_funding'] for r in output),
        'max_planned_gross_multiple': max(r['planned_notional_over_equity'] for r in output),
        'unique_buying_power_rejections': sum(r['unique_buying_power_rejections'] for r in output),
        'rejected_orders_in_over_2x_batches': sum(r['unique_buying_power_rejections'] for r in output if r['planned_exceeds_funding']),
        'opening_orders_without_fill': sum(r['orders_with_no_fill'] for r in output)}}


def run(root, output):
    completion = read(root/'assessment_v1/completion.json')
    if not completion['complete'] or completion['native_cases'] != 8 or completion['exact_bridges'] != 8:
        raise ValueError('Incomplete firm checkpoint')
    verify([completion['assessment']])
    assessed = read(completion['assessment']['path'])
    verify(assessed['bindings'])
    inventory = root.parent/'v21_observer_assessment_preflight/parent_report_inventory.json'
    results = {}; bindings = [bound(inventory), completion['assessment'], bound(root/'assessment_v1/completion.json')]
    for name in ('control', 'control_prior125'):
        for period in ('is', 'oos'):
            for scenario in ('baseline', 'combined'):
                case = root/'cases'/name/period/scenario
                evidence_path = case/'observer_evidence.json'
                account_path = root/'assessment_v1/accounts'/name/period/(scenario+'.json')
                account = read(account_path)
                evidence = read(evidence_path)
                result = opening_batches(evidence, account)
                admitted = []
                for row in account['previews']['rows']:
                    for forecast in row['forecast_diagnostics']:
                        if forecast.get('admitted'):
                            admitted.append({'opportunity_id':row['opportunity_id'],
                                'recipe':forecast['recipe'], 'copied_submission':row['copied_submission'],
                                'reasons':row['reasons']})
                result.update(metrics=account['metrics'], activity=account['activity'],
                    forecasts_admitted_native_previews=admitted,
                    preview_counts={k:v for k,v in account['previews'].items() if k!='rows'})
                results['/'.join((name,period,scenario))] = result
                bindings += [bound(evidence_path), bound(account_path), bound(case/'identity_bridge.json'),
                             bound(case/'independent_account_verification.json'),bound(case/'reclaim_source_proof.json')]
    output.mkdir(parents=True, exist_ok=False)
    payload = {'created_at_utc':datetime.now(timezone.utc).isoformat(), 'accounts':results,
        'bindings':bindings, 'program':bound(__file__), 'study_complete':False,
        'judgment': {
            'unchanged_parents':[100,125], 'reclaim_entry_grid_supported':False,
            'broad_single_position_or_atr_cut_supported':False,
            'next_mechanism':'Investigate causal pending-entry notional reservations against the existing 2x funding boundary; preserve broker funding, signals, management and costs.',
            'why':'Opening demand can exceed actual funded capacity before simultaneous orders fill. OOS has more affected sessions and rejected orders. This is a mechanical intervention candidate, not proof of better returns.',
            'qualification':'Freeze one funding-boundary rule, prove native same-state intervention and changed-quantity full-cost execution before finite portfolio trials; no return promise or threshold grid.',
            'limitations':['Actual rejections include partially filled orders and are dependent.',
                'Observed batch PnL is descriptive, never hypothetical savings.',
                'Large planned orders preserve important IS winners and remain net positive OOS; no retrospective bucket exclusion.',
                'Existing parent full reports are authenticated reuse, not newly rendered observer reports.',
                'April-July is reused development; no independent validation or promotion.']}}
    write(output/'review.json',payload)
    lines=['V21 FIRM ECONOMIC CHECKPOINT', '',
        'All eight native identities and the read-only assessment completed with source/account/raw90 proofs.',
        'Keep unchanged prior100 and prior125. No new reclaim or exit bundle is qualified.', '',
        'Opening demand / unchanged 2x funded capacity:']
    for key,value in results.items():
        s=value['summary'];lines.append(f"{key}: {s['planned_over_2x_batches']}/{s['opening_batches']} opening batches over 2x; max {s['max_planned_gross_multiple']:.6f}x; {s['unique_buying_power_rejections']} unique rejected orders, {s['rejected_orders_in_over_2x_batches']} in those batches.")
    lines += ['',payload['judgment']['next_mechanism'],payload['judgment']['why'],*payload['judgment']['limitations']]
    (output/'report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(output/'completion.json',{'complete':True,'study_complete':False,'review':bound(output/'review.json'),'report':bound(output/'report.txt')})
    print(json.dumps({k:v['summary'] for k,v in results.items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.root,a.output)
