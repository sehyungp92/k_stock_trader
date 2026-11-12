"""Describe a completed fixed-feature comparison using existing branch ledgers.

No policy search, refit, new outcome generation or retrospective eligibility.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtests.strategies.kalcb.retention_research_io_v1 import read, write, bound, verify

OLD = Path('E:/CodexResearch/k_stock_trader/holding_account_20261013/completion_assessment_v1')
FIELDS = ('value_bps','dd_change_bps','own_initial_bps','other_initial_bps',
          'preexisting_pending_bps','new_descendant_bps','fee_change_bps',
          'gross_occupancy_change','mean_cash_change_krw','material_entry_change')


def key(r):
    return r['day'], r.get('clock',r.get('available_at')), r['entry_order_id'], r['arm']


def summarise(rows):
    by_day = defaultdict(list)
    for r in rows:
        by_day[r['day']].append(r)
    means = {f:sum(sum(r[f] for r in v)/len(v) for v in by_day.values())/len(by_day)
             if by_day else None for f in FIELDS}
    return dict(actions=len(rows), days=len(by_day), mean_by_session=means,
                positive_continuations_cut=sum(r['hold_own_remaining_pnl_krw']>0 for r in rows),
                recovery_paths_cut=sum(r['features']['return_atr']<0 and r['hold_own_remaining_pnl_krw']>0 for r in rows),
                parent_top_decile_winners_cut=sum(r['parent_top_decile_winner'] for r in rows),
                nonbinding_native_buying_power=sum(r['features']['buying_power_fraction']>=1 for r in rows),
                changed_material_entry_count=sum(r['material_entry_change']!=0 for r in rows),
                pending_remainders_cancelled=sum(r['cancelled_buy_remainder'] for r in rows),
                partial_action_fill_count=sum(r['delay_last_minutes']>0 for r in rows),
                intended_vs_actual_quantity_mismatches=sum(r['intended_qty']!=r['actual_action_qty'] for r in rows))


def main(root):
    result = read(root/'result.json')
    if not result['completed']:
        raise ValueError('Comparison incomplete')
    inputs = [bound(root/'result.json'),bound(__file__)]; reports = {}; checked = {}
    for scenario in result['reports']:
        pp = root/'fit'/(scenario+'_predictions.json'); ap = OLD/('is_'+scenario+'_actions.jsonl.gz')
        inputs.extend((bound(pp),bound(ap)))
        predictions = read(pp)
        with gzip.open(ap, 'rt', encoding='utf-8') as f:
            details = {key(r):r for r in map(json.loads,f)}
        reports[scenario] = {}
        population_comparators = {arm:summarise([r for r in details.values() if r['arm']==arm and r['mature']])
                                  for arm in ('half','exit')}
        for policy in ('A0','A1'):
            selected = predictions['selected_'+policy]; rows = []
            for s in selected:
                r = details[key(s)]
                for field in ('value_bps','dd_change_bps','material_entry_change','fee_change_krw'):
                    if r[field] != s['target'][field]:
                        raise ValueError('Attribution target mismatch')
                for field in ('hold_binding','action_binding'):
                    b = r[field]
                    if b['path'] not in checked:
                        verify([b]); checked[b['path']] = b
                if abs(sum(r['components_krw'].values())-r['whole_account_delta_krw'])>.011:
                    raise ValueError('Component decomposition does not reconcile')
                rows.append(r)
            report = summarise(rows)
            by_refit = defaultdict(list)
            for s in selected:
                by_refit[s['decision']['training_through']].append(details[key(s)])
            report['by_refit'] = {d:summarise(v) for d,v in sorted(by_refit.items())}
            tail = sorted(zip(rows,selected),key=lambda x:x[0]['value_bps'])
            report['worst_three'] = [dict(day=r['day'],symbol=r['symbol'],clock=r['clock'],arm=r['arm'],
                predicted_wealth_bps=s['decision']['predicted_value_bps'], actual_wealth_bps=r['value_bps'],
                actual_net_krw=r['whole_account_delta_krw'], hold_remaining_krw=r['hold_own_remaining_pnl_krw'],
                components_krw=r['components_krw'], fee_change_krw=r['fee_change_krw']) for r,s in tail[:3]]
            reports[scenario][policy] = report
        reports[scenario]['same_mature_population'] = population_comparators
    inputs.extend(checked.values()); verify(inputs)
    out = root/'attribution.json'
    if out.exists():
        raise ValueError('Preserve earlier attribution')
    write(out,dict(at_utc=datetime.now(timezone.utc).isoformat(), reports=reports,inputs=inputs,
                   selected_branch_bindings_verified=len(checked),
                   scope='Existing authenticated suffix attribution, selected by the fixed prospective policies. Local values overlap; no additive portfolio return. Hindsight winner/recovery categories are descriptive only.'))
    print(json.dumps({s:{p:d[p]['mean_by_session'] for p in ('A0','A1')} for s,d in reports.items()}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
