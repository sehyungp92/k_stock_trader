"""C12-D: read-only earned-capital fixed/VETO overlap, no counterfactual PnL."""
from pathlib import Path
import argparse
import json
import traceback
from kalcb_opportunity_closeout_v1 import bound, read, write, verify

REPO = Path(__file__).resolve().parents[1]
BASE = Path('E:/CodexResearch/k_stock_trader')
C6 = BASE / 'kalcb_capital_sensitivity_20261016/v2'
C10 = BASE / 'kalcb_opening_veto_qualification_20261016/v3'
C12 = BASE / 'kalcb_opportunity_lifecycle_20261016/funded_interaction_v1'


def main(out):
    out.mkdir(exist_ok=False)
    a, b = read(C6/'assessment.json'), read(C10/'assessment.json')
    bindings = [bound(__file__), bound(REPO/'scripts/kalcb_opportunity_closeout_v1.py'),
        bound(REPO/'docs/optimisation/kalcb/experiments/kalcb-exp-0036-allocation-interaction-design.md')]
    for root in (C6, C10):
        done = read(root/'completion.json'); assert done['complete']
        verify(done['assessment']); bindings += [bound(root/'completion.json'), done['assessment']]
    signal_binding = read(C12/'protocol.json')['signal_map']; verify(signal_binding); bindings.append(signal_binding)
    signals = read(signal_binding['path'])
    for stress in ('baseline', 'cost', 'combined'):
        assert a['comparisons'][stress]['qualified_matched_comparison'] and b['capital'][stress]['matched_qualified']
        for act in (a['activity']['fixed_'+stress], a['activity']['compound_'+stress], b['capital'][stress]['activity']):
            bindings += act['bindings']
    for binding in bindings: verify(binding)
    write(out/'registration.json', dict(bindings=bindings, scope='C12-D saved-account overlap, no new fit or policy/replay'))
    cells = {}
    for stress in ('baseline', 'cost', 'combined'):
        fixed = a['activity']['fixed_'+stress]['opportunities']
        prior = a['activity']['compound_'+stress]['opportunities']
        veto = b['capital'][stress]['activity']['opportunities']
        def flagged(k):
            symbol, at = k.split('/', 1); signal = signals.get(at[:10]+'/'+symbol)
            return at[11:16] == '09:25' and bool(signal and signal['predictions']['baseline'] < -23 and signal['predictions']['combined'] < -33)
        def useful(v): return v['material'] and v['net_pnl_krw'] > 0
        added = {k: dict(fixed=v, under_veto=veto.get(k), fixed_negative_opening=flagged(k),
                       veto_same_day_flags=sorted(s for s,z in signals.items() if s[:10]==k.split('/')[1][:10] and z['predictions']['baseline'] < -23 and z['predictions']['combined'] < -33))
                 for k,v in fixed.items() if k not in prior}
        fresh = {k:v for k,v in added.items() if useful(v['fixed'])}
        flagged_fixed = {k:v for k,v in fixed.items() if flagged(k)}
        rows = dict(all_fixed_added=added, fixed_added_profitable_material=fresh,
            fixed_added_profitable_material_absent_veto={k:v for k,v in fresh.items() if not v['under_veto']},
            fixed_negative_opening_campaigns=flagged_fixed,
            fixed_negative_opening_net=sum(v['net_pnl_krw'] for v in flagged_fixed.values()),
            fixed_negative_opening_profitable_material=sum(useful(v) for v in flagged_fixed.values()),
            fixed_summary=a['activity']['fixed_'+stress]['summary'], prior_summary=a['activity']['compound_'+stress]['summary'],
            veto_summary=b['capital'][stress]['activity']['summary'],
            fixed_net_vs_prior=a['activity']['fixed_'+stress]['summary']['net_pnl_krw']-a['activity']['compound_'+stress]['summary']['net_pnl_krw'],
            veto_net_vs_prior=b['capital'][stress]['net_delta_krw'])
        cells[stress] = rows
    for binding in bindings: verify(binding)
    write(out/'assessment.json', dict(cells=cells, participation='No matched comparison: compound/VETO paths unqualified.',
        scope='Observed saved account overlaps only. No sum or deletion is combined-policy performance; original VETO score is not a changed-quantity value prediction.',
        economic_objective_achieved=False))
    write(out/'completion.json', dict(complete=True, registration=bound(out/'registration.json'), assessment=bound(out/'assessment.json'), fits=0, native_accounts=0))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    try: main(args.output)
    except Exception as error:
        if args.output.exists(): write(args.output/'failure.json',dict(error=repr(error), traceback=traceback.format_exc()))
        raise
