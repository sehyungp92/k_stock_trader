"""Complete finite blend/checkpoint/sizing review; no automatic promotion."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_full_oos_recovery import ROOT, read, write, bound
from scripts.kalcb_recovery_blend_checkpoint_round import DEST, ROUND, RECIPES, SIZING, check
from scripts.kalcb_recovery_stage_review import paired
from scripts.kalcb_recovery_report import segments


def concentration(directory):
    totals = defaultdict(float)
    for t in read(directory/'trades.json'):
        totals[t['symbol']] += t['net_pnl']
    ordered = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    total = sum(totals.values())
    return {'issuer_net_pnl_krw': dict(ordered), 'net_pnl_without_top_winning_issuers_krw': {
        str(k): total-sum(max(0, v) for _, v in ordered[:k]) for k in (1, 3, 5)},
        'interpretation': 'Attribution only, not a counterfactual account replay or a proposed issuer exclusion.'}


def review():
    if (ROOT/'stage5_review.json').exists():
        raise ValueError('Preserve completed review')
    check(DEST)
    if not read(DEST/'replay_completion.json')['complete']:
        raise ValueError('Incomplete registered round')
    prior3 = read(ROOT/'stage3_review.json'); prior4 = read(ROOT/'stage4_review.json')
    records = {p.parent.name: read(p) for p in (ROOT/ROUND/'cases').glob('*/record.json')}
    rejected = {p.parent.name: read(p) for p in (ROOT/ROUND/'cases').glob('*/rejected_record.json')}
    if set(records)|set(rejected) != {'control', *RECIPES, *SIZING}:
        raise ValueError('Incomplete registered trial population')
    parents = {'expanding': prior3['trials']['original_target_full_history'],
               'decay': prior4['trials']['original_decay60'], 'blend50': records['blend_decay50']}
    details = {}; comparisons = []
    for name, r in records.items():
        cells = {}
        for key, ref in r['cases'].items():
            directory = Path(ref['path']).parent
            result = read(directory/'result.json'); trades = read(directory/'trades.json')
            if not result['execution_validity_pass'] or any(result[k] for k in ('residual_sessions', 'terminal_inventory', 'terminal_pending')):
                raise ValueError('Invalid native screening record')
            gross = sum(t['gross_pnl'] for t in trades)
            fees = sum(t['commission'] for t in trades); net = sum(t['net_pnl'] for t in trades)
            if abs(gross-fees-net) > .01 or abs(net-result['return_pct']*1e6) > .01:
                raise ValueError('Cash mismatch')
            cells[key] = {'gross_after_slippage': gross, 'commission_tax': fees, 'net': net,
                          'audit_status': result['audit_status'], 'attribution': segments(directory),
                          'concentration': concentration(directory)}
        details[name] = {**r, 'cells': cells, 'paired_to_canonical': {
            s: paired(Path(r['cases']['oos_'+s]['path']).parent, ROOT/'control'/s) for s in ('baseline', 'combined')}}
        controls = (['expanding', 'decay'] if name.startswith('blend_decay') else
                    ['blend50'] if name == 'blend50_atr5' else ['decay'] if name.startswith('decay60') else
                    ['expanding'] if name == 'expanding_checkpoint_cal' else [])
        for control in controls:
            p = parents[control]
            comparisons.append({'name': name, 'control': p['recipe'], 'metric_delta': {
                k: r['metrics'][k]-p['metrics'][k] for k in r['metrics'] if k.endswith(('_return', '_dd', '_trades', '_active'))},
                'paired': {s: paired(Path(r['cases']['oos_'+s]['path']).parent,
                                    Path(p['cases']['oos_'+s]['path']).parent) for s in ('baseline', 'combined')}})
    all_records = [*read(ROOT/'stage2_review.json')['trials'],
                   *[r for n, r in prior3['trials'].items() if n != 'control'],
                   *[r for n, r in prior4['trials'].items() if n != 'control'],
                   *[r for n, r in details.items() if n != 'control']]
    objectives = [(f'oos_{s}_{m}', -1 if m == 'dd' else 1) for s in ('baseline', 'combined') for m in ('return', 'dd', 'trades')]
    def dominates(a, b):
        return all(sign*a[k] >= sign*b[k] for k, sign in objectives) and any(sign*a[k] > sign*b[k] for k, sign in objectives)
    frontier = [r['recipe'] for r in all_records if not any(dominates(t['metrics'], r['metrics']) for t in all_records)]
    findings = [
        'All eight registered recipes completed new native IS and full69-session OOS accounts. Source-audited finalist qualification is separate from these physical screening checks.',
        'Fixed expanding/decay blends do not monotonically interpolate portfolio performance. The25percent blend preserves much of IS return but remains worse than the expanding parent in both OOS returns. Increasing decay improves OOS baseline across25/50/75percent, at a substantial IS cost; central blend IS is weak. This is parameter instability, not evidence of an optimum.',
        'Decay60 ATR5 improves IS return and drawdown and combined OOS return/drawdown versus unscaled decay60. OOS baseline return drops from+0.6747% to-0.8296%, while drawdown falls from10.4047% to10.1218% and trades rise271to282. Combined OOS-9.5943%/13.9043%/293 is a material loss reduction relative to canonical-26.2801%/27.2333%/300, not positive stressed expectancy.',
        'Decay60 ATR4 reaches299/301 OOS baseline/combined trades and9.4977% baseline DD, but returns-3.0336%/-11.8099% and lowers IS return. Keep this neighborhood evidence and frequency tradeoff; do not discard only for missing a DD target.',
        'Checkpoint residual calibration on expanding slightly improves IS but worsens OOS return and DD. On decay it makes a small baseline gain to+1.2363% and reduces trades; combined OOS DD worsens. Candidate-level support is not yet direct broker-effective evidence. This small increment is not sufficient to prefer the extra mechanism.',
        'Freeze four representative qualification profiles: continuous original expanding, expanding withATR5, original decay60, and decay60 withATR5. These span exact IS preservation, moderate sizing, greater baseline OOS return, and improved stressed return/activity. They are qualification subjects, not promoted winners; reject inflation or unsupported increments.',
        'Retain all61 trial entries, including controls/aliases, component backouts, costs, concentration and paired block uncertainty. This count is not61 independent discoveries. Original April-July history is reused development. Canonical round1 and live trading remain unchanged.',
        'Qualification must bind source/config/model/label/calibration history; run baseline, participation, cost and combined for both periods; reconcile native fills/cash/inventory and verify screening bridges. Use actual new IS replay for expandingATR5. Enforce support/parameter/uncertainty evidence before any promotion.'
    ]
    payload = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'program': bound(__file__),
               'prior': bound(ROOT/'stage4_review.json'), 'completion': bound(DEST/'replay_completion.json'),
               'trials': details, 'rejected': rejected, 'component_comparisons': comparisons,
               'candidate_support': read(DEST/'build/candidate_support.json'),
               'oos_frontier_including_prior_trials': frontier, 'trial_entries_including_aliases': len(all_records),
               'findings': findings, 'scope': 'Reused development; independent100m IS/OOS accounts. Paired resampling is descriptive and unadjusted for adaptive selection.'}
    write(ROOT/'stage5_review.json', payload)
    lines = ['KALCB BLEND/CHECKPOINT/SIZING REVIEW', payload['scope'], '', *findings, '',
             'Recipe | IS baseline return/DD/trades | IS combined | OOS baseline | OOS combined']
    for name, r in details.items():
        m = r['metrics']; cells = [name]
        for period in ('is', 'oos'):
            for scenario in ('baseline', 'combined'):
                k = period+'_'+scenario
                cells.append(f"{m[k+'_return']:.4f}/{m[k+'_dd']:.4f}/{m[k+'_trades']}")
        lines.append(' | '.join(cells))
    (ROOT/'stage5_review.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'review': str(ROOT/'stage5_review.txt'), 'records': len(details), 'rejected': list(rejected), 'trial_entries': len(all_records)}), flush=True)


if __name__ == '__main__':
    review()
