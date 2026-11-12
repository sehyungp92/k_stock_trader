"""Preserve completed targeted evidence without overwriting the first-stage review."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.kalcb_full_oos_recovery import ROOT, read, write, bound
from scripts.kalcb_recovery_stage_review import paired


def run(root):
    if (root/'stage2_review.json').exists():
        raise ValueError('Preserve completed stage2 review; use a successor for new evidence')
    summary = read(root/'recovery_results.json')
    trials = []
    lookup = {}
    for group, data in summary['groups'].items():
        for trial in data['trials']:
            if trial['recipe'] == 'control':
                continue
            item = {**trial, 'group': group, 'paired_to_canonical': {}}
            for scenario in ('baseline', 'combined'):
                folder = Path(trial['cases']['oos_'+scenario]['path']).parent
                item['paired_to_canonical'][scenario] = paired(folder, root/'control'/scenario)
            trials.append(item); lookup[trial['recipe']] = item
    if len(trials) != 42 or len(summary['groups']['targeted_round']['trials']) != 17:
        raise ValueError('Expected complete registered26+16 trial population')
    objectives = [(f'oos_{s}_{m}', -1 if m == 'dd' else 1) for s in ('baseline', 'combined') for m in ('return', 'dd', 'trades')]
    def dominates(a, b):
        return all(sign*a[key] >= sign*b[key] for key, sign in objectives) and any(sign*a[key] > sign*b[key] for key, sign in objectives)
    frontier = [r['recipe'] for r in trials if not any(dominates(t['metrics'], r['metrics']) for t in trials)]
    contrasts = [('continuous_atr_quantity_5pct', 'continuous_all_models'),
                 ('continuous_atr_quantity_4pct', 'continuous_atr_quantity_5pct'),
                 ('continuous_atr5_no_mfe', 'continuous_atr_quantity_5pct'),
                 ('expanding_frozen_cal_atr5', 'scheduled_expanding_frozen_cal'),
                 ('continuous_cal_half', 'continuous_all_models'),
                 ('continuous_cal_frozen', 'continuous_all_models')]
    comparisons = []
    for candidate, component in contrasts:
        a, b = lookup[candidate], lookup[component]
        comparisons.append({'candidate': candidate, 'component_control': component,
            'metrics_delta': {k: a['metrics'][k]-b['metrics'][k] for k in a['metrics'] if k.endswith(('_return', '_dd', '_trades', '_active'))},
            'paired': {s: paired(Path(a['cases']['oos_'+s]['path']).parent, Path(b['cases']['oos_'+s]['path']).parent) for s in ('baseline', 'combined')}})
    findings = [
        'All42 noncontrol variants retain negative full69-session OOS baseline and combined returns. No profitable robust replacement is established.',
        'Continuous causal refits plus ATR4/5 sizing reduce losses and drawdown materially; fewer trades and lower historical IS returns remain explicit costs.',
        'Continuous ATR5: OOS -1.0389% return/11.4047% DD/219 trades; combined -9.7968%/13.8435%/225. Its historical ATR5 IS reference is66.6119%/7.5465%/369, combined40.0728%/8.1581%/373; these IS accounts do not validate an OOS-only changed fitting schedule.',
        'Expanding frozen-calibration ATR5 keeps activity above control:301/304 OOS trades, versus290/300, while return/DD improve to-7.2743%/13.1337% and-13.4348%/16.6870%. Its IS cost and deployment-clock interpretation remain.',
        'The tiny MFE-removal increment on continuous ATR5 worsens DD; retain as sparse exploratory component evidence, not an independently established improvement.',
        'Relaxing continuous calibration restores trades but worsens returns/DD. Greater activity alone does not recover alpha.',
        'Both DD preferences8/10 are soft: the Pareto list uses return, drawdown and trades, with no threshold rejection. It is an OOS development tradeoff list, not an IS-preserving or validated winner list.',
        'Paired circular-block sensitivity is descriptive and unadjusted for selection/multiple testing. Dropped-session contributions are attribution, never simulated tradable filters.',
        'Target-alignment follow-up separately changes economic outcomes/calibration and physical-risk outcomes, retaining one global history clock. Both IS and OOS will be newly replayed before interaction experiments.',
    ]
    payload = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'summary': bound(root/'recovery_results.json'), 'program': bound(__file__),
        'trial_count': len(trials), 'targeted_count': 16, 'findings': findings, 'oos_six_objective_pareto': frontier,
        'component_comparisons': comparisons, 'trials': trials,
        'qualification': 'Native screening executions passed; complete source/config/model/label qualification of variants is still pending. No canonical replacement.',
        'scope': 'Reused April1-July10 development evidence. Earlier stage1 artifacts retained unchanged.'}
    write(root/'stage2_review.json', payload)
    lines = ['KALCB COMPLETE TARGETED-STAGE REVIEW', payload['scope'], '', *findings, '',
        'Recipe | baseline return/DD/trades | combined return/DD/trades | five-day paired relative-wealth CI baseline/combined']
    for r in trials:
        m = r['metrics']; cells = [r['recipe']]
        for s in ('baseline', 'combined'):
            prefix = 'oos_'+s
            cells.append(f"{m[prefix+'_return']:.4f}/{m[prefix+'_dd']:.4f}/{m[prefix+'_trades']}")
        cells.extend(json.dumps(r['paired_to_canonical'][s]['paired_circular_block_resampling']['5']['relative_wealth_gain_ci95_pct']) for s in ('baseline', 'combined'))
        lines.append(' | '.join(cells))
    (root/'stage2_review.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'report': str(root/'stage2_review.txt'), 'trials': len(trials), 'pareto': frontier}))


if __name__ == '__main__':
    run(ROOT)
