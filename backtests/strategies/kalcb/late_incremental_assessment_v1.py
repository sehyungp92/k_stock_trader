"""Read-only unit and support attribution for the frozen incremental experiment.

Outcome conversion applies only to isolated labels, never funded portfolio PnL.
No producers or native workers are imported by this assessment.
"""
from collections import Counter
from datetime import datetime
from pathlib import Path
import json
import math
from statistics import mean

from .retention_research_io_v1 import read, bound, verify

SCENARIOS = ('baseline', 'combined')
PERIODS = {'is': ('2025-08-07', '2026-03-31'), 'oos': ('2026-04-01', '2026-07-10')}


def lines(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def quarter_fraction(row):
    """Match integer native quantity rounding, not an approximate 25% exposure."""
    quantity = int(row['action']['quantity'])
    if quantity <= 0:
        raise ValueError('Nonpositive planned quantity')
    fraction = float(row['features']['action_notional_equity']) * int(quantity * .25) / quantity
    if not math.isfinite(fraction) or fraction <= 0:
        raise ValueError('Nonpositive planned quarter notional')
    return fraction


def forecast_index(path, fractions, prior, kind, eligibility):
    result = {}
    for row in lines(path):
        key = row['opportunity_id']
        if key in result or not key.endswith('/q0.25'):
            raise ValueError('Duplicate or wrong-quantity forecast')
        if row['valid_from'] > key[:10] or row['feature_available_at'][:10] != key[:10]:
            raise ValueError('Future fit or mismatched feature day')
        if row['prior'] != prior or row['signal_kind'] != kind:
            raise ValueError('Wrong prior or signal lineage')
        if row['population_eligible'] != (eligibility.get(key) is True):
            raise ValueError('Population membership drift')
        fraction = fractions[key]
        for scenario in SCENARIOS:
            equity = float(row[scenario]); notional = float(row['notional_forecast'][scenario])
            if not all(math.isfinite(x) for x in (equity, notional)) or not math.isclose(equity, notional*fraction, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError('Equity/notional forecast denominator drift')
            reserve = row['reserve_bps'][scenario]
            if reserve is not None and (not math.isfinite(reserve) or reserve < 0):
                raise ValueError('Invalid reserve')
            if row['calibration_supported'] and reserve is None:
                raise ValueError('Supported map missing reserve')
        result[key] = row
    return result


def passes_reserve(row):
    return (row['calibration_supported'] and row['population_eligible'] and
            row['failure_probability'] < .001 and row['capacity_pass'] and
            all(row['reserve_bps'][s] is not None and row['notional_forecast'][s] > row['reserve_bps'][s] for s in SCENARIOS))


def calibration_group(forecasts, labels, fractions, keys, scenario):
    valid = [k for k in keys if labels[scenario][k].get('label_valid')]
    avg = lambda values: mean(values) if values else None
    predicted = [forecasts[k][scenario] for k in valid]
    outcomes = [labels[scenario][k]['net_value_equity_bps'] for k in valid]
    predicted_notional = [forecasts[k]['notional_forecast'][scenario] for k in valid]
    outcomes_notional = [labels[scenario][k]['net_value_equity_bps']/fractions[k] for k in valid]
    return {'forecast_rows': len(keys), 'valid_labels': len(valid),
        'invalid_labels_retained': len(keys)-len(valid),
        'isolated_status_counts': dict(Counter(labels[scenario][k]['status'] for k in keys)),
        'predicted_equity_bps_mean': avg(predicted), 'realized_isolated_equity_bps_mean': avg(outcomes),
        'predicted_notional_bps_mean': avg(predicted_notional), 'realized_isolated_notional_bps_mean': avg(outcomes_notional),
        'equity_overprediction_bps_mean': avg([p-y for p,y in zip(predicted,outcomes)]),
        'notional_overprediction_bps_mean': avg([p-y for p,y in zip(predicted_notional,outcomes_notional)]),
        'positive_isolated_share': avg([x>0 for x in outcomes])}


def label_calibration(root, recipes):
    root = Path(root); protocol = read(root/'protocol.json'); source = Path(protocol['label_root'])
    done = read(root/'models/completion.json')
    if not done['complete']:
        raise ValueError('Incomplete models')
    verify([{'path': str(root/'models'/f), 'sha256': h} for f,h in done['files'].items()])
    prepared = read(root/'prepared/prepared.json')
    if not prepared['complete'] or prepared['first_oos_prior_sessions'] != 217:
        raise ValueError('Incomplete continuous preparation')
    verify([{'path': str(root/'prepared/opportunities.jsonl'), 'sha256': prepared['opportunities_sha256']}])
    fractions = {}
    for row in lines(root/'prepared/opportunities.jsonl'):
        if not row['action']['ready'] or not row['guard_at_rank1']:
            continue
        key = row['opportunity_id'] + '/q0.25'
        if key in fractions:
            raise ValueError('Duplicate prepared action')
        fractions[key] = quarter_fraction(row)
    labels = {s:{} for s in SCENARIOS}; label_bindings = []
    for scenario in SCENARIOS:
        folder = source/'labels'/scenario; manifest = read(folder/'completion.json')
        if not manifest['complete'] or manifest['smoke'] or manifest['scenario'] != scenario:
            raise ValueError('Wrong full-cost label scope')
        verify([{'path':str(folder/f),'sha256':h} for f,h in manifest['files'].items()])
        label_bindings.append(bound(folder/'completion.json'))
        for file in manifest['files']:
            if not file.endswith('.jsonl'):
                continue
            for row in lines(folder/file):
                key = row['opportunity_id']
                if key in labels[scenario]:
                    raise ValueError('Duplicate quantity label')
                labels[scenario][key] = row
    eligibility = read(root/'models/eligibility.json')
    chronology = read(root/'models/chronology.json')
    for proof in chronology['calibration']:
        latest = proof.get('latest_label_available_at') or proof.get('label_available_through')
        if latest and latest[:10] >= proof['valid_from']:
            raise ValueError('Immature calibration outcome')
    groups = []; funnels = []; selected = {}; score_bindings = []
    for name, recipe in recipes.items():
        if not recipe['quantity']:
            continue
        prior = recipe['prior']; kind = recipe['signal']
        model_path = root/'models'/f'{prior}_{kind}.jsonl'
        fc = forecast_index(model_path, fractions, prior, kind, eligibility[str(prior)]['mapping'])
        score_bindings.append(bound(model_path)); selected[name] = {}
        for key in fc:
            if any(key not in labels[s] or labels[s][key]['quantity_scale'] != .25 for s in SCENARIOS):
                raise ValueError('Missing exact-quarter label')
        for period, (first,last) in PERIODS.items():
            keys = [k for k in fc if first <= k[:10] <= last]
            eligible = [k for k in keys if fc[k]['population_eligible']]
            physical = [k for k in eligible if fc[k]['failure_probability'] < .001 and fc[k]['capacity_pass']]
            positive = [k for k in physical if all(fc[k]['notional_forecast'][s] > 0 for s in SCENARIOS)]
            supported = [k for k in physical if fc[k]['calibration_supported']]
            qualified = [k for k in keys if passes_reserve(fc[k])]
            admissions = list(lines(root/'policies'/name/period/'later60.new.jsonl'))
            if len({a['action_id'] for a in admissions}) != len(admissions):
                raise ValueError('Duplicate policy admission')
            for admission in admissions:
                key = admission['action_id']; score = fc[key]
                if key not in qualified or admission['quantity_scale'] != .25 or admission['signal_kind'] != kind:
                    raise ValueError('Unsupported policy admission')
                if score['feature_available_at'] > admission['available_at'] or score['valid_from'] != admission['valid_from']:
                    raise ValueError('Admission timing drift')
            selected[name][period] = admissions
            funnels.append({'name':name, 'period':period, 'forecast_rows':len(keys),
                'known_new_snapshot_issuer_rows':len(eligible), 'physical_capacity_pass_rows':len(physical),
                'both_positive_physical_rows':len(positive), 'calibration_reserve_supported_physical_rows':len(supported),
                'reserve_pass_rows_before_final_preview':len(qualified), 'policy_admitted_issuer_days':len(admissions),
                'positive_forecasts':[{'action_id':k,'notional_forecast':fc[k]['notional_forecast'],
                    'reserve_bps':fc[k]['reserve_bps'],'supported':fc[k]['calibration_supported']} for k in positive],
                'scope':'Nested counts except positive/support branches. Snapshot eligibility is not actual account availability. Final rank and native broker gates remain binding.'})
            for population in (True, False):
                for support in (True, False):
                    subset = [k for k in keys if fc[k]['population_eligible']==population and fc[k]['calibration_supported']==support]
                    for scenario in SCENARIOS:
                        groups.append({'name':name, 'prior':prior, 'signal_kind':kind, 'period':period, 'scenario':scenario,
                            'known_new_snapshot_issuer':population, 'calibration_and_reserve_supported':support,
                            **calibration_group(fc,labels,fractions,subset,scenario)})
    return {'scope':'Descriptive exact-quarter isolated outcomes. Equity and planned-notional bps are reported separately. Never funded recoverable PnL or an independent sample across recipes.',
        'unit_rule':'quarter_fraction = original planned notional/equity * floor(original quantity*0.25)/original quantity; isolated notional bps = isolated equity bps / quarter_fraction',
        'all_forecast_rows_by_population_and_support':groups, 'selection_funnels':funnels, 'policy_selected_actions':selected,
        'incremental_support_proofs':[p for p in chronology['calibration'] if p.get('prior') in (100,125)],
        'calibration_limit':'Reserve estimates conditional session-mean overprediction, not a calibrated individual prediction interval.',
        'models':bound(root/'models/completion.json'), 'model_scores':score_bindings,
        'chronology':bound(root/'models/chronology.json'), 'eligibility':bound(root/'models/eligibility.json'),
        'prepared':bound(root/'prepared/prepared.json'), 'label_manifests':label_bindings}
