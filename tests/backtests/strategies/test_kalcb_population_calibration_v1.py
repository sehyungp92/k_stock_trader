"""Population support and information-availability contracts, before any replay."""
from copy import deepcopy
from datetime import date, timedelta
import pytest
from backtests.strategies.kalcb.population_calibration_v1 import (
    fit_population, historical_record, descriptor, corrected_values, cohort_key)

SESSIONS = [(date(2025, 1, 1) + timedelta(days=i)).isoformat() for i in range(80)]
FIRST = SESSIONS[40]


def row(i, *, q=.25, core=2., support=10., tier='top2', value=6.):
    d = SESSIONS[i]
    return {'family': 'expanding', 'day': d, 'action_id': f'{d}/TEST/q{q:g}',
            'quantity': q, 'descriptor': {'tier': tier}, 'available_at': d + 'T09:30:00+09:00',
            'valid_from': d, 'failure_probability': .0001, 'capacity_pass': True,
            'calibrated_core': {'baseline': core, 'combined': core},
            'calibrated_support': {'baseline': support, 'combined': support},
            'selected_management_labels': {s: {'label_valid': True,
                'label_available_at': d + 'T15:35:00+09:00', 'net_value_equity_bps': value}
                for s in ('baseline', 'combined')}}


def records(rs, origin='frozen_core'):
    return [historical_record(x, x, origin) for x in rs]


def fit(rs):
    return fit_population(rs, 'expanding', FIRST, SESSIONS)


def cell(proof, r=None, origin='frozen_core'):
    r = r or row(40)
    return proof['cohorts']['baseline/' + cohort_key(descriptor(r, r, origin))]


def test_current_future_and_embargoed_outcomes_cannot_change_fit():
    base = records([row(i) for i in range(20)])
    proof = fit(base)
    # All values poisonously large; none was available for this fit.
    later = records([row(i, value=float('nan')) for i in (39, 40, 41, 60)])
    assert fit(base + later) == proof


def test_immature_outcomes_are_excluded_before_value_inspection():
    base = records([row(i) for i in range(20)])
    late = records([row(30, value=float('nan'))])[0]
    for label in late['labels'].values():
        label['label_available_at'] = FIRST + 'T15:35:00+09:00'
    candidate = fit(base + [late])
    assert candidate['cohorts'] == fit(base)['cohorts']
    assert candidate['exclusions']['baseline_immature'] == 1


def test_no_borrowing_across_forecast_signs_origins_or_quantity():
    original = [row(i) for i in range(20)]
    base = records(original)
    expected = cell(fit(base))
    negatives = [row(i, q=.5, core=-2., support=-2., value=-20.) for i in range(20)]
    other_quantity = [row(i, q=1., value=100.) for i in range(20)]
    combined = fit(base + records(negatives) + records(other_quantity) + records(original, 'new_forecast'))
    assert cell(combined) == expected
    assert cell(combined, origin='new_forecast')['offset_bps'] == -expected['offset_bps']
    assert expected['pool_issuer_days'] == 20


def test_local_unsupported_cell_is_identity_despite_supported_pool():
    base = records([row(i) for i in range(20)] + [row(i, tier='rest') for i in range(20, 27)])
    c = cell(fit(base), row(40, tier='rest'))
    assert c['pool_issuer_days'] == 27 and c['cell_issuer_days'] == 7
    assert not c['supported'] and c['offset_bps'] == 0.


def test_duplicate_population_action_is_rejected():
    base = records([row(i) for i in range(20)])
    with pytest.raises(ValueError, match='Duplicate'):
        fit(base + [deepcopy(base[0])])


def test_physical_gate_and_invalid_labels_are_preserved():
    base = records([row(i) for i in range(20)])
    bad = records([row(i, value=99999.) for i in range(20, 24)])
    bad[0]['failure_probability'] = .001
    bad[1]['capacity_pass'] = False
    for record in bad[2:]:
        for label in record['labels'].values():
            label['label_valid'] = False
    assert fit(base + bad)['cohorts'] == fit(base)['cohorts']


def test_forecast_and_label_clocks_fail_closed():
    r = records([row(2)])[0]
    r['valid_from'] = FIRST
    with pytest.raises(ValueError, match='Future forecast'):
        fit([r])
    r = records([row(2)])[0]
    r['labels']['baseline']['label_available_at'] = '2025-01-03T15:35:00'
    with pytest.raises(ValueError, match='Naive'):
        fit([r])


def test_decision_application_needs_no_outcomes_and_obeys_frozen_clock():
    proof = fit(records([row(i) for i in range(20)]))
    r = row(40)
    del r['selected_management_labels']
    result = corrected_values(r, r, 'frozen_core', proof)
    assert result['baseline'] > 2.
    future = row(45)
    with pytest.raises(ValueError, match='clock'):
        corrected_values(future, future, 'frozen_core', proof)


def test_counterpart_forecast_must_be_available_at_same_decision():
    r, other = row(40), row(40)
    other['valid_from'] = SESSIONS[41]
    with pytest.raises(ValueError, match='Future decision'):
        descriptor(r, other, 'frozen_core')


def test_valid_outcome_cannot_precede_its_candidate():
    r = records([row(2)])[0]
    r['labels']['baseline']['label_available_at'] = SESSIONS[2] + 'T08:00:00+09:00'
    with pytest.raises(ValueError, match='precedes'):
        fit([r])
