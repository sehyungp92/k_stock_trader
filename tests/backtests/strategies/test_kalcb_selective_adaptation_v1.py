from copy import deepcopy
from datetime import date, timedelta
import pytest
from backtests.strategies.kalcb.selective_adaptation_v1 import residual_fit, corrected, supported_quantity, descriptors


def fixture_data():
    sessions = [str(date(2025, 1, 1)+timedelta(days=i)) for i in range(90)]
    obs = []; labels = {s: {} for s in ('baseline', 'combined')}
    for i, d in enumerate(sessions):
        k = d+'/A/q1'
        obs.append(dict(opportunity_id=k, base_opportunity_id=d+'/A', trade_date=d, valid_from=d,
                        quantity_scale=1., tier='top2', agreement='agree', baseline_value_bps=20., combined_value_bps=10.))
        for s in labels:
            labels[s][k] = dict(label_valid=True, label_available_at=d+'T16:00:00+09:00', net_value_equity_bps=-5.)
    return sessions, obs, labels


def test_future_and_embargo_outcomes_do_not_change_prefix():
    sessions, obs, labels = fixture_data()
    first = sessions[60]; a = residual_fit(obs, labels, sessions, first)
    altered = deepcopy(labels)
    for s in altered:
        for r in obs[59:]: altered[s][r['opportunity_id']]['net_value_equity_bps'] = 1e12
    assert a == residual_fit(obs, altered, sessions, first)
    assert a['latest_label_available_at'] < first


def test_immature_old_label_is_not_used():
    sessions, obs, labels = fixture_data()
    k = obs[40]['opportunity_id']
    for s in labels: labels[s][k]['label_available_at'] = sessions[70]+'T16:00:00+09:00'
    a = residual_fit(obs, labels, sessions, sessions[60])
    for s in labels: labels[s][k]['net_value_equity_bps'] = 1e12
    assert a == residual_fit(obs, labels, sessions, sessions[60])


def test_duplicate_quantity_observations_rejected():
    sessions, obs, labels = fixture_data()
    with pytest.raises(ValueError, match='Dependent'):
        residual_fit(obs+[obs[40]], labels, sessions, sessions[60])


def test_sparse_history_identity_and_supported_negative_correction():
    sessions, obs, labels = fixture_data()
    a = residual_fit(obs, labels, sessions, sessions[5])
    b = residual_fit(obs, labels, sessions, sessions[60])
    key = 'baseline/q1/top2/agree'
    assert a['cohorts'][key]['offset_bps'] == 0
    assert not a['cohorts'][key]['supported']
    assert b['cohorts'][key]['offset_bps'] < 0
    assert b['cohorts'][key]['supported']


def test_negative_mature_losses_are_retained():
    sessions, obs, labels = fixture_data()
    a = residual_fit(obs, labels, sessions, sessions[60])
    for s in labels: labels[s][obs[40]['opportunity_id']]['net_value_equity_bps'] = -10000
    b = residual_fit(obs, labels, sessions, sessions[60])
    assert b['cohorts']['baseline/q1/top2/agree']['offset_bps'] < a['cohorts']['baseline/q1/top2/agree']['offset_bps']


def test_quantity_budget_only_accepts_observed_grid():
    row = {'candidate': {'daily_atr': 10., 'prior_day_close': 100.}, 'features': {'action_quantity_scale': .5}}
    assert supported_quantity(row)
    row['features']['action_quantity_scale'] = 1.
    assert not supported_quantity(row)
    row['features']['action_quantity_scale'] = .4
    with pytest.raises(ValueError, match='Unlabelled'): supported_quantity(row)


def test_descriptors_are_outcome_free_and_quantity_grouped():
    items = [dict(opportunity_id='x'+str(q), base_opportunity_id='x', features={'action_quantity_scale': q}) for q in (.25,.5,1.)]
    values = {r['opportunity_id']: {'baseline_value_bps': 1., 'combined_value_bps': 1.} for r in items}
    other = {k: {'baseline_value_bps': 1., 'combined_value_bps': -1.} for k in values}
    ds = descriptors(items, values, other)
    assert {d['rank'] for d in ds.values()} == {1}
    assert {d['agreement'] for d in ds.values()} == {'disagree'}


def test_invalid_native_case_is_retained_unscored(tmp_path):
    from backtests.strategies.kalcb.selective_recovery_plugin_v1 import SelectiveRecoveryPlugin
    plugin = object.__new__(SelectiveRecoveryPlugin)
    plugin.root = tmp_path; plugin.records = {}; plugin.spec = {'recipes': {'bad': {}}}
    plugin.native_case = lambda *args: {'research_rejected': True, 'error': 'unattainable fill'}
    record = plugin.evaluate_names(['bad'])[0]
    assert record['rejected'] and not record['metrics']['physical_valid']
    assert record['cases'] == {} and len(record['failures']) == 4


def test_implementation_exception_stops_framework_evaluation(tmp_path):
    from backtests.strategies.kalcb.selective_recovery_plugin_v1 import SelectiveRecoveryPlugin
    plugin = object.__new__(SelectiveRecoveryPlugin)
    plugin.root = tmp_path; plugin.records = {}; plugin.spec = {'recipes': {'bad': {}}}
    def fail(*args): raise RuntimeError('missing evidence')
    plugin.native_case = fail
    with pytest.raises(RuntimeError, match='missing evidence'): plugin.evaluate_names(['bad'])
