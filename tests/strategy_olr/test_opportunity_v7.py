import pytest
from scripts.olr_opportunity_v7 import cohort, summarize


def test_only_score_mapping_is_eligible():
    assert cohort(False, True, True, [], ['afternoon_score_band_rule_miss']) == 'score_mapping_only'
    assert cohort(False, True, True, ['afternoon_ret_below_floor'], ['afternoon_score_band_rule_miss']) == 'other_exclusion'
    assert cohort(False, False, True, [], ['afternoon_score_band_rule_miss']) == 'other_exclusion'
    assert cohort(False, True, False, [], ['afternoon_score_band_rule_miss']) == 'other_exclusion'


def test_no_f_or_late_confirmation_recycling():
    for reason in ('sector_participation_below_floor', 'no_pullback_trigger', 'late_confirmation', 'missing_completed_afternoon_bars'):
        assert cohort(False, True, True, [], [reason]) == 'other_exclusion'


def test_selected_must_pass_all_guards():
    assert cohort(True, True, True, [], []) == 'incumbent'
    with pytest.raises(ValueError):
        cohort(True, True, True, ['missing'], [])


def test_no_unfilled_alpha_or_portfolio_permission():
    result = summarize([], [])
    assert not result['payoff_study_eligible']
    assert not result['portfolio_experiment_eligible']
    assert not result['pnl_consumed']


def test_more_candidates_on_same_day_do_not_add_days():
    rows = [dict(day='2025-09-01', cohort='score_mapping_only') for _ in range(60)]
    rows += [dict(day='2025-09-01', cohort='incumbent')]
    result = summarize(rows, ['2025-09-01'])
    assert result['previously_empty_supply_days'] == 0
    assert result['additional_supply_days'] == 1
    assert not result['payoff_study_eligible']
