import pytest
from backtests.strategies.olr.economic_evidence import compare,suite_status,acquisition_by_opportunity


def case(**kwargs):
    return dict(grid='same-grid',valid=True,return_pct=57.,full_dd=8.,common_dd=7.9,
                trades=185,days=87,acquired={('old','day'):1_000_000},**kwargs)


def test_lower_drawdown_alone_does_not_qualify_a_participation_cut():
    control=case();candidate={**control,'return_pct':20.,'full_dd':5.,'common_dd':4.9,'trades':140,'days':70}
    result=compare(control,candidate)
    assert result['gates']['full_dd_lower']
    assert not result['gates']['return_preservation']
    assert not result['full_objective_cell_pass']


def test_one_successful_nominal_cell_is_not_a_qualified_successor():
    control=case();candidate={**control,'return_pct':58.,'full_dd':7.,'common_dd':6.9,'trades':186,'days':88}
    cell=compare(control,candidate)
    assert cell['full_objective_cell_pass']
    assert not suite_status([cell],retained_reference_gates_pass=True,forecast_validation_pass=True)['funded_successor_qualified']


def test_mismatched_risk_grids_are_rejected():
    with pytest.raises(ValueError,match='different comparison grids'):
        compare(case(),{**case(),'grid':'other'})


def test_token_entries_do_not_count_as_meaningful_frequency_success():
    control=case();candidate={**control,'return_pct':58.,'full_dd':7.,'common_dd':6.9,
        'trades':186,'days':88,'acquired':{**control['acquired'],('new','day'):100.}}
    result=compare(control,candidate,window='oos')
    assert result['gates']['completed_entries']
    assert not result['gates']['meaningful_incremental_notional']
    assert not result['full_objective_cell_pass']


def test_complete_cells_cannot_override_failed_forecast_or_retained_gates():
    cells=[dict(window=w,scenario=s,full_objective_cell_pass=True)
           for w in ('train','oos') for s in ('baseline','combined','baseline_capped')]
    assert not suite_status(cells,retained_reference_gates_pass=True)['funded_successor_qualified']
    assert not suite_status(cells,forecast_validation_pass=True)['funded_successor_qualified']


def test_waiting_same_signal_is_not_counted_as_a_new_opportunity():
    before=dict(symbol='005930',entry_decision_time='2025-09-01T14:30:00+09:00',qty=100,entry_price=10000)
    after={**before,'entry_decision_time':'2025-09-01T14:40:00+09:00'}
    control={**case(),'acquired':acquisition_by_opportunity([before])}
    candidate={**case(),'acquired':acquisition_by_opportunity([after])}
    result=compare(control,candidate,window='oos')
    assert result['notional']==dict(added=0,removed=0,net=0,median_added=0)
    assert not result['gates']['meaningful_incremental_notional']


def test_duplicate_session_lifecycles_cannot_be_silently_merged():
    trade=dict(symbol='005930',entry_decision_time='2025-09-01T14:30:00+09:00',qty=100,entry_price=10000)
    with pytest.raises(ValueError,match='explicit opportunity lineage'):
        acquisition_by_opportunity([trade,trade])


def test_future_journal_export_preserves_all_mark_phases_without_changing_fills():
    from scripts.olr_economic_study import retain_full_mark_journal
    old="if row['event'] in ('minute_day_scope_v5','minute_exit_signal_v5','fill','submit','cancel','expire','reject'): pass"
    new=retain_full_mark_journal(old)
    assert "'opening_marks','closing_marks'" in new
    assert "'fill','submit','cancel','expire','reject'" in new
    assert retain_full_mark_journal(new)==new
    with pytest.raises(ValueError,match='export contract'):
        retain_full_mark_journal('unexpected exporter')
