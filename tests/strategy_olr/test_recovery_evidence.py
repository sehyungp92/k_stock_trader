import pytest

from backtests.strategies.olr.recovery_evidence import HYGIENE_FIELDS,hygiene_metrics
from backtests.auto.shared.phase_runner import _metric_contract_criteria


def test_native_hygiene_counts_reach_shared_gates_without_suppressing_failures():
    clean={key:0 for key in HYGIENE_FIELDS}
    audits=[clean,{**clean,'invalid_order_rejection_count':1,'rejected_order_count':1}]
    metrics=hygiene_metrics(audits,[{'terminal_inventory':{}},{'terminal_inventory':{}}])
    metrics.update(primary_promotion_metric='return_pct',return_pct=55.,
        official_metric_basis='SimBroker.equity_curve_bar_level_mtm')
    gates={gate.name:gate.passed for gate in _metric_contract_criteria(metrics)}
    assert gates['hard_same_bar_fill_count']
    assert gates['hard_forced_replay_close_count']
    assert not gates['hard_invalid_order_rejection_count']
    assert metrics['invalid_order_rejection_count']==1


def test_missing_or_invalid_native_counts_cannot_be_reported_as_zero():
    clean={key:0 for key in HYGIENE_FIELDS}
    with pytest.raises(KeyError):
        hygiene_metrics([{}],[{'terminal_inventory':{}}])
    for invalid in (-1,float('nan'),.5):
        with pytest.raises(ValueError):
            hygiene_metrics([{**clean,'same_bar_fill_count':invalid}],[{'terminal_inventory':{}}])
