import copy
import importlib.util
import json
from pathlib import Path

import pytest

from backtests.strategies.kalcb.late_incremental_assessment_v1 import (
    quarter_fraction, forecast_index, calibration_group, passes_reserve,
)

KEY = '2026-04-01/TEST/minute60/q0.25'


def forecast():
    return {'opportunity_id': KEY, 'valid_from': '2026-03-31',
        'feature_available_at': '2026-04-01T10:00:00+09:00', 'prior':100,
        'signal_kind':'aligned48', 'population_eligible':True, 'calibration_supported':True,
        'baseline':6., 'combined':5., 'notional_forecast':{'baseline':60., 'combined':50.},
        'reserve_bps':{'baseline':48., 'combined':48.}, 'failure_probability':.0001, 'capacity_pass':True}


def load(tmp_path, rows):
    path=tmp_path/'scores.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    return forecast_index(path,{KEY:.1},100,'aligned48',{KEY:True})


def test_integer_quantity_denominator():
    row={'action':{'quantity':7},'features':{'action_notional_equity':.7}}
    assert quarter_fraction(row)==pytest.approx(.1)
    assert quarter_fraction(row)!=.7*.25


@pytest.mark.parametrize('qty',[0,1,3])
def test_no_zero_quantity_denominator(qty):
    with pytest.raises(ValueError):
        quarter_fraction({'action':{'quantity':qty},'features':{'action_notional_equity':.7}})


def test_unsupported_and_physical_rules():
    row=forecast(); assert passes_reserve(row)
    for changes in ({'calibration_supported':False}, {'population_eligible':False},
                    {'capacity_pass':False}, {'failure_probability':.001},
                    {'reserve_bps':{'baseline':60.,'combined':48.}}):
        assert not passes_reserve({**row,**changes})


def test_exact_units_and_duplicate_rejected(tmp_path):
    row=forecast(); assert load(tmp_path,[row])[KEY]['baseline']==6.
    with pytest.raises(ValueError,match='Duplicate'): load(tmp_path,[row,row])
    row['baseline']=60.
    with pytest.raises(ValueError,match='denominator'): load(tmp_path,[row])


def test_future_and_membership_poison(tmp_path):
    row=forecast(); row['valid_from']='2026-04-02'
    with pytest.raises(ValueError,match='Future'): load(tmp_path,[row])
    row=forecast(); row['population_eligible']=False
    with pytest.raises(ValueError,match='membership'): load(tmp_path,[row])


def test_independent_unit_conversion_and_invalid_retention():
    rows={KEY:forecast(), KEY+'bad':forecast()}
    labels={'baseline':{KEY:{'label_valid':True,'status':'filled','net_value_equity_bps':-2.},
        KEY+'bad':{'label_valid':False,'status':'physical_failure','net_value_equity_bps':999.}}}
    result=calibration_group(rows,labels,{KEY:.1,KEY+'bad':.1},list(rows),'baseline')
    assert result['invalid_labels_retained']==1
    assert result['realized_isolated_equity_bps_mean']==-2.
    assert result['realized_isolated_notional_bps_mean']==-20.
    assert result['equity_overprediction_bps_mean']==8.
    assert result['notional_overprediction_bps_mean']==80.


def assessment_module():
    path=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011/v15_incremental/assessment_v1/assess.py')
    spec=importlib.util.spec_from_file_location('incremental_assessment_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_alias_bootstrap_exact_zero_and_bad_pair():
    a=assessment_module()
    returns=[.01,-.02,.005,0,.02]*4
    for block in (5,10):
        proof=a.paired_blocks(returns,returns,block)
        assert proof['full_period_net_return_delta_pp_95_interval']==[0.,0.]
        assert proof['observed_delta_pp']==0.
    with pytest.raises(ValueError): a.paired_blocks(returns,returns[:-1],5)


def test_report_adapter_uses_incremental_validator_and_shared_lifecycle():
    from backtests.strategies.kalcb import late_incremental_full_diagnostics_v1 as report
    from backtests.strategies.kalcb import late_incremental_validation_v1 as validator
    from scripts import kalcb_late_incremental_reports_v1 as driver
    assert report.validate_native is validator.validate_native
    assert report.validate_case is validator.validate_case
    assert 'PhaseRunner' in driver.render.__code__.co_names
    assert report.LateIncrementalScreeningReportPlugin.run_enhanced_diagnostics is report.LateIncrementalScreeningReportPlugin.run_phase_diagnostics
