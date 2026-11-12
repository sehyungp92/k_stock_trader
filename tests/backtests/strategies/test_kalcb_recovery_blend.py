"""A forecast blend cannot silently combine incompatible causal policies."""
from copy import deepcopy
import pytest
from scripts.kalcb_recovery_blend_checkpoint_round import blend_values


def forecasts():
    row={'opportunity_id':'2026-04-01/005930/q0.5','trade_date':'2026-04-01',
         'valid_from':'2026-03-10','valid_through':'2026-04-06','quantity_scale':.5,
         'capacity_ok':True,'failure_probability':.0001,'baseline_value_bps':8.,'combined_value_bps':2.}
    return {row['opportunity_id']:row},{row['opportunity_id']:{**row,'baseline_value_bps':-4.,'combined_value_bps':-6.}}


@pytest.mark.parametrize('weight',[0.,1.])
def test_endpoints_exactly_reproduce_component_forecasts(weight):
    a,b=forecasts()
    assert blend_values(a,b,weight)==(a if weight==0 else b)


def test_blend_combines_signed_raw_cash_forecasts_without_mutating_inputs():
    a,b=forecasts();before=deepcopy((a,b));result=blend_values(a,b,.5)
    r=next(iter(result.values()))
    assert r['baseline_value_bps']==2. and r['combined_value_bps']==-2.
    assert (a,b)==before
    assert r['valid_from']=='2026-03-10' and r['quantity_scale']==.5


@pytest.mark.parametrize('field,value',[
    ('valid_from','2026-04-01'),('valid_through','2026-07-10'),('trade_date','2026-04-02'),
    ('quantity_scale',1.),('failure_probability',.8),('capacity_ok',False)])
def test_clock_quantity_and_physical_eligibility_mismatch_fails_closed(field,value):
    a,b=forecasts();next(iter(b.values()))[field]=value
    with pytest.raises(ValueError,match='Different decision clocks'):blend_values(a,b,.5)


def test_missing_actions_and_nonfinite_values_fail_closed():
    a,b=forecasts()
    with pytest.raises(ValueError,match='action populations'):blend_values(a,{},.5)
    next(iter(b.values()))['baseline_value_bps']=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):blend_values(a,b,.5)


def test_unregistered_parameter_is_rejected():
    a,b=forecasts()
    with pytest.raises(ValueError,match='Unregistered'):blend_values(a,b,.6)
