from dataclasses import asdict
from datetime import date,timedelta
import numpy as np
import pytest
from backtests.strategies.kalcb.holding_action_v1 import FEATURES
from backtests.strategies.kalcb.holding_account_model_v1 import fit,choose,readiness,selected_contrasts


def sample(day,entry='one',value=2.,risk=0.):
    return {'observation':{'day':day,'features':{k:1. for k in FEATURES},'qty':10,'symbol':'a',
        'entry_order_id':entry,'available_at':day+'T10:00:00+09:00'},
        'contrasts':{arm:{'target':{'value_bps':value,'dd_change_bps':risk}} for arm in ('half','exit')}}


def test_actual_grouped_fit_and_maturity_support_rejection():
    days=[str(date(2025,1,1)+timedelta(days=i)) for i in range(60)]
    rows=[sample(d,str(i)) for d in days for i in range(2)]
    model=fit(rows,days)
    future=sample('2025-04-01')['observation']
    assert choose(model,future)['arm'] in ('half','exit')
    with pytest.raises(ValueError,match='Unmatured'):choose(model,rows[-1]['observation'])
    future['features']['book_weight']=2.
    assert choose(model,future)['arm']=='hold'
    with pytest.raises(ValueError,match='Immature'):fit(rows+[sample('2025-04-01')],days)


@pytest.mark.parametrize('value,risk,passes',[(2.,0.,True),(-2.,0.,False),(2.,1.,False)])
def test_readiness_uses_action_magnitude_and_risk_not_classification(value,risk,passes):
    rows=[{'day':str(date(2025,1,1)+timedelta(days=i//2)),
        'target':{'value_bps':value,'dd_change_bps':risk}} for i in range(40)]
    assert readiness(rows)['passed']==passes
    assert not readiness(rows[:5])['passed']


def test_overlapping_actions_are_not_summed_and_lifetimes_deduplicate():
    rows=[sample('2025-01-01','one'),sample('2025-01-01','two'),sample('2025-01-01','one')]
    rows[-1]['observation']['available_at']='2025-01-01T10:15:00+09:00'
    decisions=[{'arm':'half','predicted_value_bps':2.},{'arm':'exit','predicted_value_bps':1.},
        {'arm':'exit','predicted_value_bps':3.}]
    selected=selected_contrasts(rows,decisions)
    assert len(selected)==1 and selected[0]['entry_order_id']=='one'


def test_empty_readiness_is_failure():
    assert not readiness([])['passed']
