from dataclasses import replace
from datetime import timedelta,time
import inspect
import pytest
from backtests.strategies.kalcb.reclaim_risk_population_v1 import auxiliary,labels


@pytest.mark.parametrize('ready,geometry,expected',[(True,False,True),(True,True,False),(False,False,False),(False,True,False)])
def test_auxiliary_population_excludes_target_and_no_exposure(ready,geometry,expected):
    assert auxiliary({'quarter_preview':{'ready':ready},'guard_at_rank1':geometry}) is expected


def test_risk_data_cannot_be_mislabeled_return_or_entry_evidence():
    source=inspect.getsource(labels)
    assert 'return_model_eligible=False' in source and 'reclaim_eligible=False' in source
    assert "geometry_labels_replayed':False" in source
    assert 'audited_episode(' in source and 'portfolio_cases' in source


@pytest.mark.parametrize('scenario',['baseline','combined'])
def test_same_native_order_has_identical_execution_with_different_geometry_annotation(scenario):
    from tests.backtests.strategies.test_kalcb_reclaim_labels_v1 import native_fixture
    from backtests.strategies.kalcb import executed_value as ev
    from scripts.kalcb_action_value_study import action_features
    from scripts.kalcb_executed_value_study import run_episode
    from strategy_kalcb.core.preview import preview_entry
    from strategy_kalcb.core.core_models import KALCBPortfolioView
    snap,prefix,cfg,at=native_fixture();cfg=replace(cfg,flatten_time=time(14,45),exit_hard_stop_enabled=False);day=str(snap.trade_date)
    preview=preview_entry(prefix,snap,cfg,KALCBPortfolioView(ev.REFERENCE_EQUITY,equity=ev.REFERENCE_EQUITY),available_at=at)
    assert preview['ready']
    row={'opportunity_id':day+'/005930/minute90/q0.25','base_opportunity_id':day+'/005930/minute90',
         'trade_date':day,'symbol':'005930','candidate':snap.candidates[0].to_json_dict(),'guard_at_rank1':True,
         'feature_available_at':at.isoformat(),'features':action_features(preview,prefix,.25),
         'action':{'ready':True,'quantity':sum(a['quantity'] for a in preview['intents'])}}
    replay=prefix+[replace(prefix[-1],timestamp=prefix[0].timestamp+timedelta(minutes=5*i),
        open=104.5,high=105,low=104.25,close=104.75,volume=100000) for i in range(18,78)]
    one=run_episode(row,replay,cfg,scenario,'test-source',None,[day])
    two=run_episode({**row,'guard_at_rank1':False},replay,cfg,scenario,'test-source',None,[day])
    assert one==two and one[0]['status']=='filled'


def test_failure_free_geometry_cannot_meet_unchanged_gate():
    from backtests.strategies.kalcb.action_value import FeasibilityModel
    import numpy as np
    for count in (1,100,308):
        model=FeasibilityModel().fit(np.zeros((count,1)),np.zeros(count,dtype=bool))
        assert model.predict(np.zeros((1,1)))[0]>=.001
