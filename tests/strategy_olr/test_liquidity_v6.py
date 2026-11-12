from copy import deepcopy
from datetime import datetime
import pytest
from strategy_olr.core.liquidity_v6 import CausalLiquidityPanel, request_bound

def fixture():
    days=['2026-01-05','2026-01-06','2026-01-07','2026-01-08','2026-01-09']
    rows=[dict(day=d,symbol=s,prefix_volume=1000,auction_volume=100,
               prefix_available_at=d+'T14:30:00+09:00',auction_available_at=d+'T15:35:00+09:00',
               source_hash='authenticated-fixture') for d in days for s in ('A','B')]
    return rows,days,['A','B']

@pytest.mark.parametrize('model',['broad_median','normalized_shrinkage'])
def test_future_observations_cannot_change_prediction(model):
    rows,days,names=fixture();a=CausalLiquidityPanel(rows,days,names)
    for r in rows:
        if r['day']>='2026-01-08':r['auction_volume']=999999
        if r['day']>'2026-01-08':r['prefix_volume']=888888
    b=CausalLiquidityPanel(rows,days,names)
    assert a.forecast('2026-01-08','A',model)==b.forecast('2026-01-08','A',model)

def test_policy_and_source_order_invariance():
    rows,days,names=fixture();a=CausalLiquidityPanel(rows,days,names)
    b=CausalLiquidityPanel(list(reversed(rows)),days,list(reversed(names)))
    assert a.forecast(days[-1],'A','normalized_shrinkage')==b.forecast(days[-1],'A','normalized_shrinkage')

def test_missing_target_does_not_bridge_sessions():
    rows,days,names=fixture()
    for r in rows:
        if r['symbol']=='A' and r['day']==days[1]:r['auction_volume']=None
    p=CausalLiquidityPanel(rows,days,names)
    assert not any(d==days[1] and s=='A' for d,s,v in p.pairs)
    assert len([x for x in p.pairs if x[1]=='A'])==3

def test_no_three_observation_discontinuity_for_equal_volumes():
    rows,days,names=fixture();p=CausalLiquidityPanel(rows,days,names)
    assert [p.forecast(d,'A','broad_median')['predicted_volume'] for d in days[1:]]==[100]*4

def test_unknown_forecast_is_zero_not_unbounded():
    rows,days,names=fixture();e=CausalLiquidityPanel(rows,days,names).forecast(days[0],'A','broad_median')
    assert request_bound(e,.01,datetime.fromisoformat(days[0]+'T14:30:00+09:00'))==0

def test_zero_auction_is_a_real_observation():
    rows,days,names=fixture()
    for r in rows:r['auction_volume']=0
    assert CausalLiquidityPanel(rows,days,names).forecast(days[-1],'A','broad_median')['predicted_volume']==0

@pytest.mark.parametrize('stamp',['2026-01-09T14:29:00+09:00','2026-01-10T14:30:00+09:00','2026-01-09T14:30:00'])
def test_availability_and_staleness(stamp):
    rows,days,names=fixture();e=CausalLiquidityPanel(rows,days,names).forecast(days[-1],'A','normalized_shrinkage')
    with pytest.raises(ValueError):request_bound(e,.01,datetime.fromisoformat(stamp))

def test_duplicate_or_unbound_rows_rejected():
    rows,days,names=fixture()
    with pytest.raises(ValueError):CausalLiquidityPanel(rows+[rows[0]],days,names)
    rows[0]['source_hash']=''
    with pytest.raises(ValueError):CausalLiquidityPanel(rows,days,names)

def test_shrinkage_matches_constant_scale():
    rows,days,names=fixture();e=CausalLiquidityPanel(rows,days,names).forecast(days[-1],'A','normalized_shrinkage')
    assert abs(e['predicted_volume']-100)<1e-10
    assert e['latest_label_day']<days[-1]

def test_manifest_universe_is_not_a_path_or_shared_database_expansion():
    from scripts.olr_structural_source_v6 import source_universe
    names=[str(i).zfill(6) for i in range(103)]
    assert source_universe(dict.fromkeys(names),names+['999999'])==(names,['999999'])
    with pytest.raises(ValueError):source_universe('universe_103.yaml',names)
    with pytest.raises(ValueError):source_universe(names,names[:-1])
