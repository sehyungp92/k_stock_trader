import importlib.util
from pathlib import Path
import pytest
from datetime import datetime,timedelta

spec=importlib.util.spec_from_file_location('census',Path(__file__).parents[3]/'backtests/strategies/kalcb/opening_checkpoint_census_v1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def bars():
    start=datetime.fromisoformat('2026-04-01T09:00:00+09:00')
    return [{'timestamp':(start+timedelta(minutes=5*i)).isoformat(),'open':100,'high':102,'low':98,'close':100,'volume':10} for i in range(9)]


def test_future_bars_cannot_change_features():
    rows=bars();before=m.features('2026-04-01',rows,2,-202)
    rows.append(dict(rows[-1],timestamp='2026-04-01T09:45:00+09:00',close=10000000,volume=0))
    assert m.features('2026-04-01',rows,2,-202)==before


@pytest.mark.parametrize('change',['missing','duplicate','zero_volume'])
def test_invalid_prefix_abstains(change):
    rows=bars()
    if change=='missing':rows.pop(0)
    elif change=='duplicate':rows[1]=rows[0]
    else:rows[1]['volume']=0
    with pytest.raises(ValueError):m.features('2026-04-01',rows,2,-202)


def test_native_vwap_typical_price_and_zero_signs():
    rows=bars();rows[-1]['close']=99
    x=m.features('2026-04-01',rows,2,-200)
    assert x['state']=='negative/below_vwap'
    assert x['session_vwap']==pytest.approx((8*100+(102+98+99)/3)/9)


def test_same_clock_fill_excluded_and_fees_retained():
    rows=[{'timestamp':'2026-04-01T09:30:00+09:00','qty':2,'price':100,'side':'BUY','metadata':{'fee':1}},
          {'timestamp':'2026-04-01T09:45:00+09:00','qty':2,'price':90,'side':'SELL','metadata':{'fee':1}}]
    assert m.inventory_before(rows,'2026-04-01T09:45:00+09:00')==(2,-200,1)
