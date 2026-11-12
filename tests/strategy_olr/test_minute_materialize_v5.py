import importlib.util
from pathlib import Path
import pandas as pd
import pytest

PATH=Path(__file__).resolve().parents[2]/'scripts/olr_minute_exit_materialize_v5.py'
spec=importlib.util.spec_from_file_location('minute_source_materialize_test',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def frame():
    return pd.DataFrame([dict(timestamp=pd.Timestamp('2026-01-08T09:00:00+09:00'),open=100,high=102,low=99,close=101,volume=50)])


def test_source_normalization_preserves_exact_values_and_requires_timezone():
    x=frame();m.same(x,x)
    y=x.copy();y.timestamp=y.timestamp.dt.tz_convert('UTC');m.same(x,y)
    y.timestamp=y.timestamp.dt.tz_localize(None)
    with pytest.raises(ValueError,match='timezone'):m.normalize(y)


@pytest.mark.parametrize('field,value',[('volume',-1),('open',0),('high',100),('low',102),('close',float('nan'))])
def test_invalid_minute_observation_fails_closed(field,value):
    x=frame();x.loc[0,field]=value
    with pytest.raises(AssertionError):m.normalize(x)


def test_duplicate_or_one_changed_share_is_not_accepted():
    x=frame()
    with pytest.raises(AssertionError):m.normalize(pd.concat([x,x],ignore_index=True))
    y=x.copy();y.loc[0,'volume']+=1
    with pytest.raises(AssertionError):m.same(x,y)
