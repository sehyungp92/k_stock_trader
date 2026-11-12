import pandas as pd
from scripts.olr_minute_exit_source_preflight_v5 import compare


def source():
    ts=pd.date_range('2026-01-05 09:00',periods=10,freq='min',tz='Asia/Seoul')
    one=pd.DataFrame({'open':range(100,110),'high':range(101,111),'low':range(99,109),'close':range(100,110),'volume':[100]*10},index=ts)
    native=one.resample('5min').agg(dict(open='first',high='max',low='min',close='last',volume='sum'))
    return one,native


def test_exact_aggregation_and_missing_or_changed_native_bins_are_detected():
    one,native=source();assert not compare(one,native)[1].any()
    changed=native.copy();changed.iloc[0,changed.columns.get_loc('volume')]+=1
    assert compare(one,changed)[1].sum()==1
    assert compare(one,native.iloc[:1])[1].sum()==1
    assert compare(one.iloc[:5],native)[1].sum()==1


def test_missing_real_minute_is_not_filled_or_interpolated():
    one,native=source();sparse=one.drop(one.index[2])
    assert compare(sparse,native)[1].sum()==1
    matching=sparse.resample('5min').agg(dict(open='first',high='max',low='min',close='last',volume='sum'))
    assert not compare(sparse,matching)[1].any()
    assert len(sparse)==9
