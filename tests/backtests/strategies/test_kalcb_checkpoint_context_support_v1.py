from copy import deepcopy
from datetime import datetime,timedelta
import importlib.util
from pathlib import Path
import pytest

P=Path(__file__).resolve().parents[3]/'backtests/strategies/kalcb/checkpoint_context_support_v1.py'
spec=importlib.util.spec_from_file_location('pure_context_support',P)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture(symbol='A',sector='S'):
    start=datetime.fromisoformat('2025-08-07T09:00:00+09:00')
    bars=[dict(symbol=symbol,timestamp=(start+timedelta(minutes=5*i)).isoformat(),
               open=100.,high=102.,low=98.,close=100.+(i>=6),volume=100.) for i in range(9)]
    row=dict(symbol=symbol,sector=sector,trade_date='2025-08-07',
             feature_available_at='2025-08-07T09:30:00+09:00',
             candidate={'sector':sector,'trade_date':'2025-08-07'},
             opening_bars=deepcopy(bars[:6]),features={'daily_atr_pct':.02})
    return row,bars


def test_future_bar_cannot_change_context():
    a,ab=fixture();b,bb=fixture('B');r=m.context_rows([a,b],{'A':ab,'B':bb})
    future=deepcopy(ab[-1]);future.update(timestamp='2025-08-07T09:45:00+09:00',close=999999)
    assert r==m.context_rows([a,b],{'A':ab+[future],'B':bb})


@pytest.mark.parametrize('fault',['missing','duplicate','zero','nan','ohlc','symbol'])
def test_raw_prefix_faults_are_not_silently_imputed(fault):
    a,ab=fixture()
    if fault=='missing':ab.pop(7)
    elif fault=='duplicate':ab.insert(7,deepcopy(ab[6]))
    elif fault=='zero':ab[7]['volume']=0
    elif fault=='nan':ab[7]['close']=float('nan')
    elif fault=='ohlc':ab[7]['close']=1000
    else:ab[7]['symbol']='OTHER'
    with pytest.raises(ValueError):m.prefix_statistics(a,ab)


def test_original_opening_source_mismatch_rejected():
    a,ab=fixture();a['opening_bars'][1]['volume']+=1
    with pytest.raises(ValueError,match='source mismatch'):m.prefix_statistics(a,ab)


def test_unknown_sector_cannot_be_reassigned():
    a,ab=fixture();a['sector']='FUTURE'
    with pytest.raises(ValueError,match='sector/date'):m.prefix_statistics(a,ab)


def test_leave_one_out_not_self_breadth():
    a,ab=fixture();b,bb=fixture('B')
    for bar in bb[6:]:bar['close']=99
    rows,_=m.context_rows([a,b],{'A':ab,'B':bb})
    x=rows[0]
    assert x['market_peer_symbols']==['B'] and x['sector_peer_symbols']==['B']
    assert x['context_features']['relative_post_entry_market_return_atr']==pytest.approx(1.)
    assert x['context_features']['peer_sector_vwap_breadth_change']==0


def test_future_or_nonpopulation_members_do_not_enter():
    a,ab=fixture();b,bb=fixture('B');_,future=fixture('FUTURE')
    assert m.context_rows([a,b],{'A':ab,'B':bb})==m.context_rows([a,b],{'A':ab,'B':bb,'FUTURE':future})


def test_missing_sector_support_abstains():
    a,ab=fixture();b,bb=fixture('B','DIFFERENT')
    rows,_=m.context_rows([a,b],{'A':ab,'B':bb})
    assert all(not r['context_supported'] and r['context_features'] is None for r in rows)


def test_population_duplicate_rejected():
    a,ab=fixture()
    with pytest.raises(ValueError,match='Duplicate'):m.context_rows([a,a],{'A':ab})


def test_actual_opening_clock_used():
    a,ab=fixture()
    for b in ab:b['timestamp']=b['timestamp'].replace('T09:','T10:')
    a['feature_available_at']='2025-08-07T10:30:00+09:00';a['opening_bars']=deepcopy(ab[:6])
    assert m.prefix_statistics(a,ab)['available_at']=='2025-08-07T10:45:00+09:00'


def test_embargo_and_source_support_not_label_support():
    days=['2025-08-05','2025-08-06','2025-08-07']
    rows=[{'day':d,'context_supported':True} for d in days]
    value=m.chronological_support(rows,days,[('2025-08-07','2025-08-07')])[0]
    assert value['prior_source_sessions']==1 and value['observed_through']=='2025-08-05'
    assert not value['mature_action_label_support_verified']
