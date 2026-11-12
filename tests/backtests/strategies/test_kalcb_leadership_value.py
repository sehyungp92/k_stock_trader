from copy import deepcopy
import pytest
from backtests.strategies.kalcb.leadership_value import rank_at_cutoff,SOURCES

def fixture():
    return [{'symbol':s,'trade_date':'2025-08-07','feature_available_at':'2025-08-07T09:30:00+09:00',
        'features':{k:float(i) for k in SOURCES.values()},'quality_target':i*1e9} for i,s in enumerate('ABC')]

def test_rank_cannot_use_later_prefix_or_payoff_fields():
    rows=fixture();expected=rank_at_cutoff(rows,'2025-08-07','2025-08-07T09:30:00+09:00')
    rows.append({**deepcopy(rows[-1]),'symbol':'F','feature_available_at':'2025-08-07T09:35:00+09:00'})
    rows.append({**deepcopy(rows[-1]),'symbol':'G','trade_date':'2025-08-08','feature_available_at':'2025-08-08T09:30:00+09:00'})
    for r in rows:r['quality_target']=-1e100
    assert rank_at_cutoff(rows,'2025-08-07','2025-08-07T09:30:00+09:00')==expected
    assert expected['C']['leader_daily_return_rank']==5/6
    assert expected['A']['leader_emergence']==0

def test_ties_and_missing_values_do_not_depend_on_symbol_order():
    rows=fixture()
    for r in rows:r['features']['daily_return_20d']=1
    rows[0]['features']['first30_ret']=None
    result=rank_at_cutoff(rows,'2025-08-07','2025-08-07T09:30:00+09:00')
    assert all(f['leader_daily_return_rank']==.5 for f in result.values())
    assert result['A']['leader_persistence'] is None
    assert rank_at_cutoff(rows[::-1],'2025-08-07','2025-08-07T09:30:00+09:00')==result
    with pytest.raises(ValueError,match='Duplicate'):
        rank_at_cutoff(rows+rows[:1],'2025-08-07','2025-08-07T09:30:00+09:00')
