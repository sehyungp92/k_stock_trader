from datetime import datetime,timezone
import pytest
from strategy_olr.core.marked_book_paths import Book,project,compare_actions

AT=datetime(2026,1,10,tzinfo=timezone.utc)

def scenario():
    return dict(date='2026-01-08',known_at='2026-01-09T00:00:00+00:00',source_hash='fixture',path=[
        dict(offset=1,prices={'A':120,'B':120},capacity={'A':2,'B':2}),
        dict(offset=2,prices={'A':80,'B':80},capacity={'A':2,'B':2}),
        dict(offset=3,prices={'A':110,'B':110},capacity={'A':10,'B':10})])

def test_path_risk_keeps_profit_peak_and_common_shock():
    b=Book(0,2000,{'A':100,'B':100},{'A':10,'B':10})
    q=project(b,scenario(),AT)
    assert q['final_equity']==2200
    assert q['max_drawdown_fraction']==pytest.approx(1-1600/2400)

def test_trim_is_quantity_consistent_and_cash_released_only_on_service():
    b=Book(0,2000,{'A':100,'B':100},{'A':10,'B':10})
    out=compare_actions(b,[scenario()],AT,{'hold':{},'trim':{'A':5}},commission_bps=0,tax_bps=0,slippage_bps=0)
    t=out['trim'][0]
    assert [p['fills'].get('A',0) for p in t['path']]==[2,2,1]
    assert t['path'][0]['cash']==240 and t['remaining_inventory']['A']==5
    assert t['remaining_orders']['A']==0

def test_missing_names_and_unmatured_scenarios_fail_closed():
    b=Book(0,2000,{'A':100,'B':100},{'A':10,'B':10});s=scenario()
    s['path'][0]['capacity'].pop('B')
    with pytest.raises(ValueError,match='Missing'):project(b,s,AT)
    s=scenario();s['known_at']=AT.isoformat()
    with pytest.raises(ValueError,match='matured'):project(b,s,AT)

def test_true_historic_high_water_never_resets_on_recovery():
    b=Book(0,3000,{'A':100,'B':100},{'A':10,'B':10})
    assert project(b,scenario(),AT)['max_drawdown_fraction']==pytest.approx(1-1600/3000)
