import pytest

from scripts.olr_round1_mechanism_diagnostics import account_slice


def test_calendar_slice_uses_preceding_nav_and_includes_carried_position_marks():
    daily=[{'day':'2026-05-28','equity':110}, {'day':'2026-05-29','equity':120},
           {'day':'2026-06-01','equity':108}, {'day':'2026-06-02','equity':114}]
    trades=[{'entry_fill_time':'2026-05-28T15:00'}, {'entry_fill_time':'2026-06-02T15:00'}]
    may=account_slice(daily,trades,100,0,1)
    june=account_slice(daily,trades,100,2,3)
    assert june['opening_equity']==120
    assert june['return_pct']==pytest.approx(-5)
    assert june['drawdown_pct']==pytest.approx(10)
    assert june['net_nav_change']==-6 and june['entries']==1
    assert (1+may['return_pct']/100)*(1+june['return_pct']/100)==pytest.approx(1.14)
