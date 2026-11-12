from datetime import date, datetime, time, timedelta
from pathlib import Path
import importlib.util
from types import SimpleNamespace
import pytest
from strategy_common import session_coverage as q
from strategy_common.clock import KST

pytestmark=pytest.mark.skipif(not hasattr(q,'_calendar_grid'),reason='Runs against the isolated revision19 replay engine')


@pytest.mark.parametrize('minutes',[1,5,7,30])
@pytest.mark.parametrize('opening',[time(9),time(10),time(9,3,30)])
@pytest.mark.parametrize('periodic',[False,True])
def test_pure_grid_matches_original_every_observation_pattern(monkeypatch,minutes,opening,periodic):
    day=date(2026,10,8)
    # Test with a direct copy of the original function, not a duplicated model.
    path=Path('E:/CodexResearch/k_stock_trader/olr_round1_comprehensive_20261009/generations_v4/train/inputs/files/strategy_common/session_coverage.py')
    if not path.exists():pytest.skip('Requires the retained original native implementation')
    spec=importlib.util.spec_from_file_location('old_coverage',path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    calendar=SimpleNamespace(has_coverage=lambda d:True,has_session_coverage=lambda d:True,
        is_trading_day=lambda d:True,suspension_source=lambda s,d:None,session_hours=lambda d:(opening,time(15,30)),
        closing_auction_minutes=10,closing_match_time=lambda s,d:time(15,35),
        periodic_auction_sources=lambda s,d:('source',) if periodic else (),
        intraday_halts_for=lambda s,d:((time(11),time(11,17),'halt'),),session_schedule_hash='test')
    monkeypatch.setattr(q,'get_trading_calendar',lambda:calendar)
    monkeypatch.setattr(old,'get_trading_calendar',lambda:calendar)
    q._calendar_grid.cache_clear()
    stamps=[datetime.combine(day,time(9),KST)+timedelta(minutes=i*minutes) for i in range(400//minutes)]
    for values in ([],stamps,stamps[::2],stamps+stamps[:1]):
        assert q.session_bar_coverage(values,day,minutes,symbol='005930')==old.session_bar_coverage(values,day,minutes,symbol='005930')
    assert q._calendar_grid.cache_info().hits==3
    # Changing one calendar fact must use a new key and preserve contradictions.
    calendar.intraday_halts_for=lambda s,d:((time(11),time(11,33),'new-halt'),)
    assert q.session_bar_coverage(stamps,day,minutes,symbol='005930')==old.session_bar_coverage(stamps,day,minutes,symbol='005930')
    assert q._calendar_grid.cache_info().misses==2
