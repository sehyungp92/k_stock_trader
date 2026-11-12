from datetime import date, datetime, time, timedelta
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from opportunity_clock_contract import close_anchored_feature_bars, olr_completeness_clock
from strategy_common import source_quality
from strategy_common.clock import KST


@pytest.mark.parametrize('opening,closing,expected', [(time(9),time(15,30),time(14,30)),
    (time(10),time(15,30),time(14,30)), (time(10),time(16,30),time(15,30))])
def test_clock_matches_olr_feature_anchor(opening,closing,expected):
    day=date(2026,1,2)
    calendar=SimpleNamespace(session_hours=lambda d:(opening,closing))
    with patch('strategy_common.execution_session.get_trading_calendar',return_value=calendar), \
         patch('strategy_common.source_quality.missing_prefix',return_value=[]) as audit:
        result,excluded=close_anchored_feature_bars({(day,'123456'):()},time(14,30))
    assert audit.call_args.args[-1] == datetime.combine(day,expected,KST)
    assert (day,'123456') in result and not excluded


def test_missing_predecision_bars_are_still_rejected():
    day=date(2026,1,2)
    with patch('strategy_common.source_quality.missing_prefix',return_value=['missing']):
        result,excluded=close_anchored_feature_bars({(day,'123456'):()},time(14,30))
    assert not result and excluded==['123456:2026-01-02']


def test_future_presence_does_not_change_eligibility():
    day=date(2026,1,2);cutoff=datetime.combine(day,time(14,30),KST)
    prefix=(SimpleNamespace(timestamp=cutoff-timedelta(minutes=5)),)
    future=SimpleNamespace(timestamp=cutoff+timedelta(minutes=20))
    calendar=SimpleNamespace(session_hours=lambda d:(time(10),time(15,30)))
    def check(symbol,d,timestamps,available):
        assert available==cutoff
        return [] if cutoff-timedelta(minutes=5) in tuple(timestamps) else ['missing']
    with patch('strategy_common.execution_session.get_trading_calendar',return_value=calendar), \
         patch('strategy_common.source_quality.missing_prefix',side_effect=check):
        a=close_anchored_feature_bars({(day,'123456'):prefix},time(14,30))
        z=close_anchored_feature_bars({(day,'123456'):prefix+(future,)},time(14,30))
    assert set(a[0])==set(z[0]) and a[1]==z[1]==[]


def test_scope_restored_after_error():
    original=source_quality.complete_feature_bars
    with pytest.raises(RuntimeError):
        with olr_completeness_clock():
            assert source_quality.complete_feature_bars is close_anchored_feature_bars
            raise RuntimeError('fixture')
    assert source_quality.complete_feature_bars is original
