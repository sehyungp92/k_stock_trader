from datetime import date, datetime, time, timedelta

import pandas as pd
import pytest

from strategy_common.clock import KST
from strategy_common.session_coverage import session_bar_coverage
from backtests.core.completed_bar_policy import is_regular_session, session_close_for
from deployment.olr_kalcb.readiness import _validate_generated_at, KALCB_FINAL_ARTIFACT_STAGE, OLR_FINAL_ARTIFACT_STAGE
from deployment.olr_kalcb.runtime import default_session_schedule
from scripts.update_kis_intraday import _normalize_kis_minute_row, _previous_day_close, parquet_symbol_covers_date_window, normalize_bar_frame


def test_exceptional_hours_reach_download_availability_and_operator():
    from oms.adapter import KISExecutionAdapter
    adapter = KISExecutionAdapter(None)
    day = date(2025, 11, 13)
    assert adapter._is_order_session_open(datetime.combine(day, time(16), KST))
    assert not adapter._is_order_session_open(datetime.combine(day, time(9, 30), KST))
    assert _previous_day_close(datetime(2025, 11, 14, 9, tzinfo=KST)).time() == time(16, 40)
    assert _previous_day_close(datetime(2025, 11, 17, 9, tzinfo=KST)).date() == date(2025, 11, 14)
    assert session_close_for(day).time() == time(16, 30)
    assert is_regular_session(datetime.combine(day, time(16), KST))
    assert not is_regular_session(datetime.combine(day, time(9, 30), KST))
    _validate_generated_at(datetime.combine(day, time(9, 55), KST), "KALCB", day, KALCB_FINAL_ARTIFACT_STAGE)
    with pytest.raises(ValueError, match="15:30"):
        _validate_generated_at(datetime.combine(day, time(14, 45), KST), "OLR", day, OLR_FINAL_ARTIFACT_STAGE)
    schedule = {s.name: s.run_at_kst for s in default_session_schedule("paper", trade_date=day)}
    assert schedule["kalcb_runtime_start"] == "10:00"
    assert schedule["olr_final_artifact"] == "15:35"
    jan = {s.name: s.run_at_kst for s in default_session_schedule("paper", trade_date=date(2026, 1, 2))}
    assert jan["kalcb_runtime_start"] == "10:00" and jan["olr_final_artifact"] == "14:35"


def test_session_completeness_rejects_missing_tail_and_interior(tmp_path):
    day = date(2025, 11, 13)
    start = datetime.combine(day, time(10), KST)
    timestamps = [start + timedelta(minutes=i) for i in range(380)] + [start + timedelta(minutes=390)]
    assert session_bar_coverage(timestamps, day)["status"] == "pass"
    assert session_bar_coverage(timestamps[:-1], day)["status"] == "unverified"
    assert session_bar_coverage(timestamps[:5] + timestamps[6:], day)["status"] == "unverified"
    folder = tmp_path / "005930"
    folder.mkdir()
    path = folder / "005930_1m_fixture.parquet"
    pd.DataFrame({"timestamp": timestamps[:-1]}).to_parquet(path)
    assert not parquet_symbol_covers_date_window(tmp_path, "005930", ["1m"], day, day)
    pd.DataFrame({"timestamp": timestamps}).to_parquet(path)
    assert parquet_symbol_covers_date_window(tmp_path, "005930", ["1m"], day, day)


@pytest.mark.parametrize("volume", [{"acml_vol": "1500"}, {}, {"cntg_vol": "nan"}, {"cntg_vol": "-1"}, {"cntg_vol": "1.5"}])
def test_unknown_or_cumulative_volume_is_not_interval_volume(volume):
    with pytest.raises(ValueError, match="volume"):
        _normalize_kis_minute_row({"stck_bsop_date": "20251113", "stck_cntg_hour": "100000", "stck_prpr": "100", **volume})


def test_zero_is_valid_observed_volume_but_missing_is_not():
    row = _normalize_kis_minute_row({"stck_bsop_date": "20251113", "stck_cntg_hour": "100000", "stck_prpr": "100", "cntg_vol": "0", "acml_vol": "1000"})
    assert row["volume"] == 0
    del row["volume"]
    with pytest.raises(ValueError, match="volume"):
        normalize_bar_frame(pd.DataFrame([row]))
