from copy import deepcopy
from datetime import date

import pandas as pd
import pytest

from backtests.core.adjusted_recovery import reconstruct_adjusted_session


@pytest.fixture
def adjusted_session():
    day, symbol = date(2026, 3, 5), "005930"
    stamps = [*pd.date_range("2026-03-05 09:00", periods=21, freq="min", tz="Asia/Seoul"),
              pd.Timestamp("2026-03-05 15:30", tz="Asia/Seoul")]
    prices = [10000 + i * 200 for i in range(21)] + [12000]
    volume = [100 * (i + 1) for i in range(22)]
    full = pd.DataFrame(dict(timestamp=stamps, open=prices, high=prices, low=prices, close=prices, volume=volume))
    old = full.iloc[:-1].copy()
    raw = full.copy()
    raw.timestamp -= pd.Timedelta(seconds=44)
    for f in ("open", "high", "low", "close"):
        raw[f] = (raw[f] * .8 + .5).astype(int)
    raw.volume = (raw.volume * 1.25 + .5).astype(int)
    total, pages = 0, []
    for i, row in raw.iterrows():
        total += int(full.iloc[i].close * full.iloc[i].volume)
        pages.append({"stck_bsop_date": day.strftime("%Y%m%d"), "stck_cntg_hour": row.timestamp.strftime("%H%M%S"),
                      "acml_tr_pbmn": str(total)})
    reference = dict(open=10000, high=14000, low=10000, close=12000, volume=sum(volume))
    receipts = []
    for adjusted in (False, True):
        factor = .8 if adjusted else 1
        record = {"stck_bsop_date": day.strftime("%Y%m%d"),
                  "stck_oprc": str(int(reference["open"] * factor)),
                  "stck_hgpr": str(int(reference["high"] * factor)),
                  "stck_lwpr": str(int(reference["low"] * factor)),
                  "stck_clpr": str(int(reference["close"] * factor)),
                  "acml_vol": str(int(reference["volume"] / factor)), "acml_tr_pbmn": str(total)}
        receipts.append(dict(symbol=symbol, date=day.isoformat(), adjusted=adjusted, status="ok", output2=[record],
            endpoint="/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol,
                    "FID_INPUT_DATE_1": day.strftime("%Y%m%d"), "FID_INPUT_DATE_2": day.strftime("%Y%m%d"),
                    "FID_ORG_ADJ_PRC": "0" if adjusted else "1"}))
    return [raw, old, day, symbol, *receipts, reference, pages], full


def test_exact_basis_and_clock_reconstruction_preserves_archive(adjusted_session):
    args, expected = adjusted_session
    actual, proof = reconstruct_adjusted_session(*args)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    assert proof["offset_seconds"] == 44
    assert proof["matching_ohlcv_anchors"] == 21
    assert proof["revised_rows"] == 0
    assert proof["closing_auction"]["shares"] == 2200
    assert proof["turnover_bounds_verified_minutes"] == 22


@pytest.mark.parametrize("problem", ["price", "shares", "money", "daily", "daily_flag", "adjusted_daily", "timestamp"])
def test_contradictory_sources_cannot_be_reconciled_by_fitting(adjusted_session, problem):
    args, _ = deepcopy(adjusted_session)
    if problem == "price":
        args[0].loc[5, "close"] += 20
    elif problem == "shares":
        args[0].loc[5, "volume"] += 20
    elif problem == "money":
        args[-1][-1]["acml_tr_pbmn"] = str(int(args[-1][-1]["acml_tr_pbmn"]) + 1)
    elif problem == "daily":
        args[-2]["close"] += 1
    elif problem == "daily_flag":
        args[4]["params"]["FID_ORG_ADJ_PRC"] = "0"
    elif problem == "adjusted_daily":
        args[5]["output2"][0]["stck_oprc"] = "1"
    elif problem == "timestamp":
        args[0].loc[5, "timestamp"] += pd.Timedelta(seconds=1)
    with pytest.raises(ValueError):
        reconstruct_adjusted_session(*args)


def test_daily_residual_is_reported_and_never_allocated_to_auction(adjusted_session):
    args, expected = adjusted_session
    args[4]["output2"][0]["acml_vol"] = str(args[-2]["volume"] + 100)
    args[5]["output2"][0]["acml_vol"] = str(int((args[-2]["volume"] + 100) / .8))
    args[-2]["volume"] += 100
    for receipt in args[4:6]:
        receipt["output2"][0]["acml_tr_pbmn"] = str(int(receipt["output2"][0]["acml_tr_pbmn"]) + 1200000)
    actual, proof = reconstruct_adjusted_session(*args)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    assert proof["daily_volume_excess_over_regular_session"] == 100
    assert proof["closing_auction"]["shares"] == 2200


def test_ambiguous_new_price_is_rejected_instead_of_rounded(adjusted_session):
    args, _ = adjusted_session
    row = args[0].iloc[0].copy()
    row["timestamp"] = pd.Timestamp("2026-03-05 09:21", tz="Asia/Seoul") - pd.Timedelta(seconds=44)
    for f in ("open", "high", "low", "close"):
        row[f] = 10001  # two integer originals; neither is in the exact archive dictionary
    row["volume"] = 125
    args[0] = pd.concat([args[0].iloc[:-1], pd.DataFrame([row]), args[0].iloc[-1:]], ignore_index=True)
    args[-1].insert(-1, {"stck_bsop_date": "20260305", "stck_cntg_hour": "092016", "acml_tr_pbmn": "1"})
    with pytest.raises(ValueError, match="no unambiguous supported inverse"):
        reconstruct_adjusted_session(*args)
