from copy import deepcopy
from datetime import date
import hashlib
import json

import pandas as pd
import pytest

from backtests.core.daily_volume_residual import verify_daily_close_residual
from backtests.core.anchored_recovery import verified_reconstruction


def source():
    stamps = [*pd.date_range("2026-04-20 09:00", periods=30, freq="min", tz="Asia/Seoul"),
              pd.Timestamp("2026-04-20 15:30", tz="Asia/Seoul")]
    frame = pd.DataFrame(dict(timestamp=stamps, open=100, high=100, low=100, close=100, volume=range(1, 32)))
    reference = dict(date="2026-04-20", open=100, high=100, low=100, close=100, volume=int(frame.volume.sum()) + 1)
    receipt = dict(status="ok", symbol="005930", date="2026-04-20", adjusted=False,
        endpoint="/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
        params=dict(FID_ORG_ADJ_PRC="1", FID_COND_MRKT_DIV_CODE="J", FID_INPUT_ISCD="005930"),
        output2=[dict(stck_bsop_date="20260420", stck_oprc="100", stck_hgpr="100", stck_lwpr="100",
                      stck_clpr="100", acml_vol=str(reference["volume"]), acml_tr_pbmn=str(reference["volume"]*100))])
    total, rows = 0, []
    for r in frame.itertuples():
        total += r.volume*100
        rows.append(dict(stck_bsop_date="20260420", stck_cntg_hour=r.timestamp.strftime("%H%M%S"),
            stck_oprc="100", stck_hgpr="100", stck_lwpr="100", stck_prpr="100", cntg_vol=str(r.volume),
            acml_tr_pbmn=str(total)))
    return frame, reference, receipt, rows


def test_exact_residual_is_reconciled_without_inventing_a_bar():
    frame, reference, receipt, rows = source()
    before = frame.copy(deep=True)
    proof = verify_daily_close_residual(frame, reference, receipt, rows, date(2026, 4, 20), "005930")
    assert proof["daily_excess_shares"] == 1
    assert not proof["residual_added_to_bars"]
    assert not proof["independent_intraday_completeness_verified"]
    pd.testing.assert_frame_equal(before, frame)


@pytest.mark.parametrize("bad", ["turnover", "adjusted", "daily_volume", "conflict", "omitted_minute", "price", "fraction"])
def test_daily_excess_cannot_excuse_unproven_minutes_or_basis(bad):
    frame, reference, receipt, rows = source()
    if bad == "turnover":
        receipt["output2"][0]["acml_tr_pbmn"] = str(int(receipt["output2"][0]["acml_tr_pbmn"])+1)
    elif bad == "adjusted":
        receipt["adjusted"] = True
    elif bad == "daily_volume":
        reference["volume"] += 1
    elif bad == "conflict":
        rows.append({**rows[0], "acml_tr_pbmn": "101"})
    elif bad == "omitted_minute":
        rows.pop(4)
    elif bad == "price":
        frame.loc[0, "high"] = 101
    else:
        receipt["output2"][0]["acml_tr_pbmn"] = "49700.1"
    with pytest.raises((ValueError, AssertionError)):
        verify_daily_close_residual(frame, reference, receipt, rows, date(2026, 4, 20), "005930")


def test_sparse_auction_admission_rechecks_source_and_does_not_allocate_residual(tmp_path):
    frame, reference, receipt, rows = source()
    frame.iloc[:-1].to_parquet(tmp_path / "anchor.parquet", index=False)
    frame.to_parquet(tmp_path / "raw.parquet", index=False)
    pd.DataFrame([reference]).to_parquet(tmp_path / "daily.parquet", index=False)
    (tmp_path / "daily.json").write_text(json.dumps(receipt))
    payload = dict(endpoint="inquire-time-dailychartprice", symbol="005930", market="J", volume_field="cntg_vol", rows=rows)
    (tmp_path / "minutes.json").write_text(json.dumps(payload))
    row = dict(symbol="005930", date="2026-04-20", method="verified_sparse_closing_auction",
        anchor="anchor.parquet", raw="raw.parquet", daily_source="daily.parquet",
        original_daily_receipt="daily.json", provider_responses=["minutes.json"],
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()})
    old, new, proof, _ = verified_reconstruction(tmp_path, row)
    assert new.volume.sum() - old.volume.sum() == 31
    assert proof["coverage_basis"] == "verified_closing_print_on_repeated_sparse_source"
    assert proof["daily_volume_residual"]["daily_excess_shares"] == 1
    changed = deepcopy(receipt)
    changed["output2"][0]["acml_tr_pbmn"] = str(int(changed["output2"][0]["acml_tr_pbmn"])-100)
    (tmp_path / "daily.json").write_text(json.dumps(changed))
    row["files"]["daily.json"] = hashlib.sha256((tmp_path / "daily.json").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="residual"):
        verified_reconstruction(tmp_path, row)
