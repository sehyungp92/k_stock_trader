"""Conflicting source vintages cannot silently become one execution session."""
import pandas as pd
import pytest

from scripts.update_kis_intraday import merge_1m_frames, normalize_bar_frame


def frame(price=100, volume=20, start="2026-05-29 09:00"):
    stamps = pd.date_range(start, periods=382, freq="min", tz="Asia/Seoul")
    return pd.DataFrame(dict(timestamp=stamps, open=price, high=price,
                             low=price, close=price, volume=volume))


def merge(*frames):
    return merge_1m_frames(*frames, start_dt=pd.Timestamp("2026-05-01", tz="Asia/Seoul"),
                          end_dt=pd.Timestamp("2026-07-01", tz="Asia/Seoul"))


@pytest.mark.parametrize("field", ["open", "high", "low", "close", "volume"])
def test_conflicting_overlap_is_rejected_in_either_source_order(field):
    original, revised = frame(), frame()
    revised.loc[183, field] += 1
    for sources in ((original, revised), (revised, original)):
        with pytest.raises(ValueError, match="Conflicting minute OHLCV"):
            merge(*sources)


def test_adjusted_vintage_cannot_interleave_with_original():
    with pytest.raises(ValueError, match="Conflicting minute OHLCV"):
        merge(frame(), frame(95, 21))


def test_identical_overlap_and_new_session_are_deterministic():
    old, new = frame(), frame(start="2026-06-01 09:00")
    actual = merge(old, old.iloc[200:], new)
    pd.testing.assert_frame_equal(actual, pd.concat([old, new], ignore_index=True), check_dtype=False)


def test_declared_basis_mismatch_rejected_even_without_overlap():
    old, new = frame(), frame(start="2026-06-01 09:00")
    old.attrs["source_basis_id"] = "original"
    new.attrs["source_basis_id"] = "adjusted-20260604"
    with pytest.raises(ValueError, match="Incompatible declared"):
        merge(old, new)


def test_normalization_preserves_equal_timestamp_source_order():
    old, new = frame(), frame(95, 21)
    result = normalize_bar_frame(pd.concat([old, new], ignore_index=True))
    assert result.groupby("timestamp").close.apply(list).tolist() == [[100, 95]] * 382


def test_provider_conflicting_duplicates_are_rejected_and_raw_receipt_preserved(tmp_path, monkeypatch):
    import json
    from scripts import update_kis_intraday as updater
    row = dict(stck_bsop_date="20260601", stck_cntg_hour="090000", stck_oprc="100",
               stck_hgpr="100", stck_lwpr="100", stck_prpr="100", cntg_vol="20")
    monkeypatch.setattr(updater, "fetch_kis_1m_page", lambda *a, **kw: [row, {**row, "cntg_vol": "21"}])
    stamp = pd.Timestamp("2026-06-01 09:00", tz="Asia/Seoul")
    with pytest.raises(ValueError, match="Conflicting provider minute"):
        updater.fetch_kis_1m_bars(object(), "068270", stamp, stamp, market_code="J",
                                  extra_sleep_sec=0, max_pages=2, raw_page_root=tmp_path)
    receipt = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert len(receipt["rows"]) == 2
    assert receipt["retrieved_at"]
    assert receipt["price_basis"] == "provider_adjustment_vintage_not_yet_qualified"
