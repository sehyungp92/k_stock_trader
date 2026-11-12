from copy import deepcopy
from datetime import date
from fractions import Fraction

import pandas as pd
import pytest

from backtests.core.anchored_basis_repair import original_price, reconstruct_original_session


@pytest.fixture
def sample():
    def frame(day):
        stamps = [*pd.date_range(day + " 09:00", periods=21, freq="min", tz="Asia/Seoul"),
                  pd.Timestamp(day + " 15:30", tz="Asia/Seoul")]
        return pd.DataFrame(dict(timestamp=stamps, open=150_000, high=150_000, low=150_000,
                                 close=150_000, volume=200))
    anchor = frame("2026-06-01"); adjusted = anchor.copy()
    for field in ("open", "high", "low", "close"): adjusted[field] = 120_000
    adjusted.volume = 250
    raw = adjusted.copy(); raw.timestamp += pd.Timedelta(days=1)
    original_daily = dict(open=150_000, high=150_000, low=150_000, close=150_000,
                          volume=4500, turnover=675_000_000)
    adjusted_daily = dict(open=120_000, high=120_000, low=120_000, close=120_000,
                          volume=5625, turnover=675_000_000)
    provider = []
    for i, row in raw.iterrows():
        provider.append(dict(stck_bsop_date="20260602", stck_cntg_hour=row.timestamp.strftime("%H%M%S"),
            stck_oprc="120000", stck_hgpr="120000", stck_lwpr="120000", stck_prpr="120000",
            cntg_vol="250", acml_tr_pbmn=str((i + 1) * 30_000_000)))
    return [raw, anchor, adjusted, original_daily, adjusted_daily, provider, date(2026, 6, 2), "068270"]


def test_exact_repair_preserves_residual_outside_bars(sample):
    result, proof = reconstruct_original_session(*sample)
    assert len(result) == 22
    assert result.volume.sum() == 4400
    assert (result.close == 150_000).all()
    assert proof["daily_excess_shares"] == 100
    assert proof["daily_excess_turnover"] == 15_000_000
    assert not proof["residual_added_to_bars"]


@pytest.mark.parametrize("change", ["volume", "price", "turnover", "daily", "residual", "clock", "duplicate", "anchor"])
def test_contradictions_cannot_be_silently_repaired(sample, change):
    args = deepcopy(sample)
    if change == "volume": args[0].loc[4, "volume"] += 1
    elif change == "price": args[0].loc[4, "close"] += 1
    elif change == "turnover": args[5][4]["acml_tr_pbmn"] = "1"
    elif change == "daily": args[3]["high"] += 100
    elif change == "residual": args[3]["volume"] += 1
    elif change == "clock": args[0].loc[0, "timestamp"] += pd.Timedelta(seconds=1)
    elif change == "duplicate": args[0] = pd.concat([args[0], args[0].iloc[-1:]], ignore_index=True)
    elif change == "anchor": args[1].loc[1, "close"] = 170_000
    with pytest.raises((ValueError, AssertionError)):
        reconstruct_original_session(*args)


def test_lattice_constraint_is_unique_inverse_not_rounding():
    price, method = original_price(180_434, Fraction(367827, 385900), Fraction(72841, 76420))
    assert price == 189_300
    assert method == "unique_krx_midpoint_lattice_inverse"
    with pytest.raises(ValueError, match="No unique"):
        original_price(120000, Fraction(79, 100), Fraction(81, 100))
    with pytest.raises(ValueError, match="Outside"):
        original_price(200000, Fraction(95, 100), Fraction(96, 100))
