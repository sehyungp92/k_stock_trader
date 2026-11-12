"""Bounded reconstruction of adjusted KIS minutes on a preserved archive basis.

This is an empirical, source-bound reconstruction, not a general corporate-action
normalizer. It never changes existing minutes or allocates daily residual volume.
"""
from collections import Counter
from fractions import Fraction
import math

import pandas as pd

from kis_core.trading_calendar import get_trading_calendar


PRICES = ("open", "high", "low", "close")
DAILY_FIELDS = dict(open="stck_oprc", high="stck_hgpr", low="stck_lwpr",
                    close="stck_clpr", volume="acml_vol")
HALF = Fraction(1, 2)


def _integer(value):
    if not math.isfinite(float(value)) or int(value) != value or value < 0:
        raise ValueError("Noninteger or negative source observation")
    return int(value)


def _bounds(overlap):
    lower, upper = Fraction(0), Fraction(1)
    for row in overlap.to_dict("records"):
        for field in PRICES:
            original, adjusted = _integer(row[field + "_old"]), _integer(row[field + "_new"])
            if original <= 0 or adjusted <= 0:
                raise ValueError("Nonpositive source price")
            lower = max(lower, (adjusted - HALF) / original)
            upper = min(upper, (adjusted + HALF) / original)
        original, adjusted = _integer(row["volume_old"]), _integer(row["volume_new"])
        if original <= 0 or adjusted <= 0:
            raise ValueError("Recovery requires observed positive-volume minutes")
        # adjusted shares = round(original shares / price factor).
        lower = max(lower, original / (adjusted + HALF))
        upper = min(upper, original / (adjusted - HALF))
        if lower >= upper:
            raise ValueError("No common rounded price/share conversion")
    return lower, upper


def _candidates(value, lower, upper, volume=False):
    # Closed bounds deliberately retain boundary alternatives. Ambiguity must be
    # resolved by another source observation, never by choosing a midpoint factor.
    value = _integer(value)
    if volume:
        first, last = (value - HALF) * lower, (value + HALF) * upper
    else:
        first, last = (value - HALF) / upper, (value + HALF) / lower
    return math.ceil(first), math.floor(last)


def _daily(receipt, symbol, day, adjusted):
    params = receipt.get("params", {})
    if (receipt.get("status") != "ok" or receipt.get("symbol") != symbol
            or receipt.get("date") != day.isoformat()
            or receipt.get("adjusted") is not adjusted
            or receipt.get("endpoint") != "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
            or params.get("FID_COND_MRKT_DIV_CODE") != "J"
            or params.get("FID_INPUT_ISCD") != symbol
            or params.get("FID_ORG_ADJ_PRC") != ("0" if adjusted else "1")
            or params.get("FID_INPUT_DATE_1") != day.strftime("%Y%m%d")
            or params.get("FID_INPUT_DATE_2") != day.strftime("%Y%m%d")):
        raise ValueError("Unbound daily adjustment control")
    rows = [r for r in receipt["output2"] if r.get("stck_bsop_date") == day.strftime("%Y%m%d")]
    if len(rows) != 1:
        raise ValueError("Absent or ambiguous daily control")
    return {f: int(rows[0][key]) for f, key in DAILY_FIELDS.items()}, int(rows[0]["acml_tr_pbmn"])


def reconstruct_adjusted_session(raw, anchor, day, symbol, original_receipt,
                                 adjusted_receipt, daily_reference, provider_rows):
    old = anchor[anchor.timestamp.dt.date == day].copy().reset_index(drop=True)
    if len(old) < 20 or old.timestamp.duplicated().any() or raw.timestamp.duplicated().any():
        raise ValueError("Insufficient or ambiguous archive anchors")
    if raw.empty or any(t.date() != day for t in raw.timestamp):
        raise ValueError("Recovery must contain exactly one requested session")
    original, daily_money = _daily(original_receipt, symbol, day, False)
    adjusted, adjusted_money = _daily(adjusted_receipt, symbol, day, True)
    if adjusted_money != daily_money:
        raise ValueError("Daily adjustment changes traded monetary value")
    if any(original[f] != float(daily_reference[f]) for f in DAILY_FIELDS):
        raise ValueError("Unadjusted provider daily differs from frozen daily basis")
    solutions = []
    for offset in range(-120, 121):
        shifted = raw.copy()
        shifted.timestamp += pd.Timedelta(seconds=offset)
        if any(t.second or t.microsecond for t in shifted.timestamp):
            continue
        overlap = old.merge(shifted, on="timestamp", suffixes=("_old", "_new"))
        if len(overlap) != len(old):
            continue
        try:
            lower, upper = _bounds(overlap)
        except ValueError:
            continue
        solutions.append((offset, shifted, overlap, lower, upper))
    if len(solutions) != 1:
        raise ValueError("No unique timestamp and rounded-basis reconstruction")
    offset, shifted, overlap, lower, upper = solutions[0]
    for f in PRICES:
        # A secondary cross-endpoint corroboration only. These retained daily
        # controls differ from the exact minute factor by up to two KRW/shares.
        # No decoded value or factor is estimated from this tolerance.
        if (original[f] * lower > adjusted[f] + 2
                or original[f] * upper < adjusted[f] - 2):
            raise ValueError("Minute conversion contradicts adjusted daily prices")
    if (original["volume"] / upper > adjusted["volume"] + 2
            or original["volume"] / lower < adjusted["volume"] - 2):
        raise ValueError("Minute conversion contradicts adjusted daily shares")
    price_book = {}
    for row in overlap.to_dict("records"):
        for field in PRICES:
            observed, decoded = int(row[field + "_new"]), int(row[field + "_old"])
            if observed in price_book and price_book[observed] != decoded:
                raise ValueError("Adjusted price has conflicting archived values")
            price_book[observed] = decoded
    money = {}
    for row in provider_rows:
        if row["stck_bsop_date"] != day.strftime("%Y%m%d"):
            continue
        stamp = pd.Timestamp(day.isoformat() + " " + row["stck_cntg_hour"], tz="Asia/Seoul") + pd.Timedelta(seconds=offset)
        amount = int(row["acml_tr_pbmn"])
        if stamp in money and money[stamp] != amount:
            raise ValueError("Conflicting cumulative turnover observations")
        money[stamp] = amount
    old_rows = old.set_index("timestamp").to_dict("index")
    decoded_rows, methods = [], Counter()
    opening, closing = get_trading_calendar().session_hours(day)
    if shifted.timestamp.iloc[0].time() != opening:
        raise ValueError("Full-session cumulative turnover needs the opening print")
    previous_money, auction = 0, None
    for row in shifted.to_dict("records"):
        stamp = row["timestamp"]
        if stamp not in money:
            raise ValueError("Missing cumulative turnover observation")
        increment = money[stamp] - previous_money
        previous_money = money[stamp]
        is_close = stamp == shifted.timestamp.iloc[-1] and stamp.time() >= closing
        decoded = {"timestamp": stamp}
        for field in PRICES:
            minimum, maximum = _candidates(row[field], lower, upper)
            if stamp in old_rows:
                value = int(old_rows[stamp][field])
            elif is_close:
                value = original["close"]
                methods["closing_price_from_unadjusted_daily"] += 1
            elif int(row[field]) in price_book:
                value = price_book[int(row[field])]
                methods["exact_archived_price_dictionary"] += 1
            elif minimum == maximum:
                value = minimum
                methods["unique_integer_price_inverse"] += 1
            else:
                raise ValueError("New adjusted price has no unambiguous supported inverse")
            if not minimum <= value <= maximum:
                raise ValueError("Decoded price does not round-trip through source interval")
            decoded[field] = float(value)
        minimum, maximum = _candidates(row["volume"], lower, upper, volume=True)
        if stamp in old_rows:
            quantity = int(old_rows[stamp]["volume"])
        elif is_close:
            if len({row[f] for f in PRICES}) != 1 or increment % original["close"]:
                raise ValueError("Closing auction quantity is not exactly supported by turnover")
            quantity = increment // original["close"]
            auction = {"timestamp": stamp.isoformat(), "price": original["close"],
                       "shares": quantity, "turnover_increment": increment}
            methods["closing_shares_from_turnover_divided_by_price"] += 1
        elif minimum == maximum:
            quantity = minimum
            methods["unique_integer_share_inverse"] += 1
        else:
            raise ValueError("Adjusted volume has an ambiguous original share count")
        if not minimum <= quantity <= maximum or quantity <= 0:
            raise ValueError("Decoded shares do not round-trip through source interval")
        decoded["volume"] = quantity
        if not decoded["low"] * quantity <= increment <= decoded["high"] * quantity:
            raise ValueError("Unadjusted minute OHLCV contradicts cumulative traded value")
        decoded_rows.append(decoded)
    combined = pd.DataFrame(decoded_rows)
    # Per-value inverses can each be feasible at different points of the factor
    # interval. Require one common factor for all decoded rows as well.
    _bounds(combined.merge(shifted, on="timestamp", suffixes=("_old", "_new")))
    pd.testing.assert_frame_equal(combined[combined.timestamp.isin(old.timestamp)].reset_index(drop=True),
                                  old[["timestamp", *PRICES, "volume"]], check_dtype=False)
    if any(original[f] != val for f, val in dict(open=combined.open.iloc[0], high=combined.high.max(),
                                                low=combined.low.min(), close=combined.close.iloc[-1]).items()):
        raise ValueError("Reconstructed session differs from unadjusted daily OHLC")
    if auction is None or original["volume"] < combined.volume.sum() or daily_money < previous_money:
        raise ValueError("Invalid closing or daily cumulative totals")
    new = combined[~combined.timestamp.isin(old.timestamp)]
    return combined, {"method": "adjusted_provider_reconstruction", "offset_seconds": offset,
        "old_rows": len(old), "matching_ohlcv_anchors": len(overlap), "added_rows": len(new), "revised_rows": 0,
        "added_timestamps": [t.isoformat() for t in new.timestamp],
        "price_factor_interval": [str(lower), str(upper)],
        "share_factor_interval": [str(1 / upper), str(1 / lower)],
        "rounding_model": "nearest_integer_half_up; conservatively closed factor bounds",
        "price_dictionary_entries": len(price_book), "new_value_methods": dict(methods),
        "turnover_bounds_verified_minutes": len(combined), "closing_auction": auction,
        "daily_ohlc_match": True,
        "adjusted_daily_control_tolerance_units": 2,
        "adjusted_daily_control_used_to_estimate_conversion": False,
        "daily_volume_excess_over_regular_session": int(original["volume"] - combined.volume.sum()),
        "daily_turnover_excess_over_regular_session": int(daily_money - previous_money),
        "daily_residual_cause_verified": False, "vendor_timestamp_contract_verified": False,
        "scope": "this symbol/session and these source bytes; no general adjustment contract inferred"}
