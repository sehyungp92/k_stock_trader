"""Reconcile a daily-volume excess without assigning it to an execution bar."""
from decimal import Decimal

import pandas as pd

from kis_core.trading_calendar import get_trading_calendar
from scripts.update_kis_intraday import _normalize_kis_minute_row

FIELDS = ["timestamp", "open", "high", "low", "close", "volume"]
DAILY_FIELDS = dict(open="stck_oprc", high="stck_hgpr", low="stck_lwpr",
                    close="stck_clpr", volume="acml_vol")


def integer(value):
    number = Decimal(str(value))
    if not number.is_finite() or number != number.to_integral_value() or number < 0:
        raise ValueError("Expected nonnegative integer source value")
    return int(number)


def verify_daily_close_residual(frame, reference, receipt, provider_rows, day, symbol):
    """Check full source identity and exact closing-price residual arithmetic.

    This does not assert that every absent minute had zero trading or identify
    an after-hours trade's timestamp. Residual shares remain outside bar volume.
    """
    params = receipt.get("params", {})
    if (receipt.get("status") != "ok" or receipt.get("symbol") != symbol
            or receipt.get("date") != day.isoformat() or receipt.get("adjusted") is not False
            or params.get("FID_ORG_ADJ_PRC") != "1"
            or params.get("FID_COND_MRKT_DIV_CODE") != "J"
            or params.get("FID_INPUT_ISCD") != symbol
            or receipt.get("endpoint") != "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"):
        raise ValueError("Residual requires an original-price KIS daily control")
    daily = [r for r in receipt.get("output2", []) if r.get("stck_bsop_date") == day.strftime("%Y%m%d")]
    if len(daily) != 1:
        raise ValueError("Daily control absent or ambiguous")
    daily = daily[0]
    if any(integer(reference[k]) != integer(daily[v]) for k, v in DAILY_FIELDS.items()):
        raise ValueError("Daily control differs from retained reference")
    rows = {}
    for source in provider_rows:
        if source.get("stck_bsop_date") != day.strftime("%Y%m%d"):
            continue
        parsed = _normalize_kis_minute_row(source)
        if parsed is None:
            raise ValueError("Unparsed target-session source row")
        stamp = parsed["timestamp"]
        value = {**parsed, "turnover": integer(source["acml_tr_pbmn"])}
        if stamp in rows and rows[stamp] != value:
            raise ValueError("Conflicting cumulative source observations")
        rows[stamp] = value
    if not rows:
        raise ValueError("Full minute source absent")
    observed = pd.DataFrame([rows[t] for t in sorted(rows)])
    pd.testing.assert_frame_equal(frame[FIELDS].reset_index(drop=True), observed[FIELDS], check_dtype=False)
    calendar = get_trading_calendar()
    opening, closing = calendar.session_hours(day)
    match = calendar.closing_match_time(symbol, day)
    if (observed.timestamp.iloc[0].time() != opening or observed.timestamp.iloc[-1].time() != match
            or any(t.second or t.microsecond or not (opening <= t.time() <= closing or t.time() == match)
                   for t in observed.timestamp)):
        raise ValueError("Source does not span the declared regular session and closing match")
    actual = dict(open=frame.open.iloc[0], high=frame.high.max(), low=frame.low.min(), close=frame.close.iloc[-1])
    if any(integer(actual[k]) != integer(reference[k]) for k in actual):
        raise ValueError("Regular-session OHLC differs from daily control")
    previous = 0
    for row in observed.itertuples():
        delta = integer(row.turnover) - previous
        if not integer(row.low) * integer(row.volume) <= delta <= integer(row.high) * integer(row.volume):
            raise ValueError("Minute turnover is incompatible with its price and shares")
        previous = integer(row.turnover)
    last = observed.iloc[-1]
    if not (last.open == last.high == last.low == last.close):
        raise ValueError("Closing observation is not single-price")
    shares = integer(reference["volume"]) - sum(integer(v) for v in frame.volume)
    turnover = integer(daily["acml_tr_pbmn"]) - previous
    if shares <= 0 or turnover != shares * integer(reference["close"]):
        raise ValueError("Daily residual does not reconcile exactly at the closing price")
    return {"status": "daily_close_price_residual_reconciled", "daily_excess_shares": shares,
            "daily_excess_turnover": turnover, "closing_price": integer(reference["close"]),
            "source_minutes_checked": len(observed), "residual_added_to_bars": False,
            "after_hours_attribution": "consistent_with_post_close_trading_not_directly_timestamped",
            "independent_intraday_completeness_verified": False,
            "historical_qualification": False}
