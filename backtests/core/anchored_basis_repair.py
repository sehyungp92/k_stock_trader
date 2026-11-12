"""Exact original-unit reconstruction for a bounded KRX price band.

Prices in the 50,000--199,999 KRW band use 100 KRW limit ticks and may
trade at their midpoint (50 KRW lattice). This constraint resolves integer
rounding ambiguity; it is not nearest-tick rounding. Every decoded value must
be the ONLY admissible inverse. No corporate-action share ratio is inferred.
"""
from collections import Counter
import pandas as pd

from backtests.core.adjusted_recovery import _bounds, _candidates, PRICES
from backtests.core.daily_volume_residual import FIELDS, integer
from scripts.update_kis_intraday import _normalize_kis_minute_row
from kis_core.trading_calendar import get_trading_calendar

GRID_SOURCES = (
    "https://global.krx.co.kr/contents/GLB/06/0602/0602010201/GLB0602010201T3.jsp",
    "https://regulation.krx.co.kr/contents/RGL/03/03010208/RGL03010208.jsp",
)


def original_price(observed, lower, upper):
    first, last = _candidates(observed, lower, upper)
    if first < 50_000 or last >= 200_000:
        raise ValueError("Outside the documented repair price band")
    possible = [value for value in range(first, last + 1) if value % 50 == 0]
    if len(possible) != 1:
        raise ValueError("No unique original price on the permitted trading lattice")
    return possible[0], "unique_integer_inverse" if first == last else "unique_krx_midpoint_lattice_inverse"


def reconstruct_original_session(raw, original_anchor, adjusted_anchor, original_daily,
                                 adjusted_daily, provider_rows, day, symbol):
    """Verify source rows, all inverses, common factor and monetary conservation."""
    for frame in (raw, original_anchor, adjusted_anchor):
        if frame.empty or frame.timestamp.duplicated().any() or not frame.timestamp.is_monotonic_increasing:
            raise ValueError("Missing, unordered or duplicated source observations")
    if len(original_anchor) < 20 or len(original_anchor) != len(adjusted_anchor):
        raise ValueError("Insufficient paired original/adjusted anchors")
    if list(original_anchor.timestamp) != list(adjusted_anchor.timestamp):
        raise ValueError("Paired anchors must use identical clocks")
    if any(t.date() != day or t.second or t.microsecond for t in raw.timestamp):
        raise ValueError("Reconstruction must be a single exact-minute session")
    lower, upper = _bounds(original_anchor.merge(adjusted_anchor, on="timestamp", suffixes=("_old", "_new")))
    controls = {}
    for source in provider_rows:
        if source.get("stck_bsop_date") != day.strftime("%Y%m%d"):
            continue
        parsed = _normalize_kis_minute_row(source)
        if parsed is None:
            raise ValueError("Unparsed requested source observation")
        value = {**parsed, "turnover": integer(source["acml_tr_pbmn"])}
        stamp = parsed["timestamp"]
        if stamp in controls and controls[stamp] != value:
            raise ValueError("Conflicting provider observations")
        controls[stamp] = value
    observed = pd.DataFrame([controls[t] for t in sorted(controls)])
    pd.testing.assert_frame_equal(raw[FIELDS].reset_index(drop=True), observed[FIELDS], check_dtype=False)
    calendar = get_trading_calendar()
    opening, closing = calendar.session_hours(day)
    match = calendar.closing_match_time(symbol, day)
    if (raw.timestamp.iloc[0].time() != opening or raw.timestamp.iloc[-1].time() != match
            or any(not (opening <= t.time() <= closing or t.time() == match) for t in raw.timestamp)):
        raise ValueError("Incomplete session boundaries or unsupported observation time")
    methods, rows, previous = Counter(), [], 0
    for row in raw.to_dict("records"):
        result = {"timestamp": row["timestamp"]}
        for field in PRICES:
            result[field], method = original_price(row[field], lower, upper)
            methods[method] += 1
        first, last = _candidates(row["volume"], lower, upper, volume=True)
        if first != last or first <= 0:
            raise ValueError("No unique positive original integer share count")
        result["volume"] = first
        total = controls[row["timestamp"]]["turnover"]
        delta = total - previous; previous = total
        if not result["low"] * first <= delta <= result["high"] * first:
            raise ValueError("Decoded minute contradicts observed traded value")
        if not result["low"] <= min(result["open"], result["close"]) <= max(result["open"], result["close"]) <= result["high"]:
            raise ValueError("Invalid decoded OHLC ordering")
        rows.append(result)
    decoded = pd.DataFrame(rows)
    common = pd.concat([
        original_anchor.merge(adjusted_anchor, on="timestamp", suffixes=("_old", "_new")),
        decoded.merge(raw, on="timestamp", suffixes=("_old", "_new")),
    ], ignore_index=True)
    lower, upper = _bounds(common)
    actual = dict(open=decoded.open.iloc[0], high=decoded.high.max(), low=decoded.low.min(), close=decoded.close.iloc[-1])
    if any(integer(actual[k]) != integer(original_daily[k]) for k in PRICES):
        raise ValueError("Reconstruction differs from original daily OHLC")
    for field in PRICES:
        if (integer(original_daily[field]) * lower > integer(adjusted_daily[field]) + 2
                or integer(original_daily[field]) * upper < integer(adjusted_daily[field]) - 2):
            raise ValueError("Daily adjusted prices contradict the anchored factor")
    if (integer(original_daily["volume"]) / upper > integer(adjusted_daily["volume"]) + 2
            or integer(original_daily["volume"]) / lower < integer(adjusted_daily["volume"]) - 2):
        raise ValueError("Daily adjusted shares contradict the anchored factor")
    if integer(original_daily["turnover"]) != integer(adjusted_daily["turnover"]):
        raise ValueError("Daily turnover changes with the adjustment flag")
    excess = integer(original_daily["volume"]) - int(decoded.volume.sum())
    excess_money = integer(original_daily["turnover"]) - previous
    if excess < 0 or excess_money != excess * integer(original_daily["close"]):
        raise ValueError("Daily residual lacks exact closing-price reconciliation")
    last = decoded.iloc[-1]
    if len({last[f] for f in PRICES}) != 1:
        raise ValueError("Closing match is not a single price")
    # If the anchor session is being reconstructed, require full byte-value parity.
    if original_anchor.timestamp.iloc[0].date() == day:
        pd.testing.assert_frame_equal(decoded[FIELDS], original_anchor[FIELDS].reset_index(drop=True), check_dtype=False)
    return decoded, {"pass": True, "symbol": symbol, "date": day.isoformat(), "rows": len(decoded),
        "methods": dict(methods), "unique_original_share_inverses": len(decoded),
        "common_price_factor_interval": list(map(str, (lower, upper))),
        "factor_inferred_from_corporate_entitlement": False,
        "price_lattice": {"quantum": 50, "band": [50_000, 200_000], "sources": list(GRID_SOURCES)},
        "daily_ohlc_match": True, "turnover_bounds_verified_minutes": len(decoded),
        "daily_excess_shares": excess, "daily_excess_turnover": excess_money,
        "residual_added_to_bars": False, "scope": "Bounded source reconstruction; no general vendor adjustment contract"}
