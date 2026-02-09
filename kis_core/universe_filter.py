"""
Shared universe pre-filter safety net.

Validates tickers by basic eligibility (price, market type, market cap, ADTV)
before strategies build their state dicts. Catches suspended, delisted, or
illiquid stocks that would otherwise cause runtime errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

from loguru import logger

if TYPE_CHECKING:
    from .kis_client import KoreaInvestAPI


@dataclass
class UniverseFilterConfig:
    mcap_min: float = 20e9        # 20B KRW
    mcap_max: float = 0           # 0 = disabled
    adtv_min: float = 3e9         # 3B KRW
    exclude_non_equity: bool = True  # reject anything not KOSPI/KOSDAQ common stock
    skip_api_errors: bool = True     # keep ticker if API call fails (fail-open)


# Preferred-share suffix characters
_PREF_SUFFIXES = {"5", "K"}

# Allowed market classifications from KIS rprs_mrkt_kor_name field
_EQUITY_MARKETS = {"KOSPI", "KOSDAQ"}

# Market-cap field candidates in KIS inquire-price response.
# hts_avls is the HTS 시가총액 field returned in 억원 (1e8 KRW) units.
_MCAP_FIELDS_억 = ("hts_avls",)
# Fallback fields that may already be in KRW (kept for forward-compat).
_MCAP_FIELDS_KRW = ("total_mrkt_val", "mrkt_cap")

_억 = 1e8  # 1억원 = 100,000,000 KRW


def _extract_mcap(data: dict) -> float | None:
    """Extract market cap (in KRW) from price response, trying known field names."""
    # Primary: hts_avls is in 억원 → convert to KRW
    for f in _MCAP_FIELDS_억:
        val = data.get(f)
        if val is not None:
            try:
                v = float(val)
                if v > 0:
                    return v * _억
            except (ValueError, TypeError):
                continue
    # Fallback: fields assumed to be in raw KRW
    for f in _MCAP_FIELDS_KRW:
        val = data.get(f)
        if val is not None:
            try:
                v = float(val)
                if v > 0:
                    return v
            except (ValueError, TypeError):
                continue
    return None


def filter_universe(
    api: "KoreaInvestAPI",
    tickers: List[str],
    config: UniverseFilterConfig | None = None,
) -> Tuple[List[str], List[dict]]:
    """
    Filter tickers by basic eligibility.

    Returns (valid_tickers, rejections) where each rejection is
    {"ticker": str, "reason": str, "value": float}.
    """
    if config is None:
        config = UniverseFilterConfig()

    valid: List[str] = []
    rejected: List[dict] = []

    for ticker in tickers:
        reason = _check_ticker(api, ticker, config)
        if reason is None:
            valid.append(ticker)
        else:
            rejected.append(reason)

    logger.info(
        f"Universe filter: {len(valid)} passed, {len(rejected)} rejected "
        f"out of {len(tickers)}"
    )
    return valid, rejected


def _check_ticker(
    api: "KoreaInvestAPI",
    ticker: str,
    config: UniverseFilterConfig,
) -> dict | None:
    """Check a single ticker. Returns rejection dict or None if valid."""

    # --- Check 1: Preferred share (local, no API call) ---
    if ticker and ticker[-1] in _PREF_SUFFIXES:
        return {"ticker": ticker, "reason": "PREFERRED_SHARE", "value": 0.0}

    # --- Check 2: Price + market type + market cap (single API call) ---
    try:
        data = api.get_current_price(ticker)
    except Exception as e:
        logger.debug(f"Universe filter: API error for {ticker}: {e}")
        if config.skip_api_errors:
            return None  # fail-open
        return {"ticker": ticker, "reason": "API_ERROR", "value": 0.0}

    if data is None:
        if config.skip_api_errors:
            return None  # fail-open
        return {"ticker": ticker, "reason": "NO_PRICE", "value": 0.0}

    # Price check (suspended / delisted)
    price = 0.0
    try:
        price = float(data.get("stck_prpr", 0))
    except (ValueError, TypeError):
        pass

    if price == 0:
        return {"ticker": ticker, "reason": "NO_PRICE", "value": 0.0}

    # Market type check (KOSPI/KOSDAQ only)
    if config.exclude_non_equity:
        mrkt_name = data.get("rprs_mrkt_kor_name") or ""
        if mrkt_name and not mrkt_name.startswith(("KOSPI", "KOSDAQ", "KSQ")):
            return {"ticker": ticker, "reason": "NOT_EQUITY", "value": 0.0}
        # If field is absent → skip check (fail-open)

    # Market cap check
    mcap = _extract_mcap(data)
    if mcap is not None:
        if mcap < config.mcap_min:
            return {"ticker": ticker, "reason": "LOW_MCAP", "value": mcap}
        if config.mcap_max > 0 and mcap > config.mcap_max:
            return {"ticker": ticker, "reason": "HIGH_MCAP", "value": mcap}

    # --- Check 3: ADTV (separate API call) ---
    if config.adtv_min > 0:
        try:
            adtv = api.get_adtv_20d(ticker)
        except Exception as e:
            logger.debug(f"Universe filter: ADTV error for {ticker}: {e}")
            if config.skip_api_errors:
                return None  # fail-open
            return {"ticker": ticker, "reason": "API_ERROR", "value": 0.0}

        if adtv < config.adtv_min:
            return {"ticker": ticker, "reason": "LOW_ADTV", "value": adtv}

    return None  # all checks passed
