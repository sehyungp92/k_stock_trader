"""Tests for universe_filter module."""

import pytest
from unittest.mock import MagicMock

from kis_core.universe_filter import (
    UniverseFilterConfig,
    filter_universe,
)


def _make_api(
    price_data: dict | None = None,
    adtv_values: dict | None = None,
    price_error_tickers: set | None = None,
    adtv_error_tickers: set | None = None,
):
    """Build a mock API with configurable per-ticker responses."""
    api = MagicMock()
    price_data = price_data or {}
    adtv_values = adtv_values or {}
    price_error_tickers = price_error_tickers or set()
    adtv_error_tickers = adtv_error_tickers or set()

    def _get_current_price(ticker):
        if ticker in price_error_tickers:
            raise ConnectionError("API unavailable")
        return price_data.get(ticker)

    def _get_adtv_20d(ticker):
        if ticker in adtv_error_tickers:
            raise ConnectionError("API unavailable")
        return adtv_values.get(ticker, 0.0)

    api.get_current_price = MagicMock(side_effect=_get_current_price)
    api.get_adtv_20d = MagicMock(side_effect=_get_adtv_20d)
    return api


def _kospi_stock(price=70000, mcap=300e9):
    """Standard KOSPI common stock response. mcap is in KRW, stored as hts_avls in 억원."""
    return {
        "stck_prpr": price,
        "rprs_mrkt_kor_name": "KOSPI",
        "hts_avls": mcap / 1e8,
    }


def _kosdaq_stock(price=30000, mcap=50e9):
    """Standard KOSDAQ common stock response. mcap is in KRW, stored as hts_avls in 억원."""
    return {
        "stck_prpr": price,
        "rprs_mrkt_kor_name": "KOSDAQ",
        "hts_avls": mcap / 1e8,
    }


class TestPreferredShareDetection:
    """Check 1: preferred share suffix (local, no API call)."""

    def test_suffix_5_rejected(self):
        api = _make_api()
        valid, rejected = filter_universe(api, ["005935"])
        assert valid == []
        assert len(rejected) == 1
        assert rejected[0]["reason"] == "PREFERRED_SHARE"
        # No API calls made for local check
        api.get_current_price.assert_not_called()

    def test_suffix_K_rejected(self):
        api = _make_api()
        valid, rejected = filter_universe(api, ["00593K"])
        assert valid == []
        assert rejected[0]["reason"] == "PREFERRED_SHARE"

    def test_common_stock_not_rejected(self):
        api = _make_api(
            price_data={"005930": _kospi_stock()},
            adtv_values={"005930": 500e9},
        )
        valid, rejected = filter_universe(api, ["005930"])
        assert valid == ["005930"]
        assert rejected == []


class TestPriceCheck:
    """Check 2a: price == 0 or None → suspended/delisted."""

    def test_none_response_rejected(self):
        api = _make_api(price_data={})  # get_current_price returns None
        cfg = UniverseFilterConfig(skip_api_errors=False)
        valid, rejected = filter_universe(api, ["999999"], cfg)
        assert valid == []
        assert rejected[0]["reason"] == "NO_PRICE"

    def test_none_response_failopen(self):
        api = _make_api(price_data={})
        cfg = UniverseFilterConfig(skip_api_errors=True)
        valid, rejected = filter_universe(api, ["999999"], cfg)
        assert valid == ["999999"]

    def test_price_zero_rejected(self):
        api = _make_api(
            price_data={"000000": {"stck_prpr": 0, "rprs_mrkt_kor_name": "KOSPI"}},
        )
        valid, rejected = filter_universe(api, ["000000"])
        assert valid == []
        assert rejected[0]["reason"] == "NO_PRICE"


class TestMarketTypeCheck:
    """Check 2b: rprs_mrkt_kor_name not in {KOSPI, KOSDAQ}."""

    def test_kospi_passes(self):
        api = _make_api(
            price_data={"005930": _kospi_stock()},
            adtv_values={"005930": 500e9},
        )
        valid, _ = filter_universe(api, ["005930"])
        assert "005930" in valid

    def test_kosdaq_passes(self):
        api = _make_api(
            price_data={"035720": _kosdaq_stock()},
            adtv_values={"035720": 10e9},
        )
        valid, _ = filter_universe(api, ["035720"])
        assert "035720" in valid

    def test_etf_rejected(self):
        api = _make_api(
            price_data={"069500": {
                "stck_prpr": 35000,
                "rprs_mrkt_kor_name": "ETF",
                "hts_avls": 5000,
            }},
        )
        valid, rejected = filter_universe(api, ["069500"])
        assert valid == []
        assert rejected[0]["reason"] == "NOT_EQUITY"

    def test_konex_rejected(self):
        api = _make_api(
            price_data={"123456": {
                "stck_prpr": 5000,
                "rprs_mrkt_kor_name": "KONEX",
                "hts_avls": 100,
            }},
        )
        _, rejected = filter_universe(api, ["123456"])
        assert rejected[0]["reason"] == "NOT_EQUITY"

    def test_missing_market_field_failopen(self):
        """If rprs_mrkt_kor_name is absent, skip check (fail-open)."""
        api = _make_api(
            price_data={"005930": {
                "stck_prpr": 70000,
                "hts_avls": 3000,
            }},
            adtv_values={"005930": 500e9},
        )
        valid, _ = filter_universe(api, ["005930"])
        assert "005930" in valid

    def test_exclude_non_equity_disabled(self):
        """When exclude_non_equity=False, ETFs pass the market check."""
        cfg = UniverseFilterConfig(exclude_non_equity=False, mcap_min=0, adtv_min=0)
        api = _make_api(
            price_data={"069500": {
                "stck_prpr": 35000,
                "rprs_mrkt_kor_name": "ETF",
            }},
        )
        valid, _ = filter_universe(api, ["069500"], cfg)
        assert "069500" in valid


class TestMarketCapCheck:
    """Check 2c: market cap thresholds."""

    def test_low_mcap_rejected(self):
        api = _make_api(
            price_data={"111111": {
                "stck_prpr": 5000,
                "rprs_mrkt_kor_name": "KOSDAQ",
                "hts_avls": 100,  # 100억 = 10B KRW < 20B min
            }},
        )
        _, rejected = filter_universe(api, ["111111"])
        assert rejected[0]["reason"] == "LOW_MCAP"
        assert rejected[0]["value"] == 10e9

    def test_high_mcap_rejected(self):
        cfg = UniverseFilterConfig(mcap_max=100e9)
        api = _make_api(
            price_data={"005930": {
                "stck_prpr": 70000,
                "rprs_mrkt_kor_name": "KOSPI",
                "hts_avls": 5000,  # 5000억 = 500B KRW > 100B max
            }},
        )
        _, rejected = filter_universe(api, ["005930"], cfg)
        assert rejected[0]["reason"] == "HIGH_MCAP"

    def test_mcap_max_disabled_by_default(self):
        """mcap_max=0 means no upper cap."""
        api = _make_api(
            price_data={"005930": _kospi_stock(mcap=9999e9)},
            adtv_values={"005930": 500e9},
        )
        valid, _ = filter_universe(api, ["005930"])
        assert "005930" in valid

    def test_missing_mcap_field_failopen(self):
        """If no mcap field in response, skip mcap check."""
        api = _make_api(
            price_data={"005930": {
                "stck_prpr": 70000,
                "rprs_mrkt_kor_name": "KOSPI",
            }},
            adtv_values={"005930": 500e9},
        )
        valid, _ = filter_universe(api, ["005930"])
        assert "005930" in valid


class TestADTVCheck:
    """Check 3: ADTV >= adtv_min."""

    def test_low_adtv_rejected(self):
        api = _make_api(
            price_data={"222222": _kospi_stock()},
            adtv_values={"222222": 1e9},  # 1B < 3B min
        )
        _, rejected = filter_universe(api, ["222222"])
        assert rejected[0]["reason"] == "LOW_ADTV"
        assert rejected[0]["value"] == 1e9

    def test_sufficient_adtv_passes(self):
        api = _make_api(
            price_data={"005930": _kospi_stock()},
            adtv_values={"005930": 500e9},
        )
        valid, _ = filter_universe(api, ["005930"])
        assert "005930" in valid

    def test_adtv_disabled_when_zero(self):
        cfg = UniverseFilterConfig(adtv_min=0)
        api = _make_api(
            price_data={"005930": _kospi_stock()},
        )
        valid, _ = filter_universe(api, ["005930"], cfg)
        assert "005930" in valid
        api.get_adtv_20d.assert_not_called()


class TestAPIErrorHandling:
    """API errors are fail-open by default."""

    def test_price_api_error_failopen(self):
        api = _make_api(price_error_tickers={"005930"})
        valid, _ = filter_universe(api, ["005930"])
        assert "005930" in valid

    def test_price_api_error_failclosed(self):
        cfg = UniverseFilterConfig(skip_api_errors=False)
        api = _make_api(price_error_tickers={"005930"})
        valid, rejected = filter_universe(api, ["005930"], cfg)
        assert valid == []
        assert rejected[0]["reason"] == "API_ERROR"

    def test_adtv_api_error_failopen(self):
        api = _make_api(
            price_data={"005930": _kospi_stock()},
            adtv_error_tickers={"005930"},
        )
        valid, _ = filter_universe(api, ["005930"])
        assert "005930" in valid

    def test_adtv_api_error_failclosed(self):
        cfg = UniverseFilterConfig(skip_api_errors=False)
        api = _make_api(
            price_data={"005930": _kospi_stock()},
            adtv_error_tickers={"005930"},
        )
        _, rejected = filter_universe(api, ["005930"], cfg)
        assert rejected[0]["reason"] == "API_ERROR"


class TestMultipleTickers:
    """End-to-end with mixed valid/invalid tickers."""

    def test_mixed_universe(self):
        api = _make_api(
            price_data={
                "005930": _kospi_stock(price=70000, mcap=300e9),  # valid
                "035720": _kosdaq_stock(price=30000, mcap=50e9),  # valid
                "000000": {"stck_prpr": 0},                       # suspended
                "069500": {"stck_prpr": 35000, "rprs_mrkt_kor_name": "ETF", "hts_avls": 5000},  # ETF
                "111111": {"stck_prpr": 5000, "rprs_mrkt_kor_name": "KOSDAQ", "hts_avls": 50},  # low mcap (50억=5B KRW)
            },
            adtv_values={
                "005930": 500e9,
                "035720": 10e9,
            },
        )
        tickers = ["005930", "005935", "035720", "000000", "069500", "111111"]
        valid, rejected = filter_universe(api, tickers)

        assert valid == ["005930", "035720"]
        reasons = {r["ticker"]: r["reason"] for r in rejected}
        assert reasons["005935"] == "PREFERRED_SHARE"
        assert reasons["000000"] == "NO_PRICE"
        assert reasons["069500"] == "NOT_EQUITY"
        assert reasons["111111"] == "LOW_MCAP"

    def test_empty_universe(self):
        api = _make_api()
        valid, rejected = filter_universe(api, [])
        assert valid == []
        assert rejected == []

    def test_ordering_preserved(self):
        """Valid tickers should maintain their original order."""
        api = _make_api(
            price_data={
                "C": _kospi_stock(),
                "A": _kospi_stock(),
                "B": _kospi_stock(),
            },
            adtv_values={"C": 10e9, "A": 10e9, "B": 10e9},
        )
        valid, _ = filter_universe(api, ["C", "A", "B"])
        assert valid == ["C", "A", "B"]


class TestShortCircuit:
    """Verify checks short-circuit (no unnecessary API calls)."""

    def test_preferred_share_no_api_call(self):
        api = _make_api()
        filter_universe(api, ["005935"])
        api.get_current_price.assert_not_called()
        api.get_adtv_20d.assert_not_called()

    def test_no_price_skips_adtv(self):
        api = _make_api(
            price_data={"000000": {"stck_prpr": 0}},
        )
        filter_universe(api, ["000000"])
        api.get_adtv_20d.assert_not_called()

    def test_not_equity_skips_adtv(self):
        api = _make_api(
            price_data={"069500": {"stck_prpr": 35000, "rprs_mrkt_kor_name": "ETF"}},
        )
        filter_universe(api, ["069500"])
        api.get_adtv_20d.assert_not_called()
