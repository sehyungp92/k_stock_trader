"""Tests for Nulrimok Watchlist Artifact Schema."""

import pytest
from strategy_nulrimok.dse.artifact import TickerArtifact, PositionArtifact, WatchlistArtifact


class TestWatchlistArtifact:
    """Tests for WatchlistArtifact: to_dict, get_ticker, all_tickers."""

    def test_to_dict(self):
        """to_dict returns correct serialized structure."""
        wa = WatchlistArtifact(
            date="2024-01-15",
            regime_tier="A",
            regime_score=0.75,
            candidates=[TickerArtifact(ticker="005930")],
            tradable=["005930"],
            active_set=["005930"],
        )
        d = wa.to_dict()
        assert d["date"] == "2024-01-15"
        assert d["regime_tier"] == "A"
        assert d["regime_score"] == 0.75
        assert len(d["candidates"]) == 1
        assert d["candidates"][0]["ticker"] == "005930"
        assert d["tradable"] == ["005930"]
        assert d["active_set"] == ["005930"]

    def test_to_dict_with_positions(self):
        """to_dict includes serialized positions."""
        pa = PositionArtifact(ticker="005930", entry_time="09:30:00", avg_price=72000, qty=100)
        wa = WatchlistArtifact(date="2024-01-15", positions=[pa])
        d = wa.to_dict()
        assert len(d["positions"]) == 1
        assert d["positions"][0]["ticker"] == "005930"
        assert d["positions"][0]["avg_price"] == 72000

    def test_to_dict_empty(self):
        """to_dict on empty artifact returns expected defaults."""
        wa = WatchlistArtifact(date="2024-01-15")
        d = wa.to_dict()
        assert d["date"] == "2024-01-15"
        assert d["candidates"] == []
        assert d["positions"] == []
        assert d["tradable"] == []
        assert d["overflow"] == []

    def test_get_ticker_found(self):
        """get_ticker returns matching TickerArtifact."""
        wa = WatchlistArtifact(
            date="2024-01-15",
            candidates=[TickerArtifact(ticker="005930"), TickerArtifact(ticker="000660")],
        )
        result = wa.get_ticker("005930")
        assert result is not None
        assert result.ticker == "005930"

    def test_get_ticker_not_found(self):
        """get_ticker returns None when ticker not in candidates."""
        wa = WatchlistArtifact(date="2024-01-15", candidates=[])
        assert wa.get_ticker("NOPE") is None

    def test_get_ticker_returns_correct_instance(self):
        """get_ticker returns the exact same object from candidates."""
        ta = TickerArtifact(ticker="005930", sector="IT")
        wa = WatchlistArtifact(date="2024-01-15", candidates=[ta])
        result = wa.get_ticker("005930")
        assert result is ta

    def test_all_tickers(self):
        """all_tickers returns list of ticker strings in order."""
        wa = WatchlistArtifact(
            date="2024-01-15",
            candidates=[TickerArtifact(ticker="A"), TickerArtifact(ticker="B")],
        )
        assert wa.all_tickers == ["A", "B"]

    def test_all_tickers_empty(self):
        """all_tickers returns empty list when no candidates."""
        wa = WatchlistArtifact(date="2024-01-15")
        assert wa.all_tickers == []

    def test_default_values(self):
        """Default values for optional fields."""
        wa = WatchlistArtifact(date="2024-01-15")
        assert wa.regime_tier == "C"
        assert wa.regime_score == 0.0
        assert wa.candidates == []
        assert wa.tradable == []
        assert wa.active_set == []
        assert wa.overflow == []
        assert wa.positions == []


class TestTickerArtifact:
    """Tests for TickerArtifact dataclass defaults."""

    def test_defaults(self):
        """Default tier is C, risk_multiplier 0, tradable False."""
        ta = TickerArtifact(ticker="005930")
        assert ta.regime_tier == "C"
        assert ta.risk_multiplier == 0.0
        assert ta.tradable is False
        assert ta.recommended_risk == 0.005
        assert ta.band_lower == 0.0
        assert ta.band_upper == 0.0
        assert ta.avwap_ref == 0.0

    def test_custom_fields(self):
        """Custom field values are preserved."""
        ta = TickerArtifact(
            ticker="005930",
            regime_tier="A",
            risk_multiplier=1.5,
            sector="IT",
            tradable=True,
            band_lower=95,
            band_upper=105,
            avwap_ref=100,
        )
        assert ta.regime_tier == "A"
        assert ta.risk_multiplier == 1.5
        assert ta.sector == "IT"
        assert ta.tradable is True

    def test_anchor_date_optional(self):
        """anchor_date defaults to None."""
        ta = TickerArtifact(ticker="005930")
        assert ta.anchor_date is None


class TestPositionArtifact:
    """Tests for PositionArtifact dataclass."""

    def test_creation(self):
        """Basic creation with required fields."""
        pa = PositionArtifact(ticker="005930", entry_time="09:30:00", avg_price=72000, qty=100)
        assert pa.ticker == "005930"
        assert pa.entry_time == "09:30:00"
        assert pa.avg_price == 72000
        assert pa.qty == 100

    def test_defaults(self):
        """Default flags are False/zero."""
        pa = PositionArtifact(ticker="005930", entry_time="09:30:00", avg_price=72000, qty=100)
        assert pa.exit_at_open is False
        assert pa.flow_reversal_flag is False
        assert pa.stop == 0.0

    def test_flow_reversal(self):
        """Flow reversal flags can be set."""
        pa = PositionArtifact(
            ticker="005930", entry_time="09:30:00", avg_price=72000, qty=100,
            flow_reversal_flag=True, exit_at_open=True,
        )
        assert pa.flow_reversal_flag is True
        assert pa.exit_at_open is True
