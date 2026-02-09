"""Tests for Nulrimok Daily Selection Engine."""

import pytest
from datetime import date
from unittest.mock import MagicMock, patch

from strategy_nulrimok.dse.artifact import WatchlistArtifact, TickerArtifact, PositionArtifact


class TestWatchlistArtifact:
    """Tests for WatchlistArtifact dataclass."""

    def test_default_values(self):
        """Test default artifact values."""
        artifact = WatchlistArtifact(
            date="2024-01-15",
            regime_tier="B",
            regime_score=0.65,
        )

        assert artifact.date == "2024-01-15"
        assert artifact.regime_tier == "B"
        assert artifact.regime_score == 0.65
        assert artifact.candidates == []
        assert artifact.tradable == []
        assert artifact.active_set == []
        assert artifact.overflow == []
        assert artifact.positions == []

    def test_with_candidates(self):
        """Test artifact with candidates."""
        ticker = TickerArtifact(
            ticker="005930",
            sector="IT",
        )
        artifact = WatchlistArtifact(
            date="2024-01-15",
            regime_tier="A",
            regime_score=0.80,
            candidates=[ticker],
            tradable=["005930"],
            active_set=["005930"],
        )

        assert len(artifact.candidates) == 1
        assert artifact.candidates[0].ticker == "005930"
        assert "005930" in artifact.tradable
        assert "005930" in artifact.active_set

    def test_to_dict(self):
        """Test artifact serialization."""
        artifact = WatchlistArtifact(
            date="2024-01-15",
            regime_tier="A",
            regime_score=0.80,
        )

        result = artifact.to_dict()

        assert result["date"] == "2024-01-15"
        assert result["regime_tier"] == "A"
        assert result["regime_score"] == 0.80


class TestTickerArtifact:
    """Tests for TickerArtifact dataclass."""

    def test_required_fields(self):
        """Test ticker artifact required fields."""
        artifact = TickerArtifact(
            ticker="005930",
            sector="IT",
        )

        assert artifact.ticker == "005930"
        assert artifact.sector == "IT"


class TestPositionArtifact:
    """Tests for PositionArtifact dataclass."""

    def test_position_artifact(self):
        """Test position artifact creation."""
        artifact = PositionArtifact(
            ticker="005930",
            entry_time="09:30:00",
            avg_price=72000,
            qty=100,
            stop=70000,
            flow_reversal_flag=False,
            exit_at_open=False,
        )

        assert artifact.ticker == "005930"
        assert artifact.avg_price == 72000
        assert artifact.flow_reversal_flag is False

    def test_flow_reversal_flag(self):
        """Test flow reversal flag."""
        artifact = PositionArtifact(
            ticker="005930",
            entry_time="09:30:00",
            avg_price=72000,
            qty=100,
            stop=70000,
            flow_reversal_flag=True,
            exit_at_open=True,
        )

        assert artifact.flow_reversal_flag is True
        assert artifact.exit_at_open is True


class TestRegimeTierHandling:
    """Tests for regime tier handling."""

    def test_regime_c_returns_empty(self):
        """Test regime C returns empty artifact."""
        # Regime C = crisis, should return minimal artifact
        artifact = WatchlistArtifact(
            date="2024-01-15",
            regime_tier="C",
            regime_score=0.20,
        )

        assert artifact.regime_tier == "C"
        assert artifact.candidates == []
        assert artifact.tradable == []

    def test_regime_a_allows_trading(self):
        """Test regime A allows full trading."""
        artifact = WatchlistArtifact(
            date="2024-01-15",
            regime_tier="A",
            regime_score=0.85,
            tradable=["005930", "000660", "035420"],
            active_set=["005930", "000660"],
            overflow=["035420"],
        )

        assert artifact.regime_tier == "A"
        assert len(artifact.tradable) == 3
        assert len(artifact.active_set) == 2

    def test_regime_b_allows_trading(self):
        """Test regime B allows trading with caution."""
        artifact = WatchlistArtifact(
            date="2024-01-15",
            regime_tier="B",
            regime_score=0.55,
            tradable=["005930"],
            active_set=["005930"],
        )

        assert artifact.regime_tier == "B"
        assert len(artifact.tradable) == 1


class TestActiveSetRotation:
    """Tests for active set management."""

    def test_active_set_size_limit(self):
        """Test active set respects size limit (K)."""
        # Active set should be limited to K symbols
        tradable = ["005930", "000660", "035420", "051910", "006400", "035720"]
        active_set = tradable[:5]  # K=5
        overflow = tradable[5:]

        assert len(active_set) == 5
        assert len(overflow) == 1
        assert "035720" in overflow

    def test_overflow_contains_extra_tradable(self):
        """Test overflow contains extra tradable symbols."""
        tradable = ["005930", "000660", "035420", "051910", "006400", "035720", "105560"]
        active_set = tradable[:5]
        overflow = tradable[5:]

        assert len(overflow) == 2
        assert "035720" in overflow
        assert "105560" in overflow
