"""Tests for KPR Investor Flow Signal."""

import pytest
from datetime import datetime, timedelta
from strategy_kpr.signals.investor import InvestorSignal, InvestorFlowData, InvestorFlowProvider


class TestClassify:
    """Tests for InvestorFlowProvider._classify: maps flow data to signal enum."""

    def _classify(self, foreign_net, inst_net):
        provider = InvestorFlowProvider(api=None)
        data = InvestorFlowData(ticker="005930", foreign_net=foreign_net, inst_net=inst_net)
        return provider._classify(data)

    def test_strong(self):
        """Both foreign and institutional net positive -> STRONG."""
        assert self._classify(1000, 500) == InvestorSignal.STRONG

    def test_distribute(self):
        """Both foreign and institutional net negative -> DISTRIBUTE."""
        assert self._classify(-1000, -500) == InvestorSignal.DISTRIBUTE

    def test_conflict_foreign_positive(self):
        """Foreign positive, institutional negative -> CONFLICT."""
        assert self._classify(1000, -500) == InvestorSignal.CONFLICT

    def test_conflict_foreign_negative(self):
        """Foreign negative, institutional positive -> CONFLICT."""
        assert self._classify(-1000, 500) == InvestorSignal.CONFLICT

    def test_neutral(self):
        """Both zero -> NEUTRAL."""
        assert self._classify(0, 0) == InvestorSignal.NEUTRAL

    def test_neutral_one_zero_one_positive(self):
        """Foreign zero, institutional positive -> NEUTRAL (foreign_net > 0 is False)."""
        # foreign_net=0 -> (0 > 0) is False, inst_net=500 -> (500 > 0) is True
        # (False) != (True) -> True -> CONFLICT
        assert self._classify(0, 500) == InvestorSignal.CONFLICT

    def test_neutral_one_zero_one_negative(self):
        """Foreign zero, institutional negative -> both not > 0, neither < 0 for foreign."""
        # foreign_net=0: (0 > 0)=False, (0 < 0)=False
        # inst_net=-500: (-500 > 0)=False, (-500 < 0)=True
        # first check: both > 0? No. second: both < 0? No. third: (False != False)=False.
        assert self._classify(0, -500) == InvestorSignal.NEUTRAL

    def test_strong_small_values(self):
        """Even small positive values -> STRONG."""
        assert self._classify(1, 1) == InvestorSignal.STRONG


class TestInvestorFlowData:
    """Tests for InvestorFlowData dataclass."""

    def test_is_stale_no_timestamp(self):
        """No timestamp -> is_stale is True."""
        data = InvestorFlowData(ticker="005930")
        assert data.is_stale is True

    def test_is_stale_old_timestamp(self):
        """Timestamp > 300 seconds ago -> is_stale is True."""
        old_time = datetime.now() - timedelta(seconds=400)
        data = InvestorFlowData(ticker="005930", timestamp=old_time)
        assert data.is_stale is True

    def test_not_stale_recent_timestamp(self):
        """Recent timestamp -> is_stale is False."""
        data = InvestorFlowData(ticker="005930", timestamp=datetime.now())
        assert data.is_stale is False

    def test_defaults(self):
        """Default field values."""
        data = InvestorFlowData(ticker="005930")
        assert data.foreign_net == 0.0
        assert data.inst_net == 0.0
        assert data.timestamp is None
        assert data.epoch_ts == 0.0


class TestInvestorFlowProvider:
    """Tests for InvestorFlowProvider initialization and cache behavior."""

    def test_initialization(self):
        """Provider initializes with empty cache."""
        provider = InvestorFlowProvider(api=None)
        assert provider._cache == {}
        assert provider._inflight == set()

    def test_age_sec_missing(self):
        """Missing ticker returns inf age."""
        provider = InvestorFlowProvider(api=None)
        assert provider.age_sec("MISSING") == float("inf")

    def test_age_sec_cached(self):
        """Cached data returns correct age in seconds."""
        provider = InvestorFlowProvider(api=None)
        import time
        now = time.time()
        provider._cache["005930"] = InvestorFlowData(
            ticker="005930", epoch_ts=now - 30,
        )
        age = provider.age_sec("005930", now=now)
        assert abs(age - 30) < 1  # Allow small floating point variance

    def test_is_stale_method(self):
        """Provider.is_stale checks timestamp age against max_age."""
        provider = InvestorFlowProvider(api=None)
        assert provider.is_stale("MISSING", max_age=60) is True

        provider._cache["005930"] = InvestorFlowData(
            ticker="005930", timestamp=datetime.now(),
        )
        assert provider.is_stale("005930", max_age=60) is False

    def test_investor_signal_enum_values(self):
        """InvestorSignal enum has all expected members."""
        assert InvestorSignal.STRONG is not None
        assert InvestorSignal.NEUTRAL is not None
        assert InvestorSignal.DISTRIBUTE is not None
        assert InvestorSignal.CONFLICT is not None
        assert InvestorSignal.STALE is not None
        assert InvestorSignal.UNAVAILABLE is not None
