"""Tests for the PCIM intraday halt decision helper."""

from __future__ import annotations

import pytest

from strategy_pcim.config.constants import INTRADAY_HALT_KOSPI_DD_PCT
from strategy_pcim.guards import should_trigger_intraday_halt


def test_no_halt_when_realtime_drawdown_is_above_threshold():
    should_halt, drawdown = should_trigger_intraday_halt(
        kospi_prev_close=100.0,
        kospi_now=100.0 * (1 + (INTRADAY_HALT_KOSPI_DD_PCT / 2)),
        halt_threshold=INTRADAY_HALT_KOSPI_DD_PCT,
    )

    assert should_halt is False
    assert drawdown is not None
    assert drawdown > INTRADAY_HALT_KOSPI_DD_PCT


def test_halt_when_realtime_drawdown_reaches_threshold():
    should_halt, drawdown = should_trigger_intraday_halt(
        kospi_prev_close=100.0,
        kospi_now=100.0 * (1 + INTRADAY_HALT_KOSPI_DD_PCT),
        halt_threshold=INTRADAY_HALT_KOSPI_DD_PCT,
    )

    assert should_halt is True
    assert drawdown == pytest.approx(INTRADAY_HALT_KOSPI_DD_PCT)


def test_invalid_prices_do_not_trigger_halt():
    assert should_trigger_intraday_halt(None, 97.0, INTRADAY_HALT_KOSPI_DD_PCT) == (False, None)
    assert should_trigger_intraday_halt(100.0, 0.0, INTRADAY_HALT_KOSPI_DD_PCT) == (False, None)
