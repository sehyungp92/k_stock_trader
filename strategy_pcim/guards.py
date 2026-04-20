"""Pure decision helpers for PCIM runtime guards."""

from __future__ import annotations


def should_trigger_intraday_halt(
    kospi_prev_close: float | None,
    kospi_now: float,
    halt_threshold: float,
) -> tuple[bool, float | None]:
    """Return whether the KOSPI drawdown should trigger an intraday halt."""
    if not kospi_prev_close or kospi_prev_close <= 0:
        return False, None
    if kospi_now <= 0:
        return False, None

    drawdown = (kospi_now - kospi_prev_close) / kospi_prev_close
    return drawdown <= halt_threshold, drawdown
