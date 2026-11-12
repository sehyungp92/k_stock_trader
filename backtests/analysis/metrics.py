from __future__ import annotations

import math
from statistics import mean, pstdev
from datetime import date, timedelta

from backtests.engine.contracts import METRIC_VERSION
from strategy_common.clock import ensure_kst
from kis_core.trading_calendar import get_trading_calendar

from strategy_common.events import TradeOutcome


def max_drawdown_pct(equity_curve: list[float]) -> float:
    peak = -math.inf
    worst = 0.0
    for value in equity_curve:
        peak = max(peak, float(value))
        if peak > 0:
            worst = max(worst, (peak - float(value)) / peak)
    return worst


def compute_trade_metrics(
    trades: list[TradeOutcome],
    equity_curve: list[float],
    *,
    initial_equity: float,
    timestamps: list | None = None,
    mark_coverage: list | None = None,
    daily_annualization: float = 252.0,
    declared_window: dict | None = None,
    initial_cash_only: bool = False,
) -> dict:
    total = len(trades)
    net_values = [float(trade.net_pnl) for trade in trades]
    gross_values = [float(trade.gross_pnl) for trade in trades]
    wins = [value for value in net_values if value > 0]
    losses = [value for value in net_values if value < 0]
    net_gains = sum(wins)
    net_losses = abs(sum(losses))
    gross_profit = sum(value for value in gross_values if value > 0)
    gross_loss = abs(sum(value for value in gross_values if value < 0))
    net_profit = sum(net_values)
    daily_values, coverage = session_equity(equity_curve, timestamps, mark_coverage,
        declared_window=declared_window, initial_cash_only=initial_cash_only)
    returns = [(current - previous) / previous for previous, current in zip(daily_values, daily_values[1:]) if previous]
    vol = pstdev(returns) if len(returns) > 1 else 0.0
    avg_return = mean(returns) if returns else 0.0
    sharpe = (avg_return / vol) * math.sqrt(daily_annualization) if vol > 0 else 0.0
    dd = max_drawdown_pct(equity_curve)
    return {
        "total_trades": float(total),
        "net_profit": float(net_profit),
        "net_gains": float(net_gains),
        "net_losses": float(net_losses),
        "gross_profit": float(gross_profit),
        "gross_loss": float(gross_loss),
        "profit_factor": float(net_gains / net_losses) if net_losses > 0 else (999.0 if net_gains > 0 else 0.0),
        "win_rate": float(len(wins) / total) if total else 0.0,
        "expectancy": float(net_profit / total) if total else 0.0,
        "avg_r": float(mean([trade.r_multiple for trade in trades])) if trades else 0.0,
        "expected_total_r": float(sum(trade.r_multiple for trade in trades)),
        "max_drawdown_pct": float(dd),
        "net_return_pct": float(net_profit / initial_equity) if initial_equity else 0.0,
        "sharpe": float(sharpe),
        "mfe_capture": _mfe_capture(trades),
        "same_bar_fill_count": None,
        "metric_version": METRIC_VERSION,
        "sharpe_sampling": "exchange_session_close_to_close",
        "sharpe_annualization": daily_annualization,
        "daily_equity_coverage": coverage,
        "daily_return_count": len(returns),
    }


def session_equity(equity_curve, timestamps, mark_coverage=None, *, declared_window=None, initial_cash_only=False):
    if timestamps is None:
        return [], {"status": "unavailable", "reason": "missing_timestamps"}
    if len(equity_curve) != len(timestamps) + 1:
        raise ValueError("Equity must include exactly one initial sample before timestamped samples")
    if any(a > b for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("Equity timestamps must be ordered")
    if mark_coverage is not None and len(mark_coverage) != len(timestamps):
        raise ValueError("Mark coverage must align with equity timestamps")
    calendar = get_trading_calendar()
    samples = {}
    for index, (timestamp, equity) in enumerate(zip(timestamps, equity_curve[1:], strict=True)):
        local = ensure_kst(timestamp)
        if calendar.is_trading_day(local.date()):
            samples[local.date()] = (float(equity), mark_coverage[index] if mark_coverage is not None else None)
    values = [float(equity_curve[0])] if equity_curve else []
    gaps, stale = [], []
    previous_day = None
    previous_coverage = None
    ordered = sorted(samples.items())
    if declared_window:
        start, end = (date.fromisoformat(str(declared_window[key])) for key in ("start", "end"))
        if end < start or any(day < start or day > end for day in samples):
            raise ValueError("Equity observations fall outside the declared window")
        expanded = []
        previous = (float(equity_curve[0]), {"cash_only": True}) if initial_cash_only else None
        day = start
        while day <= end:
            if calendar.is_trading_day(day):
                if day in samples:
                    previous = samples[day]
                    expanded.append((day, previous))
                elif previous and previous[1] and previous[1].get("cash_only"):
                    expanded.append((day, previous))
                else:
                    gaps.append(day.isoformat())
            day += timedelta(days=1)
        ordered = expanded
    for day, (equity, detail) in ordered:
        if previous_day is not None and not declared_window:
            missing = calendar.next_trading_day(previous_day)
            while missing < day:
                if previous_coverage and previous_coverage.get("cash_only"):
                    values.append(values[-1])
                else:
                    gaps.append(missing.isoformat())
                missing = calendar.next_trading_day(missing)
        if detail is None or detail.get("stale_symbols") or (
                not detail.get("cash_only") and not detail.get("closing_marks_complete")):
            stale.append(day.isoformat())
        values.append(equity)
        previous_day, previous_coverage = day, detail
    # Never annualize irregular multi-session observations as daily returns.
    return ([] if gaps else values), {"status": "complete" if not gaps and not stale else "unverified",
        "missing_sessions": gaps, "stale_or_unknown_mark_sessions": stale,
        "session_count": len(samples), "return_session_count": max(len(values) - 1, 0),
        "declared_window": declared_window,
        "cash_only_padded_sessions": len(ordered) - len(samples)}


def _mfe_capture(trades: list[TradeOutcome]) -> float:
    captures: list[float] = []
    for trade in trades:
        if trade.mfe <= 0 or trade.exit_price is None:
            continue
        captures.append(max(0.0, trade.exit_price - trade.entry_price) / trade.mfe)
    return float(mean(captures)) if captures else 0.0
