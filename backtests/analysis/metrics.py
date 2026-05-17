from __future__ import annotations

import math
from statistics import mean, pstdev

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
) -> dict[str, float]:
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
    returns = [
        (equity_curve[index] - equity_curve[index - 1]) / equity_curve[index - 1]
        for index in range(1, len(equity_curve))
        if equity_curve[index - 1]
    ]
    vol = pstdev(returns) if len(returns) > 1 else 0.0
    avg_return = mean(returns) if returns else 0.0
    sharpe = (avg_return / vol) * math.sqrt(252 * 390) if vol > 0 else 0.0
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
        "same_bar_fill_count": 0.0,
    }


def _mfe_capture(trades: list[TradeOutcome]) -> float:
    captures: list[float] = []
    for trade in trades:
        if trade.mfe <= 0 or trade.exit_price is None:
            continue
        captures.append(max(0.0, trade.exit_price - trade.entry_price) / trade.mfe)
    return float(mean(captures)) if captures else 0.0
