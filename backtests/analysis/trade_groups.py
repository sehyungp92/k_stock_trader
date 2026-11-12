"""Count a position lifetime once even when entries and exits fill in pieces."""
from dataclasses import replace


def trade_group_key(trade):
    # Average cost can change during fragmented entries; it is not an identity.
    return (trade.strategy_id, trade.symbol, trade.entry_fill_time.isoformat())


def collapse_exit_legs(trades):
    grouped = {}
    for trade in trades:
        grouped.setdefault(trade_group_key(trade), []).append(trade)
    return [combine_trade_legs(legs) if len(legs) > 1 else legs[0] for legs in grouped.values()]


def combine_trade_legs(legs):
    ordered = sorted(legs, key=lambda t: t.exit_fill_time or t.entry_fill_time)
    first, final = ordered[0], ordered[-1]
    quantity = sum(int(t.qty) for t in ordered)
    if quantity <= 0:
        return final
    partial_count = sum(t.exit_reason.lower() in {"partial_profit", "partial_target"}
        or str(t.cohort_metadata.get("exit_cohort", "")).lower() == "partial" for t in ordered)
    leg_rows = []
    for trade in ordered:
        risk = float(trade.route_metadata.get("risk_per_share", 0) or 0) * trade.qty
        leg_rows.append({"qty": trade.qty, "exit_time": trade.exit_fill_time.isoformat() if trade.exit_fill_time else "",
            "exit_price": float(trade.exit_price or trade.entry_price), "exit_reason": trade.exit_reason,
            "net_r": trade.net_pnl / risk if risk > 0 else trade.r_multiple})
    metadata = {"partial_taken": partial_count > 0, "partial_exit_count": partial_count, "exit_leg_count": len(ordered)}
    return replace(first, qty=quantity,
        entry_price=sum(t.entry_price * t.qty for t in ordered) / quantity,
        exit_price=sum(float(t.exit_price or t.entry_price) * t.qty for t in ordered) / quantity,
        exit_fill_time=final.exit_fill_time, exit_reason=final.exit_reason,
        gross_pnl=sum(t.gross_pnl for t in ordered), net_pnl=sum(t.net_pnl for t in ordered),
        commission=sum(t.commission for t in ordered), realized=all(t.realized for t in ordered),
        route_metadata={**first.route_metadata, **metadata, "exit_legs": leg_rows, "final_exit_reason": final.exit_reason},
        cohort_metadata={**final.cohort_metadata, **metadata},
        mfe=max(t.mfe for t in ordered), mae=min(t.mae for t in ordered))
