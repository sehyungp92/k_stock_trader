"""Project execution observations into realized economics and closed lifecycles.

Cost corrections replace an observation's economics; they are never another fill.
Daily economics are attributed to the original execution day, including restatements.
"""
from collections import defaultdict
from datetime import datetime, timezone
import math


def _number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Non-finite execution economics")
    return value


def event_day(row, field="exit_time"):
    value = row.get(field)
    if not value:
        return ""
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError("Execution timestamps require a timezone")
    return stamp.astimezone(timezone.utc).date().isoformat()


def project_executions(rows):
    """Deduplicate observations and apply only consecutive, bound revisions."""
    versions = defaultdict(dict)
    for row in rows:
        if row.get("stage") != "exit":
            continue
        key = (row.get("bot_id", ""), row["trade_id"], row.get("execution_id") or "legacy_close")
        revision = int(row.get("cost_revision", 0))
        old = versions[key].get(revision)
        if old is not None and old != row:
            raise ValueError("Conflicting execution identity/revision")
        versions[key][revision] = row
    result = []
    for revisions in versions.values():
        if sorted(revisions) != list(range(max(revisions) + 1)):
            raise ValueError("Missing execution cost revision")
        first = revisions[0]
        for row in revisions.values():
            for name in ("exit_time", "entry_price", "exit_price", "position_size", "remaining_qty", "gross_pnl"):
                if row.get(name) != first.get(name):
                    raise ValueError("Cost revision changed execution facts")
        row = dict(revisions[max(revisions)])
        fee = row.get("fees_paid")
        fee = _number(fee) if fee is not None else None
        # Legacy pnl is net only when a numeric fee was explicitly supplied.
        gross = row.get("gross_pnl")
        if gross is None:
            gross = (_number(row["exit_price"]) - _number(row["entry_price"])) * _number(row["position_size"])
            if row.get("side") == "SHORT":
                gross = -gross
        row.update(gross_pnl=_number(gross), fees_paid=fee,
                   net_pnl=round(_number(gross) - fee, 4) if fee is not None else None)
        result.append(row)
    return sorted(result, key=lambda r: (r["exit_time"], r["trade_id"], r.get("execution_id", "")))


def closed_lifecycles(executions):
    groups = defaultdict(list)
    for row in executions:
        groups[(row.get("bot_id", ""), row["trade_id"])].append(row)
    completed = []
    for fragments in groups.values():
        last = fragments[-1]
        if last.get("remaining_qty") not in (0, 0.0, None):
            continue
        if last.get("remaining_qty") is None and last.get("execution_id"):
            continue  # An identified execution needs an explicit closing quantity.
        cumulative = 0.0
        for row in fragments:
            cumulative += _number(row["position_size"])
            if row.get("lifetime_acquired_qty"):
                if (not math.isclose(cumulative, _number(row["cumulative_exited_qty"]))
                    or not math.isclose(cumulative + _number(row["remaining_qty"]), _number(row["lifetime_acquired_qty"]))):
                    raise ValueError("Incomplete execution lifecycle history")
        for prior, row in zip(fragments, fragments[1:]):
            if row.get("lifetime_acquired_qty") and prior.get("lifetime_acquired_qty"):
                if row["lifetime_acquired_qty"] < prior["lifetime_acquired_qty"]:
                    raise ValueError("Acquired quantity cannot decrease")
                continue
            if prior.get("remaining_qty") is None or not math.isclose(
                _number(prior["remaining_qty"]), _number(row["position_size"]) + _number(row["remaining_qty"])
            ):
                raise ValueError("Discontinuous trade exit quantities")
        outcome = dict(last)
        outcome.update(economics(fragments))
        outcome["pnl"] = outcome["net_pnl"]
        outcome["position_size"] = sum(_number(r["position_size"]) for r in fragments)
        cost = sum(_number(r["entry_price"]) * _number(r["position_size"]) for r in fragments)
        outcome["entry_price"] = cost / outcome["position_size"]
        outcome["exit_price"] = sum(_number(r["exit_price"]) * _number(r["position_size"]) for r in fragments) / outcome["position_size"]
        outcome["pnl_pct"] = outcome["gross_pnl"] / cost * 100 if cost else None
        outcome["fees_paid"] = outcome["total_fees"]
        outcome["execution_count"] = len(fragments)
        outcome["stage"] = "completed"
        outcome["schema_version"] = "trade_outcome_v1"
        outcome["event_type"] = "trade"
        outcome["pnl_basis"] = "net_with_supplied_costs" if outcome["net_pnl"] is not None else "gross_costs_unknown"
        completed.append(outcome)
    return completed


def economics(executions):
    unknown = sum(r["fees_paid"] is None for r in executions)
    known_fees = round(sum(r["fees_paid"] for r in executions if r["fees_paid"] is not None), 4)
    gross = round(sum(r["gross_pnl"] for r in executions), 4)
    return {"gross_pnl": gross, "net_pnl": None if unknown else round(gross - known_fees, 4),
            "total_fees": None if unknown else known_fees, "known_fees": known_fees,
            "unknown_cost_execution_count": unknown, "execution_count": len(executions),
            "costs_complete": unknown == 0}


def outcome_stats(completed):
    pnls = [r["net_pnl"] for r in completed if r["net_pnl"] is not None]
    wins, losses = [p for p in pnls if p > 0], [p for p in pnls if p < 0]
    unknown = len(completed) - len(pnls)
    return {"total_trades": len(completed), "win_count": len(wins), "loss_count": len(losses),
            "breakeven_count": sum(p == 0 for p in pnls), "unclassified_trade_count": unknown,
            "win_rate": None if unknown else (len(wins) / len(pnls) if pnls else 0.0),
            "best_trade_pnl": None if unknown else max(pnls, default=0),
            "worst_trade_pnl": None if unknown else min(pnls, default=0),
            "avg_win": sum(wins) / len(wins) if wins else 0.0,
            "avg_loss": sum(losses) / len(losses) if losses else 0.0,
            "profit_factor": None if unknown or (wins and not losses) else (sum(wins) / abs(sum(losses)) if losses else 0.0)}


def daily_accounting(rows, day):
    executions = project_executions(rows)
    today = [r for r in executions if event_day(r) == day]
    closed = [r for r in closed_lifecycles(executions) if event_day(r) == day]
    return {**economics(today), **outcome_stats(closed)}, today, closed
