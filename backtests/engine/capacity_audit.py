"""Reconstruct capacity from source events and the journal, independently of quotes."""
from collections import defaultdict
from datetime import datetime, timedelta
import math

from strategy_common.market import MarketBar
from .capacity import continuous_interval, closing_auction_event


def audit_capacity(broker):
    policy = broker.capacity_policy
    if policy is None:
        return {"status": "not_enabled", "errors": []}
    attempts = [r for r in broker.journal if r["event"] == "capacity"]
    needed = {(r["order"]["symbol"], r["timestamp"], r["phase"]) for r in attempts}
    needed.update((f.symbol, f.timestamp.isoformat(), f.metadata.get("execution_phase")) for f in broker.fills)
    quotes, volumes, previous = {}, {}, {}
    # Classification is the shared source/session policy; quantity arithmetic is
    # deliberately independent of BarParticipationScenario.quote/budget.
    for batch in getattr(broker, "expected_batches", ()):
        for raw in batch:
            stamp = str(raw["timestamp"])
            symbol = raw["symbol"]
            phases = [phase for phase in ("open", "intrabar", "auction") if (symbol, stamp, phase) in needed]
            bar = MarketBar(**{**raw, "timestamp": datetime.fromisoformat(stamp)})
            prior = previous.get(symbol)
            for phase in phases:
                quantity, source, reason = 0, None, "available"
                if phase == "open":
                    basis = "previous_contiguous_continuous_5m"
                    if not continuous_interval(bar):
                        reason = "outside_continuous_interval"
                    elif (not continuous_interval(prior) or prior.symbol != symbol
                          or prior.timestamp.date() != bar.timestamp.date()
                          or prior.timestamp + timedelta(minutes=5) != bar.timestamp):
                        reason = "missing_prior_continuous_interval"
                    else:
                        source = prior
                elif phase == "auction":
                    basis = "single_price_closing_auction_event"
                    if closing_auction_event(bar):
                        source = bar
                    else:
                        reason = "unsupported_auction_observation"
                else:
                    basis, reason = "unsupported", "unsupported_execution_phase"
                if source is not None:
                    quantity = max(0, math.floor(float(source.volume) * policy.participation))
                quotes[(symbol, stamp, phase)] = {"budget": quantity,
                    "predictor_volume": float(source.volume) if source else 0,
                    "predictor_timestamp": source.timestamp.isoformat() if source else None,
                    "volume_basis": basis, "reason": reason}
                volumes[(symbol, stamp)] = float(bar.volume)
            # Invalid/missing intervals cannot become a valid predictor. Keeping
            # them here also makes the next adjacency check fail closed.
            previous[symbol] = bar
    errors, offers = [], {}
    used, bar_used = defaultdict(int), defaultdict(int)
    groups, attempt_groups = defaultdict(list), defaultdict(lambda: defaultdict(int))
    by_order = {}
    positions, marks = defaultdict(int), {}
    peak_gross, peak_equity = 0.0, float(broker.initial_equity)
    cash = float(broker.initial_equity)
    for row in broker.journal:
        kind = row["event"]
        if kind in {"submit", "amend"} and row.get("order", {}).get("order_type") not in {"MARKET", "CLOSE_AUCTION"}:
            errors.append("unsupported_capacity_order_type")
        if kind in {"opening_marks", "closing_marks"}:
            marks.update(row["prices"])
        if kind == "capacity":
            order = row["order"]
            key = (order["symbol"], row["timestamp"], row["phase"])
            quote = quotes.get(key)
            requested = int(row["requested_qty"])
            allowed = min(requested, max(0, (quote or {}).get("budget", 0) - used[key]))
            if quote is None or row["quote"] != quote or row["used_qty"] != used[key] or row["allowed_qty"] != allowed or requested < 0:
                errors.append("capacity_quote_mismatch")
            offer_key = (order["order_id"], row["timestamp"], row["phase"])
            if offer_key in offers:
                errors.append("duplicate_capacity_offer")
            offers[offer_key] = allowed
            group = attempt_groups[f"{row['phase']}_{order['side'].lower()}"]
            group["attempts"] += 1
            group["requested_attempt_shares"] += requested
            group["offered_attempt_shares"] += allowed
            group["zero_offer_attempts"] += int(allowed == 0)
            group["partial_offer_attempts"] += int(0 < allowed < requested)
            group["missing_predictor_attempts"] += int(row["quote"]["reason"] == "missing_prior_continuous_interval")
            if row["quote"]["reason"].startswith("unsupported_"):
                errors.append(row["quote"]["reason"])
            by_order.setdefault(order["order_id"], {"requested": requested, "filled": 0, "first_attempt": row["timestamp"]})
        elif kind == "fill":
            fill = row["fill"]
            stamp = fill["timestamp"].isoformat() if hasattr(fill["timestamp"], "isoformat") else fill["timestamp"]
            phase = fill["metadata"].get("execution_phase")
            key = (fill["symbol"], stamp, phase)
            offer = offers.pop((fill["order_id"], stamp, phase), None)
            qty = int(fill["qty"])
            if offer is None or qty > offer:
                errors.append("missing_or_exceeded_capacity_offer")
            used[key] += qty
            bar_used[key[:2]] += qty
            if used[key] > quotes.get(key, {}).get("budget", 0):
                errors.append("shared_capacity_exceeded")
            if bar_used[key[:2]] > volumes.get(key[:2], 0):
                errors.append("shared_bar_volume_exceeded")
            if fill["order_id"] in by_order:
                by_order[fill["order_id"]]["filled"] += qty
                by_order[fill["order_id"]].setdefault("first_fill", stamp)
            positions[(fill["strategy_id"], fill["symbol"])] += qty if fill["side"] == "BUY" else -qty
            cash = row["cash_after"]
        if kind in {"fill", "opening_marks", "closing_marks"}:
            gross = sum(qty * marks.get(symbol, 0) for (_, symbol), qty in positions.items())
            peak_gross = max(peak_gross, gross)
            peak_equity = max(peak_equity, cash + gross)
    violations = []
    for key, qty in used.items():
        volume = volumes.get(key[:2], 0)
        ratio = qty / volume if volume > 0 else None
        if ratio is not None:
            groups[key[2]].append(ratio)
        if volume <= 0 or bar_used[key[:2]] > volume:
            violations.append({"symbol": key[0], "timestamp": key[1], "phase": key[2],
                "filled_shares": qty, "bar_total_filled_shares": bar_used[key[:2]], "printed_shares": volume})
    forecast_ratios = [q["predictor_volume"] / volumes[key[:2]] for key, q in quotes.items()
        if key[2] == "open" and q["reason"] == "available" and volumes.get(key[:2], 0) > 0]
    delays = [(datetime.fromisoformat(r["first_fill"]) - datetime.fromisoformat(r["first_attempt"])).total_seconds()
              for r in by_order.values() if "first_fill" in r]
    def distribution(values):
        ordered = sorted(values)
        return {"count": len(ordered), "p50": ordered[(len(ordered)-1)//2] if ordered else None,
            "p95": ordered[int((len(ordered)-1)*.95)] if ordered else None, "max": max(ordered, default=None)}
    return {"status": "failed" if errors else "model_consistent", "errors": sorted(set(errors)),
        "contract": policy.contract(), "phase_liquidity_verified": False,
        "attempts_by_phase_and_side": {k: dict(v) for k, v in sorted(attempt_groups.items())},
        "combined_participation_by_phase": {phase: {"groups": len(values),
            "p50": sorted(values)[(len(values)-1)//2], "p95": sorted(values)[int((len(values)-1)*.95)],
            "max": max(values), "above_1pct": sum(v > .01 for v in values),
            "above_100pct": sum(v > 1 for v in values)} for phase, values in sorted(groups.items())},
        "impossible_volume_events": violations, "attempted_orders": len(by_order),
        "opening_forecast_to_realized_bar_volume": distribution(forecast_ratios),
        "first_eligible_attempt_to_first_fill_seconds": distribution(delays),
        "orders_with_fills": sum(v["filled"] > 0 for v in by_order.values()),
        "unfilled_order_shares": sum(max(v["requested"] - v["filled"], 0) for v in by_order.values()),
        "capital": {"initial_equity": broker.initial_equity, "peak_equity": peak_equity,
            "peak_gross_exposure": peak_gross, "final_equity": broker.equity_curve[-1]},
        "terminal_open_shares": sum(p.qty for p in broker.positions.values()),
        "terminal_pending_shares": sum(int(o.qty or 0) for o in broker.orders)}
