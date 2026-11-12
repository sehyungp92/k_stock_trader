"""Causal research contracts for KALCB executed-value forecasts and admission.

The value labels are standalone, fixed-equity policy outcomes. Portfolio replay
is still required: these labels cannot be added to obtain portfolio performance.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from datetime import date, datetime, time
import hashlib
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "kalcb-executed-value-v1"
REFERENCE_EQUITY = 100_000_000.0
KST = ZoneInfo("Asia/Seoul")
FEATURES = (
    "first30_ret", "first30_vwap_ret", "first30_gap", "first30_rel_volume_log",
    "first30_close_location", "first30_open_drawdown", "first30_low_vs_prev_close",
    "first30_range_atr", "daily_return_5d", "daily_return_20d", "daily_atr_pct",
    "daily_adv20_krw_log", "stock_sector_daily_ret5_spread",
    "sector_intraday_ret", "sector_intraday_breadth", "sector_daily_score_pct",
    "first30_volume_log", "last_opening_volume_share", "opening_volume_min_share",
    "opening_capacity_ratio",
)
SCENARIOS = {
    "baseline": (.01, 3.0, "research-baseline"),
    "participation": (.005, 3.0, "participation-stress"),
    "cost": (.01, 13.0, "cost-stress"),
    "combined": (.005, 13.0, "combined-stress"),
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def feature_matrix(rows, columns=FEATURES):
    import numpy as np
    if not columns or set(columns) - set(FEATURES):
        raise ValueError("Predictors must belong to the causal feature allowlist")
    return np.asarray([[float(row["features"].get(k)) if row["features"].get(k) is not None
                        else np.nan for k in columns] for row in rows], dtype=float)


def eligible(row, seed, q3, *, rank):
    from backtests.strategies.kalcb.fixed_trade_plan_phase import _configured_entry_routes, _route_candidate_passes
    meta = {**row["guard_features"], "candidate_rank": rank, "frontier_rank": rank,
            "frontier_initial_active": True, "frontier_selection_score": 0.0}
    def passes(mutations):
        return any(_route_candidate_passes(route, mutations, meta)[0]
                   for route in _configured_entry_routes(mutations))
    return passes(seed) or (passes(q3) and row["sector"] != "FINANCIAL"
                           and float(meta.get("stock_sector_daily_ret5_spread") or 0) >= -.036622)


def admit(rows, scores, predicate, *, construction, budget=8, shortlist=16, threshold=None):
    """Ranking and guard order are explicit; no outcomes influence selection."""
    if len(rows) != len(scores) or budget < 1 or shortlist < budget:
        raise ValueError("Invalid admission inputs")
    if construction not in {"after", "before"}:
        raise ValueError("Unknown eligibility construction")
    ordered = sorted(zip(rows, scores), key=lambda pair: (-float(pair[1]), pair[0]["symbol"]))
    selected, reasons = [], {}
    for raw_rank, (row, score) in enumerate(ordered, 1):
        key = row["opportunity_id"]
        if not math.isfinite(float(score)):
            raise ValueError("Non-finite forecast")
        if construction == "after" and raw_rank > shortlist:
            reasons[key] = "outside_shortlist"
            continue
        guard_rank = raw_rank if construction == "after" else len(selected) + 1
        if not predicate(row, guard_rank):
            reasons[key] = "guard"
        elif threshold is not None and score <= threshold:
            reasons[key] = "net_value_threshold"
        elif len(selected) >= budget:
            reasons[key] = "admission_budget"
        else:
            selected.append((row, float(score)))
            reasons[key] = "admitted"
    return selected, reasons


def snapshot_for(day, selected, *, source, forecast=None):
    from strategy_kalcb.models import KALCBDailyCandidate, KALCBDailySnapshot
    candidates = []
    for rank, (row, score) in enumerate(selected, 1):
        candidate = KALCBDailyCandidate.from_json_dict(row["candidate"])
        candidates.append(replace(candidate, selection_score=score,
            metadata={**candidate.metadata, "candidate_rank": rank, "frontier_rank": rank,
                      "frontier_initial_active": True, "frontier_role": "initial_active",
                      "frontier_selection_score": score, "first30_score": score,
                      "feature_available_at": row.get("feature_available_at"),
                      "value_forecast_provenance": forecast or {}}))
    generated_at = max((datetime.fromisoformat(row["feature_available_at"])
                        for row, _ in selected if row.get("feature_available_at")),
                       default=datetime.combine(date.fromisoformat(day), time.min, KST))
    return KALCBDailySnapshot(date.fromisoformat(day), tuple(candidates), source,
        generated_at,
        metadata={"active_symbols": [c.symbol for c in candidates], "active_symbol_count": len(candidates),
                  "candidate_pool_count": len(candidates), "selection_count": len(candidates),
                  "frontier_symbol_count": len(candidates), "research_only": True,
                  "value_policy_version": VERSION})


def ledger_value(fills, costs, *, terminal_inventory=0, pending=0):
    """Cash-flow target, independently reconciled with the broker by callers."""
    cash, fees, bought, sold = 0.0, 0.0, 0, 0
    orders = defaultdict(list)
    for fill in fills:
        side, qty, price = fill.side, int(fill.qty), float(fill.price)
        if qty <= 0 or not math.isfinite(price) or price <= 0 or side not in {"BUY", "SELL"}:
            raise ValueError("Invalid fill")
        fee = qty * price * (costs.commission_bps + (costs.tax_bps_on_sell if side == "SELL" else 0)) / 10000
        fees += fee
        cash += qty * price * (1 if side == "SELL" else -1) - fee
        if side == "BUY":
            bought += qty
            orders[fill.order_id].append(fill)
        else:
            sold += qty
        if sold > bought:
            raise ValueError("Sell precedes owned inventory")
    if bought - sold != terminal_inventory:
        raise ValueError("Fill/inventory mismatch")
    unresolved = bool(terminal_inventory or pending)
    return {"net_cash_flow_krw": cash, "net_value_krw": None if unresolved else cash,
            "net_value_equity_bps": None if unresolved else cash / REFERENCE_EQUITY * 10000,
            "fees_and_tax_krw": fees, "buy_qty": bought, "sell_qty": sold,
            "entry_orders_filled": len(orders), "entry_orders_multislice": sum(len(v) > 1 for v in orders.values()),
            "entry_orders_incomplete": sum(sum(f.qty for f in fs) < max(int(f.metadata["order_qty"]) for f in fs)
                                            for fs in orders.values()),
            "unresolved": unresolved}


def physical_fill_failures(fills):
    """Ex-post impossibility test, never a future-volume execution input."""
    quantities, volumes = defaultdict(int), {}
    for fill in fills:
        metadata = fill["metadata"] if isinstance(fill, dict) else fill.metadata
        qty = fill["qty"] if isinstance(fill, dict) else fill.qty
        bar = metadata["execution_bar"]
        key = (bar["symbol"], bar["timestamp"])
        volume = float(bar["volume"])
        if key in volumes and volumes[key] != volume:
            raise ValueError("Inconsistent execution-bar volume")
        volumes[key] = volume
        quantities[key] += int(qty)
    return [{"symbol": key[0], "timestamp": key[1], "quantity": qty, "realized_bar_volume": volumes[key],
             "reason": "shared_bar_volume_exceeded"}
            for key, qty in sorted(quantities.items()) if qty > volumes[key]]


def mature_training_indices(rows, sessions, valid_from, *, min_sessions=60, embargo=1):
    first = sessions.index(valid_from)
    allowed = set(sessions[:max(0, first - embargo)])
    cutoff = datetime.combine(date.fromisoformat(valid_from), time.min, KST)
    indices = []
    for i, row in enumerate(rows):
        available = row.get("label_available_at")
        if row["trade_date"] in allowed and available and datetime.fromisoformat(available) < cutoff:
            indices.append(i)
    if len({rows[i]["trade_date"] for i in indices}) < min_sessions:
        return []
    return indices


def outer_windows(sessions, *, min_train=60, block=20, embargo=1):
    if sessions != sorted(set(sessions)) or min_train < 1 or block < 1 or embargo < 0:
        raise ValueError("Invalid session calendar/window")
    return [(sessions[i], sessions[min(i + block, len(sessions)) - 1])
            for i in range(min_train + embargo, len(sessions), block)]


def choose_threshold(records, thresholds=(None, 0.0, 2.0, 5.0, 10.0), *, minimum_days=20):
    """Calibrate on earlier OOF action outcomes; final decisions need portfolio replay.

    Inputs are already eligible/budgeted OOF actions. Thresholds are reference-equity
    basis points. Cash wins negative/tied utility; otherwise prefer more positive
    trades at equal value. Liquidation-failing calibration policies are infeasible.
    """
    days = {r["trade_date"] for r in records}
    if len(days) < minimum_days:
        return 0.0, {"method": "predeclared_zero_until_sufficient_oof_history", "sessions": len(days)}
    trials = [{"threshold": "cash", "sum_net_value_bps": 0.0, "admissions": 0, "feasible": True}]
    for threshold in thresholds:
        selected = [r for r in records if threshold is None or r["score"] > threshold]
        feasible = all(not r["deadline_violation"] and r["net_value_equity_bps"] is not None for r in selected)
        trials.append({"threshold": threshold, "sum_net_value_bps": sum(r["net_value_equity_bps"] or 0 for r in selected),
                       "admissions": len(selected), "feasible": feasible})
    best = max((t for t in trials if t["feasible"]),
               key=lambda t: (t["sum_net_value_bps"], t["admissions"] if t["sum_net_value_bps"] > 0 else -t["admissions"]))
    return best["threshold"], {"method": "earlier_oof_standalone_value_not_portfolio_utility",
                              "sessions": len(days), "trials": trials, "selected": best}
