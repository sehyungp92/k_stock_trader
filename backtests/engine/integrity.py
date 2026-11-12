"""Independent checks of a recorded path, never a second strategy simulator."""
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta
import math
import hashlib
import json

from backtests.auto.shared.cache_keys import stable_signature, source_generation_hash
from backtests.engine.contracts import AUDIT_VERSION, METRIC_VERSION, EXECUTION_VERSION
from kis_core.trading_calendar import get_trading_calendar
from strategy_common.execution_session import bar_execution_issue, bar_reopens_after_label, bar_has_periodic_auction, bar_is_delayed_closing_auction
from strategy_common.market import MarketBar

REQUIRED_COVERAGE = frozenset({"ledger", "clock", "calendar", "special_session_hours",
    "terminal_marks", "input_provenance", "intrabar_entry_chronology", "capacity",
    "dated_references", "price_volume_basis", "source_completeness", "replay_completeness", "portfolio_admission", "daily_marks", "halt_boundary_chronology", "periodic_auction_execution", "strategy_state"})


def audit_broker(broker, *, identity=None):
    identity = dict(identity if identity is not None else getattr(broker, "evaluation_identity", {}))
    identity["qualification_snapshot_hash"] = getattr(broker, "qualification_snapshot_hash", None)
    identity["qualification_day_dispositions_hash"] = stable_signature(getattr(broker, "qualification_day_dispositions", {}))
    broker.evaluation_identity = identity
    errors = []
    expected = getattr(broker, "expected_batches", None)
    expected_bars = {}
    expected_marks = {}
    from .capacity_audit import audit_capacity
    capacity_report = audit_capacity(broker)
    errors.extend(capacity_report["errors"])
    capacity_report["selection_shadow_reports"] = [audit_capacity(shadow)
        for shadow in getattr(broker, "capacity_shadow_brokers", ())]
    for report in capacity_report["selection_shadow_reports"]:
        errors.extend("selection_shadow:" + error for error in report["errors"])
        if broker.capacity_policy is not None and report.get("contract") != broker.capacity_policy.contract():
            errors.append("selection_shadow:capacity_policy_mismatch")
    if expected is not None:
        digest = hashlib.sha256()
        count = 0
        for batch in expected:
            for bar in batch:
                digest.update(json.dumps(bar, sort_keys=True, default=str).encode())
                count += 1
                key = (bar["symbol"], str(bar["timestamp"]))
                if key in expected_bars:
                    errors.append("duplicate_input_event")
                expected_bars[key] = bar
                source = dict(bar)
                source["timestamp"] = datetime.fromisoformat(source["timestamp"])
                if not bar_execution_issue(MarketBar(**source)):
                    for field in ("open", "close"):
                        if field == "open" and (bar_reopens_after_label(MarketBar(**source)) or bar_has_periodic_auction(MarketBar(**source)) or bar_is_delayed_closing_auction(MarketBar(**source))):
                            continue
                        expected_marks.setdefault((key[1], field), {})[key[0]] = bar[field]
        if count != broker.source_event_count or digest.hexdigest() != broker.source_events.hexdigest():
            errors.append("source_event_disposition_mismatch")
    active = {}
    strategy_budget_admissions = {}
    quantities = defaultdict(int)
    cash = float(broker.initial_equity)
    seen = set()
    recorded_fills = []
    same_decision = 0
    demand = defaultdict(int)
    volumes = {}
    marks = {}
    batch_timestamp = None
    phase_index = -1
    filled_quantities = defaultdict(int)
    shared_demand = defaultdict(int)
    bar_demand = defaultdict(int)
    expected_equity = [cash]
    policy_admissions = {}
    halt_boundaries = 0
    periodic_auction_batches = 0
    rejection_reasons = defaultdict(int)
    invalid_rejections = 0
    capacity_allowances = {}
    deadline_positions = {}
    state_checkpoints = defaultdict(set)
    native_budget_audit = None
    if getattr(broker, 'native_budget', None) is not None:
        from backtests.strategies.portfolio_synergy.native_budget import audit_native_budgets
        native_budget_audit = audit_native_budgets(broker)
        errors.extend('native_budget_' + error for error in native_budget_audit['errors'])
    calendar = get_trading_calendar()

    def record_deadline(stamp, completed=False):
        closing = calendar.session_hours(stamp.date())[1]
        deadline = stamp.replace(hour=closing.hour, minute=closing.minute, second=0, microsecond=0) - timedelta(minutes=10)
        available = stamp + timedelta(minutes=5) if completed else stamp
        day = stamp.date().isoformat()
        if available >= deadline and day not in deadline_positions:
            deadline_positions[day] = {f"{strategy}:{symbol}": qty for (strategy, symbol), qty in quantities.items() if qty}

    for row in broker.journal:
        kind = row["event"]
        if kind == "strategy_state":
            strategy = row["strategy_id"]
            stamp = row["timestamp"]
            if stamp in state_checkpoints[strategy] or stamp != batch_timestamp:
                errors.append("strategy_state_clock_mismatch")
            state_checkpoints[strategy].add(stamp)
            inventory = {symbol: qty for (owner, symbol), qty in quantities.items() if owner == strategy and qty}
            if row["positions"] != inventory:
                errors.append("strategy_state_position_mismatch")
            entries = {key: order for key, order in active.items() if order["strategy_id"] == strategy and order["side"] == "BUY"}
            if set(row["pending_entries"]) != set(entries):
                errors.append("strategy_state_entry_order_mismatch")
            for key, item in row["pending_entries"].items():
                order = entries.get(key)
                if order is None:
                    continue
                meta = order.get("metadata", {})
                intended = filled_quantities[key] + int(order["qty"])
                reserved = float(meta.get("reserved_notional", 0))
                if not reserved:
                    reserved = intended * float(meta.get("estimated_entry_price", 0)) * (1 + float(meta.get("entry_cost_buffer_pct", 0)))
                expected_reserve = reserved * int(order["qty"]) / intended
                if (item["symbol"] != order["symbol"] or item["remaining_qty"] != order["qty"]
                        or not math.isclose(item["reserved_notional"], expected_reserve, abs_tol=1e-5)):
                    errors.append("strategy_state_entry_reservation_mismatch")
            for key, symbol in row["pending_exits"].items():
                order = active.get(key, {})
                if order.get("strategy_id") != strategy or order.get("symbol") != symbol or order.get("side") != "SELL":
                    errors.append("strategy_state_exit_order_mismatch")
            continue
        if kind in {"opening_marks", "closing_marks"}:
            field = "open" if kind == "opening_marks" else "close"
            if expected is not None and row["prices"] != expected_marks.get((row["timestamp"], field), {}):
                errors.append("source_mark_mismatch")
            marks.update(row["prices"])
            if kind == "opening_marks":
                record_deadline(datetime.fromisoformat(row["timestamp"]))
                batch_timestamp, phase_index = row["timestamp"], -1
                affected = set()
                periodic = set()
                for order in active.values():
                    raw = expected_bars.get((order["symbol"], batch_timestamp))
                    if raw is not None and bar_reopens_after_label(MarketBar(**{**raw, "timestamp": datetime.fromisoformat(raw["timestamp"])})):
                        affected.add(order["symbol"])
                    if raw is not None and bar_has_periodic_auction(MarketBar(**{**raw, "timestamp": datetime.fromisoformat(raw["timestamp"])})):
                        periodic.add(order["symbol"])
                halt_boundaries += len(affected)
                periodic_auction_batches += len(periodic)
            else:
                record_deadline(datetime.fromisoformat(row["timestamp"]), completed=True)
                expected_equity.append(cash + sum(q * marks.get(symbol, 0) for (_, symbol), q in quantities.items()))
            continue
        order = row.get("order", {})
        order_id = order.get("order_id")
        if kind == "capacity":
            requested = int(order.get("qty") or quantities[(order["strategy_id"], order["symbol"])])
            if order != active.get(order_id) or row["requested_qty"] != requested:
                errors.append("capacity_inactive_order_or_quantity")
            capacity_allowances[order_id] = row["allowed_qty"]
            continue
        if kind == "native_budget_admission":
            config = getattr(broker, "native_budget", None)
            admission = policy_admissions.get(order_id)
            if (config is None or admission is None or order != active.get(order_id)
                    or row['config'] != asdict(config) or row['timestamp'] != admission['timestamp']
                    or row['offered_qty'] != admission['admitted_qty']
                    or not math.isclose(row['unit_cost'], admission['unit_cost'], abs_tol=1e-7)
                    or not 0 <= row['approved_qty'] <= row['offered_qty']):
                errors.append('unbound_native_budget_admission')
            strategy_budget_admissions[order_id] = row['approved_qty']
            continue
        if kind == "strategy_budget_admission":
            config = getattr(broker, "allocation_config", None)
            admission = policy_admissions.get(order_id)
            if config is None or admission is None or order != active.get(order_id):
                errors.append("unbound_strategy_budget_admission")
                continue
            sid = order["strategy_id"]
            fraction = config.olr_budget_fraction if sid == "OLR" else config.kalcb_budget_fraction
            gross = sum(q * marks.get(symbol, 0) for (_, symbol), q in quantities.items())
            own = sum(q * marks.get(symbol, 0) for (s, symbol), q in quantities.items() if s == sid)
            budget = max(0, cash + gross) * min(fraction, broker.strategy_leverage[sid])
            maximum = min(admission["admitted_qty"], max(0, math.floor(max(0, budget-own) / admission["unit_cost"])))
            if (row["approved_qty"] != maximum or row["timestamp"] != admission["timestamp"]
                    or not math.isclose(row["budget"], budget, abs_tol=1e-5)
                    or not math.isclose(row["own_exposure"], own, abs_tol=1e-5)
                    or not math.isclose(row["unit_cost"], admission["unit_cost"], abs_tol=1e-7)):
                errors.append("incorrect_strategy_budget_admission")
            strategy_budget_admissions[order_id] = maximum
            continue
        if kind == "portfolio_admission":
            strategy_budget_admissions.pop(order_id, None)
            policy = getattr(broker, "portfolio_policy", None)
            if policy is None or row["policy"] != asdict(policy.config) or row["order"] != active.get(order_id):
                errors.append("unbound_portfolio_admission")
                continue
            sectors = getattr(broker, "policy_sectors", {})
            sector = sectors.get(order["symbol"], "UNKNOWN")
            exposures = {"gross": 0.0, "symbol": 0.0, "sector": 0.0}
            for (_, symbol), qty in quantities.items():
                value = qty * marks.get(symbol, 0)
                exposures["gross"] += value
                exposures["symbol"] += value if symbol == order["symbol"] else 0
                exposures["sector"] += value if sectors.get(symbol, "UNKNOWN") == sector else 0
            equity = cash + exposures["gross"]
            if row["sectors"] != sectors or row["sector"] != sector or any(not math.isclose(exposures[k], row["exposure"][k], abs_tol=1e-5) for k in exposures):
                errors.append("portfolio_exposure_mismatch")
            if not math.isclose(cash, row["cash"], abs_tol=1e-5) or not math.isclose(equity, row["equity"], abs_tol=1e-5):
                errors.append("portfolio_unavailable_cash_or_equity")
            config = row["policy"]
            prediction = row.get("quality_prediction")
            model = getattr(broker, "quality_model", None)
            decision_at = datetime.fromisoformat(str(order["submitted_at"]))
            expected_prediction = model.prediction(order["strategy_id"], order.get("metadata", {}), decision_at) if model else None
            if prediction != expected_prediction:
                errors.append("portfolio_quality_prediction_mismatch")
            if prediction is not None:
                if (model is None or prediction.get("label_hash") != model.label_hash
                        or datetime.fromisoformat(prediction["latest_label_exit"]).date() >= decision_at.date()
                        or not math.isfinite(prediction["score_bps"])):
                    errors.append("portfolio_unbound_or_noncausal_quality")
            scale = max(equity, 0) / config["notional_cap_reference_equity"] if config["notional_cap_mode"] in {"equity_scaled", "equity_capped"} else 1
            if config["notional_cap_mode"] == "equity_capped":
                scale = min(scale, 1)
            funding = cash
            if config.get("funding_mode", "cash") == "native_buying_power":
                own = sum(q * marks.get(symbol, 0) for (strategy, symbol), q in quantities.items()
                          if strategy == order["strategy_id"])
                funding = max(0, min(equity * broker.buying_power_leverage - exposures["gross"],
                    equity * broker.strategy_leverage[order["strategy_id"]] - own))
            if "available_funding" in row and not math.isclose(funding, row["available_funding"], abs_tol=1e-5):
                errors.append("portfolio_unavailable_funding")
            available = min(funding, config["max_gross_notional"] * scale - exposures["gross"],
                config["max_symbol_notional"] * scale - exposures["symbol"], config["max_sector_notional"] * scale - exposures["sector"])
            continuation = filled_quantities[order_id] > 0
            offered, unit_cost = row["offered_qty"], row["unit_cost"]
            if not math.isfinite(unit_cost) or unit_cost <= 0 or not 0 < offered <= int(order["qty"]):
                errors.append("invalid_policy_quantity_or_price")
                continue
            maximum = min(offered, max(0, math.floor((available + 1e-9) / unit_cost)))
            if not continuation and ((config.get("same_symbol_policy", "block") == "block" and exposures["symbol"] > 0) or equity <= 0):
                maximum = 0
            if row["continuation"] != continuation or row["admitted_qty"] != maximum:
                errors.append("incorrect_portfolio_admission")
            policy_admissions[order_id] = row
            continue
        if kind in {"submit", "amend"}:
            if kind == "amend" and order_id not in active:
                errors.append("amend_without_active_order")
            active[order_id] = order
            continue
        if kind == "reject":
            reason = row.get("reason", "unknown")
            rejection_reasons[reason] += 1
            justified = False
            if order_id in active and reason == "portfolio_policy":
                admission = policy_admissions.get(order_id, {})
                justified = strategy_budget_admissions.get(order_id, admission.get("admitted_qty")) == 0
            elif order_id in active and reason == "buying_power":
                qty, price = row.get("attempted_qty", 0), row.get("execution_price", 0)
                costs = getattr(broker, "strategy_costs", {}).get(order.get("strategy_id"), broker.costs)
                gross = sum(q * marks.get(symbol, 0) for (_, symbol), q in quantities.items())
                available = (cash + gross) * broker.buying_power_leverage - gross
                if hasattr(broker, "strategy_leverage"):
                    own = sum(q * marks.get(symbol, 0) for (strategy, symbol), q in quantities.items()
                              if strategy == order.get("strategy_id"))
                    available = min(available, (cash + gross) * broker.strategy_leverage[order["strategy_id"]] - own)
                source = expected_bars.get((order["symbol"], row.get("timestamp"))) if expected is not None else row.get("execution_bar")
                if expected is not None and row.get("execution_bar") != source:
                    errors.append("rejection_source_event_mismatch")
                expected_price = None
                if source:
                    base = float(source["open"])
                    typ = order["order_type"]
                    if typ == "STOP" and order.get("stop_price") is not None:
                        base = max(base, float(order["stop_price"]))
                    elif typ == "LIMIT" and order.get("limit_price") is not None:
                        base = min(base, float(order["limit_price"]))
                    elif typ == "CLOSE_AUCTION":
                        base = float(source["close"])
                    bps = costs.slippage_bps
                    if typ == "CLOSE_AUCTION":
                        bps = max(0, bps + float(order.get("metadata", {}).get("auction_adverse_bps", costs.auction_slippage_bps) or 0))
                    expected_price = base * (1 + bps / 10000)
                    if typ == "LIMIT":
                        expected_price = min(expected_price, float(order["limit_price"]))
                offered = policy_admissions.get(order_id, {}).get("admitted_qty", capacity_allowances.get(order_id, order.get("qty")))
                justified = (order.get("side") == "BUY" and qty == offered
                    and 0 < qty <= int(order.get("qty", 0))
                    and expected_price is not None and math.isclose(price, expected_price, abs_tol=1e-7)
                    and math.isfinite(price) and price > 0 and row.get("marks") == marks
                    and math.isclose(row.get("cash", float("nan")), cash, abs_tol=1e-5)
                    and qty * price * (1 + costs.commission_bps / 10000) > available + 1e-9)
            if not justified:
                invalid_rejections += 1
                errors.append("unverified_order_rejection:" + str(reason))
        if kind in {"cancel", "expire", "reject"}:
            active.pop(order_id, None)
            continue
        if kind != "fill":
            continue
        fill = row["fill"]
        recorded_fills.append(fill)
        meta = fill["metadata"]
        execution_id = meta.get("execution_id")
        if not execution_id or execution_id in seen:
            errors.append("missing_or_duplicate_execution")
        seen.add(execution_id)
        if order_id not in active or active.get(order_id) != order:
            errors.append("inactive_or_changed_order")
        if int(fill["qty"]) >= int(order.get("qty") or fill["qty"]):
            active.pop(order_id, None)
        before = filled_quantities[order_id]
        if meta.get("filled_qty_before") != before or meta.get("filled_qty_after") != before + int(fill["qty"]):
            errors.append("order_progress_mismatch")
        filled_quantities[order_id] += int(fill["qty"])
        bar_data = dict(meta.get("execution_bar") or {})
        if expected is not None and bar_data != expected_bars.get((fill["symbol"], str(bar_data.get("timestamp")))):
            errors.append("fill_source_event_mismatch")
        try:
            bar_data["timestamp"] = datetime.fromisoformat(bar_data["timestamp"])
            bar = MarketBar(**bar_data)
            if bar_execution_issue(bar) or bar_reopens_after_label(bar) or bar_has_periodic_auction(bar) or bar.symbol != fill["symbol"] or bar.timestamp != fill["timestamp"]:
                errors.append("non_executable_fill")
        except (ValueError, KeyError, TypeError):
            errors.append("missing_execution_bar")
            continue
        price, qty = float(fill["price"]), int(fill["qty"])
        if fill["side"] == "BUY" and getattr(broker, "portfolio_policy", None) is not None:
            admission = policy_admissions.pop(order_id, None)
            costs = broker.strategy_costs[fill["strategy_id"]]
            if (admission is None or qty > admission["admitted_qty"]
                or admission["timestamp"] != fill["timestamp"].isoformat()
                or not math.isclose(admission["unit_cost"], price * (1 + costs.commission_bps / 10000), abs_tol=1e-7)):
                errors.append("missing_or_mismatched_policy_admission")
            if getattr(broker, "allocation_config", None) is not None or getattr(broker, "native_budget", None) is not None:
                budget_qty = strategy_budget_admissions.pop(order_id, None)
                if budget_qty is None or qty > budget_qty:
                    errors.append("missing_or_exceeded_strategy_budget_admission")
        side, phase = fill["side"], meta.get("execution_phase")
        next_phase = {"open": 0, "intrabar": 1, "auction": 2}.get(phase, -1)
        if next_phase < phase_index or bar.timestamp.isoformat() != batch_timestamp:
            errors.append("out_of_order_execution_phase")
        phase_index = next_phase
        if row.get("marks") != marks:
            errors.append("unavailable_phase_marks")
        if not math.isfinite(price) or price <= 0 or qty <= 0 or qty > int(order.get("qty") or 0):
            errors.append("invalid_quantity_or_price")
        submitted = order["submitted_at"]
        if isinstance(submitted, str):
            submitted = datetime.fromisoformat(submitted)
        available = datetime.fromisoformat(order["metadata"].get("decision_available_at", submitted.isoformat()))
        same_open_protection = (side == "SELL" and phase == "intrabar"
            and order["metadata"].get("activation_phase") == "open_fill" and submitted == bar.timestamp)
        if available > bar.timestamp or (submitted >= bar.timestamp and not same_open_protection):
            errors.append("execution_before_activation")
            same_decision += int(side == "BUY")
        typ = order["order_type"]
        if bar_is_delayed_closing_auction(bar) and typ != "CLOSE_AUCTION":
            errors.append("continuous_fill_at_delayed_auction")
        stop, limit = order.get("stop_price"), order.get("limit_price")
        if typ == "MARKET":
            expected_base = bar.open
            if phase != "open":
                errors.append("market_phase")
        elif typ == "CLOSE_AUCTION":
            expected_base = bar.close
            target = str(order["metadata"].get("auction_fill_time", "15:30"))
            calendar = get_trading_calendar()
            scheduled = calendar.session_hours(submitted.date())[1].strftime("%H:%M")
            if order["metadata"].get("expiry_basis") == "closing_auction" and target == scheduled:
                target = calendar.closing_match_time(bar.symbol, submitted.date()).strftime("%H:%M")
            if phase != "auction" or bar.timestamp.strftime("%H:%M") != target:
                errors.append("auction_phase")
        elif typ == "STOP" and stop:
            expected_base = (float(stop) if same_open_protection else
                min(float(stop), bar.open) if side == "SELL" else max(float(stop), bar.open))
            if (side == "SELL" and bar.low > stop) or (side == "BUY" and bar.high < stop):
                errors.append("untriggered_stop")
        elif typ == "LIMIT" and limit:
            expected_base = min(float(limit), bar.open) if side == "BUY" else max(float(limit), bar.open)
            if (side == "BUY" and bar.low > limit) or (side == "SELL" and bar.high < limit):
                errors.append("unreachable_limit")
        else:
            errors.append("unsupported_order")
            continue
        base = float(meta.get("base_price", float("nan")))
        if not math.isclose(base, expected_base, abs_tol=1e-7) or not bar.low <= base <= bar.high:
            errors.append("incorrect_base_price")
        configured = getattr(broker, "strategy_costs", {}).get(fill["strategy_id"], broker.costs)
        costs = asdict(configured)
        if row.get("costs") != costs:
            errors.append("incorrect_cost_contract")
        bps = costs["slippage_bps"]
        if typ == "CLOSE_AUCTION":
            bps = max(0, bps + float(order["metadata"].get("auction_adverse_bps", costs["auction_slippage_bps"]) or 0))
        expected_price = base * (1 + (1 if side == "BUY" else -1) * bps / 10000)
        if typ == "LIMIT":
            expected_price = min(expected_price, limit) if side == "BUY" else max(expected_price, limit)
        if not math.isclose(price, expected_price, abs_tol=1e-7):
            errors.append("incorrect_slippage")
        fees = qty * price * (costs["commission_bps"] + (costs["tax_bps_on_sell"] if side == "SELL" else 0)) / 10000
        if not math.isclose(fees, float(meta.get("fee", -1)), abs_tol=1e-7):
            errors.append("incorrect_fees")
        if not math.isclose(cash, row["cash_before"], abs_tol=1e-5):
            errors.append("cash_before_mismatch")
        key = (fill["strategy_id"], fill["symbol"])
        if side == "BUY":
            notional = sum(q * marks.get(symbol, 0) for (_, symbol), q in quantities.items())
            available_cash = cash if broker.buying_power_leverage <= 1 else (cash + notional) * broker.buying_power_leverage - notional
            if hasattr(broker, "strategy_leverage"):
                strategy_notional = sum(q * marks.get(symbol, 0) for (sid, symbol), q in quantities.items()
                    if sid == fill["strategy_id"])
                available_cash = min(available_cash,
                    (cash + notional) * broker.strategy_leverage[fill["strategy_id"]] - strategy_notional)
            if qty * price + fees > available_cash + 1e-5:
                errors.append("unavailable_buying_power")
            quantities[key] += qty
            cash -= qty * price + fees
        else:
            quantities[key] -= qty
            if quantities[key] < 0:
                errors.append("oversold_position")
            cash += qty * price - fees
        if not math.isclose(cash, row["cash_after"], abs_tol=1e-5):
            errors.append("cash_after_mismatch")
        group = (bar.symbol, bar.timestamp.isoformat(), side, phase)
        demand[group] += qty
        volumes[group] = bar.volume
        shared = (bar.symbol, bar.timestamp.isoformat(), phase)
        shared_demand[shared] += qty
        bar_demand[shared[:2]] += qty
        if broker.capacity_policy is not None and bar_demand[shared[:2]] > bar.volume:
            errors.append("shared_bar_volume_exceeded")
    if stable_signature(recorded_fills) != stable_signature([asdict(fill) for fill in broker.fills]):
        errors.append("fill_ledger_mismatch")
    if {k: v for k, v in quantities.items() if v} != {k: p.qty for k, p in broker.positions.items()}:
        errors.append("terminal_quantity_mismatch")
    if not math.isclose(cash, broker.cash, abs_tol=1e-5):
        errors.append("terminal_cash_mismatch")
    if len(expected_equity) != len(broker.equity_curve) or any(
            not math.isclose(a, b, abs_tol=1e-5) for a, b in zip(expected_equity, broker.equity_curve)):
        errors.append("equity_path_mismatch")
    if marks != broker.last_prices:
        errors.append("terminal_marks_mismatch")
    if set(active) != {o.order_id for o in broker.orders}:
        errors.append("missing_order_disposition")
    if sum(rejection_reasons.values()) != len(broker.rejected_orders):
        errors.append("rejected_order_disposition_mismatch")
    if expected is not None and halt_boundaries != broker.bar_issues.get("unsupported_halt_boundary_chronology", 0):
        errors.append("halt_boundary_counter_mismatch")
    if expected is not None and periodic_auction_batches != broker.bar_issues.get("unsupported_periodic_auction_execution", 0):
        errors.append("periodic_auction_counter_mismatch")
    olr_scope = "OLR" in getattr(broker, "qualification_snapshot_scopes", {})
    state_schedule = getattr(broker, "expected_strategy_state_timestamps", {})
    if olr_scope:
        expected_states = state_schedule.get("OLR")
        if (expected_states is None or len(expected_states) != len(set(expected_states))
                or state_checkpoints["OLR"] != set(expected_states)
                or not set(expected_states).issubset({ts.isoformat() for ts in broker.timestamps})):
            errors.append("strategy_state_missing_checkpoint")
    calendar = get_trading_calendar()
    from .qualification_evidence import verify_qualification
    verified_coverage, qualification_details = verify_qualification(broker, identity, capacity_report=capacity_report)
    from backtests.analysis.metrics import session_equity
    declared_window = getattr(broker, "evaluation_window", None)
    requested_window = (identity.get("configuration") or {}).get("date_range")
    if declared_window and requested_window and declared_window != requested_window:
        errors.append("declared_metric_window_mismatch")
    _, daily_coverage = session_equity(broker.equity_curve, broker.timestamps, broker.mark_coverage or None,
        declared_window=declared_window, initial_cash_only=True)
    coverage = {
        "strategy_state": "failed" if any(e.startswith("strategy_state_") for e in errors) else "pass" if olr_scope else "not_applicable",
        "ledger": "failed" if errors else "pass" if expected is not None else "unverified",
        "clock": "pass" if broker.execution_clock == EXECUTION_VERSION else "unverified",
        "calendar": "pass" if all(calendar.has_coverage(ts.date()) for ts in broker.timestamps) else "unverified",
        "special_session_hours": "pass" if calendar.session_hours_verified and all(calendar.has_session_coverage(ts.date()) for ts in broker.timestamps) else "unverified",
        "terminal_marks": "pass" if not broker.mark_coverage or not broker.mark_coverage[-1]["stale_symbols"] else "unverified",
        "daily_marks": "pass" if daily_coverage["status"] == "complete" and broker.timestamps else "unverified",
        "intrabar_entry_chronology": "unverified" if broker.bar_issues.get("unsupported_intrabar_entry_chronology") else "not_applicable",
        "halt_boundary_chronology": ("pass" if broker.capacity_policy is not None else "unverified") if halt_boundaries else "not_applicable",
        "periodic_auction_execution": ("pass" if broker.capacity_policy is not None else "unverified") if periodic_auction_batches else "not_applicable",
        "portfolio_admission": "failed" if any("portfolio" in error or "policy" in error or "strategy_budget" in error for error in errors) else "pass" if getattr(broker, "portfolio_policy", None) is not None else "not_applicable",
        **verified_coverage,
    }
    ratios = [demand[key] / volumes[key] for key in demand]
    capacity_groups = {}
    for phase in ("open", "intrabar", "auction"):
        for side in ("BUY", "SELL"):
            values = sorted(demand[key] / volumes[key] for key in demand if key[2:] == (side, phase))
            if values:
                capacity_groups[f"{phase}_{side.lower()}"] = {
                    "count": len(values), "p50": values[(len(values) - 1) // 2],
                    "p95": values[int((len(values) - 1) * .95)], "max": values[-1],
                    "above_1pct": sum(v > .01 for v in values),
                    "above_10pct": sum(v > .1 for v in values),
                    "above_100pct": sum(v > 1 for v in values)}
    receipt = {"audit_version": AUDIT_VERSION, "metric_version": METRIC_VERSION,
        "source_generation_hash": source_generation_hash(),
        "execution_version": broker.execution_clock, "identity": dict(identity or {}),
        "source_events_hash": broker.source_events.hexdigest(), "source_event_count": broker.source_event_count,
        "ledger_hash": stable_signature(broker.journal), "coverage": coverage,
        "qualification": qualification_details, "errors": sorted(set(errors))}
    receipt["continuous_session_end_inventory"] = deadline_positions
    receipt["strategy_state_checkpoint_counts"] = {key: len(value) for key, value in state_checkpoints.items()}
    receipt["strategy_state_schedule_hash"] = stable_signature(state_schedule)
    receipt["daily_equity_coverage"] = daily_coverage
    receipt["receipt_hash"] = stable_signature(receipt)
    passed = not errors and all(value in {"pass", "not_applicable"} for value in coverage.values())
    return {"audit_pass": passed, "audit_status": "passed" if passed else "failed" if errors else "unverified",
        "execution_validity_pass": not errors, "audit_receipt": receipt,
        "audit_version": AUDIT_VERSION, "metric_version": METRIC_VERSION, "execution_version": broker.execution_clock,
        "same_bar_fill_count": float(same_decision), "rejected_order_count": float(len(broker.rejected_orders)),
        "invalid_order_rejection_count": float(invalid_rejections),
        "rejected_order_reason_counts": dict(rejection_reasons),
        "intraday_residual_session_count": float(sum(any(key.startswith("KALCB:") for key in positions)
            for positions in deadline_positions.values())),
        "intraday_deadline_observed_sessions": len(deadline_positions),
        "forced_replay_close_count": float(broker.same_day_forced_exit_count),
        "native_budget_audit": native_budget_audit,
        "capacity_model_audit": capacity_report,
        "invalid_bar_counts": dict(broker.bar_issues),
        "capacity_evidence": {"basis": "whole_bar_volume_not_opening_or_auction_liquidity",
            "scenario": broker.capacity_policy.contract() if broker.capacity_policy else None,
            "price_share_adjustment_basis": "verified_scoped_source_contract" if coverage["price_volume_basis"] == "pass" else "unverified_source_basis", "by_phase_and_side": capacity_groups,
            "max_quantity_to_volume": max(ratios, default=0), "execution_groups": len(ratios),
            "above_1pct": sum(r > .01 for r in ratios), "above_10pct": sum(r > .1 for r in ratios),
            "above_100pct": sum(r > 1 for r in ratios)}}


def receipt_matches(metrics, *, expected_request=None, expected_mutations=None):
    receipt = dict(metrics.get("audit_receipt") or {})
    digest = receipt.pop("receipt_hash", None)
    identity = receipt.get("identity") or {}
    expected_request = expected_request or metrics.get("evaluation_request")
    expected_mutations = expected_mutations or metrics.get("evaluation_mutations")
    return (bool(digest) and digest == stable_signature(receipt)
        and bool(expected_request) and identity.get("request") == expected_request
        and bool(expected_mutations) and identity.get("mutations") == expected_mutations
        and receipt.get("audit_version") == AUDIT_VERSION
        and receipt.get("metric_version") == METRIC_VERSION
        and receipt.get("execution_version") == EXECUTION_VERSION
        and receipt.get("source_generation_hash") == source_generation_hash()
        and not receipt.get("errors")
        and set(receipt.get("coverage", {})) == REQUIRED_COVERAGE
        and all(value in {"pass", "not_applicable"} for value in receipt.get("coverage", {}).values()))
