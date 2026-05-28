from __future__ import annotations

import math
from typing import Any, Literal

from kis_core.tick_table import tick_size

from .config import KALCBConfig
from .models import EntryType, KALCBDailyCandidate


TickIntent = Literal["buy_limit", "sell_limit", "buy_stop", "sell_stop", "protective_stop"]


def round_price_for_krx(price: float, intent: TickIntent) -> float:
    price_f = max(float(price), 0.0)
    if price_f <= 0:
        return 0.0
    if intent == "buy_limit":
        return _floor_tick(price_f)
    if intent in {"sell_limit", "buy_stop", "sell_stop", "protective_stop"}:
        return _ceil_tick(price_f)
    raise ValueError(f"Unsupported tick intent: {intent}")


def momentum_stop_price(
    entry_price: float,
    or_low: float,
    entry_bar_low: float,
    daily_atr: float,
    config: KALCBConfig,
) -> float:
    structural_stop = min(float(or_low), float(entry_bar_low)) if or_low > 0 else float(entry_bar_low)
    atr_stop = float(entry_price) - config.stop_atr_multiple * max(float(daily_atr), 0.0)
    raw_stop = max(structural_stop, atr_stop)
    if raw_stop >= entry_price:
        raw_stop = float(entry_price) * 0.985
    return round_price_for_krx(raw_stop, "protective_stop")


def regime_size_mult(regime_tier: str, config: KALCBConfig) -> float:
    tier = str(regime_tier or "C").upper()
    if tier == "A":
        return config.regime_mult_a
    if tier == "B":
        return config.regime_mult_b
    return config.regime_mult_c


def conditional_entry_blocked(
    candidate: KALCBDailyCandidate,
    entry_type: EntryType,
    momentum_score: int,
    config: KALCBConfig,
    score_detail: dict[str, int] | None = None,
) -> bool:
    entry_key = _entry_type_key(entry_type)
    score = int(momentum_score)
    if _contains_key(config.entry_score_blocklist, f"{entry_key}:{score}"):
        return True
    if _contains_key(config.entry_score_blocklist, f"*:{score}"):
        return True
    return _detail_matches_any(config.entry_detail_blocklist, entry_key, score, score_detail)


def conditional_entry_size_mult(
    candidate: KALCBDailyCandidate,
    entry_type: EntryType,
    momentum_score: int,
    config: KALCBConfig,
    score_detail: dict[str, int] | None = None,
) -> float:
    del candidate
    entry_key = _entry_type_key(entry_type)
    score = int(momentum_score)
    mult = 1.0
    mult *= _lookup_mult(config.entry_score_size_mults, f"{entry_key}:{score}")
    mult *= _lookup_mult(config.entry_score_size_mults, f"*:{score}")
    mult *= _lookup_detail_mult(config.entry_detail_size_mults, entry_key, score, score_detail)
    return max(0.0, mult)


def compute_entry_qty(
    *,
    cash: float,
    open_notional: float,
    portfolio_equity: float | None = None,
    portfolio_drawdown_pct: float | None = None,
    portfolio_session_return_pct: float | None = None,
    open_positions: int = 0,
    entry_price: float,
    stop_price: float,
    config: KALCBConfig,
    candidate: KALCBDailyCandidate,
    entry_type: EntryType,
    momentum_score: int,
    score_detail: dict[str, int] | None = None,
) -> int:
    entry = max(float(entry_price), 0.0)
    if entry <= 0 or cash <= 0:
        return 0
    risk_per_share = max(entry - float(stop_price), 1.0)
    reg_mult = regime_size_mult(candidate.regime_tier, config)
    if reg_mult <= 0:
        return 0
    type_mult = config.pdh_size_mult if entry_type == EntryType.PDH_BREAKOUT else 1.0
    cohort_mult = conditional_entry_size_mult(candidate, entry_type, momentum_score, config, score_detail)
    route_risk_mult = max(float(config.entry_plan_route_risk_mult), 0.0)
    route_notional_mult = max(float(config.entry_plan_route_notional_mult), 0.0)
    route_participation_mult = max(float(config.entry_plan_route_participation_mult), 0.0)
    risk_budget = float(cash) * config.risk_per_trade_pct * reg_mult * type_mult * cohort_mult * route_risk_mult
    risk_qty = int(risk_budget / risk_per_share)
    effective_notional_pct = effective_max_position_notional_pct(
        config=config,
        open_notional=open_notional,
        portfolio_equity=portfolio_equity,
        portfolio_drawdown_pct=portfolio_drawdown_pct,
        portfolio_session_return_pct=portfolio_session_return_pct,
        open_positions=open_positions,
    )
    max_notional = max(float(cash) * effective_notional_pct * route_notional_mult, 0.0)
    buying_power = max(float(cash) * config.intraday_leverage - float(open_notional), 0.0)
    notional_qty = int(min(max_notional, buying_power) / entry)
    participation_qty = int(max(candidate.average_30m_volume, 1.0) * config.max_participation_30m * route_participation_mult)
    if participation_qty <= 0:
        return 0
    qty = min(risk_qty, notional_qty, participation_qty)
    return max(0, qty)


def effective_max_position_notional_pct(
    *,
    config: KALCBConfig,
    open_notional: float = 0.0,
    portfolio_equity: float | None = None,
    portfolio_drawdown_pct: float | None = None,
    portfolio_session_return_pct: float | None = None,
    open_positions: int = 0,
) -> float:
    base = max(float(config.max_position_notional_pct), 0.0)
    if not config.risk_dynamic_notional_enabled:
        return base
    dynamic = max(float(config.risk_dynamic_max_position_notional_pct), 0.0)
    if dynamic <= base:
        return base

    drawdown = float(portfolio_drawdown_pct or 0.0)
    if config.risk_dynamic_max_drawdown_pct > 0 and drawdown > float(config.risk_dynamic_max_drawdown_pct):
        return base

    session_return = float(portfolio_session_return_pct or 0.0)
    if config.risk_dynamic_min_session_return_pct > -9.0 and session_return < float(config.risk_dynamic_min_session_return_pct):
        return base

    if config.risk_dynamic_max_open_positions > 0 and int(open_positions) >= int(config.risk_dynamic_max_open_positions):
        return base

    equity = float(portfolio_equity or 0.0)
    if config.risk_dynamic_max_open_notional_pct > 0 and equity > 0:
        if float(open_notional or 0.0) / equity >= float(config.risk_dynamic_max_open_notional_pct):
            return base

    return dynamic


def _floor_tick(price: float) -> float:
    current = float(price)
    for _ in range(4):
        ts = tick_size(current)
        rounded = math.floor(current / ts) * ts
        if tick_size(rounded or current) == ts:
            return float(rounded)
        current = rounded
    return float(current)


def _ceil_tick(price: float) -> float:
    current = float(price)
    for _ in range(4):
        ts = tick_size(current)
        rounded = math.ceil(current / ts) * ts
        if tick_size(rounded) == ts:
            return float(rounded)
        current = rounded
    return float(current)


def _entry_type_key(entry_type: EntryType | str) -> str:
    raw = getattr(entry_type, "value", entry_type)
    text = str(raw).upper().strip()
    if text.startswith("KRX_"):
        return text
    return f"KRX_{text}"


def _control_key(value: object) -> str:
    text = str(value).strip()
    if ":" not in text:
        return text.upper()
    left, right = text.split(":", 1)
    left_text = left.strip().upper()
    if left_text != "*" and not left_text.startswith("KRX_"):
        left_text = f"KRX_{left_text}"
    return f"{left_text}:{right.strip().upper()}"


def _contains_key(values: object, key: str) -> bool:
    wanted = _control_key(key)
    if isinstance(values, str):
        iterable = [part.strip() for part in values.split(",") if part.strip()]
    else:
        iterable = list(values or [])
    return any(_control_key(value) == wanted for value in iterable)


def _lookup_mult(mapping: object, key: str) -> float:
    if not isinstance(mapping, dict):
        return 1.0
    wanted = _control_key(key)
    for raw_key, raw_value in mapping.items():
        if _control_key(raw_key) != wanted:
            continue
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return 1.0
    return 1.0


def _detail_matches_any(values: object, entry_type_key: str, score: int, score_detail: dict[str, int] | None) -> bool:
    if isinstance(values, str):
        iterable = [part.strip() for part in values.split(",") if part.strip()]
    else:
        iterable = list(values or [])
    return any(_detail_control_matches(value, entry_type_key, score, score_detail) for value in iterable)


def _lookup_detail_mult(mapping: object, entry_type_key: str, score: int, score_detail: dict[str, int] | None) -> float:
    if not isinstance(mapping, dict):
        return 1.0
    mult = 1.0
    for raw_key, raw_value in mapping.items():
        if not _detail_control_matches(raw_key, entry_type_key, score, score_detail):
            continue
        try:
            mult *= float(raw_value)
        except (TypeError, ValueError):
            continue
    return mult


def _detail_control_matches(raw_key: object, entry_type_key: str, score: int, score_detail: dict[str, int] | None) -> bool:
    parts = [part.strip() for part in str(raw_key).split(":") if part.strip()]
    if len(parts) == 2:
        type_part, detail_part = parts
        score_part = "*"
    elif len(parts) == 3:
        type_part, score_part, detail_part = parts
    else:
        return False
    type_key = _control_key(f"{type_part}:0").split(":", 1)[0]
    if type_key != "*" and type_key != entry_type_key:
        return False
    if score_part != "*":
        try:
            if int(score_part) != int(score):
                return False
        except (TypeError, ValueError):
            return False
    return _score_detail_present(score_detail, detail_part)


def _score_detail_present(score_detail: dict[str, int] | None, detail_part: str) -> bool:
    want_absent = str(detail_part).startswith("!")
    detail_name = str(detail_part)[1:] if want_absent else str(detail_part)
    normalized = detail_name.strip().lower()
    values = score_detail or {}
    present = any(str(key).strip().lower() == normalized and bool(value) for key, value in values.items())
    return not present if want_absent else present
