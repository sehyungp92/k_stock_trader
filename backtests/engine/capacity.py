"""Causal, declared research capacity; never a claim of observed queue liquidity."""
from dataclasses import dataclass
from datetime import timedelta
import math

from kis_core.trading_calendar import get_trading_calendar
from strategy_common.clock import ensure_kst
from strategy_common.execution_session import bar_execution_issue, bar_has_periodic_auction, bar_reopens_after_label

CAPACITY_VERSION = "causal-bar-participation-v2"
CAPACITY_SCENARIOS = (
    ("research-baseline", .01, 0.0),
    ("participation-stress", .005, 0.0),
    ("cost-stress", .01, 10.0),
    ("combined-stress", .005, 10.0),
)


@dataclass(frozen=True)
class CapacityQuote:
    budget: int
    predictor_volume: float
    predictor_timestamp: str | None
    volume_basis: str
    reason: str


def continuous_interval(bar):
    """Only a complete native continuous five-minute interval can be a predictor."""
    if bar is None or bar.timeframe != "5m" or bar_execution_issue(bar):
        return False
    stamp = ensure_kst(bar.timestamp)
    opening, closing = get_trading_calendar().session_hours(stamp.date())
    start = stamp.replace(hour=opening.hour, minute=opening.minute, second=0, microsecond=0)
    end = stamp.replace(hour=closing.hour, minute=closing.minute, second=0, microsecond=0) - timedelta(minutes=10)
    return (start <= stamp and stamp + timedelta(minutes=5) <= end
            and not bar_has_periodic_auction(bar) and not bar_reopens_after_label(bar))


def closing_auction_event(bar):
    stamp = ensure_kst(bar.timestamp)
    calendar = get_trading_calendar()
    return (not bar_execution_issue(bar) and not bar_has_periodic_auction(bar)
            and stamp.time().replace(tzinfo=None) == calendar.closing_match_time(bar.symbol, stamp.date())
            and bar.open == bar.high == bar.low == bar.close)


@dataclass(frozen=True)
class BarParticipationScenario:
    participation: float
    name: str

    def __post_init__(self):
        if not self.name or not math.isfinite(self.participation) or not 0 < self.participation <= 1:
            raise ValueError("Capacity scenario requires a name and participation in (0,1]")

    def quote(self, bar, phase, previous_bar):
        if phase == "open":
            basis = "previous_contiguous_continuous_5m"
            if not continuous_interval(bar):
                return CapacityQuote(0, 0, None, basis, "outside_continuous_interval")
            if (not continuous_interval(previous_bar) or previous_bar.symbol != bar.symbol
                    or ensure_kst(previous_bar.timestamp).date() != ensure_kst(bar.timestamp).date()
                    or previous_bar.timestamp + timedelta(minutes=5) != bar.timestamp):
                return CapacityQuote(0, 0, None, basis, "missing_prior_continuous_interval")
            source = previous_bar
        elif phase == "auction":
            basis = "single_price_closing_auction_event"
            if not closing_auction_event(bar):
                return CapacityQuote(0, 0, None, basis, "unsupported_auction_observation")
            source = bar
        else:
            return CapacityQuote(0, 0, None, "unsupported", "unsupported_execution_phase")
        return CapacityQuote(max(0, math.floor(source.volume * self.participation)),
            float(source.volume), source.timestamp.isoformat(), basis, "available")

    def budget(self, bar, phase, previous_bar):
        return self.quote(bar, phase, previous_bar).budget

    def contract(self):
        return {"kind": "declared_bar_participation_model", "version": CAPACITY_VERSION, "name": self.name,
                "participation": self.participation, "phase_liquidity_verified": False,
                "qualification_scope": "historical_research_model",
                "opening_basis": "previous_contiguous_continuous_5m",
                "auction_basis": "single_price_closing_auction_event",
                "supported_phases": ["open", "auction"], "shared_across_sides": True,
                "missing_predictor": "no_fill_existing_order_lifecycle",
                "market_remainder": "retain_until_fill_cancel_or_expiry",
                "auction_remainder": "expire_at_actual_match",
                "realized_volume": "ex_post_impossibility_check_only"}


def capacity_from_config(config):
    raw = dict(config.get("capacity_scenario") or {})
    if not raw:
        return None
    if set(raw) != {"name", "participation"}:
        raise ValueError("Unsupported capacity scenario fields")
    return BarParticipationScenario(float(raw["participation"]), str(raw["name"]))
