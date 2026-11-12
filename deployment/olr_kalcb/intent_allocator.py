"""Pure, causal allocation of entry intents. This module never sends orders.

One account snapshot includes filled exposure and remaining BUY reservations.
Pending sales never fund entries. Values, when supplied, must already be net of
costs and uncertainty and refer to information available at this decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import Mapping


@dataclass(frozen=True)
class IntentAllocationConfig:
    olr_budget_fraction: float = 1.0
    kalcb_budget_fraction: float = 2.0
    kalcb_sizing_fraction: float = 1.0
    olr_sizing_fraction: float = 1.0
    entry_capacity_bars: float = 0.0
    exit_capacity_quantile: float = 0.0
    minimum_exit_observations: int = 5
    rank_mode: str = "native"
    # Zero means keep each native signal's original next-open timing only.
    defer_minutes: int = 0
    # A small fixed capital-occupation charge, used only with supported values.
    occupation_bps_per_session: float = 0.0
    preserve_kalcb_submission: bool = False

    def __post_init__(self):
        for name in ('olr_budget_fraction', 'kalcb_budget_fraction', 'kalcb_sizing_fraction', 'olr_sizing_fraction'):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 2:
                raise ValueError('Invalid strategy capital budget: '+name)
        if self.olr_budget_fraction > 1:
            raise ValueError('OLR cannot exceed its native unlevered budget')
        for name in ('entry_capacity_bars', 'occupation_bps_per_session'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError('Invalid allocation setting: '+name)
        if not 0 <= self.exit_capacity_quantile <= 1:
            raise ValueError('Exit quantile must be in [0,1]')
        if self.minimum_exit_observations < 2 or self.defer_minutes not in (0, 5, 10, 15):
            raise ValueError('Unsupported evidence count or defer horizon')
        if self.rank_mode not in ('native', 'supported_value'):
            raise ValueError('Unsupported entry ranking')
        if not isinstance(self.preserve_kalcb_submission, bool):
            raise ValueError('Native KALCB submission setting must be boolean')
        if self.preserve_kalcb_submission and self.rank_mode != 'native':
            raise ValueError('Native KALCB submissions cannot use external value ranking')


@dataclass(frozen=True)
class EntryIntent:
    intent_id: str
    strategy_id: str
    symbol: str
    sector: str
    desired_qty: int
    unit_cost: float
    available_at: datetime
    expires_at: datetime
    native_rank: int = 1_000_000
    capacity_qty: int | None = None
    robust_net_bps: float | None = None
    expected_holding_sessions: float = 1.0
    value_asof: datetime | None = None
    value_source: str = ''
    shared_entry_capacity_qty: int | None = None
    minimum_qty: int = 0

    def __post_init__(self):
        if self.strategy_id not in ('KALCB', 'OLR'):
            raise ValueError('Unknown strategy')
        if (not self.intent_id or self.desired_qty <= 0 or not math.isfinite(self.unit_cost)
                or self.unit_cost <= 0 or self.available_at.tzinfo is None
                or self.expires_at.tzinfo is None or self.expires_at < self.available_at):
            raise ValueError('Invalid entry intent')
        if self.capacity_qty is not None and self.capacity_qty < 0:
            raise ValueError('Invalid capacity')
        if self.shared_entry_capacity_qty is not None and self.shared_entry_capacity_qty < 0:
            raise ValueError('Invalid shared entry capacity')
        if self.minimum_qty < 0:
            raise ValueError('Invalid minimum allocation')
        if not math.isfinite(self.expected_holding_sessions) or self.expected_holding_sessions < 0:
            raise ValueError('Invalid holding horizon')
        if self.robust_net_bps is not None:
            if (not math.isfinite(self.robust_net_bps) or not self.value_source or self.value_asof is None
                    or self.value_asof.tzinfo is None or self.value_asof > self.available_at):
                raise ValueError('Unbound or noncausal value estimate')


@dataclass(frozen=True)
class AllocationAccount:
    equity: float
    gross_limit: float
    symbol_limit: float
    sector_limit: float
    funding_limit: float
    strategy_limits: Mapping[str, float]
    gross_committed: float = 0.0
    symbol_committed: Mapping[str, float] = field(default_factory=dict)
    sector_committed: Mapping[str, float] = field(default_factory=dict)
    strategy_committed: Mapping[str, float] = field(default_factory=dict)
    occupied: frozenset[tuple[str, str]] = frozenset()
    sector_slots: Mapping[tuple[str, str], int] = field(default_factory=dict)
    max_positions: Mapping[str, int] = field(default_factory=dict)
    max_per_sector: Mapping[str, int] = field(default_factory=dict)
    block_duplicates: bool = False

    def __post_init__(self):
        values=[self.equity,self.gross_limit,self.symbol_limit,self.sector_limit,
                self.funding_limit,self.gross_committed,*self.strategy_limits.values(),
                *self.symbol_committed.values(),*self.sector_committed.values(),*self.strategy_committed.values()]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('Nonfinite or negative account allocation state')


@dataclass(frozen=True)
class IntentDecision:
    intent_id: str
    approved_qty: int
    reserved_notional: float
    reason: str
    rank_basis: str
    marginal_net_bps: float | None
    binding_limit: str


def allocate_entry_intents(intents, account: AllocationAccount, at: datetime,
                          config: IntentAllocationConfig) -> list[IntentDecision]:
    """Greedy marginal allocation for linear supported values and hard caps.

    This is not a global optimizer for arbitrary non-linear risk or integer
    bundles. Unsupported forecasts force native ordering for the whole cohort;
    incomparable scores are never mixed. All eligible intents remain in output.
    """
    intents=tuple(intents)
    if at.tzinfo is None or len({x.intent_id for x in intents}) != len(intents):
        raise ValueError('Timezone-aware, unique allocation cohort required')
    eligible=[x for x in intents if x.available_at <= at <= x.expires_at]
    supported=bool(eligible) and config.rank_mode=='supported_value' and all(x.robust_net_bps is not None for x in eligible)
    values={x.intent_id:(x.robust_net_bps-config.occupation_bps_per_session*x.expected_holding_sessions)
            if supported and x.robust_net_bps is not None else None for x in intents}
    basis='supported_net_value' if supported else 'native_fallback'
    def key(x):
        # Timestamps determine eligibility, not a permanent seniority right
        # among unsubmitted intents whose valid horizons overlap now.
        return (-(values[x.intent_id] or 0) if supported else 0,
                x.native_rank, x.available_at, x.strategy_id, x.symbol, x.intent_id)
    gross=account.gross_committed; funding=account.funding_limit
    symbols=dict(account.symbol_committed);sectors=dict(account.sector_committed)
    strategies=dict(account.strategy_committed);occupied=set(account.occupied)
    sector_slots=dict(account.sector_slots);decisions=[];allocated_shares={}
    for intent in sorted(intents,key=key):
        reason='accepted';binding='native_size';qty=0
        score=values[intent.intent_id];sid=intent.strategy_id;sym=intent.symbol;sec=intent.sector
        if account.equity <= 0: reason='missing_account_equity'
        elif intent.available_at > at: reason='not_yet_available'
        elif intent.expires_at < at: reason='expired_intent'
        elif supported and score <= 0: reason='nonpositive_supported_value'
        elif (sid,sym) in occupied: reason='own_position_or_order'
        elif account.block_duplicates and symbols.get(sym,0)>0: reason='duplicate_symbol'
        elif sum(s==sid for s,_ in occupied)>=account.max_positions.get(sid,1_000_000): reason='native_position_slots'
        elif sector_slots.get((sid,sec),0)>=account.max_per_sector.get(sid,1_000_000): reason='native_sector_slots'
        else:
            limits={'native_size':intent.desired_qty,'gross':math.floor(max(0,account.gross_limit-gross)/intent.unit_cost),
                    'symbol':math.floor(max(0,account.symbol_limit-symbols.get(sym,0))/intent.unit_cost),
                    'sector':math.floor(max(0,account.sector_limit-sectors.get(sec,0))/intent.unit_cost),
                    'strategy_budget':math.floor(max(0,account.strategy_limits[sid]-strategies.get(sid,0))/intent.unit_cost),
                    'funding':math.floor(max(0,funding)/intent.unit_cost)}
            if intent.capacity_qty is not None:limits['forecast_capacity']=intent.capacity_qty
            if intent.shared_entry_capacity_qty is not None:
                limits['shared_entry_capacity']=max(0,intent.shared_entry_capacity_qty-allocated_shares.get(sym,0))
            binding,qty=min(limits.items(),key=lambda p:p[1])
            reason='insufficient_'+binding if qty<=0 else 'resized' if qty<intent.desired_qty else 'accepted'
            if 0 < qty < intent.minimum_qty:
                qty=0;reason='below_meaningful_allocation';binding='minimum_qty'
        amount=qty*intent.unit_cost
        if qty:
            gross+=amount;funding-=amount;symbols[sym]=symbols.get(sym,0)+amount
            sectors[sec]=sectors.get(sec,0)+amount;strategies[sid]=strategies.get(sid,0)+amount
            occupied.add((sid,sym));sector_slots[sid,sec]=sector_slots.get((sid,sec),0)+1
            allocated_shares[sym]=allocated_shares.get(sym,0)+qty
        decisions.append(IntentDecision(intent.intent_id,qty,amount,reason,basis,score,binding))
    return decisions
