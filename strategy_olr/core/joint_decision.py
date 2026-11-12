"""Research-only exact allocation over sourced, causal action response paths.

This module never estimates alpha, calls a broker, or enables a live policy.
Each action supplies a *net* quantity-dependent path on the same historical
scenario/time axes. Summing independent per-name drawdowns is forbidden.
Native execution must separately validate any selected funded policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import product
from math import isfinite, prod
from typing import Mapping

import numpy as np


def _finite(value, name, *, nonnegative=False):
    if not isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f'Invalid {name}')


@dataclass(frozen=True)
class ScenarioAxis:
    ids: tuple[str, ...]
    weights: tuple[float, ...]
    offsets: tuple[float, ...]
    known_at: datetime
    source_sha256: str

    def __post_init__(self):
        if not self.ids or len(set(self.ids)) != len(self.ids) or len(self.ids) != len(self.weights):
            raise ValueError('Unique joint scenarios and matching probabilities required')
        if any(not isfinite(w) or w <= 0 for w in self.weights) or abs(sum(self.weights)-1) > 1e-10:
            raise ValueError('Positive scenario probabilities must sum to one')
        if not self.offsets or self.offsets[0] != 0 or tuple(sorted(set(self.offsets))) != self.offsets:
            raise ValueError('A shared ordered time axis beginning at zero is required')
        if any(not isfinite(x) or x < 0 for x in self.offsets):
            raise ValueError('Invalid scenario time axis')
        if self.known_at.tzinfo is None or len(self.source_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.source_sha256):
            raise ValueError('Bound, timezone-aware scenario evidence required')


@dataclass(frozen=True)
class Account:
    equity: float
    high_water: float
    cash: float
    pending_buy_reservations: float
    # Remaining hard envelopes after current holdings AND pending orders.
    remaining_limits: tuple[tuple[str, float], ...] = ()

    def __post_init__(self):
        for name in ('equity', 'high_water', 'cash', 'pending_buy_reservations'):
            _finite(getattr(self, name), name, nonnegative=True)
        if self.equity <= 0 or self.high_water < self.equity:
            raise ValueError('Positive actual equity and its actual high-water mark required')
        names = [name for name, _ in self.remaining_limits]
        if len(set(names)) != len(names):
            raise ValueError('Duplicate hard envelope')
        for name, value in self.remaining_limits:
            _finite(value, name, nonnegative=True)

    @property
    def available_cash(self):
        # A pending SELL is deliberately absent: only acknowledged cash funds orders.
        return max(0., self.cash-self.pending_buy_reservations)


@dataclass(frozen=True)
class Action:
    action_id: str
    group: str
    quantity: int
    reserve_cash: float
    net_paths: tuple[tuple[float, ...], ...]
    scenario_ids: tuple[str, ...]
    available_at: datetime
    source_sha256: str
    resources: tuple[tuple[str, float], ...] = ()
    # Probability of a completed, economically meaningful new lifecycle in
    # each scenario. A submitted request is not an executed trade.
    completed_entry_probabilities: tuple[float, ...] = ()
    # Diagnostic labels need explicit missingness. Unknown is never zero alpha.
    supported: bool = True
    kind: str = 'buy'
    management_completion_deltas: tuple[float, ...] = ()

    def __post_init__(self):
        if not self.action_id or not self.group or type(self.quantity) is not int or self.quantity < 0:
            raise ValueError('Unique action identity and integer nonnegative quantity required')
        if self.kind not in ('buy','hold','cancel','trim','exit'):
            raise ValueError('Unknown economic action')
        if self.kind in ('hold','cancel') and self.quantity or self.kind in ('trim','exit') and not self.quantity:
            raise ValueError('Invalid management quantity')
        _finite(self.reserve_cash, 'cash reservation', nonnegative=True)
        if len(self.completed_entry_probabilities)!=len(self.scenario_ids) or any(
                not isfinite(p) or not 0<=p<=1 for p in self.completed_entry_probabilities):
            raise ValueError('Supported completed-entry probabilities required on every scenario')
        if self.available_at.tzinfo is None or len(self.source_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.source_sha256):
            raise ValueError('Action values require sourced decision-time evidence')
        if len({name for name, _ in self.resources}) != len(self.resources):
            raise ValueError('Duplicate action resource')
        for name, value in self.resources:
            _finite(value, name, nonnegative=True)
        if self.quantity == 0 and (self.reserve_cash or any(self.completed_entry_probabilities) or any(v for _, v in self.resources)):
            raise ValueError('No-order action cannot reserve capital or create an entry')
        if self.kind=='buy' and self.quantity > 0 and self.reserve_cash <= 0:
            raise ValueError('Acquisition must reserve real cash including costs')
        if self.kind!='buy' and (self.reserve_cash or any(self.completed_entry_probabilities)):
            raise ValueError('Management cannot release acknowledged cash or count as a new completed entry')
        if self.kind in ('cancel','trim','exit'):
            if len(self.management_completion_deltas)!=len(self.scenario_ids) or any(
                    not isfinite(v) or not -1<=v<=1 for v in self.management_completion_deltas):
                raise ValueError('Management requires explicit completion changes against the held baseline')
            if self.kind=='cancel' and any(v>0 for v in self.management_completion_deltas):
                raise ValueError('Cancellation cannot create a completed lifecycle')
        elif self.management_completion_deltas:
            raise ValueError('Only management actions change baseline completions')

    @property
    def baseline(self):
        return self.quantity==0 and self.kind in ('buy','hold')


@dataclass(frozen=True)
class Constraints:
    tail_mass: float = .1
    maximum_tail_drawdown_fraction: float = 1.
    minimum_mean_net_pnl: float = float('-inf')
    minimum_expected_completed_entries: float = 0.
    maximum_combinations: int = 100_000
    # A breached absolute ceiling cannot be repaired by a future action.
    # Explicit restoration chooses the least additional tail deterioration.
    restore_feasibility: bool = False
    recovery_tail_budget_fraction: float = 0.

    def __post_init__(self):
        if not 0 < self.tail_mass <= 1 or not 0 <= self.maximum_tail_drawdown_fraction <= 1:
            raise ValueError('Invalid risk constraint')
        if self.minimum_mean_net_pnl != float('-inf'):
            _finite(self.minimum_mean_net_pnl, 'minimum net value')
        if not isfinite(self.minimum_expected_completed_entries) or self.minimum_expected_completed_entries < 0:
            raise ValueError('Invalid entry constraint')
        if type(self.maximum_combinations) is not int or self.maximum_combinations <= 0:
            raise ValueError('Finite enumeration budget required')
        if not isfinite(self.recovery_tail_budget_fraction) or not 0<=self.recovery_tail_budget_fraction<=1:
            raise ValueError('Invalid prospective recovery risk budget')


@dataclass(frozen=True)
class Evaluation:
    action_ids: tuple[str, ...]
    mean_net_pnl: float
    tail_drawdown_fraction: float
    worst_drawdown_fraction: float
    reserve_cash: float
    expected_completed_entries: float
    resources: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class Decision:
    selected: Evaluation | None
    frontier: tuple[Evaluation, ...]
    enumerated: int
    funded_combinations: int
    admissible_combinations: int
    reason: str
    scenario_source: str


def upper_tail_mean(values, probabilities, mass):
    """Weighted upper-tail expected shortfall, including a fractional boundary."""
    if not 0 < mass <= 1:
        raise ValueError('Invalid tail mass')
    if len(values) != len(probabilities) or not len(values):
        raise ValueError('Aligned nonempty tail inputs required')
    if any(not isfinite(float(x)) for x in values) or any(not isfinite(float(w)) or w <= 0 for w in probabilities):
        raise ValueError('Invalid tail input')
    if abs(sum(probabilities)-1) > 1e-10:
        raise ValueError('Tail probabilities must sum to one')
    left, total = mass, 0.
    for value, probability in sorted(zip(values, probabilities), reverse=True):
        used = min(left, probability)
        total += used*value
        left -= used
        if left <= 1e-12:
            break
    return total/mass


def _paths(value, axis):
    x = np.asarray(value, dtype=float)
    if x.shape != (len(axis.ids), len(axis.offsets)) or not np.isfinite(x).all():
        raise ValueError('Every complete net path must use the identical scenario/time axes')
    return x


def _dominates(a, b):
    no_worse = (a.mean_net_pnl >= b.mean_net_pnl-1e-8
                and a.tail_drawdown_fraction <= b.tail_drawdown_fraction+1e-12
                and a.expected_completed_entries >= b.expected_completed_entries)
    better = (a.mean_net_pnl > b.mean_net_pnl+1e-8
              or a.tail_drawdown_fraction < b.tail_drawdown_fraction-1e-12
              or a.expected_completed_entries > b.expected_completed_entries)
    return no_worse and better


def decide(account: Account, axis: ScenarioAxis, held_net_paths,
           groups: Mapping[str, tuple[Action, ...]], at: datetime,
           constraints: Constraints = Constraints(), *, held_completed_entries=None) -> Decision:
    """Exact discrete frontier; no fitted forecast or hidden risk preference.

Held paths include all outstanding commitments and costs, measured relative
to actual decision NAV. A management path is an incremental change against
that identical held/pending baseline, including execution costs and delays.
Sale/cancel submissions never release cash or hard envelopes in this decision;
only actual acknowledgements may change the next account snapshot. Every
group must supply the zero-change alternative. The native bridge currently
routes BUY choices only; management requires a separately qualified dispatcher.
"""
    if at.tzinfo is None or axis.known_at > at:
        raise ValueError('Future scenario information')
    held = _paths(held_net_paths, axis)
    if held_completed_entries is None:
        if any(a.kind in ('cancel','trim','exit') for options in groups.values() for a in options):
            raise ValueError('Management comparisons require baseline completion evidence')
        completions=np.zeros(len(axis.ids))
    else:
        completions=np.asarray(held_completed_entries,dtype=float)
        if completions.shape!=(len(axis.ids),) or not np.isfinite(completions).all() or np.any(completions<0):
            raise ValueError('Invalid held/pending completion evidence')
    if not np.allclose(held[:, 0], 0, rtol=0, atol=1e-8):
        raise ValueError('Held paths must begin at the actual decision NAV')
    names = sorted(groups)
    if not names:
        raise ValueError('At least one decision group required')
    arrays, ids = {}, set()
    for group in names:
        options = groups[group]
        if not options or not any(a.baseline for a in options):
            raise ValueError('Each group needs a real no-order alternative')
        for a in options:
            if a.group != group or a.action_id in ids or a.scenario_ids != axis.ids:
                raise ValueError('Action/scenario identity mismatch')
            ids.add(a.action_id)
            if not a.supported or a.available_at > at:
                raise ValueError('Unsupported or future action forecast; use an explicit native fallback outside the solver')
            x = _paths(a.net_paths, axis)
            if not np.allclose(x[:, 0], 0, rtol=0, atol=1e-8):
                raise ValueError('New action paths must begin at the actual decision NAV')
            if a.baseline and np.any(x != 0):
                raise ValueError('No-order net cash outcome must be zero')
            arrays[a.action_id] = x
    count = prod(len(groups[n]) for n in names)
    if count > constraints.maximum_combinations:
        raise ValueError('Enumeration budget exceeded; do not silently approximate')
    limits = dict(account.remaining_limits)
    frontier, candidates, feasible_resources = [], [], []
    current_drawdown=1-account.equity/account.high_water
    recovery=(constraints.restore_feasibility and current_drawdown>constraints.maximum_tail_drawdown_fraction)
    effective_limit=(current_drawdown+constraints.recovery_tail_budget_fraction*account.equity/account.high_water
        if recovery and constraints.recovery_tail_budget_fraction else constraints.maximum_tail_drawdown_fraction)
    funded = 0
    for choices in product(*(groups[n] for n in names)):
        cash = sum(a.reserve_cash for a in choices)
        resources = {}
        for a in choices:
            for name, value in a.resources:
                if name not in limits:
                    raise ValueError('Undeclared shared resource: '+name)
                resources[name] = resources.get(name, 0.)+value
        if cash > account.available_cash+1e-8 or any(v > limits[k]+1e-8 for k, v in resources.items()):
            continue
        funded += 1
        paths = held.copy()
        completed=completions.copy()
        for a in choices:
            paths += arrays[a.action_id]
            completed+=a.completed_entry_probabilities
            if a.management_completion_deltas:completed+=a.management_completion_deltas
        if np.any(completed < -1e-10):
            raise ValueError('Management removes more completions than the sourced baseline')
        nav = account.equity+paths
        if np.any(nav <= 0):
            continue
        high = np.maximum.accumulate(np.maximum(nav, account.high_water), axis=1)
        dd = np.max(1-nav/high, axis=1)
        row = Evaluation(tuple(a.action_id for a in choices), float(np.dot(axis.weights, paths[:, -1])),
            upper_tail_mean(dd, axis.weights, constraints.tail_mass), float(dd.max()), cash,
            float(np.dot(axis.weights,completed)), tuple(sorted(resources.items())))
        feasible_resources.append(row)
        if not any(_dominates(old, row) for old in frontier):
            frontier = [old for old in frontier if not _dominates(row, old)]
            # Keep one deterministic representative of equal economic points.
            equal = next((old for old in frontier if abs(old.mean_net_pnl-row.mean_net_pnl)<1e-8
                          and abs(old.tail_drawdown_fraction-row.tail_drawdown_fraction)<1e-12
                          and old.expected_completed_entries==row.expected_completed_entries), None)
            if equal is None:
                frontier.append(row)
            elif (row.reserve_cash,row.action_ids) < (equal.reserve_cash,equal.action_ids):
                frontier.remove(equal);frontier.append(row)
        if (row.tail_drawdown_fraction <= effective_limit+1e-12
                and row.mean_net_pnl >= constraints.minimum_mean_net_pnl
                and row.expected_completed_entries >= constraints.minimum_expected_completed_entries):
            candidates.append(row)
    selected = min(candidates, key=lambda r:(-r.mean_net_pnl,r.tail_drawdown_fraction,
                   -r.expected_completed_entries,r.reserve_cash,r.action_ids), default=None)
    reason = 'selected' if selected else 'no_feasible_supported_allocation'
    if selected is not None and recovery:reason='bounded_risk_recovery'
    if selected is None and constraints.restore_feasibility and feasible_resources:
        selected = min(feasible_resources, key=lambda r:(r.tail_drawdown_fraction,
            r.worst_drawdown_fraction,-r.mean_net_pnl,-r.expected_completed_entries,r.reserve_cash,r.action_ids))
        reason = 'risk_feasibility_restoration'
    return Decision(selected, tuple(sorted(frontier,key=lambda r:(r.tail_drawdown_fraction,-r.mean_net_pnl,r.action_ids))),
        count, funded, len(candidates), reason, axis.source_sha256)
