"""Strict account semantics for research holding observations.

No live hook. Version 1 remains frozen. Buying power retains the native broker's
pre-reservation definition; pending buys are reported separately, not silently
subtracted to change the fixed-feature experiment.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
import math


@dataclass(frozen=True)
class AccountObservation:
    available_at: str
    session_date: str
    equity_krw: float
    session_start_equity_krw: float
    running_peak_equity_krw: float
    marked_inventory_krw: float
    native_buying_power_krw: float

    def __post_init__(self):
        at = datetime.fromisoformat(self.available_at)
        if at.tzinfo is None or str(at.date()) != self.session_date:
            raise ValueError('Missing timezone or stale account session baseline')
        values = (self.equity_krw, self.session_start_equity_krw,
                  self.running_peak_equity_krw, self.marked_inventory_krw,
                  self.native_buying_power_krw)
        if not all(math.isfinite(x) for x in values):
            raise ValueError('Nonfinite account state')
        if min(values[:3]) <= 0 or min(values[3:]) < 0:
            raise ValueError('Invalid equity, inventory or buying power')
        if self.equity_krw > self.running_peak_equity_krw + .01:
            raise ValueError('Running peak precedes current native mark')

    @property
    def session_return(self):
        """Fraction, not percentage points: 0.01 means +1%."""
        return self.equity_krw / self.session_start_equity_krw - 1.0

    def features(self):
        return dict(session_return=self.session_return,
                    book_weight=self.marked_inventory_krw / self.equity_krw,
                    buying_power_fraction=self.native_buying_power_krw / self.equity_krw,
                    account_drawdown=1 - self.equity_krw / self.running_peak_equity_krw)


def from_native(adapter, broker, available_at):
    meta = adapter.state.meta
    # A missing/reset baseline must never become a plausible zero return.
    start = float(meta['portfolio_session_start_equity'])
    day = meta['portfolio_session_date']
    if str(adapter.state.session_date) != day:
        raise ValueError('Adapter session and account baseline disagree')
    book = sum(p.qty * broker.last_prices[s] for (_, s), p in broker.positions.items())
    return AccountObservation(available_at.isoformat(), day, broker._portfolio_equity(),
                              start, max(broker.equity_curve), book,
                              broker._buying_power_available())


def observations(adapter, broker, batch):
    """Use the established real observation path; change only session_return."""
    from .holding_action_v1 import observations as original
    rows = original(adapter, broker, batch)
    eligible = [r for r in rows if r['reason'] == 'eligible']
    if not eligible:
        return rows
    account = from_native(adapter, broker, batch[0].timestamp + timedelta(minutes=5))
    for row in eligible:
        for name, value in account.features().items():
            if name != 'session_return' and not math.isclose(
                    row['features'][name], value, abs_tol=1e-12, rel_tol=1e-12):
                raise ValueError('Additional account feature mismatch: ' + name)
        row['features']['session_return'] = account.session_return
    return rows


def from_saved_prefix(prefix, row):
    """Read complete saved native state, with identity checked by the caller."""
    state = prefix['state']; adapter = state['adapter']['state']; broker = state['broker']
    meta = dict(adapter['meta']); marks = dict(broker['last_prices'])
    if adapter['session_date'] != meta['portfolio_session_date']:
        raise ValueError('Saved adapter session and baseline disagree')
    book = sum(p['qty'] * marks[key[1]] for key, p in broker['positions'])
    equity = broker['cash'] + book
    leverage = broker['buying_power_leverage']
    buying_power = max(broker['cash'] if leverage <= 1 else equity * leverage - book, 0.)
    if not math.isclose(equity, row['equity'], abs_tol=.01):
        raise ValueError('Saved prefix equity differs from native observation')
    return AccountObservation(row['available_at'], meta['portfolio_session_date'], equity,
                              float(meta['portfolio_session_start_equity']),
                              row['running_peak'], book, buying_power)
