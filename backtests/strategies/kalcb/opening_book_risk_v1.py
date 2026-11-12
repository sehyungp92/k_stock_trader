"""Synchronized historical downside and bounded whole-book allocation.

An empirical stress proxy, not a stop guarantee or a new alpha forecast. Inputs
are 60 completed sessions ending BEFORE the previous session (one embargo).
Within each session, all symbols use actual synchronized half-hour OPEN prices
after first30 and before the auction. The archive mixes five-minute and
half-hour rows; opens have unambiguous clocks in both. Coarse bar closes must
never be mislabeled as five-minute observations. This is a coarse stress proxy;
actual portfolio drawdown is still evaluated at every native account mark.
Only entry quantities change; no synthetic exit, fill or future winner filter.
"""
from datetime import date, datetime, timedelta
import math
import sqlite3
import numpy as np


class HistoricalPaths:
    def __init__(self, databases, session_hours=None):
        self.connections = [sqlite3.connect('file:' + str(p).replace('\\', '/') + '?mode=ro', uri=True)
                            for p in databases]
        self.days = sorted({r[0] for c in self.connections for r in
                            c.execute('select distinct day from bars')})
        self.cache = {}
        if session_hours is None:
            from kis_core.trading_calendar import get_trading_calendar
            session_hours = get_trading_calendar().session_hours
        self.session_hours = session_hours

    def close(self):
        for c in self.connections:
            c.close()

    def session(self, day, symbol):
        key = day, symbol
        if key not in self.cache:
            rows = {}
            lower = day+'T'
            upper = str(date.fromisoformat(day)+timedelta(days=1))+'T'
            for c in self.connections:
                # IS indexes (symbol,ts); OOS indexes (day,symbol,ts). Keep both
                # predicates so either immutable archive gets an indexed seek.
                for ts, op, close in c.execute('select ts,o,c from bars where day=? and symbol=? and ts>=? and ts<? order by ts', (day, symbol, lower, upper)):
                    value = float(op), float(close)
                    if any(x <= 0 or not math.isfinite(x) for x in value):
                        raise ValueError('Invalid historical source price')
                    if ts in rows and rows[ts] != value:
                        raise ValueError('Conflicting historical source bars')
                    rows[ts] = value
            self.cache[key] = rows
        return self.cache[key]

    def matrix(self, day, symbols):
        prior = [d for d in self.days if d < day]
        selected = prior[:-1][-60:]
        if len(selected) != 60:
            return None, {'reason': 'fewer_than_60_embargoed_sessions', 'sessions': len(selected)}
        values = []
        omitted = []
        for past in selected:
            rows = [self.session(past, s) for s in symbols]
            opened, closed = self.session_hours(date.fromisoformat(past))
            opening = datetime.combine(date.fromisoformat(past), opened)
            closing = datetime.combine(date.fromisoformat(past), closed)
            first = opening+timedelta(minutes=30)
            expected = []
            clock = first
            while clock < closing-timedelta(minutes=10):
                expected.append(clock.strftime('%H:%M'))
                clock += timedelta(minutes=30)
            if not expected or len(expected)>12:
                return None, {'reason':'unsupported_session_length','day':past}
            by_clock = [{t[11:16]:v for t,v in r.items()} for r in rows]
            if any(first.strftime('%H:%M') not in r for r in by_clock):
                return None, {'reason':'missing_historical_open_reference','day':past,'symbols':symbols}
            common = [t for t in expected if all(t in r for r in by_clock)]
            if len(common)<2:
                return None, {'reason':'missing_historical_path','day':past,'symbols':symbols}
            missing = [t for t in expected if t not in common]
            if missing:
                omitted.append({'day':past,'unobserved_common_clocks':missing})
            base = [r[first.strftime('%H:%M')][0] for r in by_clock]
            if any(x <= 0 or not math.isfinite(x) for x in base):
                raise ValueError('Invalid historical opening source')
            path = [[r[t][0]/b-1.0 for r,b in zip(by_clock,base)] for t in common]
            # Duplicate the final observed scenario for rectangular arithmetic;
            # this cannot change a session minimum. No extra market bar or fill.
            path += [path[-1]]*(12-len(path))
            values.append(path)
        # Keep the cache bounded to the current rolling history.
        self.cache = {k: v for k, v in self.cache.items() if k[0] >= selected[0]}
        array = np.asarray(values, dtype=float)
        if not np.isfinite(array).all():
            raise ValueError('Nonfinite historical stress')
        return array, {'reason': 'supported', 'sessions': 60, 'embargo_session': prior[-1],
                       'history_from': selected[0], 'history_through': selected[-1],
                       'scenario_slots_per_session': 12, 'tail_sessions': 6,
                       'unobserved_clocks':omitted,
                       'reference':'actual_first30_next_open',
                       'limitation':'Half-hour open stress does not bound intervening price excursions or entry gaps. '
                                    'No missing prices or intrabar fills imputed; interrupted days remain included.'}


def downside(paths, weights, fixed=None, cost_fraction=0.0):
    """Mean worst six of sixty daily adverse account excursions, all in equity units."""
    weights = np.asarray(weights, dtype=float)
    fixed = np.zeros(paths.shape[:2]) if fixed is None else np.asarray(fixed, dtype=float)
    pnl = np.einsum('dts,s->dt', paths, weights) + fixed
    loss = np.maximum(0.0, -np.min(pnl, axis=1)) + cost_fraction * float(weights.sum())
    tail = max(1, math.ceil(len(loss) * .10))
    return float(np.sort(loss)[-tail:].mean())


def allocate(intents, paths, *, equity, buying_power, budget, fixed=None, cost_fraction=0.0):
    """One declared recipe. Preserve requested caps; compare to matched-risk scaling.

    Diminishing utility of retained forecast value prevents winner-take-all
    allocation. Forecasts are priors from the unchanged parent, not newly fitted
    estimates of portfolio value. Full-account native evidence decides efficacy.
    """
    from scipy.optimize import minimize
    if not intents or equity <= 0 or budget <= 0 or buying_power < 0:
        raise ValueError('Invalid allocation domain')
    q = np.array([r['qty'] for r in intents], dtype=int)
    prices = np.array([r['price'] for r in intents], dtype=float)
    scores = np.array([r['score'] for r in intents], dtype=float)
    if (q <= 0).any() or (prices <= 0).any() or not np.isfinite(prices).all() or not np.isfinite(scores).all() or (scores <= 0).any():
        raise ValueError('Unsupported opening forecast/quantity')
    weights = q * prices / equity
    room = buying_power / equity
    risk = lambda f: downside(paths, weights * f, fixed, cost_fraction)
    # Fee reserve is included in the funding constraint. Price gaps remain
    # subject to the real broker, not an invented guaranteed execution cap.
    funding = lambda f: float(np.dot(weights, f) * (1 + cost_fraction))
    parent_risk = risk(np.ones(len(q)))
    if parent_risk <= budget and funding(np.ones(len(q))) <= room:
        return dict(joint=q.tolist(), uniform=q.tolist(), changed=False,
                    parent_risk=parent_risk, joint_risk=parent_risk, uniform_risk=parent_risk,
                    optimizer='unconstrained_parent_preserved')
    if risk(np.zeros(len(q))) > budget:
        return dict(joint=[0]*len(q), uniform=[0]*len(q), changed=True,
                    parent_risk=parent_risk, joint_risk=risk(np.zeros(len(q))), uniform_risk=risk(np.zeros(len(q))),
                    optimizer='existing_book_exceeds_budget_no_new_addition')
    def scale_to(target):
        low, high = 0., 1.
        for _ in range(55):
            mid = (low+high)/2
            if risk(np.full(len(q), mid)) <= target and funding(np.full(len(q), mid)) <= room:
                low = mid
            else:
                high = mid
        return low
    start = np.full(len(q), scale_to(budget) * .999999)
    utility = scores / scores.sum()
    fit = minimize(lambda f: -float(np.dot(utility, np.log1p(f))), start,
        method='SLSQP', bounds=[(0., 1.)]*len(q),
        constraints=[{'type':'ineq', 'fun':lambda f: budget-risk(f)},
                     {'type':'ineq', 'fun':lambda f: room-funding(f)}],
        options={'maxiter':300, 'ftol':1e-10})
    if not fit.success or risk(fit.x) > budget+1e-8 or funding(fit.x) > room+1e-8:
        raise ValueError('Joint allocation failed feasibility: '+str(fit.message))
    joint = np.floor(q * np.clip(fit.x, 0, 1)).astype(int)
    # Long-only historical paths need not be monotone in each name (hedges).
    # Integer rounding is therefore explicitly checked rather than assumed safe.
    if risk(joint/q) > budget+1e-8 or funding(joint/q) > room+1e-8:
        raise ValueError('Integer allocation violates account budget')
    target = risk(joint/q)
    uniform = np.floor(q * scale_to(target)).astype(int)
    return dict(joint=joint.tolist(), uniform=uniform.tolist(), changed=bool((joint != q).any()),
        parent_risk=parent_risk, joint_risk=target, uniform_risk=risk(uniform/q),
        joint_notional=float(np.dot(joint, prices)), uniform_notional=float(np.dot(uniform, prices)),
        optimizer=str(fit.message), utility_before=float(np.log(2)),
        utility_after=float(np.dot(utility, np.log1p(joint/q))))
