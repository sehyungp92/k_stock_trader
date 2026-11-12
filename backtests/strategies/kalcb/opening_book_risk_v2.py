"""Same allocation rule, with an equivalent convex fallback after solver failure only."""
import numpy as np
from .opening_book_risk_v1 import HistoricalPaths, downside


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
        from .convex_allocation_solver_v1 import solve
        fit = solve(paths, weights, utility, fixed, cost_fraction, budget, room, start)
        if risk(fit.x) > budget+1e-8 or funding(fit.x) > room+1e-8:
            raise ValueError('Certified continuous allocation violates original constraints')
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
