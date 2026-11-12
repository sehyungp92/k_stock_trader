"""Numerical fallback for the unchanged convex opening-allocation problem.

Daily maximum loss and worst-tail sorting are represented by linear epigraph
constraints. The utility, funding, caps, fixed book and six-of-sixty tail budget
are unchanged. A linear oracle checks a numerical first-order optimality gap.
"""
import math
import numpy as np
from scipy.optimize import OptimizeResult, linprog, minimize


def solve(paths, weights, utility, fixed, cost_fraction, budget, room, start):
    days, clocks, names = paths.shape
    tail = max(1, math.ceil(days*.10))
    fixed = np.zeros((days,clocks)) if fixed is None else np.asarray(fixed)
    # Variables: retained fractions, tail threshold / budget, and each day's
    # excess loss / budget. Eliminating the daily-max variable is exact.
    size = names+1+days
    rows, limits = [], []
    for day in range(days):
        for clock in range(clocks+1):
            row = np.zeros(size)
            row[:names] = cost_fraction*weights/budget
            rhs = 0.
            if clock < clocks:
                row[:names] -= paths[day,clock]*weights/budget
                rhs = fixed[day,clock]/budget
            row[names] = row[names+1+day] = -1.
            rows.append(row); limits.append(rhs)
    risk = np.zeros(size); risk[names] = 1.; risk[names+1:] = 1./tail
    rows.append(risk); limits.append(1.)
    funding = np.zeros(size); funding[:names] = weights*(1+cost_fraction)
    rows.append(funding); limits.append(room)
    matrix, rhs = np.asarray(rows), np.asarray(limits)
    bounds = [(0.,1.)]*names+[(0.,None)]*(days+1)
    def objective(x):
        return -float(np.dot(utility,np.log1p(x[:names])))
    def gradient(x):
        g=np.zeros(size); g[:names]=-utility/(1+x[:names]); return g
    losses = np.maximum(0.,-np.min(np.einsum('dts,s->dt',paths,weights*start)+fixed,axis=1))
    losses += cost_fraction*float(np.dot(weights,start))
    threshold = np.sort(losses)[-tail]/budget
    initial = np.r_[start,threshold,np.maximum(losses/budget-threshold,0.)]
    fitted = minimize(objective,initial,jac=gradient,method='SLSQP',bounds=bounds,
        constraints=[{'type':'ineq','fun':lambda x:rhs-matrix@x,'jac':lambda x:-matrix}],
        options={'maxiter':1000,'ftol':1e-12})
    feasible=lambda x: np.isfinite(x).all() and np.max(matrix@x-rhs)<=1e-8 and np.min(x)>=-1e-10 and np.max(x[:names])<=1+1e-10
    x=fitted.x if feasible(fitted.x) else initial
    if not feasible(x):
        raise ValueError('Convex fallback has no feasible initial state')
    for iteration in range(501):
        g=gradient(x)
        oracle=linprog(g,A_ub=matrix,b_ub=rhs,bounds=bounds,method='highs',
            options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9})
        if not oracle.success:
            raise ValueError('Convex optimality oracle failed: '+oracle.message)
        gap=float(np.dot(g,x)-oracle.fun)
        if gap<=1e-8 and feasible(x):
            return OptimizeResult(x=np.clip(x[:names],0,1),success=True,
                message=f'Equivalent convex epigraph fallback; optimality_gap={max(gap,0.):.3g}; refinements={iteration}',
                optimality_gap=max(gap,0.),max_constraint_residual=float(np.max(matrix@x-rhs)))
        direction=oracle.x-x
        # Exact one-dimensional concave-utility maximization along a feasible
        # segment. This changes only numerical search, not the economic rule.
        derivative=lambda t:float(np.dot(utility,direction[:names]/(1+x[:names]+t*direction[:names])))
        lo,hi=0.,1.
        if derivative(1.)>=0:
            step=1.
        else:
            for _ in range(60):
                mid=(lo+hi)/2
                if derivative(mid)>0:lo=mid
                else:hi=mid
            step=(lo+hi)/2
        x=x+step*direction
    raise ValueError(f'Convex fallback did not meet optimality-gap tolerance: {gap}')
