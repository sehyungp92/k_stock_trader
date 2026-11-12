"""Independent numerical and recovery regressions; no economic gate changes."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from backtests.strategies.kalcb.convex_allocation_solver_v1 import solve
from backtests.strategies.kalcb.opening_book_risk_v1 import allocate as original, downside
from backtests.strategies.kalcb.opening_book_risk_v2 import allocate as repaired


def test_symmetric_risk_has_known_concave_optimum():
    paths = np.full((60, 3, 2), -.1)
    fit = solve(paths, np.ones(2), np.array([.5, .5]), None, 0., .1, 3., np.array([.4, .4]))
    assert fit.x == pytest.approx([.5, .5], abs=2e-7)
    assert fit.optimality_gap <= 1e-8
    assert downside(paths, fit.x) <= .1 + 1e-10


def test_funding_and_roundtrip_cost_have_known_optimum():
    paths = np.zeros((60, 2, 2))
    fit = solve(paths, np.ones(2), np.array([.5, .5]), None, .01, 1., .808, np.array([.3, .3]))
    assert fit.x == pytest.approx([.4, .4], abs=2e-7)
    assert np.sum(fit.x)*1.01 <= .808+1e-10


def test_fixed_holdings_consume_the_same_budget():
    paths = np.full((60, 2, 1), -.1)
    fixed = np.full((60, 2), -.025)
    fit = solve(paths, np.ones(1), np.ones(1), fixed, .01, .08, 2., np.array([.1]))
    assert fit.x[0] == pytest.approx(.5, abs=2e-7)
    assert downside(paths, fit.x, fixed, .01) <= .08+1e-10


@pytest.mark.parametrize('seed', [17, 29, 41])
def test_synchronized_maximum_and_tail_dominate_independent_dense_grid(seed):
    rng = np.random.default_rng(seed)
    paths = rng.normal(-.01, .06, (60, 4, 2))
    fixed = rng.normal(0., .002, (60, 4))
    weights = np.array([.8, 1.2]); utility = np.array([.7, .3])
    fit = solve(paths, weights, utility, fixed, .003, .055, 1.1, np.zeros(2))
    assert downside(paths, weights*fit.x, fixed, .003) <= .055+1e-9
    assert np.dot(weights,fit.x)*1.003 <= 1.1+1e-9
    grid = np.array([(a,b) for a in np.linspace(0,1,81) for b in np.linspace(0,1,81)])
    feasible = [f for f in grid if downside(paths,weights*f,fixed,.003)<=.055 and np.dot(weights,f)*1.003<=1.1]
    assert np.dot(utility,np.log1p(fit.x)) >= max(np.dot(utility,np.log1p(f)) for f in feasible)-1e-8


@pytest.mark.parametrize('budget,room,cost',[(.5,3.,0.),(.06,2.,0.),(.5,1.,.01)])
def test_successful_original_paths_are_exactly_unchanged(budget,room,cost):
    intents = [dict(symbol='a',qty=1000,price=100.,score=2.),dict(symbol='b',qty=1000,price=100.,score=1.)]
    paths = np.empty((60,12,2));paths[:,:,0]=-.02;paths[:,:,1]=-.1
    args = dict(equity=100000,buying_power=room*100000,budget=budget,cost_fraction=cost)
    assert repaired(intents,paths,**args) == original(intents,paths,**args)


def test_infeasible_fixed_book_fails_instead_of_relaxing_budget():
    with pytest.raises(ValueError, match='no feasible initial state'):
        solve(np.zeros((60,2,1)),np.ones(1),np.ones(1),np.full((60,2),-.2),0.,.1,2.,np.zeros(1))


def test_missing_optimality_certificate_fails_closed(monkeypatch):
    from scipy.optimize import OptimizeResult
    from backtests.strategies.kalcb import convex_allocation_solver_v1 as solver
    monkeypatch.setattr(solver,'linprog',lambda *a,**k:OptimizeResult(success=False,message='test oracle failure'))
    with pytest.raises(ValueError,match='optimality oracle failed'):
        solve(np.full((60,2,1),-.1),np.ones(1),np.ones(1),None,0.,.05,2.,np.zeros(1))


def test_exact_captured_native_failure_is_repaired_without_relaxed_limits():
    location = os.environ.get('KALCB_SOLVER_FAILURE_ROOT')
    if not location:
        pytest.skip('Native research fixture is external and explicitly hash-bound in the study')
    root = Path(location)
    record = json.loads((root/'failed_solver_inputs_v1.json').read_text())
    arrays = np.load(root/'failed_solver_inputs_v1.npz')
    with pytest.raises(ValueError,match='Iteration limit reached'):
        original(record['intents'],arrays['paths'],fixed=arrays['fixed'],**record['kwargs'])
    result = repaired(record['intents'],arrays['paths'],fixed=arrays['fixed'],**record['kwargs'])
    assert result['joint'] == [51,261] and result['uniform'] == [67,207]
    assert result['joint_risk'] <= record['kwargs']['budget']
    assert result['uniform_risk'] <= result['joint_risk']
    assert result['joint_notional']*(1+record['kwargs']['cost_fraction']) <= record['kwargs']['buying_power']
    assert 'Equivalent convex epigraph fallback' in result['optimizer']


def test_only_original_failure_branch_differs_in_allocation_source():
    import ast
    import inspect
    old = ast.parse(inspect.getsource(original)).body[0]
    new = ast.parse(inspect.getsource(repaired)).body[0]
    assert len(old.body)==len(new.body)
    different = [(a,b) for a,b in zip(old.body,new.body) if ast.dump(a)!=ast.dump(b)]
    assert len(different)==1
    a,b = different[0]
    assert isinstance(a,ast.If) and ast.dump(a.test)==ast.dump(b.test)
    assert len(a.body)==1 and isinstance(a.body[0],ast.Raise)


def test_replay_and_worker_change_only_the_versioned_import():
    root=Path(__file__).resolve().parents[1]
    for old,new,before,after in (
        ('backtests/strategies/kalcb/opening_account_replay_v1.py','backtests/strategies/kalcb/opening_account_replay_v2.py',
         'from .opening_book_risk_v1 import HistoricalPaths, allocate','from .opening_book_risk_v2 import HistoricalPaths, allocate'),
        ('scripts/kalcb_opening_account_worker_v1.py','scripts/kalcb_opening_account_worker_v2.py',
         'opening_account_replay_v1 import OpeningAccountReplay','opening_account_replay_v2 import OpeningAccountReplay')):
        assert (root/new).read_text()==(root/old).read_text().replace(before,after)


@pytest.mark.parametrize('tamper',[False,True])
def test_mixed_case_profiles_remain_individually_bound_and_validated(tmp_path,monkeypatch,tamper):
    from backtests.strategies.kalcb import opening_account_plugin_v3 as plugin
    from backtests.strategies.kalcb.retention_research_io_v1 import write,bound
    instance=object.__new__(plugin.OpeningAccountPlugin)
    instance.root=tmp_path
    instance.records={'uniform':{'cases':{},'input_profiles':{}}}
    record=instance.records['uniform'];observed=[]
    for period in ('is','oos'):
        for scenario in plugin.SCENARIOS:
            out=tmp_path/'cases/uniform'/period/scenario
            write(out/'result.json',{})
            # Distinct source identities; never substitute a single period profile.
            source=tmp_path/f'{period}_{scenario}.json';write(source,{'case':f'{period}/{scenario}'})
            item={k:bound(source) for k in ('manifest','policy','plan')}
            write(out/'input_profile.json',item)
            record['cases'][f'{period}/{scenario}']=bound(out/'result.json')
            record['input_profiles'][f'{period}/{scenario}']=bound(out/'input_profile.json')
    if tamper:
        write(tmp_path/'cases/uniform/oos/combined/input_profile.json',{})
    monkeypatch.setattr(plugin,'validate_native',lambda folder,item,scenario,save:observed.append(item['plan']['path']))
    if tamper:
        with pytest.raises(ValueError):instance.phase_acceptance_criteria(final_metrics={'recipe_identity':'uniform'})
    else:
        assert instance.phase_acceptance_criteria(final_metrics={'recipe_identity':'uniform'})[0].passed
        assert len(set(observed))==8
