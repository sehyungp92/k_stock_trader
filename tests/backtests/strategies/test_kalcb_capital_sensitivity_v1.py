"""Historical bridges and denominator defects for the capital sensitivity."""
import importlib.util
import json
from pathlib import Path
import shutil
import pytest
from backtests.strategies.kalcb import capital_sensitivity_diagnostics_v1 as new
from backtests.strategies.kalcb.fixed_allocation_funding_v1 import funding_review
from backtests.strategies.kalcb.late_artifact_validation_v2 import validate_case
from scripts.kalcb_population_assessment_v1 import account
from scripts.kalcb_frontier_activity_attribution_v1 import opportunities

ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_frontier_completion_20261016/v3')
INDEX=json.loads((ROOT/'account_index.json').read_text())
REPO=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('previous_frontier',REPO/'docs/optimisation/kalcb/evidence/reconstruct_frontier.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
FUNCTIONS=[(new.metrics,previous.metrics),(new.account,account),(new.opportunities,opportunities),(new.funding_review,funding_review)]


@pytest.mark.parametrize('scenario',['baseline','combined'])
@pytest.mark.parametrize('functions',FUNCTIONS)
def test_parameterized_helpers_reproduce_real_historical_accounts(scenario,functions):
    folder=Path(INDEX['fixed_prior100/oos/'+scenario]);after,before=functions
    assert after(folder,100_000_000.)==before(folder)


@pytest.mark.parametrize('functions',FUNCTIONS)
def test_wrong_initial_capital_is_detected(functions):
    with pytest.raises((ValueError,AssertionError)):
        functions[0](Path(INDEX['fixed_prior100/oos/baseline']),175_000_000.)


def test_higher_capital_denominators_and_cash_reconciliation(tmp_path):
    # Synthetic diagnostic fixture only, never accepted as native/source evidence.
    source=Path(INDEX['fixed_prior100/oos/baseline']);folder=tmp_path/'synthetic';folder.mkdir()
    for name in ('result.json','equity.json','daily.json','trades.json','fills.json','orders.jsonl.gz','independent_account_verification.json'):
        shutil.copyfile(source/name,folder/name)
    initial=175_000_000.;extra=75_000_000.
    eq=new.read(folder/'equity.json');daily=new.read(folder/'daily.json');result=new.read(folder/'result.json')
    eq=[[s,v+extra] for s,v in eq]
    for row in daily:row['equity']+=extra
    peak=initial;dd=0.
    for _,value in eq:peak=max(peak,value);dd=max(dd,100*(1-value/peak))
    result['return_pct']=100*(eq[-1][1]/initial-1);result['drawdown_pct']=dd
    for name,value in [('equity.json',eq),('daily.json',daily),('result.json',result)]:new.write(folder/name,value)
    assert new.metrics(folder,initial)['return_pct']==result['return_pct']
    assert new.funding_review(folder,initial)['terminal_cash_krw']==pytest.approx(eq[-1][1])
    assert new.account(folder,initial)['net_pnl_krw']==pytest.approx(eq[-1][1]-initial)
    assert new.opportunities(folder,initial)['summary']['net_pnl_krw']==pytest.approx(eq[-1][1]-initial)
    with pytest.raises((ValueError,AssertionError)):new.metrics(folder)


def test_unchanged_independent_verifier_rejects_wrong_capital():
    folder=Path(INDEX['fixed_prior100/oos/baseline']);item=new.read(ROOT/'registration.json')['profiles']['fixed_prior100/oos']
    result=new.read(folder/'result.json')
    profile=dict(execution_source=dict(source_generation_hash=item['source_generation_hash']),audit_manifest=item['manifest']['path'],
        policy=dict(name='interaction_cost_margin',manifest_sha256=item['policy']['sha256'],directory=str(Path(item['policy']['path']).parent)),
        initial_equity_krw=100_000_000.,date_range=dict(start='2026-04-01',end='2026-07-10',sessions=69),metrics=dict(baseline=result))
    assert validate_case(folder,profile,'baseline')['validation']['passed']
    profile['initial_equity_krw']=175_000_000.
    with pytest.raises(ValueError,match='cash reconciliation'):validate_case(folder,profile,'baseline')
