"""Resolve every historical profile through the unchanged real native verifier."""
from pathlib import Path
import pytest
from scripts import kalcb_fixed_veto_v2 as r


@pytest.mark.parametrize('arm',['prior','fixed','veto'])
@pytest.mark.parametrize('scenario',r.SCENARIOS)
def test_all_real_predecessor_profile_and_verdict_interfaces(arm,scenario):
    if arm=='veto':
        folder=r.C10/'cases'/('capital_'+scenario)
        expected=r.read(r.C10/'assessment.json')['capital'][scenario]['verdict']['qualified']
    else:
        name=('compound_' if arm=='prior' else 'fixed_')+scenario
        row=r.read(r.C6/'assessment.json')['case_ledger'][name]
        folder=Path(row['folder']);expected=row['qualified']
    item=r.registered_profile(arm,scenario,folder)
    verdict=r.classify_case(folder,item,save=False)
    assert verdict['qualified']==expected
    if expected:
        initial=r.read(item['plan']['path'])['capital_sensitivity']['initial_equity']
        assert r.diag.funding_review(folder,initial)['verified']
