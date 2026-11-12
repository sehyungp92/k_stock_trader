import pytest
from scripts.kalcb_recovery_final_artifacts_v1 import require_ready
from scripts.kalcb_recovery_qualification_round_v4 import PROFILES, PERIODS, SCENARIOS


def receipts():
    return {'complete': True, 'results': [
        {'name':n,'period':p,'scenario':s,'metrics':{'audit_status':'passed','return_pct':999}}
        for n in PROFILES for p in PERIODS for s in SCENARIOS
    ]}, {'complete':True,'passed':True}


def test_complete_32_scenario_population_and_forecasts_are_required():
    require_ready(*receipts())


@pytest.mark.parametrize('defect', ['missing','duplicate','failed','unverified','forecast_failed','forecast_incomplete'])
def test_attractive_return_does_not_bypass_qualification(defect):
    cases, forecasts = receipts()
    if defect == 'missing': cases['results'].pop()
    elif defect == 'duplicate': cases['results'][-1] = cases['results'][0]
    elif defect in ('failed','unverified'): cases['results'][0]['metrics']['audit_status'] = defect
    elif defect == 'forecast_failed': forecasts['passed'] = False
    else: forecasts['complete'] = False
    with pytest.raises(ValueError):
        require_ready(cases, forecasts)
