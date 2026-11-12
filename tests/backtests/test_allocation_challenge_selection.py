import copy
import pytest
from scripts.portfolio_allocation_challenge import BATTERY, PROTOCOL, OVERNIGHT_PROTOCOL, passes, choose, midpoint_grid, budget_jobs
from scripts.portfolio_allocation_profile import verify_challenge_profile


def candidate(ret=80,dd=10,trades=523,passed=True):
    return {s:{'return_pct':ret,'drawdown_pct':dd,'trades':trades,'gate':{'pass':passed}} for s in BATTERY}


def test_random_seed_never_becomes_candidate_even_when_most_profitable():
    assert choose({'symbol':candidate(),'random11':candidate(300)})[0]=='symbol'


def test_renaming_a_random_policy_cannot_make_it_eligible():
    random=candidate(300)
    for m in random.values():m['allocation_mode']='random'
    assert choose({'symbol':candidate(),'renamed':random})[0]=='symbol'


def test_one_adverse_failure_excludes_whole_candidate():
    bad=candidate(200);bad['auction_nonfill_combined']['gate']['pass']=False
    assert choose({'good':candidate(),'bad':bad})[0]=='good'


def test_incomplete_battery_cannot_win():
    partial=candidate(200);partial.pop('cost')
    assert choose({'complete':candidate(),'partial':partial})[0]=='complete'


def test_near_tie_prefers_frequency_not_largest_point_return():
    assert choose({'more_entries':candidate(80,trades=525),'more_return':candidate(80.9,trades=520)})[0]=='more_entries'


@pytest.mark.parametrize('mode',['native_rank','proportional'])
def test_allocation_return_gain_does_not_excuse_lower_frequency(mode):
    assert choose({'k075_o075':candidate(),mode:candidate(85,trades=522)})[0]=='k075_o075'


@pytest.mark.parametrize('metric,value',[('drawdown_pct',12.6),('trades',514),('overnight_max',.81),('overnight_mean',.31),('worst_session_pct',-4.6)])
def test_adverse_risk_gate_is_multidimensional(metric,value):
    m={'drawdown_pct':12.,'trades':523,'overnight_max':.75,'overnight_mean':.23,'worst_session_pct':-3.4}
    assert passes(m,'auction_nonfill_combined',PROTOCOL)['pass']
    m[metric]=value
    assert not passes(m,'auction_nonfill_combined',PROTOCOL)['pass']


def test_baseline_gate_remains_stricter():
    m={'drawdown_pct':11.1,'trades':523,'overnight_max':.75,'overnight_mean':.23,'worst_session_pct':-3.4}
    assert not passes(m,'baseline',PROTOCOL)['pass']
    assert passes(m,'combined',PROTOCOL)['pass']


@pytest.mark.parametrize('field,value',[('live_deployment_enabled',True),('independent_performance_established',True),('status','unverified')])
def test_profile_cannot_promote_to_live_or_independent_claim(field,value):
    p={'status':'research_selected_full_native_audits_passed','live_deployment_enabled':False,'independent_performance_established':False}
    p[field]=value
    with pytest.raises(ValueError,match='research-only'):verify_challenge_profile(p)


def test_publication_rehashes_frozen_source_bytes(tmp_path):
    import json
    from scripts.portfolio_allocation_profile import verify_frozen_bytes
    from scripts.replay_selected_portfolio import digest
    from kis_core.input_manifest import signature
    files=tmp_path/'files';files.mkdir();source=files/'signal.json'
    source.write_text('{"value":1}',encoding='utf-8')
    payload={'files':{'signal.json':digest(source)}}
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({**payload,'manifest_hash':signature(payload)}))
    verify_frozen_bytes({'manifest':str(manifest)},{})
    source.write_text('{"value":2}',encoding='utf-8')
    with pytest.raises(ValueError,match='frozen generation bytes'):
        verify_frozen_bytes({'manifest':str(manifest)},{})


def test_overnight_followup_keeps_every_original_risk_gate():
    assert OVERNIGHT_PROTOCOL['risk']==PROTOCOL['risk']


def test_midpoint_only_tests_a_registered_feasibility_boundary():
    data={f'k{k*100:g}_o{o*100:g}':candidate() for k,o in OVERNIGHT_PROTOCOL['budget_grid']}
    assert midpoint_grid(data)==[]
    data['k87.5_o50']['auction_nonfill_combined']['gate']['pass']=False
    assert midpoint_grid(data)==[[.8125,.5]]
    data['k75_o50']['auction_nonfill_combined']['gate']['pass']=False
    assert midpoint_grid(data)==[]


def test_budget_generation_changes_only_registered_factors_and_stress():
    base={'native_budget':{'kalcb_sizing_fraction':.75,'olr_fraction':.75,'max_kalcb_stop_risk_fraction':.02},
        'olr_capacity':{'capacity_redistribute':True},'policy':{'max_gross_notional':50_000_000}}
    before=copy.deepcopy(base);recipes,jobs=budget_jobs(base,OVERNIGHT_PROTOCOL['budget_grid'])
    assert base==before and len(jobs)==32 and len(recipes)==12
    for recipe in recipes.values():
        assert recipe['policy']==base['policy'] and recipe['olr_capacity']==base['olr_capacity']
        assert recipe['native_budget']['allocation_mode']=='symbol'
        assert recipe['native_budget']['max_kalcb_stop_risk_fraction']==.02


def test_resume_does_not_require_unfinished_midpoints_to_decide_coarse_stage(tmp_path,monkeypatch):
    from scripts import portfolio_allocation_challenge as driver
    data={f'k{k*100:g}_o{o*100:g}':candidate() for k,o in OVERNIGHT_PROTOCOL['budget_grid']}
    for battery in data.values():
        for m in battery.values():m['completion']={'path':'completed-case','sha256':'bound'}
    # A interrupted midpoint folder is allowed to exist. Coarse decisions use
    # complete coarse cases, even while that folder has no completion artifact.
    (tmp_path/'midpoint').mkdir()
    def observed(root,partial=False):
        if not partial:raise FileNotFoundError('Unfinished midpoint completion')
        return data
    monkeypatch.setattr(driver,'run',lambda *a,**k:None)
    monkeypatch.setattr(driver,'collect',observed)
    monkeypatch.setattr(driver,'select',lambda root:{'winner':None})
    driver.overnight_pipeline(tmp_path)
    assert driver.read(tmp_path/'stage_decision.json')['registered_midpoints_triggered']==[]
    assert driver.read(tmp_path/'pipeline_progress.json')['stage']=='completed_without_risk_feasible_replacement'
