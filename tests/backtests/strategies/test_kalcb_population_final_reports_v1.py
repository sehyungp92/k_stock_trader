import json
from copy import deepcopy
import pytest
from backtests.strategies.kalcb.qualified_artifacts import digest
from backtests.strategies.kalcb.population_full_diagnostics_v1 import population_report_context
from scripts.kalcb_funded_qualification_assessment_v1 import require_complete, difference_in_deltas
from scripts.kalcb_population_final_artifacts_v1 import authenticate_capsule
from backtests.auto.shared.cache_keys import stable_signature
from scripts.kalcb_population_report_supervisor_v1 import dependency_ready


def context(tmp_path,prior=125):
    path=tmp_path/'funding.json'
    evidence={'passed':True,'recipe':'expanding_removal_prior'+str(prior),'period':'oos',
        'reference_krw':100_000_000.,'external_capital_flows':0,'cash_resets':0,
        'four_identity_bridges_passed':True,'cases':{'baseline':{},'combined':{}},'bindings':[]}
    path.write_text(json.dumps(evidence))
    profile={'date_range':{'start':'2026-04-01','end':'2026-07-10','sessions':69},
        'primary_profile':evidence['recipe'],
        'recovery_report_context':{'period':'oos','recipe':'expanding','independent_validation':False,
            'june_july_used_for_research':True,'training_contract':{'minimum_training_sessions':60,
            'refit_sessions':20,'embargo_sessions':1,'continuous_original_fitting_clock':True,
            'mature_labels_only':True,'earlier_oof_calibration_only':True}},
        'population_report_context':{'recipe':evidence['recipe'],'parent':'expanding','prior':prior,
            'calibration_contract':{'lookback_sessions':60,'refit_sessions':5,'embargo_sessions':1,
            'minimum_count':20,'minimum_sessions':8,'local_minimum_count':8,'local_minimum_sessions':5,
            'unsupported_identity':True,'mature_labels_only':True,'checkpoint_readmission':True,
            'selection':'exact_supported_nonpositive_removal_only'}},
        'funded_allocation_evidence':{'path':str(path),'sha256':digest(path)}}
    return profile,{'files':{str(path.resolve()):digest(path)}}


def test_exact_variant_and_completed_funding_context(tmp_path):
    p,c=context(tmp_path)
    result=population_report_context(p,c)
    assert result['recipe']=='expanding_removal_prior125'
    assert result['parent_recipe']=='expanding'
    assert 'prior count125' in result['training']
    assert result['canonical_replaced'] is False and result['production_enabled'] is False
    assert 'no independent validation claim' in result['validation_scope']


@pytest.mark.parametrize('mutation',['prior','period','funding','history'])
def test_mislabelled_or_unqualified_report_fails(tmp_path,mutation):
    p,c=context(tmp_path)
    if mutation=='prior':p['population_report_context']['prior']=100
    elif mutation=='period':p['date_range']['start']='2026-06-01'
    elif mutation=='history':p['population_report_context']['calibration_contract']['mature_labels_only']=False
    else:
        path=tmp_path/'funding.json';f=json.loads(path.read_text());f['four_identity_bridges_passed']=False
        path.write_text(json.dumps(f));p['funded_allocation_evidence']['sha256']=digest(path);c['files'][str(path.resolve())]=digest(path)
    with pytest.raises(ValueError):population_report_context(p,c)


def test_partial_batch_cannot_be_assessed(tmp_path):
    (tmp_path/'stage_completion.json').write_text(json.dumps({'complete':True,'native_cases':35}))
    (tmp_path/'progress.json').write_text('[]')
    with pytest.raises(ValueError,match='Incomplete'):require_complete(tmp_path)


def test_duplicate_native_cases_cannot_be_assessed(tmp_path):
    rows=[{'mode':mode,'name':'duplicated','period':'is','scenario':'baseline','returncode':0}
          for mode,n in [('compounding',8),('identity',4),('fixed',24)] for _ in range(n)]
    (tmp_path/'stage_completion.json').write_text(json.dumps({'complete':True,'native_cases':36}))
    (tmp_path/'progress.json').write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='Duplicated'):require_complete(tmp_path)


def test_policy_effect_is_compared_within_each_funded_regime():
    def a(ret,dd,trades):return {'metrics':{'return_pct':ret,'drawdown_pct':dd,'trades':trades,'active_days':20}}
    # The fixed candidate return can be smaller while its gain over its own
    # parent is larger. Never compare candidates across capital regimes alone.
    delta=difference_in_deltas(a(9,8,40),a(-3,13,50),a(10,10,42),a(0,14,55))
    assert delta=={'return_pct':2,'drawdown_pct':-1,'trades':3,'active_days':0}


def test_inherited_lineage_capsule_rejects_changed_evidence(tmp_path):
    evidence=tmp_path/'model.json';evidence.write_text('{"fit":"past"}')
    payload={'files':{str(evidence):digest(evidence)}}
    capsule=tmp_path/'acceptance.json'
    capsule.write_text(json.dumps({**payload,'manifest_hash':stable_signature(payload)}))
    authenticate_capsule(capsule)
    evidence.write_text('{"fit":"future"}')
    with pytest.raises(ValueError,match='Changed binding'):authenticate_capsule(capsule)


def test_changed_acceptance_file_list_cannot_hide_missing_lineage(tmp_path):
    payload={'files':{'some-model.json':'expected'}}
    capsule=tmp_path/'acceptance.json'
    capsule.write_text(json.dumps({'files':{},'manifest_hash':stable_signature(payload)}))
    with pytest.raises(ValueError,match='Changed inherited'):authenticate_capsule(capsule)


def test_report_pipeline_waits_for_healthy_process_to_exit(tmp_path):
    (tmp_path/'status.json').write_text(json.dumps({'stage':'awaiting_final_assessment'}))
    (tmp_path/'stage_completion.json').write_text(json.dumps({'complete':True,'native_cases':36}))
    class Process:
        def create_time(self):return 17.
        def cmdline(self):return ['python','native']
    identity={'pid':123,'process_created':17.,'command':['python','native']}
    assert dependency_ready(tmp_path,identity,lambda pid:Process()) is False
    assert dependency_ready(tmp_path,identity,lambda pid:None) is True


@pytest.mark.parametrize('changed',['command','creation'])
def test_pid_reuse_cannot_authorize_a_report_start(tmp_path,changed):
    (tmp_path/'status.json').write_text(json.dumps({'stage':'native_fixed'}))
    class Process:
        def create_time(self):return 18. if changed=='creation' else 17.
        def cmdline(self):return ['python','different'] if changed=='command' else ['python','native']
    identity={'pid':123,'process_created':17.,'command':['python','native']}
    with pytest.raises(ValueError,match='identity changed'):dependency_ready(tmp_path,identity,lambda pid:Process())


def test_failed_prerequisite_cannot_be_treated_as_completed(tmp_path):
    (tmp_path/'status.json').write_text(json.dumps({'stage':'attention_required','error':'audit failed'}))
    (tmp_path/'stage_completion.json').write_text(json.dumps({'complete':True,'native_cases':36}))
    with pytest.raises(ValueError,match='needs attention'):dependency_ready(tmp_path,{},lambda pid:None)
