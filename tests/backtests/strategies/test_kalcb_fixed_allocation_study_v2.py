from copy import deepcopy
import pytest
from scripts.kalcb_fixed_allocation_study_v2 import recipes,stress_recipes
from scripts.kalcb_population_qualification_v1 import recipes as prior_recipes
from backtests.strategies.kalcb.fixed_allocation_plugin_v1 import FixedAllocationPlugin
from backtests.strategies.kalcb.fixed_allocation_reference_v1 import verify_identity_bridge
from backtests.strategies.kalcb.retention_research_io_v1 import write,read
from backtests.auto.shared.phase_runner import PhaseRunner
from backtests.auto.shared.round_manager import RoundManager


def test_registration_is_finite_with_eight_missing_stress_cases_and_twenty_four_budget_cases():
    old=prior_recipes();original=deepcopy(old);new=stress_recipes(old)
    before={(n,p,s) for n,v in old.items() for p in ('is','oos') for s in v['scenarios']}
    after={(n,p,s) for n,v in new.items() for p in ('is','oos') for s in v['scenarios']}
    assert old==original and before<=after and len(before)==48 and len(after)==56
    assert len(after-before)==8
    assert all(new[n]['prior']==125 and s in ('cost','participation') for n,p,s in after-before)
    assert len(recipes())==6 and sum(2*len(v['scenarios']) for v in recipes().values())==24
    assert {v['prior'] for v in recipes().values()}=={None,100,125}


def test_shared_phase_records_all_six_funded_profiles_without_promoting(tmp_path):
    write(tmp_path/'study.json',{'recipes':recipes(),'code':str(tmp_path),'source_generation_hash':'fixture',
        'bound_inputs':[],'source_programs':[],'global_IS_reference':{'baseline':73.1303145472,'combined':47.8121465627}})
    calls=[]
    def native(name,period,scenario):
        calls.append((name,period,scenario));folder=tmp_path/'cases'/name/period/scenario
        result={'return_pct':1.,'drawdown_pct':2.,'trades':3,'active_days':2,'audit_status':'passed'}
        write(folder/'result.json',result);write(folder/'independent_account_verification.json',{'passed':True});return result
    plugin=FixedAllocationPlugin(tmp_path/'study.json',native);manager=RoundManager('stock',plugin.name,base_dir=tmp_path/'framework')
    PhaseRunner(plugin,manager.get_round_dir(1),round_name='Six funded profiles',round_manager=manager,
        round_num=1,max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()
    assert len(calls)==len(set(calls))==24
    assert len(read(tmp_path/'allocation_ledger.json')['trials'])==6
    assert read(tmp_path/'allocation_ledger.json')['no_promotion']


@pytest.mark.parametrize('drift',['fee','mark'])
def test_disabled_benchmark_bridge_rejects_fee_and_timestamped_mark_drift(tmp_path,drift):
    fill={'order_id':'1','strategy_id':'KALCB','symbol':'A','side':'BUY','qty':1,'price':100.,
          'timestamp':'2026-04-01T09:30:00+09:00','reason':'entry','metadata':{'fee':1.,'execution_phase':'continuous'}}
    for folder in (tmp_path/'reference',tmp_path/'actual'):
        write(folder/'daily.json',[{'day':'2026-04-01','equity':100.}])
        write(folder/'equity.json',[['2026-04-01T09:30:00+09:00',100.]])
        write(folder/'fills.json',[fill])
    assert verify_identity_bridge(tmp_path/'actual',tmp_path/'reference')['passed']
    if drift=='fee':
        fill['metadata']['fee']=0.;write(tmp_path/'actual/fills.json',[fill])
    else:write(tmp_path/'actual/equity.json',[['2026-04-01T09:35:00+09:00',100.]])
    with pytest.raises(ValueError,match='drift'):verify_identity_bridge(tmp_path/'actual',tmp_path/'reference')
