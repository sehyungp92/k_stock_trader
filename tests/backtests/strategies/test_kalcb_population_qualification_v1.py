import json
from pathlib import Path
import pytest
from scripts.kalcb_population_qualification_v1 import recipes
from backtests.strategies.kalcb.population_qualification_plugin_v1 import PopulationQualificationPlugin
from backtests.strategies.kalcb.retention_research_io_v1 import write,read
from backtests.auto.shared.phase_runner import PhaseRunner
from backtests.auto.shared.round_manager import RoundManager


def setup(tmp_path,failed=False):
    study=tmp_path/'study.json'
    write(study,{'recipes':recipes(),'code':str(tmp_path),'source_generation_hash':'test',
                 'bound_inputs':[],'source_programs':[]})
    calls=[]
    def native(name,period,scenario):
        calls.append((name,period,scenario))
        folder=tmp_path/'cases'/name/period/scenario
        write(folder/'independent_account_verification.json',{'passed':not failed})
        result={'return_pct':1.,'drawdown_pct':2.,'trades':3,'active_days':2,'audit_status':'passed'}
        write(folder/'result.json',result)
        return result
    return PopulationQualificationPlugin(study,native),calls


def test_record_rejects_unreconciled_account(tmp_path):
    plugin,_=setup(tmp_path,failed=True)
    with pytest.raises(ValueError,match='Unverified account'):
        plugin.evaluate_names(['control'])
    assert not (tmp_path/'cases/control/record.json').exists()


def test_shared_phase_runs_control_observers_before_variants_and_saves_all_stresses(tmp_path):
    plugin,calls=setup(tmp_path)
    manager=RoundManager('stock',plugin.name,base_dir=tmp_path/'framework')
    PhaseRunner(plugin,manager.get_round_dir(1),round_name='Qualification integration',round_manager=manager,
                round_num=1,max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()
    controls=[c for c in calls if c[0].startswith('control')]
    assert calls[:16]==controls
    assert len(calls)==48 and len(set(calls))==48
    for parent in ('expanding','expanding_atr5'):
        record=read(tmp_path/'cases'/(parent+'_removal_prior100')/'record.json')
        assert len(record['cases'])==8
        assert record['metrics']['oos_cost_return']==1.
        assert record['metrics']['is_participation_dd']==2.
    ledger=read(tmp_path/'qualification_ledger.json')
    assert ledger['funded_constant_capital_and_final_review_pending']
    assert ledger['all_scores_zero_control_retained']


def test_corrupted_study_cannot_produce_diagnostics(tmp_path):
    plugin,_=setup(tmp_path)
    plugin.evaluate_names(['control'])
    data=read(plugin.study);data['recipes']['control']['parent']='decay60'
    write(plugin.study,data)
    with pytest.raises(ValueError,match='study changed'):
        plugin.diagnostic()
