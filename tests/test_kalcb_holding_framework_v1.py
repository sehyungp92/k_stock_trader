from pathlib import Path
from backtests.strategies.kalcb.retention_research_io_v1 import write,read,bound
from backtests.strategies.kalcb import holding_account_plugin_v1 as hp


def test_failed_readiness_finalizes_without_fabricating_candidate_accounts(tmp_path,monkeypatch):
    from backtests.auto.shared.phase_runner import PhaseRunner
    from backtests.auto.shared.round_manager import RoundManager
    write(tmp_path/'study.json',{'code':str(Path.cwd()),'source_generation_hash':'fixture-source',
        'bound_inputs':[],'source_programs':[],'recipes':{'control':{},'selective':{}},'periods':{}})
    calls=[];checks=[]
    def native(name,period,scenario):
        calls.append((name,period,scenario));folder=tmp_path/'cases'/name/period/scenario
        result={'return_pct':50. if period=='is' else 5.,'drawdown_pct':10.,'trades':100,'active_days':50,
            'material_funded_entries':90}
        write(folder/'result.json',result);write(folder/'input_profile.json',{'fixture':True})
        write(folder/'execution_audit.json',{'audit_receipt':{'source_generation_hash':'fixture-source',
            'identity':{'input_manifest':{'manifest_hash':'fixture-manifest'},'qualification_snapshot_hash':'fixture-snapshot'}},
            **{k:0 for k in ('same_bar_fill_count','forced_replay_close_count','rejected_order_count','invalid_order_rejection_count')}})
        return {**result,'result_binding':bound(folder/'result.json')}
    monkeypatch.setattr(hp,'build',lambda root:{'economic_readiness_pass':False,'readiness':{'baseline':{'passed':False}}})
    monkeypatch.setattr(hp,'validate_native',lambda folder,*args,**kwargs:checks.append(str(folder)))
    plugin=hp.HoldingAccountPlugin(tmp_path/'study.json',native)
    manager=RoundManager('stock',plugin.name,base_dir=tmp_path/'framework')
    state=PhaseRunner(plugin,manager.get_round_dir(1),round_name='Holding fixture',round_manager=manager,
        round_num=1,max_rounds=1,max_retries=0,max_diagnostic_retries=0).run_all_phases()
    assert set(state.completed_phases)=={1}
    assert len(calls)==8 and {c[0] for c in calls}=={'control'}
    record=plugin.records['selective']
    assert not record['native_policy_executed'] and not record['metrics']['economic_qualified']
    assert record['metrics']['policy_account_metrics_are_control_reference']
    assert not plugin.diagnostic()['objective_achieved_on_reused_development']
    assert len(checks)>=8 and all('control' in path for path in checks)
