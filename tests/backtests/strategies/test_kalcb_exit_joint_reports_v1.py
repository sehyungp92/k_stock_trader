from pathlib import Path
import pytest
from backtests.strategies.kalcb import exit_joint_full_diagnostics_v1 as report
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound

BASE=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011/v16_exit_qualification')


@pytest.mark.parametrize('mode,name,period,scenario',[
    ('compounding','prior100_failure8','oos','cost'),
    ('fixed','control','oos','baseline'),
    ('competing','prior125','oos','participation'),
])
def test_actual_native_source_account_and_report_metrics(tmp_path,monkeypatch,mode,name,period,scenario):
    if mode=='competing':
        p=read(BASE/'protocol.json');folder=Path(p['competing_prior125_references'][period][scenario])
        source=read(Path('E:/CodexResearch/k_stock_trader/kalcb_selective_recovery_20261010/v7_funded_qualification/protocol.json'))
        entry={'folder':str(folder),'item':source['profiles']['expanding_removal_prior125'][period]}
    else:
        folder=BASE/mode/'cases'/name/period/scenario
        entry={'folder':str(folder),'input_profile':bound(folder/'qualification_input_profile.json')}
    profile=mode+'_'+name
    protocol={'profiles':{profile:{period:{scenario:entry}}},'recipes':{profile:{'prior':125 if mode=='competing' else 100,'mode':mode,'role':'actual test case'}},
        'assessment':bound(BASE/'economic_decision.json'),'native_protocol':bound(BASE/'protocol.json')}
    write(tmp_path/'protocol.json',protocol);monkeypatch.setattr(report,'authenticated_protocol',lambda _:protocol)
    plugin=report.ExitJointReportPlugin(tmp_path,profile,period,scenario)
    native=read(folder/'result.json')
    assert plugin.metrics['official_mtm_net_return_pct']*100==pytest.approx(native['return_pct'])
    assert plugin.metrics['max_drawdown_pct']*100==pytest.approx(native['drawdown_pct'])
    assert plugin.bundle['cases'][scenario]['validation']['passed']
    with pytest.raises(ValueError):plugin.canonicalize_mutations({'max_positions':999})


def test_locked_shared_framework_and_all_stress_selector():
    from scripts import kalcb_exit_joint_reports_v1 as driver
    from backtests.strategies.kalcb import late_artifact_validation_v2 as validator
    assert report.validate_native is validator.validate_native
    assert 'PhaseRunner' in driver.render.__code__.co_names
    assert report.ExitJointReportPlugin.run_enhanced_diagnostics is report.ExitJointReportPlugin.run_phase_diagnostics


def test_report_design_retains_stresses_funding_and_rejected_trials(monkeypatch):
    from scripts import kalcb_exit_joint_reports_register_v1 as register
    monkeypatch.setattr(register,'bound',lambda p:{'path':str(p),'sha256':'test-only'})
    class AnyAccount:
        def __contains__(self,key):return True
    recipes,profiles=register.profile_design(BASE,{'accounts':AnyAccount()})
    assert len(recipes)==8
    assert sum(len(c) for ps in profiles.values() for c in ps.values())==48
    assert set(profiles['compounding_prior100_failure8']['oos'])=={'baseline','participation','cost','combined'}
    assert set(profiles['fixed_prior100_failure8']['oos'])=={'baseline','combined'}
    assert profiles['rejected_prior125_failure8']['oos']['combined']['assessment_kind']=='screening'
    assert all(not name.startswith('identity') for name in profiles)
