"""Adversarial promotion checks for the registered clock comparison."""
import copy
import pytest
from scripts import olr_minute_exit_qualify_v5 as q


@pytest.fixture
def evidence(monkeypatch):
    g=dict(is_return_loss_max_pp_original=3,is_return_loss_max_pp_guard=3,is_return_loss_max_pp_selected=1.5,
        is_dd_increase_max_pp_guard=2,oos_return_improvement_min_pp_selected=.25,oos_dd_reduction_min_pp_selected=.75,
        is_trades_min=186,oos_trades_min=47,is_entry_days_min=88,oos_entry_days_min=29)
    monkeypatch.setattr(q.n,'read',lambda p:{'gates':g})
    monkeypatch.setattr(q.repair,'frequency_economics',lambda a,b:dict(net_incremental_notional=2e6,median_added_notional=2e6))
    metric=lambda ret,dd,entries,days:dict(return_pct=ret,drawdown_pct=dd,complete_fine_drawdown_pct=dd+1,trades=entries,entry_days=days)
    old={k:dict(train=metric(53.638,8.176,186,88),oos=metric(12.19,16.28,46,29)) for k in ('windows','stress','capped')}
    fine=copy.deepcopy(old)
    refs=dict(original={'train':metric(55.047,7.203,176,88)},original_stress={'train':metric(55.047,7.203,176,88)},
        guard={k:dict(train=metric(54.535,8.175,177,88)) for k in ('windows','stress')})
    return metric(54,8.2,186,88),metric(14,14,47,30),old,fine,refs


def test_full_success_requires_every_grid(evidence):
    a,z,old,fine,refs=evidence
    assert all(q.gates(a,z,old,fine,refs,'windows').values())
    z['complete_fine_drawdown_pct']=fine['windows']['oos']['complete_fine_drawdown_pct']
    checks=q.gates(a,z,old,fine,refs,'windows')
    assert checks['oos_drawdown'] and checks['matched_oos_common_drawdown']
    assert not checks['matched_oos_complete_fine_drawdown']
    assert not all(q.risk_gates(checks).values())


@pytest.mark.parametrize('group',['windows','stress','capped'])
def test_old_comparison_cannot_replace_fine_control(evidence,group):
    a,z,old,fine,refs=evidence
    fine[group]['oos'].update(return_pct=15,drawdown_pct=13,complete_fine_drawdown_pct=14)
    checks=q.gates(a,z,old,fine,refs,group)
    assert checks['oos_return'] and checks['oos_drawdown']
    assert not checks['matched_oos_return'] and not checks['matched_oos_common_drawdown']
    assert not checks['matched_oos_complete_fine_drawdown']


@pytest.mark.parametrize('group',['windows','stress','capped'])
def test_risk_status_cannot_relax_returns(evidence,group):
    a,z,old,fine,refs=evidence
    a['return_pct']=45
    assert not all(q.risk_gates(q.gates(a,z,old,fine,refs,group)).values())


def test_component_frequency_exception_does_not_promote(evidence):
    a,z,old,fine,refs=evidence
    z['trades']=46;z['entry_days']=28
    checks=q.gates(a,z,old,fine,refs,'windows')
    assert not all(checks.values()) and all(q.risk_gates(checks).values())


def test_tiny_additional_fills_do_not_satisfy_frequency(evidence,monkeypatch):
    a,z,old,fine,refs=evidence
    monkeypatch.setattr(q.repair,'frequency_economics',lambda a,b:dict(net_incremental_notional=999999,median_added_notional=2e6))
    checks=q.gates(a,z,old,fine,refs,'windows')
    assert checks['oos_frequency'] and not checks['oos_meaningful_added_notional']
