import importlib.util
from pathlib import Path
import pytest
from scripts import kalcb_mechanism_assessment_v1 as a


def fixture(root):
    recipes={f'p{p}_{h}':{'prior':p,'hold':h} for p in (100,125) for h in (0,6,8)}
    rows=[{'name':n,'period':p,'scenario':s,'is_identity':r['hold']==0}
          for n,r in recipes.items() for p in ('is','oos') for s in ('baseline','combined')]
    a.write(root/'protocol.json',{'recipes':recipes})
    a.write(root/'status.json',{'stage':'complete'})
    a.write(root/'stage_completion.json',{'complete':True,'native_cases':24})
    a.write(root/'bridge_gate.json',{'passed':True})
    a.write(root/'progress.json',rows)
    return rows


def test_gate_waits_for_exact_process_even_with_complete_files(tmp_path,monkeypatch):
    fixture(tmp_path);monkeypatch.setattr(a,'exact_process',lambda _:True)
    assert not a.check_gate(tmp_path,{})
    monkeypatch.setattr(a,'exact_process',lambda _:False)
    assert a.check_gate(tmp_path,{})


@pytest.mark.parametrize('poison',['duplicate','missing','bridge','failed','dead_early'])
def test_gate_rejects_incomplete_or_failed_evidence(tmp_path,monkeypatch,poison):
    rows=fixture(tmp_path);monkeypatch.setattr(a,'exact_process',lambda _:False)
    if poison=='duplicate':rows[-1]=rows[0];a.write(tmp_path/'progress.json',rows)
    if poison=='missing':a.write(tmp_path/'progress.json',rows[:-1])
    if poison=='bridge':a.write(tmp_path/'bridge_gate.json',{'passed':False})
    if poison=='failed':a.write(tmp_path/'status.json',{'stage':'attention_required'})
    if poison=='dead_early':(tmp_path/'stage_completion.json').unlink()
    with pytest.raises((RuntimeError,ValueError)):a.check_gate(tmp_path,{})


def test_exact_identity_requires_command_and_creation(monkeypatch):
    import psutil
    class Process:
        def create_time(self):return 123.0
        def cmdline(self):return ['python','correct']
    monkeypatch.setattr(psutil,'Process',lambda _:Process())
    assert a.exact_process({'pid':1,'process_created':123,'command':['python','correct']})
    assert not a.exact_process({'pid':1,'process_created':124,'command':['python','correct']})
    assert not a.exact_process({'pid':1,'process_created':123,'command':['python','other']})


def test_tail_join_aggregates_partial_exits_once(tmp_path):
    parent=tmp_path/'p';candidate=tmp_path/'c'
    a.write(parent/'trades.json',[{'symbol':'A','entry_fill_time':'t','net_pnl':10,'qty':3},
        {'symbol':'A','entry_fill_time':'t','net_pnl':20,'qty':2}])
    a.write(candidate/'trades.json',[{'symbol':'A','entry_fill_time':'t','net_pnl':15,'qty':5}])
    result=a.tail_comparison(candidate,parent)
    assert result['parent_top10_net_pnl']==30
    assert result['candidate_same_entry_net_pnl']==15
    assert len(result['parent_top10_entries'])==1


def test_generic_attribution_and_zero_difference_uncertainty_use_real_parent():
    spec=importlib.util.spec_from_file_location('registered_generic',a.GENERIC)
    generic=importlib.util.module_from_spec(spec);spec.loader.exec_module(generic)
    folder=a.GENERIC.parent.parent/'cases/control/oos/combined'
    attr=generic.account_attribution(folder)
    assert len(attr['daily_dates'])==69
    for block in (5,10):
        interval=generic.paired_blocks(attr['daily_returns'],attr['daily_returns'],block)
        assert interval['full_period_net_return_delta_pp_95_interval']==[0,0]
        assert interval['observed_delta_pp']==0
