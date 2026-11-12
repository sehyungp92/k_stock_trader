import copy,gzip,json
from pathlib import Path
import pytest
from backtests.strategies.kalcb import exit_qualification_assessment_v1 as a


def fixture(root):
    stages={mode:{name:{'scenarios':(['baseline','participation','cost','combined'] if mode=='compounding' else ['baseline','combined'])}
        for name in (['control','prior100_failure6','prior100_failure8'] if mode=='compounding' else ['control','prior100_failure8'])}
        for mode in ('compounding','identity','fixed')}
    a.write(root/'protocol.json',{'stages':stages})
    rows=[dict(mode=m,name=n,period=p,scenario=s,authenticated_reuse=m=='compounding' and s in ('baseline','combined'),
        exact_bridge=m=='identity' or n=='control',metrics={'audit_status':'passed'},result={},input_profile={},funding={},evidence={},log=None)
        for m,n,p,s in sorted(a.expected_cases({'stages':stages}))]
    a.write(root/'progress.json',rows);a.write(root/'stage_completion.json',dict(complete=True,native_cases=40,fresh_native_cases=28))
    return rows


def test_complete_unique_design(tmp_path,monkeypatch):
    fixture(tmp_path);monkeypatch.setattr(a,'verify',lambda _:None)
    assert len(a.require_design(tmp_path))==40


@pytest.mark.parametrize('poison',['duplicate','missing','identity','funded_parent','stress_parent','reuse','audit'])
def test_failed_gates(tmp_path,monkeypatch,poison):
    rows=fixture(tmp_path);monkeypatch.setattr(a,'verify',lambda _:None)
    if poison=='duplicate':rows[-1]=rows[0]
    if poison=='missing':rows.pop()
    if poison in ('identity','funded_parent','stress_parent'):
        mode={'identity':'identity','funded_parent':'fixed','stress_parent':'compounding'}[poison]
        next(r for r in rows if r['mode']==mode and r['name']=='control')['exact_bridge']=False
    if poison=='reuse':rows[0]['authenticated_reuse']=not rows[0]['authenticated_reuse']
    if poison=='audit':rows[0]['metrics']['audit_status']='failed'
    a.write(tmp_path/'progress.json',rows)
    with pytest.raises(ValueError):a.require_design(tmp_path)


def test_allocation_rejects_cash_reset_and_wrong_mode(tmp_path):
    obj=dict(version='fixed-allocation-reference-v1',enabled=True,reference=100000000.,rows=[],ledger_cash_mutations=0,external_capital_flows=0)
    a.write(tmp_path/'allocation_reference.json',obj);assert a.allocation_contract(tmp_path,'fixed')['enabled']
    with pytest.raises(ValueError):a.allocation_contract(tmp_path,'identity')
    obj['ledger_cash_mutations']=1;a.write(tmp_path/'allocation_reference.json',obj)
    with pytest.raises(ValueError):a.allocation_contract(tmp_path,'fixed')


def test_four_path_delta_is_not_candidate_rescaling():
    c={'metric_delta':dict(return_pct=2,drawdown_pct=-.3,trades=1,active_days=0)}
    f={'metric_delta':dict(return_pct=1,drawdown_pct=-.2,trades=0,active_days=0)}
    assert a.difference_in_deltas(f,c)['return_pct']==-1
    assert a.difference_in_deltas(f,c)['drawdown_pct']==pytest.approx(.1)


def test_partial_exits_join_only_once(tmp_path):
    p=tmp_path/'p';c=tmp_path/'c'
    a.write(p/'trades.json',[dict(symbol='A',entry_fill_time='t',net_pnl=10,qty=3),dict(symbol='A',entry_fill_time='t',net_pnl=20,qty=2)])
    a.write(c/'trades.json',[dict(symbol='A',entry_fill_time='t',net_pnl=15,qty=5)])
    x=a.tail_comparison(c,p);assert x['parent_top10_net_pnl']==30 and x['candidate_same_entry_net_pnl']==15


def test_real_direct_support_and_parent_collision_rejection(tmp_path):
    src=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011/v16_exit_qualification/compounding/cases/prior100_failure8/oos/baseline')
    import shutil
    for f in ('mechanism_evidence.json','fills.json','orders.jsonl.gz'):shutil.copyfile(src/f,tmp_path/f)
    x=a.direct_support(tmp_path);assert x['authenticated_actual_support']['filled_positions']==22
    assert x['authenticated_actual_support']['partial_executions']>22
    evidence=a.read(tmp_path/'mechanism_evidence.json');evidence['summary']['executions'][0]['parent_exit_reason']='parent_exit'
    a.write(tmp_path/'mechanism_evidence.json',evidence)
    with pytest.raises(ValueError,match='priority'):a.direct_support(tmp_path)


def test_real_native_attribution_and_paired_zero():
    folder=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011/v16_exit_qualification/fixed/cases/control/oos/baseline')
    attr=a.account_attribution(folder);assert len(attr['daily_dates'])==69
    for block in (5,10):assert a.paired_blocks(attr['daily_returns'],attr['daily_returns'],block)['full_period_net_return_delta_pp_95_interval']==[0,0]
