import json

import pytest

from scripts import portfolio_native_preservation_study as runner


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def fixture(tmp_path,monkeypatch):
    parent=tmp_path/'old';root=tmp_path/'new'
    write(parent/'study.json',{'source_generation_hash':'old'})
    selection={'baseline_max_dd_pct':11.,'combined_max_dd_pct':12.5,'near_return_band_pct_points':1.}
    write(parent/'finalist.json',{'return_floors':[113.,72.],'reference_frontier':['kalcb_size_200'],
                                 'results':{},'rejected':{},'winner':None})
    names=['joint_control','native_a','native_b']
    write(root/'study.json',{'source_generation_hash':'new','candidates':{n:{'allocation':{}} for n in names}})
    write(root/'registration.json',{'parent_study':str(parent/'study.json'),
        'parent_study_sha256':runner.digest(parent/'study.json'),
        'parent_finalist_sha256':runner.digest(parent/'finalist.json'),'selection':selection,
        'jobs':[{'name':n+'_'+s} for n in ['kalcb_size_200',*names] for s in ('baseline','combined')],
        'portfolios':names,'bridge_cases':{'kalcb_size_200':'kalcb_size_200','joint_control':'old_joint'},
        'frequency_floors':[369,371]})
    values={}
    for generation in (parent,root):
        for name,pair in {'kalcb_size_200':[113.,72.],'joint_control':[116.,63.],
                          'old_joint':[116.,63.],'native_a':[117.,77.], 'native_b':[118.,77.5]}.items():
            for i,scenario in enumerate(('baseline','combined')):
                values[generation/'cases'/(name+'_'+scenario)]={
                    'return_pct':pair[i],'drawdown_pct':10.+i,'trades':520 if name=='native_a' else 500,
                    'active_days':130,'strategy_counts':{'KALCB':350,'OLR':150}}
    monkeypatch.setattr(runner,'verify_case',lambda path,study:values[path])
    for path,value in values.items():
        for filename in ('result.json','completion.json','execution_audit.json'):write(path/filename,value)
    return root,parent,values


def test_native_followup_keeps_risk_gates_and_uses_frequency_tie_break(tmp_path,monkeypatch):
    root,_,_=fixture(tmp_path,monkeypatch)
    runner.select(root/'study.json')
    final=json.loads((root/'finalist.json').read_text())
    assert final['winner']=='native_a'
    assert final['frequency_floors']==[369,371]
    assert final['rejected']['joint_control']==['standalone_return_frontier']


def test_native_followup_rejects_changed_bridge_economics(tmp_path,monkeypatch):
    root,_,values=fixture(tmp_path,monkeypatch)
    values[root/'cases/joint_control_combined']['strategy_counts']={'KALCB':349,'OLR':151}
    with pytest.raises(ValueError,match='Source bridge changed economics'):
        runner.select(root/'study.json')
    assert not (root/'finalist.json').exists()


def test_native_followup_does_not_relax_drawdown_gate(tmp_path,monkeypatch):
    root,_,values=fixture(tmp_path,monkeypatch)
    for name in ('native_a','native_b'):
        values[root/'cases'/(name+'_combined')]['drawdown_pct']=12.50001
    runner.select(root/'study.json')
    assert json.loads((root/'finalist.json').read_text())['winner'] is None
