import json
import pytest
from scripts.kalcb_reclaim_observer_assessment_wait_v1 import dependency_ready


def state(root,stage='awaiting_observer_assessment',count=8):
    identity={'pid':9,'process_created':1,'command':['python','native']}
    rows=[{'name':n,'period':p,'scenario':s,'is_identity':True,'metrics':{'audit_status':'passed'}}
        for n in ('control','control_prior125') for p in ('is','oos') for s in ('baseline','combined')][:count]
    for name,data in [('supervisor_start.json',identity),('status.json',{'stage':stage,'child':None}),
        ('stage_completion.json',{'complete':True,'native_cases':8,'exact_parent_bridges':8}),('progress.json',rows)]:
        (root/name).write_text(json.dumps(data))
    return identity


def test_waits_even_after_eight_cells_while_exact_supervisor_lives(tmp_path):
    identity=state(tmp_path)
    assert not dependency_ready(tmp_path,identity,lambda _:True)
    assert dependency_ready(tmp_path,identity,lambda _:False)


@pytest.mark.parametrize('mode',['changed_identity','failed','partial_exit'])
def test_never_registers_invalid_dependency(tmp_path,mode):
    identity=state(tmp_path,'attention_required' if mode=='failed' else 'awaiting_observer_assessment',7 if mode=='partial_exit' else 8)
    if mode=='changed_identity':identity['process_created']=2
    with pytest.raises(ValueError):dependency_ready(tmp_path,identity,lambda _:False)
