import json
from types import SimpleNamespace
import pytest
from scripts.kalcb_late_entry_launcher_v1 import same_process,ready


def test_pid_reuse_and_command_change_do_not_match():
    i={'process_created':3.,'command':['python','preflight']}
    assert same_process(i,SimpleNamespace(create_time=lambda:3.,cmdline=lambda:['python','preflight']))
    assert not same_process(i,SimpleNamespace(create_time=lambda:4.,cmdline=lambda:['python','preflight']))
    assert not same_process(i,SimpleNamespace(create_time=lambda:3.,cmdline=lambda:['python','run']))


def test_all_four_bridges_and_filled_smoke_are_required(tmp_path):
    assert not ready(tmp_path)
    p=tmp_path/'preflight/completion.json';p.parent.mkdir()
    value={'passed':True,'native_identity_bridges':3,'fresh_filled_labels':{'baseline':{'filled':1},'combined':{'filled':1}}}
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError):ready(tmp_path)
    value['native_identity_bridges']=4;p.write_text(json.dumps(value));assert ready(tmp_path)
    value['fresh_filled_labels']['combined']['filled']=0;p.write_text(json.dumps(value))
    with pytest.raises(ValueError):ready(tmp_path)
