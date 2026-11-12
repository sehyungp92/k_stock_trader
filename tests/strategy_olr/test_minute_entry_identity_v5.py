"""An exit-only flag cannot change entry identity or suppress feature differences."""
import copy
import pytest
from scripts.olr_minute_exit_snapshots_v5 import entry_request
from scripts.olr_minute_exit_entry_identity_v5 import normalize_research_identity


def request(flag):
    m={'olr.research.minute_exit':flag,'olr.afternoon.min_score':4,'olr.risk.risk_per_trade_pct':.01}
    return {'config':{'initial_mutations':copy.deepcopy(m),'initial_equity':100000000},'mutations':m}


def rows():
    return [{'metadata':{'research_config_hash':'new','candidate_config_hash':'selection'},
             'candidates':[{'score':4,'quantity':10,'metadata':{'research_config_hash':'new','rejection':'unchanged'}}]}]


def test_exit_flag_preserves_exact_entry_request_and_original():
    treatment=request(True);saved=copy.deepcopy(treatment)
    assert entry_request(treatment)==entry_request(request(False))
    assert treatment==saved


@pytest.mark.parametrize('key,value',[('olr.afternoon.min_score',5),('olr.risk.risk_per_trade_pct',.02)])
def test_real_selection_and_size_changes_remain_visible(key,value):
    treatment=request(True);treatment['mutations'][key]=value
    assert entry_request(treatment)!=entry_request(request(False))


@pytest.mark.parametrize('target',['snapshot','candidate'])
def test_unexplained_config_hash_rejected(target):
    value=rows();record=value[0] if target=='snapshot' else value[0]['candidates'][0]
    record['metadata']['research_config_hash']='forged'
    with pytest.raises(ValueError):normalize_research_identity(value,'new','old')


def test_comparison_projection_preserves_raw_records():
    value=rows();saved=copy.deepcopy(value);result=normalize_research_identity(value,'new','old')
    assert value==saved
    assert result[0]['metadata']['research_config_hash']=='old'
    assert result[0]['candidates'][0]['metadata']['research_config_hash']=='old'
    assert result[0]['metadata']['candidate_config_hash']=='selection'


@pytest.mark.parametrize('key,value',[('score',5),('quantity',11),('rejection','changed')])
def test_projection_never_masks_real_entry_fields(key,value):
    baseline=rows();changed=rows()
    record=changed[0]['candidates'][0]
    if key=='rejection':record['metadata'][key]=value
    else:record[key]=value
    assert normalize_research_identity(baseline,'new','old')!=normalize_research_identity(changed,'new','old')


def test_resumed_parity_keeps_original_legacy_target(tmp_path):
    from scripts.olr_minute_exit_preflight_v5 import legacy_snapshot_path,n
    old=tmp_path/'legacy.json';old.write_text('original retained entries')
    new=tmp_path/'new.json';new.write_text('rebuilt entries')
    identity={'old_snapshot_path':str(old),'old_snapshot_sha256':n.sha(old)}
    manifest={'files':{'legacy.json':n.sha(old),'new.json':n.sha(new)}}
    assert legacy_snapshot_path(identity,tmp_path,manifest)==old
    old.write_text('changed retained entries')
    with pytest.raises(ValueError):legacy_snapshot_path(identity,tmp_path,manifest)


def test_legacy_target_cannot_escape_manifest(tmp_path):
    from scripts.olr_minute_exit_preflight_v5 import legacy_snapshot_path,n
    path=tmp_path/'unbound.json';path.write_text('unbound')
    identity={'old_snapshot_path':str(path),'old_snapshot_sha256':n.sha(path)}
    with pytest.raises(ValueError):legacy_snapshot_path(identity,tmp_path,{'files':{}})
