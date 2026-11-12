from copy import deepcopy
import pytest

from backtests.auto.shared.cache_keys import stable_signature
from backtests.strategies.kalcb import population_research_io_v1 as io


def fixture_parent(tmp_path):
    parent=tmp_path/'parent';files=parent/'files';files.mkdir(parents=True)
    io.write(parent/'config.json',{'keep':1})
    io.write(parent/'context.json',{'output_root':'previous'})
    plan={'config':str(parent/'config.json'),'audit_configuration':str(parent/'context.json'),
          'input_sha256':{str(parent/'config.json'):io.digest(parent/'config.json'),
                          str(parent/'context.json'):io.digest(parent/'context.json')}}
    io.write(parent/'plan.json',plan)
    policy=files/'research/kalcb_executed_value/policies';policy.mkdir(parents=True)
    io.write(policy/'policies_manifest.json',{'old':True})
    io.write(policy/'obsolete.json',{'old':True})
    io.write(files/'economic.json',{'fixed':True})
    payload={'files':{str(p.relative_to(files)).replace('\\','/'):io.digest(p) for p in files.rglob('*') if p.is_file()},
             'evidence':{},'source_generation_hash':'economic'}
    io.write(parent/'manifest.json',{**payload,'manifest_hash':stable_signature(payload)})
    return {'manifest':io.bound(parent/'manifest.json'),'plan':io.bound(parent/'plan.json'),
            'policy':io.bound(policy/'policies_manifest.json'),'source_generation_hash':'economic'}


def test_freeze_only_replaces_policy_and_documentation(tmp_path,monkeypatch):
    old=fixture_parent(tmp_path);before=io.digest(old['manifest']['path'])
    out=tmp_path/'study';io.write(out/'protocol.json',{'registered':True})
    policy=tmp_path/'new';io.write(policy/'policies_manifest.json',{'new':True})
    io.write(policy/'actual.json',{'decisions':[1,2]})
    monkeypatch.setattr(io,'source_hash',lambda code:'economic')
    item=io.freeze_case(out,'candidate','oos',{'parent':'expanding','config':{}},old,policy,set())
    assert io.digest(old['manifest']['path'])==before
    frozen=io.read(item['manifest']['path'])['files']
    assert 'research/kalcb_executed_value/policies/obsolete.json' not in frozen
    assert frozen['economic.json']==io.read(old['manifest']['path'])['files']['economic.json']
    assert item['policy']['sha256']==io.digest(policy/'policies_manifest.json')
    assert io.read(io.read(item['plan']['path'])['config'])=={'keep':1}


def test_changed_economic_runtime_rejected(tmp_path,monkeypatch):
    old=fixture_parent(tmp_path);out=tmp_path/'study'
    io.write(out/'protocol.json',{});policy=tmp_path/'new';io.write(policy/'policies_manifest.json',{})
    monkeypatch.setattr(io,'source_hash',lambda code:'changed')
    with pytest.raises(ValueError,match='Economic runtime changed'):
        io.freeze_case(out,'candidate','is',{'parent':'expanding','config':{}},old,policy,set())


def test_native_bridge_detects_fill_fee_and_mark_drift(tmp_path):
    fill={'order_id':'o','strategy_id':'s','symbol':'x','side':'BUY','qty':1,'price':100,
          'timestamp':'2026-04-01T09:31:00+09:00','reason':'entry','metadata':{'fee':1}}
    a=tmp_path/'a';b=tmp_path/'b'
    for folder in (a,b):
        io.write(folder/'equity.json',[{'timestamp':'t','equity':100}])
        io.write(folder/'daily.json',[{'day':'2026-04-01','equity':100}])
        io.write(folder/'fills.json',[fill])
    assert io.bridge_accounts(a,b)['passed']
    changed=deepcopy(fill);changed['metadata']['fee']=2;io.write(a/'fills.json',[changed])
    with pytest.raises(ValueError,match='fill bridge drift'):io.bridge_accounts(a,b)
    io.write(a/'fills.json',[fill]);io.write(a/'equity.json',[{'timestamp':'t','equity':101}])
    with pytest.raises(ValueError,match='account bridge drift'):io.bridge_accounts(a,b)
