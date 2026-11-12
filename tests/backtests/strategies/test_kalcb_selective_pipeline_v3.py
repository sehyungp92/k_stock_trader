"""Fail-closed adoption of asynchronously published immutable native inputs."""
from pathlib import Path
import pytest
from scripts import kalcb_selective_recovery_v3 as study
from backtests.strategies.kalcb.executed_value import signature


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    out=tmp_path/'consumer';source=tmp_path/'producer';name='expanding_conditional';period='is'
    policy=source/'build/policies'/name/period/'policies_manifest.json'
    study.write(policy,{'test':'policy'})
    study.write(source/'protocol.json',{'test':'producer'})
    config=tmp_path/'canonical.json';study.write(config,{'test':'config'})
    monkeypatch.setattr(study,'CANONICAL',config)
    folder=source/'cases'/name/period
    start,end,_=study.aligned.PERIODS[period]
    study.write(folder/'plan.json',{'start':start,'end':end,'config':str(config),
        'selective_research':{'recipe':name,'period':period,'protocol':study.bound(source/'protocol.json')}})
    manifest={'source_generation_hash':'expected','files':{'research/kalcb_executed_value/policies/policies_manifest.json':study.digest(policy)}}
    study.write(folder/'inputs/manifest.json',{**manifest,'manifest_hash':signature(manifest)})
    (source/'prepare.log').write_text('{"prepared":"expanding_conditional","period":"is"}\n',encoding='utf-8')
    study.write(out/'protocol.json',{'producer':{'root':str(source)},'expected_policies':{name+'/'+period:study.bound(policy)},'expected_sources':{'is':'expected'}})
    return out,source,folder,name,period


def test_published_profile_adopted_without_copying_runtime(prepared):
    out,source,folder,name,period=prepared
    item=study.ready_profile(out,name,period)
    assert item['input_folder']==str(folder)
    assert Path(item['folder'])!=folder
    assert study.ready_profile(out,name,period)==item


def test_published_wrong_recipe_rejected(prepared):
    out,source,folder,name,period=prepared
    plan=study.read(folder/'plan.json');plan['selective_research']['recipe']='wrong'
    study.write(folder/'plan.json',plan)
    with pytest.raises(ValueError,match='recipe'):study.ready_profile(out,name,period)


def test_published_manifest_corruption_rejected(prepared):
    out,source,folder,name,period=prepared
    manifest=study.read(folder/'inputs/manifest.json');manifest['source_generation_hash']='changed'
    study.write(folder/'inputs/manifest.json',manifest)
    with pytest.raises(ValueError,match='manifest'):study.ready_profile(out,name,period)


def test_published_policy_substitution_rejected(prepared):
    out,source,folder,name,period=prepared
    manifest=study.read(folder/'inputs/manifest.json');manifest.pop('manifest_hash')
    manifest['files']['research/kalcb_executed_value/policies/policies_manifest.json']='wrong'
    study.write(folder/'inputs/manifest.json',{**manifest,'manifest_hash':signature(manifest)})
    with pytest.raises(ValueError,match='policy substitution'):study.ready_profile(out,name,period)


def test_future_period_substitution_rejected(prepared):
    out,source,folder,name,period=prepared
    plan=study.read(folder/'plan.json');plan['end']='2026-07-10';study.write(folder/'plan.json',plan)
    with pytest.raises(ValueError,match='period'):study.ready_profile(out,name,period)


def test_wrong_source_with_valid_manifest_hash_rejected(prepared):
    out,source,folder,name,period=prepared
    manifest=study.read(folder/'inputs/manifest.json');manifest.pop('manifest_hash')
    manifest['source_generation_hash']='wrong'
    study.write(folder/'inputs/manifest.json',{**manifest,'manifest_hash':signature(manifest)})
    with pytest.raises(ValueError,match='source generation'):study.ready_profile(out,name,period)


def test_unpublished_bundle_waits_for_healthy_producer(prepared,monkeypatch):
    from types import SimpleNamespace
    import psutil
    out,source,folder,name,period=prepared
    (source/'prepare.log').write_text('',encoding='utf-8')
    p=study.read(out/'protocol.json');p['producer'].update(pid=123,process_created=42.,command=['producer'])
    study.write(out/'protocol.json',p)
    monkeypatch.setattr(psutil,'Process',lambda pid:SimpleNamespace(create_time=lambda:42.,cmdline=lambda:['producer']))
    assert study.ready_profile(out,name,period) is None
    assert not (out/'cases'/name/period/'input_profile.json').exists()


def test_unpublished_bundle_after_producer_exit_stops(prepared,monkeypatch):
    import psutil
    out,source,folder,name,period=prepared
    (source/'prepare.log').write_text('',encoding='utf-8')
    p=study.read(out/'protocol.json');p['producer'].update(pid=123,process_created=42.,command=['producer'])
    study.write(out/'protocol.json',p)
    def missing(pid):raise psutil.NoSuchProcess(pid)
    monkeypatch.setattr(psutil,'Process',missing)
    with pytest.raises(RuntimeError,match='before publishing'):study.ready_profile(out,name,period)
