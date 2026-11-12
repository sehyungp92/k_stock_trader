import json
from pathlib import Path
import pytest
from backtests.auto.shared.cache_keys import stable_signature
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound,digest
from backtests.strategies.kalcb.fixed_allocation_research_io_v1 import freeze_allocation,OVERLAY,WORKER
from backtests.strategies.kalcb.fixed_allocation_plugin_v1 import FixedAllocationPlugin
from backtests.auto.shared.phase_runner import PhaseRunner
from backtests.auto.shared.round_manager import RoundManager


def fixture(tmp_path,monkeypatch):
    out=tmp_path/'next';out.mkdir();write(out/'protocol.json',{'registered':'fixture'})
    old=tmp_path/'old';write(old/'config.json',{'intraday_leverage':2})
    write(old/'audit_configuration.json',{'prior':'unaltered'})
    write(old/'plan.json',{'config':str(old/'config.json'),'audit_configuration':str(old/'audit_configuration.json'),
        'input_sha256':{str(old/'config.json'):digest(old/'config.json'),str(old/'audit_configuration.json'):digest(old/'audit_configuration.json')}})
    files=old/'inputs/files';write(files/'research/kalcb_executed_value/policies/policies_manifest.json',{'files':{}})
    (files/'source.py').write_text('original = 1\n')
    payload={'files':{str(p.relative_to(files)).replace('\\','/'):digest(p) for p in files.rglob('*') if p.is_file()},'source_generation_hash':'old','evidence':{}}
    write(old/'inputs/manifest.json',{**payload,'manifest_hash':stable_signature(payload)})
    profile={'folder':str(old),'source_generation_hash':'old','manifest':bound(old/'inputs/manifest.json'),
             'plan':bound(old/'plan.json'),'policy':bound(files/'research/kalcb_executed_value/policies/policies_manifest.json')}
    programs={}
    for i,relative in enumerate((OVERLAY,WORKER)):
        p=tmp_path/f'new{i}.py';p.write_text(f'new = {i}\n');programs[relative]=bound(p)
    monkeypatch.setattr('backtests.strategies.kalcb.fixed_allocation_research_io_v1.source_hash',lambda p:'new')
    return out,profile,programs


def test_publisher_changes_source_identity_and_preserves_frozen_inputs(tmp_path,monkeypatch):
    out,old,programs=fixture(tmp_path,monkeypatch);before=read(old['manifest']['path'])
    item=freeze_allocation(out,'control','is',old,programs,set(),mode='fixed')
    manifest=read(item['manifest']['path']);plan=read(item['plan']['path'])
    assert manifest['source_generation_hash']=='new'
    assert manifest['files']['source.py']==before['files']['source.py']
    assert item['policy']['sha256']==old['policy']['sha256']
    assert read(plan['config'])==read(read(old['plan']['path'])['config'])
    assert plan['fixed_allocation_reference']['mode']=='fixed'
    assert read(old['manifest']['path'])==before


def test_changed_inherited_bytes_fail_closed_without_republishing_parent(tmp_path,monkeypatch):
    out,old,programs=fixture(tmp_path,monkeypatch)
    (Path(old['folder'])/'inputs/files/source.py').write_text('changed = 2\n')
    with pytest.raises(ValueError,match='Changed inherited'):
        freeze_allocation(out,'control','is',old,programs,set(),mode='fixed')


def test_identity_publication_requires_audited_reference(tmp_path,monkeypatch):
    out,old,programs=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='audited reference'):
        freeze_allocation(out,'control','is',old,programs,set(),mode='identity')


def test_fixed_comparison_uses_shared_phase_and_preserves_all_cases(tmp_path):
    recipes={'control':{'parent':'expanding','scenarios':['baseline','combined']},
             'control_expanding_atr5':{'parent':'expanding_atr5','scenarios':['baseline','combined']},
             'expanding_removal_prior100':{'parent':'expanding','scenarios':['baseline','combined']},
             'expanding_atr5_removal_prior100':{'parent':'expanding_atr5','scenarios':['baseline','combined']}}
    write(tmp_path/'study.json',{'recipes':recipes,'code':str(tmp_path),'source_generation_hash':'fixture',
        'bound_inputs':[],'source_programs':[],'global_IS_reference':{'baseline':73.1303145472,'combined':47.8121465627}})
    calls=[]
    def native(name,period,scenario):
        calls.append((name,period,scenario));folder=tmp_path/'cases'/name/period/scenario
        result={'return_pct':1.,'drawdown_pct':2.,'trades':3,'active_days':2,'audit_status':'passed'}
        write(folder/'result.json',result);write(folder/'independent_account_verification.json',{'passed':True})
        return result
    plugin=FixedAllocationPlugin(tmp_path/'study.json',native)
    manager=RoundManager('stock',plugin.name,base_dir=tmp_path/'framework')
    PhaseRunner(plugin,manager.get_round_dir(1),round_name='Fixed allocation test',round_manager=manager,
        round_num=1,max_rounds=1,min_delta=.00001,max_retries=0,max_diagnostic_retries=0).run_all_phases()
    assert len(calls)==len(set(calls))==16
    assert [c[0] for c in calls[:8]]==['control']*4+['control_expanding_atr5']*4
    ledger=read(tmp_path/'allocation_ledger.json')
    assert len(ledger['trials'])==4 and ledger['no_promotion']


def test_funding_review_counts_real_rejections_and_unique_filled_orders(tmp_path):
    import gzip
    from backtests.strategies.kalcb.fixed_allocation_funding_v1 import funding_review
    write(tmp_path/'independent_account_verification.json',{'passed':True})
    write(tmp_path/'daily.json',[{'equity':100_000_010.}])
    write(tmp_path/'fills.json',[
        {'order_id':'buy','side':'BUY','qty':1,'price':100.,'metadata':{'execution_id':'b','fee':0}},
        {'order_id':'sell','side':'SELL','qty':1,'price':110.,'metadata':{'execution_id':'s','fee':0}}])
    buy={'order_id':'buy','side':'BUY','symbol':'A','metadata':{'decision_available_at':'2026-04-01T09:30:00+09:00'}}
    rejected={**buy,'order_id':'unfunded'}
    events=[{'event':'submit','order':buy},{'event':'submit','order':rejected}]
    events += [{'event':'reject','order':rejected,'reason':'buying_power','timestamp':'2026-04-01T09:30:00+09:00','attempted_qty':100,'cash':20.}]*2
    with gzip.open(tmp_path/'orders.jsonl.gz','wt',encoding='utf-8') as f:
        for row in events:f.write(json.dumps(row)+'\n')
    review=funding_review(tmp_path)
    assert review['physically_filled_buy_orders']==1
    assert review['unique_rejected_buy_orders']==1 and len(review['rejected_buy_events'])==2
    assert review['terminal_cash_krw']==100_000_010.
    assert review['submitted_buy_orders_without_fill']==1


def test_mtm_episode_review_retains_recovered_and_terminal_drawdowns():
    from scripts.kalcb_removal_qualification_assessment_v1 import drawdown_episodes
    result=drawdown_episodes(list(zip(['a','b','c','d','e','f'],[100,110,90,108,111,95])),100)
    assert result[0]['peak_at']=='b' and result[0]['trough_at']=='c'
    assert result[0]['drawdown_pct']==pytest.approx(100*(1-90/110))
    assert result[0]['recovered_at']=='e'
    assert result[1]['trough_at']=='f' and result[1]['recovered_at'] is None
