from copy import deepcopy
from datetime import date, datetime, timedelta
from pathlib import Path
import json
import pytest
from strategy_common.clock import KST
from backtests.strategies.kalcb.late_source_audit_v2 import verify_rows


def fixture():
    day=date(2026,4,1);start=datetime(2026,4,1,9,tzinfo=KST)
    original=[{'trade_date':str(day),'symbol':s,'sector':'S'} for s in ('A','B')]
    indexed={}
    for s in ('A','B'):
        for i in range(12):
            if s=='B' and i==11:continue
            indexed[(s,(start+timedelta(minutes=5*i)).isoformat(),'5m')]=(100,102,99,101,1000,True)
    bars=[{'symbol':'A','timestamp':(start+timedelta(minutes=5*i)).isoformat(),
           'open':100,'high':102,'low':99,'close':101,'volume':1000} for i in range(12)]
    features=[{**original[0],'feature_available_at':(start+timedelta(minutes=60)).isoformat(),
        'opening_bars':bars,'features':{'checkpoint_sector_return':.01,'checkpoint_sector_breadth':1.}}]
    snapshots=[{'trade_date':str(day),'candidates':[{'symbol':'A','metadata':{'entry_checkpoint_at':features[0]['feature_available_at']}}]}]
    return original,features,indexed,{day},snapshots


def test_independently_excludes_missing_hour_but_not_parent_prefix():
    result=verify_rows(*fixture())
    assert result['complete_consumed_prefixes']==1
    assert result['incomplete_excluded']==[('B','2026-04-01')]


@pytest.mark.parametrize('mutation',['included_missing','bad_sector','bad_price','missing_valid','bad_candidate','future_clock'])
def test_rejects_unsupported_first_hour_consumption(mutation):
    original,features,indexed,days,snapshots=fixture()
    if mutation=='included_missing':features.append({**deepcopy(features[0]),'symbol':'B'})
    elif mutation=='bad_sector':features[0]['features']['checkpoint_sector_return']=.02
    elif mutation=='bad_price':features[0]['opening_bars'][-1]['close']=102
    elif mutation=='missing_valid':features=[]
    elif mutation=='bad_candidate':snapshots[0]['candidates'][0]['symbol']='B'
    elif mutation=='future_clock':features[0]['feature_available_at']='2026-04-01T10:05:00+09:00'
    with pytest.raises(ValueError):verify_rows(original,features,indexed,days,snapshots)


def test_successor_preserves_old_scope_and_adds_independent_source_check(tmp_path):
    from backtests.strategies.kalcb.late_entry_io_v4 import make_overrides
    profile=Path('E:/CodexResearch/k_stock_trader/kalcb_selective_recovery_20261010/v8_population_reports/artifacts/profiles/expanding_removal_prior100/is/research_profile.json')
    code=Path(json.loads(profile.read_text())['execution_source']['code'])
    files=make_overrides(code,tmp_path/'overrides',{})
    for item in files.values():compile(Path(item['path']).read_text(),item['path'],'exec')
    source=Path(files['backtests/engine/qualification_evidence.py']['path']).read_text()
    assert 'verify_first_hour(generation, indexed, required, broker)' in source
    assert "expected=('09:45','15:20')" in source
    assert 'coverage["source_completeness"] = "pass"' not in source[source.index('if scope.get("version") == "causal-timed-candidates60-v2"'):source.index('details["required_sessions"]')]


def test_continuous_source_boundary_excludes_duplicate_forward_halo(monkeypatch):
    from types import SimpleNamespace
    from backtests.strategies.kalcb import late_source_audit_v2 as module
    original,features,indexed,days,snapshots=fixture()
    halo={**original[0],'trade_date':'2026-03-31'}
    data={'history':[halo],'forward':[halo]+original,'features':features}
    monkeypatch.setattr(module,'verified_rows',lambda generation,path:iter(data[path]))
    binding={'version':'complete-first-hour-population-v1','original_populations':['history','forward'],
             'features':'features','forward_from':'2026-04-01'}
    generation=SimpleNamespace(payload={'evidence':{'first_hour_feature_sources':binding}})
    result=module.verify_first_hour(generation,indexed,{('A',next(iter(days)),'5m')},SimpleNamespace(qualification_snapshots=snapshots))
    assert result['considered']==2
