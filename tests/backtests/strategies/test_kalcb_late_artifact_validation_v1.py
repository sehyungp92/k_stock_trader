import ast
import copy
import inspect
import json
from pathlib import Path
import pytest
from backtests.auto.shared.cache_keys import stable_signature
from backtests.strategies.kalcb import late_artifact_validation_v1 as adapter
from backtests.strategies.kalcb import qualified_artifacts as original
from backtests.strategies.kalcb.late_completed_inputs_v1 import file_manifest
from backtests.strategies.kalcb.retention_research_io_v1 import digest

ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_late_entry_20261011/v6')


@pytest.fixture(scope='module')
def artifacts():
    item=json.loads((ROOT/'cases/control/is/input_profile.json').read_text())
    out=ROOT/'cases/control/is/baseline'
    data={name:json.loads((out/name).read_text()) for name in original.REPLAY_FILES if name.endswith('.json')}
    profile={'execution_source':{'source_generation_hash':item['source_generation_hash']},
      'audit_manifest':item['manifest']['path'],'policy':{'name':'interaction_cost_margin',
        'manifest_sha256':item['policy']['sha256'],'directory':str(Path(item['policy']['path']).parent)},
      'initial_equity_krw':100000000,'date_range':{'start':'2025-08-07','end':'2026-03-31','sessions':156},
      'metrics':{'baseline':data['result.json']}}
    return item,out,data,profile


def test_original_account_reconciliation_body_retained():
    old=inspect.getsource(original.validate_case)
    start=old.index('    require(receipt["identity"]["configuration"]["source_dependency_scope"]')
    end=old.index('\n    require(receipt["identity"]["input_manifest"]',start)
    expected=old[:start]+'    validate_scope(receipt, profile)'+old[end:]
    assert ast.dump(ast.parse(expected))==ast.dump(ast.parse(inspect.getsource(adapter.validate_case)))


def test_real_first_hour_control_reconciles_and_matches_parent(artifacts):
    from backtests.strategies.kalcb.population_research_io_v1 import bridge_accounts
    item,out,_,_=artifacts
    assert adapter.validate_native(out,item,'baseline',save=False)['audit_status']=='passed'
    protocol=json.loads((ROOT/'protocol.json').read_text())
    proof=bridge_accounts(out,Path(protocol['profiles']['100']['is']['references']['baseline']).parent)
    assert proof['passed'] and proof['equity_marks']==11995


def test_legacy_control_still_reconciles():
    root=ROOT.parent/'v5';item=json.loads((root/'cases/identity_100/oos/input_profile.json').read_text())
    assert adapter.validate_native(root/'cases/identity_100/oos/baseline',item,'baseline',save=False)['audit_status']=='passed'


@pytest.mark.parametrize('mutation', ['old_scope','short_feature_scope','missing_population','population_count',
    'changed_receipt','bad_fee','early_fill','duplicate_fill','altered_mtm'])
def test_rejects_invalid_source_and_economic_evidence(monkeypatch,artifacts,mutation):
    _,out,source,profile=artifacts;data=copy.deepcopy(source)
    receipt=data['execution_audit.json']['audit_receipt']
    if mutation=='old_scope':receipt['identity']['configuration']['source_dependency_scope']=adapter.OLD_SCOPE
    elif mutation=='short_feature_scope':receipt['identity']['configuration']['source_dependency_scope']['feature_intraday_before']='09:30'
    elif mutation=='missing_population':receipt['qualification'].pop('first_hour_population')
    elif mutation=='population_count':receipt['qualification']['first_hour_population']['considered']+=1
    elif mutation=='changed_receipt':receipt['coverage']['capacity']='unverified'
    elif mutation=='bad_fee':data['fills.json'][0]['metadata']['fee']+=1
    elif mutation=='early_fill':data['fills.json'][0]['metadata']['decision_available_at']='2030-01-01T10:00:00+09:00'
    elif mutation=='duplicate_fill':data['fills.json'].insert(1,copy.deepcopy(data['fills.json'][0]))
    elif mutation=='altered_mtm':data['equity.json'][-1][1]+=1000
    if mutation in ('old_scope','short_feature_scope','missing_population','population_count'):
        receipt.pop('receipt_hash');receipt['receipt_hash']=stable_signature(receipt)
    base=adapter.read
    monkeypatch.setattr(adapter,'read',lambda path:data[Path(path).name] if Path(path).parent==out else base(path))
    with pytest.raises(ValueError):adapter.validate_case(out,profile,'baseline')


def test_changed_or_incomplete_producer_is_not_reused(tmp_path):
    p=tmp_path/'labels.jsonl';p.write_text('original')
    done={'complete':True,'files':{p.name:digest(p)}}
    assert file_manifest(tmp_path,done)
    p.write_text('changed')
    with pytest.raises(ValueError):file_manifest(tmp_path,done)
    with pytest.raises(ValueError):file_manifest(tmp_path,{**done,'complete':False})


def test_successor_keeps_shared_lifecycle_and_never_reruns_producers():
    text=Path('scripts/kalcb_late_entry_study_v7.py').read_text();tree=ast.parse(text)
    assert 'PhaseRunner(' in text and "completed_input_reuse" in text
    assert "for stage in ('prepare'" not in text
    assert not any(isinstance(n,ast.ImportFrom) and n.module and 'kalcb_late_entry_study' in n.module for n in ast.walk(tree))
