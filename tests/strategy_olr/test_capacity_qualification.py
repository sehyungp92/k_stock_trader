import pytest
import json
from copy import deepcopy

from backtests.auto.shared.greedy_optimizer import run_greedy
from backtests.auto.shared.types import Experiment
from backtests.strategies.olr.capacity_qualification import (
    OLRCapacityQualificationPlugin, paired_failures)


CONTROL = dict(return_pct=40., drawdown_pct=10., trades=180,
    execution_validity_pass=True, terminal_inventory={}, terminal_pending=0)


@pytest.mark.parametrize('change,reason', [
    ({'drawdown_pct':10.01}, 'drawdown'),
    ({'trades':170}, 'frequency'),
    ({'return_pct':39.}, 'return'),
    ({'terminal_deferred_intents':1}, 'execution_or_unfinished_inventory'),
    ({'execution_validity_pass':False}, 'execution_or_unfinished_inventory'),
])
def test_qualification_checks_each_material_constraint(change, reason):
    assert reason in paired_failures({**CONTROL, 'return_pct':60., **change}, CONTROL)


def test_return_winner_failing_risk_is_rejected_by_shared_selection():
    plugin = object.__new__(OLRCapacityQualificationPlugin)
    def record(mutations):
        candidate = mutations['recipe'] == 'high_return_bad_risk'
        result = {**CONTROL, 'return_pct':80., 'drawdown_pct':15.} if candidate else CONTROL
        failures = paired_failures(result, CONTROL) if candidate else []
        return {'score':100+result['return_pct'], 'failures':failures, 'metrics':result}
    plugin.evaluate = record
    result = run_greedy([Experiment('high_return_bad_risk', {'recipe':'high_return_bad_risk'})],
        {'recipe':'control'}, plugin.create_evaluate_batch(1, {'recipe':'control'}), max_rounds=1)
    assert result.final_mutations == {'recipe':'control'}
    assert not result.accepted_count


def test_native_audit_adapter_rejects_changed_identity_missing_coverage_and_false_pass(tmp_path, monkeypatch):
    from backtests.strategies.olr import capacity_qualification as module
    manifest={'source_generation_hash':'frozen'}
    manifest_hash=module.signature(manifest)
    manifest['manifest_hash']=manifest_hash
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    study={'configuration':{'source_dependency_scopes':{'OLR':['x']}},
           'source_generation_hash':'frozen','manifest':str(tmp_path/'manifest.json')}
    (tmp_path/'study.json').write_text(json.dumps(study))
    request={'study_hash':'study','recipe':{'olr_capacity':{'capacity_exit_quantile':.5}},
        'arguments':{'study':str(tmp_path/'study.json'),'scenario':'baseline','standalone':'OLR',
            'absolute_caps':False,'no_upward_compounding':False,'full_audit':True}}
    (tmp_path/'request.json').write_text(json.dumps(request))
    identity={k:request[k] for k in ('study_hash','recipe')}
    identity.update({k:request['arguments'][k] for k in ('scenario','standalone','absolute_caps','no_upward_compounding')})
    identity.update(configuration=study['configuration'],qualification_snapshot_hash='snapshot',
        input_manifest={'manifest_hash':manifest_hash,'source_generation_hash':'frozen'})
    receipt={'identity':identity,'audit_version':module.AUDIT_VERSION,
        'metric_version':module.METRIC_VERSION,'execution_version':module.EXECUTION_VERSION,
        'source_generation_hash':'frozen','coverage':{k:'pass' for k in module.REQUIRED_COVERAGE},'errors':[]}
    receipt['receipt_hash']=module.stable_signature(receipt)
    audit={'audit_pass':True,'execution_validity_pass':True,'audit_status':'passed','audit_receipt':receipt}
    monkeypatch.setattr(module,'source_generation_hash',lambda:'frozen')
    assert module.native_audit_valid(audit,tmp_path)
    changed=deepcopy(audit);changed['audit_receipt']['identity']['recipe']={}
    body=changed['audit_receipt'];body.pop('receipt_hash');body['receipt_hash']=module.stable_signature(body)
    assert not module.native_audit_valid(changed,tmp_path)
    changed=deepcopy(audit);changed['audit_receipt']['coverage'].pop(next(iter(module.REQUIRED_COVERAGE)))
    body=changed['audit_receipt'];body.pop('receipt_hash');body['receipt_hash']=module.stable_signature(body)
    assert not module.native_audit_valid(changed,tmp_path)
    assert not module.native_audit_valid({**audit,'audit_pass':False},tmp_path)
