import json
import pytest

from backtests.strategies.kalcb.oos_validation import KALCBOOSValidationPlugin


def test_oos_cannot_change_profile_or_window():
    plugin=object.__new__(KALCBOOSValidationPlugin)
    plugin.initial_mutations={'profile':'locked_round1','window':'20260601_20260710'}
    assert plugin.canonicalize_mutations({})==plugin.initial_mutations
    for changes in ({'window':'best_subperiod'},{'profile':'better_runner_up'},{'size':.5}):
        with pytest.raises(ValueError,match='cannot change'):
            plugin.canonicalize_mutations(changes)


@pytest.mark.parametrize('failure',['failed_audit','changed_source'])
def test_profitable_cached_oos_failure_cannot_be_accepted(tmp_path,failure):
    plugin=object.__new__(KALCBOOSValidationPlugin)
    plugin.root=tmp_path
    plugin.source_fingerprint='locked_source'
    p=tmp_path/'cases/baseline';p.mkdir(parents=True)
    (p/'completion.json').write_text(json.dumps({'source_generation_hash':
        'different_source' if failure=='changed_source' else 'locked_source'}))
    (p/'result.json').write_text(json.dumps({'return_pct':999,
        'execution_validity_pass':True,'audit_status':'failed' if failure=='failed_audit' else 'passed'}))
    with pytest.raises(ValueError,match='Stale completed|Failed source audit'):
        plugin._run_case('baseline')


def test_oos_context_cannot_silently_reuse_development_dates(tmp_path):
    from scripts.kalcb_executed_value_replay import audit_configuration
    from backtests.strategies.kalcb.qualified_artifacts import digest
    path=tmp_path/'audit_context.json'
    context={'start':'2026-06-01','end':'2026-07-10','data_snapshot_end':'2026-07-10'}
    path.write_text(json.dumps(context))
    plan={'audit_configuration':str(path),'historical_reference':'must_not_be_read',
          'input_sha256':{str(path):digest(path)}}
    assert audit_configuration(plan)==context
    path.write_text('{"end":"2026-05-12"}')
    with pytest.raises(ValueError,match='Unbound or changed'):
        audit_configuration(plan)
