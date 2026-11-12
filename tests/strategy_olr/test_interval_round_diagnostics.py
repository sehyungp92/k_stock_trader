import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backtests.strategies.olr import interval_round_diagnostics as report


def fixture(tmp_path):
    from tests.strategy_olr.test_native_round_diagnostics import fixture_case
    result, audit, window = fixture_case(tmp_path)
    with gzip.open(tmp_path / 'journal.jsonl.gz', 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    # Completed marks become available one minute later; do not move fills.
    with gzip.open(tmp_path / 'minute_events.jsonl.gz', 'wt') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    nav = dict(final_equity=1048.69, all_event_drawdown_pct=100 * 1.21 / 1049.9, points=6,
               events={'opening_marks': 2, 'closing_marks': 2, 'fill': 2})
    (tmp_path / 'minute_nav_evidence.json').write_text(json.dumps(nav))
    return window


def test_full_event_reconstruction_is_observational(tmp_path):
    window = fixture(tmp_path)
    before = {p.name: report.digest(p) for p in tmp_path.iterdir()}
    value = report.reconstruct(tmp_path, 1000, window)
    assert value['metrics']['official_mtm_net_return_pct'] == pytest.approx(.04869)
    # The sell cost follows the opening mark peak; close-only DD would miss it.
    assert value['metrics']['max_drawdown_pct'] == pytest.approx(1.21 / 1049.9)
    assert value['metrics']['maximum_currency_decline'] == pytest.approx(1.21)
    assert value['metrics']['entry_lifecycle_count'] == 1
    assert before == {p.name: report.digest(p) for p in tmp_path.iterdir()}


@pytest.mark.parametrize('fault', ['fee', 'cash', 'owner', 'inventory', 'clock', 'nav', 'daily', 'duplicate'])
def test_corrupt_or_optimistic_exports_fail(tmp_path, fault):
    window = fixture(tmp_path)
    if fault in ('fee', 'cash', 'owner', 'inventory', 'clock'):
        p = tmp_path / 'minute_events.jsonl.gz'
        with gzip.open(p, 'rt') as stream:
            rows = [json.loads(line) for line in stream]
        if fault == 'fee': rows[1]['fill']['metadata']['fee'] = 0
        if fault == 'cash': rows[1]['cash_after'] += 1
        if fault == 'owner': rows[1]['fill']['strategy_id'] = 'OTHER'
        if fault == 'inventory': rows[4]['fill']['qty'] = 6
        if fault == 'clock': rows[1]['fill']['timestamp'] = '2026-01-02T14:00:00+09:00'
        with gzip.open(p, 'wt') as stream:
            for row in rows: stream.write(json.dumps(row) + '\n')
    elif fault == 'nav':
        p = tmp_path / 'minute_nav_evidence.json'
        value = report.read(p); value['all_event_drawdown_pct'] = 0
        p.write_text(json.dumps(value))
    elif fault == 'daily':
        p = tmp_path / 'daily.json'; p.write_text(json.dumps(report.read(p)[1:]))
    else:
        p = tmp_path / 'trades.json'; p.write_text(json.dumps(report.read(p) * 2))
    with pytest.raises(ValueError):
        report.reconstruct(tmp_path, 1000, window)


def test_failed_real_worker_blocks_shared_final_publication(tmp_path):
    from backtests.auto.shared.phase_runner import PhaseRunner
    from backtests.auto.shared.phase_state import PhaseState
    from backtests.auto.shared.round_manager import RoundManager
    # No mocked passing gate: the real validation subprocess rejects this
    # attractive but unbound result before any numbered artifact is published.
    plan_path = tmp_path / 'plan.json'
    root = str(Path(__file__).resolve().parents[2])
    plan = dict(schema=report.SCHEMA, saved_config={}, saved_mutations={},
                cases={k: dict(path=str(tmp_path), execution_root=root,
                    source_generation_hash='forged-high-return') for k in report.CASE_NAMES})
    plan_path.write_text(json.dumps(plan))
    cfg = dict(initial_mutations={}, native_case_plan_sha256=report.digest(plan_path),
               promotion_status=report.STATUS, independent_oos_pass=False, live_promotion=False,
               diagnostic_scope={}, diagnostic_limitations=[], date_range={})
    manager = RoundManager('stock', 'olr', base_dir=tmp_path / 'output')
    directory = manager.get_round_dir(2)
    plugin = report.IntervalDiagnosticsPlugin(cfg, plan_path, output_dir=directory)
    runner = PhaseRunner(plugin, directory, round_manager=manager, round_num=2)
    with pytest.raises(ValueError, match='Native validation failed'):
        runner.run_end_of_round(PhaseState(cumulative_mutations={}))
    assert not manager.manifest_path.exists()
    assert not (directory / 'optimized_config.json').exists()


def test_saved_policy_and_complete_matrix_are_mandatory(tmp_path):
    plan = dict(schema=report.SCHEMA, cases={}, saved_config={}, saved_mutations={})
    path = tmp_path / 'plan.json'; path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='Incomplete'):
        report.IntervalDiagnosticsPlugin({}, path, output_dir=tmp_path)


def test_generic_optimization_cannot_silently_reinterpret_native_round():
    from backtests.strategies.olr.plugin import OLROptimizationPlugin
    with pytest.raises(ValueError, match='frozen native execution feed'):
        OLROptimizationPlugin({'native_case_plan_sha256': 'bound-plan', 'capability_level': 'synthetic'})


@pytest.mark.parametrize('fault', [None, 'clock', 'structural', 'source', 'coverage', 'pass', 'signature'])
def test_interval_receipt_requires_exact_clock_source_coverage_and_signature(fault):
    from copy import deepcopy
    from backtests.auto.shared.cache_keys import stable_signature
    from backtests.engine.contracts import AUDIT_VERSION, METRIC_VERSION, EXECUTION_VERSION
    from backtests.engine.integrity import REQUIRED_COVERAGE
    study = dict(configuration={'source_dependency_scopes': {'OLR': {'scope': 'causal'}, 'OTHER': {}}},
                 minute_source_definition='minute.json', structural_settings='control.json', source_generation_hash='source')
    request = dict(study_hash='study', recipe={'name': 'retained'},
                   arguments=dict(scenario='baseline', no_upward_compounding=False, full_audit=True))
    configuration = deepcopy(study['configuration'])
    configuration['source_dependency_scopes'].pop('OTHER')
    configuration.update(minute_clock=dict(version='olr-minute-clock-v5', source_definition='minute.json', fast_exit=False),
                         structural_decision_v1={'settings_path': 'control.json'})
    receipt = dict(identity=dict(study_hash='study', recipe=request['recipe'], scenario='baseline', standalone='OLR',
        absolute_caps=False, no_upward_compounding=False, configuration=configuration,
        input_manifest={'manifest_hash': 'manifest', 'source_generation_hash': 'source'}, qualification_snapshot_hash='snapshot'),
        audit_version=AUDIT_VERSION, metric_version=METRIC_VERSION, execution_version=EXECUTION_VERSION,
        source_generation_hash='source', errors=[], coverage={k: 'pass' for k in REQUIRED_COVERAGE})
    audit = dict(audit_pass=True, execution_validity_pass=True, audit_status='passed', audit_receipt=receipt)
    if fault == 'clock': configuration['minute_clock']['fast_exit'] = True
    if fault == 'structural': configuration.pop('structural_decision_v1')
    if fault == 'source': receipt['source_generation_hash'] = 'other'
    if fault == 'coverage': receipt['coverage'].pop(next(iter(REQUIRED_COVERAGE)))
    if fault == 'pass': audit['audit_pass'] = False
    receipt['receipt_hash'] = stable_signature(receipt)
    if fault == 'signature': receipt['receipt_hash'] = 'forged'
    if fault:
        with pytest.raises(ValueError): report.require_native_receipt(audit, request, study, 'manifest')
    else:
        report.require_native_receipt(audit, request, study, 'manifest')
