"""R1 storage equivalence and actual historical qualification parity."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import tracemalloc
import pytest

from backtests.strategies.kalcb import qualification_storage_v1 as storage
from tests.backtests.test_qualification_evidence import evidence_fixture

HISTORICAL = Path('E:/CodexResearch/k_stock_trader/kalcb_reconsideration_20261017/v1/runtime/is/inputs/files')


def test_storage_types_order_duplicates_and_cleanup():
    paths = []

    @storage.audit_storage
    def exercise(fail=False):
        index, groups = storage.DiskEventIndex(), storage.DiskStampGroups()
        paths.append(index.store.path)
        stamp = datetime(2026, 1, 5, 9, 0, tzinfo=timezone(timedelta(hours=9)))
        key = ('005930', stamp.isoformat(), '5m')
        value = (100, 101.5, 99, 100.25, 123456, True)
        assert key not in index and index.get(key) is None
        index[key] = value
        assert key in index and index.get(key) == value
        assert tuple(type(x) for x in index.get(key)) == tuple(type(x) for x in value)
        group = ('005930', stamp.date(), '5m')
        for i in (2, 0, 1):
            groups[group].append(stamp + timedelta(minutes=5*i))
        assert groups.get(group) == [stamp + timedelta(minutes=5*i) for i in (2, 0, 1)]
        assert groups.get(('missing',), ()) == ()
        if fail:
            raise ValueError('expected cleanup path')
    exercise()
    with pytest.raises(ValueError, match='expected cleanup'):
        exercise(True)
    assert all(not p.parent.exists() for p in paths)
    assert storage._CURRENT.get() is None


def test_large_index_does_not_retain_python_rows():
    @storage.audit_storage
    def exercise():
        index, groups = storage.DiskEventIndex(), storage.DiskStampGroups()
        tracemalloc.start()
        try:
            for i in range(50_000):
                key = ('005930', str(i), '5m')
                assert key not in index
                index[key] = (100., 102., 99., 101., i, True)
                groups[('005930', i//79, '5m')].append(i)
            assert index.get(('005930', '49999', '5m'))[-2] == 49999
            assert len(groups.get(('005930', 0, '5m'))) == 79
            _, peak = tracemalloc.get_traced_memory()
            assert peak < 8*1024**2
            assert index.store.path.stat().st_size > 1024**2
        finally:
            tracemalloc.stop()
    exercise()


def historical_verifier(patched):
    source = (HISTORICAL/'backtests/engine/qualification_evidence.py').read_text(encoding='utf-8')
    if patched:
        source = storage.patched_verifier(source)
    namespace = {'__name__': 'backtests.engine.qualification_evidence', '__package__': 'backtests.engine'}
    exec(compile(source, '<historical-c15-audit>', 'exec'), namespace)
    return namespace['verify_qualification']


@pytest.mark.parametrize('corruption', [None, 'events', 'basis', 'daily', 'capacity'])
def test_complete_historical_audit_parity(evidence_fixture, monkeypatch, corruption):
    from backtests.strategies.kalcb import worker
    from backtests.engine import qualification_evidence
    from backtests.core.input_manifest import _VERIFIED_BYTES
    generation, bundle, prefix = evidence_fixture
    if corruption:
        (generation.root/prefix/(corruption+'.json')).write_text('[]', encoding='utf-8')
    monkeypatch.setattr(worker, '_worker_config', dict(capability_level='synthetic',
        input_manifest_path=str(generation.path), symbols=['005930'], start='2026-01-05', end='2026-01-05'))
    monkeypatch.setattr(worker, '_worker_replay_bundle', bundle)
    monkeypatch.setattr(worker, '_worker_hard_rejects', {'min_trades': 1})
    results = []
    for patched in (False, True):
        _VERIFIED_BYTES.clear()
        monkeypatch.setattr(qualification_evidence, 'verify_qualification', historical_verifier(patched))
        result = worker.score_candidate(('r1-parity', {}, {}))
        results.append(result.metrics['audit_receipt'])
        assert bool(result.metrics['audit_pass']) == (corruption is None)
    assert results[0] == results[1]


def test_repair_is_only_storage_and_matrix_is_unchanged():
    from scripts import kalcb_reconsideration_v1 as before, kalcb_reconsideration_v2 as after
    controls = {p+'/'+s: 'control' for p in ('is', 'oos') for s in ('baseline', 'combined')}
    assert before.cases(controls) == after.cases(controls)
    source = (HISTORICAL/'backtests/engine/qualification_evidence.py').read_text(encoding='utf-8')
    modified = storage.patched_verifier(source)
    assert modified.count('@audit_storage') == 1
    with pytest.raises(ValueError):
        storage.patched_verifier(modified)
    repo = Path(__file__).resolve().parents[1]
    for rel in (after.WORKER, after.CONTROLLER):
        assert hashlib.sha256((repo/rel).read_bytes()).digest() == hashlib.sha256((HISTORICAL/rel).read_bytes()).digest()
