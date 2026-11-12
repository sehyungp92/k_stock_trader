import ast
from pathlib import Path
import pytest
from backtests.strategies.kalcb.late_signal_reuse_v1 import seed_verified_cache,require_closed
from backtests.strategies.kalcb.retention_research_io_v1 import digest


def test_cache_hashes_before_reuse_and_preserves_source(tmp_path):
    original=tmp_path/'original';original.write_bytes(b'frozen')
    sha=digest(original);source=tmp_path/'cache';blob=source/sha[:2]/sha
    blob.parent.mkdir(parents=True);blob.write_bytes(original.read_bytes())
    target=tmp_path/'successor';receipt=seed_verified_cache(source,target)
    assert receipt['verified_blobs']==1 and digest(target/sha[:2]/sha)==sha
    assert blob.read_bytes()==b'frozen'
    with pytest.raises(ValueError):seed_verified_cache(source,target)


def test_corrupt_blob_is_not_reused(tmp_path):
    source=tmp_path/'cache';blob=source/'00'/('0'*64);blob.parent.mkdir(parents=True);blob.write_bytes(b'bad')
    with pytest.raises(ValueError):seed_verified_cache(source,tmp_path/'target')


def test_active_process_cannot_be_reused():
    import psutil
    p=psutil.Process()
    with pytest.raises(ValueError):require_closed({'pid':p.pid,'process_created':p.create_time()})


def test_consumer_uses_shared_framework_without_producer_or_old_runner():
    source=Path('scripts/kalcb_late_signal_study_v3.py').read_text();tree=ast.parse(source)
    assert 'PhaseRunner(' in source and "'fresh_native_cases':31" in source
    assert "'produce',code,state" not in source and 'late_signal_producer' not in source
    assert 'late_artifact_validation_v2 import validate_native' in source
    assert 'All eight unchanged parent bridges must pass first' in source
    assert not any(isinstance(n,ast.ImportFrom) and n.module and 'scripts.kalcb_' in n.module for n in ast.walk(tree))
