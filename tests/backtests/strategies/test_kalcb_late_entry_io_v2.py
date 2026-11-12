from pathlib import Path
import pytest
from backtests.strategies.kalcb.late_entry_io_v2 import cached_link
from backtests.strategies.kalcb.retention_research_io_v1 import digest


def test_content_cache_reuses_exact_bytes_and_detects_corruption(tmp_path):
    source=tmp_path/'source';source.write_bytes(b'frozen market observation')
    sha=digest(source);cache=tmp_path/'cache';first=tmp_path/'first'
    cached_link(source,first,sha,cache);assert first.read_bytes()==source.read_bytes()
    # Authentication is against content, not a mutable original path.
    source.write_bytes(b'changed source');second=tmp_path/'second'
    cached_link(source,second,sha,cache);assert digest(second)==sha
    (cache/sha[:2]/sha).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Corrupt'):cached_link(source,tmp_path/'third',sha,cache)


def test_wrong_source_cannot_enter_the_content_cache(tmp_path):
    p=tmp_path/'source';p.write_bytes(b'x')
    with pytest.raises(ValueError,match='Changed source'):cached_link(p,tmp_path/'target','0'*64,tmp_path/'cache')
    assert not (tmp_path/'target').exists()


def test_frozen_dependency_is_explicitly_registered():
    from scripts.kalcb_late_entry_study_v3 import PROGRAMS
    assert 'backtests/strategies/kalcb/retention_research_io_v1.py' in PROGRAMS
    assert 'backtests/strategies/kalcb/selective_recovery_plugin_v1.py' in PROGRAMS
    assert 'scripts/kalcb_late_policy_builder_v3.py' in PROGRAMS
