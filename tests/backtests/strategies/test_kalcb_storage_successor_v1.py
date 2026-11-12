from pathlib import Path
from types import SimpleNamespace
import ast
import pytest
from backtests.strategies.kalcb.retention_research_io_v1 import digest,write,read
from backtests.strategies.kalcb.late_entry_io_v5 import cached_link
from backtests.strategies.kalcb.incremental_storage_reuse_v1 import seed_cache


def test_external_cache_links_same_volume_bytes_and_preserves_hash(tmp_path):
    source=tmp_path/'source';source.write_bytes(b'causal immutable input'*100)
    h=digest(source);target=tmp_path/'runtime/data/input';cache=tmp_path/'cache'
    cached_link(source,target,h,cache)
    assert digest(target)==h
    assert target.stat().st_ino==source.stat().st_ino
    second=tmp_path/'runtime2/input';cached_link(source,second,h,cache)
    assert second.stat().st_ino==target.stat().st_ino


def test_wrong_source_hash_cannot_publish_an_authenticated_blob(tmp_path):
    source=tmp_path/'source';source.write_bytes(b'wrong')
    with pytest.raises(ValueError,match='Changed source'):
        cached_link(source,tmp_path/'target','a'*64,tmp_path/'cache')
    assert not (tmp_path/'target').exists()
    assert (tmp_path/'cache/aa'/('a'*64+'.partial')).exists()


def test_corrupt_cache_cannot_be_reused(tmp_path):
    source=tmp_path/'source';source.write_bytes(b'original');h=digest(source)
    blob=tmp_path/'cache'/h[:2]/h;blob.parent.mkdir(parents=True);blob.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Corrupt'):
        cached_link(source,tmp_path/'target',h,tmp_path/'cache')
    assert not (tmp_path/'target').exists()


def test_failed_partial_cache_is_never_seeded(tmp_path):
    source=tmp_path/'source';blob=source/'aa'/('a'*64+'.partial');blob.parent.mkdir(parents=True);blob.write_bytes(b'incomplete')
    with pytest.raises(ValueError,match='Partial'):
        seed_cache(source,tmp_path/'target')
    assert blob.read_bytes()==b'incomplete'


def test_native_guard_uses_argument_identity_not_shell_script_substrings(monkeypatch):
    from scripts.kalcb_late_incremental_study_v2 import native_processes
    commands=[['python','-m','scripts.kalcb_executed_value_replay'],
        ['python','E:/frozen/scripts/kalcb_exit_allocation_worker_v1.py'],
        ['pwsh','-Command',"print('scripts.kalcb_executed_value_replay')"],
        ['python','scripts/kalcb_late_incremental_study_v2.py','run']]
    processes=[SimpleNamespace(pid=999990+i,info=dict(cmdline=x,create_time=1)) for i,x in enumerate(commands)]
    monkeypatch.setattr('psutil.process_iter',lambda _:processes)
    assert [x['pid'] for x in native_processes()]==[999990,999991]


def test_successor_source_does_not_import_an_orchestration_runner():
    for file in ('scripts/kalcb_late_incremental_study_v2.py','scripts/kalcb_exit_qualification_study_v2.py'):
        source=Path(file).read_text(encoding='utf-8')
        imports=[n.module for n in ast.walk(ast.parse(source)) if isinstance(n,ast.ImportFrom)]
        assert not any(m and m.startswith('scripts.') for m in imports)
        assert 'backtests.auto.shared.phase_runner' in imports


def test_qualification_economic_design_is_unchanged():
    from scripts.kalcb_exit_qualification_study_v1 import recipes as old
    from scripts.kalcb_exit_qualification_study_v2 import recipes as new
    assert all(old(mode)==new(mode) for mode in ('compounding','identity','fixed'))
    assert sum(2*len(r['scenarios']) for m in ('compounding','identity','fixed') for r in new(m).values())==40


def test_allocation_publication_keeps_scenario_bound_identity_with_external_cache(tmp_path,monkeypatch):
    from tests.backtests.strategies.test_kalcb_fixed_allocation_packaging_v1 import fixture
    from backtests.strategies.kalcb.exit_allocation_io_v2 import freeze_allocation,WORKER
    from backtests.strategies.kalcb.fixed_allocation_research_io_v1 import WORKER as OLD_WORKER
    out,old,programs=fixture(tmp_path,monkeypatch);programs[WORKER]=programs.pop(OLD_WORKER)
    write(out/'protocol.json',dict(runtime_root=str(tmp_path/'external_runtime')))
    monkeypatch.setattr('backtests.strategies.kalcb.exit_allocation_io_v2.source_hash',lambda p:'new')
    for scenario in ('baseline','combined'):
        ref=tmp_path/scenario
        for f in ('result.json','daily.json','equity.json','fills.json'):write(ref/f,{'scenario':scenario})
        write(ref/'independent_account_verification.json',{'passed':True})
    first=freeze_allocation(out,'control_baseline','is',old,programs,set(),mode='identity',bridge=tmp_path/'baseline')
    assert first['policy']['sha256']==old['policy']['sha256']
    with pytest.raises(ValueError,match='scenario bridge'):
        freeze_allocation(out,'control_baseline','is',old,programs,set(),mode='identity',bridge=tmp_path/'combined')
