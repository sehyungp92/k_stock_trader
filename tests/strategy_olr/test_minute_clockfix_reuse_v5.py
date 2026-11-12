"""Exact source-reuse authentication for a new engine-only generation."""
from pathlib import Path
import hashlib
import pytest
from scripts.olr_minute_exit_clockfix_v5 import verify_source_reuse


def fixture(tmp_path):
    old,new=tmp_path/'old',tmp_path/'new';old.mkdir();new.mkdir()
    files={}
    for name in ('source.bin','parser.py','driver.py'):
        content=name.encode();(old/name).write_bytes(content);(new/name).write_bytes(content)
        files[name]=hashlib.sha256(content).hexdigest()
    proof=dict(all_source_rows_recomputed=True,origins_read={str(old/'source.bin'):files['source.bin']})
    imports=dict(passed=True,native_imports={'parser.py':files['parser.py']})
    return old,new,dict(files=dict(files)),dict(files=dict(files)),proof,imports,{'driver.py'}


def test_exact_source_and_parser_reuse(tmp_path):
    args=fixture(tmp_path);verify_source_reuse(*args)


@pytest.mark.parametrize('tamper',['new_source','old_source','new_parser','manifest','foreign_origin','hidden_extra'])
def test_source_reuse_rejects_changed_or_unbound_evidence(tmp_path,tamper):
    args=list(fixture(tmp_path));old,new,om,nm,proof,imports,allowed=args
    if tamper=='new_source':(new/'source.bin').write_bytes(b'changed')
    elif tamper=='old_source':(old/'source.bin').write_bytes(b'changed')
    elif tamper=='new_parser':(new/'parser.py').write_bytes(b'changed')
    elif tamper=='manifest':nm['files']['source.bin']='0'*64
    elif tamper=='foreign_origin':proof['origins_read'][str(tmp_path/'foreign.bin')]='0'*64
    else:nm['files']['extra.bin']='0'*64
    with pytest.raises((AssertionError,ValueError)):verify_source_reuse(*args)


def test_imported_parser_cannot_be_exempted_as_engine_change(tmp_path):
    args=list(fixture(tmp_path));args[-1].add('parser.py')
    with pytest.raises(AssertionError):verify_source_reuse(*args)


def test_explicit_manifest_alias_resolves_historical_external_label(tmp_path):
    args=list(fixture(tmp_path));origin=str(tmp_path/'original_archive.parquet')
    digest=args[2]['files']['source.bin'];args[4]['origins_read']={origin:digest}
    verify_source_reuse(*args,aliases={origin:'source.bin'})


@pytest.mark.parametrize('alias',['../foreign.bin','missing.bin','driver.py'])
def test_alias_cannot_escape_or_bypass_a_bound_origin_hash(tmp_path,alias):
    args=list(fixture(tmp_path));origin=str(tmp_path/'original_archive.parquet')
    args[4]['origins_read']={origin:args[2]['files']['source.bin']}
    with pytest.raises((AssertionError,ValueError,KeyError)):
        verify_source_reuse(*args,aliases={origin:alias})


def test_alias_cannot_substitute_a_different_bound_file(tmp_path):
    args=list(fixture(tmp_path));origin=str(tmp_path/'original_archive.parquet')
    args[4]['origins_read']={origin:args[2]['files']['source.bin']}
    with pytest.raises(AssertionError):verify_source_reuse(*args,aliases={origin:'parser.py'})
