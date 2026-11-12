import hashlib,json,os
from pathlib import Path
import pytest
from backtests.strategies.kalcb.exact_source_reader_v1 import ExactSourceReader,DIRECTORIES


def original(root):
    paths=[p for d in DIRECTORIES for p in (root/d).rglob('*.py')]
    paths.extend((root/'kis_core/data').glob('*.yaml'))
    rows=[{'path':p.relative_to(root).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(paths)]
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def fixture(root):
    for name,text in [('backtests/a.py','a=1'),('scripts/Empty.py',''),('kis_core/data/session.yaml','A: 1'),('strategy_kalcb/core/Z.py','pass')]:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)


def test_all_byte_digest_matches_original_algorithm_including_empty_files(tmp_path):
    fixture(tmp_path)
    with ExactSourceReader(tmp_path,original(tmp_path)) as reader:
        assert reader()==original(tmp_path)
        assert reader()==original(tmp_path)
        assert reader.calls==3


@pytest.mark.parametrize('change',['same_size_restored_mtime','size','new_file','new_yaml','rename'])
def test_every_call_detects_changed_bytes_or_membership(tmp_path,change):
    fixture(tmp_path)
    with ExactSourceReader(tmp_path,original(tmp_path)) as reader:
        p=tmp_path/'backtests/a.py';stat=p.stat()
        if change=='same_size_restored_mtime':
            try:
                with p.open('r+b') as f:f.write(b'a=2');f.flush();os.fsync(f.fileno())
            except PermissionError:
                assert reader()==original(tmp_path);return
            os.utime(p,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        elif change=='size':
            try:
                with p.open('ab') as f:f.write(b'\n')
            except PermissionError:
                assert reader()==original(tmp_path);return
        elif change=='new_file':(tmp_path/'scripts/new.py').write_text('pass')
        elif change=='new_yaml':(tmp_path/'kis_core/data/new.yaml').write_text('A: 2')
        elif change=='rename':
            # Windows prevents replacement of mapped open files: blocking the
            # mutation is also fail-closed. On other systems membership catches it.
            try:p.rename(tmp_path/'backtests/b.py')
            except PermissionError:return
        with pytest.raises(ValueError):reader()


def test_source_handles_are_released_after_reader_closes(tmp_path):
    fixture(tmp_path)
    with ExactSourceReader(tmp_path,original(tmp_path)) as reader:assert reader()==original(tmp_path)
    (tmp_path/'backtests/a.py').write_text('a=2')
    assert (tmp_path/'backtests/a.py').read_text()=='a=2'
