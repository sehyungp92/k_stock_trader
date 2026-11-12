from threading import Barrier
import pytest
from scripts import olr_structural_controller as controller


def test_shared_failure_does_not_start_queued_cases(tmp_path,monkeypatch):
    rendezvous=Barrier(2);started=[]
    def native_case(root,*,mode):
        started.append(mode)
        if mode in ('first','second'):rendezvous.wait(timeout=5)
        if mode=='first':raise RuntimeError('shared audit contract failed')
        return mode
    monkeypatch.setattr(controller,'case',native_case)
    with pytest.raises(RuntimeError,match='shared audit contract'):
        controller.group(tmp_path,{name:dict(mode=name) for name in ('first','second','third','fourth')})
    assert set(started)=={'first','second'}
    assert controller.read(tmp_path/'controller_status.json')['phase']=='engineering_failure_draining_active_cases'


def test_existing_completion_still_requires_execution_validity(tmp_path,monkeypatch):
    output=tmp_path/'cases/train/control_baseline';output.mkdir(parents=True)
    (output/'completion.json').write_text('{}',encoding='utf-8')
    monkeypatch.setattr(controller,'completed_case',lambda path:dict(valid=False))
    with pytest.raises(RuntimeError,match='qualification failed'):
        controller.case(tmp_path,'train','control')
