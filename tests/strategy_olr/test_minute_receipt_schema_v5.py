"""Receipt path repair preserves native failures and does not accept fallback labels."""
import copy
from pathlib import Path
import pytest
from scripts import olr_minute_exit_receipt_resume_v5 as r


def test_exact_single_path_change_and_no_other_semantic_change():
    original=(r.t.REPO/'scripts/olr_minute_exit_qualify_v5.py').read_text(encoding='utf-8')
    fixed=r.corrected_source(original)
    assert fixed.replace(r.NEW,r.OLD)==original
    assert len(fixed.splitlines())==len(original.splitlines())
    for invalid in (original+r.OLD,original.replace(r.OLD,'clock=other'),fixed):
        with pytest.raises(ValueError):r.corrected_source(invalid)


@pytest.fixture
def reader(tmp_path,monkeypatch):
    source=r.corrected_source((r.t.REPO/'scripts/olr_minute_exit_qualify_v5.py').read_text(encoding='utf-8'))
    path=tmp_path/'runtime.py';path.write_text(source,encoding='utf-8');q=r.load_module(path)
    monkeypatch.setattr(q,'_audit',lambda *a,**kw:{})
    monkeypatch.setattr(q.b,'output',lambda *a,**kw:tmp_path)
    return q


@pytest.mark.parametrize('mode',['missing_nested','failed_status','errors','changed_thresholds','wrong_policy'])
def test_reader_rejects_missing_or_failed_nested_audit(reader,monkeypatch,mode):
    q=reader;clock=dict(status='pass',errors=[],unchanged_thresholds=True,control_or_treatment='clock_control')
    native=dict(audit_receipt={'minute_clock_audit':copy.deepcopy(clock)},minute_clock_audit=clock)
    if mode=='missing_nested':native.pop('audit_receipt')
    elif mode=='failed_status':native['audit_receipt']['minute_clock_audit']['status']='failed'
    elif mode=='errors':native['audit_receipt']['minute_clock_audit']['errors']=['clock_reversal']
    elif mode=='changed_thresholds':native['audit_receipt']['minute_clock_audit']['unchanged_thresholds']=False
    else:native['audit_receipt']['minute_clock_audit']['control_or_treatment']='treatment'
    monkeypatch.setattr(q.n,'read',lambda path:native)
    with pytest.raises((KeyError,AssertionError)):q.audit({'mutations':{q.t.FLAG:False}},'train')


def test_underlying_native_failure_is_not_bypassed(reader,monkeypatch):
    def fail(*a,**kw):raise ValueError('native source failure')
    monkeypatch.setattr(reader,'_audit',fail)
    with pytest.raises(ValueError,match='native source failure'):
        reader.audit({'mutations':{reader.t.FLAG:False}},'train')
