from pathlib import Path
import pytest
from backtests.strategies.kalcb.retention_research_io_v1 import read,write,bound
from backtests.strategies.kalcb import holding_account_recovery_v2 as recovery


@pytest.mark.parametrize('kind',['partial','oos_substitution','duplicate'])
def test_recovery_rejects_partial_wrong_or_duplicate_matrix(tmp_path,kind):
    write(tmp_path/'protocol.json',{'programs':[],'bindings':[],'database_bindings':[]})
    write(tmp_path/'supervisor_failure.json',{'preserve_evidence':True})
    rows=[{'name':'control','period':'is','scenario':s} for s in recovery.SCENARIOS]
    if kind=='partial':rows.pop()
    elif kind=='oos_substitution':rows[-1]['period']='oos'
    else:rows[-1]=rows[0]
    write(tmp_path/'progress.json',rows)
    with pytest.raises(ValueError,match='Only the four complete IS'):
        recovery.authenticate_predecessor(tmp_path)


def test_reuse_keeps_original_absolute_profile_and_private_files(tmp_path,monkeypatch):
    source=tmp_path/'old';dest=tmp_path/'new';root=tmp_path/'study'
    profile={'manifest':{'path':'E:/original/manifest.json','sha256':'abc'}}
    write(source/'input_profile.json',profile);write(source/'result.json',{'return_pct':4.})
    write(root/'protocol.json',{'fixture':True})
    files=[{'relative':p.name,**bound(p)} for p in source.iterdir()]
    spec={'source':str(source),'files':files,'original_profile':profile}
    monkeypatch.setattr(recovery,'validate_native',lambda *a,**k:None)
    recovery.reuse_complete(root,spec,dest,'baseline')
    assert read(dest/'input_profile.json')==profile
    write(dest/'result.json',{'return_pct':5.})
    assert read(source/'result.json')=={'return_pct':4.}
    with pytest.raises(ValueError,match='already exists'):
        recovery.reuse_complete(root,spec,dest,'baseline')


def test_reuse_rejects_changed_completed_artifact(tmp_path,monkeypatch):
    source=tmp_path/'old';write(source/'result.json',{'return_pct':4.})
    spec={'source':str(source),'files':[{'relative':'result.json',**bound(source/'result.json')}],
        'original_profile':{}}
    write(source/'result.json',{'return_pct':5.})
    with pytest.raises(ValueError,match='Changed binding'):
        recovery.reuse_complete(tmp_path,spec,tmp_path/'new','baseline')
    assert not (tmp_path/'new').exists()


def test_audit_receipt_cannot_be_substituted(tmp_path):
    from backtests.strategies.kalcb.pending_entry_account_pilot_v1 import neutral
    write(tmp_path/'holding_complete.json',{'complete':True})
    write(tmp_path/'execution_audit.json',{'audit_receipt':{'identity':{'scope':'real'}}})
    write(tmp_path/'terminal_audit_transition.json',{'passed':True,'after_sha256':'a',
        'expected_after_sha256':'a','published_evaluation_identity':neutral({'scope':'different'})})
    write(tmp_path/'completion.json',{'complete':True,'holding':bound(tmp_path/'holding_complete.json'),
        'terminal_audit_transition':bound(tmp_path/'terminal_audit_transition.json')})
    with pytest.raises(ValueError,match='Unverified complete terminal'):
        recovery.verify_terminal_transition(tmp_path)
