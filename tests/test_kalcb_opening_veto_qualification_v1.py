"""Real inherited controls, issuer-day denominators and strict rejection handling."""
from pathlib import Path
import pytest
from scripts import kalcb_opening_veto_qualification_v1 as q


@pytest.mark.parametrize('period',['is','oos'])
@pytest.mark.parametrize('scenario',q.SCENARIOS)
def test_real_parent_profiles_and_frontier_identity(period,scenario):
    controls,items=q.control_profiles();key=period+'/'+scenario
    q.validate_native(controls[key],items[key],scenario,save=False)
    index=q.read(q.FRONT/'account_index.json')
    assert q.verify_identity_bridge(controls[key],index['compound_prior100/'+key])['passed']
    assert items[key]['policy']['sha256']==q.read(q.C/'runtime'/period/'profile.json')['policy']['sha256']


def test_issuer_day_counts_losses_once_and_uses_first_decision_equity():
    def row(pnl,equity=100_000_000,notional=2_000_000):
        return dict(net_pnl_krw=pnl,decision_equity_krw=equity,entry_notional_krw=notional)
    a={'opportunities':{
        '000001/2026-04-01T10:30:00+09:00':row(-15000,50_000_000),
        '000001/2026-04-01T09:30:00+09:00':row(22000),
        '000002/2026-04-01T09:30:00+09:00':row(12000),
        '000003/2026-04-01T09:30:00+09:00':row(12000,notional=100),
    }}
    assert q.useful_days(a)==1 # 7000 is below first-equity 10000, not above later 5000.
    a['opportunities']['000002/2026-04-01T10:30:00+09:00']=row(12000)
    assert q.useful_days(a)==1 # two profitable lifetimes remain one issuer-day.


def test_standard_nonflat_rejection_never_becomes_a_pass(tmp_path,monkeypatch):
    def reject(*args,**kwargs):raise ValueError('Unresolved native account')
    monkeypatch.setattr(q,'validate_native',reject)
    q.write(tmp_path/'result.json',dict(residual_sessions=1,terminal_inventory=0,terminal_pending=0))
    assert q.classify_standard(tmp_path,{},'baseline')['qualified'] is False
    q.write(tmp_path/'independent_account_verification.json',dict(passed=True))
    with pytest.raises(ValueError,match='contradicts'):q.classify_standard(tmp_path,{},'baseline')


def test_standard_other_errors_are_not_relabelled_as_residual(tmp_path,monkeypatch):
    def reject(*args,**kwargs):raise ValueError('Wrong source')
    monkeypatch.setattr(q,'validate_native',reject)
    with pytest.raises(ValueError,match='Wrong source'):q.classify_standard(tmp_path,{},'baseline')
