from types import SimpleNamespace
from copy import deepcopy
import pytest

from backtests.strategies.kalcb import account_fork_v2 as states
from backtests.strategies.kalcb.holding_account_replay_v2 import HoldingAccountReplay


def setup_controller(tmp_path,monkeypatch,poison=None):
    from backtests.engine import integrity
    adapter=SimpleNamespace(frontier_shadow=None,snapshots={},timed_snapshots={},state={'qty':0})
    broker=SimpleNamespace(cash=123.,orders=[],fills=[],equity_curve=[123.])
    identity={'scope':'native-test','qualification_snapshot_hash':'test'}
    def audit(broker,*,identity=None):
        broker.evaluation_identity=deepcopy(identity)
        if poison=='cash':broker.cash+=1
        if poison=='adapter':adapter.state['qty']=1
        if poison=='history':broker.fills.append({'unexpected':True})
        if poison=='identity':broker.evaluation_identity['scope']='wrong'
        return {'audit_receipt':{'identity':identity},'audit_pass':True}
    monkeypatch.setattr(integrity,'audit_broker',audit)
    c=HoldingAccountReplay(tmp_path)
    c.install();c.parent=(adapter,broker);c.day='2026-07-10'
    c.expected_hold=states.receipt(adapter,broker)['sha256']
    return c,adapter,broker,identity


def test_terminal_hold_and_complete_audit_transition_are_both_checked(tmp_path,monkeypatch):
    from backtests.engine import integrity
    c,a,b,i=setup_controller(tmp_path,monkeypatch)
    try:
        expected=c.expected_hold
        integrity.audit_broker(b,identity=i)
        c.finish()
        assert c.checked_days==[{'day':'2026-07-10','exact_hold_eod_state':expected}]
        assert c.audit_transition['before_sha256']==expected
        assert c.audit_transition['after_sha256']!=expected
        assert c.audit_transition['after_sha256']==c.audit_transition['expected_after_sha256']
    finally:c.restore()


@pytest.mark.parametrize('poison',['cash','adapter','history','identity'])
def test_audit_cannot_hide_any_other_state_change(tmp_path,monkeypatch,poison):
    from backtests.engine import integrity
    c,a,b,i=setup_controller(tmp_path,monkeypatch,poison)
    try:
        with pytest.raises(ValueError,match='beyond its exact published identity'):
            integrity.audit_broker(b,identity=i)
    finally:c.restore()


def test_real_pre_audit_mismatch_still_fails(tmp_path,monkeypatch):
    from backtests.engine import integrity
    c,a,b,i=setup_controller(tmp_path,monkeypatch)
    b.cash+=1
    try:
        with pytest.raises(ValueError,match='No-action EOD full-state identity failed'):
            integrity.audit_broker(b,identity=i)
        assert c.audit_transition is None
    finally:c.restore()


def test_export_cannot_change_state_after_audit(tmp_path,monkeypatch):
    from backtests.engine import integrity
    c,a,b,i=setup_controller(tmp_path,monkeypatch)
    try:
        integrity.audit_broker(b,identity=i);b.cash+=1
        with pytest.raises(ValueError,match='State changed after'):
            c.finish()
    finally:c.restore()


def test_native_terminal_audit_is_required(tmp_path,monkeypatch):
    c,a,b,i=setup_controller(tmp_path,monkeypatch)
    try:
        with pytest.raises(ValueError,match='Missing verified native terminal audit'):
            c.finish()
    finally:c.restore()
