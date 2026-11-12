from datetime import datetime
import pytest
from backtests.strategies.kalcb.funded_horizon_v1 import HorizonReplay,enabled_signal


def signal():
    return {'day':'2026-04-01','origin':'2026-03-10','available_at':'2026-04-01T09:30:00+09:00',
            'predictions':{'baseline':-30.,'combined':-40.}}


def controller(mode):
    c=HorizonReplay(mode,{'2026-04-01/000001':signal()});c.day='2026-04-01'
    c.stamp=datetime.fromisoformat('2026-04-01T09:25:00+09:00');c.allocate=lambda s,q,m:q
    return c


def test_disabled_is_exact_and_never_uses_future_signal():
    c=controller('disabled');c.signals['2026-04-01/000001']['available_at']='2027-01-01T09:30:00+09:00'
    assert c.quantity('000001',37,{'entry_route':'first30_open_anchor'})==37
    assert not c.records


def test_session_veto_preserves_distinct_later_route_and_other_names():
    c=controller('veto')
    assert c.quantity('000001',37,{'entry_route':'first30_open_anchor'})==0
    assert c.quantity('000001',37,{'entry_route':'later60'})==37
    assert c.quantity('000002',37,{'entry_route':'first30_open_anchor'})==37
    assert len(c.records)==1


def test_renew_preserves_entry_and_suppresses_only_treated_first30_retry():
    c=controller('renew');m={'entry_route':'first30_open_anchor'}
    assert c.quantity('000001',37,m)==37
    c.finished.add('2026-04-01/000001')
    assert c.quantity('000001',37,m)==0
    assert c.quantity('000001',37,{'entry_route':'later60'})==37


def test_signal_requires_causal_availability_and_both_declared_cost_hurdles():
    r=signal();at=datetime.fromisoformat(r['available_at']);assert enabled_signal(r,at)
    r['predictions']['baseline']=-23.;assert not enabled_signal(r,at)
    r=signal();r['available_at']='2026-04-01T09:35:00+09:00'
    with pytest.raises(ValueError,match='Future opening'):enabled_signal(r,at)


@pytest.mark.parametrize('scenario',['baseline','combined'])
def test_renew_uses_real_native_partial_fill_cancel_and_next_event_exit(scenario):
    from dataclasses import replace
    from tests.backtests.strategies.test_kalcb_checkpoint_contrast_labels_v1 import fixture
    from scripts.kalcb_executed_value_study import run_episode
    row,bars,cfg,_,_=fixture();cfg=replace(cfg,frontier_shadow_enabled=False)
    q=row['action']['quantity'];ref=bars[5].close
    bars[5]=replace(bars[5],volume=max(200,int(q/.04)))
    bars[6]=replace(bars[6],open=ref,high=ref,low=ref*.99,close=ref*.99,volume=max(200,int(q/.04)))
    day=row['trade_date'];key=day+'/'+bars[0].symbol;r=signal()
    r.update(day=day,origin='2025-01-01',available_at=day+'T09:30:00+09:00')
    c=HorizonReplay('renew',{key:r});c.attempts.add(key)
    # This exercises the action adapter; full frozen bridges separately exercise admission hook wiring.
    c.install()
    try:run_episode(row,bars,cfg,scenario,'horizon-native-fixture',None,[day])
    finally:c.restore()
    actions=[a for a in c.records if a['kind']=='renew_exit']
    assert len(actions)==1 and actions[0]['action']['cancelled_buy_remainder']
    assert actions[0]['row']['qty']>0 and key in c.finished
