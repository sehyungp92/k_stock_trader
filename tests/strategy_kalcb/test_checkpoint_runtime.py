from dataclasses import replace
from datetime import timedelta
import pytest
from strategy_kalcb.engine import KALCBEngine
from strategy_kalcb.core.core_models import KALCBPortfolioView
from strategy_kalcb.core.serializers import snapshot_state, restore_state
from strategy_common.actions import action_to_json_dict
from tests.strategy_kalcb.test_checkpoint_entry import setup


def test_live_adapter_and_native_replay_have_same_checkpoint_intent_after_restart():
    from backtests.strategies.kalcb.runner import KALCBReplayAdapter
    from backtests.engine.sim_broker import SimBroker, BrokerCosts
    from backtests.engine.replay import drive_timestamp_batch
    snap, bars, cfg, at = setup()
    base = replace(snap, candidates=(), generated_at=at.replace(hour=0, minute=0))
    live = KALCBEngine(config=cfg, candidate_snapshot=base)
    for b in bars[:8]:
        live.observe_candidate_bar(b)
    live.state = restore_state(snapshot_state(live.state))
    with pytest.raises(ValueError):
        live.activate_candidate_snapshot(snap, at - timedelta(seconds=1))
    live.activate_candidate_snapshot(snap, at)
    actions = []
    def submit(action):
        actions.append(action_to_json_dict(action))
        return 'live-order'
    live.on_bar(bars[8], KALCBPortfolioView(cash=1_000_000, equity=1_000_000), submit)
    costs = BrokerCosts(0, 0, 0)
    broker = SimBroker(1_000_000, costs)
    adapter = KALCBReplayAdapter(cfg, {snap.trade_date: base}, initial_equity=1_000_000, costs=costs,
        timed_snapshots={snap.trade_date: [snap]}, observation_symbols=(bars[0].symbol,))
    events = []
    for b in bars[:9]:
        events.extend(drive_timestamp_batch(broker, (b,), (adapter,)))
    native = [action_to_json_dict(a) for e in events if e.decision_code == 'entry' for a in e.actions]
    assert actions and actions == native
    assert not broker.fills


def test_legacy_selector_fails_closed_for_new_round_configuration():
    from backtests.strategies.kalcb.runner import run_kalcb_backtest
    with pytest.raises(ValueError, match='authenticated policy'):
        run_kalcb_backtest({'entry_checkpoint_enabled': True, 'capability_level': 'synthetic'})
