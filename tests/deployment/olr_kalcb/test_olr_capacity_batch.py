import asyncio
from dataclasses import replace
import json

import pytest

from deployment.olr_kalcb.offline_replay import ReplayInputLoader
from tests.deployment.olr_kalcb.test_session_driver import _driver, _bar, _NoActionEngine


class BatchEngine(_NoActionEngine):
    def on_bars(self, bars, portfolio, submit):
        self.state['cohort']=[b.symbol for b in bars]
        return []


def test_runtime_cohort_is_durable_and_loader_reconstructs_verified_bars(tmp_path):
    engine=BatchEngine();driver=_driver(tmp_path,engine,strategy_id='OLR')
    bars=(_bar('005930'),_bar('000660'))
    asyncio.run(driver.handle_bars(bars))
    assert engine.state['cohort']==['005930','000660']
    root=tmp_path/'session'
    # Minimal manifest needed to inspect captured event order independently.
    (root/'session_manifest.json').write_text(json.dumps({'trade_date':'2026-02-02','mode':'dry_run'}))
    events=ReplayInputLoader(root).load_runtime_events()
    assert len(events)==1 and events[0][1]=='strategy_bar_batch'
    sid,loaded=events[0][2]
    assert sid=='OLR' and [b.to_json_dict() for b in loaded]==[b.to_json_dict() for b in bars]


def test_driver_rejects_mixed_cohort_before_core_or_orders(tmp_path):
    engine=BatchEngine();driver=_driver(tmp_path,engine,strategy_id='OLR')
    bars=(_bar('005930'),replace(_bar('000660'),is_completed=False))
    with pytest.raises(ValueError,match='Invalid completed'):
        asyncio.run(driver.handle_bars(bars))
    assert 'cohort' not in engine.state

def test_seed_is_replayable_without_creating_orders(tmp_path):
    from datetime import datetime, timedelta
    from strategy_olr.engine import OLREngine
    from strategy_olr.config import OLRConfig
    from strategy_common.clock import KST
    engine=OLREngine(config=OLRConfig(capacity_broad_auction_history=True))
    driver=_driver(tmp_path,engine,strategy_id='OLR')
    stamp=datetime(2026,1,30,15,30,tzinfo=KST)
    seed=replace(_bar('005930'),timestamp=stamp,open=100,high=100,low=100,close=100,source_fingerprint='provider-snapshot')
    cutoff=datetime(2026,2,2,8,tzinfo=KST)
    result=asyncio.run(driver.handle_capacity_seed((seed,),cutoff))
    assert result.accepted_intent_count==0
    root=tmp_path/'session'
    (root/'session_manifest.json').write_text(json.dumps({'trade_date':'2026-02-02','mode':'dry_run'}))
    events=ReplayInputLoader(root).load_runtime_events()
    assert events[0][1]=='strategy_capacity_seed'
    _,bars,at=events[0][2]
    other=OLREngine(config=engine.config);other.seed_capacity_history(bars,at)
    assert other.state.meta==engine.state.meta
