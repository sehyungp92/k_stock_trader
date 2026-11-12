"""Small tests run with the frozen native package first on sys.path."""
from dataclasses import replace
from datetime import datetime,timedelta
import json
from pathlib import Path
import sys
from types import SimpleNamespace

source=Path(sys.argv[1]).resolve();sys.path.insert(0,str(source))
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.core import logic
from backtests.strategies.kalcb import runner
from strategy_common.market import MarketBar
from strategy_common.actions import SubmitEntry,SubmitProtectiveStop
from strategy_common.clock import KST
from backtests.engine.sim_broker import SimBroker,BrokerCosts
from backtests.engine.replay import drive_timestamp_batch
from strategy_kalcb.opening_failure_v1 import evidence

cfg=KALCBConfig.from_mapping(json.loads(Path(sys.argv[2]).read_text()))
route=cfg.entry_plan_routes[0]
assert logic._entry_route_config(replace(cfg,entry_plan_min_quality_votes=1),route).entry_plan_min_quality_votes==6
assert logic._entry_route_config(cfg,{**route,'min_quality_votes':5}).entry_plan_min_quality_votes==5
assert logic._entry_route_config(cfg,{**route,'min_quality_votes':7}).entry_plan_min_quality_votes==7
old=logic.step_kalcb_core
assert runner.step_kalcb_core is old
logic.step_kalcb_core=lambda *a:None
assert runner.step_kalcb_core is not logic.step_kalcb_core # Witness: patching origin alone is ineffective.
logic.step_kalcb_core=old
assert runner.step_kalcb_core is logic.step_kalcb_core

start=datetime(2026,1,5,9,0,tzinfo=KST)
def bar(i,close=100,low=90,completed=True):
    return MarketBar('005930',start+timedelta(minutes=5*i),'5m',100,max(110,close+10),low,close,100000,is_completed=completed)
prefix=[bar(i,100 if i<6 else 94) for i in range(12)]
position=SimpleNamespace(metadata={'entry_route':'first30_open_anchor'},hold_bars=8,qty_open=10,
    exit_in_flight=False,entry_price=101,entry_time=start+timedelta(minutes=30))
ss=SimpleNamespace(bars_today=prefix)
proof=evidence(ss,position,prefix[-1],8)
assert proof and proof['available_at']==(prefix[-1].timestamp+timedelta(minutes=5)).isoformat()
ss.bars_today=prefix+[bar(20,100000,90000)]
assert evidence(ss,position,prefix[-1],8)==proof
try:evidence(ss,position,replace(prefix[-1],is_completed=False),8)
except ValueError:pass
else:raise AssertionError('Incomplete observation accepted')

class Adapter:
    snapshots={};frontier_shadow=None
    def __init__(self):self.sent=False;self.children=False
    def on_bar(self,b,broker):
        if not self.sent:
            broker.submit(SubmitEntry('KALCB',b.symbol,10,'MARKET',None,95,'probe',
                {'fill_timing':'next_5m_open','hard_stop_enabled':False}),b.timestamp);self.sent=True
        return []
    def on_fills(self,broker):
        if broker.fills and not self.children:
            broker.submit(SubmitProtectiveStop('KALCB','005930',10,95,'active_child'),broker.fills[-1].timestamp)
            self.children=True
        return []
a=Adapter();b=SimBroker(1e8,BrokerCosts(2,18,3))
drive_timestamp_batch(b,(bar(6),),(a,))
assert len(b.orders)==1 and not b.fills
assert b.orders[0].metadata['decision_available_at']==bar(7).timestamp.isoformat()
drive_timestamp_batch(b,(bar(7,close=100,low=94),),(a,))
assert len(b.fills)==2 and b.fills[0].timestamp==bar(7).timestamp
assert [f.side for f in b.fills]==['BUY','SELL']
assert b.fills[1].metadata['execution_phase']=='intrabar'
print(json.dumps(dict(passed=True,checks=['route_override','import_alias_negative_witness','completed_prefix',
    'future_tail_independence','incomplete_bar_rejection','decision_to_next_open','active_protective_child_semantics'])))
