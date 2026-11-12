"""Synthetic known-answer timing and funding checks; no market study outcomes."""
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timedelta
from dataclasses import replace
import json,sys,importlib.util,hashlib

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('E:/CodexResearch/k_stock_trader/kalcb_research_decision_20261018/v1')
out=ROOT/'horizon_unit_v2';out.mkdir(exist_ok=False)
def bound(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
helper=REPO/'backtests/strategies/kalcb/research_decision_horizon_v1.py'
(out/'registration.json').write_text(json.dumps(dict(program=bound(__file__),helper=bound(helper),scope='Synthetic only, no retained-market replay'),indent=2))
code=ROOT/'native_v1/is/inputs/files';sys.path.insert(0,str(code))
spec=importlib.util.spec_from_file_location('c18_horizon_unit_helper',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
from strategy_kalcb.core import logic
from strategy_kalcb.config import KALCBConfig
from strategy_common.market import MarketBar
from strategy_common.clock import KST
from strategy_common.actions import SubmitEntry,SubmitExit
from backtests.engine.sim_broker import SimBroker,BrokerCosts

cfg=KALCBConfig.from_mapping(json.loads(Path('E:/CodexResearch/k_stock_trader/kalcb_round3_20261017/attempt_2/native_v4/runtime/is/policies/v6_h0/100000000/config.json').read_text()))
cfg=replace(cfg,exit_liquidity_guard_enabled=False,exit_late_trend_extension_minutes=0)
t=datetime(2026,1,5,9,30,tzinfo=KST)
def bar(at,price=100,symbol='005930'):
    return MarketBar(symbol,at,'5m',price,price,price,price,100000,is_completed=True)
position=SimpleNamespace(symbol='005930',metadata={},entry_time=t,entry_price=100.,regime_tier='A',hold_bars=1,qty_open=10)
ss=SimpleNamespace(candidate=SimpleNamespace(prior_day_close=100.),bars_today=[bar(t)])
original=logic._exit_reason;off=module.DiagnosticHorizon();off.install();assert logic._exit_reason is original;off.restore();assert logic._exit_reason is original
hook=module.DiagnosticHorizon(30);hook.install()
try:
    assert logic._exit_reason(ss,position,bar(t+timedelta(minutes=20)),cfg,current_r=0,mfe_r=0)==''
    assert logic._exit_reason(ss,position,bar(t+timedelta(minutes=25)),cfg,current_r=0,mfe_r=0)=='research_scheduled_horizon'
    assert hook.records[-1]['available_at']==(t+timedelta(minutes=30)).isoformat()
finally:hook.restore()
assert logic._exit_reason is original
position.entry_time=t.replace(hour=14,minute=0);hook=module.DiagnosticHorizon(120);hook.install()
try:assert logic._exit_reason(ss,position,bar(t.replace(hour=14,minute=45)),cfg,current_r=0,mfe_r=0).startswith('eod')
finally:hook.restore()

def submit_buy(b,symbol,qty,stamp):
    return b.submit(SubmitEntry('KALCB',symbol,qty,'MARKET',None,9.,'synthetic',
        {'hard_stop_enabled':False,'fill_timing':'next_5m_open','decision_available_at':(stamp+timedelta(minutes=5)).isoformat(),'entry_price_ref':10.,'risk_per_share':1.}),stamp)
def submit_sell(b,symbol,qty,stamp):
    return b.submit(SubmitExit('KALCB',symbol,qty,'MARKET',None,'synthetic',
        {'decision_available_at':(stamp+timedelta(minutes=5)).isoformat()}),stamp)
b=SimBroker(100.,BrokerCosts(0,0,0),buying_power_leverage=1)
b.process_batch((bar(t,10,'005930'),bar(t,10,'000660')));submit_buy(b,'005930',8,t)
assert not b.fills and b.cash==100.
b.process_batch((bar(t+timedelta(minutes=5),10,'005930'),));assert b.cash==20.
submit_buy(b,'000660',8,t+timedelta(minutes=5))
b.process_batch((bar(t+timedelta(minutes=10),10,'005930'),bar(t+timedelta(minutes=10),10,'000660')))
assert sum(f.qty for f in b.fills if f.side=='BUY')<=10 and b.cash>=0
before=b.cash;submit_sell(b,'005930',8,t+timedelta(minutes=10));assert b.cash==before
for minutes in [15,20]:b.process_batch((bar(t+timedelta(minutes=minutes),10,'005930'),bar(t+timedelta(minutes=minutes),10,'000660')))
assert abs(b.cash+sum(p.qty*10 for p in b.positions.values())-100)<1e-9
assert all(f.timestamp>=datetime.fromisoformat(f.metadata['decision_available_at']) for f in b.fills)

values=[]
for symbol,price in [('005930',11.),('000660',12.)]:
    one=SimBroker(100.,BrokerCosts(0,0,0),buying_power_leverage=1)
    one.process_batch((bar(t,10,symbol),));submit_buy(one,symbol,10,t)
    one.process_batch((bar(t+timedelta(minutes=5),10,symbol),));assert one.cash==0
    submit_sell(one,symbol,10,t+timedelta(minutes=5));assert one.cash==0
    one.process_batch((bar(t+timedelta(minutes=10),price,symbol),))
    independent=100+sum((-1 if f.side=='BUY' else 1)*f.qty*f.price for f in one.fills)
    assert abs(one.cash-independent)<1e-9 and not one.positions and not one.orders
    values.append(one.cash)
assert values==[110.,120.]
result=dict(passed=True,known_funded_options=values,exact_toy_optimum=120.,
    relaxed_incompatible_sum_profit=30.,funded_max_profit=20.,
    checks=['off_function_identity','completed_clock_horizon','native_mandatory_EOD_precedence',
        'cash_not_released_on_exit_submission','native_integer_funding','independent_cash','no_early_fill',
        'tiny_exhaustive_funded_choices_not_incompatible_sum'],market_evaluations=0)
(out/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
