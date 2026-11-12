from dataclasses import replace
from pathlib import Path
import sys,types
import pytest
from backtests.strategies.kalcb.pending_funding_io_v1 import patch_core


def install(monkeypatch):
    from strategy_kalcb.core import preview
    from backtests.strategies.kalcb import runner
    module=types.ModuleType('strategy_kalcb.core._funding_scoped_test');module.__package__='strategy_kalcb.core'
    monkeypatch.setitem(sys.modules,module.__name__,module)
    exec(compile(patch_core(Path('strategy_kalcb/core/logic.py').read_text(encoding='utf-8')),'funding_core.py','exec'),module.__dict__)
    monkeypatch.setattr(preview,'step_kalcb_core',module.step_kalcb_core)
    monkeypatch.setattr(runner,'step_kalcb_core',module.step_kalcb_core)
    return preview.preview_entry,module


def fixture(opening):
    from tests.backtests.strategies.test_kalcb_reclaim_labels_v1 import native_fixture
    from strategy_kalcb.core.core_models import KALCBPortfolioView
    from strategy_kalcb.core.state import KALCBState,SymbolStage
    snap,bars,cfg,at=native_fixture();cfg=replace(cfg,intraday_leverage=2)
    if opening:
        at=bars[0].timestamp.replace(minute=30)
        c=replace(snap.candidates[0],metadata={'candidate_rank':1,'action_quantity_scale':.25})
        snap=replace(snap,candidates=(c,),generated_at=bars[0].timestamp.replace(hour=0))
        cfg=replace(cfg,entry_checkpoint_enabled=False,entry_plan_routes=({'name':'first30_open_anchor','mode':'first30_open'},))
        bars=bars[:6]
    state=KALCBState();state.session_date=snap.trade_date
    pending=state.symbol_state('PENDING');pending.stage=SymbolStage.ENTRY_QUEUED;pending.pending_entry_order_id='queued'
    pending.pending_entry_metadata={'risk_reserved_entry_qty':1,'entry_price_ref':1999700}
    return snap,bars,cfg,at,KALCBPortfolioView(1000000,equity=1000000),state


@pytest.mark.parametrize('opening',[False,True])
def test_disabled_exact_and_checkpoint_untouched(monkeypatch,opening):
    from strategy_kalcb.core.preview import preview_entry
    s,b,c,a,p,st=fixture(opening);old=preview_entry(b,s,c,p,available_at=a,state=st)
    assert old['ready']
    run,_=install(monkeypatch)
    assert run(b,s,c,p,available_at=a,state=st)==old
    new=run(b,s,replace(c,risk_max_gross_equity_pct=2),p,available_at=a,state=st)
    if opening:
        x=new['intents'][0];assert x['quantity']==int(300/b[-1].close)<old['intents'][0]['quantity']
        proof=x['metadata']['pending_funding_reservation'];assert proof['original_qty']==old['intents'][0]['quantity']
    else:assert new==old


def test_pending_release_and_zero_abstention(monkeypatch):
    from strategy_kalcb.core.state import SymbolStage
    run,_=install(monkeypatch);s,b,c,a,p,st=fixture(True);c=replace(c,risk_max_gross_equity_pct=2)
    pending=st.symbols['PENDING'];pending.pending_entry_metadata['entry_price_ref']=2000001
    zero=run(b,s,c,p,available_at=a,state=st);assert not zero['intents']
    pending.stage=SymbolStage.WATCHING;pending.pending_entry_order_id='';pending.pending_entry_metadata={}
    full=run(b,s,c,p,available_at=a,state=st);assert full['ready']
    assert full['intents'][0]['metadata']['pending_funding_reservation']['original_qty']==full['intents'][0]['quantity']


def test_exact_bridge_gate_rejects_missing_or_failed(tmp_path):
    from scripts.kalcb_pending_funding_study_v1 import require_controls
    from backtests.strategies.kalcb.retention_research_io_v1 import write
    recipes={'control':{'reservation':0},'control_prior125':{'reservation':0},'candidate':{'reservation':2}}
    with pytest.raises(FileNotFoundError):require_controls(tmp_path,recipes)
    for n in ('control','control_prior125'):
        for p in ('is','oos'):
            for s in ('baseline','combined'):
                for f in ('identity_bridge.json','independent_account_verification.json'):write(tmp_path/'cases'/n/p/s/f,{'passed':True})
    require_controls(tmp_path,recipes)
    write(tmp_path/'cases/control_prior125/oos/combined/identity_bridge.json',{'passed':False})
    with pytest.raises(ValueError):require_controls(tmp_path,recipes)


def test_core_patch_refuses_double_application():
    text=patch_core(Path('strategy_kalcb/core/logic.py').read_text(encoding='utf-8'))
    with pytest.raises(ValueError):patch_core(text)


def test_capped_native_order_fills_next_bar_with_costs(monkeypatch):
    from datetime import timedelta
    from backtests.strategies.kalcb.runner import KALCBReplayAdapter
    from backtests.engine.sim_broker import SimBroker,BrokerCosts
    from backtests.engine.replay import drive_timestamp_batch
    install(monkeypatch);snap,bars,cfg,at,_,state=fixture(True)
    cfg=replace(cfg,risk_max_gross_equity_pct=2);costs=BrokerCosts(2,18,3)
    broker=SimBroker(1000000,costs,buying_power_leverage=2)
    adapter=KALCBReplayAdapter(cfg,{snap.trade_date:snap},initial_equity=1000000,costs=costs)
    for b in bars[:-1]:drive_timestamp_batch(broker,(b,),(adapter,))
    # Synthetic outstanding-reservation fixture; it is never economic evidence.
    adapter.state.symbols['PENDING']=state.symbols['PENDING']
    drive_timestamp_batch(broker,(bars[-1],),(adapter,));assert not broker.fills
    assert broker.orders and broker.orders[0].qty==2
    nxt=replace(bars[-1],timestamp=at,open=101,high=102,low=100.5,close=101.5,volume=10000)
    drive_timestamp_batch(broker,(nxt,),(adapter,))
    buys=[f for f in broker.fills if f.side=='BUY'];assert len(buys)==1 and buys[0].qty==2
    f=buys[0];assert f.timestamp==at and f.price==pytest.approx(101*1.0003)
    assert broker.cash==pytest.approx(1000000-2*f.price-f.metadata['fee'])
    assert not broker.same_bar_fill_violations


def test_plugin_executes_eight_controls_before_any_variant(monkeypatch):
    from backtests.strategies.kalcb.pending_funding_plugin_v1 import PendingFundingPlugin
    from backtests.strategies.kalcb.selective_recovery_plugin_v1 import SelectiveRecoveryPlugin
    p=object.__new__(PendingFundingPlugin);p.records={};p.spec={'recipes':{'control':{'reservation':0},'control_prior125':{'reservation':0},'candidate':{'reservation':2}}};calls=[]
    def evaluate(self,names):
        for name in names:calls.append(name);self.records[name]={'recipe':name}
        return [self.records[n] for n in names]
    monkeypatch.setattr(SelectiveRecoveryPlugin,'evaluate_names',evaluate)
    p.evaluate_names(['candidate']);assert calls==['control','control_prior125','candidate']
