from pathlib import Path
import pytest
from scripts.kalcb_pending_funding_preflight_v1 import patch_core


def test_patch_is_narrow_unique_and_has_disabled_identity():
    source=Path('strategy_kalcb/core/logic.py').read_text(encoding='utf-8')
    result=patch_core(source)
    assert result.count('reserved_gross = _governed_open_notional')==1
    assert 'if config.risk_max_gross_equity_pct:\n' in result
    assert 'open_notional=portfolio.open_notional,' in result
    assert result.count('remaining / bar.close')==1
    compile(result,'frozen_research_core.py','exec')


def test_unexpected_source_is_rejected():
    with pytest.raises(ValueError,match='anchor'):patch_core('')


def test_patch_cannot_be_applied_twice():
    source=Path('strategy_kalcb/core/logic.py').read_text(encoding='utf-8')
    with pytest.raises(ValueError,match='anchor'):patch_core(patch_core(source))


def patched(monkeypatch):
    import types,sys
    from strategy_kalcb.core import preview
    name='strategy_kalcb.core._pending_funding_test'
    module=types.ModuleType(name);module.__package__='strategy_kalcb.core'
    monkeypatch.setitem(sys.modules,name,module)
    exec(compile(patch_core(Path('strategy_kalcb/core/logic.py').read_text(encoding='utf-8')),
        'research_core.py','exec'),module.__dict__)
    monkeypatch.setattr(preview,'step_kalcb_core',module.step_kalcb_core)
    return preview.preview_entry


@pytest.mark.parametrize('enabled',[False,True])
def test_real_core_pending_reservation_and_release(monkeypatch,enabled):
    from dataclasses import replace
    from tests.backtests.strategies.test_kalcb_reclaim_labels_v1 import native_fixture
    from strategy_kalcb.core.core_models import KALCBPortfolioView
    from strategy_kalcb.core.state import KALCBState,SymbolStage
    from strategy_kalcb.core.preview import preview_entry
    snap,bars,cfg,at=native_fixture();cfg=replace(cfg,intraday_leverage=2)
    account=KALCBPortfolioView(1000000,equity=1000000)
    original=preview_entry(bars,snap,cfg,account,available_at=at)
    qty=original['intents'][0]['quantity'];room_qty=qty//2
    run=patched(monkeypatch)
    state=KALCBState();state.session_date=snap.trade_date
    queued=state.symbol_state('PENDING');queued.stage=SymbolStage.ENTRY_QUEUED
    queued.pending_entry_order_id='pending-native-order'
    queued.pending_entry_metadata={'risk_reserved_entry_qty':1,
        'entry_price_ref':2000000-room_qty*bars[-1].close}
    changed=run(bars,snap,replace(cfg,risk_max_gross_equity_pct=2 if enabled else 0),
        account,available_at=at,state=state)
    assert changed['intents'][0]['quantity']==(room_qty if enabled else qty)
    # Releasing a cancelled reservation restores the original action; the
    # input state itself was not mutated by either read-only preview.
    assert queued.pending_entry_order_id=='pending-native-order'
    queued.stage=SymbolStage.WATCHING;queued.pending_entry_order_id='';queued.pending_entry_metadata={}
    released=run(bars,snap,replace(cfg,risk_max_gross_equity_pct=2 if enabled else 0),account,available_at=at,state=state)
    assert released['intents'][0]['quantity']==qty


def test_disabled_patched_core_reproduces_actual_intent_metadata(monkeypatch):
    from tests.backtests.strategies.test_kalcb_reclaim_labels_v1 import native_fixture
    from strategy_kalcb.core.core_models import KALCBPortfolioView
    from strategy_kalcb.core.preview import preview_entry
    snap,bars,cfg,at=native_fixture();account=KALCBPortfolioView(1000000,equity=1000000)
    expected=preview_entry(bars,snap,cfg,account,available_at=at)
    assert patched(monkeypatch)(bars,snap,cfg,account,available_at=at)==expected
