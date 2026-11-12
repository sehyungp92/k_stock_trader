import asyncio
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock

from strategy_pcim.main import _cancel_and_handle_partial_fills, _reconcile_pending_entry_allocations
from strategy_pcim.positions.manager import PCIMPosition, PositionManager


def test_cutoff_reconciles_full_intended_allocation_against_partial_local_state():
    manager = PositionManager()
    manager.add_position(PCIMPosition("005930", date(2026, 2, 2), 100, 4, 2))
    manager.track_pending("005930", "intent", 10, 2)
    remote = SimpleNamespace(get_allocation=lambda _: 10,
        allocations={"PCIM": SimpleNamespace(qty=10, cost_basis=106)})
    oms = SimpleNamespace(submit_intent=AsyncMock(), get_position=AsyncMock(return_value=remote),
        get_all_positions=AsyncMock(return_value={"005930": remote}))
    asyncio.run(_cancel_and_handle_partial_fills({"005930": 10}, manager, oms, None))
    assert manager.positions["005930"].remaining_qty == 10
    assert manager.positions["005930"].entry_price == 106
    assert "005930" in manager.pending_orders  # A cancel request is not an acknowledgment.
    remote.allocations["PCIM"].qty = 12
    asyncio.run(_reconcile_pending_entry_allocations(oms, manager, date(2026, 2, 2)))
    assert manager.positions["005930"].qty == 12
    assert manager.positions["005930"].remaining_qty == 12
