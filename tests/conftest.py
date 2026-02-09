"""
Root pytest fixtures for k_stock_trader tests.

Provides mock objects, factories, and sample data for all test modules.
"""

from __future__ import annotations
import pytest
from datetime import datetime, time, date
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.mocks.mock_kis_api import MockKoreaInvestAPI, MockPosition
from tests.mocks.mock_oms_client import MockOMSClient, MockIntentResult, MockIntentStatus


# ---------------------------------------------------------------------------
# KIS API Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_kis_api():
    """Create MockKoreaInvestAPI with default prices."""
    prices = {
        "005930": 72000,  # Samsung Electronics
        "000660": 130000,  # SK Hynix
        "035420": 210000,  # NAVER
        "051910": 400000,  # LG Chem
        "006400": 300000,  # Samsung SDI
        "035720": 55000,  # Kakao
    }
    return MockKoreaInvestAPI(prices=prices)


@pytest.fixture
def mock_kis_api_with_positions():
    """Create MockKoreaInvestAPI with existing positions."""
    prices = {
        "005930": 72000,
        "000660": 130000,
    }
    positions = [
        MockPosition(symbol="005930", qty=100, avg_price=70000, current_price=72000),
    ]
    return MockKoreaInvestAPI(prices=prices, positions=positions)


@pytest.fixture
def mock_kis_api_failing():
    """Create MockKoreaInvestAPI that fails orders."""
    return MockKoreaInvestAPI(fail_orders=True)


# ---------------------------------------------------------------------------
# OMS Client Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_oms_client():
    """Create MockOMSClient with default success behavior."""
    return MockOMSClient(default_status="EXECUTED")


@pytest.fixture
def mock_oms_client_rejecting():
    """Create MockOMSClient that rejects all intents."""
    return MockOMSClient(fail_intents=True)


@pytest.fixture
def mock_oms_client_deferring():
    """Create MockOMSClient that defers all intents."""
    return MockOMSClient(defer_intents=True)


# ---------------------------------------------------------------------------
# Sector Exposure Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sector_map():
    """Sample symbol-to-sector mapping."""
    return {
        "005930": "IT",  # Samsung Electronics
        "000660": "IT",  # SK Hynix
        "035420": "IT",  # NAVER
        "051910": "Chemicals",  # LG Chem
        "006400": "Chemicals",  # Samsung SDI
        "035720": "Internet",  # Kakao
        "105560": "Financials",  # KB Financial
    }


@pytest.fixture
def mock_sector_exposure(sector_map):
    """Create SectorExposure with default config."""
    from kis_core.sector_exposure import SectorExposure, SectorExposureConfig

    config = SectorExposureConfig(
        mode="both",
        max_positions_per_sector=2,
        max_sector_pct=0.30,
        unknown_sector_policy="allow",
    )
    return SectorExposure(sector_map, config)


# ---------------------------------------------------------------------------
# Intent Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def intent_result_factory():
    """Factory for creating IntentResult objects."""
    def create(
        status: str = "EXECUTED",
        message: str = "",
        order_id: Optional[str] = None,
        modified_qty: Optional[int] = None,
        cooldown_until: Optional[float] = None,
    ) -> MockIntentResult:
        return MockIntentResult(
            intent_id="test-intent-id",
            status=MockIntentStatus(status),
            message=message,
            order_id=order_id or "ORD00000001",
            modified_qty=modified_qty,
            cooldown_until=cooldown_until,
        )
    return create


@pytest.fixture
def sample_enter_intent():
    """Create sample ENTER intent."""
    from oms.intent import Intent, IntentType, Urgency, TimeHorizon, IntentConstraints, RiskPayload

    return Intent(
        intent_type=IntentType.ENTER,
        strategy_id="KMP",
        symbol="005930",
        desired_qty=100,
        urgency=Urgency.HIGH,
        time_horizon=TimeHorizon.INTRADAY,
        constraints=IntentConstraints(
            stop_price=72100,
            limit_price=72300,
            expiry_ts=None,
        ),
        risk_payload=RiskPayload(
            entry_px=72100,
            stop_px=71500,
            hard_stop_px=71000,
            rationale_code="or_break",
            confidence="GREEN",
        ),
        signal_hash="test_signal_001",
    )


@pytest.fixture
def sample_exit_intent():
    """Create sample EXIT intent."""
    from oms.intent import Intent, IntentType, Urgency, TimeHorizon, RiskPayload

    return Intent(
        intent_type=IntentType.EXIT,
        strategy_id="KMP",
        symbol="005930",
        desired_qty=100,
        urgency=Urgency.NORMAL,
        time_horizon=TimeHorizon.INTRADAY,
        risk_payload=RiskPayload(
            rationale_code="stop_hit",
        ),
    )


# ---------------------------------------------------------------------------
# Bar/OHLCV Data Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_1m_bar():
    """Sample 1-minute OHLCV bar."""
    return {
        "open": 72000,
        "high": 72200,
        "low": 71900,
        "close": 72100,
        "volume": 15000,
        "timestamp": datetime(2024, 1, 15, 9, 30, 0),
    }


@pytest.fixture
def sample_daily_bars():
    """Sample daily bars (120 days)."""
    import pandas as pd
    import numpy as np

    np.random.seed(42)
    n = 120
    base_price = 70000

    # Generate random walk
    returns = np.random.normal(0.0005, 0.02, n)
    prices = base_price * np.cumprod(1 + returns)

    data = {
        "open": prices * (1 - np.random.uniform(0, 0.01, n)),
        "high": prices * (1 + np.random.uniform(0, 0.02, n)),
        "low": prices * (1 - np.random.uniform(0, 0.02, n)),
        "close": prices,
        "volume": np.random.randint(100000, 500000, n),
    }

    dates = pd.date_range(end=datetime(2024, 1, 15), periods=n, freq="B")
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# State Store Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def state_store():
    """Create empty StateStore."""
    from oms.state import StateStore
    return StateStore()


@pytest.fixture
def state_store_with_equity():
    """Create StateStore with equity set."""
    from oms.state import StateStore

    store = StateStore()
    store.equity = 100_000_000
    store.buyable_cash = 50_000_000
    return store


@pytest.fixture
def state_store_with_position():
    """Create StateStore with existing position."""
    from oms.state import StateStore, StrategyAllocation

    store = StateStore()
    store.equity = 100_000_000
    store.buyable_cash = 50_000_000

    pos = store.get_position("005930")
    pos.real_qty = 100
    pos.avg_price = 70000
    pos.allocations["KMP"] = StrategyAllocation(
        strategy_id="KMP",
        qty=100,
        cost_basis=70000,
    )

    return store


# ---------------------------------------------------------------------------
# Risk Config Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def risk_config():
    """Create default RiskConfig."""
    from oms.risk import RiskConfig
    return RiskConfig()


@pytest.fixture
def risk_config_strict():
    """Create strict RiskConfig with tight limits."""
    from oms.risk import RiskConfig

    return RiskConfig(
        daily_loss_warn_pct=0.01,
        daily_loss_halt_pct=0.02,
        max_gross_exposure_pct=0.50,
        max_position_pct=0.10,
        max_positions_count=5,
        max_sector_pct=0.20,
    )


# ---------------------------------------------------------------------------
# Time Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def market_open_time():
    """Return market open time (09:00 KST)."""
    return time(9, 0, 0)


@pytest.fixture
def or_lock_time():
    """Return OR lock time (09:15 KST)."""
    return time(9, 15, 0)


@pytest.fixture
def entry_window_time():
    """Return entry window time (09:30 KST)."""
    return time(9, 30, 0)


@pytest.fixture
def mock_now_kst():
    """Create mock datetime in KST."""
    return datetime(2024, 1, 15, 9, 30, 0)


# ---------------------------------------------------------------------------
# Strategy State Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kmp_symbol_state():
    """Create KMP SymbolState for testing."""
    from strategy_kmp.core.state import SymbolState, State

    state = SymbolState(code="005930")
    state.fsm = State.IDLE
    state.sector = "IT"
    state.sma20 = 70000
    state.sma60 = 68000
    state.prev_close = 71000
    state.trend_ok = True
    state.or_high = 72000
    state.or_low = 71500
    state.or_mid = 71750
    state.or_locked = True
    state.vwap = 71800
    state.value15 = 5_000_000_000  # 5B KRW
    state.surge = 3.0
    state.avg_1m_vol = 10000
    state.curr_1m_vol = 25000
    state.rvol_1m = 2.5
    state.bid = 71900
    state.ask = 72000
    state.spread = 100
    state.spread_pct = 0.0014

    return state


@pytest.fixture
def kpr_symbol_state():
    """Create KPR SymbolState for testing."""
    from strategy_kpr.core.state import SymbolState, FSMState, Tier

    state = SymbolState(code="005930")
    state.fsm = FSMState.IDLE
    state.tier = Tier.HOT
    state.sector = "IT"
    state.hod = 73000
    state.lod = 69000
    state.vwap = 71500
    state.drop_from_open = 0.04
    state.in_vwap_band = True

    return state


# ---------------------------------------------------------------------------
# Async Test Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Common Test Data
# ---------------------------------------------------------------------------

@pytest.fixture
def equity():
    """Standard equity for tests."""
    return 100_000_000  # 100M KRW


@pytest.fixture
def buyable_cash():
    """Standard buyable cash for tests."""
    return 50_000_000  # 50M KRW
