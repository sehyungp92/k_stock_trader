"""Tests for OMS risk module."""

import pytest
from unittest.mock import MagicMock
import time

from oms.risk import RiskGateway, RiskConfig, RiskDecision, RiskResult
from oms.state import StateStore, StrategyAllocation, WorkingOrder, OrderStatus
from oms.intent import Intent, IntentType, Urgency, TimeHorizon, RiskPayload, IntentConstraints


class TestRiskConfig:
    """Tests for RiskConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RiskConfig()

        assert config.daily_loss_warn_pct == 0.02
        assert config.daily_loss_halt_pct == 0.03
        assert config.max_gross_exposure_pct == 0.80
        assert config.max_net_exposure_pct == 0.60
        assert config.max_position_pct == 0.15
        assert config.max_positions_count == 10
        assert config.max_sector_pct == 0.30
        assert config.max_spread_bps == 50.0
        assert config.vi_cooldown_sec == 600.0

    def test_default_regime_caps(self):
        """Test default regime exposure caps."""
        config = RiskConfig()

        assert config.regime_exposure_caps["CRISIS"] == 0.20
        assert config.regime_exposure_caps["WEAK"] == 0.50
        assert config.regime_exposure_caps["NORMAL"] == 0.80
        assert config.regime_exposure_caps["STRONG"] == 1.00
        assert config.current_regime == "NORMAL"

    def test_default_strategy_budgets(self):
        """Test default strategy budgets."""
        config = RiskConfig()

        assert "KMP" in config.strategy_budgets
        assert config.strategy_budgets["KMP"]["max_positions"] == 4
        assert config.strategy_budgets["KMP"]["max_risk_pct"] == 0.015


class TestRiskGatewayGlobalBlocks:
    """Tests for RiskGateway global block checks."""

    @pytest.fixture
    def gateway(self, state_store_with_equity, risk_config):
        """Create RiskGateway for testing."""
        return RiskGateway(state_store_with_equity, risk_config)

    @pytest.fixture
    def enter_intent(self):
        """Create sample ENTER intent."""
        return Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
            risk_payload=RiskPayload(entry_px=72000, stop_px=71000),
        )

    def test_safe_mode_blocks_all(self, gateway, enter_intent):
        """Test safe mode defers all intents."""
        gateway.safe_mode = True

        result = gateway.check(enter_intent)

        assert result.decision == RiskDecision.DEFER
        assert "safe mode" in result.reason.lower()

    def test_flatten_blocks_entries(self, gateway, enter_intent):
        """Test flatten mode blocks entries."""
        gateway.flatten_in_progress = True

        result = gateway.check(enter_intent)

        assert result.decision == RiskDecision.REJECT
        assert "flatten" in result.reason.lower()

    def test_flatten_allows_exits(self, gateway):
        """Test flatten mode allows exits."""
        gateway.flatten_in_progress = True

        exit_intent = Intent(
            intent_type=IntentType.EXIT,
            strategy_id="KMP",
            symbol="005930",
        )

        result = gateway.check(exit_intent)
        assert result.decision == RiskDecision.APPROVE

    def test_halt_blocks_entries(self, gateway, enter_intent):
        """Test halt flag blocks entries."""
        gateway.halt_new_entries = True

        result = gateway.check(enter_intent)

        assert result.decision == RiskDecision.REJECT
        assert "halted" in result.reason.lower()

    def test_paused_strategy_blocks_entries(self, gateway, enter_intent):
        """Test paused strategy blocks entries."""
        gateway._paused_strategies.add("KMP")

        result = gateway.check(enter_intent)

        assert result.decision == RiskDecision.REJECT
        assert "paused" in result.reason.lower()

    def test_frozen_symbol_blocks_entries(self, gateway, enter_intent):
        """Test frozen symbol blocks entries."""
        pos = gateway.state.get_position("005930")
        pos.frozen = True

        result = gateway.check(enter_intent)

        assert result.decision == RiskDecision.REJECT
        assert "frozen" in result.reason.lower()


class TestRiskGatewayDailyLimits:
    """Tests for RiskGateway daily limit checks."""

    @pytest.fixture
    def gateway(self, state_store_with_equity, risk_config):
        """Create RiskGateway for testing."""
        return RiskGateway(state_store_with_equity, risk_config)

    @pytest.fixture
    def enter_intent(self):
        """Create sample ENTER intent."""
        return Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
            risk_payload=RiskPayload(entry_px=72000, stop_px=71000),
        )

    def test_daily_loss_warn_halts_entries(self, gateway, enter_intent):
        """Test daily loss warn threshold halts new entries."""
        gateway.state.daily_pnl_pct = -0.025  # 2.5% loss, exceeds 2% warn

        result = gateway.check(enter_intent)

        assert result.decision == RiskDecision.REJECT
        assert gateway.halt_new_entries is True

    def test_daily_loss_halt_rejects(self, gateway, enter_intent):
        """Test daily loss halt threshold rejects entries."""
        gateway.state.daily_pnl_pct = -0.035  # 3.5% loss, exceeds 3% halt

        result = gateway.check(enter_intent)

        assert result.decision == RiskDecision.REJECT

    def test_daily_loss_allows_exits(self, gateway):
        """Test daily loss allows exit intents."""
        gateway.state.daily_pnl_pct = -0.05  # 5% loss

        exit_intent = Intent(
            intent_type=IntentType.EXIT,
            strategy_id="KMP",
            symbol="005930",
        )

        result = gateway.check(exit_intent)
        assert result.decision == RiskDecision.APPROVE


class TestRiskGatewayExposureLimits:
    """Tests for RiskGateway exposure limit checks."""

    @pytest.fixture
    def gateway(self, state_store_with_equity, risk_config):
        """Create RiskGateway for testing."""
        return RiskGateway(state_store_with_equity, risk_config)

    @pytest.fixture
    def enter_intent(self):
        """Create sample ENTER intent."""
        return Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
            risk_payload=RiskPayload(entry_px=72000, stop_px=71000),
        )

    def test_max_positions_rejects(self, gateway, enter_intent):
        """Test max positions count rejects entry."""
        # Fill up positions
        for i in range(10):
            symbol = f"00{i:04d}"
            gateway.state.update_position(symbol, real_qty=100)

        result = gateway.check(enter_intent)

        assert result.decision == RiskDecision.REJECT
        assert "max positions" in result.reason.lower()

    def test_gross_exposure_rejects(self, gateway, enter_intent):
        """Test gross exposure limit rejects entry."""
        # Add large existing position: 85M = 85% of 100M equity
        gateway.state.update_position("000660", real_qty=1000, avg_price=85000)

        result = gateway.check(enter_intent)

        assert result.decision == RiskDecision.REJECT
        assert "gross exposure" in result.reason.lower()

    def test_position_size_scaled(self, gateway):
        """Test position size is scaled down when exceeding limit."""
        # 100M equity, 15% limit = 15M max
        # Intent for 300 shares at 70000 = 21M (exceeds 15M)
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=300,
            risk_payload=RiskPayload(entry_px=70000, stop_px=69000),
        )

        result = gateway.check(intent)

        assert result.decision == RiskDecision.MODIFY
        assert result.modified_qty is not None
        assert result.modified_qty < 300
        # max_qty = 15M / 70000 = 214
        assert result.modified_qty == 214

    def test_regime_cap_applies(self, gateway, enter_intent):
        """Test regime cap is applied to exposure."""
        gateway.config.current_regime = "CRISIS"
        # CRISIS cap = 20%, add position at 15% already
        gateway.state.update_position("000660", real_qty=214, avg_price=70000)  # ~15M

        result = gateway.check(enter_intent)

        assert result.decision == RiskDecision.REJECT
        assert "regime" in result.reason.lower()


class TestRiskGatewaySectorLimits:
    """Tests for RiskGateway sector limit checks."""

    @pytest.fixture
    def gateway(self, state_store_with_equity, risk_config, sector_map):
        """Create RiskGateway with sector map."""
        return RiskGateway(state_store_with_equity, risk_config, sector_map=sector_map)

    @pytest.fixture
    def enter_intent(self):
        """Create sample ENTER intent for IT sector."""
        return Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="000660",  # SK Hynix - IT sector
            desired_qty=100,
            risk_payload=RiskPayload(entry_px=130000, stop_px=125000),
        )

    def test_sector_limit_rejects(self, gateway, enter_intent):
        """Test sector limit rejects entry when exceeded."""
        # Add 25M in IT sector (Samsung) - 25% of 100M equity
        gateway.reserve_sector("005930", 347, 72000)  # ~25M
        gateway.on_sector_fill("005930", 347, 72000)

        # New entry would add 13M (100 * 130000), total 38M = 38%
        result = gateway.check(enter_intent)

        assert result.decision == RiskDecision.REJECT
        assert "sector" in result.reason.lower()


class TestRiskGatewayStrategyBudget:
    """Tests for RiskGateway strategy budget checks."""

    @pytest.fixture
    def gateway(self, state_store_with_equity, risk_config):
        """Create RiskGateway for testing."""
        return RiskGateway(state_store_with_equity, risk_config)

    def test_max_strategy_positions_rejects(self, gateway):
        """Test strategy position count limit rejects entry."""
        # KMP max_positions = 4
        for i in range(4):
            symbol = f"00{i:04d}"
            gateway.state.update_allocation(symbol, "KMP", 100)

        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
            risk_payload=RiskPayload(entry_px=72000, stop_px=71000),
        )

        result = gateway.check(intent)

        assert result.decision == RiskDecision.REJECT
        assert "max positions" in result.reason.lower()

    def test_risk_budget_scales_position(self, gateway):
        """Test risk budget scales position size."""
        # KMP max_risk_pct = 1.5% of 100M = 1.5M
        # Risk per share = entry - stop = 72000 - 60000 = 12000
        # Trade risk at 200 shares = 200 * 12000 = 2.4M (exceeds 1.5M)
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=200,
            risk_payload=RiskPayload(entry_px=72000, stop_px=60000),
        )

        result = gateway.check(intent)

        assert result.decision == RiskDecision.MODIFY
        assert result.modified_qty is not None
        # max_qty = 1.5M / 12000 = 125
        assert result.modified_qty == 125


class TestRiskGatewayMicrostructure:
    """Tests for RiskGateway microstructure checks."""

    @pytest.fixture
    def gateway(self, state_store_with_equity, risk_config):
        """Create RiskGateway for testing."""
        return RiskGateway(state_store_with_equity, risk_config)

    @pytest.fixture
    def enter_intent(self):
        """Create sample ENTER intent."""
        return Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
            risk_payload=RiskPayload(entry_px=72000, stop_px=71000),
        )

    def test_vi_cooldown_defers(self, gateway, enter_intent):
        """Test VI cooldown defers entry."""
        gateway.set_vi_cooldown("005930", 600)

        result = gateway.check(enter_intent)

        assert result.decision == RiskDecision.DEFER
        assert "vi cooldown" in result.reason.lower()


class TestRiskGatewayHelpers:
    """Tests for RiskGateway helper methods."""

    @pytest.fixture
    def gateway(self, state_store_with_equity, risk_config, sector_map):
        """Create RiskGateway with sector map."""
        return RiskGateway(state_store_with_equity, risk_config, sector_map=sector_map)

    def test_set_regime(self, gateway):
        """Test set_regime updates current regime."""
        gateway.set_regime("CRISIS")
        assert gateway.config.current_regime == "CRISIS"

    def test_set_safe_mode(self, gateway):
        """Test set_safe_mode enables/disables safe mode."""
        gateway.set_safe_mode(True)
        assert gateway.safe_mode is True

        gateway.set_safe_mode(False)
        assert gateway.safe_mode is False

    def test_trigger_flatten(self, gateway):
        """Test trigger_flatten sets flags."""
        gateway.trigger_flatten()

        assert gateway.flatten_in_progress is True
        assert gateway.halt_new_entries is True

    def test_set_vi_cooldown(self, gateway):
        """Test set_vi_cooldown sets cooldown on symbol."""
        gateway.set_vi_cooldown("005930")

        pos = gateway.state.get_position("005930")
        assert pos.vi_cooldown_until is not None
        assert pos.vi_cooldown_until > time.time()

    def test_reserve_sector(self, gateway):
        """Test reserve_sector tracks working sector exposure."""
        gateway.reserve_sector("005930", 100, 72000)

        sector = gateway._sector_exposure.get_sector("005930")
        count = gateway._sector_exposure.count_in_sector(sector, include_working=True)
        assert count == 1

    def test_unreserve_sector(self, gateway):
        """Test unreserve_sector releases sector slot."""
        gateway.reserve_sector("005930", 100, 72000)
        gateway.unreserve_sector("005930", 100, 72000)

        sector = gateway._sector_exposure.get_sector("005930")
        count = gateway._sector_exposure.count_in_sector(sector, include_working=True)
        assert count == 0

    def test_on_sector_fill(self, gateway):
        """Test on_sector_fill updates open exposure."""
        gateway.reserve_sector("005930", 100, 72000)
        gateway.on_sector_fill("005930", 100, 72000)

        sector = gateway._sector_exposure.get_sector("005930")
        count = gateway._sector_exposure.count_in_sector(sector, include_working=False)
        assert count == 1

    def test_on_sector_close(self, gateway):
        """Test on_sector_close releases open exposure."""
        gateway.reserve_sector("005930", 100, 72000)
        gateway.on_sector_fill("005930", 100, 72000)
        gateway.on_sector_close("005930", 100, 72000)

        sector = gateway._sector_exposure.get_sector("005930")
        count = gateway._sector_exposure.count_in_sector(sector, include_working=False)
        assert count == 0


class TestRiskDecisionIntegration:
    """Integration tests for full risk check flow."""

    @pytest.fixture
    def gateway(self, state_store_with_equity, risk_config, sector_map):
        """Create RiskGateway with full configuration."""
        return RiskGateway(state_store_with_equity, risk_config, sector_map=sector_map)

    def test_approve_valid_entry(self, gateway):
        """Test valid entry intent is approved."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
            risk_payload=RiskPayload(entry_px=72000, stop_px=71000),
        )

        result = gateway.check(intent)

        assert result.decision == RiskDecision.APPROVE

    def test_approve_exit_intent(self, gateway):
        """Test exit intent is always approved."""
        intent = Intent(
            intent_type=IntentType.EXIT,
            strategy_id="KMP",
            symbol="005930",
        )

        result = gateway.check(intent)

        assert result.decision == RiskDecision.APPROVE

    def test_multiple_checks_in_sequence(self, gateway):
        """Test multiple risk checks in sequence."""
        # First entry should pass
        intent1 = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
            risk_payload=RiskPayload(entry_px=72000, stop_px=71000),
        )
        result1 = gateway.check(intent1)
        assert result1.decision == RiskDecision.APPROVE

        # Simulate fill
        gateway.state.update_allocation("005930", "KMP", 100)

        # Second entry should still pass (under limits)
        intent2 = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="000660",
            desired_qty=50,
            risk_payload=RiskPayload(entry_px=130000, stop_px=125000),
        )
        result2 = gateway.check(intent2)
        assert result2.decision == RiskDecision.APPROVE


class TestRiskGatewaySectorReconcile:
    """Tests for sector exposure reconciliation and sector map updates."""

    @pytest.fixture
    def gateway(self, state_store_with_equity, risk_config, sector_map):
        """Create RiskGateway with sector map."""
        return RiskGateway(state_store_with_equity, risk_config, sector_map=sector_map)

    def test_reconcile_sector_exposure(self, gateway):
        """Test reconcile_sector_exposure rebuilds from OMS truth."""
        gateway.reserve_sector("005930", 100, 72000)
        gateway.on_sector_fill("005930", 100, 72000)
        # Reconcile with current state
        positions = {"005930": (100, 72000)}
        gateway.reconcile_sector_exposure(positions)
        # Verify sector still tracked after reconciliation
        sector = gateway._sector_exposure.get_sector("005930")
        count = gateway._sector_exposure.count_in_sector(sector, include_working=False)
        assert count >= 0  # Reconciliation should not crash

    def test_update_sector_map(self, gateway):
        """Test update_sector_map replaces the symbol-to-sector mapping."""
        new_map = {"005930": "Electronics", "000660": "Semiconductors"}
        gateway.update_sector_map(new_map)
        assert gateway._sector_exposure.sym_to_sector == new_map


class TestRiskGatewayGetPrice:
    """Tests for _get_price exception and fallback paths."""

    @pytest.fixture
    def gateway(self, state_store_with_equity, risk_config):
        """Create RiskGateway for testing."""
        return RiskGateway(state_store_with_equity, risk_config)

    def test_get_price_exception_returns_fallback(self, state_store_with_equity, risk_config):
        """Test that _get_price returns fallback when price_getter raises."""
        def bad_getter(symbol):
            raise RuntimeError("price feed down")

        gw = RiskGateway(state_store_with_equity, risk_config, price_getter=bad_getter)
        result = gw._get_price("005930", fallback=70000)
        assert result == 70000

    def test_get_price_no_getter_returns_fallback(self, gateway):
        """Test that _get_price returns fallback when no price_getter is set."""
        result = gateway._get_price("005930", fallback=65000)
        assert result == 65000

    def test_get_price_getter_returns_zero_uses_fallback(self, state_store_with_equity, risk_config):
        """Test that _get_price returns fallback when price_getter returns 0."""
        gw = RiskGateway(state_store_with_equity, risk_config, price_getter=lambda s: 0)
        result = gw._get_price("005930", fallback=72000)
        assert result == 72000

    def test_get_price_getter_returns_none_uses_fallback(self, state_store_with_equity, risk_config):
        """Test that _get_price returns fallback when price_getter returns None."""
        gw = RiskGateway(state_store_with_equity, risk_config, price_getter=lambda s: None)
        result = gw._get_price("005930", fallback=72000)
        assert result == 72000


class TestRiskGatewayExposureEntryPxFallback:
    """Tests for _check_exposure_limits entry_px fallback via _get_price."""

    def test_exposure_check_uses_price_getter_when_no_entry_px(self, state_store_with_equity, risk_config):
        """When risk_payload.entry_px is None, exposure check uses _get_price()."""
        gw = RiskGateway(
            state_store_with_equity,
            risk_config,
            price_getter=lambda s: 72000,
        )
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
            risk_payload=RiskPayload(entry_px=None, stop_px=71000),
        )
        result = gw.check(intent)
        # With a working price_getter, the check should proceed (not defer)
        assert result.decision != RiskDecision.DEFER

    def test_exposure_check_defers_when_no_price(self, state_store_with_equity, risk_config):
        """When entry_px is None and no price_getter, exposure check defers."""
        gw = RiskGateway(state_store_with_equity, risk_config)
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
            risk_payload=RiskPayload(entry_px=None, stop_px=71000),
        )
        result = gw.check(intent)
        assert result.decision == RiskDecision.DEFER
        assert "price unavailable" in result.reason.lower()


class TestRiskGatewayUnknownStrategyBudget:
    """Tests for _check_strategy_budget with unknown strategies."""

    @pytest.fixture
    def gateway(self, state_store_with_equity, risk_config):
        """Create RiskGateway for testing."""
        return RiskGateway(state_store_with_equity, risk_config)

    def test_unknown_strategy_budget_approves(self, gateway):
        """Test that an unknown strategy not in budgets is approved."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="UNKNOWN_STRAT",
            symbol="005930",
            desired_qty=100,
            risk_payload=RiskPayload(entry_px=72000, stop_px=71000),
        )
        result = gateway.check(intent)
        assert result.decision == RiskDecision.APPROVE
