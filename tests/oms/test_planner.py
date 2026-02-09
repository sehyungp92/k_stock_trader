"""Tests for OMS planner module."""

import pytest

from oms.planner import OrderPlanner, OrderPlan, OrderType
from oms.intent import Intent, IntentType, Urgency, TimeHorizon, IntentConstraints, RiskPayload


class TestOrderType:
    """Tests for OrderType enum."""

    def test_all_types_defined(self):
        """Test all order types are defined."""
        assert OrderType.MARKET
        assert OrderType.LIMIT
        assert OrderType.STOP_LIMIT
        assert OrderType.MARKETABLE_LIMIT


class TestOrderPlan:
    """Tests for OrderPlan dataclass."""

    def test_default_values(self):
        """Test default plan values."""
        plan = OrderPlan()

        assert plan.plan_id is not None
        assert plan.symbol == ""
        assert plan.side == ""
        assert plan.qty == 0
        assert plan.order_type == OrderType.LIMIT
        assert plan.limit_price is None
        assert plan.stop_price is None
        assert plan.submit_by is None
        assert plan.cancel_after is None
        assert plan.intent_ids == []
        assert plan.strategy_id == ""
        assert plan.max_chase_bps == 30.0

    def test_with_values(self):
        """Test plan with values."""
        plan = OrderPlan(
            symbol="005930",
            side="BUY",
            qty=100,
            order_type=OrderType.STOP_LIMIT,
            limit_price=72300,
            stop_price=72100,
            cancel_after=30.0,
            intent_ids=["intent-1"],
            strategy_id="KMP",
        )

        assert plan.symbol == "005930"
        assert plan.side == "BUY"
        assert plan.qty == 100
        assert plan.order_type == OrderType.STOP_LIMIT


class TestOrderPlannerCreatePlan:
    """Tests for OrderPlanner.create_plan method."""

    @pytest.fixture
    def planner(self):
        """Create OrderPlanner for testing."""
        return OrderPlanner()

    def test_stop_limit_for_breakout(self, planner):
        """Test stop-limit order for breakout entry."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
            constraints=IntentConstraints(
                stop_price=72100,
                limit_price=72300,
            ),
        )

        plan = planner.create_plan(
            symbol="005930",
            side="BUY",
            qty=100,
            intent=intent,
            current_price=72000,
        )

        assert plan.order_type == OrderType.STOP_LIMIT
        assert plan.stop_price == 72100
        assert plan.limit_price == 72300
        assert plan.cancel_after == 30.0

    def test_stop_limit_default_limit_price(self, planner):
        """Test stop-limit with default limit price calculation."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
            constraints=IntentConstraints(
                stop_price=72100,
                # No limit_price specified
            ),
        )

        plan = planner.create_plan(
            symbol="005930",
            side="BUY",
            qty=100,
            intent=intent,
            current_price=72000,
        )

        assert plan.order_type == OrderType.STOP_LIMIT
        # Default limit = stop * 1.003 = 72100 * 1.003 = 72316.3
        assert plan.limit_price == pytest.approx(72316.3, abs=1)

    def test_marketable_limit_for_high_urgency(self, planner):
        """Test marketable limit for high urgency."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
            urgency=Urgency.HIGH,
        )

        plan = planner.create_plan(
            symbol="005930",
            side="BUY",
            qty=100,
            intent=intent,
            current_price=72000,
        )

        assert plan.order_type == OrderType.MARKETABLE_LIMIT
        # BUY: limit = current * 1.002 = 72000 * 1.002 = 72144
        assert plan.limit_price == pytest.approx(72144, abs=1)
        assert plan.cancel_after == 10.0

    def test_marketable_limit_sell_side(self, planner):
        """Test marketable limit for sell side."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
            urgency=Urgency.HIGH,
        )

        plan = planner.create_plan(
            symbol="005930",
            side="SELL",
            qty=100,
            intent=intent,
            current_price=72000,
        )

        assert plan.order_type == OrderType.MARKETABLE_LIMIT
        # SELL: limit = current * 0.998 = 72000 * 0.998 = 71856
        assert plan.limit_price == pytest.approx(71856, abs=1)

    def test_standard_limit_for_normal_urgency(self, planner):
        """Test standard limit for normal urgency."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
            urgency=Urgency.NORMAL,
        )

        plan = planner.create_plan(
            symbol="005930",
            side="BUY",
            qty=100,
            intent=intent,
            current_price=72000,
        )

        assert plan.order_type == OrderType.LIMIT
        assert plan.limit_price == 72000
        assert plan.cancel_after == 120.0

    def test_limit_with_constraint_price(self, planner):
        """Test limit uses constraint price when available."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
            urgency=Urgency.NORMAL,
            constraints=IntentConstraints(limit_price=71500),
        )

        plan = planner.create_plan(
            symbol="005930",
            side="BUY",
            qty=100,
            intent=intent,
            current_price=72000,
        )

        assert plan.order_type == OrderType.LIMIT
        assert plan.limit_price == 71500

    def test_plan_attribution(self, planner):
        """Test plan attribution fields."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
        )

        plan = planner.create_plan(
            symbol="005930",
            side="BUY",
            qty=100,
            intent=intent,
            current_price=72000,
        )

        assert intent.intent_id in plan.intent_ids
        assert plan.strategy_id == "KMP"


class TestOrderPlannerCreateExitPlan:
    """Tests for OrderPlanner.create_exit_plan method."""

    @pytest.fixture
    def planner(self):
        """Create OrderPlanner for testing."""
        return OrderPlanner()

    def test_exit_plan_is_market(self, planner):
        """Test exit plan uses MARKET order."""
        plan = planner.create_exit_plan(
            symbol="005930",
            qty=100,
            strategy_id="KMP",
            intent_id="intent-1",
            urgency=Urgency.NORMAL,
        )

        assert plan.order_type == OrderType.MARKET

    def test_exit_plan_sell_side(self, planner):
        """Test exit plan is SELL side."""
        plan = planner.create_exit_plan(
            symbol="005930",
            qty=100,
            strategy_id="KMP",
            intent_id="intent-1",
            urgency=Urgency.NORMAL,
        )

        assert plan.side == "SELL"

    def test_exit_plan_short_timeout(self, planner):
        """Test exit plan has short cancel timeout."""
        plan = planner.create_exit_plan(
            symbol="005930",
            qty=100,
            strategy_id="KMP",
            intent_id="intent-1",
            urgency=Urgency.NORMAL,
        )

        assert plan.cancel_after == 5.0

    def test_exit_plan_attribution(self, planner):
        """Test exit plan attribution."""
        plan = planner.create_exit_plan(
            symbol="005930",
            qty=100,
            strategy_id="KMP",
            intent_id="intent-1",
            urgency=Urgency.HIGH,
        )

        assert plan.symbol == "005930"
        assert plan.qty == 100
        assert plan.strategy_id == "KMP"
        assert "intent-1" in plan.intent_ids
