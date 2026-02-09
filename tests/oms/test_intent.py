"""Tests for OMS intent module."""

import pytest
from datetime import datetime
import time
from unittest.mock import patch

from oms.intent import (
    Intent,
    IntentType,
    IntentStatus,
    IntentResult,
    Urgency,
    TimeHorizon,
    IntentConstraints,
    RiskPayload,
)


class TestIntentType:
    """Tests for IntentType enum."""

    def test_all_types_defined(self):
        """Test all intent types are defined."""
        assert IntentType.ENTER
        assert IntentType.REDUCE
        assert IntentType.EXIT
        assert IntentType.SET_TARGET
        assert IntentType.CANCEL_ORDERS
        assert IntentType.MODIFY_RISK
        assert IntentType.FLATTEN


class TestUrgency:
    """Tests for Urgency enum."""

    def test_all_urgencies_defined(self):
        """Test all urgency levels are defined."""
        assert Urgency.LOW
        assert Urgency.NORMAL
        assert Urgency.HIGH


class TestTimeHorizon:
    """Tests for TimeHorizon enum."""

    def test_all_horizons_defined(self):
        """Test all time horizons are defined."""
        assert TimeHorizon.INTRADAY
        assert TimeHorizon.SWING


class TestIntentStatus:
    """Tests for IntentStatus enum."""

    def test_all_statuses_defined(self):
        """Test all intent statuses are defined."""
        assert IntentStatus.PENDING
        assert IntentStatus.ACCEPTED
        assert IntentStatus.APPROVED
        assert IntentStatus.MODIFIED
        assert IntentStatus.REJECTED
        assert IntentStatus.DEFERRED
        assert IntentStatus.EXECUTED
        assert IntentStatus.CANCELLED


class TestIntentConstraints:
    """Tests for IntentConstraints dataclass."""

    def test_default_values(self):
        """Test default constraint values."""
        constraints = IntentConstraints()

        assert constraints.max_slippage_bps is None
        assert constraints.max_spread_bps is None
        assert constraints.limit_price is None
        assert constraints.stop_price is None
        assert constraints.expiry_ts is None

    def test_with_values(self):
        """Test constraints with values."""
        constraints = IntentConstraints(
            max_slippage_bps=30.0,
            max_spread_bps=50.0,
            limit_price=72300,
            stop_price=72100,
            expiry_ts=time.time() + 30,
        )

        assert constraints.max_slippage_bps == 30.0
        assert constraints.limit_price == 72300


class TestRiskPayload:
    """Tests for RiskPayload dataclass."""

    def test_default_values(self):
        """Test default risk payload values."""
        payload = RiskPayload()

        assert payload.entry_px is None
        assert payload.stop_px is None
        assert payload.hard_stop_px is None
        assert payload.rationale_code == ""
        assert payload.confidence == "YELLOW"

    def test_with_values(self):
        """Test risk payload with values."""
        payload = RiskPayload(
            entry_px=72000,
            stop_px=71000,
            hard_stop_px=70500,
            rationale_code="or_break",
            confidence="GREEN",
        )

        assert payload.entry_px == 72000
        assert payload.stop_px == 71000
        assert payload.confidence == "GREEN"


class TestIntent:
    """Tests for Intent dataclass."""

    def test_required_fields(self):
        """Test intent with only required fields."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
        )

        assert intent.intent_type == IntentType.ENTER
        assert intent.strategy_id == "KMP"
        assert intent.symbol == "005930"
        assert intent.desired_qty is None
        assert intent.target_qty is None

    def test_strategy_id_normalized(self):
        """Test strategy_id is normalized to uppercase."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="kmp",
            symbol="005930",
        )

        assert intent.strategy_id == "KMP"

    def test_strategy_id_stripped(self):
        """Test strategy_id is stripped of whitespace."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="  KMP  ",
            symbol="005930",
        )

        assert intent.strategy_id == "KMP"

    def test_default_urgency(self):
        """Test default urgency is NORMAL."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
        )

        assert intent.urgency == Urgency.NORMAL

    def test_default_time_horizon(self):
        """Test default time horizon is INTRADAY."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
        )

        assert intent.time_horizon == TimeHorizon.INTRADAY

    def test_auto_generated_intent_id(self):
        """Test intent_id is auto-generated."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
        )

        assert intent.intent_id is not None
        assert len(intent.intent_id) > 0

    def test_auto_generated_timestamp(self):
        """Test timestamp is auto-generated."""
        before = time.time()
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
        )
        after = time.time()

        assert intent.timestamp >= before
        assert intent.timestamp <= after


class TestIntentIdempotencyKey:
    """Tests for Intent idempotency key generation."""

    @patch('oms.intent._kst_trade_date')
    def test_enter_with_signal_hash(self, mock_date):
        """Test ENTER intent uses signal_hash in idempotency key."""
        mock_date.return_value = "20240115"

        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            signal_hash="test_signal_123",
        )

        assert intent.idempotency_key == "KMP:005930:ENTER:20240115:test_signal_123:0"

    @patch('oms.intent._kst_trade_date')
    def test_enter_with_rationale_fallback(self, mock_date):
        """Test ENTER intent falls back to rationale_code."""
        mock_date.return_value = "20240115"

        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            risk_payload=RiskPayload(rationale_code="or_break"),
        )

        assert intent.idempotency_key == "KMP:005930:ENTER:20240115:or_break:0"

    @patch('oms.intent._kst_trade_date')
    def test_enter_default_suffix(self, mock_date):
        """Test ENTER intent uses 'default' when no hash or rationale."""
        mock_date.return_value = "20240115"

        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
        )

        assert intent.idempotency_key == "KMP:005930:ENTER:20240115:default:0"

    @patch('oms.intent._kst_trade_date')
    def test_exit_with_rationale(self, mock_date):
        """Test EXIT intent uses rationale_code."""
        mock_date.return_value = "20240115"

        intent = Intent(
            intent_type=IntentType.EXIT,
            strategy_id="KMP",
            symbol="005930",
            risk_payload=RiskPayload(rationale_code="stop_hit"),
        )

        assert intent.idempotency_key == "KMP:005930:EXIT:20240115:stop_hit:0"

    @patch('oms.intent._kst_trade_date')
    def test_exit_default_rationale(self, mock_date):
        """Test EXIT intent uses 'manual' when no rationale."""
        mock_date.return_value = "20240115"

        intent = Intent(
            intent_type=IntentType.EXIT,
            strategy_id="KMP",
            symbol="005930",
        )

        assert intent.idempotency_key == "KMP:005930:EXIT:20240115:manual:0"

    @patch('oms.intent._kst_trade_date')
    def test_operational_intent_unique(self, mock_date):
        """Test operational intents use unique suffix."""
        mock_date.return_value = "20240115"

        intent1 = Intent(
            intent_type=IntentType.CANCEL_ORDERS,
            strategy_id="KMP",
            symbol="005930",
        )

        intent2 = Intent(
            intent_type=IntentType.CANCEL_ORDERS,
            strategy_id="KMP",
            symbol="005930",
        )

        # Each operational intent should have unique key
        assert intent1.idempotency_key != intent2.idempotency_key

    def test_custom_idempotency_key_preserved(self):
        """Test custom idempotency key is preserved."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            idempotency_key="custom:key:123",
        )

        assert intent.idempotency_key == "custom:key:123"


class TestIntentValidation:
    """Tests for Intent validation."""

    def test_valid_enter_intent(self):
        """Test valid ENTER intent passes validation."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
        )

        valid, error = intent.validate()
        assert valid is True
        assert error == ""

    def test_missing_symbol(self):
        """Test missing symbol fails validation."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="",
            desired_qty=100,
        )

        valid, error = intent.validate()
        assert valid is False
        assert "symbol" in error.lower()

    def test_missing_strategy_id(self):
        """Test missing strategy_id fails validation."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="",
            symbol="005930",
            desired_qty=100,
        )

        valid, error = intent.validate()
        assert valid is False
        assert "strategy_id" in error.lower()

    def test_enter_without_qty(self):
        """Test ENTER without qty fails validation."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
        )

        valid, error = intent.validate()
        assert valid is False
        assert "qty" in error.lower()

    def test_enter_with_target_qty(self):
        """Test ENTER with target_qty passes validation."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            target_qty=100,
        )

        valid, error = intent.validate()
        assert valid is True

    def test_reduce_without_qty(self):
        """Test REDUCE without qty fails validation."""
        intent = Intent(
            intent_type=IntentType.REDUCE,
            strategy_id="KMP",
            symbol="005930",
        )

        valid, error = intent.validate()
        assert valid is False

    def test_exit_without_qty_ok(self):
        """Test EXIT without qty passes validation."""
        intent = Intent(
            intent_type=IntentType.EXIT,
            strategy_id="KMP",
            symbol="005930",
        )

        valid, error = intent.validate()
        assert valid is True

    def test_expired_intent(self):
        """Test expired intent fails validation."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
            constraints=IntentConstraints(expiry_ts=time.time() - 10),
        )

        valid, error = intent.validate()
        assert valid is False
        assert "expired" in error.lower()

    def test_future_expiry_ok(self):
        """Test future expiry passes validation."""
        intent = Intent(
            intent_type=IntentType.ENTER,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=100,
            constraints=IntentConstraints(expiry_ts=time.time() + 30),
        )

        valid, error = intent.validate()
        assert valid is True


class TestIntentResult:
    """Tests for IntentResult dataclass."""

    def test_required_fields(self):
        """Test IntentResult with required fields."""
        result = IntentResult(
            intent_id="test-id",
            status=IntentStatus.EXECUTED,
        )

        assert result.intent_id == "test-id"
        assert result.status == IntentStatus.EXECUTED
        assert result.message == ""
        assert result.modified_qty is None
        assert result.order_id is None
        assert result.cooldown_until is None

    def test_with_all_fields(self):
        """Test IntentResult with all fields."""
        result = IntentResult(
            intent_id="test-id",
            status=IntentStatus.MODIFIED,
            message="Scaled to 100",
            modified_qty=100,
            order_id="ORD001",
            cooldown_until=time.time() + 60,
        )

        assert result.message == "Scaled to 100"
        assert result.modified_qty == 100
        assert result.order_id == "ORD001"
        assert result.cooldown_until is not None


class TestIntentIdempotencyExtended:
    """Extended tests for idempotency key generation across intent types."""

    @patch('oms.intent._kst_trade_date')
    def test_flatten_with_rationale(self, mock_date):
        """Test FLATTEN intent uses rationale_code in idempotency key."""
        mock_date.return_value = "20240115"
        intent = Intent(
            intent_type=IntentType.FLATTEN,
            strategy_id="KMP",
            symbol="005930",
            risk_payload=RiskPayload(rationale_code="emergency"),
        )
        assert "FLATTEN:20240115:emergency" in intent.idempotency_key

    @patch('oms.intent._kst_trade_date')
    def test_reduce_with_rationale(self, mock_date):
        """Test REDUCE intent uses rationale_code in idempotency key."""
        mock_date.return_value = "20240115"
        intent = Intent(
            intent_type=IntentType.REDUCE,
            strategy_id="KMP",
            symbol="005930",
            desired_qty=50,
            risk_payload=RiskPayload(rationale_code="partial_take"),
        )
        assert "REDUCE:20240115:partial_take" in intent.idempotency_key

    @patch('oms.intent._kst_trade_date')
    def test_set_target_unique(self, mock_date):
        """Test SET_TARGET intent uses unique suffix (operational intent)."""
        mock_date.return_value = "20240115"
        intent = Intent(
            intent_type=IntentType.SET_TARGET,
            strategy_id="KMP",
            symbol="005930",
            target_qty=200,
        )
        assert "SET_TARGET" in intent.idempotency_key
        # Operational intents use unique suffix from intent_id
        assert intent.intent_id[:8] in intent.idempotency_key

    @patch('oms.intent._kst_trade_date')
    def test_modify_risk_unique(self, mock_date):
        """Test MODIFY_RISK intent uses unique suffix (operational intent)."""
        mock_date.return_value = "20240115"
        intent = Intent(
            intent_type=IntentType.MODIFY_RISK,
            strategy_id="KMP",
            symbol="005930",
        )
        assert "MODIFY_RISK" in intent.idempotency_key
        # Each MODIFY_RISK should have unique key
        assert intent.intent_id[:8] in intent.idempotency_key
