"""Tests for order_logger module."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from instrumentation.src.order_logger import OrderEvent, OrderLogger


class TestOrderEvent:
    def test_construction_all_fields(self):
        event = OrderEvent(
            order_id="KIS_001",
            bot_id="test_bot",
            pair="005930",
            side="LONG",
            order_type="MARKET",
            status="FILLED",
            requested_qty=10,
            filled_qty=10,
            fill_price=72500,
            timestamp="2026-03-06T14:30:00+09:00",
        )
        assert event.order_id == "KIS_001"
        assert event.bot_id == "test_bot"
        assert event.pair == "005930"
        assert event.side == "LONG"
        assert event.status == "FILLED"

    def test_to_dict_excludes_none(self):
        event = OrderEvent(
            order_id="KIS_001",
            bot_id="test_bot",
            pair="005930",
        )
        d = event.to_dict()
        assert "order_id" in d
        # requested_price is None, should be excluded
        assert "requested_price" not in d
        assert "fill_price" not in d
        assert "slippage_bps" not in d
        assert "latency_ms" not in d

    def test_slippage_computation_in_to_dict(self):
        """Slippage is computed during log_order, not in OrderEvent itself."""
        event = OrderEvent(
            order_id="KIS_001",
            bot_id="test_bot",
            pair="005930",
            fill_price=72600,
            requested_price=72500,
            slippage_bps=13.79,  # pre-computed
        )
        d = event.to_dict()
        assert d["slippage_bps"] == 13.79

    def test_market_order_no_slippage(self):
        """MARKET order with no requested_price → slippage_bps stays None."""
        event = OrderEvent(
            order_id="KIS_001",
            bot_id="test_bot",
            pair="005930",
            order_type="MARKET",
            fill_price=72500,
            requested_price=None,
        )
        d = event.to_dict()
        assert "slippage_bps" not in d

    def test_rejected_status(self):
        event = OrderEvent(
            order_id="KIS_001",
            bot_id="test_bot",
            pair="005930",
            status="REJECTED",
            requested_qty=10,
            filled_qty=0.0,
            reject_reason="exposure_limit_exceeded",
        )
        d = event.to_dict()
        assert d["status"] == "REJECTED"
        assert d["reject_reason"] == "exposure_limit_exceeded"
        assert d["filled_qty"] == 0.0


class TestOrderLogger:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = {
            "bot_id": "test_bot",
            "data_dir": self.tmpdir,
        }
        self.logger = OrderLogger(self.config)

    def test_log_order_creates_file(self):
        event = self.logger.log_order(
            order_id="KIS_001",
            pair="005930",
            order_type="MARKET",
            status="SUBMITTED",
            requested_qty=10,
        )
        assert event.order_id == "KIS_001"
        assert event.bot_id == "test_bot"

        # Verify file written
        order_dir = Path(self.tmpdir) / "orders"
        files = list(order_dir.glob("orders_*.jsonl"))
        assert len(files) == 1

        with open(files[0]) as f:
            record = json.loads(f.readline())
        assert record["order_id"] == "KIS_001"
        assert record["status"] == "SUBMITTED"

    def test_slippage_computed_on_fill(self):
        event = self.logger.log_order(
            order_id="KIS_002",
            pair="005930",
            order_type="LIMIT",
            status="FILLED",
            requested_qty=10,
            filled_qty=10,
            requested_price=72500,
            fill_price=72600,
        )
        # (72600 - 72500) / 72500 * 10000 = 13.79 bps
        assert event.slippage_bps == 13.79

    def test_slippage_none_for_market_order(self):
        event = self.logger.log_order(
            order_id="KIS_003",
            pair="005930",
            order_type="MARKET",
            status="FILLED",
            requested_qty=10,
            filled_qty=10,
            fill_price=72500,
            requested_price=None,
        )
        assert event.slippage_bps is None

    def test_rejected_order(self):
        event = self.logger.log_order(
            order_id="KIS_004",
            pair="005930",
            order_type="LIMIT",
            status="REJECTED",
            requested_qty=10,
            reject_reason="exposure_limit",
        )
        assert event.status == "REJECTED"
        assert event.reject_reason == "exposure_limit"
        assert event.filled_qty == 0.0

    def test_jsonl_write_valid_format(self):
        """Each line should be valid JSON."""
        self.logger.log_order(
            order_id="KIS_A", pair="005930", order_type="MARKET",
            status="SUBMITTED", requested_qty=5,
        )
        self.logger.log_order(
            order_id="KIS_A", pair="005930", order_type="MARKET",
            status="FILLED", requested_qty=5, filled_qty=5, fill_price=72500,
        )

        order_dir = Path(self.tmpdir) / "orders"
        files = list(order_dir.glob("orders_*.jsonl"))
        with open(files[0]) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 2
        for line in lines:
            record = json.loads(line)
            assert "order_id" in record

    def test_event_metadata_populated(self):
        event = self.logger.log_order(
            order_id="KIS_005",
            pair="005930",
            order_type="MARKET",
            status="SUBMITTED",
            requested_qty=10,
            exchange_timestamp=datetime(2026, 3, 6, 5, 30, tzinfo=timezone.utc),
        )
        assert "event_id" in event.event_metadata
        assert "bot_id" in event.event_metadata
        assert event.event_metadata["bot_id"] == "test_bot"

    def test_experiment_fields_propagated(self):
        config = {
            "bot_id": "test_bot",
            "data_dir": self.tmpdir,
            "experiment_id": "exp_fast_ma",
            "experiment_variant": "control",
        }
        logger = OrderLogger(config)
        event = logger.log_order(
            order_id="KIS_006",
            pair="005930",
            order_type="MARKET",
            status="SUBMITTED",
            requested_qty=10,
        )
        assert event.experiment_id == "exp_fast_ma"
        assert event.experiment_variant == "control"

    def test_orders_directory_created(self):
        order_dir = Path(self.tmpdir) / "orders"
        assert order_dir.exists()
