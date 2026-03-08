"""Tests for OrderBookContext and OrderBookLogger."""
import json
import tempfile
from pathlib import Path

from instrumentation.src.orderbook_logger import OrderBookLogger, OrderBookContext


class TestOrderBookContext:
    def test_spread_bps_auto_computed(self):
        """spread_bps computed from best_bid/best_ask."""
        ctx = OrderBookContext(
            bot_id="b", pair="p", timestamp="t",
            best_bid=72400.0, best_ask=72500.0,
        )
        mid = (72400 + 72500) / 2
        expected = (72500 - 72400) / mid * 10000
        assert abs(ctx.spread_bps - round(expected, 2)) < 0.01

    def test_spread_bps_zero_when_provided(self):
        """If spread_bps already set, don't overwrite."""
        ctx = OrderBookContext(
            bot_id="b", pair="p", timestamp="t",
            best_bid=72400.0, best_ask=72500.0, spread_bps=15.0,
        )
        assert ctx.spread_bps == 15.0

    def test_imbalance_ratio_computed(self):
        """bid_depth / ask_depth."""
        ctx = OrderBookContext(
            bot_id="b", pair="p", timestamp="t",
            best_bid=100.0, best_ask=101.0,
            bid_depth_10bps=5000.0, ask_depth_10bps=2500.0,
        )
        assert ctx.imbalance_ratio == 2.0

    def test_imbalance_ratio_zero_no_depth(self):
        """Returns 0.0 when no depth data."""
        ctx = OrderBookContext(
            bot_id="b", pair="p", timestamp="t",
            best_bid=100.0, best_ask=101.0,
        )
        assert ctx.imbalance_ratio == 0.0

    def test_trade_context_values(self):
        """trade_context field correctly set."""
        for context in ("entry", "exit", "signal_eval"):
            ctx = OrderBookContext(
                bot_id="b", pair="p", timestamp="t",
                best_bid=100.0, best_ask=101.0,
                trade_context=context,
            )
            assert ctx.trade_context == context

    def test_to_dict_includes_imbalance(self):
        """to_dict includes computed imbalance_ratio."""
        ctx = OrderBookContext(
            bot_id="b", pair="p", timestamp="t",
            best_bid=100.0, best_ask=101.0,
            bid_depth_10bps=1000.0, ask_depth_10bps=1000.0,
        )
        d = ctx.to_dict()
        assert "imbalance_ratio" in d
        assert d["imbalance_ratio"] == 1.0

    def test_event_id_deterministic(self):
        """Same inputs produce same event_id."""
        c1 = OrderBookContext(
            bot_id="b", pair="p", timestamp="t",
            best_bid=100.0, best_ask=101.0,
        )
        c2 = OrderBookContext(
            bot_id="b", pair="p", timestamp="t",
            best_bid=100.0, best_ask=101.0,
        )
        assert c1.event_id == c2.event_id


class TestOrderBookLogger:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_context_written_to_jsonl(self):
        """log_context writes valid JSON line."""
        lg = OrderBookLogger(data_dir=self.tmpdir, bot_id="test_bot")
        ctx = lg.log_context(
            pair="005930", best_bid=72400.0, best_ask=72500.0,
            trade_context="entry", related_trade_id="t_001",
        )
        assert ctx.bot_id == "test_bot"

        files = list(Path(self.tmpdir).joinpath("orderbook").glob("*.jsonl"))
        assert len(files) == 1
        data = json.loads(files[0].read_text().strip())
        assert data["pair"] == "005930"
        assert data["best_bid"] == 72400.0
        assert data["trade_context"] == "entry"
        assert data["related_trade_id"] == "t_001"
        assert data["spread_bps"] > 0
