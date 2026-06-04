"""Tests for FilterDecisionEvent and FilterLogger."""
import json
import tempfile
from pathlib import Path

from instrumentation.src.filter_logger import FilterLogger, FilterDecisionEvent
from instrumentation.src.lineage import LineageContext


class TestFilterDecisionEvent:
    def test_event_id_deterministic(self):
        """Same inputs produce same event_id."""
        e1 = FilterDecisionEvent(
            bot_id="bot1", pair="005930", timestamp="2026-03-15T09:17:00",
            filter_name="rvol_min", passed=True, threshold=2.0, actual_value=3.2,
        )
        e2 = FilterDecisionEvent(
            bot_id="bot1", pair="005930", timestamp="2026-03-15T09:17:00",
            filter_name="rvol_min", passed=True, threshold=2.0, actual_value=3.2,
        )
        assert e1.event_id == e2.event_id
        assert len(e1.event_id) == 16

    def test_margin_pct_computed_correctly(self):
        """(actual - threshold) / |threshold| * 100."""
        e = FilterDecisionEvent(
            bot_id="b", pair="p", timestamp="t",
            filter_name="rvol_min", passed=True,
            threshold=2.0, actual_value=3.2,
        )
        assert e.margin_pct == 60.0  # (3.2 - 2.0) / 2.0 * 100

    def test_margin_pct_negative_when_blocked(self):
        """Negative margin when actual below threshold."""
        e = FilterDecisionEvent(
            bot_id="b", pair="p", timestamp="t",
            filter_name="rvol_min", passed=False,
            threshold=2.0, actual_value=1.0,
        )
        assert e.margin_pct == -50.0  # (1.0 - 2.0) / 2.0 * 100

    def test_margin_pct_none_for_boolean_filters(self):
        """threshold == 0 returns None."""
        e = FilterDecisionEvent(
            bot_id="b", pair="p", timestamp="t",
            filter_name="vi_blocked", passed=False,
            threshold=0.0, actual_value=1.0,
        )
        assert e.margin_pct is None

    def test_to_dict_includes_margin(self):
        """to_dict includes computed margin_pct."""
        e = FilterDecisionEvent(
            bot_id="b", pair="p", timestamp="t",
            filter_name="f", passed=True, threshold=100.0, actual_value=150.0,
        )
        d = e.to_dict()
        assert "margin_pct" in d
        assert d["margin_pct"] == 50.0


class TestFilterLogger:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_filter_event_written_to_jsonl(self):
        """log_decision writes valid JSON line."""
        lg = FilterLogger(data_dir=self.tmpdir, bot_id="test_bot")
        event = lg.log_decision(
            pair="005930", filter_name="rvol_min",
            passed=True, threshold=2.0, actual_value=3.2,
            signal_name="alpha_value_surge", strategy_type="alpha",
        )
        assert event.passed is True

        files = list(Path(self.tmpdir).joinpath("filter_decisions").glob("*.jsonl"))
        assert len(files) == 1
        data = json.loads(files[0].read_text().strip())
        assert data["filter_name"] == "rvol_min"
        assert data["margin_pct"] == 60.0

    def test_pass_and_block_both_captured(self):
        """Events emitted regardless of passed value."""
        lg = FilterLogger(data_dir=self.tmpdir, bot_id="test_bot")
        lg.log_decision(pair="A", filter_name="f1", passed=True, threshold=1.0, actual_value=2.0)
        lg.log_decision(pair="B", filter_name="f2", passed=False, threshold=1.0, actual_value=0.5)

        files = list(Path(self.tmpdir).joinpath("filter_decisions").glob("*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text().strip().split("\n")
        assert len(lines) == 2

        d1 = json.loads(lines[0])
        d2 = json.loads(lines[1])
        assert d1["passed"] is True
        assert d2["passed"] is False

    def test_filter_event_carries_full_lineage(self):
        lineage = LineageContext(
            strategy_id="KALCB",
            deployment_id="deploy-unit",
            code_sha="abc123",
            strategy_version="strategy-unit",
            config_version="cfg-unit",
            portfolio_config_version="portfolio-unit",
            risk_config_version="risk-unit",
            allocation_version="allocation-unit",
            strategy_registry_version="registry-unit",
        )
        lg = FilterLogger(data_dir=self.tmpdir, bot_id="test_bot", lineage=lineage)
        event = lg.log_decision(
            pair="005930",
            filter_name="spread_gate",
            passed=False,
            threshold=50.0,
            actual_value=75.0,
            strategy_type="KALCB",
            event_ref="event-unit",
            decision_id="decision-unit",
            input_refs=["spread_bps"],
        )

        assert event.deployment_id == "deploy-unit"
        assert event.strategy_version == "strategy-unit"
        assert event.config_version == "cfg-unit"
        data = json.loads(next(Path(self.tmpdir).joinpath("filter_decisions").glob("*.jsonl")).read_text().strip())
        assert data["deployment_id"] == "deploy-unit"
        assert data["risk_config_version"] == "risk-unit"
        assert data["event_metadata"]["deployment_id"] == "deploy-unit"
