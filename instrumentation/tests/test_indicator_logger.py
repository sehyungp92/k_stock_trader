"""Tests for IndicatorSnapshot and IndicatorLogger."""
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from instrumentation.src.indicator_logger import IndicatorLogger, IndicatorSnapshot
from instrumentation.src.lineage import LineageContext


class TestIndicatorSnapshot:
    def test_event_id_deterministic(self):
        """Same inputs produce same event_id."""
        s1 = IndicatorSnapshot(
            bot_id="bot1", pair="005930", timestamp="2026-03-15T09:17:00",
            indicators={"sma_20": 72500.0}, signal_name="alpha_value_surge",
            signal_strength=0.78, decision="enter", strategy_type="alpha",
        )
        s2 = IndicatorSnapshot(
            bot_id="bot1", pair="005930", timestamp="2026-03-15T09:17:00",
            indicators={"sma_20": 72500.0}, signal_name="alpha_value_surge",
            signal_strength=0.78, decision="enter", strategy_type="alpha",
        )
        assert s1.event_id == s2.event_id
        assert len(s1.event_id) == 16

    def test_different_inputs_different_ids(self):
        """Different inputs produce different event_ids."""
        s1 = IndicatorSnapshot(
            bot_id="bot1", pair="005930", timestamp="2026-03-15T09:17:00",
            indicators={}, signal_name="alpha_value_surge",
            signal_strength=0.78, decision="enter", strategy_type="alpha",
        )
        s2 = IndicatorSnapshot(
            bot_id="bot1", pair="005931", timestamp="2026-03-15T09:17:00",
            indicators={}, signal_name="alpha_value_surge",
            signal_strength=0.78, decision="enter", strategy_type="alpha",
        )
        assert s1.event_id != s2.event_id

    def test_to_dict_roundtrip(self):
        """All indicator values captured in dict."""
        indicators = {"sma_20": 72500.0, "atr_14": 1850.0, "rvol": 3.2}
        s = IndicatorSnapshot(
            bot_id="bot1", pair="005930", timestamp="2026-03-15T09:17:00",
            indicators=indicators, signal_name="alpha_value_surge",
            signal_strength=0.78, decision="enter", strategy_type="alpha",
            bar_id="bar1", context={"extra": "value"},
        )
        d = s.to_dict()
        assert d["indicators"] == indicators
        assert d["signal_name"] == "alpha_value_surge"
        assert d["bar_id"] == "bar1"
        assert d["context"] == {"extra": "value"}

    def test_decision_field(self):
        """Decision field stored correctly."""
        for decision in ("enter", "skip", "exit"):
            s = IndicatorSnapshot(
                bot_id="b", pair="p", timestamp="t",
                indicators={}, signal_name="s",
                signal_strength=0.0, decision=decision, strategy_type="alpha",
            )
            assert s.decision == decision


class TestIndicatorLogger:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_snapshot_written_to_jsonl(self):
        """log_snapshot writes valid JSON line to correct file."""
        lg = IndicatorLogger(data_dir=self.tmpdir, bot_id="test_bot")
        snap = lg.log_snapshot(
            pair="005930",
            indicators={"sma_20": 72500.0, "atr_14": 1850.0},
            signal_name="alpha_value_surge",
            signal_strength=0.78,
            decision="enter",
            strategy_type="alpha",
        )
        assert snap.bot_id == "test_bot"

        files = list(Path(self.tmpdir).joinpath("indicators").glob("*.jsonl"))
        assert len(files) == 1
        data = json.loads(files[0].read_text().strip())
        assert data["pair"] == "005930"
        assert data["indicators"]["sma_20"] == 72500.0

    def test_exchange_timestamp_used(self):
        """Exchange timestamp overrides default."""
        lg = IndicatorLogger(data_dir=self.tmpdir, bot_id="test_bot")
        ts = datetime(2026, 3, 15, 9, 17, 0, tzinfo=timezone.utc)
        snap = lg.log_snapshot(
            pair="005930", indicators={}, signal_name="s",
            signal_strength=0.0, decision="skip", strategy_type="alpha",
            exchange_timestamp=ts,
        )
        assert "2026-03-15T09:17:00" in snap.timestamp

    def test_strategy_specific_indicators(self):
        """ALPHA snapshot has SMA/ATR/RVol, BETA has VWAP fields."""
        lg = IndicatorLogger(data_dir=self.tmpdir, bot_id="test_bot")

        alpha = lg.log_snapshot(
            pair="005930",
            indicators={"sma_20": 72500.0, "atr_14": 1850.0, "rvol": 3.2},
            signal_name="alpha_value_surge", signal_strength=0.8,
            decision="enter", strategy_type="alpha",
        )
        assert "sma_20" in alpha.indicators

        beta = lg.log_snapshot(
            pair="005930",
            indicators={"vwap": 71000.0, "vwap_depth_pct": 0.03},
            signal_name="beta_vwap_pullback", signal_strength=0.6,
            decision="enter", strategy_type="beta",
        )
        assert "vwap" in beta.indicators

    def test_indicator_snapshot_carries_full_lineage(self):
        lineage = LineageContext(
            strategy_id="OLR",
            deployment_id="deploy-unit",
            code_sha="abc123",
            strategy_version="strategy-unit",
            config_version="cfg-unit",
            portfolio_config_version="portfolio-unit",
            risk_config_version="risk-unit",
            allocation_version="allocation-unit",
            strategy_registry_version="registry-unit",
        )
        lg = IndicatorLogger(data_dir=self.tmpdir, bot_id="test_bot", lineage=lineage)
        snap = lg.log_snapshot(
            pair="005930",
            indicators={"vwap_ret": 0.012, "range_atr": 1.4},
            signal_name="olr_afternoon",
            signal_strength=0.7,
            decision="enter",
            strategy_type="OLR",
            event_ref="event-unit",
            decision_id="decision-unit",
        )

        assert snap.deployment_id == "deploy-unit"
        assert snap.strategy_version == "strategy-unit"
        assert snap.config_version == "cfg-unit"
        data = json.loads(next(Path(self.tmpdir).joinpath("indicators").glob("*.jsonl")).read_text().strip())
        assert data["deployment_id"] == "deploy-unit"
        assert data["risk_config_version"] == "risk-unit"
        assert data["event_metadata"]["deployment_id"] == "deploy-unit"
