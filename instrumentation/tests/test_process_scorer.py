"""Tests for process_scorer module."""

import tempfile
from pathlib import Path

import yaml

from instrumentation.src.process_scorer import ProcessScorer, ProcessScore, ROOT_CAUSES


def _make_rules():
    return {
        "global": {
            "max_entry_latency_ms": 5000,
            "max_slippage_multiplier": 2.0,
            "expected_slippage_bps": 10,
            "min_signal_strength": 0.3,
            "strong_signal_threshold": 0.7,
        },
        "strategies": {
            "kmp": {
                "preferred_regimes": ["trending_up"],
                "adverse_regimes": ["ranging", "trending_down"],
                "expected_slippage_bps": 5,
            },
            "kpr": {
                "preferred_regimes": ["volatile", "trending_down"],
                "adverse_regimes": ["trending_up"],
            },
        },
    }


class TestProcessScorer:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.rules_path = Path(self.tmpdir) / "rules.yaml"
        with open(self.rules_path, "w") as f:
            yaml.dump(_make_rules(), f)
        self.scorer = ProcessScorer(str(self.rules_path))

    def test_perfect_trade_scores_high(self):
        trade = {
            "trade_id": "t1",
            "regime": "trending_up",
            "signal_strength": 0.85,
            "entry_latency_ms": 200,
            "entry_slippage_bps": 3.0,
            "exit_slippage_bps": 3.0,
            "exit_reason": "TAKE_PROFIT",
            "pnl": 500,
        }
        score = self.scorer.score_trade(trade, "kmp")
        assert score.process_quality_score >= 80
        assert score.classification == "good_process"
        assert len(score.positive_factors) > 0

    def test_bad_trade_scores_low(self):
        trade = {
            "trade_id": "t2",
            "regime": "ranging",
            "signal_strength": 0.1,
            "entry_latency_ms": 10000,
            "entry_slippage_bps": 30.0,
            "exit_slippage_bps": 25.0,
            "exit_reason": "STOP_LOSS",
            "pnl": -200,
        }
        score = self.scorer.score_trade(trade, "kmp")
        assert score.process_quality_score < 50
        assert "regime_mismatch" in score.root_causes
        assert "weak_signal" in score.root_causes
        assert "late_entry" in score.root_causes
        assert score.classification == "bad_process"

    def test_regime_mismatch_penalty(self):
        trade = {
            "trade_id": "t3",
            "regime": "ranging",
            "signal_strength": 0.8,
            "pnl": 100,
        }
        score = self.scorer.score_trade(trade, "kmp")
        assert "regime_mismatch" in score.root_causes
        # -20 from regime, starts at 100, so <= 80
        assert score.process_quality_score <= 80

    def test_weak_signal_penalty(self):
        trade = {
            "trade_id": "t4",
            "signal_strength": 0.1,
            "pnl": 0,
        }
        score = self.scorer.score_trade(trade, "kmp")
        assert "weak_signal" in score.root_causes

    def test_late_entry_penalty(self):
        trade = {
            "trade_id": "t5",
            "entry_latency_ms": 10000,
            "pnl": 0,
        }
        score = self.scorer.score_trade(trade, "kmp")
        assert "late_entry" in score.root_causes

    def test_high_slippage_penalty(self):
        trade = {
            "trade_id": "t6",
            "entry_slippage_bps": 50.0,
            "pnl": 0,
        }
        score = self.scorer.score_trade(trade, "kmp")
        assert "high_entry_slippage" in score.root_causes

    def test_manual_exit_penalty(self):
        trade = {
            "trade_id": "t7",
            "exit_reason": "MANUAL",
            "pnl": 0,
        }
        score = self.scorer.score_trade(trade, "kmp")
        assert "manual_exit" in score.root_causes

    def test_good_process_loss_is_normal_loss(self):
        """Good process but negative PnL = normal_loss result_tag."""
        trade = {
            "trade_id": "t8",
            "regime": "trending_up",
            "signal_strength": 0.85,
            "entry_latency_ms": 200,
            "entry_slippage_bps": 3.0,
            "exit_reason": "STOP_LOSS",
            "pnl": -100,
        }
        score = self.scorer.score_trade(trade, "kmp")
        assert score.process_quality_score >= 70
        assert score.result_tag == "normal_loss"

    def test_bad_process_win_is_exceptional_win(self):
        """Bad process but profitable = exceptional_win (got lucky)."""
        trade = {
            "trade_id": "t9",
            "regime": "ranging",
            "signal_strength": 0.05,
            "entry_latency_ms": 20000,
            "entry_slippage_bps": 50.0,
            "exit_slippage_bps": 50.0,
            "exit_reason": "MANUAL",
            "pnl": 500,
        }
        score = self.scorer.score_trade(trade, "kmp")
        assert score.process_quality_score < 40
        assert score.result_tag == "exceptional_win"

    def test_good_process_win_is_normal_win(self):
        trade = {
            "trade_id": "t10",
            "regime": "trending_up",
            "signal_strength": 0.9,
            "entry_latency_ms": 100,
            "entry_slippage_bps": 2.0,
            "exit_reason": "TAKE_PROFIT",
            "pnl": 500,
        }
        score = self.scorer.score_trade(trade, "kmp")
        assert score.result_tag == "normal_win"

    def test_all_root_causes_from_taxonomy(self):
        """No root cause should exist outside the controlled taxonomy."""
        trade = {"trade_id": "t11", "pnl": 0}
        score = self.scorer.score_trade(trade, "kmp")
        for cause in score.root_causes:
            assert cause in ROOT_CAUSES, f"'{cause}' not in ROOT_CAUSES taxonomy"

    def test_score_clamped_0_to_100(self):
        """Score should never exceed 100 or go below 0."""
        # Stack all penalties
        trade = {
            "trade_id": "t12",
            "regime": "ranging",
            "signal_strength": 0.01,
            "entry_latency_ms": 50000,
            "entry_slippage_bps": 100,
            "exit_slippage_bps": 100,
            "exit_reason": "MANUAL",
            "pnl": -500,
        }
        score = self.scorer.score_trade(trade, "kmp")
        assert 0 <= score.process_quality_score <= 100

    def test_missing_fields_handled_gracefully(self):
        """Scorer should not crash on minimal input."""
        trade = {"trade_id": "t13"}
        score = self.scorer.score_trade(trade, "kmp")
        assert isinstance(score, ProcessScore)
        assert 0 <= score.process_quality_score <= 100

    def test_scorer_never_crashes(self):
        """Even with garbage input, scorer returns a ProcessScore."""
        score = self.scorer.score_trade({"trade_id": "bad", "signal_strength": "not_a_number"}, "kmp")
        assert isinstance(score, ProcessScore)

    def test_strategy_specific_rules(self):
        """KPR has different preferred/adverse regimes than KMP."""
        trade = {
            "trade_id": "t14",
            "regime": "trending_up",
            "signal_strength": 0.8,
            "pnl": 100,
        }
        # For KPR, trending_up is adverse
        score_kpr = self.scorer.score_trade(trade, "kpr")
        assert "regime_mismatch" in score_kpr.root_causes

        # For KMP, trending_up is preferred
        score_kmp = self.scorer.score_trade(trade, "kmp")
        assert "regime_mismatch" not in score_kmp.root_causes

    def test_default_rules_fallback(self):
        """If rules file doesn't exist, scorer uses defaults."""
        scorer = ProcessScorer("/nonexistent/path.yaml")
        score = scorer.score_trade({"trade_id": "t15"}, "kmp")
        assert isinstance(score, ProcessScore)
