"""
Process Scorer — deterministic trade quality evaluation.

Scores each trade 0-100 based on how well the process was followed,
independent of PnL outcome. This separates luck from skill.

Designed for Korean equity strategies (KMP, KPR, PCIM, NULRIMOK).
Funding rate checks are skipped entirely (equity market, no funding rates).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

# ---------------------------------------------------------------------------
# Controlled root-cause tags (21 total)
# ---------------------------------------------------------------------------

ROOT_CAUSES: list[str] = [
    "regime_mismatch",
    "weak_signal",
    "late_entry",
    "high_entry_slippage",
    "high_exit_slippage",
    "premature_stop",
    "manual_exit",
    "oversized_position",
    "undersized_position",
    "missed_entry",
    "missed_exit",
    "stale_data",
    "execution_timeout",
    "partial_fill",
    "adverse_news",
    "liquidity_gap",
    "spread_blow_out",
    "wrong_direction",
    "duplicate_entry",
    "risk_limit_breach",
    "unknown",
]

# ---------------------------------------------------------------------------
# ProcessScore dataclass
# ---------------------------------------------------------------------------


@dataclass
class ProcessScore:
    """Immutable result of scoring a single trade."""

    trade_id: str
    process_quality_score: int  # 0-100
    root_causes: list[str] = field(default_factory=list)
    evidence_refs: dict[str, Any] = field(default_factory=dict)
    positive_factors: list[str] = field(default_factory=list)
    negative_factors: list[str] = field(default_factory=list)
    classification: str = "neutral"  # good_process | neutral | bad_process
    result_tag: str = "normal_loss"  # normal_win | normal_loss | exceptional_win

    def __post_init__(self) -> None:
        self.process_quality_score = max(0, min(100, self.process_quality_score))
        # Validate root causes against controlled list
        self.root_causes = [rc for rc in self.root_causes if rc in ROOT_CAUSES]


# ---------------------------------------------------------------------------
# ProcessScorer
# ---------------------------------------------------------------------------

_DEFAULT_RULES_PATH = Path(__file__).resolve().parent.parent / "config" / "process_scoring_rules.yaml"


class ProcessScorer:
    """Deterministic, rule-based trade process evaluator.

    Usage::

        scorer = ProcessScorer()
        score = scorer.score_trade(trade_dict, strategy_type="kmp")
        print(score.process_quality_score, score.classification)

    The scorer never crashes — on bad input it returns a degraded score
    with ``unknown`` root cause.
    """

    def __init__(self, rules_path: str | Path | None = None) -> None:
        self._rules_path = Path(rules_path) if rules_path else _DEFAULT_RULES_PATH
        self._raw_rules: dict[str, Any] = {}
        self._load_rules()

    # ------------------------------------------------------------------
    # Rule loading
    # ------------------------------------------------------------------

    def _load_rules(self) -> None:
        try:
            with open(self._rules_path, "r", encoding="utf-8") as fh:
                self._raw_rules = yaml.safe_load(fh) or {}
            logger.debug("Loaded process scoring rules from {}", self._rules_path)
        except Exception as exc:
            logger.warning("Failed to load rules from {}: {}. Using defaults.", self._rules_path, exc)
            self._raw_rules = {}

    def _get_rules(self, strategy_type: str) -> dict[str, Any]:
        """Merge global rules with strategy-specific overrides.

        Strategy keys override global keys when both exist.
        """
        global_rules: dict[str, Any] = dict(self._raw_rules.get("global", {}))
        strategy_rules: dict[str, Any] = (
            self._raw_rules.get("strategies", {}).get(strategy_type.lower(), {}) or {}
        )
        merged = {**global_rules, **strategy_rules}
        return merged

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_trade(self, trade_dict: dict[str, Any], strategy_type: str) -> ProcessScore:
        """Score a single trade deterministically.

        Parameters
        ----------
        trade_dict : dict
            Must contain at minimum ``trade_id``.  Optional keys used for
            scoring: ``regime``, ``signal_strength``, ``entry_latency_ms``,
            ``entry_slippage_bps``, ``exit_slippage_bps``, ``exit_reason``,
            ``pnl``, ``stop_distance_pct``, ``price_moved_pct``.
        strategy_type : str
            One of ``kmp``, ``kpr``, ``pcim``, ``nulrimok`` (case-insensitive).

        Returns
        -------
        ProcessScore
            Deterministic score object. Never raises.
        """
        try:
            return self._score_trade_inner(trade_dict, strategy_type)
        except Exception as exc:
            trade_id = str(trade_dict.get("trade_id", "unknown"))
            logger.error("ProcessScorer crashed on trade {}: {}", trade_id, exc)
            return ProcessScore(
                trade_id=trade_id,
                process_quality_score=50,
                root_causes=["unknown"],
                evidence_refs={"error": str(exc)},
                positive_factors=[],
                negative_factors=["scorer_error"],
                classification="neutral",
                result_tag="normal_loss",
            )

    # ------------------------------------------------------------------
    # Internal scoring pipeline
    # ------------------------------------------------------------------

    def _score_trade_inner(self, trade_dict: dict[str, Any], strategy_type: str) -> ProcessScore:
        rules = self._get_rules(strategy_type)
        trade_id = str(trade_dict.get("trade_id", "unknown"))

        score = 100
        root_causes: list[str] = []
        evidence: dict[str, Any] = {}
        positives: list[str] = []
        negatives: list[str] = []

        # --- 1. Regime fit check (-20 for adverse, +0 for preferred) -----
        score, root_causes, evidence, positives, negatives = self._check_regime(
            trade_dict, rules, score, root_causes, evidence, positives, negatives,
        )

        # --- 2. Signal strength check (-25 for weak, +0 for strong) ------
        score, root_causes, evidence, positives, negatives = self._check_signal_strength(
            trade_dict, rules, score, root_causes, evidence, positives, negatives,
        )

        # --- 3. Entry latency check (-15 for late) -----------------------
        score, root_causes, evidence, positives, negatives = self._check_entry_latency(
            trade_dict, rules, score, root_causes, evidence, positives, negatives,
        )

        # --- 4. Entry slippage check (-10 for spike, +0 for good) --------
        score, root_causes, evidence, positives, negatives = self._check_entry_slippage(
            trade_dict, rules, score, root_causes, evidence, positives, negatives,
        )

        # --- 5. Exit slippage check (-10) --------------------------------
        score, root_causes, evidence, positives, negatives = self._check_exit_slippage(
            trade_dict, rules, score, root_causes, evidence, positives, negatives,
        )

        # --- 6. Exit reason analysis (-10 for MANUAL, check premature) ---
        score, root_causes, evidence, positives, negatives = self._check_exit_reason(
            trade_dict, rules, score, root_causes, evidence, positives, negatives,
        )

        # --- 7. Funding rate check — SKIPPED (equity market) -------------
        # No funding rates in Korean equity. Intentionally a no-op.

        # --- Final classification ----------------------------------------
        score = max(0, min(100, score))

        if score >= 70:
            classification = "good_process"
        elif score >= 40:
            classification = "neutral"
        else:
            classification = "bad_process"

        # --- Result tag (combines score + PnL) ---------------------------
        pnl = trade_dict.get("pnl")
        result_tag = self._compute_result_tag(score, pnl)

        return ProcessScore(
            trade_id=trade_id,
            process_quality_score=score,
            root_causes=root_causes,
            evidence_refs=evidence,
            positive_factors=positives,
            negative_factors=negatives,
            classification=classification,
            result_tag=result_tag,
        )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_regime(
        self,
        trade: dict,
        rules: dict,
        score: int,
        root_causes: list,
        evidence: dict,
        positives: list,
        negatives: list,
    ) -> tuple:
        regime = trade.get("regime")
        if regime is None:
            return score, root_causes, evidence, positives, negatives

        preferred = rules.get("preferred_regimes", [])
        adverse = rules.get("adverse_regimes", [])

        if regime in adverse:
            score -= 20
            root_causes.append("regime_mismatch")
            negatives.append(f"Traded in adverse regime: {regime}")
            evidence["regime"] = regime
            evidence["adverse_regimes"] = adverse
        elif regime in preferred:
            positives.append(f"Traded in preferred regime: {regime}")
            evidence["regime"] = regime

        return score, root_causes, evidence, positives, negatives

    def _check_signal_strength(
        self,
        trade: dict,
        rules: dict,
        score: int,
        root_causes: list,
        evidence: dict,
        positives: list,
        negatives: list,
    ) -> tuple:
        signal = trade.get("signal_strength")
        if signal is None:
            return score, root_causes, evidence, positives, negatives

        try:
            signal = float(signal)
        except (TypeError, ValueError):
            return score, root_causes, evidence, positives, negatives

        min_strength = float(rules.get("min_signal_strength", 0.3))
        strong_threshold = float(rules.get("strong_signal_threshold", 0.7))

        evidence["signal_strength"] = signal

        if signal < min_strength:
            score -= 25
            root_causes.append("weak_signal")
            negatives.append(f"Weak signal ({signal:.2f} < {min_strength:.2f})")
        elif signal >= strong_threshold:
            positives.append(f"Strong signal ({signal:.2f} >= {strong_threshold:.2f})")

        return score, root_causes, evidence, positives, negatives

    def _check_entry_latency(
        self,
        trade: dict,
        rules: dict,
        score: int,
        root_causes: list,
        evidence: dict,
        positives: list,
        negatives: list,
    ) -> tuple:
        latency = trade.get("entry_latency_ms")
        if latency is None:
            return score, root_causes, evidence, positives, negatives

        try:
            latency = float(latency)
        except (TypeError, ValueError):
            return score, root_causes, evidence, positives, negatives

        max_latency = float(rules.get("max_entry_latency_ms", 5000))
        evidence["entry_latency_ms"] = latency

        if latency > max_latency:
            score -= 15
            root_causes.append("late_entry")
            negatives.append(f"Late entry ({latency:.0f}ms > {max_latency:.0f}ms)")
        else:
            positives.append(f"Timely entry ({latency:.0f}ms)")

        return score, root_causes, evidence, positives, negatives

    def _check_entry_slippage(
        self,
        trade: dict,
        rules: dict,
        score: int,
        root_causes: list,
        evidence: dict,
        positives: list,
        negatives: list,
    ) -> tuple:
        slippage = trade.get("entry_slippage_bps")
        if slippage is None:
            return score, root_causes, evidence, positives, negatives

        try:
            slippage = float(slippage)
        except (TypeError, ValueError):
            return score, root_causes, evidence, positives, negatives

        expected = float(rules.get("expected_slippage_bps", 10))
        max_mult = float(rules.get("max_slippage_multiplier", 2.0))
        threshold = expected * max_mult

        evidence["entry_slippage_bps"] = slippage
        evidence["expected_slippage_bps"] = expected

        if slippage > threshold:
            score -= 10
            root_causes.append("high_entry_slippage")
            negatives.append(f"High entry slippage ({slippage:.1f}bps > {threshold:.1f}bps)")
        else:
            positives.append(f"Good entry execution ({slippage:.1f}bps)")

        return score, root_causes, evidence, positives, negatives

    def _check_exit_slippage(
        self,
        trade: dict,
        rules: dict,
        score: int,
        root_causes: list,
        evidence: dict,
        positives: list,
        negatives: list,
    ) -> tuple:
        slippage = trade.get("exit_slippage_bps")
        if slippage is None:
            return score, root_causes, evidence, positives, negatives

        try:
            slippage = float(slippage)
        except (TypeError, ValueError):
            return score, root_causes, evidence, positives, negatives

        expected = float(rules.get("expected_slippage_bps", 10))
        max_mult = float(rules.get("max_slippage_multiplier", 2.0))
        threshold = expected * max_mult

        evidence["exit_slippage_bps"] = slippage

        if slippage > threshold:
            score -= 10
            root_causes.append("high_exit_slippage")
            negatives.append(f"High exit slippage ({slippage:.1f}bps > {threshold:.1f}bps)")
        else:
            positives.append(f"Good exit execution ({slippage:.1f}bps)")

        return score, root_causes, evidence, positives, negatives

    def _check_exit_reason(
        self,
        trade: dict,
        rules: dict,
        score: int,
        root_causes: list,
        evidence: dict,
        positives: list,
        negatives: list,
    ) -> tuple:
        exit_reason = trade.get("exit_reason")
        if exit_reason is None:
            return score, root_causes, evidence, positives, negatives

        exit_reason_upper = str(exit_reason).upper()
        evidence["exit_reason"] = exit_reason

        # Manual exits are a process violation
        if exit_reason_upper == "MANUAL":
            score -= 10
            root_causes.append("manual_exit")
            negatives.append("Manual exit override")
            return score, root_causes, evidence, positives, negatives

        # Check for premature stop: stopped out but price reversed favorably
        if exit_reason_upper in ("STOP", "STOP_LOSS"):
            stop_distance = trade.get("stop_distance_pct")
            price_moved = trade.get("price_moved_pct")

            if stop_distance is not None and price_moved is not None:
                try:
                    stop_distance = float(stop_distance)
                    price_moved = float(price_moved)
                except (TypeError, ValueError):
                    return score, root_causes, evidence, positives, negatives

                # Premature stop: price moved back favorably beyond stop distance
                # (i.e., the stop was too tight — price recovered past where it hit stop)
                if price_moved > 0 and price_moved > stop_distance:
                    score -= 10
                    root_causes.append("premature_stop")
                    negatives.append(
                        f"Premature stop (price moved +{price_moved:.2f}% "
                        f"after {stop_distance:.2f}% stop)"
                    )
                    evidence["stop_distance_pct"] = stop_distance
                    evidence["price_moved_pct"] = price_moved

        # Clean exits are a positive signal
        if exit_reason_upper in ("TARGET", "TAKE_PROFIT", "SIGNAL", "EOD"):
            positives.append(f"Clean exit: {exit_reason}")

        return score, root_causes, evidence, positives, negatives

    # ------------------------------------------------------------------
    # Result tag
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_result_tag(score: int, pnl: Any) -> str:
        """Combine process score with PnL to produce a result tag.

        Tags:
        - ``normal_win``        — good/neutral process + profit
        - ``normal_loss``       — any process + loss (or unknown PnL)
        - ``exceptional_win``   — bad process but still profitable (luck)
        """
        if pnl is None:
            return "normal_loss"

        try:
            pnl = float(pnl)
        except (TypeError, ValueError):
            return "normal_loss"

        is_profitable = pnl > 0

        if not is_profitable:
            return "normal_loss"

        # Profitable trade
        if score < 40:
            # Bad process + profit = got lucky
            return "exceptional_win"
        else:
            return "normal_win"
