from __future__ import annotations

from pathlib import Path
from typing import Any

from backtests.auto.shared.plugin import PhaseAnalysisPolicy, PhaseSpec
from backtests.auto.shared.phase_state import PhaseState
from backtests.auto.shared.types import EndOfRoundArtifacts, Experiment, GateCriterion, GreedyResult, PhaseAnalysis
from backtests.strategies.common.plugin_base import SharedStrategyPluginMixin
from backtests.auto.shared.types import ScoredCandidate

from .diagnostics import format_kmp_diagnostics
from .phase_candidates import BASE_MUTATIONS, PHASE_FOCUS, get_phase_candidates
from .phase_scoring import PHASE_HARD_REJECTS, PHASE_SCORING_WEIGHTS, ULTIMATE_TARGETS, gate_criteria, kmp_reject_reason, score_kmp_phase
from .runner import run_kmp_backtest

# Synthetic KMP single-candidate replay measured in this repo at sub-second runtime.
_KMP_HEARTBEAT_SECONDS = 30.0
_KMP_PER_CANDIDATE_TIMEOUT_SECONDS = 30.0
_KMP_MINIMUM_TIMEOUT_SECONDS = 60.0
_KMP_MAX_EVAL_BATCH_SIZE = 8


class KMPOptimizationPlugin(SharedStrategyPluginMixin):
    name = "kmp"
    num_phases = 6
    ultimate_targets = ULTIMATE_TARGETS
    initial_mutations = dict(BASE_MUTATIONS)

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        output_dir: Path | None = None,
        max_workers: int | None = 1,
        capability_level: str = "synthetic",
    ):
        self.config = dict(config or {})
        self.config.setdefault("capability_level", capability_level)
        self.max_workers = max_workers
        self.capability_level = self.config.get("capability_level", capability_level)
        self.output_dir = Path(output_dir) if output_dir else None
        baseline = run_kmp_backtest(self.config, self.initial_mutations)
        self.source_fingerprint = baseline.source_fingerprint
        self._evaluation_cache: dict[str, ScoredCandidate] = {}
        self._metrics_cache: dict[str, dict[str, float]] = {}
        self._last_result = baseline

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec:
        del state
        focus, focus_metrics = PHASE_FOCUS[phase]
        return PhaseSpec(
            focus=focus,
            candidates=get_phase_candidates(phase),
            gate_criteria_fn=lambda metrics, current_phase=phase: gate_criteria(current_phase, metrics, PHASE_HARD_REJECTS[current_phase]),
            scoring_weights=dict(PHASE_SCORING_WEIGHTS.get(phase, {})),
            hard_rejects=dict(PHASE_HARD_REJECTS.get(phase, {})),
            analysis_policy=PhaseAnalysisPolicy(
                focus_metrics=focus_metrics,
                min_effective_score_delta_pct=0.001,
                diagnostic_gap_fn=self.get_diagnostic_gaps,
                suggest_experiments_fn=self.suggest_experiments,
                redesign_scoring_weights_fn=self.redesign_scoring_weights,
            ),
            max_rounds=3,
            prune_threshold=0.10,
            reject_streak_limit=2,
        )

    def create_evaluate_batch(
        self,
        phase: int,
        cumulative_mutations: dict[str, Any],
        *,
        scoring_weights: dict[str, float] | None = None,
        hard_rejects: dict[str, float] | None = None,
    ):
        from .worker import init_worker, score_candidate

        base_result = run_kmp_backtest(self.config, cumulative_mutations)
        self._last_result = base_result
        reject = kmp_reject_reason(phase, base_result.metrics, hard_rejects)
        baseline = ScoredCandidate(
            name="__baseline__",
            score=0.0 if reject else score_kmp_phase(phase, base_result.metrics, scoring_weights),
            rejected=bool(reject),
            reject_reason=reject,
            metrics=base_result.metrics,
        )
        return self._wrap_cached_evaluator(
            phase=phase,
            cumulative_mutations=cumulative_mutations,
            scoring_weights=scoring_weights,
            hard_rejects=hard_rejects,
            init_worker=init_worker,
            score_candidate=score_candidate,
            initargs=(self.config, phase, hard_rejects, scoring_weights),
            heartbeat_seconds=_KMP_HEARTBEAT_SECONDS,
            per_candidate_timeout_seconds=_KMP_PER_CANDIDATE_TIMEOUT_SECONDS,
            minimum_timeout_seconds=_KMP_MINIMUM_TIMEOUT_SECONDS,
            max_eval_batch_size=_KMP_MAX_EVAL_BATCH_SIZE,
            description=f"kmp phase {phase}",
            baseline_result=baseline,
        )

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        self._last_result = run_kmp_backtest(self.config, mutations)
        return self._last_result.metrics

    def run_phase_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result: GreedyResult) -> str:
        del state, metrics, greedy_result
        return format_kmp_diagnostics(self._last_result)

    def run_enhanced_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result: GreedyResult) -> str:
        base = self.run_phase_diagnostics(phase, state, metrics, greedy_result)
        return base + "\nEnhanced checks: surge alpha, gate blocks, MFE capture, VI/spread opportunity cost."

    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
        final = run_kmp_backtest(self.config, state.cumulative_mutations)
        text = format_kmp_diagnostics(final)
        return EndOfRoundArtifacts(
            final_diagnostics_text=text,
            dimension_reports={
                "signal_extraction": "Opening-range and surge candidates are evaluated before entry mechanics.",
                "signal_discrimination": "RVOL, spread, VI, and drawdown gates are scored separately from signal extraction.",
                "entry_mechanism": "Signals submit neutral actions and the simulator fills no earlier than the next bar.",
                "trade_management": "Sizing mutations compete under one shared simulated ledger.",
                "exit_mechanism": "Targets, stops, and EOD flatten route through the same broker path.",
            },
            overall_verdict="Synthetic KMP optimisation completed; official promotion still requires tick/orderbook/VI replay bundles.",
        )

    def get_diagnostic_gaps(self, phase: int, metrics: dict[str, float]) -> list[str]:
        gaps: list[str] = []
        if self.capability_level == "synthetic":
            gaps.append("Official KMP runs require tick, bid/ask, spread, VI, and tick-imbalance bundles.")
        if metrics.get("mfe_capture", 0.0) < 0.25:
            gaps.append("Accepted KMP signals have weak MFE capture; inspect entry and exit timing.")
        return gaps

    def suggest_experiments(self, phase: int, metrics: dict[str, float], weaknesses: list[str], state: PhaseState) -> list[Experiment]:
        del weaknesses, state
        suggestions: list[Experiment] = []
        if phase == 2 and metrics.get("profit_factor", 0.0) < 1.2:
            suggestions.append(Experiment("spread_vi_filter_rebalance", {"spread_bps": 6.0}))
        if phase == 5 and metrics.get("mfe_capture", 0.0) < 0.35:
            suggestions.append(Experiment("faster_profit_capture", {"target_pct": 0.018}))
        return suggestions

    def redesign_scoring_weights(self, phase: int, current_weights: dict[str, float] | None, analysis: PhaseAnalysis, gate_result) -> dict[str, float] | None:
        if analysis.scoring_assessment not in {"INEFFECTIVE", "MISALIGNED", "MARGINAL"}:
            return None
        weights = dict(current_weights or PHASE_SCORING_WEIGHTS.get(phase, {}))
        for criterion in gate_result.criteria:
            if criterion.passed:
                continue
            key = criterion.name.replace("hard_", "")
            if key in weights:
                weights[key] = weights.get(key, 0.0) * 1.35
        total = sum(abs(value) for value in weights.values())
        return {key: value / total for key, value in weights.items()} if total else weights

