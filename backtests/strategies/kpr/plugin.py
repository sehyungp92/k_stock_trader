from __future__ import annotations

from pathlib import Path
from typing import Any

from backtests.auto.shared.phase_state import PhaseState
from backtests.auto.shared.plugin import PhaseAnalysisPolicy, PhaseSpec
from backtests.auto.shared.types import EndOfRoundArtifacts, Experiment, GreedyResult, PhaseAnalysis, ScoredCandidate
from backtests.strategies.common.plugin_base import SharedStrategyPluginMixin

from .diagnostics import format_kpr_diagnostics
from .phase_candidates import BASE_MUTATIONS, PHASE_FOCUS, get_phase_candidates
from .phase_scoring import PHASE_HARD_REJECTS, PHASE_SCORING_WEIGHTS, ULTIMATE_TARGETS, gate_criteria, kpr_reject_reason, score_kpr_phase
from .runner import run_kpr_backtest

# Synthetic KPR single-candidate replay measured in this repo at sub-second runtime.
_KPR_HEARTBEAT_SECONDS = 30.0
_KPR_PER_CANDIDATE_TIMEOUT_SECONDS = 45.0
_KPR_MINIMUM_TIMEOUT_SECONDS = 90.0
_KPR_MAX_EVAL_BATCH_SIZE = 8


class KPROptimizationPlugin(SharedStrategyPluginMixin):
    name = "kpr"
    num_phases = 6
    ultimate_targets = ULTIMATE_TARGETS
    initial_mutations = dict(BASE_MUTATIONS)

    def __init__(self, config: dict[str, Any] | None = None, *, output_dir: Path | None = None, max_workers: int | None = 1, capability_level: str = "synthetic"):
        self.config = dict(config or {})
        self.config.setdefault("capability_level", capability_level)
        self.max_workers = max_workers
        self.capability_level = self.config.get("capability_level", capability_level)
        self.output_dir = Path(output_dir) if output_dir else None
        baseline = run_kpr_backtest(self.config, self.initial_mutations)
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

    def create_evaluate_batch(self, phase: int, cumulative_mutations: dict[str, Any], *, scoring_weights: dict[str, float] | None = None, hard_rejects: dict[str, float] | None = None):
        from .worker import init_worker, score_candidate

        base_result = run_kpr_backtest(self.config, cumulative_mutations)
        self._last_result = base_result
        reject = kpr_reject_reason(phase, base_result.metrics, hard_rejects)
        baseline = ScoredCandidate("__baseline__", 0.0 if reject else score_kpr_phase(phase, base_result.metrics, scoring_weights), bool(reject), reject, base_result.metrics)
        return self._wrap_cached_evaluator(
            phase=phase,
            cumulative_mutations=cumulative_mutations,
            scoring_weights=scoring_weights,
            hard_rejects=hard_rejects,
            init_worker=init_worker,
            score_candidate=score_candidate,
            initargs=(self.config, phase, hard_rejects, scoring_weights),
            heartbeat_seconds=_KPR_HEARTBEAT_SECONDS,
            per_candidate_timeout_seconds=_KPR_PER_CANDIDATE_TIMEOUT_SECONDS,
            minimum_timeout_seconds=_KPR_MINIMUM_TIMEOUT_SECONDS,
            max_eval_batch_size=_KPR_MAX_EVAL_BATCH_SIZE,
            description=f"kpr phase {phase}",
            baseline_result=baseline,
        )

    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]:
        self._last_result = run_kpr_backtest(self.config, mutations)
        return self._last_result.metrics

    def run_phase_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result: GreedyResult) -> str:
        del phase, state, metrics, greedy_result
        return format_kpr_diagnostics(self._last_result)

    def run_enhanced_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result: GreedyResult) -> str:
        return self.run_phase_diagnostics(phase, state, metrics, greedy_result) + "\nEnhanced checks: VWAP depth, panic/drift cohorts, confidence pillars, stale flow, partial exits."

    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts:
        final = run_kpr_backtest(self.config, state.cumulative_mutations)
        return EndOfRoundArtifacts(
            final_diagnostics_text=format_kpr_diagnostics(final),
            dimension_reports={
                "signal_extraction": "VWAP depth and HOD-drop setups are measured separately.",
                "signal_discrimination": "Investor, program, and micro-pressure confidence remain explicit feature requirements.",
                "entry_mechanism": "Reclaim acceptance submits neutral actions with next-bar fills.",
                "trade_management": "Sizing and stale-flow penalties are scored under shared capital.",
                "exit_mechanism": "Partial, trailing, full target, and time exits route through the same broker.",
            },
            overall_verdict="Synthetic KPR optimisation completed; official promotion requires investor/program/micro replay bundles.",
        )

    def get_diagnostic_gaps(self, phase: int, metrics: dict[str, float]) -> list[str]:
        del phase
        gaps = []
        if self.capability_level == "synthetic":
            gaps.append("Official KPR runs require investor, program, and micro-pressure historical bundles.")
        if metrics.get("mfe_capture", 0.0) < 0.25:
            gaps.append("Partial/trailing exits need deeper MFE capture diagnostics.")
        return gaps

    def suggest_experiments(self, phase: int, metrics: dict[str, float], weaknesses: list[str], state: PhaseState) -> list[Experiment]:
        del weaknesses, state
        if phase == 3 and metrics.get("profit_factor", 0.0) < 1.2:
            return [Experiment("confirmed_reclaim_acceptance", {"required_closes": 2})]
        if phase == 5 and metrics.get("mfe_capture", 0.0) < 0.35:
            return [Experiment("fast_trailing_exit", {"trail_variant": "fast"})]
        return []

    def redesign_scoring_weights(self, phase: int, current_weights: dict[str, float] | None, analysis: PhaseAnalysis, gate_result) -> dict[str, float] | None:
        if analysis.scoring_assessment not in {"INEFFECTIVE", "MISALIGNED", "MARGINAL"}:
            return None
        weights = dict(current_weights or PHASE_SCORING_WEIGHTS.get(phase, {}))
        for criterion in gate_result.criteria:
            key = criterion.name.replace("hard_", "")
            if not criterion.passed and key in weights:
                weights[key] *= 1.35
        total = sum(abs(value) for value in weights.values())
        return {key: value / total for key, value in weights.items()} if total else weights

