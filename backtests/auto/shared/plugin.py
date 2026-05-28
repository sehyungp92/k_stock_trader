from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .phase_state import PhaseState
from .types import EndOfRoundArtifacts, Experiment, GateCriterion, GreedyResult, PhaseAnalysis, PhaseDecision, ScoredCandidate

DiagnosticGapFn = Callable[[int, dict[str, float]], list[str]]
SuggestExperimentsFn = Callable[[int, dict[str, float], list[str], PhaseState], list[Experiment]]
RedesignWeightsFn = Callable[[int, dict[str, float] | None, PhaseAnalysis, Any], dict[str, float] | None]
ExtraFn = Callable[[int, dict[str, float], PhaseState, GreedyResult], dict[str, Any]]
ExtraReportFn = Callable[[dict[str, Any]], list[str]]
DecideActionFn = Callable[[int, dict[str, float], PhaseState, GreedyResult, Any, dict[str, float] | None, PhaseAnalysis, int, int], PhaseDecision | None]


@dataclass(slots=True)
class PhaseAnalysisPolicy:
    focus_metrics: list[str] = field(default_factory=list)
    min_effective_score_delta_pct: float = 0.01
    diagnostic_gap_fn: DiagnosticGapFn | None = None
    suggest_experiments_fn: SuggestExperimentsFn | None = None
    redesign_scoring_weights_fn: RedesignWeightsFn | None = None
    build_extra_analysis_fn: ExtraFn | None = None
    format_extra_analysis_fn: ExtraReportFn | None = None
    decide_action_fn: DecideActionFn | None = None


@dataclass(slots=True)
class PhaseSpec:
    focus: str
    candidates: list[Experiment]
    gate_criteria_fn: Callable[[dict[str, float]], list[GateCriterion]]
    scoring_weights: dict[str, float] | None
    hard_rejects: dict[str, float]
    analysis_policy: PhaseAnalysisPolicy
    max_rounds: int | None = None
    prune_threshold: float | None = None
    reject_streak_limit: int | None = None
    phase_metric_basis: str = ""
    primary_promotion_metric: str = ""
    proxy_metric_keys: tuple[str, ...] = field(default_factory=tuple)
    official_metric_keys: tuple[str, ...] = field(default_factory=tuple)
    promotion_requires_audit_pass: bool = False

    @property
    def redesign_scoring_weights_fn(self) -> RedesignWeightsFn | None:
        return self.analysis_policy.redesign_scoring_weights_fn


class StrategyPlugin(Protocol):
    name: str
    num_phases: int
    ultimate_targets: dict[str, float]
    initial_mutations: dict[str, Any] | None

    def get_phase_spec(self, phase: int, state: PhaseState) -> PhaseSpec: ...
    def create_evaluate_batch(self, phase: int, cumulative_mutations: dict[str, Any], *, scoring_weights: dict[str, float] | None = None, hard_rejects: dict[str, float] | None = None) -> Callable[[list[Experiment], dict[str, Any]], list[ScoredCandidate]]: ...
    def compute_final_metrics(self, mutations: dict[str, Any]) -> dict[str, float]: ...
    def run_phase_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result: GreedyResult) -> str: ...
    def run_enhanced_diagnostics(self, phase: int, state: PhaseState, metrics: dict[str, float], greedy_result: GreedyResult) -> str: ...
    def build_end_of_round_artifacts(self, state: PhaseState) -> EndOfRoundArtifacts: ...
