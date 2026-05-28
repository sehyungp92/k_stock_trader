from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Experiment:
    name: str
    mutations: dict[str, Any]


@dataclass(frozen=True, slots=True)
class GreedyRound:
    round_num: int
    candidates_tested: int
    best_name: str
    best_score: float
    best_delta_pct: float
    kept: bool
    rejected_count: int = 0


@dataclass(slots=True)
class GreedyResult:
    base_score: float
    final_score: float
    final_mutations: dict[str, Any]
    kept_features: list[str]
    rounds: list[GreedyRound]
    final_metrics: dict[str, float]
    total_candidates: int
    accepted_count: int
    elapsed_seconds: float
    candidate_evaluations: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class GateCriterion:
    name: str
    target: float
    actual: float
    passed: bool


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    criteria: tuple[GateCriterion, ...]
    failure_category: str | None = None
    recommendations: tuple[str, ...] = ()


@dataclass(slots=True)
class PhaseAnalysis:
    phase: int
    goal_progress: dict[str, dict] = field(default_factory=dict)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    scoring_assessment: str = ""
    diagnostic_gaps: list[str] = field(default_factory=list)
    suggested_experiments: list[Experiment] = field(default_factory=list)
    recommendation: str = ""
    recommendation_reason: str = ""
    report: str = ""
    scoring_weight_overrides: dict[str, float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PhaseDecision:
    action: str
    reason: str
    scoring_weight_overrides: dict[str, float] | None = None
    scoring_assessment_override: str | None = None
    extra_diagnostic_gaps: list[str] = field(default_factory=list)
    extra_suggested_experiments: list[Experiment] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EndOfRoundArtifacts:
    final_diagnostics_text: str
    dimension_reports: dict[str, str]
    overall_verdict: str
    extra_sections: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    name: str
    score: float
    rejected: bool = False
    reject_reason: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
