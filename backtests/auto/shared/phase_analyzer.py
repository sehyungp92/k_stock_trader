from __future__ import annotations

import json

from .greedy_optimizer import _delta_ratio
from .phase_gates import _lower_is_better
from .phase_state import PhaseState
from .plugin import PhaseAnalysisPolicy
from .types import Experiment, GateResult, GreedyResult, PhaseAnalysis


def _metric_value(metrics: dict[str, float], name: str) -> float:
    return float(metrics.get(name, 0.0))


def _target_progress(name: str, actual: float, target: float) -> float:
    if target == 0:
        return 0.0
    if _lower_is_better(name):
        if actual <= 0:
            return 200.0
        return min((target / actual) * 100.0, 200.0)
    return min((actual / target) * 100.0, 200.0)


def _metric_improved(name: str, previous: float, current: float) -> bool:
    return current < previous if _lower_is_better(name) else current > previous


def _assess_scoring(
    greedy_result: GreedyResult,
    gate_result: GateResult,
    *,
    score_delta_ratio: float,
    focus_metrics: list[str],
    goal_progress: dict[str, dict],
    previous_metrics: dict[str, float] | None,
    metrics: dict[str, float],
    min_effective_score_delta_pct: float,
) -> str:
    if gate_result.passed:
        return "EFFECTIVE"
    if greedy_result.accepted_count == 0 or gate_result.failure_category == "candidates_exhausted":
        return "INEFFECTIVE"
    if score_delta_ratio < min_effective_score_delta_pct:
        return "MARGINAL"
    if focus_metrics:
        far = sum(1 for name in focus_metrics if goal_progress.get(name, {}).get("pct_of_target", 0.0) < 60.0)
        if far >= max(1, (len(focus_metrics) + 1) // 2):
            return "MISALIGNED"
    if previous_metrics and focus_metrics:
        improved = sum(
            1 for name in focus_metrics
            if _metric_improved(name, _metric_value(previous_metrics, name), _metric_value(metrics, name))
        )
        if improved < max(1, (len(focus_metrics) + 1) // 2):
            return "MISALIGNED"
    return "MARGINAL" if gate_result.failure_category == "diagnostic_needed" else "EFFECTIVE"


def analyze_phase(
    phase: int,
    greedy_result: GreedyResult,
    metrics: dict[str, float],
    state: PhaseState,
    gate_result: GateResult,
    *,
    ultimate_targets: dict[str, float],
    policy: PhaseAnalysisPolicy,
    current_weights: dict[str, float] | None = None,
    max_scoring_retries: int = 2,
    max_diagnostic_retries: int = 1,
) -> PhaseAnalysis:
    analysis = PhaseAnalysis(phase=phase)
    for name, target in ultimate_targets.items():
        actual = _metric_value(metrics, name)
        analysis.goal_progress[name] = {
            "target": target,
            "actual": actual,
            "pct_of_target": _target_progress(name, actual, target),
        }
    for name, progress in analysis.goal_progress.items():
        label = f"{name}: {progress['actual']:.4f} ({progress['pct_of_target']:.0f}% of {progress['target']:.4f})"
        if progress["pct_of_target"] >= 80.0:
            analysis.strengths.append(label)
        elif progress["pct_of_target"] < 60.0:
            analysis.weaknesses.append(label)

    previous_metrics = state.get_phase_metrics(phase - 1) if phase > 1 else None
    analysis.scoring_assessment = _assess_scoring(
        greedy_result,
        gate_result,
        score_delta_ratio=_delta_ratio(greedy_result.final_score, greedy_result.base_score),
        focus_metrics=policy.focus_metrics,
        goal_progress=analysis.goal_progress,
        previous_metrics=previous_metrics,
        metrics=metrics,
        min_effective_score_delta_pct=policy.min_effective_score_delta_pct,
    )
    analysis.diagnostic_gaps = _dedupe_strings(policy.diagnostic_gap_fn(phase, metrics) if policy.diagnostic_gap_fn else [])
    analysis.suggested_experiments = _dedupe_experiments(
        policy.suggest_experiments_fn(phase, metrics, analysis.weaknesses, state)
        if policy.suggest_experiments_fn
        else []
    )
    if policy.build_extra_analysis_fn:
        analysis.extra = policy.build_extra_analysis_fn(phase, metrics, state, greedy_result)

    scoring_retries = state.scoring_retries.get(phase, 0)
    diagnostic_retries = state.diagnostic_retries.get(phase, 0)
    decision = None
    if policy.decide_action_fn:
        proposed = policy.decide_action_fn(
            phase,
            metrics,
            state,
            greedy_result,
            gate_result,
            current_weights,
            analysis,
            max_scoring_retries,
            max_diagnostic_retries,
        )
        if proposed:
            if proposed.scoring_assessment_override:
                analysis.scoring_assessment = proposed.scoring_assessment_override
            analysis.diagnostic_gaps = _dedupe_strings([*analysis.diagnostic_gaps, *proposed.extra_diagnostic_gaps])
            analysis.suggested_experiments = _dedupe_experiments([*analysis.suggested_experiments, *proposed.extra_suggested_experiments])
            decision = proposed

    if decision:
        analysis.recommendation = decision.action
        analysis.recommendation_reason = decision.reason
        if decision.scoring_weight_overrides:
            analysis.scoring_weight_overrides = decision.scoring_weight_overrides
    elif gate_result.passed:
        analysis.recommendation = "advance"
        analysis.recommendation_reason = "Gate passed."
    elif gate_result.failure_category == "diagnostic_needed" and diagnostic_retries < max_diagnostic_retries:
        analysis.recommendation = "improve_diagnostics"
        analysis.recommendation_reason = "Near misses need enhanced diagnostics."
    elif analysis.scoring_assessment in {"MISALIGNED", "INEFFECTIVE", "MARGINAL"} and scoring_retries < max_scoring_retries:
        analysis.recommendation = "improve_scoring"
        analysis.recommendation_reason = "Greedy score is not aligned strongly enough with phase goals."
        analysis.scoring_weight_overrides = _redesign_weights(policy, phase, current_weights, analysis, gate_result)
    else:
        analysis.recommendation = "advance"
        analysis.recommendation_reason = "Retry budget exhausted; carry results forward."

    analysis.report = _build_report(analysis, policy)
    return analysis


def _redesign_weights(
    policy: PhaseAnalysisPolicy,
    phase: int,
    current_weights: dict[str, float] | None,
    analysis: PhaseAnalysis,
    gate_result: GateResult,
) -> dict[str, float] | None:
    if policy.redesign_scoring_weights_fn:
        redesigned = policy.redesign_scoring_weights_fn(phase, current_weights, analysis, gate_result)
        if redesigned is not None:
            return redesigned
    if not current_weights:
        return None
    weights = dict(current_weights)
    for name in policy.focus_metrics:
        if name in weights:
            weights[name] *= 1.25
    for weakness in analysis.weaknesses:
        key = weakness.split(":", 1)[0]
        if key in weights:
            weights[key] *= 1.15
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()} if total else weights


def _build_report(analysis: PhaseAnalysis, policy: PhaseAnalysisPolicy) -> str:
    lines = [
        "=" * 70,
        f"PHASE {analysis.phase} ANALYSIS",
        "=" * 70,
        "",
        f"Recommendation: {analysis.recommendation}",
        f"Reason: {analysis.recommendation_reason}",
        f"Scoring assessment: {analysis.scoring_assessment}",
    ]
    if analysis.strengths:
        lines.extend(["", "Strengths:", *(f"  + {item}" for item in analysis.strengths)])
    if analysis.weaknesses:
        lines.extend(["", "Weaknesses:", *(f"  - {item}" for item in analysis.weaknesses)])
    if analysis.diagnostic_gaps:
        lines.extend(["", "Diagnostic Gaps:", *(f"  ! {item}" for item in analysis.diagnostic_gaps)])
    if analysis.suggested_experiments:
        lines.extend(["", "Suggested Experiments:"])
        lines.extend(f"  - {item.name}: {json.dumps(item.mutations, default=str)}" for item in analysis.suggested_experiments[:10])
    if analysis.extra:
        lines.extend(["", "Extra Analysis:"])
        if policy.format_extra_analysis_fn:
            lines.extend(f"  {line}" for line in policy.format_extra_analysis_fn(analysis.extra))
        else:
            lines.extend(f"  {line}" for line in json.dumps(analysis.extra, indent=2, default=str).splitlines())
    lines.extend(["", "Goal Progress:"])
    lines.extend(
        f"  {name}: {progress['actual']:.4f} / {progress['target']:.4f} ({progress['pct_of_target']:.0f}%)"
        for name, progress in analysis.goal_progress.items()
    )
    return "\n".join(lines)


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _dedupe_experiments(values: list[Experiment]) -> list[Experiment]:
    seen: set[str] = set()
    result: list[Experiment] = []
    for value in values:
        if value.name in seen:
            continue
        seen.add(value.name)
        result.append(value)
    return result

