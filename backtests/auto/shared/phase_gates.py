from __future__ import annotations

from .types import GateCriterion, GateResult, GreedyResult

_LOWER_IS_BETTER_TOKENS = (
    "drawdown",
    "latency",
    "false_positive",
    "hard_stop",
    "same_bar",
    "net_gross_gap",
)


def _lower_is_better(name: str) -> bool:
    normalized = name.lower()
    if normalized.startswith("max_") or normalized.endswith("_max"):
        return True
    return "dd" in normalized.replace(".", "_").split("_") or any(token in normalized for token in _LOWER_IS_BETTER_TOKENS)


def evaluate_gate(criteria: list[GateCriterion], greedy_result: GreedyResult | None = None) -> GateResult:
    if all(item.passed for item in criteria):
        return GateResult(True, tuple(criteria))
    category = _categorize(criteria, greedy_result)
    return GateResult(False, tuple(criteria), category, tuple(_recommend(criteria, category)))


def _categorize(criteria: list[GateCriterion], greedy_result: GreedyResult | None) -> str:
    for item in criteria:
        if item.passed or item.target == 0:
            continue
        if item.name.startswith("hard_"):
            return "structural_issue"
        if _lower_is_better(item.name):
            if item.actual > item.target * 2.0:
                return "structural_issue"
        elif item.actual < item.target * 0.5:
            return "structural_issue"
    if greedy_result and greedy_result.accepted_count == 0:
        return "candidates_exhausted"
    near = 0
    for item in criteria:
        if item.passed or item.target == 0 or item.name.startswith("hard_"):
            continue
        if _lower_is_better(item.name):
            near += item.actual <= item.target * 1.15
        else:
            near += item.actual >= item.target * 0.85
    return "diagnostic_needed" if near >= 2 else "scoring_ineffective"


def _recommend(criteria: list[GateCriterion], category: str) -> list[str]:
    failing = [item for item in criteria if not item.passed]
    if category == "structural_issue":
        return [f"{item.name} is far from target; revisit structural parameters." for item in failing]
    if category == "candidates_exhausted":
        return ["No candidates were accepted; expand or reshape the experiment pool."]
    if category == "diagnostic_needed":
        return ["Multiple criteria are close to target; run enhanced diagnostics before changing scoring."]
    return ["Score moved more than gate metrics; rebalance scoring toward failing criteria."]
