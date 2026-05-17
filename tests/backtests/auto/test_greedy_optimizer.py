from __future__ import annotations

from backtests.auto.shared.greedy_optimizer import run_greedy
from backtests.auto.shared.types import Experiment, ScoredCandidate


def test_greedy_pruning_compares_candidates_to_previous_score_not_round_winner():
    candidates = [
        Experiment("A", {"a": True}),
        Experiment("B", {"b": True}),
    ]

    def evaluate(batch: list[Experiment], current: dict) -> list[ScoredCandidate]:
        results: list[ScoredCandidate] = []
        for candidate in batch:
            merged = dict(current)
            merged.update(candidate.mutations)
            if candidate.name == "__baseline__":
                score = 100.0 + (20.0 if merged.get("a") else 0.0) + (20.0 if merged.get("b") else 0.0)
            elif merged.get("a") and merged.get("b"):
                score = 140.0
            elif candidate.name == "A":
                score = 120.0
            else:
                score = 99.0
            results.append(ScoredCandidate(candidate.name, score))
        return results

    result = run_greedy(
        candidates,
        {},
        evaluate,
        max_rounds=2,
        min_delta=0.001,
        prune_threshold=0.05,
        reject_streak_limit=2,
    )

    assert result.kept_features == ["A", "B"]
    assert result.final_score == 140.0
