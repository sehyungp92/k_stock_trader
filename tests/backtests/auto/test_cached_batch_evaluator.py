from __future__ import annotations

import pytest

from backtests.auto.shared.plugin_utils import CachedBatchEvaluator, ResilientBatchEvaluator
from backtests.auto.shared.types import Experiment


def test_cached_batch_evaluator_can_reject_timeout_chunks() -> None:
    class TimeoutDelegate:
        terminated = False

        def __call__(self, candidates, current_mutations):
            del candidates, current_mutations
            raise TimeoutError("too slow")

        def terminate(self) -> None:
            self.terminated = True

    delegate = TimeoutDelegate()
    evaluator = CachedBatchEvaluator(delegate, max_batch_size=2, reject_on_timeout=True)

    scored = evaluator(
        [
            Experiment("slow_a", {"x": 1}),
            Experiment("slow_b", {"x": 2}),
        ],
        {},
    )

    assert [item.name for item in scored] == ["slow_a", "slow_b"]
    assert all(item.rejected for item in scored)
    assert all("evaluation_timeout" in item.reject_reason for item in scored)
    assert delegate.terminated


def test_cached_batch_evaluator_raises_timeout_by_default() -> None:
    def delegate(candidates, current_mutations):
        del candidates, current_mutations
        raise TimeoutError("too slow")

    evaluator = CachedBatchEvaluator(delegate)

    with pytest.raises(TimeoutError):
        evaluator([Experiment("slow", {"x": 1})], {})


def test_resilient_batch_evaluator_rebuilds_after_terminate() -> None:
    builds = []

    class Delegate:
        def __init__(self, generation: int):
            self.generation = generation
            self.terminated = False

        def __call__(self, candidates, current_mutations):
            del current_mutations
            return [f"{self.generation}:{candidate.name}" for candidate in candidates]

        def terminate(self) -> None:
            self.terminated = True

    def preferred_factory():
        generation = len(builds) + 1
        delegate = Delegate(generation)
        builds.append(delegate)
        return delegate

    evaluator = ResilientBatchEvaluator(
        preferred_factory,
        preferred_factory,
        description="test",
        fallback_on_timeout=False,
    )

    assert evaluator([Experiment("a", {})], {}) == ["1:a"]
    evaluator.terminate()
    assert builds[0].terminated
    assert evaluator([Experiment("b", {})], {}) == ["2:b"]


def test_cached_resilient_evaluator_rejects_timeout_without_local_fallback() -> None:
    class TimeoutDelegate:
        terminated = False

        def __call__(self, candidates, current_mutations):
            del candidates, current_mutations
            raise RuntimeError("evaluation phase 7 exceeded timeout after 720s")

        def terminate(self) -> None:
            self.terminated = True

    preferred = TimeoutDelegate()
    fallback_called = False

    def preferred_factory():
        return preferred

    def fallback_factory():
        nonlocal fallback_called
        fallback_called = True
        return preferred

    resilient = ResilientBatchEvaluator(
        preferred_factory,
        fallback_factory,
        description="test",
        fallback_on_timeout=False,
    )
    evaluator = CachedBatchEvaluator(resilient, reject_on_timeout=True)

    scored = evaluator([Experiment("slow", {"x": 1})], {})

    assert scored[0].name == "slow"
    assert scored[0].rejected
    assert "evaluation_timeout" in scored[0].reject_reason
    assert preferred.terminated
    assert not fallback_called
