from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Callable

from backtests.auto.shared.cache_keys import build_cache_key
from backtests.auto.shared.plugin_utils import (
    CachedBatchEvaluator,
    ResilientBatchEvaluator,
    SharedPoolBatchEvaluator,
    create_process_pool,
    mutation_signature,
    shutdown_process_pool,
)
from backtests.auto.shared.types import ScoredCandidate

logger = logging.getLogger(__name__)


class LocalBatchEvaluator:
    def __init__(self, init_worker: Callable, score_candidate: Callable, initargs: tuple[Any, ...]):
        init_worker(*initargs)
        self._score_candidate = score_candidate

    def __call__(self, candidates, current_mutations):
        return [
            self._score_candidate((candidate.name, candidate.mutations, current_mutations))
            for candidate in candidates
        ]

    def close(self) -> None:
        return None


class SharedStrategyPluginMixin:
    _shared_pool = None

    def _get_or_create_pool(self, init_worker: Callable, initargs: tuple[Any, ...]):
        if self._shared_pool is None:
            self._shared_pool = create_process_pool(self.max_workers, initializer=init_worker, initargs=initargs)
        return self._shared_pool

    def close_pool(self) -> None:
        shutdown_process_pool(self._shared_pool)
        self._shared_pool = None

    def _destroy_pool(self) -> None:
        shutdown_process_pool(self._shared_pool, force=True)
        self._shared_pool = None

    def _wrap_cached_evaluator(
        self,
        *,
        phase: int,
        cumulative_mutations: dict[str, Any],
        scoring_weights: dict[str, float] | None,
        hard_rejects: dict[str, float] | None,
        init_worker: Callable,
        score_candidate: Callable,
        initargs: tuple[Any, ...],
        heartbeat_seconds: float,
        per_candidate_timeout_seconds: float,
        minimum_timeout_seconds: float,
        max_eval_batch_size: int,
        description: str,
        baseline_result: ScoredCandidate,
    ):
        baseline_key = mutation_signature(cumulative_mutations)
        seed = {baseline_key: baseline_result}

        def local_factory():
            return LocalBatchEvaluator(init_worker, score_candidate, initargs)

        def preferred_factory():
            if int(self.max_workers or 1) <= 1 or (sys.platform == "win32" and not _supports_spawn()):
                return local_factory()
            pool = self._get_or_create_pool(init_worker, initargs)
            return SharedPoolBatchEvaluator(
                pool,
                worker_fn=score_candidate,
                build_args=lambda candidates, current: [
                    (candidate.name, candidate.mutations, current)
                    for candidate in candidates
                ],
                on_terminate=self._destroy_pool,
                on_close=None,
                description=description,
                logger=logger,
                heartbeat_seconds=heartbeat_seconds,
                per_candidate_timeout_seconds=per_candidate_timeout_seconds,
                minimum_timeout_seconds=minimum_timeout_seconds,
            )

        resilient = ResilientBatchEvaluator(
            preferred_factory,
            local_factory,
            description=description,
            logger=logger,
        )
        signature_prefix = build_cache_key(
            f"{self.name}.phase_eval",
            source_fingerprint=self.source_fingerprint,
            extra={
                "phase": phase,
                "scoring_weights": scoring_weights or {},
                "hard_rejects": hard_rejects or {},
                "capability_level": self.capability_level,
            },
        )
        return CachedBatchEvaluator(
            resilient,
            cache=self._evaluation_cache,
            seed_results=seed,
            signature_prefix=signature_prefix,
            metrics_cache=self._metrics_cache,
            max_batch_size=max_eval_batch_size,
        )


def _supports_spawn() -> bool:
    if sys.platform != "win32":
        return True
    main_module = sys.modules.get("__main__")
    main_path = getattr(main_module, "__file__", "")
    return bool(main_path) and not str(main_path).startswith("<")

