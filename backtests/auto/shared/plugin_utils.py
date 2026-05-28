from __future__ import annotations

import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .phase_state import PhaseState
from .types import Experiment, GreedyResult, GreedyRound, ScoredCandidate

DEFAULT_AUTO_WORKERS = 3
DEFAULT_POOL_HEARTBEAT_SECONDS = 150.0
DEFAULT_POOL_TIMEOUT_PER_CANDIDATE_SECONDS = 300.0
DEFAULT_POOL_MIN_TIMEOUT_SECONDS = 300.0
MIN_POOL_PROGRESS_STEP = 5
_FALLBACK_EXCEPTIONS = (PermissionError, OSError, EOFError, BrokenPipeError, RuntimeError, TimeoutError)


def _timeout_like(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    if exc.__class__.__name__ == "TimeoutError":
        return True
    message = str(exc).lower()
    return "exceeded timeout" in message or "timed out" in message


def deserialize_experiments(raw: list[dict[str, Any]] | None) -> list[Experiment]:
    items: list[Experiment] = []
    for item in raw or []:
        if isinstance(item, Experiment):
            items.append(item)
        elif isinstance(item, dict):
            items.append(Experiment(name=str(item.get("name", "")), mutations=dict(item.get("mutations", {}))))
    return items


def greedy_result_from_state(state: PhaseState, *, phase: int, final_metrics: dict[str, float]) -> GreedyResult:
    result = state.phase_results.get(phase, {})
    rounds = [GreedyRound(**item) for item in result.get("rounds", [])]
    return GreedyResult(
        base_score=float(result.get("base_score", 0.0)),
        final_score=float(result.get("final_score", 0.0)),
        final_mutations=dict(result.get("final_mutations", state.cumulative_mutations)),
        kept_features=list(result.get("kept_features", [])),
        rounds=rounds,
        final_metrics=dict(final_metrics),
        total_candidates=int(result.get("total_candidates", 0)),
        accepted_count=int(result.get("accepted_count", len(result.get("kept_features", [])))),
        elapsed_seconds=float(result.get("elapsed_seconds", 0.0)),
    )


def resolve_worker_processes(max_workers: int | None) -> int:
    if max_workers is not None:
        return max(1, int(max_workers))
    return min(DEFAULT_AUTO_WORKERS, max(1, os.cpu_count() or DEFAULT_AUTO_WORKERS))


def create_process_pool(max_workers: int | None, *, initializer=None, initargs: tuple[Any, ...] = ()):
    context = mp.get_context("spawn") if os.name == "nt" else mp.get_context()
    return context.Pool(processes=resolve_worker_processes(max_workers), initializer=initializer, initargs=tuple(initargs))


def shutdown_process_pool(pool, *, force: bool = False) -> None:
    if pool is None:
        return
    try:
        if force:
            pool.terminate()
        else:
            pool.close()
        pool.join()
    except Exception:
        try:
            pool.terminate()
            pool.join()
        except Exception:
            pass


def pool_map_with_heartbeat(
    pool,
    worker_fn,
    args: list[Any],
    *,
    description: str,
    logger: logging.Logger | None = None,
    heartbeat_seconds: float = DEFAULT_POOL_HEARTBEAT_SECONDS,
    per_candidate_timeout_seconds: float = DEFAULT_POOL_TIMEOUT_PER_CANDIDATE_SECONDS,
    minimum_timeout_seconds: float = DEFAULT_POOL_MIN_TIMEOUT_SECONDS,
):
    if not args:
        return []
    pending = {index: pool.apply_async(worker_fn, (arg,)) for index, arg in enumerate(args)}
    results: list[Any | None] = [None] * len(args)
    started = time.monotonic()
    timeout = max(float(minimum_timeout_seconds), len(args) * float(per_candidate_timeout_seconds))
    next_heartbeat = started + max(1.0, heartbeat_seconds)
    completed = 0
    progress_step = max(MIN_POOL_PROGRESS_STEP, math.ceil(len(args) / 10))
    next_progress = progress_step
    while completed < len(args):
        elapsed = time.monotonic() - started
        if elapsed >= timeout:
            raise TimeoutError(f"{description} exceeded timeout after {elapsed:.0f}s")
        progressed = False
        for index, async_result in list(pending.items()):
            if not async_result.ready():
                continue
            results[index] = async_result.get()
            pending.pop(index)
            completed += 1
            progressed = True
        if logger and progressed and (completed >= next_progress or completed == len(args)):
            logger.info("%s progress: %d/%d completed", description, completed, len(args))
            while next_progress <= completed:
                next_progress += progress_step
            next_heartbeat = time.monotonic() + max(1.0, heartbeat_seconds)
        elif logger and time.monotonic() >= next_heartbeat:
            logger.info("%s still running after %.0fs (%d/%d completed)", description, elapsed, completed, len(args))
            next_heartbeat = time.monotonic() + max(1.0, heartbeat_seconds)
        if not progressed:
            time.sleep(min(max(heartbeat_seconds / 3.0, 1.0), 5.0))
    if any(item is None for item in results):
        raise RuntimeError(f"{description} completed with missing worker results")
    return [item for item in results if item is not None]


def mutation_signature(mutations: dict[str, Any]) -> str:
    raw = json.dumps(mutations, sort_keys=True, separators=(",", ":"), default=_default).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def clone_scored_candidate(result: ScoredCandidate, *, name: str | None = None) -> ScoredCandidate:
    return ScoredCandidate(
        name=name or result.name,
        score=float(result.score),
        rejected=bool(result.rejected),
        reject_reason=str(result.reject_reason or ""),
        metrics=dict(result.metrics or {}),
    )


class CachedBatchEvaluator:
    def __init__(
        self,
        delegate,
        *,
        cache: dict[str, ScoredCandidate] | None = None,
        seed_results: dict[str, ScoredCandidate] | None = None,
        signature_prefix: str = "",
        metrics_cache: dict[str, dict[str, float]] | None = None,
        max_batch_size: int | None = None,
        reject_on_timeout: bool = False,
    ):
        self._delegate = delegate
        self._cache = cache if cache is not None else {}
        self._signature_prefix = signature_prefix
        self._metrics_cache = metrics_cache
        self._max_batch_size = max_batch_size if max_batch_size and max_batch_size > 0 else None
        self._reject_on_timeout = bool(reject_on_timeout)
        self._progress_callback = None
        for key, result in (seed_results or {}).items():
            self._cache[self._cache_key(key)] = clone_scored_candidate(result)
            if self._metrics_cache is not None and result.metrics:
                self._metrics_cache[key] = dict(result.metrics)

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]) -> list[ScoredCandidate]:
        results: list[ScoredCandidate | None] = [None] * len(candidates)
        pending: list[Experiment] = []
        pending_keys: list[str] = []
        key_to_indexes: dict[str, list[int]] = {}
        for index, candidate in enumerate(candidates):
            merged = dict(current_mutations)
            merged.update(candidate.mutations)
            key = mutation_signature(merged)
            cached = self._cache.get(self._cache_key(key))
            if cached:
                results[index] = clone_scored_candidate(cached, name=candidate.name)
                continue
            key_to_indexes.setdefault(key, []).append(index)
            if len(key_to_indexes[key]) == 1:
                pending.append(candidate)
                pending_keys.append(key)
        cached_count = len(candidates) - len(pending)
        self._emit({"event": "batch_start", "total_candidates": len(candidates), "pending_candidates": len(pending), "cached_candidates": cached_count})
        if pending:
            batch_size = self._max_batch_size or len(pending)
            for start in range(0, len(pending), batch_size):
                chunk = pending[start:start + batch_size]
                chunk_keys = pending_keys[start:start + batch_size]
                self._emit({"event": "chunk_start", "chunk_size": len(chunk), "chunk_candidate_names": [item.name for item in chunk]})
                try:
                    scored = self._delegate(chunk, current_mutations)
                except TimeoutError as exc:
                    if not self._reject_on_timeout:
                        raise
                    terminate = getattr(self._delegate, "terminate", None)
                    if callable(terminate):
                        terminate()
                    reason = f"evaluation_timeout: {exc}"
                    self._emit({"event": "chunk_timeout", "chunk_size": len(chunk), "chunk_candidate_names": [item.name for item in chunk], "reason": reason})
                    scored = [
                        ScoredCandidate(
                            name=item.name,
                            score=0.0,
                            rejected=True,
                            reject_reason=reason,
                            metrics={"evaluation_timeout": 1.0},
                        )
                        for item in chunk
                    ]
                if len(scored) != len(chunk):
                    raise RuntimeError("Batch evaluator returned the wrong number of results")
                for key, prototype, scored_result in zip(chunk_keys, chunk, scored, strict=True):
                    template = clone_scored_candidate(scored_result, name=prototype.name)
                    self._cache[self._cache_key(key)] = template
                    if self._metrics_cache is not None and scored_result.metrics:
                        self._metrics_cache[key] = dict(scored_result.metrics)
                    for index in key_to_indexes.get(key, []):
                        results[index] = clone_scored_candidate(template, name=candidates[index].name)
                self._emit({"event": "chunk_complete", "chunk_size": len(chunk)})
        self._emit({"event": "batch_complete", "total_candidates": len(candidates), "cached_candidates": cached_count})
        return [item for item in results if item is not None]

    def close(self) -> None:
        close = getattr(self._delegate, "close", None)
        if callable(close):
            close()

    def set_progress_callback(self, callback) -> None:
        self._progress_callback = callback
        setter = getattr(self._delegate, "set_progress_callback", None)
        if callable(setter):
            setter(callback)

    def _cache_key(self, key: str) -> str:
        return f"{self._signature_prefix}:{key}" if self._signature_prefix else key

    def _emit(self, payload: dict[str, Any]) -> None:
        if callable(self._progress_callback):
            self._progress_callback(payload)


class ResilientBatchEvaluator:
    def __init__(
        self,
        preferred_factory: Callable[[], Any],
        fallback_factory: Callable[[], Any],
        *,
        description: str,
        logger: logging.Logger | None = None,
        retryable_exceptions: tuple[type[BaseException], ...] = _FALLBACK_EXCEPTIONS,
        fallback_on_timeout: bool = True,
    ):
        self._preferred_factory = preferred_factory
        self._fallback_factory = fallback_factory
        self._description = description
        self._logger = logger or logging.getLogger(__name__)
        self._fallback_on_timeout = bool(fallback_on_timeout)
        self._retryable_exceptions = tuple(
            exc for exc in retryable_exceptions
            if fallback_on_timeout or exc is not TimeoutError
        )
        self._using_fallback = False
        self._delegate = self._build_delegate()

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]):
        try:
            return self._delegate(candidates, current_mutations)
        except self._retryable_exceptions as exc:
            if not self._fallback_on_timeout and _timeout_like(exc):
                raise TimeoutError(str(exc)) from exc
            if self._using_fallback:
                raise
            self._using_fallback = True
            self._logger.warning("%s failed; retrying locally: %s", self._description, exc)
            self._close(force=True)
            self._delegate = self._fallback_factory()
            return self._delegate(candidates, current_mutations)

    def close(self) -> None:
        self._close(force=False)

    def terminate(self) -> None:
        self._close(force=True)
        self._using_fallback = False
        self._delegate = self._build_delegate()

    def _build_delegate(self):
        try:
            return self._preferred_factory()
        except self._retryable_exceptions as exc:
            self._using_fallback = True
            self._logger.warning("%s initialization failed; using local evaluator: %s", self._description, exc)
            return self._fallback_factory()

    def _close(self, *, force: bool) -> None:
        delegate = getattr(self, "_delegate", None)
        if delegate is None:
            return
        if force and callable(getattr(delegate, "terminate", None)):
            delegate.terminate()
        elif callable(getattr(delegate, "close", None)):
            delegate.close()


class SharedPoolBatchEvaluator:
    def __init__(
        self,
        pool,
        *,
        worker_fn,
        build_args: Callable[[list[Experiment], dict[str, Any]], list[Any]],
        on_terminate: Callable[[], None] | None,
        on_close: Callable[[], None] | None = None,
        description: str,
        logger: logging.Logger | None = None,
        heartbeat_seconds: float = DEFAULT_POOL_HEARTBEAT_SECONDS,
        per_candidate_timeout_seconds: float = DEFAULT_POOL_TIMEOUT_PER_CANDIDATE_SECONDS,
        minimum_timeout_seconds: float = DEFAULT_POOL_MIN_TIMEOUT_SECONDS,
    ):
        self._pool = pool
        self._worker_fn = worker_fn
        self._build_args = build_args
        self._on_terminate = on_terminate
        self._on_close = on_close
        self._description = description
        self._logger = logger
        self._heartbeat_seconds = heartbeat_seconds
        self._per_candidate_timeout_seconds = per_candidate_timeout_seconds
        self._minimum_timeout_seconds = minimum_timeout_seconds

    def __call__(self, candidates: list[Experiment], current_mutations: dict[str, Any]):
        return pool_map_with_heartbeat(
            self._pool,
            self._worker_fn,
            self._build_args(candidates, current_mutations),
            description=self._description,
            logger=self._logger,
            heartbeat_seconds=self._heartbeat_seconds,
            per_candidate_timeout_seconds=self._per_candidate_timeout_seconds,
            minimum_timeout_seconds=self._minimum_timeout_seconds,
        )

    def close(self) -> None:
        if callable(self._on_close):
            self._on_close()

    def terminate(self) -> None:
        if callable(self._on_terminate):
            self._on_terminate()


def _default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return str(value)
