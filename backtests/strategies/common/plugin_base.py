from __future__ import annotations

import concurrent.futures
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable

from backtests.auto.shared.cache_keys import build_cache_key
from backtests.auto.shared.plugin_utils import (
    CachedBatchEvaluator,
    ResilientBatchEvaluator,
    SharedPoolBatchEvaluator,
    create_process_pool,
    mutation_signature,
    resolve_worker_processes,
    shutdown_process_pool,
)
from backtests.auto.shared.types import ScoredCandidate

logger = logging.getLogger(__name__)

OFFICIAL_MTM_BASIS = "SimBroker.equity_curve_bar_level_mtm"
PHASE_FRAMEWORK_VERSION = "shared-phase-auto-v2-official-mtm-contract"


def attach_official_metric_contract(
    metrics: dict[str, Any],
    *,
    primary_metric: str = "official_mtm_net_return_pct",
    requires_audit_pass: bool = False,
    audit_pass: bool | None = None,
    audit_status: str = "direct_official_replay",
    official_replay_pass: bool = True,
    execution_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics.setdefault("official_metric_basis", OFFICIAL_MTM_BASIS)
    metrics["primary_promotion_metric"] = str(primary_metric or metrics.get("primary_promotion_metric") or "")
    metrics["primary_promotion_value"] = metrics.get(
        metrics["primary_promotion_metric"],
        metrics.get("primary_promotion_value"),
    )
    promotion_basis = (
        metrics["official_metric_basis"]
        if primary_metric
        else metrics.get("primary_promotion_basis") or metrics["official_metric_basis"]
    )
    metrics["primary_promotion_basis"] = str(promotion_basis)
    metrics["promotion_requires_audit_pass"] = bool(
        requires_audit_pass or metrics.get("promotion_requires_audit_pass", False)
    )
    metrics["official_replay_pass"] = bool(metrics.get("official_replay_pass", official_replay_pass))
    metrics["audit_pass"] = bool(metrics.get("audit_pass", False if audit_pass is None else audit_pass))
    metrics["audit_status"] = str(metrics.get("audit_status") or audit_status)
    contract = dict(metrics.get("metric_contract") or {})
    official_metrics = list(contract.get("official_metrics") or [primary_metric])
    if primary_metric and primary_metric not in official_metrics:
        official_metrics.insert(0, primary_metric)
    contract.update(
        {
            "primary_promotion_metric": metrics["primary_promotion_metric"],
            "primary_promotion_value": metrics["primary_promotion_value"],
            "primary_promotion_basis": metrics["primary_promotion_basis"],
            "promotion_requires_audit_pass": metrics["promotion_requires_audit_pass"],
            "official_replay_pass": metrics["official_replay_pass"],
            "audit_pass": metrics["audit_pass"],
            "audit_status": metrics["audit_status"],
            "official_metrics": official_metrics,
            "proxy_metrics": list(contract.get("proxy_metrics") or []),
            "legacy_closed_trade_metrics": list(contract.get("legacy_closed_trade_metrics") or ["net_return_pct"]),
            "closed_trade_return_basis": metrics.get("net_return_pct_basis", ""),
            "required_hygiene_metrics": list(
                contract.get("required_hygiene_metrics")
                or [
                    "same_bar_fill_count",
                    "forced_replay_close_count",
                    "rejected_order_count",
                    "end_open_position_count",
                ]
            ),
        }
    )
    if execution_contract:
        contract["execution_contract"] = dict(execution_contract)
        metrics["execution_contract"] = dict(execution_contract)
    metrics["metric_contract"] = contract
    return metrics


def build_execution_contract(
    plugin_or_context: Any,
    metrics: dict[str, Any] | None = None,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the behavior identity used for promotion and cache compatibility."""
    metrics = dict(metrics or {})
    if isinstance(plugin_or_context, dict):
        plugin = None
        context = dict(plugin_or_context)
        config = dict(context.get("config") or {})
    else:
        plugin = plugin_or_context
        context = dict(getattr(plugin, "execution_context", {}) or {})
        config = dict(getattr(plugin, "config", {}) or {})
    metric_contract = metrics.get("metric_contract") if isinstance(metrics.get("metric_contract"), dict) else {}
    nested_contract = metrics.get("execution_contract") or metric_contract.get("execution_contract") or {}
    nested_context = dict(nested_contract) if isinstance(nested_contract, dict) else {}

    def pick(*keys: str, default: Any = "") -> Any:
        for key in keys:
            if key in metrics and metrics.get(key) not in (None, ""):
                return metrics[key]
            if key in nested_context and nested_context.get(key) not in (None, ""):
                return nested_context[key]
            if key in context and context.get(key) not in (None, ""):
                return context[key]
            if key in config and config.get(key) not in (None, ""):
                return config[key]
        return default

    last_result = getattr(plugin, "_last_result", None) if plugin is not None else None

    def result_attr(name: str) -> Any:
        if last_result is None:
            return ""
        return getattr(last_result, name, "")

    strategy = pick("strategy", default=getattr(plugin, "name", "") if plugin is not None else "")
    initial_equity = pick("initial_equity", default=getattr(plugin, "initial_equity", ""))
    source_fingerprint = (
        metrics.get("source_fingerprint")
        or nested_context.get("source_fingerprint")
        or result_attr("source_fingerprint")
        or context.get("source_fingerprint")
        or (getattr(plugin, "source_fingerprint", "") if plugin is not None else "")
    )
    feature_hash = (
        metrics.get("feature_manifest_hash")
        or metrics.get("feature_bundle_hash")
        or nested_context.get("feature_manifest_hash")
        or nested_context.get("feature_bundle_hash")
        or result_attr("feature_bundle_hash")
        or context.get("feature_manifest_hash")
        or context.get("feature_bundle_hash")
        or (getattr(plugin, "feature_manifest_hash", "") if plugin is not None else "")
    )
    candidate_hash = (
        metrics.get("candidate_snapshot_hash")
        or nested_context.get("candidate_snapshot_hash")
        or result_attr("candidate_snapshot_hash")
        or context.get("candidate_snapshot_hash")
    )
    contract = {
        "strategy": strategy,
        "phase_framework_version": pick("phase_framework_version", default=PHASE_FRAMEWORK_VERSION),
        "strategy_core_version": pick("strategy_core_version"),
        "source_fingerprint": source_fingerprint,
        "feature_manifest_hash": feature_hash,
        "candidate_snapshot_hash": candidate_hash,
        "date_window": {
            "start": pick("start_date", "train_start", "holdout_start"),
            "end": pick("end_date", "train_end"),
        },
        "initial_equity": initial_equity,
        "cost_policy": pick("cost_policy", "cost_model", default={}),
        "fill_timing": pick("live_parity_fill_timing", "fill_timing"),
        "auction_mode": pick("auction_mode", "auction_timing_mode"),
        "capability_level": pick("capability_level"),
        "replay_mode": pick("replay_mode"),
        "primary_promotion_metric": pick("primary_promotion_metric", default="official_mtm_net_return_pct"),
        "primary_promotion_basis": pick("primary_promotion_basis", "official_metric_basis", default=OFFICIAL_MTM_BASIS),
    }
    if extra:
        contract.update(extra)
    return _drop_empty_contract_values(contract)


def _drop_empty_contract_values(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {
            key: _drop_empty_contract_values(item)
            for key, item in value.items()
            if item not in (None, "")
        }
        return {key: item for key, item in cleaned.items() if item not in ({}, [])}
    if isinstance(value, (list, tuple)):
        return [_drop_empty_contract_values(item) for item in value if item not in (None, "")]
    if isinstance(value, Path):
        return str(value)
    return value


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


class ThreadBatchEvaluator:
    def __init__(
        self,
        init_worker: Callable,
        score_candidate: Callable,
        initargs: tuple[Any, ...],
        *,
        max_workers: int | None,
        description: str,
        heartbeat_seconds: float,
        per_candidate_timeout_seconds: float,
        minimum_timeout_seconds: float,
    ):
        init_worker(*initargs)
        self._score_candidate = score_candidate
        self._max_workers = resolve_worker_processes(max_workers)
        self._description = description
        self._heartbeat_seconds = float(heartbeat_seconds)
        self._per_candidate_timeout_seconds = float(per_candidate_timeout_seconds)
        self._minimum_timeout_seconds = float(minimum_timeout_seconds)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix=description.replace(" ", "-"),
        )

    def __call__(self, candidates, current_mutations):
        if not candidates:
            return []
        args = [
            (candidate.name, candidate.mutations, current_mutations)
            for candidate in candidates
        ]
        results = [None] * len(args)
        futures = {
            self._executor.submit(self._score_candidate, arg): index
            for index, arg in enumerate(args)
        }
        timeout = max(self._minimum_timeout_seconds, len(args) * self._per_candidate_timeout_seconds)
        started = time.monotonic()
        next_heartbeat = started + max(1.0, self._heartbeat_seconds)
        completed = 0
        try:
            while futures:
                elapsed = time.monotonic() - started
                if elapsed >= timeout:
                    raise TimeoutError(f"{self._description} exceeded timeout after {elapsed:.0f}s")
                done, _ = concurrent.futures.wait(
                    tuple(futures),
                    timeout=min(max(self._heartbeat_seconds / 3.0, 1.0), 5.0),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                progressed = False
                for future in done:
                    index = futures.pop(future)
                    results[index] = future.result()
                    completed += 1
                    progressed = True
                if progressed:
                    next_heartbeat = time.monotonic() + max(1.0, self._heartbeat_seconds)
                elif time.monotonic() >= next_heartbeat:
                    logger.info(
                        "%s still running after %.0fs (%d/%d completed, %d/%d worker thread(s) active)",
                        self._description,
                        elapsed,
                        completed,
                        len(args),
                        min(len(futures), self._max_workers),
                        self._max_workers,
                    )
                    next_heartbeat = time.monotonic() + max(1.0, self._heartbeat_seconds)
        except Exception:
            for future in futures:
                future.cancel()
            raise
        if any(result is None for result in results):
            raise RuntimeError(f"{self._description} completed with missing thread results")
        return [result for result in results if result is not None]

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)


class SharedStrategyPluginMixin:
    _shared_pool = None

    def _remember_backtest_result(self, mutations: dict[str, Any], result: Any) -> Any:
        self._last_result = result
        self._last_mutation_signature = mutation_signature(mutations)
        metrics_cache = getattr(self, "_metrics_cache", None)
        if isinstance(metrics_cache, dict) and getattr(result, "metrics", None):
            metrics_cache[self._last_mutation_signature] = dict(result.metrics)
        return result

    def _last_result_for_mutations(self, mutations: dict[str, Any]) -> Any | None:
        if getattr(self, "_last_mutation_signature", None) != mutation_signature(mutations):
            return None
        return getattr(self, "_last_result", None)

    def _get_or_create_pool(self, init_worker: Callable, initargs: tuple[Any, ...]):
        init_signature = build_cache_key(
            "shared_strategy_pool_init",
            extra={"initargs": initargs},
        )
        if self._shared_pool is not None and getattr(self, "_shared_pool_init_signature", "") != init_signature:
            shutdown_process_pool(self._shared_pool)
            self._shared_pool = None
            self._shared_pool_init_signature = ""
        if self._shared_pool is None:
            self._shared_pool = create_process_pool(self.max_workers, initializer=init_worker, initargs=initargs)
            self._shared_pool_init_signature = init_signature
        return self._shared_pool

    def close_pool(self) -> None:
        shutdown_process_pool(self._shared_pool)
        self._shared_pool = None
        self._shared_pool_init_signature = ""

    def _destroy_pool(self) -> None:
        shutdown_process_pool(self._shared_pool, force=True)
        self._shared_pool = None
        self._shared_pool_init_signature = ""

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
        reject_on_timeout: bool | None = None,
    ):
        baseline_key = mutation_signature(cumulative_mutations)
        seed = {baseline_key: baseline_result}
        reject_timeouts = bool(
            reject_on_timeout
            if reject_on_timeout is not None
            else getattr(self, "reject_evaluation_timeouts", False)
        )

        def local_factory():
            return LocalBatchEvaluator(init_worker, score_candidate, initargs)

        def preferred_factory():
            if int(self.max_workers or 1) <= 1 or (sys.platform == "win32" and not _supports_spawn()):
                return local_factory()
            if sys.platform == "win32" and bool(getattr(self, "prefer_thread_evaluator_on_windows", False)):
                return ThreadBatchEvaluator(
                    init_worker,
                    score_candidate,
                    initargs,
                    max_workers=self.max_workers,
                    description=description,
                    heartbeat_seconds=heartbeat_seconds,
                    per_candidate_timeout_seconds=per_candidate_timeout_seconds,
                    minimum_timeout_seconds=minimum_timeout_seconds,
                )
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
            fallback_on_timeout=not reject_timeouts,
        )
        signature_prefix = build_cache_key(
            f"{self.name}.phase_eval",
            source_fingerprint=self.source_fingerprint,
            extra={
                "phase": phase,
                "scoring_weights": scoring_weights or {},
                "hard_rejects": hard_rejects or {},
                "capability_level": self.capability_level,
                **_cache_identity_context(self),
            },
        )
        return CachedBatchEvaluator(
            resilient,
            cache=self._evaluation_cache,
            seed_results=seed,
            signature_prefix=signature_prefix,
            metrics_cache=self._metrics_cache,
            max_batch_size=max_eval_batch_size,
            reject_on_timeout=reject_timeouts,
        )


def _supports_spawn() -> bool:
    if sys.platform != "win32":
        return True
    main_module = sys.modules.get("__main__")
    main_path = getattr(main_module, "__file__", "")
    return bool(main_path) and not str(main_path).startswith("<")


def _cache_identity_context(plugin: Any) -> dict[str, Any]:
    context = dict(getattr(plugin, "execution_context", {}) or {})
    last_result = getattr(plugin, "_last_result", None)
    metrics = dict(getattr(last_result, "metrics", {}) or {})
    return {
        "feature_manifest_hash": getattr(plugin, "feature_manifest_hash", "") or context.get("feature_manifest_hash", ""),
        "strategy_core_version": context.get("strategy_core_version", ""),
        "candidate_snapshot_hash": getattr(last_result, "candidate_snapshot_hash", ""),
        "execution_contract": build_execution_contract(plugin, metrics),
        "initial_equity": context.get("initial_equity", getattr(plugin, "initial_equity", "")),
        "cost_policy": context.get("cost_policy", getattr(plugin, "cost_policy", "")),
        "fill_timing": context.get("live_parity_fill_timing", context.get("fill_timing", "")),
        "auction_mode": context.get("auction_mode", ""),
        "date_window": {
            "start": context.get("start_date", getattr(plugin, "start_date", "")),
            "end": context.get("end_date", getattr(plugin, "end_date", "")),
        },
    }
