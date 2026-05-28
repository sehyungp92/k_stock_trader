from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .phase_state import _atomic_write_json
from .types import Experiment, GreedyResult, GreedyRound, ScoredCandidate

EvaluateFn = Callable[[list[Experiment], dict[str, object]], list[ScoredCandidate]]


def _delta_ratio(score: float, baseline: float) -> float:
    return (score - baseline) / baseline if baseline > 0 else score - baseline


def _checkpoint_identity(base_mutations: dict[str, object], candidate_names: list[str], context: dict[str, object] | None) -> str:
    raw = json.dumps(
        {"base_mutations": base_mutations, "candidate_names": sorted(candidate_names), "context": context or {}},
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def load_greedy_checkpoint(path: Path):
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return (
        dict(data.get("current_mutations", {})),
        list(data.get("kept_features", [])),
        float(data.get("current_score", 0.0)),
        set(data.get("remaining_names", [])),
        [GreedyRound(**item) for item in data.get("rounds", [])],
        data.get("checkpoint_identity", ""),
    )


def save_greedy_checkpoint(
    current_mutations: dict[str, object],
    kept_features: list[str],
    current_score: float,
    remaining_names: list[str],
    rounds: list[GreedyRound],
    checkpoint_identity: str,
    path: Path,
) -> None:
    _atomic_write_json(
        {
            "current_mutations": current_mutations,
            "kept_features": kept_features,
            "current_score": current_score,
            "remaining_names": remaining_names,
            "rounds": [asdict(item) for item in rounds],
            "checkpoint_identity": checkpoint_identity,
        },
        path,
    )


def run_greedy(
    candidates: list[Experiment],
    base_mutations: dict[str, object],
    evaluate_batch: EvaluateFn,
    *,
    max_rounds: int = 20,
    min_delta: float = 0.001,
    prune_threshold: float = 0.05,
    reject_streak_limit: int = 2,
    checkpoint_path: Path | None = None,
    checkpoint_context: dict[str, object] | None = None,
    logger: logging.Logger | None = None,
    progress_callback=None,
) -> GreedyResult:
    started = time.time()
    log = logger or logging.getLogger(__name__)
    current_mutations = dict(base_mutations)
    kept_features: list[str] = []
    rounds: list[GreedyRound] = []
    remaining = list(candidates)
    reject_streak: dict[str, int] = {}
    identity = _checkpoint_identity(base_mutations, [item.name for item in candidates], checkpoint_context)
    baseline = evaluate_batch([Experiment("__baseline__", {})], current_mutations)
    base_score = baseline[0].score if baseline else 0.0
    current_score = base_score
    candidate_evaluations: list[dict[str, object]] = []
    _emit(progress_callback, {"event": "baseline_complete", "base_score": base_score})

    if checkpoint_path:
        resumed = load_greedy_checkpoint(checkpoint_path)
        if resumed and resumed[-1] == identity:
            current_mutations, kept_features, current_score, remaining_names, rounds, _ = resumed
            remaining = [item for item in candidates if item.name in remaining_names]
            log.info("Resumed greedy checkpoint with %d remaining candidates.", len(remaining))

    try:
        for round_num in range(len(rounds) + 1, max_rounds + 1):
            if not remaining:
                break
            _emit(progress_callback, {"event": "round_start", "round_num": round_num, "candidate_count": len(remaining)})
            scored = evaluate_batch(remaining, current_mutations)
            rejected_names = {item.name for item in scored if item.rejected}
            for candidate in remaining:
                reject_streak[candidate.name] = reject_streak.get(candidate.name, 0) + 1 if candidate.name in rejected_names else 0
            valid = [item for item in scored if not item.rejected]
            if not valid:
                candidate_evaluations.extend(
                    _candidate_evaluations(round_num, scored, current_score=current_score, best_name="", kept_name="")
                )
                rounds.append(GreedyRound(round_num, len(remaining), "", current_score, 0.0, False, len(scored)))
                break
            best = max(valid, key=lambda item: item.score)
            delta = _delta_ratio(best.score, current_score)
            kept = best.score > current_score and delta >= min_delta
            candidate_evaluations.extend(
                _candidate_evaluations(
                    round_num,
                    scored,
                    current_score=current_score,
                    best_name=best.name,
                    kept_name=best.name if kept else "",
                )
            )
            rounds.append(GreedyRound(round_num, len(remaining), best.name, best.score, delta * 100.0, kept, len(scored) - len(valid)))
            if not kept:
                break
            chosen = next(item for item in remaining if item.name == best.name)
            previous_score = current_score
            current_mutations.update(chosen.mutations)
            current_score = best.score
            kept_features.append(best.name)
            score_by_name = {item.name: _delta_ratio(item.score, previous_score) for item in valid}
            remaining = [
                item for item in remaining
                if item.name != best.name
                and score_by_name.get(item.name, 0.0) >= -prune_threshold
                and reject_streak.get(item.name, 0) < reject_streak_limit
            ]
            if checkpoint_path:
                save_greedy_checkpoint(
                    current_mutations,
                    kept_features,
                    current_score,
                    [item.name for item in remaining],
                    rounds,
                    identity,
                    checkpoint_path,
                )
            _emit(progress_callback, {"event": "round_complete", "round_num": round_num, "best_name": best.name, "current_score": current_score})
    finally:
        close = getattr(evaluate_batch, "close", None)
        if callable(close):
            close()
    if checkpoint_path and checkpoint_path.exists():
        checkpoint_path.unlink()
    return GreedyResult(
        base_score=base_score,
        final_score=current_score,
        final_mutations=current_mutations,
        kept_features=kept_features,
        rounds=rounds,
        final_metrics={},
        total_candidates=len(candidates),
        accepted_count=len(kept_features),
        elapsed_seconds=time.time() - started,
        candidate_evaluations=candidate_evaluations,
    )


def _emit(callback, payload: dict[str, object]) -> None:
    if callable(callback):
        callback(payload)


def _candidate_evaluations(
    round_num: int,
    scored: list[ScoredCandidate],
    *,
    current_score: float,
    best_name: str,
    kept_name: str,
) -> list[dict[str, object]]:
    return [
        {
            "round_num": int(round_num),
            "name": item.name,
            "score": float(item.score),
            "score_delta_pct": _delta_ratio(float(item.score), float(current_score)) * 100.0,
            "rejected": bool(item.rejected),
            "reject_reason": item.reject_reason,
            "is_best": item.name == best_name,
            "kept": item.name == kept_name,
            "metrics": _scalar_metrics(item.metrics),
        }
        for item in scored
    ]


def _scalar_metrics(metrics: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[str(key)] = value
    return result
