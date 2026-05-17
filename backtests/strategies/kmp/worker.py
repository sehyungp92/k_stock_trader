from __future__ import annotations

from typing import Any

from backtests.core.replay_bundle import EventReplayBundle
from backtests.auto.shared.types import ScoredCandidate
from backtests.strategies.common.synthetic import make_synthetic_replay_bundle

from .phase_scoring import kmp_reject_reason, score_kmp_phase
from .runner import run_kmp_backtest

_worker_config: dict[str, Any] = {}
_worker_phase: int = 1
_worker_hard_rejects: dict[str, float] = {}
_worker_scoring_weights: dict[str, float] | None = None
_worker_replay_bundle: EventReplayBundle | None = None


def init_worker(
    config: dict[str, Any] | None,
    phase: int,
    hard_rejects: dict[str, float] | None,
    scoring_weights: dict[str, float] | None,
) -> None:
    global _worker_config, _worker_phase, _worker_hard_rejects, _worker_scoring_weights, _worker_replay_bundle
    _worker_config = dict(config or {})
    _worker_phase = int(phase)
    _worker_hard_rejects = dict(hard_rejects or {})
    _worker_scoring_weights = dict(scoring_weights) if scoring_weights else None
    _worker_replay_bundle = (
        make_synthetic_replay_bundle("kmp", _worker_config)
        if str(_worker_config.get("capability_level", "synthetic")).lower() == "synthetic"
        else None
    )


def score_candidate(args) -> ScoredCandidate:
    name, candidate_mutations, current_mutations = args
    mutations = dict(current_mutations or {})
    mutations.update(candidate_mutations or {})
    result = run_kmp_backtest(_worker_config, mutations, replay_bundle=_worker_replay_bundle)
    metrics = result.metrics
    reject = kmp_reject_reason(_worker_phase, metrics, _worker_hard_rejects)
    if reject:
        return ScoredCandidate(name=name, score=0.0, rejected=True, reject_reason=reject, metrics=metrics)
    return ScoredCandidate(name=name, score=score_kmp_phase(_worker_phase, metrics, _worker_scoring_weights), metrics=metrics)
