from __future__ import annotations

from typing import Any

from backtests.auto.shared.types import ScoredCandidate
from backtests.core.replay_bundle import EventReplayBundle

from .phase_scoring import olr_reject_reason, score_olr_phase
from .replay_cache import load_olr_real_replay_bundle, olr_candidate_bundle_changes, warm_olr_stage1_replay_cache
from .runner import run_olr_backtest, snapshots_from_bundle


_worker_config: dict[str, Any] = {}
_worker_phase: int = 1
_worker_hard_rejects: dict[str, float] = {}
_worker_scoring_weights: dict[str, float] | None = None
_worker_replay_bundle: EventReplayBundle | None = None
_worker_base_mutations: dict[str, Any] = {}


def init_worker(
    config: dict[str, Any] | None,
    phase: int,
    hard_rejects: dict[str, float] | None,
    scoring_weights: dict[str, float] | None,
    base_mutations: dict[str, Any] | None = None,
) -> None:
    global _worker_config, _worker_phase, _worker_hard_rejects, _worker_scoring_weights, _worker_replay_bundle, _worker_base_mutations
    _worker_config = dict(config or {})
    _worker_phase = int(phase)
    _worker_hard_rejects = dict(hard_rejects or {})
    _worker_scoring_weights = dict(scoring_weights) if scoring_weights else None
    _worker_base_mutations = dict(base_mutations or {})
    capability_level = str(_worker_config.get("capability_level", "real_replay")).lower()
    if capability_level == "synthetic":
        _worker_replay_bundle = None
    elif capability_level in {"real", "real_replay", "parquet", "krx_replay"}:
        _worker_replay_bundle = load_olr_real_replay_bundle(_worker_config, _worker_base_mutations)
        if _worker_phase in {1, 2, 6}:
            warm_olr_stage1_replay_cache(_worker_config, _worker_base_mutations)
    else:
        _worker_replay_bundle = None


def score_candidate(args) -> ScoredCandidate:
    name, candidate_mutations, current_mutations = args
    from .plugin import _augment_snapshot_label_metrics, _augment_trade_alpha_metrics, _canonicalize_olr_mutations

    mutations = dict(current_mutations or {})
    mutations.update(candidate_mutations or {})
    mutations = _canonicalize_olr_mutations(mutations)
    capability_level = str(_worker_config.get("capability_level", "real_replay")).lower()
    replay_bundle = _resolve_replay_bundle(candidate_mutations or {}, mutations, capability_level)
    result = run_olr_backtest(_worker_config, mutations, replay_bundle=replay_bundle)
    metrics = dict(result.metrics)
    _augment_snapshot_label_metrics(metrics, snapshots_from_bundle(replay_bundle) if replay_bundle is not None else {})
    _augment_trade_alpha_metrics(metrics, result.trades)
    metrics["phase_candidate_metric_basis"] = "direct_official_training_replay_holdout_excluded"
    metrics["paper_live_parity_required"] = True
    metrics["paper_live_parity_status"] = "required_before_promotion"
    score = score_olr_phase(_worker_phase, metrics, _worker_scoring_weights)
    reject = olr_reject_reason(_worker_phase, metrics, _worker_hard_rejects)
    if reject:
        return ScoredCandidate(name=name, score=score, rejected=True, reject_reason=reject, metrics=metrics)
    return ScoredCandidate(name=name, score=score, metrics=metrics)


def _resolve_replay_bundle(
    candidate_mutations: dict[str, Any],
    merged_mutations: dict[str, Any],
    capability_level: str,
) -> EventReplayBundle | None:
    if capability_level == "synthetic":
        return None
    if capability_level in {"real", "real_replay", "parquet", "krx_replay"}:
        if olr_candidate_bundle_changes(candidate_mutations):
            return load_olr_real_replay_bundle(_worker_config, merged_mutations)
        return _worker_replay_bundle
    return None
