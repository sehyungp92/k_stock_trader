from __future__ import annotations

from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

from backtests.auto.shared.cache_keys import build_cache_key, stable_signature
from backtests.core.replay_bundle import EventReplayBundle
from strategy_common.market import MarketBar
from strategy_common.sector_daily import SECTOR_DAILY_VERSION
from strategy_olr.artifact_store import OLRArtifactStore
from strategy_olr.config import OLRConfig, OLR_CORE_VERSION
from strategy_olr.research import (
    FINAL_CANDIDATE_CONFIG_HASH_VERSION,
    afternoon_selection_from_contexts,
    final_candidate_config_fingerprint,
)

from .research_sweep import (
    DEFAULT_EXPECTED_UNIVERSE_SIZE,
    OLRResearchSweepDataset,
    _training_config,
    afternoon_contexts_for_snapshots,
    prepare_research_sweep_dataset,
    research_snapshots_for_dataset,
    snapshots_for_experiment,
)
from .runner import attach_overnight_labels_to_snapshots, compile_olr_replay_bundle


_BUNDLE_CACHE: dict[str, EventReplayBundle] = {}
_DATASET_CACHE: dict[str, OLRResearchSweepDataset] = {}
_RESEARCH_SNAPSHOT_CACHE: dict[str, dict[date, Any]] = {}
_STAGE1_SNAPSHOT_CACHE: dict[str, dict[date, Any]] = {}
_AFTERNOON_CONTEXT_CACHE: dict[str, dict[date, dict[str, Any]]] = {}
_OFFICIAL_AUCTION_EXIT_RECOVERY_SESSIONS = 2

_STAGE1_MUTATION_PREFIXES = (
    "olr.universe",
    "olr.frontier",
    "olr.discovery",
    "olr.premarket",
    "olr.research",
    "olr.signal",
)

_CANDIDATE_MUTATION_PREFIXES = (
    *_STAGE1_MUTATION_PREFIXES,
    "olr.afternoon",
)

_EXECUTION_ONLY_CONFIG_FIELDS = {
    "live_parity_fill_timing",
    "entry_mode",
    "exit_mode",
    "allocation_mode",
    "target_gross_exposure",
    "max_position_pct",
    "rank_decay",
    "min_selected",
    "auction_fill_time",
    "auction_limit_offset_bps",
    "auction_adverse_bps",
    "auction_nonfill_rate",
    "market_entry_price_buffer_bps",
    "trade_entry_plan",
    "trade_exit_plan",
    "slippage_bps",
    "commission_bps",
    "tax_bps_on_sell",
    "overnight_slot_count",
}


def load_olr_real_replay_bundle(
    config: dict[str, Any] | None = None,
    mutations: dict[str, Any] | None = None,
) -> EventReplayBundle:
    raw_config = dict(config or {})
    raw_mutations = dict(mutations or {})
    holdout_days = int(raw_config.get("holdout_days", 42) or 42)
    training_config = _training_config(raw_config, holdout_days)
    cfg = OLRConfig.from_mapping(training_config, raw_mutations)
    dataset = _load_dataset(training_config, cfg)
    stage1_config_hash = _stage1_config_hash(cfg, raw_mutations)
    candidate_config_hash = _candidate_config_hash(cfg, raw_mutations)
    bundle_key = build_cache_key(
        "olr.real_event_replay_bundle",
        source_fingerprint=dataset.source_fingerprint,
        mutations=raw_mutations,
        mutation_prefixes=_CANDIDATE_MUTATION_PREFIXES,
        extra={
            "data_root": str(dataset.data_root.resolve()),
            "daily_data_root": str(dataset.daily_data_root.resolve()),
            "timeframe": dataset.timeframe,
            "train_start": dataset.train_start.isoformat(),
            "train_end": dataset.train_end.isoformat(),
            "holdout_start": dataset.holdout_start.isoformat(),
            "stage1_config_hash": stage1_config_hash,
            "candidate_config_hash": candidate_config_hash,
            "artifact_root": str(raw_config.get("artifact_root", "data/strategy/olr")),
            "core_version": OLR_CORE_VERSION,
            "sector_daily_version": SECTOR_DAILY_VERSION,
            "bar_scope": "stage2_candidates_plus_train_only_auction_exit_recovery",
        },
    )
    cached = _BUNDLE_CACHE.get(bundle_key)
    if cached is not None:
        return cached

    eligible_dates = _eligible_snapshot_dates(dataset.trading_dates)
    snapshots = _load_or_build_stage2_snapshots(
        dataset,
        cfg,
        raw_mutations,
        eligible_dates,
        stage1_config_hash,
        candidate_config_hash,
        raw_config,
    )
    snapshots = attach_overnight_labels_to_snapshots(snapshots, dataset.overnight_labels_by_key)
    replay_bars, bar_scope = _filtered_training_bars_for_snapshots(dataset, snapshots)
    source_fingerprint = stable_signature(
        {
            "dataset": dataset.source_fingerprint,
            "candidate_config_hash": candidate_config_hash,
            "snapshots": {day.isoformat(): snapshot.artifact_hash for day, snapshot in snapshots.items()},
            "bar_scope": bar_scope,
            "holdout_excluded": True,
        }
    )
    bundle = compile_olr_replay_bundle(
        replay_bars,
        snapshots,
        source_fingerprint=source_fingerprint,
        data_root=dataset.data_root,
        config=training_config,
    )
    bundle.metadata.update(
        {
            "replay_mode": "olr_core_simbroker_cached_training",
            "capability_level": "real_replay",
            "source_fingerprint": dataset.source_fingerprint,
            "candidate_config_hash": candidate_config_hash,
            "olr_stage1_config_hash": stage1_config_hash,
            "olr_candidate_artifact_hashes": {day.isoformat(): snapshot.artifact_hash for day, snapshot in snapshots.items()},
            "olr_candidate_artifact_hash": stable_signature({day.isoformat(): snapshot.artifact_hash for day, snapshot in snapshots.items()}),
            "olr_feature_bundle_hash": stable_signature(
                {
                    "dataset": dataset.source_fingerprint,
                    "candidate_config_hash": candidate_config_hash,
                    "snapshots": {day.isoformat(): snapshot.artifact_hash for day, snapshot in snapshots.items()},
                    "core_version": OLR_CORE_VERSION,
                }
            ),
            "sector_daily_version": SECTOR_DAILY_VERSION,
            "holdout_excluded": True,
            "holdout_policy": "Dates >= holdout_start are excluded from phased auto unless include_holdout is explicitly used outside this cache.",
            "training_window": {
                "start": dataset.train_start.isoformat(),
                "end": dataset.train_end.isoformat(),
                "holdout_start": dataset.holdout_start.isoformat(),
                "eligible_snapshot_start": eligible_dates[0].isoformat() if eligible_dates else "",
                "eligible_snapshot_end": eligible_dates[-1].isoformat() if eligible_dates else "",
                "sessions": len(dataset.trading_dates),
                "eligible_sessions": len(eligible_dates),
                "replayed_selected_symbol_bars": len(replay_bars),
                "bar_scope": bar_scope,
            },
            "candidate_replay_scope": "Stage 1/2 snapshots are source-fingerprinted from train-only data; replay bars include selected Stage 2 symbols on trade date plus train-only recovery sessions.",
            "causality_policy": {
                "daily_row_cutoff": "row_date < trade_date",
                "flow_row_cutoff": "row_date < trade_date",
                "intraday_selection_cutoff": "timestamp < 14:30 KST",
                "holdout_excluded": True,
            },
        }
    )
    _BUNDLE_CACHE[bundle_key] = bundle
    return bundle


def warm_olr_real_replay_cache(config: dict[str, Any] | None = None, mutations: dict[str, Any] | None = None) -> None:
    load_olr_real_replay_bundle(config, mutations)


def warm_olr_stage1_replay_cache(config: dict[str, Any] | None = None, mutations: dict[str, Any] | None = None) -> None:
    raw_config = dict(config or {})
    raw_mutations = dict(mutations or {})
    holdout_days = int(raw_config.get("holdout_days", 42) or 42)
    training_config = _training_config(raw_config, holdout_days)
    cfg = OLRConfig.from_mapping(training_config, raw_mutations)
    dataset = _load_dataset(training_config, cfg)
    stage1_config_hash = _stage1_config_hash(cfg, raw_mutations)
    stage1 = _stage1_snapshots(dataset, raw_mutations, stage1_config_hash, raw_config)
    _afternoon_contexts(dataset, stage1, stage1_config_hash)


def olr_candidate_bundle_changes(mutations: dict[str, Any] | None) -> bool:
    for key in dict(mutations or {}):
        if any(key == prefix or key.startswith(f"{prefix}.") for prefix in _CANDIDATE_MUTATION_PREFIXES):
            return True
    return False


def _load_dataset(training_config: dict[str, Any], cfg: OLRConfig) -> OLRResearchSweepDataset:
    key = build_cache_key(
        "olr.research_dataset",
        extra={
            "config": _dataset_cache_config(training_config),
            "expected_universe_size": int(cfg.complete_universe_size or DEFAULT_EXPECTED_UNIVERSE_SIZE),
        },
    )
    cached = _DATASET_CACHE.get(key)
    if cached is not None:
        return cached
    dataset = prepare_research_sweep_dataset(
        training_config,
        holdout_days=int(training_config.get("holdout_days", 42) or 42),
        expected_universe_size=int(cfg.complete_universe_size or DEFAULT_EXPECTED_UNIVERSE_SIZE),
        include_holdout=False,
    )
    _DATASET_CACHE[key] = dataset
    return dataset


def _dataset_cache_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: config.get(key)
        for key in (
            "data_root",
            "daily_data_root",
            "timeframe",
            "universe",
            "universe_file",
            "symbols",
            "holdout_days",
            "date_range",
            "start_date",
            "end_date",
            "train_start",
            "train_end",
            "initial_equity",
        )
        if key in config
    }


def _load_or_build_stage2_snapshots(
    dataset: OLRResearchSweepDataset,
    cfg: OLRConfig,
    mutations: dict[str, Any],
    eligible_dates: tuple[date, ...],
    stage1_config_hash: str,
    candidate_config_hash: str,
    raw_config: dict[str, Any],
) -> dict[date, Any]:
    artifact_root = Path(raw_config.get("artifact_root", "data/strategy/olr")) / "candidate_snapshots" / candidate_config_hash[:16]
    store = OLRArtifactStore(artifact_root)
    snapshots: dict[date, Any] = {}
    missing: list[date] = []
    for trade_date in eligible_dates:
        cached = store.load_snapshot(trade_date)
        if (
            cached is not None
            and cached.source_fingerprint == dataset.source_fingerprint
            and cached.metadata.get("candidate_config_hash") == candidate_config_hash
            and cached.metadata.get("final_candidate_config_hash") == candidate_config_hash
        ):
            snapshots[trade_date] = cached
        else:
            missing.append(trade_date)
    if missing:
        stage1 = _stage1_snapshots(dataset, mutations, stage1_config_hash, raw_config)
        contexts = _afternoon_contexts(dataset, stage1, stage1_config_hash)
        for trade_date in eligible_dates:
            if trade_date in snapshots:
                continue
            base = stage1.get(trade_date)
            if base is None:
                continue
            selected = afternoon_selection_from_contexts(base, contexts.get(trade_date, {}), cfg)
            selected.metadata.update(
                {
                    "candidate_config_hash": candidate_config_hash,
                    "final_candidate_config_hash": candidate_config_hash,
                    "final_candidate_config_hash_version": FINAL_CANDIDATE_CONFIG_HASH_VERSION,
                    "stage1_config_hash": stage1_config_hash,
                    "source_fingerprint": dataset.source_fingerprint,
                    "holdout_excluded": True,
                }
            )
            store.save_snapshot(selected)
            snapshots[trade_date] = selected
    return dict(sorted(snapshots.items()))


def _stage1_snapshots(
    dataset: OLRResearchSweepDataset,
    mutations: dict[str, Any],
    stage1_config_hash: str,
    raw_config: dict[str, Any],
) -> dict[date, Any]:
    key = stable_signature(
        {
            "dataset": dataset.source_fingerprint,
            "stage1_config_hash": stage1_config_hash,
            "stage1_scope": "daily_research_selection_snapshots",
        }
    )
    cached = _STAGE1_SNAPSHOT_CACHE.get(key)
    if cached is not None:
        return cached
    loaded = _load_stage1_snapshots_from_store(dataset, stage1_config_hash, raw_config)
    if loaded:
        _STAGE1_SNAPSHOT_CACHE[key] = loaded
        return loaded
    research_cache = _research_snapshots(dataset, mutations, stage1_config_hash)
    snapshots = snapshots_for_experiment(dataset, mutations, research_snapshots=research_cache)
    _save_stage1_snapshots_to_store(snapshots, dataset, stage1_config_hash, raw_config)
    _STAGE1_SNAPSHOT_CACHE[key] = snapshots
    return snapshots


def _load_stage1_snapshots_from_store(
    dataset: OLRResearchSweepDataset,
    stage1_config_hash: str,
    raw_config: dict[str, Any],
) -> dict[date, Any]:
    store = _stage1_artifact_store(raw_config, stage1_config_hash)
    snapshots: dict[date, Any] = {}
    for trade_date in dataset.trading_dates:
        cached = store.load_snapshot(trade_date)
        if (
            cached is not None
            and cached.source_fingerprint == dataset.source_fingerprint
            and cached.metadata.get("stage1_config_hash") == stage1_config_hash
        ):
            snapshots[trade_date] = cached
    if len(snapshots) == len(dataset.trading_dates):
        return dict(sorted(snapshots.items()))
    return {}


def _save_stage1_snapshots_to_store(
    snapshots: dict[date, Any],
    dataset: OLRResearchSweepDataset,
    stage1_config_hash: str,
    raw_config: dict[str, Any],
) -> None:
    store = _stage1_artifact_store(raw_config, stage1_config_hash)
    for snapshot in snapshots.values():
        snapshot.metadata.update(
            {
                "stage1_config_hash": stage1_config_hash,
                "source_fingerprint": dataset.source_fingerprint,
                "holdout_excluded": True,
                "stage1_artifact_scope": "pre_afternoon_daily_research_selection",
            }
        )
        store.save_snapshot(snapshot)


def _stage1_artifact_store(raw_config: dict[str, Any], stage1_config_hash: str) -> OLRArtifactStore:
    artifact_root = Path(raw_config.get("artifact_root", "data/strategy/olr")) / "stage1_snapshots" / stage1_config_hash[:16]
    return OLRArtifactStore(artifact_root)


def _research_snapshots(dataset: OLRResearchSweepDataset, mutations: dict[str, Any], stage1_config_hash: str) -> dict[date, Any]:
    key = stable_signature(
        {
            "dataset": dataset.source_fingerprint,
            "stage1_config_hash": stage1_config_hash,
            "research_cache_scope": "compact_daily_research_snapshots",
        }
    )
    cached = _RESEARCH_SNAPSHOT_CACHE.get(key)
    if cached is not None:
        return cached
    snapshots = research_snapshots_for_dataset(dataset, mutations)
    _RESEARCH_SNAPSHOT_CACHE[key] = snapshots
    return snapshots


def _afternoon_contexts(
    dataset: OLRResearchSweepDataset,
    stage1: dict[date, Any],
    stage1_config_hash: str,
) -> dict[date, dict[str, Any]]:
    key = stable_signature(
        {
            "dataset": dataset.source_fingerprint,
            "stage1_config_hash": stage1_config_hash,
            "context_scope": "completed_5m_bars_before_14_30",
        }
    )
    cached = _AFTERNOON_CONTEXT_CACHE.get(key)
    if cached is not None:
        return cached
    contexts = afternoon_contexts_for_snapshots(dataset, stage1)
    _AFTERNOON_CONTEXT_CACHE[key] = contexts
    return contexts


def _eligible_snapshot_dates(trading_dates: Iterable[date]) -> tuple[date, ...]:
    dates = tuple(sorted(trading_dates))
    recovery_sessions = max(1, int(_OFFICIAL_AUCTION_EXIT_RECOVERY_SESSIONS))
    if len(dates) <= recovery_sessions:
        return ()
    return dates[:-recovery_sessions]


def _filtered_training_bars_for_snapshots(
    dataset: OLRResearchSweepDataset,
    snapshots: dict[date, Any],
) -> tuple[tuple[MarketBar, ...], dict[str, Any]]:
    dates = tuple(sorted(dataset.trading_dates))
    selected_pairs = _all_snapshot_candidate_pairs(snapshots)
    needed: set[tuple[date, str]] = set()
    for day, symbol in selected_pairs:
        needed.add((day, symbol))
        for followup_day in _official_followup_session_dates(dates, day):
            needed.add((followup_day, symbol))
    bars: list[MarketBar] = []
    for key in sorted(needed):
        bars.extend(dataset.bars_by_key.get(key, ()))
    ordered = tuple(sorted(bars, key=lambda bar: (bar.timestamp, bar.symbol)))
    bar_scope = {
        "selected_pairs": [(day.isoformat(), symbol) for day, symbol in sorted(selected_pairs)],
        "needed_pairs": [(day.isoformat(), symbol) for day, symbol in sorted(needed)],
        "bar_count": len(ordered),
        "auction_exit_recovery_sessions": _OFFICIAL_AUCTION_EXIT_RECOVERY_SESSIONS,
        "scope": "all_stage2_candidates_trade_date_plus_train_only_auction_exit_recovery",
    }
    return ordered, bar_scope


def _all_snapshot_candidate_pairs(snapshots: dict[date, Any]) -> set[tuple[date, str]]:
    pairs: set[tuple[date, str]] = set()
    for day, snapshot in sorted(snapshots.items()):
        for candidate in tuple(getattr(snapshot, "candidates", ()) or ()):
            if not bool(getattr(candidate, "tradable", True)):
                continue
            symbol = str(getattr(candidate, "symbol", "")).zfill(6)
            if symbol:
                pairs.add((day, symbol))
    return pairs


def _official_followup_session_dates(ordered_dates: Sequence[date], trade_date: date) -> tuple[date, ...]:
    ordered = tuple(sorted(ordered_dates))
    try:
        index = ordered.index(trade_date)
    except ValueError:
        return ()
    stop = min(len(ordered), index + 1 + _OFFICIAL_AUCTION_EXIT_RECOVERY_SESSIONS)
    return ordered[index + 1 : stop]


def _candidate_config_hash(cfg: OLRConfig, _mutations: dict[str, Any]) -> str:
    return final_candidate_config_fingerprint(cfg)


def _stage1_config_hash(cfg: OLRConfig, mutations: dict[str, Any]) -> str:
    return stable_signature(
        {
            "core_version": OLR_CORE_VERSION,
            "sector_daily_version": SECTOR_DAILY_VERSION,
            "stage1_selection_config": _stage1_config_payload(cfg),
            "mutations": {
                key: mutations[key]
                for key in sorted(mutations)
                if any(key == prefix or key.startswith(f"{prefix}.") for prefix in _STAGE1_MUTATION_PREFIXES)
            },
        }
    )


def _stage1_config_payload(cfg: OLRConfig) -> dict[str, Any]:
    return {
        key: value
        for key, value in asdict(cfg).items()
        if not key.startswith("afternoon_")
        and key != "overnight_slot_count"
        and key not in _EXECUTION_ONLY_CONFIG_FIELDS
    }
