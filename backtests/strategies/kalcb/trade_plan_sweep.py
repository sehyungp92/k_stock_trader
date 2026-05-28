from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
import threading
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Iterable

from backtests.auto.shared.cache_keys import fingerprint_paths, stable_signature
from backtests.auto.shared.phase_state import _utc_now_iso
from backtests.analysis.metrics import compute_trade_metrics
from backtests.config import load_yaml_config, normalize_runtime_config
from backtests.engine.replay import run_replay
from backtests.engine.sim_broker import BrokerCosts
from strategy_common.actions import action_to_json_dict
from strategy_common.clock import KST
from strategy_common.events import TradeOutcome as BrokerTradeOutcome
from strategy_common.market import MarketBar
from strategy_kalcb.config import KALCBConfig, KALCB_CORE_VERSION
from strategy_kalcb.models import KALCBDailyCandidate, KALCBDailySnapshot

from .runner import KALCBReplayAdapter, _collapse_exit_legs, _trade_net_r
from .first30_signal_sweep import (
    FIRST30_END,
    First30Context,
    First30Spec,
    KALCBFirst30Dataset,
    Selection,
    _row_date,
    _round_trip_cost_pct,
    prepare_first30_dataset,
)
from .premarket_frontier_sweep import (
    FrontierSpec,
    build_premarket_features,
    name_frontier,
    select_first30_from_frontier,
    select_frontier,
    select_frontier_ranked,
)


TRADE_PLAN_SWEEP_VERSION = "kalcb-fixed-candidate-trade-plan-sweep-v3"
PREPARED_CONTEXT_CACHE_VERSION = "slim-v6-joint-context"
TRADE_PLAN_OBJECTIVE_VERSION = "portfolio-net-v2"
PRIMARY_OBJECTIVE_METRIC = "broker_net_return_pct"
LEGACY_SLOT_METRIC = "slot_cumulative_net_return_pct"
EQUAL_SLOT_METRIC = "equal_slot_net_return_pct"
EQUAL_SLOT_GROSS_METRIC = "equal_slot_gross_return_pct"
PORTFOLIO_EQUIVALENT_METRIC = "portfolio_equivalent_net_return_pct"
EXPOSURE_NORMALIZED_SLOT_METRIC = "exposure_normalized_slot_net_return_pct"
EXPOSURE_NORMALIZED_SLOT_CUMULATIVE_METRIC = "exposure_normalized_slot_cumulative_net_return_pct"
PORTFOLIO_RISK_POLICY = {
    "name": "aggressive_contained_kiaric_comparable_v1",
    "risk_per_trade_pct": 0.0070,
    "risk_per_trade_pct_cap": 0.0070,
    "max_position_notional_pct": 0.45,
    "max_position_notional_pct_cap": 0.45,
    "max_positions_cap": 8,
    "max_per_sector_cap": 8,
    "heat_cap_pct": 0.04,
    "hard_max_drawdown_pct": 0.08,
    "intraday_leverage": 2.0,
    "max_participation_30m": 0.01,
}
DEFAULT_OPTIMIZED_SOURCE = Path(
    "data/backtests/output/kalcb/premarket_frontier_sweeps_full_flow/"
    "kalcb_premarket_frontier_sweep_8ddb4881bd89.json"
)
DEFAULT_OUTPUT_DIR = Path("data/backtests/output/kalcb/trade_plan_sweeps/fixed_optimized_train")
DEFAULT_COMPILED_CACHE_DIR = Path("data/backtests/cache/kalcb/trade_plan_replay")
BASELINE_ENTRY_MODE = "first30_0930"
EOD_EXIT_REASON = "eod_flatten"
EXPECTED_OPTIMIZED_FRONTIER_NAME = (
    "rs_trend_fx30_r53_r20m100to999_r60m100_cl200_adv0_atr999_vol0_"
    "flow5999_flowz999_flowacc999_for5999_inst5999_forz999_instz999_"
    "agree0_div999_secflow999_secpart0_mkt999_flowdata"
)
EXPECTED_OPTIMIZED_FIRST30_NAME = (
    "hybrid_top1_ret0p2_vwap0_gap0p3to10_rv0_cl0_dd20_rng999_"
    "r53_r20m100to999_r60m100_lowPrevm2_flow5999_for5999_inst5999_"
    "flowzm1_agree999_div999_secflow999_mkt999"
)

@dataclass(frozen=True, slots=True)
class FixedCandidateSource:
    source_path: str
    source_file_hash: str
    source_sweep_hash: str
    source_row_name: str
    frontier: FrontierSpec
    first30: First30Spec
    source_section: str = "top_slot_return"
    source_rank: int = 0
    calibration_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EntrySpec:
    name: str
    mode: str
    max_signal_bars: int = 1
    min_bar_ret: float = -9.99
    min_vwap_ret: float = -9.99
    min_breakout_pct: float = 0.0
    max_pullback_from_vwap_pct: float = 0.01
    min_reclaim_ret: float = -9.99
    min_reclaim_closes: int = 1
    min_close_location: float = 0.0
    min_or_position: float = 0.0
    max_avwap_extension_pct: float = 9.99
    after_bar: int = 0
    require_above_prev_close: bool = False
    reclaim_level_source: str = "legacy"


@dataclass(frozen=True, slots=True)
class ExitSpec:
    name: str
    stop_mode: str = "atr"
    stop_atr_mult: float = 0.80
    stop_pct: float = 0.006
    hard_stop_enabled: bool = False
    target_r: float = 0.0
    partial_trigger_r: float = 0.0
    partial_fraction: float = 0.0
    partial_stop_r: float = 0.0
    breakeven_trigger_r: float = 0.0
    breakeven_stop_r: float = 0.0
    trail_start_r: float = 0.0
    trail_gap_r: float = 0.0
    vwap_fail_bars: int = 0
    vwap_fail_pct: float = 0.0
    failed_followthrough_bars: int = 0
    failed_followthrough_mfe_r: float = 0.0
    failed_followthrough_close_r: float = 0.0
    no_mfe_bars: int = 0
    no_mfe_thresh_r: float = 0.0
    max_hold_bars: int = 0


@dataclass(frozen=True, slots=True)
class TradePlanSpec:
    name: str
    entry: EntrySpec
    exit: ExitSpec


@dataclass(frozen=True, slots=True)
class EntrySignal:
    fill_index: int
    signal_index: int
    reason: str


@dataclass(frozen=True, slots=True)
class ExitLeg:
    fraction: float
    price: float
    reason: str
    bar_index: int


@dataclass(frozen=True, slots=True)
class TradeOutcome:
    trade_date: date
    symbol: str
    entry_time: Any
    entry_price: float
    stop_price: float
    risk_per_share: float
    gross_return_pct: float
    net_return_pct: float
    mfe_r: float
    mae_r: float
    mfe_capture: float
    bars_held: int
    exit_reason: str
    ambiguous_bar_count: int
    stopped: bool
    target_hit: bool
    partial_hit: bool
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    entry_type: str = ""
    frontier_role: str = ""
    candidate_rank: int = 0
    frontier_rank: int = 0


@dataclass(frozen=True, slots=True)
class PlanResult:
    spec: TradePlanSpec
    score: float
    rejected: bool
    reject_reason: str
    train_metrics: dict[str, float]
    fold_metrics: tuple[dict[str, Any], ...]
    promotion_pass: bool = False
    replay_digest: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CompiledCoreReplay:
    bars: tuple[MarketBar, ...]
    snapshots: dict[date, KALCBDailySnapshot]
    session_dates: tuple[date, ...]
    selection_counts: dict[date, int]
    initial_equity: float
    source_fingerprint: str
    candidate_artifact_hash: str


@dataclass(frozen=True, slots=True)
class PreparedTradePlanContext:
    candidate_source: FixedCandidateSource
    training_config: dict[str, Any]
    cfg: KALCBConfig
    dataset: KALCBFirst30Dataset
    contexts: dict[date, tuple[First30Context, ...]]
    context_by_key: dict[tuple[date, str], First30Context]
    selections: list[Selection]
    frontier: dict[date, tuple[str, ...]]
    selection_counts: dict[date, int]
    train_dates: tuple[date, ...]
    folds: list[tuple[date, date]]
    compiled_replay: CompiledCoreReplay
    baseline_spec: TradePlanSpec
    baseline_train: dict[str, float]
    baseline_fold_metrics: tuple[dict[str, Any], ...]
    cache_key: str
    cache_metadata: dict[str, Any]


_PROCESS_WORKER_COMPILED_REPLAY: CompiledCoreReplay | None = None
_PROCESS_WORKER_CFG: KALCBConfig | None = None
_PROCESS_WORKER_TRAIN_DATES: tuple[date, ...] | None = None
_PROCESS_WORKER_FOLDS: list[tuple[date, date]] | None = None
_PROCESS_WORKER_SELECTION_COUNTS: dict[date, int] | None = None
_PROCESS_WORKER_BASELINE_TRAIN: dict[str, float] | None = None
_PROCESS_WORKER_BASELINE_FOLDS: tuple[dict[str, Any], ...] | None = None


def _init_process_eval_worker(
    compiled_replay: CompiledCoreReplay,
    cfg: KALCBConfig,
    train_dates: tuple[date, ...],
    folds: list[tuple[date, date]],
    selection_counts: dict[date, int],
    baseline_train: dict[str, float],
    baseline_fold_metrics: tuple[dict[str, Any], ...],
) -> None:
    global _PROCESS_WORKER_COMPILED_REPLAY
    global _PROCESS_WORKER_CFG
    global _PROCESS_WORKER_TRAIN_DATES
    global _PROCESS_WORKER_FOLDS
    global _PROCESS_WORKER_SELECTION_COUNTS
    global _PROCESS_WORKER_BASELINE_TRAIN
    global _PROCESS_WORKER_BASELINE_FOLDS

    _PROCESS_WORKER_COMPILED_REPLAY = compiled_replay
    _PROCESS_WORKER_CFG = cfg
    _PROCESS_WORKER_TRAIN_DATES = train_dates
    _PROCESS_WORKER_FOLDS = folds
    _PROCESS_WORKER_SELECTION_COUNTS = selection_counts
    _PROCESS_WORKER_BASELINE_TRAIN = baseline_train
    _PROCESS_WORKER_BASELINE_FOLDS = baseline_fold_metrics


def _evaluate_spec_from_process_worker(spec: TradePlanSpec) -> PlanResult:
    if (
        _PROCESS_WORKER_COMPILED_REPLAY is None
        or _PROCESS_WORKER_CFG is None
        or _PROCESS_WORKER_TRAIN_DATES is None
        or _PROCESS_WORKER_FOLDS is None
        or _PROCESS_WORKER_SELECTION_COUNTS is None
        or _PROCESS_WORKER_BASELINE_TRAIN is None
        or _PROCESS_WORKER_BASELINE_FOLDS is None
    ):
        raise RuntimeError("process worker was not initialised")
    return _evaluate_spec_from_compiled(
        spec,
        _PROCESS_WORKER_COMPILED_REPLAY,
        _PROCESS_WORKER_CFG,
        _PROCESS_WORKER_TRAIN_DATES,
        _PROCESS_WORKER_FOLDS,
        _PROCESS_WORKER_SELECTION_COUNTS,
        _PROCESS_WORKER_BASELINE_TRAIN,
        _PROCESS_WORKER_BASELINE_FOLDS,
    )


def load_or_build_prepared_context(
    config: dict[str, Any],
    *,
    optimized_source: str | Path = DEFAULT_OPTIMIZED_SOURCE,
    candidate_section: str = "top_slot_return",
    candidate_rank: int = 0,
    strict_candidate_source: bool = True,
    output_dir: str | Path | None = None,
    train_only: bool = True,
    fold_count: int = 2,
    compiled_cache_dir: str | Path | None = None,
    force_rebuild_cache: bool = False,
    refresh_cached_baseline: bool = True,
    status_callback: Callable[..., None] | None = None,
) -> PreparedTradePlanContext:
    """Load or materialize the causal replay bundle used by official sweeps.

    The cached object is only the immutable research/replay input: fixed candidate
    snapshots, completed 5m bars, folds, and baseline metrics. Plan evaluation
    still calls KALCBReplayAdapter -> step_kalcb_core -> SimBroker.
    """

    def status(stage: str, **extra: Any) -> None:
        if status_callback is not None:
            status_callback(stage, **extra)

    training_config = _training_only_config(dict(config), train_only=train_only)
    candidate_source = load_fixed_candidate_source(
        optimized_source,
        section=candidate_section,
        rank=candidate_rank,
        strict_expected=strict_candidate_source,
    )
    cfg = KALCBConfig.from_mapping(training_config, {})
    cache_dir = Path(compiled_cache_dir) if compiled_cache_dir is not None else DEFAULT_COMPILED_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    preflight_fingerprint = _preflight_context_fingerprint(training_config, candidate_source, train_only=train_only, fold_count=fold_count)
    cache_key = stable_signature(
        {
            "namespace": "kalcb_trade_plan_prepared_context",
            "sweep_version": TRADE_PLAN_SWEEP_VERSION,
            "cache_version": PREPARED_CONTEXT_CACHE_VERSION,
            "core_version": KALCB_CORE_VERSION,
            "preflight_source_fingerprint": preflight_fingerprint,
        }
    )
    cache_path = cache_dir / f"prepared_trade_plan_context_{cache_key[:16]}.pkl"
    meta_path = cache_dir / f"prepared_trade_plan_context_{cache_key[:16]}.json"
    status(
        "compiled_replay_cache_check",
        cache_key=cache_key[:16],
        cache_path=str(cache_path),
        force_rebuild=bool(force_rebuild_cache),
    )
    if not force_rebuild_cache:
        cached = _read_prepared_context_cache(
            cache_path,
            meta_path,
            cache_key=cache_key,
            preflight_fingerprint=preflight_fingerprint,
        )
        if cached is not None:
            if refresh_cached_baseline:
                cached = _refresh_prepared_context_runtime_policy(
                    cached,
                    training_config=training_config,
                    cfg=cfg,
                    fold_count=fold_count,
                )
            status(
                "compiled_replay_cache_hit",
                cache_key=cache_key[:16],
                sessions=len(cached.train_dates),
                bars=len(cached.compiled_replay.bars),
                selections=len(cached.selections),
            )
            return cached

    status("compiled_replay_cache_miss", cache_key=cache_key[:16])
    status("preparing_dataset", cache_key=cache_key[:16])
    dataset = prepare_first30_dataset(training_config)
    status("building_first30_contexts", sessions=len(dataset.trading_dates), symbols=len(dataset.symbols))
    contexts = _build_contexts(dataset)
    context_by_key = {(day, ctx.symbol): ctx for day, items in contexts.items() for ctx in items}
    status("selecting_fixed_candidates", sessions=len(dataset.trading_dates), contexts=len(context_by_key))
    selections, frontier = build_fixed_candidate_selections(candidate_source, dataset, contexts)
    frontier_scores = _frontier_scores_by_day(candidate_source, contexts)
    selection_counts = _selection_counts(selections, dataset.trading_dates)
    baseline_spec = baseline_trade_plan_spec()
    train_dates = tuple(dataset.trading_dates)
    folds = _resolve_folds(train_dates, fold_count)
    status("compiling_shared_core_replay", selections=len(selections), folds=len(folds))
    compiled_replay = compile_core_replay(
        selections,
        dataset,
        context_by_key,
        train_dates,
        selection_counts,
        cfg,
        frontier_by_day=frontier,
        frontier_scores_by_day=frontier_scores,
        source_calibration_metadata=candidate_source.calibration_metadata,
    )
    baseline_train = evaluate_plan(
        baseline_spec,
        selections,
        dataset,
        context_by_key,
        cfg,
        train_dates,
        selection_counts,
        compiled_replay=compiled_replay,
    )
    baseline_fold_metrics = _fold_metrics(
        baseline_spec,
        selections,
        dataset,
        context_by_key,
        cfg,
        folds,
        selection_counts,
        compiled_replay=compiled_replay,
    )
    metadata = {
        "cache_key": cache_key,
        "cache_key_short": cache_key[:16],
        "cache_hit": False,
        "created_at": _utc_now_iso(),
        "cache_path": str(cache_path),
        "metadata_path": str(meta_path),
        "output_dir": str(output_dir or ""),
        "sweep_version": TRADE_PLAN_SWEEP_VERSION,
        "cache_version": PREPARED_CONTEXT_CACHE_VERSION,
        "strategy_core_version": KALCB_CORE_VERSION,
        "shared_decision_core": "live_shared_core",
        "research_only": False,
        "preflight_source_fingerprint": preflight_fingerprint,
        "candidate_source_file_hash": candidate_source.source_file_hash,
        "candidate_source_sweep_hash": candidate_source.source_sweep_hash,
        "candidate_source_calibration_metadata": dict(candidate_source.calibration_metadata),
        "intraday_source_fingerprint": dataset.source_fingerprint,
        "daily_source_fingerprint": dataset.daily_source_fingerprint,
        "compiled_replay_fingerprint": compiled_replay.source_fingerprint,
        "candidate_artifact_hash": compiled_replay.candidate_artifact_hash,
        "training_window": {
            "start": train_dates[0].isoformat() if train_dates else "",
            "end": train_dates[-1].isoformat() if train_dates else "",
            "sessions": len(train_dates),
        },
        "counts": {
            "symbols": len(dataset.symbols),
            "contexts": len(context_by_key),
            "selections": len(selections),
            "compiled_bars": len(compiled_replay.bars),
            "snapshots": len(compiled_replay.snapshots),
        },
        "causality_policy": {
            "premarket_frontier": "daily/index/flow rows strictly date < trade_date",
            "first30_gate": "completed 09:00-09:25 KST bars only",
            "trading_core": "KALCBReplayAdapter -> step_kalcb_core -> SimBroker",
        },
    }
    context = PreparedTradePlanContext(
        candidate_source=candidate_source,
        training_config=training_config,
        cfg=cfg,
        dataset=dataset,
        contexts=contexts,
        context_by_key=context_by_key,
        selections=selections,
        frontier=frontier,
        selection_counts=selection_counts,
        train_dates=train_dates,
        folds=folds,
        compiled_replay=compiled_replay,
        baseline_spec=baseline_spec,
        baseline_train=baseline_train,
        baseline_fold_metrics=baseline_fold_metrics,
        cache_key=cache_key,
        cache_metadata=metadata,
    )
    _write_prepared_context_cache(cache_path, meta_path, context, metadata)
    status(
        "compiled_replay_cache_written",
        cache_key=cache_key[:16],
        sessions=len(train_dates),
        bars=len(compiled_replay.bars),
        selections=len(selections),
    )
    return context


def _refresh_prepared_context_runtime_policy(
    context: PreparedTradePlanContext,
    *,
    training_config: dict[str, Any],
    cfg: KALCBConfig,
    fold_count: int,
) -> PreparedTradePlanContext:
    train_dates = tuple(context.train_dates)
    folds = _resolve_folds(train_dates, fold_count)
    baseline_spec = baseline_trade_plan_spec()
    baseline_train = evaluate_plan(
        baseline_spec,
        context.selections,
        context.dataset,
        context.context_by_key,
        cfg,
        train_dates,
        context.selection_counts,
        compiled_replay=context.compiled_replay,
    )
    baseline_fold_metrics = _fold_metrics(
        baseline_spec,
        context.selections,
        context.dataset,
        context.context_by_key,
        cfg,
        folds,
        context.selection_counts,
        compiled_replay=context.compiled_replay,
    )
    metadata = {
        **dict(context.cache_metadata),
        "baseline_recomputed_at": _utc_now_iso(),
        "portfolio_risk_policy": dict(PORTFOLIO_RISK_POLICY),
    }
    return replace(
        context,
        training_config=training_config,
        cfg=cfg,
        folds=folds,
        baseline_spec=baseline_spec,
        baseline_train=baseline_train,
        baseline_fold_metrics=baseline_fold_metrics,
        cache_metadata=metadata,
    )


def _preflight_context_fingerprint(
    training_config: dict[str, Any],
    candidate_source: FixedCandidateSource,
    *,
    train_only: bool,
    fold_count: int,
) -> str:
    data_root = Path(training_config.get("data_root", "data/kis_intraday_parquet"))
    daily_root = Path(training_config.get("daily_data_root", "data/krx_daily_parquet"))
    universe_path = Path(str(training_config.get("universe") or ""))
    intraday_paths = [
        data_root,
        data_root / "conversion_manifest.json",
        data_root / "manifest.json",
    ]
    daily_paths = [
        daily_root,
        daily_root / "manifest.json",
        daily_root / "tables" / "sector_map.parquet",
    ]
    config_paths = [universe_path] if str(universe_path) not in {"", "."} else []
    source_payload = {
        "path": candidate_source.source_path,
        "file_hash": candidate_source.source_file_hash,
        "sweep_hash": candidate_source.source_sweep_hash,
        "section": candidate_source.source_section,
        "rank": candidate_source.source_rank,
        "frontier": asdict(candidate_source.frontier),
        "first30": asdict(candidate_source.first30),
    }
    if candidate_source.calibration_metadata:
        source_payload["calibration_metadata"] = candidate_source.calibration_metadata
    return stable_signature(
        {
            "training_config": training_config,
            "train_only": bool(train_only),
            "fold_count": int(fold_count),
            "candidate_source": source_payload,
            "roots": {
                "intraday": {
                    "path": str(data_root),
                    "fingerprint": fingerprint_paths(intraday_paths, root=data_root.parent if data_root.parent.exists() else None),
                },
                "daily": {
                    "path": str(daily_root),
                    "fingerprint": fingerprint_paths(daily_paths, root=daily_root.parent if daily_root.parent.exists() else None),
                },
                "config": fingerprint_paths(config_paths, root=Path.cwd()) if config_paths else "",
            },
        }
    )


def _read_prepared_context_cache(
    cache_path: Path,
    meta_path: Path,
    *,
    cache_key: str,
    preflight_fingerprint: str,
) -> PreparedTradePlanContext | None:
    if not cache_path.exists() or not meta_path.exists():
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("cache_key") != cache_key:
            return None
        if metadata.get("preflight_source_fingerprint") != preflight_fingerprint:
            return None
        if metadata.get("sweep_version") != TRADE_PLAN_SWEEP_VERSION:
            return None
        if metadata.get("cache_version") != PREPARED_CONTEXT_CACHE_VERSION:
            return None
        if metadata.get("strategy_core_version") != KALCB_CORE_VERSION:
            return None
        with cache_path.open("rb") as handle:
            context = pickle.load(handle)
        if not isinstance(context, PreparedTradePlanContext):
            return None
        if context.cache_key != cache_key:
            return None
        baseline_train = dict(context.baseline_train)
        if PRIMARY_OBJECTIVE_METRIC in baseline_train and "primary_objective_net_return_pct" not in baseline_train:
            _add_return_divergence_metrics(baseline_train)
        loaded_metadata = dict(context.cache_metadata)
        loaded_metadata.update(
            {
                **metadata,
                "cache_hit": True,
                "loaded_at": _utc_now_iso(),
                "cache_path": str(cache_path),
                "metadata_path": str(meta_path),
            }
        )
        return replace(context, baseline_train=baseline_train, cache_metadata=loaded_metadata)
    except Exception:
        return None


def _write_prepared_context_cache(
    cache_path: Path,
    meta_path: Path,
    context: PreparedTradePlanContext,
    metadata: dict[str, Any],
) -> None:
    cache_context = _slim_prepared_context_for_cache(context)
    cache_tmp = cache_path.with_name(cache_path.name + ".tmp")
    meta_tmp = meta_path.with_name(meta_path.name + ".tmp")
    with cache_tmp.open("wb") as handle:
        pickle.dump(cache_context, handle, protocol=pickle.HIGHEST_PROTOCOL)
    meta_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8")
    cache_tmp.replace(cache_path)
    meta_tmp.replace(meta_path)


def _slim_prepared_context_for_cache(context: PreparedTradePlanContext) -> PreparedTradePlanContext:
    return replace(
        context,
        dataset=_slim_dataset_for_cache(context.dataset),
        contexts={},
        context_by_key={},
    )


def _slim_dataset_for_cache(dataset: KALCBFirst30Dataset) -> KALCBFirst30Dataset:
    return replace(
        dataset,
        daily_by_symbol={},
        flow_by_symbol={},
        index_by_code={},
        foreign_flow_by_symbol={},
        institutional_flow_by_symbol={},
        bars_by_key={},
    )


def run_trade_plan_sweep(
    config: dict[str, Any],
    *,
    optimized_source: str | Path = DEFAULT_OPTIMIZED_SOURCE,
    candidate_section: str = "top_slot_return",
    candidate_rank: int = 0,
    strict_candidate_source: bool = True,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    train_only: bool = True,
    max_workers: int = 4,
    fold_count: int = 2,
    coarse_entry_limit: int = 720,
    coarse_exit_limit: int = 240,
    entry_seed_top_n: int = 12,
    deep_refine_top_n: int = 16,
    deep_refine_max_specs: int = 8_000,
    finalist_count: int = 50,
    audit_max_workers: int | None = None,
    worker_backend: str = "thread",
    compiled_cache_dir: str | Path | None = None,
    force_rebuild_cache: bool = False,
    focused_seed_rows: str | Path | None = None,
    focused_seed_limit: int = 0,
    focused_seed_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _clear_progress_files(out)
    _write_run_status(out, "loading_or_building_compiled_context")
    stop_heartbeat = _start_status_heartbeat(out, "preparing_compiled_replay_context")
    try:
        prepared = load_or_build_prepared_context(
            config,
            optimized_source=optimized_source,
            candidate_section=candidate_section,
            candidate_rank=candidate_rank,
            strict_candidate_source=strict_candidate_source,
            output_dir=out,
            train_only=train_only,
            fold_count=fold_count,
            compiled_cache_dir=compiled_cache_dir,
            force_rebuild_cache=force_rebuild_cache,
            status_callback=lambda stage, **extra: _write_run_status(out, stage, **extra),
        )
    finally:
        stop_heartbeat()
    candidate_source = prepared.candidate_source
    training_config = prepared.training_config
    cfg = prepared.cfg
    dataset = prepared.dataset
    contexts = prepared.contexts
    context_by_key = prepared.context_by_key
    selections = prepared.selections
    frontier = prepared.frontier
    selection_counts = prepared.selection_counts
    baseline_spec = prepared.baseline_spec
    train_dates = prepared.train_dates
    folds = prepared.folds
    compiled_replay = prepared.compiled_replay
    baseline_train = prepared.baseline_train
    baseline_fold_metrics = prepared.baseline_fold_metrics
    _write_run_status(
        out,
        "baseline_complete",
        baseline_broker_net=round(100.0 * baseline_train.get(PRIMARY_OBJECTIVE_METRIC, 0.0), 3),
        baseline_exposure_normalized_slot_net=round(100.0 * baseline_train.get(EXPOSURE_NORMALIZED_SLOT_METRIC, baseline_train.get(PORTFOLIO_EQUIVALENT_METRIC, 0.0)), 3),
        baseline_equal_slot_net=round(100.0 * baseline_train.get(EQUAL_SLOT_METRIC, baseline_train.get(LEGACY_SLOT_METRIC, 0.0)), 3),
        baseline_trades=int(baseline_train.get("trade_count", 0.0)),
        compiled_cache_hit=bool(prepared.cache_metadata.get("cache_hit")),
        compiled_cache_key=str(prepared.cache_metadata.get("cache_key_short") or prepared.cache_key[:16]),
    )

    focused_seed_source = Path(focused_seed_rows) if focused_seed_rows else None
    focused_seed_eval_rows: list[PlanResult] = []
    entry_rows: list[PlanResult] = []
    exit_rows: list[PlanResult] = []
    entry_seed_rows: list[PlanResult] = []
    entry_seeds: list[EntrySpec] = []
    if focused_seed_source is not None:
        seed_rows = _load_focused_seed_rows(
            focused_seed_source,
            limit=focused_seed_limit,
            names=tuple(str(name) for name in (focused_seed_names or ()) if str(name)),
        )
        if not seed_rows:
            raise ValueError(f"No focused seed rows loaded from {focused_seed_source}")
        seed_specs = [row.spec for row in seed_rows]
        _write_run_status(
            out,
            "focused_seed",
            total=len(seed_specs),
            source=str(focused_seed_source),
            seed_limit=int(focused_seed_limit or 0),
        )
        focused_seed_eval_rows = _evaluate_specs(
            seed_specs,
            selections,
            dataset,
            context_by_key,
            cfg,
            train_dates,
            folds,
            selection_counts,
            baseline_train,
            baseline_fold_metrics,
            out,
            "focused_seed",
            max_workers=max_workers,
            compiled_replay=compiled_replay,
            worker_backend=worker_backend,
        )
        focused_seed_eval_rows.sort(key=_plan_sort_key)
        broad_rows = list(focused_seed_eval_rows)
    else:
        entry_specs = _limit_specs(build_entry_specs(), coarse_entry_limit)
        entry_exit = eod_flatten_exit_spec()
        entry_plans = [name_plan(TradePlanSpec("", entry, entry_exit)) for entry in entry_specs]
        _write_run_status(out, "entry_coarse", total=len(entry_plans))
        entry_rows = _evaluate_specs(
            entry_plans,
            selections,
            dataset,
            context_by_key,
            cfg,
            train_dates,
            folds,
            selection_counts,
            baseline_train,
            baseline_fold_metrics,
            out,
            "entry_coarse",
            max_workers=max_workers,
            compiled_replay=compiled_replay,
            worker_backend=worker_backend,
        )
        entry_rows.sort(key=_plan_sort_key)
        entry_seed_rows = _select_exit_entry_seed_rows(entry_rows, entry_seed_top_n)
        entry_seeds = [row.spec.entry for row in entry_seed_rows]

        exit_specs = _limit_specs(build_exit_specs(), coarse_exit_limit)
        exit_plans = [
            name_plan(TradePlanSpec("", entry, exit_spec))
            for entry in entry_seeds
            for exit_spec in exit_specs
        ]
        _write_run_status(out, "exit_coarse", total=len(exit_plans), entry_seeds=len(entry_seeds), exit_specs=len(exit_specs))
        exit_rows = _evaluate_specs(
            exit_plans,
            selections,
            dataset,
            context_by_key,
            cfg,
            train_dates,
            folds,
            selection_counts,
            baseline_train,
            baseline_fold_metrics,
            out,
            "exit_coarse",
            max_workers=max_workers,
            compiled_replay=compiled_replay,
            worker_backend=worker_backend,
        )

        broad_rows = _merge_results(entry_rows, exit_rows)
    broad_rows.sort(key=_plan_sort_key)
    refine_specs = build_refined_trade_plan_specs(
        broad_rows if focused_seed_source is not None else broad_rows[: max(0, int(deep_refine_top_n))],
        exclude_names={row.spec.name for row in broad_rows},
        max_specs=deep_refine_max_specs,
    )
    refine_rows: list[PlanResult] = []
    if refine_specs:
        _write_run_status(out, "deep_refine", total=len(refine_specs))
        refine_rows = _evaluate_specs(
            refine_specs,
            selections,
            dataset,
            context_by_key,
            cfg,
            train_dates,
            folds,
            selection_counts,
            baseline_train,
            baseline_fold_metrics,
            out,
            "deep_refine",
            max_workers=max_workers,
            compiled_replay=compiled_replay,
            worker_backend=worker_backend,
        )
    rows = _merge_results(broad_rows, refine_rows)
    rows.sort(key=_plan_sort_key)
    finalists = rows[: max(1, int(finalist_count))]
    promotion_rows = [row for row in finalists if row.promotion_pass]
    audit_targets = promotion_rows[:25] if promotion_rows else finalists[:10]
    audit_workers = max(1, int(audit_max_workers if audit_max_workers is not None else max_workers))
    _write_run_status(out, "full_audit_finalists", total=len(audit_targets), max_workers=audit_workers)
    audit_rows = _audit_replay_rows(
        audit_targets,
        compiled_replay,
        cfg,
        train_dates,
        selection_counts,
        max_workers=audit_workers,
    )

    payload = {
        "strategy": "kalcb",
        "sweep_version": TRADE_PLAN_SWEEP_VERSION,
        "objective_version": TRADE_PLAN_OBJECTIVE_VERSION,
        "research_only": False,
        "shared_decision_core": "live_shared_core",
        "strategy_core_version": KALCB_CORE_VERSION,
        "created_at": _utc_now_iso(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "training_window": {
            "start": train_dates[0].isoformat() if train_dates else "",
            "end": train_dates[-1].isoformat() if train_dates else "",
            "sessions": len(train_dates),
        },
        "holdout_policy": "No holdout dates are evaluated or used. This artifact is train-only.",
        "causality_policy": {
            "premarket_frontier": "daily/index/flow rows strictly date < trade_date",
            "first30_gate": "completed 09:00-09:25 KST bars only",
            "entry": "signals from completed bar t fill no earlier than bar t+1 open; first30_0930 fills after 09:25 completion",
            "same_bar_stop_target": "stop-first conservative ordering",
            "trailing_updates": "completed-bar trail/breakeven updates become active on the next bar",
            "strategy_core": "official sweep replays through KALCBReplayAdapter -> step_kalcb_core -> SimBroker",
            "research_boundary": "research snapshots can be built by live generator or replay cache, but final trading candidates must be KALCBDailySnapshot artifacts from daily_selection_from_snapshot/run_daily_selection contract",
        },
        "objective": "training-only shared-core optimization over fixed KALCB Optimized final first30 picks; broker portfolio net return is the primary objective, Slot Return is diagnostic",
        "execution_policy": {
            "worker_backend": "thread_shared_compiled_replay",
            "requested_worker_backend": str(worker_backend or "thread"),
            "max_workers": max(1, min(int(max_workers), 4)),
            "reason": "avoid Windows process-pool duplication of the compiled replay while preserving shared-core replay decisions",
            "compiled_replay_cache": "source-fingerprinted causal replay input bundle; official decisions still run through the shared live core",
            "compiled_replay_cache_hit": bool(prepared.cache_metadata.get("cache_hit")),
        },
        "focused_seed": {
            "enabled": focused_seed_source is not None,
            "source": str(focused_seed_source or ""),
            "requested_limit": int(focused_seed_limit or 0),
            "loaded_count": len(focused_seed_eval_rows),
            "seed_names": [row.spec.name for row in focused_seed_eval_rows],
        },
        "portfolio_risk_policy": {
            **PORTFOLIO_RISK_POLICY,
            "effective_core_mutations": _portfolio_risk_mutations(cfg),
            "rationale": "aggressive but controlled KRX intraday research stance: roughly 0.30% risk per full trade, 1x capital, capped concentration, and capped open stop-risk heat",
        },
        "fixed_candidate_source": {
            "source_path": candidate_source.source_path,
            "source_file_hash": candidate_source.source_file_hash,
            "source_sweep_hash": candidate_source.source_sweep_hash,
            "source_row_name": candidate_source.source_row_name,
            "source_section": candidate_source.source_section,
            "source_rank": candidate_source.source_rank,
            "frontier": asdict(candidate_source.frontier),
            "first30": asdict(candidate_source.first30),
        },
        "source_fingerprints": {
            "intraday": dataset.source_fingerprint,
            "daily_lrs": dataset.daily_source_fingerprint,
            "combined": stable_signature([dataset.source_fingerprint, dataset.daily_source_fingerprint]),
            "optimized_source": candidate_source.source_sweep_hash,
            "optimized_source_file": candidate_source.source_file_hash,
            "candidate_artifact_hash": compiled_replay.candidate_artifact_hash,
            "compiled_replay": compiled_replay.source_fingerprint,
            "compiled_replay_cache_key": prepared.cache_key,
            "compiled_replay_preflight": str(prepared.cache_metadata.get("preflight_source_fingerprint") or ""),
        },
        "compiled_replay_cache": prepared.cache_metadata,
        "trade_plan_policy_hash": stable_signature([TRADE_PLAN_SWEEP_VERSION, KALCB_CORE_VERSION, [_spec_payload(row.spec) for row in rows[:25]]]) if "rows" in locals() else "",
        "selection_stats": _selection_stats(selections, frontier, selection_counts, train_dates),
        "sweep_counts": {
            "focused_seed": len(focused_seed_eval_rows),
            "entry_coarse": len(entry_rows),
            "exit_coarse": len(exit_rows),
            "deep_refine": len(refine_rows),
            "candidate_count": len(rows),
            "promoted_count": len(promotion_rows),
            "exit_entry_seed_count": len(entry_seeds),
        },
        "exit_entry_seeds": [_entry_seed_payload(row) for row in entry_seed_rows],
        "cost_policy": {
            "round_trip_cost_pct": _round_trip_cost_pct(cfg),
            "slippage_bps_each_side": cfg.slippage_bps,
            "commission_bps_each_side": cfg.commission_bps,
            "tax_bps_on_sell": cfg.tax_bps_on_sell,
        },
        "baseline": {
            "spec": _spec_payload(baseline_spec),
            "train_metrics": baseline_train,
            "fold_metrics": baseline_fold_metrics,
        },
        "promotion_policy": {
            "primary_metric": PRIMARY_OBJECTIVE_METRIC,
            "objective_version": TRADE_PLAN_OBJECTIVE_VERSION,
            "portfolio_net_relative_improvement_min": 0.10,
            "metric_contract": {
                "broker_net_return_pct": "official executable SimBroker closed-trade net PnL over initial equity; primary optimization and promotion metric",
                "portfolio_equivalent_net_return_pct": "exposure-normalized slot proxy from broker net PnL over initial equity; should match broker net except for accounting basis changes",
                "exposure_normalized_slot_net_return_pct": "alias of portfolio_equivalent_net_return_pct for human-readable reporting",
                "slot_cumulative_net_return_pct": "legacy costed equal-slot/opportunity metric; not exposure-normalized and not comparable to broker net",
                "equal_slot_net_return_pct": "alias of slot_cumulative_net_return_pct",
                "slot_to_broker_net_return_ratio": "legacy diagnostic comparing equal-slot to broker net; can be far from 1.0 without indicating replay divergence",
                "exposure_normalized_slot_to_broker_ratio": "broker-comparable parity diagnostic; should be near 1.0",
            },
            "active_day_net_must_not_worsen": True,
            "broker_max_drawdown_absolute_worsening_limit": 0.05,
            "hard_max_drawdown_ceiling": PORTFOLIO_RISK_POLICY["hard_max_drawdown_pct"],
            "fold_stability": "portfolio-equivalent/broker net fold returns must be stable versus baseline",
            "finalist_audit_policy": "promoted rows are rerun in full audit mode; if no row promotes, top finalists are audited diagnostically",
            "fast_audit_metric_tolerance": 1e-10,
            "fast_suppression_scope": "fast mode suppresses entry_rejected diagnostic events only; full audit must match fills, trades, trading decisions, and metrics",
            "result": "promote" if promotion_rows else "no_promotion",
        },
        "audit_replays": audit_rows,
        "fast_suppression_audit": _audit_summary(audit_rows),
        "audit_pass": all(bool(row.get("audit_pass")) for row in audit_rows) if audit_rows else True,
        "top_train": [_row_payload(row) for row in rows[:100]],
        "top_promoted": [_row_payload(row) for row in promotion_rows[:25]],
        "rows": [_row_payload(row) for row in rows],
    }
    payload["sweep_hash"] = stable_signature(
        {
            "version": TRADE_PLAN_SWEEP_VERSION,
            "objective_version": TRADE_PLAN_OBJECTIVE_VERSION,
            "fixed_candidate_source": payload["fixed_candidate_source"],
            "source_fingerprints": payload["source_fingerprints"],
            "training_window": payload["training_window"],
            "baseline": payload["baseline"],
            "top_train": payload["top_train"][:10],
        }
    )
    json_path = out / f"kalcb_trade_plan_sweep_{payload['sweep_hash'][:12]}.json"
    md_path = out / f"kalcb_trade_plan_sweep_{payload['sweep_hash'][:12]}.md"
    payload["artifact_paths"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    _write_progress(out, len(rows), len(rows), rows, "completed")
    _write_run_status(out, "completed", sweep_hash=payload["sweep_hash"], json=str(json_path), markdown=str(md_path))
    return payload


def load_fixed_candidate_source(
    path: str | Path = DEFAULT_OPTIMIZED_SOURCE,
    *,
    section: str = "top_slot_return",
    rank: int = 0,
    strict_expected: bool = True,
) -> FixedCandidateSource:
    source_path = _resolve_optimized_source_path(Path(path))
    source_file_hash, source_sweep_hash, row = _read_optimized_source_row(source_path, section=section, rank=rank)
    if not row:
        raise ValueError(f"No {section}[{rank}] row found in optimized source: {source_path}")
    frontier = _frontier_from_payload(row.get("frontier") or {})
    first30 = _first30_from_payload(row.get("first30") or {})
    if strict_expected:
        if frontier.name != EXPECTED_OPTIMIZED_FRONTIER_NAME:
            raise ValueError(f"Optimized source top row does not match expected frontier name: {frontier.name}")
        if first30.name != EXPECTED_OPTIMIZED_FIRST30_NAME:
            raise ValueError(f"Optimized source top row does not match expected first30 selector name: {first30.name}")
        if frontier.mode != "rs_trend" or frontier.frontier_size != 30 or not frontier.require_flow_available:
            raise ValueError(f"Optimized source top row does not match expected KALCB Optimized frontier: {frontier.name}")
        if first30.score_mode != "hybrid" or first30.top_n != 1:
            raise ValueError(f"Optimized source top row does not match expected KALCB Optimized first30 selector: {first30.name}")
    return FixedCandidateSource(
        source_path=str(source_path),
        source_file_hash=source_file_hash,
        source_sweep_hash=source_sweep_hash,
        source_row_name=str(row.get("name") or f"{frontier.name}__{first30.name}"),
        frontier=frontier,
        first30=first30,
        source_section=str(section),
        source_rank=int(rank),
        calibration_metadata=_calibration_metadata_from_row(row),
    )


def _calibration_metadata_from_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "calibration_version",
        "path_calibration_score_version",
        "path_calibrated_score",
        "path_score_components",
        "path_risk_metrics",
        "path_quality_model",
        "interaction_regime_model_version",
        "interaction_regime_model",
        "fold_path_risk_metrics",
        "original_source_section",
        "original_source_rank",
        "original_source_name",
    )
    metadata = {key: row.get(key) for key in keys if key in row}
    return metadata if any(value not in (None, {}, [], "") for value in metadata.values()) else {}


def _resolve_optimized_source_path(path: Path) -> Path:
    if path.exists():
        return path
    archive_root = Path("data/backtests/output/kalcb/_archive")
    if not archive_root.exists():
        return path
    matches = sorted(candidate for candidate in archive_root.rglob(path.name) if candidate.is_file())
    if not matches:
        return path
    suffix = Path(*path.parts[-4:]) if len(path.parts) >= 4 else path
    for candidate in matches:
        if str(candidate).endswith(str(suffix)):
            return candidate
    return matches[0]


def _read_optimized_source_row(path: Path, *, section: str, rank: int) -> tuple[str, str, dict[str, Any]]:
    if path.stat().st_size <= 64_000_000:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        rows = payload.get(section) or []
        row = rows[max(0, int(rank))] if isinstance(rows, list) and len(rows) > max(0, int(rank)) else {}
        return hashlib.sha256(raw).hexdigest(), str(payload.get("sweep_hash") or payload.get("calibration_hash") or ""), row if isinstance(row, dict) else {}
    hasher = hashlib.sha256()
    marker = f'"{section}"'
    sweep_hash = ""
    row: dict[str, Any] | None = None
    buffer = ""
    marker_seen = False
    array_seen = False
    row_index = 0
    decoder = json.JSONDecoder()
    with path.open("rb") as handle:
        for raw_chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(raw_chunk)
            if row is not None and sweep_hash:
                continue
            chunk = raw_chunk.decode("utf-8", errors="ignore")
            buffer += chunk
            if not sweep_hash:
                match = re.search(r'"(?:sweep_hash|calibration_hash)"\s*:\s*"([^"]+)"', buffer)
                if match:
                    sweep_hash = match.group(1)
            if row is None:
                if not marker_seen:
                    marker_index = buffer.find(marker)
                    if marker_index >= 0:
                        marker_seen = True
                        buffer = buffer[marker_index:]
                    elif len(buffer) > 2_000_000:
                        buffer = buffer[-2_000_000:]
                        continue
                    else:
                        continue
                if not array_seen:
                    array_start = buffer.find("[")
                    if array_start < 0:
                        continue
                    array_seen = True
                    buffer = buffer[array_start + 1 :]
                while row is None:
                    buffer = buffer.lstrip()
                    while buffer.startswith(","):
                        buffer = buffer[1:].lstrip()
                    if buffer.startswith("]"):
                        break
                    try:
                        obj, end = decoder.raw_decode(buffer)
                    except json.JSONDecodeError:
                        if len(buffer) > 8_000_000:
                            buffer = buffer[-8_000_000:]
                        break
                    if row_index == max(0, int(rank)):
                        row = obj if isinstance(obj, dict) else {}
                        buffer = ""
                        break
                    row_index += 1
                    buffer = buffer[end:]
    if row is None:
        row = {}
    return hasher.hexdigest(), sweep_hash, row


def build_fixed_candidate_selections(
    candidate_source: FixedCandidateSource,
    dataset: KALCBFirst30Dataset,
    contexts: dict[date, tuple[First30Context, ...]],
) -> tuple[list[Selection], dict[date, tuple[str, ...]]]:
    features = build_premarket_features(contexts)
    frontier = select_frontier(candidate_source.frontier, features)
    selections = select_first30_from_frontier(frontier, candidate_source.first30, contexts)
    selections.sort(key=lambda item: (item.trade_date, item.symbol))
    return selections, frontier


def _frontier_scores_by_day(
    candidate_source: FixedCandidateSource,
    contexts: dict[date, tuple[First30Context, ...]],
) -> dict[date, dict[str, float]]:
    features = build_premarket_features(contexts)
    ranked = select_frontier_ranked(candidate_source.frontier, features)
    return {
        day: {symbol: float(score) for symbol, score in rows}
        for day, rows in ranked.items()
    }


def baseline_trade_plan_spec() -> TradePlanSpec:
    return name_plan(TradePlanSpec("", _name_entry(EntrySpec("", BASELINE_ENTRY_MODE)), eod_flatten_exit_spec()))


def eod_flatten_exit_spec() -> ExitSpec:
    return _name_exit(ExitSpec(name="", stop_mode="atr", hard_stop_enabled=False))


def build_entry_specs() -> list[EntrySpec]:
    specs: list[EntrySpec] = [EntrySpec("", BASELINE_ENTRY_MODE)]
    for min_ret in (-0.002, 0.0, 0.002, 0.004):
        for min_vwap in (-0.002, 0.0, 0.001):
            for close_loc in (0.0, 0.50, 0.60, 0.70):
                specs.append(EntrySpec("", "opening_drive", min_bar_ret=min_ret, min_vwap_ret=min_vwap, min_close_location=close_loc, min_or_position=0.35))
                specs.append(EntrySpec("", "opening_drive", min_bar_ret=min_ret, min_vwap_ret=min_vwap, min_close_location=close_loc, min_or_position=0.50, require_above_prev_close=True))
    for bars in (1, 2, 3, 6, 10):
        for min_ret in (-0.001, 0.0, 0.001, 0.002, 0.004):
            for min_vwap in (-0.003, -0.001, 0.0, 0.001, 0.002):
                for close_loc in (0.0, 0.50, 0.60):
                    specs.append(EntrySpec("", "confirm_next_bar", max_signal_bars=bars, min_bar_ret=min_ret, min_vwap_ret=min_vwap, min_close_location=close_loc))
                    specs.append(EntrySpec("", "post_or_momentum", max_signal_bars=bars, min_bar_ret=min_ret, min_vwap_ret=min_vwap, min_close_location=close_loc, min_or_position=0.45))
                    specs.append(EntrySpec("", "post_or_momentum", max_signal_bars=bars, min_bar_ret=min_ret, min_vwap_ret=min_vwap, min_close_location=close_loc, min_or_position=0.60))
    for mode in ("breakout", "or_breakout", "pdh_breakout", "combined_breakout"):
        for bars in (2, 4, 8, 12, 18):
            for breakout in (0.0, 0.0005, 0.001, 0.002, 0.004):
                for close_loc in (0.0, 0.50, 0.60, 0.70):
                    specs.append(EntrySpec("", mode, max_signal_bars=bars, min_breakout_pct=breakout, min_close_location=close_loc))
    for mode in ("avwap_reclaim", "pullback_acceptance"):
        for bars in (3, 6, 10, 16):
            for pullback in (0.002, 0.004, 0.008, 0.012, 0.016):
                for reclaim in (-0.001, 0.0, 0.001, 0.002):
                    specs.append(EntrySpec("", mode, max_signal_bars=bars, max_pullback_from_vwap_pct=pullback, min_reclaim_ret=reclaim, min_vwap_ret=reclaim))
    for mode in ("or_mid_reclaim", "or_high_reclaim", "pdh_reclaim"):
        for bars in (4, 8, 12, 18):
            for reclaim in (0.0, 0.0005, 0.001, 0.002):
                for close_loc in (0.0, 0.50, 0.60):
                    specs.append(EntrySpec("", mode, max_signal_bars=bars, min_reclaim_ret=reclaim, min_close_location=close_loc))
    for after in (1, 2, 4, 6):
        for bars in (6, 10, 16, 24):
            for breakout in (0.0, 0.001, 0.002):
                for min_vwap in (-0.001, 0.0, 0.001):
                    specs.append(EntrySpec("", "deferred_continuation", max_signal_bars=bars, after_bar=after, min_breakout_pct=breakout, min_vwap_ret=min_vwap))
    return _dedupe_entries(_name_entry(spec) for spec in specs)


def build_exit_specs() -> list[ExitSpec]:
    specs: list[ExitSpec] = [ExitSpec("", stop_mode="atr", hard_stop_enabled=False)]
    stop_modes = ("first30_low", "signal_low", "entry_low", "atr", "fixed_pct", "vwap")
    for stop_mode in stop_modes:
        specs.append(ExitSpec("", stop_mode=stop_mode, hard_stop_enabled=True))
        for target in (0.35, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00):
            specs.append(ExitSpec("", stop_mode=stop_mode, hard_stop_enabled=True, target_r=target))
        for trigger, fraction, stop_r in (
            (0.25, 0.50, 0.0),
            (0.35, 0.50, 0.0),
            (0.50, 0.50, 0.10),
            (0.75, 0.33, 0.20),
            (1.00, 0.33, 0.35),
        ):
            specs.append(ExitSpec("", stop_mode=stop_mode, hard_stop_enabled=True, partial_trigger_r=trigger, partial_fraction=fraction, partial_stop_r=stop_r))
            specs.append(ExitSpec("", stop_mode=stop_mode, hard_stop_enabled=True, partial_trigger_r=trigger, partial_fraction=fraction, partial_stop_r=stop_r, target_r=1.25))
        for trigger, stop_r in ((0.25, 0.0), (0.35, 0.0), (0.50, 0.10), (0.75, 0.20), (1.00, 0.35)):
            specs.append(ExitSpec("", stop_mode=stop_mode, hard_stop_enabled=True, breakeven_trigger_r=trigger, breakeven_stop_r=stop_r, target_r=1.25))
        for start, gap in ((0.35, 0.30), (0.50, 0.35), (0.75, 0.45), (1.00, 0.60), (1.25, 0.75)):
            specs.append(ExitSpec("", stop_mode=stop_mode, hard_stop_enabled=True, trail_start_r=start, trail_gap_r=gap))
            specs.append(ExitSpec("", stop_mode=stop_mode, hard_stop_enabled=True, partial_trigger_r=0.50, partial_fraction=0.50, partial_stop_r=0.0, trail_start_r=start, trail_gap_r=gap))
    for stop_mode in ("atr", "first30_low", "fixed_pct", "vwap"):
        for bars in (3, 4, 6, 8):
            for thresh in (0.10, 0.15, 0.25):
                specs.append(ExitSpec("", stop_mode=stop_mode, hard_stop_enabled=True, no_mfe_bars=bars, no_mfe_thresh_r=thresh, target_r=1.0))
        for fail_bars in (2, 3, 4, 6):
            for mfe in (0.10, 0.20, 0.35):
                specs.append(ExitSpec("", stop_mode=stop_mode, hard_stop_enabled=True, failed_followthrough_bars=fail_bars, failed_followthrough_mfe_r=mfe, failed_followthrough_close_r=0.0))
        for vwap_bars in (1, 2, 3):
            for fail_pct in (0.0, 0.001, 0.002, 0.004):
                specs.append(ExitSpec("", stop_mode=stop_mode, hard_stop_enabled=True, vwap_fail_bars=vwap_bars, vwap_fail_pct=fail_pct, target_r=1.0))
    for hold_bars in (6, 10, 12, 16, 24, 36):
        specs.append(ExitSpec("", stop_mode="atr", hard_stop_enabled=False, max_hold_bars=hold_bars))
        specs.append(ExitSpec("", stop_mode="first30_low", hard_stop_enabled=True, max_hold_bars=hold_bars))
    return _dedupe_exits(_name_exit(spec) for spec in specs)


def build_refined_trade_plan_specs(
    seed_rows: list[PlanResult],
    *,
    exclude_names: set[str] | None = None,
    max_specs: int = 30_000,
) -> list[TradePlanSpec]:
    excluded = set(exclude_names or set())
    out: dict[str, TradePlanSpec] = {}
    for row in seed_rows:
        for entry in _refined_entry_specs(row.spec.entry):
            for exit_spec in _refined_exit_specs(row.spec.exit):
                spec = name_plan(TradePlanSpec("", entry, exit_spec))
                if spec.name in excluded or spec.name in out:
                    continue
                out[spec.name] = spec
                if len(out) >= max(0, int(max_specs)):
                    return list(out.values())
    return list(out.values())


def evaluate_plan(
    spec: TradePlanSpec,
    selections: list[Selection],
    dataset: KALCBFirst30Dataset,
    context_by_key: dict[tuple[date, str], First30Context],
    cfg: KALCBConfig,
    dates: Iterable[date],
    selection_counts: dict[date, int],
    *,
    compiled_replay: CompiledCoreReplay | None = None,
) -> dict[str, float]:
    if compiled_replay is not None:
        outcomes, metrics = _core_outcomes_and_metrics(spec, compiled_replay, cfg, tuple(dates), selection_counts, audit=False)
        del outcomes
        return metrics
    outcomes = collect_trade_outcomes(spec, selections, dataset, context_by_key, cfg, dates, compiled_replay=compiled_replay)
    return summarize_outcomes(outcomes, session_dates=tuple(dates), selection_counts=selection_counts)


def collect_trade_outcomes(
    spec: TradePlanSpec,
    selections: list[Selection],
    dataset: KALCBFirst30Dataset,
    context_by_key: dict[tuple[date, str], First30Context],
    cfg: KALCBConfig,
    dates: Iterable[date],
    *,
    compiled_replay: CompiledCoreReplay | None = None,
) -> list[TradeOutcome]:
    return collect_core_trade_outcomes(spec, selections, dataset, context_by_key, cfg, dates, compiled_replay=compiled_replay)


def collect_standalone_trade_outcomes(
    spec: TradePlanSpec,
    selections: list[Selection],
    dataset: KALCBFirst30Dataset,
    context_by_key: dict[tuple[date, str], First30Context],
    cfg: KALCBConfig,
    dates: Iterable[date],
) -> list[TradeOutcome]:
    date_set = set(dates)
    outcomes: list[TradeOutcome] = []
    for selection in selections:
        if selection.trade_date not in date_set:
            continue
        ctx = context_by_key.get((selection.trade_date, selection.symbol))
        bars = dataset.bars_by_key.get((selection.trade_date, selection.symbol), ())
        if ctx is None or not bars:
            continue
        prior_day_high = _prior_day_high(dataset, selection.symbol, selection.trade_date)
        outcome = simulate_trade(selection.trade_date, selection.symbol, bars, ctx, spec.entry, spec.exit, cfg, prior_day_high=prior_day_high)
        if outcome is not None:
            outcomes.append(outcome)
    return outcomes


def compile_core_replay(
    selections: list[Selection],
    dataset: KALCBFirst30Dataset,
    context_by_key: dict[tuple[date, str], First30Context],
    dates: Iterable[date],
    selection_counts: dict[date, int],
    cfg: KALCBConfig,
    *,
    frontier_by_day: dict[date, tuple[str, ...]] | None = None,
    frontier_scores_by_day: dict[date, dict[str, float]] | None = None,
    candidate_metadata_by_key: dict[tuple[date, str], dict[str, Any]] | None = None,
    source_calibration_metadata: dict[str, Any] | None = None,
) -> CompiledCoreReplay:
    date_set = set(dates)
    snapshots: dict[date, KALCBDailySnapshot] = {}
    bars: list[MarketBar] = []
    selected_by_day: dict[date, list[Selection]] = {}
    for selection in selections:
        if selection.trade_date in date_set:
            selected_by_day.setdefault(selection.trade_date, []).append(selection)
    for day in sorted(date_set):
        candidates: list[KALCBDailyCandidate] = []
        included_symbols: set[str] = set()
        frontier_symbols = tuple((frontier_by_day or {}).get(day, ()))
        frontier_rank_by_symbol = {symbol: rank for rank, symbol in enumerate(frontier_symbols, start=1)}
        frontier_score_by_symbol = dict((frontier_scores_by_day or {}).get(day, {}))
        initial_active_symbols: list[str] = []
        for rank, selection in enumerate(sorted(selected_by_day.get(day, ()), key=lambda item: (-item.score, item.symbol)), start=1):
            ctx = context_by_key.get((selection.trade_date, selection.symbol))
            day_bars = dataset.bars_by_key.get((selection.trade_date, selection.symbol), ())
            if ctx is None or not day_bars:
                continue
            included_symbols.add(selection.symbol)
            initial_active_symbols.append(selection.symbol)
            bars.extend(day_bars)
            candidates.append(
                _candidate_from_context(
                    ctx,
                    selection,
                    dataset,
                    cfg,
                    rank,
                    frontier_rank=frontier_rank_by_symbol.get(selection.symbol, rank),
                    frontier_score=frontier_score_by_symbol.get(selection.symbol, 0.0),
                    frontier_initial_active=True,
                    frontier_role="initial_active",
                    extra_candidate_metadata=dict((candidate_metadata_by_key or {}).get((selection.trade_date, selection.symbol)) or {}),
                    source_calibration_metadata=source_calibration_metadata,
                )
            )
        if frontier_symbols:
            for symbol in frontier_symbols:
                if symbol in included_symbols:
                    continue
                ctx = context_by_key.get((day, symbol))
                day_bars = dataset.bars_by_key.get((day, symbol), ())
                if ctx is None or not day_bars:
                    continue
                included_symbols.add(symbol)
                bars.extend(day_bars)
                rank = len(candidates) + 1
                selection = Selection(day, symbol, 0.0, "frontier_shadow")
                candidates.append(
                    _candidate_from_context(
                        ctx,
                        selection,
                        dataset,
                        cfg,
                        rank,
                        frontier_rank=frontier_rank_by_symbol.get(symbol, rank),
                        frontier_score=frontier_score_by_symbol.get(symbol, 0.0),
                        frontier_initial_active=False,
                        frontier_role="frontier_shadow",
                        extra_candidate_metadata=dict((candidate_metadata_by_key or {}).get((day, symbol)) or {}),
                        source_calibration_metadata=source_calibration_metadata,
                    )
                )
        if candidates:
            snapshots[day] = KALCBDailySnapshot(
                trade_date=day,
                candidates=tuple(candidates),
                source_fingerprint=stable_signature([dataset.source_fingerprint, dataset.daily_source_fingerprint, day.isoformat()]),
                generated_at=datetime.combine(day, dtime.min, tzinfo=KST),
                metadata={
                    "research_only": False,
                    "fixed_candidate_generation": "kalcb_optimized_frontier_first30",
                    "selection_contract": "daily_selection_from_snapshot/run_daily_selection compatible snapshot",
                    "active_symbols": list(initial_active_symbols),
                    "active_symbol_count": len(initial_active_symbols),
                    "candidate_pool_count": len(candidates),
                    "frontier_symbol_count": len(frontier_symbols),
                    "selection_count": selection_counts.get(day, 0),
                    "core_version": KALCB_CORE_VERSION,
                },
            )
    ordered_bars = tuple(sorted(bars, key=lambda item: (item.timestamp, item.symbol)))
    candidate_hash = stable_signature({day.isoformat(): snapshot.artifact_hash for day, snapshot in sorted(snapshots.items())})
    return CompiledCoreReplay(
        bars=ordered_bars,
        snapshots=snapshots,
        session_dates=tuple(sorted(date_set)),
        selection_counts=dict(selection_counts),
        initial_equity=float((dataset.config or {}).get("initial_equity", 100_000_000.0) or 100_000_000.0),
        source_fingerprint=stable_signature([dataset.source_fingerprint, dataset.daily_source_fingerprint, candidate_hash]),
        candidate_artifact_hash=candidate_hash,
    )


def collect_core_trade_outcomes(
    spec: TradePlanSpec,
    selections: list[Selection],
    dataset: KALCBFirst30Dataset,
    context_by_key: dict[tuple[date, str], First30Context],
    cfg: KALCBConfig,
    dates: Iterable[date],
    *,
    compiled_replay: CompiledCoreReplay | None = None,
) -> list[TradeOutcome]:
    date_tuple = tuple(dates)
    compiled = compiled_replay or compile_core_replay(selections, dataset, context_by_key, date_tuple, _selection_counts(selections, dataset.trading_dates), cfg)
    if not compiled.snapshots or not compiled.bars:
        return []
    replay = _run_core_replay(spec, compiled, cfg, audit=False)
    return _broker_trades_to_slot_outcomes(_collapse_exit_legs(replay.trades), cfg)


def _core_outcomes_and_metrics(
    spec: TradePlanSpec,
    compiled: CompiledCoreReplay,
    cfg: KALCBConfig,
    dates: tuple[date, ...],
    selection_counts: dict[date, int],
    *,
    audit: bool,
) -> tuple[list[TradeOutcome], dict[str, float]]:
    outcomes, metrics, _digest = _core_outcomes_metrics_digest(spec, compiled, cfg, dates, selection_counts, audit=audit)
    return outcomes, metrics


def _core_outcomes_metrics_digest(
    spec: TradePlanSpec,
    compiled: CompiledCoreReplay,
    cfg: KALCBConfig,
    dates: tuple[date, ...],
    selection_counts: dict[date, int],
    *,
    audit: bool,
) -> tuple[list[TradeOutcome], dict[str, float], dict[str, Any]]:
    replay = _run_core_replay(spec, compiled, cfg, audit=audit)
    trades = _collapse_exit_legs(replay.trades)
    outcomes = _broker_trades_to_slot_outcomes(trades, cfg)
    metrics = summarize_outcomes(outcomes, session_dates=dates, selection_counts=selection_counts)
    _add_compiled_candidate_pool_metrics(metrics, compiled, dates, len(outcomes))
    broker_metrics = compute_trade_metrics(trades, replay.equity_curve, initial_equity=compiled.initial_equity)
    final_equity = float(replay.equity_curve[-1]) if replay.equity_curve else float(compiled.initial_equity)
    official_mtm_net = (final_equity / float(compiled.initial_equity) - 1.0) if compiled.initial_equity else 0.0
    metrics.update(
        {
            "broker_net_return_pct": float(broker_metrics.get("net_return_pct", 0.0)),
            "official_mtm_net_return_pct": float(official_mtm_net),
            "final_equity": float(final_equity),
            "end_open_position_count": float(len(replay.broker.positions)),
            "broker_net_profit": float(broker_metrics.get("net_profit", 0.0)),
            "broker_max_drawdown_pct": float(broker_metrics.get("max_drawdown_pct", 0.0)),
            "broker_expected_total_r": float(broker_metrics.get("expected_total_r", 0.0)),
            "broker_avg_r": float(broker_metrics.get("avg_r", 0.0)),
            "broker_mfe_capture": float(broker_metrics.get("mfe_capture", 0.0)),
            "broker_trade_count": float(broker_metrics.get("total_trades", 0.0)),
            "same_bar_fill_count": float(replay.broker.same_bar_fill_violations),
            "mark_to_market_equity_points": float(len(replay.equity_curve)),
            "broker_net_return_pct_basis": "closed_trade_net_pnl_over_initial_equity",
            "net_return_pct_basis": "closed_trade_net_pnl_over_initial_equity",
            "official_metric_basis": "SimBroker.equity_curve_bar_level_mtm",
        }
    )
    _add_portfolio_equivalent_metrics(metrics, outcomes, dates, compiled.initial_equity)
    _add_return_divergence_metrics(metrics)
    return outcomes, metrics, _replay_digest(replay, trades)


def _add_compiled_candidate_pool_metrics(
    metrics: dict[str, float],
    compiled: CompiledCoreReplay,
    dates: tuple[date, ...],
    trade_count: int,
) -> None:
    pool_counts = [
        int((compiled.snapshots.get(day).metadata or {}).get("candidate_pool_count", 0) or 0)
        for day in dates
        if compiled.snapshots.get(day) is not None
    ]
    active_counts = [
        int((compiled.snapshots.get(day).metadata or {}).get("active_symbol_count", 0) or 0)
        for day in dates
        if compiled.snapshots.get(day) is not None
    ]
    candidate_pool_count = float(sum(pool_counts))
    active_candidate_count = float(sum(active_counts))
    metrics["candidate_pool_count"] = candidate_pool_count
    metrics["candidate_pool_days"] = float(len(pool_counts))
    metrics["avg_candidate_pool_per_day"] = _avg(pool_counts)
    metrics["initial_active_candidate_count"] = active_candidate_count
    metrics["frontier_expansion_candidate_count"] = max(0.0, candidate_pool_count - active_candidate_count)
    metrics["candidate_pool_conversion"] = float(trade_count) / max(candidate_pool_count, 1.0)
    metrics["initial_active_conversion"] = float(trade_count) / max(active_candidate_count, 1.0)


def _add_return_divergence_metrics(metrics: dict[str, float]) -> None:
    slot_net = float(metrics.get(LEGACY_SLOT_METRIC, 0.0))
    broker_net = float(metrics.get(PRIMARY_OBJECTIVE_METRIC, metrics.get(PORTFOLIO_EQUIVALENT_METRIC, slot_net)))
    portfolio_equivalent = float(metrics.get(PORTFOLIO_EQUIVALENT_METRIC, broker_net))
    metrics[EQUAL_SLOT_METRIC] = slot_net
    metrics[EQUAL_SLOT_GROSS_METRIC] = float(metrics.get("slot_cumulative_gross_return_pct", 0.0))
    metrics[EXPOSURE_NORMALIZED_SLOT_METRIC] = portfolio_equivalent
    metrics[EXPOSURE_NORMALIZED_SLOT_CUMULATIVE_METRIC] = float(
        metrics.get("portfolio_equivalent_cumulative_net_return_pct", portfolio_equivalent)
    )
    metrics["primary_objective_net_return_pct"] = broker_net
    metrics["slot_minus_broker_net_return_pct"] = slot_net - broker_net
    metrics["slot_to_broker_net_return_ratio"] = slot_net / broker_net if abs(broker_net) > 1e-9 else 0.0
    metrics["equal_slot_minus_broker_net_return_pct"] = metrics["slot_minus_broker_net_return_pct"]
    metrics["equal_slot_to_broker_net_return_ratio"] = metrics["slot_to_broker_net_return_ratio"]
    metrics["portfolio_equivalent_minus_broker_net_return_pct"] = portfolio_equivalent - broker_net
    metrics["exposure_normalized_slot_minus_broker_net_return_pct"] = portfolio_equivalent - broker_net
    metrics["exposure_normalized_slot_to_broker_ratio"] = portfolio_equivalent / broker_net if abs(broker_net) > 1e-9 else 0.0


def _add_portfolio_equivalent_metrics(
    metrics: dict[str, float],
    outcomes: list[TradeOutcome],
    session_dates: Iterable[date],
    initial_equity: float,
) -> None:
    basis = max(float(initial_equity or 0.0), 1.0)
    date_tuple = tuple(session_dates)
    date_set = set(date_tuple)
    by_day: dict[date, float] = {day: 0.0 for day in date_tuple}
    total = 0.0
    for outcome in outcomes:
        if outcome.trade_date not in date_set:
            continue
        pnl = float(outcome.net_pnl)
        by_day[outcome.trade_date] = by_day.get(outcome.trade_date, 0.0) + pnl
        total += pnl
    daily_returns = [by_day.get(day, 0.0) / basis for day in date_tuple]
    metrics[PORTFOLIO_EQUIVALENT_METRIC] = float(total / basis)
    metrics["portfolio_equivalent_cumulative_net_return_pct"] = _compound(daily_returns)
    metrics["portfolio_equivalent_max_drawdown_pct"] = abs(_max_drawdown(daily_returns))
    metrics["portfolio_equivalent_calendar_day_net_pct"] = sum(daily_returns) / max(float(len(daily_returns)), 1.0)


def _audit_replay_rows(
    rows: list[PlanResult],
    compiled: CompiledCoreReplay,
    cfg: KALCBConfig,
    train_dates: tuple[date, ...],
    selection_counts: dict[date, int],
    *,
    max_workers: int = 1,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    worker_count = max(1, min(int(max_workers), 4))
    if worker_count <= 1 or len(rows) == 1:
        return [_audit_replay_row(row, compiled, cfg, train_dates, selection_counts) for row in rows]
    out: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_audit_replay_row, row, compiled, cfg, train_dates, selection_counts): row
            for row in rows
        }
        for future in as_completed(futures):
            out.append(future.result())
    out.sort(key=lambda item: item.get("name", ""))
    return out


def _audit_replay_row(
    row: PlanResult,
    compiled: CompiledCoreReplay,
    cfg: KALCBConfig,
    train_dates: tuple[date, ...],
    selection_counts: dict[date, int],
) -> dict[str, Any]:
    _audit_outcomes, audit_metrics, audit_digest = _core_outcomes_metrics_digest(
        row.spec,
        compiled,
        cfg,
        train_dates,
        selection_counts,
        audit=True,
    )
    metric_names = (
        "slot_cumulative_net_return_pct",
        "equal_slot_net_return_pct",
        "slot_cumulative_gross_return_pct",
        "equal_slot_gross_return_pct",
        "calendar_day_net_pct",
        "active_day_net_pct",
        "avg_trade_net_pct",
        "avg_mfe_r",
        "avg_mfe_capture",
        "max_drawdown_net_pct",
        "broker_net_return_pct",
        "official_mtm_net_return_pct",
        "final_equity",
        "end_open_position_count",
        "broker_max_drawdown_pct",
        "primary_objective_net_return_pct",
        "portfolio_equivalent_net_return_pct",
        "exposure_normalized_slot_net_return_pct",
        "portfolio_equivalent_cumulative_net_return_pct",
        "exposure_normalized_slot_cumulative_net_return_pct",
        "portfolio_equivalent_max_drawdown_pct",
        "slot_minus_broker_net_return_pct",
        "slot_to_broker_net_return_ratio",
        "equal_slot_minus_broker_net_return_pct",
        "equal_slot_to_broker_net_return_ratio",
        "portfolio_equivalent_minus_broker_net_return_pct",
        "exposure_normalized_slot_minus_broker_net_return_pct",
        "exposure_normalized_slot_to_broker_ratio",
        "same_bar_fill_count",
    )
    deltas = {
        name: float(audit_metrics.get(name, 0.0)) - float(row.train_metrics.get(name, 0.0))
        for name in metric_names
    }
    fast_digest = dict(row.replay_digest or {})
    fill_hash_match = fast_digest.get("fill_hash", "") == audit_digest.get("fill_hash", "")
    trade_hash_match = fast_digest.get("trade_hash", "") == audit_digest.get("trade_hash", "")
    trading_decision_hash_match = fast_digest.get("trading_decision_hash", "") == audit_digest.get("trading_decision_hash", "")
    audit_pass = (
        all(abs(value) <= 1e-10 for value in deltas.values())
        and fill_hash_match
        and trade_hash_match
        and trading_decision_hash_match
    )
    return {
        "name": row.spec.name,
        "promotion_pass": row.promotion_pass,
        "audit_pass": audit_pass,
        "max_abs_metric_delta": max((abs(value) for value in deltas.values()), default=0.0),
        "metric_deltas": deltas,
        "audit_metrics": audit_metrics,
        "fast_replay_digest": fast_digest,
        "audit_replay_digest": audit_digest,
        "fill_hash_match": fill_hash_match,
        "trade_hash_match": trade_hash_match,
        "trading_decision_hash_match": trading_decision_hash_match,
        "fast_decision_count": int(fast_digest.get("decision_count", 0) or 0),
        "audit_decision_count": int(audit_digest.get("decision_count", 0) or 0),
        "suppressed_entry_rejection_count": max(
            0,
            int(audit_digest.get("entry_rejection_count", 0) or 0) - int(fast_digest.get("entry_rejection_count", 0) or 0),
        ),
        "trade_count": int(audit_digest.get("trade_count", 0) or 0),
    }


def _replay_digest(replay: Any, trades: list[BrokerTradeOutcome]) -> dict[str, Any]:
    fill_rows = [
        {
            "index": index,
            "strategy_id": fill.strategy_id,
            "symbol": fill.symbol,
            "side": fill.side,
            "qty": int(fill.qty),
            "price": float(fill.price),
            "timestamp": fill.timestamp.isoformat(),
            "reason": fill.reason,
            "metadata": dict(fill.metadata),
        }
        for index, fill in enumerate(replay.broker.fills, start=1)
    ]
    trade_rows = [trade.to_json_dict() for trade in trades]
    trading_decision_events = [
        decision
        for decision in replay.decisions
        if decision.decision_code != "entry_rejected"
    ]
    trading_decisions = [decision.to_json_dict() for decision in trading_decision_events]
    strategy_actions = [
        {
            "decision_index": decision_index,
            "action_index": action_index,
            "decision_code": decision.decision_code,
            "decision_reason": decision.reason,
            "payload": action_to_json_dict(action),
        }
        for decision_index, decision in enumerate(trading_decision_events, start=1)
        for action_index, action in enumerate(decision.actions, start=1)
    ]
    entry_rejections = sum(1 for decision in replay.decisions if decision.decision_code == "entry_rejected")
    entry_rejection_reasons = Counter(
        str(decision.reason or "unknown")
        for decision in replay.decisions
        if decision.decision_code == "entry_rejected"
    )
    entry_rejection_failed_gates: Counter[str] = Counter()
    for decision in replay.decisions:
        if decision.decision_code != "entry_rejected":
            continue
        metadata = dict(decision.metadata or {})
        gates = metadata.get("gates") or metadata.get("filter_decisions") or ()
        first_failed = ""
        for gate in gates:
            if not isinstance(gate, dict):
                continue
            if bool(gate.get("applicable", True)) and not bool(gate.get("passed", True)):
                first_failed = str(gate.get("filter_name") or "")
                break
        entry_rejection_failed_gates[first_failed or str(decision.reason or "unknown")] += 1
    return {
        "fill_count": len(fill_rows),
        "fill_hash": stable_signature(fill_rows),
        "trade_count": len(trade_rows),
        "trade_hash": stable_signature(trade_rows),
        "decision_count": len(replay.decisions),
        "trading_decision_count": len(trading_decisions),
        "trading_decision_hash": stable_signature(trading_decisions),
        "strategy_action_count": len(strategy_actions),
        "strategy_action_hash": stable_signature(strategy_actions),
        "entry_rejection_count": entry_rejections,
        "entry_rejection_reasons": dict(sorted(entry_rejection_reasons.items())),
        "top_entry_rejection_reasons": [
            {"reason": reason, "count": int(count)}
            for reason, count in entry_rejection_reasons.most_common(12)
        ],
        "top_entry_failed_gates": [
            {"gate": gate, "count": int(count)}
            for gate, count in entry_rejection_failed_gates.most_common(12)
        ],
        "same_bar_fill_count": int(replay.broker.same_bar_fill_violations),
        "equity_point_count": len(replay.equity_curve),
    }


def _audit_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "status": "not_run",
            "audit_count": 0,
            "pass": True,
            "max_abs_metric_delta": 0.0,
            "fill_hash_mismatches": 0,
            "trade_hash_mismatches": 0,
            "trading_decision_hash_mismatches": 0,
            "suppressed_entry_rejection_count": 0,
        }
    return {
        "status": "pass" if all(bool(row.get("audit_pass")) for row in rows) else "fail",
        "audit_count": len(rows),
        "pass": all(bool(row.get("audit_pass")) for row in rows),
        "max_abs_metric_delta": max(float(row.get("max_abs_metric_delta", 0.0) or 0.0) for row in rows),
        "fill_hash_mismatches": sum(1 for row in rows if not row.get("fill_hash_match")),
        "trade_hash_mismatches": sum(1 for row in rows if not row.get("trade_hash_match")),
        "trading_decision_hash_mismatches": sum(1 for row in rows if not row.get("trading_decision_hash_match")),
        "suppressed_entry_rejection_count": sum(int(row.get("suppressed_entry_rejection_count", 0) or 0) for row in rows),
        "scope": "Only entry_rejected diagnostics may differ between fast and full audit paths.",
    }


def _run_core_replay(spec: TradePlanSpec, compiled: CompiledCoreReplay, cfg: KALCBConfig, *, audit: bool):
    plan_cfg = _core_config_for_spec(cfg, spec, audit=audit)
    costs = BrokerCosts(commission_bps=plan_cfg.commission_bps, tax_bps_on_sell=plan_cfg.tax_bps_on_sell, slippage_bps=plan_cfg.slippage_bps)
    adapter = KALCBReplayAdapter(plan_cfg, _clone_snapshots_for_replay(compiled.snapshots), initial_equity=compiled.initial_equity, costs=costs)
    result = run_replay(
        compiled.bars,
        adapter,
        initial_equity=compiled.initial_equity,
        costs=costs,
        close_open_positions=False,
        bars_are_ordered=True,
        buying_power_leverage=max(float(plan_cfg.intraday_leverage), 1.0),
    )
    result.decisions.extend(adapter._sync_new_fills(result.broker))
    result.trades = _collapse_exit_legs(result.trades)
    return result


def _clone_snapshots_for_replay(snapshots: dict[date, KALCBDailySnapshot]) -> dict[date, KALCBDailySnapshot]:
    return {day: _clone_snapshot(snapshot) for day, snapshot in snapshots.items()}


def _clone_snapshot(snapshot: KALCBDailySnapshot) -> KALCBDailySnapshot:
    return KALCBDailySnapshot(
        trade_date=snapshot.trade_date,
        candidates=tuple(_clone_candidate(candidate) for candidate in snapshot.candidates),
        source_fingerprint=snapshot.source_fingerprint,
        generated_at=snapshot.generated_at,
        strategy_id=snapshot.strategy_id,
        metadata=dict(snapshot.metadata),
    )


def _clone_candidate(candidate: KALCBDailyCandidate) -> KALCBDailyCandidate:
    return replace(
        candidate,
        reject_reasons=tuple(candidate.reject_reasons),
        metadata=dict(candidate.metadata),
    )


def _core_config_for_spec(cfg: KALCBConfig, spec: TradePlanSpec, *, audit: bool) -> KALCBConfig:
    entry_mode = "first30_open" if spec.entry.mode == BASELINE_ENTRY_MODE else spec.entry.mode
    if entry_mode == "confirm_next_bar":
        entry_mode = "post_or_momentum"
    risk_policy = _portfolio_risk_mutations(cfg)
    mutations: dict[str, Any] = {
        "kalcb.entry.plan_mode": entry_mode,
        "kalcb.entry.max_signal_bars": max(1, int(spec.entry.max_signal_bars)),
        "kalcb.entry.after_bar": max(0, int(spec.entry.after_bar)),
        "kalcb.entry.min_bar_ret": float(spec.entry.min_bar_ret),
        "kalcb.entry.min_vwap_ret": float(spec.entry.min_vwap_ret),
        "kalcb.entry.min_breakout_pct": float(spec.entry.min_breakout_pct),
        "kalcb.entry.max_pullback_from_vwap_pct": float(spec.entry.max_pullback_from_vwap_pct),
        "kalcb.entry.min_reclaim_ret": float(spec.entry.min_reclaim_ret),
        "kalcb.entry.min_reclaim_closes": int(spec.entry.min_reclaim_closes),
        "kalcb.entry.min_close_location": float(spec.entry.min_close_location),
        "kalcb.entry.min_or_position": float(spec.entry.min_or_position),
        "kalcb.entry.max_avwap_extension_pct": float(spec.entry.max_avwap_extension_pct),
        "kalcb.entry.require_above_prev_close": bool(spec.entry.require_above_prev_close),
        "kalcb.entry.reclaim_level_source": str(spec.entry.reclaim_level_source or "legacy"),
        "kalcb.entry.rvol_threshold": 0.0,
        "kalcb.entry.rvol_max": 999.0,
        "kalcb.entry.cpr_threshold": float(spec.entry.min_close_location),
        "kalcb.entry.momentum_score_min": 0,
        "kalcb.entry.breakout_distance_cap_r": 0.0,
        "kalcb.entry.orb_entry_range_cap_r": 0.0,
        "kalcb.entry.fast_replay_suppress_rejections": not audit,
        "kalcb.exit.hard_stop_enabled": bool(spec.exit.hard_stop_enabled),
        "kalcb.exit.stop_mode": spec.exit.stop_mode,
        "kalcb.risk.stop_atr_multiple": float(spec.exit.stop_atr_mult),
        "kalcb.exit.stop_pct": float(spec.exit.stop_pct),
        "kalcb.exit.target_r": float(spec.exit.target_r),
        "kalcb.exit.use_partial_takes": bool(spec.exit.partial_trigger_r > 0 and spec.exit.partial_fraction > 0),
        "kalcb.exit.partial_r_trigger": float(spec.exit.partial_trigger_r or 1.0),
        "kalcb.exit.partial_fraction": float(spec.exit.partial_fraction or 0.5),
        "kalcb.exit.partial_breakeven_buffer_r": float(spec.exit.partial_stop_r),
        "kalcb.exit.breakeven_trigger_r": float(spec.exit.breakeven_trigger_r),
        "kalcb.exit.breakeven_stop_r": float(spec.exit.breakeven_stop_r),
        "kalcb.exit.trail_start_r": float(spec.exit.trail_start_r),
        "kalcb.exit.trail_gap_r": float(spec.exit.trail_gap_r),
        "kalcb.exit.no_mfe_bars": int(spec.exit.no_mfe_bars),
        "kalcb.exit.no_mfe_thresh_r": float(spec.exit.no_mfe_thresh_r),
        "kalcb.exit.failed_followthrough_bars": int(spec.exit.failed_followthrough_bars),
        "kalcb.exit.failed_followthrough_mfe_r": float(spec.exit.failed_followthrough_mfe_r),
        "kalcb.exit.failed_followthrough_close_r": float(spec.exit.failed_followthrough_close_r),
        "kalcb.exit.vwap_fail_bars": int(spec.exit.vwap_fail_bars),
        "kalcb.exit.vwap_fail_pct": float(spec.exit.vwap_fail_pct),
        "kalcb.exit.max_hold_bars": int(spec.exit.max_hold_bars),
        "kalcb.exit.quick_exit_enabled": False,
        "kalcb.exit.failure_stop_enabled": False,
        "kalcb.exit.mfe_conviction_enabled": False,
        "kalcb.exit.adaptive_trail_enabled": False,
        "kalcb.exit.flow_reversal_enabled": False,
        "kalcb.carry.mode": "off",
        "kalcb.frontier.shadow_enabled": False,
        "kalcb.frontier.rotation_enabled": False,
        "kalcb.session.ws_budget": max(1, cfg.ws_budget),
        **risk_policy,
    }
    return cfg.with_mutations(mutations)


def _portfolio_risk_mutations(cfg: KALCBConfig) -> dict[str, Any]:
    policy = PORTFOLIO_RISK_POLICY
    max_positions = min(max(1, int(cfg.ws_budget)), int(policy["max_positions_cap"]))
    max_per_sector = min(max_positions, int(policy["max_per_sector_cap"]))
    risk_per_trade = min(
        max(float(cfg.risk_per_trade_pct), float(policy["risk_per_trade_pct"])),
        float(policy["risk_per_trade_pct_cap"]),
    )
    max_notional = min(
        max(float(cfg.max_position_notional_pct), float(policy["max_position_notional_pct"])),
        float(policy["max_position_notional_pct_cap"]),
    )
    heat_cap = max(float(cfg.heat_cap_r), float(policy["heat_cap_pct"]) / max(risk_per_trade, 1e-9))
    return {
        "kalcb.risk.risk_per_trade_pct": risk_per_trade,
        "kalcb.risk.max_position_notional_pct": max_notional,
        "kalcb.risk.max_positions": max_positions,
        "kalcb.risk.max_per_sector": max_per_sector,
        "kalcb.risk.heat_cap_r": heat_cap,
        "kalcb.risk.intraday_leverage": float(policy["intraday_leverage"]),
        "kalcb.risk.max_participation_30m": max(float(cfg.max_participation_30m), float(policy["max_participation_30m"])),
    }


def _candidate_from_context(
    ctx: First30Context,
    selection: Selection,
    dataset: KALCBFirst30Dataset,
    cfg: KALCBConfig,
    rank: int,
    *,
    frontier_rank: int | None = None,
    frontier_score: float = 0.0,
    frontier_initial_active: bool = True,
    frontier_role: str = "initial_active",
    source_calibration_metadata: dict[str, Any] | None = None,
    extra_candidate_metadata: dict[str, Any] | None = None,
) -> KALCBDailyCandidate:
    prior = _prior_daily_row(dataset, selection.symbol, selection.trade_date) or {}
    prior_high = _float(prior.get("high"), ctx.daily.prev_close)
    prior_low = _float(prior.get("low"), ctx.daily.prev_close)
    prior_close = _float(prior.get("close"), ctx.daily.prev_close)
    avg30 = max(float(ctx.intraday.expected_30m_volume or ctx.intraday.volume or 0.0), 1.0)
    rel_volume_log = math.log1p(max(float(ctx.rel_volume or 0.0), 0.0))
    gap_retention_ratio = float(ctx.low_vs_prev_close) / max(abs(float(ctx.gap)), 1e-6) if ctx.gap > 0.0 else 0.0
    sector_daily_feature = getattr(ctx, "sector_daily", None)
    sector_intraday_feature = getattr(ctx, "sector_intraday", None)
    sector_daily_metadata = dict(sector_daily_feature.metadata()) if sector_daily_feature is not None else {}
    sector_intraday_metadata = dict(sector_intraday_feature.metadata()) if sector_intraday_feature is not None else {}
    sector_daily_ret_5d = float(sector_daily_metadata.get("sector_daily_ret_5d") or 0.0)
    sector_daily_ret_20d = float(sector_daily_metadata.get("sector_daily_ret_20d") or 0.0)
    sector_intraday_ret = float(sector_intraday_metadata.get("sector_intraday_ret") or 0.0)
    sector_intraday_rel_volume = max(float(sector_intraday_metadata.get("sector_intraday_rel_volume") or 1.0), 1e-6)
    sector_intraday_breadth = float(sector_intraday_metadata.get("sector_intraday_breadth") or 0.5)
    daily_acceleration = float(ctx.daily.return_5d) - 0.25 * float(ctx.daily.return_20d)
    daily_momentum_pct = 100.0 * (
        0.30 * _bounded(0.5 + float(ctx.daily.return_20d) / 0.40, 0.0, 1.0)
        + 0.18 * _bounded(0.5 + float(ctx.daily.return_60d) / 0.70, 0.0, 1.0)
        + 0.17 * _bounded(float(ctx.daily.close20_loc), 0.0, 1.0)
        + 0.12 * _bounded(float(ctx.daily.close60_loc), 0.0, 1.0)
        + 0.13 * _bounded(0.5 + math.log(max(float(ctx.daily.volume_ratio_20d), 0.1)) / 4.0, 0.0, 1.0)
        + 0.10 * _bounded(0.5 + daily_acceleration / 0.12, 0.0, 1.0)
    )
    first30_quality_pct = 100.0 * (
        0.24 * _bounded(0.5 + float(ctx.first30_ret) / 0.06, 0.0, 1.0)
        + 0.18 * _bounded(0.5 + float(ctx.vwap_ret) / 0.04, 0.0, 1.0)
        + 0.18 * _bounded(float(ctx.close_location), 0.0, 1.0)
        + 0.16 * _bounded(math.log1p(max(float(ctx.rel_volume), 0.0)) / math.log(21.0), 0.0, 1.0)
        + 0.14 * _bounded(float(gap_retention_ratio), 0.0, 1.25) / 1.25
        + 0.10 * _bounded(0.5 + float(ctx.low_vs_prev_close) / 0.08, 0.0, 1.0)
    )
    first30_sector_ret_spread = float(ctx.first30_ret) - sector_intraday_ret
    first30_sector_relvol_ratio = float(ctx.rel_volume) / sector_intraday_rel_volume
    first30_sector_leadership_pct = 100.0 * (
        0.45 * _bounded(0.5 + first30_sector_ret_spread / 0.06, 0.0, 1.0)
        + 0.25 * _bounded(math.log(max(first30_sector_relvol_ratio, 0.1)) / 3.0 + 0.5, 0.0, 1.0)
        + 0.20 * _bounded(float(ctx.close_location), 0.0, 1.0)
        + 0.10 * _bounded(float(gap_retention_ratio), 0.0, 1.25) / 1.25
    )
    stock_sector_daily_ret20_spread = float(ctx.daily.return_20d) - sector_daily_ret_20d
    stock_sector_daily_ret5_spread = float(ctx.daily.return_5d) - sector_daily_ret_5d
    daily_sector_alignment_pct = 100.0 * (
        0.45 * _bounded(0.5 + stock_sector_daily_ret20_spread / 0.40, 0.0, 1.0)
        + 0.25 * _bounded(0.5 + stock_sector_daily_ret5_spread / 0.16, 0.0, 1.0)
        + 0.20 * _bounded(float(sector_daily_metadata.get("sector_daily_score_pct") or 50.0) / 100.0, 0.0, 1.0)
        + 0.10 * _bounded(float(sector_daily_metadata.get("sector_daily_participation") or 0.0), 0.0, 1.0)
    )
    continuation_joint_quality_pct = 100.0 * (
        0.34 * first30_quality_pct / 100.0
        + 0.23 * daily_momentum_pct / 100.0
        + 0.18 * first30_sector_leadership_pct / 100.0
        + 0.15 * daily_sector_alignment_pct / 100.0
        + 0.10 * _bounded(sector_intraday_breadth, 0.0, 1.0)
    )
    metadata = {
        "source": "fixed_kalcb_optimized_first30",
        "research_replay_contract": "snapshot_shape_matches_live_daily_selection_output",
        "research_lookahead_policy": "premarket_rows_date_lt_trade_date_first30_0900_0925_only",
        "active_symbol_count": 1,
        "candidate_rank": rank,
        "frontier_rank": int(frontier_rank or rank),
        "frontier_initial_active": bool(frontier_initial_active),
        "frontier_role": str(frontier_role),
        "frontier_selection_score": float(frontier_score),
        "first30_score": selection.score,
        "first30_family": selection.family,
        "first30_ret": ctx.first30_ret,
        "first30_vwap_ret": ctx.vwap_ret,
        "first30_gap": ctx.gap,
        "first30_rel_volume": ctx.rel_volume,
        "first30_close_location": ctx.close_location,
        "first30_open_drawdown": ctx.open_drawdown,
        "first30_low_vs_prev_close": ctx.low_vs_prev_close,
        "first30_range_atr": ctx.range_atr,
        "first30_gap_retention_ratio": gap_retention_ratio,
        "first30_gap_relvol": ctx.gap * rel_volume_log,
        "first30_low_vs_prev_relvol": ctx.low_vs_prev_close * rel_volume_log,
        "daily_return_5d": float(ctx.daily.return_5d),
        "daily_return_20d": float(ctx.daily.return_20d),
        "daily_return_60d": float(ctx.daily.return_60d),
        "daily_volume_ratio_20d": float(ctx.daily.volume_ratio_20d),
        "daily_close20_loc": float(ctx.daily.close20_loc),
        "daily_close60_loc": float(ctx.daily.close60_loc),
        "daily_above_sma20": bool(ctx.daily.above_sma20),
        "daily_above_sma60": bool(ctx.daily.above_sma60),
        "daily_acceleration_5v20": daily_acceleration,
        "daily_momentum_pct": daily_momentum_pct,
        "sector_flow_participation": float(ctx.flow.sector_participation),
        "sector_participation": float(sector_daily_metadata.get("sector_daily_participation", ctx.flow.sector_participation) or 0.0),
        "daily_sector_alignment_pct": daily_sector_alignment_pct,
        "stock_sector_daily_ret20_spread": stock_sector_daily_ret20_spread,
        "stock_sector_daily_ret5_spread": stock_sector_daily_ret5_spread,
        "first30_quality_pct": first30_quality_pct,
        "first30_sector_ret_spread": first30_sector_ret_spread,
        "first30_sector_relvol_ratio": first30_sector_relvol_ratio,
        "first30_sector_leadership_pct": first30_sector_leadership_pct,
        "first30_gap_relvol_sector_breadth": ctx.gap * rel_volume_log * sector_intraday_breadth,
        "first30_gap_retention_sector_breadth": gap_retention_ratio * sector_intraday_breadth,
        "continuation_joint_quality_pct": continuation_joint_quality_pct,
        **sector_daily_metadata,
        **sector_intraday_metadata,
        **dict(extra_candidate_metadata or {}),
        "core_version": KALCB_CORE_VERSION,
        "ws_budget": cfg.ws_budget,
    }
    if source_calibration_metadata:
        metadata["source_calibration"] = dict(source_calibration_metadata)
    return KALCBDailyCandidate(
        symbol=selection.symbol,
        trade_date=selection.trade_date,
        prior_day_high=prior_high,
        prior_day_low=prior_low,
        prior_day_close=prior_close,
        daily_atr=max(float(ctx.daily.atr14 or 0.0), 1.0),
        expected_5m_volume=max(avg30 / 6.0, 1.0),
        average_30m_volume=avg30,
        sector=ctx.sector or "UNKNOWN",
        regime_tier="A" if ctx.market.score >= 0 else "B",
        selection_score=float(selection.score),
        rs_percentile=float((extra_candidate_metadata or {}).get("rs_percentile") or (extra_candidate_metadata or {}).get("relative_strength_pct") or 0.0),
        accumulation_score=float(ctx.flow.combined_5d if ctx.flow.available else 0.0),
        flow_score=float(ctx.flow.sector_flow_5d if ctx.flow.available else 0.0),
        tradable=True,
        source_fingerprint=stable_signature([dataset.source_fingerprint, dataset.daily_source_fingerprint, selection.trade_date.isoformat(), selection.symbol]),
        metadata=metadata,
    )


def _broker_trades_to_slot_outcomes(trades: list[BrokerTradeOutcome], cfg: KALCBConfig) -> list[TradeOutcome]:
    del cfg
    outcomes: list[TradeOutcome] = []
    for trade in trades:
        notional = max(float(trade.entry_price) * max(int(trade.qty), 1), 1e-9)
        risk = max(float(trade.route_metadata.get("risk_per_share") or 0.0), 1.0)
        gross_ret = float(trade.gross_pnl) / notional
        net_ret = float(trade.net_pnl) / notional
        route = dict(trade.route_metadata or {})
        outcomes.append(
            TradeOutcome(
                trade_date=trade.entry_fill_time.astimezone(KST).date(),
                symbol=trade.symbol,
                entry_time=trade.entry_fill_time,
                entry_price=float(trade.entry_price),
                stop_price=float(trade.route_metadata.get("initial_stop") or trade.route_metadata.get("stop_price") or 0.0),
                risk_per_share=risk,
                gross_return_pct=gross_ret,
                net_return_pct=net_ret,
                mfe_r=max(float(trade.mfe), 0.0) / risk,
                mae_r=min(float(trade.mae), 0.0) / risk,
                mfe_capture=max(float(trade.exit_price or trade.entry_price) - float(trade.entry_price), 0.0) / max(float(trade.mfe), 1e-9) if trade.mfe > 0 else 0.0,
                bars_held=max(1, int((trade.exit_fill_time - trade.entry_fill_time).total_seconds() // 300) + 1) if trade.exit_fill_time else 1,
                exit_reason=str(trade.exit_reason),
                ambiguous_bar_count=0,
                stopped=str(trade.exit_reason).lower() in {"initial_stop", "hard_stop", "protective_stop"},
                target_hit=str(trade.exit_reason).lower() in {"target", "target_r"},
                partial_hit=bool(trade.cohort_metadata.get("partial_taken", False)),
                gross_pnl=float(trade.gross_pnl),
                net_pnl=float(trade.net_pnl),
                entry_type=str(route.get("entry_type") or ""),
                frontier_role=str(route.get("frontier_role") or ""),
                candidate_rank=int(route.get("candidate_rank") or 0),
                frontier_rank=int(route.get("frontier_rank") or 0),
            )
        )
    return outcomes


def simulate_trade(
    trade_date: date,
    symbol: str,
    bars: tuple[MarketBar, ...],
    ctx: First30Context,
    entry_spec: EntrySpec,
    exit_spec: ExitSpec,
    cfg: KALCBConfig,
    *,
    prior_day_high: float | None = None,
    candidate_metadata: dict[str, Any] | None = None,
) -> TradeOutcome | None:
    ordered_bars = tuple(bar for bar in sorted(bars, key=lambda item: item.timestamp) if bar.timestamp.astimezone(KST).time() <= cfg.flatten_time)
    if not ordered_bars:
        return None
    signal = find_entry_signal(ordered_bars, ctx, entry_spec, cfg, prior_day_high=prior_day_high, candidate_metadata=candidate_metadata)
    if signal is None or signal.fill_index >= len(ordered_bars):
        return None
    entry_bar = ordered_bars[signal.fill_index]
    entry = max(float(entry_bar.open), 1e-9)
    stop = initial_stop_price(ordered_bars, ctx, signal, entry, exit_spec, cfg)
    if stop >= entry:
        stop = entry * (1.0 - 0.006)
    risk = max(entry - stop, entry * 0.001, 1.0)
    exit_legs, stats = simulate_exits(ordered_bars, signal.fill_index, entry, stop, risk, exit_spec, cfg)
    gross = sum(float(leg.fraction) * (float(leg.price) / entry - 1.0) for leg in exit_legs)
    net = gross - _round_trip_cost_pct(cfg)
    future = ordered_bars[signal.fill_index :]
    high = max(float(bar.high) for bar in future)
    low = min(float(bar.low) for bar in future)
    mfe_r = max(0.0, (high - entry) / risk)
    mae_r = (low - entry) / risk
    mfe_pct = max(high / entry - 1.0, 0.0)
    mfe_capture = gross / max(mfe_pct, 1e-9) if mfe_pct > 0.0 else 0.0
    final_leg = exit_legs[-1]
    return TradeOutcome(
        trade_date=trade_date,
        symbol=symbol,
        entry_time=entry_bar.timestamp,
        entry_price=entry,
        stop_price=stop,
        risk_per_share=risk,
        gross_return_pct=gross,
        net_return_pct=net,
        mfe_r=mfe_r,
        mae_r=mae_r,
        mfe_capture=mfe_capture,
        bars_held=max(1, int(final_leg.bar_index) - signal.fill_index + 1),
        exit_reason=final_leg.reason,
        ambiguous_bar_count=int(stats["ambiguous_bar_count"]),
        stopped=bool(stats["stopped"]),
        target_hit=bool(stats["target_hit"]),
        partial_hit=bool(stats["partial_hit"]),
    )


def find_entry_signal(
    bars: tuple[MarketBar, ...],
    ctx: First30Context,
    spec: EntrySpec,
    cfg: KALCBConfig,
    *,
    prior_day_high: float | None = None,
    candidate_metadata: dict[str, Any] | None = None,
) -> EntrySignal | None:
    first_index = _first_fill_index(bars)
    if first_index is None:
        return None
    if spec.mode == BASELINE_ENTRY_MODE:
        signal_index = max(0, first_index - 1)
        if bars[signal_index].timestamp.astimezone(KST).time() >= FIRST30_END:
            return None
        return EntrySignal(first_index, signal_index, BASELINE_ENTRY_MODE)
    if spec.mode == "opening_drive":
        signal_index = max(0, first_index - 1)
        signal_bar = bars[signal_index]
        if _first30_gate_passes(ctx, signal_bar, spec):
            return EntrySignal(first_index, signal_index, "opening_drive")
        return None
    start = min(len(bars) - 1, first_index + max(0, int(spec.after_bar)))
    stop = min(len(bars) - 2, first_index + max(1, int(spec.max_signal_bars)) - 1)
    touched_vwap = False
    touched_levels: dict[str, bool] = {"or_mid": False, "or_high": False, "pdh": False}
    touched_reclaim_sources: dict[str, bool] = {}
    for signal_index in range(start, stop + 1):
        bar = bars[signal_index]
        prior = bars[signal_index - 1] if signal_index > 0 else None
        vwap = _running_vwap(bars[: signal_index + 1])
        close_location = _close_location(bar)
        or_high = max(float(ctx.intraday.high), 1e-9)
        or_low = max(float(ctx.intraday.low), 1e-9)
        or_mid = or_low + 0.5 * max(or_high - or_low, 1e-9)
        pdh = float(prior_day_high or 0.0)
        reclaim_source, reclaim_level = _resolve_trade_plan_reclaim_level(
            spec,
            avwap=vwap,
            or_high=or_high,
            or_mid=or_mid,
            pdh=pdh,
            candidate_metadata=candidate_metadata,
        )
        if float(bar.low) <= vwap * (1.0 + spec.max_pullback_from_vwap_pct):
            touched_vwap = True
        touched_levels["or_mid"] = touched_levels["or_mid"] or _touched_level(bar, prior, or_mid, spec.max_pullback_from_vwap_pct)
        touched_levels["or_high"] = touched_levels["or_high"] or _touched_level(bar, prior, or_high, spec.max_pullback_from_vwap_pct)
        touched_levels["pdh"] = touched_levels["pdh"] or (pdh > 0 and _touched_level(bar, prior, pdh, spec.max_pullback_from_vwap_pct))
        touched_reclaim = reclaim_level > 0 and _touched_level(bar, prior, reclaim_level, spec.max_pullback_from_vwap_pct)
        if reclaim_source == "session_vwap" and touched_reclaim:
            touched_vwap = True
        elif reclaim_source in {"or_mid", "or_high", "pdh"} and touched_reclaim:
            touched_levels[{"or_mid": "or_mid", "or_high": "or_high", "pdh": "pdh"}[reclaim_source]] = True
        elif reclaim_source != "legacy":
            touch_key = f"{spec.mode}:{reclaim_source}"
            touched_reclaim_sources[touch_key] = bool(touched_reclaim_sources.get(touch_key, False) or touched_reclaim)
        if not _common_entry_bar_passes(bar, ctx, vwap, close_location, spec):
            continue
        source_touched = (
            touched_vwap
            if reclaim_source == "session_vwap"
            else touched_levels[reclaim_source]
            if reclaim_source in touched_levels
            else touched_reclaim_sources.get(f"{spec.mode}:{reclaim_source}", False)
        )
        min_closes = max(1, int(spec.min_reclaim_closes or 1))
        if spec.mode == "confirm_next_bar":
            return _next_fill(signal_index, "confirm_next_bar", len(bars))
        if spec.mode in {"breakout", "or_breakout", "pdh_breakout", "combined_breakout"}:
            if _breakout_passes(spec.mode, bar, ctx, or_high, pdh, spec):
                return _next_fill(signal_index, spec.mode, len(bars))
        elif spec.mode == "post_or_momentum":
            if _or_position(bar, or_high, or_low) >= max(spec.min_or_position, 0.45):
                return _next_fill(signal_index, "post_or_momentum", len(bars))
        elif spec.mode == "avwap_reclaim":
            level = reclaim_level if reclaim_source != "legacy" else vwap
            if (
                source_touched
                and float(bar.close) >= level * (1.0 + max(spec.min_reclaim_ret, -0.05))
                and _reclaim_close_count(bars[: signal_index + 1], level, spec.min_reclaim_ret) >= min_closes
            ):
                return _next_fill(signal_index, "avwap_reclaim", len(bars))
        elif spec.mode == "pullback_acceptance":
            reclaim = float(bar.close) / max(float(bar.open), 1e-9) - 1.0
            level = reclaim_level if reclaim_source != "legacy" else vwap
            if (
                source_touched
                and reclaim >= spec.min_reclaim_ret
                and float(bar.close) >= min(float(bar.open), level)
                and _reclaim_close_count(bars[: signal_index + 1], level, spec.min_reclaim_ret) >= min_closes
            ):
                return _next_fill(signal_index, "pullback_acceptance", len(bars))
        elif spec.mode == "or_mid_reclaim":
            level = reclaim_level if reclaim_level > 0 else or_mid
            if (
                source_touched
                and float(bar.close) >= level * (1.0 + max(spec.min_reclaim_ret, -0.05))
                and _reclaim_close_count(bars[: signal_index + 1], level, spec.min_reclaim_ret) >= min_closes
            ):
                return _next_fill(signal_index, "or_mid_reclaim", len(bars))
        elif spec.mode == "or_high_reclaim":
            level = reclaim_level if reclaim_level > 0 else or_high
            if (
                source_touched
                and float(bar.close) >= level * (1.0 + max(spec.min_reclaim_ret, -0.05))
                and _reclaim_close_count(bars[: signal_index + 1], level, spec.min_reclaim_ret) >= min_closes
            ):
                return _next_fill(signal_index, "or_high_reclaim", len(bars))
        elif spec.mode == "pdh_reclaim":
            level = reclaim_level if reclaim_level > 0 else pdh
            if (
                level > 0
                and source_touched
                and float(bar.close) >= level * (1.0 + max(spec.min_reclaim_ret, -0.05))
                and _reclaim_close_count(bars[: signal_index + 1], level, spec.min_reclaim_ret) >= min_closes
            ):
                return _next_fill(signal_index, "pdh_reclaim", len(bars))
        elif spec.mode == "deferred_continuation":
            prior_high = max((float(item.high) for item in bars[first_index:signal_index]), default=or_high)
            if float(bar.close) >= prior_high * (1.0 + spec.min_breakout_pct):
                return _next_fill(signal_index, "deferred_continuation", len(bars))
    return None


def simulate_exits(
    bars: tuple[MarketBar, ...],
    entry_index: int,
    entry: float,
    initial_stop: float,
    risk: float,
    spec: ExitSpec,
    cfg: KALCBConfig,
) -> tuple[tuple[ExitLeg, ...], dict[str, float | bool]]:
    remaining = 1.0
    active_stop = initial_stop
    pending_stop = initial_stop
    high_water = entry
    partial_done = False
    legs: list[ExitLeg] = []
    ambiguous = 0
    stopped = False
    target_hit = False
    partial_hit = False
    vwap_fail_streak = 0
    for index in range(entry_index, len(bars)):
        bar = bars[index]
        active_targets = []
        if spec.partial_trigger_r > 0 and not partial_done and remaining > 0:
            active_targets.append(entry + spec.partial_trigger_r * risk)
        if spec.target_r > 0 and remaining > 0:
            active_targets.append(entry + spec.target_r * risk)
        if spec.hard_stop_enabled and active_targets and float(bar.low) <= active_stop and float(bar.high) >= min(active_targets):
            ambiguous += 1
        if spec.hard_stop_enabled and float(bar.low) <= active_stop:
            legs.append(ExitLeg(remaining, min(active_stop, float(bar.open)) if float(bar.open) < active_stop else active_stop, "hard_stop", index))
            stopped = True
            remaining = 0.0
            break
        if spec.partial_trigger_r > 0 and not partial_done and remaining > 0 and float(bar.high) >= entry + spec.partial_trigger_r * risk:
            fraction = min(max(spec.partial_fraction, 0.0), remaining)
            if fraction > 0:
                legs.append(ExitLeg(fraction, entry + spec.partial_trigger_r * risk, "partial_target", index))
                remaining -= fraction
                partial_done = True
                partial_hit = True
                pending_stop = max(pending_stop, entry + spec.partial_stop_r * risk)
        if spec.target_r > 0 and remaining > 0 and float(bar.high) >= entry + spec.target_r * risk:
            legs.append(ExitLeg(remaining, entry + spec.target_r * risk, "target", index))
            target_hit = True
            remaining = 0.0
            break
        high_water = max(high_water, float(bar.high))
        next_exit = _completed_bar_exit_reason(bars, index, entry_index, entry, risk, high_water, spec, vwap_fail_streak)
        vwap_fail_streak = int(next_exit.pop("vwap_fail_streak"))
        reason = str(next_exit.get("reason") or "")
        if reason and remaining > 0:
            fill_index = min(index + 1, len(bars) - 1)
            fill_price = float(bars[fill_index].open) if fill_index > index else float(bar.close)
            legs.append(ExitLeg(remaining, fill_price, reason, fill_index))
            remaining = 0.0
            break
        if spec.max_hold_bars > 0 and index - entry_index + 1 >= spec.max_hold_bars and remaining > 0:
            fill_index = min(index + 1, len(bars) - 1)
            fill_price = float(bars[fill_index].open) if fill_index > index else float(bar.close)
            legs.append(ExitLeg(remaining, fill_price, "max_hold", fill_index))
            remaining = 0.0
            break
        if bar.timestamp.astimezone(KST).time() >= cfg.flatten_time and remaining > 0:
            legs.append(ExitLeg(remaining, float(bar.close), EOD_EXIT_REASON, index))
            remaining = 0.0
            break
        pending_stop = max(pending_stop, _next_bar_stop_from_completed_bar(entry, risk, active_stop, high_water, spec))
        active_stop = pending_stop
    if remaining > 0:
        legs.append(ExitLeg(remaining, float(bars[-1].close), EOD_EXIT_REASON, len(bars) - 1))
    return tuple(legs), {
        "ambiguous_bar_count": float(ambiguous),
        "stopped": stopped,
        "target_hit": target_hit,
        "partial_hit": partial_hit,
    }


def initial_stop_price(
    bars: tuple[MarketBar, ...],
    ctx: First30Context,
    signal: EntrySignal,
    entry: float,
    spec: ExitSpec,
    cfg: KALCBConfig,
) -> float:
    signal_bar = bars[max(0, min(signal.signal_index, len(bars) - 1))]
    entry_bar = bars[max(0, min(signal.fill_index, len(bars) - 1))]
    atr = max(float(ctx.daily.atr14), entry * 0.006, 1.0)
    if spec.stop_mode == "first30_low":
        return min(float(ctx.intraday.low), entry - entry * 0.003)
    if spec.stop_mode == "signal_low":
        return min(float(signal_bar.low), entry - entry * 0.003)
    if spec.stop_mode == "entry_low":
        return min(float(entry_bar.low), entry - entry * 0.003)
    if spec.stop_mode == "fixed_pct":
        return entry * (1.0 - max(spec.stop_pct, 0.001))
    if spec.stop_mode == "vwap":
        vwap = _running_vwap(bars[: signal.fill_index + 1])
        return min(vwap * 0.997, entry - entry * 0.003)
    return entry - float(spec.stop_atr_mult or cfg.stop_atr_multiple) * atr


def summarize_outcomes(
    outcomes: list[TradeOutcome],
    *,
    session_dates: tuple[date, ...],
    selection_counts: dict[date, int],
) -> dict[str, float]:
    date_set = set(session_dates)
    outcomes = [outcome for outcome in outcomes if outcome.trade_date in date_set]
    by_day: dict[date, list[TradeOutcome]] = {}
    for outcome in outcomes:
        by_day.setdefault(outcome.trade_date, []).append(outcome)
    daily_net: list[float] = []
    daily_gross: list[float] = []
    selected_day_net: list[float] = []
    active_day_net: list[float] = []
    for day in session_dates:
        denom = max(int(selection_counts.get(day, 0)), 1)
        day_outcomes = by_day.get(day, [])
        net = sum(item.net_return_pct for item in day_outcomes) / denom
        gross = sum(item.gross_return_pct for item in day_outcomes) / denom
        daily_net.append(net)
        daily_gross.append(gross)
        if selection_counts.get(day, 0) > 0:
            selected_day_net.append(net)
        if day_outcomes:
            active_day_net.append(net)
    selected_count = float(sum(selection_counts.get(day, 0) for day in session_dates))
    return {
        "selected_count": selected_count,
        "selected_days": float(sum(1 for day in session_dates if selection_counts.get(day, 0) > 0)),
        "trade_count": float(len(outcomes)),
        "active_days": float(len(by_day)),
        "session_count": float(len(session_dates)),
        "signal_conversion": len(outcomes) / max(selected_count, 1.0),
        "active_day_share": len(by_day) / max(float(len(session_dates)), 1.0),
        "selected_day_share": sum(1 for day in session_dates if selection_counts.get(day, 0) > 0) / max(float(len(session_dates)), 1.0),
        "avg_trades_per_session": len(outcomes) / max(float(len(session_dates)), 1.0),
        "avg_trade_net_pct": _avg(outcome.net_return_pct for outcome in outcomes),
        "active_day_net_pct": _avg(active_day_net),
        "selected_day_net_pct": _avg(selected_day_net),
        "calendar_day_net_pct": _avg(daily_net),
        "calendar_day_gross_pct": _avg(daily_gross),
        "slot_cumulative_net_return_pct": _compound(daily_net),
        "slot_cumulative_gross_return_pct": _compound(daily_gross),
        "max_drawdown_net_pct": _max_drawdown(daily_net),
        "net_win_share": _share(outcome.net_return_pct > 0.0 for outcome in outcomes),
        "avg_mfe_r": _avg(outcome.mfe_r for outcome in outcomes),
        "median_mfe_r": _median(outcome.mfe_r for outcome in outcomes),
        "avg_mae_r": _avg(outcome.mae_r for outcome in outcomes),
        "mae_le_neg_1_share": _share(outcome.mae_r <= -1.0 for outcome in outcomes),
        "mfe_ge_1_share": _share(outcome.mfe_r >= 1.0 for outcome in outcomes),
        "stopout_share": _share(outcome.stopped for outcome in outcomes),
        "target_hit_share": _share(outcome.target_hit for outcome in outcomes),
        "partial_hit_share": _share(outcome.partial_hit for outcome in outcomes),
        "avg_mfe_capture": _avg(outcome.mfe_capture for outcome in outcomes),
        "avg_bars_held": _avg(outcome.bars_held for outcome in outcomes),
        "ambiguous_bar_count": float(sum(outcome.ambiguous_bar_count for outcome in outcomes)),
        "accepted_loser_summary": _accepted_loser_summary(outcomes),
        "mfe_capture_by_frontier_role": _cohort_metrics(outcomes, "frontier_role"),
        "mfe_capture_by_entry_type": _cohort_metrics(outcomes, "entry_type"),
        "per_candidate_metrics": _per_candidate_metrics(outcomes),
        **_exit_reason_metrics(outcomes),
    }


def _accepted_loser_summary(outcomes: list[TradeOutcome]) -> dict[str, float]:
    losers = [item for item in outcomes if item.net_return_pct < 0.0]
    return {
        "count": float(len(losers)),
        "share": float(len(losers)) / max(float(len(outcomes)), 1.0),
        "avg_mae_r": _avg(item.mae_r for item in losers),
        "avg_mfe_r": _avg(item.mfe_r for item in losers),
        "mae_le_neg_1_share": _share(item.mae_r <= -1.0 for item in losers),
    }


def _cohort_metrics(outcomes: list[TradeOutcome], attr: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[TradeOutcome]] = {}
    for outcome in outcomes:
        key = str(getattr(outcome, attr, "") or "unknown")
        grouped.setdefault(key, []).append(outcome)
    return {
        key: {
            "trades": float(len(rows)),
            "avg_net_pct": _avg(item.net_return_pct for item in rows),
            "win_share": _share(item.net_return_pct > 0.0 for item in rows),
            "avg_mfe_capture": _avg(item.mfe_capture for item in rows),
            "mae_le_neg_1_share": _share(item.mae_r <= -1.0 for item in rows),
        }
        for key, rows in sorted(grouped.items())
    }


def _per_candidate_metrics(outcomes: list[TradeOutcome]) -> list[dict[str, float | str]]:
    grouped: dict[str, list[TradeOutcome]] = {}
    for outcome in outcomes:
        grouped.setdefault(outcome.symbol, []).append(outcome)
    rows = [
        {
            "symbol": symbol,
            "trades": float(len(items)),
            "net_pct": sum(item.net_return_pct for item in items),
            "win_share": _share(item.net_return_pct > 0.0 for item in items),
            "avg_mfe_capture": _avg(item.mfe_capture for item in items),
            "mae_le_neg_1_share": _share(item.mae_r <= -1.0 for item in items),
        }
        for symbol, items in grouped.items()
    ]
    rows.sort(key=lambda item: (float(item["net_pct"]), str(item["symbol"])))
    return rows


def _evaluate_specs(
    specs: list[TradePlanSpec],
    selections: list[Selection],
    dataset: KALCBFirst30Dataset,
    context_by_key: dict[tuple[date, str], First30Context],
    cfg: KALCBConfig,
    train_dates: tuple[date, ...],
    folds: list[tuple[date, date]],
    selection_counts: dict[date, int],
    baseline_train: dict[str, float],
    baseline_fold_metrics: tuple[dict[str, Any], ...],
    output_dir: Path,
    stage: str,
    *,
    max_workers: int,
    compiled_replay: CompiledCoreReplay | None = None,
    worker_backend: str = "thread",
) -> list[PlanResult]:
    if not specs:
        return []
    local_compiled = compiled_replay or compile_core_replay(selections, dataset, context_by_key, train_dates, selection_counts, cfg)
    cached = {
        name: _refresh_plan_result(row, baseline_train, baseline_fold_metrics)
        for name, row in _load_stage_result_rows(output_dir, stage).items()
    }
    rows: list[PlanResult] = [cached[spec.name] for spec in specs if spec.name in cached]
    remaining_specs = [spec for spec in specs if spec.name not in cached]
    if rows:
        _write_progress(output_dir, len(rows), len(specs), rows, stage)
    if not remaining_specs:
        rows.sort(key=lambda item: item.spec.name)
        return rows
    if max_workers <= 1:
        for spec in remaining_specs:
            row = _evaluate_spec(spec, selections, dataset, context_by_key, cfg, train_dates, folds, selection_counts, baseline_train, baseline_fold_metrics, local_compiled)
            rows.append(row)
            _record_progress(output_dir, len(rows), len(specs), rows, row, stage)
        rows.sort(key=lambda item: item.spec.name)
        return rows
    worker_count = max(1, min(int(max_workers), 4))
    if compiled_replay is not None and str(worker_backend or "").lower() == "process":
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_process_eval_worker,
            initargs=(
                local_compiled,
                cfg,
                train_dates,
                folds,
                selection_counts,
                baseline_train,
                baseline_fold_metrics,
            ),
        ) as executor:
            futures = {executor.submit(_evaluate_spec_from_process_worker, spec): spec for spec in remaining_specs}
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                _record_progress(output_dir, len(rows), len(specs), rows, row, stage)
        rows.sort(key=lambda item: item.spec.name)
        return rows
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _evaluate_spec,
                spec,
                selections,
                dataset,
                context_by_key,
                cfg,
                train_dates,
                folds,
                selection_counts,
                baseline_train,
                baseline_fold_metrics,
                local_compiled,
            ): spec
            for spec in remaining_specs
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            _record_progress(output_dir, len(rows), len(specs), rows, row, stage)
    rows.sort(key=lambda item: item.spec.name)
    return rows


def _refresh_plan_result(
    row: PlanResult,
    baseline_train: dict[str, float],
    baseline_fold_metrics: tuple[dict[str, Any], ...],
) -> PlanResult:
    metrics = dict(row.train_metrics)
    if PRIMARY_OBJECTIVE_METRIC in metrics and (
        "primary_objective_net_return_pct" not in metrics
        or EQUAL_SLOT_METRIC not in metrics
        or EXPOSURE_NORMALIZED_SLOT_METRIC not in metrics
    ):
        _add_return_divergence_metrics(metrics)
    score, reject = score_plan(metrics, row.fold_metrics, baseline_train)
    promoted = (not reject) and promotion_pass(metrics, row.fold_metrics, baseline_train, baseline_fold_metrics)
    return replace(
        row,
        score=round(0.0 if reject else score, 6),
        rejected=bool(reject),
        reject_reason=reject,
        train_metrics=metrics,
        promotion_pass=promoted,
    )


def _evaluate_spec(
    spec: TradePlanSpec,
    selections: list[Selection],
    dataset: KALCBFirst30Dataset,
    context_by_key: dict[tuple[date, str], First30Context],
    cfg: KALCBConfig,
    train_dates: tuple[date, ...],
    folds: list[tuple[date, date]],
    selection_counts: dict[date, int],
    baseline_train: dict[str, float],
    baseline_fold_metrics: tuple[dict[str, Any], ...],
    compiled_replay: CompiledCoreReplay | None = None,
) -> PlanResult:
    if compiled_replay is not None:
        return _evaluate_spec_from_compiled(
            spec,
            compiled_replay,
            cfg,
            train_dates,
            folds,
            selection_counts,
            baseline_train,
            baseline_fold_metrics,
        )
    else:
        outcomes = collect_trade_outcomes(spec, selections, dataset, context_by_key, cfg, train_dates, compiled_replay=compiled_replay)
        metrics = summarize_outcomes(outcomes, session_dates=train_dates, selection_counts=selection_counts)
    fold_rows = _fold_metrics_from_outcomes(outcomes, dataset, folds, selection_counts)
    score, reject = score_plan(metrics, fold_rows, baseline_train)
    promoted = (not reject) and promotion_pass(metrics, fold_rows, baseline_train, baseline_fold_metrics)
    return PlanResult(
        spec=spec,
        score=round(0.0 if reject else score, 6),
        rejected=bool(reject),
        reject_reason=reject,
        train_metrics=metrics,
        fold_metrics=fold_rows,
        promotion_pass=promoted,
    )


def _evaluate_spec_from_compiled(
    spec: TradePlanSpec,
    compiled_replay: CompiledCoreReplay,
    cfg: KALCBConfig,
    train_dates: tuple[date, ...],
    folds: list[tuple[date, date]],
    selection_counts: dict[date, int],
    baseline_train: dict[str, float],
    baseline_fold_metrics: tuple[dict[str, Any], ...],
) -> PlanResult:
    try:
        outcomes, metrics, replay_digest = _core_outcomes_metrics_digest(spec, compiled_replay, cfg, train_dates, selection_counts, audit=False)
    except ValueError as exc:
        return _invalid_plan_result(spec, train_dates, selection_counts, f"invalid_config: {exc}")
    fold_rows = _fold_metrics_from_outcomes_for_dates(outcomes, train_dates, folds, selection_counts, initial_equity=compiled_replay.initial_equity)
    score, reject = score_plan(metrics, fold_rows, baseline_train)
    promoted = (not reject) and promotion_pass(metrics, fold_rows, baseline_train, baseline_fold_metrics)
    return PlanResult(
        spec=spec,
        score=round(0.0 if reject else score, 6),
        rejected=bool(reject),
        reject_reason=reject,
        train_metrics=metrics,
        fold_metrics=fold_rows,
        promotion_pass=promoted,
        replay_digest=replay_digest,
    )


def _invalid_plan_result(
    spec: TradePlanSpec,
    train_dates: tuple[date, ...],
    selection_counts: dict[date, int],
    reason: str,
) -> PlanResult:
    selected_count = float(sum(int(selection_counts.get(day, 0)) for day in train_dates))
    selected_days = float(sum(1 for day in train_dates if int(selection_counts.get(day, 0)) > 0))
    metrics = {
        "selected_count": selected_count,
        "selected_days": selected_days,
        "trade_count": 0.0,
        "active_days": 0.0,
        "session_count": float(len(train_dates)),
        "signal_conversion": 0.0,
        "active_day_share": 0.0,
        "selected_day_share": selected_days / max(float(len(train_dates)), 1.0),
        "avg_trades_per_session": 0.0,
        "slot_cumulative_net_return_pct": 0.0,
        "slot_cumulative_gross_return_pct": 0.0,
        "equal_slot_net_return_pct": 0.0,
        "equal_slot_gross_return_pct": 0.0,
        "selected_day_net_pct": 0.0,
        "active_day_net_pct": 0.0,
        "calendar_day_net_pct": 0.0,
        "max_drawdown_net_pct": 0.0,
        "broker_net_return_pct": 0.0,
        "primary_objective_net_return_pct": 0.0,
        "portfolio_equivalent_net_return_pct": 0.0,
        "exposure_normalized_slot_net_return_pct": 0.0,
        "portfolio_equivalent_cumulative_net_return_pct": 0.0,
        "exposure_normalized_slot_cumulative_net_return_pct": 0.0,
        "portfolio_equivalent_max_drawdown_pct": 0.0,
        "slot_minus_broker_net_return_pct": 0.0,
        "slot_to_broker_net_return_ratio": 0.0,
        "equal_slot_minus_broker_net_return_pct": 0.0,
        "equal_slot_to_broker_net_return_ratio": 0.0,
        "portfolio_equivalent_minus_broker_net_return_pct": 0.0,
        "exposure_normalized_slot_minus_broker_net_return_pct": 0.0,
        "exposure_normalized_slot_to_broker_ratio": 0.0,
        "broker_max_drawdown_pct": 0.0,
        "official_mtm_net_return_pct": 0.0,
        "final_equity": 0.0,
        "end_open_position_count": 0.0,
        "same_bar_fill_count": 0.0,
    }
    return PlanResult(
        spec=spec,
        score=0.0,
        rejected=True,
        reject_reason=reason,
        train_metrics=metrics,
        fold_metrics=tuple(),
        promotion_pass=False,
        replay_digest={},
    )


def score_plan(metrics: dict[str, float], fold_rows: tuple[dict[str, Any], ...], baseline_train: dict[str, float]) -> tuple[float, str]:
    selected = metrics.get("selected_count", 0.0)
    trades = metrics.get("trade_count", 0.0)
    if selected <= 0:
        return 0.0, "no_selected_candidates"
    if trades < 60:
        return 0.0, f"too_few_trades ({trades:.0f} < 60)"
    if metrics.get("signal_conversion", 0.0) < 0.15:
        return 0.0, f"too_low_conversion ({metrics.get('signal_conversion', 0.0):.3f} < 0.150)"
    portfolio_net = _objective_net_return(metrics)
    baseline_portfolio = _objective_net_return(baseline_train)
    portfolio_dd = abs(_objective_drawdown(metrics))
    dd_ceiling = float(PORTFOLIO_RISK_POLICY["hard_max_drawdown_pct"])
    if portfolio_dd > dd_ceiling:
        return 0.0, f"max_drawdown_ceiling ({portfolio_dd:.3f} > {dd_ceiling:.3f})"
    portfolio_equivalent = float(metrics.get(PORTFOLIO_EQUIVALENT_METRIC, portfolio_net))
    selected_net = float(metrics.get("portfolio_equivalent_calendar_day_net_pct", metrics.get("selected_day_net_pct", 0.0)))
    active_net = float(metrics.get("active_day_net_pct", 0.0))
    fold_portfolio = [float(row["metrics"].get(PORTFOLIO_EQUIVALENT_METRIC, row["metrics"].get(PRIMARY_OBJECTIVE_METRIC, 0.0))) for row in fold_rows]
    worst_fold_portfolio = min(fold_portfolio, default=portfolio_equivalent)
    median_fold_portfolio = median(fold_portfolio) if fold_portfolio else portfolio_equivalent
    portfolio_equiv_dd = abs(float(metrics.get("portfolio_equivalent_max_drawdown_pct", metrics.get("max_drawdown_net_pct", 0.0))))
    score = (
        1000.0 * portfolio_net
        + 80.0 * _relative_improvement(portfolio_net, baseline_portfolio)
        + 180.0 * portfolio_equivalent
        + 30.0 * selected_net
        + 8.0 * active_net
        + 120.0 * worst_fold_portfolio
        + 40.0 * median_fold_portfolio
        + 8.0 * float(metrics.get("avg_mfe_capture", 0.0))
        + 8.0 * float(metrics.get("net_win_share", 0.0))
        + 4.0 * float(metrics.get("mfe_ge_1_share", 0.0))
        + 0.20 * float(metrics.get("broker_expected_total_r", 0.0))
        - 650.0 * portfolio_dd
        - 40.0 * portfolio_equiv_dd
        - 8.0 * float(metrics.get("mae_le_neg_1_share", 0.0))
        - 4.0 * float(metrics.get("ambiguous_bar_count", 0.0)) / max(trades, 1.0)
    )
    if portfolio_net < baseline_portfolio:
        score -= 60.0 * min(1.0, (baseline_portfolio - portfolio_net) / max(abs(baseline_portfolio), 0.01))
    return score, ""


def _objective_net_return(metrics: dict[str, float]) -> float:
    if PRIMARY_OBJECTIVE_METRIC in metrics:
        return float(metrics.get(PRIMARY_OBJECTIVE_METRIC, 0.0))
    return float(metrics.get(LEGACY_SLOT_METRIC, 0.0))


def _objective_drawdown(metrics: dict[str, float]) -> float:
    if "broker_max_drawdown_pct" in metrics:
        return float(metrics.get("broker_max_drawdown_pct", 0.0))
    return abs(float(metrics.get("max_drawdown_net_pct", 0.0)))


def _relative_improvement(value: float, baseline: float) -> float:
    return (float(value) - float(baseline)) / max(abs(float(baseline)), 0.01)


def promotion_pass(
    metrics: dict[str, float],
    fold_rows: tuple[dict[str, Any], ...],
    baseline_train: dict[str, float],
    baseline_fold_metrics: tuple[dict[str, Any], ...],
) -> bool:
    baseline_portfolio = _objective_net_return(baseline_train)
    portfolio_net = _objective_net_return(metrics)
    if _objective_drawdown(metrics) > float(PORTFOLIO_RISK_POLICY["hard_max_drawdown_pct"]):
        return False
    if baseline_portfolio > 0 and portfolio_net < baseline_portfolio * 1.10:
        return False
    if baseline_portfolio <= 0 and portfolio_net < baseline_portfolio + 0.01:
        return False
    if metrics.get("portfolio_equivalent_calendar_day_net_pct", metrics.get("active_day_net_pct", 0.0)) < baseline_train.get("portfolio_equivalent_calendar_day_net_pct", baseline_train.get("active_day_net_pct", 0.0)):
        return False
    if _objective_drawdown(metrics) > _objective_drawdown(baseline_train) + 0.05:
        return False
    fold_portfolio = [float(row["metrics"].get(PORTFOLIO_EQUIVALENT_METRIC, row["metrics"].get(PRIMARY_OBJECTIVE_METRIC, 0.0))) for row in fold_rows]
    if fold_portfolio and min(fold_portfolio) < 0.0:
        return False
    baseline_fold_portfolio = [float(row["metrics"].get(PORTFOLIO_EQUIVALENT_METRIC, row["metrics"].get(PRIMARY_OBJECTIVE_METRIC, 0.0))) for row in baseline_fold_metrics]
    if fold_portfolio and baseline_fold_portfolio and median(fold_portfolio) < median(baseline_fold_portfolio):
        return False
    return True


def _fold_metrics(
    spec: TradePlanSpec,
    selections: list[Selection],
    dataset: KALCBFirst30Dataset,
    context_by_key: dict[tuple[date, str], First30Context],
    cfg: KALCBConfig,
    folds: list[tuple[date, date]],
    selection_counts: dict[date, int],
    *,
    compiled_replay: CompiledCoreReplay | None = None,
) -> tuple[dict[str, Any], ...]:
    outcomes = collect_trade_outcomes(spec, selections, dataset, context_by_key, cfg, dataset.trading_dates, compiled_replay=compiled_replay)
    initial_equity = compiled_replay.initial_equity if compiled_replay is not None else 0.0
    return _fold_metrics_from_outcomes(outcomes, dataset, folds, selection_counts, initial_equity=initial_equity)


def _fold_metrics_from_outcomes(
    outcomes: list[TradeOutcome],
    dataset: KALCBFirst30Dataset,
    folds: list[tuple[date, date]],
    selection_counts: dict[date, int],
    *,
    initial_equity: float = 0.0,
) -> tuple[dict[str, Any], ...]:
    return _fold_metrics_from_outcomes_for_dates(outcomes, dataset.trading_dates, folds, selection_counts, initial_equity=initial_equity)


def _fold_metrics_from_outcomes_for_dates(
    outcomes: list[TradeOutcome],
    session_dates: Iterable[date],
    folds: list[tuple[date, date]],
    selection_counts: dict[date, int],
    *,
    initial_equity: float = 0.0,
) -> tuple[dict[str, Any], ...]:
    all_dates = tuple(session_dates)
    rows = []
    for index, (start, end) in enumerate(folds, start=1):
        dates = tuple(day for day in all_dates if start <= day <= end)
        metrics = summarize_outcomes(outcomes, session_dates=dates, selection_counts=selection_counts)
        if initial_equity > 0:
            _add_portfolio_equivalent_metrics(metrics, outcomes, dates, initial_equity)
            _add_return_divergence_metrics(metrics)
        rows.append({"fold": index, "start": start.isoformat(), "end": end.isoformat(), "metrics": metrics})
    return tuple(rows)


def _completed_bar_exit_reason(
    bars: tuple[MarketBar, ...],
    index: int,
    entry_index: int,
    entry: float,
    risk: float,
    high_water: float,
    spec: ExitSpec,
    vwap_fail_streak: int,
) -> dict[str, Any]:
    bar = bars[index]
    held = index - entry_index + 1
    close_r = (float(bar.close) - entry) / risk
    mfe_r = (high_water - entry) / risk
    if spec.vwap_fail_bars > 0:
        vwap = _running_vwap(bars[: index + 1])
        vwap_fail_streak = vwap_fail_streak + 1 if float(bar.close) < vwap * (1.0 - spec.vwap_fail_pct) else 0
        if vwap_fail_streak >= spec.vwap_fail_bars:
            return {"reason": "vwap_fail", "vwap_fail_streak": vwap_fail_streak}
    if spec.failed_followthrough_bars > 0 and held >= spec.failed_followthrough_bars and mfe_r < spec.failed_followthrough_mfe_r and close_r <= spec.failed_followthrough_close_r:
        return {"reason": "failed_followthrough", "vwap_fail_streak": vwap_fail_streak}
    if spec.no_mfe_bars > 0 and held >= spec.no_mfe_bars and mfe_r < spec.no_mfe_thresh_r:
        return {"reason": "no_mfe_time_stop", "vwap_fail_streak": vwap_fail_streak}
    return {"reason": "", "vwap_fail_streak": vwap_fail_streak}


def _next_bar_stop_from_completed_bar(entry: float, risk: float, current_stop: float, high_water: float, spec: ExitSpec) -> float:
    stop = current_stop
    mfe_r = (high_water - entry) / risk
    if spec.breakeven_trigger_r > 0 and mfe_r >= spec.breakeven_trigger_r:
        stop = max(stop, entry + spec.breakeven_stop_r * risk)
    if spec.trail_start_r > 0 and mfe_r >= spec.trail_start_r:
        stop = max(stop, high_water - spec.trail_gap_r * risk)
    return stop


def _first30_gate_passes(ctx: First30Context, signal_bar: MarketBar, spec: EntrySpec) -> bool:
    vwap_ret = ctx.vwap_ret
    width = max(float(ctx.intraday.high) - float(ctx.intraday.low), 1e-9)
    or_position = (float(ctx.intraday.close) - float(ctx.intraday.low)) / width
    if ctx.first30_ret < spec.min_bar_ret:
        return False
    if vwap_ret < spec.min_vwap_ret:
        return False
    if _close_location(signal_bar) < spec.min_close_location:
        return False
    if or_position < spec.min_or_position:
        return False
    if spec.require_above_prev_close and float(ctx.intraday.close) <= float(ctx.daily.prev_close):
        return False
    if spec.max_avwap_extension_pct < 9 and ctx.intraday.vwap > 0 and (float(ctx.intraday.close) - ctx.intraday.vwap) / ctx.intraday.vwap > spec.max_avwap_extension_pct:
        return False
    return True


def _common_entry_bar_passes(bar: MarketBar, ctx: First30Context, vwap: float, close_location: float, spec: EntrySpec) -> bool:
    if float(bar.close) / max(float(bar.open), 1e-9) - 1.0 < spec.min_bar_ret:
        return False
    if float(bar.close) / max(vwap, 1e-9) - 1.0 < spec.min_vwap_ret:
        return False
    if close_location < spec.min_close_location:
        return False
    if spec.require_above_prev_close and float(bar.close) <= float(ctx.daily.prev_close):
        return False
    if spec.max_avwap_extension_pct < 9 and vwap > 0 and (float(bar.close) - vwap) / vwap > spec.max_avwap_extension_pct:
        return False
    return True


def _breakout_passes(mode: str, bar: MarketBar, ctx: First30Context, or_high: float, pdh: float, spec: EntrySpec) -> bool:
    close = float(bar.close)
    above_or = close >= or_high * (1.0 + spec.min_breakout_pct)
    above_pdh = pdh > 0 and close >= pdh * (1.0 + spec.min_breakout_pct)
    if mode == "or_breakout":
        return above_or
    if mode == "pdh_breakout":
        return above_pdh
    if mode == "combined_breakout":
        return above_or and above_pdh
    return above_or or above_pdh


def _training_only_config(config: dict[str, Any], *, train_only: bool) -> dict[str, Any]:
    out = dict(config)
    out["use_full_available_window"] = False
    if train_only:
        baseline = dict(out.get("baseline") or {})
        if out.get("end"):
            return out
        if baseline.get("holdout_start"):
            start_text = str(baseline["holdout_start"])
            holdout_start = date.fromisoformat(start_text.split("T", 1)[0])
            out["end"] = (holdout_start - timedelta(days=1)).isoformat()
    return out


def _build_contexts(dataset: KALCBFirst30Dataset) -> dict[date, tuple[First30Context, ...]]:
    from .first30_signal_sweep import build_contexts

    return build_contexts(dataset)


def _selection_counts(selections: list[Selection], dates: Iterable[date]) -> dict[date, int]:
    counts = {day: 0 for day in dates}
    for selection in selections:
        counts[selection.trade_date] = counts.get(selection.trade_date, 0) + 1
    return counts


def _selection_stats(
    selections: list[Selection],
    frontier: dict[date, tuple[str, ...]],
    selection_counts: dict[date, int],
    dates: tuple[date, ...],
) -> dict[str, Any]:
    frontier_counts = [len(frontier.get(day, ())) for day in dates]
    counts = [selection_counts.get(day, 0) for day in dates]
    return {
        "selected_count": int(sum(counts)),
        "selected_days": int(sum(1 for value in counts if value > 0)),
        "avg_selected_per_session": _avg(counts),
        "max_selected_per_session": max(counts) if counts else 0,
        "frontier_avg_size": _avg(frontier_counts),
        "frontier_max_size": max(frontier_counts) if frontier_counts else 0,
        "first_selection": selections[0].trade_date.isoformat() if selections else "",
        "last_selection": selections[-1].trade_date.isoformat() if selections else "",
    }


def _prior_day_high(dataset: KALCBFirst30Dataset, symbol: str, trade_date: date) -> float:
    prior = _prior_daily_row(dataset, symbol, trade_date)
    if not prior:
        return 0.0
    return float(prior.get("high") or 0.0)


def _prior_daily_row(dataset: KALCBFirst30Dataset, symbol: str, trade_date: date) -> dict[str, Any] | None:
    rows = [row for row in dataset.daily_by_symbol.get(symbol, []) if _row_date(row) < trade_date]
    if not rows:
        return None
    return max(rows, key=_row_date)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _bounded(value: float, low: float, high: float) -> float:
    return min(max(float(value), float(low)), float(high))


def _frontier_from_payload(payload: dict[str, Any]) -> FrontierSpec:
    values = {key: value for key, value in payload.items() if key in FrontierSpec.__dataclass_fields__}
    return name_frontier(FrontierSpec(**values))


def _first30_from_payload(payload: dict[str, Any]) -> First30Spec:
    values = {key: value for key, value in payload.items() if key in First30Spec.__dataclass_fields__}
    return First30Spec(**values)


def _resolve_folds(dates: tuple[date, ...], fold_count: int) -> list[tuple[date, date]]:
    count = max(0, int(fold_count))
    if count <= 0 or len(dates) < count:
        return []
    folds: list[tuple[date, date]] = []
    for index in range(count):
        start = round(index * len(dates) / count)
        end = round((index + 1) * len(dates) / count)
        chunk = dates[start:end]
        if chunk:
            folds.append((chunk[0], chunk[-1]))
    return folds


def _first_fill_index(bars: tuple[MarketBar, ...]) -> int | None:
    for index, bar in enumerate(bars):
        if bar.timestamp.astimezone(KST).time() >= FIRST30_END:
            return index
    return None


def _next_fill(signal_index: int, reason: str, bar_count: int) -> EntrySignal | None:
    fill_index = signal_index + 1
    if fill_index >= bar_count:
        return None
    return EntrySignal(fill_index, signal_index, reason)


def _legacy_reclaim_level_source(mode: str) -> str:
    return {
        "avwap_reclaim": "session_vwap",
        "pullback_acceptance": "session_vwap",
        "or_mid_reclaim": "or_mid",
        "or_high_reclaim": "or_high",
        "pdh_reclaim": "pdh",
    }.get(str(mode), "legacy")


def _metadata_level(candidate_metadata: dict[str, Any] | None, key: str) -> float:
    metadata = dict(candidate_metadata or {})
    value = metadata.get(key)
    if value not in (None, ""):
        return _float(value)
    campaign = metadata.get("structural_campaign")
    if isinstance(campaign, dict) and campaign.get(key) not in (None, ""):
        return _float(campaign.get(key))
    return 0.0


def _resolve_trade_plan_reclaim_level(
    spec: EntrySpec,
    *,
    avwap: float,
    or_high: float,
    or_mid: float,
    pdh: float,
    candidate_metadata: dict[str, Any] | None,
) -> tuple[str, float]:
    source = str(spec.reclaim_level_source or "legacy")
    if source == "legacy":
        source = _legacy_reclaim_level_source(spec.mode)
    levels = {
        "session_vwap": float(avwap or 0.0),
        "or_high": float(or_high or 0.0),
        "or_mid": float(or_mid or 0.0),
        "pdh": float(pdh or 0.0),
        "campaign_avwap": _metadata_level(candidate_metadata, "campaign_avwap"),
        "campaign_box_high": _metadata_level(candidate_metadata, "campaign_box_high"),
        "campaign_box_mid": _metadata_level(candidate_metadata, "campaign_box_mid"),
        "campaign_breakout_level": _metadata_level(candidate_metadata, "campaign_breakout_level"),
    }
    return source, float(levels.get(source) or 0.0)


def _running_vwap(bars: tuple[MarketBar, ...]) -> float:
    volume = sum(max(float(bar.volume), 0.0) for bar in bars)
    if volume <= 0:
        return max(float(bars[-1].close), 1e-9) if bars else 1e-9
    value = sum(((float(bar.high) + float(bar.low) + float(bar.close)) / 3.0) * max(float(bar.volume), 0.0) for bar in bars)
    return max(value / volume, 1e-9)


def _close_location(bar: MarketBar) -> float:
    width = max(float(bar.high) - float(bar.low), 1e-9)
    return (float(bar.close) - float(bar.low)) / width


def _or_position(bar: MarketBar, or_high: float, or_low: float) -> float:
    return (float(bar.close) - or_low) / max(or_high - or_low, 1e-9)


def _touched_level(bar: MarketBar, prior: MarketBar | None, level: float, tolerance_pct: float) -> bool:
    if level <= 0:
        return False
    if float(bar.low) <= level * (1.0 + tolerance_pct):
        return True
    return prior is not None and float(prior.close) <= level * (1.0 + tolerance_pct)


def _reclaim_close_count(bars: tuple[MarketBar, ...], level: float, min_close_ret: float = 0.0) -> int:
    if level <= 0.0:
        return 0
    threshold = level * (1.0 + max(float(min_close_ret or 0.0), 0.0))
    count = 0
    for bar in reversed(bars):
        if float(bar.close) < threshold:
            break
        count += 1
    return count


def _exit_reason_metrics(outcomes: list[TradeOutcome]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for outcome in outcomes:
        counts[outcome.exit_reason] = counts.get(outcome.exit_reason, 0) + 1
    return {f"exit_reason_{key}_share": value / max(float(len(outcomes)), 1.0) for key, value in sorted(counts.items())}


def _refined_entry_specs(seed: EntrySpec) -> list[EntrySpec]:
    specs = [seed]
    for bars in _near_int(seed.max_signal_bars, (1, 2, 4), 1, 30):
        for min_ret in _near_float(seed.min_bar_ret, (-0.002, -0.001, 0.0, 0.001, 0.002, 0.004), 0.001, -0.01, 0.02):
            for min_vwap in _near_float(seed.min_vwap_ret, (-0.003, -0.001, 0.0, 0.001, 0.002), 0.001, -0.01, 0.02):
                specs.append(replace(seed, name="", max_signal_bars=bars, min_bar_ret=min_ret, min_vwap_ret=min_vwap))
    for breakout in _near_float(seed.min_breakout_pct, (0.0, 0.0005, 0.001, 0.002, 0.004), 0.0005, 0.0, 0.02):
        for close_loc in _near_float(seed.min_close_location, (0.0, 0.45, 0.50, 0.60, 0.70, 0.80), 0.05, 0.0, 0.95):
            specs.append(replace(seed, name="", min_breakout_pct=breakout, min_close_location=close_loc))
    for pullback in _near_float(seed.max_pullback_from_vwap_pct, (0.002, 0.004, 0.008, 0.012, 0.016), 0.002, 0.0, 0.04):
        for reclaim in _near_float(seed.min_reclaim_ret, (-0.001, 0.0, 0.001, 0.002), 0.0005, -0.01, 0.02):
            specs.append(replace(seed, name="", max_pullback_from_vwap_pct=pullback, min_reclaim_ret=reclaim))
    return _dedupe_entries(_name_entry(spec) for spec in specs)


def _refined_exit_specs(seed: ExitSpec) -> list[ExitSpec]:
    specs = [seed]
    stop_modes = _ordered_unique((seed.stop_mode, "first30_low", "signal_low", "entry_low", "atr", "fixed_pct", "vwap"))
    hard_values = _ordered_unique((seed.hard_stop_enabled, True, False))
    for stop_mode in stop_modes:
        for hard_stop in hard_values:
            for atr in _near_float(seed.stop_atr_mult, (0.30, 0.50, 0.65, 0.80, 1.00), 0.10, 0.10, 1.50)[:4]:
                for stop_pct in _near_float(seed.stop_pct, (0.003, 0.005, 0.006, 0.008, 0.012), 0.002, 0.001, 0.04)[:4]:
                    specs.append(replace(seed, name="", stop_mode=stop_mode, hard_stop_enabled=hard_stop, stop_atr_mult=atr, stop_pct=stop_pct))
    for target in _near_float(seed.target_r, (0.0, 0.35, 0.50, 0.75, 1.0, 1.25, 1.5, 2.0), 0.15, 0.0, 3.0):
        specs.append(replace(seed, name="", target_r=target))
    for trigger in _near_float(seed.partial_trigger_r or 0.50, (0.25, 0.35, 0.50, 0.75, 1.0), 0.10, 0.0, 2.0):
        for fraction in (0.33, 0.50, 0.67):
            for stop_r in _near_float(seed.partial_stop_r, (0.0, 0.10, 0.20, 0.35), 0.05, 0.0, 1.0)[:4]:
                specs.append(replace(seed, name="", partial_trigger_r=trigger, partial_fraction=fraction, partial_stop_r=stop_r, target_r=seed.target_r))
    for start in _near_float(seed.trail_start_r or 0.75, (0.35, 0.50, 0.75, 1.0, 1.25), 0.10, 0.0, 3.0):
        for gap in _near_float(seed.trail_gap_r or 0.45, (0.25, 0.35, 0.45, 0.60, 0.80), 0.10, 0.05, 2.0):
            specs.append(replace(seed, name="", trail_start_r=start, trail_gap_r=gap, target_r=0.0))
    for bars in _ordered_unique((seed.max_hold_bars, 4, 6, 8, 10, 12, 16, 24, 36)):
        if int(bars) > 0:
            specs.append(replace(seed, name="", max_hold_bars=int(bars)))
    return _dedupe_exits(_name_exit(spec) for spec in specs)


def name_plan(spec: TradePlanSpec) -> TradePlanSpec:
    entry = _name_entry(spec.entry)
    exit_spec = _name_exit(spec.exit)
    return TradePlanSpec(name=f"{entry.name}__{exit_spec.name}", entry=entry, exit=exit_spec)


def _name_entry(spec: EntrySpec) -> EntrySpec:
    parts = [
        spec.mode,
        f"sig{spec.max_signal_bars}",
        f"br{_label_pct(spec.min_bar_ret)}",
        f"vw{_label_pct(spec.min_vwap_ret)}",
        f"bo{_label_pct(spec.min_breakout_pct)}",
        f"pb{_label_pct(spec.max_pullback_from_vwap_pct)}",
        f"rec{_label_pct(spec.min_reclaim_ret)}",
        f"rc{int(spec.min_reclaim_closes)}",
        f"cl{_label_num(spec.min_close_location)}",
        f"or{_label_num(spec.min_or_position)}",
        f"cap{_label_pct(spec.max_avwap_extension_pct)}",
        f"a{spec.after_bar}",
    ]
    if spec.require_above_prev_close:
        parts.append("abovePrev")
    return replace(spec, name="_".join(parts).replace("-", "m").replace(".", "p"))


def _name_exit(spec: ExitSpec) -> ExitSpec:
    spec = _canonical_exit_spec(spec)
    parts = [
        spec.stop_mode,
        "stop" if spec.hard_stop_enabled else "nostop",
        f"atr{_label_num(spec.stop_atr_mult)}",
        f"pct{_label_pct(spec.stop_pct)}",
        f"t{_label_num(spec.target_r)}",
        f"pt{_label_num(spec.partial_trigger_r)}x{_label_num(spec.partial_fraction)}",
        f"ps{_label_num(spec.partial_stop_r)}",
        f"be{_label_num(spec.breakeven_trigger_r)}to{_label_num(spec.breakeven_stop_r)}",
        f"tr{_label_num(spec.trail_start_r)}g{_label_num(spec.trail_gap_r)}",
        f"vf{spec.vwap_fail_bars}",
        f"ff{spec.failed_followthrough_bars}",
        f"nm{spec.no_mfe_bars}",
        f"mh{spec.max_hold_bars}",
    ]
    return replace(spec, name="_".join(parts).replace("-", "m").replace(".", "p"))


def _canonical_exit_spec(spec: ExitSpec) -> ExitSpec:
    """Collapse only parameters that cannot change shared-core behavior.

    Stop fields are intentionally preserved even when hard stops are disabled:
    the shared KALCB core still uses the stop anchor for risk-per-share and
    quantity sizing, so e.g. fixed_pct_nostop and atr_nostop are distinct
    executable trade plans.
    """
    updates: dict[str, Any] = {"name": ""}
    if spec.target_r <= 0:
        updates["target_r"] = 0.0
    if spec.partial_trigger_r <= 0 or spec.partial_fraction <= 0:
        updates.update({"partial_trigger_r": 0.0, "partial_fraction": 0.0, "partial_stop_r": 0.0})
    if spec.breakeven_trigger_r <= 0:
        updates.update({"breakeven_trigger_r": 0.0, "breakeven_stop_r": 0.0})
    if spec.trail_start_r <= 0 or spec.trail_gap_r <= 0:
        updates.update({"trail_start_r": 0.0, "trail_gap_r": 0.0})
    if spec.no_mfe_bars <= 0:
        updates.update({"no_mfe_bars": 0, "no_mfe_thresh_r": 0.0})
    if spec.failed_followthrough_bars <= 0:
        updates.update(
            {
                "failed_followthrough_bars": 0,
                "failed_followthrough_mfe_r": 0.0,
                "failed_followthrough_close_r": 0.0,
            }
        )
    if spec.vwap_fail_bars <= 0:
        updates.update({"vwap_fail_bars": 0, "vwap_fail_pct": 0.0})
    if spec.max_hold_bars <= 0:
        updates["max_hold_bars"] = 0
    return replace(spec, **updates)


def _dedupe_entries(specs: Iterable[EntrySpec]) -> list[EntrySpec]:
    out: list[EntrySpec] = []
    seen: set[str] = set()
    for spec in specs:
        named = _name_entry(spec)
        data = asdict(named)
        data.pop("name", None)
        key = json.dumps(data, sort_keys=True, separators=(",", ":"))
        if key in seen:
            continue
        seen.add(key)
        out.append(named)
    return out


def _dedupe_exits(specs: Iterable[ExitSpec]) -> list[ExitSpec]:
    out: list[ExitSpec] = []
    seen: set[str] = set()
    for spec in specs:
        named = _name_exit(spec)
        data = asdict(named)
        data.pop("name", None)
        key = json.dumps(data, sort_keys=True, separators=(",", ":"))
        if key in seen:
            continue
        seen.add(key)
        out.append(named)
    return out


def _limit_specs(items: list[Any], limit: int) -> list[Any]:
    count = int(limit or 0)
    if count <= 0 or count >= len(items):
        return list(items)
    return _even_sample(items, count)


def _even_sample(items: list[Any], count: int) -> list[Any]:
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[0]]
    selected = []
    used: set[int] = set()
    last = len(items) - 1
    for index in range(count):
        position = round(index * last / (count - 1))
        while position in used and position < last:
            position += 1
        while position in used and position > 0:
            position -= 1
        used.add(position)
        selected.append(items[position])
    return selected


def _unique_entries(rows: list[PlanResult], count: int) -> list[EntrySpec]:
    out: list[EntrySpec] = []
    seen: set[str] = set()
    for row in sorted(rows, key=_plan_sort_key):
        data = asdict(row.spec.entry)
        data.pop("name", None)
        key = json.dumps(data, sort_keys=True, separators=(",", ":"))
        if key in seen:
            continue
        seen.add(key)
        out.append(row.spec.entry)
        if len(out) >= max(1, int(count)):
            break
    return out


def _select_exit_entry_seed_rows(rows: list[PlanResult], count: int) -> list[PlanResult]:
    """Pick broad but viable entry families for expensive exit-management sweeps.

    Exit and trade-management specs cannot create more entries, so spending the
    full exit catalog on entry variants that already fail conversion/trade-count
    gates is pure compute burn.  Keep the wide entry-only pass, then promote a
    compact, deterministic set of viable families for deeper management work.
    """

    target = max(1, int(count))
    ranked = sorted(rows, key=_plan_sort_key)
    selected: list[PlanResult] = []
    seen: set[str] = set()
    family_counts: dict[str, int] = {}

    def add(row: PlanResult) -> bool:
        key = _entry_spec_key(row.spec.entry)
        if key in seen:
            return False
        seen.add(key)
        selected.append(row)
        family = _entry_family(row.spec.entry)
        family_counts[family] = family_counts.get(family, 0) + 1
        return True

    for row in ranked:
        if row.spec.entry.mode == BASELINE_ENTRY_MODE:
            add(row)
            break

    for family in _ENTRY_SEED_FAMILY_ORDER:
        if len(selected) >= target:
            break
        if family_counts.get(family, 0) > 0:
            continue
        for row in ranked:
            if _entry_family(row.spec.entry) == family and _entry_seed_viable(row):
                add(row)
                break

    for row in ranked:
        if len(selected) >= target:
            break
        family = _entry_family(row.spec.entry)
        if family_counts.get(family, 0) >= _entry_family_cap(family, target):
            continue
        if _entry_seed_viable(row):
            add(row)

    for row in ranked:
        if len(selected) >= target:
            break
        add(row)
    return selected


_ENTRY_SEED_FAMILY_ORDER = (
    "first30_open",
    "opening_drive",
    "post_or_momentum",
    "breakout",
    "reclaim",
    "deferred_continuation",
)


def _entry_seed_viable(row: PlanResult) -> bool:
    metrics = row.train_metrics
    return (
        not row.rejected
        and float(metrics.get("trade_count", 0.0)) >= 60.0
        and float(metrics.get("signal_conversion", 0.0)) >= 0.15
    )


def _entry_family(entry: EntrySpec) -> str:
    if entry.mode == BASELINE_ENTRY_MODE:
        return "first30_open"
    if entry.mode in {"confirm_next_bar", "post_or_momentum"}:
        return "post_or_momentum"
    if entry.mode in {"breakout", "or_breakout", "pdh_breakout", "combined_breakout"}:
        return "breakout"
    if entry.mode in {"avwap_reclaim", "pullback_acceptance", "or_mid_reclaim", "or_high_reclaim", "pdh_reclaim"}:
        return "reclaim"
    return entry.mode


def _entry_family_cap(family: str, target: int) -> int:
    if family == "first30_open":
        return 1
    if family == "opening_drive":
        return max(2, min(5, target - 1))
    if family == "post_or_momentum":
        return max(1, min(3, target // 3 + 1))
    return max(1, min(2, target // 4 + 1))


def _entry_spec_key(entry: EntrySpec) -> str:
    data = asdict(entry)
    data.pop("name", None)
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _entry_seed_payload(row: PlanResult) -> dict[str, Any]:
    payload = _progress_row(row)
    payload["entry_family"] = _entry_family(row.spec.entry)
    payload["entry"] = asdict(row.spec.entry)
    return payload


def _merge_results(*groups: list[PlanResult]) -> list[PlanResult]:
    out: dict[str, PlanResult] = {}
    for group in groups:
        for row in group:
            current = out.get(row.spec.name)
            if current is None or _plan_sort_key(row) < _plan_sort_key(current):
                out[row.spec.name] = row
    return list(out.values())


def _plan_sort_key(row: PlanResult) -> tuple[Any, ...]:
    metrics = row.train_metrics
    worst_fold = min(
        (float(fold["metrics"].get(PORTFOLIO_EQUIVALENT_METRIC, fold["metrics"].get(PRIMARY_OBJECTIVE_METRIC, 0.0))) for fold in row.fold_metrics),
        default=metrics.get(PORTFOLIO_EQUIVALENT_METRIC, metrics.get(PRIMARY_OBJECTIVE_METRIC, 0.0)),
    )
    return (
        row.rejected,
        -_objective_net_return(metrics),
        -float(metrics.get("broker_expected_total_r", 0.0)),
        _objective_drawdown(metrics),
        -float(metrics.get(PORTFOLIO_EQUIVALENT_METRIC, 0.0)),
        -float(metrics.get("portfolio_equivalent_calendar_day_net_pct", metrics.get("selected_day_net_pct", 0.0))),
        -worst_fold,
        abs(float(metrics.get("portfolio_equivalent_max_drawdown_pct", metrics.get("max_drawdown_net_pct", 0.0)))),
        -float(metrics.get("avg_mfe_capture", 0.0)),
        row.spec.name,
    )


def _row_payload(row: PlanResult) -> dict[str, Any]:
    return {
        "name": row.spec.name,
        "score": row.score,
        "rejected": row.rejected,
        "reject_reason": row.reject_reason,
        "promotion_pass": row.promotion_pass,
        "spec": _spec_payload(row.spec),
        "train_metrics": row.train_metrics,
        "fold_metrics": row.fold_metrics,
        "fast_replay_digest": row.replay_digest,
    }


def _row_from_payload(payload: dict[str, Any]) -> PlanResult:
    spec_payload = dict(payload.get("spec") or {})
    entry_payload = dict(spec_payload.get("entry") or {})
    exit_payload = dict(spec_payload.get("exit") or {})
    spec = TradePlanSpec(
        name=str(spec_payload.get("name") or payload.get("name") or ""),
        entry=EntrySpec(**entry_payload),
        exit=ExitSpec(**exit_payload),
    )
    return PlanResult(
        spec=spec,
        score=float(payload.get("score", 0.0) or 0.0),
        rejected=bool(payload.get("rejected", False)),
        reject_reason=str(payload.get("reject_reason", "") or ""),
        train_metrics=dict(payload.get("train_metrics") or {}),
        fold_metrics=tuple(dict(row) for row in (payload.get("fold_metrics") or ())),
        promotion_pass=bool(payload.get("promotion_pass", False)),
        replay_digest=dict(payload.get("fast_replay_digest") or {}),
    )


def _load_focused_seed_rows(
    path: str | Path,
    *,
    limit: int = 0,
    names: tuple[str, ...] = (),
) -> list[PlanResult]:
    source = Path(path)
    rows: list[PlanResult] = []
    if source.suffix.lower() == ".jsonl":
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                rows.append(_row_from_payload(json.loads(text)))
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            raw_rows = []
            for key in ("rows", "top_promoted", "top_train"):
                raw_rows.extend(payload.get(key) or [])
            rows = [_row_from_payload(dict(row)) for row in raw_rows]
        elif isinstance(payload, list):
            rows = [_row_from_payload(dict(row)) for row in payload]
    if names:
        wanted = set(names)
        rows = [row for row in rows if row.spec.name in wanted]
    rows.sort(key=_plan_sort_key)
    distinct: list[PlanResult] = []
    seen: set[tuple[float, float, float, float, float]] = set()
    for row in rows:
        key = _outcome_group_key(row)
        if key in seen:
            continue
        seen.add(key)
        distinct.append(row)
        if limit and len(distinct) >= max(1, int(limit)):
            break
    return distinct


def _outcome_group_key(row: PlanResult) -> tuple[float, float, float, float, float]:
    metrics = row.train_metrics
    return (
        round(_objective_net_return(metrics), 10),
        round(float(metrics.get(EXPOSURE_NORMALIZED_SLOT_METRIC, metrics.get(PORTFOLIO_EQUIVALENT_METRIC, 0.0))), 10),
        round(float(metrics.get(EQUAL_SLOT_METRIC, metrics.get(LEGACY_SLOT_METRIC, 0.0))), 10),
        round(float(metrics.get("trade_count", 0.0)), 6),
        round(abs(_objective_drawdown(metrics)), 10),
    )


def _spec_payload(spec: TradePlanSpec) -> dict[str, Any]:
    return {"name": spec.name, "entry": asdict(spec.entry), "exit": asdict(spec.exit)}


def _near_int(value: int, deltas: tuple[int, ...], floor: int, ceiling: int) -> list[int]:
    return [int(item) for item in _ordered_unique(max(floor, min(ceiling, int(value) + int(delta))) for delta in (0, *deltas, *(-d for d in deltas)))]


def _near_float(value: float, anchors: tuple[float, ...], delta: float, floor: float, ceiling: float) -> list[float]:
    raw = [float(value), *anchors, float(value) - delta, float(value) + delta]
    return [float(item) for item in _ordered_unique(round(max(floor, min(ceiling, float(item))), 6) for item in raw if math.isfinite(float(item)))]


def _ordered_unique(values: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = json.dumps(value, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _avg(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return float(mean(items)) if items else 0.0


def _median(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    return float(median(items)) if items else 0.0


def _share(values: Iterable[bool]) -> float:
    items = [bool(value) for value in values]
    return sum(1 for value in items if value) / max(float(len(items)), 1.0)


def _compound(values: list[float]) -> float:
    equity = 1.0
    for value in values:
        equity *= 1.0 + float(value)
    return equity - 1.0


def _max_drawdown(values: list[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for value in values:
        equity *= 1.0 + float(value)
        peak = max(peak, equity)
        if peak > 0:
            max_dd = min(max_dd, equity / peak - 1.0)
    return max_dd


def _label_pct(value: float) -> str:
    if abs(float(value)) >= 9.0:
        return "na"
    return str(int(round(float(value) * 1000.0))).replace("-", "m")


def _label_num(value: float) -> str:
    if abs(float(value)) >= 9.0:
        return "na"
    return f"{float(value):.3g}".replace("-", "m").replace(".", "p")


def _record_progress(output_dir: Path, completed: int, total: int, rows: list[PlanResult], row: PlanResult, stage: str) -> None:
    _append_stage_result_row(output_dir, stage, row)
    if completed not in {1, 2, 3, 5, 10, total} and completed % 100 != 0:
        return
    _write_progress(output_dir, completed, total, rows, stage)
    event = {"updated_at": _utc_now_iso(), "stage": stage, "completed": completed, "total": total, "row": _progress_row(row)}
    with (output_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")
    print(
        "[kalcb-trade-plan-sweep] "
        f"{stage} {completed}/{total} {row.spec.name} "
        f"broker={100.0 * row.train_metrics.get(PRIMARY_OBJECTIVE_METRIC, 0.0):.2f}% "
        f"exp_slot={100.0 * row.train_metrics.get(EXPOSURE_NORMALIZED_SLOT_METRIC, row.train_metrics.get(PORTFOLIO_EQUIVALENT_METRIC, 0.0)):.2f}% "
        f"equal_slot={100.0 * row.train_metrics.get(EQUAL_SLOT_METRIC, row.train_metrics.get(LEGACY_SLOT_METRIC, 0.0)):.1f}% "
        f"net/day={100.0 * row.train_metrics.get('selected_day_net_pct', 0.0):.3f}% "
        f"trades={row.train_metrics.get('trade_count', 0.0):.0f} reject={row.reject_reason}",
        flush=True,
    )


def _stage_rows_path(output_dir: Path, stage: str) -> Path:
    safe_stage = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in stage)
    return output_dir / f"rows_{safe_stage}.jsonl"


def _append_stage_result_row(output_dir: Path, stage: str, row: PlanResult) -> None:
    with _stage_rows_path(output_dir, stage).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_row_payload(row), sort_keys=True, default=str) + "\n")


def _load_stage_result_rows(output_dir: Path, stage: str) -> dict[str, PlanResult]:
    path = _stage_rows_path(output_dir, stage)
    if not path.exists():
        return {}
    rows: dict[str, PlanResult] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                row = _row_from_payload(json.loads(text))
            except Exception:
                continue
            rows[row.spec.name] = row
    return rows


def _clear_progress_files(output_dir: Path) -> None:
    for name in ("progress.jsonl", "run_status.json"):
        try:
            (output_dir / name).unlink()
        except FileNotFoundError:
            pass
    for path in output_dir.glob("progress_*.json"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _write_run_status(output_dir: Path, stage: str, **extra: Any) -> None:
    payload = {"updated_at": _utc_now_iso(), "stage": stage, **extra}
    path = output_dir / "run_status.json"
    tmp = output_dir / "run_status.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)
    with (output_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": "status", **payload}, sort_keys=True, default=str) + "\n")
    details = " ".join(f"{key}={value}" for key, value in extra.items())
    print(f"[kalcb-trade-plan-sweep] status {stage} {details}".rstrip(), flush=True)


def _start_status_heartbeat(output_dir: Path, stage: str, *, interval_seconds: int = 45):
    stop = threading.Event()

    def loop() -> None:
        beat = 0
        while not stop.wait(interval_seconds):
            beat += 1
            _write_run_status(output_dir, stage, heartbeat=beat)

    thread = threading.Thread(target=loop, name=f"kalcb-{stage}-heartbeat", daemon=True)
    thread.start()

    def stop_heartbeat() -> None:
        stop.set()
        thread.join(timeout=2.0)

    return stop_heartbeat


def _write_progress(output_dir: Path, completed: int, total: int, rows: list[PlanResult], stage: str) -> None:
    ranked = sorted(rows, key=_plan_sort_key)
    payload = {
        "updated_at": _utc_now_iso(),
        "stage": stage,
        "completed": int(completed),
        "total": int(total),
        "percent": round(100.0 * completed / total, 3) if total else 100.0,
        "best_train_so_far": _progress_row(ranked[0]) if ranked else None,
        "top_train": [_progress_row(row) for row in ranked[:15]],
    }
    path = output_dir / f"progress_{stage}.json"
    tmp = output_dir / f"progress_{stage}.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def _progress_row(row: PlanResult) -> dict[str, Any]:
    return {
        "name": row.spec.name,
        "score": row.score,
        "rejected": row.rejected,
        "reject_reason": row.reject_reason,
        "promotion_pass": row.promotion_pass,
        "broker_net_return_pct": row.train_metrics.get(PRIMARY_OBJECTIVE_METRIC, 0.0),
        "broker_max_drawdown_pct": row.train_metrics.get("broker_max_drawdown_pct", 0.0),
        "portfolio_equivalent_net_return_pct": row.train_metrics.get(PORTFOLIO_EQUIVALENT_METRIC, 0.0),
        "exposure_normalized_slot_net_return_pct": row.train_metrics.get(EXPOSURE_NORMALIZED_SLOT_METRIC, row.train_metrics.get(PORTFOLIO_EQUIVALENT_METRIC, 0.0)),
        "portfolio_equivalent_max_drawdown_pct": row.train_metrics.get("portfolio_equivalent_max_drawdown_pct", 0.0),
        "slot_cumulative_net_return_pct": row.train_metrics.get("slot_cumulative_net_return_pct", 0.0),
        "equal_slot_net_return_pct": row.train_metrics.get(EQUAL_SLOT_METRIC, row.train_metrics.get("slot_cumulative_net_return_pct", 0.0)),
        "slot_minus_broker_net_return_pct": row.train_metrics.get("slot_minus_broker_net_return_pct", 0.0),
        "slot_to_broker_net_return_ratio": row.train_metrics.get("slot_to_broker_net_return_ratio", 0.0),
        "equal_slot_to_broker_net_return_ratio": row.train_metrics.get("equal_slot_to_broker_net_return_ratio", row.train_metrics.get("slot_to_broker_net_return_ratio", 0.0)),
        "exposure_normalized_slot_to_broker_ratio": row.train_metrics.get("exposure_normalized_slot_to_broker_ratio", 0.0),
        "selected_day_net_pct": row.train_metrics.get("selected_day_net_pct", 0.0),
        "trade_count": row.train_metrics.get("trade_count", 0.0),
        "max_drawdown_net_pct": row.train_metrics.get("max_drawdown_net_pct", 0.0),
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# KALCB Fixed-Candidate Trade Plan Sweep",
        "",
        f"Sweep hash: `{payload['sweep_hash']}`",
        f"Train: {payload['training_window']['start']} to {payload['training_window']['end']} ({payload['training_window']['sessions']} sessions)",
        f"Candidate source: `{payload['fixed_candidate_source']['source_row_name']}`",
        f"Candidates: {payload['sweep_counts']['candidate_count']} ({payload['sweep_counts']['entry_coarse']} entry, {payload['sweep_counts']['exit_coarse']} exit, {payload['sweep_counts']['deep_refine']} refine)",
        f"Promotion result: `{payload['promotion_policy']['result']}`",
        f"Fast/full audit: `{payload.get('fast_suppression_audit', {}).get('status', 'not_run')}` "
        f"(max metric delta {payload.get('fast_suppression_audit', {}).get('max_abs_metric_delta', 0.0):.3g})",
        "",
        "Metric contract: `broker_net_return_pct` is the official executable objective. "
        "`exposure_normalized_slot_net_return_pct` is the broker-comparable slot proxy. "
        "`equal_slot_net_return_pct` / legacy `slot_cumulative_net_return_pct` is a costed equal-opportunity diagnostic, not exposure-normalized.",
        "",
        "## Baseline",
        "",
        _metrics_line(payload["baseline"]["train_metrics"], "Train"),
        "",
        "## Top Train",
        "",
        "| Rank | Promote | Broker Net | Exposure-Norm Net | Equal-Slot Net | Exp/Broker | Equal/Broker | Selected-Day Net | Active-Day Net | Trades | MFE Capture | Broker DD | Spec |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for index, row in enumerate(payload.get("top_train", [])[:30], start=1):
        metrics = row["train_metrics"]
        lines.append(
            f"| {index} | {int(bool(row.get('promotion_pass')))} | "
            f"{100.0 * metrics.get(PRIMARY_OBJECTIVE_METRIC, 0.0):.2f}% | "
            f"{100.0 * metrics.get(EXPOSURE_NORMALIZED_SLOT_METRIC, metrics.get(PORTFOLIO_EQUIVALENT_METRIC, 0.0)):.2f}% | "
            f"{100.0 * metrics.get(EQUAL_SLOT_METRIC, metrics.get(LEGACY_SLOT_METRIC, 0.0)):.1f}% | "
            f"{metrics.get('exposure_normalized_slot_to_broker_ratio', 0.0):.3f} | "
            f"{metrics.get('equal_slot_to_broker_net_return_ratio', metrics.get('slot_to_broker_net_return_ratio', 0.0)):.3f} | "
            f"{100.0 * metrics.get('selected_day_net_pct', 0.0):.3f}% | "
            f"{100.0 * metrics.get('active_day_net_pct', 0.0):.3f}% | "
            f"{metrics.get('trade_count', 0.0):.0f} | "
            f"{metrics.get('avg_mfe_capture', 0.0):.3f} | "
            f"{100.0 * metrics.get('broker_max_drawdown_pct', 0.0):.2f}% | {row['name']} |"
        )
    return "\n".join(lines) + "\n"


def _metrics_line(metrics: dict[str, float], label: str) -> str:
    return (
        f"- {label}: broker net {100.0 * metrics.get(PRIMARY_OBJECTIVE_METRIC, 0.0):.2f}%, "
        f"exposure-normalized net {100.0 * metrics.get(EXPOSURE_NORMALIZED_SLOT_METRIC, metrics.get(PORTFOLIO_EQUIVALENT_METRIC, 0.0)):.2f}%, "
        f"equal-slot net {100.0 * metrics.get(EQUAL_SLOT_METRIC, metrics.get(LEGACY_SLOT_METRIC, 0.0)):.1f}%, "
        f"selected-day net {100.0 * metrics.get('selected_day_net_pct', 0.0):.3f}%, "
        f"active-day net {100.0 * metrics.get('active_day_net_pct', 0.0):.3f}%, "
        f"trades {metrics.get('trade_count', 0.0):.0f}, "
        f"broker DD {100.0 * metrics.get('broker_max_drawdown_pct', 0.0):.2f}%"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep KALCB trade plans over fixed KALCB Optimized candidates.")
    parser.add_argument("--config", default="config/optimization/kalcb.yaml")
    parser.add_argument("--optimized-source", default=str(DEFAULT_OPTIMIZED_SOURCE))
    parser.add_argument("--candidate-section", default="top_slot_return")
    parser.add_argument("--candidate-rank", type=int, default=0)
    parser.add_argument("--allow-non-default-candidate-source", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--fold-count", type=int, default=2)
    parser.add_argument("--coarse-entry-limit", type=int, default=720)
    parser.add_argument("--coarse-exit-limit", type=int, default=240)
    parser.add_argument("--entry-seed-top-n", type=int, default=12)
    parser.add_argument("--deep-refine-top-n", type=int, default=16)
    parser.add_argument("--deep-refine-max-specs", type=int, default=8_000)
    parser.add_argument("--finalist-count", type=int, default=50)
    parser.add_argument("--audit-max-workers", type=int, default=None)
    parser.add_argument("--worker-backend", choices=("thread", "process"), default="thread")
    parser.add_argument("--compiled-cache-dir", default=None)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--focused-seed-rows", default=None)
    parser.add_argument("--focused-seed-limit", type=int, default=0)
    parser.add_argument("--focused-seed-names", default="")
    args = parser.parse_args(argv)
    config = normalize_runtime_config("kalcb", load_yaml_config(args.config))
    payload = run_trade_plan_sweep(
        config,
        optimized_source=args.optimized_source,
        candidate_section=args.candidate_section,
        candidate_rank=args.candidate_rank,
        strict_candidate_source=not args.allow_non_default_candidate_source,
        output_dir=args.output_dir,
        train_only=bool(args.train_only),
        max_workers=args.max_workers,
        fold_count=args.fold_count,
        coarse_entry_limit=args.coarse_entry_limit,
        coarse_exit_limit=args.coarse_exit_limit,
        entry_seed_top_n=args.entry_seed_top_n,
        deep_refine_top_n=args.deep_refine_top_n,
        deep_refine_max_specs=args.deep_refine_max_specs,
        finalist_count=args.finalist_count,
        audit_max_workers=args.audit_max_workers,
        worker_backend=args.worker_backend,
        compiled_cache_dir=args.compiled_cache_dir,
        force_rebuild_cache=bool(args.force_rebuild_cache),
        focused_seed_rows=args.focused_seed_rows,
        focused_seed_limit=args.focused_seed_limit,
        focused_seed_names=tuple(name for name in str(args.focused_seed_names or "").split("||") if name),
    )
    print(json.dumps({"artifact_paths": payload["artifact_paths"], "sweep_hash": payload["sweep_hash"], "top_train": payload["top_train"][:3]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
