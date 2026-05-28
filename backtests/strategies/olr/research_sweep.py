from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Sequence

import pandas as pd
import yaml

from backtests.auto.shared.cache_keys import fingerprint_paths, stable_signature
from backtests.auto.shared.phase_state import _utc_now_iso
from backtests.auto.shared.types import Experiment
from backtests.config import load_yaml_config, normalize_runtime_config
from strategy_common.clock import KST
from strategy_common.daily_lrs_parquet import (
    load_daily_flow,
    load_daily_foreign_flow,
    load_daily_institutional_flow,
    load_daily_ohlcv,
    load_index_ohlcv,
    load_sector_map,
)
from strategy_common.market import MarketBar
from strategy_common.sector_intraday import AFTERNOON_CUTOFF, SectorIntradayPanel, build_sector_intraday_panel, load_canonical_sector_map
from strategy_olr.config import OLRConfig, OLR_CORE_VERSION
from strategy_olr.execution import OLRAllocationPlan, OLRTradeOutcome, summarize_olr_portfolio_proxy
from strategy_olr.models import OLRAfternoonContext, OLRDailySnapshot, OLRResearchSnapshot
from strategy_olr.research import (
    _research_config_fingerprint,
    afternoon_selection_from_snapshot,
    afternoon_selection_from_contexts,
    build_research_snapshot,
    build_afternoon_contexts,
    daily_selection_from_snapshot,
)


RESEARCH_SWEEP_VERSION = "olr-103-overnight-research-sweep-v1"
STAGE2_PORTFOLIO_PROXY_VERSION = "olr-stage2-close-auction-next-close-portfolio-proxy-v5"
DEFAULT_OUTPUT_DIR = Path("data/backtests/output/olr/research_sweeps")
DEFAULT_HOLDOUT_DAYS = 42
DEFAULT_EXPECTED_UNIVERSE_SIZE = 103
DEFAULT_STAGE1_STAGE2_SEED_COUNT = 5
_WINDOW_CACHE: dict[str, "OLRReplayWindow"] = {}
_FRAME_CACHE: dict[str, pd.DataFrame] = {}


@dataclass(frozen=True, slots=True)
class OLRReplayWindow:
    earliest: date
    latest: date
    train_start: date
    train_end: date
    holdout_start: date


def _resolve_replay_window(config: dict[str, Any], data_root: Path, timeframe: str, symbols: list[str]) -> OLRReplayWindow:
    cache_key = stable_signature(
        {
            "data_root": str(data_root),
            "timeframe": timeframe,
            "symbols": symbols,
            "holdout_days": int(config.get("holdout_days", DEFAULT_HOLDOUT_DAYS)),
        }
    )
    cached = _WINDOW_CACHE.get(cache_key)
    if cached is None:
        earliest: date | None = None
        latest: date | None = None
        for symbol in symbols:
            for path in sorted(Path(data_root).glob(f"{symbol}/{symbol}_{timeframe}_*.parquet")):
                frame = pd.read_parquet(path, columns=["timestamp"])
                if frame.empty:
                    continue
                min_date = frame["timestamp"].min().date()
                max_date = frame["timestamp"].max().date()
                earliest = min_date if earliest is None or min_date < earliest else earliest
                latest = max_date if latest is None or max_date > latest else latest
        if earliest is None or latest is None:
            raise FileNotFoundError(f"No {timeframe} parquet data found under {data_root}")
        holdout_start = latest - timedelta(days=int(config.get("holdout_days", DEFAULT_HOLDOUT_DAYS)))
        cached = OLRReplayWindow(earliest, latest, earliest, holdout_start - timedelta(days=1), holdout_start)
        _WINDOW_CACHE[cache_key] = cached

    date_range = dict(config.get("date_range") or {})
    train_start = date.fromisoformat(str(date_range["start"])) if date_range.get("start") else cached.train_start
    if date_range.get("end"):
        train_end = date.fromisoformat(str(date_range["end"]))
    elif config.get("holdout_start"):
        train_end = date.fromisoformat(str(config["holdout_start"])) - timedelta(days=1)
    elif bool(config.get("use_full_available_window", False)):
        train_end = cached.latest
    else:
        train_end = cached.train_end
    train_end = min(train_end, cached.latest)
    if train_end < train_start:
        raise ValueError(f"OLR replay train window is empty: {train_start} > {train_end}")
    return OLRReplayWindow(cached.earliest, cached.latest, train_start, train_end, cached.holdout_start)


def _load_symbol_frame(data_root: Path, symbol: str, timeframe: str, end: date) -> pd.DataFrame:
    key = stable_signature({"data_root": str(data_root), "symbol": symbol, "timeframe": timeframe, "end": end.isoformat()})
    cached = _FRAME_CACHE.get(key)
    if cached is not None:
        return cached
    frames: list[pd.DataFrame] = []
    for path in sorted(Path(data_root).glob(f"{symbol}/{symbol}_{timeframe}_*.parquet")):
        frame = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close", "volume"])
        frame = frame[frame["timestamp"].dt.date <= end]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        out = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    else:
        out = pd.concat(frames, ignore_index=True)
        out = out.drop_duplicates(subset=["timestamp"], keep="last")
        out = out.sort_values("timestamp").reset_index(drop=True)
    _FRAME_CACHE[key] = out
    return out

RESEARCH_MUTATION_KEYS = {
    "olr.frontier.active_selection_mode",
    "olr.premarket.frontier_size",
    "olr.premarket.min_adv20_krw",
    "olr.premarket.min_foreign5_z",
    "olr.research.top_long_count",
    "olr.research.min_price_krw",
    "olr.research.min_adv20_krw",
    "olr.research.min_history_days",
    "olr.research.max_median_spread_pct",
    "olr.research.require_spread",
    "olr.research.weights.relative_strength",
    "olr.research.weights.daily_trend",
    "olr.research.weights.compression",
    "olr.research.weights.accumulation",
    "olr.research.weights.stock_regime",
    "olr.research.weights.sector_regime",
    "olr.research.weights.sector_participation",
    "olr.research.weights.daily_signal",
    "olr.research.weights.flow",
    "olr.research.weights.foreign_flow",
    "olr.research.weights.institutional_flow",
    "olr.research.weights.flow_agreement",
    "olr.research.min_rs_percentile",
    "olr.research.min_trend_score",
    "olr.research.min_compression_score",
    "olr.research.min_accumulation_score",
    "olr.research.min_sector_participation",
    "olr.research.min_sector_daily_score_pct",
    "olr.research.max_box_range_pct",
    "olr.research.min_flow_5d",
    "olr.research.min_foreign_flow_5d",
    "olr.research.min_institutional_flow_5d",
    "olr.research.min_flow_z",
    "olr.research.min_flow_agreement",
    "olr.research.max_flow_divergence",
    "olr.research.min_sector_flow_5d",
    "olr.research.min_sector_foreign_flow_5d",
    "olr.research.min_sector_institutional_flow_5d",
    "olr.research.min_sector_flow_agreement",
    "olr.signal.daily_signal_family",
    "olr.signal.daily_min_score",
    "olr.signal.daily_rescue_min_score",
    "olr.signal.daily_max_score",
    "olr.signal.signal_floor",
    "olr.signal.flow_policy",
    "olr.signal.rescue_size_mult",
    "olr.signal.allow_secular",
    "olr.signal.secular_sizing_mult",
    "olr.signal.cdd_max",
    "olr.signal.gap_max_pct",
    "olr.signal.min_candidates_day",
    "olr.signal.rank_gate_mode",
    "olr.signal.daily_structure_weight",
    "olr.signal.min_relative_strength_pct",
    "olr.signal.max_relative_strength_pct",
    "olr.signal.min_parent_20d_return_pct",
    "olr.signal.max_parent_20d_return_pct",
    "olr.signal.min_market_breadth_pct",
    "olr.signal.min_market_heat_score",
    "olr.signal.structure_sizing_enabled",
    "olr.signal.structure_size_mult_min",
    "olr.signal.structure_size_mult_max",
    "olr.signal.rsi2_trigger_thresh",
    "olr.signal.rsi5_trigger_thresh",
    "olr.signal.cdd_min_for_rsi5",
    "olr.signal.depth_atr_trigger",
    "olr.signal.bb_pctb_trigger",
    "olr.signal.volume_climax_trigger",
    "olr.signal.relative_strength_trigger_pct",
    "olr.signal.roc5_drop_trigger_pct",
    "olr.signal.gap_down_trigger_pct",
}

AFTERNOON_MUTATION_KEYS = {
    "olr.afternoon.score_mode",
    "olr.afternoon.top_n",
    "olr.afternoon.min_ret",
    "olr.afternoon.min_vwap_ret",
    "olr.afternoon.max_ret",
    "olr.afternoon.max_vwap_ret",
    "olr.afternoon.min_gap",
    "olr.afternoon.max_gap",
    "olr.afternoon.min_rel_volume",
    "olr.afternoon.min_close_location",
    "olr.afternoon.max_open_drawdown",
    "olr.afternoon.max_range_atr",
    "olr.afternoon.min_high_from_open",
    "olr.afternoon.min_low_vs_prev_close",
    "olr.afternoon.min_prior_ret5",
    "olr.afternoon.min_prior_ret20",
    "olr.afternoon.max_prior_ret20",
    "olr.afternoon.min_prior_ret60",
    "olr.afternoon.min_bar_count",
    "olr.afternoon.min_flow_5d",
    "olr.afternoon.min_foreign_flow_5d",
    "olr.afternoon.min_institutional_flow_5d",
    "olr.afternoon.min_flow_z",
    "olr.afternoon.min_foreign_z",
    "olr.afternoon.min_institutional_z",
    "olr.afternoon.min_flow_agreement",
    "olr.afternoon.max_flow_divergence",
    "olr.afternoon.min_sector_flow",
    "olr.afternoon.min_sector_foreign_flow",
    "olr.afternoon.min_sector_institutional_flow",
    "olr.afternoon.min_intraday_sector_score_pct",
    "olr.afternoon.weight_intraday_sector",
    "olr.afternoon.min_market_score",
    "olr.afternoon.require_close_above_prev",
    "olr.afternoon.use_lagged_flow_score",
    "olr.afternoon.min_lagged_flow_score",
    "olr.afternoon.score_calibration_mode",
    "olr.afternoon.exhaustion_penalty",
    "olr.afternoon.max_exhaustion_score",
    "olr.afternoon.min_score",
    "olr.afternoon.max_score",
    "olr.afternoon.blocked_sectors",
    "olr.afternoon.allowed_sectors",
    "olr.overnight.slot_count",
}


@dataclass(frozen=True, slots=True)
class OLRResearchSweepDataset:
    config: dict[str, Any]
    source_fingerprint: str
    daily_source_fingerprint: str
    intraday_source_fingerprint: str
    data_root: Path
    daily_data_root: Path
    timeframe: str
    symbols: tuple[str, ...]
    requested_symbols: tuple[str, ...]
    excluded_symbols: dict[str, str]
    intraday_available_symbols: tuple[str, ...]
    intraday_unavailable_symbols: tuple[str, ...]
    daily_by_symbol: dict[str, list[dict[str, Any]]]
    flow_by_symbol: dict[str, list[dict[str, Any]]]
    foreign_flow_by_symbol: dict[str, list[dict[str, Any]]]
    institutional_flow_by_symbol: dict[str, list[dict[str, Any]]]
    index_by_code: dict[str, list[dict[str, Any]]]
    sector_map: dict[str, str]
    trading_dates: tuple[date, ...]
    bars_by_key: dict[tuple[date, str], tuple[MarketBar, ...]]
    train_start: date
    train_end: date
    holdout_start: date
    coverage_report: dict[str, Any]
    overnight_labels_by_key: dict[tuple[date, str], "OvernightLabel"] = field(default_factory=dict)
    sector_intraday_panel_1430: SectorIntradayPanel | None = None


@dataclass(frozen=True, slots=True)
class OvernightLabel:
    trade_date: date
    symbol: str
    entry_close: float
    next_close: float
    next_high: float
    next_low: float


@dataclass(frozen=True, slots=True)
class OvernightOpportunity:
    trade_date: date
    symbol: str
    rank: int
    valid: bool
    net_return_pct: float = 0.0
    gross_return_pct: float = 0.0
    mfe_r: float = 0.0
    mfe_pct: float = 0.0
    mae_r: float = 0.0


@dataclass(frozen=True, slots=True)
class ResearchSweepResult:
    experiment: Experiment
    score: float
    full_score: float
    median_fold_score: float
    worst_fold_score: float
    rejected: bool
    reject_reason: str
    metrics: dict[str, float]
    folds: tuple[dict[str, Any], ...]
    artifact_hash: str
    objective: str = "research_label"
    parent_stage1_name: str = ""
    parent_stage1_mutations: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Stage1ResumeBundle:
    source_path: Path
    source_payload_hash: str
    source_sweep_hash: str
    source_sweep_version: str
    source_strategy_core_version: str
    source_stage2_portfolio_proxy_version: str
    stage1_rows: tuple[ResearchSweepResult, ...]
    stage1_stage2_seeds: tuple[ResearchSweepResult, ...]
    stage1_refinement_seeds: tuple[ResearchSweepResult, ...]
    stage1_candidate_count: int
    stage1_coarse_candidate_count: int
    stage1_refinement_candidate_count: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "source_artifact": str(self.source_path),
            "source_payload_hash": self.source_payload_hash,
            "source_sweep_hash": self.source_sweep_hash,
            "source_sweep_version": self.source_sweep_version,
            "source_strategy_core_version": self.source_strategy_core_version,
            "source_stage2_portfolio_proxy_version": self.source_stage2_portfolio_proxy_version,
            "seed_count": len(self.stage1_stage2_seeds),
            "stage1_candidate_count": self.stage1_candidate_count,
            "stage1_coarse_candidate_count": self.stage1_coarse_candidate_count,
            "stage1_refinement_candidate_count": self.stage1_refinement_candidate_count,
            "policy": "Reuse Stage 1 seed mutations only; rebuild Stage 2 snapshots, contexts, metrics, and audits under current code.",
        }


def run_research_sweep(
    config: dict[str, Any] | None = None,
    *,
    mutations: dict[str, Any] | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    holdout_days: int = DEFAULT_HOLDOUT_DAYS,
    fold_days: int | None = None,
    fold_count: int = 2,
    top_n: int = 10,
    max_candidates: int | None = None,
    refine_top_n: int = 3,
    max_refinement_candidates: int | None = 96,
    stage1_stage2_seed_count: int = DEFAULT_STAGE1_STAGE2_SEED_COUNT,
    max_workers: int = 2,
    expected_universe_size: int = DEFAULT_EXPECTED_UNIVERSE_SIZE,
    allow_universe_size_override: bool = False,
    audit_finalist_count: int = 5,
    audit_metric_tolerance: float = 1e-12,
    resume_stage1_artifact: str | Path | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _clear_progress_files(out)
    training_config = _training_config(dict(config or {}), holdout_days)
    base_mutations = _base_mutations(training_config, mutations)
    _write_run_status(out, "preparing_dataset")
    dataset = prepare_research_sweep_dataset(
        training_config,
        holdout_days=holdout_days,
        expected_universe_size=expected_universe_size,
        allow_universe_size_override=allow_universe_size_override,
    )
    folds = _resolve_folds(list(dataset.trading_dates), fold_days=fold_days, fold_count=fold_count)
    _write_run_status(out, "compiling_research_snapshots", sessions=len(dataset.trading_dates), symbols=len(dataset.symbols))
    research_snapshot_cache = research_snapshots_for_dataset(dataset, base_mutations)

    stage1_resume: Stage1ResumeBundle | None = None
    if resume_stage1_artifact is not None:
        stage1_resume = _load_stage1_resume_bundle(resume_stage1_artifact, stage1_stage2_seed_count=stage1_stage2_seed_count)
        refreshed_seeds = [
            evaluate_stage1_experiment(row.experiment, dataset, base_mutations, folds, research_snapshots=research_snapshot_cache)
            for row in stage1_resume.stage1_stage2_seeds
        ]
        refreshed_by_name = {row.experiment.name: row for row in refreshed_seeds}
        stage1_rows = [refreshed_by_name.get(row.experiment.name, row) for row in stage1_resume.stage1_rows]
        for row in refreshed_seeds:
            if row.experiment.name not in {item.experiment.name for item in stage1_rows}:
                stage1_rows.append(row)
        stage1_rows.sort(key=lambda row: (-row.score, row.rejected, row.experiment.name))
        stage1_refinement_seeds = [refreshed_by_name.get(row.experiment.name, row) for row in stage1_resume.stage1_refinement_seeds]
        stage1_refinement_candidates: list[Experiment] = []
        stage1_candidates: list[Experiment] = []
        stage1_stage2_seeds = refreshed_seeds
        _write_run_status(
            out,
            "resuming_stage1_for_stage2",
            source_artifact=str(stage1_resume.source_path),
            source_sweep_hash=stage1_resume.source_sweep_hash,
            seed_count=len(stage1_stage2_seeds),
            refreshed_seed_count=len(refreshed_seeds),
        )
    else:
        stage1_candidates = build_research_sweep_candidates()
        if max_candidates is not None:
            stage1_candidates = stage1_candidates[: max(0, int(max_candidates))]
        baseline = Experiment("__baseline__", {})
        stage1_coarse = [baseline, *stage1_candidates]
        _write_run_status(out, "stage1_coarse", total=len(stage1_coarse))
        stage1_rows = _evaluate_stage1_candidates(
            stage1_coarse,
            dataset,
            base_mutations,
            folds,
            research_snapshot_cache,
            out,
            stage="stage1_coarse",
            completed_offset=0,
            total=len(stage1_coarse),
            max_workers=max_workers,
        )
        stage1_rows.sort(key=lambda row: (-row.score, row.rejected, row.experiment.name))
        stage1_refinement_seeds = [row for row in stage1_rows if not row.rejected][: max(0, int(refine_top_n))]
        stage1_refinement_candidates = build_research_refinement_candidates(
            stage1_refinement_seeds,
            existing_mutations=[row.experiment.mutations for row in stage1_rows],
            max_candidates=max_refinement_candidates,
        )
        stage1_refine_total = len(stage1_rows) + len(stage1_refinement_candidates)
        _write_run_status(out, "stage1_refine", total=len(stage1_refinement_candidates))
        stage1_refine_rows = _evaluate_stage1_candidates(
            stage1_refinement_candidates,
            dataset,
            base_mutations,
            folds,
            research_snapshot_cache,
            out,
            stage="stage1_refine",
            completed_offset=len(stage1_rows),
            total=stage1_refine_total,
            max_workers=max_workers,
            seed_rows=stage1_rows,
        )
        stage1_rows.extend(stage1_refine_rows)
        stage1_rows.sort(key=lambda row: (-row.score, row.rejected, row.experiment.name))
        stage1_stage2_seeds = [row for row in stage1_rows if not row.rejected][: max(1, int(stage1_stage2_seed_count))]
        if not stage1_stage2_seeds:
            stage1_stage2_seeds = stage1_rows[:1]

    stage2_rows: list[ResearchSweepResult] = []
    stage1_snapshots_by_name: dict[str, dict[date, OLRDailySnapshot]] = {}
    stage1_contexts_by_name: dict[str, dict[date, dict[str, OLRAfternoonContext]]] = {}
    for seed_index, stage1_seed in enumerate(stage1_stage2_seeds, start=1):
        stage1_mutations = dict(base_mutations)
        stage1_mutations.update(stage1_seed.experiment.mutations)
        _write_run_status(
            out,
            "compiling_stage1_seed_for_stage2",
            seed_index=seed_index,
            seed_count=len(stage1_stage2_seeds),
            name=stage1_seed.experiment.name,
        )
        stage1_seed_snapshots = snapshots_for_experiment(dataset, stage1_mutations, research_snapshots=research_snapshot_cache)
        stage1_afternoon_contexts = afternoon_contexts_for_snapshots(
            dataset,
            stage1_seed_snapshots,
            OLRConfig.from_mapping(dataset.config, stage1_mutations),
        )
        stage1_snapshots_by_name[stage1_seed.experiment.name] = stage1_seed_snapshots
        stage1_contexts_by_name[stage1_seed.experiment.name] = stage1_afternoon_contexts

        stage2_candidates = build_afternoon_sweep_candidates(stage1_seed)
        if max_refinement_candidates is not None:
            stage2_candidates = stage2_candidates[: max(0, int(max_refinement_candidates))]
        stage2_baseline = Experiment("__stage2_baseline__", {})
        stage2_all = _stage2_experiments_for_stage1(stage1_seed, [stage2_baseline, *stage2_candidates])
        _write_run_status(
            out,
            "stage2_portfolio_afternoon",
            seed_index=seed_index,
            seed_count=len(stage1_stage2_seeds),
            stage1_candidate=stage1_seed.experiment.name,
            total=len(stage2_all),
        )
        seed_stage2_rows = _evaluate_stage2_candidates(
            stage2_all,
            dataset,
            stage1_mutations,
            folds,
            stage1_seed_snapshots,
            stage1_afternoon_contexts,
            out,
            stage=f"stage2_portfolio_seed{seed_index}",
            max_workers=max_workers,
            parent_stage1_name=stage1_seed.experiment.name,
            parent_stage1_mutations=stage1_seed.experiment.mutations,
        )
        stage2_rows.extend(seed_stage2_rows)
    stage2_rows.sort(key=lambda row: (-row.score, row.rejected, row.experiment.name))
    selected = [row for row in stage2_rows if not row.rejected][: max(1, int(top_n))]
    seed_row = selected[0] if selected else stage2_rows[0]
    selected_stage1_name = seed_row.parent_stage1_name or stage1_stage2_seeds[0].experiment.name
    selected_stage1_row = next((row for row in stage1_stage2_seeds if row.experiment.name == selected_stage1_name), stage1_stage2_seeds[0])
    stage1_mutations = dict(base_mutations)
    stage1_mutations.update(selected_stage1_row.experiment.mutations)
    phase_seed = dict(stage1_mutations)
    phase_seed.update(seed_row.experiment.mutations)
    audit_targets = selected[: max(0, int(audit_finalist_count))]
    if not audit_targets and stage2_rows:
        audit_targets = stage2_rows[:1]
    _write_run_status(out, "full_audit_finalists", total=len(audit_targets))
    audit_research_snapshot_cache = research_snapshots_for_dataset(dataset, base_mutations)
    stage1_seed_audits = [
        _audit_stage1_result(row, dataset, base_mutations, folds, research_snapshots=audit_research_snapshot_cache, tolerance=audit_metric_tolerance)
        for row in stage1_stage2_seeds
    ]
    stage1_seed_audit = next(
        (row for row in stage1_seed_audits if row.get("name") == selected_stage1_row.experiment.name),
        stage1_seed_audits[0] if stage1_seed_audits else {},
    )
    audit_rows = _audit_stage2_results(
        audit_targets,
        dataset,
        base_mutations,
        folds,
        research_snapshots=audit_research_snapshot_cache,
        tolerance=audit_metric_tolerance,
    )
    audit_pass = all(bool(row.get("audit_pass", True)) for row in stage1_seed_audits) and all(bool(row.get("audit_pass")) for row in audit_rows)
    payload = {
        "strategy": "olr",
        "strategy_core_version": OLR_CORE_VERSION,
        "sweep_version": RESEARCH_SWEEP_VERSION,
        "sweep_type": "overnight_leader_rotation_research_training_only",
        "created_at": _utc_now_iso(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "official_performance": False,
        "training_window": {
            "start": dataset.trading_dates[0].isoformat(),
            "end": dataset.trading_dates[-1].isoformat(),
            "sessions": len(dataset.trading_dates),
        },
        "holdout_policy": {
            "holdout_days": int(holdout_days),
            "train_only": True,
            "selection_uses_holdout": False,
            "execution_uses_holdout": False,
            "sizing_uses_holdout": False,
        },
        "holdout_contract": {
            "holdout_days": int(holdout_days),
            "selection_uses_holdout": False,
            "policy": "Dates >= holdout_start are excluded from the research sweep training window.",
        },
        "causality_policy": {
            "daily_row_cutoff": "row_date < trade_date",
            "flow_row_cutoff": "row_date < trade_date",
            "intraday_selection_cutoff": "timestamp < 14:30 KST",
            "overnight_label": "day-D close to day-D+1 close; labels are offline scoring only",
            "official_performance": False,
        },
        "implementation_lessons_contract": _implementation_lessons_contract(),
        "metric_contract": _metric_contract(),
        "fast_replay_policy": {
            "enabled": True,
            "mode": "compiled_causal_research_replay",
            "suppressed_work": "daily/flow row parsing, overnight label lookup, and afternoon context construction are cached; selector calls stay shared with live",
            "selection_functions": [
                "strategy_olr.research.daily_selection_from_snapshot",
                "strategy_olr.research.afternoon_selection_from_contexts",
            ],
            "live_wrapper_function": "strategy_olr.research.afternoon_selection_from_snapshot",
            "stage2_context_builder": "strategy_olr.research.build_afternoon_contexts",
            "stage2_portfolio_proxy_version": STAGE2_PORTFOLIO_PROXY_VERSION,
            "full_audit_rebuild_policy": "Audit rebuilds the full causal ResearchSnapshot cache once, then replays finalist selectors from that independently rebuilt cache.",
            "full_audit_finalist_count": len(audit_rows),
            "audit_metric_tolerance": float(audit_metric_tolerance),
            "fill_parity_scope": "not_applicable_research_only",
        },
        "stage1_seed_audit": stage1_seed_audit,
        "stage1_seed_audits": stage1_seed_audits,
        "full_audit_replays": audit_rows,
        "audit_pass": audit_pass,
        "coverage": dataset.coverage_report,
        "fold_days": int(fold_days) if fold_days is not None else None,
        "fold_count": int(fold_count),
        "folds": [{"start": start.isoformat(), "end": end.isoformat()} for start, end in folds],
        "base_mutations": base_mutations,
        "stage1_resume": stage1_resume.to_payload() if stage1_resume is not None else {"enabled": False},
        "stage1_candidate_count": stage1_resume.stage1_candidate_count if stage1_resume is not None else len(stage1_rows),
        "stage1_coarse_candidate_count": stage1_resume.stage1_coarse_candidate_count if stage1_resume is not None else 1 + len(stage1_candidates),
        "stage1_refinement_candidate_count": stage1_resume.stage1_refinement_candidate_count if stage1_resume is not None else len(stage1_refinement_candidates),
        "stage1_refinement_seeds": [
            {"name": row.experiment.name, "score": row.score, "mutations": row.experiment.mutations}
            for row in stage1_refinement_seeds
        ],
        "stage1_stage2_seed_count": len(stage1_stage2_seeds),
        "stage1_stage2_seeds": [_row_payload(row) for row in stage1_stage2_seeds],
        "stage1_frontier": [_row_payload(row) for row in stage1_rows[: max(10, int(top_n))]],
        "selected_stage1_seed": _row_payload(selected_stage1_row),
        "stage2_candidate_count": len(stage2_rows),
        "stage2_frontier": [_row_payload(row) for row in stage2_rows[: max(10, int(top_n))]],
        "selected_stage2_seed": _row_payload(seed_row),
        "selected_combined_seed": {
            "stage1_candidate": selected_stage1_row.experiment.name,
            "stage2_candidate": seed_row.experiment.name,
            "score": seed_row.score,
            "mutations": phase_seed,
            "metric_basis": "stage2_fixed_execution_portfolio_proxy_not_official_performance",
        },
        "selected_count": len(selected),
        "selection_frontier": [_row_payload(row) for row in selected],
        "phase_auto_seed": {
            **training_config,
            "initial_mutations": phase_seed,
            "research_sweep_seed": {
                "stage1_candidate": selected_stage1_row.experiment.name,
                "stage2_candidate": seed_row.experiment.name,
                "score": seed_row.score,
                "metrics": seed_row.metrics,
                "policy": "Use this OLR research seed for later overnight execution design; holdout remains untouched.",
                "metric_basis": "stage2_fixed_execution_portfolio_proxy_not_official_performance",
                "thin_layer_contract": "Replay and live wrappers must call strategy_olr.research daily_selection_from_snapshot() and afternoon_selection_from_snapshot().",
            },
        },
    }
    payload["sweep_hash"] = stable_signature(
        {
            "training_config": training_config,
            "stage1_resume": stage1_resume.to_payload() if stage1_resume is not None else {"enabled": False},
            "stage1": [{"name": row.experiment.name, "score": row.score, "mutations": row.experiment.mutations} for row in stage1_rows],
            "stage2": [
                {
                    "name": row.experiment.name,
                    "score": row.score,
                    "stage1": row.parent_stage1_name,
                    "mutations": row.experiment.mutations,
                }
                for row in stage2_rows
            ],
        }
    )
    json_path = out / f"olr_research_sweep_{payload['sweep_hash'][:12]}.json"
    md_path = out / f"olr_research_sweep_{payload['sweep_hash'][:12]}.md"
    seed_path = out / f"olr_research_seed_{payload['sweep_hash'][:12]}.json"
    payload["artifact_paths"] = {"json": str(json_path), "markdown": str(md_path), "phase_auto_seed": str(seed_path)}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    seed_path.write_text(json.dumps(payload["phase_auto_seed"], indent=2, sort_keys=True, default=str), encoding="utf-8")
    _write_run_status(out, "completed", sweep_hash=payload["sweep_hash"], json=str(json_path), markdown=str(md_path), audit_pass=audit_pass)
    return payload


def build_research_sweep_candidates() -> list[Experiment]:
    raw: list[tuple[str, dict[str, Any]]] = []
    raw.extend((f"top{count}", {"olr.research.top_long_count": count}) for count in (10, 15, 20, 30, 40, 60))
    raw.extend((f"adv{int(value / 1_000_000_000)}b", {"olr.research.min_adv20_krw": value}) for value in (1_000_000_000, 2_000_000_000, 5_000_000_000, 10_000_000_000))
    raw.extend(
        [
            ("rs_trend", _weights(rs=0.30, trend=0.24, comp=0.08, accum=0.08, stock=0.08, sector=0.05, part=0.05, signal=0.07, flow=0.03, foreign=0.01, inst=0.01, agree=0.0)),
            ("flow_confirmed", _weights(rs=0.18, trend=0.16, comp=0.08, accum=0.08, stock=0.06, sector=0.06, part=0.06, signal=0.08, flow=0.12, foreign=0.06, inst=0.04, agree=0.02)),
            ("sector_participation", _weights(rs=0.18, trend=0.16, comp=0.08, accum=0.08, stock=0.06, sector=0.12, part=0.14, signal=0.08, flow=0.05, foreign=0.02, inst=0.02, agree=0.01)),
            ("daily_signal", {"olr.research.weights.daily_signal": 0.20, "olr.signal.daily_min_score": 55.0}),
            ("score_active", {"olr.frontier.active_selection_mode": "score"}),
            ("liquidity_active", {"olr.frontier.active_selection_mode": "liquidity"}),
            ("campaign_active", {"olr.frontier.active_selection_mode": "campaign"}),
            ("hot_active", {"olr.frontier.active_selection_mode": "hot"}),
            ("premarket_adv5b", {"olr.premarket.min_adv20_krw": 5_000_000_000.0}),
            ("premarket_foreign_z_m1", {"olr.premarket.min_foreign5_z": -1.0}),
            ("rs_floor60", {"olr.research.min_rs_percentile": 60.0}),
            ("rs_floor70", {"olr.research.min_rs_percentile": 70.0}),
            ("trend_floor55", {"olr.research.min_trend_score": 55.0}),
            ("accum_floor0", {"olr.research.min_accumulation_score": 0.0}),
            ("sector_participation50", {"olr.research.min_sector_participation": 0.50}),
            ("sector_daily45", {"olr.research.min_sector_daily_score_pct": 45.0}),
            ("sector_daily55", {"olr.research.min_sector_daily_score_pct": 55.0}),
            ("sector_daily65", {"olr.research.min_sector_daily_score_pct": 65.0}),
            ("flow_5d_positive", {"olr.research.min_flow_5d": 0.0}),
            ("foreign_5d_positive", {"olr.research.min_foreign_flow_5d": 0.0}),
            ("inst_5d_positive", {"olr.research.min_institutional_flow_5d": 0.0}),
            ("flow_agreement_positive", {"olr.research.min_flow_agreement": 0.0}),
            ("flow_z_positive", {"olr.research.min_flow_z": 0.0}),
            ("sector_flow_positive", {"olr.research.min_sector_flow_5d": 0.0}),
            ("sector_foreign_flow_positive", {"olr.research.min_sector_foreign_flow_5d": 0.0}),
            ("sector_inst_flow_positive", {"olr.research.min_sector_institutional_flow_5d": 0.0}),
            ("sector_flow_agreement_positive", {"olr.research.min_sector_flow_agreement": 0.0}),
            ("box_range12", {"olr.research.max_box_range_pct": 0.12}),
            ("spread_required", {"olr.research.require_spread": True}),
            ("score_floor46", {"olr.signal.daily_min_score": 46.0}),
            ("score_floor58", {"olr.signal.daily_min_score": 58.0}),
            ("rescue_floor30", {"olr.signal.daily_rescue_min_score": 30.0}),
            ("signal_floor40", {"olr.signal.signal_floor": 40.0}),
            ("score_cap88", {"olr.signal.daily_max_score": 88.0}),
            ("flow_policy_positive", {"olr.signal.flow_policy": "require_positive"}),
            ("no_secular", {"olr.signal.allow_secular": False}),
            ("cdd_max4", {"olr.signal.cdd_max": 4}),
            ("cdd_max8", {"olr.signal.cdd_max": 8}),
            ("rsi5_cdd3", {"olr.signal.cdd_min_for_rsi5": 3}),
            ("gap_max4", {"olr.signal.gap_max_pct": 4.0}),
            ("rsi2_10", {"olr.signal.rsi2_trigger_thresh": 10.0}),
            ("rsi2_20", {"olr.signal.rsi2_trigger_thresh": 20.0}),
            ("rsi5_25", {"olr.signal.rsi5_trigger_thresh": 25.0}),
            ("rsi5_35", {"olr.signal.rsi5_trigger_thresh": 35.0}),
            ("depth_atr10", {"olr.signal.depth_atr_trigger": 1.0}),
            ("depth_atr20", {"olr.signal.depth_atr_trigger": 2.0}),
            ("bb_pctb00", {"olr.signal.bb_pctb_trigger": 0.0}),
            ("bb_pctb10", {"olr.signal.bb_pctb_trigger": 0.10}),
            ("volume_climax15", {"olr.signal.volume_climax_trigger": 1.5}),
            ("volume_climax25", {"olr.signal.volume_climax_trigger": 2.5}),
            ("rs_trigger60", {"olr.signal.relative_strength_trigger_pct": 60.0}),
            ("rs_trigger80", {"olr.signal.relative_strength_trigger_pct": 80.0}),
            ("roc5_m5", {"olr.signal.roc5_drop_trigger_pct": -5.0}),
            ("gapdown_m1", {"olr.signal.gap_down_trigger_pct": -1.0}),
            ("structure025", {"olr.signal.daily_structure_weight": 0.25}),
            ("structure045", {"olr.signal.daily_structure_weight": 0.45}),
            ("structure_sizing_on", {"olr.signal.structure_sizing_enabled": True}),
            ("rs_floor35", {"olr.signal.min_relative_strength_pct": 35.0}),
            ("rs_ceiling85", {"olr.signal.max_relative_strength_pct": 85.0}),
            ("parent_tight", {"olr.signal.min_parent_20d_return_pct": -6.0, "olr.signal.max_parent_20d_return_pct": 24.0}),
            ("heat45", {"olr.signal.min_market_heat_score": 45.0}),
            ("breadth45", {"olr.signal.min_market_breadth_pct": 45.0}),
        ]
    )
    candidates = [Experiment(name, mutations) for name, mutations in raw]
    _validate_mutations(candidates, RESEARCH_MUTATION_KEYS)
    return candidates


def build_research_refinement_candidates(
    seed_rows: list[ResearchSweepResult],
    *,
    existing_mutations: list[dict[str, Any]] | None = None,
    max_candidates: int | None = 96,
) -> list[Experiment]:
    if not seed_rows or max_candidates == 0:
        return []
    existing = {_mutation_signature(mutations) for mutations in (existing_mutations or [])}
    raw: list[tuple[str, dict[str, Any]]] = []
    for index, row in enumerate(seed_rows, start=1):
        seed = dict(row.experiment.mutations)
        prefix = f"ref{index}_{_safe_name(row.experiment.name)}"
        for top in (10, 12, 15, 20, 25, 30, 40, 50, 60):
            raw.append((f"{prefix}_top{top}", {**seed, "olr.research.top_long_count": top}))
        for mode in ("score", "hybrid", "liquidity", "campaign", "hot"):
            raw.append((f"{prefix}_{mode}", {**seed, "olr.frontier.active_selection_mode": mode}))
        for rs in (45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0):
            raw.append((f"{prefix}_rs{_num_label(rs)}", {**seed, "olr.research.min_rs_percentile": rs}))
        for trend in (35.0, 45.0, 55.0, 65.0, 70.0, 75.0):
            raw.append((f"{prefix}_trend{_num_label(trend)}", {**seed, "olr.research.min_trend_score": trend}))
        for flow in (-0.02, -0.01, 0.0, 0.01, 0.02):
            raw.append((f"{prefix}_flow{_num_label(flow)}", {**seed, "olr.research.min_flow_5d": flow}))
            raw.append((f"{prefix}_fflow{_num_label(flow)}", {**seed, "olr.research.min_foreign_flow_5d": flow}))
            raw.append((f"{prefix}_iflow{_num_label(flow)}", {**seed, "olr.research.min_institutional_flow_5d": flow}))
        for value in (0.0, 0.20, 0.35, 0.45, 0.55, 0.65):
            raw.append((f"{prefix}_struct{_num_label(value)}", {**seed, "olr.signal.daily_structure_weight": value}))
        for value in (8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 24.0):
            raw.append((f"{prefix}_rsi2{_num_label(value)}", {**seed, "olr.signal.rsi2_trigger_thresh": value}))
        for value in (0.0, 30.0, 40.0, 50.0, 60.0):
            raw.append((f"{prefix}_sig_rsmin{_num_label(value)}", {**seed, "olr.signal.min_relative_strength_pct": value}))
        for family, weights in _refinement_weight_families():
            raw.append((f"{prefix}_{family}", {**seed, **weights}))

    out: list[Experiment] = []
    seen = set(existing)
    for name, mutations in raw:
        clean = {key: value for key, value in mutations.items() if key in RESEARCH_MUTATION_KEYS}
        signature = _mutation_signature(clean)
        if signature in seen:
            continue
        seen.add(signature)
        out.append(Experiment(name, clean))
        if max_candidates is not None and len(out) >= max(0, int(max_candidates)):
            break
    _validate_mutations(out, RESEARCH_MUTATION_KEYS)
    return out


def build_afternoon_sweep_candidates(seed: ResearchSweepResult | None = None) -> list[Experiment]:
    raw: list[tuple[str, dict[str, Any]]] = []
    modes = ("momentum", "vwap_strength", "gap_hold", "flow_confirmed", "efficient", "hybrid", "daily_plus_intraday")
    filters = [
        {"olr.afternoon.min_ret": 0.0},
        {"olr.afternoon.min_ret": 0.002, "olr.afternoon.min_vwap_ret": 0.0},
        {"olr.afternoon.min_ret": 0.005, "olr.afternoon.min_rel_volume": 0.75},
        {"olr.afternoon.min_vwap_ret": 0.001, "olr.afternoon.min_close_location": 0.60},
        {"olr.afternoon.min_rel_volume": 1.25, "olr.afternoon.min_close_location": 0.65},
        {"olr.afternoon.max_open_drawdown": 0.015},
        {"olr.afternoon.max_range_atr": 1.25},
        {"olr.afternoon.max_vwap_ret": 0.035},
        {"olr.afternoon.max_ret": 0.055},
        {"olr.afternoon.max_exhaustion_score": 1.75},
        {"olr.afternoon.score_calibration_mode": "exhaustion_adjusted", "olr.afternoon.exhaustion_penalty": 15.0},
        {"olr.afternoon.score_calibration_mode": "exhaustion_adjusted", "olr.afternoon.exhaustion_penalty": 25.0, "olr.afternoon.max_exhaustion_score": 2.25},
        {"olr.afternoon.min_gap": -0.02, "olr.afternoon.max_gap": 0.08},
        {"olr.afternoon.min_lagged_flow_score": 0.0},
        {"olr.afternoon.min_prior_ret5": 0.03},
        {"olr.afternoon.min_prior_ret20": 0.0},
        {"olr.afternoon.max_prior_ret20": 0.30, "olr.afternoon.min_prior_ret5": 0.03},
        {"olr.afternoon.min_low_vs_prev_close": -0.02, "olr.afternoon.require_close_above_prev": True},
        {"olr.afternoon.min_flow_5d": 0.0},
        {"olr.afternoon.min_foreign_flow_5d": 0.0},
        {"olr.afternoon.min_institutional_flow_5d": 0.0},
        {"olr.afternoon.min_foreign_flow_5d": 0.0, "olr.afternoon.min_institutional_flow_5d": 0.0},
        {"olr.afternoon.min_flow_z": 0.0},
        {"olr.afternoon.min_foreign_z": 0.0},
        {"olr.afternoon.min_institutional_z": 0.0},
        {"olr.afternoon.min_flow_agreement": 0.0},
        {"olr.afternoon.max_flow_divergence": 0.01},
        {"olr.afternoon.min_sector_flow": 0.0},
        {"olr.afternoon.min_intraday_sector_score_pct": 55.0},
        {"olr.afternoon.min_intraday_sector_score_pct": 65.0},
        {"olr.afternoon.weight_intraday_sector": 0.001},
        {"olr.afternoon.weight_intraday_sector": 0.002},
        {"olr.afternoon.min_sector_foreign_flow": 0.0},
        {"olr.afternoon.min_sector_institutional_flow": 0.0},
        {"olr.afternoon.min_flow_5d": 0.0, "olr.afternoon.min_sector_flow": 0.0},
        {"olr.afternoon.min_flow_agreement": 0.0, "olr.afternoon.min_sector_flow": 0.0},
        {"olr.afternoon.min_flow_z": 0.0, "olr.afternoon.min_prior_ret5": 0.03},
        {"olr.afternoon.min_market_score": 0.0, "olr.afternoon.min_prior_ret5": 0.03},
        {"olr.afternoon.min_market_score": 55.0, "olr.afternoon.min_gap": 0.002, "olr.afternoon.max_gap": 0.10},
        {"olr.afternoon.min_market_score": 45.0},
        {"olr.afternoon.min_bar_count": 48},
    ]
    for mode in modes:
        for top_n in (2, 4, 6, 8, 10, 12):
            for values in filters:
                mutations = {"olr.afternoon.score_mode": mode, "olr.afternoon.top_n": top_n, **values}
                raw.append((_afternoon_name(mode, top_n, values), mutations))
    candidates = [Experiment(name, mutations) for name, mutations in raw]
    _validate_mutations(candidates, AFTERNOON_MUTATION_KEYS)
    return candidates


def _stage2_experiments_for_stage1(stage1_seed: ResearchSweepResult, experiments: list[Experiment]) -> list[Experiment]:
    prefix = _safe_name(stage1_seed.experiment.name)
    return [Experiment(f"{prefix}__{experiment.name}", dict(experiment.mutations)) for experiment in experiments]


def prepare_research_sweep_dataset(
    config: dict[str, Any],
    *,
    holdout_days: int = DEFAULT_HOLDOUT_DAYS,
    expected_universe_size: int = DEFAULT_EXPECTED_UNIVERSE_SIZE,
    allow_universe_size_override: bool = False,
    include_holdout: bool = False,
) -> OLRResearchSweepDataset:
    raw_config = dict(config or {})
    raw_config["holdout_days"] = int(holdout_days)
    data_root = Path(raw_config.get("data_root", "data/kis_intraday_parquet"))
    daily_root = Path(raw_config.get("daily_data_root", "data/krx_daily_parquet"))
    timeframe = str(raw_config.get("timeframe", "5m") or "5m")
    if timeframe != "5m":
        raise ValueError("OLR research sweep requires 5m parquet input")
    requested = tuple(_requested_symbols(raw_config, data_root, timeframe))
    window = _resolve_replay_window(raw_config, data_root, timeframe, list(requested))
    symbols, excluded = resolve_complete_daily_flow_intraday_universe(
        raw_config,
        daily_root,
        data_root,
        window.train_end,
        timeframe=timeframe,
        expected_universe_size=expected_universe_size,
        allow_universe_size_override=allow_universe_size_override,
    )
    frames = {symbol: _load_symbol_frame(data_root, symbol, timeframe, window.train_end) for symbol in symbols}
    intraday_available = tuple(symbol for symbol, frame in frames.items() if not frame.empty)
    intraday_unavailable = tuple(symbol for symbol, frame in frames.items() if frame.empty)
    daily_by_symbol = _load_daily_rows(daily_root, symbols, window.train_end)
    flow_by_symbol = _load_flow_rows(daily_root, symbols, window.train_end)
    foreign_flow_by_symbol = _load_foreign_flow_rows(daily_root, symbols, window.train_end)
    institutional_flow_by_symbol = _load_institutional_flow_rows(daily_root, symbols, window.train_end)
    index_by_code = _load_index_rows(daily_root, window.train_end)
    sector_map = load_canonical_sector_map(raw_config, fallback=load_sector_map(daily_root))
    bars_by_key = _bars_by_key_from_frames(frames, window.train_start, window.train_end, source_fingerprint="olr-replay")
    trading_dates = tuple(
        day
        for day in _trading_dates(frames, window.train_start, window.train_end)
        if include_holdout or day < window.holdout_start
    )
    if not trading_dates:
        raise ValueError("OLR research sweep found no 5m bars in the selected training window")
    overnight_labels = _overnight_label_cache(daily_by_symbol, symbols, trading_dates)
    daily_source = _daily_source_fingerprint(daily_root, symbols, window.train_end)
    intraday_source = _intraday_source_fingerprint(data_root, intraday_available, timeframe, window.train_start, window.train_end)
    sector_intraday_panel_1430 = build_sector_intraday_panel(
        bars_by_key,
        sector_map,
        trade_dates=trading_dates,
        cutoff=AFTERNOON_CUTOFF,
        symbols=symbols,
    )
    coverage = _coverage_report(symbols, excluded, daily_by_symbol, flow_by_symbol, foreign_flow_by_symbol, institutional_flow_by_symbol, sector_map, bars_by_key, trading_dates)
    return OLRResearchSweepDataset(
        config=raw_config,
        source_fingerprint=stable_signature([daily_source, intraday_source]),
        daily_source_fingerprint=daily_source,
        intraday_source_fingerprint=intraday_source,
        data_root=data_root,
        daily_data_root=daily_root,
        timeframe=timeframe,
        symbols=tuple(symbols),
        requested_symbols=requested,
        excluded_symbols=excluded,
        intraday_available_symbols=intraday_available,
        intraday_unavailable_symbols=intraday_unavailable,
        daily_by_symbol=daily_by_symbol,
        flow_by_symbol=flow_by_symbol,
        foreign_flow_by_symbol=foreign_flow_by_symbol,
        institutional_flow_by_symbol=institutional_flow_by_symbol,
        index_by_code=index_by_code,
        sector_map=sector_map,
        trading_dates=trading_dates,
        bars_by_key=bars_by_key,
        train_start=window.train_start,
        train_end=window.train_end,
        holdout_start=window.holdout_start,
        coverage_report=coverage,
        overnight_labels_by_key=overnight_labels,
        sector_intraday_panel_1430=sector_intraday_panel_1430,
    )


def resolve_complete_daily_flow_intraday_universe(
    config: dict[str, Any],
    daily_root: str | Path,
    data_root: str | Path,
    train_end: date,
    *,
    timeframe: str = "5m",
    expected_universe_size: int = DEFAULT_EXPECTED_UNIVERSE_SIZE,
    allow_universe_size_override: bool = False,
) -> tuple[tuple[str, ...], dict[str, str]]:
    root = Path(daily_root)
    intraday_root = Path(data_root)
    requested = _requested_symbols(config, intraday_root, timeframe)
    sector_map = load_sector_map(root)
    selected: list[str] = []
    excluded: dict[str, str] = {}
    for symbol in requested:
        missing: list[str] = []
        if load_daily_ohlcv(root, symbol, end=train_end).empty:
            missing.append("daily_ohlcv")
        if load_daily_flow(root, symbol, end=train_end).empty:
            missing.append("daily_flow")
        if load_daily_foreign_flow(root, symbol, end=train_end).empty:
            missing.append("daily_foreign_flow")
        if load_daily_institutional_flow(root, symbol, end=train_end).empty:
            missing.append("daily_institutional_flow")
        if symbol not in sector_map:
            missing.append("sector_map")
        if not _has_intraday_data(intraday_root, symbol, timeframe):
            missing.append(f"intraday_{timeframe}")
        if missing:
            excluded[symbol] = "missing_" + "_".join(missing)
        else:
            selected.append(symbol)
    expected = int(expected_universe_size)
    if expected > 0 and len(selected) != expected and not allow_universe_size_override:
        raise ValueError(f"OLR research sweep requires exactly {expected} complete daily-flow-intraday symbols; got {len(selected)}")
    return tuple(sorted(selected)), dict(sorted(excluded.items()))


def evaluate_stage1_experiment(
    experiment: Experiment,
    dataset: OLRResearchSweepDataset,
    base_mutations: dict[str, Any],
    folds: list[tuple[date, date]],
    *,
    research_snapshots: dict[date, OLRResearchSnapshot] | None = None,
) -> ResearchSweepResult:
    _validate_mutations([experiment], RESEARCH_MUTATION_KEYS)
    merged = dict(base_mutations)
    merged.update(experiment.mutations)
    snapshots = snapshots_for_experiment(dataset, merged, research_snapshots=research_snapshots)
    return _evaluate_snapshot_set(experiment, dataset, merged, snapshots, folds, objective="stage1_research_label")


def evaluate_stage2_experiment(
    experiment: Experiment,
    dataset: OLRResearchSweepDataset,
    stage1_mutations: dict[str, Any],
    folds: list[tuple[date, date]],
    *,
    stage1_snapshots: dict[date, OLRDailySnapshot] | None = None,
    stage1_afternoon_contexts: dict[date, dict[str, OLRAfternoonContext]] | None = None,
    research_snapshots: dict[date, OLRResearchSnapshot] | None = None,
    parent_stage1_name: str = "",
    parent_stage1_mutations: dict[str, Any] | None = None,
) -> ResearchSweepResult:
    _validate_mutations([experiment], AFTERNOON_MUTATION_KEYS)
    merged = dict(stage1_mutations)
    merged.update(experiment.mutations)
    stage1 = stage1_snapshots or snapshots_for_experiment(dataset, stage1_mutations, research_snapshots=research_snapshots)
    cfg = OLRConfig.from_mapping(dataset.config, merged)
    if stage1_afternoon_contexts is None:
        snapshots = {day: afternoon_selection_from_snapshot(snapshot, dataset.bars_by_key, cfg, sector_map=dataset.sector_map) for day, snapshot in stage1.items()}
    else:
        snapshots = {
            day: afternoon_selection_from_contexts(snapshot, stage1_afternoon_contexts.get(day, {}), cfg)
            for day, snapshot in stage1.items()
        }
    return _evaluate_snapshot_set(
        experiment,
        dataset,
        merged,
        snapshots,
        folds,
        objective="stage2_portfolio_proxy",
        parent_stage1_name=parent_stage1_name,
        parent_stage1_mutations=parent_stage1_mutations or {},
    )


def _evaluate_stage1_candidates(
    experiments: list[Experiment],
    dataset: OLRResearchSweepDataset,
    base_mutations: dict[str, Any],
    folds: list[tuple[date, date]],
    research_snapshots: dict[date, OLRResearchSnapshot],
    output_dir: Path,
    *,
    stage: str,
    completed_offset: int,
    total: int,
    max_workers: int,
    seed_rows: list[ResearchSweepResult] | None = None,
) -> list[ResearchSweepResult]:
    if not experiments:
        return []
    rows: list[ResearchSweepResult] = []
    progress_rows = list(seed_rows or [])
    worker_count = max(1, min(int(max_workers), 2))
    if worker_count <= 1 or len(experiments) == 1:
        for experiment in experiments:
            row = evaluate_stage1_experiment(experiment, dataset, base_mutations, folds, research_snapshots=research_snapshots)
            rows.append(row)
            progress_rows.append(row)
            _record_progress(output_dir, stage, completed_offset + len(rows), total, progress_rows, row)
        return rows
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(evaluate_stage1_experiment, experiment, dataset, base_mutations, folds, research_snapshots=research_snapshots): experiment
            for experiment in experiments
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            progress_rows.append(row)
            _record_progress(output_dir, stage, completed_offset + len(rows), total, progress_rows, row)
    return rows


def _evaluate_stage2_candidates(
    experiments: list[Experiment],
    dataset: OLRResearchSweepDataset,
    stage1_mutations: dict[str, Any],
    folds: list[tuple[date, date]],
    stage1_snapshots: dict[date, OLRDailySnapshot],
    stage1_afternoon_contexts: dict[date, dict[str, OLRAfternoonContext]],
    output_dir: Path,
    *,
    stage: str,
    max_workers: int,
    parent_stage1_name: str = "",
    parent_stage1_mutations: dict[str, Any] | None = None,
) -> list[ResearchSweepResult]:
    if not experiments:
        return []
    rows: list[ResearchSweepResult] = []
    worker_count = max(1, min(int(max_workers), 2))
    if worker_count <= 1 or len(experiments) == 1:
        for experiment in experiments:
            row = evaluate_stage2_experiment(
                experiment,
                dataset,
                stage1_mutations,
                folds,
                stage1_snapshots=stage1_snapshots,
                stage1_afternoon_contexts=stage1_afternoon_contexts,
                parent_stage1_name=parent_stage1_name,
                parent_stage1_mutations=parent_stage1_mutations or {},
            )
            rows.append(row)
            _record_progress(output_dir, stage, len(rows), len(experiments), rows, row)
        return rows
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                evaluate_stage2_experiment,
                experiment,
                dataset,
                stage1_mutations,
                folds,
                stage1_snapshots=stage1_snapshots,
                stage1_afternoon_contexts=stage1_afternoon_contexts,
                parent_stage1_name=parent_stage1_name,
                parent_stage1_mutations=parent_stage1_mutations or {},
            ): experiment
            for experiment in experiments
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            _record_progress(output_dir, stage, len(rows), len(experiments), rows, row)
    return rows


def snapshots_for_experiment(
    dataset: OLRResearchSweepDataset,
    mutations: dict[str, Any] | None = None,
    *,
    research_snapshots: dict[date, OLRResearchSnapshot] | None = None,
) -> dict[date, OLRDailySnapshot]:
    cfg = OLRConfig.from_mapping(dataset.config, mutations)
    snapshots: dict[date, OLRDailySnapshot] = {}
    for trade_date in dataset.trading_dates:
        if research_snapshots is None:
            research = _build_research_snapshot_for_date(dataset, trade_date, cfg)
        else:
            base = research_snapshots.get(trade_date)
            if base is None:
                continue
            research = _research_snapshot_with_config(base, cfg)
        snapshots[trade_date] = _with_prev_session_sector_intraday(
            daily_selection_from_snapshot(research, cfg),
            dataset,
        )
    return snapshots


def _with_prev_session_sector_intraday(
    snapshot: OLRDailySnapshot,
    dataset: OLRResearchSweepDataset,
) -> OLRDailySnapshot:
    panel = dataset.sector_intraday_panel_1430
    if panel is None:
        return snapshot
    prior_day = _previous_trading_date(dataset.trading_dates, snapshot.trade_date)
    if prior_day is None:
        return snapshot
    candidates = []
    for candidate in snapshot.candidates:
        feature = panel.feature_for(prior_day, candidate.symbol, sector=dataset.sector_map.get(candidate.symbol, candidate.sector))
        candidates.append(
            replace(
                candidate,
                metadata={
                    **dict(candidate.metadata),
                    **feature.metadata(prefix="prev_session"),
                    "prev_session_sector_intraday_cutoff": panel.cutoff_label,
                    "prev_session_sector_intraday_date": prior_day.isoformat(),
                },
            )
        )
    return replace(
        snapshot,
        candidates=tuple(candidates),
        metadata={
            **dict(snapshot.metadata),
            "prev_session_sector_intraday_cutoff": panel.cutoff_label,
            "prev_session_sector_intraday_version": panel.version,
        },
    )


def _previous_trading_date(trading_dates: Sequence[date], trade_date: date) -> date | None:
    prior = [day for day in trading_dates if day < trade_date]
    return prior[-1] if prior else None


def afternoon_contexts_for_snapshots(
    dataset: OLRResearchSweepDataset,
    snapshots: dict[date, OLRDailySnapshot],
    config: OLRConfig | None = None,
) -> dict[date, dict[str, OLRAfternoonContext]]:
    cfg = config or OLRConfig.from_mapping(dataset.config, {})
    return {day: build_afternoon_contexts(snapshot, dataset.bars_by_key, cfg, sector_map=dataset.sector_map) for day, snapshot in snapshots.items()}


def research_snapshots_for_dataset(
    dataset: OLRResearchSweepDataset,
    mutations: dict[str, Any] | None = None,
) -> dict[date, OLRResearchSnapshot]:
    cfg = OLRConfig.from_mapping(dataset.config, mutations)
    daily_lookback = max(100, int(cfg.research_min_history_days) + 5)
    return {
        trade_date: _compact_research_snapshot(
            _build_research_snapshot_for_date(dataset, trade_date, cfg, daily_lookback=daily_lookback),
            daily_lookback=daily_lookback,
        )
        for trade_date in dataset.trading_dates
    }


def _build_research_snapshot_for_date(
    dataset: OLRResearchSweepDataset,
    trade_date: date,
    config: OLRConfig,
    *,
    daily_lookback: int | None = None,
) -> OLRResearchSnapshot:
    metadata = {
        "requested_universe_count": len(dataset.requested_symbols),
        "complete_universe_count": len(dataset.symbols),
        "unavailable_symbols": list(dataset.excluded_symbols),
    }
    if daily_lookback is not None:
        metadata["research_snapshot_daily_lookback"] = int(daily_lookback)
        metadata["research_snapshot_flow_lookback"] = max(30, int(daily_lookback))
    return build_research_snapshot(
        dataset.daily_by_symbol,
        trade_date,
        config,
        sector_map=dataset.sector_map,
        flow_by_symbol=dataset.flow_by_symbol,
        foreign_flow_by_symbol=dataset.foreign_flow_by_symbol,
        institutional_flow_by_symbol=dataset.institutional_flow_by_symbol,
        index_ohlcv_by_symbol=dataset.index_by_code,
        source_fingerprint=dataset.source_fingerprint,
        metadata=metadata,
    )


def _research_snapshot_with_config(snapshot: OLRResearchSnapshot, config: OLRConfig) -> OLRResearchSnapshot:
    metadata = dict(snapshot.metadata)
    metadata["research_config_hash"] = _research_config_fingerprint(config)
    return replace(snapshot, metadata=metadata)


def _compact_research_snapshot(snapshot: OLRResearchSnapshot, *, daily_lookback: int) -> OLRResearchSnapshot:
    symbols = {
        symbol: replace(
            item,
            daily_rows=_compact_daily_rows(item.daily_rows, daily_lookback),
            flow_rows=_compact_flow_rows(item.flow_rows, daily_lookback),
            foreign_flow_rows=_compact_flow_rows(item.foreign_flow_rows, daily_lookback),
            institutional_flow_rows=_compact_flow_rows(item.institutional_flow_rows, daily_lookback),
        )
        for symbol, item in snapshot.symbols.items()
    }
    metadata = {
        **dict(snapshot.metadata),
        "fast_replay_cache_compacted": True,
        "fast_replay_daily_lookback": int(daily_lookback),
        "fast_replay_flow_rows_compacted": True,
    }
    return replace(snapshot, symbols=symbols, metadata=metadata)


def _compact_daily_rows(rows: tuple[dict[str, Any], ...], lookback: int) -> tuple[dict[str, Any], ...]:
    keys = ("date", "trade_date", "timestamp", "open", "high", "low", "close", "volume")
    compacted = []
    for row in tuple(rows or ())[-max(1, int(lookback)) :]:
        compacted.append({key: row.get(key) for key in keys if key in row})
    return tuple(compacted)


def _compact_flow_rows(rows: tuple[dict[str, Any], ...], lookback: int) -> tuple[dict[str, Any], ...]:
    keys = ("date", "trade_date", "timestamp", "foreign_net", "institutional_net", "inst_net")
    compacted = []
    for row in tuple(rows or ())[-max(1, int(lookback)) :]:
        compacted.append({key: row.get(key) for key in keys if key in row})
    return tuple(compacted)


def overnight_metrics_from_snapshots(
    dataset: OLRResearchSweepDataset,
    snapshots: dict[date, OLRDailySnapshot],
    config: OLRConfig,
) -> dict[str, float]:
    rows: list[OvernightOpportunity] = []
    slot = max(1, int(config.overnight_slot_count))
    for session, snapshot in sorted(snapshots.items()):
        for rank, candidate in enumerate([item for item in snapshot.candidates if item.tradable][:slot], start=1):
            rows.append(_overnight_opportunity(dataset, candidate.symbol, session, rank=rank, atr=float(candidate.daily_atr), config=config))
    valid = [row for row in rows if row.valid]
    active_days = {row.trade_date for row in valid}
    active_good = [row for row in valid if row.mfe_r >= 0.75]
    daily_returns = []
    by_day: dict[date, list[OvernightOpportunity]] = {}
    for row in valid:
        by_day.setdefault(row.trade_date, []).append(row)
    for day in sorted(snapshots):
        day_rows = by_day.get(day, [])
        daily_returns.append(sum(row.net_return_pct for row in day_rows) / slot)
    avg_net = _avg(row.net_return_pct for row in valid)
    avg_mfe = _avg(row.mfe_r for row in valid)
    return_component = 100.0 * _return_score(avg_net, target=0.004)
    mfe_component = 100.0 * _clip(avg_mfe / 2.0)
    hit_component = 100.0 * _clip(_ratio(len(active_good), len(valid)) / 0.50)
    active_component = 100.0 * _clip(_ratio(len(active_days), len(snapshots)) / 0.70)
    downside_component = 100.0 * (
        0.5 * (1.0 - _clip(_ratio(sum(1 for row in valid if row.mae_r <= -1.0), len(valid)) / 0.50))
        + 0.5 * (1.0 - _clip(_ratio(sum(1 for row in valid if row.mfe_r < 0.10), len(valid)) / 0.25))
    )
    score = 0.40 * return_component + 0.30 * mfe_component + 0.10 * hit_component + 0.10 * active_component + 0.10 * downside_component
    return {
        "snapshot_count": float(len(snapshots)),
        "candidate_days": float(len(rows)),
        "valid_candidate_days": float(len(valid)),
        "active_days": float(len(active_days)),
        "active_day_share": _ratio(len(active_days), len(snapshots)),
        "avg_active_net_return_pct": avg_net,
        "avg_active_gross_return_pct": _avg(row.gross_return_pct for row in valid),
        "calendar_slot_net_return_pct": _avg(daily_returns),
        "slot_cumulative_net_return_pct": _compound(daily_returns),
        "avg_active_mfe_r": avg_mfe,
        "avg_active_mfe_pct": _avg(row.mfe_pct for row in valid),
        "avg_active_mae_r": _avg(row.mae_r for row in valid),
        "active_mfe_ge_0_75_share": _ratio(len(active_good), len(valid)),
        "active_bad_mae_le_neg_1_share": _ratio(sum(1 for row in valid if row.mae_r <= -1.0), len(valid)),
        "active_low_mfe_lt_0_1_share": _ratio(sum(1 for row in valid if row.mfe_r < 0.10), len(valid)),
        "overnight_component_return": return_component,
        "overnight_component_mfe": mfe_component,
        "overnight_component_mfe_hit_rate": hit_component,
        "overnight_component_active_day": active_component,
        "overnight_component_downside": downside_component,
        "overnight_score": max(0.0, score),
    }


def stage2_portfolio_metrics_from_snapshots(
    dataset: OLRResearchSweepDataset,
    snapshots: dict[date, OLRDailySnapshot],
    config: OLRConfig,
) -> dict[str, float]:
    slot = max(1, int(config.overnight_slot_count))
    outcomes: list[OLRTradeOutcome] = []
    selection_counts: dict[date, int] = {}
    candidate_counts: list[int] = []
    for session, snapshot in sorted(snapshots.items()):
        tradable = [item for item in snapshot.candidates if item.tradable][:slot]
        selection_counts[session] = len(tradable)
        candidate_counts.append(len(tradable))
        for rank, candidate in enumerate(tradable, start=1):
            outcome = _fixed_close_auction_outcome(
                dataset,
                candidate.symbol,
                session,
                rank=rank,
                atr=float(candidate.daily_atr),
                score=float(candidate.selection_score or 0.0),
                sector=str(candidate.sector or "UNKNOWN"),
                config=config,
            )
            if outcome is not None:
                outcomes.append(outcome)
    allocation = _stage2_portfolio_proxy_allocation()
    metrics = summarize_olr_portfolio_proxy(
        outcomes,
        session_dates=tuple(sorted(snapshots)),
        selection_counts=selection_counts,
        slot_count=slot,
        allocation=allocation,
        initial_equity=float(dataset.config.get("initial_equity", 10_000_000.0) or 10_000_000.0),
        config=config,
    )
    active_counts = [value for value in candidate_counts if value > 0]
    too_many_penalty = _clip(max(0.0, _avg(active_counts) - 4.0) / 8.0)
    metrics.update(
        {
            "stage2_portfolio_proxy_version": 1.0,
            "stage2_portfolio_proxy_timing_policy": STAGE2_PORTFOLIO_PROXY_VERSION,
            "stage2_selector_cutoff_time": 14.30,
            "stage2_proxy_entry_time": 15.30,
            "stage2_proxy_exit_time": 15.30,
            "stage2_proxy_target_gross_exposure": float(allocation.target_gross_exposure),
            "stage2_proxy_max_position_pct": float(allocation.max_position_pct),
            "stage2_proxy_min_selected": float(allocation.min_selected),
            "stage2_avg_selected_per_session": _avg(candidate_counts),
            "stage2_avg_selected_per_active_day": _avg(active_counts),
            "stage2_too_many_names_penalty": too_many_penalty,
            **_stage2_score_band_metrics(outcomes),
        }
    )
    return metrics


def _stage2_portfolio_proxy_allocation() -> OLRAllocationPlan:
    return OLRAllocationPlan(
        "stage2_capped_equal_cap0p5_min2",
        mode="capped_equal",
        target_gross_exposure=1.0,
        max_position_pct=0.50,
        min_selected=2,
    )


def _fixed_close_auction_outcome(
    dataset: OLRResearchSweepDataset,
    symbol: str,
    trade_date: date,
    *,
    rank: int,
    atr: float,
    score: float = 0.0,
    sector: str = "UNKNOWN",
    config: OLRConfig,
) -> OLRTradeOutcome | None:
    label = dataset.overnight_labels_by_key.get((trade_date, symbol))
    if label is None:
        day_row = _row_on(dataset.daily_by_symbol.get(symbol, []), trade_date)
        next_row = _next_row(dataset.daily_by_symbol.get(symbol, []), trade_date)
        if day_row is None or next_row is None:
            return None
        entry = max(_float(day_row.get("close")), 1e-9)
        close = _float(next_row.get("close"))
        high = _float(next_row.get("high"), close)
        low = _float(next_row.get("low"), close)
        next_date = _row_date(next_row)
    else:
        entry = max(float(label.entry_close), 1e-9)
        close = float(label.next_close)
        high = float(label.next_high)
        low = float(label.next_low)
        next_date = label.trade_date if label.trade_date > trade_date else _next_trading_date(dataset, trade_date)
    if entry <= 0.0 or close <= 0.0:
        return None
    risk = max(float(atr) * 0.50, entry * 0.01, 1.0)
    gross = close / entry - 1.0
    net = gross - _round_trip_cost_pct(config)
    entry_time = datetime.combine(trade_date, datetime.min.time(), tzinfo=KST).replace(hour=15, minute=30)
    exit_time = datetime.combine(next_date, datetime.min.time(), tzinfo=KST).replace(hour=15, minute=30)
    return OLRTradeOutcome(
        trade_date=trade_date,
        symbol=symbol,
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=entry,
        exit_price=close,
        stop_price=max(0.0, entry - risk),
        risk_per_share=risk,
        gross_return_pct=gross,
        net_return_pct=net,
        mfe_r=max(0.0, (high - entry) / risk),
        mae_r=(low - entry) / risk,
        mfe_capture=_ratio(net, max(high / entry - 1.0, 1e-9)),
        bars_held=0,
        entry_reason="stage2_fixed_close_auction_proxy",
        exit_reason="stage2_fixed_next_close_proxy",
        ambiguous_bar_count=0,
        stopped=False,
        target_hit=False,
        partial_hit=False,
        metadata={
            "candidate_rank": int(rank),
            "candidate_score": float(score),
            "candidate_sector": str(sector or "UNKNOWN"),
            "metric_basis": STAGE2_PORTFOLIO_PROXY_VERSION,
        },
    )


def _stage2_score_band_metrics(outcomes: Sequence[OLRTradeOutcome]) -> dict[str, float]:
    scored = [
        outcome
        for outcome in outcomes
        if math.isfinite(float((outcome.metadata or {}).get("candidate_score", 0.0) or 0.0))
    ]
    total = len(scored)
    if total < 4:
        return {
            "stage2_score_band_sample_count": float(total),
            "stage2_score_monotonicity": 0.0,
            "stage2_score_bottom_quartile_net_pct": 0.0,
            "stage2_score_mid_half_net_pct": 0.0,
            "stage2_score_top_quartile_net_pct": 0.0,
            "stage2_score_top_loss_share": 0.0,
            "stage2_negative_selected_share": _ratio(sum(1 for item in scored if item.net_return_pct <= 0.0), max(total, 1)),
            "stage2_alpha_capture_pct": 0.0,
        }
    scored.sort(key=lambda item: float((item.metadata or {}).get("candidate_score", 0.0) or 0.0))
    bucket_count = min(4, total)
    buckets: list[list[OLRTradeOutcome]] = []
    for bucket_index in range(bucket_count):
        start = int(round(bucket_index * total / bucket_count))
        end = int(round((bucket_index + 1) * total / bucket_count))
        buckets.append(scored[start:end])
    bucket_nets = [_avg(item.net_return_pct for item in bucket) for bucket in buckets]
    monotone_steps = sum(1 for left, right in zip(bucket_nets, bucket_nets[1:]) if right >= left)
    monotonicity = monotone_steps / max(len(bucket_nets) - 1, 1)
    top = buckets[-1]
    mid = [item for bucket in buckets[1:-1] for item in bucket]
    alpha_available = sum(max(0.0, (item.mfe_r * item.risk_per_share) / max(item.entry_price, 1e-9)) for item in scored)
    alpha_realized = sum(item.net_return_pct for item in scored)
    return {
        "stage2_score_band_sample_count": float(total),
        "stage2_score_monotonicity": float(monotonicity),
        "stage2_score_bottom_quartile_net_pct": float(bucket_nets[0]),
        "stage2_score_mid_half_net_pct": _avg(item.net_return_pct for item in mid),
        "stage2_score_top_quartile_net_pct": float(bucket_nets[-1]),
        "stage2_score_top_loss_share": _ratio(sum(1 for item in top if item.net_return_pct <= 0.0), len(top)),
        "stage2_negative_selected_share": _ratio(sum(1 for item in scored if item.net_return_pct <= 0.0), total),
        "stage2_alpha_capture_pct": _ratio(alpha_realized, alpha_available),
    }


def _next_trading_date(dataset: OLRResearchSweepDataset, trade_date: date) -> date:
    dates = list(dataset.trading_dates)
    try:
        index = dates.index(trade_date)
    except ValueError:
        return trade_date + timedelta(days=1)
    if index + 1 < len(dates):
        return dates[index + 1]
    return trade_date + timedelta(days=1)


def score_overnight_metrics(metrics: dict[str, float]) -> tuple[float, str]:
    snapshots = float(metrics.get("snapshot_count", 0.0) or 0.0)
    valid = float(metrics.get("valid_candidate_days", 0.0) or 0.0)
    min_snapshots = 1.0 if snapshots <= 2.0 else 50.0
    min_valid = 1.0 if snapshots <= 2.0 else 60.0
    if snapshots < min_snapshots:
        return 0.0, f"too_few_snapshots ({snapshots:.0f} < {min_snapshots:.0f})"
    if valid < min_valid:
        return 0.0, f"too_few_valid_candidate_days ({valid:.0f} < {min_valid:.0f})"
    if snapshots > 2.0 and float(metrics.get("active_day_share", 0.0) or 0.0) < 0.20:
        return 0.0, "too_sparse"
    if float(metrics.get("avg_active_mfe_r", 0.0) or 0.0) < 0.05:
        return 0.0, "too_low_candidate_mfe"
    return max(0.0, float(metrics.get("overnight_score", 0.0) or 0.0)), ""


def score_stage2_portfolio_metrics(metrics: dict[str, float]) -> tuple[float, str]:
    base_score, base_reject = score_overnight_metrics(metrics)
    if base_reject:
        return 0.0, base_reject
    proxy_net = float(metrics.get("portfolio_proxy_net_return_pct", 0.0) or 0.0)
    if proxy_net <= 0.0:
        return 0.0, "non_positive_stage2_portfolio_proxy"
    active_net = float(metrics.get("portfolio_proxy_active_day_net_pct", 0.0) or 0.0)
    active_gross = float(metrics.get("portfolio_proxy_avg_active_gross_exposure_pct", 0.0) or 0.0)
    if active_gross < 0.45:
        return 0.0, "underdeployed_stage2_portfolio_proxy"
    drawdown = abs(float(metrics.get("portfolio_proxy_max_drawdown_pct", 0.0) or 0.0))
    band_sample = float(metrics.get("stage2_score_band_sample_count", 0.0) or 0.0)
    top_band = float(metrics.get("stage2_score_top_quartile_net_pct", 0.0) or 0.0)
    mid_band = float(metrics.get("stage2_score_mid_half_net_pct", 0.0) or 0.0)
    top_loss = float(metrics.get("stage2_score_top_loss_share", 0.0) or 0.0)
    if band_sample >= 80.0 and top_band <= 0.0 and top_loss >= 0.55:
        return 0.0, "stage2_top_score_band_negative"
    discrimination_quality = (
        0.45 * _clip(float(metrics.get("stage2_score_monotonicity", 0.0) or 0.0))
        + 0.25 * (1.0 - _clip(top_loss))
        + 0.20 * (1.0 - _clip(float(metrics.get("stage2_negative_selected_share", 0.0) or 0.0)))
        + 0.10 * _clip(float(metrics.get("stage2_alpha_capture_pct", 0.0) or 0.0) / 0.55)
    )
    if band_sample >= 80.0 and top_band < mid_band:
        discrimination_quality *= 0.80
    components = {
        "compound": 100.0 * _clip((proxy_net + 0.05) / 1.05),
        "active_return": 100.0 * _return_score(active_net, target=0.004),
        "opportunity": 100.0 * _clip(float(metrics.get("avg_active_mfe_r", 0.0) or 0.0) / 2.0),
        "frequency": 100.0 * _clip(float(metrics.get("active_day_share", 0.0) or 0.0) / 0.70),
        "downside": 100.0 * (1.0 - _clip(drawdown / 0.30)),
        "breadth": 100.0 * (1.0 - float(metrics.get("stage2_too_many_names_penalty", 0.0) or 0.0)),
        "discrimination": 100.0 * discrimination_quality,
    }
    score = (
        0.35 * components["compound"]
        + 0.12 * components["active_return"]
        + 0.13 * components["opportunity"]
        + 0.10 * components["frequency"]
        + 0.10 * components["downside"]
        + 0.05 * components["breadth"]
        + 0.15 * components["discrimination"]
    )
    return max(0.0, min(100.0, 0.82 * score + 0.18 * base_score)), ""


def _evaluate_snapshot_set(
    experiment: Experiment,
    dataset: OLRResearchSweepDataset,
    mutations: dict[str, Any],
    snapshots: dict[date, OLRDailySnapshot],
    folds: list[tuple[date, date]],
    *,
    objective: str,
    parent_stage1_name: str = "",
    parent_stage1_mutations: dict[str, Any] | None = None,
) -> ResearchSweepResult:
    cfg = OLRConfig.from_mapping(dataset.config, mutations)
    full_metrics = overnight_metrics_from_snapshots(dataset, snapshots, cfg)
    if objective == "stage2_portfolio_proxy":
        full_metrics.update(stage2_portfolio_metrics_from_snapshots(dataset, snapshots, cfg))
        full_score, reject_reason = score_stage2_portfolio_metrics(full_metrics)
    else:
        full_score, reject_reason = score_overnight_metrics(full_metrics)
    fold_rows = []
    for index, (start, end) in enumerate(folds, start=1):
        fold_snapshots = {day: snapshot for day, snapshot in snapshots.items() if start <= day <= end}
        fold_metrics = overnight_metrics_from_snapshots(dataset, fold_snapshots, cfg)
        if objective == "stage2_portfolio_proxy":
            fold_metrics.update(stage2_portfolio_metrics_from_snapshots(dataset, fold_snapshots, cfg))
            fold_score, fold_reject = score_stage2_portfolio_metrics(fold_metrics)
        else:
            fold_score, fold_reject = score_overnight_metrics(fold_metrics)
        fold_rows.append(
            {
                "fold": index,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "score": round(fold_score, 6),
                "rejected": bool(fold_reject),
                "reject_reason": fold_reject,
                "metrics": _compact_metrics(fold_metrics),
            }
        )
    fold_scores = [float(row["score"]) for row in fold_rows]
    median_fold = float(median(fold_scores)) if fold_scores else full_score
    worst_fold = float(min(fold_scores)) if fold_scores else full_score
    stability_score = 0.55 * full_score + 0.30 * median_fold + 0.15 * worst_fold
    rejected = bool(reject_reason) or (len(fold_rows) >= 2 and sum(1 for row in fold_rows if row["rejected"]) > len(fold_rows) // 2)
    if rejected and not reject_reason:
        reject_reason = "unstable_across_folds"
    return ResearchSweepResult(
        experiment=experiment,
        score=round(0.0 if rejected else stability_score, 6),
        full_score=round(full_score, 6),
        median_fold_score=round(median_fold, 6),
        worst_fold_score=round(worst_fold, 6),
        rejected=rejected,
        reject_reason=reject_reason,
        metrics=_compact_metrics(full_metrics),
        folds=tuple(fold_rows),
        artifact_hash=_snapshot_set_hash(snapshots),
        objective=objective,
        parent_stage1_name=parent_stage1_name,
        parent_stage1_mutations=dict(parent_stage1_mutations or {}),
    )


def _overnight_opportunity(
    dataset: OLRResearchSweepDataset,
    symbol: str,
    trade_date: date,
    *,
    rank: int,
    atr: float,
    config: OLRConfig,
) -> OvernightOpportunity:
    label = dataset.overnight_labels_by_key.get((trade_date, symbol))
    if label is None:
        day_row = _row_on(dataset.daily_by_symbol.get(symbol, []), trade_date)
        next_row = _next_row(dataset.daily_by_symbol.get(symbol, []), trade_date)
        if day_row is None or next_row is None:
            return OvernightOpportunity(trade_date=trade_date, symbol=symbol, rank=rank, valid=False)
        entry = max(_float(day_row.get("close")), 1e-9)
        close = _float(next_row.get("close"))
        high = _float(next_row.get("high"), close)
        low = _float(next_row.get("low"), close)
    else:
        entry = max(float(label.entry_close), 1e-9)
        close = float(label.next_close)
        high = float(label.next_high)
        low = float(label.next_low)
    if entry <= 0.0 or close <= 0.0:
        return OvernightOpportunity(trade_date=trade_date, symbol=symbol, rank=rank, valid=False)
    risk = max(float(atr) * 0.50, entry * 0.01, 1.0)
    gross = close / entry - 1.0
    cost = _round_trip_cost_pct(config)
    return OvernightOpportunity(
        trade_date=trade_date,
        symbol=symbol,
        rank=rank,
        valid=True,
        gross_return_pct=gross,
        net_return_pct=gross - cost,
        mfe_r=max(0.0, (high - entry) / risk),
        mfe_pct=high / entry - 1.0,
        mae_r=(low - entry) / risk,
    )


def _overnight_label_cache(
    daily_by_symbol: dict[str, list[dict[str, Any]]],
    symbols: tuple[str, ...],
    trading_dates: tuple[date, ...],
) -> dict[tuple[date, str], OvernightLabel]:
    wanted = set(trading_dates)
    out: dict[tuple[date, str], OvernightLabel] = {}
    for symbol in symbols:
        dated = [(_row_date(row), row) for row in daily_by_symbol.get(symbol, []) if _try_row_date(row) is not None]
        dated.sort(key=lambda item: item[0])
        for index, (day, row) in enumerate(dated[:-1]):
            if day not in wanted:
                continue
            next_row = dated[index + 1][1]
            close = _float(next_row.get("close"))
            entry = _float(row.get("close"))
            if entry <= 0.0 or close <= 0.0:
                continue
            out[(day, symbol)] = OvernightLabel(
                trade_date=day,
                symbol=symbol,
                entry_close=entry,
                next_close=close,
                next_high=_float(next_row.get("high"), close),
                next_low=_float(next_row.get("low"), close),
            )
    return out


def _row_on(rows: list[dict[str, Any]], target: date) -> dict[str, Any] | None:
    for row in rows:
        if _row_date(row) == target:
            return row
    return None


def _next_row(rows: list[dict[str, Any]], target: date) -> dict[str, Any] | None:
    dated = sorted((_row_date(row), row) for row in rows if _try_row_date(row) is not None)
    for day, row in dated:
        if day > target:
            return row
    return None


def _training_config(config: dict[str, Any], holdout_days: int) -> dict[str, Any]:
    out = dict(config)
    out["holdout_days"] = int(holdout_days)
    out["use_full_available_window"] = False
    out.pop("end", None)
    out.pop("holdout_start", None)
    date_range = dict(out.get("date_range") or {})
    date_range.pop("end", None)
    if date_range:
        out["date_range"] = date_range
    else:
        out.pop("date_range", None)
    return out


def _base_mutations(config: dict[str, Any], mutations: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(config.get("initial_mutations"), dict):
        out.update(config["initial_mutations"])
    out.update(dict(mutations or {}))
    return out


def _resolve_folds(dates: list[date], *, fold_days: int | None, fold_count: int) -> list[tuple[date, date]]:
    ordered = sorted(set(dates))
    if not ordered or fold_count <= 0:
        return []
    if fold_days is not None and int(fold_days) > 0:
        folds = []
        start = ordered[0]
        latest = ordered[-1]
        while start <= latest:
            end = min(start + timedelta(days=int(fold_days) - 1), latest)
            members = [day for day in ordered if start <= day <= end]
            if members:
                folds.append((members[0], members[-1]))
            start = end + timedelta(days=1)
        return folds
    if fold_count == 1:
        return [(ordered[0], ordered[-1])]
    folds: list[tuple[date, date]] = []
    total = len(ordered)
    for index in range(int(fold_count)):
        start_idx = round(index * total / int(fold_count))
        end_idx = round((index + 1) * total / int(fold_count)) - 1
        if start_idx < total:
            folds.append((ordered[start_idx], ordered[min(max(end_idx, start_idx), total - 1)]))
    return folds


def _requested_symbols(config: dict[str, Any], data_root: Path, timeframe: str) -> list[str]:
    raw = config.get("universe_file") or config.get("universe") or config.get("symbols")
    if isinstance(raw, (list, tuple)):
        return sorted(str(symbol).zfill(6) for symbol in raw)
    if raw:
        path = Path(str(raw))
        if not path.exists():
            path = Path("config") / str(raw)
        if not path.exists():
            raise FileNotFoundError(f"OLR universe file not found: {raw}")
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        values = (payload.get("symbols") or payload.get("universe")) if isinstance(payload, dict) else payload
        return sorted(str(symbol).zfill(6) for symbol in values)
    return sorted(path.parent.name for path in Path(data_root).glob(f"*/*_{timeframe}_*.parquet"))


def _has_intraday_data(data_root: Path, symbol: str, timeframe: str) -> bool:
    return any(Path(data_root).glob(f"{symbol}/{symbol}_{timeframe}_*.parquet"))


def _load_daily_rows(root: Path, symbols: Iterable[str], end: date) -> dict[str, list[dict[str, Any]]]:
    return {symbol: _frame_rows(load_daily_ohlcv(root, symbol, end=end)) for symbol in symbols}


def _load_flow_rows(root: Path, symbols: Iterable[str], end: date) -> dict[str, list[dict[str, Any]]]:
    return {symbol: _frame_rows(load_daily_flow(root, symbol, end=end)) for symbol in symbols}


def _load_foreign_flow_rows(root: Path, symbols: Iterable[str], end: date) -> dict[str, list[dict[str, Any]]]:
    return {symbol: _frame_rows(load_daily_foreign_flow(root, symbol, end=end)) for symbol in symbols}


def _load_institutional_flow_rows(root: Path, symbols: Iterable[str], end: date) -> dict[str, list[dict[str, Any]]]:
    return {symbol: _frame_rows(load_daily_institutional_flow(root, symbol, end=end)) for symbol in symbols}


def _load_index_rows(root: Path, end: date) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for code in ("KOSPI", "KOSDAQ"):
        rows = _frame_rows(load_index_ohlcv(root, code, end=end))
        if rows:
            out[code] = rows
    return out


def _frame_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    return [dict(row._asdict()) for row in frame.sort_values("date").itertuples(index=False)]


def _bars_by_key_from_frames(
    frames: dict[str, pd.DataFrame],
    start: date,
    end: date,
    *,
    source_fingerprint: str,
) -> dict[tuple[date, str], tuple[MarketBar, ...]]:
    grouped: dict[tuple[date, str], list[MarketBar]] = {}
    for symbol, frame in frames.items():
        if frame.empty:
            continue
        sliced = frame[(frame["timestamp"].dt.date >= start) & (frame["timestamp"].dt.date <= end)]
        for row in sliced.itertuples(index=False):
            ts = row.timestamp.to_pydatetime() if hasattr(row.timestamp, "to_pydatetime") else row.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=KST)
            bar = MarketBar(
                symbol=symbol,
                timestamp=ts.astimezone(KST),
                timeframe="5m",
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                is_completed=True,
                source="kis_krx_parquet",
                source_fingerprint=source_fingerprint,
            )
            grouped.setdefault((bar.timestamp.date(), symbol), []).append(bar)
    return {key: tuple(sorted(values, key=lambda bar: bar.timestamp)) for key, values in grouped.items()}


def _trading_dates(frames: dict[str, pd.DataFrame], start: date, end: date) -> list[date]:
    dates = set()
    for frame in frames.values():
        if frame.empty:
            continue
        for day in frame["timestamp"].dt.date.unique():
            if start <= day <= end:
                dates.add(day)
    return sorted(dates)


def _daily_source_fingerprint(root: Path, symbols: tuple[str, ...], end: date) -> str:
    paths = []
    for table in ("daily_ohlcv", "daily_flow", "daily_foreign_flow", "daily_institutional_flow"):
        for symbol in symbols:
            paths.extend((Path(root) / table / symbol).glob("*.parquet"))
    paths.extend((Path(root) / "index_ohlcv").glob("*/*.parquet"))
    paths.extend((Path(root) / "tables").glob("*.parquet"))
    return stable_signature({"mode": "olr_daily_sources", "paths": fingerprint_paths(paths, root=root), "symbols": symbols, "end": end.isoformat()})


def _intraday_source_fingerprint(root: Path, symbols: tuple[str, ...], timeframe: str, start: date, end: date) -> str:
    paths = []
    for symbol in symbols:
        paths.extend((Path(root) / symbol).glob(f"{symbol}_{timeframe}_*.parquet"))
    return stable_signature({"mode": "olr_intraday_sources", "paths": fingerprint_paths(paths, root=root), "symbols": symbols, "start": start.isoformat(), "end": end.isoformat()})


def _coverage_report(
    symbols: tuple[str, ...],
    excluded: dict[str, str],
    daily_by_symbol: dict[str, list[dict[str, Any]]],
    flow_by_symbol: dict[str, list[dict[str, Any]]],
    foreign_flow_by_symbol: dict[str, list[dict[str, Any]]],
    institutional_flow_by_symbol: dict[str, list[dict[str, Any]]],
    sector_map: dict[str, str],
    bars_by_key: dict[tuple[date, str], tuple[MarketBar, ...]],
    trading_dates: tuple[date, ...],
) -> dict[str, Any]:
    return {
        "complete_symbols": len(symbols),
        "excluded_symbols": excluded,
        "daily_ohlcv_symbols": sum(1 for symbol in symbols if daily_by_symbol.get(symbol)),
        "daily_flow_symbols": sum(1 for symbol in symbols if flow_by_symbol.get(symbol)),
        "daily_foreign_flow_symbols": sum(1 for symbol in symbols if foreign_flow_by_symbol.get(symbol)),
        "daily_institutional_flow_symbols": sum(1 for symbol in symbols if institutional_flow_by_symbol.get(symbol)),
        "sector_map_symbols": sum(1 for symbol in symbols if symbol in sector_map),
        "intraday_symbol_sessions": float(len(bars_by_key)),
        "training_sessions": len(trading_dates),
    }


def _snapshot_set_hash(snapshots: dict[date, OLRDailySnapshot]) -> str:
    return stable_signature({day.isoformat(): snapshot.artifact_hash for day, snapshot in sorted(snapshots.items())})


def _audit_stage1_result(
    fast_row: ResearchSweepResult,
    dataset: OLRResearchSweepDataset,
    base_mutations: dict[str, Any],
    folds: list[tuple[date, date]],
    *,
    research_snapshots: dict[date, OLRResearchSnapshot] | None = None,
    tolerance: float,
) -> dict[str, Any]:
    full_row = evaluate_stage1_experiment(fast_row.experiment, dataset, base_mutations, folds, research_snapshots=research_snapshots)
    return _audit_result_payload("stage1", fast_row, full_row, tolerance=tolerance)


def _audit_stage2_results(
    fast_rows: list[ResearchSweepResult],
    dataset: OLRResearchSweepDataset,
    base_mutations: dict[str, Any],
    folds: list[tuple[date, date]],
    *,
    research_snapshots: dict[date, OLRResearchSnapshot] | None = None,
    tolerance: float,
) -> list[dict[str, Any]]:
    out = []
    stage1_cache: dict[str, tuple[dict[date, OLRDailySnapshot], dict[date, dict[str, OLRAfternoonContext]]]] = {}
    for row in fast_rows:
        stage1_mutations = dict(base_mutations)
        stage1_mutations.update(dict(row.parent_stage1_mutations or {}))
        stage1_key = stable_signature(stage1_mutations)
        if stage1_key not in stage1_cache:
            full_stage1_snapshots = snapshots_for_experiment(dataset, stage1_mutations, research_snapshots=research_snapshots)
            full_contexts = afternoon_contexts_for_snapshots(
                dataset,
                full_stage1_snapshots,
                OLRConfig.from_mapping(dataset.config, stage1_mutations),
            )
            stage1_cache[stage1_key] = (full_stage1_snapshots, full_contexts)
        full_stage1_snapshots, full_contexts = stage1_cache[stage1_key]
        full_row = evaluate_stage2_experiment(
            row.experiment,
            dataset,
            stage1_mutations,
            folds,
            stage1_snapshots=full_stage1_snapshots,
            stage1_afternoon_contexts=full_contexts,
            parent_stage1_name=row.parent_stage1_name,
            parent_stage1_mutations=row.parent_stage1_mutations,
        )
        out.append(_audit_result_payload("stage2", row, full_row, tolerance=tolerance))
    return out


def _audit_result_payload(
    stage: str,
    fast_row: ResearchSweepResult,
    full_row: ResearchSweepResult,
    *,
    tolerance: float,
) -> dict[str, Any]:
    metric_keys = sorted(set(fast_row.metrics) | set(full_row.metrics))
    deltas = {
        key: float(full_row.metrics.get(key, 0.0) or 0.0) - float(fast_row.metrics.get(key, 0.0) or 0.0)
        for key in metric_keys
    }
    score_deltas = {
        "score": float(full_row.score) - float(fast_row.score),
        "full_score": float(full_row.full_score) - float(fast_row.full_score),
        "median_fold_score": float(full_row.median_fold_score) - float(fast_row.median_fold_score),
        "worst_fold_score": float(full_row.worst_fold_score) - float(fast_row.worst_fold_score),
    }
    max_abs_metric_delta = max((abs(value) for value in deltas.values()), default=0.0)
    max_abs_score_delta = max((abs(value) for value in score_deltas.values()), default=0.0)
    artifact_match = fast_row.artifact_hash == full_row.artifact_hash
    rejection_match = fast_row.rejected == full_row.rejected and fast_row.reject_reason == full_row.reject_reason
    audit_pass = (
        artifact_match
        and rejection_match
        and max_abs_metric_delta <= float(tolerance)
        and max_abs_score_delta <= float(tolerance)
    )
    return {
        "stage": stage,
        "name": fast_row.experiment.name,
        "objective": fast_row.objective,
        "stage1_seed": fast_row.parent_stage1_name,
        "audit_pass": audit_pass,
        "artifact_hash_match": artifact_match,
        "rejection_match": rejection_match,
        "fast_artifact_hash": fast_row.artifact_hash,
        "full_artifact_hash": full_row.artifact_hash,
        "max_abs_metric_delta": max_abs_metric_delta,
        "max_abs_score_delta": max_abs_score_delta,
        "metric_deltas": deltas,
        "score_deltas": score_deltas,
        "fast_score": fast_row.score,
        "full_score": full_row.score,
        "scope": "fast cache suppresses raw row-slice rebuild work only; selected artifacts and research/portfolio-proxy metrics must match full rebuild",
    }


def _compact_metrics(metrics: dict[str, float]) -> dict[str, float]:
    keep = (
        "snapshot_count",
        "candidate_days",
        "valid_candidate_days",
        "active_days",
        "active_day_share",
        "avg_active_net_return_pct",
        "calendar_slot_net_return_pct",
        "slot_cumulative_net_return_pct",
        "avg_active_mfe_r",
        "avg_active_mfe_pct",
        "avg_active_mae_r",
        "active_mfe_ge_0_75_share",
        "active_bad_mae_le_neg_1_share",
        "active_low_mfe_lt_0_1_share",
        "overnight_score",
        "portfolio_proxy_daily_net_pct",
        "portfolio_proxy_net_return_pct",
        "portfolio_proxy_max_drawdown_pct",
        "portfolio_proxy_avg_gross_exposure_pct",
        "portfolio_proxy_avg_active_gross_exposure_pct",
        "portfolio_proxy_active_day_net_pct",
        "portfolio_proxy_qty_zero_count",
        "stage2_proxy_target_gross_exposure",
        "stage2_proxy_max_position_pct",
        "stage2_proxy_min_selected",
        "stage2_avg_selected_per_session",
        "stage2_avg_selected_per_active_day",
        "stage2_too_many_names_penalty",
        "stage2_score_band_sample_count",
        "stage2_score_monotonicity",
        "stage2_score_bottom_quartile_net_pct",
        "stage2_score_mid_half_net_pct",
        "stage2_score_top_quartile_net_pct",
        "stage2_score_top_loss_share",
        "stage2_negative_selected_share",
        "stage2_alpha_capture_pct",
    )
    return {key: _float(metrics.get(key)) for key in keep if key in metrics}


def _load_stage1_resume_bundle(path: str | Path, *, stage1_stage2_seed_count: int) -> Stage1ResumeBundle:
    source_path = Path(path).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"OLR Stage 1 resume artifact not found: {source_path}")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if str(payload.get("strategy", "olr")).lower() != "olr":
        raise ValueError(f"Stage 1 resume artifact must be for OLR: {source_path}")

    seed_payloads = list(payload.get("stage1_stage2_seeds") or ())
    if not seed_payloads and isinstance(payload.get("selected_stage1_seed"), dict):
        seed_payloads = [payload["selected_stage1_seed"]]
    if not seed_payloads:
        raise ValueError(f"OLR Stage 1 resume artifact has no stage1_stage2_seeds or selected_stage1_seed: {source_path}")

    limit = max(1, int(stage1_stage2_seed_count))
    stage1_stage2_seeds = tuple(_result_from_payload(row, default_objective="stage1_research_label") for row in seed_payloads[:limit])
    frontier_payloads = list(payload.get("stage1_frontier") or ())
    stage1_rows = _dedupe_results(
        [
            *[_result_from_payload(row, default_objective="stage1_research_label") for row in frontier_payloads],
            *stage1_stage2_seeds,
        ]
    )
    refinement_seed_payloads = list(payload.get("stage1_refinement_seeds") or ())
    stage1_refinement_seeds = tuple(
        _result_from_payload(row, default_objective="stage1_research_label") for row in refinement_seed_payloads
    )
    fast_replay_policy = payload.get("fast_replay_policy") if isinstance(payload.get("fast_replay_policy"), dict) else {}
    return Stage1ResumeBundle(
        source_path=source_path.resolve(),
        source_payload_hash=stable_signature(payload),
        source_sweep_hash=str(payload.get("sweep_hash", "")),
        source_sweep_version=str(payload.get("sweep_version", "")),
        source_strategy_core_version=str(payload.get("strategy_core_version", "")),
        source_stage2_portfolio_proxy_version=str(fast_replay_policy.get("stage2_portfolio_proxy_version", "")),
        stage1_rows=tuple(stage1_rows),
        stage1_stage2_seeds=stage1_stage2_seeds,
        stage1_refinement_seeds=stage1_refinement_seeds,
        stage1_candidate_count=int(payload.get("stage1_candidate_count") or len(stage1_rows)),
        stage1_coarse_candidate_count=int(payload.get("stage1_coarse_candidate_count") or 0),
        stage1_refinement_candidate_count=int(payload.get("stage1_refinement_candidate_count") or 0),
    )


def _result_from_payload(row: dict[str, Any], *, default_objective: str) -> ResearchSweepResult:
    stage1_seed = row.get("stage1_seed") if isinstance(row.get("stage1_seed"), dict) else {}
    return ResearchSweepResult(
        experiment=Experiment(str(row.get("name") or "__loaded_seed__"), dict(row.get("mutations") or {})),
        score=round(_float(row.get("score")), 6),
        full_score=round(_float(row.get("full_score"), _float(row.get("score"))), 6),
        median_fold_score=round(_float(row.get("median_fold_score"), _float(row.get("score"))), 6),
        worst_fold_score=round(_float(row.get("worst_fold_score"), _float(row.get("score"))), 6),
        rejected=bool(row.get("rejected", False)),
        reject_reason=str(row.get("reject_reason") or ""),
        metrics=dict(row.get("metrics") or {}),
        folds=tuple(row.get("folds") or ()),
        artifact_hash=str(row.get("artifact_hash") or ""),
        objective=str(row.get("objective") or default_objective),
        parent_stage1_name=str(stage1_seed.get("name") or ""),
        parent_stage1_mutations=dict(stage1_seed.get("mutations") or {}),
    )


def _dedupe_results(rows: Iterable[ResearchSweepResult]) -> tuple[ResearchSweepResult, ...]:
    out: list[ResearchSweepResult] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.experiment.name, stable_signature(row.experiment.mutations))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    out.sort(key=lambda item: (-item.score, item.rejected, item.experiment.name))
    return tuple(out)


def _row_payload(row: ResearchSweepResult) -> dict[str, Any]:
    payload = {
        "name": row.experiment.name,
        "score": row.score,
        "full_score": row.full_score,
        "median_fold_score": row.median_fold_score,
        "worst_fold_score": row.worst_fold_score,
        "rejected": row.rejected,
        "reject_reason": row.reject_reason,
        "objective": row.objective,
        "mutations": dict(row.experiment.mutations),
        "metrics": dict(row.metrics),
        "folds": list(row.folds),
        "artifact_hash": row.artifact_hash,
    }
    if row.parent_stage1_name:
        payload["stage1_seed"] = {
            "name": row.parent_stage1_name,
            "mutations": dict(row.parent_stage1_mutations),
        }
    return payload


def _record_progress(
    output_dir: Path,
    stage: str,
    completed: int,
    total: int,
    rows: list[ResearchSweepResult],
    row: ResearchSweepResult,
) -> None:
    _write_progress(output_dir, stage, completed, total, rows)
    event = {"updated_at": _utc_now_iso(), "stage": stage, "completed": int(completed), "total": int(total), "row": _progress_row(row)}
    with (output_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")


def _clear_progress_files(output_dir: Path) -> None:
    for name in ("progress.jsonl", "run_status.json"):
        path = output_dir / name
        if path.exists():
            path.unlink()
    for path in output_dir.glob("progress_*.json"):
        path.unlink()


def _write_run_status(output_dir: Path, stage: str, **extra: Any) -> None:
    payload = {"updated_at": _utc_now_iso(), "stage": stage, **extra}
    tmp = output_dir / "run_status.json.tmp"
    path = output_dir / "run_status.json"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)
    with (output_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _write_progress(output_dir: Path, stage: str, completed: int, total: int, rows: list[ResearchSweepResult]) -> None:
    ranked = sorted(rows, key=lambda item: (-item.score, item.rejected, item.experiment.name))
    payload = {
        "updated_at": _utc_now_iso(),
        "stage": stage,
        "completed": int(completed),
        "total": int(total),
        "best_so_far": _progress_row(ranked[0]) if ranked else None,
        "top_rows": [_progress_row(row) for row in ranked[:15]],
    }
    path = output_dir / f"progress_{stage}.json"
    tmp = output_dir / f"progress_{stage}.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def _progress_row(row: ResearchSweepResult) -> dict[str, Any]:
    return {
        "name": row.experiment.name,
        "score": row.score,
        "rejected": row.rejected,
        "reject_reason": row.reject_reason,
        "objective": row.objective,
        "stage1_seed": row.parent_stage1_name,
        "net": row.metrics.get("avg_active_net_return_pct", 0.0),
        "slot": row.metrics.get("slot_cumulative_net_return_pct", 0.0),
        "portfolio_proxy": row.metrics.get("portfolio_proxy_net_return_pct", 0.0),
        "portfolio_proxy_dd": row.metrics.get("portfolio_proxy_max_drawdown_pct", 0.0),
        "mfe": row.metrics.get("avg_active_mfe_r", 0.0),
        "active_days": row.metrics.get("active_days", 0.0),
    }


def _implementation_lessons_contract() -> dict[str, Any]:
    return {
        "status": "research_only_thin_selector",
        "shared_selection_api": {
            "stage1": "strategy_olr.research.daily_selection_from_snapshot",
            "stage2": "strategy_olr.research.afternoon_selection_from_snapshot",
            "stage2_cached_contexts": "strategy_olr.research.build_afternoon_contexts -> strategy_olr.research.afternoon_selection_from_contexts",
            "live_stage1_wrapper": "strategy_olr.research_generator.generate_candidate_snapshot",
            "live_stage2_wrapper": "strategy_olr.research_generator.generate_afternoon_candidate_snapshot",
            "replay_stage1_wrapper": "backtests.strategies.olr.research_sweep.snapshots_for_experiment",
            "replay_stage2_wrapper": "backtests.strategies.olr.research_sweep.evaluate_stage2_experiment",
        },
        "reference_pattern": {
            "live_data_builder": "_references/trading/strategies/stock/artifact_generator.py",
            "replay_data_builder": "_references/trading/backtests/stock/engine/research_replay.py",
            "contract": "Data builders differ; both must converge on ResearchSnapshot -> daily_selection_from_snapshot()/run_daily_selection().",
        },
        "live_backtest_divergence_policy": "Only data acquisition and artifact persistence may differ; selection functions, causal cutoffs, and config aliases are shared.",
        "completed_bar_policy": {
            "daily": "daily and flow rows must satisfy row_date < trade_date",
            "stage2_intraday": "5m bars must be completed and timestamp < 14:30 KST",
            "same_day_daily_flow": "not visible to either stage",
        },
        "execution_core_status": "not_applicable_yet_research_module_only",
        "production_gate": "No official performance until a later execution core/backtest adapter/paper-parity path consumes the selected artifacts.",
    }


def _metric_contract() -> dict[str, Any]:
    return {
        "basis": "research_selection_label_only",
        "stage2_basis": "fixed close-auction to next-close portfolio proxy for selection only",
        "primary_promotion_metric": "",
        "primary_promotion_basis": "research_selection_label_only",
        "promotion_requires_audit_pass": True,
        "official_replay_pass": False,
        "audit_pass": False,
        "audit_status": "research_only_proxy",
        "official_metrics": [],
        "proxy_metrics": ["portfolio_proxy_net_return_pct", "avg_active_net_return_pct"],
        "headline_allowed": False,
        "official_performance": False,
        "return_label": "day-D close to day-D+1 close after 14:30 selection; label is not an executable fill model",
        "stage2_portfolio_proxy": {
            "version": STAGE2_PORTFOLIO_PROXY_VERSION,
            "allocation": "1.0 target gross, 50% max position, min 2 selected names",
            "purpose": "Rank afternoon candidate sources by deployable fixed-baseline opportunity before Stage 3 execution optimization.",
            "official_performance": False,
        },
        "mfe_label": "next-session high versus day-D close, normalized by candidate ATR risk",
        "cost_model": "round-trip slippage/commission/tax approximation used only for research ranking",
        "risk_metrics": "No Sharpe/Calmar/drawdown headline metrics are emitted by this research sweep; later execution backtests must use MTM equity.",
        "artifact_hygiene": "Outputs carry source fingerprints, causal cutoffs, holdout policy, and official_performance=false.",
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# OLR Research Sweep",
        "",
        f"- Sweep hash: `{payload.get('sweep_hash')}`",
        f"- Training window: {payload['training_window']['start']} to {payload['training_window']['end']}",
        "- Causality: daily/flow rows use `row_date < trade_date`; afternoon bars use `timestamp < 14:30 KST`.",
        f"- Fast/full audit: `{'pass' if payload.get('audit_pass') else 'fail'}`.",
        f"- Stage 1 seeds evaluated in Stage 2: `{payload.get('stage1_stage2_seed_count', 1)}`.",
        "",
        "## Stage 1 Frontier",
        _table(payload.get("stage1_frontier", [])),
        "",
        "## Stage 2 Frontier",
        _table(payload.get("stage2_frontier", [])),
        "",
        "## Metric Basis",
        "- Stage 1 uses research labels only; not official performance.",
        "- Stage 2 ranks fixed close-auction to next-close portfolio proxy plus MFE/downside/coverage; not official performance.",
        "- Later execution backtests must consume these artifacts through a shared core and MTM accounting before promotion.",
    ]
    return "\n".join(lines) + "\n"


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| name | score | proxy | net | mfe | reject |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for row in rows[:20]:
        metrics = row.get("metrics") or {}
        lines.append(
            f"| {row.get('name')} | {row.get('score', 0.0):.3f} | "
            f"{100.0 * metrics.get('portfolio_proxy_net_return_pct', 0.0):.1f}% | "
            f"{100.0 * metrics.get('avg_active_net_return_pct', 0.0):.3f}% | "
            f"{metrics.get('avg_active_mfe_r', 0.0):.3f} | {row.get('reject_reason') or ''} |"
        )
    return "\n".join(lines)


def _weights(
    *,
    rs: float,
    trend: float,
    comp: float,
    accum: float,
    stock: float,
    sector: float,
    part: float,
    signal: float,
    flow: float,
    foreign: float,
    inst: float,
    agree: float,
) -> dict[str, float]:
    return {
        "olr.research.weights.relative_strength": rs,
        "olr.research.weights.daily_trend": trend,
        "olr.research.weights.compression": comp,
        "olr.research.weights.accumulation": accum,
        "olr.research.weights.stock_regime": stock,
        "olr.research.weights.sector_regime": sector,
        "olr.research.weights.sector_participation": part,
        "olr.research.weights.daily_signal": signal,
        "olr.research.weights.flow": flow,
        "olr.research.weights.foreign_flow": foreign,
        "olr.research.weights.institutional_flow": inst,
        "olr.research.weights.flow_agreement": agree,
    }


def _refinement_weight_families() -> tuple[tuple[str, dict[str, float]], ...]:
    return (
        ("sector_rs_weights", _weights(rs=0.28, trend=0.18, comp=0.10, accum=0.10, stock=0.07, sector=0.09, part=0.18, signal=0.06, flow=0.02, foreign=0.01, inst=0.01, agree=0.0)),
        ("rs_trend_clean_weights", _weights(rs=0.30, trend=0.24, comp=0.08, accum=0.10, stock=0.08, sector=0.06, part=0.08, signal=0.04, flow=0.01, foreign=0.005, inst=0.005, agree=0.0)),
        ("flow_leader_weights", _weights(rs=0.16, trend=0.14, comp=0.06, accum=0.08, stock=0.06, sector=0.08, part=0.10, signal=0.06, flow=0.14, foreign=0.06, inst=0.04, agree=0.02)),
        ("pullback_signal_weights", _weights(rs=0.14, trend=0.12, comp=0.08, accum=0.08, stock=0.06, sector=0.06, part=0.06, signal=0.24, flow=0.08, foreign=0.04, inst=0.03, agree=0.01)),
    )


def _afternoon_name(mode: str, top_n: int, values: dict[str, Any]) -> str:
    parts = [mode, f"top{top_n}"]
    for key, value in sorted(values.items()):
        label = key.split(".")[-1].replace("afternoon_", "")
        parts.append(f"{label}{_num_label(value)}")
    return "_".join(parts)


def _num_label(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    try:
        return str(round(float(value), 4)).replace("-", "m").replace(".", "p")
    except (TypeError, ValueError):
        return str(value)


def _validate_mutations(experiments: list[Experiment], allowed: set[str]) -> None:
    for experiment in experiments:
        invalid = sorted(set(experiment.mutations) - allowed)
        if invalid:
            raise ValueError(f"OLR research sweep candidate {experiment.name} has non-research mutations: {invalid}")


def _mutation_signature(mutations: dict[str, Any]) -> str:
    return json.dumps(dict(sorted((str(key), value) for key, value in dict(mutations or {}).items())), sort_keys=True, separators=(",", ":"), default=str)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value))[:80] or "candidate"


def _row_date(row: dict[str, Any]) -> date:
    value = row.get("date") or row.get("trade_date") or row.get("timestamp")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _try_row_date(row: dict[str, Any]) -> date | None:
    try:
        return _row_date(row)
    except (TypeError, ValueError):
        return None


def _round_trip_cost_pct(config: OLRConfig) -> float:
    bps = 2.0 * float(config.slippage_bps) + 2.0 * float(config.commission_bps) + float(config.tax_bps_on_sell)
    return max(0.0, bps) / 10_000.0


def _return_score(value: float, *, target: float) -> float:
    span = max(float(target), 1e-9)
    return _clip((float(value) + span) / (2.0 * span))


def _compound(returns: Iterable[float]) -> float:
    value = 1.0
    for ret in returns:
        value *= 1.0 + float(ret)
    return value - 1.0


def _avg(values: Iterable[float]) -> float:
    seq = [float(value) for value in values]
    return mean(seq) if seq else 0.0


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep OLR research selectors.")
    parser.add_argument("--config", default="config/optimization/olr.yaml")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--holdout-days", type=int, default=DEFAULT_HOLDOUT_DAYS)
    parser.add_argument("--expected-universe-size", type=int, default=DEFAULT_EXPECTED_UNIVERSE_SIZE)
    parser.add_argument("--allow-universe-size-override", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--refine-top-n", type=int, default=3)
    parser.add_argument("--max-refinement-candidates", type=int, default=96)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--fold-count", type=int, default=2)
    parser.add_argument("--stage1-stage2-seed-count", type=int, default=DEFAULT_STAGE1_STAGE2_SEED_COUNT)
    parser.add_argument(
        "--resume-stage1-artifact",
        default=None,
        help="Reuse Stage 1 seeds from a prior OLR research sweep artifact and rebuild Stage 2 under current code.",
    )
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--audit-finalist-count", type=int, default=5)
    args = parser.parse_args(argv)
    config = normalize_runtime_config("olr", load_yaml_config(args.config))
    payload = run_research_sweep(
        config,
        output_dir=args.output_dir,
        holdout_days=args.holdout_days,
        expected_universe_size=args.expected_universe_size,
        allow_universe_size_override=args.allow_universe_size_override,
        max_candidates=args.max_candidates,
        refine_top_n=args.refine_top_n,
        max_refinement_candidates=args.max_refinement_candidates,
        top_n=args.top_n,
        fold_count=args.fold_count,
        stage1_stage2_seed_count=args.stage1_stage2_seed_count,
        max_workers=args.max_workers,
        audit_finalist_count=args.audit_finalist_count,
        resume_stage1_artifact=args.resume_stage1_artifact,
    )
    print(json.dumps({"strategy": "olr", "sweep_hash": payload["sweep_hash"], "artifact_paths": payload["artifact_paths"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
