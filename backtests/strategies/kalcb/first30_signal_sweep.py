from __future__ import annotations

import argparse
import bisect
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import pandas as pd

from backtests.auto.shared.cache_keys import fingerprint_paths, stable_signature
from backtests.auto.shared.phase_state import _utc_now_iso
from backtests.config import load_yaml_config, normalize_runtime_config
from strategy_common.clock import KST
from strategy_common.daily_lrs_parquet import (
    load_daily_flow,
    load_daily_foreign_flow,
    load_daily_institutional_flow,
    load_daily_ohlcv,
    load_index_ohlcv,
    load_manifest,
    load_sector_map,
)
from strategy_common.market import MarketBar
from strategy_common.sector_daily import SECTOR_DAILY_VERSION, SectorDailyFeature, SectorDailyMember, score_sector_daily_members
from strategy_common.sector_intraday import (
    FIRST30_CUTOFF,
    SECTOR_INTRADAY_VERSION,
    SectorIntradayFeature,
    SectorIntradayMember,
    cutoff_label_for,
    score_sector_members,
)
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.first30 import (
    FIRST30_BAR_COUNT,
    FIRST30_END,
    build_first30_features,
    completed_first30_bars,
)

from .replay_cache import (
    _load_symbol_frame,
    _real_source_fingerprint,
    _resolve_replay_window,
    _resolve_sector_map,
    _resolve_symbols,
    _trading_dates,
)


FIRST30_SWEEP_VERSION = "kalcb-causal-first30-sweep-v2"
DEFAULT_OUTPUT_DIR = Path("data/backtests/output/kalcb/first30_signal_sweeps")
DEFAULT_HOLDOUT_DAYS = 42
STAGE2_PORTFOLIO_POLICY = {
    "name": "aggressive_contained_first30_proxy_v1",
    "risk_per_trade_pct": 0.0070,
    "max_position_notional_pct": 0.45,
    "intraday_leverage": 2.0,
    "max_positions": 8,
}

_ROW_INDEX_CACHE: dict[tuple[Any, ...], tuple[tuple[date, ...], tuple[dict[str, Any], ...]]] = {}
_DAILY_FEATURE_CACHE: dict[tuple[Any, ...], DailyFeature | None] = {}
_FLOW_FEATURE_CACHE: dict[tuple[Any, ...], FlowFeature] = {}
_MARKET_FEATURE_CACHE: dict[tuple[int, date], MarketFeature] = {}
_INDEX_STATS_CACHE: dict[tuple[int, date], dict[str, Any]] = {}
_DAILY_FEATURE_BY_DAY_CACHE: dict[tuple[Any, ...], dict[date, DailyFeature | None]] = {}
_FLOW_FEATURE_BY_DAY_CACHE: dict[tuple[Any, ...], dict[date, FlowFeature]] = {}
_SORTED_BARS_CACHE: dict[int, tuple[MarketBar, ...]] = {}
_FIRST30_BARS_CACHE: dict[int, tuple[MarketBar, ...]] = {}
_POST_ENTRY_BARS_CACHE: dict[tuple[int, dt_time], tuple[MarketBar, ...]] = {}


@dataclass(frozen=True, slots=True)
class DailyFeature:
    symbol: str
    trade_date: date
    prev_close: float
    atr14: float
    return_5d: float
    return_20d: float
    return_60d: float
    adv20_krw: float
    volume_ratio_20d: float
    close20_loc: float
    close60_loc: float
    above_sma20: bool
    above_sma60: bool


@dataclass(frozen=True, slots=True)
class FlowFeature:
    available: bool
    foreign_1d: float = 0.0
    foreign_3d: float = 0.0
    foreign_5d: float = 0.0
    foreign_20d: float = 0.0
    inst_1d: float = 0.0
    inst_3d: float = 0.0
    inst_5d: float = 0.0
    inst_20d: float = 0.0
    combined_1d: float = 0.0
    combined_3d: float = 0.0
    combined_5d: float = 0.0
    combined_20d: float = 0.0
    combined_notional_5d: float = 0.0
    positive_days_5d: float = 0.0
    foreign_positive_days_5d: float = 0.0
    inst_positive_days_5d: float = 0.0
    acceleration: float = 0.0
    foreign_acceleration: float = 0.0
    inst_acceleration: float = 0.0
    z_score: float = 0.0
    foreign_z: float = 0.0
    inst_z: float = 0.0
    agreement_5d: float = 0.0
    divergence_5d: float = 0.0
    sponsorship_balance_5d: float = 0.0
    sector_flow_5d: float = 0.0
    sector_foreign_5d: float = 0.0
    sector_inst_5d: float = 0.0
    sector_agreement_5d: float = 0.0
    sector_participation: float = 0.0


@dataclass(frozen=True, slots=True)
class MarketFeature:
    kospi_ret_1d: float = 0.0
    kospi_ret_5d: float = 0.0
    kospi_ret_20d: float = 0.0
    kosdaq_ret_1d: float = 0.0
    kosdaq_ret_5d: float = 0.0
    kosdaq_ret_20d: float = 0.0
    kospi_above_sma20: bool = False
    kospi_above_sma60: bool = False
    kosdaq_above_sma20: bool = False
    kosdaq_above_sma60: bool = False
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class First30Intraday:
    open: float
    high: float
    low: float
    close: float
    vwap: float
    volume: float
    expected_30m_volume: float


@dataclass(frozen=True, slots=True)
class First30Context:
    day: date
    symbol: str
    sector: str
    daily: DailyFeature
    flow: FlowFeature
    market: MarketFeature
    intraday: First30Intraday
    bars: tuple[MarketBar, ...]
    post_bars: tuple[MarketBar, ...]
    first30_ret: float
    vwap_ret: float
    gap: float
    rel_volume: float
    close_location: float
    open_drawdown: float
    low_vs_prev_close: float
    range_atr: float
    sector_daily: SectorDailyFeature | None = None
    sector_intraday: SectorIntradayFeature | None = None


@dataclass(frozen=True, slots=True)
class Selection:
    trade_date: date
    symbol: str
    score: float
    family: str


@dataclass(frozen=True, slots=True)
class OpportunityRow:
    trade_date: date
    symbol: str
    family: str
    score: float
    gross_eod_pct: float
    net_eod_pct: float
    mfe_r: float
    mae_r: float
    entry_price: float = 0.0
    risk_pct: float = 0.0


@dataclass(frozen=True, slots=True)
class KALCBFirst30Dataset:
    config: dict[str, Any]
    source_fingerprint: str
    daily_source_fingerprint: str
    data_root: Path
    daily_data_root: Path
    timeframe: str
    symbols: tuple[str, ...]
    data_available_symbols: tuple[str, ...]
    daily_available_symbols: tuple[str, ...]
    unavailable_symbols: tuple[str, ...]
    daily_by_symbol: dict[str, list[dict[str, Any]]]
    flow_by_symbol: dict[str, list[dict[str, Any]]]
    index_by_code: dict[str, list[dict[str, Any]]]
    trading_dates: tuple[date, ...]
    bars_by_key: dict[tuple[date, str], tuple[MarketBar, ...]]
    sector_map: dict[str, str]
    foreign_flow_by_symbol: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    institutional_flow_by_symbol: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class First30Spec:
    name: str
    score_mode: str
    top_n: int
    min_first30_ret: float = 0.0
    min_vwap_ret: float = 0.0
    min_gap: float = -0.20
    max_gap: float = 0.20
    min_rel_volume: float = 0.0
    min_close_location: float = 0.0
    max_open_drawdown: float = 0.20
    max_range_atr: float = 99.0
    min_prior_ret5: float = -1.0
    min_prior_ret20: float = -1.0
    min_prior_ret60: float = -1.0
    max_prior_ret20: float = 9.99
    min_low_vs_prev_close: float = -0.20
    min_flow_5d: float = -9.99
    min_foreign_flow_5d: float = -9.99
    min_inst_flow_5d: float = -9.99
    min_flow_z: float = -9.99
    min_flow_agreement: float = -9.99
    max_flow_divergence: float = 9.99
    min_sector_flow: float = -9.99
    min_market_score: float = -9.99
    require_close_above_prev: bool = False


@dataclass(frozen=True, slots=True)
class SweepResult:
    spec: First30Spec
    score: float
    full_score: float
    median_fold_score: float
    worst_fold_score: float
    rejected: bool
    reject_reason: str
    metrics: dict[str, float]
    folds: tuple[dict[str, Any], ...]


def run_first30_signal_sweep(
    config: dict[str, Any],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    holdout_days: int = DEFAULT_HOLDOUT_DAYS,
    max_workers: int = 4,
    refine_top_n: int = 6,
    max_coarse_specs: int | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    training_config = _training_config(dict(config), holdout_days)
    dataset = prepare_first30_dataset(training_config)
    cfg = KALCBConfig.from_mapping(training_config, {})
    contexts = build_contexts(dataset)
    folds = _resolve_folds(list(dataset.trading_dates), fold_count=2)

    coarse_specs = build_coarse_specs()
    if max_coarse_specs is not None:
        coarse_specs = _even_sample(coarse_specs, max(1, int(max_coarse_specs)))
    coarse_rows = _evaluate_specs(
        coarse_specs,
        contexts,
        dataset,
        cfg,
        folds,
        out,
        stage="coarse",
        completed_offset=0,
        total=len(coarse_specs),
        max_workers=max_workers,
    )
    coarse_rows.sort(key=lambda row: (-row.score, row.rejected, row.spec.name))
    seeds = [row for row in coarse_rows if not row.rejected][: max(0, int(refine_top_n))]
    refinement_specs = build_refinement_specs(
        [row.spec for row in seeds],
        existing={_spec_signature(row.spec) for row in coarse_rows},
    )
    refinement_rows = _evaluate_specs(
        refinement_specs,
        contexts,
        dataset,
        cfg,
        folds,
        out,
        stage="refinement",
        completed_offset=len(coarse_rows),
        total=len(coarse_rows) + len(refinement_specs),
        max_workers=max_workers,
        seed_rows=coarse_rows,
    )
    rows = [*coarse_rows, *refinement_rows]
    rows.sort(key=lambda row: (-row.score, row.rejected, row.spec.name))
    top_portfolio = sorted(rows, key=lambda row: (row.rejected, -row.metrics.get("portfolio_proxy_net_return_pct", 0.0), row.spec.name))[:25]
    top_slot = sorted(rows, key=lambda row: (row.rejected, -row.metrics.get("slot_cumulative_gross_return_pct", 0.0), row.spec.name))[:25]
    top_mfe = sorted(rows, key=lambda row: (row.rejected, -row.metrics.get("avg_mfe_r", 0.0), row.spec.name))[:25]
    payload = {
        "strategy": "kalcb",
        "sweep_version": FIRST30_SWEEP_VERSION,
        "created_at": _utc_now_iso(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "training_window": {
            "start": dataset.trading_dates[0].isoformat(),
            "end": dataset.trading_dates[-1].isoformat(),
            "sessions": len(dataset.trading_dates),
        },
        "holdout_days": int(holdout_days),
        "causality_policy": {
            "selection_inputs": "prior completed daily/flow/index rows plus completed 09:00-09:25 KST bars only",
            "entry": "09:30 KST bar open",
            "evaluation": "post-entry 09:30-to-configured-flatten bars only",
            "official_performance": False,
        },
        "source_fingerprints": {
            "intraday": dataset.source_fingerprint,
            "daily_lrs": dataset.daily_source_fingerprint,
            "combined": stable_signature([dataset.source_fingerprint, dataset.daily_source_fingerprint]),
            "sector_daily_version": SECTOR_DAILY_VERSION,
            "sector_intraday_version": SECTOR_INTRADAY_VERSION,
        },
        "cost_policy": {
            "round_trip_cost_pct": _round_trip_cost_pct(cfg),
            "slippage_bps_each_side": cfg.slippage_bps,
            "commission_bps_each_side": cfg.commission_bps,
            "tax_bps_on_sell": cfg.tax_bps_on_sell,
        },
        "data_policy": {
            "data_root": str(dataset.data_root),
            "daily_data_root": str(dataset.daily_data_root),
            "symbols": len(dataset.symbols),
            "daily_available_symbols": len(dataset.daily_available_symbols),
            "combined_flow_available_symbols": len(dataset.flow_by_symbol),
            "foreign_flow_available_symbols": len(dataset.foreign_flow_by_symbol),
            "institutional_flow_available_symbols": len(dataset.institutional_flow_by_symbol),
            "sector_daily_context": "prior completed daily/flow rows only",
            "sector_intraday_context": "completed 09:00-09:25 KST peer bars excluding current symbol",
            "unavailable_symbols": list(dataset.unavailable_symbols),
            "index_codes": sorted(dataset.index_by_code),
        },
        "objective": "portfolio-aware net 09:30/EOD proxy first, Avg MFE R second; gross Slot Return remains an opportunity diagnostic",
        "portfolio_proxy_policy": STAGE2_PORTFOLIO_POLICY,
        "coarse_count": len(coarse_rows),
        "refinement_count": len(refinement_rows),
        "candidate_count": len(rows),
        "top_results": [_row_payload(row) for row in rows[:25]],
        "top_portfolio_proxy": [_row_payload(row) for row in top_portfolio],
        "top_slot_return": [_row_payload(row) for row in top_slot],
        "top_mfe": [_row_payload(row) for row in top_mfe],
        "rows": [_row_payload(row) for row in rows],
    }
    payload["sweep_hash"] = stable_signature(
        {
            "version": FIRST30_SWEEP_VERSION,
            "source_fingerprints": payload["source_fingerprints"],
            "training_window": payload["training_window"],
            "top_portfolio": payload["top_portfolio_proxy"][:10],
            "top": payload["top_results"][:10],
        }
    )
    json_path = out / f"kalcb_first30_signal_sweep_{payload['sweep_hash'][:12]}.json"
    md_path = out / f"kalcb_first30_signal_sweep_{payload['sweep_hash'][:12]}.md"
    payload["artifact_paths"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    _write_progress(out, "completed", len(rows), len(rows), rows)
    return payload


def prepare_first30_dataset(config: dict[str, Any]) -> KALCBFirst30Dataset:
    raw_config = dict(config)
    cfg = KALCBConfig.from_mapping(raw_config, {})
    data_root = Path(raw_config.get("data_root", "data/kis_intraday_parquet"))
    daily_root = Path(raw_config.get("daily_data_root", "data/krx_daily_parquet"))
    timeframe = str(raw_config.get("timeframe", cfg.timeframe) or "5m")
    if timeframe != "5m":
        raise ValueError("KALCB first30 sweeps require 5m parquet input")
    symbols = _resolve_symbols(raw_config, data_root, timeframe)
    window = _resolve_replay_window(raw_config, data_root, timeframe, symbols)
    intraday_fingerprint = _real_source_fingerprint(data_root, symbols, timeframe, window.train_start, window.train_end)
    frames = {symbol: _load_symbol_frame(data_root, symbol, timeframe, window.train_end) for symbol in symbols}
    data_available = tuple(symbol for symbol, frame in frames.items() if not frame.empty)
    unavailable = tuple(symbol for symbol, frame in frames.items() if frame.empty)
    daily_by_symbol = _load_daily_rows(daily_root, data_available, window.train_end)
    flow_by_symbol = _load_flow_rows(daily_root, data_available, window.train_end)
    foreign_flow_by_symbol = _load_foreign_flow_rows(daily_root, data_available, window.train_end)
    institutional_flow_by_symbol = _load_institutional_flow_rows(daily_root, data_available, window.train_end)
    index_by_code = _load_index_rows(daily_root, window.train_end)
    trading_dates = tuple(_trading_dates(frames, window.train_start, window.train_end))
    bars_by_key = _bars_by_key_from_frames(frames, window.train_start, window.train_end, source_fingerprint=intraday_fingerprint)
    if not trading_dates or not bars_by_key:
        raise ValueError("KALCB first30 sweep found no intraday bars in the selected training window")
    sector_map = {**_resolve_sector_map(raw_config), **load_sector_map(daily_root)}
    daily_available = tuple(symbol for symbol in data_available if daily_by_symbol.get(symbol))
    return KALCBFirst30Dataset(
        config=raw_config,
        source_fingerprint=intraday_fingerprint,
        daily_source_fingerprint=_daily_source_fingerprint(
            daily_root,
            daily_by_symbol,
            flow_by_symbol,
            foreign_flow_by_symbol,
            institutional_flow_by_symbol,
            index_by_code,
            sector_map,
        ),
        data_root=data_root,
        daily_data_root=daily_root,
        timeframe=timeframe,
        symbols=tuple(symbols),
        data_available_symbols=data_available,
        daily_available_symbols=daily_available,
        unavailable_symbols=unavailable,
        daily_by_symbol=daily_by_symbol,
        flow_by_symbol=flow_by_symbol,
        index_by_code=index_by_code,
        trading_dates=trading_dates,
        bars_by_key=bars_by_key,
        sector_map={str(symbol).zfill(6): str(sector).upper() for symbol, sector in sector_map.items()},
        foreign_flow_by_symbol=foreign_flow_by_symbol,
        institutional_flow_by_symbol=institutional_flow_by_symbol,
    )


def build_contexts(dataset: KALCBFirst30Dataset) -> dict[date, tuple[First30Context, ...]]:
    by_day: dict[date, list[First30Context]] = {}
    cfg = KALCBConfig.from_mapping(dataset.config, {})
    for day in dataset.trading_dates:
        market = _market_feature(dataset, day)
        provisional: list[tuple[str, DailyFeature, FlowFeature, Any, First30Intraday, tuple[MarketBar, ...], tuple[MarketBar, ...]]] = []
        sector_flows: dict[str, list[FlowFeature]] = {}
        for symbol in dataset.data_available_symbols:
            daily = daily_feature(dataset, symbol, day)
            if daily is None:
                continue
            bars = dataset.bars_by_key.get((day, symbol), ())
            first30 = shared_first30_feature(daily, bars)
            intraday = (
                First30Intraday(
                    open=first30.open,
                    high=first30.high,
                    low=first30.low,
                    close=first30.close,
                    vwap=first30.vwap,
                    volume=first30.volume,
                    expected_30m_volume=first30.expected_30m_volume,
                )
                if first30 is not None
                else None
            )
            post_bars = _post_entry_bars(bars, cfg)
            if intraday is None or not post_bars:
                continue
            flow = flow_feature(dataset, symbol, day)
            sector = dataset.sector_map.get(symbol, "UNKNOWN")
            sector_flows.setdefault(sector, []).append(flow)
            provisional.append((symbol, daily, flow, first30, intraday, bars, post_bars))
        sector_stats = {
            sector: (
                _avg(item.combined_5d for item in items if item.available),
                _avg(item.foreign_5d for item in items if item.available),
                _avg(item.inst_5d for item in items if item.available),
                _avg(item.agreement_5d for item in items if item.available),
                _ratio(sum(1 for item in items if item.combined_5d > 0.0), len(items)),
            )
            for sector, items in sector_flows.items()
        }
        sector_daily_panel = score_sector_daily_members(
            [
                SectorDailyMember(
                    symbol=symbol,
                    sector=dataset.sector_map.get(symbol, "UNKNOWN"),
                    trade_date=day,
                    ret_5d=daily.return_5d,
                    ret_20d=daily.return_20d,
                    ret_60d=daily.return_60d,
                    above_sma20=daily.above_sma20,
                    rel_volume=daily.volume_ratio_20d,
                    flow_5d=flow.combined_5d,
                    foreign_flow_5d=flow.foreign_5d,
                    institutional_flow_5d=flow.inst_5d,
                    flow_agreement_5d=flow.agreement_5d,
                    flow_available=flow.available,
                )
                for symbol, daily, flow, _first30, _intraday, _bars, _post_bars in provisional
            ],
            trade_date=day,
            target_symbols=[symbol for symbol, *_rest in provisional],
        )
        sector_first30_panel = score_sector_members(
            [
                SectorIntradayMember(
                    symbol=symbol,
                    sector=dataset.sector_map.get(symbol, "UNKNOWN"),
                    trade_date=day,
                    ret=first30.first30_ret,
                    vwap_ret=first30.vwap_ret,
                    close_location=first30.range_close_location,
                    rel_volume=first30.rel_volume,
                    volume=intraday.volume,
                    bar_count=FIRST30_BAR_COUNT,
                )
                for symbol, _daily, _flow, first30, intraday, _bars, _post_bars in provisional
            ],
            trade_date=day,
            cutoff_label=cutoff_label_for(FIRST30_CUTOFF),
            target_symbols=[symbol for symbol, *_rest in provisional],
        )
        for symbol, daily, flow, first30, intraday, bars, post_bars in provisional:
            sector = dataset.sector_map.get(symbol, "UNKNOWN")
            sector_flow, sector_foreign, sector_inst, sector_agreement, sector_participation = sector_stats.get(sector, (0.0, 0.0, 0.0, 0.0, 0.0))
            sector_daily = sector_daily_panel.feature_for(day, symbol, sector=sector)
            sector_intraday = sector_first30_panel.feature_for(day, symbol, sector=sector)
            enriched_flow = replace(
                flow,
                sector_flow_5d=sector_flow,
                sector_foreign_5d=sector_foreign,
                sector_inst_5d=sector_inst,
                sector_agreement_5d=sector_agreement,
                sector_participation=sector_participation,
            )
            by_day.setdefault(day, []).append(
                First30Context(
                    day=day,
                    symbol=symbol,
                    sector=sector,
                    daily=daily,
                    flow=enriched_flow,
                    market=market,
                    intraday=intraday,
                    bars=bars,
                    post_bars=post_bars,
                    first30_ret=first30.first30_ret,
                    vwap_ret=first30.vwap_ret,
                    gap=first30.gap,
                    rel_volume=first30.rel_volume,
                    close_location=first30.range_close_location,
                    open_drawdown=first30.open_drawdown,
                    low_vs_prev_close=first30.low_vs_prev_close,
                    range_atr=first30.range_atr,
                    sector_daily=sector_daily,
                    sector_intraday=sector_intraday,
                )
            )
    return {day: tuple(items) for day, items in by_day.items()}


def daily_feature(dataset: KALCBFirst30Dataset, symbol: str, trade_date: date) -> DailyFeature | None:
    symbol = str(symbol).zfill(6)
    key = (*_daily_series_cache_key(dataset, symbol), trade_date)
    if key in _DAILY_FEATURE_CACHE:
        return _DAILY_FEATURE_CACHE[key]
    series = _daily_feature_series(dataset, symbol)
    if trade_date in series:
        feature = series[trade_date]
        _DAILY_FEATURE_CACHE[key] = feature
        return feature
    rows = prior_daily_rows(dataset, symbol, trade_date)
    feature = _daily_feature_from_prior(symbol, trade_date, rows)
    _DAILY_FEATURE_CACHE[key] = feature
    return feature


def _daily_feature_series(dataset: KALCBFirst30Dataset, symbol: str) -> dict[date, DailyFeature | None]:
    symbol = str(symbol).zfill(6)
    key = _daily_series_cache_key(dataset, symbol)
    cached = _DAILY_FEATURE_BY_DAY_CACHE.get(key)
    if cached is not None:
        return cached
    dates, rows = _indexed_symbol_rows(dataset, "daily", symbol)
    by_day: dict[date, DailyFeature | None] = {}
    for trade_date in dataset.trading_dates:
        index = bisect.bisect_left(dates, trade_date)
        by_day[trade_date] = _daily_feature_from_prior(symbol, trade_date, rows[:index])
    _DAILY_FEATURE_BY_DAY_CACHE[key] = by_day
    return by_day


def _daily_feature_from_prior(symbol: str, trade_date: date, rows: Iterable[dict[str, Any]]) -> DailyFeature | None:
    rows = list(rows)
    if len(rows) < 60:
        return None
    prev_close = _float(rows[-1].get("close"))
    if prev_close <= 0.0:
        return None
    last20 = rows[-20:]
    last60 = rows[-60:]
    sma20 = _avg(_float(row.get("close")) for row in last20)
    sma60 = _avg(_float(row.get("close")) for row in last60)
    avg_vol20 = _avg(max(_float(row.get("volume")), 0.0) for row in last20)
    high20 = max(_float(row.get("high")) for row in last20)
    low20 = min(_float(row.get("low")) for row in last20)
    high60 = max(_float(row.get("high")) for row in last60)
    low60 = min(_float(row.get("low")) for row in last60)
    feature = DailyFeature(
        symbol=symbol,
        trade_date=trade_date,
        prev_close=prev_close,
        atr14=max(_atr(rows, 14), prev_close * 0.01),
        return_5d=_return_pct(rows, 5),
        return_20d=_return_pct(rows, 20),
        return_60d=_return_pct(rows, 60),
        adv20_krw=_avg(max(_float(row.get("close")), 0.0) * max(_float(row.get("volume")), 0.0) for row in last20),
        volume_ratio_20d=max(_float(rows[-1].get("volume")), 0.0) / max(avg_vol20, 1.0),
        close20_loc=(prev_close - low20) / max(high20 - low20, 1e-9),
        close60_loc=(prev_close - low60) / max(high60 - low60, 1e-9),
        above_sma20=prev_close >= sma20,
        above_sma60=prev_close >= sma60,
    )
    return feature


def flow_feature(dataset: KALCBFirst30Dataset, symbol: str, trade_date: date) -> FlowFeature:
    symbol = str(symbol).zfill(6)
    key = (*_flow_series_cache_key(dataset, symbol), trade_date)
    if key in _FLOW_FEATURE_CACHE:
        return _FLOW_FEATURE_CACHE[key]
    series = _flow_feature_series(dataset, symbol)
    if trade_date in series:
        feature = series[trade_date]
        _FLOW_FEATURE_CACHE[key] = feature
        return feature
    flows = prior_flow_rows(dataset, symbol, trade_date)
    foreign_flows = prior_foreign_flow_rows(dataset, symbol, trade_date)
    institutional_flows = prior_institutional_flow_rows(dataset, symbol, trade_date)
    daily = prior_daily_rows(dataset, symbol, trade_date)
    feature = _flow_feature_from_prior(flows, foreign_flows, institutional_flows, daily)
    _FLOW_FEATURE_CACHE[key] = feature
    return feature


def _flow_feature_series(dataset: KALCBFirst30Dataset, symbol: str) -> dict[date, FlowFeature]:
    symbol = str(symbol).zfill(6)
    key = _flow_series_cache_key(dataset, symbol)
    cached = _FLOW_FEATURE_BY_DAY_CACHE.get(key)
    if cached is not None:
        return cached
    _, daily_rows = _indexed_symbol_rows(dataset, "daily", symbol)
    _, flow_rows = _indexed_symbol_rows(dataset, "flow", symbol)
    _, foreign_rows = _indexed_symbol_rows(dataset, "foreign_flow", symbol)
    _, institutional_rows = _indexed_symbol_rows(dataset, "institutional_flow", symbol)
    normalized = _normalized_flow_rows(flow_rows, foreign_rows, institutional_rows, daily_rows)
    by_day: dict[date, FlowFeature] = {}
    flow_dates = tuple(item[0] for item in normalized)
    for trade_date in dataset.trading_dates:
        index = bisect.bisect_left(flow_dates, trade_date)
        by_day[trade_date] = _flow_feature_from_normalized(normalized[:index])
    _FLOW_FEATURE_BY_DAY_CACHE[key] = by_day
    return by_day


def _flow_feature_from_prior(
    flows: Iterable[dict[str, Any]],
    foreign_flows: Iterable[dict[str, Any]],
    institutional_flows: Iterable[dict[str, Any]],
    daily: Iterable[dict[str, Any]],
) -> FlowFeature:
    if not (flows or foreign_flows or institutional_flows) or not daily:
        return FlowFeature(available=False)
    return _flow_feature_from_normalized(_normalized_flow_rows(flows, foreign_flows, institutional_flows, daily))


def _normalized_flow_rows(
    flows: Iterable[dict[str, Any]],
    foreign_flows: Iterable[dict[str, Any]],
    institutional_flows: Iterable[dict[str, Any]],
    daily: Iterable[dict[str, Any]],
) -> tuple[tuple[date, float, float, float, float, float, float, float], ...]:
    daily_by_date = {_row_date(row): row for row in daily}
    foreign_by_date = {_row_date(row): _flow_value(row, "foreign_net") for row in foreign_flows}
    inst_by_date = {_row_date(row): _flow_value(row, "institutional_net", "inst_net") for row in institutional_flows}
    for flow in flows:
        flow_date = _row_date(flow)
        foreign_by_date.setdefault(flow_date, _flow_value(flow, "foreign_net"))
        inst_by_date.setdefault(flow_date, _flow_value(flow, "inst_net", "institutional_net"))
    normalized: list[tuple[date, float, float, float, float, float, float, float]] = []
    notional = []
    for flow_date in sorted(set(foreign_by_date) | set(inst_by_date)):
        daily_row = daily_by_date.get(flow_date)
        if daily_row is None:
            continue
        close = max(_float(daily_row.get("close")), 0.0)
        volume = max(_float(daily_row.get("volume")), 1.0)
        foreign = foreign_by_date.get(flow_date, 0.0)
        inst = inst_by_date.get(flow_date, 0.0)
        combined = foreign + inst
        foreign_norm = foreign / volume
        inst_norm = inst / volume
        combined_norm = combined / volume
        agreement = min(max(foreign_norm, 0.0), max(inst_norm, 0.0))
        divergence = max(0.0, -foreign_norm * inst_norm) ** 0.5 if foreign_norm * inst_norm < 0.0 else abs(foreign_norm - inst_norm) * 0.25
        normalized.append((flow_date, combined_norm, foreign_norm, inst_norm, agreement, divergence, combined * close, volume * close))
    return tuple(sorted(normalized, key=lambda item: item[0]))


def _flow_feature_from_normalized(normalized: Iterable[tuple[date, float, float, float, float, float, float, float]]) -> FlowFeature:
    normalized = tuple(normalized)
    if not normalized:
        return FlowFeature(available=False)
    combined_values = [item[1] for item in normalized]
    foreign_values = [item[2] for item in normalized]
    inst_values = [item[3] for item in normalized]
    agreement_values = [item[4] for item in normalized]
    divergence_values = [item[5] for item in normalized]
    notional_5d = sum(item[6] for item in normalized[-5:]) / max(sum(item[7] for item in normalized[-5:]), 1.0)
    recent20 = combined_values[-20:]
    prior20 = combined_values[-40:-20]
    mean20 = _avg(recent20)
    std20 = _std(recent20)
    accel_base = _avg(prior20) if prior20 else mean20
    foreign_mean20 = _avg(foreign_values[-20:])
    inst_mean20 = _avg(inst_values[-20:])
    foreign_std20 = _std(foreign_values[-20:])
    inst_std20 = _std(inst_values[-20:])
    foreign_accel_base = _avg(foreign_values[-40:-20]) if foreign_values[-40:-20] else foreign_mean20
    inst_accel_base = _avg(inst_values[-40:-20]) if inst_values[-40:-20] else inst_mean20
    foreign_5d = _avg(foreign_values[-5:])
    inst_5d = _avg(inst_values[-5:])
    feature = FlowFeature(
        available=True,
        foreign_1d=foreign_values[-1],
        foreign_3d=_avg(foreign_values[-3:]),
        foreign_5d=foreign_5d,
        foreign_20d=foreign_mean20,
        inst_1d=inst_values[-1],
        inst_3d=_avg(inst_values[-3:]),
        inst_5d=inst_5d,
        inst_20d=inst_mean20,
        combined_1d=combined_values[-1],
        combined_3d=_avg(combined_values[-3:]),
        combined_5d=_avg(combined_values[-5:]),
        combined_20d=mean20,
        combined_notional_5d=notional_5d,
        positive_days_5d=float(sum(1 for value in combined_values[-5:] if value > 0.0)),
        foreign_positive_days_5d=float(sum(1 for value in foreign_values[-5:] if value > 0.0)),
        inst_positive_days_5d=float(sum(1 for value in inst_values[-5:] if value > 0.0)),
        acceleration=_avg(combined_values[-3:]) - accel_base,
        foreign_acceleration=_avg(foreign_values[-3:]) - foreign_accel_base,
        inst_acceleration=_avg(inst_values[-3:]) - inst_accel_base,
        z_score=(combined_values[-1] - mean20) / std20 if std20 > 0.0 else 0.0,
        foreign_z=(foreign_values[-1] - foreign_mean20) / foreign_std20 if foreign_std20 > 0.0 else 0.0,
        inst_z=(inst_values[-1] - inst_mean20) / inst_std20 if inst_std20 > 0.0 else 0.0,
        agreement_5d=_avg(agreement_values[-5:]),
        divergence_5d=_avg(divergence_values[-5:]),
        sponsorship_balance_5d=foreign_5d - inst_5d,
    )
    return feature


def first30_intraday_feature(daily: DailyFeature, bars: tuple[MarketBar, ...]) -> First30Intraday | None:
    feature = shared_first30_feature(daily, bars)
    if feature is None:
        return None
    return First30Intraday(
        open=feature.open,
        high=feature.high,
        low=feature.low,
        close=feature.close,
        vwap=feature.vwap,
        volume=feature.volume,
        expected_30m_volume=feature.expected_30m_volume,
    )


def shared_first30_feature(daily: DailyFeature, bars: tuple[MarketBar, ...]):
    return build_first30_features(
        bars,
        prior_close=daily.prev_close,
        daily_atr=daily.atr14,
        expected_30m_volume=max(_expected_30m_volume(daily), 1.0),
    )


def build_coarse_specs() -> list[First30Spec]:
    specs: list[First30Spec] = []
    modes = ("gap_hold", "momentum", "vwap_strength", "flow_confirmed", "efficient", "hybrid")
    top_ns = (1, 2, 3, 4, 6, 8, 10, 12)
    filters = [
        {"min_gap": 0.002, "max_gap": 0.10, "min_first30_ret": 0.0, "min_vwap_ret": 0.0, "min_prior_ret5": 0.03, "min_low_vs_prev_close": -0.02},
        {"min_gap": 0.002, "max_gap": 0.08, "min_first30_ret": 0.0, "min_vwap_ret": 0.0, "min_prior_ret5": 0.03, "min_low_vs_prev_close": -0.01},
        {"min_gap": 0.003, "max_gap": 0.10, "min_first30_ret": 0.002, "min_vwap_ret": 0.0, "min_prior_ret5": 0.03, "min_low_vs_prev_close": -0.02},
        {"min_gap": 0.005, "max_gap": 0.10, "min_first30_ret": 0.0, "min_prior_ret5": 0.05},
        {"min_gap": 0.0, "max_gap": 0.06, "min_first30_ret": 0.003, "min_vwap_ret": 0.001, "min_prior_ret5": 0.03},
        {"min_first30_ret": 0.005, "min_vwap_ret": 0.001, "min_prior_ret5": 0.05},
        {"min_first30_ret": 0.008, "min_vwap_ret": 0.002},
        {"min_vwap_ret": 0.003, "min_rel_volume": 0.75, "min_prior_ret5": 0.03},
        {"min_rel_volume": 1.25, "min_close_location": 0.60},
        {"min_rel_volume": 2.0, "min_close_location": 0.70},
        {"min_close_location": 0.75, "min_low_vs_prev_close": -0.01},
        {"max_open_drawdown": 0.006, "min_first30_ret": 0.0},
        {"max_open_drawdown": 0.010, "min_vwap_ret": 0.0},
        {"max_range_atr": 0.75, "min_first30_ret": 0.0},
        {"max_range_atr": 1.00, "min_vwap_ret": 0.0},
        {"min_prior_ret5": 0.05, "min_prior_ret20": 0.0},
        {"min_prior_ret5": 0.10},
        {"min_prior_ret20": 0.05},
        {"max_prior_ret20": 0.30, "min_prior_ret5": 0.03},
        {"max_prior_ret20": 0.50, "min_gap": 0.002, "max_gap": 0.10},
        {"min_flow_5d": 0.0},
        {"min_foreign_flow_5d": 0.0},
        {"min_inst_flow_5d": 0.0},
        {"min_foreign_flow_5d": 0.0, "min_inst_flow_5d": 0.0},
        {"min_flow_z": 0.0},
        {"min_flow_agreement": 0.0},
        {"max_flow_divergence": 0.01},
        {"min_sector_flow": 0.0},
        {"min_flow_5d": 0.0, "min_sector_flow": 0.0},
        {"min_flow_agreement": 0.0, "min_sector_flow": 0.0},
        {"min_flow_z": 0.0, "min_prior_ret5": 0.03},
        {"min_market_score": 0.0, "min_prior_ret5": 0.03},
        {"min_market_score": 0.2, "min_gap": 0.002, "max_gap": 0.10},
        {"require_close_above_prev": True, "min_first30_ret": 0.0},
        {"min_gap": -0.01, "max_gap": 0.05, "min_first30_ret": 0.003, "min_vwap_ret": 0.001},
        {"min_gap": -0.02, "max_gap": 0.10, "min_first30_ret": 0.005, "min_prior_ret5": 0.05},
    ]
    for mode in modes:
        for top_n in top_ns:
            for values in filters:
                specs.append(_spec(mode, top_n, values))
    return _dedupe_specs(specs)


def build_refinement_specs(seeds: list[First30Spec], *, existing: set[str]) -> list[First30Spec]:
    specs: list[First30Spec] = []
    for seed in seeds:
        for top_n in _near(seed.top_n, (1, 2, 3, 4, 5, 6, 8, 10, 12, 16)):
            for min_ret in _near(seed.min_first30_ret, (0.0, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.010, 0.0125, 0.015)):
                specs.append(replace(seed, name="", top_n=int(top_n), min_first30_ret=float(min_ret)))
        for min_vwap in _near(seed.min_vwap_ret, (-0.002, -0.001, 0.0, 0.001, 0.002, 0.003, 0.005, 0.0075)):
            specs.append(replace(seed, name="", min_vwap_ret=float(min_vwap)))
        for min_gap in _near(seed.min_gap, (-0.02, -0.01, 0.0, 0.002, 0.003, 0.005, 0.008)):
            for max_gap in _near(seed.max_gap, (0.03, 0.05, 0.08, 0.10, 0.12, 0.15)):
                if float(max_gap) > float(min_gap):
                    specs.append(replace(seed, name="", min_gap=float(min_gap), max_gap=float(max_gap)))
        for relvol in _near(seed.min_rel_volume, (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)):
            specs.append(replace(seed, name="", min_rel_volume=float(relvol)))
        for close_loc in _near(seed.min_close_location, (0.0, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90)):
            specs.append(replace(seed, name="", min_close_location=float(close_loc)))
        for low_hold in _near(seed.min_low_vs_prev_close, (-0.05, -0.03, -0.02, -0.01, -0.005, 0.0)):
            specs.append(replace(seed, name="", min_low_vs_prev_close=float(low_hold)))
        for flow in _near(seed.min_flow_5d, (-9.99, -0.02, -0.01, 0.0, 0.01, 0.02)):
            specs.append(replace(seed, name="", min_flow_5d=float(flow)))
        for foreign_flow in _near(seed.min_foreign_flow_5d, (-9.99, -0.02, -0.01, 0.0, 0.01, 0.02)):
            specs.append(replace(seed, name="", min_foreign_flow_5d=float(foreign_flow)))
        for inst_flow in _near(seed.min_inst_flow_5d, (-9.99, -0.02, -0.01, 0.0, 0.01, 0.02)):
            specs.append(replace(seed, name="", min_inst_flow_5d=float(inst_flow)))
        for flow_z in _near(seed.min_flow_z, (-9.99, -1.0, -0.5, 0.0, 0.5, 1.0)):
            specs.append(replace(seed, name="", min_flow_z=float(flow_z)))
        for agreement in _near(seed.min_flow_agreement, (-9.99, -0.005, 0.0, 0.002, 0.005, 0.01)):
            specs.append(replace(seed, name="", min_flow_agreement=float(agreement)))
        for divergence in _near(seed.max_flow_divergence, (0.005, 0.01, 0.02, 0.04, 9.99)):
            specs.append(replace(seed, name="", max_flow_divergence=float(divergence)))
        for sector_flow in _near(seed.min_sector_flow, (-9.99, -0.01, 0.0, 0.01, 0.02)):
            specs.append(replace(seed, name="", min_sector_flow=float(sector_flow)))
        for market_score in _near(seed.min_market_score, (-9.99, -0.2, 0.0, 0.2, 0.4)):
            specs.append(replace(seed, name="", min_market_score=float(market_score)))
    named = [_name_spec(spec) for spec in specs]
    return [spec for spec in _dedupe_specs(named) if _spec_signature(spec) not in existing]


def select(spec: First30Spec, contexts: dict[date, tuple[First30Context, ...]]) -> list[Selection]:
    selections: list[Selection] = []
    for day, items in contexts.items():
        scored = [(score_candidate(spec, ctx), ctx.symbol) for ctx in items if passes(spec, ctx)]
        scored.sort(key=lambda item: (-item[0], item[1]))
        selections.extend(Selection(day, symbol, score, spec.score_mode) for score, symbol in scored[: max(1, spec.top_n)])
    return selections


def passes(spec: First30Spec, ctx: First30Context) -> bool:
    if ctx.first30_ret < spec.min_first30_ret:
        return False
    if ctx.vwap_ret < spec.min_vwap_ret:
        return False
    if ctx.gap < spec.min_gap or ctx.gap > spec.max_gap:
        return False
    if ctx.rel_volume < spec.min_rel_volume:
        return False
    if ctx.close_location < spec.min_close_location:
        return False
    if abs(min(ctx.open_drawdown, 0.0)) > spec.max_open_drawdown:
        return False
    if ctx.range_atr > spec.max_range_atr:
        return False
    if ctx.daily.return_5d < spec.min_prior_ret5:
        return False
    if ctx.daily.return_20d < spec.min_prior_ret20 or ctx.daily.return_20d > spec.max_prior_ret20:
        return False
    if ctx.daily.return_60d < spec.min_prior_ret60:
        return False
    if ctx.low_vs_prev_close < spec.min_low_vs_prev_close:
        return False
    if ctx.flow.combined_5d < spec.min_flow_5d:
        return False
    if ctx.flow.foreign_5d < spec.min_foreign_flow_5d:
        return False
    if ctx.flow.inst_5d < spec.min_inst_flow_5d:
        return False
    if ctx.flow.z_score < spec.min_flow_z:
        return False
    if ctx.flow.agreement_5d < spec.min_flow_agreement:
        return False
    if ctx.flow.divergence_5d > spec.max_flow_divergence:
        return False
    if ctx.flow.sector_flow_5d < spec.min_sector_flow:
        return False
    if ctx.market.score < spec.min_market_score:
        return False
    if spec.require_close_above_prev and ctx.intraday.close <= ctx.daily.prev_close:
        return False
    if spec.score_mode in {"momentum", "hybrid", "efficient", "flow_confirmed"} and ctx.first30_ret < 0.0:
        return False
    if spec.score_mode in {"vwap_strength", "flow_confirmed"} and ctx.vwap_ret < 0.0:
        return False
    if spec.score_mode == "gap_hold" and (ctx.gap < 0.002 or ctx.first30_ret < 0.0 or ctx.low_vs_prev_close < -0.02):
        return False
    if spec.score_mode == "flow_confirmed" and not (ctx.flow.combined_5d > 0.0 or ctx.flow.z_score > 0.0 or ctx.flow.sector_flow_5d > 0.0):
        return False
    return True


def score_candidate(spec: First30Spec, ctx: First30Context) -> float:
    rel_volume_bonus = min(ctx.rel_volume, 5.0) * 0.001
    close_bonus = ctx.close_location * 0.002
    flow_bonus = (
        0.025 * max(ctx.flow.combined_5d, 0.0)
        + 0.015 * max(ctx.flow.foreign_5d, 0.0)
        + 0.015 * max(ctx.flow.inst_5d, 0.0)
        + 0.020 * max(ctx.flow.agreement_5d, 0.0)
        + 0.001 * max(ctx.flow.z_score, 0.0)
        + 0.04 * max(ctx.flow.sector_flow_5d, 0.0)
        - 0.010 * max(ctx.flow.divergence_5d, 0.0)
    )
    market_bonus = 0.002 * max(ctx.market.score, 0.0)
    if spec.score_mode == "gap_hold":
        return ctx.gap + ctx.first30_ret + 0.25 * ctx.vwap_ret + max(ctx.low_vs_prev_close, 0.0) + close_bonus + flow_bonus
    if spec.score_mode == "momentum":
        return ctx.first30_ret + 0.25 * ctx.vwap_ret + 0.05 * max(ctx.gap, 0.0) + rel_volume_bonus + close_bonus + market_bonus
    if spec.score_mode == "vwap_strength":
        return ctx.vwap_ret + 0.40 * max(ctx.first30_ret, 0.0) + 0.05 * max(ctx.gap, 0.0) + rel_volume_bonus + flow_bonus
    if spec.score_mode == "flow_confirmed":
        return 0.55 * max(ctx.first30_ret, 0.0) + 0.30 * max(ctx.vwap_ret, 0.0) + flow_bonus + 0.001 * ctx.flow.positive_days_5d + close_bonus
    if spec.score_mode == "efficient":
        return (ctx.first30_ret + 0.50 * ctx.vwap_ret + 0.05 * max(ctx.gap, 0.0)) / max(ctx.range_atr, 0.15) + rel_volume_bonus + flow_bonus
    if spec.score_mode == "daily_sector_leadership":
        stock_leadership, sector_confirmation, opening_quality = _daily_sector_opening_components(ctx)
        return 0.42 * stock_leadership + 0.26 * sector_confirmation + 0.32 * opening_quality + 0.05 * max(ctx.first30_ret, 0.0)
    if spec.score_mode == "daily_sector_gap_retention":
        stock_leadership, sector_confirmation, opening_quality = _daily_sector_opening_components(ctx)
        gap_retention = _gap_retention_ratio(ctx)
        return 0.38 * _clip(gap_retention / 1.25) + 0.22 * _clip(math.log1p(max(ctx.rel_volume, 0.0)) / 2.0) + 0.22 * stock_leadership + 0.18 * sector_confirmation + 0.04 * opening_quality
    if spec.score_mode == "daily_sector_relvol_leadership":
        stock_leadership, sector_confirmation, opening_quality = _daily_sector_opening_components(ctx)
        gap_relvol = max(ctx.gap, 0.0) * math.log1p(max(ctx.rel_volume, 0.0))
        return 0.36 * _clip(gap_relvol / 0.035) + 0.30 * stock_leadership + 0.18 * sector_confirmation + 0.16 * opening_quality
    return ctx.first30_ret + ctx.vwap_ret + 0.20 * max(ctx.gap, 0.0) + rel_volume_bonus + close_bonus + flow_bonus + market_bonus


def _daily_sector_opening_components(ctx: First30Context) -> tuple[float, float, float]:
    sector = ctx.sector_daily
    sector_score_pct = float(getattr(sector, "score_pct", 50.0) if sector is not None else 50.0)
    sector_participation = float(getattr(sector, "participation", 0.0) if sector is not None else 0.0)
    sector_ret5 = float(getattr(sector, "ret_5d", 0.0) if sector is not None else 0.0)
    sector_ret20 = float(getattr(sector, "ret_20d", 0.0) if sector is not None else 0.0)
    spread5 = float(ctx.daily.return_5d) - sector_ret5
    spread20 = float(ctx.daily.return_20d) - sector_ret20
    daily_acceleration = float(ctx.daily.return_5d) - 0.25 * float(ctx.daily.return_20d)
    rel_volume_log = math.log1p(max(float(ctx.rel_volume), 0.0))
    gap_relvol = max(float(ctx.gap), 0.0) * rel_volume_log
    gap_retention = _gap_retention_ratio(ctx)
    stock_leadership = (
        0.34 * _clip(0.5 + spread20 / 0.40)
        + 0.24 * _clip(0.5 + spread5 / 0.16)
        + 0.22 * _clip(0.5 + daily_acceleration / 0.12)
        + 0.20 * _clip(float(ctx.daily.close20_loc))
    )
    sector_confirmation = 0.62 * _clip(sector_score_pct / 100.0) + 0.38 * _clip(sector_participation)
    opening_quality = (
        0.34 * _clip(gap_relvol / 0.035)
        + 0.28 * _clip(gap_retention / 1.25)
        + 0.22 * _clip(rel_volume_log / 2.0)
        + 0.16 * _clip(0.5 + float(ctx.first30_ret) / 0.06)
    )
    return stock_leadership, sector_confirmation, opening_quality


def _gap_retention_ratio(ctx: First30Context) -> float:
    return float(ctx.low_vs_prev_close) / max(abs(float(ctx.gap)), 1e-6) if float(ctx.gap) > 0.0 else 0.0


def evaluate_selections(
    dataset: KALCBFirst30Dataset,
    selections: list[Selection],
    config: KALCBConfig,
) -> list[OpportunityRow]:
    cost = _round_trip_cost_pct(config)
    rows: list[OpportunityRow] = []
    for selection in selections:
        bars = dataset.bars_by_key.get((selection.trade_date, selection.symbol), ())
        daily = daily_feature(dataset, selection.symbol, selection.trade_date)
        if daily is None:
            continue
        post = _post_entry_bars(bars, config)
        if not post:
            continue
        entry = max(float(post[0].open), 1e-9)
        first = _first30_bars(bars)
        signal_low = min((float(bar.low) for bar in first), default=float(post[0].low))
        risk = _baseline_risk(entry, signal_low, float(post[0].low), daily.atr14, config)
        high = max(float(bar.high) for bar in post)
        low = min(float(bar.low) for bar in post)
        close = float(post[-1].close)
        gross = close / entry - 1.0
        rows.append(
            OpportunityRow(
                trade_date=selection.trade_date,
                symbol=selection.symbol,
                family=selection.family,
                score=selection.score,
                gross_eod_pct=gross,
                net_eod_pct=gross - cost,
                mfe_r=max(0.0, (high - entry) / risk),
                mae_r=(low - entry) / risk,
                entry_price=entry,
                risk_pct=risk / max(entry, 1e-9),
            )
        )
    return rows


def summarize(name: str, rows: list[OpportunityRow], *, session_dates: Iterable[date], slot_count: int) -> dict[str, Any]:
    sessions = list(session_dates)
    by_day: dict[date, list[OpportunityRow]] = {}
    for row in rows:
        by_day.setdefault(row.trade_date, []).append(row)
    slot = max(1, int(slot_count))
    daily_gross = [sum(item.gross_eod_pct for item in by_day.get(day, [])) / slot for day in sessions]
    daily_net = [sum(item.net_eod_pct for item in by_day.get(day, [])) / slot for day in sessions]
    portfolio_gross, portfolio_net, portfolio_positions = _portfolio_proxy_daily_returns(by_day, sessions)
    active_gross = [_avg(item.gross_eod_pct for item in items) for _, items in sorted(by_day.items())]
    active_net = [_avg(item.net_eod_pct for item in items) for _, items in sorted(by_day.items())]
    mfe_values = [row.mfe_r for row in rows]
    mae_values = [row.mae_r for row in rows]
    return {
        "name": name,
        "candidate_days": len(rows),
        "active_days": len(by_day),
        "session_count": len(sessions),
        "slot_count": slot,
        "active_day_share": len(by_day) / max(float(len(sessions)), 1.0),
        "avg_candidates_per_session": len(rows) / max(float(len(sessions)), 1.0),
        "avg_candidates_per_active_day": len(rows) / max(float(len(by_day)), 1.0),
        "avg_gross_eod_pct": _avg(row.gross_eod_pct for row in rows),
        "avg_net_eod_pct": _avg(row.net_eod_pct for row in rows),
        "active_day_gross_pct": _avg(active_gross),
        "active_day_net_pct": _avg(active_net),
        "calendar_day_gross_pct": sum(daily_gross) / max(float(len(sessions)), 1.0),
        "calendar_day_net_pct": sum(daily_net) / max(float(len(sessions)), 1.0),
        "slot_cumulative_gross_return_pct": _compound(daily_gross),
        "slot_cumulative_net_return_pct": _compound(daily_net),
        "slot_max_drawdown_gross_pct": _max_drawdown(daily_gross),
        "slot_max_drawdown_net_pct": _max_drawdown(daily_net),
        "portfolio_proxy_gross_return_pct": _compound(portfolio_gross),
        "portfolio_proxy_net_return_pct": _compound(portfolio_net),
        "portfolio_proxy_calendar_day_net_pct": sum(portfolio_net) / max(float(len(sessions)), 1.0),
        "portfolio_proxy_active_day_net_pct": _avg(value for value, count in zip(portfolio_net, portfolio_positions) if count > 0),
        "portfolio_proxy_max_drawdown_pct": _max_drawdown(portfolio_net),
        "portfolio_proxy_avg_positions_per_session": sum(portfolio_positions) / max(float(len(sessions)), 1.0),
        "portfolio_proxy_avg_positions_per_active_day": sum(portfolio_positions) / max(float(sum(1 for count in portfolio_positions if count > 0)), 1.0),
        "avg_mfe_r": _avg(mfe_values),
        "median_mfe_r": _med(mfe_values),
        "mfe_ge_0_5_share": _ratio(sum(1 for value in mfe_values if value >= 0.5), len(mfe_values)),
        "mfe_ge_0_75_share": _ratio(sum(1 for value in mfe_values if value >= 0.75), len(mfe_values)),
        "mfe_ge_1_0_share": _ratio(sum(1 for value in mfe_values if value >= 1.0), len(mfe_values)),
        "avg_mae_r": _avg(mae_values),
        "mae_le_neg_1_share": _ratio(sum(1 for value in mae_values if value <= -1.0), len(mae_values)),
        "gross_win_share": _ratio(sum(1 for row in rows if row.gross_eod_pct > 0.0), len(rows)),
        "net_win_share": _ratio(sum(1 for row in rows if row.net_eod_pct > 0.0), len(rows)),
    }


def _portfolio_proxy_daily_returns(
    by_day: dict[date, list[OpportunityRow]],
    sessions: list[date],
) -> tuple[list[float], list[float], list[int]]:
    """Approximate executable 09:30/EOD first30 deployment without exit tuning."""
    policy = STAGE2_PORTFOLIO_POLICY
    risk_budget = float(policy["risk_per_trade_pct"])
    notional_cap = float(policy["max_position_notional_pct"])
    leverage = float(policy["intraday_leverage"])
    max_positions = max(1, int(policy["max_positions"]))
    daily_gross: list[float] = []
    daily_net: list[float] = []
    position_counts: list[int] = []
    for day in sessions:
        rows = sorted(by_day.get(day, []), key=lambda item: (-float(item.score), item.symbol))
        selected = rows[:max_positions]
        count = len(selected)
        position_counts.append(count)
        if count <= 0:
            daily_gross.append(0.0)
            daily_net.append(0.0)
            continue
        per_slot_leverage = leverage / max(float(count), 1.0)
        gross = 0.0
        net = 0.0
        for row in selected:
            risk_pct = max(float(row.risk_pct), 0.0005)
            risk_cap = risk_budget / risk_pct
            weight = max(0.0, min(notional_cap, per_slot_leverage, risk_cap))
            gross += weight * float(row.gross_eod_pct)
            net += weight * float(row.net_eod_pct)
        daily_gross.append(gross)
        daily_net.append(net)
    return daily_gross, daily_net, position_counts


def _evaluate_specs(
    specs: list[First30Spec],
    contexts: dict[date, tuple[First30Context, ...]],
    dataset: KALCBFirst30Dataset,
    cfg: KALCBConfig,
    folds: list[tuple[date, date]],
    output_dir: Path,
    *,
    stage: str,
    completed_offset: int,
    total: int,
    max_workers: int,
    seed_rows: list[SweepResult] | None = None,
) -> list[SweepResult]:
    if not specs:
        return []
    rows: list[SweepResult] = []
    # This loop is CPU-bound and shares large cached context objects. Threads add
    # GIL contention and have proven less reliable than deterministic iteration.
    for spec in specs:
        row = evaluate_spec(spec, contexts, dataset, cfg, folds)
        rows.append(row)
        _record_progress(output_dir, stage, completed_offset + len(rows), total, [*(seed_rows or []), *rows], row)
    return rows


def evaluate_spec(
    spec: First30Spec,
    contexts: dict[date, tuple[First30Context, ...]],
    dataset: KALCBFirst30Dataset,
    cfg: KALCBConfig,
    folds: list[tuple[date, date]] | None = None,
) -> SweepResult:
    selections = select(spec, contexts)
    rows = evaluate_selections(dataset, selections, cfg)
    full = summarize(spec.name, rows, session_dates=dataset.trading_dates, slot_count=spec.top_n)
    full_score, reject_reason = score_summary(full)
    fold_rows: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(folds or [], start=1):
        fold_dates = [day for day in dataset.trading_dates if start <= day <= end]
        fold_data = [row for row in rows if start <= row.trade_date <= end]
        summary = summarize(f"{spec.name}_fold{index}", fold_data, session_dates=fold_dates, slot_count=spec.top_n)
        score, reject = score_summary(summary)
        fold_rows.append(
            {
                "fold": index,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "score": round(score, 6),
                "rejected": bool(reject),
                "reject_reason": reject,
                "metrics": _compact_summary(summary),
            }
        )
    fold_scores = [float(row["score"]) for row in fold_rows]
    median_fold = median(fold_scores) if fold_scores else full_score
    worst_fold = min(fold_scores) if fold_scores else full_score
    stability_score = 0.55 * full_score + 0.30 * median_fold + 0.15 * worst_fold
    rejected = bool(reject_reason) or sum(1 for row in fold_rows if row["rejected"]) > max(0, len(fold_rows) // 2)
    if rejected and not reject_reason:
        reject_reason = "unstable_across_folds"
    return SweepResult(
        spec=spec,
        score=round(0.0 if rejected else stability_score, 6),
        full_score=round(full_score, 6),
        median_fold_score=round(median_fold, 6),
        worst_fold_score=round(worst_fold, 6),
        rejected=rejected,
        reject_reason=reject_reason,
        metrics=_compact_summary(full),
        folds=tuple(fold_rows),
    )


def score_summary(summary: dict[str, Any]) -> tuple[float, str]:
    candidate_days = float(summary.get("candidate_days", 0.0) or 0.0)
    active_share = float(summary.get("active_day_share", 0.0) or 0.0)
    if candidate_days < 60.0:
        return 0.0, f"too_few_candidate_days ({candidate_days:.0f} < 60)"
    if active_share < 0.20:
        return 0.0, f"too_sparse ({active_share:.3f} < 0.200)"
    portfolio_net = float(summary.get("portfolio_proxy_net_return_pct", summary.get("slot_cumulative_net_return_pct", 0.0)) or 0.0)
    avg_mfe = float(summary.get("avg_mfe_r", 0.0) or 0.0)
    calendar_net = float(summary.get("portfolio_proxy_calendar_day_net_pct", summary.get("calendar_day_net_pct", 0.0)) or 0.0)
    active_net = float(summary.get("portfolio_proxy_active_day_net_pct", summary.get("active_day_net_pct", 0.0)) or 0.0)
    mfe_075 = float(summary.get("mfe_ge_0_75_share", 0.0) or 0.0)
    win = float(summary.get("net_win_share", 0.0) or 0.0)
    mae_bad = float(summary.get("mae_le_neg_1_share", 0.0) or 0.0)
    names = float(summary.get("avg_candidates_per_active_day", 0.0) or 0.0)
    portfolio_dd = abs(float(summary.get("portfolio_proxy_max_drawdown_pct", summary.get("slot_max_drawdown_net_pct", 0.0)) or 0.0))
    score = (
        0.38 * _clip(portfolio_net / 0.45)
        + 0.15 * _return_score(calendar_net, target=0.0025)
        + 0.12 * _return_score(active_net, target=0.004)
        + 0.18 * _clip(avg_mfe / 1.50)
        + 0.08 * _clip(mfe_075)
        + 0.05 * _return_score(win - 0.50, target=0.08)
        + 0.05 * _clip(active_share / 0.70)
    )
    penalty = (
        0.06 * _clip(mae_bad / 0.40)
        + 0.035 * _clip(portfolio_dd / 0.12)
        + 0.020 * _clip(max(0.0, names - STAGE2_PORTFOLIO_POLICY["max_positions"]) / STAGE2_PORTFOLIO_POLICY["max_positions"])
    )
    return max(0.0, 100.0 * (score - penalty)), ""


def _daily_series_cache_key(dataset: KALCBFirst30Dataset, symbol: str) -> tuple[Any, ...]:
    rows = dataset.daily_by_symbol.get(str(symbol).zfill(6))
    return (id(dataset), str(symbol).zfill(6), id(rows), len(rows or ()))


def _flow_series_cache_key(dataset: KALCBFirst30Dataset, symbol: str) -> tuple[Any, ...]:
    symbol = str(symbol).zfill(6)
    daily_rows = dataset.daily_by_symbol.get(symbol)
    flow_rows = dataset.flow_by_symbol.get(symbol)
    foreign_rows = dataset.foreign_flow_by_symbol.get(symbol)
    institutional_rows = dataset.institutional_flow_by_symbol.get(symbol)
    return (
        id(dataset),
        symbol,
        id(daily_rows),
        len(daily_rows or ()),
        id(flow_rows),
        len(flow_rows or ()),
        id(foreign_rows),
        len(foreign_rows or ()),
        id(institutional_rows),
        len(institutional_rows or ()),
    )


def prior_daily_rows(dataset: KALCBFirst30Dataset, symbol: str, trade_date: date) -> list[dict[str, Any]]:
    return _prior_rows(dataset, "daily", symbol, trade_date)


def prior_flow_rows(dataset: KALCBFirst30Dataset, symbol: str, trade_date: date) -> list[dict[str, Any]]:
    return _prior_rows(dataset, "flow", symbol, trade_date)


def prior_foreign_flow_rows(dataset: KALCBFirst30Dataset, symbol: str, trade_date: date) -> list[dict[str, Any]]:
    return _prior_rows(dataset, "foreign_flow", symbol, trade_date)


def prior_institutional_flow_rows(dataset: KALCBFirst30Dataset, symbol: str, trade_date: date) -> list[dict[str, Any]]:
    return _prior_rows(dataset, "institutional_flow", symbol, trade_date)


def _prior_rows(dataset: KALCBFirst30Dataset, table: str, symbol: str, trade_date: date) -> list[dict[str, Any]]:
    dates, rows = _indexed_symbol_rows(dataset, table, str(symbol).zfill(6))
    index = bisect.bisect_left(dates, trade_date)
    return list(rows[:index])


def _indexed_symbol_rows(dataset: KALCBFirst30Dataset, table: str, symbol: str) -> tuple[tuple[date, ...], tuple[dict[str, Any], ...]]:
    if table == "daily":
        raw_rows = dataset.daily_by_symbol.get(symbol, [])
    elif table == "flow":
        raw_rows = dataset.flow_by_symbol.get(symbol, [])
    elif table == "foreign_flow":
        raw_rows = dataset.foreign_flow_by_symbol.get(symbol, [])
    elif table == "institutional_flow":
        raw_rows = dataset.institutional_flow_by_symbol.get(symbol, [])
    else:
        raw_rows = []
    key = (id(dataset), table, symbol, id(raw_rows), len(raw_rows))
    cached = _ROW_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    paired = sorted(((_row_date(row), row) for row in raw_rows), key=lambda item: item[0])
    indexed = (tuple(item[0] for item in paired), tuple(item[1] for item in paired))
    _ROW_INDEX_CACHE[key] = indexed
    return indexed


def _market_feature(dataset: KALCBFirst30Dataset, trade_date: date) -> MarketFeature:
    key = (id(dataset), trade_date)
    cached = _MARKET_FEATURE_CACHE.get(key)
    if cached is not None:
        return cached
    values = {code: _index_stats(dataset.index_by_code.get(code, []), trade_date) for code in ("KOSPI", "KOSDAQ")}
    kospi = values.get("KOSPI", {})
    kosdaq = values.get("KOSDAQ", {})
    score = (
        0.20 * _signed_clip(kospi.get("ret_1d", 0.0), 0.02)
        + 0.20 * _signed_clip(kosdaq.get("ret_1d", 0.0), 0.02)
        + 0.20 * _signed_clip(kospi.get("ret_5d", 0.0), 0.06)
        + 0.20 * _signed_clip(kosdaq.get("ret_5d", 0.0), 0.06)
        + 0.10 * (1.0 if kospi.get("above_sma20") else -1.0)
        + 0.10 * (1.0 if kosdaq.get("above_sma20") else -1.0)
    )
    feature = MarketFeature(
        kospi_ret_1d=kospi.get("ret_1d", 0.0),
        kospi_ret_5d=kospi.get("ret_5d", 0.0),
        kospi_ret_20d=kospi.get("ret_20d", 0.0),
        kosdaq_ret_1d=kosdaq.get("ret_1d", 0.0),
        kosdaq_ret_5d=kosdaq.get("ret_5d", 0.0),
        kosdaq_ret_20d=kosdaq.get("ret_20d", 0.0),
        kospi_above_sma20=bool(kospi.get("above_sma20")),
        kospi_above_sma60=bool(kospi.get("above_sma60")),
        kosdaq_above_sma20=bool(kosdaq.get("above_sma20")),
        kosdaq_above_sma60=bool(kosdaq.get("above_sma60")),
        score=float(score),
    )
    _MARKET_FEATURE_CACHE[key] = feature
    return feature


def _index_stats(rows: list[dict[str, Any]], trade_date: date) -> dict[str, Any]:
    key = (id(rows), trade_date)
    cached = _INDEX_STATS_CACHE.get(key)
    if cached is not None:
        return cached
    paired = sorted(((_row_date(row), row) for row in rows), key=lambda item: item[0])
    dates = tuple(item[0] for item in paired)
    all_rows = tuple(item[1] for item in paired)
    index = bisect.bisect_left(dates, trade_date)
    prior = list(all_rows[:index])
    if len(prior) < 60:
        _INDEX_STATS_CACHE[key] = {}
        return {}
    close = _float(prior[-1].get("close"))
    sma20 = _avg(_float(row.get("close")) for row in prior[-20:])
    sma60 = _avg(_float(row.get("close")) for row in prior[-60:])
    stats = {
        "ret_1d": _return_pct(prior, 1),
        "ret_5d": _return_pct(prior, 5),
        "ret_20d": _return_pct(prior, 20),
        "above_sma20": close >= sma20,
        "above_sma60": close >= sma60,
    }
    _INDEX_STATS_CACHE[key] = stats
    return stats


def _load_daily_rows(root: Path, symbols: Iterable[str], end: date) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for symbol in symbols:
        frame = load_daily_ohlcv(root, symbol, end=end)
        rows = _rows_from_frame(frame)
        if rows:
            out[str(symbol).zfill(6)] = rows
    return out


def _load_flow_rows(root: Path, symbols: Iterable[str], end: date) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for symbol in symbols:
        frame = load_daily_flow(root, symbol, end=end)
        rows = _rows_from_frame(frame)
        if rows:
            out[str(symbol).zfill(6)] = rows
    return out


def _load_foreign_flow_rows(root: Path, symbols: Iterable[str], end: date) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for symbol in symbols:
        frame = load_daily_foreign_flow(root, symbol, end=end)
        rows = _rows_from_frame(frame)
        if rows:
            out[str(symbol).zfill(6)] = rows
    return out


def _load_institutional_flow_rows(root: Path, symbols: Iterable[str], end: date) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for symbol in symbols:
        frame = load_daily_institutional_flow(root, symbol, end=end)
        rows = _rows_from_frame(frame)
        if rows:
            out[str(symbol).zfill(6)] = rows
    return out


def _load_index_rows(root: Path, end: date) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for code in ("KOSPI", "KOSDAQ"):
        rows = _rows_from_frame(load_index_ohlcv(root, code, end=end))
        if rows:
            out[code] = rows
    return out


def _rows_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows = []
    for row in frame.sort_values("date").to_dict("records"):
        item = dict(row)
        item["date"] = _row_date(item).isoformat()
        if "ticker" in item:
            item["ticker"] = str(item["ticker"]).zfill(6)
        rows.append(item)
    return rows


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
    return {key: tuple(sorted(values, key=lambda item: item.timestamp)) for key, values in grouped.items()}


def _daily_source_fingerprint(
    root: Path,
    daily_by_symbol: dict[str, list[dict[str, Any]]],
    flow_by_symbol: dict[str, list[dict[str, Any]]],
    foreign_flow_by_symbol: dict[str, list[dict[str, Any]]],
    institutional_flow_by_symbol: dict[str, list[dict[str, Any]]],
    index_by_code: dict[str, list[dict[str, Any]]],
    sector_map: dict[str, str],
) -> str:
    manifest = load_manifest(root)
    paths: list[Path] = [Path(root) / "manifest.json", Path(root) / "tables" / "sector_map.parquet"]
    for symbol in daily_by_symbol:
        paths.extend((Path(root) / "daily_ohlcv" / symbol).glob("*.parquet"))
    for symbol in flow_by_symbol:
        paths.extend((Path(root) / "daily_flow" / symbol).glob("*.parquet"))
    for symbol in foreign_flow_by_symbol:
        paths.extend((Path(root) / "daily_foreign_flow" / symbol).glob("*.parquet"))
    for symbol in institutional_flow_by_symbol:
        paths.extend((Path(root) / "daily_institutional_flow" / symbol).glob("*.parquet"))
    for code in index_by_code:
        paths.extend((Path(root) / "index_ohlcv" / code).glob("*.parquet"))
    return stable_signature(
        {
            "root": str(root.resolve()),
            "manifest_source_fingerprint": manifest.get("source_fingerprint"),
            "dataset_version": manifest.get("dataset_version"),
            "manifest_tables": manifest.get("tables"),
            "paths": fingerprint_paths([path for path in paths if path.exists()], root=root),
            "daily_symbols": sorted(daily_by_symbol),
            "flow_symbols": sorted(flow_by_symbol),
            "foreign_flow_symbols": sorted(foreign_flow_by_symbol),
            "institutional_flow_symbols": sorted(institutional_flow_by_symbol),
            "index_codes": sorted(index_by_code),
            "sector_map_hash": stable_signature(sector_map),
        }
    )


def _training_config(config: dict[str, Any], holdout_days: int) -> dict[str, Any]:
    out = dict(config)
    out["holdout_days"] = int(holdout_days)
    out.pop("holdout_start", None)
    out.pop("use_full_available_window", None)
    date_range = dict(out.get("date_range") or {})
    date_range.pop("end", None)
    out["date_range"] = date_range
    return out


def _first30_bars(bars: tuple[MarketBar, ...]) -> tuple[MarketBar, ...]:
    key = id(bars)
    cached = _FIRST30_BARS_CACHE.get(key)
    if cached is not None:
        return cached
    result = completed_first30_bars(_sorted_bars(bars), required_count=FIRST30_BAR_COUNT)
    _FIRST30_BARS_CACHE[key] = result
    return result


def _post_entry_bars(bars: tuple[MarketBar, ...], config: KALCBConfig) -> tuple[MarketBar, ...]:
    key = (id(bars), config.flatten_time)
    cached = _POST_ENTRY_BARS_CACHE.get(key)
    if cached is not None:
        return cached
    rows = []
    for bar in _sorted_bars(bars):
        t = bar.timestamp.astimezone(KST).time()
        if FIRST30_END <= t <= config.flatten_time:
            rows.append(bar)
    result = tuple(rows)
    _POST_ENTRY_BARS_CACHE[key] = result
    return result


def _sorted_bars(bars: tuple[MarketBar, ...]) -> tuple[MarketBar, ...]:
    key = id(bars)
    cached = _SORTED_BARS_CACHE.get(key)
    if cached is not None:
        return cached
    result = tuple(sorted(bars, key=lambda item: item.timestamp))
    _SORTED_BARS_CACHE[key] = result
    return result


def _baseline_risk(entry_price: float, first30_low: float, entry_low: float, daily_atr: float, config: KALCBConfig) -> float:
    structural_stop = min(float(first30_low), float(entry_low))
    atr_stop = float(entry_price) - config.stop_atr_multiple * max(float(daily_atr), 0.0)
    stop = max(structural_stop, atr_stop)
    if stop >= entry_price:
        stop = float(entry_price) * 0.985
    return max(float(entry_price) - stop, float(entry_price) * 0.001)


def _round_trip_cost_pct(config: KALCBConfig) -> float:
    bps = 2.0 * float(config.slippage_bps) + 2.0 * float(config.commission_bps) + float(config.tax_bps_on_sell)
    return max(0.0, bps) / 10_000.0


def _resolve_folds(dates: list[date], *, fold_count: int) -> list[tuple[date, date]]:
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


def _spec(mode: str, top_n: int, values: dict[str, Any]) -> First30Spec:
    return _name_spec(First30Spec(name="", score_mode=mode, top_n=top_n, **values))


def _name_spec(spec: First30Spec) -> First30Spec:
    parts = [
        spec.score_mode,
        f"top{spec.top_n}",
        f"ret{_pct_label(spec.min_first30_ret)}",
        f"vwap{_pct_label(spec.min_vwap_ret)}",
        f"gap{_pct_label(spec.min_gap)}to{_pct_label(spec.max_gap)}",
        f"rv{_num_label(spec.min_rel_volume)}",
        f"cl{_num_label(spec.min_close_location)}",
        f"dd{_pct_label(spec.max_open_drawdown)}",
        f"rng{_num_label(spec.max_range_atr)}",
        f"r5{_pct_label(spec.min_prior_ret5)}",
        f"r20{_pct_label(spec.min_prior_ret20)}to{_pct_label(spec.max_prior_ret20)}",
        f"r60{_pct_label(spec.min_prior_ret60)}",
        f"lowPrev{_pct_label(spec.min_low_vs_prev_close)}",
        f"flow5{_num_label(spec.min_flow_5d)}",
        f"for5{_num_label(spec.min_foreign_flow_5d)}",
        f"inst5{_num_label(spec.min_inst_flow_5d)}",
        f"flowz{_num_label(spec.min_flow_z)}",
        f"agree{_num_label(spec.min_flow_agreement)}",
        f"div{_num_label(spec.max_flow_divergence)}",
        f"secflow{_num_label(spec.min_sector_flow)}",
        f"mkt{_num_label(spec.min_market_score)}",
    ]
    if spec.require_close_above_prev:
        parts.append("abovePrev")
    return replace(spec, name="_".join(parts).replace("-", "m").replace(".", "p"))


def _dedupe_specs(specs: list[First30Spec]) -> list[First30Spec]:
    out: list[First30Spec] = []
    seen: set[str] = set()
    for spec in specs:
        named = _name_spec(spec)
        signature = _spec_signature(named)
        if signature in seen:
            continue
        seen.add(signature)
        out.append(named)
    return out


def _spec_signature(spec: First30Spec) -> str:
    data = asdict(spec)
    data.pop("name", None)
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def _compact_summary(summary: dict[str, Any]) -> dict[str, float]:
    keep = (
        "candidate_days",
        "active_days",
        "active_day_share",
        "avg_candidates_per_session",
        "avg_candidates_per_active_day",
        "active_day_gross_pct",
        "active_day_net_pct",
        "calendar_day_gross_pct",
        "calendar_day_net_pct",
        "slot_cumulative_gross_return_pct",
        "slot_cumulative_net_return_pct",
        "slot_max_drawdown_gross_pct",
        "slot_max_drawdown_net_pct",
        "portfolio_proxy_gross_return_pct",
        "portfolio_proxy_net_return_pct",
        "portfolio_proxy_calendar_day_net_pct",
        "portfolio_proxy_active_day_net_pct",
        "portfolio_proxy_max_drawdown_pct",
        "portfolio_proxy_avg_positions_per_session",
        "portfolio_proxy_avg_positions_per_active_day",
        "avg_mfe_r",
        "median_mfe_r",
        "mfe_ge_0_5_share",
        "mfe_ge_0_75_share",
        "mfe_ge_1_0_share",
        "mae_le_neg_1_share",
        "gross_win_share",
        "net_win_share",
    )
    return {key: _float(summary.get(key)) for key in keep if key in summary}


def _row_payload(row: SweepResult) -> dict[str, Any]:
    return {
        "name": row.spec.name,
        "score": row.score,
        "full_score": row.full_score,
        "median_fold_score": row.median_fold_score,
        "worst_fold_score": row.worst_fold_score,
        "rejected": row.rejected,
        "reject_reason": row.reject_reason,
        "spec": asdict(row.spec),
        "metrics": row.metrics,
        "folds": list(row.folds),
    }


def _record_progress(output_dir: Path, stage: str, completed: int, total: int, rows: list[SweepResult], row: SweepResult) -> None:
    if completed not in {1, 2, 3, 5, 10, total} and completed % 50 != 0:
        return
    _write_progress(output_dir, stage, completed, total, rows)
    event = {"updated_at": _utc_now_iso(), "stage": stage, "completed": int(completed), "total": int(total), "row": _progress_row(row)}
    with (output_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")
    print(
        "[kalcb-first30-sweep] "
        f"{stage} {completed}/{total} {row.spec.name} score={row.score:.3f} "
        f"portfolio={100.0 * row.metrics.get('portfolio_proxy_net_return_pct', 0.0):.1f}% "
        f"slot={100.0 * row.metrics.get('slot_cumulative_gross_return_pct', 0.0):.1f}% "
        f"mfe={row.metrics.get('avg_mfe_r', 0.0):.3f} reject={row.reject_reason}",
        flush=True,
    )


def _write_progress(output_dir: Path, stage: str, completed: int, total: int, rows: list[SweepResult]) -> None:
    ranked = sorted((_progress_row(row) for row in rows), key=lambda item: (-float(item.get("score", 0.0)), bool(item.get("rejected")), str(item.get("name"))))
    payload = {
        "updated_at": _utc_now_iso(),
        "stage": stage,
        "completed": int(completed),
        "total": int(total),
        "percent": round(100.0 * float(completed) / float(total), 3) if total else 100.0,
        "best_so_far": ranked[0] if ranked else None,
        "top_rows": ranked[:20],
    }
    path = output_dir / "progress.json"
    tmp = output_dir / "progress.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def _progress_row(row: SweepResult) -> dict[str, Any]:
    return {
        "name": row.spec.name,
        "score": row.score,
        "rejected": row.rejected,
        "reject_reason": row.reject_reason,
        "spec": asdict(row.spec),
        "metrics": row.metrics,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# KALCB Causal First30 Sweep",
        "",
        f"Sweep hash: `{payload['sweep_hash']}`",
        f"Window: {payload['training_window']['start']} to {payload['training_window']['end']} ({payload['training_window']['sessions']} sessions)",
        "",
        "## Top Combined",
        "",
        _table(payload["top_results"]),
        "",
        "## Top Slot Return",
        "",
        _table(payload["top_slot_return"]),
        "",
        "## Top Portfolio Proxy",
        "",
        _table(payload.get("top_portfolio_proxy", [])),
        "",
        "## Top Avg MFE",
        "",
        _table(payload["top_mfe"]),
    ]
    return "\n".join(lines) + "\n"


def _table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Rank | Config | Score | Active Days | Names/Sess | Portfolio Net | DD | Gross Slot | Net Slot | Avg MFE | +0.75R |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(rows[:25], start=1):
        metrics = row["metrics"]
        lines.append(
            "| "
            f"{index} | {row['name']} | {row['score']:.3f} | {metrics.get('active_days', 0):.0f} | "
            f"{metrics.get('avg_candidates_per_session', 0.0):.3f} | "
            f"{100.0 * metrics.get('portfolio_proxy_net_return_pct', 0.0):.1f}% | "
            f"{100.0 * metrics.get('portfolio_proxy_max_drawdown_pct', 0.0):.1f}% | "
            f"{100.0 * metrics.get('slot_cumulative_gross_return_pct', 0.0):.1f}% | "
            f"{100.0 * metrics.get('slot_cumulative_net_return_pct', 0.0):.1f}% | "
            f"{metrics.get('avg_mfe_r', 0.0):.3f} | "
            f"{metrics.get('mfe_ge_0_75_share', 0.0):.3f} |"
        )
    return "\n".join(lines)


def _row_date(row: dict[str, Any]) -> date:
    raw = row.get("date")
    if isinstance(raw, date):
        return raw
    return date.fromisoformat(str(raw)[:10])


def _return_pct(rows: list[dict[str, Any]], lookback: int) -> float:
    if len(rows) <= lookback:
        return 0.0
    current = _float(rows[-1].get("close"))
    prior = _float(rows[-lookback - 1].get("close"))
    return current / prior - 1.0 if current > 0.0 and prior > 0.0 else 0.0


def _atr(rows: list[dict[str, Any]], period: int) -> float:
    if len(rows) < 2:
        return 0.0
    trs = []
    for index in range(1, len(rows)):
        high = _float(rows[index].get("high"))
        low = _float(rows[index].get("low"))
        prev_close = _float(rows[index - 1].get("close"))
        trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    return _avg(trs[-period:])


def _expected_30m_volume(daily: DailyFeature) -> float:
    return max(daily.adv20_krw / max(daily.prev_close, 1e-9) / 13.0, 1.0)


def _near(value: float, grid: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(sorted(grid, key=lambda item: (abs(float(item) - float(value)), float(item)))[:4])


def _even_sample(items: list[Any], count: int) -> list[Any]:
    if count <= 0:
        return []
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[0]]
    last = len(items) - 1
    return [items[round(index * last / (count - 1))] for index in range(count)]


def _compound(values: Iterable[float]) -> float:
    equity = 1.0
    for value in values:
        equity *= 1.0 + float(value)
    return equity - 1.0


def _max_drawdown(values: Iterable[float]) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for value in values:
        equity *= 1.0 + float(value)
        peak = max(peak, equity)
        if peak > 0:
            max_dd = min(max_dd, equity / peak - 1.0)
    return max_dd


def _return_score(value: float, *, target: float) -> float:
    span = max(float(target), 1e-9)
    return _clip((float(value) + span) / (2.0 * span))


def _signed_clip(value: float, span: float) -> float:
    return max(-1.0, min(1.0, float(value) / max(float(span), 1e-9)))


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _avg(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return float(mean(rows)) if rows else 0.0


def _med(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    return float(median(rows)) if rows else 0.0


def _std(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    if len(rows) < 2:
        return 0.0
    avg = _avg(rows)
    return (sum((value - avg) ** 2 for value in rows) / (len(rows) - 1)) ** 0.5


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _flow_value(row: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in row and row.get(key) is not None:
            return _float(row.get(key))
    return 0.0


def _pct_label(value: float) -> str:
    if abs(float(value)) >= 9:
        return "999"
    return f"{100.0 * float(value):.2f}".rstrip("0").rstrip(".")


def _num_label(value: float) -> str:
    if abs(float(value)) >= 9:
        return "999"
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep causal KALCB first30 selectors.")
    parser.add_argument("--config", default="config/optimization/kalcb.yaml")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--holdout-days", type=int, default=DEFAULT_HOLDOUT_DAYS)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--refine-top-n", type=int, default=6)
    parser.add_argument("--max-coarse-specs", type=int, default=None)
    args = parser.parse_args(argv)
    config = normalize_runtime_config("kalcb", load_yaml_config(args.config))
    payload = run_first30_signal_sweep(
        config,
        output_dir=args.output_dir,
        holdout_days=args.holdout_days,
        max_workers=args.max_workers,
        refine_top_n=args.refine_top_n,
        max_coarse_specs=args.max_coarse_specs,
    )
    print(
        json.dumps(
            {
                "sweep_hash": payload["sweep_hash"],
                "artifact_paths": payload["artifact_paths"],
                "top_results": payload["top_results"][:10],
                "top_portfolio_proxy": payload["top_portfolio_proxy"][:10],
                "top_slot_return": payload["top_slot_return"][:10],
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
