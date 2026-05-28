from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from backtests.auto.shared.cache_keys import build_cache_key, fingerprint_paths, stable_signature
from backtests.core.replay_bundle import EventReplayBundle
from backtests.core.replay_events import ReplayEvent
from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_common.sector_daily import SECTOR_DAILY_VERSION
from strategy_common.sector_intraday import AFTERNOON_CUTOFF, FIRST30_CUTOFF, SECTOR_INTRADAY_VERSION, SectorIntradayPanel, build_sector_intraday_panel
from strategy_kalcb.artifact_store import KALCBArtifactStore
from strategy_kalcb.config import KALCBConfig, KALCB_CORE_VERSION
from strategy_kalcb.models import KALCBDailyCandidate, KALCBDailySnapshot
from strategy_kalcb.research import (
    KALCB_CANONICAL_SOURCE,
    RESEARCH_MODEL_VERSION,
    build_research_snapshot,
    candidate_config_fingerprint,
    daily_selection_from_snapshot,
    finalize_candidate_snapshot,
    _build_frontier_order as _strategy_build_frontier_order,
    _frontier_rest_budget_symbols_per_5m as _strategy_frontier_rest_budget_symbols_per_5m,
    _rank_candidates as _strategy_rank_candidates,
    _select_active_seed as _strategy_select_active_seed,
)

from .features import build_feature_bundle_hash_for_snapshots


_BUNDLE_CACHE: dict[str, EventReplayBundle] = {}
_FRAME_CACHE: dict[str, pd.DataFrame] = {}
_DAILY_ROWS_CACHE: dict[str, list[dict[str, Any]]] = {}
_WINDOW_CACHE: dict[str, "KALCBReplayWindow"] = {}


@dataclass(frozen=True, slots=True)
class KALCBReplayWindow:
    earliest: date
    latest: date
    train_start: date
    train_end: date
    holdout_start: date


def load_kalcb_real_replay_bundle(config: dict[str, Any] | None = None, mutations: dict[str, Any] | None = None) -> EventReplayBundle:
    raw_config = dict(config or {})
    raw_mutations = dict(mutations or {})
    cfg = KALCBConfig.from_mapping(raw_config, raw_mutations)
    data_root = Path(raw_config.get("data_root", "data/kis_intraday_parquet"))
    timeframe = str(raw_config.get("timeframe", cfg.timeframe) or "5m")
    if timeframe != "5m":
        raise ValueError("KALCB real replay requires 5m parquet input")
    symbols = _resolve_symbols(raw_config, data_root, timeframe)
    if not symbols:
        raise FileNotFoundError(f"No KALCB 5m parquet symbols found under {data_root}")
    window = _resolve_replay_window(raw_config, data_root, timeframe, symbols)
    data_fingerprint = _real_source_fingerprint(data_root, symbols, timeframe, window.train_start, window.train_end)
    sector_map = _resolve_sector_map(raw_config)
    candidate_config_hash = _candidate_config_hash(cfg, raw_mutations, sector_map)
    bundle_key = build_cache_key(
        "kalcb.real_event_replay_bundle",
        source_fingerprint=data_fingerprint,
        mutations=raw_mutations,
        mutation_exact_keys=(
            "kalcb.session.opening_range_bars",
            "kalcb.session.ws_budget",
            "kalcb.live.ws_budget",
            "kalcb.frontier.enabled",
            "kalcb.frontier.size",
            "kalcb.frontier.selection_mode",
            "kalcb.frontier.active_selection_mode",
            "kalcb.frontier.rotation_min_frontier_trades",
            "kalcb.frontier.rotation_min_frontier_avg_r",
            "kalcb.frontier.rotation_min_frontier_total_r",
            "kalcb.frontier.rotation_min_proof_symbols",
            "kalcb.discovery.frontier_size",
            "kalcb.discovery.selection_mode",
            "kalcb.discovery.active_selection_mode",
            "kalcb.research.top_long_count",
            "kalcb.research.min_price_krw",
            "kalcb.research.min_adv20_krw",
            "kalcb.research.min_history_days",
            "kalcb.research.weights.relative_strength",
            "kalcb.research.weights.daily_trend",
            "kalcb.research.weights.compression",
            "kalcb.research.weights.accumulation",
            "kalcb.research.weights.stock_regime",
            "kalcb.research.weights.sector_regime",
            "kalcb.research.weights.sector_participation",
            "kalcb.research.min_rs_percentile",
            "kalcb.research.min_trend_score",
            "kalcb.research.min_compression_score",
            "kalcb.research.min_accumulation_score",
            "kalcb.research.min_sector_participation",
            "kalcb.research.min_sector_daily_score_pct",
            "kalcb.research.max_box_range_pct",
        ),
        extra={
            "data_root": str(data_root.resolve()),
            "timeframe": timeframe,
            "symbols": symbols,
            "train_start": window.train_start.isoformat(),
            "train_end": window.train_end.isoformat(),
            "candidate_config_hash": candidate_config_hash,
            "sector_map_hash": stable_signature(sector_map),
            "artifact_root": str(raw_config.get("artifact_root", "data/strategy/kalcb")),
            "core_version": KALCB_CORE_VERSION,
            "sector_daily_version": SECTOR_DAILY_VERSION,
            "sector_intraday_version": SECTOR_INTRADAY_VERSION,
        },
    )
    cached = _BUNDLE_CACHE.get(bundle_key)
    if cached is not None:
        return cached

    frames = {symbol: _load_symbol_frame(data_root, symbol, timeframe, window.train_end) for symbol in symbols}
    data_available_symbols = [symbol for symbol, frame in frames.items() if not frame.empty]
    unavailable_symbols = [symbol for symbol, frame in frames.items() if frame.empty]
    daily_by_symbol = {symbol: _daily_rows_from_frame(symbol, frame) for symbol, frame in frames.items() if not frame.empty}
    trading_dates = _trading_dates(frames, window.train_start, window.train_end)
    if not trading_dates:
        raise ValueError("KALCB real replay found no 5m bars in the selected train window")
    bars_by_key = _bars_by_key_from_frames(frames, window.train_start, window.train_end, source_fingerprint=data_fingerprint)
    sector_first30_panel = build_sector_intraday_panel(
        bars_by_key,
        sector_map,
        trade_dates=trading_dates,
        cutoff=FIRST30_CUTOFF,
        symbols=symbols,
    )
    sector_prev_panel = build_sector_intraday_panel(
        bars_by_key,
        sector_map,
        trade_dates=trading_dates,
        cutoff=AFTERNOON_CUTOFF,
        symbols=symbols,
    )

    artifact_root = Path(raw_config.get("artifact_root", "data/strategy/kalcb"))
    store_root = artifact_root / "candidate_snapshots" / candidate_config_hash[:16]
    store = KALCBArtifactStore(store_root)
    snapshots: dict[date, KALCBDailySnapshot] = {}
    for trade_date in trading_dates:
        snapshot = _load_or_build_snapshot(
            trade_date,
            daily_by_symbol,
            cfg,
            source_fingerprint=data_fingerprint,
            candidate_config_hash=candidate_config_hash,
            requested_universe_count=len(symbols),
            data_available_symbols=data_available_symbols,
            unavailable_symbols=unavailable_symbols,
            sector_map=sector_map,
            store=store,
        )
        snapshot = _with_sector_intraday_metadata(
            snapshot,
            sector_first30_panel=sector_first30_panel,
            sector_prev_panel=sector_prev_panel,
            trading_dates=trading_dates,
            sector_map=sector_map,
        )
        store.save_snapshot(snapshot)
        if snapshot.candidates:
            snapshots[trade_date] = snapshot
    if not snapshots:
        raise ValueError("KALCB real replay produced no source-fingerprinted candidate snapshots")

    events = _events_from_frames(frames, snapshots, window.train_start, window.train_end, source_fingerprint=data_fingerprint)
    if not events:
        raise ValueError("KALCB real replay produced no replay events after ws_budget candidate filtering")

    candidate_hashes = {day.isoformat(): snapshot.artifact_hash for day, snapshot in snapshots.items()}
    feature_hash = build_feature_bundle_hash_for_snapshots(snapshots, data_fingerprint)
    metadata = {
        "replay_mode": "real_kis_krx_parquet",
        "capability_level": "real_replay",
        "data_root": str(data_root),
        "timeframe": timeframe,
        "timestamp_basis": str(raw_config.get("timestamp_basis", "kis_recorded_completed")),
        "source_fingerprint": data_fingerprint,
        "candidate_config_hash": candidate_config_hash,
        "candidate_artifact_root": str(store_root),
        "sector_map_size": len(sector_map),
        "sector_map_hash": stable_signature(sector_map),
        "kalcb_candidate_snapshots": {day.isoformat(): snapshot.to_json_dict() for day, snapshot in snapshots.items()},
        "kalcb_candidate_artifact_hashes": candidate_hashes,
        "kalcb_candidate_artifact_hash": stable_signature(candidate_hashes),
        "kalcb_feature_bundle_hash": feature_hash,
        "available_features": _real_replay_available_features(),
        "sector_daily_version": SECTOR_DAILY_VERSION,
        "sector_intraday_version": sector_first30_panel.version,
        "optional_features": _optional_feature_status(),
        "fallback_features": {"foreign_institutional_flow": "unavailable_in_v1", "program_flow": "unavailable_in_v1"},
        "universe": symbols,
        "universe_size": len(symbols),
        "data_available_symbols": data_available_symbols,
        "data_available_symbol_count": len(data_available_symbols),
        "unavailable_symbols": unavailable_symbols,
        "unavailable_symbol_count": len(unavailable_symbols),
        "candidate_pool_max": max(int(snapshot.metadata.get("candidate_pool_count") or len(snapshot.candidates)) for snapshot in snapshots.values()),
        "active_symbol_max": max(int(snapshot.metadata.get("active_symbol_count") or len(snapshot.candidates)) for snapshot in snapshots.values()),
        "frontier_symbol_max": max(len(snapshot.candidates) for snapshot in snapshots.values()),
        "ws_budget": cfg.ws_budget,
        "frontier_enabled": cfg.frontier_enabled,
        "frontier_size": cfg.frontier_size,
        "frontier_selection_mode": cfg.frontier_selection_mode,
        "frontier_active_selection_mode": cfg.frontier_active_selection_mode,
        "frontier_rest_budget_symbols_per_5m": _frontier_rest_budget_symbols_per_5m(cfg),
        "train_start": window.train_start.isoformat(),
        "train_end": window.train_end.isoformat(),
        "holdout_start": window.holdout_start.isoformat(),
        "holdout_policy": "Dates >= holdout_start are excluded unless date_range.end/start overrides it.",
        "live_parity_fill_timing": cfg.live_parity_fill_timing,
        "auction_mode": cfg.auction_mode,
        "sessions": len(snapshots),
    }
    bundle = EventReplayBundle(
        events=tuple(events),
        source_fingerprint=data_fingerprint,
        data_root=data_root,
        metadata=metadata,
    )
    _BUNDLE_CACHE[bundle_key] = bundle
    return bundle


def warm_kalcb_real_replay_cache(config: dict[str, Any] | None = None, mutations: dict[str, Any] | None = None) -> None:
    load_kalcb_real_replay_bundle(config, mutations)


def real_replay_metadata(config: dict[str, Any] | None = None, mutations: dict[str, Any] | None = None) -> dict[str, Any]:
    return dict(load_kalcb_real_replay_bundle(config, mutations).metadata)


def _resolve_symbols(config: dict[str, Any], data_root: Path, timeframe: str) -> list[str]:
    raw = config.get("universe") or config.get("symbols")
    requested: list[str] = []
    explicit_universe = raw is not None
    if raw:
        if isinstance(raw, str):
            maybe_path = Path(raw)
            if not maybe_path.exists() and not maybe_path.is_absolute():
                maybe_path = Path("config") / raw
            if maybe_path.exists() and maybe_path.suffix.lower() in {".yaml", ".yml"}:
                payload = yaml.safe_load(maybe_path.read_text(encoding="utf-8")) or {}
                values = (payload.get("symbols") or payload.get("universe")) if isinstance(payload, dict) else payload
                requested = [str(item).zfill(6) for item in (values or [])]
            elif raw.endswith((".yaml", ".yml")):
                raise FileNotFoundError(f"KALCB explicit universe file was not found: {raw}")
            else:
                requested = [item.strip().zfill(6) for item in raw.split(",") if item.strip()]
        else:
            requested = [str(symbol).zfill(6) for symbol in raw]
    if not requested:
        if explicit_universe:
            raise ValueError(f"KALCB explicit universe resolved to no symbols: {raw!r}")
        requested = sorted(path.parent.name for path in Path(data_root).glob(f"*/*_{timeframe}_*.parquet"))
    symbols = []
    missing: list[str] = []
    for symbol in requested:
        if list(Path(data_root).glob(f"{symbol}/{symbol}_{timeframe}_*.parquet")):
            symbols.append(symbol)
        else:
            missing.append(symbol)
    if explicit_universe and missing:
        raise FileNotFoundError(
            f"KALCB explicit universe has {len(missing)} symbols without {timeframe} parquet data: {', '.join(missing[:10])}"
        )
    return sorted(dict.fromkeys(symbols))


def _resolve_replay_window(config: dict[str, Any], data_root: Path, timeframe: str, symbols: list[str]) -> KALCBReplayWindow:
    cache_key = stable_signature(
        {
            "data_root": str(data_root),
            "timeframe": timeframe,
            "symbols": symbols,
            "holdout_days": int(config.get("holdout_days", 21)),
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
        holdout_start = latest - timedelta(days=int(config.get("holdout_days", 21)))
        cached = KALCBReplayWindow(earliest, latest, earliest, holdout_start - timedelta(days=1), holdout_start)
        _WINDOW_CACHE[cache_key] = cached

    date_range = dict(config.get("date_range") or {})
    train_start = _parse_config_date(date_range.get("start") or config.get("start")) or cached.train_start
    train_end = _parse_config_date(date_range.get("end") or config.get("end"))
    if train_end is None:
        if config.get("holdout_start"):
            train_end = _parse_config_date(config["holdout_start"]) - timedelta(days=1)
        elif bool(config.get("use_full_available_window", False)):
            train_end = cached.latest
        else:
            train_end = cached.train_end
    train_end = min(train_end, cached.latest)
    if train_end < train_start:
        raise ValueError(f"KALCB replay train window is empty: {train_start} > {train_end}")
    return KALCBReplayWindow(cached.earliest, cached.latest, train_start, train_end, cached.holdout_start)


def _parse_config_date(value: Any) -> date | None:
    if not value:
        return None
    text = str(value)
    if "T" in text:
        return datetime.fromisoformat(text).date()
    return date.fromisoformat(text)


def _real_source_fingerprint(data_root: Path, symbols: list[str], timeframe: str, train_start: date, train_end: date) -> str:
    paths = []
    for symbol in symbols:
        paths.extend(Path(data_root).glob(f"{symbol}/{symbol}_{timeframe}_*.parquet"))
    return stable_signature(
        {
            "mode": "kalcb_real_kis_krx_parquet",
            "paths": fingerprint_paths(paths, root=data_root),
            "timeframe": timeframe,
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "symbols": symbols,
        }
    )


def _candidate_config_hash(config: KALCBConfig, mutations: dict[str, Any], sector_map: dict[str, str] | None = None) -> str:
    return candidate_config_fingerprint(config, mutations, sector_map)


def _resolve_sector_map(config: dict[str, Any]) -> dict[str, str]:
    raw = config.get("sector_map")
    if isinstance(raw, str):
        raw = _load_sector_map_from_path(_resolve_config_path(raw))
    elif raw is None:
        path = _resolve_config_path(str(config.get("sector_map_path") or "config/olr/sector_map.yaml"))
        raw = _load_sector_map_from_path(path) if path.exists() else {}
    return {str(symbol).zfill(6): str(sector).strip().upper() for symbol, sector in dict(raw or {}).items() if str(sector).strip()}


def _resolve_config_path(raw: str) -> Path:
    path = Path(raw)
    if not path.exists() and not path.is_absolute():
        config_path = Path("config") / path
        if config_path.exists():
            path = config_path
    return path


def _load_sector_map_from_path(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return {}
    raw = payload.get("sector_map", payload)
    return dict(raw or {}) if isinstance(raw, dict) else {}


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


def _daily_rows_from_frame(symbol: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    key = stable_signature({"symbol": symbol, "rows": len(frame), "first": _frame_boundary(frame, "min"), "last": _frame_boundary(frame, "max")})
    cached = _DAILY_ROWS_CACHE.get(key)
    if cached is not None:
        return cached
    if frame.empty:
        rows: list[dict[str, Any]] = []
    else:
        grouped = frame.groupby(frame["timestamp"].dt.date, sort=True).agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        rows = [
            {
                "date": pd.Timestamp(day).date().isoformat() if not isinstance(day, date) else day.isoformat(),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(row.volume),
            }
            for day, row in grouped.iterrows()
        ]
    _DAILY_ROWS_CACHE[key] = rows
    return rows


def _frame_boundary(frame: pd.DataFrame, op: str) -> str:
    if frame.empty:
        return ""
    value = frame["timestamp"].min() if op == "min" else frame["timestamp"].max()
    return value.isoformat()


def _trading_dates(frames: dict[str, pd.DataFrame], start: date, end: date) -> list[date]:
    dates: set[date] = set()
    for frame in frames.values():
        if frame.empty:
            continue
        session_dates = frame["timestamp"].dt.date
        dates.update(day for day in session_dates if start <= day <= end)
    return sorted(dates)


def _load_or_build_snapshot(
    trade_date: date,
    daily_by_symbol: dict[str, list[dict[str, Any]]],
    config: KALCBConfig,
    *,
    source_fingerprint: str,
    candidate_config_hash: str,
    requested_universe_count: int,
    data_available_symbols: list[str],
    unavailable_symbols: list[str],
    sector_map: dict[str, str],
    store: KALCBArtifactStore,
) -> KALCBDailySnapshot:
    cached = store.load_snapshot(trade_date)
    if (
        cached is not None
        and cached.source_fingerprint == source_fingerprint
        and cached.metadata.get("candidate_config_hash") == candidate_config_hash
    ):
        return cached

    research_snapshot = build_research_snapshot(
        daily_by_symbol,
        trade_date,
        config,
        sector_map=sector_map,
        generated_at=datetime.combine(trade_date, datetime.min.time(), tzinfo=KST),
        source_fingerprint=source_fingerprint,
        metadata={
            "candidate_config_hash": candidate_config_hash,
            "source": KALCB_CANONICAL_SOURCE,
            "sector_map_hash": stable_signature(sector_map),
        },
    )
    base_snapshot = daily_selection_from_snapshot(research_snapshot, config)
    snapshot = finalize_candidate_snapshot(
        base_snapshot,
        config=config,
        candidate_config_hash=candidate_config_hash,
        source=KALCB_CANONICAL_SOURCE,
        sector_map_hash=stable_signature(sector_map),
        requested_universe_count=requested_universe_count,
        data_available_symbols=data_available_symbols,
        unavailable_symbols=unavailable_symbols,
        source_universe_count=len(daily_by_symbol),
        sector_map_size=len(sector_map),
        generated_at=datetime.combine(trade_date, datetime.min.time(), tzinfo=KST),
    )
    store.save_snapshot(snapshot)
    return snapshot


def _events_from_frames(
    frames: dict[str, pd.DataFrame],
    snapshots: dict[date, KALCBDailySnapshot],
    start: date,
    end: date,
    *,
    source_fingerprint: str,
) -> list[ReplayEvent]:
    active_by_date = {day: set(snapshot.by_symbol()) for day, snapshot in snapshots.items()}
    events: list[ReplayEvent] = []
    for symbol, frame in frames.items():
        if frame.empty:
            continue
        sliced = frame[(frame["timestamp"].dt.date >= start) & (frame["timestamp"].dt.date <= end)]
        for row in sliced.itertuples(index=False):
            ts = row.timestamp.to_pydatetime() if hasattr(row.timestamp, "to_pydatetime") else row.timestamp
            day = ts.date()
            if symbol not in active_by_date.get(day, set()):
                continue
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
            events.append(ReplayEvent.from_bar(bar))
    return sorted(events, key=lambda event: (event.timestamp, event.symbol))


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


def _with_sector_intraday_metadata(
    snapshot: KALCBDailySnapshot,
    *,
    sector_first30_panel: SectorIntradayPanel,
    sector_prev_panel: SectorIntradayPanel,
    trading_dates: list[date],
    sector_map: dict[str, str],
) -> KALCBDailySnapshot:
    prior_day = _previous_trading_date(trading_dates, snapshot.trade_date)
    candidates: list[KALCBDailyCandidate] = []
    for candidate in snapshot.candidates:
        sector = sector_map.get(candidate.symbol, candidate.sector)
        first30_feature = sector_first30_panel.feature_for(snapshot.trade_date, candidate.symbol, sector=sector)
        metadata = {
            **dict(candidate.metadata),
            **first30_feature.metadata(),
            "sector_intraday_cutoff": sector_first30_panel.cutoff_label,
        }
        if prior_day is not None:
            prev_feature = sector_prev_panel.feature_for(prior_day, candidate.symbol, sector=sector)
            metadata.update(
                {
                    **prev_feature.metadata(prefix="prev_session"),
                    "prev_session_sector_intraday_cutoff": sector_prev_panel.cutoff_label,
                    "prev_session_sector_intraday_date": prior_day.isoformat(),
                }
            )
        candidates.append(replace(candidate, metadata=metadata))
    return replace(
        snapshot,
        candidates=tuple(candidates),
        metadata={
            **dict(snapshot.metadata),
            "sector_intraday_cutoff": sector_first30_panel.cutoff_label,
            "sector_intraday_version": sector_first30_panel.version,
            "prev_session_sector_intraday_cutoff": sector_prev_panel.cutoff_label,
        },
    )


def _previous_trading_date(trading_dates: list[date], trade_date: date) -> date | None:
    prior = [day for day in trading_dates if day < trade_date]
    return prior[-1] if prior else None


def _build_frontier_order(
    candidates: list[KALCBDailyCandidate],
    active_seed: list[KALCBDailyCandidate],
    config: KALCBConfig,
    limit: int,
) -> list[KALCBDailyCandidate]:
    return _strategy_build_frontier_order(candidates, active_seed, config, limit)


def _select_active_seed(candidates: list[KALCBDailyCandidate], config: KALCBConfig) -> list[KALCBDailyCandidate]:
    return _strategy_select_active_seed(candidates, config)


def _rank_candidates(candidates: list[KALCBDailyCandidate], component: str) -> list[KALCBDailyCandidate]:
    return _strategy_rank_candidates(candidates, component)


def _frontier_rest_budget_symbols_per_5m(config: KALCBConfig) -> int:
    return _strategy_frontier_rest_budget_symbols_per_5m(config)


def _real_replay_available_features() -> tuple[str, ...]:
    return (
        "completed_5m_signal_bars",
        "prior_completed_daily_ohlcv",
        "intraday_rvol_curve",
        "session_vwap",
        "opening_range",
        "candidate_artifact",
        "market_regime_inputs",
        "sector_map",
        "sector_participation",
        "sector_intraday_panel",
        "krx_hot_candidate_frontier",
        "frontier_shadow_replay",
        "krx_tick_table",
    )


def _optional_feature_status() -> dict[str, bool]:
    return {
        "foreign_institutional_flow": False,
        "program_flow": False,
        "sector_map": True,
        "sector_participation": True,
        "sector_intraday_panel": True,
        "benchmark_breadth": False,
        "bid_ask_spread_snapshots": False,
        "market_regime_inputs": True,
    }
