from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from deployment.olr_kalcb.artifacts import generate_kalcb_daily, generate_olr_afternoon, generate_olr_daily
from deployment.olr_kalcb.hashing import file_sha256
from deployment.olr_kalcb.offline_replay import load_market_bars_for_replay
from strategy_common.clock import KST
from strategy_common.daily_lrs_parquet import (
    available_daily_symbols,
    load_daily_flow,
    load_daily_foreign_flow,
    load_daily_institutional_flow,
    load_daily_ohlcv,
    load_index_ohlcv,
    load_manifest,
)
from strategy_common.market import MarketBar
from strategy_common.sector_map import load_canonical_sector_map
from strategy_kalcb.config import KALCBConfig
from strategy_olr.artifact_store import OLR_STAGE1_ARTIFACT_STAGE
from strategy_olr.config import OLRConfig
from strategy_olr.research import load_candidate_snapshot

DEFAULT_BASELINE_MANIFEST = Path(os.environ.get("OLR_KALCB_BASELINE_MANIFEST", "data/live_readiness/olr_kalcb/2026-05-28/baseline_manifest.json"))
DEFAULT_DAILY_UNIVERSE_FILE = Path(os.environ.get("OLR_KALCB_DAILY_UNIVERSE_FILE", "config/olr_kalcb/olr_deployment_universe_103.yaml"))
DEFAULT_DAILY_ROOT = Path(os.environ.get("OLR_KALCB_DAILY_ROOT", "data/krx_daily_parquet"))
DEFAULT_INTRADAY_ROOT = Path(os.environ.get("OLR_KALCB_INTRADAY_ROOT", "data/kis_intraday_parquet"))
DEFAULT_SECTOR_MAP = Path(os.environ.get("OLR_KALCB_SECTOR_MAP", "config/olr/sector_map.yaml"))
DEFAULT_KALCB_ARTIFACT_ROOT = Path("data/strategy/kalcb")
DEFAULT_OLR_ARTIFACT_ROOT = Path("data/strategy/olr")
DEFAULT_DAILY_LOOKBACK_DAYS = 420
DEFAULT_CUTOFF = time(14, 30)
DEFAULT_MARKET_OPEN = time(9, 0)


@dataclass(frozen=True, slots=True)
class DailyUniverseReport:
    expected_symbols: tuple[str, ...]
    loaded_symbols: tuple[str, ...]
    missing_symbols: tuple[str, ...]
    unexpected_symbols: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DailyFreshnessReport:
    latest: date
    earliest_symbol_latest: date
    per_symbol_latest: dict[str, date]


@dataclass(frozen=True, slots=True)
class AfternoonCoverageReport:
    required_symbols: tuple[str, ...]
    missing_symbols: tuple[str, ...]
    underfilled_symbols: dict[str, int]
    min_bars_per_symbol: int


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "daily":
            payload = run_daily(args)
        elif args.command == "afternoon":
            payload = run_afternoon(args)
        else:
            parser.error(f"unsupported command {args.command!r}")
            return 2
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        _emit({"passed": False, "error_type": type(exc).__name__, "error": str(exc)}, getattr(args, "output_json", None))
        return 2
    _emit(payload, args.output_json)
    return 0 if payload.get("passed") else 1


def run_daily(args: argparse.Namespace) -> dict[str, Any]:
    trade_date = date.fromisoformat(args.trade_date)
    daily_end = date.fromisoformat(args.daily_end) if args.daily_end else trade_date - timedelta(days=1)
    daily_start = daily_end - timedelta(days=int(args.daily_lookback_days))
    fixture_generated_at = _parse_fixture_generated_at(getattr(args, "fixture_generated_at", None))
    configs = _load_strategy_configs(args.baseline_manifest)
    sector_map = _load_sector_map(args.sector_map)
    strategies = {str(item).upper().strip() for item in args.strategies}
    bundle = _load_daily_bundle(
        daily_root=args.daily_root,
        daily_json=args.daily_json,
        daily_universe_file=getattr(args, "daily_universe_file", None),
        trade_date=trade_date,
        start=daily_start,
        end=daily_end,
        symbols=args.symbols,
    )
    if not bundle["daily_by_symbol"]:
        raise ValueError("no daily OHLCV rows were loaded for artifact generation")
    universe = _validate_daily_universe(
        bundle,
        configs,
        strategies,
        allow_partial=bool(getattr(args, "allow_partial_universe", False)),
    )
    freshness = _validate_daily_freshness(
        bundle["daily_by_symbol"],
        trade_date,
        daily_end=daily_end,
        explicit_daily_end=bool(args.daily_end),
        max_lag_days=int(args.max_daily_lag_days),
        expected_symbols=universe.loaded_symbols,
    )

    results = []
    if "KALCB" in strategies:
        _snapshot, result = generate_kalcb_daily(
            bundle["daily_by_symbol"],
            trade_date,
            config=configs["KALCB"]["config"],
            sector_map=sector_map,
            daily_flow_by_symbol=bundle["daily_flow_by_symbol"],
            daily_foreign_flow_by_symbol=bundle["daily_foreign_flow_by_symbol"],
            daily_institutional_flow_by_symbol=bundle["daily_institutional_flow_by_symbol"],
            artifact_root=_resolve_path(args.kalcb_artifact_root),
            source_fingerprint=args.source_fingerprint or None,
            config_mutations=configs["KALCB"]["mutations"],
            generated_at=fixture_generated_at,
        )
        results.append(_result_payload(result))
    if "OLR" in strategies:
        _snapshot, result = generate_olr_daily(
            bundle["daily_by_symbol"],
            trade_date,
            config=configs["OLR"]["config"],
            sector_map=sector_map,
            flow_by_symbol=bundle["daily_flow_by_symbol"],
            foreign_flow_by_symbol=bundle["daily_foreign_flow_by_symbol"],
            institutional_flow_by_symbol=bundle["daily_institutional_flow_by_symbol"],
            index_ohlcv_by_symbol=bundle["index_ohlcv_by_symbol"],
            artifact_root=_resolve_path(args.olr_artifact_root),
            source_fingerprint=args.source_fingerprint or None,
            generated_at=fixture_generated_at,
        )
        results.append(_result_payload(result))

    return {
        "command": "daily",
        "passed": True,
        "trade_date": trade_date.isoformat(),
        "daily_start": daily_start.isoformat(),
        "daily_end": daily_end.isoformat(),
        "latest_daily_row": freshness.latest.isoformat(),
        "earliest_symbol_latest_row": freshness.earliest_symbol_latest.isoformat(),
        "daily_source": bundle["source"],
        "daily_universe_file": bundle.get("daily_universe_file", ""),
        "expected_symbols_sha256": bundle["expected_symbols_sha256"],
        "symbols_loaded": len(bundle["daily_by_symbol"]),
        "expected_symbols": len(universe.expected_symbols),
        "missing_symbols": list(universe.missing_symbols),
        "unexpected_symbols": list(universe.unexpected_symbols),
        "sector_map_count": len(sector_map),
        "baseline_manifest": str(_resolve_path(args.baseline_manifest)),
        "fixture_generated_at": fixture_generated_at.isoformat() if fixture_generated_at else "",
        "results": results,
        "next_gate": f"python scripts/run_olr_kalcb_runtime_session.py preflight --trade-date {trade_date.isoformat()} --mode artifact_only_stage1",
    }


def run_afternoon(args: argparse.Namespace) -> dict[str, Any]:
    trade_date = date.fromisoformat(args.trade_date)
    configs = _load_strategy_configs(args.baseline_manifest)
    sector_map = _load_sector_map(args.sector_map)
    olr_root = _resolve_path(args.olr_artifact_root)
    stage1 = load_candidate_snapshot(
        trade_date,
        artifact_root=olr_root,
        artifact_stage=OLR_STAGE1_ARTIFACT_STAGE,
    )
    if stage1 is None:
        raise FileNotFoundError(f"OLR stage1 artifact missing for {trade_date.isoformat()}")
    stage1_symbols = tuple(str(candidate.symbol).zfill(6) for candidate in stage1.candidates)
    target_symbols = _normalize_symbol_list(args.symbols) if args.symbols else stage1_symbols
    bars = _load_intraday_bars(args, trade_date, target_symbols)
    if not bars:
        raise ValueError("no completed intraday 5m bars were loaded for OLR afternoon artifact generation")
    coverage = _validate_afternoon_coverage(
        bars,
        stage1_symbols,
        min_bars_per_symbol=_afternoon_min_bars(args, configs["OLR"]["config"]),
        allow_partial=bool(getattr(args, "allow_partial_afternoon_bars", False)),
    )
    _snapshot, result = generate_olr_afternoon(
        trade_date,
        bars,
        candidate_snapshot=stage1,
        artifact_root=olr_root,
        config=configs["OLR"]["config"],
        sector_map=sector_map,
    )
    return {
        "command": "afternoon",
        "passed": True,
        "trade_date": trade_date.isoformat(),
        "cutoff_kst": args.cutoff,
        "symbols_loaded": len(bars),
        "bar_count": sum(len(item) for item in bars.values()),
        "stage1_symbols": len(stage1_symbols),
        "min_bars_per_stage1_symbol": coverage.min_bars_per_symbol,
        "missing_stage1_bar_symbols": list(coverage.missing_symbols),
        "underfilled_stage1_bar_symbols": coverage.underfilled_symbols,
        "sector_map_count": len(sector_map),
        "baseline_manifest": str(_resolve_path(args.baseline_manifest)),
        "result": _result_payload(result),
        "next_gate": f"python scripts/run_olr_kalcb_runtime_session.py preflight --trade-date {trade_date.isoformat()} --mode artifact_only",
    }


def _load_strategy_configs(baseline_manifest: str | Path) -> dict[str, dict[str, Any]]:
    manifest_path = _resolve_path(baseline_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    configs: dict[str, dict[str, Any]] = {}
    for strategy_id in ("KALCB", "OLR"):
        record = _config_record(manifest, strategy_id)
        if record is None:
            raise ValueError(f"baseline manifest is missing {strategy_id} optimized config artifact")
        path = _resolve_path(record["path"])
        raw = json.loads(path.read_text(encoding="utf-8"))
        mutations = _config_mutations(raw)
        config = KALCBConfig.from_mapping(mutations) if strategy_id == "KALCB" else OLRConfig.from_mapping(mutations)
        configs[strategy_id] = {"config": config, "mutations": mutations, "path": str(path)}
    return configs


def _config_record(manifest: Mapping[str, Any], strategy_id: str) -> dict[str, Any] | None:
    sid = strategy_id.lower()
    for row in manifest.get("artifacts") or ():
        item = dict(row or {})
        label = str(item.get("label") or "").lower()
        path = str(item.get("path") or "").lower()
        if sid in label and "optimized_config" in label:
            return item
        if sid in path and path.endswith("optimized_config.json"):
            return item
    return None


def _config_mutations(raw: Mapping[str, Any]) -> dict[str, Any]:
    candidate: Any = dict(raw or {})
    while isinstance(candidate, Mapping):
        if isinstance(candidate.get("mutations"), Mapping):
            candidate = candidate["mutations"]
            continue
        if isinstance(candidate.get("payload"), Mapping):
            candidate = candidate["payload"]
            continue
        break
    return dict(candidate or {}) if isinstance(candidate, Mapping) else {}


def _load_daily_bundle(
    *,
    daily_root: str | Path,
    daily_json: str | Path | None,
    daily_universe_file: str | Path | None,
    trade_date: date,
    start: date,
    end: date,
    symbols: Sequence[str] | None,
) -> dict[str, Any]:
    if daily_json:
        payload = _load_mapping(daily_json)
        raw_daily_by_symbol = _normalize_rows_by_symbol(dict(payload.get("daily_by_symbol", payload)))
        expected_symbols = _daily_expected_symbols(
            cli_symbols=symbols,
            universe_file=daily_universe_file,
            payload=payload,
            fallback_symbols=raw_daily_by_symbol,
        )
        daily_by_symbol = _filter_rows_by_expected_symbols(raw_daily_by_symbol, expected_symbols)
        return {
            "source": {"type": "json", "path": str(_resolve_path(daily_json))},
            "daily_universe_file": str(_resolve_path(daily_universe_file)) if daily_universe_file else "",
            "expected_symbols": expected_symbols,
            "expected_symbols_sha256": _symbol_list_sha256(expected_symbols),
            "daily_by_symbol": daily_by_symbol,
            "daily_flow_by_symbol": _filter_rows_by_expected_symbols(_normalize_rows_by_symbol(payload.get("daily_flow_by_symbol", {})), expected_symbols),
            "daily_foreign_flow_by_symbol": _filter_rows_by_expected_symbols(_normalize_rows_by_symbol(payload.get("daily_foreign_flow_by_symbol", {})), expected_symbols),
            "daily_institutional_flow_by_symbol": _filter_rows_by_expected_symbols(_normalize_rows_by_symbol(payload.get("daily_institutional_flow_by_symbol", {})), expected_symbols),
            "index_ohlcv_by_symbol": _normalize_rows_by_symbol(payload.get("index_ohlcv_by_symbol", {}), zfill_keys=False),
        }

    root = _resolve_path(daily_root)
    manifest = load_manifest(root)
    wanted = _daily_expected_symbols(
        cli_symbols=symbols,
        universe_file=daily_universe_file,
        payload=manifest,
        fallback_symbols=available_daily_symbols(root),
    )
    daily_by_symbol: dict[str, list[dict[str, Any]]] = {}
    flow_by_symbol: dict[str, list[dict[str, Any]]] = {}
    foreign_by_symbol: dict[str, list[dict[str, Any]]] = {}
    institutional_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for symbol in wanted:
        daily_rows = _frame_records(load_daily_ohlcv(root, symbol, start=start, end=end), drop=("ticker",))
        if not daily_rows:
            continue
        daily_by_symbol[symbol] = daily_rows
        flow_by_symbol[symbol] = _frame_records(load_daily_flow(root, symbol, start=start, end=end), drop=("ticker",))
        foreign_by_symbol[symbol] = _frame_records(load_daily_foreign_flow(root, symbol, start=start, end=end), drop=("ticker",))
        institutional_by_symbol[symbol] = _frame_records(load_daily_institutional_flow(root, symbol, start=start, end=end), drop=("ticker",))
    index_frame = load_index_ohlcv(root, None, start=start, end=end)
    index_by_symbol = {
        str(code): _frame_records(group.drop(columns=["index_code"], errors="ignore"))
        for code, group in index_frame.groupby("index_code")
    } if not index_frame.empty and "index_code" in index_frame.columns else {}
    return {
        "source": {"type": "daily_lrs_parquet", "root": str(root), "manifest": _manifest_summary(manifest), "trade_date": trade_date.isoformat()},
        "daily_universe_file": str(_resolve_path(daily_universe_file)) if daily_universe_file else "",
        "expected_symbols": tuple(wanted),
        "expected_symbols_sha256": _symbol_list_sha256(wanted),
        "daily_by_symbol": daily_by_symbol,
        "daily_flow_by_symbol": flow_by_symbol,
        "daily_foreign_flow_by_symbol": foreign_by_symbol,
        "daily_institutional_flow_by_symbol": institutional_by_symbol,
        "index_ohlcv_by_symbol": index_by_symbol,
    }


def _load_intraday_bars(
    args: argparse.Namespace,
    trade_date: date,
    symbols: Sequence[str] | None,
) -> dict[tuple[date, str], tuple[MarketBar, ...]]:
    cutoff = _parse_cutoff(args.cutoff)
    if args.bars_json:
        payload = _load_mapping(args.bars_json)
        raw = payload.get("bars_by_symbol", payload)
        return _bars_from_json(raw, trade_date, cutoff)
    if args.bars_parquet:
        path = _resolve_path(args.bars_parquet)
        fingerprint = file_sha256(path)
        bars = [
            _bar_with_default_source(bar, source="operator_bars_parquet", source_fingerprint=fingerprint)
            for bar in load_market_bars_for_replay(path)
        ]
        return _group_bars(_filter_bars(bars, trade_date, cutoff))
    return _bars_from_intraday_root(_resolve_path(args.intraday_root), trade_date, cutoff, symbols)


def _bars_from_json(raw: Any, trade_date: date, cutoff: time) -> dict[tuple[date, str], tuple[MarketBar, ...]]:
    bars: list[MarketBar] = []
    for symbol, rows in dict(raw or {}).items():
        for row in list(rows or ()):
            bars.append(_bar_from_mapping(str(symbol).zfill(6), row, source="operator_bars_json", source_fingerprint=""))
    return _group_bars(_filter_bars(bars, trade_date, cutoff))


def _bars_from_intraday_root(root: Path, trade_date: date, cutoff: time, symbols: Sequence[str] | None) -> dict[tuple[date, str], tuple[MarketBar, ...]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas and pyarrow are required to load intraday parquet bars") from exc
    wanted = _normalize_symbol_list(symbols) if symbols else sorted(path.name for path in root.iterdir() if path.is_dir())
    bars: list[MarketBar] = []
    for symbol in wanted:
        symbol_dir = root / symbol
        if not symbol_dir.is_dir():
            continue
        for path in _intraday_5m_files_for_trade_date(symbol_dir, symbol, trade_date):
            fingerprint = file_sha256(path)
            frame = pd.read_parquet(path)
            for row in frame.to_dict("records"):
                bars.append(_bar_from_mapping(symbol, row, source=str(path), source_fingerprint=fingerprint))
    return _group_bars(_filter_bars(bars, trade_date, cutoff))


def _filter_bars(bars: Sequence[MarketBar], trade_date: date, cutoff: time) -> list[MarketBar]:
    out = []
    for bar in bars:
        ts = bar.timestamp.astimezone(KST) if bar.timestamp.tzinfo else bar.timestamp.replace(tzinfo=KST)
        if ts.date() != trade_date or ts.time() >= cutoff or not bar.is_completed:
            continue
        out.append(replace(bar, timestamp=ts))
    return sorted(out, key=lambda item: (item.symbol, item.timestamp))


def _group_bars(bars: Sequence[MarketBar]) -> dict[tuple[date, str], tuple[MarketBar, ...]]:
    grouped: dict[tuple[date, str], list[MarketBar]] = {}
    for bar in bars:
        key = (bar.timestamp.astimezone(KST).date(), str(bar.symbol).zfill(6))
        grouped.setdefault(key, []).append(bar)
    return {key: tuple(value) for key, value in sorted(grouped.items())}


def _bar_from_mapping(symbol: str, row: Mapping[str, Any], *, source: str, source_fingerprint: str) -> MarketBar:
    timestamp = row.get("timestamp") or row.get("datetime") or row.get("bar_time")
    if hasattr(timestamp, "to_pydatetime"):
        timestamp = timestamp.to_pydatetime()
    elif not isinstance(timestamp, datetime):
        timestamp = datetime.fromisoformat(str(timestamp))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=KST)
    return MarketBar(
        symbol=str(row.get("symbol") or row.get("ticker") or symbol).zfill(6),
        timestamp=timestamp,
        timeframe=str(row.get("timeframe") or "5m"),
        open=float(row.get("open")),
        high=float(row.get("high")),
        low=float(row.get("low")),
        close=float(row.get("close")),
        volume=float(row.get("volume") or 0.0),
        is_completed=bool(row.get("is_completed", True)),
        source=str(row.get("source") or source),
        source_fingerprint=str(row.get("source_fingerprint") or source_fingerprint),
        metadata=dict(row.get("metadata") or {}),
    )


def _bar_with_default_source(bar: MarketBar, *, source: str, source_fingerprint: str) -> MarketBar:
    return replace(
        bar,
        source=bar.source or source,
        source_fingerprint=bar.source_fingerprint or source_fingerprint,
    )


def _load_sector_map(path: str | Path) -> dict[str, str]:
    return load_canonical_sector_map({"sector_map_path": str(_resolve_path(path))})


def _normalize_rows_by_symbol(raw: Any, *, zfill_keys: bool = True) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for symbol, items in dict(raw or {}).items():
        key = str(symbol).zfill(6) if zfill_keys else str(symbol)
        rows[key] = [dict(item) for item in list(items or ())]
    return rows


def _normalize_symbol_list(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        raw = raw.get("symbols") or raw.get("universe") or raw.get("values") or ()
    if isinstance(raw, str):
        values = [raw]
    else:
        values = list(raw or ())
    return tuple(dict.fromkeys(str(symbol).zfill(6) for symbol in values if str(symbol or "").strip()))


def _daily_expected_symbols(
    *,
    cli_symbols: Sequence[str] | None,
    universe_file: str | Path | None,
    payload: Mapping[str, Any],
    fallback_symbols: Any,
) -> tuple[str, ...]:
    if cli_symbols:
        return _normalize_symbol_list(cli_symbols)
    if universe_file:
        loaded = _load_mapping(universe_file)
        symbols = _normalize_symbol_list(loaded)
        if not symbols:
            raise ValueError(f"daily universe file has no symbols: {_resolve_path(universe_file)}")
        _validate_daily_universe_manifest(loaded, symbols, universe_file)
        return symbols
    for key in ("expected_symbols", "symbols", "olr_universe", "kalcb_universe", "universe"):
        symbols = _normalize_symbol_list(payload.get(key))
        if symbols:
            return symbols
    if isinstance(fallback_symbols, Mapping):
        return tuple(sorted(str(symbol).zfill(6) for symbol in fallback_symbols))
    return tuple(sorted(_normalize_symbol_list(fallback_symbols)))


def _filter_rows_by_expected_symbols(
    rows_by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
    expected_symbols: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    expected = set(expected_symbols)
    return {symbol: [dict(row) for row in rows] for symbol, rows in dict(rows_by_symbol).items() if symbol in expected}


def _validate_daily_universe_manifest(payload: Mapping[str, Any], symbols: Sequence[str], universe_file: str | Path) -> None:
    source = str(_resolve_path(universe_file))
    declared_count = _first_int(payload, "symbol_count", "expected_symbol_count", "complete_universe_size")
    if declared_count is not None and declared_count != len(symbols):
        raise ValueError(
            "daily universe file symbol_count mismatch: "
            f"path={source} declared={declared_count} loaded={len(symbols)}"
        )
    declared_hash = _first_present(payload, "symbols_sha256", "symbol_list_sha256")
    if declared_hash is not None:
        actual_hash = _symbol_list_sha256(symbols)
        if str(declared_hash).lower() != actual_hash:
            raise ValueError(
                "daily universe file symbols_sha256 mismatch: "
                f"path={source} declared={declared_hash} actual={actual_hash}"
            )


def _first_int(payload: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key in payload and payload[key] is not None:
            return int(payload[key])
    return None


def _first_present(payload: Mapping[str, Any], *keys: str) -> Any | None:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _symbol_list_sha256(symbols: Sequence[str]) -> str:
    normalized = _normalize_symbol_list(symbols)
    return hashlib.sha256(("\n".join(normalized) + "\n").encode("utf-8")).hexdigest()


def _validate_daily_universe(
    bundle: Mapping[str, Any],
    configs: Mapping[str, Mapping[str, Any]],
    strategies: set[str],
    *,
    allow_partial: bool,
) -> DailyUniverseReport:
    expected = tuple(str(symbol).zfill(6) for symbol in bundle.get("expected_symbols") or sorted(bundle["daily_by_symbol"]))
    loaded = tuple(sorted(str(symbol).zfill(6) for symbol in bundle["daily_by_symbol"]))
    missing = tuple(sorted(set(expected) - set(loaded)))
    unexpected = tuple(sorted(set(loaded) - set(expected)))
    if missing and not allow_partial:
        raise ValueError(
            "daily OHLCV input is missing required universe symbols: "
            f"missing_count={len(missing)} sample={list(missing[:10])}"
        )
    if unexpected and not allow_partial:
        raise ValueError(
            "daily OHLCV input contains symbols outside the approved universe: "
            f"unexpected_count={len(unexpected)} sample={list(unexpected[:10])}"
        )
    if "OLR" in strategies and not allow_partial:
        expected_count = int(configs["OLR"]["config"].complete_universe_size)
        if len(loaded) != expected_count:
            raise ValueError(
                "OLR live artifact requires the optimized complete universe size: "
                f"expected={expected_count} loaded={len(loaded)}"
            )
    return DailyUniverseReport(
        expected_symbols=expected,
        loaded_symbols=loaded,
        missing_symbols=missing,
        unexpected_symbols=unexpected,
    )


def _validate_daily_freshness(
    daily_by_symbol: Mapping[str, Sequence[Mapping[str, Any]]],
    trade_date: date,
    *,
    daily_end: date,
    explicit_daily_end: bool,
    max_lag_days: int,
    expected_symbols: Sequence[str],
) -> DailyFreshnessReport:
    per_symbol_latest: dict[str, date] = {}
    missing_dated_rows: list[str] = []
    for symbol in expected_symbols:
        row_dates = [_row_date(row) for row in daily_by_symbol.get(symbol, ())]
        dates = [row_date for row_date in row_dates if row_date is not None]
        if not dates:
            missing_dated_rows.append(symbol)
            continue
        per_symbol_latest[symbol] = max(dates)
    if missing_dated_rows:
        raise ValueError(
            "daily OHLCV input has no dated rows for required symbols: "
            f"missing_count={len(missing_dated_rows)} sample={missing_dated_rows[:10]}"
        )
    if not per_symbol_latest:
        raise ValueError("daily OHLCV input has no dated rows")
    latest = max(per_symbol_latest.values())
    if latest >= trade_date:
        raise ValueError(f"daily OHLCV input includes same-day or future rows: latest={latest.isoformat()} trade_date={trade_date.isoformat()}")
    stale_symbols = {
        symbol: row_date
        for symbol, row_date in per_symbol_latest.items()
        if (trade_date - row_date).days > max_lag_days
    }
    if explicit_daily_end:
        stale_symbols.update({
            symbol: row_date
            for symbol, row_date in per_symbol_latest.items()
            if row_date < daily_end
        })
    if stale_symbols:
        sample = {symbol: row_date.isoformat() for symbol, row_date in list(sorted(stale_symbols.items()))[:10]}
        raise ValueError(
            f"daily OHLCV input is stale for trade date {trade_date.isoformat()}: "
            f"stale_symbol_count={len(stale_symbols)} max_daily_lag_days={max_lag_days} sample={sample}"
        )
    return DailyFreshnessReport(
        latest=latest,
        earliest_symbol_latest=min(per_symbol_latest.values()),
        per_symbol_latest=per_symbol_latest,
    )


def _validate_afternoon_coverage(
    bars: Mapping[tuple[date, str], Sequence[MarketBar]],
    required_symbols: Sequence[str],
    *,
    min_bars_per_symbol: int,
    allow_partial: bool,
) -> AfternoonCoverageReport:
    required = tuple(dict.fromkeys(str(symbol).zfill(6) for symbol in required_symbols))
    counts = {
        str(symbol).zfill(6): len(tuple(items))
        for (_day, symbol), items in bars.items()
    }
    missing = tuple(symbol for symbol in required if counts.get(symbol, 0) <= 0)
    underfilled = {
        symbol: counts.get(symbol, 0)
        for symbol in required
        if 0 < counts.get(symbol, 0) < min_bars_per_symbol
    }
    if (missing or underfilled) and not allow_partial:
        raise ValueError(
            "OLR afternoon artifact requires completed 5m bar coverage for every stage1 candidate: "
            f"required_symbols={len(required)} min_bars_per_symbol={min_bars_per_symbol} "
            f"missing={list(missing[:10])} underfilled={dict(list(underfilled.items())[:10])}"
        )
    return AfternoonCoverageReport(
        required_symbols=required,
        missing_symbols=missing,
        underfilled_symbols=underfilled,
        min_bars_per_symbol=min_bars_per_symbol,
    )


def _afternoon_min_bars(args: argparse.Namespace, config: OLRConfig) -> int:
    explicit = getattr(args, "min_afternoon_bars_per_symbol", None)
    if explicit is not None:
        return max(1, int(explicit))
    return max(int(config.afternoon_min_bar_count), _expected_5m_bar_count(_parse_cutoff(args.cutoff)))


def _expected_5m_bar_count(cutoff: time) -> int:
    start_minutes = DEFAULT_MARKET_OPEN.hour * 60 + DEFAULT_MARKET_OPEN.minute
    cutoff_minutes = cutoff.hour * 60 + cutoff.minute
    return max(1, (cutoff_minutes - start_minutes) // 5)


def _intraday_5m_files_for_trade_date(symbol_dir: Path, symbol: str, trade_date: date) -> tuple[Path, ...]:
    files = sorted(symbol_dir.glob(f"{symbol}_5m_*.parquet"))
    covering = [path for path in files if _intraday_file_covers_trade_date(path, trade_date)]
    if covering:
        return tuple(covering)
    return tuple(files[-1:]) if files else ()


def _intraday_file_covers_trade_date(path: Path, trade_date: date) -> bool:
    parts = path.stem.split("_")
    if len(parts) < 4:
        return True
    try:
        start = datetime.strptime(parts[-2], "%Y%m%d").date()
        end = datetime.strptime(parts[-1], "%Y%m%d").date()
    except ValueError:
        return True
    return start <= trade_date <= end


def _row_date(row: Mapping[str, Any]) -> date | None:
    value = row.get("date") or row.get("timestamp")
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value)
    return datetime.fromisoformat(text).date() if "T" in text else date.fromisoformat(text[:10])


def _parse_fixture_generated_at(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=KST)


def _frame_records(frame: Any, *, drop: Sequence[str] = ()) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    data = frame.drop(columns=list(drop), errors="ignore").to_dict("records")
    return [_jsonable_row(row) for row in data]


def _jsonable_row(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError:
        pd = None
    out: dict[str, Any] = {}
    for key, value in dict(row).items():
        if pd is not None and pd.isna(value):
            out[key] = None
        elif hasattr(value, "isoformat"):
            out[key] = value.isoformat()
        else:
            out[key] = value
    return out


def _result_payload(result: Any) -> dict[str, Any]:
    payload = asdict(result)
    payload["path"] = str(result.path) if result.path else ""
    return payload


def _manifest_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    tables = dict(manifest.get("tables") or {})
    return {
        "dataset_version": manifest.get("dataset_version"),
        "source_label": manifest.get("source_label"),
        "source_fingerprint": manifest.get("source_fingerprint"),
        "start": manifest.get("start"),
        "end": manifest.get("end"),
        "tables": {
            name: {
                "rows": dict(summary or {}).get("rows"),
                "min_date": dict(summary or {}).get("min_date"),
                "max_date": dict(summary or {}).get("max_date"),
            }
            for name, summary in tables.items()
        },
    }


def _load_mapping(path: str | Path) -> dict[str, Any]:
    resolved = _resolve_path(path)
    text = resolved.read_text(encoding="utf-8")
    if resolved.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        payload = yaml.safe_load(text) or {}
    else:
        payload = json.loads(text or "{}")
    if not isinstance(payload, Mapping):
        raise ValueError(f"{resolved} must contain a JSON/YAML object")
    return dict(payload)


def _parse_cutoff(raw: str) -> time:
    hour, minute = str(raw).split(":", 1)
    return time(int(hour), int(minute))


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else REPO_ROOT / resolved


def _emit(payload: Mapping[str, Any], output_json: str | Path | None = None) -> None:
    if output_json:
        target = _resolve_path(output_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate OLR/KALCB deployment artifacts from prepared VPS data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    daily = subparsers.add_parser("daily", help="Generate KALCB finalized daily and OLR stage1 daily artifacts.")
    _add_common_args(daily)
    daily.add_argument("--daily-root", default=str(DEFAULT_DAILY_ROOT), help="Prepared daily LRS parquet root.")
    daily.add_argument("--daily-json", help="Optional JSON input containing daily_by_symbol and optional flow maps.")
    daily.add_argument("--daily-universe-file", default=str(DEFAULT_DAILY_UNIVERSE_FILE), help="YAML/JSON file with the approved daily universe symbols.")
    daily.add_argument("--daily-end", help="Last daily row date to include. Defaults to trade date minus one day.")
    daily.add_argument("--daily-lookback-days", type=int, default=DEFAULT_DAILY_LOOKBACK_DAYS)
    daily.add_argument("--max-daily-lag-days", type=int, default=4, help="Reject daily inputs older than this many calendar days before trade date.")
    daily.add_argument("--strategies", nargs="+", default=["KALCB", "OLR"], choices=("KALCB", "OLR"))
    daily.add_argument("--source-fingerprint", help="Override source fingerprint; normally omitted so generators hash causal rows.")
    daily.add_argument("--fixture-generated-at", help="Fixture rehearsal only: override daily artifact generated_at timestamp, ISO format.")
    daily.add_argument("--allow-partial-universe", action="store_true", help="Fixture/shadow only: allow missing or subset daily universe inputs.")

    afternoon = subparsers.add_parser("afternoon", help="Generate the OLR final afternoon artifact from completed 5m bars.")
    _add_common_args(afternoon)
    afternoon.add_argument("--intraday-root", default=str(DEFAULT_INTRADAY_ROOT), help="Prepared per-symbol intraday parquet root.")
    afternoon.add_argument("--bars-parquet", help="Optional consolidated completed 5m bars parquet.")
    afternoon.add_argument("--bars-json", help="Optional JSON input containing bars_by_symbol.")
    afternoon.add_argument("--cutoff", default="14:30", help="KST cutoff; bars at or after this time are excluded.")
    afternoon.add_argument("--min-afternoon-bars-per-symbol", type=int, help="Minimum completed 5m bars required for every OLR stage1 candidate. Defaults to full 09:00-to-cutoff coverage.")
    afternoon.add_argument("--allow-partial-afternoon-bars", action="store_true", help="Fixture/shadow only: allow missing OLR stage1 candidate bars.")
    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trade-date", required=True, help="KRX trade date, YYYY-MM-DD.")
    parser.add_argument("--baseline-manifest", default=str(DEFAULT_BASELINE_MANIFEST), help="Frozen approved optimized-config manifest.")
    parser.add_argument("--sector-map", default=str(DEFAULT_SECTOR_MAP), help="Approved sector map YAML/JSON.")
    parser.add_argument("--kalcb-artifact-root", default=str(DEFAULT_KALCB_ARTIFACT_ROOT))
    parser.add_argument("--olr-artifact-root", default=str(DEFAULT_OLR_ARTIFACT_ROOT))
    parser.add_argument("--symbols", nargs="+", help="Optional symbol subset for fixture rehearsals or targeted regeneration.")
    parser.add_argument("--output-json", help="Optional path to save result JSON.")


if __name__ == "__main__":
    raise SystemExit(main())
