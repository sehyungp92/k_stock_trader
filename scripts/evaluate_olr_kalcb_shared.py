from __future__ import annotations

# ruff: noqa: E402 -- repository scripts add the project root before local imports.

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtests.config import load_yaml_config
from backtests.strategies.portfolio_synergy.source_replay import (
    PromotedReplayInputs,
    load_promoted_kalcb_replay,
    load_promoted_olr_snapshots,
    overlap_attribution,
    require_parity,
    run_kalcb_standalone,
    run_native_risk_shared_replay,
    run_olr_standalone,
    shared_vs_standalone_attribution,
    summarize_replay,
)
from strategy_common.clock import KST
from strategy_common.market import MarketBar
from strategy_olr.config import OLRConfig
from strategy_olr.models import OLRDailySnapshot


PINNED_INTRADAY_END = "20260512"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay promoted KALCB + OLR through their native cores on one shared account."
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=ROOT / "config/olr_kalcb/shared_replay.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec = _read_json(args.spec)
    initial_equity = float(spec["initial_equity"])
    cache_dir = _repo_path(spec["cache_dir"])

    kalcb_spec = dict(spec["kalcb"])
    kalcb_raw = load_yaml_config(_repo_path(kalcb_spec["optimization_config"]))
    kalcb_optimized = _read_json(_repo_path(kalcb_spec["optimized_result"]))
    kalcb_config, kalcb_replay, kalcb_lineage = load_promoted_kalcb_replay(
        ROOT,
        kalcb_raw,
        kalcb_optimized,
        cache_dir=cache_dir,
        snapshot_root=_repo_path(kalcb_spec["candidate_snapshot_root"]),
        intraday_root=ROOT / "data/kis_intraday_parquet",
        data_snapshot_end=PINNED_INTRADAY_END,
        market_bar_digest_path=_repo_path(kalcb_spec["market_bar_digest"]),
    )

    olr_spec = dict(spec["olr"])
    olr_raw = load_yaml_config(_repo_path(olr_spec["optimization_config"]))
    olr_optimized = _read_json(_repo_path(olr_spec["optimized_result"]))
    olr_mutations = dict(olr_optimized.get("mutations") or {})
    olr_config = OLRConfig.from_mapping(olr_raw, olr_mutations)
    olr_snapshots, olr_identity = load_promoted_olr_snapshots(
        _repo_path(olr_spec["candidate_snapshot_root"]),
        daily_data_root=_repo_path(olr_spec["daily_data_root"]),
        training_end=str((olr_optimized.get("train_window") or {}).get("date_end") or ""),
        expected_candidate_snapshot_hash=str(
            (olr_optimized.get("execution_contract") or {}).get("candidate_snapshot_hash") or ""
        ),
        expected_candidate_config_hash=str(olr_spec["candidate_config_hash"]),
        round_generated_at_utc=str(olr_spec["round_generated_at_utc"]),
    )
    olr_pairs = _olr_replay_pairs(olr_snapshots, kalcb_replay.session_dates)
    olr_bars = tuple(_load_pinned_bars(olr_pairs))
    olr_lineage = {
        "round": int(olr_optimized.get("round") or 0),
        "bars": len(olr_bars),
        **olr_identity,
    }

    kalcb_standalone = run_kalcb_standalone(kalcb_config, kalcb_replay)
    kalcb_summary = summarize_replay(kalcb_standalone, initial_equity=kalcb_replay.initial_equity)
    require_parity(kalcb_summary, kalcb_spec["expected"], strategy="KALCB")

    olr_initial_equity = float((olr_optimized.get("execution_contract") or {}).get("initial_equity") or 10_000_000.0)
    olr_standalone = run_olr_standalone(
        olr_config,
        olr_snapshots,
        olr_bars,
        initial_equity=olr_initial_equity,
    )
    olr_summary = summarize_replay(olr_standalone, initial_equity=olr_initial_equity)
    require_parity(olr_summary, olr_spec["expected"], strategy="OLR")

    inputs = PromotedReplayInputs(
        kalcb_config=kalcb_config,
        kalcb_replay=kalcb_replay,
        olr_config=olr_config,
        olr_snapshots=olr_snapshots,
        olr_bars=olr_bars,
        lineage={"KALCB": kalcb_lineage, "OLR": olr_lineage},
    )
    shared = run_native_risk_shared_replay(inputs, initial_equity=initial_equity)
    shared_summary = summarize_replay(shared, initial_equity=initial_equity)
    output = {
        "schema_version": spec["schema_version"],
        "result_basis": "exact_promoted_consumed_content_native_strategy_risk_one_shared_account",
        "standalone_parity": {
            "status": "pass",
            "KALCB": asdict(kalcb_summary),
            "OLR": asdict(olr_summary),
        },
        "shared": asdict(shared_summary),
        "shared_vs_standalone": shared_vs_standalone_attribution(
            shared_summary,
            {"KALCB": kalcb_summary, "OLR": olr_summary},
        ),
        "overlap": overlap_attribution(shared),
        "risk_contract": {
            "KALCB_native_intraday_leverage": float(kalcb_config.intraday_leverage),
            "OLR_native_intraday_leverage": 1.0,
            "combined_gross_leverage": max(float(kalcb_config.intraday_leverage), 1.0),
            "additional_portfolio_symbol_or_sector_caps": False,
        },
        "olr_comparison_resolution": {
            "status": olr_identity["identity_status"],
            "historical_profit_factor": float(olr_spec["expected"]["profit_factor"]),
            "current_reproduced_profit_factor": olr_summary.profit_factor,
            "strategy_decay_attribution_allowed": olr_identity["strategy_decay_attribution_allowed"],
        },
        "lineage": inputs.lineage,
        "pinned_intraday_snapshot_end": PINNED_INTRADAY_END,
    }
    output_path = args.output or _repo_path(spec["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


def _olr_replay_pairs(
    snapshots: Mapping[date, OLRDailySnapshot],
    calendar: tuple[date, ...],
) -> set[tuple[date, str]]:
    calendar_index = {day: index for index, day in enumerate(calendar)}
    pairs: set[tuple[date, str]] = set()
    for day, snapshot in snapshots.items():
        if day not in calendar_index:
            raise ValueError(f"OLR snapshot date is outside the KALCB training calendar: {day}")
        offset = calendar_index[day]
        for candidate in snapshot.candidates:
            symbol = str(candidate.symbol).zfill(6)
            for replay_day in calendar[offset : min(offset + 3, len(calendar))]:
                pairs.add((replay_day, symbol))
    return pairs


def _load_pinned_bars(pairs: set[tuple[date, str]]) -> list[MarketBar]:
    import pandas as pd

    dates_by_symbol: dict[str, set[date]] = {}
    for day, symbol in pairs:
        dates_by_symbol.setdefault(symbol, set()).add(day)
    bars: list[MarketBar] = []
    data_root = ROOT / "data/kis_intraday_parquet"
    for symbol, dates in sorted(dates_by_symbol.items()):
        paths = sorted((data_root / symbol).glob(f"{symbol}_5m_*_{PINNED_INTRADAY_END}.parquet"))
        if not paths:
            raise FileNotFoundError(f"No pinned 5m parquet for OLR symbol {symbol}")
        frame = pd.read_parquet(paths[-1], columns=["timestamp", "open", "high", "low", "close", "volume"])
        frame = frame[frame["timestamp"].dt.date.isin(dates)]
        for row in frame.itertuples(index=False):
            timestamp = row.timestamp.to_pydatetime() if hasattr(row.timestamp, "to_pydatetime") else row.timestamp
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=KST)
            timestamp = timestamp.astimezone(KST)
            bars.append(
                MarketBar(
                    symbol=symbol,
                    timestamp=timestamp,
                    timeframe="5m",
                    open=float(row.open),
                    high=float(row.high),
                    low=float(row.low),
                    close=float(row.close),
                    volume=float(row.volume),
                    is_completed=True,
                    source="kis_krx_parquet",
                    source_fingerprint=f"pinned_{PINNED_INTRADAY_END}",
                )
            )
    return sorted(bars, key=lambda bar: (bar.timestamp, bar.symbol))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _repo_path(raw: Any) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    main()
