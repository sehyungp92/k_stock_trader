from __future__ import annotations

import json
from argparse import Namespace
from datetime import date, timedelta

import pytest

from scripts import generate_olr_kalcb_artifacts as generator


def test_artifact_generator_daily_and_afternoon_from_json(tmp_path):
    trade_date = date(2026, 2, 2)
    baseline = _baseline_manifest(tmp_path)
    daily_json = tmp_path / "daily.json"
    daily_json.write_text(json.dumps({"daily_by_symbol": _daily_rows(trade_date)}), encoding="utf-8")
    bars_json = tmp_path / "bars.json"
    bars_json.write_text(json.dumps({"bars_by_symbol": _bars(trade_date)}), encoding="utf-8")
    sector_map = tmp_path / "sector_map.json"
    sector_map.write_text(json.dumps({"sector_map": {"005930": "SEMIS", "000660": "SEMIS", "035420": "IT"}}), encoding="utf-8")
    kalcb_root = tmp_path / "kalcb"
    olr_root = tmp_path / "olr"

    daily = generator.run_daily(
        Namespace(
            trade_date=trade_date.isoformat(),
            daily_end=None,
            daily_lookback_days=120,
            max_daily_lag_days=4,
            baseline_manifest=str(baseline),
            sector_map=str(sector_map),
            daily_root="unused",
            daily_json=str(daily_json),
            daily_universe_file=None,
            symbols=None,
            strategies=["KALCB", "OLR"],
            kalcb_artifact_root=str(kalcb_root),
            olr_artifact_root=str(olr_root),
            source_fingerprint=None,
            allow_partial_universe=False,
        )
    )

    assert daily["passed"] is True
    assert (kalcb_root / f"candidate_snapshot_{trade_date.isoformat()}.json").is_file()
    assert (olr_root / "stage1" / f"candidate_snapshot_{trade_date.isoformat()}.json").is_file()

    afternoon = generator.run_afternoon(
        Namespace(
            trade_date=trade_date.isoformat(),
            baseline_manifest=str(baseline),
            sector_map=str(sector_map),
            bars_json=str(bars_json),
            bars_parquet=None,
            intraday_root="unused",
            symbols=None,
            cutoff="14:30",
            min_afternoon_bars_per_symbol=2,
            allow_partial_afternoon_bars=False,
            kalcb_artifact_root=str(kalcb_root),
            olr_artifact_root=str(olr_root),
        )
    )

    assert afternoon["passed"] is True
    assert afternoon["bar_count"] == 6
    assert (olr_root / "final" / f"candidate_snapshot_{trade_date.isoformat()}.json").is_file()


def test_artifact_generator_rejects_partial_olr_daily_universe(tmp_path):
    trade_date = date(2026, 2, 2)
    baseline = _baseline_manifest(tmp_path)
    rows = _daily_rows(trade_date)
    rows.pop("035420")
    daily_json = tmp_path / "daily.json"
    daily_json.write_text(json.dumps({"daily_by_symbol": rows}), encoding="utf-8")
    sector_map = tmp_path / "sector_map.json"
    sector_map.write_text(json.dumps({"sector_map": {"005930": "SEMIS", "000660": "SEMIS"}}), encoding="utf-8")

    with pytest.raises(ValueError, match="complete universe size"):
        generator.run_daily(
            Namespace(
                trade_date=trade_date.isoformat(),
                daily_end=None,
                daily_lookback_days=120,
                max_daily_lag_days=4,
                baseline_manifest=str(baseline),
                sector_map=str(sector_map),
                daily_root="unused",
                daily_json=str(daily_json),
                daily_universe_file=None,
                symbols=None,
                strategies=["OLR"],
                kalcb_artifact_root=str(tmp_path / "kalcb"),
                olr_artifact_root=str(tmp_path / "olr"),
                source_fingerprint=None,
                allow_partial_universe=False,
            )
        )


def test_artifact_generator_rejects_stale_symbol_daily_rows(tmp_path):
    trade_date = date(2026, 2, 2)
    baseline = _baseline_manifest(tmp_path)
    rows = _daily_rows(trade_date)
    rows["035420"] = _rows(trade_date - timedelta(days=8), 35_000, 80)
    daily_json = tmp_path / "daily.json"
    daily_json.write_text(json.dumps({"daily_by_symbol": rows}), encoding="utf-8")
    sector_map = tmp_path / "sector_map.json"
    sector_map.write_text(json.dumps({"sector_map": {"005930": "SEMIS", "000660": "SEMIS", "035420": "IT"}}), encoding="utf-8")

    with pytest.raises(ValueError, match="stale_symbol_count"):
        generator.run_daily(
            Namespace(
                trade_date=trade_date.isoformat(),
                daily_end=None,
                daily_lookback_days=120,
                max_daily_lag_days=4,
                baseline_manifest=str(baseline),
                sector_map=str(sector_map),
                daily_root="unused",
                daily_json=str(daily_json),
                daily_universe_file=None,
                symbols=None,
                strategies=["KALCB", "OLR"],
                kalcb_artifact_root=str(tmp_path / "kalcb"),
                olr_artifact_root=str(tmp_path / "olr"),
                source_fingerprint=None,
                allow_partial_universe=False,
            )
        )


def test_artifact_generator_enforces_daily_universe_manifest_count_and_hash(tmp_path):
    trade_date = date(2026, 2, 2)
    baseline = _baseline_manifest(tmp_path)
    symbols = ("005930", "000660", "035420")
    digest = generator._symbol_list_sha256(symbols)
    universe = tmp_path / "olr_deployment_universe.yaml"
    universe.write_text(
        "\n".join(
            [
                "symbol_count: 3",
                f"symbols_sha256: {digest}",
                "symbols:",
                '  - "005930"',
                '  - "000660"',
                '  - "035420"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    daily_json = tmp_path / "daily.json"
    daily_json.write_text(json.dumps({"daily_by_symbol": _daily_rows(trade_date)}), encoding="utf-8")
    sector_map = tmp_path / "sector_map.json"
    sector_map.write_text(json.dumps({"sector_map": {"005930": "SEMIS", "000660": "SEMIS", "035420": "IT"}}), encoding="utf-8")

    daily = generator.run_daily(
        Namespace(
            trade_date=trade_date.isoformat(),
            daily_end=None,
            daily_lookback_days=120,
            max_daily_lag_days=4,
            baseline_manifest=str(baseline),
            sector_map=str(sector_map),
            daily_root="unused",
            daily_json=str(daily_json),
            daily_universe_file=str(universe),
            symbols=None,
            strategies=["OLR"],
            kalcb_artifact_root=str(tmp_path / "kalcb"),
            olr_artifact_root=str(tmp_path / "olr"),
            source_fingerprint=None,
            allow_partial_universe=False,
        )
    )

    assert daily["expected_symbols"] == 3
    assert daily["expected_symbols_sha256"] == digest
    assert daily["daily_universe_file"] == str(universe.resolve())

    universe.write_text(
        "\n".join(
            [
                "symbol_count: 3",
                f"symbols_sha256: {'0' * 64}",
                "symbols:",
                '  - "005930"',
                '  - "000660"',
                '  - "035420"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="symbols_sha256 mismatch"):
        generator.run_daily(
            Namespace(
                trade_date=trade_date.isoformat(),
                daily_end=None,
                daily_lookback_days=120,
                max_daily_lag_days=4,
                baseline_manifest=str(baseline),
                sector_map=str(sector_map),
                daily_root="unused",
                daily_json=str(daily_json),
                daily_universe_file=str(universe),
                symbols=None,
                strategies=["OLR"],
                kalcb_artifact_root=str(tmp_path / "kalcb2"),
                olr_artifact_root=str(tmp_path / "olr2"),
                source_fingerprint=None,
                allow_partial_universe=False,
            )
        )


def test_artifact_generator_cli_defaults_to_approved_olr_deployment_universe():
    args = generator._parser().parse_args(["daily", "--trade-date", "2026-02-02"])

    assert args.daily_universe_file.replace("\\", "/").endswith("config/olr_kalcb/olr_deployment_universe_103.yaml")


def test_artifact_generator_rejects_missing_stage1_afternoon_bars(tmp_path):
    trade_date = date(2026, 2, 2)
    baseline = _baseline_manifest(tmp_path)
    daily_json = tmp_path / "daily.json"
    daily_json.write_text(json.dumps({"daily_by_symbol": _daily_rows(trade_date)}), encoding="utf-8")
    sector_map = tmp_path / "sector_map.json"
    sector_map.write_text(json.dumps({"sector_map": {"005930": "SEMIS", "000660": "SEMIS", "035420": "IT"}}), encoding="utf-8")
    kalcb_root = tmp_path / "kalcb"
    olr_root = tmp_path / "olr"
    generator.run_daily(
        Namespace(
            trade_date=trade_date.isoformat(),
            daily_end=None,
            daily_lookback_days=120,
            max_daily_lag_days=4,
            baseline_manifest=str(baseline),
            sector_map=str(sector_map),
            daily_root="unused",
            daily_json=str(daily_json),
            daily_universe_file=None,
            symbols=None,
            strategies=["OLR"],
            kalcb_artifact_root=str(kalcb_root),
            olr_artifact_root=str(olr_root),
            source_fingerprint=None,
            allow_partial_universe=False,
        )
    )
    bars_json = tmp_path / "bars.json"
    bars = _bars(trade_date)
    stage1_payload = json.loads((olr_root / "stage1" / f"candidate_snapshot_{trade_date.isoformat()}.json").read_text(encoding="utf-8"))
    missing_symbol = stage1_payload["candidates"][0]["symbol"]
    bars.pop(missing_symbol)
    bars_json.write_text(json.dumps({"bars_by_symbol": bars}), encoding="utf-8")

    with pytest.raises(ValueError, match="every stage1 candidate"):
        generator.run_afternoon(
            Namespace(
                trade_date=trade_date.isoformat(),
                baseline_manifest=str(baseline),
                sector_map=str(sector_map),
                bars_json=str(bars_json),
                bars_parquet=None,
                intraday_root="unused",
                symbols=None,
                cutoff="14:30",
                min_afternoon_bars_per_symbol=2,
                allow_partial_afternoon_bars=False,
                kalcb_artifact_root=str(kalcb_root),
                olr_artifact_root=str(olr_root),
            )
        )


def test_intraday_loader_selects_only_trade_date_covering_5m_files(tmp_path):
    trade_date = date(2026, 2, 2)
    symbol_dir = tmp_path / "005930"
    symbol_dir.mkdir()
    old = symbol_dir / "005930_5m_20250101_20250131.parquet"
    current = symbol_dir / "005930_5m_20260201_20260228.parquet"
    other_timeframe = symbol_dir / "005930_1m_20260201_20260228.parquet"
    for path in (old, current, other_timeframe):
        path.write_text("placeholder", encoding="utf-8")

    assert generator._intraday_5m_files_for_trade_date(symbol_dir, "005930", trade_date) == (current,)


def _baseline_manifest(tmp_path):
    kalcb_config = tmp_path / "kalcb_optimized_config.json"
    kalcb_config.write_text(
        json.dumps(
            {
                "mutations": {
                    "kalcb.research.min_adv20_krw": 1_000_000,
                    "kalcb.research.top_long_count": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    olr_config = tmp_path / "olr_optimized_config.json"
    olr_config.write_text(
        json.dumps(
            {
                "mutations": {
                    "olr.research.min_adv20_krw": 1_000_000,
                    "olr.universe.complete_size": 3,
                    "olr.research.top_long_count": 3,
                    "olr.signal.daily_min_score": 0.0,
                    "olr.afternoon.top_n": 2,
                    "olr.afternoon.min_bar_count": 2,
                    "olr.afternoon.min_ret": -999.0,
                    "olr.afternoon.min_vwap_ret": -999.0,
                    "olr.afternoon.min_market_score": -999.0,
                }
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "baseline_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {"label": "kalcb_round_test_optimized_config", "path": str(kalcb_config)},
                    {"label": "olr_round_test_optimized_config", "path": str(olr_config)},
                ]
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _daily_rows(trade_date):
    return {
        "005930": _rows(trade_date, 50_000, 150),
        "000660": _rows(trade_date, 45_000, 100),
        "035420": _rows(trade_date, 35_000, 80),
    }


def _rows(trade_date, start, drift):
    first = trade_date - timedelta(days=90)
    rows = []
    for index in range(90):
        day = first + timedelta(days=index)
        close = start + drift * index
        rows.append(
            {
                "date": day.isoformat(),
                "open": close - 50,
                "high": close + 500,
                "low": close - 500,
                "close": close,
                "volume": 2_000_000,
            }
        )
    return rows


def _bars(trade_date):
    return {
        symbol: [
            _bar(symbol, trade_date, "09:00", 100.0, 103.0, 99.0, 102.0),
            _bar(symbol, trade_date, "14:25", 102.0, 105.0, 101.0, 104.0),
        ]
        for symbol in ("005930", "000660", "035420")
    }


def _bar(symbol, trade_date, hhmm, open_, high, low, close):
    return {
        "symbol": symbol,
        "timestamp": f"{trade_date.isoformat()}T{hhmm}:00+09:00",
        "timeframe": "5m",
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": 10_000,
        "is_completed": True,
    }
