from __future__ import annotations

from pathlib import Path

import pytest

from backtests.strategies.kalcb.shadow_ledger_reranker import (
    aggregate_route_shadow_rows,
    build_same_day_reranker_artifacts,
    feature_coverage,
    fit_reranker_profile,
    prepare_shadow_ledger_rows,
    score_shadow_ledger_rows,
)


def test_route_probe_aggregation_by_day_symbol_route() -> None:
    rows = [
        _shadow_trade("2026-01-05", "AAA", r=2.0, mfe=5.0, mae=-0.5, route="first30_open"),
        _shadow_trade("2026-01-05", "AAA", r=-1.0, mfe=1.5, mae=-2.0, route="first30_open", exit_reason="stop"),
        _shadow_trade("2026-01-05", "BBB", r=3.0, mfe=4.0, mae=-0.2, route="first30_open"),
    ]

    aggregated = aggregate_route_shadow_rows(rows, route_family="first30_open")

    aaa = aggregated[("2026-01-05", "AAA")]
    assert aaa["route_family"] == "first30_open"
    assert aaa["shadow_trade_count"] == 2
    assert aaa["shadow_total_r"] == pytest.approx(1.0)
    assert aaa["shadow_max_mfe_r"] == pytest.approx(5.0)
    assert aaa["shadow_min_mae_r"] == pytest.approx(-2.0)
    assert aaa["shadow_exit_reasons"] == {"eod": 1, "stop": 1}


def test_replacement_labels_and_daily_sector_feature_copying() -> None:
    missing_context = _candidate("2026-01-05", "BBB", sector="BIO", actual_total=1.0, weakest=-2.0)
    missing_context.pop("sector_daily_score_pct")
    missing_context.pop("sector_intraday_score_pct")
    rows = [
        _candidate(
            "2026-01-05",
            "AAA",
            sector="SEMIS",
            actual_total=1.0,
            weakest=-2.0,
            route_outcomes={
                "avwap_reclaim": {
                    "shadow_trade_count": 1,
                    "shadow_total_r": 5.0,
                    "shadow_max_mfe_r": 7.0,
                    "shadow_min_mae_r": -0.6,
                    "shadow_avg_mfe_capture": 0.70,
                    "shadow_avg_giveback_r": 2.0,
                }
            },
            sector_daily_score_pct=91.0,
            sector_intraday_score_pct=84.0,
            stock_sector_daily_ret5_spread=0.13,
        ),
        missing_context,
    ]

    prepared = prepare_shadow_ledger_rows(rows, max_per_sector=8)
    row = next(item for item in prepared if item["symbol"] == "AAA")

    assert row["best_route_family"] == "avwap_reclaim"
    assert row["same_day_replacement_value_r"] == pytest.approx(4.0)
    assert row["marginal_slot_replacement_value_r"] == pytest.approx(7.0)
    assert row["sector_daily_score_pct"] == pytest.approx(91.0)
    assert row["sector_intraday_score_pct"] == pytest.approx(84.0)
    assert row["stock_sector_daily_ret5_spread"] == pytest.approx(0.13)

    coverage = feature_coverage(prepared, ("sector_daily_score_pct", "sector_intraday_score_pct"))
    assert coverage["sector_daily_score_pct"]["coverage"] == pytest.approx(0.5)
    assert coverage["sector_intraday_score_pct"]["coverage"] == pytest.approx(0.5)


def test_sector_crowding_penalty_and_deterministic_per_day_ranking() -> None:
    train = prepare_shadow_ledger_rows(
        [
            _candidate("2026-01-05", "AAA", sector="SEMIS", actual_total=0.0, weakest=-1.0, shadow_total=4.0),
            _candidate("2026-01-05", "BBB", sector="SEMIS", actual_total=0.0, weakest=-1.0, shadow_total=3.0),
            _candidate("2026-01-05", "CCC", sector="SEMIS", actual_total=0.0, weakest=-1.0, shadow_total=2.0),
        ],
        max_per_sector=2,
    )
    profile = fit_reranker_profile(train)

    scored_once = score_shadow_ledger_rows(train, profile)
    scored_twice = score_shadow_ledger_rows(train, profile)

    assert all(row["candidate_sector_crowding_pressure"] == pytest.approx(0.5) for row in scored_once)
    assert all(row["max_per_sector_pressure"] == pytest.approx(0.0) for row in scored_once)
    assert all(row["reranker_penalties"]["over_concentration"] > 0.0 for row in scored_once)
    assert [row["symbol"] for row in scored_once] == [row["symbol"] for row in scored_twice]
    assert [row["reranker_rank_in_day"] for row in scored_once] == [1, 2, 3]
    assert [row["symbol"] for row in scored_once] == ["AAA", "BBB", "CCC"]


def test_delayed_route_eligibility_is_an_explicit_score_component() -> None:
    delayed = _candidate(
        "2026-01-05",
        "AAA",
        route_family_static_eligible_modes=["avwap_reclaim"],
        route_outcomes={
            "avwap_reclaim": {
                "shadow_trade_count": 1,
                "shadow_total_r": 3.0,
                "shadow_max_mfe_r": 5.0,
                "shadow_min_mae_r": -0.5,
                "shadow_avg_mfe_capture": 0.60,
                "shadow_avg_giveback_r": 1.0,
            }
        },
    )
    first30_only = _candidate("2026-01-05", "BBB", shadow_total=3.0)
    rows = prepare_shadow_ledger_rows([delayed, first30_only], max_per_sector=8)

    scored = {row["symbol"]: row for row in score_shadow_ledger_rows(rows, fit_reranker_profile(rows))}

    assert scored["AAA"]["reranker_components"]["delayed_route_eligibility"] > 0.0
    assert scored["BBB"]["reranker_components"]["delayed_route_eligibility"] == pytest.approx(0.0)


def test_holdout_fields_are_not_used_to_fit_train_profile(tmp_path: Path) -> None:
    train = [_candidate("2026-01-05", "AAA", sector="SEMIS", actual_total=0.0, weakest=-1.0, shadow_total=4.0)]
    holdout = [_candidate("2026-04-02", "ZZZ", sector="HOLDOUT_ONLY", actual_total=0.0, weakest=-1.0, shadow_total=8.0)]

    summary = build_same_day_reranker_artifacts(train, holdout, output_dir=tmp_path, max_per_sector=8)

    profile = summary["profile"]
    assert profile["source_window"] == "train"
    assert "SEMIS" in profile["sector_priors"]
    assert "HOLDOUT_ONLY" not in profile["sector_priors"]
    assert summary["holdout_sector_validation"]["usage"] == "validation_veto_not_train_score_feature"
    assert Path(summary["artifact_paths"]["train_jsonl"]).exists()
    assert Path(summary["artifact_paths"]["holdout_jsonl"]).exists()
    assert Path(summary["artifact_paths"]["summary_json"]).exists()
    assert Path(summary["artifact_paths"]["report_md"]).exists()


def _shadow_trade(
    day: str,
    symbol: str,
    *,
    r: float,
    mfe: float,
    mae: float,
    route: str,
    exit_reason: str = "eod",
) -> dict[str, object]:
    return {
        "entry_date": day,
        "symbol": symbol,
        "r": r,
        "mfe_r": mfe,
        "mae_r": mae,
        "giveback_r": mfe - r,
        "exit_reason": exit_reason,
        "entry_route_mode": route,
    }


def _candidate(
    day: str,
    symbol: str,
    *,
    sector: str = "SEMIS",
    actual_total: float = 0.0,
    weakest: float = 0.0,
    shadow_total: float | None = None,
    route_outcomes: dict[str, dict[str, object]] | None = None,
    **features: object,
) -> dict[str, object]:
    outcomes = dict(route_outcomes or {})
    if shadow_total is not None:
        outcomes.setdefault(
            "first30_open",
            {
                "shadow_trade_count": 1,
                "shadow_total_r": shadow_total,
                "shadow_max_mfe_r": max(shadow_total + 1.0, 1.0),
                "shadow_min_mae_r": -0.5,
                "shadow_avg_mfe_capture": 0.50,
                "shadow_avg_giveback_r": 1.0,
            },
        )
    row: dict[str, object] = {
        "window": "train",
        "trade_date": day,
        "symbol": symbol,
        "sector": sector,
        "frontier_rank": 3,
        "candidate_rank": 1,
        "frontier_role": "shadow",
        "first30_rel_volume": 6.0,
        "first30_signal_bar_cpr": 0.82,
        "same_day_actual_total_r": actual_total,
        "same_day_weakest_actual_r": weakest,
        "same_day_actual_sector_counts": {sector: 1},
        "route_outcomes": outcomes,
        "sector_daily_score_pct": 70.0,
        "sector_daily_participation": 0.65,
        "sector_intraday_score_pct": 70.0,
        "sector_intraday_breadth": 0.60,
    }
    row.update(features)
    return row
