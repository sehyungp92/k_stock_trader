from __future__ import annotations

from pathlib import Path

import pytest

from backtests.strategies.kalcb.candidate_surfacing_recovery import (
    CAUSAL_FEATURE_KEYS,
    LEAKAGE_FEATURE_BLOCKLIST,
    PoolVariant,
    build_conservative_route_family_mutations,
    build_matched_incumbent_route_mutations,
    build_candidate_pools,
    describe_stage08_route_bundle,
    fit_causal_ranker_profile,
    score_causal_feature_rows,
    summarize_pool_recall,
)


def test_causal_feature_allowlist_excludes_oracle_and_path_labels() -> None:
    assert set(CAUSAL_FEATURE_KEYS).isdisjoint(LEAKAGE_FEATURE_BLOCKLIST)


def test_holdout_scoring_uses_frozen_train_profile_and_ignores_labels() -> None:
    train = [
        _row("2026-01-05", "AAA", relvol=1.0, cpr=0.40, target=0.10),
        _row("2026-01-05", "BBB", relvol=3.0, cpr=0.65, target=0.60),
        _row("2026-01-06", "CCC", relvol=7.0, cpr=0.90, target=0.95),
    ]
    profile = fit_causal_ranker_profile(train)
    holdout = _row("2026-04-01", "ZZZ", relvol=5.0, cpr=0.80, target=0.0)
    leaked = dict(holdout)
    leaked.update({"net_r": 999.0, "mfe_r": 999.0, "label_composite_oracle_recall": 999.0})

    score_clean = score_causal_feature_rows([holdout], profile)[0]["causal_ranker_score"]
    score_leaked = score_causal_feature_rows([leaked], profile)[0]["causal_ranker_score"]

    assert profile["source_window"] == "train"
    assert score_clean == pytest.approx(score_leaked)


def test_low_coverage_and_zero_variance_features_are_dropped() -> None:
    rows = [
        _row("2026-01-05", "AAA", relvol=1.0, cpr=0.60, target=0.1, gap=0.02),
        _row("2026-01-05", "BBB", relvol=2.0, cpr=0.70, target=0.4, gap=0.02),
        _row("2026-01-05", "CCC", relvol=3.0, cpr=0.80, target=0.8, gap=0.02),
    ]
    for row in rows:
        row["sector_intraday_rel_volume"] = None

    profile = fit_causal_ranker_profile(rows)

    assert profile["feature_stats"]["sector_intraday_rel_volume"]["status"] == "dropped_low_coverage"
    assert profile["feature_stats"]["first30_gap"]["status"] == "dropped_zero_iqr"


def test_candidate_pool_ranking_is_deterministic_and_marks_top8_active() -> None:
    rows = [
        _scored("2026-01-05", "CCC", score=1.0),
        _scored("2026-01-05", "AAA", score=1.0),
        _scored("2026-01-05", "BBB", score=0.5),
    ]

    pools = build_candidate_pools(rows, {}, [PoolVariant("unit_top3", 3, active_count=2)])
    symbols = [row["symbol"] for row in pools["unit_top3"]]

    assert symbols == ["AAA", "CCC", "BBB"]
    assert [row["pool_rank"] for row in pools["unit_top3"]] == [1, 2, 3]
    assert [row["pool_active"] for row in pools["unit_top3"]] == [True, True, False]
    assert [row["frontier_role_for_replay"] for row in pools["unit_top3"]] == ["initial_active", "initial_active", "frontier_shadow"]


def test_blend_variant_preserves_existing_frontier_then_fills_with_ranker() -> None:
    rows = [
        _scored("2026-01-05", "AAA", score=4.0),
        _scored("2026-01-05", "BBB", score=3.0),
        _scored("2026-01-05", "CCC", score=2.0),
        _scored("2026-01-05", "DDD", score=1.0),
    ]

    pools = build_candidate_pools(
        rows,
        {"2026-01-05": ("DDD", "CCC")},
        [PoolVariant("blend", 4, active_count=2, kind="blend_existing_50pct")],
    )

    assert [row["symbol"] for row in pools["blend"]] == ["DDD", "CCC", "AAA", "BBB"]


def test_recall_diagnostics_include_route_eligible_share() -> None:
    scored = [
        _scored("2026-01-05", "AAA", score=2.0, oracle_score=10.0, route="pullback_acceptance", relvol=6.0, cpr=0.8),
        _scored("2026-01-05", "BBB", score=1.0, oracle_score=5.0, route="first30_open", relvol=1.0, cpr=0.4),
    ]
    pools = build_candidate_pools(scored, {}, [PoolVariant("unit_top2", 2, active_count=1)])
    routes = [
        {
            "name": "unit_pullback",
            "mode": "pullback_acceptance",
            "require_initial_active": False,
            "context_min": {"first30_rel_volume": 5.0},
            "context_max": {"frontier_rank": 2.0},
        }
    ]

    summary = summarize_pool_recall(scored, pools, routes, {})
    variant = summary["variants"]["unit_top2"]

    assert variant["best_oracle_in_pool_share"] == pytest.approx(1.0)
    assert variant["best_quality_delayed_oracle_in_pool_share"] == pytest.approx(1.0)
    assert variant["route_eligible_share"] == pytest.approx(0.5)


def test_stage08_route_bundle_uses_delayed_families_without_first30_expansion() -> None:
    seed = {
        "kalcb.entry.routes": [
            {
                "name": "seed_pullback",
                "mode": "pullback_acceptance",
                "priority": 0,
                "require_initial_active": False,
                "max_frontier_rank": 8,
                "max_session_trades": 1,
                "risk_mult": 0.015,
                "notional_mult": 0.015,
                "context_min": {"first30_rel_volume": 5.3518},
            },
            {
                "name": "first30_anchor",
                "mode": "first30_open",
                "priority": 100,
                "require_initial_active": True,
                "max_frontier_rank": 8,
                "risk_mult": 0.99,
            },
        ]
    }

    mutations = build_conservative_route_family_mutations(seed)
    bundle = describe_stage08_route_bundle(mutations)
    routes = mutations["kalcb.entry.routes"]

    assert bundle["modes"] == ["pullback_acceptance", "avwap_reclaim", "or_high_reclaim"]
    assert bundle["first30_open_enabled"] is False
    assert all(route["require_initial_active"] is False for route in routes)
    assert all(route["max_session_trades"] == 1 for route in routes)
    assert all(route["context_min"]["first30_rel_volume"] == pytest.approx(5.3518) for route in routes)
    assert sum(route["risk_mult"] for route in routes) == pytest.approx(0.015)


def test_stage08_matched_incumbent_bundle_preserves_first30_anchor() -> None:
    seed = {
        "kalcb.entry.routes": [
            {"name": "seed_pullback", "mode": "pullback_acceptance", "risk_mult": 0.015},
            {"name": "first30_anchor", "mode": "first30_open", "risk_mult": 0.99, "require_initial_active": True},
        ]
    }

    mutations = build_matched_incumbent_route_mutations(seed)
    bundle = describe_stage08_route_bundle(mutations)

    assert bundle["modes"] == ["pullback_acceptance", "first30_open"]
    assert bundle["first30_open_enabled"] is True
    assert mutations["kalcb.entry.routes"] == seed["kalcb.entry.routes"]


def test_stage08_replay_helper_does_not_use_trade_plan_spec_config_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    source = (repo_root / "backtests" / "strategies" / "kalcb" / "candidate_surfacing_recovery.py").read_text(encoding="utf-8")

    assert "_core_config_for_spec" not in source


def _row(day: str, symbol: str, *, relvol: float, cpr: float, target: float, gap: float = 0.01) -> dict[str, object]:
    row: dict[str, object] = {
        "window": "train",
        "trade_date": day,
        "symbol": symbol,
        "sector": "TECH",
        "label_composite_oracle_recall": target,
        "first30_rel_volume": relvol,
        "first30_signal_bar_cpr": cpr,
        "first30_gap": gap,
    }
    for index, key in enumerate(CAUSAL_FEATURE_KEYS):
        row.setdefault(key, relvol * 0.1 + cpr * 0.2 + index * 0.001)
    row["first30_rel_volume"] = relvol
    row["first30_signal_bar_cpr"] = cpr
    row["first30_gap"] = gap
    return row


def _scored(
    day: str,
    symbol: str,
    *,
    score: float,
    oracle_score: float | None = None,
    route: str = "pullback_acceptance",
    relvol: float = 3.0,
    cpr: float = 0.7,
) -> dict[str, object]:
    return {
        "window": "train",
        "trade_date": day,
        "symbol": symbol,
        "sector": "TECH",
        "causal_ranker_score": score,
        "causal_rank_in_day": 1,
        "first30_rel_volume": relvol,
        "first30_signal_bar_cpr": cpr,
        "leading_sector_cluster": False,
        "oracle_label_available": oracle_score is not None,
        "oracle_score": oracle_score,
        "oracle_route_family": route,
        "label_top_decile_oracle": 1.0 if oracle_score and oracle_score >= 10.0 else 0.0,
        "net_r": oracle_score or 0.0,
        "mfe_r": oracle_score or 0.0,
    }
