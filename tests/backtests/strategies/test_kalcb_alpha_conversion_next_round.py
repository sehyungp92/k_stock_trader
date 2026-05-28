from __future__ import annotations

import pytest

from scripts.kalcb_local_minimum_recovery import (
    NEXT_ROUND_CHALLENGER,
    NEXT_ROUND_SEED,
    _find_named_rows,
    _summarize_oracle_rows,
    build_path_state_exit_experiments,
)


def test_next_round_named_rows_are_found_without_requiring_top_level_shape() -> None:
    payload = {
        "evaluations": {
            "rows": [
                {"name": NEXT_ROUND_SEED, "mutations": {"a": 1}, "train": {"trade_count": 132}},
                {"name": "noise", "mutations": {"b": 2}},
            ]
        },
        "recommendation": {"candidate": {"name": NEXT_ROUND_CHALLENGER, "mutations": {"c": 3}}},
    }

    rows = _find_named_rows(payload, {NEXT_ROUND_SEED, NEXT_ROUND_CHALLENGER})

    assert set(rows) == {NEXT_ROUND_SEED, NEXT_ROUND_CHALLENGER}
    assert rows[NEXT_ROUND_SEED]["mutations"] == {"a": 1}
    assert rows[NEXT_ROUND_CHALLENGER]["mutations"] == {"c": 3}


def test_path_state_exit_experiments_are_route_family_scoped() -> None:
    experiments = build_path_state_exit_experiments({"seed": True}, {"challenger": True})

    assert {item.family for item in experiments} == {"path_state_exits"}
    assert all(item.replace_base for item in experiments)
    assert len(experiments) == 4
    for experiment in experiments:
        modes = experiment.mutations["kalcb.exit.path_quality_entry_route_modes"]
        assert modes == ["first30_open", "pullback_acceptance", "avwap_reclaim", "or_high_reclaim"]
        assert experiment.mutations["kalcb.exit.path_quality_enabled"] is True
        assert "kalcb.exit.target_r" not in experiment.mutations


def test_full_universe_oracle_summary_flags_out_of_pool_missed_alpha() -> None:
    rows = [
        _oracle_row("2026-01-05", "AAA", in_pool=True, score=5.0, net_r=2.0, mfe_r=6.0),
        _oracle_row("2026-01-05", "ZZZ", in_pool=False, score=9.0, net_r=4.0, mfe_r=12.0),
        _oracle_row("2026-01-06", "BBB", in_pool=True, score=7.0, net_r=3.0, mfe_r=8.0),
        _oracle_row("2026-01-06", "YYY", in_pool=False, score=6.0, net_r=2.0, mfe_r=7.0),
    ]

    summary = _summarize_oracle_rows(rows, session_count=2)

    assert summary["days_best_overall_outside_candidate_pool"] == 1
    assert summary["days_out_of_pool_beats_best_in_pool"] == 1
    assert summary["avg_out_of_pool_net_r_advantage"] == pytest.approx(2.0)
    assert summary["avg_out_of_pool_mfe_r_advantage"] == pytest.approx(6.0)
    assert summary["candidate_surfacing_verdict"] == "candidate_surfacing_likely_missing_material_alpha"


def _oracle_row(day: str, symbol: str, *, in_pool: bool, score: float, net_r: float, mfe_r: float) -> dict[str, object]:
    return {
        "trade_date": day,
        "symbol": symbol,
        "route_family": "pullback_acceptance",
        "in_candidate_pool": in_pool,
        "oracle_score": score,
        "net_r": net_r,
        "mfe_r": mfe_r,
    }
