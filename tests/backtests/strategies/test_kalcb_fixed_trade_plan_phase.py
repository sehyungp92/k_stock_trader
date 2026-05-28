from __future__ import annotations

from backtests.strategies.kalcb.fixed_trade_plan_phase import (
    KALCBFixedTradePlanOptimizationPlugin,
    OFFICIAL_PROMOTION_METRIC,
    SCORE_COMPONENTS,
    POOL_SOURCE_ACTIVE_COUNT_MUTATION,
    POOL_SOURCE_LABEL_MUTATION,
    POOL_SOURCE_PATH_MUTATION,
    SOURCE_PATH_MUTATION,
    SOURCE_RANK_MUTATION,
    SOURCE_SECTION_MUTATION,
    _initial_mutations_for_output,
    _paper_live_parity_requirements,
    _quality_guardrail_reject_reason,
    _route_candidate_passes,
    _resolve_fixed_candidate_source,
    _validation_gate_reject_reason,
    get_phase_candidates,
    score_fixed,
)
from backtests.auto.shared.phase_state import PhaseState
from backtests.strategies.common.plugin_base import attach_official_metric_contract


def test_paper_live_contract_requires_entry_route_attempts_metadata() -> None:
    payload = _paper_live_parity_requirements(
        {
            "fast_suppression_audit": {
                "pass": True,
                "fast_replay_digest": {
                    "trading_decision_hash": "decisions",
                    "strategy_action_hash": "actions",
                    "fill_hash": "fills",
                    "trade_hash": "trades",
                    "same_bar_fill_count": 0,
                },
            }
        },
        {
            "fill_timing": "next_5m_open",
            "auction_mode": "non_auction_continuous",
            "strategy_core_version": "unit",
        },
        mutations={"kalcb.entry.routes": [{"name": "secondary_post_or", "mode": "pullback_reclaim"}]},
    )

    action_fields = payload["evidence_schema"]["paper_strategy_actions.jsonl"]

    assert payload["required_before_promotion"] is True
    assert payload["required_strategy_action_fields"] == action_fields
    assert "entry_route" in action_fields
    assert "entry_route_mode" in action_fields
    assert "entry_route_priority" in action_fields
    assert "entry_route_attempts" in action_fields
    assert "entry_route_risk_mult" in action_fields
    assert "entry_route_notional_mult" in action_fields
    assert "entry_route_participation_mult" in action_fields
    assert "entry_route_max_session_trades" in action_fields
    assert "entry_route_context_min_keys" in action_fields
    assert "entry_route_context_max_keys" in action_fields
    assert "entry_route_context_exclude_keys" in action_fields
    assert "entry_route_session_count_before" in action_fields
    assert "first30_gap" in action_fields
    assert "first30_gap_retention_ratio" in action_fields
    assert "first30_gap_relvol" in action_fields
    assert "entry_path_completed_bars" in action_fields
    assert "h3_current_r" in action_fields
    assert "h6_current_r" in action_fields
    assert "daily_return_20d" in action_fields
    assert "daily_momentum_pct" in action_fields
    assert "daily_sector_alignment_pct" in action_fields
    assert "first30_sector_leadership_pct" in action_fields
    assert "continuation_joint_quality_pct" in action_fields
    assert "sector_daily_score_pct" in action_fields
    assert "sector_intraday_score_pct" in action_fields
    assert "session_sector_intraday_positive_share" in action_fields
    assert "effective_max_position_notional_pct" in action_fields
    assert "exit_path_quality_context" in action_fields
    assert payload["acceptance_criteria"]["entry_route_metadata_mismatch_count"] == 0
    assert payload["acceptance_criteria"]["exit_path_quality_context_mismatch_count"] == 0


def test_fixed_phase_promotion_contract_requires_audited_official_mtm() -> None:
    plugin = object.__new__(KALCBFixedTradePlanOptimizationPlugin)
    spec = plugin.get_phase_spec(1, PhaseState())

    assert spec.primary_promotion_metric == OFFICIAL_PROMOTION_METRIC
    assert spec.promotion_requires_audit_pass is True
    assert OFFICIAL_PROMOTION_METRIC in spec.official_metric_keys


def test_official_metric_contract_does_not_allow_audit_requirement_downgrade() -> None:
    metrics = {
        "official_mtm_net_return_pct": 0.05,
        "broker_net_return_pct": 0.10,
        "primary_promotion_metric": "broker_net_return_pct",
        "primary_promotion_basis": "closed_trade_net_pnl_over_initial_equity",
        "promotion_requires_audit_pass": False,
    }

    out = attach_official_metric_contract(
        metrics,
        primary_metric=OFFICIAL_PROMOTION_METRIC,
        requires_audit_pass=True,
        audit_pass=True,
    )

    assert out["primary_promotion_metric"] == OFFICIAL_PROMOTION_METRIC
    assert out["primary_promotion_value"] == 0.05
    assert out["primary_promotion_basis"] == "SimBroker.equity_curve_bar_level_mtm"
    assert out["promotion_requires_audit_pass"] is True
    assert out["metric_contract"]["primary_promotion_metric"] == OFFICIAL_PROMOTION_METRIC
    assert out["metric_contract"]["promotion_requires_audit_pass"] is True


def test_route_candidate_passes_supports_categorical_context_exclusion() -> None:
    route = {"name": "non_financial", "mode": "first30_open", "context_exclude": {"sector": ["FINANCIAL"]}}
    meta = {"first30_ret": 0.01, "sector": "FINANCIAL"}

    passed, reason = _route_candidate_passes(route, {}, meta)

    assert passed is False
    assert reason == "entry_context_exclude:sector"


def test_quality_guardrail_rejects_frequency_without_expectancy() -> None:
    baseline = {
        "trade_count": 95,
        "broker_net_return_pct": 0.95,
        "avg_trade_net_pct": 0.013,
        "worst_fold_net": 0.16,
        "avg_mfe_capture": 0.42,
    }
    candidate = {
        "trade_count": 106,
        "broker_net_return_pct": 0.94,
        "avg_trade_net_pct": 0.012,
        "worst_fold_net": 0.14,
        "avg_mfe_capture": 0.40,
    }

    assert _quality_guardrail_reject_reason(candidate, baseline) == "frequency_without_expectancy_guardrail"


def test_validation_gate_rejects_train_only_improvements() -> None:
    baseline = {"trade_count": 100, "broker_net_return_pct": 0.20, "avg_trade_net_pct": 0.004, "broker_max_drawdown_pct": 0.04}
    candidate = {"trade_count": 104, "broker_net_return_pct": 0.25, "avg_trade_net_pct": 0.0045, "broker_max_drawdown_pct": 0.041}
    validation_baseline = {"trade_count": 20, "broker_net_return_pct": 0.03, "avg_trade_net_pct": 0.003, "broker_max_drawdown_pct": 0.03}
    validation_candidate = {"trade_count": 21, "broker_net_return_pct": 0.02, "avg_trade_net_pct": 0.003, "broker_max_drawdown_pct": 0.03}

    assert (
        _validation_gate_reject_reason(candidate, baseline, validation_candidate, validation_baseline)
        == "validation_gate_holdout_net_not_positive"
    )


def test_previous_round_mutations_merge_with_configured_baseline(tmp_path) -> None:
    round4 = tmp_path / "round_4"
    round5 = tmp_path / "round_5"
    round4.mkdir()
    round5.mkdir()
    (round4 / "optimized_config.json").write_text(
        '{"mutations": {"kalcb.entry.min_bar_ret": 0.01}}',
        encoding="utf-8",
    )

    merged = _initial_mutations_for_output(
        {"initial_mutations": {"kalcb.session.flatten_time": "15:15", "kalcb.entry.min_bar_ret": 0.0}},
        round5,
    )

    assert merged["kalcb.session.flatten_time"] == "15:15"
    assert merged["kalcb.entry.min_bar_ret"] == 0.01


def test_previous_round_source_resolves_from_optimized_mutation_contract(tmp_path) -> None:
    round1 = tmp_path / "round_1"
    round2 = tmp_path / "round_2"
    round1.mkdir()
    round2.mkdir()
    source = tmp_path / "candidate_frontier.json"
    source.write_text("{}", encoding="utf-8")
    (round1 / "optimized_config.json").write_text(
        (
            '{"mutations": {'
            f'"{SOURCE_PATH_MUTATION}": "{source.as_posix()}", '
            f'"{SOURCE_SECTION_MUTATION}": "train_ranked", '
            f'"{SOURCE_RANK_MUTATION}": 3'
            "}}"
        ),
        encoding="utf-8",
    )

    resolved = _resolve_fixed_candidate_source({}, round2)

    assert resolved.path == str(source)
    assert resolved.section == "train_ranked"
    assert resolved.rank == 3


def test_previous_guarded_pool_source_is_merged_into_next_round_baseline(tmp_path) -> None:
    round1 = tmp_path / "round_1"
    round2 = tmp_path / "round_2"
    round1.mkdir()
    round2.mkdir()
    pool = round1 / "train_guarded_prefilter_pool_rows.jsonl"
    pool.write_text("", encoding="utf-8")
    (round1 / "optimized_config.json").write_text('{"mutations": {"kalcb.entry.min_bar_ret": 0.01}}', encoding="utf-8")
    (round1 / "full_diagnostics_index.json").write_text(
        (
            '{"artifacts": {"train_guarded_prefilter_pool_rows": "'
            + pool.as_posix()
            + '"}, "source_hashes": {"holdout_pool": {"policy": {"active_count": 16}}}}'
        ),
        encoding="utf-8",
    )
    (round1 / "diagnostics_summary.json").write_text(
        '{"selected_guard": {"label": "stock_sector_daily_ret5_spread>=-0.036622 & sector!=FINANCIAL"}}',
        encoding="utf-8",
    )

    merged = _initial_mutations_for_output({"initial_mutations": {}}, round2)

    assert merged[POOL_SOURCE_PATH_MUTATION] == str(pool)
    assert merged[POOL_SOURCE_ACTIVE_COUNT_MUTATION] == 16
    assert "sector!=FINANCIAL" in merged[POOL_SOURCE_LABEL_MUTATION]


def test_fixed_score_stays_within_component_limit_and_rewards_total_r() -> None:
    baseline = {
        "broker_net_return_pct": 1.20,
        "broker_expected_total_r": 1800.0,
        "trade_count": 205.0,
        "active_days": 60.0,
        "worst_fold_net": 0.08,
        "avg_mfe_capture": 0.46,
        "broker_max_drawdown_pct": 0.055,
        "mae_le_neg_1_share": 0.55,
        "avg_mae_r": -8.0,
    }
    richer = dict(baseline, broker_expected_total_r=2200.0, trade_count=235.0)

    assert len(SCORE_COMPONENTS) <= 7
    assert score_fixed(richer) > score_fixed(baseline)


def test_consolidated_phase_candidates_are_targeted_and_stop_after_phase_6() -> None:
    phase_counts = {phase: len(get_phase_candidates(phase)) for phase in range(1, 7)}

    assert all(count > 0 for count in phase_counts.values())
    assert any("q85" in candidate.name for candidate in get_phase_candidates(1))
    assert any("dailysector" in candidate.name for candidate in get_phase_candidates(1))
    phase1_route = get_phase_candidates(1)[0].mutations["kalcb.entry.routes"][0]
    assert phase1_route["mode"] == "first30_open"
    assert phase1_route["risk_mult"] <= 1.03
    assert phase1_route["context_max"]["frontier_rank"] <= 12
    assert any("cap60" in candidate.name for candidate in get_phase_candidates(2))
    assert any("frontier_pathproof" in candidate.name for candidate in get_phase_candidates(3))
    assert any("conditional_target" in candidate.name for candidate in get_phase_candidates(4))
    assert any("mfe" in candidate.name for candidate in get_phase_candidates(5))
    assert any("combo" in candidate.name for candidate in get_phase_candidates(6))
    phase3_route = get_phase_candidates(3)[0].mutations["kalcb.entry.routes"][1]
    assert phase3_route["risk_mult"] <= 0.04
    assert phase3_route["context_min"]["h6_current_r"] > 0
    assert phase3_route["context_min"]["daily_acceleration_5v20"] > 0
    assert get_phase_candidates(7) == []
    assert get_phase_candidates(11) == []
