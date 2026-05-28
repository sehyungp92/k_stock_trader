from __future__ import annotations

from typing import Any

from backtests.auto.shared.types import Experiment


BASE_MUTATIONS: dict[str, Any] = {}

PHASE_FOCUS: dict[int, tuple[str, list[str]]] = {
    1: (
        "Stage 2 score-band calibration, sector drag rejection, and rank monotonicity",
        [
            "olr_selected_negative_label_share",
            "olr_score_top_bottom_label_spread_pct",
            "olr_score_top_loss_share",
            "olr_discrimination_quality",
            "official_mtm_net_return_pct",
        ],
    ),
    2: (
        "Stage 1 pool quality without starving the downstream selector",
        [
            "selected_candidate_count",
            "selected_avg_mfe_label_pct",
            "olr_selected_positive_label_share",
            "olr_discrimination_quality",
            "official_mtm_net_return_pct",
        ],
    ),
    3: (
        "Post-decision entry timing to improve label-to-fill alpha capture",
        [
            "entry_fill_count",
            "entry_conversion_rate",
            "olr_alpha_capture",
            "olr_positive_mfe_share",
            "official_mtm_net_return_pct",
        ],
    ),
    4: (
        "Managed exit layer for MFE capture and loser containment",
        [
            "mfe_capture",
            "olr_alpha_capture",
            "olr_low_mfe_trade_share",
            "max_drawdown_pct",
            "official_mtm_net_return_pct",
        ],
    ),
    5: (
        "Allocation shape after selector calibration",
        [
            "gross_exposure_avg_pct",
            "total_trades",
            "profit_factor",
            "max_drawdown_pct",
            "official_mtm_net_return_pct",
        ],
    ),
    6: (
        "Integrated alpha-capture candidates with paper/live parity guardrails",
        [
            "official_mtm_net_return_pct",
            "expected_total_r",
            "olr_alpha_capture",
            "olr_discrimination_quality",
            "max_drawdown_pct",
        ],
    ),
}


def get_phase_candidates(phase: int) -> list[Experiment]:
    builders = {
        1: _phase1_stage2_discrimination,
        2: _phase2_stage1_pool_quality,
        3: _phase3_entry_capture,
        4: _phase4_exit_management,
        5: _phase5_allocation_shape,
        6: _phase6_integrated_candidates,
    }
    return _dedupe(builders.get(phase, lambda: [])())


def _phase1_stage2_discrimination() -> list[Experiment]:
    # Round 1's filled holdout had a negative top score quartile and a rank-1
    # loss. These candidates test mid-band caps, exhaustion penalties, and the
    # defense/consumer sector drag without changing the Stage 1 baseline.
    return [
        _exp("s2_midband_score100_650", {"olr.afternoon.min_score": 100.0, "olr.afternoon.max_score": 650.0}),
        _exp("s2_midband_score120_500", {"olr.afternoon.min_score": 120.0, "olr.afternoon.max_score": 500.0}),
        _exp("s2_midband_score140_650_top6", {"olr.afternoon.min_score": 140.0, "olr.afternoon.max_score": 650.0, "olr.afternoon.top_n": 6}),
        _exp("s2_midband_score120_500_top4", {"olr.afternoon.min_score": 120.0, "olr.afternoon.max_score": 500.0, "olr.afternoon.top_n": 4}),
        _exp("s2_exhaustion_pen15_cap225", {"olr.afternoon.score_calibration_mode": "exhaustion_adjusted", "olr.afternoon.exhaustion_penalty": 15.0, "olr.afternoon.max_exhaustion_score": 2.25}),
        _exp("s2_exhaustion_pen25_cap175", {"olr.afternoon.score_calibration_mode": "exhaustion_adjusted", "olr.afternoon.exhaustion_penalty": 25.0, "olr.afternoon.max_exhaustion_score": 1.75}),
        _exp("s2_exhaustion_midband", {"olr.afternoon.score_calibration_mode": "exhaustion_adjusted", "olr.afternoon.exhaustion_penalty": 20.0, "olr.afternoon.max_score": 650.0}),
        _exp("s2_block_defense_consumer", {"olr.afternoon.blocked_sectors": ["DEFENSE", "CONSUMER"]}),
        _exp("s2_block_defense_consumer_battery", {"olr.afternoon.blocked_sectors": ["DEFENSE", "CONSUMER", "BATTERY"]}),
        _exp("s2_semis_chem_elec_allow", {"olr.afternoon.allowed_sectors": ["SEMICONDUCTORS", "CHEMICALS", "ELECTRONICS", "TELECOM", "HEAVY INDUSTRY"]}),
        _exp("s2_flow_sector_foreign", {"olr.afternoon.min_flow_5d": 0.0, "olr.afternoon.min_sector_flow": 0.0, "olr.afternoon.min_foreign_flow_5d": 0.0}),
        _exp("s2_flow_agreement_divergence", {"olr.afternoon.min_flow_agreement": 0.0, "olr.afternoon.max_flow_divergence": 0.01}),
        _exp("s2_close_quality_relvol", {"olr.afternoon.min_rel_volume": 0.75, "olr.afternoon.min_close_location": 0.60, "olr.afternoon.max_open_drawdown": 0.030}),
        _exp("s2_prior5_market45", {"olr.afternoon.min_prior_ret5": 0.03, "olr.afternoon.min_market_score": 45.0}),
        _exp("s2_vwap_strength_midband", {"olr.afternoon.score_mode": "vwap_strength", "olr.afternoon.min_vwap_ret": 0.0, "olr.afternoon.min_close_location": 0.55, "olr.afternoon.max_score": 650.0}),
        _exp("s2_gap_hold_midband", {"olr.afternoon.score_mode": "gap_hold", "olr.afternoon.min_gap": 0.002, "olr.afternoon.max_gap": 0.08, "olr.afternoon.max_score": 650.0}),
        _exp("s2_flow_confirmed_top6", {"olr.afternoon.score_mode": "flow_confirmed", "olr.afternoon.top_n": 6, "olr.afternoon.min_flow_5d": 0.0, "olr.afternoon.min_sector_flow": 0.0}),
        _exp("s2_daily_plus_intraday_top6", {"olr.afternoon.score_mode": "daily_plus_intraday", "olr.afternoon.top_n": 6, "olr.afternoon.max_score": 650.0}),
    ]


def _phase2_stage1_pool_quality() -> list[Experiment]:
    # Stage 1 has real opportunity, but weak ordering. These candidates stay
    # close to the round-1 sector-participation/hot seed while testing whether
    # pool breadth, lagged flow, and structure improve downstream discrimination.
    return [
        _exp("s1_top20_hot_sector50", {"olr.research.top_long_count": 20, "olr.frontier.active_selection_mode": "hot", "olr.research.min_sector_participation": 0.50}),
        _exp("s1_top40_hot_sector50", {"olr.research.top_long_count": 40, "olr.frontier.active_selection_mode": "hot", "olr.research.min_sector_participation": 0.50}),
        _exp("s1_sector60_hot", {"olr.research.min_sector_participation": 0.60, "olr.frontier.active_selection_mode": "hot"}),
        _exp("s1_sector50_score_mode", {"olr.research.min_sector_participation": 0.50, "olr.frontier.active_selection_mode": "score"}),
        _exp("s1_flow_positive", {"olr.research.min_flow_5d": 0.0, "olr.research.min_foreign_flow_5d": 0.0}),
        _exp("s1_sector_flow_positive", {"olr.research.min_sector_flow_5d": 0.0, "olr.research.min_sector_foreign_flow_5d": 0.0}),
        _exp("s1_rs60_sector50", {"olr.research.min_rs_percentile": 60.0, "olr.research.min_sector_participation": 0.50}),
        _exp("s1_trend55_flow", {"olr.research.min_trend_score": 55.0, "olr.research.min_flow_5d": 0.0}),
        _exp("s1_structure025_hot", {"olr.signal.daily_structure_weight": 0.25, "olr.frontier.active_selection_mode": "hot"}),
        _exp("s1_flow_policy_positive_top30", {"olr.signal.flow_policy": "require_positive", "olr.research.top_long_count": 30}),
        _exp("s1_flow_leader_weights", _weights(rs=0.16, trend=0.14, comp=0.06, accum=0.08, stock=0.06, sector=0.08, part=0.10, signal=0.06, flow=0.14, foreign=0.06, inst=0.04, agree=0.02)),
        _exp("s1_sector_rs_weights", _weights(rs=0.28, trend=0.18, comp=0.10, accum=0.10, stock=0.07, sector=0.09, part=0.18, signal=0.06, flow=0.02, foreign=0.01, inst=0.01, agree=0.0)),
    ]


def _phase3_entry_capture() -> list[Experiment]:
    # Entry tax was not the main issue, but 14:40 fills beat 14:35 fills in
    # Round 1. The target is quality-preserving delay/confirmation, not more
    # routes for their own sake.
    return [
        _exp("entry_close_auction", _entry("close_auction", "close_auction")),
        _exp("entry_confirm_b1_ret0_vw0_cl50", _entry("confirm_b1_ret0_vw0_cl50", "confirm_next_bar", max_signal_bars=1, min_bar_ret=0.0, min_vwap_ret=0.0, min_close_location=0.50)),
        _exp("entry_confirm_b2_ret0_vw0_cl60", _entry("confirm_b2_ret0_vw0_cl60", "confirm_next_bar", max_signal_bars=2, min_bar_ret=0.0, min_vwap_ret=0.0, min_close_location=0.60)),
        _exp("entry_confirm_b4_ret0_vw0_cl60", _entry("confirm_b4_ret0_vw0_cl60", "confirm_next_bar", max_signal_bars=4, min_bar_ret=0.0, min_vwap_ret=0.0, min_close_location=0.60)),
        _exp("entry_confirm_b4_ret0_vw0_cl50_above", _entry("confirm_b4_above_decision", "confirm_next_bar", max_signal_bars=4, min_bar_ret=0.0, min_vwap_ret=0.0, min_close_location=0.50, require_above_decision_close=True)),
        _exp("entry_confirm_b6_vwap_cap25", _entry("confirm_b6_vwap_cap25", "confirm_next_bar", max_signal_bars=6, min_bar_ret=-0.001, min_vwap_ret=0.0, min_close_location=0.50, max_vwap_extension_pct=0.025)),
        _exp("entry_late_cont_after1_b4_bo05", _entry("late_cont_after1_b4_bo05", "late_continuation", after_bar=1, max_signal_bars=4, min_breakout_pct=0.0005, min_vwap_ret=0.0, min_close_location=0.50)),
        _exp("entry_late_cont_after2_b8_bo10", _entry("late_cont_after2_b8_bo10", "late_continuation", after_bar=2, max_signal_bars=8, min_breakout_pct=0.001, min_vwap_ret=0.0, min_close_location=0.50)),
        _exp("entry_decision_high_b4_bo10", _entry("decision_high_b4_bo10", "decision_high_breakout", max_signal_bars=4, min_breakout_pct=0.001, min_close_location=0.60)),
        _exp("entry_momentum_breakout_b4_bo10", _entry("momentum_breakout_b4_bo10", "momentum_breakout", max_signal_bars=4, min_breakout_pct=0.001, min_close_location=0.60)),
        _exp("entry_vwap_reclaim_b6_pb4", _entry("vwap_reclaim_b6_pb4", "vwap_reclaim", max_signal_bars=6, max_pullback_from_vwap_pct=0.004, min_reclaim_ret=0.0, min_vwap_ret=0.0, min_close_location=0.50)),
        _exp("entry_pullback_accept_b6_pb8", _entry("pullback_accept_b6_pb8", "pullback_acceptance", max_signal_bars=6, max_pullback_from_vwap_pct=0.008, min_reclaim_ret=0.0, min_vwap_ret=0.0, min_close_location=0.50)),
    ]


def _phase4_exit_management() -> list[Experiment]:
    # Paired replay analysis showed broad hard stops and low-MFE exits rescue
    # some losers but truncate substantially more winner R. Target Phase 4 at
    # high-MFE capture only: no initial hard stop, no early loser churn, and
    # completed-bar MFE fade exits that preserve next-close upside until profit
    # has actually appeared.
    return [
        _exp("exit_mfe_fade1_g125", _exit("mfe_fade1_g125", "managed", hard_stop_enabled=False, mfe_fade_start_r=1.00, mfe_fade_gap_r=1.25, mfe_fade_floor_r=0.0)),
        _exp("exit_mfe_fade1_g150", _exit("mfe_fade1_g150", "managed", hard_stop_enabled=False, mfe_fade_start_r=1.00, mfe_fade_gap_r=1.50, mfe_fade_floor_r=0.0)),
        _exp("exit_mfe_fade125_g125", _exit("mfe_fade125_g125", "managed", hard_stop_enabled=False, mfe_fade_start_r=1.25, mfe_fade_gap_r=1.25, mfe_fade_floor_r=0.0)),
        _exp("exit_mfe_fade125_g150", _exit("mfe_fade125_g150", "managed", hard_stop_enabled=False, mfe_fade_start_r=1.25, mfe_fade_gap_r=1.50, mfe_fade_floor_r=0.0)),
        _exp("exit_mfe_fade15_g125", _exit("mfe_fade15_g125", "managed", hard_stop_enabled=False, mfe_fade_start_r=1.50, mfe_fade_gap_r=1.25, mfe_fade_floor_r=0.0)),
        _exp("exit_mfe_fade15_g150", _exit("mfe_fade15_g150", "managed", hard_stop_enabled=False, mfe_fade_start_r=1.50, mfe_fade_gap_r=1.50, mfe_fade_floor_r=0.0)),
        _exp("exit_mfe_fade2_g125", _exit("mfe_fade2_g125", "managed", hard_stop_enabled=False, mfe_fade_start_r=2.00, mfe_fade_gap_r=1.25, mfe_fade_floor_r=0.0)),
        _exp("exit_mfe_fade2_g150", _exit("mfe_fade2_g150", "managed", hard_stop_enabled=False, mfe_fade_start_r=2.00, mfe_fade_gap_r=1.50, mfe_fade_floor_r=0.0)),
        _exp("exit_mfe_fade25_g125", _exit("mfe_fade25_g125", "managed", hard_stop_enabled=False, mfe_fade_start_r=2.50, mfe_fade_gap_r=1.25, mfe_fade_floor_r=0.0)),
        _exp("exit_mfe_fade3_g075", _exit("mfe_fade3_g075", "managed", hard_stop_enabled=False, mfe_fade_start_r=3.00, mfe_fade_gap_r=0.75, mfe_fade_floor_r=0.0)),
        _exp("exit_target275", _exit("target275", "managed", hard_stop_enabled=False, target_r=2.75)),
        _exp("exit_target4", _exit("target4", "managed", hard_stop_enabled=False, target_r=4.00)),
        _exp("exit_target5", _exit("target5", "managed", hard_stop_enabled=False, target_r=5.00)),
        _exp("exit_fade2_g125_target4", _exit("fade2_g125_target4", "managed", hard_stop_enabled=False, mfe_fade_start_r=2.00, mfe_fade_gap_r=1.25, mfe_fade_floor_r=0.0, target_r=4.00)),
        _exp("exit_fade25_g125_target4", _exit("fade25_g125_target4", "managed", hard_stop_enabled=False, mfe_fade_start_r=2.50, mfe_fade_gap_r=1.25, mfe_fade_floor_r=0.0, target_r=4.00)),
    ]


def _phase5_allocation_shape() -> list[Experiment]:
    # Phase 4 trade slices show rank 1 carries most realized R, ranks 2-3 are
    # useful, and rank 4 is close to flat. Test controlled rank concentration
    # and moderate gross expansion instead of flattening into weak ranks.
    return [
        _exp("alloc_rank_cap050_d125", _alloc("rank_cap050_d125", "rank_weighted", max_position_pct=0.50, rank_decay=1.25)),
        _exp("alloc_rank_cap050_d150", _alloc("rank_cap050_d150", "rank_weighted", max_position_pct=0.50, rank_decay=1.50)),
        _exp("alloc_rank_cap050_d200", _alloc("rank_cap050_d200", "rank_weighted", max_position_pct=0.50, rank_decay=2.00)),
        _exp("alloc_rank_cap055_d125", _alloc("rank_cap055_d125", "rank_weighted", max_position_pct=0.55, rank_decay=1.25)),
        _exp("alloc_rank_cap055_d150", _alloc("rank_cap055_d150", "rank_weighted", max_position_pct=0.55, rank_decay=1.50)),
        _exp("alloc_rank_cap060_d100", _alloc("rank_cap060_d100", "rank_weighted", max_position_pct=0.60, rank_decay=1.00)),
        _exp("alloc_rank_cap060_d125", _alloc("rank_cap060_d125", "rank_weighted", max_position_pct=0.60, rank_decay=1.25)),
        _exp("alloc_rank_cap060_d150", _alloc("rank_cap060_d150", "rank_weighted", max_position_pct=0.60, rank_decay=1.50)),
        _exp("alloc_rank_cap060_d200", _alloc("rank_cap060_d200", "rank_weighted", max_position_pct=0.60, rank_decay=2.00)),
        _exp("alloc_rank_cap065_d150", _alloc("rank_cap065_d150", "rank_weighted", max_position_pct=0.65, rank_decay=1.50)),
        _exp("alloc_rank_cap065_d200", _alloc("rank_cap065_d200", "rank_weighted", max_position_pct=0.65, rank_decay=2.00)),
        _exp("alloc_rank_cap060_g110_d150", _alloc("rank_cap060_g110_d150", "rank_weighted", target_gross_exposure=1.10, max_position_pct=0.60, rank_decay=1.50)),
        _exp("alloc_rank_cap060_g120_d150", _alloc("rank_cap060_g120_d150", "rank_weighted", target_gross_exposure=1.20, max_position_pct=0.60, rank_decay=1.50)),
        _exp("slot3_rank_cap060_d150", {"olr.overnight.slot_count": 3, **_alloc("slot3_rank_cap060_d150", "rank_weighted", max_position_pct=0.60, rank_decay=1.50)}),
        _exp("slot5_rank_cap050_d100", {"olr.overnight.slot_count": 5, **_alloc("slot5_rank_cap050_d100", "rank_weighted", max_position_pct=0.50, rank_decay=1.00)}),
        _exp("slot5_rank_cap055_d150", {"olr.overnight.slot_count": 5, **_alloc("slot5_rank_cap055_d150", "rank_weighted", max_position_pct=0.55, rank_decay=1.50)}),
    ]


def _phase6_integrated_candidates() -> list[Experiment]:
    # Score is U-shaped after the MFE-fade exit: the 400-650 band is weak, but
    # the high-score tail remains valuable. Use the shared selector notch to
    # reject that band, then pair it with slot-5 frequency and the stronger
    # rank-concentrated allocation from Phase 5.
    return [
        _exp("combo_slot5_notch400_650_rank065_g120", {"olr.overnight.slot_count": 5, "olr.afternoon.reject_score_min": 400.0, "olr.afternoon.reject_score_max": 650.0, **_alloc("slot5_notch400_650_rank065_g120", "rank_weighted", target_gross_exposure=1.20, max_position_pct=0.65, rank_decay=1.50)}),
        _exp("combo_slot5_notch400_650_rank070_g110", {"olr.overnight.slot_count": 5, "olr.afternoon.reject_score_min": 400.0, "olr.afternoon.reject_score_max": 650.0, **_alloc("slot5_notch400_650_rank070_g110", "rank_weighted", target_gross_exposure=1.10, max_position_pct=0.70, rank_decay=1.50)}),
        _exp("combo_slot5_notch350_650_rank060_g120", {"olr.overnight.slot_count": 5, "olr.afternoon.reject_score_min": 350.0, "olr.afternoon.reject_score_max": 650.0, **_alloc("slot5_notch350_650_rank060_g120", "rank_weighted", target_gross_exposure=1.20, max_position_pct=0.60, rank_decay=1.50)}),
        _exp("combo_slot5_notch400_650_rank060_g120", {"olr.overnight.slot_count": 5, "olr.afternoon.reject_score_min": 400.0, "olr.afternoon.reject_score_max": 650.0, **_alloc("slot5_notch400_650_rank060_g120", "rank_weighted", target_gross_exposure=1.20, max_position_pct=0.60, rank_decay=1.50)}),
        _exp("combo_slot5_notch400_650_rank065_g110", {"olr.overnight.slot_count": 5, "olr.afternoon.reject_score_min": 400.0, "olr.afternoon.reject_score_max": 650.0, **_alloc("slot5_notch400_650_rank065_g110", "rank_weighted", target_gross_exposure=1.10, max_position_pct=0.65, rank_decay=1.50)}),
        _exp("combo_slot5_notch350_650_rank065_g110", {"olr.overnight.slot_count": 5, "olr.afternoon.reject_score_min": 350.0, "olr.afternoon.reject_score_max": 650.0, **_alloc("slot5_notch350_650_rank065_g110", "rank_weighted", target_gross_exposure=1.10, max_position_pct=0.65, rank_decay=1.50)}),
        _exp("combo_slot5_notch400_650", {"olr.overnight.slot_count": 5, "olr.afternoon.reject_score_min": 400.0, "olr.afternoon.reject_score_max": 650.0}),
        _exp("combo_slot5_notch350_650", {"olr.overnight.slot_count": 5, "olr.afternoon.reject_score_min": 350.0, "olr.afternoon.reject_score_max": 650.0}),
        _exp("combo_notch400_700", {"olr.afternoon.reject_score_min": 400.0, "olr.afternoon.reject_score_max": 700.0}),
        _exp("combo_notch500_700", {"olr.afternoon.reject_score_min": 500.0, "olr.afternoon.reject_score_max": 700.0}),
        _exp("combo_slot5_frequency", {"olr.overnight.slot_count": 5}),
        _exp("combo_block_auto_ent_consumer", {"olr.afternoon.blocked_sectors": ["AUTOMOTIVE", "ENTERTAINMENT", "CONSUMER"]}),
        _exp("combo_slot5_block_auto_ent", {"olr.overnight.slot_count": 5, "olr.afternoon.blocked_sectors": ["AUTOMOTIVE", "ENTERTAINMENT"]}),
    ]


def _entry(name: str, mode: str, **values: Any) -> dict[str, Any]:
    return {"olr.trade_plan.entry": {"name": name, "mode": mode, **values}}


def _exit(name: str, mode: str, **values: Any) -> dict[str, Any]:
    return {"olr.trade_plan.exit": {"name": name, "mode": mode, **values}}


def _alloc(
    name: str,
    mode: str,
    *,
    target_gross_exposure: float = 1.0,
    max_position_pct: float = 0.50,
    min_selected: int = 1,
    rank_decay: float = 1.0,
) -> dict[str, Any]:
    return {
        "olr.allocation.mode": mode,
        "olr.allocation.target_gross_exposure": target_gross_exposure,
        "olr.allocation.max_position_pct": max_position_pct,
        "olr.allocation.min_selected": min_selected,
        "olr.allocation.rank_decay": rank_decay,
    }


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


def _exp(name: str, mutations: dict[str, Any]) -> Experiment:
    return Experiment(name, dict(mutations))


def _dedupe(experiments: list[Experiment]) -> list[Experiment]:
    out: list[Experiment] = []
    seen: set[str] = set()
    for experiment in experiments:
        signature = repr(sorted(dict(experiment.mutations).items(), key=lambda item: str(item[0])))
        if experiment.name in seen or signature in seen:
            continue
        seen.add(experiment.name)
        seen.add(signature)
        out.append(experiment)
    return out
