from __future__ import annotations

from copy import deepcopy
from typing import Any

from backtests.auto.shared.types import Experiment, GateCriterion

from .phase_scoring import SCORE_WEIGHTS

RISK_STANCE = "controlled_aggressive"
STRATEGY_ORDER: tuple[str, str] = ("kalcb", "olr")

INITIAL_MUTATIONS: dict[str, Any] = {
    "portfolio.risk_stance": RISK_STANCE,
    "portfolio.reference_risk_pct": 0.0055,
    "portfolio.max_daily_gross": 9.99,
    "portfolio.max_sector_gross": 9.99,
    "portfolio.same_symbol_policy": "allow",
    "portfolio.capacity_rank_mode": "chronological",
    "portfolio.olr_close_reserve_gross": 0.0,
    "portfolio.olr_close_reserve_priority_floor": 1.15,
    "portfolio.daily_loss_stop_r": 0.0,
    "portfolio.weekly_loss_stop_r": 0.0,
    "portfolio.dynamic_drawdown_tiers": [
        [0.06, 1.00],
        [0.09, 0.80],
        [0.12, 0.55],
        [0.15, 0.00],
    ],
    "portfolio.use_dynamic_drawdown_tiers": False,
    "portfolio.agreement_boost_mult": 1.00,
    "portfolio.disagreement_haircut_mult": 1.00,
    "portfolio.strategy_r_share_cap": 0.72,
    "portfolio.frequency.enable_olr_shadow_slot6": False,
    "portfolio.frequency.enable_kalcb_secondary": False,
    "kalcb.size_mult": 1.00,
    "kalcb.confirmed_size_mult": 1.00,
    "olr.size_mult": 1.00,
    "olr.confirmed_size_mult": 1.00,
    "blockers.block_olr_after_kalcb_failed_followthrough": False,
    "blockers.block_olr_after_kalcb_quick_exit_r_lt": 0.0,
    "blockers.haircut_olr_after_kalcb_negative_path": 1.00,
    "blockers.boost_olr_after_kalcb_strong_eod": 1.00,
    "blockers.boost_olr_after_kalcb_strong_live_path": 1.00,
    "blockers.kalcb_rank6_10_requires_olr_sector_confirm": False,
    "blockers.kalcb_rank6_10_half_size_confirmed": False,
    "blockers.sector_failure_haircut_mult": 1.00,
    "source.kalcb.round": 3,
    "source.olr.round": 5,
    "source.holdout_excluded": True,
}

PHASE_FOCUS: dict[int, str] = {
    1: "Latest KALCB/OLR artifact baseline and accounting integrity",
    2: "Neutral capacity and low-friction portfolio caps",
    3: "Selective blockers that should block weaker trades first",
    4: "Causal cross-strategy confirmation and sizing overlays",
    5: "Quality-preserving frequency expansion",
    6: "Controlled-aggressive sizing and drawdown governors",
    7: "Robustness, ablation, and final promotion audit",
}

PHASE_GATES: dict[int, dict[str, float]] = {
    1: {
        "min_trades_per_21_sessions": 38.0,
        "max_drawdown_pct": 0.14,
        "min_profit_factor": 1.70,
        "min_strategy_trade_capture": 0.90,
    },
    2: {
        "min_trades_per_21_sessions": 37.0,
        "max_drawdown_pct": 0.14,
        "min_profit_factor": 1.70,
        "max_block_rate": 0.16,
        "max_positive_alpha_block_rate": 0.12,
        "min_strategy_trade_capture": 0.82,
    },
    3: {
        "min_trades_per_21_sessions": 35.0,
        "max_drawdown_pct": 0.14,
        "min_profit_factor": 1.75,
        "max_block_rate": 0.28,
        "max_positive_alpha_block_rate": 0.18,
        "min_strategy_trade_capture": 0.70,
    },
    4: {
        "min_trades_per_21_sessions": 35.0,
        "max_drawdown_pct": 0.14,
        "min_profit_factor": 1.75,
        "max_block_rate": 0.28,
        "max_positive_alpha_block_rate": 0.18,
        "min_strategy_trade_capture": 0.70,
    },
    5: {
        "min_trades_per_21_sessions": 39.0,
        "max_drawdown_pct": 0.145,
        "min_profit_factor": 1.70,
        "max_block_rate": 0.30,
        "max_positive_alpha_block_rate": 0.20,
        "min_strategy_trade_capture": 0.66,
    },
    6: {
        "min_trades_per_21_sessions": 39.0,
        "max_drawdown_pct": 0.14,
        "min_profit_factor": 1.70,
        "max_block_rate": 0.30,
        "max_positive_alpha_block_rate": 0.20,
        "min_strategy_trade_capture": 0.66,
    },
    7: {
        "min_trades_per_21_sessions": 39.0,
        "max_drawdown_pct": 0.14,
        "min_profit_factor": 1.70,
        "max_block_rate": 0.28,
        "max_positive_alpha_block_rate": 0.18,
        "min_strategy_trade_capture": 0.66,
    },
}

ULTIMATE_TARGETS: dict[str, float] = {
    "official_mtm_net_return_pct": 4.40,
    "trades_per_21_sessions": 40.0,
    "profit_factor": 2.20,
    "block_selectivity_edge_r": 0.25,
    "positive_alpha_block_rate": 0.18,
    "max_drawdown_pct": 0.14,
    "min_strategy_trade_capture": 0.66,
}

_PHASE_CANDIDATES: dict[int, list[tuple[str, dict[str, Any]]]] = {
    1: [
        ("p1_equal_risk_baseline", {"portfolio.reference_risk_pct": 0.0055}),
        ("p1_kalcb_olr_55_45_risk", {"kalcb.size_mult": 1.06, "olr.size_mult": 0.94}),
        ("p1_olr_kalcb_55_45_risk", {"kalcb.size_mult": 0.94, "olr.size_mult": 1.06}),
        ("p1_light_agreement_boost", {"portfolio.agreement_boost_mult": 1.08}),
    ],
    2: [
        ("p2_gross_cap_2p10", {"portfolio.max_daily_gross": 2.10}),
        ("p2_gross_cap_2p25", {"portfolio.max_daily_gross": 2.25}),
        ("p2_gross_cap_2p10_expected_alpha_rank", {"portfolio.max_daily_gross": 2.10, "portfolio.capacity_rank_mode": "expected_alpha_density"}),
        ("p2_capacity_rank_expected_alpha_density", {"portfolio.capacity_rank_mode": "expected_alpha_density"}),
        ("p2_olr_close_reserve_035", {"portfolio.olr_close_reserve_gross": 0.35}),
        ("p2_olr_close_reserve_050", {"portfolio.olr_close_reserve_gross": 0.50}),
        ("p2_sector_cap_1p00", {"portfolio.max_sector_gross": 1.00}),
        ("p2_sector_cap_1p10", {"portfolio.max_sector_gross": 1.10}),
        ("p2_same_symbol_half_size", {"portfolio.same_symbol_policy": "half_size"}),
        ("p2_same_symbol_expected_alpha_block", {"portfolio.same_symbol_policy": "block"}),
        ("p2_daily_loss_3p25r", {"portfolio.daily_loss_stop_r": 3.25}),
        ("p2_weekly_loss_8r", {"portfolio.weekly_loss_stop_r": 8.0}),
    ],
    3: [
        ("p3_olr_block_failed_followthrough_same_symbol", {"blockers.block_olr_after_kalcb_failed_followthrough": True}),
        ("p3_olr_block_quick_exit_below_minus2r", {"blockers.block_olr_after_kalcb_quick_exit_r_lt": -2.0}),
        ("p3_olr_haircut_negative_path_50", {"blockers.haircut_olr_after_kalcb_negative_path": 0.50}),
        ("p3_olr_haircut_negative_path_65", {"blockers.haircut_olr_after_kalcb_negative_path": 0.65}),
        ("p3_kalcb_rank6_10_requires_olr_sector_confirm", {"blockers.kalcb_rank6_10_requires_olr_sector_confirm": True}),
        ("p3_kalcb_rank6_10_confirmed_half_size", {"blockers.kalcb_rank6_10_half_size_confirmed": True}),
        ("p3_sector_failure_haircut_65", {"blockers.sector_failure_haircut_mult": 0.65}),
        ("p3_sector_failure_haircut_75", {"blockers.sector_failure_haircut_mult": 0.75}),
    ],
    4: [
        ("p4_olr_boost_after_kalcb_strong_live_path_115", {"blockers.boost_olr_after_kalcb_strong_live_path": 1.15}),
        ("p4_agreement_boost_118", {"portfolio.agreement_boost_mult": 1.18}),
        ("p4_agreement_boost_125_probe", {"portfolio.agreement_boost_mult": 1.25}),
        ("p4_disagreement_haircut_70", {"portfolio.disagreement_haircut_mult": 0.70}),
        ("p4_disagreement_haircut_55", {"portfolio.disagreement_haircut_mult": 0.55}),
        ("p4_confirmed_kalcb_size_112", {"kalcb.confirmed_size_mult": 1.12}),
        ("p4_confirmed_olr_size_112", {"olr.confirmed_size_mult": 1.12}),
    ],
    5: [
        ("p5_olr_slot6_guarded", {"portfolio.frequency.enable_olr_shadow_slot6": True}),
        ("p5_kalcb_secondary_confirmed", {"portfolio.frequency.enable_kalcb_secondary": True}),
        (
            "p5_combined_frequency_frontier_guarded",
            {
                "portfolio.frequency.enable_olr_shadow_slot6": True,
                "portfolio.frequency.enable_kalcb_secondary": True,
                "portfolio.max_daily_gross": 2.30,
                "portfolio.max_sector_gross": 1.10,
                "portfolio.capacity_rank_mode": "expected_alpha_density",
            },
        ),
        (
            "p5_combined_frequency_with_rank_gate",
            {
                "portfolio.frequency.enable_olr_shadow_slot6": True,
                "portfolio.frequency.enable_kalcb_secondary": True,
                "blockers.kalcb_rank6_10_requires_olr_sector_confirm": True,
            },
        ),
    ],
    6: [
        ("p6_reference_risk_60bp", {"portfolio.reference_risk_pct": 0.0060}),
        ("p6_reference_risk_62bp_probe", {"portfolio.reference_risk_pct": 0.0062}),
        ("p6_kalcb_confirmed_size_125", {"kalcb.confirmed_size_mult": 1.25}),
        ("p6_olr_confirmed_size_125", {"olr.confirmed_size_mult": 1.25}),
        ("p6_strategy_r_share_cap_65", {"portfolio.strategy_r_share_cap": 0.65}),
        ("p6_dynamic_drawdown_tiers", {"portfolio.use_dynamic_drawdown_tiers": True}),
        (
            "p6_aggressive_controlled_package",
            {
                "portfolio.reference_risk_pct": 0.0060,
                "portfolio.agreement_boost_mult": 1.20,
                "portfolio.use_dynamic_drawdown_tiers": True,
            },
        ),
    ],
    7: [
        ("p7_cost_stress_guard", {"portfolio.cost_stress_bps": 2.0}),
        ("p7_block_selectivity_guard", {"portfolio.require_block_selectivity_guard": True}),
        ("p7_final_frequency_balance", {"portfolio.max_daily_gross": 2.25, "portfolio.max_sector_gross": 1.10}),
        ("p7_soft_conflict_final", {"portfolio.disagreement_haircut_mult": 0.65}),
    ],
}


def get_phase_candidates(phase: int) -> list[Experiment]:
    return [Experiment(name, deepcopy(mutations)) for name, mutations in _PHASE_CANDIDATES[phase]]


def get_phase_focus(phase: int) -> str:
    return PHASE_FOCUS[phase]


def get_phase_gates(phase: int) -> dict[str, float]:
    return dict(PHASE_GATES[phase])


def get_score_weights() -> dict[str, float]:
    return dict(SCORE_WEIGHTS)


def gate_criteria(metrics: dict[str, Any], phase: int) -> list[GateCriterion]:
    gates = get_phase_gates(phase)
    criteria: list[GateCriterion] = []
    for name, target in gates.items():
        actual = _metric(metrics, _metric_name_for_gate(name))
        if name.startswith("max_"):
            passed = actual <= float(target)
        elif name.startswith("min_"):
            passed = actual >= float(target)
        else:
            passed = actual >= float(target)
        criteria.append(GateCriterion(name, float(target), actual, passed))
    return criteria


def hard_rejects_for_phase(phase: int) -> dict[str, float]:
    gates = get_phase_gates(phase)
    return {
        "max_same_bar_fill_count": 0.0,
        "max_forced_replay_close_count": 0.0,
        "max_rejected_order_count": 0.0,
        "require_accepted_avg_r_gt_blocked_avg_r": 1.0,
        **gates,
    }


def phase_summary() -> list[dict[str, Any]]:
    return [
        {
            "phase": phase,
            "focus": PHASE_FOCUS[phase],
            "candidate_count": len(_PHASE_CANDIDATES[phase]),
            "candidate_names": [name for name, _ in _PHASE_CANDIDATES[phase]],
        }
        for phase in sorted(PHASE_FOCUS)
    ]


def _metric(metrics: dict[str, Any], key: str) -> float:
    try:
        return float(metrics.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _metric_name_for_gate(name: str) -> str:
    aliases = {
        "min_trades_per_21_sessions": "trades_per_21_sessions",
        "min_profit_factor": "profit_factor",
        "max_block_rate": "block_rate",
        "max_positive_alpha_block_rate": "positive_alpha_block_rate",
        "min_strategy_trade_capture": "min_strategy_trade_capture",
    }
    if name in aliases:
        return aliases[name]
    if name.startswith("max_") and name[4:] in {
        "strategy_trade_share",
        "strategy_r_share",
    }:
        return name[4:]
    return name
