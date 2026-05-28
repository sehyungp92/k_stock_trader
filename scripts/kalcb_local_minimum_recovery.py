from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backtests.config import load_yaml_config, normalize_runtime_config
from backtests.auto.shared.phase_state import _utc_now_iso
from backtests.strategies.kalcb.fixed_trade_plan_phase import (
    GAP_RETENTION_TRAIN_THRESHOLDS,
    KALCBFixedTradePlanOptimizationPlugin,
    _candidate_frontier_role,
    _candidate_snapshot_metadata,
    _configured_entry_routes,
    _mutation_key,
    _route_candidate_passes,
)
from backtests.strategies.kalcb.shadow_ledger_reranker import (
    LEDGER_CONTEXT_FEATURE_KEYS,
    ROUTE_FAMILIES,
    SHADOW_LEDGER_RERANKER_USAGE_CONTRACT,
    SHADOW_LEDGER_RERANKER_VERSION,
    aggregate_route_shadow_rows,
    build_same_day_reranker_artifacts,
    feature_coverage,
    prepare_shadow_ledger_rows,
    read_jsonl,
    route_family_outcome_coverage,
    write_jsonl,
)
from backtests.strategies.kalcb.candidate_surfacing_recovery import (
    CANDIDATE_SURFACING_RECOVERY_VERSION,
    CANDIDATE_SURFACING_USAGE_CONTRACT,
    build_candidate_surfacing_recovery_artifacts,
)
from backtests.strategies.kalcb.structural_campaign_surfacing import (
    STRUCTURAL_CAMPAIGN_SURFACING_VERSION,
    STRUCTURAL_CAMPAIGN_USAGE_CONTRACT,
    active_budget_by_day_from_context,
    attach_causal_calibration_scores,
    build_alcb_breakout_replay_artifacts,
    build_alcb_faithfulness_funnel_artifacts,
    build_structural_campaign_artifact_rows,
    build_structural_campaign_surfacing_artifacts,
    structural_campaign_feature_artifact_row,
)
from backtests.strategies.kalcb.first30_signal_sweep import (
    build_contexts as build_first30_contexts,
    prepare_first30_dataset,
)
from backtests.strategies.kalcb.trade_plan_sweep import (
    BASELINE_ENTRY_MODE,
    EntrySpec,
    ExitSpec,
    _prior_day_high,
    simulate_trade,
)
from strategy_kalcb.config import KALCBConfig


ROUND_ROOT = REPO_ROOT / "data" / "backtests" / "output" / "kalcb"
CONFIG_PATH = REPO_ROOT / "config" / "optimization" / "kalcb.yaml"
DEFAULT_OUT_DIR = ROUND_ROOT / "round_5" / "local_minimum_recovery"

METRIC_KEYS = (
    "broker_net_return_pct",
    "official_mtm_net_return_pct",
    "broker_max_drawdown_pct",
    "trade_count",
    "active_days",
    "active_budget_candidate_count",
    "initial_active_candidate_count",
    "candidate_pool_count",
    "full_candidate_pool_count",
    "static_route_eligible_count",
    "avg_trade_net_pct",
    "avg_mfe_capture",
    "avg_mfe_r",
    "avg_mae_r",
    "mae_le_neg_1_share",
    "worst_fold_net",
    "median_fold_net",
    "same_bar_fill_count",
    "forced_replay_close_count",
    "rejected_order_count",
    "end_open_position_count",
)

EXECUTION_SEQUENCE = (
    "cumulative_mutation_ablation",
    "route_family_perturbation",
    "shadow_opportunity_ledger",
    "shadow_same_day_reranker",
    "full_universe_missed_opportunity_oracle",
    "next_round_seed_and_path_state_exits",
    "causal_candidate_surfacing_recovery",
    "structural_campaign_surfacing",
    "targeted_indicator_additions",
    "structured_auto_rounds",
)

NEXT_ROUND_SEED = "auto_pullback_q85_rank8_r0p015"
NEXT_ROUND_CHALLENGER = "auto_pullback_q85_rank8_r0p02_target60"


@dataclass(frozen=True)
class RecoveryExperiment:
    stage: str
    family: str
    name: str
    purpose: str
    mutations: dict[str, Any]
    replace_base: bool = False

    def materialize(self, base: dict[str, Any]) -> dict[str, Any]:
        if self.replace_base:
            return dict(self.mutations)
        out = dict(base)
        out.update(dict(self.mutations))
        return out


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _num(value: Any) -> float:
    try:
        out = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def _pct(value: Any) -> str:
    return f"{100.0 * _num(value):.2f}%"


def _clean_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics.get(key) for key in METRIC_KEYS if key in metrics}


def _metric_delta(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {key: _num(metrics.get(key)) - _num(baseline.get(key)) for key in METRIC_KEYS}


def _load_round_mutations(round_num: int) -> dict[str, Any]:
    path = ROUND_ROOT / f"round_{round_num}" / "optimized_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing optimized config for round {round_num}: {path}")
    payload = _read_json(path)
    mutations = payload.get("mutations")
    if not isinstance(mutations, dict):
        raise ValueError(f"optimized_config.json for round {round_num} does not contain mutations")
    return dict(mutations)


def _round_source_ref(round_num: int) -> dict[str, Any]:
    summary_path = ROUND_ROOT / f"round_{round_num}" / "diagnostics_summary.json"
    if summary_path.exists():
        source = dict((_read_json(summary_path).get("source") or {}))
    else:
        source = {}
    mutations = _load_round_mutations(round_num)
    return {
        "path": str(source.get("path") or mutations.get("_kalcb.source.path") or ""),
        "section": str(source.get("section") or mutations.get("_kalcb.source.section") or "top_portfolio_proxy"),
        "rank": int(source.get("rank") if source.get("rank") is not None else mutations.get("_kalcb.source.rank") or 0),
        "row_name": str(source.get("row_name") or ""),
    }


def _quality_vote_off() -> dict[str, Any]:
    return {
        "kalcb.entry.min_quality_votes": 0,
        "kalcb.entry.quality_min_bar_ret": -9.99,
        "kalcb.entry.quality_min_first30_signal_cpr": -9.99,
        "kalcb.entry.quality_min_first30_rel_volume": -9.99,
        "kalcb.entry.quality_min_first30_range_atr": -9.99,
        "kalcb.entry.quality_max_first30_range_atr": 0.0,
        "kalcb.entry.quality_min_flow_score": -9.99,
        "kalcb.entry.quality_min_accumulation_score": -9.99,
        "kalcb.entry.quality_max_frontier_rank": 0,
    }


def _path_quality_off() -> dict[str, Any]:
    return {
        "kalcb.exit.path_quality_enabled": False,
        "kalcb.exit.path_quality_min_hold_bars": 0,
        "kalcb.exit.path_quality_min_mfe_r": 0.0,
        "kalcb.exit.path_quality_min_giveback_r": 0.0,
        "kalcb.exit.path_quality.context_min": {},
        "kalcb.exit.path_quality.context_max": {},
        "kalcb.exit.path_quality_entry_route_modes": [],
    }


def _first30_anchor(priority: int = 100, risk_mult: float = 0.99) -> dict[str, Any]:
    return {
        "name": "first30_open_anchor",
        "mode": "first30_open",
        "priority": int(priority),
        "require_initial_active": True,
        "risk_mult": float(risk_mult),
        "notional_mult": float(risk_mult),
    }


def _shadow_route(
    *,
    name: str,
    mode: str,
    rank: int,
    risk_mult: float,
    max_session_trades: int,
    context_min: dict[str, float] | None = None,
    quality_votes: int = 6,
    cpr: float = 0.75,
    relvol: float = 2.0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    route: dict[str, Any] = {
        "name": name,
        "mode": mode,
        "priority": 0,
        "require_initial_active": False,
        "max_frontier_rank": int(rank),
        "max_session_trades": int(max_session_trades),
        "risk_mult": float(risk_mult),
        "notional_mult": float(risk_mult),
        "min_bar_ret": 0.0 if mode != "first30_open" else 0.01,
        "min_vwap_ret": 0.0,
        "min_quality_votes": int(quality_votes),
        "quality_min_first30_signal_cpr": float(cpr),
        "quality_min_first30_rel_volume": float(relvol),
        "quality_min_accumulation_score": -0.05,
    }
    if context_min:
        route["context_min"] = dict(context_min)
    if mode == "or_high_reclaim":
        route.update({"after_bar": 1, "max_signal_bars": 15, "min_reclaim_ret": 0.0})
    elif mode == "avwap_reclaim":
        route.update({"after_bar": 1, "max_signal_bars": 18, "min_reclaim_ret": 0.001, "max_pullback_from_vwap_pct": 0.006})
    elif mode == "pullback_acceptance":
        route.update({"after_bar": 1, "max_signal_bars": 18, "min_reclaim_ret": 0.0005, "max_pullback_from_vwap_pct": 0.008})
    elif mode == "deferred_continuation":
        route.update({"after_bar": 3, "max_signal_bars": 24, "min_breakout_pct": 0.0005, "min_close_location": 0.60})
    if extra:
        route.update(dict(extra))
    return route


def _route_plan(*routes: dict[str, Any], fallback_risk: float = 0.99) -> dict[str, Any]:
    return {
        "kalcb.entry.frontier_branch_universe": True,
        "kalcb.entry.routes": [*(dict(route) for route in routes), _first30_anchor(risk_mult=fallback_risk)],
    }


def build_ablation_experiments(rounds: dict[int, dict[str, Any]], base: dict[str, Any]) -> list[RecoveryExperiment]:
    exps: list[RecoveryExperiment] = []
    for round_num in (2, 3, 4):
        exps.append(
            RecoveryExperiment(
                "ablation",
                "round_rollback",
                f"rollback_full_round{round_num}",
                f"Replace the cumulative round-5 stack with round {round_num} optimized mutations.",
                rounds[round_num],
                replace_base=True,
            )
        )

    exps.extend(
        [
            RecoveryExperiment("ablation", "route_overlay", "remove_route_overlay", "Revert round-5 route overlay to the baseline first30 plan.", {"kalcb.entry.routes": []}),
            RecoveryExperiment("ablation", "path_quality", "remove_path_quality", "Disable the accepted round-5 path-quality exit.", _path_quality_off()),
            RecoveryExperiment(
                "ablation",
                "quality_stack",
                "remove_quality_vote_stack",
                "Remove round-4 quality vote stack and restore pure first30 filter surface.",
                _quality_vote_off(),
            ),
            RecoveryExperiment(
                "ablation",
                "quality_stack",
                "relax_quality_vote_5",
                "Relax quality-vote minimum from 6 to 5 while preserving the same dimensions.",
                {"kalcb.entry.min_quality_votes": 5},
            ),
            RecoveryExperiment(
                "ablation",
                "quality_stack",
                "tighten_quality_vote_7",
                "Tighten quality votes to test whether the current stack is still too permissive.",
                {"kalcb.entry.min_quality_votes": 7},
            ),
            RecoveryExperiment("ablation", "entry_floor", "min_bar_ret_0000", "Remove first30 return floor.", {"kalcb.entry.min_bar_ret": 0.0}),
            RecoveryExperiment("ablation", "entry_floor", "min_bar_ret_0005", "Restore round-3 first30 return floor.", {"kalcb.entry.min_bar_ret": 0.005}),
            RecoveryExperiment("ablation", "entry_floor", "min_bar_ret_0075", "Perturb first30 return floor between round 3 and 4.", {"kalcb.entry.min_bar_ret": 0.0075}),
            RecoveryExperiment("ablation", "entry_floor", "min_bar_ret_0125", "Tighten first30 return floor to test selectivity sensitivity.", {"kalcb.entry.min_bar_ret": 0.0125}),
            RecoveryExperiment("ablation", "target", "target_off", "Remove high-extension target and isolate EOD/failed-followthrough management.", {"kalcb.exit.target_r": 0.0}),
            RecoveryExperiment("ablation", "target", "target_36r", "Restore round-3 target.", {"kalcb.exit.target_r": 36.0}),
            RecoveryExperiment("ablation", "target", "target_45r", "Perturb target below round-5 final.", {"kalcb.exit.target_r": 45.0}),
            RecoveryExperiment("ablation", "target", "target_60r", "Perturb target near train-positive but holdout-fragile surface.", {"kalcb.exit.target_r": 60.0}),
            RecoveryExperiment(
                "ablation",
                "failed_followthrough",
                "failed_followthrough_off",
                "Disable failed-followthrough to test whether it cuts recoverable paths.",
                {"kalcb.exit.failed_followthrough_bars": 0, "kalcb.exit.failed_followthrough_mfe_r": 0.0, "kalcb.exit.failed_followthrough_close_r": 0.0},
            ),
            RecoveryExperiment(
                "ablation",
                "failed_followthrough",
                "failed_followthrough_round2",
                "Restore round-2 failed-followthrough.",
                {"kalcb.exit.failed_followthrough_bars": 8, "kalcb.exit.failed_followthrough_mfe_r": 1.0, "kalcb.exit.failed_followthrough_close_r": -0.5},
            ),
            RecoveryExperiment(
                "ablation",
                "failed_followthrough",
                "failed_followthrough_round3",
                "Restore round-3 failed-followthrough.",
                {"kalcb.exit.failed_followthrough_bars": 10, "kalcb.exit.failed_followthrough_mfe_r": 1.25, "kalcb.exit.failed_followthrough_close_r": -0.25},
            ),
            RecoveryExperiment("ablation", "risk", "risk_round2_cap40_r065", "Restore round-2 risk cap and risk per trade.", {"kalcb.risk.max_position_notional_pct": 0.40, "kalcb.risk.risk_per_trade_pct": 0.0065}),
            RecoveryExperiment("ablation", "risk", "risk_round3_cap50_r055", "Restore round-3/4 risk cap and risk per trade.", {"kalcb.risk.max_position_notional_pct": 0.50, "kalcb.risk.risk_per_trade_pct": 0.0055}),
            RecoveryExperiment("ablation", "risk", "risk_cap55_r050", "Test intermediate notional expansion with lower risk per trade.", {"kalcb.risk.max_position_notional_pct": 0.55, "kalcb.risk.risk_per_trade_pct": 0.0050}),
            RecoveryExperiment("ablation", "risk", "risk_cap60_r045", "Retest positive-both cap60 surface under explicit gates.", {"kalcb.risk.max_position_notional_pct": 0.60, "kalcb.risk.risk_per_trade_pct": 0.0045}),
        ]
    )

    round3_frequency = dict(base)
    round3_frequency.update(
        {
            "kalcb.entry.min_bar_ret": 0.005,
            "kalcb.exit.target_r": 36.0,
            "kalcb.entry.routes": [],
            **_quality_vote_off(),
            **_path_quality_off(),
        }
    )
    exps.append(
        RecoveryExperiment(
            "ablation",
            "cumulative_combo",
            "restore_round3_frequency_surface_keep_round5_source",
            "Keep current source/risk context but restore the broader round-3 entry/target surface.",
            round3_frequency,
            replace_base=True,
        )
    )
    return exps


def build_route_family_experiments(base: dict[str, Any]) -> list[RecoveryExperiment]:
    del base
    t = GAP_RETENTION_TRAIN_THRESHOLDS
    relvol_q85 = {"first30_rel_volume": t["first30_rel_volume_q85"]}
    lowprev_q85 = {"first30_rel_volume": t["first30_rel_volume_q85"], "first30_low_vs_prev_close": t["first30_low_vs_prev_close_q85"]}
    sector_q75 = {"first30_rel_volume": t["first30_rel_volume_q85"], "sector_daily_score_pct": t["sector_daily_score_pct_q75"]}
    return [
        RecoveryExperiment(
            "routes",
            "first30_shadow",
            "shadow_first30_relvol_q85_rank8_cap1_r003",
            "Narrow high-relvol rank<=8 shadow first30 route, capped at one trade per route/session.",
            _route_plan(_shadow_route(name="shadow_first30_relvol_q85_rank8", mode="first30_open", rank=8, risk_mult=0.03, max_session_trades=1, context_min=relvol_q85)),
        ),
        RecoveryExperiment(
            "routes",
            "first30_shadow",
            "shadow_first30_relvol_q85_rank5_cap1_r003",
            "Same high-relvol branch but only rank<=5.",
            _route_plan(_shadow_route(name="shadow_first30_relvol_q85_rank5", mode="first30_open", rank=5, risk_mult=0.03, max_session_trades=1, context_min=relvol_q85)),
        ),
        RecoveryExperiment(
            "routes",
            "first30_shadow",
            "shadow_first30_lowprev_relvol_q85_rank8_cap1_r003",
            "Require both high relvol and gap-retention low-vs-prev proof.",
            _route_plan(_shadow_route(name="shadow_first30_lowprev_relvol_rank8", mode="first30_open", rank=8, risk_mult=0.03, max_session_trades=1, context_min=lowprev_q85)),
        ),
        RecoveryExperiment(
            "routes",
            "first30_shadow",
            "shadow_first30_sector_relvol_q85_rank8_cap1_r003",
            "Require high relvol plus strong daily sector score.",
            _route_plan(_shadow_route(name="shadow_first30_sector_relvol_rank8", mode="first30_open", rank=8, risk_mult=0.03, max_session_trades=1, context_min=sector_q75)),
        ),
        RecoveryExperiment(
            "routes",
            "or_high_reclaim",
            "shadow_or_high_reclaim_relvol_q85_rank8_cap1_r002",
            "Delayed OR-high reclaim branch for non-initial active high-relvol names.",
            _route_plan(_shadow_route(name="shadow_or_high_reclaim_rank8", mode="or_high_reclaim", rank=8, risk_mult=0.02, max_session_trades=1, context_min=relvol_q85)),
        ),
        RecoveryExperiment(
            "routes",
            "avwap_reclaim",
            "shadow_avwap_reclaim_relvol_q85_rank8_cap1_r002",
            "Delayed AVWAP reclaim branch for non-initial active high-relvol names.",
            _route_plan(_shadow_route(name="shadow_avwap_reclaim_rank8", mode="avwap_reclaim", rank=8, risk_mult=0.02, max_session_trades=1, context_min=relvol_q85)),
        ),
        RecoveryExperiment(
            "routes",
            "pullback_acceptance",
            "shadow_pullback_acceptance_relvol_q85_rank8_cap1_r002",
            "Pullback acceptance branch for high-relvol top-rank shadows.",
            _route_plan(_shadow_route(name="shadow_pullback_acceptance_rank8", mode="pullback_acceptance", rank=8, risk_mult=0.02, max_session_trades=1, context_min=relvol_q85)),
        ),
        RecoveryExperiment(
            "routes",
            "deferred_continuation",
            "shadow_deferred_pathproof_relvol_rank12_cap1_r002",
            "Deferred path-proof branch; used only after the first30 thrust keeps working intraday.",
            _route_plan(
                _shadow_route(
                    name="shadow_deferred_pathproof_rank12",
                    mode="deferred_continuation",
                    rank=12,
                    risk_mult=0.02,
                    max_session_trades=1,
                    context_min={
                        "first30_rel_volume": t["first30_rel_volume_q85"],
                        "h3_current_r": t["h3_current_r_q65"],
                        "h6_current_r": t["h6_current_r_q65"],
                    },
                    quality_votes=5,
                    cpr=0.70,
                    relvol=1.25,
                )
            ),
        ),
    ]


def build_indicator_experiments() -> list[RecoveryExperiment]:
    t = GAP_RETENTION_TRAIN_THRESHOLDS
    return [
        RecoveryExperiment(
            "indicators",
            "path_risk",
            "shadow_pathrisk_h3_h6_relvol_rank8",
            "If shadow names show MFE but high MAE, admit only those with early h3/h6 path proof.",
            _route_plan(
                _shadow_route(
                    name="shadow_pathrisk_h3_h6_rank8",
                    mode="deferred_continuation",
                    rank=8,
                    risk_mult=0.02,
                    max_session_trades=1,
                    context_min={
                        "first30_rel_volume": t["first30_rel_volume_q85"],
                        "h3_current_r": t["h3_current_r_q65"],
                        "h6_current_r": t["h6_current_r_q65"],
                    },
                    quality_votes=5,
                    cpr=0.70,
                    relvol=1.25,
                )
            ),
        ),
        RecoveryExperiment(
            "indicators",
            "sector_confirmation",
            "shadow_sector_daily_relvol_rank8",
            "If failures concentrate by sector, require strong daily sector participation before shadow admission.",
            _route_plan(
                _shadow_route(
                    name="shadow_sector_daily_rank8",
                    mode="first30_open",
                    rank=8,
                    risk_mult=0.03,
                    max_session_trades=1,
                    context_min={
                        "first30_rel_volume": t["first30_rel_volume_q85"],
                        "sector_daily_score_pct": t["sector_daily_score_pct_q75"],
                    },
                )
            ),
        ),
        RecoveryExperiment(
            "indicators",
            "exit_path_state",
            "path_quality_tighter_below_or_high2",
            "If MFE leakage dominates, test path-state exit conditioning rather than another global target.",
            {
                "kalcb.exit.path_quality_enabled": True,
                "kalcb.exit.path_quality_min_hold_bars": 30,
                "kalcb.exit.path_quality_min_mfe_r": 6.0,
                "kalcb.exit.path_quality_min_giveback_r": 4.0,
                "kalcb.exit.path_quality.context_min": {"below_or_high_streak": 2.0},
                "kalcb.exit.path_quality_entry_route_modes": ["first30_open"],
            },
        ),
    ]


def build_structured_auto_experiments(base: dict[str, Any]) -> list[RecoveryExperiment]:
    del base
    t = GAP_RETENTION_TRAIN_THRESHOLDS
    relvol_q85 = {"first30_rel_volume": t["first30_rel_volume_q85"]}
    exps: list[RecoveryExperiment] = []
    for mode, family, label in (
        ("pullback_acceptance", "pullback_acceptance", "pullback"),
        ("avwap_reclaim", "avwap_reclaim", "avwap"),
    ):
        for risk in (0.015, 0.02, 0.025):
            risk_label = str(risk).replace(".", "p")
            exps.append(
                RecoveryExperiment(
                    "auto",
                    family,
                    f"auto_{label}_q85_rank8_r{risk_label}",
                    "Structured route-timing grid around the route-family survivors.",
                    _route_plan(_shadow_route(name=f"auto_{label}_rank8_r{risk_label}", mode=mode, rank=8, risk_mult=risk, max_session_trades=1, context_min=relvol_q85)),
                )
            )
        for rank in (5, 10):
            exps.append(
                RecoveryExperiment(
                    "auto",
                    family,
                    f"auto_{label}_q85_rank{rank}_r0p02",
                    "Rank-cap perturbation for the route-family survivor.",
                    _route_plan(_shadow_route(name=f"auto_{label}_rank{rank}_r0p02", mode=mode, rank=rank, risk_mult=0.02, max_session_trades=1, context_min=relvol_q85)),
                )
            )
        exps.append(
            RecoveryExperiment(
                "auto",
                f"{family}_target",
                f"auto_{label}_q85_rank8_r0p02_target60",
                "Route survivor with target-60 overlay from ablation leads.",
                {**_route_plan(_shadow_route(name=f"auto_{label}_rank8_r0p02_target60", mode=mode, rank=8, risk_mult=0.02, max_session_trades=1, context_min=relvol_q85)), "kalcb.exit.target_r": 60.0},
            )
        )
    exps.append(
        RecoveryExperiment(
            "auto",
            "route_combo",
            "auto_combo_pullback_avwap_q85_rank8_r0p015_each",
            "Two-route combination with half-step risk to test route complementarity without broadening frontier eligibility.",
            _route_plan(
                _shadow_route(name="auto_combo_pullback_rank8_r0p015", mode="pullback_acceptance", rank=8, risk_mult=0.015, max_session_trades=1, context_min=relvol_q85, extra={"priority": 0}),
                _shadow_route(name="auto_combo_avwap_rank8_r0p015", mode="avwap_reclaim", rank=8, risk_mult=0.015, max_session_trades=1, context_min=relvol_q85, extra={"priority": 1}),
            ),
        )
    )
    exps.append(
        RecoveryExperiment(
            "auto",
            "or_high_reclaim",
            "auto_or_high_q85_rank8_r0p02_target60",
            "Secondary OR-high reclaim branch with the target-60 overlay.",
            {**_route_plan(_shadow_route(name="auto_or_high_rank8_r0p02_target60", mode="or_high_reclaim", rank=8, risk_mult=0.02, max_session_trades=1, context_min=relvol_q85)), "kalcb.exit.target_r": 60.0},
        )
    )
    return exps


def structured_auto_round_plan() -> dict[str, Any]:
    return {
        "objective": "escape the round2-5 local minimum by making frequency and executable opportunity first-class objectives",
        "primary_gates": {
            "train_min_trades": 105,
            "train_min_active_days": 65,
            "train_max_drawdown_pct": 0.08,
            "holdout_net_delta_floor": -0.005,
            "holdout_max_drawdown_pct": 0.08,
            "route_eligible_conversion_min": 0.06,
            "same_bar_fill_count": 0,
            "end_open_position_count": 0,
        },
        "phases": [
            {"phase": 1, "name": "cumulative_ablation", "goal": "identify which round2-5 mutation groups created the narrow basin"},
            {"phase": 2, "name": "route_family_perturbation", "goal": "admit narrow high-conviction shadow routes with per-session caps"},
            {"phase": 3, "name": "shadow_ledger_attribution", "goal": "rank blocked candidates by causal path MFE/MAE and same-day opportunity cost"},
            {"phase": 4, "name": "targeted_indicators", "goal": "add only indicators that explain ablation/ledger failures"},
            {"phase": 5, "name": "path_management", "goal": "condition exits on path-state leakage, not global target sweeps"},
            {"phase": 6, "name": "risk_and_capacity", "goal": "resize only survivor routes and keep drawdown under the hard ceiling"},
            {"phase": 7, "name": "locked_holdout_and_parity", "goal": "require untouched holdout survival and paper/live parity artifacts"},
        ],
    }


def shadow_ledger_plan() -> dict[str, Any]:
    return {
        "objective": "attribute untraded frontier candidates to executable route blockage, path quality, daily/sector context, and same-day portfolio opportunity cost",
        "probe_routes": [spec["route"]["name"] for spec in _shadow_ledger_probe_specs()],
        "route_families": list(ROUTE_FAMILIES),
        "windows": ["train", "holdout"],
        "row_keys": [
            "blocked_reason",
            "route_eligible",
            "route_block_reasons",
            "route_outcomes",
            "best_route_shadow_total_r",
            "best_route_shadow_max_mfe_r",
            "best_route_shadow_min_mae_r",
            "same_day_replacement_value_r",
            "marginal_slot_replacement_value_r",
            "same_day_improver",
            "sector",
            "sector_daily_score_pct",
            "sector_intraday_score_pct",
            "first30_rel_volume",
            "first30_signal_bar_cpr",
            "first30_range_atr",
        ],
        "outputs": [
            "shadow_opportunity_ledger_train.jsonl",
            "shadow_opportunity_ledger_holdout.jsonl",
            "06_same_day_reranker/shadow_same_day_reranker_train.jsonl",
            "06_same_day_reranker/shadow_same_day_reranker_holdout.jsonl",
            "06_same_day_reranker/shadow_same_day_reranker_summary.json",
            "06_same_day_reranker/shadow_same_day_reranker_report.md",
        ],
        "usage_contract": SHADOW_LEDGER_RERANKER_USAGE_CONTRACT,
    }


class RecoveryEvaluator:
    def __init__(self, config_path: Path, output_dir: Path, source_ref: dict[str, Any], *, max_workers: int = 1):
        self.progress_path = output_dir / "progress.jsonl"
        config = normalize_runtime_config("kalcb", load_yaml_config(config_path))
        config["fixed_trade_plan_phase_auto"] = True
        config["skip_initial_baseline_eval"] = True
        config["force_rebuild_cache"] = False
        config["fixed_candidate_source"] = dict(source_ref)
        self.log("context_init_start", output_dir=str(output_dir / "train_replay"), source_ref=source_ref)
        self.plugin = KALCBFixedTradePlanOptimizationPlugin(
            config,
            output_dir=output_dir / "train_replay",
            max_workers=max_workers,
            capability_level="real_replay",
        )
        self.log(
            "context_init_done",
            train_sessions=len(self.plugin.context.train_dates),
            source_fingerprint=self.plugin.source_fingerprint,
        )

    def log(self, event: str, **extra: Any) -> None:
        payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, **extra}
        print(json.dumps(payload, sort_keys=True, default=str), flush=True)
        _append_jsonl(self.progress_path, payload)

    def evaluate_pair(self, mutations: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        train = self.plugin.evaluate_mutations(mutations)
        holdout = self.plugin.evaluate_validation_mutations(mutations)
        return train, holdout


def _recovery_score(train: dict[str, Any], holdout: dict[str, Any], base_train: dict[str, Any], base_holdout: dict[str, Any]) -> dict[str, Any]:
    train_delta = _metric_delta(train, base_train)
    holdout_delta = _metric_delta(holdout, base_holdout)
    trade_target = 105.0
    train_trades = _num(train.get("trade_count"))
    holdout_trades = _num(holdout.get("trade_count"))
    frequency = min(train_trades / trade_target, 1.25)
    train_net = train_delta["broker_net_return_pct"]
    holdout_net = holdout_delta["broker_net_return_pct"]
    dd_penalty = max(_num(train.get("broker_max_drawdown_pct")) - 0.08, 0.0) * 8.0
    holdout_dd_penalty = max(_num(holdout.get("broker_max_drawdown_pct")) - 0.08, 0.0) * 8.0
    score = 100.0 * (0.35 * train_net + 0.25 * holdout_net + 0.20 * (frequency - 1.0) + 0.10 * train_delta["avg_mfe_capture"] + 0.10 * holdout_delta["avg_mfe_capture"] - dd_penalty - holdout_dd_penalty)
    pass_research = (
        _num(train.get("same_bar_fill_count")) == 0.0
        and _num(train.get("end_open_position_count")) == 0.0
        and _num(holdout.get("same_bar_fill_count")) == 0.0
        and _num(holdout.get("end_open_position_count")) == 0.0
        and _num(train.get("broker_max_drawdown_pct")) <= 0.08
        and _num(holdout.get("broker_max_drawdown_pct")) <= 0.08
        and holdout_net >= -0.005
        and train_trades >= 0.80 * _num(base_train.get("trade_count"))
        and holdout_trades >= max(8.0, 0.70 * _num(base_holdout.get("trade_count")))
    )
    return {
        "score": score,
        "research_survivor": pass_research,
        "train_delta": train_delta,
        "holdout_delta": holdout_delta,
        "frequency_ratio_to_105": frequency,
    }


def run_evaluations(
    evaluator: RecoveryEvaluator,
    base: dict[str, Any],
    experiments: list[RecoveryExperiment],
    *,
    max_evals: int | None,
) -> dict[str, Any]:
    evaluator.log("baseline_start")
    base_train, base_holdout = evaluator.evaluate_pair(base)
    evaluator.log("baseline_done", train_net=base_train.get("broker_net_return_pct"), holdout_net=base_holdout.get("broker_net_return_pct"))
    rows: list[dict[str, Any]] = []
    selected = experiments[: max_evals if max_evals is not None else len(experiments)]
    for index, experiment in enumerate(selected, start=1):
        mutations = experiment.materialize(base)
        evaluator.log("candidate_start", index=index, total=len(selected), stage=experiment.stage, name=experiment.name)
        started = time.monotonic()
        train, holdout = evaluator.evaluate_pair(mutations)
        score = _recovery_score(train, holdout, base_train, base_holdout)
        rows.append(
            {
                **asdict(experiment),
                "mutation_count": len(mutations),
                "mutation_hash": _mutation_key(mutations),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "train": _clean_metrics(train),
                "holdout": _clean_metrics(holdout),
                **score,
            }
        )
        evaluator.log("candidate_done", name=experiment.name, score=score["score"], survivor=score["research_survivor"])
    return {
        "baseline": {"train": _clean_metrics(base_train), "holdout": _clean_metrics(base_holdout)},
        "rows": rows,
        "survivors": [row for row in rows if row.get("research_survivor")],
    }


def _shadow_ledger_probe_specs() -> list[dict[str, Any]]:
    t = GAP_RETENTION_TRAIN_THRESHOLDS
    relvol_q85 = {"first30_rel_volume": t["first30_rel_volume_q85"]}
    return [
        {
            "route_family": "first30_open",
            "route": _shadow_route(
                name="ledger_first30_open_relvol_q85_rank8",
                mode="first30_open",
                rank=8,
                risk_mult=1.0,
                max_session_trades=0,
                context_min=relvol_q85,
                quality_votes=6,
                cpr=0.75,
                relvol=2.0,
            ),
        },
        {
            "route_family": "pullback_acceptance",
            "route": _shadow_route(
                name="ledger_pullback_acceptance_relvol_q85_rank8",
                mode="pullback_acceptance",
                rank=8,
                risk_mult=1.0,
                max_session_trades=0,
                context_min=relvol_q85,
                quality_votes=6,
                cpr=0.75,
                relvol=2.0,
            ),
        },
        {
            "route_family": "avwap_reclaim",
            "route": _shadow_route(
                name="ledger_avwap_reclaim_relvol_q85_rank8",
                mode="avwap_reclaim",
                rank=8,
                risk_mult=1.0,
                max_session_trades=0,
                context_min=relvol_q85,
                quality_votes=6,
                cpr=0.75,
                relvol=2.0,
            ),
        },
        {
            "route_family": "or_high_reclaim",
            "route": _shadow_route(
                name="ledger_or_high_reclaim_relvol_q85_rank8",
                mode="or_high_reclaim",
                rank=8,
                risk_mult=1.0,
                max_session_trades=0,
                context_min=relvol_q85,
                quality_votes=6,
                cpr=0.75,
                relvol=2.0,
            ),
        },
    ]


def _shadow_ledger_probe_mutations(base: dict[str, Any], route: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    out.update(
        {
            "kalcb.frontier.shadow_enabled": True,
            "kalcb.frontier.shadow_max_positions": 128,
            "kalcb.entry.frontier_branch_universe": True,
            "kalcb.entry.routes": [dict(route)],
        }
    )
    return out


def _realized_trade_context(detail_rows: tuple[dict[str, Any], ...]) -> tuple[set[tuple[str, str]], dict[str, dict[str, Any]]]:
    realized: set[tuple[str, str]] = set()
    by_day: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total_r": 0.0, "trade_count": 0, "weakest_r": None, "sector_counts": Counter()}
    )
    for row in detail_rows:
        day = str(row.get("entry_date") or "")[:10]
        symbol = str(row.get("symbol") or "")
        realized.add((day, symbol))
        r_value = _num(row.get("r"))
        sector = str(row.get("sector") or "UNKNOWN")
        stats = by_day[day]
        stats["total_r"] = _num(stats.get("total_r")) + r_value
        stats["trade_count"] = int(stats.get("trade_count") or 0) + 1
        stats["weakest_r"] = r_value if stats.get("weakest_r") is None else min(_num(stats.get("weakest_r")), r_value)
        stats["sector_counts"][sector] += 1
    normalized: dict[str, dict[str, Any]] = {}
    for day, stats in by_day.items():
        normalized[day] = {
            "total_r": _num(stats.get("total_r")),
            "trade_count": int(stats.get("trade_count") or 0),
            "weakest_r": _num(stats.get("weakest_r")),
            "sector_counts": dict(stats.get("sector_counts") or {}),
        }
    return realized, normalized


def _max_per_sector_from_mutations(mutations: dict[str, Any]) -> int:
    return max(1, int(_num(mutations.get("kalcb.risk.max_per_sector") or mutations.get("max_per_sector") or 8)))


def _route_family_static_checks(
    probe_specs: list[dict[str, Any]],
    base: dict[str, Any],
    meta: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    checks: list[dict[str, Any]] = []
    eligible_modes: list[str] = []
    for spec in probe_specs:
        route = dict(spec.get("route") or {})
        family = str(spec.get("route_family") or route.get("mode") or "")
        passed, reason = _route_candidate_passes(route, base, meta)
        checks.append(
            {
                "route_family": family,
                "route": str(route.get("name") or route.get("mode") or family),
                "passed": bool(passed),
                "reason": reason,
            }
        )
        if passed:
            eligible_modes.append(family)
    return checks, eligible_modes


def build_shadow_ledger(
    evaluator: RecoveryEvaluator,
    base: dict[str, Any],
    output_dir: Path,
    *,
    window: str,
) -> dict[str, Any]:
    plugin = evaluator.plugin if window == "train" else evaluator.plugin._validation_plugin()
    base_metrics = plugin.evaluate_mutations(base)
    base_detail = plugin._evaluation_details.get(_mutation_key(base))
    context = plugin._context_for_mutations(base)
    probe_specs = _shadow_ledger_probe_specs()
    route_outcomes_by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    probe_summaries: dict[str, dict[str, Any]] = {}
    for spec in probe_specs:
        family = str(spec["route_family"])
        mutations = _shadow_ledger_probe_mutations(base, dict(spec["route"]))
        evaluator.log("shadow_ledger_probe_start", window=window, route_family=family)
        probe_metrics = plugin.evaluate_mutations(mutations)
        aggregated = aggregate_route_shadow_rows(probe_metrics.get("frontier_shadow_trade_rows") or [], route_family=family)
        for key, outcome in aggregated.items():
            route_outcomes_by_key[key][family] = dict(outcome)
        probe_summaries[family] = {
            "mutations_hash": _mutation_key(mutations),
            "route": dict(spec["route"]),
            "metrics": _clean_metrics(probe_metrics),
            "frontier_shadow_summary": {
                key: probe_metrics.get(key)
                for key in (
                    "frontier_shadow_trade_count",
                    "frontier_shadow_expected_total_r",
                    "frontier_shadow_avg_r",
                    "frontier_shadow_nonselected_trade_count",
                    "frontier_shadow_nonselected_total_r",
                    "frontier_shadow_nonselected_avg_r",
                )
                if key in probe_metrics
            },
            "route_outcome_candidate_count": len(aggregated),
        }
        evaluator.log(
            "shadow_ledger_probe_done",
            window=window,
            route_family=family,
            shadow_trades=probe_summaries[family]["frontier_shadow_summary"].get("frontier_shadow_trade_count"),
            outcome_candidates=len(aggregated),
        )
    realized, actual_day_context = _realized_trade_context(base_detail.trade_rows if base_detail else tuple())
    routes = _configured_entry_routes(base)
    max_per_sector = _max_per_sector_from_mutations(base)
    ledger_path = output_dir / f"shadow_opportunity_ledger_{window}.jsonl"
    rows: list[dict[str, Any]] = []
    for day, snapshot in sorted(context.compiled_replay.snapshots.items(), key=lambda item: str(item[0])):
        day_label = day.isoformat() if hasattr(day, "isoformat") else str(day)[:10]
        day_actual = dict(actual_day_context.get(day_label) or {})
        actual_sector_counts = dict(day_actual.get("sector_counts") or {})
        for candidate in tuple(getattr(snapshot, "candidates", ()) or ()):
            meta = _candidate_snapshot_metadata(candidate, day_label)
            symbol = str(meta.get("symbol") or "")
            key = (day_label, symbol)
            route_checks = []
            eligible = False
            for route in routes:
                passed, reason = _route_candidate_passes(route, base, meta)
                route_checks.append({"route": str(route.get("name") or route.get("mode") or "route"), "passed": passed, "reason": reason})
                eligible = eligible or passed
            first_blocker = "eligible" if eligible else next((item["reason"] for item in route_checks if not item["passed"]), "eligible")
            family_checks, family_eligible = _route_family_static_checks(probe_specs, base, meta)
            actual_total = _num(day_actual.get("total_r"))
            actual_trade_count = int(day_actual.get("trade_count") or 0)
            weakest_actual = _num(day_actual.get("weakest_r"))
            role = _candidate_frontier_role(meta)
            sector = str(meta.get("sector") or "UNKNOWN")
            sector_actual_count = int(actual_sector_counts.get(sector, 0) or 0)
            row = {
                "window": window,
                "trade_date": day_label,
                "symbol": symbol,
                "sector": sector,
                "frontier_role": role,
                "frontier_rank": int(_num(meta.get("frontier_rank"))),
                "candidate_rank": int(_num(meta.get("candidate_rank"))),
                "first30_ret": meta.get("first30_ret"),
                "first30_vwap_ret": meta.get("first30_vwap_ret"),
                "first30_rel_volume": meta.get("first30_rel_volume"),
                "first30_signal_bar_cpr": meta.get("first30_signal_bar_cpr", meta.get("first30_close_location")),
                "first30_range_close_location": meta.get("first30_range_close_location", meta.get("first30_close_location")),
                "first30_range_atr": meta.get("first30_range_atr"),
                "current_route_eligible": eligible,
                "current_route_checks": route_checks,
                "first_blocker": first_blocker,
                "current_realized": key in realized,
                "route_family_static_checks": family_checks,
                "route_family_static_eligible_modes": family_eligible,
                "route_outcomes": dict(route_outcomes_by_key.get(key) or {}),
                "same_day_actual_total_r": actual_total,
                "same_day_actual_trade_count": actual_trade_count,
                "same_day_actual_sector_counts": actual_sector_counts,
                "same_day_weakest_actual_r": weakest_actual,
                "same_day_candidate_sector_actual_count": sector_actual_count,
                "candidate_max_per_sector_pressure": max(0.0, (sector_actual_count + 1.0) / max(float(max_per_sector), 1.0) - 1.0),
            }
            for field in LEDGER_CONTEXT_FEATURE_KEYS:
                if field not in row:
                    row[field] = meta.get(field)
            rows.append(row)
    prepared_rows = prepare_shadow_ledger_rows(rows, max_per_sector=max_per_sector)
    write_jsonl(ledger_path, prepared_rows)
    counts = Counter(str(row.get("frontier_role") or "unknown") for row in prepared_rows)
    blocker_counts = Counter(str(row.get("first_blocker") or "unknown") for row in prepared_rows if not row.get("current_route_eligible"))
    outcome_rows = [row for row in prepared_rows if row.get("route_outcome_available")]
    same_day_improvers = sum(1 for row in outcome_rows if _num(row.get("same_day_replacement_value_r")) > 0.0)
    top_rows = sorted(
        outcome_rows,
        key=lambda item: (_num(item.get("same_day_replacement_value_r")), _num(item.get("best_route_shadow_total_r")), _num(item.get("best_route_shadow_max_mfe_r"))),
        reverse=True,
    )
    sector_shadow_r: dict[str, float] = defaultdict(float)
    for row in outcome_rows:
        sector_shadow_r[str(row.get("sector") or "UNKNOWN")] += _num(row.get("best_route_shadow_total_r"))
    summary = {
        "window": window,
        "ledger_path": str(ledger_path),
        "base_metrics": _clean_metrics(base_metrics),
        "route_probe_summaries": probe_summaries,
        "route_family_outcome_coverage": route_family_outcome_coverage(prepared_rows),
        "feature_coverage": feature_coverage(prepared_rows, LEDGER_CONTEXT_FEATURE_KEYS),
        "candidate_count_by_role": dict(counts),
        "top_current_route_blockers": blocker_counts.most_common(12),
        "same_day_improver_count": same_day_improvers,
        "route_outcome_candidate_count": len(outcome_rows),
        "max_per_sector": max_per_sector,
        "top_shadow_rows": top_rows[:20],
        "sector_shadow_total_r": sorted(({"sector": key, "shadow_total_r": value} for key, value in sector_shadow_r.items()), key=lambda item: item["shadow_total_r"], reverse=True)[:20],
    }
    return summary


def _same_day_reranker_output_dir(output_dir: Path) -> Path:
    return output_dir if output_dir.name == "06_same_day_reranker" else output_dir / "06_same_day_reranker"


def build_same_day_reranker_stage(
    output_dir: Path,
    base: dict[str, Any],
    *,
    round_num: int,
) -> dict[str, Any]:
    train_path = output_dir / "shadow_opportunity_ledger_train.jsonl"
    holdout_path = output_dir / "shadow_opportunity_ledger_holdout.jsonl"
    if not train_path.exists() or not holdout_path.exists():
        missing = [str(path) for path in (train_path, holdout_path) if not path.exists()]
        raise FileNotFoundError(f"Same-day reranker requires both shadow ledgers; missing {missing}")
    summary = build_same_day_reranker_artifacts(
        read_jsonl(train_path),
        read_jsonl(holdout_path),
        output_dir=_same_day_reranker_output_dir(output_dir),
        max_per_sector=_max_per_sector_from_mutations(base),
    )
    _update_round_research_diagnostics(round_num, summary)
    return summary


def _next_round_output_dir(output_dir: Path) -> Path:
    return output_dir if output_dir.name == "07_alpha_conversion_next_round" else output_dir / "07_alpha_conversion_next_round"


def _structured_auto_artifact(output_dir: Path) -> Path:
    direct = output_dir / "kalcb_local_minimum_recovery.json"
    nested = output_dir / "05_structured_auto_round" / "kalcb_local_minimum_recovery.json"
    return nested if nested.exists() else direct


def _find_named_rows(payload: Any, names: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            name = str(node.get("name") or "")
            if name in names and name not in found and isinstance(node.get("mutations"), dict):
                found[name] = dict(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    return found


def _materialized_candidate(base: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    out.update(dict(row.get("mutations") or {}))
    return out


def _compact_auto_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": row.get("name"),
        "score": row.get("score"),
        "research_survivor": row.get("research_survivor"),
        "train": _clean_metrics(dict(row.get("train") or {})),
        "holdout": _clean_metrics(dict(row.get("holdout") or {})),
        "train_delta": _clean_metrics(dict(row.get("train_delta") or {})),
        "holdout_delta": _clean_metrics(dict(row.get("holdout_delta") or {})),
    }


def _path_state_exit_mutations(*, hold_bars: int, min_mfe_r: float, min_giveback_r: float, context_min: dict[str, float]) -> dict[str, Any]:
    return {
        "kalcb.exit.path_quality_enabled": True,
        "kalcb.exit.path_quality_min_hold_bars": int(hold_bars),
        "kalcb.exit.path_quality_min_mfe_r": float(min_mfe_r),
        "kalcb.exit.path_quality_min_giveback_r": float(min_giveback_r),
        "kalcb.exit.path_quality.context_min": dict(context_min),
        "kalcb.exit.path_quality.context_max": {},
        "kalcb.exit.path_quality_entry_route_modes": ["first30_open", "pullback_acceptance", "avwap_reclaim", "or_high_reclaim"],
    }


def build_path_state_exit_experiments(seed: dict[str, Any], challenger: dict[str, Any]) -> list[RecoveryExperiment]:
    profiles = [
        (
            "pathq_route_orhigh2_h24_mfe6_gb4",
            _path_state_exit_mutations(
                hold_bars=24,
                min_mfe_r=6.0,
                min_giveback_r=4.0,
                context_min={"below_or_high_streak": 2.0},
            ),
            "Exit after real MFE if post-entry closes lose OR-high for two bars.",
        ),
        (
            "pathq_route_vwap2_h18_mfe5_gb3",
            _path_state_exit_mutations(
                hold_bars=18,
                min_mfe_r=5.0,
                min_giveback_r=3.0,
                context_min={"below_vwap_streak": 2.0},
            ),
            "Exit after real MFE if the accepted route fails back under VWAP.",
        ),
    ]
    roots = ((NEXT_ROUND_SEED, seed), (NEXT_ROUND_CHALLENGER, challenger))
    experiments: list[RecoveryExperiment] = []
    for root_name, root_mutations in roots:
        for profile_name, profile_mutations, purpose in profiles:
            mutations = dict(root_mutations)
            mutations.update(profile_mutations)
            experiments.append(
                RecoveryExperiment(
                    stage="next_round",
                    family="path_state_exits",
                    name=f"{root_name}__{profile_name}",
                    purpose=purpose,
                    mutations=mutations,
                    replace_base=True,
                )
            )
    return experiments


def _oracle_entry_specs() -> tuple[tuple[str, EntrySpec], ...]:
    return (
        ("first30_open", EntrySpec("oracle_first30_open", BASELINE_ENTRY_MODE)),
        (
            "pullback_acceptance",
            EntrySpec(
                "oracle_pullback_acceptance",
                "pullback_acceptance",
                after_bar=1,
                max_signal_bars=18,
                max_pullback_from_vwap_pct=0.008,
                min_reclaim_ret=0.0005,
                min_vwap_ret=0.0,
            ),
        ),
        (
            "avwap_reclaim",
            EntrySpec(
                "oracle_avwap_reclaim",
                "avwap_reclaim",
                after_bar=1,
                max_signal_bars=18,
                max_pullback_from_vwap_pct=0.008,
                min_reclaim_ret=0.0005,
                min_vwap_ret=0.0,
            ),
        ),
        (
            "or_high_reclaim",
            EntrySpec(
                "oracle_or_high_reclaim",
                "or_high_reclaim",
                after_bar=1,
                max_signal_bars=18,
                max_pullback_from_vwap_pct=0.008,
                min_reclaim_ret=0.0005,
                min_vwap_ret=0.0,
            ),
        ),
    )


def _oracle_exit_spec(base: dict[str, Any]) -> ExitSpec:
    stop_mode = str(base.get("kalcb.exit.stop_mode") or "fixed_pct")
    return ExitSpec(
        "oracle_eod_path",
        stop_mode=stop_mode,
        stop_atr_mult=float(base.get("kalcb.risk.stop_atr_multiple") or 0.80),
        stop_pct=float(base.get("kalcb.exit.stop_pct") or 0.003),
        hard_stop_enabled=False,
        target_r=0.0,
    )


def _context_oracle_features(ctx: Any) -> dict[str, Any]:
    daily_meta = dict(ctx.sector_daily.metadata()) if getattr(ctx, "sector_daily", None) is not None else {}
    intraday_meta = dict(ctx.sector_intraday.metadata()) if getattr(ctx, "sector_intraday", None) is not None else {}
    sector_ret5 = _num(daily_meta.get("sector_daily_ret_5d"))
    sector_ret20 = _num(daily_meta.get("sector_daily_ret_20d"))
    sector_intraday_ret = _num(intraday_meta.get("sector_intraday_ret"))
    return {
        "sector": str(ctx.sector or "UNKNOWN"),
        "first30_ret": float(ctx.first30_ret),
        "first30_vwap_ret": float(ctx.vwap_ret),
        "first30_rel_volume": float(ctx.rel_volume),
        "first30_signal_bar_cpr": float(ctx.close_location),
        "first30_range_atr": float(ctx.range_atr),
        "sector_daily_score_pct": daily_meta.get("sector_daily_score_pct"),
        "sector_daily_participation": daily_meta.get("sector_daily_participation"),
        "sector_daily_breadth_20d": daily_meta.get("sector_daily_breadth_20d"),
        "stock_sector_daily_ret5_spread": float(ctx.daily.return_5d) - sector_ret5,
        "stock_sector_daily_ret20_spread": float(ctx.daily.return_20d) - sector_ret20,
        "sector_intraday_score_pct": intraday_meta.get("sector_intraday_score_pct"),
        "sector_intraday_ret": sector_intraday_ret,
        "sector_intraday_breadth": intraday_meta.get("sector_intraday_breadth"),
        "sector_intraday_participation": intraday_meta.get("sector_intraday_participation"),
        "first30_sector_ret_spread": float(ctx.first30_ret) - sector_intraday_ret,
        "leading_sector_cluster": _num(daily_meta.get("sector_daily_score_pct")) >= 80.0
        and _num(intraday_meta.get("sector_intraday_score_pct")) >= 70.0,
    }


def _oracle_score(row: dict[str, Any]) -> float:
    relvol = min(math.log1p(max(_num(row.get("first30_rel_volume")), 0.0)) / math.log(21.0), 1.5)
    cpr = max(min(_num(row.get("first30_signal_bar_cpr")), 1.0), 0.0)
    delayed_bonus = 0.5 if str(row.get("route_family")) != "first30_open" else 0.0
    return (
        _num(row.get("net_r"))
        + 0.22 * min(_num(row.get("mfe_r")), 25.0)
        - 0.32 * abs(min(_num(row.get("mae_r")), 0.0))
        + 1.4 * max(_num(row.get("mfe_capture")), 0.0)
        + 1.2 * relvol
        + 0.8 * (cpr - 0.5)
        + delayed_bonus
    )


def _simulate_oracle_row(ctx: Any, route_family: str, entry_spec: EntrySpec, exit_spec: ExitSpec, cfg: Any, dataset: Any) -> dict[str, Any] | None:
    bars = tuple(getattr(ctx, "bars", ()) or ())
    if not bars:
        return None
    outcome = simulate_trade(
        ctx.day,
        ctx.symbol,
        bars,
        ctx,
        entry_spec,
        exit_spec,
        cfg,
        prior_day_high=_prior_day_high(dataset, ctx.symbol, ctx.day),
    )
    if outcome is None:
        return None
    risk_pct = float(outcome.risk_per_share) / max(float(outcome.entry_price), 1e-9)
    net_r = float(outcome.net_return_pct) / max(risk_pct, 1e-9)
    gross_r = float(outcome.gross_return_pct) / max(risk_pct, 1e-9)
    row = {
        "trade_date": ctx.day.isoformat(),
        "symbol": str(ctx.symbol),
        "route_family": route_family,
        "entry_time": outcome.entry_time.isoformat() if hasattr(outcome.entry_time, "isoformat") else str(outcome.entry_time),
        "entry_price": float(outcome.entry_price),
        "risk_pct": risk_pct,
        "gross_r": gross_r,
        "net_r": net_r,
        "mfe_r": float(outcome.mfe_r),
        "mae_r": float(outcome.mae_r),
        "mfe_capture": float(outcome.mfe_capture),
        "bars_held": int(outcome.bars_held),
        "exit_reason": outcome.exit_reason,
        **_context_oracle_features(ctx),
    }
    row["oracle_score"] = _oracle_score(row)
    return row


def _best_by_day(rows: list[dict[str, Any]], *, in_pool: bool | None = None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if in_pool is not None and bool(row.get("in_candidate_pool")) != in_pool:
            continue
        day = str(row.get("trade_date") or "")
        current = out.get(day)
        key = (_num(row.get("oracle_score")), _num(row.get("net_r")), _num(row.get("mfe_r")), str(row.get("symbol") or ""))
        if current is None or key > (_num(current.get("oracle_score")), _num(current.get("net_r")), _num(current.get("mfe_r")), str(current.get("symbol") or "")):
            out[day] = row
    return out


def _avg(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def _quality_qualified_oracle_row(row: dict[str, Any]) -> bool:
    return _num(row.get("first30_rel_volume")) >= 2.0 and _num(row.get("first30_signal_bar_cpr")) >= 0.55


def _summarize_oracle_rows_core(rows: list[dict[str, Any]], session_count: int) -> dict[str, Any]:
    best_all = _best_by_day(rows)
    best_in_pool = _best_by_day(rows, in_pool=True)
    best_out_pool = _best_by_day(rows, in_pool=False)
    missed_days: list[dict[str, Any]] = []
    for day, out_row in best_out_pool.items():
        in_row = best_in_pool.get(day)
        score_advantage = _num(out_row.get("oracle_score")) - _num((in_row or {}).get("oracle_score"))
        net_advantage = _num(out_row.get("net_r")) - _num((in_row or {}).get("net_r"))
        mfe_advantage = _num(out_row.get("mfe_r")) - _num((in_row or {}).get("mfe_r"))
        if in_row is None or score_advantage > 0.0:
            missed_days.append(
                {
                    "trade_date": day,
                    "out_of_pool": out_row,
                    "best_in_pool": in_row,
                    "score_advantage": score_advantage,
                    "net_r_advantage": net_advantage,
                    "mfe_r_advantage": mfe_advantage,
                }
            )
    missed_days.sort(key=lambda item: (_num(item.get("score_advantage")), _num(item.get("mfe_r_advantage"))), reverse=True)
    out_pool_net_adv = [_num(item.get("net_r_advantage")) for item in missed_days]
    out_pool_mfe_adv = [_num(item.get("mfe_r_advantage")) for item in missed_days]
    route_counts = Counter(str(row.get("route_family") or "unknown") for row in rows)
    top_all_outside = sum(1 for row in best_all.values() if not row.get("in_candidate_pool"))
    missed_share = len(missed_days) / max(float(session_count), 1.0)
    return {
        "session_count": int(session_count),
        "route_outcome_rows": len(rows),
        "symbols_with_route_outcomes": len({(row.get("trade_date"), row.get("symbol")) for row in rows}),
        "route_family_counts": dict(route_counts),
        "candidate_pool_route_outcome_rows": sum(1 for row in rows if row.get("in_candidate_pool")),
        "out_of_pool_route_outcome_rows": sum(1 for row in rows if not row.get("in_candidate_pool")),
        "days_with_any_route_outcome": len(best_all),
        "days_best_overall_outside_candidate_pool": top_all_outside,
        "days_out_of_pool_beats_best_in_pool": len(missed_days),
        "out_of_pool_missed_day_share": missed_share,
        "avg_out_of_pool_net_r_advantage": _avg(out_pool_net_adv),
        "median_out_of_pool_net_r_advantage": _median(out_pool_net_adv),
        "avg_out_of_pool_mfe_r_advantage": _avg(out_pool_mfe_adv),
        "top_missed_opportunity_days": missed_days[:25],
        "candidate_surfacing_verdict": (
            "candidate_surfacing_likely_missing_material_alpha"
            if missed_share >= 0.20 and _avg(out_pool_mfe_adv) > 2.0
            else "candidate_surfacing_gap_present_but_not_primary_without_more_evidence"
        ),
    }


def _summarize_oracle_rows(rows: list[dict[str, Any]], session_count: int) -> dict[str, Any]:
    summary = _summarize_oracle_rows_core(rows, session_count)
    quality_rows = [row for row in rows if _quality_qualified_oracle_row(row)]
    delayed_rows = [row for row in rows if str(row.get("route_family") or "") != "first30_open"]
    quality_delayed_rows = [row for row in delayed_rows if _quality_qualified_oracle_row(row)]
    summary["quality_filter"] = {"first30_rel_volume_min": 2.0, "first30_signal_bar_cpr_min": 0.55}
    summary["quality_qualified"] = _summarize_oracle_rows_core(quality_rows, session_count)
    summary["delayed_route_only"] = _summarize_oracle_rows_core(delayed_rows, session_count)
    summary["quality_delayed_route_only"] = _summarize_oracle_rows_core(quality_delayed_rows, session_count)
    return summary


def build_full_universe_missed_opportunity(
    evaluator: RecoveryEvaluator,
    base: dict[str, Any],
    output_dir: Path,
    *,
    window: str,
) -> dict[str, Any]:
    plugin = evaluator.plugin if window == "train" else evaluator.plugin._validation_plugin()
    base_metrics = plugin.evaluate_mutations(base)
    base_detail = plugin._evaluation_details.get(_mutation_key(base))
    context = plugin._context_for_mutations(base)
    cfg = plugin._config_for_mutations(base)
    dataset = prepare_first30_dataset(dict(plugin.config))
    contexts = build_first30_contexts(dataset)
    exit_spec = _oracle_exit_spec(base)
    realized, actual_day_context = _realized_trade_context(base_detail.trade_rows if base_detail else tuple())
    pool_by_day = {
        day: {str(candidate.symbol) for candidate in getattr(snapshot, "candidates", ())}
        for day, snapshot in context.compiled_replay.snapshots.items()
    }
    rows: list[dict[str, Any]] = []
    for day in dataset.trading_dates:
        candidate_pool = pool_by_day.get(day, set())
        actual = dict(actual_day_context.get(day.isoformat()) or {})
        best_by_symbol: dict[str, dict[str, Any]] = {}
        for ctx in contexts.get(day, ()):
            best: dict[str, Any] | None = None
            for route_family, entry_spec in _oracle_entry_specs():
                row = _simulate_oracle_row(ctx, route_family, entry_spec, exit_spec, cfg, dataset)
                if row is None:
                    continue
                if best is None or (_num(row.get("oracle_score")), _num(row.get("net_r"))) > (
                    _num(best.get("oracle_score")),
                    _num(best.get("net_r")),
                ):
                    best = row
            if best is None:
                continue
            best["window"] = window
            best["in_candidate_pool"] = str(ctx.symbol) in candidate_pool
            best["current_realized"] = (day.isoformat(), str(ctx.symbol)) in realized
            best["same_day_actual_total_r"] = _num(actual.get("total_r"))
            best["same_day_actual_trade_count"] = int(actual.get("trade_count") or 0)
            best["same_day_weakest_actual_r"] = _num(actual.get("weakest_r"))
            best_by_symbol[str(ctx.symbol)] = best
        rows.extend(best_by_symbol.values())
    rows.sort(key=lambda item: (str(item.get("trade_date") or ""), -_num(item.get("oracle_score")), str(item.get("symbol") or "")))
    path = output_dir / f"full_universe_missed_opportunities_{window}.jsonl"
    write_jsonl(path, rows)
    summary = _summarize_oracle_rows(rows, len(dataset.trading_dates))
    summary.update(
        {
            "window": window,
            "artifact_path": str(path),
            "base_metrics": _clean_metrics(base_metrics),
            "universe_scope": "configured_kalcb_dataset_symbols_with_first30_contexts_not_all_krx",
            "candidate_pool_scope": "compiled_round_candidate_snapshot_symbols",
            "oracle_usage_contract": "research_only_ex_post_path_outcome_diagnostic_not_live_selection_feature",
            "configured_symbol_count": len(dataset.symbols),
            "context_symbol_day_count": sum(len(items) for items in contexts.values()),
        }
    )
    return summary


def _render_full_universe_oracle_report(summary: dict[str, Any]) -> str:
    lines = [
        "# KALCB Full-Universe Missed-Opportunity Oracle",
        "",
        "Research-only diagnostic. It compares configured-universe first30 contexts against the compiled round candidate pool using ex-post route/path outcomes.",
        "",
    ]
    for window in ("train", "holdout"):
        row = dict(summary.get(window) or {})
        if not row:
            continue
        lines.extend(
            [
                f"## {window.title()}",
                "",
                f"- Sessions: {row.get('session_count', 0)}",
                f"- Route-outcome rows: {row.get('route_outcome_rows', 0)}",
                f"- In-pool rows: {row.get('candidate_pool_route_outcome_rows', 0)}",
                f"- Out-of-pool rows: {row.get('out_of_pool_route_outcome_rows', 0)}",
                f"- Days where out-of-pool beat best in-pool: {row.get('days_out_of_pool_beats_best_in_pool', 0)}",
                f"- Avg out-of-pool MFE advantage: {_num(row.get('avg_out_of_pool_mfe_r_advantage')):.2f}R",
                f"- Verdict: {row.get('candidate_surfacing_verdict')}",
                "",
                "Qualified subviews:",
                f"- Relvol/CPR qualified: rows={(row.get('quality_qualified') or {}).get('route_outcome_rows', 0)}, missed_days={(row.get('quality_qualified') or {}).get('days_out_of_pool_beats_best_in_pool', 0)}, avg_MFE_adv={_num((row.get('quality_qualified') or {}).get('avg_out_of_pool_mfe_r_advantage')):.2f}R",
                f"- Delayed-route only: rows={(row.get('delayed_route_only') or {}).get('route_outcome_rows', 0)}, missed_days={(row.get('delayed_route_only') or {}).get('days_out_of_pool_beats_best_in_pool', 0)}, avg_MFE_adv={_num((row.get('delayed_route_only') or {}).get('avg_out_of_pool_mfe_r_advantage')):.2f}R",
                f"- Relvol/CPR delayed-route: rows={(row.get('quality_delayed_route_only') or {}).get('route_outcome_rows', 0)}, missed_days={(row.get('quality_delayed_route_only') or {}).get('days_out_of_pool_beats_best_in_pool', 0)}, avg_MFE_adv={_num((row.get('quality_delayed_route_only') or {}).get('avg_out_of_pool_mfe_r_advantage')):.2f}R",
                "",
                "| Date | Out-of-pool | Route | Score adv | NetR adv | MFE adv | Best in-pool |",
                "|---|---:|---|---:|---:|---:|---:|",
            ]
        )
        for item in list(row.get("top_missed_opportunity_days") or [])[:12]:
            out = dict(item.get("out_of_pool") or {})
            inp = dict(item.get("best_in_pool") or {})
            lines.append(
                f"| {item.get('trade_date')} | {out.get('symbol')} | {out.get('route_family')} | "
                f"{_num(item.get('score_advantage')):.2f} | {_num(item.get('net_r_advantage')):.2f} | "
                f"{_num(item.get('mfe_r_advantage')):.2f} | {inp.get('symbol', '')} |"
            )
        lines.append("")
    return "\n".join(lines)


def _render_next_round_report(payload: dict[str, Any]) -> str:
    seed = dict(payload.get("seed") or {})
    challenger = dict(payload.get("challenger") or {})
    path_eval = dict(payload.get("path_state_exit_evaluations") or {})
    rows = sorted(path_eval.get("rows") or [], key=lambda item: _num(item.get("score")), reverse=True)
    lines = [
        "# KALCB Alpha-Conversion Next Round",
        "",
        f"- Conservative seed: `{seed.get('name')}`",
        f"- Upside challenger: `{challenger.get('name')}`",
        "- Exit work: route-family path-state exits only; no broad target/giveback sweep.",
        "",
        "## Seed Metrics",
        "",
        f"- Seed train/holdout net: {_pct((seed.get('train') or {}).get('broker_net_return_pct'))} / {_pct((seed.get('holdout') or {}).get('broker_net_return_pct'))}",
        f"- Seed trades train/holdout: {_num((seed.get('train') or {}).get('trade_count')):.0f} / {_num((seed.get('holdout') or {}).get('trade_count')):.0f}",
        f"- Challenger train/holdout net: {_pct((challenger.get('train') or {}).get('broker_net_return_pct'))} / {_pct((challenger.get('holdout') or {}).get('broker_net_return_pct'))}",
        f"- Challenger trades train/holdout: {_num((challenger.get('train') or {}).get('trade_count')):.0f} / {_num((challenger.get('holdout') or {}).get('trade_count')):.0f}",
        "",
        "## Path-State Exit Probes",
        "",
    ]
    if rows:
        for row in rows:
            lines.append(
                f"- `{row.get('name')}`: score={_num(row.get('score')):.2f}, survivor={row.get('research_survivor')}, "
                f"train_delta={_pct((row.get('train_delta') or {}).get('broker_net_return_pct'))}, "
                f"holdout_delta={_pct((row.get('holdout_delta') or {}).get('broker_net_return_pct'))}, "
                f"train_capture_delta={_pct((row.get('train_delta') or {}).get('avg_mfe_capture'))}"
            )
    else:
        lines.append("- No path-state exit probes were evaluated in this run.")
    lines.extend(
        [
            "",
            "## Forward Contract",
            "",
            "- Use the seed as the conservative next auto-round starting point.",
            "- Keep the target60 challenger only as upside evidence; do not promote it if holdout degradation or drawdown expands.",
            "- Use full-universe oracle misses to decide whether the next frontier/search phase needs candidate surfacing changes before more route tuning.",
        ]
    )
    return "\n".join(lines)


def _alpha_conversion_digest(summary: dict[str, Any]) -> dict[str, Any]:
    paths = {key: _repo_display_path(value) for key, value in dict(summary.get("artifact_paths") or {}).items()}

    def compact_oracle(row: dict[str, Any]) -> dict[str, Any]:
        out = {key: value for key, value in row.items() if key not in {"top_missed_opportunity_days", "base_metrics"}}
        if out.get("artifact_path"):
            out["artifact_path"] = _repo_display_path(str(out["artifact_path"]))
        for key in ("quality_qualified", "delayed_route_only", "quality_delayed_route_only"):
            if isinstance(out.get(key), dict):
                out[key] = {
                    sub_key: sub_value
                    for sub_key, sub_value in dict(out[key]).items()
                    if sub_key not in {"top_missed_opportunity_days", "base_metrics"}
                }
        return out

    return {
        "created_at": summary.get("created_at") or _utc_now_iso(),
        "purpose": "next_round_seed_full_universe_oracle_and_route_family_path_state_exit_diagnostics",
        "artifact_paths": paths,
        "seed_name": (summary.get("seed") or {}).get("name"),
        "challenger_name": (summary.get("challenger") or {}).get("name"),
        "seed": _compact_auto_row(dict(summary.get("seed") or {})),
        "challenger": _compact_auto_row(dict(summary.get("challenger") or {})),
        "full_universe_oracle": {
            "train": compact_oracle(dict((summary.get("full_universe_oracle") or {}).get("train") or {})),
            "holdout": compact_oracle(dict((summary.get("full_universe_oracle") or {}).get("holdout") or {})),
        },
        "path_state_exit_survivors": [
            {"name": row.get("name"), "score": row.get("score"), "train_delta": row.get("train_delta"), "holdout_delta": row.get("holdout_delta")}
            for row in ((summary.get("path_state_exit_evaluations") or {}).get("survivors") or [])
        ],
    }


def _update_alpha_conversion_final_diagnostics(round_dir: Path, digest: dict[str, Any]) -> None:
    path = round_dir / "round_final_diagnostics.txt"
    if not path.exists():
        return
    oracle = dict(digest.get("full_universe_oracle") or {})
    train = dict(oracle.get("train") or {})
    holdout = dict(oracle.get("holdout") or {})
    train_quality = dict(train.get("quality_qualified") or {})
    train_delayed = dict(train.get("delayed_route_only") or {})
    holdout_quality = dict(holdout.get("quality_qualified") or {})
    holdout_delayed = dict(holdout.get("delayed_route_only") or {})
    paths = dict(digest.get("artifact_paths") or {})
    body = "\n".join(
        [
            "",
            f"Seed: {digest.get('seed_name')} (conservative pullback q85 rank<=8 risk 0.015)",
            f"Upside challenger: {digest.get('challenger_name')} (pullback q85 rank<=8 risk 0.02 target60)",
            "",
            "Full-universe missed-opportunity oracle:",
            f"  Train: days_out_of_pool_beats_best_in_pool={train.get('days_out_of_pool_beats_best_in_pool', 0)}, avg_MFE_adv={_num(train.get('avg_out_of_pool_mfe_r_advantage')):.2f}R, verdict={train.get('candidate_surfacing_verdict', '')}",
            f"    Qualified train: relvol/CPR missed_days={train_quality.get('days_out_of_pool_beats_best_in_pool', 0)}, delayed_route_missed_days={train_delayed.get('days_out_of_pool_beats_best_in_pool', 0)}",
            f"  Holdout: days_out_of_pool_beats_best_in_pool={holdout.get('days_out_of_pool_beats_best_in_pool', 0)}, avg_MFE_adv={_num(holdout.get('avg_out_of_pool_mfe_r_advantage')):.2f}R, verdict={holdout.get('candidate_surfacing_verdict', '')}",
            f"    Qualified holdout: relvol/CPR missed_days={holdout_quality.get('days_out_of_pool_beats_best_in_pool', 0)}, delayed_route_missed_days={holdout_delayed.get('days_out_of_pool_beats_best_in_pool', 0)}",
            "",
            "Path-state exit work: evaluated route-family OR-high/VWAP failure exits tied to MFE and giveback; no broad target/giveback sweep.",
            "",
            "Artifact pointers:",
            f"  summary_json={paths.get('summary_json', '')}",
            f"  report_md={paths.get('report_md', '')}",
            f"  oracle_report_md={paths.get('oracle_report_md', '')}",
        ]
    )
    path.write_text(_replace_marked_section(path.read_text(encoding="utf-8"), "17. Alpha-Conversion Next Round", body), encoding="utf-8")


def _update_alpha_conversion_diagnostics(round_num: int, summary: dict[str, Any]) -> None:
    round_dir = ROUND_ROOT / f"round_{round_num}"
    digest = _alpha_conversion_digest(summary)
    for name in ("diagnostics_summary.json", "full_diagnostics_index.json"):
        path = round_dir / name
        payload = _read_json(path) if path.exists() else {}
        payload["alpha_conversion_next_round"] = digest
        for key, value in dict(digest.get("artifact_paths") or {}).items():
            payload[f"alpha_conversion_next_round_{key}_path"] = value
        _write_json(path, payload)
    _update_alpha_conversion_final_diagnostics(round_dir, digest)

    manifest_path = ROUND_ROOT / "rounds_manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        for row in manifest.get("rounds") or []:
            if int(row.get("round") or 0) != int(round_num):
                continue
            row.setdefault("research_artifacts", {})["alpha_conversion_next_round"] = digest
            row["updated_at_utc"] = _utc_now_iso()
            break
        _write_json(manifest_path, manifest)


def build_alpha_conversion_next_round_stage(
    evaluator: RecoveryEvaluator,
    base: dict[str, Any],
    output_dir: Path,
    *,
    round_num: int,
    max_evals: int | None,
) -> dict[str, Any]:
    out = _next_round_output_dir(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    auto_path = _structured_auto_artifact(output_dir)
    if not auto_path.exists():
        raise FileNotFoundError(f"Next-round stage requires structured auto artifact: {auto_path}")
    rows = _find_named_rows(_read_json(auto_path), {NEXT_ROUND_SEED, NEXT_ROUND_CHALLENGER})
    missing = sorted({NEXT_ROUND_SEED, NEXT_ROUND_CHALLENGER} - set(rows))
    if missing:
        raise ValueError(f"Structured auto artifact is missing required next-round rows: {missing}")
    seed_row = rows[NEXT_ROUND_SEED]
    challenger_row = rows[NEXT_ROUND_CHALLENGER]
    seed = _materialized_candidate(base, seed_row)
    challenger = _materialized_candidate(base, challenger_row)

    seed_path = out / "next_round_seed_auto_pullback_q85_rank8_r0p015.json"
    challenger_path = out / "next_round_challenger_auto_pullback_q85_rank8_r0p02_target60.json"
    _write_json(seed_path, {"name": NEXT_ROUND_SEED, "role": "conservative_seed", "mutations": seed, "source_row": _compact_auto_row(seed_row)})
    _write_json(challenger_path, {"name": NEXT_ROUND_CHALLENGER, "role": "upside_challenger", "mutations": challenger, "source_row": _compact_auto_row(challenger_row)})

    oracle = {
        "train": build_full_universe_missed_opportunity(evaluator, base, out, window="train"),
        "holdout": build_full_universe_missed_opportunity(evaluator, base, out, window="holdout"),
    }
    oracle_summary_path = out / "full_universe_missed_opportunity_summary.json"
    oracle_report_path = out / "full_universe_missed_opportunity_report.md"
    _write_json(oracle_summary_path, oracle)
    oracle_report_path.write_text(_render_full_universe_oracle_report(oracle), encoding="utf-8")

    exit_experiments = build_path_state_exit_experiments(seed, challenger)
    path_eval = run_evaluations(evaluator, base, exit_experiments, max_evals=max_evals)
    summary = {
        "created_at": _utc_now_iso(),
        "strategy": "kalcb",
        "round": round_num,
        "seed": {**_compact_auto_row(seed_row), "mutations_path": str(seed_path)},
        "challenger": {**_compact_auto_row(challenger_row), "mutations_path": str(challenger_path)},
        "full_universe_oracle": oracle,
        "path_state_exit_experiments": [asdict(item) for item in exit_experiments],
        "path_state_exit_evaluations": path_eval,
        "next_round_contract": {
            "conservative_seed": NEXT_ROUND_SEED,
            "upside_challenger": NEXT_ROUND_CHALLENGER,
            "exit_policy": "route_family_path_state_rules_only_no_broad_target_or_giveback_sweep",
            "candidate_surfacing_gate": "use full-universe oracle to decide whether out-of-pool winners require frontier/source changes before more route tuning",
        },
    }
    summary_path = out / "alpha_conversion_next_round_summary.json"
    report_path = out / "alpha_conversion_next_round_report.md"
    summary["artifact_paths"] = {
        "summary_json": str(summary_path),
        "report_md": str(report_path),
        "seed_config_json": str(seed_path),
        "challenger_config_json": str(challenger_path),
        "oracle_summary_json": str(oracle_summary_path),
        "oracle_report_md": str(oracle_report_path),
        "oracle_train_jsonl": str(out / "full_universe_missed_opportunities_train.jsonl"),
        "oracle_holdout_jsonl": str(out / "full_universe_missed_opportunities_holdout.jsonl"),
    }
    _write_json(summary_path, summary)
    report_path.write_text(_render_next_round_report(summary), encoding="utf-8")
    _update_alpha_conversion_diagnostics(round_num, summary)
    return summary


def _candidate_surfacing_output_dir(output_dir: Path) -> Path:
    return output_dir if output_dir.name == "08_candidate_surfacing_recovery" else output_dir / "08_candidate_surfacing_recovery"


def _structural_campaign_output_dir(output_dir: Path) -> Path:
    return output_dir if output_dir.name == "09_structural_campaign_surfacing" else output_dir / "09_structural_campaign_surfacing"


def _stage09_cached_rows(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists() or path.stat().st_size <= 0:
        return None
    try:
        first_line = ""
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    first_line = line
                    break
        first = json.loads(first_line) if first_line else {}
    except (OSError, json.JSONDecodeError):
        return None
    required_fields = {
        "campaign_box_atr_ratio",
        "campaign_box_squeeze_pct",
        "campaign_box_tier",
        "first30_rel_volume",
        "first30_signal_cpr",
    }
    if required_fields - set(first):
        return None
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(structural_campaign_feature_artifact_row(json.loads(line)))
    except (OSError, json.JSONDecodeError):
        return None
    return rows or None


def _write_stage09_cached_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(tmp, (structural_campaign_feature_artifact_row(row) for row in rows))
    tmp.replace(path)


def _iter_jsonl_rows(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _structural_campaign_digest(summary: dict[str, Any]) -> dict[str, Any]:
    optimizer = dict(summary.get("optimizer") or {})
    alcb_delta = dict(summary.get("alcb_delta_diagnostics") or {})
    breakout_replay = dict(summary.get("alcb_breakout_replay") or {})
    alcb_compact: dict[str, Any] = {}
    for window in ("train", "holdout"):
        row = dict(alcb_delta.get(window) or {})
        variants = dict(row.get("proxy_variants") or {})
        alcb_compact[window] = {
            "strict_box": variants.get("alcb_strict_box") or {},
            "or_breakout_proxy": variants.get("or_breakout_proxy") or {},
            "or_breakout_avwap_cap_1pct": variants.get("or_breakout_avwap_cap_1pct") or {},
            "active_frontier_miss": row.get("active_frontier_miss") or {},
        }
    return {
        "version": summary.get("version", STRUCTURAL_CAMPAIGN_SURFACING_VERSION),
        "usage_contract": summary.get("usage_contract", STRUCTURAL_CAMPAIGN_USAGE_CONTRACT),
        "created_at": summary.get("created_at") or _utc_now_iso(),
        "artifact_paths": {key: _repo_display_path(value) for key, value in dict(summary.get("artifact_paths") or {}).items()},
        "train": summary.get("train") or {},
        "holdout": summary.get("holdout") or {},
        "optimizer": {
            "optimizer_contract": optimizer.get("optimizer_contract"),
            "train_variant_count": optimizer.get("train_variant_count"),
            "shortlist_size": optimizer.get("shortlist_size"),
            "best_train_variant": optimizer.get("best_train_variant") or {},
            "holdout_frozen_rank1_variant": optimizer.get("holdout_frozen_rank1_variant") or optimizer.get("best_holdout_frozen_variant") or {},
            "best_holdout_frozen_variant": optimizer.get("holdout_frozen_rank1_variant") or optimizer.get("best_holdout_frozen_variant") or {},
            "holdout_selection_basis": optimizer.get("holdout_selection_basis"),
            "shortlist": [
                {
                    "train_rank": row.get("train_rank"),
                    "optimizer_variant_id": row.get("optimizer_variant_id"),
                    "pool_variant": row.get("pool_variant"),
                    "selector_variant": row.get("selector_variant"),
                    "source_score": row.get("source_score"),
                    "recall_at_active_budget": row.get("recall_at_active_budget"),
                    "recall_at_32": row.get("recall_at_32"),
                    "avg_top1_oracle_score": row.get("avg_top1_oracle_score"),
                    "top1_net_r_sum": row.get("top1_net_r_sum"),
                    "top_decile_oracle_recall": row.get("top_decile_oracle_recall"),
                    "monotonicity_score": row.get("monotonicity_score"),
                    "route_eligible_share": row.get("route_route_eligible_share"),
                    "mutations": row.get("mutations"),
                }
                for row in optimizer.get("shortlist") or []
            ],
        },
        "alcb_delta_diagnostics": {
            "version": alcb_delta.get("version"),
            "purpose": alcb_delta.get("purpose"),
            "windows": alcb_compact,
        },
        "alcb_breakout_replay": {
            "version": breakout_replay.get("version"),
            "replay_contract": breakout_replay.get("replay_contract"),
            "train_variant_count": breakout_replay.get("train_variant_count"),
            "shortlist_size": breakout_replay.get("shortlist_size"),
            "best_train_variant": breakout_replay.get("best_train_variant") or {},
            "holdout_frozen_rank1_variant": breakout_replay.get("holdout_frozen_rank1_variant") or breakout_replay.get("best_holdout_frozen_variant") or {},
            "best_holdout_frozen_variant": breakout_replay.get("holdout_frozen_rank1_variant") or breakout_replay.get("best_holdout_frozen_variant") or {},
            "holdout_selection_basis": breakout_replay.get("holdout_selection_basis"),
            "shortlist": [
                {
                    "train_rank": row.get("train_rank"),
                    "variant_id": row.get("variant_id"),
                    "pool_variant": row.get("pool_variant"),
                    "selector_mode": row.get("selector_mode"),
                    "route_family": row.get("route_family"),
                    "replay_score": row.get("replay_score"),
                    "trade_count": row.get("trade_count"),
                    "broker_net_return_pct": row.get("broker_net_return_pct"),
                    "broker_expected_total_r": row.get("broker_expected_total_r"),
                    "broker_max_drawdown_pct": row.get("broker_max_drawdown_pct"),
                    "reject_reason": row.get("reject_reason"),
                }
                for row in breakout_replay.get("shortlist") or []
            ],
        },
        "variant_manifest": summary.get("variant_manifest") or {},
    }


def _update_structural_campaign_final_diagnostics(round_dir: Path, digest: dict[str, Any]) -> None:
    path = round_dir / "round_final_diagnostics.txt"
    if not path.exists():
        return
    train = dict(digest.get("train") or {})
    holdout = dict(digest.get("holdout") or {})
    train_recall = dict(train.get("recall") or {})
    holdout_recall = dict(holdout.get("recall") or {})
    optimizer = dict(digest.get("optimizer") or {})
    best_train = dict(optimizer.get("best_train_variant") or {})
    best_holdout = dict(optimizer.get("holdout_frozen_rank1_variant") or optimizer.get("best_holdout_frozen_variant") or {})
    alcb_windows = dict((dict(digest.get("alcb_delta_diagnostics") or {}).get("windows")) or {})
    alcb_train = dict(alcb_windows.get("train") or {})
    alcb_holdout = dict(alcb_windows.get("holdout") or {})
    train_or_proxy = dict(alcb_train.get("or_breakout_proxy") or {})
    holdout_or_proxy = dict(alcb_holdout.get("or_breakout_proxy") or {})
    train_miss = dict(alcb_train.get("active_frontier_miss") or {})
    holdout_miss = dict(alcb_holdout.get("active_frontier_miss") or {})
    breakout = dict(digest.get("alcb_breakout_replay") or {})
    breakout_train = dict(breakout.get("best_train_variant") or {})
    breakout_holdout = dict(breakout.get("holdout_frozen_rank1_variant") or breakout.get("best_holdout_frozen_variant") or {})
    paths = dict(digest.get("artifact_paths") or {})
    body = "\n".join(
        [
            "",
            f"Version: {digest.get('version')}",
            f"Usage: {digest.get('usage_contract')}",
            "",
            "Structural campaign source:",
            f"  Train features={train.get('feature_row_count', 0)}, pools={train.get('pool_row_count', 0)}, oracle={train_recall.get('oracle_label_available')}, contract={train_recall.get('recall_contract', '')}.",
            f"  Holdout features={holdout.get('feature_row_count', 0)}, pools={holdout.get('pool_row_count', 0)}, oracle={holdout_recall.get('oracle_label_available')}, contract={holdout_recall.get('recall_contract', '')}.",
            f"  Train recall@32={_pct(train_recall.get('recall_at_32'))}; holdout recall@32={_pct(holdout_recall.get('recall_at_32'))}.",
            "",
            "Train-only structural optimizer:",
            f"  Variants swept={optimizer.get('train_variant_count', 0)}, frozen shortlist={optimizer.get('shortlist_size', 0)}.",
            f"  Best train `{best_train.get('optimizer_variant_id', '')}` pool={best_train.get('pool_variant', '')} selector={best_train.get('selector_variant', '')}: source_score={_num(best_train.get('source_score')):.3f}, active_recall={_pct(best_train.get('recall_at_active_budget'))}, recall@32={_pct(best_train.get('recall_at_32'))}, top1_oracle={_num(best_train.get('avg_top1_oracle_score')):.2f}, route_eligible={_pct(best_train.get('route_route_eligible_share'))}.",
            f"  Frozen holdout rank1 `{best_holdout.get('optimizer_variant_id', '')}` pool={best_holdout.get('pool_variant', '')} selector={best_holdout.get('selector_variant', '')}: source_score={_num(best_holdout.get('source_score')):.3f}, active_recall={_pct(best_holdout.get('recall_at_active_budget'))}, recall@32={_pct(best_holdout.get('recall_at_32'))}, top1_oracle={_num(best_holdout.get('avg_top1_oracle_score')):.2f}, route_eligible={_pct(best_holdout.get('route_route_eligible_share'))}.",
            "",
            "ALCB delta diagnostics:",
            f"  Train OR proxy rows={train_or_proxy.get('row_count', 0)}, best-share={_pct(train_or_proxy.get('best_oracle_in_proxy_share'))}, active-missed-best={_pct(train_miss.get('missed_best_in_frontier_share'))}.",
            f"  Holdout OR proxy rows={holdout_or_proxy.get('row_count', 0)}, best-share={_pct(holdout_or_proxy.get('best_oracle_in_proxy_share'))}, active-missed-best={_pct(holdout_miss.get('missed_best_in_frontier_share'))}.",
            "",
            "ALCB breakout replay:",
            f"  Train variants={breakout.get('train_variant_count', 0)}, frozen shortlist={breakout.get('shortlist_size', 0)}.",
            f"  Best train `{breakout_train.get('variant_id', '')}` pool={breakout_train.get('pool_variant', '')} selector={breakout_train.get('selector_mode', '')} route={breakout_train.get('route_family', '')}: score={_num(breakout_train.get('replay_score')):.3f}, trades={_num(breakout_train.get('trade_count')):.0f}, net={_pct(breakout_train.get('broker_net_return_pct'))}, totalR={_num(breakout_train.get('broker_expected_total_r')):.2f}, dd={_pct(breakout_train.get('broker_max_drawdown_pct'))}, reject={breakout_train.get('reject_reason', '')}.",
            f"  Frozen holdout rank1 `{breakout_holdout.get('variant_id', '')}` pool={breakout_holdout.get('pool_variant', '')} selector={breakout_holdout.get('selector_mode', '')} route={breakout_holdout.get('route_family', '')}: score={_num(breakout_holdout.get('replay_score')):.3f}, trades={_num(breakout_holdout.get('trade_count')):.0f}, net={_pct(breakout_holdout.get('broker_net_return_pct'))}, totalR={_num(breakout_holdout.get('broker_expected_total_r')):.2f}, dd={_pct(breakout_holdout.get('broker_max_drawdown_pct'))}, reject={breakout_holdout.get('reject_reason', '')}.",
            "",
            "Artifact pointers:",
            f"  summary_json={paths.get('diagnostics_summary_json', '')}",
            f"  report_md={paths.get('structural_campaign_report_md', '')}",
            f"  recall_summary_json={paths.get('structural_campaign_recall_summary_json', '')}",
            f"  optimizer_summary_json={paths.get('structural_campaign_optimizer_summary_json', '')}",
            f"  replay_summary_json={paths.get('structural_campaign_replay_summary_json', '')}",
            f"  alcb_delta_json={paths.get('structural_campaign_alcb_delta_diagnostics_json', '')}",
            f"  alcb_breakout_replay_json={paths.get('structural_campaign_alcb_breakout_replay_summary_json', '')}",
        ]
    )
    path.write_text(_replace_marked_section(path.read_text(encoding="utf-8"), "19. Structural Campaign Surfacing", body), encoding="utf-8")


def _update_structural_campaign_diagnostics(round_num: int, summary: dict[str, Any]) -> None:
    round_dir = ROUND_ROOT / f"round_{round_num}"
    digest = _structural_campaign_digest(summary)
    for name in ("diagnostics_summary.json", "full_diagnostics_index.json"):
        path = round_dir / name
        payload = _read_json(path) if path.exists() else {}
        payload["structural_campaign_surfacing"] = digest
        for key, value in dict(digest.get("artifact_paths") or {}).items():
            payload[f"structural_campaign_surfacing_{key}_path"] = value
        _write_json(path, payload)
    _update_structural_campaign_final_diagnostics(round_dir, digest)
    manifest_path = ROUND_ROOT / "rounds_manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        for row in manifest.get("rounds") or []:
            if int(row.get("round") or 0) != int(round_num):
                continue
            row.setdefault("research_artifacts", {})["structural_campaign_surfacing"] = digest
            row["updated_at_utc"] = _utc_now_iso()
            break
        _write_json(manifest_path, manifest)


def build_structural_campaign_surfacing_stage(
    evaluator: RecoveryEvaluator,
    base: dict[str, Any],
    output_dir: Path,
    *,
    round_num: int,
) -> dict[str, Any]:
    out = _structural_campaign_output_dir(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_plugin = evaluator.plugin
    holdout_plugin = evaluator.plugin._validation_plugin()
    train_cfg = KALCBConfig.from_mapping(dict(train_plugin.config), base)
    holdout_cfg = KALCBConfig.from_mapping(dict(holdout_plugin.config), base)
    train_existing_context = train_plugin._context_for_mutations(base)
    holdout_existing_context = holdout_plugin._context_for_mutations(base)
    oracle_dir = _next_round_output_dir(output_dir)
    train_oracle_path = oracle_dir / "full_universe_missed_opportunities_train.jsonl"
    holdout_oracle_path = oracle_dir / "full_universe_missed_opportunities_holdout.jsonl"
    causal_dir = _candidate_surfacing_output_dir(output_dir)
    train_causal_path = causal_dir / "candidate_surfacing_train_features.jsonl"
    holdout_causal_path = causal_dir / "candidate_surfacing_holdout_features.jsonl"
    train_features_path = out / "structural_campaign_features_train.jsonl"
    holdout_features_path = out / "structural_campaign_features_holdout.jsonl"
    permissive_source = {
        "kalcb.research.structural_frontier_count": 64,
        "kalcb.research.min_structural_campaign_score": 0.0,
        "kalcb.research.min_rs_percentile": 0.0,
        "kalcb.research.min_sector_daily_score_pct": 0.0,
        "kalcb.research.min_sector_participation": 0.0,
        "kalcb.research.max_box_range_pct": 0.0,
    }
    train_dataset = None
    holdout_dataset = None
    train_rows = _stage09_cached_rows(train_features_path)
    if train_rows is None:
        train_dataset = prepare_first30_dataset(dict(train_plugin.config))
        train_rows = build_structural_campaign_artifact_rows(train_dataset, None, train_cfg.with_mutations(permissive_source), window="train")
        _write_stage09_cached_rows(train_features_path, train_rows)
    holdout_rows = _stage09_cached_rows(holdout_features_path)
    if holdout_rows is None:
        holdout_dataset = prepare_first30_dataset(dict(holdout_plugin.config))
        holdout_rows = build_structural_campaign_artifact_rows(holdout_dataset, None, holdout_cfg.with_mutations(permissive_source), window="holdout")
        _write_stage09_cached_rows(holdout_features_path, holdout_rows)
    train_rows = attach_causal_calibration_scores(
        train_rows,
        _iter_jsonl_rows(train_causal_path) if train_causal_path.exists() else (),
    )
    holdout_rows = attach_causal_calibration_scores(
        holdout_rows,
        _iter_jsonl_rows(holdout_causal_path) if holdout_causal_path.exists() else (),
    )
    train_active_budget = active_budget_by_day_from_context(train_existing_context)
    holdout_active_budget = active_budget_by_day_from_context(holdout_existing_context)
    train_oracle_rows = read_jsonl(train_oracle_path) if train_oracle_path.exists() else []
    holdout_oracle_rows = read_jsonl(holdout_oracle_path) if holdout_oracle_path.exists() else []
    summary = build_structural_campaign_surfacing_artifacts(
        train_rows,
        holdout_rows,
        output_dir=out,
        cfg=train_cfg,
        train_active_budget_by_day=train_active_budget,
        holdout_active_budget_by_day=holdout_active_budget,
        train_oracle_rows=train_oracle_rows,
        holdout_oracle_rows=holdout_oracle_rows,
    )
    train_dataset = train_dataset or prepare_first30_dataset(dict(train_plugin.config))
    holdout_dataset = holdout_dataset or prepare_first30_dataset(dict(holdout_plugin.config))
    breakout_replay = build_alcb_breakout_replay_artifacts(
        train_rows,
        holdout_rows,
        train_dataset=train_dataset,
        holdout_dataset=holdout_dataset,
        output_dir=out,
        cfg=train_cfg,
        train_active_budget_by_day=train_active_budget,
        holdout_active_budget_by_day=holdout_active_budget,
    )
    summary["alcb_breakout_replay"] = {key: value for key, value in breakout_replay.items() if key != "train_rows"}
    summary["replay_summary"] = summary["alcb_breakout_replay"]
    summary.setdefault("artifact_paths", {}).update(dict(breakout_replay.get("artifact_paths") or {}))
    faithfulness_funnel = build_alcb_faithfulness_funnel_artifacts(
        train_rows,
        holdout_rows,
        train_dataset=train_dataset,
        holdout_dataset=holdout_dataset,
        output_dir=out,
        cfg=train_cfg,
        train_oracle_rows=train_oracle_rows,
        holdout_oracle_rows=holdout_oracle_rows,
    )
    summary["alcb_faithfulness_funnel"] = {key: value for key, value in faithfulness_funnel.items() if key != "train_rows"}
    summary.setdefault("artifact_paths", {}).update(dict(faithfulness_funnel.get("artifact_paths") or {}))
    replay_summary_path = Path(summary["artifact_paths"].get("structural_campaign_replay_summary_json", out / "structural_campaign_replay_summary.json"))
    replay_summary_path.write_text(json.dumps(summary["replay_summary"], indent=2, sort_keys=True, default=str), encoding="utf-8")
    for name in ("diagnostics_summary_json", "full_diagnostics_index_json"):
        path_value = summary["artifact_paths"].get(name)
        if path_value:
            Path(path_value).write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    report_path_value = summary["artifact_paths"].get("structural_campaign_report_md")
    if report_path_value:
        report_path = Path(report_path_value)
        if report_path.exists():
            train_best = dict(summary["alcb_breakout_replay"].get("best_train_variant") or {})
            holdout_best = dict(summary["alcb_breakout_replay"].get("holdout_frozen_rank1_variant") or summary["alcb_breakout_replay"].get("best_holdout_frozen_variant") or {})
            funnel_train_best = dict(summary["alcb_faithfulness_funnel"].get("best_train_variant") or {})
            funnel_holdout_best = dict(summary["alcb_faithfulness_funnel"].get("holdout_frozen_rank1_variant") or {})
            replay_body = "\n".join(
                [
                    "",
                    f"- Contract: `{summary['alcb_breakout_replay'].get('replay_contract') or ''}`",
                    f"- Train variants: {summary['alcb_breakout_replay'].get('train_variant_count', 0)}; frozen shortlist: {summary['alcb_breakout_replay'].get('shortlist_size', 0)}",
                    f"- Best train: `{train_best.get('variant_id') or ''}` pool={train_best.get('pool_variant') or ''} selector={train_best.get('selector_mode') or ''} route={train_best.get('route_family') or ''} trades={int(train_best.get('trade_count') or 0)} net={_pct(train_best.get('broker_net_return_pct'))} totalR={_num(train_best.get('broker_expected_total_r')):.2f} dd={_pct(train_best.get('broker_max_drawdown_pct'))}",
                    f"- Frozen holdout rank1: `{holdout_best.get('variant_id') or ''}` pool={holdout_best.get('pool_variant') or ''} selector={holdout_best.get('selector_mode') or ''} route={holdout_best.get('route_family') or ''} trades={int(holdout_best.get('trade_count') or 0)} net={_pct(holdout_best.get('broker_net_return_pct'))} totalR={_num(holdout_best.get('broker_expected_total_r')):.2f} dd={_pct(holdout_best.get('broker_max_drawdown_pct'))}",
                    "",
                    "## ALCB Faithfulness Funnel",
                    "",
                    f"- Contract: `{summary['alcb_faithfulness_funnel'].get('funnel_contract') or ''}`",
                    f"- Train variants: {summary['alcb_faithfulness_funnel'].get('train_variant_count', 0)}; frozen shortlist: {summary['alcb_faithfulness_funnel'].get('shortlist_size', 0)}",
                    f"- Best train: `{funnel_train_best.get('variant_id') or ''}` topN={int(funnel_train_best.get('top_n') or 0)} selector={funnel_train_best.get('selector_mode') or ''} route={funnel_train_best.get('route_bundle') or ''} trades={int(funnel_train_best.get('trade_count') or 0)} simR={_num(funnel_train_best.get('simulated_total_r')):.2f} best_pool={_num(funnel_train_best.get('best_oracle_in_pool_share')):.3f}",
                    f"- Frozen holdout rank1: `{funnel_holdout_best.get('variant_id') or ''}` topN={int(funnel_holdout_best.get('top_n') or 0)} selector={funnel_holdout_best.get('selector_mode') or ''} route={funnel_holdout_best.get('route_bundle') or ''} trades={int(funnel_holdout_best.get('trade_count') or 0)} simR={_num(funnel_holdout_best.get('simulated_total_r')):.2f} best_pool={_num(funnel_holdout_best.get('best_oracle_in_pool_share')):.3f}",
                    f"- Train top1 miss counts: `{summary['alcb_faithfulness_funnel'].get('train_top1_miss_counts') or {}}`",
                    f"- Holdout top1 miss counts: `{summary['alcb_faithfulness_funnel'].get('holdout_top1_miss_counts') or {}}`",
                ]
            )
            marker = "\n## ALCB Breakout Replay\n"
            report_text = report_path.read_text(encoding="utf-8")
            report_path.write_text(f"{report_text.split(marker, 1)[0].rstrip()}{marker}{replay_body.rstrip()}\n", encoding="utf-8")
    _update_structural_campaign_diagnostics(round_num, summary)
    return summary


def _next_round_seed_config(output_dir: Path) -> Path:
    return _next_round_output_dir(output_dir) / "next_round_seed_auto_pullback_q85_rank8_r0p015.json"


def _candidate_surfacing_digest(summary: dict[str, Any]) -> dict[str, Any]:
    paths = {key: _repo_display_path(value) for key, value in dict(summary.get("artifact_paths") or {}).items()}

    def compact_recall(window: dict[str, Any]) -> dict[str, Any]:
        recall = dict(window.get("recall") or {})
        variants = dict(recall.get("variants") or {})
        compact_variants = {}
        for name, row in variants.items():
            row = dict(row or {})
            compact_variants[name] = {
                "best_oracle_in_pool_share": row.get("best_oracle_in_pool_share"),
                "best_quality_delayed_oracle_in_pool_share": row.get("best_quality_delayed_oracle_in_pool_share"),
                "route_eligible_share": row.get("route_eligible_share"),
                "avg_pool_size": row.get("avg_pool_size"),
                "top_decile_oracle_recall": row.get("top_decile_oracle_recall"),
                "avg_in_pool_mfe_r": row.get("avg_in_pool_mfe_r"),
                "avg_out_of_pool_mfe_r": row.get("avg_out_of_pool_mfe_r"),
                "avg_leading_sector_cluster_share": row.get("avg_leading_sector_cluster_share"),
            }
        return {
            "ranker_recall_at": recall.get("ranker_recall_at"),
            "best_variant_by_quality_delayed_recall": recall.get("best_variant_by_quality_delayed_recall"),
            "best_variant_by_route_eligible_share": recall.get("best_variant_by_route_eligible_share"),
            "variants": compact_variants,
        }

    def compact_replay(window: dict[str, Any]) -> dict[str, Any]:
        replay = dict(window.get("replay") or {})
        def compact_best(row: Any) -> dict[str, Any]:
            item = dict(row or {})
            return {
                "variant": item.get("variant"),
                "metrics": _clean_metrics(dict(item.get("metrics") or {})),
                "baseline_delta": item.get("baseline_delta"),
                "compiled_replay": item.get("compiled_replay"),
                "entry_route_mode_summary": item.get("entry_route_mode_summary"),
            }

        rows = []
        for row in replay.get("rows") or []:
            row = dict(row or {})
            rows.append(
                {
                    "variant": row.get("variant"),
                    "metrics": _clean_metrics(dict(row.get("metrics") or {})),
                    "baseline_delta": row.get("baseline_delta"),
                    "trade_rows_path": _repo_display_path(row.get("trade_rows_path") or "") if row.get("trade_rows_path") else "",
                    "compiled_replay": row.get("compiled_replay"),
                    "entry_route_mode_summary": row.get("entry_route_mode_summary"),
                }
            )
        return {
            "baseline_metrics": _clean_metrics(dict(replay.get("baseline_metrics") or {})),
            "best_by_net": compact_best(replay.get("best_by_net")),
            "best_by_frequency_adjusted_net": compact_best(replay.get("best_by_frequency_adjusted_net")),
            "rows": rows,
        }

    return {
        "version": summary.get("version", CANDIDATE_SURFACING_RECOVERY_VERSION),
        "usage_contract": summary.get("usage_contract", CANDIDATE_SURFACING_USAGE_CONTRACT),
        "created_at": summary.get("created_at") or _utc_now_iso(),
        "artifact_paths": paths,
        "route_bundle": summary.get("route_bundle") or (summary.get("variant_manifest") or {}).get("route_bundle"),
        "matched_incumbent_route_bundle": summary.get("matched_incumbent_route_bundle") or (summary.get("variant_manifest") or {}).get("matched_incumbent_route_bundle"),
        "profile": {
            "used_feature_count": (summary.get("profile") or {}).get("used_feature_count"),
            "weights": (summary.get("profile") or {}).get("weights"),
            "source_window": (summary.get("profile") or {}).get("source_window"),
        },
        "train": {
            "feature_row_count": (summary.get("train") or {}).get("feature_row_count"),
            "oracle_labeled_row_count": (summary.get("train") or {}).get("oracle_labeled_row_count"),
            "recall": compact_recall(dict(summary.get("train") or {})),
            "replay": compact_replay(dict(summary.get("train") or {})),
            "matched_incumbent_replay": compact_replay({"replay": (summary.get("train") or {}).get("matched_incumbent_replay")}),
            "matched_active_budget_replay": compact_replay({"replay": (summary.get("train") or {}).get("matched_active_budget_replay")}),
        },
        "holdout": {
            "feature_row_count": (summary.get("holdout") or {}).get("feature_row_count"),
            "oracle_labeled_row_count": (summary.get("holdout") or {}).get("oracle_labeled_row_count"),
            "recall": compact_recall(dict(summary.get("holdout") or {})),
            "replay": compact_replay(dict(summary.get("holdout") or {})),
            "matched_incumbent_replay": compact_replay({"replay": (summary.get("holdout") or {}).get("matched_incumbent_replay")}),
            "matched_active_budget_replay": compact_replay({"replay": (summary.get("holdout") or {}).get("matched_active_budget_replay")}),
        },
        "root_cause_layer_attribution": summary.get("root_cause_layer_attribution"),
    }


def _best_recall_metrics(digest: dict[str, Any], window: str) -> dict[str, Any]:
    recall = dict(((digest.get(window) or {}).get("recall") or {}))
    best = dict(recall.get("best_variant_by_quality_delayed_recall") or {})
    return dict(best.get("metrics") or {})


def _best_replay_metrics(digest: dict[str, Any], window: str, replay_key: str = "replay") -> tuple[str, dict[str, Any]]:
    replay = dict(((digest.get(window) or {}).get(replay_key) or {}))
    best = dict(replay.get("best_by_frequency_adjusted_net") or replay.get("best_by_net") or {})
    return str(best.get("variant") or ""), dict(best.get("metrics") or {})


def _route_mix_label(digest: dict[str, Any], window: str, replay_key: str = "replay") -> str:
    replay = dict(((digest.get(window) or {}).get(replay_key) or {}))
    best = dict(replay.get("best_by_frequency_adjusted_net") or replay.get("best_by_net") or {})
    modes = dict(best.get("entry_route_mode_summary") or {})
    if not modes:
        return "none"
    parts = []
    for mode, metrics in sorted(modes.items()):
        row = dict(metrics or {})
        parts.append(f"{mode}:{int(_num(row.get('trades')))} trades/capture={_pct(row.get('avg_mfe_capture'))}/EOD={_pct(row.get('eod_flatten_share'))}")
    return "; ".join(parts)


def _update_candidate_surfacing_final_diagnostics(round_dir: Path, digest: dict[str, Any]) -> None:
    path = round_dir / "round_final_diagnostics.txt"
    if not path.exists():
        return
    train_recall = _best_recall_metrics(digest, "train")
    holdout_recall = _best_recall_metrics(digest, "holdout")
    train_variant, train_replay = _best_replay_metrics(digest, "train")
    holdout_variant, holdout_replay = _best_replay_metrics(digest, "holdout")
    train_matched_variant, train_matched_replay = _best_replay_metrics(digest, "train", "matched_incumbent_replay")
    holdout_matched_variant, holdout_matched_replay = _best_replay_metrics(digest, "holdout", "matched_incumbent_replay")
    train_active_variant, train_active_replay = _best_replay_metrics(digest, "train", "matched_active_budget_replay")
    holdout_active_variant, holdout_active_replay = _best_replay_metrics(digest, "holdout", "matched_active_budget_replay")
    root = dict(digest.get("root_cause_layer_attribution") or {})
    paths = dict(digest.get("artifact_paths") or {})
    route_bundle = dict(digest.get("route_bundle") or {})
    matched_route_bundle = dict(digest.get("matched_incumbent_route_bundle") or {})
    body = "\n".join(
        [
            "",
            f"Version: {digest.get('version')}",
            f"Usage: {digest.get('usage_contract')}",
            f"Replay route bundle: {route_bundle.get('name', '')}; modes={', '.join(route_bundle.get('modes') or [])}; first30_open_enabled={route_bundle.get('first30_open_enabled')}",
            f"Matched incumbent sanity bundle: {matched_route_bundle.get('name', '')}; first30_open_enabled={matched_route_bundle.get('first30_open_enabled')}",
            "",
            "Causal candidate-pool recall:",
            f"  Train best quality-delayed recall={_pct(train_recall.get('best_quality_delayed_oracle_in_pool_share'))}, route_eligible={_pct(train_recall.get('route_eligible_share'))}, top_decile_recall={_pct(train_recall.get('top_decile_oracle_recall'))}.",
            f"  Holdout best quality-delayed recall={_pct(holdout_recall.get('best_quality_delayed_oracle_in_pool_share'))}, route_eligible={_pct(holdout_recall.get('route_eligible_share'))}, top_decile_recall={_pct(holdout_recall.get('top_decile_oracle_recall'))}.",
            "",
            "Constrained replay using delayed pullback/AVWAP/OR-high routes:",
            f"  Train best replay `{train_variant}`: net={_pct(train_replay.get('broker_net_return_pct'))}, trades={_num(train_replay.get('trade_count')):.0f}, DD={_pct(train_replay.get('broker_max_drawdown_pct'))}, capture={_pct(train_replay.get('avg_mfe_capture'))}.",
            f"  Train route mix: {_route_mix_label(digest, 'train')}.",
            f"  Holdout best replay `{holdout_variant}`: net={_pct(holdout_replay.get('broker_net_return_pct'))}, trades={_num(holdout_replay.get('trade_count')):.0f}, DD={_pct(holdout_replay.get('broker_max_drawdown_pct'))}, capture={_pct(holdout_replay.get('avg_mfe_capture'))}.",
            f"  Holdout route mix: {_route_mix_label(digest, 'holdout')}.",
            "",
            "Matched incumbent execution sanity check:",
            f"  Train `{train_matched_variant}`: net={_pct(train_matched_replay.get('broker_net_return_pct'))}, trades={_num(train_matched_replay.get('trade_count')):.0f}, DD={_pct(train_matched_replay.get('broker_max_drawdown_pct'))}, capture={_pct(train_matched_replay.get('avg_mfe_capture'))}.",
            f"  Train matched route mix: {_route_mix_label(digest, 'train', 'matched_incumbent_replay')}.",
            f"  Holdout `{holdout_matched_variant}`: net={_pct(holdout_matched_replay.get('broker_net_return_pct'))}, trades={_num(holdout_matched_replay.get('trade_count')):.0f}, DD={_pct(holdout_matched_replay.get('broker_max_drawdown_pct'))}, capture={_pct(holdout_matched_replay.get('avg_mfe_capture'))}.",
            f"  Holdout matched route mix: {_route_mix_label(digest, 'holdout', 'matched_incumbent_replay')}.",
            "",
            "Active-budget matched incumbent sanity check:",
            f"  Train `{train_active_variant}`: net={_pct(train_active_replay.get('broker_net_return_pct'))}, trades={_num(train_active_replay.get('trade_count')):.0f}, active_budget={_num(train_active_replay.get('active_budget_candidate_count')):.0f}, DD={_pct(train_active_replay.get('broker_max_drawdown_pct'))}, capture={_pct(train_active_replay.get('avg_mfe_capture'))}.",
            f"  Holdout `{holdout_active_variant}`: net={_pct(holdout_active_replay.get('broker_net_return_pct'))}, trades={_num(holdout_active_replay.get('trade_count')):.0f}, active_budget={_num(holdout_active_replay.get('active_budget_candidate_count')):.0f}, DD={_pct(holdout_active_replay.get('broker_max_drawdown_pct'))}, capture={_pct(holdout_active_replay.get('avg_mfe_capture'))}.",
            "",
            "Layer attribution:",
            f"  Candidate surfacing: {root.get('candidate_surfacing', '')}",
            f"  Candidate selection: {root.get('candidate_selection', '')}",
            f"  Entry route: {root.get('entry_route', '')}",
            f"  Exit/path management: {root.get('exit_path_management', '')}",
            "",
            "Artifact pointers:",
            f"  summary_json={paths.get('summary_json', '')}",
            f"  report_md={paths.get('report_md', '')}",
            f"  recall_summary_json={paths.get('recall_summary_json', '')}",
            f"  replay_summary_json={paths.get('replay_summary_json', '')}",
        ]
    )
    path.write_text(_replace_marked_section(path.read_text(encoding="utf-8"), "18. Candidate Surfacing Recovery", body), encoding="utf-8")


def _update_candidate_surfacing_diagnostics(round_num: int, summary: dict[str, Any]) -> None:
    round_dir = ROUND_ROOT / f"round_{round_num}"
    digest = _candidate_surfacing_digest(summary)
    for name in ("diagnostics_summary.json", "full_diagnostics_index.json"):
        path = round_dir / name
        payload = _read_json(path) if path.exists() else {}
        payload["candidate_surfacing_recovery"] = digest
        for key, value in dict(digest.get("artifact_paths") or {}).items():
            payload[f"candidate_surfacing_recovery_{key}_path"] = value
        _write_json(path, payload)
    _update_candidate_surfacing_final_diagnostics(round_dir, digest)

    manifest_path = ROUND_ROOT / "rounds_manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        for row in manifest.get("rounds") or []:
            if int(row.get("round") or 0) != int(round_num):
                continue
            row.setdefault("research_artifacts", {})["candidate_surfacing_recovery"] = digest
            row["updated_at_utc"] = _utc_now_iso()
            break
        _write_json(manifest_path, manifest)


def build_candidate_surfacing_recovery_stage(
    evaluator: RecoveryEvaluator,
    base: dict[str, Any],
    output_dir: Path,
    *,
    round_num: int,
    max_evals: int | None = None,
) -> dict[str, Any]:
    out = _candidate_surfacing_output_dir(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_path = _next_round_seed_config(output_dir)
    if not seed_path.exists():
        raise FileNotFoundError(f"Candidate-surfacing stage requires next-round seed config: {seed_path}")
    seed_payload = _read_json(seed_path)
    seed = dict(seed_payload.get("mutations") or {})
    if not seed:
        raise ValueError(f"Seed config does not contain mutations: {seed_path}")

    oracle_dir = _next_round_output_dir(output_dir)
    train_oracle = oracle_dir / "full_universe_missed_opportunities_train.jsonl"
    holdout_oracle = oracle_dir / "full_universe_missed_opportunities_holdout.jsonl"
    missing = [str(path) for path in (train_oracle, holdout_oracle) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Candidate-surfacing stage requires stage-07 oracle artifacts; missing {missing}")

    train_plugin = evaluator.plugin
    holdout_plugin = evaluator.plugin._validation_plugin()
    evaluator.log("candidate_surfacing_seed_baseline_start")
    baseline_metrics = {
        "train": train_plugin.evaluate_mutations(seed),
        "holdout": holdout_plugin.evaluate_mutations(seed),
    }
    evaluator.log(
        "candidate_surfacing_seed_baseline_done",
        train_net=baseline_metrics["train"].get("broker_net_return_pct"),
        holdout_net=baseline_metrics["holdout"].get("broker_net_return_pct"),
    )
    summary = build_candidate_surfacing_recovery_artifacts(
        train_config=dict(train_plugin.config),
        holdout_config=dict(holdout_plugin.config),
        train_existing_context=train_plugin._context_for_mutations(base),
        holdout_existing_context=holdout_plugin._context_for_mutations(base),
        train_oracle_path=train_oracle,
        holdout_oracle_path=holdout_oracle,
        seed_mutations=seed,
        baseline_metrics=baseline_metrics,
        output_dir=out,
        max_replay_variants=max_evals,
    )
    _update_candidate_surfacing_diagnostics(round_num, summary)
    return summary


def _repo_display_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _compact_reranker_window(window: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "row_count",
        "route_outcome_count",
        "candidate_day_count",
        "top_ranked_day_count",
        "positive_top_replacement_days",
        "top_ranked_same_day_replacement_total_r",
        "top_ranked_marginal_slot_replacement_total_r",
        "avg_top_ranked_mfe_r",
        "avg_top_ranked_mae_r",
        "avg_top_ranked_mfe_capture",
        "top_ranked_route_family_counts",
    )
    return {key: window.get(key) for key in keys if key in window}


def _reranker_diagnostics_digest(summary: dict[str, Any]) -> dict[str, Any]:
    paths = {key: _repo_display_path(value) for key, value in dict(summary.get("artifact_paths") or {}).items()}
    return {
        "reranker_version": summary.get("reranker_version", SHADOW_LEDGER_RERANKER_VERSION),
        "usage_contract": summary.get("usage_contract", SHADOW_LEDGER_RERANKER_USAGE_CONTRACT),
        "created_at": summary.get("created_at") or _utc_now_iso(),
        "artifact_paths": paths,
        "train": _compact_reranker_window(dict(summary.get("train") or {})),
        "holdout": _compact_reranker_window(dict(summary.get("holdout") or {})),
        "route_family_outcome_coverage": summary.get("route_family_outcome_coverage"),
        "holdout_sector_validation": summary.get("holdout_sector_validation"),
        "root_cause_layer_attribution": summary.get("root_cause_layer_attribution"),
    }


def _replace_marked_section(text: str, title: str, body: str) -> str:
    start = f"\n======================================================================\n  {title}\n======================================================================\n"
    if start in text:
        prefix = text.split(start, 1)[0].rstrip()
        rest = text.split(start, 1)[1]
        next_marker = "\n======================================================================\n  "
        suffix = ""
        if next_marker in rest:
            suffix = next_marker + rest.split(next_marker, 1)[1]
        return f"{prefix}{start}{body.rstrip()}\n{suffix.lstrip()}"
    return f"{text.rstrip()}{start}{body.rstrip()}\n"


def _update_round_final_diagnostics(round_dir: Path, digest: dict[str, Any]) -> None:
    path = round_dir / "round_final_diagnostics.txt"
    if not path.exists():
        return
    root = dict(digest.get("root_cause_layer_attribution") or {})
    train = dict(digest.get("train") or {})
    holdout = dict(digest.get("holdout") or {})
    paths = dict(digest.get("artifact_paths") or {})
    body = "\n".join(
        [
            "",
            f"Version: {digest.get('reranker_version')}",
            f"Usage: {digest.get('usage_contract')}",
            f"Primary bottleneck: {root.get('primary_bottleneck', 'review')}",
            f"Candidate surfacing: {root.get('candidate_surfacing', 'review')}",
            f"Candidate selection: {root.get('candidate_selection', 'review')}",
            f"Entry route: {root.get('entry_route', 'review')}",
            f"Exit/path management: {root.get('exit_path_management', 'review')}",
            f"Holdout validation: {root.get('holdout_validation', 'review')}",
            "",
            f"Train reranker: rows={train.get('row_count', 0)}, route_outcomes={train.get('route_outcome_count', 0)}, positive_top_replacement_days={train.get('positive_top_replacement_days', 0)}, top_ranked_replacement_total_r={_num(train.get('top_ranked_same_day_replacement_total_r')):.2f}, marginal_slot_total_r={_num(train.get('top_ranked_marginal_slot_replacement_total_r')):.2f}.",
            f"Holdout reranker: rows={holdout.get('row_count', 0)}, route_outcomes={holdout.get('route_outcome_count', 0)}, positive_top_replacement_days={holdout.get('positive_top_replacement_days', 0)}, top_ranked_replacement_total_r={_num(holdout.get('top_ranked_same_day_replacement_total_r')):.2f}, marginal_slot_total_r={_num(holdout.get('top_ranked_marginal_slot_replacement_total_r')):.2f}.",
            "",
            "Artifact pointers:",
            f"  train_jsonl={paths.get('train_jsonl', '')}",
            f"  holdout_jsonl={paths.get('holdout_jsonl', '')}",
            f"  summary_json={paths.get('summary_json', '')}",
            f"  report_md={paths.get('report_md', '')}",
            "",
            "Interpretation: this reranker is a research diagnostic only. It uses ex-post shadow MFE/MAE and same-day replacement labels to separate candidate surfacing, selected-candidate value, route execution, and exit/path-management weakness; it must not be promoted as a live entry objective.",
        ]
    )
    path.write_text(_replace_marked_section(path.read_text(encoding="utf-8"), "16. Shadow-Ledger Same-Day Reranker", body), encoding="utf-8")


def _update_round_research_diagnostics(round_num: int, summary: dict[str, Any]) -> None:
    round_dir = ROUND_ROOT / f"round_{round_num}"
    digest = _reranker_diagnostics_digest(summary)
    for name in ("diagnostics_summary.json", "full_diagnostics_index.json"):
        path = round_dir / name
        if path.exists():
            payload = _read_json(path)
        else:
            payload = {}
        payload["shadow_same_day_reranker"] = digest
        for key, value in dict(digest.get("artifact_paths") or {}).items():
            payload[f"shadow_same_day_reranker_{key}_path"] = value
        _write_json(path, payload)
    _update_round_final_diagnostics(round_dir, digest)

    manifest_path = ROUND_ROOT / "rounds_manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        for row in manifest.get("rounds") or []:
            if int(row.get("round") or 0) != int(round_num):
                continue
            row.setdefault("research_artifacts", {})["shadow_same_day_reranker"] = digest
            row["updated_at_utc"] = _utc_now_iso()
            break
        _write_json(manifest_path, manifest)


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# KALCB Local-Minimum Recovery",
        "",
        "## Sequence",
        "",
    ]
    for item in payload.get("execution_sequence") or []:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Plan",
            "",
            f"- Ablation experiments: {payload['plan_counts'].get('ablation', 0)}",
            f"- Route-family experiments: {payload['plan_counts'].get('routes', 0)}",
            f"- Targeted indicator experiments: {payload['plan_counts'].get('indicators', 0)}",
            f"- Structured auto experiments: {payload['plan_counts'].get('auto', 0)}",
            "",
        ]
    )
    ledger_plan = payload.get("shadow_ledger_plan") or {}
    if ledger_plan:
        outputs = ", ".join(f"`{name}`" for name in ledger_plan.get("outputs", []))
        probes = ", ".join(f"`{name}`" for name in ledger_plan.get("probe_routes", []))
        lines.extend(
            [
                "## Shadow Ledger Plan",
                "",
                f"- Probe routes: {probes}",
                f"- Route families: {', '.join(ledger_plan.get('route_families', []))}",
                f"- Windows: {', '.join(ledger_plan.get('windows', []))}",
                f"- Outputs: {outputs}",
                "",
            ]
        )
    evaluations = payload.get("evaluations") or {}
    if evaluations:
        base = evaluations.get("baseline") or {}
        lines.extend(
            [
                "## Evaluation Baseline",
                "",
                f"- Train net: {_pct((base.get('train') or {}).get('broker_net_return_pct'))}, trades: {_num((base.get('train') or {}).get('trade_count')):.0f}",
                f"- Holdout net: {_pct((base.get('holdout') or {}).get('broker_net_return_pct'))}, trades: {_num((base.get('holdout') or {}).get('trade_count')):.0f}",
                "",
                "## Top Evaluated Candidates",
                "",
            ]
        )
        rows = sorted(evaluations.get("rows") or [], key=lambda item: _num(item.get("score")), reverse=True)
        for row in rows[:12]:
            lines.append(
                f"- `{row['name']}` [{row['stage']}/{row['family']}]: score={_num(row.get('score')):.2f}, survivor={row.get('research_survivor')}, train_delta={_pct((row.get('train_delta') or {}).get('broker_net_return_pct'))}, holdout_delta={_pct((row.get('holdout_delta') or {}).get('broker_net_return_pct'))}, trades={_num((row.get('train') or {}).get('trade_count')):.0f}"
            )
        lines.append("")
    ledgers = payload.get("shadow_ledgers") or {}
    if ledgers:
        lines.extend(["## Shadow Opportunity Ledger", ""])
        for window, summary in ledgers.items():
            family_bits = []
            for family, coverage in (summary.get("route_family_outcome_coverage") or {}).items():
                family_bits.append(f"{family}={int(_num(coverage.get('candidate_count')))}")
            lines.append(
                f"- {window}: outcome_candidates={summary.get('route_outcome_candidate_count')}, same_day_improvers={summary.get('same_day_improver_count')}, route_coverage={'; '.join(family_bits)}"
            )
        lines.append("")
    reranker = payload.get("same_day_reranker") or {}
    if reranker:
        train = reranker.get("train") or {}
        holdout = reranker.get("holdout") or {}
        lines.extend(
            [
                "## Shadow Same-Day Reranker",
                "",
                f"- Version: `{reranker.get('reranker_version')}`",
                f"- Train rows: {train.get('row_count', 0)}, route outcomes: {train.get('route_outcome_count', 0)}, positive top replacement days: {train.get('positive_top_replacement_days', 0)}",
                f"- Holdout rows: {holdout.get('row_count', 0)}, route outcomes: {holdout.get('route_outcome_count', 0)}, positive top replacement days: {holdout.get('positive_top_replacement_days', 0)}",
                f"- Report: `{(reranker.get('artifact_paths') or {}).get('report_md', '')}`",
                "",
            ]
        )
    next_round = payload.get("alpha_conversion_next_round") or {}
    if next_round:
        oracle = next_round.get("full_universe_oracle") or {}
        train_oracle = oracle.get("train") or {}
        holdout_oracle = oracle.get("holdout") or {}
        lines.extend(
            [
                "## Alpha-Conversion Next Round",
                "",
                f"- Conservative seed: `{(next_round.get('seed') or {}).get('name', '')}`",
                f"- Upside challenger: `{(next_round.get('challenger') or {}).get('name', '')}`",
                f"- Train out-of-pool missed days: {train_oracle.get('days_out_of_pool_beats_best_in_pool', 0)}",
                f"- Holdout out-of-pool missed days: {holdout_oracle.get('days_out_of_pool_beats_best_in_pool', 0)}",
                f"- Report: `{(next_round.get('artifact_paths') or {}).get('report_md', '')}`",
                "",
            ]
        )
    candidate_surfacing = payload.get("candidate_surfacing_recovery") or {}
    if candidate_surfacing:
        train = candidate_surfacing.get("train") or {}
        holdout = candidate_surfacing.get("holdout") or {}
        train_recall = (((train.get("recall") or {}).get("best_variant_by_quality_delayed_recall") or {}).get("metrics") or {})
        holdout_recall = (((holdout.get("recall") or {}).get("best_variant_by_quality_delayed_recall") or {}).get("metrics") or {})
        train_replay = ((train.get("replay") or {}).get("best_by_frequency_adjusted_net") or {})
        holdout_replay = ((holdout.get("replay") or {}).get("best_by_frequency_adjusted_net") or {})
        train_matched_replay = ((train.get("matched_incumbent_replay") or {}).get("best_by_frequency_adjusted_net") or {})
        holdout_matched_replay = ((holdout.get("matched_incumbent_replay") or {}).get("best_by_frequency_adjusted_net") or {})
        train_active_replay = ((train.get("matched_active_budget_replay") or {}).get("best_by_frequency_adjusted_net") or {})
        holdout_active_replay = ((holdout.get("matched_active_budget_replay") or {}).get("best_by_frequency_adjusted_net") or {})
        route_bundle = candidate_surfacing.get("route_bundle") or {}
        matched_route_bundle = candidate_surfacing.get("matched_incumbent_route_bundle") or {}
        lines.extend(
            [
                "## Candidate Surfacing Recovery",
                "",
                f"- Version: `{candidate_surfacing.get('version')}`",
                f"- Replay route bundle: `{route_bundle.get('name', '')}`; first30/open enabled: {route_bundle.get('first30_open_enabled')}",
                f"- Matched incumbent bundle: `{matched_route_bundle.get('name', '')}`; first30/open enabled: {matched_route_bundle.get('first30_open_enabled')}",
                f"- Train quality-delayed recall: {_pct(train_recall.get('best_quality_delayed_oracle_in_pool_share'))}; route eligible: {_pct(train_recall.get('route_eligible_share'))}",
                f"- Holdout quality-delayed recall: {_pct(holdout_recall.get('best_quality_delayed_oracle_in_pool_share'))}; route eligible: {_pct(holdout_recall.get('route_eligible_share'))}",
                f"- Train best replay: `{train_replay.get('variant', '')}` net={_pct((train_replay.get('metrics') or {}).get('broker_net_return_pct'))}, trades={_num((train_replay.get('metrics') or {}).get('trade_count')):.0f}",
                f"- Holdout best replay: `{holdout_replay.get('variant', '')}` net={_pct((holdout_replay.get('metrics') or {}).get('broker_net_return_pct'))}, trades={_num((holdout_replay.get('metrics') or {}).get('trade_count')):.0f}",
                f"- Matched train replay: `{train_matched_replay.get('variant', '')}` net={_pct((train_matched_replay.get('metrics') or {}).get('broker_net_return_pct'))}, trades={_num((train_matched_replay.get('metrics') or {}).get('trade_count')):.0f}",
                f"- Matched holdout replay: `{holdout_matched_replay.get('variant', '')}` net={_pct((holdout_matched_replay.get('metrics') or {}).get('broker_net_return_pct'))}, trades={_num((holdout_matched_replay.get('metrics') or {}).get('trade_count')):.0f}",
                f"- Active-budget train replay: `{train_active_replay.get('variant', '')}` net={_pct((train_active_replay.get('metrics') or {}).get('broker_net_return_pct'))}, trades={_num((train_active_replay.get('metrics') or {}).get('trade_count')):.0f}",
                f"- Active-budget holdout replay: `{holdout_active_replay.get('variant', '')}` net={_pct((holdout_active_replay.get('metrics') or {}).get('broker_net_return_pct'))}, trades={_num((holdout_active_replay.get('metrics') or {}).get('trade_count')):.0f}",
                f"- Train route mix: {_route_mix_label(candidate_surfacing, 'train')}",
                f"- Holdout route mix: {_route_mix_label(candidate_surfacing, 'holdout')}",
                f"- Report: `{(candidate_surfacing.get('artifact_paths') or {}).get('report_md', '')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Structured Auto Round",
            "",
            "Use the emitted `structured_auto_round_plan` as the phase contract after ablation and route-family survivors are known.",
            "",
        ]
    )
    return "\n".join(lines)


def selected_experiments(stage: str, experiments: list[RecoveryExperiment]) -> list[RecoveryExperiment]:
    if stage == "all":
        return [experiment for experiment in experiments if experiment.stage in {"ablation", "routes", "indicators", "auto"}]
    if stage in {"ablation", "routes", "indicators", "auto"}:
        return [experiment for experiment in experiments if experiment.stage == stage]
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the KALCB local-minimum recovery workflow.")
    parser.add_argument("--round", type=int, default=5, help="Optimized round to recover from.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--stage", choices=("plan", "ablation", "routes", "ledger", "same_day_rerank", "next_round", "candidate_surfacing", "structural_campaign_surfacing", "indicators", "auto", "all"), default="plan")
    parser.add_argument("--max-evals", type=int, default=None, help="Cap evaluated candidates for smoke/resume runs.")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Write plan artifacts without replay evaluation.")
    parser.add_argument("--ledger-windows", choices=("train", "holdout", "both"), default="both")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rounds = {round_num: _load_round_mutations(round_num) for round_num in (2, 3, 4, args.round)}
    base = rounds[args.round]
    experiments = [
        *build_ablation_experiments(rounds, base),
        *build_route_family_experiments(base),
        *build_indicator_experiments(),
        *build_structured_auto_experiments(base),
    ]
    payload: dict[str, Any] = {
        "strategy": "kalcb",
        "round": args.round,
        "stage": args.stage,
        "dry_run": bool(args.dry_run),
        "execution_sequence": list(EXECUTION_SEQUENCE),
        "base_mutation_count": len(base),
        "plan_counts": dict(Counter(experiment.stage for experiment in experiments)),
        "experiments": [asdict(experiment) for experiment in experiments],
        "shadow_ledger_plan": shadow_ledger_plan(),
        "structured_auto_round_plan": structured_auto_round_plan(),
    }
    run_eval = args.stage in {"ablation", "routes", "indicators", "auto", "all"} and not args.dry_run
    run_ledger = args.stage in {"ledger", "same_day_rerank", "all"} and not args.dry_run
    run_reranker = args.stage in {"same_day_rerank", "all"} and not args.dry_run
    run_next_round = args.stage in {"next_round", "all"} and not args.dry_run
    run_candidate_surfacing = args.stage in {"candidate_surfacing", "all"} and not args.dry_run
    run_structural_campaign = args.stage in {"structural_campaign_surfacing", "all"} and not args.dry_run
    evaluator: RecoveryEvaluator | None = None
    if run_eval or run_ledger or run_next_round or run_candidate_surfacing or run_structural_campaign:
        evaluator = RecoveryEvaluator(args.config, args.output_dir, _round_source_ref(args.round), max_workers=args.max_workers)
    if run_eval and evaluator is not None:
        payload["evaluations"] = run_evaluations(
            evaluator,
            base,
            selected_experiments(args.stage, experiments),
            max_evals=args.max_evals,
        )
    if run_ledger and evaluator is not None:
        windows = ("train", "holdout") if run_reranker or args.ledger_windows == "both" else (args.ledger_windows,)
        payload["shadow_ledgers"] = {window: build_shadow_ledger(evaluator, base, args.output_dir, window=window) for window in windows}
    if run_reranker:
        payload["same_day_reranker"] = build_same_day_reranker_stage(args.output_dir, base, round_num=args.round)
    if run_next_round and evaluator is not None:
        payload["alpha_conversion_next_round"] = build_alpha_conversion_next_round_stage(
            evaluator,
            base,
            args.output_dir,
            round_num=args.round,
            max_evals=args.max_evals,
        )
    if run_candidate_surfacing and evaluator is not None:
        payload["candidate_surfacing_recovery"] = _candidate_surfacing_digest(
            build_candidate_surfacing_recovery_stage(
                evaluator,
                base,
                args.output_dir,
                round_num=args.round,
                max_evals=args.max_evals,
            )
        )
    if run_structural_campaign and evaluator is not None:
        payload["structural_campaign_surfacing"] = _structural_campaign_digest(
            build_structural_campaign_surfacing_stage(
                evaluator,
                base,
                args.output_dir,
                round_num=args.round,
            )
        )
    output_json = args.output_dir / "kalcb_local_minimum_recovery.json"
    output_md = args.output_dir / "kalcb_local_minimum_recovery.md"
    _write_json(output_json, payload)
    output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"result_path": str(output_json), "summary_path": str(output_md), "plan_counts": payload["plan_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
