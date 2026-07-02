from __future__ import annotations

from dataclasses import dataclass, fields, replace
from datetime import time
from enum import Enum
from typing import Any


STRATEGY_ID = "KALCB"
KALCB_CORE_VERSION = "kalcb-core-v13"


class CarryMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    STRICT_LIVE = "strict_live"


SUPPORTED_ENTRY_PLAN_MODES = {
    "breakout",
    "first30_open",
    "opening_drive",
    "post_or_momentum",
    "or_breakout",
    "pdh_breakout",
    "combined_breakout",
    "avwap_reclaim",
    "pullback_acceptance",
    "or_mid_reclaim",
    "or_high_reclaim",
    "pdh_reclaim",
    "deferred_continuation",
}

_SUPPORTED_RECLAIM_LEVEL_SOURCES = {
    "legacy",
    "session_vwap",
    "or_high",
    "or_mid",
    "pdh",
    "campaign_avwap",
    "campaign_box_high",
    "campaign_box_mid",
    "campaign_breakout_level",
}


@dataclass(frozen=True, slots=True)
class KALCBConfig:
    """Live-first Korean ALCB-style continuation configuration.

    Defaults are seeded from the accepted stock ALCB round-3 profile and then
    constrained to KRX/KIS live parity: completed 5m bars, next-5m-open fills,
    no auction fills, and a small shared websocket allocation.
    """

    strategy_id: str = STRATEGY_ID
    market: str = "krx"
    timezone: str = "Asia/Seoul"
    timeframe: str = "5m"
    execution_timeframe: str = "5m_next_open"
    live_parity_fill_timing: str = "next_5m_open"
    auction_mode: str = "non_auction_continuous"

    session_open: time = time(9, 0)
    session_close: time = time(15, 30)
    opening_range_bars: int = 6
    entry_window_start: time = time(9, 30)
    entry_window_end: time = time(12, 0)
    flatten_time: time = time(15, 20)

    ws_budget: int = 10
    ws_max_registrations: int = 40
    ws_reserved_execution_regs: int = 1
    ws_hot_regs_per_symbol: int = 1
    rest_min_interval_paper_s: float = 0.50
    rest_min_interval_live_s: float = 0.07
    rest_egw00201_cooldown_s: float = 30.0

    frontier_enabled: bool = True
    frontier_size: int = 103
    frontier_selection_mode: str = "opportunity"
    frontier_active_selection_mode: str = "liquidity"
    frontier_rest_safety_fraction: float = 0.50
    frontier_shadow_enabled: bool = True
    frontier_shadow_max_positions: int = 128
    frontier_rotation_enabled: bool = True
    frontier_rotation_slots: int = 2
    frontier_rotation_min_shadow_trades: int = 2
    frontier_rotation_min_avg_r: float = 0.05
    frontier_rotation_min_total_r: float = 0.0
    frontier_rotation_min_frontier_trades: int = 40
    frontier_rotation_min_frontier_avg_r: float = 0.05
    frontier_rotation_min_frontier_total_r: float = 2.0
    frontier_rotation_min_proof_symbols: int = 2

    research_top_long_count: int = 20
    research_min_price_krw: float = 1_000.0
    research_min_adv20_krw: float = 2_000_000_000.0
    research_min_history_days: int = 60
    research_weight_relative_strength: float = 0.25
    research_weight_daily_trend: float = 0.20
    research_weight_compression: float = 0.15
    research_weight_accumulation: float = 0.15
    research_weight_stock_regime: float = 0.10
    research_weight_sector_regime: float = 0.08
    research_weight_sector_participation: float = 0.07
    research_min_rs_percentile: float = 0.0
    research_min_trend_score: float = 0.0
    research_min_compression_score: float = 0.0
    research_min_accumulation_score: float = -1.0
    research_min_sector_participation: float = 0.0
    research_min_sector_daily_score_pct: float = 0.0
    research_max_box_range_pct: float = 0.0
    research_structural_frontier_count: int = 0
    research_min_structural_campaign_score: float = 0.0

    rvol_threshold: float = 2.0
    rvol_max: float = 5.0
    cpr_threshold: float = 0.6
    cpr_relax_threshold: float = 0.0
    cpr_relax_min_score: int = 0
    momentum_score_min: int = 2
    adx_threshold: float = 20.0
    or_breakout_min_rvol: float = 0.0
    pdh_breakout_min_rvol: float = 0.0
    or_width_min_pct: float = 0.0015
    or_width_max_pct: float = 0.0
    breakout_distance_cap_r: float = 1.0
    orb_entry_range_cap_r: float = 1.1
    combined_breakout_score_min: int = 5
    combined_breakout_min_rvol: float = 2.5
    combined_avwap_cap_pct: float = 0.003
    pdh_avwap_cap_pct: float = 0.005
    pdh_entry_window_end: time = time(12, 0)
    or_caution_window_start: time = time(10, 30)
    or_caution_window_end: time = time(11, 30)
    or_caution_min_rvol: float = 0.0
    or_caution_max_avwap_dist_pct: float = 0.0
    or_caution_require_two_close: bool = False
    secondary_rank_start: int = 0
    secondary_min_rvol: float = 0.0
    secondary_min_score: int = 0
    secondary_require_pdh_or_late: bool = False
    secondary_late_time: time = time(11, 30)
    block_combined_regime_b: bool = True
    entry_score_blocklist: tuple[str, ...] = ("KRX_COMBINED_BREAKOUT:5",)
    entry_score_size_mults: dict[str, float] | None = None
    entry_detail_blocklist: tuple[str, ...] = ()
    entry_detail_size_mults: dict[str, float] | None = None
    entry_plan_mode: str = "breakout"
    entry_plan_routes: tuple[dict[str, Any], ...] = ()
    entry_plan_max_signal_bars: int = 1
    entry_plan_after_bar: int = 0
    entry_plan_min_bar_ret: float = -9.99
    entry_plan_min_vwap_ret: float = -9.99
    entry_plan_min_breakout_pct: float = 0.0
    entry_plan_max_pullback_from_vwap_pct: float = 0.01
    entry_plan_min_reclaim_ret: float = -9.99
    entry_plan_min_reclaim_closes: int = 1
    entry_plan_min_close_location: float = 0.0
    entry_plan_min_or_position: float = 0.0
    entry_plan_max_avwap_extension_pct: float = 9.99
    entry_plan_gap_min_pct: float = -9.99
    entry_plan_gap_max_pct: float = 9.99
    entry_plan_require_above_prev_close: bool = False
    entry_plan_min_first30_rel_volume: float = 0.0
    entry_plan_min_first30_signal_cpr: float = 0.0
    entry_plan_min_first30_open_drawdown: float = -9.99
    entry_plan_min_first30_low_vs_prev_close: float = -9.99
    entry_plan_min_first30_range_atr: float = 0.0
    entry_plan_max_first30_range_atr: float = 99.0
    entry_plan_min_frontier_score: float = -9.99
    entry_plan_max_frontier_rank: int = 0
    entry_plan_require_initial_active: bool = False
    entry_plan_min_flow_score: float = -9.99
    entry_plan_min_accumulation_score: float = -9.99
    entry_plan_min_quality_votes: int = 0
    entry_plan_quality_min_bar_ret: float = -9.99
    entry_plan_quality_min_first30_signal_cpr: float = -9.99
    entry_plan_quality_min_first30_rel_volume: float = -9.99
    entry_plan_quality_min_first30_range_atr: float = -9.99
    entry_plan_quality_max_first30_range_atr: float = 0.0
    entry_plan_quality_min_flow_score: float = -9.99
    entry_plan_quality_min_accumulation_score: float = -9.99
    entry_plan_quality_max_frontier_rank: int = 0
    entry_plan_route_risk_mult: float = 1.0
    entry_plan_route_notional_mult: float = 1.0
    entry_plan_route_participation_mult: float = 1.0
    entry_plan_route_max_session_trades: int = 0
    entry_plan_route_context_min: dict[str, float] | None = None
    entry_plan_route_context_max: dict[str, float] | None = None
    entry_plan_route_context_exclude: dict[str, tuple[str, ...]] | None = None
    entry_plan_reclaim_level_source: str = "legacy"
    entry_plan_frontier_branch_universe: bool = False
    fast_replay_suppress_rejections: bool = False

    stop_atr_multiple: float = 0.8
    risk_per_trade_pct: float = 0.0010
    max_position_notional_pct: float = 0.10
    max_positions: int = 6
    max_per_sector: int = 3
    heat_cap_r: float = 4.0
    max_participation_30m: float = 0.01
    intraday_leverage: float = 1.0
    pdh_size_mult: float = 0.75
    regime_mult_a: float = 1.0
    regime_mult_b: float = 0.7
    regime_mult_c: float = 0.0
    risk_dynamic_notional_enabled: bool = False
    risk_dynamic_max_position_notional_pct: float = 0.0
    risk_dynamic_max_drawdown_pct: float = 0.0
    risk_dynamic_min_session_return_pct: float = -9.99
    risk_dynamic_max_open_positions: int = 0
    risk_dynamic_max_open_notional_pct: float = 0.0

    use_partial_takes: bool = False
    partial_r_trigger: float = 1.25
    partial_fraction: float = 0.33
    partial_stop_to_breakeven: bool = True
    partial_breakeven_buffer_r: float = 0.10
    quick_exit_enabled: bool = True
    quick_exit_bars: int = 10
    quick_exit_min_r: float = -0.5
    failure_stop_enabled: bool = True
    failure_stop_bars: int = 10
    failure_stop_mfe_max_r: float = 0.2
    failure_stop_current_r_max: float = 0.0
    failure_stop_to_r: float = -0.25
    failure_stop_close_buffer_pct: float = 0.0005
    mfe_conviction_enabled: bool = True
    mfe_conviction_check_bars: int = 16
    mfe_conviction_min_r: float = 0.2
    mfe_conviction_floor_r: float = -0.15
    adaptive_trail_enabled: bool = True
    adaptive_trail_start_bars: int = 25
    adaptive_trail_tighten_bars: int = 25
    adaptive_trail_mid_activate_r: float = 0.20
    adaptive_trail_mid_distance_r: float = 0.40
    adaptive_trail_late_activate_r: float = 0.22
    adaptive_trail_late_distance_r: float = 0.12
    flow_reversal_enabled: bool = True
    flow_reversal_min_hold_bars: int = 12
    flow_reversal_cpr_threshold: float = 0.3
    flow_reversal_mfe_grace_r: float = 0.2
    flow_reversal_trailing_activate_r: float = 0.0
    flow_reversal_trailing_distance_r: float = 0.3
    exit_hard_stop_enabled: bool = True
    exit_stop_mode: str = "momentum"
    exit_stop_pct: float = 0.006
    exit_target_r: float = 0.0
    exit_breakeven_trigger_r: float = 0.0
    exit_breakeven_stop_r: float = 0.0
    exit_trail_start_r: float = 0.0
    exit_trail_gap_r: float = 0.0
    exit_no_mfe_bars: int = 0
    exit_no_mfe_thresh_r: float = 0.0
    exit_failed_followthrough_bars: int = 0
    exit_failed_followthrough_mfe_r: float = 0.0
    exit_failed_followthrough_close_r: float = 0.0
    exit_failed_followthrough_persistent: bool = False
    exit_vwap_fail_bars: int = 0
    exit_vwap_fail_pct: float = 0.0
    exit_vwap_fail_after_mfe_r: float = 0.0
    exit_mfe_giveback_enabled: bool = False
    exit_mfe_giveback_start_r: float = 0.0
    exit_mfe_giveback_gap_r: float = 0.0
    exit_mfe_giveback_min_hold_bars: int = 0
    exit_mfe_floor_enabled: bool = False
    exit_mfe_floor_start_r: float = 0.0
    exit_mfe_floor_floor_r: float = 0.0
    exit_mfe_floor_min_hold_bars: int = 0
    exit_mfe_floor_min_frontier_rank: int = 0
    exit_mfe_floor_max_frontier_rank: int = 0
    exit_mfe_floor_max_first30_signal_cpr: float = 0.0
    exit_mfe_floor_max_first30_rel_volume: float = 0.0
    exit_mfe_floor_max_first30_low_vs_prev_close: float = -9.99
    exit_mfe_floor_max_first30_ret: float = -9.99
    exit_mfe_floor_max_first30_range_close_location: float = 0.0
    exit_mfe_floor_entry_routes: tuple[str, ...] = ()
    exit_mfe_floor_entry_route_modes: tuple[str, ...] = ()
    exit_conditional_target_enabled: bool = False
    exit_conditional_target_r: float = 0.0
    exit_conditional_target_min_hold_bars: int = 0
    exit_conditional_target_min_frontier_rank: int = 0
    exit_conditional_target_max_frontier_rank: int = 0
    exit_conditional_target_min_first30_rel_volume: float = 0.0
    exit_conditional_target_max_first30_rel_volume: float = 0.0
    exit_conditional_target_min_first30_signal_cpr: float = 0.0
    exit_conditional_target_max_first30_signal_cpr: float = 0.0
    exit_conditional_target_entry_routes: tuple[str, ...] = ()
    exit_conditional_target_entry_route_modes: tuple[str, ...] = ()
    exit_path_quality_enabled: bool = False
    exit_path_quality_min_hold_bars: int = 0
    exit_path_quality_max_hold_bars: int = 0
    exit_path_quality_min_mfe_r: float = 0.0
    exit_path_quality_min_giveback_r: float = 0.0
    exit_path_quality_min: dict[str, float] | None = None
    exit_path_quality_max: dict[str, float] | None = None
    exit_path_quality_entry_routes: tuple[str, ...] = ()
    exit_path_quality_entry_route_modes: tuple[str, ...] = ()
    exit_late_giveback_start_bars: int = 0
    exit_late_giveback_start_r: float = 0.0
    exit_late_giveback_gap_r: float = 0.0
    exit_time_decay_bars: int = 0
    exit_time_decay_min_mfe_r: float = 0.0
    exit_time_decay_max_current_r: float = 0.0
    exit_conditional_stop_activate_r: float = 0.0
    exit_conditional_stop_gap_r: float = 0.0
    exit_conditional_stop_min_hold_bars: int = 0
    exit_shadow_failed_followthrough_bars: int = 0
    exit_shadow_failed_followthrough_mfe_r: float = 0.0
    exit_shadow_failed_followthrough_close_r: float = 0.0
    exit_shadow_failed_followthrough_persistent: bool = False
    exit_max_hold_bars: int = 0

    carry_mode: CarryMode = CarryMode.SHADOW
    carry_min_cpr: float = 0.6
    carry_min_r: float = 0.5

    commission_bps: float = 2.0
    slippage_bps: float = 3.0
    tax_bps_on_sell: float = 18.0

    def __post_init__(self) -> None:
        if self.strategy_id.upper() != STRATEGY_ID:
            raise ValueError("KALCBConfig.strategy_id must be KALCB")
        if self.timeframe.lower() != "5m":
            raise ValueError("KALCB requires completed 5m signal bars")
        if self.execution_timeframe != "5m_next_open":
            raise ValueError("KALCB official live-parity execution timeframe must be '5m_next_open'")
        if self.live_parity_fill_timing != "next_5m_open":
            raise ValueError("KALCB only supports live_parity_fill_timing='next_5m_open'")
        if self.auction_mode != "non_auction_continuous":
            raise ValueError("KALCB first production profile excludes auction fills")
        if self.opening_range_bars != 6:
            raise ValueError("KALCB opening range must start with exactly six 5m bars")
        if self.entry_plan_mode not in SUPPORTED_ENTRY_PLAN_MODES:
            raise ValueError(f"Unsupported KALCB entry_plan_mode: {self.entry_plan_mode}")
        for index, route in enumerate(self.entry_plan_routes):
            if not isinstance(route, dict):
                raise ValueError("KALCB entry_plan_routes must contain mappings")
            mode = str(route.get("mode") or route.get("plan_mode") or route.get("entry_plan_mode") or self.entry_plan_mode)
            if mode not in SUPPORTED_ENTRY_PLAN_MODES:
                raise ValueError(f"Unsupported KALCB entry_plan_routes[{index}] mode: {mode}")
            if "priority" in route:
                int(route["priority"])
        if self.ws_budget < 1:
            raise ValueError("KALCB ws_budget must leave room for at least one active symbol")
        if self.ws_max_registrations <= self.ws_reserved_execution_regs:
            raise ValueError("ws_max_registrations must exceed reserved execution registrations")
        if self.ws_budget * max(self.ws_hot_regs_per_symbol, 1) > self.ws_max_registrations - self.ws_reserved_execution_regs:
            raise ValueError("KALCB ws_budget exceeds the shared KIS websocket registration budget")
        if self.frontier_enabled:
            if self.frontier_size < self.ws_budget:
                raise ValueError("KALCB frontier_size must be at least ws_budget")
            paper_capacity = int((5 * 60 / max(self.rest_min_interval_paper_s, 1e-9)) * max(min(self.frontier_rest_safety_fraction, 1.0), 0.01))
            if self.frontier_size > max(1, paper_capacity):
                raise ValueError("KALCB frontier_size exceeds conservative KIS paper REST poll capacity per completed 5m bar")
        if self.frontier_rotation_slots < 0 or self.frontier_rotation_slots > self.ws_budget:
            raise ValueError("KALCB frontier_rotation_slots must fit inside ws_budget")
        if self.frontier_shadow_max_positions < 1:
            raise ValueError("KALCB frontier_shadow_max_positions must be positive")
        if self.frontier_rotation_min_proof_symbols < 1:
            raise ValueError("KALCB frontier_rotation_min_proof_symbols must be positive")
        if self.research_top_long_count < 1:
            raise ValueError("KALCB research_top_long_count must be positive")
        if self.research_structural_frontier_count < 0:
            raise ValueError("KALCB research_structural_frontier_count cannot be negative")
        if self.research_min_history_days < 20:
            raise ValueError("KALCB research_min_history_days must be at least 20")
        if sum(_research_weight_values(self)) <= 0:
            raise ValueError("KALCB research score weights must include at least one positive value")
        if self.research_min_accumulation_score < -1.0 or self.research_min_accumulation_score > 1.0:
            raise ValueError("KALCB research_min_accumulation_score must be between -1 and 1")
        if self.research_min_structural_campaign_score < 0.0 or self.research_min_structural_campaign_score > 10.0:
            raise ValueError("KALCB research_min_structural_campaign_score must be between 0 and 10")
        if self.use_partial_takes:
            if self.partial_r_trigger <= 0:
                raise ValueError("partial_r_trigger must be positive when partial takes are enabled")
            if not (0.0 < self.partial_fraction < 1.0):
                raise ValueError("partial_fraction must be between 0 and 1 when partial takes are enabled")
            if self.partial_breakeven_buffer_r < 0:
                raise ValueError("partial_breakeven_buffer_r cannot be negative")
        if self.entry_plan_max_frontier_rank < 0:
            raise ValueError("entry_plan_max_frontier_rank cannot be negative")
        if self.entry_plan_min_quality_votes < 0:
            raise ValueError("entry_plan_min_quality_votes cannot be negative")
        if self.entry_plan_quality_max_frontier_rank < 0:
            raise ValueError("entry_plan_quality_max_frontier_rank cannot be negative")
        if self.entry_plan_min_reclaim_closes < 1:
            raise ValueError("entry_plan_min_reclaim_closes must be at least 1")
        if self.entry_plan_route_risk_mult < 0:
            raise ValueError("entry_plan_route_risk_mult cannot be negative")
        if self.entry_plan_route_notional_mult < 0:
            raise ValueError("entry_plan_route_notional_mult cannot be negative")
        if self.entry_plan_route_participation_mult < 0:
            raise ValueError("entry_plan_route_participation_mult cannot be negative")
        if self.entry_plan_route_max_session_trades < 0:
            raise ValueError("entry_plan_route_max_session_trades cannot be negative")
        _validate_float_mapping(self.entry_plan_route_context_min, "entry_plan_route_context_min")
        _validate_float_mapping(self.entry_plan_route_context_max, "entry_plan_route_context_max")
        _validate_string_mapping(self.entry_plan_route_context_exclude, "entry_plan_route_context_exclude")
        if str(self.entry_plan_reclaim_level_source or "legacy") not in _SUPPORTED_RECLAIM_LEVEL_SOURCES:
            raise ValueError(f"Unsupported KALCB entry_plan_reclaim_level_source: {self.entry_plan_reclaim_level_source}")
        if self.entry_plan_quality_max_first30_range_atr > 0 and self.entry_plan_quality_min_first30_range_atr > -9.0:
            if self.entry_plan_quality_min_first30_range_atr > self.entry_plan_quality_max_first30_range_atr:
                raise ValueError("quality first30 range_atr min cannot exceed max")
        if self.entry_plan_min_first30_range_atr < 0 or self.entry_plan_max_first30_range_atr <= 0:
            raise ValueError("first30 range_atr thresholds must be positive")
        if self.entry_plan_min_first30_range_atr > self.entry_plan_max_first30_range_atr:
            raise ValueError("entry_plan_min_first30_range_atr cannot exceed entry_plan_max_first30_range_atr")
        if self.exit_mfe_giveback_enabled:
            if self.exit_mfe_giveback_start_r <= 0 or self.exit_mfe_giveback_gap_r <= 0:
                raise ValueError("mfe giveback requires positive start_r and gap_r")
            if self.exit_mfe_giveback_min_hold_bars < 0:
                raise ValueError("exit_mfe_giveback_min_hold_bars cannot be negative")
        if self.exit_mfe_floor_enabled:
            if self.exit_mfe_floor_start_r <= 0:
                raise ValueError("mfe floor requires positive start_r")
            if self.exit_mfe_floor_min_hold_bars < 0:
                raise ValueError("exit_mfe_floor_min_hold_bars cannot be negative")
            if self.exit_mfe_floor_min_frontier_rank < 0 or self.exit_mfe_floor_max_frontier_rank < 0:
                raise ValueError("mfe floor frontier-rank filters cannot be negative")
            if (
                self.exit_mfe_floor_min_frontier_rank > 0
                and self.exit_mfe_floor_max_frontier_rank > 0
                and self.exit_mfe_floor_min_frontier_rank > self.exit_mfe_floor_max_frontier_rank
            ):
                raise ValueError("mfe floor min frontier rank cannot exceed max frontier rank")
        if self.risk_dynamic_max_position_notional_pct < 0:
            raise ValueError("risk_dynamic_max_position_notional_pct cannot be negative")
        if self.risk_dynamic_max_drawdown_pct < 0:
            raise ValueError("risk_dynamic_max_drawdown_pct cannot be negative")
        if self.risk_dynamic_max_open_positions < 0:
            raise ValueError("risk_dynamic_max_open_positions cannot be negative")
        if self.risk_dynamic_max_open_notional_pct < 0:
            raise ValueError("risk_dynamic_max_open_notional_pct cannot be negative")
        if self.exit_conditional_target_enabled:
            if self.exit_conditional_target_r <= 0:
                raise ValueError("conditional target requires positive target_r")
            if self.exit_conditional_target_min_hold_bars < 0:
                raise ValueError("exit_conditional_target_min_hold_bars cannot be negative")
            if self.exit_conditional_target_min_frontier_rank < 0 or self.exit_conditional_target_max_frontier_rank < 0:
                raise ValueError("conditional target frontier-rank filters cannot be negative")
            if (
                self.exit_conditional_target_min_frontier_rank > 0
                and self.exit_conditional_target_max_frontier_rank > 0
                and self.exit_conditional_target_min_frontier_rank > self.exit_conditional_target_max_frontier_rank
            ):
                raise ValueError("conditional target min frontier rank cannot exceed max frontier rank")
        if self.exit_path_quality_min_hold_bars < 0 or self.exit_path_quality_max_hold_bars < 0:
            raise ValueError("exit_path_quality hold-bar thresholds cannot be negative")
        if (
            self.exit_path_quality_min_hold_bars > 0
            and self.exit_path_quality_max_hold_bars > 0
            and self.exit_path_quality_min_hold_bars > self.exit_path_quality_max_hold_bars
        ):
            raise ValueError("exit_path_quality_min_hold_bars cannot exceed max_hold_bars")
        if self.exit_path_quality_min_mfe_r < 0 or self.exit_path_quality_min_giveback_r < 0:
            raise ValueError("exit_path_quality MFE/giveback thresholds cannot be negative")
        _validate_float_mapping(self.exit_path_quality_min, "exit_path_quality_min")
        _validate_float_mapping(self.exit_path_quality_max, "exit_path_quality_max")
        if self.exit_late_giveback_start_bars < 0:
            raise ValueError("exit_late_giveback_start_bars cannot be negative")
        if self.exit_late_giveback_start_bars > 0 and (self.exit_late_giveback_start_r <= 0 or self.exit_late_giveback_gap_r <= 0):
            raise ValueError("late giveback requires positive start_r and gap_r when enabled")
        if self.exit_time_decay_bars < 0:
            raise ValueError("exit_time_decay_bars cannot be negative")
        if self.exit_conditional_stop_activate_r > 0 and self.exit_conditional_stop_gap_r <= 0:
            raise ValueError("conditional stop requires a positive gap_r when activate_r is set")
        if self.exit_conditional_stop_min_hold_bars < 0:
            raise ValueError("exit_conditional_stop_min_hold_bars cannot be negative")
        if self.exit_shadow_failed_followthrough_bars < 0:
            raise ValueError("exit_shadow_failed_followthrough_bars cannot be negative")

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any] | None = None,
        mutations: dict[str, Any] | None = None,
    ) -> "KALCBConfig":
        merged: dict[str, Any] = {}
        for source in (data or {}, mutations or {}):
            merged.update(_flatten_config(source))
        allowed = {field.name for field in fields(cls)}
        kwargs = {key: value for key, value in merged.items() if key in allowed}
        for key in (
            "session_open",
            "session_close",
            "entry_window_start",
            "entry_window_end",
            "flatten_time",
            "pdh_entry_window_end",
            "or_caution_window_start",
            "or_caution_window_end",
            "secondary_late_time",
        ):
            if key in kwargs:
                kwargs[key] = _coerce_time(kwargs[key])
        if "carry_mode" in kwargs and not isinstance(kwargs["carry_mode"], CarryMode):
            kwargs["carry_mode"] = CarryMode(str(kwargs["carry_mode"]).lower())
        if "entry_score_blocklist" in kwargs:
            kwargs["entry_score_blocklist"] = _coerce_tuple(kwargs["entry_score_blocklist"])
        if "entry_detail_blocklist" in kwargs:
            kwargs["entry_detail_blocklist"] = _coerce_tuple(kwargs["entry_detail_blocklist"])
        if "exit_mfe_floor_entry_routes" in kwargs:
            kwargs["exit_mfe_floor_entry_routes"] = _coerce_tuple(kwargs["exit_mfe_floor_entry_routes"])
        if "exit_mfe_floor_entry_route_modes" in kwargs:
            kwargs["exit_mfe_floor_entry_route_modes"] = _coerce_tuple(kwargs["exit_mfe_floor_entry_route_modes"])
        if "exit_conditional_target_entry_routes" in kwargs:
            kwargs["exit_conditional_target_entry_routes"] = _coerce_tuple(kwargs["exit_conditional_target_entry_routes"])
        if "exit_conditional_target_entry_route_modes" in kwargs:
            kwargs["exit_conditional_target_entry_route_modes"] = _coerce_tuple(kwargs["exit_conditional_target_entry_route_modes"])
        if "exit_path_quality_entry_routes" in kwargs:
            kwargs["exit_path_quality_entry_routes"] = _coerce_tuple(kwargs["exit_path_quality_entry_routes"])
        if "exit_path_quality_entry_route_modes" in kwargs:
            kwargs["exit_path_quality_entry_route_modes"] = _coerce_tuple(kwargs["exit_path_quality_entry_route_modes"])
        if "entry_plan_route_context_min" in kwargs:
            kwargs["entry_plan_route_context_min"] = _coerce_float_mapping(kwargs["entry_plan_route_context_min"])
        if "entry_plan_route_context_max" in kwargs:
            kwargs["entry_plan_route_context_max"] = _coerce_float_mapping(kwargs["entry_plan_route_context_max"])
        if "entry_plan_route_context_exclude" in kwargs:
            kwargs["entry_plan_route_context_exclude"] = _coerce_string_mapping(kwargs["entry_plan_route_context_exclude"])
        if "exit_path_quality_min" in kwargs:
            kwargs["exit_path_quality_min"] = _coerce_float_mapping(kwargs["exit_path_quality_min"])
        if "exit_path_quality_max" in kwargs:
            kwargs["exit_path_quality_max"] = _coerce_float_mapping(kwargs["exit_path_quality_max"])
        if "entry_plan_routes" in kwargs:
            kwargs["entry_plan_routes"] = _coerce_entry_routes(kwargs["entry_plan_routes"])
        return cls(**kwargs)

    def with_mutations(self, mutations: dict[str, Any] | None = None) -> "KALCBConfig":
        if not mutations:
            return self
        updated = self.from_mapping(mutations)
        flattened = _flatten_config(mutations)
        values = {field.name: getattr(self, field.name) for field in fields(self)}
        values.update({field.name: getattr(updated, field.name) for field in fields(updated) if field.name in flattened})
        return replace(self, **values)


def _research_weight_values(config: KALCBConfig) -> tuple[float, ...]:
    return (
        max(float(config.research_weight_relative_strength), 0.0),
        max(float(config.research_weight_daily_trend), 0.0),
        max(float(config.research_weight_compression), 0.0),
        max(float(config.research_weight_accumulation), 0.0),
        max(float(config.research_weight_stock_regime), 0.0),
        max(float(config.research_weight_sector_regime), 0.0),
        max(float(config.research_weight_sector_participation), 0.0),
    )


_ALIASES = {
    "kalcb.session.open": "session_open",
    "kalcb.session.close": "session_close",
    "kalcb.session.opening_range_bars": "opening_range_bars",
    "kalcb.session.entry_window_start": "entry_window_start",
    "kalcb.session.entry_window_end": "entry_window_end",
    "kalcb.session.flatten_time": "flatten_time",
    "kalcb.session.auction_mode": "auction_mode",
    "kalcb.session.ws_budget": "ws_budget",
    "kalcb.live.ws_budget": "ws_budget",
    "kalcb.live.rest_min_interval_paper_s": "rest_min_interval_paper_s",
    "kalcb.live.rest_min_interval_live_s": "rest_min_interval_live_s",
    "kalcb.frontier.enabled": "frontier_enabled",
    "kalcb.frontier.size": "frontier_size",
    "kalcb.frontier.selection_mode": "frontier_selection_mode",
    "kalcb.frontier.active_selection_mode": "frontier_active_selection_mode",
    "kalcb.frontier.rest_safety_fraction": "frontier_rest_safety_fraction",
    "kalcb.frontier.shadow_enabled": "frontier_shadow_enabled",
    "kalcb.frontier.shadow_max_positions": "frontier_shadow_max_positions",
    "kalcb.frontier.rotation_enabled": "frontier_rotation_enabled",
    "kalcb.frontier.rotation_slots": "frontier_rotation_slots",
    "kalcb.frontier.rotation_min_shadow_trades": "frontier_rotation_min_shadow_trades",
    "kalcb.frontier.rotation_min_avg_r": "frontier_rotation_min_avg_r",
    "kalcb.frontier.rotation_min_total_r": "frontier_rotation_min_total_r",
    "kalcb.frontier.rotation_min_frontier_trades": "frontier_rotation_min_frontier_trades",
    "kalcb.frontier.rotation_min_frontier_avg_r": "frontier_rotation_min_frontier_avg_r",
    "kalcb.frontier.rotation_min_frontier_total_r": "frontier_rotation_min_frontier_total_r",
    "kalcb.frontier.rotation_min_proof_symbols": "frontier_rotation_min_proof_symbols",
    "kalcb.research.top_long_count": "research_top_long_count",
    "kalcb.research.min_price_krw": "research_min_price_krw",
    "kalcb.research.min_adv20_krw": "research_min_adv20_krw",
    "kalcb.research.min_history_days": "research_min_history_days",
    "kalcb.research.weights.relative_strength": "research_weight_relative_strength",
    "kalcb.research.weights.daily_trend": "research_weight_daily_trend",
    "kalcb.research.weights.compression": "research_weight_compression",
    "kalcb.research.weights.accumulation": "research_weight_accumulation",
    "kalcb.research.weights.stock_regime": "research_weight_stock_regime",
    "kalcb.research.weights.sector_regime": "research_weight_sector_regime",
    "kalcb.research.weights.sector_participation": "research_weight_sector_participation",
    "kalcb.research.min_rs_percentile": "research_min_rs_percentile",
    "kalcb.research.min_trend_score": "research_min_trend_score",
    "kalcb.research.min_compression_score": "research_min_compression_score",
    "kalcb.research.min_accumulation_score": "research_min_accumulation_score",
    "kalcb.research.min_sector_participation": "research_min_sector_participation",
    "kalcb.research.min_sector_daily_score_pct": "research_min_sector_daily_score_pct",
    "kalcb.research.max_box_range_pct": "research_max_box_range_pct",
    "kalcb.research.structural_frontier_count": "research_structural_frontier_count",
    "kalcb.research.min_structural_campaign_score": "research_min_structural_campaign_score",
    "kalcb.discovery.frontier_size": "frontier_size",
    "kalcb.discovery.selection_mode": "frontier_selection_mode",
    "kalcb.discovery.active_selection_mode": "frontier_active_selection_mode",
    "kalcb.timeframes.signal": "timeframe",
    "kalcb.timeframes.execution": "execution_timeframe",
    "kalcb.timeframes.live_parity_fill_timing": "live_parity_fill_timing",
    "kalcb.entry.rvol_threshold": "rvol_threshold",
    "kalcb.entry.rvol_max": "rvol_max",
    "kalcb.entry.cpr_threshold": "cpr_threshold",
    "kalcb.entry.cpr_relax_threshold": "cpr_relax_threshold",
    "kalcb.entry.cpr_relax_min_score": "cpr_relax_min_score",
    "kalcb.entry.momentum_score_min": "momentum_score_min",
    "kalcb.entry.adx_threshold": "adx_threshold",
    "kalcb.entry.or_breakout_min_rvol": "or_breakout_min_rvol",
    "kalcb.entry.pdh_breakout_min_rvol": "pdh_breakout_min_rvol",
    "kalcb.entry.or_width_min_pct": "or_width_min_pct",
    "kalcb.entry.or_width_max_pct": "or_width_max_pct",
    "kalcb.entry.breakout_distance_cap_r": "breakout_distance_cap_r",
    "kalcb.entry.orb_entry_range_cap_r": "orb_entry_range_cap_r",
    "kalcb.entry.combined_breakout_score_min": "combined_breakout_score_min",
    "kalcb.entry.combined_breakout_min_rvol": "combined_breakout_min_rvol",
    "kalcb.entry.combined_avwap_cap_pct": "combined_avwap_cap_pct",
    "kalcb.entry.pdh_avwap_cap_pct": "pdh_avwap_cap_pct",
    "kalcb.entry.pdh_entry_window_end": "pdh_entry_window_end",
    "kalcb.entry.or_caution_window_start": "or_caution_window_start",
    "kalcb.entry.or_caution_window_end": "or_caution_window_end",
    "kalcb.entry.or_caution_min_rvol": "or_caution_min_rvol",
    "kalcb.entry.or_caution_max_avwap_dist_pct": "or_caution_max_avwap_dist_pct",
    "kalcb.entry.or_caution_require_two_close": "or_caution_require_two_close",
    "kalcb.entry.secondary_rank_start": "secondary_rank_start",
    "kalcb.entry.secondary_min_rvol": "secondary_min_rvol",
    "kalcb.entry.secondary_min_score": "secondary_min_score",
    "kalcb.entry.secondary_require_pdh_or_late": "secondary_require_pdh_or_late",
    "kalcb.entry.secondary_late_time": "secondary_late_time",
    "kalcb.entry.block_combined_regime_b": "block_combined_regime_b",
    "kalcb.entry.entry_score_blocklist": "entry_score_blocklist",
    "kalcb.entry.entry_score_size_mults": "entry_score_size_mults",
    "kalcb.entry.entry_detail_blocklist": "entry_detail_blocklist",
    "kalcb.entry.entry_detail_size_mults": "entry_detail_size_mults",
    "kalcb.entry.plan_mode": "entry_plan_mode",
    "kalcb.entry.entry_plan_mode": "entry_plan_mode",
    "kalcb.entry.routes": "entry_plan_routes",
    "kalcb.entry.plan_routes": "entry_plan_routes",
    "kalcb.entry.entry_plan_routes": "entry_plan_routes",
    "kalcb.entry.max_signal_bars": "entry_plan_max_signal_bars",
    "kalcb.entry.after_bar": "entry_plan_after_bar",
    "kalcb.entry.min_bar_ret": "entry_plan_min_bar_ret",
    "kalcb.entry.min_vwap_ret": "entry_plan_min_vwap_ret",
    "kalcb.entry.min_breakout_pct": "entry_plan_min_breakout_pct",
    "kalcb.entry.max_pullback_from_vwap_pct": "entry_plan_max_pullback_from_vwap_pct",
    "kalcb.entry.min_reclaim_ret": "entry_plan_min_reclaim_ret",
    "kalcb.entry.min_reclaim_closes": "entry_plan_min_reclaim_closes",
    "kalcb.entry.reclaim_min_closes": "entry_plan_min_reclaim_closes",
    "kalcb.entry.min_close_location": "entry_plan_min_close_location",
    "kalcb.entry.min_or_position": "entry_plan_min_or_position",
    "kalcb.entry.max_avwap_extension_pct": "entry_plan_max_avwap_extension_pct",
    "kalcb.entry.gap_min_pct": "entry_plan_gap_min_pct",
    "kalcb.entry.gap_max_pct": "entry_plan_gap_max_pct",
    "kalcb.entry.require_above_prev_close": "entry_plan_require_above_prev_close",
    "kalcb.entry.min_first30_rel_volume": "entry_plan_min_first30_rel_volume",
    "kalcb.entry.min_first30_signal_cpr": "entry_plan_min_first30_signal_cpr",
    "kalcb.entry.min_first30_open_drawdown": "entry_plan_min_first30_open_drawdown",
    "kalcb.entry.min_first30_low_vs_prev_close": "entry_plan_min_first30_low_vs_prev_close",
    "kalcb.entry.min_first30_range_atr": "entry_plan_min_first30_range_atr",
    "kalcb.entry.max_first30_range_atr": "entry_plan_max_first30_range_atr",
    "kalcb.entry.min_frontier_score": "entry_plan_min_frontier_score",
    "kalcb.entry.max_frontier_rank": "entry_plan_max_frontier_rank",
    "kalcb.entry.require_initial_active": "entry_plan_require_initial_active",
    "kalcb.entry.min_flow_score": "entry_plan_min_flow_score",
    "kalcb.entry.min_accumulation_score": "entry_plan_min_accumulation_score",
    "kalcb.entry.min_quality_votes": "entry_plan_min_quality_votes",
    "kalcb.entry.quality.min_votes": "entry_plan_min_quality_votes",
    "kalcb.entry.quality_min_bar_ret": "entry_plan_quality_min_bar_ret",
    "kalcb.entry.quality.min_bar_ret": "entry_plan_quality_min_bar_ret",
    "kalcb.entry.quality_min_first30_signal_cpr": "entry_plan_quality_min_first30_signal_cpr",
    "kalcb.entry.quality.min_first30_signal_cpr": "entry_plan_quality_min_first30_signal_cpr",
    "kalcb.entry.quality_min_first30_rel_volume": "entry_plan_quality_min_first30_rel_volume",
    "kalcb.entry.quality.min_first30_rel_volume": "entry_plan_quality_min_first30_rel_volume",
    "kalcb.entry.quality_min_first30_range_atr": "entry_plan_quality_min_first30_range_atr",
    "kalcb.entry.quality.min_first30_range_atr": "entry_plan_quality_min_first30_range_atr",
    "kalcb.entry.quality_max_first30_range_atr": "entry_plan_quality_max_first30_range_atr",
    "kalcb.entry.quality.max_first30_range_atr": "entry_plan_quality_max_first30_range_atr",
    "kalcb.entry.quality_min_flow_score": "entry_plan_quality_min_flow_score",
    "kalcb.entry.quality.min_flow_score": "entry_plan_quality_min_flow_score",
    "kalcb.entry.quality_min_accumulation_score": "entry_plan_quality_min_accumulation_score",
    "kalcb.entry.quality.min_accumulation_score": "entry_plan_quality_min_accumulation_score",
    "kalcb.entry.quality_max_frontier_rank": "entry_plan_quality_max_frontier_rank",
    "kalcb.entry.quality.max_frontier_rank": "entry_plan_quality_max_frontier_rank",
    "kalcb.entry.route_risk_mult": "entry_plan_route_risk_mult",
    "kalcb.entry.risk_mult": "entry_plan_route_risk_mult",
    "kalcb.entry.risk_per_trade_mult": "entry_plan_route_risk_mult",
    "kalcb.entry.route_notional_mult": "entry_plan_route_notional_mult",
    "kalcb.entry.notional_mult": "entry_plan_route_notional_mult",
    "kalcb.entry.route_participation_mult": "entry_plan_route_participation_mult",
    "kalcb.entry.participation_mult": "entry_plan_route_participation_mult",
    "kalcb.entry.route_max_session_trades": "entry_plan_route_max_session_trades",
    "kalcb.entry.max_session_trades": "entry_plan_route_max_session_trades",
    "kalcb.entry.route_context_min": "entry_plan_route_context_min",
    "kalcb.entry.context_min": "entry_plan_route_context_min",
    "kalcb.entry.regime_min": "entry_plan_route_context_min",
    "kalcb.entry.route_context_max": "entry_plan_route_context_max",
    "kalcb.entry.context_max": "entry_plan_route_context_max",
    "kalcb.entry.regime_max": "entry_plan_route_context_max",
    "kalcb.entry.route_context_exclude": "entry_plan_route_context_exclude",
    "kalcb.entry.context_exclude": "entry_plan_route_context_exclude",
    "kalcb.entry.context_not": "entry_plan_route_context_exclude",
    "kalcb.entry.reclaim_level_source": "entry_plan_reclaim_level_source",
    "kalcb.entry.level_source": "entry_plan_reclaim_level_source",
    "kalcb.entry.frontier_branch_universe": "entry_plan_frontier_branch_universe",
    "kalcb.entry.additive_frontier_universe": "entry_plan_frontier_branch_universe",
    "kalcb.entry.fast_replay_suppress_rejections": "fast_replay_suppress_rejections",
    "kalcb.risk.stop_atr_multiple": "stop_atr_multiple",
    "kalcb.risk.risk_per_trade_pct": "risk_per_trade_pct",
    "kalcb.risk.max_position_notional_pct": "max_position_notional_pct",
    "kalcb.risk.max_positions": "max_positions",
    "kalcb.risk.max_per_sector": "max_per_sector",
    "kalcb.risk.heat_cap_r": "heat_cap_r",
    "kalcb.risk.pdh_size_mult": "pdh_size_mult",
    "kalcb.risk.regime_mult_a": "regime_mult_a",
    "kalcb.risk.regime_mult_b": "regime_mult_b",
    "kalcb.risk.regime_mult_c": "regime_mult_c",
    "kalcb.risk.regime_size_multipliers.A": "regime_mult_a",
    "kalcb.risk.regime_size_multipliers.B": "regime_mult_b",
    "kalcb.risk.regime_size_multipliers.C": "regime_mult_c",
    "kalcb.risk.dynamic_notional_enabled": "risk_dynamic_notional_enabled",
    "kalcb.risk.dynamic_enabled": "risk_dynamic_notional_enabled",
    "kalcb.risk.dynamic_max_position_notional_pct": "risk_dynamic_max_position_notional_pct",
    "kalcb.risk.dynamic_notional_pct": "risk_dynamic_max_position_notional_pct",
    "kalcb.risk.dynamic_max_drawdown_pct": "risk_dynamic_max_drawdown_pct",
    "kalcb.risk.dynamic_min_session_return_pct": "risk_dynamic_min_session_return_pct",
    "kalcb.risk.dynamic_max_open_positions": "risk_dynamic_max_open_positions",
    "kalcb.risk.dynamic_max_open_notional_pct": "risk_dynamic_max_open_notional_pct",
    "kalcb.exit.use_partial_takes": "use_partial_takes",
    "kalcb.exit.partial_r_trigger": "partial_r_trigger",
    "kalcb.exit.partial_fraction": "partial_fraction",
    "kalcb.exit.partial_stop_to_breakeven": "partial_stop_to_breakeven",
    "kalcb.exit.partial_breakeven_buffer_r": "partial_breakeven_buffer_r",
    "kalcb.exit.quick_exit_enabled": "quick_exit_enabled",
    "kalcb.exit.quick_exit_bars": "quick_exit_bars",
    "kalcb.exit.quick_exit_min_r": "quick_exit_min_r",
    "kalcb.exit.failure_stop_enabled": "failure_stop_enabled",
    "kalcb.exit.failure_stop_bars": "failure_stop_bars",
    "kalcb.exit.failure_stop_mfe_max_r": "failure_stop_mfe_max_r",
    "kalcb.exit.failure_stop_current_r_max": "failure_stop_current_r_max",
    "kalcb.exit.failure_stop_to_r": "failure_stop_to_r",
    "kalcb.exit.mfe_conviction_enabled": "mfe_conviction_enabled",
    "kalcb.exit.mfe_conviction_check_bars": "mfe_conviction_check_bars",
    "kalcb.exit.mfe_conviction_min_r": "mfe_conviction_min_r",
    "kalcb.exit.mfe_conviction_floor_r": "mfe_conviction_floor_r",
    "kalcb.exit.adaptive_trail_enabled": "adaptive_trail_enabled",
    "kalcb.exit.adaptive_trail_start_bars": "adaptive_trail_start_bars",
    "kalcb.exit.adaptive_trail_tighten_bars": "adaptive_trail_tighten_bars",
    "kalcb.exit.adaptive_trail_mid_activate_r": "adaptive_trail_mid_activate_r",
    "kalcb.exit.adaptive_trail_mid_distance_r": "adaptive_trail_mid_distance_r",
    "kalcb.exit.adaptive_trail_late_activate_r": "adaptive_trail_late_activate_r",
    "kalcb.exit.adaptive_trail_late_distance_r": "adaptive_trail_late_distance_r",
    "kalcb.exit.flow_reversal_enabled": "flow_reversal_enabled",
    "kalcb.exit.flow_reversal_min_hold_bars": "flow_reversal_min_hold_bars",
    "kalcb.exit.flow_reversal_cpr_threshold": "flow_reversal_cpr_threshold",
    "kalcb.exit.flow_reversal_mfe_grace_r": "flow_reversal_mfe_grace_r",
    "kalcb.exit.flow_reversal_trailing_activate_r": "flow_reversal_trailing_activate_r",
    "kalcb.exit.flow_reversal_trailing_distance_r": "flow_reversal_trailing_distance_r",
    "kalcb.exit.hard_stop_enabled": "exit_hard_stop_enabled",
    "kalcb.exit.stop_mode": "exit_stop_mode",
    "kalcb.exit.stop_pct": "exit_stop_pct",
    "kalcb.exit.target_r": "exit_target_r",
    "kalcb.exit.breakeven_trigger_r": "exit_breakeven_trigger_r",
    "kalcb.exit.breakeven_stop_r": "exit_breakeven_stop_r",
    "kalcb.exit.trail_start_r": "exit_trail_start_r",
    "kalcb.exit.trail_gap_r": "exit_trail_gap_r",
    "kalcb.exit.no_mfe_bars": "exit_no_mfe_bars",
    "kalcb.exit.no_mfe_thresh_r": "exit_no_mfe_thresh_r",
    "kalcb.exit.failed_followthrough_bars": "exit_failed_followthrough_bars",
    "kalcb.exit.failed_followthrough_mfe_r": "exit_failed_followthrough_mfe_r",
    "kalcb.exit.failed_followthrough_close_r": "exit_failed_followthrough_close_r",
    "kalcb.exit.failed_followthrough_persistent": "exit_failed_followthrough_persistent",
    "kalcb.exit.vwap_fail_bars": "exit_vwap_fail_bars",
    "kalcb.exit.vwap_fail_pct": "exit_vwap_fail_pct",
    "kalcb.exit.vwap_fail_after_mfe_r": "exit_vwap_fail_after_mfe_r",
    "kalcb.exit.mfe_giveback_enabled": "exit_mfe_giveback_enabled",
    "kalcb.exit.mfe_giveback_start_r": "exit_mfe_giveback_start_r",
    "kalcb.exit.mfe_giveback_gap_r": "exit_mfe_giveback_gap_r",
    "kalcb.exit.mfe_giveback_min_hold_bars": "exit_mfe_giveback_min_hold_bars",
    "kalcb.exit.mfe_floor_enabled": "exit_mfe_floor_enabled",
    "kalcb.exit.mfe_floor_start_r": "exit_mfe_floor_start_r",
    "kalcb.exit.mfe_floor_floor_r": "exit_mfe_floor_floor_r",
    "kalcb.exit.mfe_floor_min_hold_bars": "exit_mfe_floor_min_hold_bars",
    "kalcb.exit.mfe_floor_min_frontier_rank": "exit_mfe_floor_min_frontier_rank",
    "kalcb.exit.mfe_floor_max_frontier_rank": "exit_mfe_floor_max_frontier_rank",
    "kalcb.exit.mfe_floor_max_first30_signal_cpr": "exit_mfe_floor_max_first30_signal_cpr",
    "kalcb.exit.mfe_floor_max_first30_rel_volume": "exit_mfe_floor_max_first30_rel_volume",
    "kalcb.exit.mfe_floor_max_first30_low_vs_prev_close": "exit_mfe_floor_max_first30_low_vs_prev_close",
    "kalcb.exit.mfe_floor_max_first30_ret": "exit_mfe_floor_max_first30_ret",
    "kalcb.exit.mfe_floor_max_first30_range_close_location": "exit_mfe_floor_max_first30_range_close_location",
    "kalcb.exit.mfe_floor_entry_routes": "exit_mfe_floor_entry_routes",
    "kalcb.exit.mfe_floor_entry_route_modes": "exit_mfe_floor_entry_route_modes",
    "kalcb.exit.conditional_target_enabled": "exit_conditional_target_enabled",
    "kalcb.exit.conditional_target_r": "exit_conditional_target_r",
    "kalcb.exit.conditional_target_min_hold_bars": "exit_conditional_target_min_hold_bars",
    "kalcb.exit.conditional_target_min_frontier_rank": "exit_conditional_target_min_frontier_rank",
    "kalcb.exit.conditional_target_max_frontier_rank": "exit_conditional_target_max_frontier_rank",
    "kalcb.exit.conditional_target_min_first30_rel_volume": "exit_conditional_target_min_first30_rel_volume",
    "kalcb.exit.conditional_target_max_first30_rel_volume": "exit_conditional_target_max_first30_rel_volume",
    "kalcb.exit.conditional_target_min_first30_signal_cpr": "exit_conditional_target_min_first30_signal_cpr",
    "kalcb.exit.conditional_target_max_first30_signal_cpr": "exit_conditional_target_max_first30_signal_cpr",
    "kalcb.exit.conditional_target_entry_routes": "exit_conditional_target_entry_routes",
    "kalcb.exit.conditional_target_entry_route_modes": "exit_conditional_target_entry_route_modes",
    "kalcb.exit.path_quality_enabled": "exit_path_quality_enabled",
    "kalcb.exit.path_quality_min_hold_bars": "exit_path_quality_min_hold_bars",
    "kalcb.exit.path_quality_max_hold_bars": "exit_path_quality_max_hold_bars",
    "kalcb.exit.path_quality_min_mfe_r": "exit_path_quality_min_mfe_r",
    "kalcb.exit.path_quality_min_giveback_r": "exit_path_quality_min_giveback_r",
    "kalcb.exit.path_quality_min": "exit_path_quality_min",
    "kalcb.exit.path_quality_max": "exit_path_quality_max",
    "kalcb.exit.path_quality_entry_routes": "exit_path_quality_entry_routes",
    "kalcb.exit.path_quality_entry_route_modes": "exit_path_quality_entry_route_modes",
    "kalcb.exit.path_quality.context_min": "exit_path_quality_min",
    "kalcb.exit.path_quality.context_max": "exit_path_quality_max",
    "kalcb.exit.late_giveback_start_bars": "exit_late_giveback_start_bars",
    "kalcb.exit.late_giveback_start_r": "exit_late_giveback_start_r",
    "kalcb.exit.late_giveback_gap_r": "exit_late_giveback_gap_r",
    "kalcb.exit.time_decay_bars": "exit_time_decay_bars",
    "kalcb.exit.time_decay_min_mfe_r": "exit_time_decay_min_mfe_r",
    "kalcb.exit.time_decay_max_current_r": "exit_time_decay_max_current_r",
    "kalcb.exit.conditional_stop_activate_r": "exit_conditional_stop_activate_r",
    "kalcb.exit.conditional_stop_gap_r": "exit_conditional_stop_gap_r",
    "kalcb.exit.conditional_stop_min_hold_bars": "exit_conditional_stop_min_hold_bars",
    "kalcb.exit.shadow_failed_followthrough_bars": "exit_shadow_failed_followthrough_bars",
    "kalcb.exit.shadow_failed_followthrough_mfe_r": "exit_shadow_failed_followthrough_mfe_r",
    "kalcb.exit.shadow_failed_followthrough_close_r": "exit_shadow_failed_followthrough_close_r",
    "kalcb.exit.shadow_failed_followthrough_persistent": "exit_shadow_failed_followthrough_persistent",
    "kalcb.exit.max_hold_bars": "exit_max_hold_bars",
    "kalcb.carry.mode": "carry_mode",
    "kalcb.carry.min_cpr": "carry_min_cpr",
    "kalcb.carry.min_r": "carry_min_r",
    "kalcb.robustness.slippage_bps": "slippage_bps",
}


def _flatten_config(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in dict(data or {}).items():
        if key == "initial_mutations" and isinstance(value, dict):
            out.update(_flatten_config(value))
            continue
        if str(key) in {"kalcb", "session", "entry", "risk", "exit", "exits", "carry", "live", "timeframes", "frontier", "discovery", "research"} and isinstance(value, dict):
            prefix = "exit" if str(key) == "exits" else str(key)
            out.update(_flatten_nested(value, "" if prefix == "kalcb" else prefix))
            continue
        normalized = _ALIASES.get(str(key), str(key))
        if normalized.startswith("kalcb."):
            normalized = _ALIASES.get(normalized, normalized.split(".")[-1])
        out[normalized] = value
    return out


def _flatten_nested(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        full = f"{prefix}.{key}" if prefix else str(key)
        alias = _ALIASES.get(f"kalcb.{full}")
        if isinstance(value, dict) and alias in {
            "entry_score_size_mults",
            "entry_detail_size_mults",
            "entry_plan_route_context_min",
            "entry_plan_route_context_max",
            "entry_plan_route_context_exclude",
            "exit_path_quality_min",
            "exit_path_quality_max",
        }:
            out[alias] = dict(value)
            continue
        if isinstance(value, dict):
            out.update(_flatten_nested(value, full))
        else:
            out[alias or str(key)] = value
    return out


def _coerce_time(value: Any) -> time:
    if isinstance(value, time):
        return value
    parts = str(value).split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid time value: {value!r}")
    return time(int(parts[0]), int(parts[1]))


def _coerce_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(item) for item in (value or ()))


def _coerce_float_mapping(value: Any) -> dict[str, float] | None:
    if value in (None, "", ()):
        return None
    if not isinstance(value, dict):
        raise ValueError("route context gates must be mappings")
    out: dict[str, float] = {}
    for key, raw in value.items():
        try:
            out[str(key)] = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"route context gate {key!r} must be numeric") from exc
    return out


def _coerce_string_mapping(value: Any) -> dict[str, tuple[str, ...]] | None:
    if value in (None, "", ()):
        return None
    if not isinstance(value, dict):
        raise ValueError("route context exclusion gates must be mappings")
    out: dict[str, tuple[str, ...]] = {}
    for key, raw in value.items():
        values = _coerce_tuple(raw)
        if values:
            out[str(key)] = values
    return out or None


def _validate_float_mapping(value: dict[str, float] | None, name: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    for key, raw in value.items():
        try:
            float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{key!r}] must be numeric") from exc


def _validate_string_mapping(value: dict[str, tuple[str, ...]] | None, name: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    for key, raw in value.items():
        values = _coerce_tuple(raw)
        if not values:
            raise ValueError(f"{name}[{key!r}] must contain at least one excluded value")


def _coerce_entry_routes(value: Any) -> tuple[dict[str, Any], ...]:
    if value in (None, "", ()):
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("entry_plan_routes must be a list of route mappings")
    routes: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("entry_plan_routes must contain route mappings")
        routes.append(dict(item))
    return tuple(routes)
