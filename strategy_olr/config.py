from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any


STRATEGY_ID = "OLR"
OLR_CORE_VERSION = "olr-research-v4"


@dataclass(frozen=True, slots=True)
class OLRConfig:
    """Research-only Korean overnight leader rotation configuration."""

    strategy_id: str = STRATEGY_ID
    market: str = "krx"
    timezone: str = "Asia/Seoul"
    timeframe: str = "5m"

    complete_universe_size: int = 103
    research_top_long_count: int = 30
    frontier_active_selection_mode: str = "hybrid"
    afternoon_top_n: int = 8
    overnight_slot_count: int = 4

    premarket_frontier_size: int = 30
    premarket_min_adv20_krw: float = 0.0
    premarket_min_foreign5_z: float = -9.99

    research_min_price_krw: float = 1_000.0
    research_min_adv20_krw: float = 2_000_000_000.0
    research_min_history_days: int = 60
    research_max_median_spread_pct: float = 0.35
    research_require_spread: bool = False

    research_weight_relative_strength: float = 0.20
    research_weight_daily_trend: float = 0.16
    research_weight_compression: float = 0.10
    research_weight_accumulation: float = 0.10
    research_weight_stock_regime: float = 0.08
    research_weight_sector_regime: float = 0.07
    research_weight_sector_participation: float = 0.07
    research_weight_daily_signal: float = 0.10
    research_weight_flow: float = 0.05
    research_weight_foreign_flow: float = 0.03
    research_weight_institutional_flow: float = 0.02
    research_weight_flow_agreement: float = 0.02

    research_min_rs_percentile: float = 0.0
    research_min_trend_score: float = 0.0
    research_min_compression_score: float = 0.0
    research_min_accumulation_score: float = -1.0
    research_min_sector_participation: float = 0.0
    research_min_sector_daily_score_pct: float = 0.0
    research_use_sector_daily_regime_score: bool = False
    research_use_sector_daily_participation: bool = False
    research_use_sector_daily_flow_gates: bool = False
    research_max_box_range_pct: float = 0.0
    research_min_flow_5d: float = -9.99
    research_min_foreign_flow_5d: float = -9.99
    research_min_institutional_flow_5d: float = -9.99
    research_min_flow_z: float = -9.99
    research_min_flow_agreement: float = -9.99
    research_max_flow_divergence: float = 9.99
    research_min_sector_flow_5d: float = -9.99
    research_min_sector_foreign_flow_5d: float = -9.99
    research_min_sector_institutional_flow_5d: float = -9.99
    research_min_sector_flow_agreement: float = -9.99

    daily_signal_family: str = "discrete_pullback_v2"
    daily_signal_min_score: float = 0.0
    daily_rescue_min_score: float = 0.0
    daily_signal_max_score: float = 100.0
    signal_floor: float = 0.0
    flow_policy: str = "soft_penalty_rescue"
    rescue_size_mult: float = 0.65
    allow_secular: bool = True
    secular_sizing_mult: float = 0.65
    cdd_max: int = 99
    gap_max_pct: float = 20.0
    min_candidates_day: int = 1
    signal_rank_gate_mode: str = "score_rank"
    daily_structure_weight: float = 0.0
    min_relative_strength_pct: float = 0.0
    max_relative_strength_pct: float = 100.0
    min_parent_20d_return_pct: float = -20.0
    max_parent_20d_return_pct: float = 999.0
    min_market_breadth_pct: float = 0.0
    min_market_heat_score: float = 0.0
    structure_sizing_enabled: bool = False
    structure_size_mult_min: float = 0.70
    structure_size_mult_max: float = 1.25
    rsi2_trigger_thresh: float = 15.0
    rsi5_trigger_thresh: float = 30.0
    cdd_min_for_rsi5: int = 2
    depth_atr_trigger: float = 1.5
    bb_pctb_trigger: float = 0.05
    volume_climax_trigger: float = 2.0
    relative_strength_trigger_pct: float = 70.0
    roc5_drop_trigger_pct: float = -3.0
    gap_down_trigger_pct: float = -2.0

    afternoon_score_mode: str = "hybrid"
    afternoon_min_ret: float = 0.0
    afternoon_min_vwap_ret: float = -0.02
    afternoon_max_ret: float = 9.99
    afternoon_max_vwap_ret: float = 9.99
    afternoon_min_gap: float = -0.20
    afternoon_max_gap: float = 0.20
    afternoon_min_rel_volume: float = 0.0
    afternoon_min_close_location: float = 0.0
    afternoon_max_open_drawdown: float = 0.20
    afternoon_max_range_atr: float = 99.0
    afternoon_min_high_from_open: float = -0.20
    afternoon_min_low_vs_prev_close: float = -0.20
    afternoon_min_prior_ret5: float = -100.0
    afternoon_min_prior_ret20: float = -100.0
    afternoon_max_prior_ret20: float = 999.0
    afternoon_min_prior_ret60: float = -100.0
    afternoon_min_bar_count: int = 1
    afternoon_min_flow_5d: float = -9.99
    afternoon_min_foreign_flow_5d: float = -9.99
    afternoon_min_institutional_flow_5d: float = -9.99
    afternoon_min_flow_z: float = -9.99
    afternoon_min_foreign_z: float = -9.99
    afternoon_min_institutional_z: float = -9.99
    afternoon_min_flow_agreement: float = -9.99
    afternoon_max_flow_divergence: float = 9.99
    afternoon_min_sector_flow: float = -9.99
    afternoon_min_sector_foreign_flow: float = -9.99
    afternoon_min_sector_institutional_flow: float = -9.99
    afternoon_min_intraday_sector_score_pct: float = 0.0
    afternoon_weight_intraday_sector: float = 0.0
    afternoon_weight_sector_confirm_quality: float = 0.0
    afternoon_weight_sector_rotation: float = 0.0
    afternoon_weight_stock_sector_leadership: float = 0.0
    afternoon_min_market_score: float = -9.99
    afternoon_require_close_above_prev: bool = False
    afternoon_use_lagged_flow_score: bool = True
    afternoon_min_lagged_flow_score: float = -999.0
    afternoon_score_calibration_mode: str = "raw"
    afternoon_exhaustion_penalty: float = 0.0
    afternoon_max_exhaustion_score: float = 999.0
    afternoon_min_score: float = -999_999.0
    afternoon_max_score: float = 999_999.0
    afternoon_reject_score_min: float = 0.0
    afternoon_reject_score_max: float = 0.0
    afternoon_score_band_rules: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    afternoon_blocked_sectors: tuple[str, ...] = ()
    afternoon_allowed_sectors: tuple[str, ...] = ()
    shadow_reranker_enabled: bool = False
    shadow_reranker_profile: dict[str, Any] = field(default_factory=dict)
    shadow_reranker_replace_score_band_rules: bool = False
    shadow_reranker_min_score: float = -999_999.0

    live_parity_fill_timing: str = "completed_5m_signal_next_bar_or_resting_close_auction"
    entry_mode: str = "close_auction"
    exit_mode: str = "next_close"
    allocation_mode: str = "selected_equal_capped"
    target_gross_exposure: float = 1.0
    max_position_pct: float = 0.25
    rank_decay: float = 1.0
    min_selected: int = 1
    auction_fill_time: str = "15:30"
    auction_limit_offset_bps: float = 150.0
    auction_adverse_bps: float = 0.0
    auction_nonfill_rate: float = 0.0
    market_entry_price_buffer_bps: float = 150.0
    trade_entry_plan: dict[str, Any] = field(default_factory=dict)
    trade_exit_plan: dict[str, Any] = field(default_factory=dict)

    slippage_bps: float = 5.0
    commission_bps: float = 2.0
    tax_bps_on_sell: float = 18.0

    def __post_init__(self) -> None:
        if isinstance(self.live_parity_fill_timing, bool):
            live_parity_fill_timing = (
                "completed_5m_signal_next_bar_or_resting_close_auction" if self.live_parity_fill_timing else "unspecified"
            )
        else:
            live_parity_fill_timing = str(self.live_parity_fill_timing or "").strip()
        object.__setattr__(self, "live_parity_fill_timing", live_parity_fill_timing)
        if not live_parity_fill_timing:
            raise ValueError("OLR live_parity_fill_timing must be an explicit timing string")
        if self.strategy_id.upper() != STRATEGY_ID:
            raise ValueError("OLRConfig.strategy_id must be OLR")
        if self.timeframe.lower() != "5m":
            raise ValueError("OLR research requires completed 5m bars")
        if self.complete_universe_size < 1:
            raise ValueError("OLR complete_universe_size must be positive")
        if self.research_top_long_count < 1:
            raise ValueError("OLR research_top_long_count must be positive")
        if self.afternoon_top_n < 1:
            raise ValueError("OLR afternoon_top_n must be positive")
        if self.overnight_slot_count < 1:
            raise ValueError("OLR overnight_slot_count must be positive")
        if self.premarket_frontier_size < 1:
            raise ValueError("OLR premarket_frontier_size must be positive")
        if self.research_min_history_days < 20:
            raise ValueError("OLR research_min_history_days must be at least 20")
        if self.daily_rescue_min_score > self.daily_signal_max_score:
            raise ValueError("OLR daily_rescue_min_score must be <= daily_signal_max_score")
        if self.daily_signal_max_score < self.daily_signal_min_score:
            raise ValueError("OLR daily_signal_max_score must be >= daily_signal_min_score")
        if self.max_relative_strength_pct < self.min_relative_strength_pct:
            raise ValueError("OLR max_relative_strength_pct must be >= min_relative_strength_pct")
        if self.structure_size_mult_max < self.structure_size_mult_min:
            raise ValueError("OLR structure_size_mult_max must be >= structure_size_mult_min")
        if self.afternoon_max_gap <= self.afternoon_min_gap:
            raise ValueError("OLR afternoon_max_gap must exceed afternoon_min_gap")
        if self.afternoon_max_ret < self.afternoon_min_ret:
            raise ValueError("OLR afternoon_max_ret must be >= afternoon_min_ret")
        if self.afternoon_max_vwap_ret < self.afternoon_min_vwap_ret:
            raise ValueError("OLR afternoon_max_vwap_ret must be >= afternoon_min_vwap_ret")
        if self.afternoon_max_prior_ret20 < self.afternoon_min_prior_ret20:
            raise ValueError("OLR afternoon_max_prior_ret20 must be >= afternoon_min_prior_ret20")
        if self.afternoon_min_bar_count < 1:
            raise ValueError("OLR afternoon_min_bar_count must be positive")
        if self.afternoon_score_calibration_mode not in {"raw", "exhaustion_adjusted"}:
            raise ValueError("OLR afternoon_score_calibration_mode must be raw or exhaustion_adjusted")
        if self.afternoon_max_exhaustion_score < 0.0:
            raise ValueError("OLR afternoon_max_exhaustion_score must be non-negative")
        if self.afternoon_max_score < self.afternoon_min_score:
            raise ValueError("OLR afternoon_max_score must be >= afternoon_min_score")
        if self.afternoon_reject_score_max and self.afternoon_reject_score_max <= self.afternoon_reject_score_min:
            raise ValueError("OLR afternoon_reject_score_max must exceed afternoon_reject_score_min when enabled")
        _validate_score_band_rules(self.afternoon_score_band_rules)
        if self.entry_mode not in {"close_auction", "decision_next_open"}:
            raise ValueError("OLR entry_mode must be close_auction or decision_next_open")
        if self.exit_mode != "next_close":
            raise ValueError("OLR exit_mode must be next_close")
        if not (0.0 <= self.target_gross_exposure <= 2.0):
            raise ValueError("OLR target_gross_exposure must be between 0 and 2")
        if not (0.0 < self.max_position_pct <= 1.0):
            raise ValueError("OLR max_position_pct must be in (0, 1]")
        if self.min_selected < 1:
            raise ValueError("OLR min_selected must be positive")
        if not (0.0 <= self.auction_nonfill_rate <= 1.0):
            raise ValueError("OLR auction_nonfill_rate must be between 0 and 1")
        if sum(_research_weight_values(self)) <= 0.0:
            raise ValueError("OLR research score weights must include at least one positive value")

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any] | None = None,
        mutations: dict[str, Any] | None = None,
    ) -> "OLRConfig":
        merged: dict[str, Any] = {}
        for source in (data or {}, mutations or {}):
            merged.update(_flatten_config(source))
        allowed = {field.name for field in fields(cls)}
        kwargs = {key: value for key, value in merged.items() if key in allowed}
        return cls(**kwargs)

    def with_mutations(self, mutations: dict[str, Any] | None = None) -> "OLRConfig":
        if not mutations:
            return self
        updated = self.from_mapping(mutations)
        flattened = _flatten_config(mutations)
        values = {field.name: getattr(self, field.name) for field in fields(self)}
        values.update({field.name: getattr(updated, field.name) for field in fields(updated) if field.name in flattened})
        return replace(self, **values)


def _research_weight_values(config: OLRConfig) -> tuple[float, ...]:
    return (
        max(float(config.research_weight_relative_strength), 0.0),
        max(float(config.research_weight_daily_trend), 0.0),
        max(float(config.research_weight_compression), 0.0),
        max(float(config.research_weight_accumulation), 0.0),
        max(float(config.research_weight_stock_regime), 0.0),
        max(float(config.research_weight_sector_regime), 0.0),
        max(float(config.research_weight_sector_participation), 0.0),
        max(float(config.research_weight_daily_signal), 0.0),
        max(float(config.research_weight_flow), 0.0),
        max(float(config.research_weight_foreign_flow), 0.0),
        max(float(config.research_weight_institutional_flow), 0.0),
        max(float(config.research_weight_flow_agreement), 0.0),
    )


_ALIASES = {
    "olr.universe.complete_size": "complete_universe_size",
    "olr.frontier.active_selection_mode": "frontier_active_selection_mode",
    "olr.discovery.active_selection_mode": "frontier_active_selection_mode",
    "olr.premarket.frontier_size": "premarket_frontier_size",
    "olr.premarket.min_adv20_krw": "premarket_min_adv20_krw",
    "olr.premarket.min_foreign5_z": "premarket_min_foreign5_z",
    "olr.research.top_long_count": "research_top_long_count",
    "olr.research.min_price_krw": "research_min_price_krw",
    "olr.research.min_adv20_krw": "research_min_adv20_krw",
    "olr.research.min_history_days": "research_min_history_days",
    "olr.research.max_median_spread_pct": "research_max_median_spread_pct",
    "olr.research.require_spread": "research_require_spread",
    "olr.research.weights.relative_strength": "research_weight_relative_strength",
    "olr.research.weights.daily_trend": "research_weight_daily_trend",
    "olr.research.weights.compression": "research_weight_compression",
    "olr.research.weights.accumulation": "research_weight_accumulation",
    "olr.research.weights.stock_regime": "research_weight_stock_regime",
    "olr.research.weights.sector_regime": "research_weight_sector_regime",
    "olr.research.weights.sector_participation": "research_weight_sector_participation",
    "olr.research.weights.daily_signal": "research_weight_daily_signal",
    "olr.research.weights.flow": "research_weight_flow",
    "olr.research.weights.foreign_flow": "research_weight_foreign_flow",
    "olr.research.weights.institutional_flow": "research_weight_institutional_flow",
    "olr.research.weights.flow_agreement": "research_weight_flow_agreement",
    "olr.research.min_rs_percentile": "research_min_rs_percentile",
    "olr.research.min_trend_score": "research_min_trend_score",
    "olr.research.min_compression_score": "research_min_compression_score",
    "olr.research.min_accumulation_score": "research_min_accumulation_score",
    "olr.research.min_sector_participation": "research_min_sector_participation",
    "olr.research.min_sector_daily_score_pct": "research_min_sector_daily_score_pct",
    "olr.research.use_sector_daily_regime_score": "research_use_sector_daily_regime_score",
    "olr.research.use_sector_daily_participation": "research_use_sector_daily_participation",
    "olr.research.use_sector_daily_flow_gates": "research_use_sector_daily_flow_gates",
    "olr.research.max_box_range_pct": "research_max_box_range_pct",
    "olr.research.min_flow_5d": "research_min_flow_5d",
    "olr.research.min_foreign_flow_5d": "research_min_foreign_flow_5d",
    "olr.research.min_institutional_flow_5d": "research_min_institutional_flow_5d",
    "olr.research.min_inst_flow_5d": "research_min_institutional_flow_5d",
    "olr.research.min_flow_z": "research_min_flow_z",
    "olr.research.min_flow_agreement": "research_min_flow_agreement",
    "olr.research.max_flow_divergence": "research_max_flow_divergence",
    "olr.research.min_sector_flow_5d": "research_min_sector_flow_5d",
    "olr.research.min_sector_foreign_flow_5d": "research_min_sector_foreign_flow_5d",
    "olr.research.min_sector_inst_flow_5d": "research_min_sector_institutional_flow_5d",
    "olr.research.min_sector_institutional_flow_5d": "research_min_sector_institutional_flow_5d",
    "olr.research.min_sector_flow_agreement": "research_min_sector_flow_agreement",
    "olr.signal.daily_signal_family": "daily_signal_family",
    "olr.signal.family": "daily_signal_family",
    "olr.signal.daily_min_score": "daily_signal_min_score",
    "olr.signal.daily_rescue_min_score": "daily_rescue_min_score",
    "olr.signal.daily_max_score": "daily_signal_max_score",
    "olr.signal.signal_floor": "signal_floor",
    "olr.signal.flow_policy": "flow_policy",
    "olr.signal.rescue_size_mult": "rescue_size_mult",
    "olr.signal.allow_secular": "allow_secular",
    "olr.signal.secular_sizing_mult": "secular_sizing_mult",
    "olr.signal.cdd_max": "cdd_max",
    "olr.signal.gap_max_pct": "gap_max_pct",
    "olr.signal.min_candidates_day": "min_candidates_day",
    "olr.signal.rank_gate_mode": "signal_rank_gate_mode",
    "olr.signal.daily_structure_weight": "daily_structure_weight",
    "olr.signal.min_relative_strength_pct": "min_relative_strength_pct",
    "olr.signal.max_relative_strength_pct": "max_relative_strength_pct",
    "olr.signal.min_parent_20d_return_pct": "min_parent_20d_return_pct",
    "olr.signal.max_parent_20d_return_pct": "max_parent_20d_return_pct",
    "olr.signal.min_market_breadth_pct": "min_market_breadth_pct",
    "olr.signal.min_market_heat_score": "min_market_heat_score",
    "olr.signal.structure_sizing_enabled": "structure_sizing_enabled",
    "olr.signal.structure_size_mult_min": "structure_size_mult_min",
    "olr.signal.structure_size_mult_max": "structure_size_mult_max",
    "olr.signal.rsi2_trigger_thresh": "rsi2_trigger_thresh",
    "olr.signal.rsi5_trigger_thresh": "rsi5_trigger_thresh",
    "olr.signal.cdd_min_for_rsi5": "cdd_min_for_rsi5",
    "olr.signal.depth_atr_trigger": "depth_atr_trigger",
    "olr.signal.bb_pctb_trigger": "bb_pctb_trigger",
    "olr.signal.volume_climax_trigger": "volume_climax_trigger",
    "olr.signal.relative_strength_trigger_pct": "relative_strength_trigger_pct",
    "olr.signal.roc5_drop_trigger_pct": "roc5_drop_trigger_pct",
    "olr.signal.gap_down_trigger_pct": "gap_down_trigger_pct",
    "olr.afternoon.score_mode": "afternoon_score_mode",
    "olr.afternoon.top_n": "afternoon_top_n",
    "olr.afternoon.min_ret": "afternoon_min_ret",
    "olr.afternoon.min_vwap_ret": "afternoon_min_vwap_ret",
    "olr.afternoon.max_ret": "afternoon_max_ret",
    "olr.afternoon.max_vwap_ret": "afternoon_max_vwap_ret",
    "olr.afternoon.min_gap": "afternoon_min_gap",
    "olr.afternoon.max_gap": "afternoon_max_gap",
    "olr.afternoon.min_rel_volume": "afternoon_min_rel_volume",
    "olr.afternoon.min_close_location": "afternoon_min_close_location",
    "olr.afternoon.max_open_drawdown": "afternoon_max_open_drawdown",
    "olr.afternoon.max_range_atr": "afternoon_max_range_atr",
    "olr.afternoon.min_high_from_open": "afternoon_min_high_from_open",
    "olr.afternoon.min_low_vs_prev_close": "afternoon_min_low_vs_prev_close",
    "olr.afternoon.min_prior_ret5": "afternoon_min_prior_ret5",
    "olr.afternoon.min_prior_ret20": "afternoon_min_prior_ret20",
    "olr.afternoon.max_prior_ret20": "afternoon_max_prior_ret20",
    "olr.afternoon.min_prior_ret60": "afternoon_min_prior_ret60",
    "olr.afternoon.min_bar_count": "afternoon_min_bar_count",
    "olr.afternoon.min_flow_5d": "afternoon_min_flow_5d",
    "olr.afternoon.min_foreign_flow_5d": "afternoon_min_foreign_flow_5d",
    "olr.afternoon.min_inst_flow_5d": "afternoon_min_institutional_flow_5d",
    "olr.afternoon.min_institutional_flow_5d": "afternoon_min_institutional_flow_5d",
    "olr.afternoon.min_flow_z": "afternoon_min_flow_z",
    "olr.afternoon.min_foreign_z": "afternoon_min_foreign_z",
    "olr.afternoon.min_institutional_z": "afternoon_min_institutional_z",
    "olr.afternoon.min_flow_agreement": "afternoon_min_flow_agreement",
    "olr.afternoon.max_flow_divergence": "afternoon_max_flow_divergence",
    "olr.afternoon.min_sector_flow": "afternoon_min_sector_flow",
    "olr.afternoon.min_sector_foreign_flow": "afternoon_min_sector_foreign_flow",
    "olr.afternoon.min_sector_inst_flow": "afternoon_min_sector_institutional_flow",
    "olr.afternoon.min_sector_institutional_flow": "afternoon_min_sector_institutional_flow",
    "olr.afternoon.min_intraday_sector_score_pct": "afternoon_min_intraday_sector_score_pct",
    "olr.afternoon.weight_intraday_sector": "afternoon_weight_intraday_sector",
    "olr.afternoon.weight_sector_confirm_quality": "afternoon_weight_sector_confirm_quality",
    "olr.afternoon.weight_sector_rotation": "afternoon_weight_sector_rotation",
    "olr.afternoon.weight_stock_sector_leadership": "afternoon_weight_stock_sector_leadership",
    "olr.afternoon.min_market_score": "afternoon_min_market_score",
    "olr.afternoon.require_close_above_prev": "afternoon_require_close_above_prev",
    "olr.afternoon.use_lagged_flow_score": "afternoon_use_lagged_flow_score",
    "olr.afternoon.min_lagged_flow_score": "afternoon_min_lagged_flow_score",
    "olr.afternoon.score_calibration_mode": "afternoon_score_calibration_mode",
    "olr.afternoon.exhaustion_penalty": "afternoon_exhaustion_penalty",
    "olr.afternoon.max_exhaustion_score": "afternoon_max_exhaustion_score",
    "olr.afternoon.min_score": "afternoon_min_score",
    "olr.afternoon.max_score": "afternoon_max_score",
    "olr.afternoon.reject_score_min": "afternoon_reject_score_min",
    "olr.afternoon.reject_score_max": "afternoon_reject_score_max",
    "olr.afternoon.score_band_rules": "afternoon_score_band_rules",
    "olr.afternoon.blocked_sectors": "afternoon_blocked_sectors",
    "olr.afternoon.allowed_sectors": "afternoon_allowed_sectors",
    "olr.shadow_reranker.enabled": "shadow_reranker_enabled",
    "olr.shadow_reranker.profile": "shadow_reranker_profile",
    "olr.shadow_reranker.replace_score_band_rules": "shadow_reranker_replace_score_band_rules",
    "olr.shadow_reranker.min_score": "shadow_reranker_min_score",
    "olr.overnight.slot_count": "overnight_slot_count",
    "olr.execution.live_parity_fill_timing": "live_parity_fill_timing",
    "olr.execution.entry_mode": "entry_mode",
    "olr.execution.exit_mode": "exit_mode",
    "olr.execution.auction_fill_time": "auction_fill_time",
    "olr.execution.auction_limit_offset_bps": "auction_limit_offset_bps",
    "olr.execution.auction_adverse_bps": "auction_adverse_bps",
    "olr.execution.auction_nonfill_rate": "auction_nonfill_rate",
    "olr.execution.market_entry_price_buffer_bps": "market_entry_price_buffer_bps",
    "olr.execution.trade_entry_plan": "trade_entry_plan",
    "olr.execution.trade_exit_plan": "trade_exit_plan",
    "olr.trade_plan.entry": "trade_entry_plan",
    "olr.trade_plan.exit": "trade_exit_plan",
    "olr.allocation.mode": "allocation_mode",
    "olr.allocation.target_gross_exposure": "target_gross_exposure",
    "olr.allocation.max_position_pct": "max_position_pct",
    "olr.allocation.rank_decay": "rank_decay",
    "olr.allocation.min_selected": "min_selected",
    "olr.cost.slippage_bps": "slippage_bps",
    "olr.cost.commission_bps": "commission_bps",
    "olr.cost.tax_bps_on_sell": "tax_bps_on_sell",
    "olr.robustness.slippage_bps": "slippage_bps",
    "olr.robustness.auction_adverse_bps": "auction_adverse_bps",
    "olr.robustness.auction_nonfill_rate": "auction_nonfill_rate",
    "olr.robustness.auction_limit_offset_bps": "auction_limit_offset_bps",
    "olr.robustness.market_entry_price_buffer_bps": "market_entry_price_buffer_bps",
}


def _validate_score_band_rules(value: Any) -> None:
    if not value:
        return
    if isinstance(value, dict):
        rules = (value,)
    elif isinstance(value, (list, tuple)):
        rules = tuple(value)
    else:
        raise ValueError("OLR afternoon_score_band_rules must be a rule mapping or a list of rule mappings")
    for index, raw_rule in enumerate(rules, start=1):
        if not isinstance(raw_rule, dict):
            raise ValueError(f"OLR afternoon_score_band_rules[{index}] must be a mapping")
        if "min_score" in raw_rule and "max_score" in raw_rule:
            if float(raw_rule["max_score"]) <= float(raw_rule["min_score"]):
                raise ValueError(f"OLR afternoon_score_band_rules[{index}] max_score must exceed min_score")
        if "max_rank" in raw_rule and int(raw_rule["max_rank"]) < 1:
            raise ValueError(f"OLR afternoon_score_band_rules[{index}] max_rank must be positive")
        if "min_rank" in raw_rule and int(raw_rule["min_rank"]) < 1:
            raise ValueError(f"OLR afternoon_score_band_rules[{index}] min_rank must be positive")
        if "sector_admission" in raw_rule and not isinstance(raw_rule["sector_admission"], dict):
            raise ValueError(f"OLR afternoon_score_band_rules[{index}] sector_admission must be a mapping")


def _flatten_config(data: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        alias = _ALIASES.get(prefix)
        if alias in {"trade_entry_plan", "trade_exit_plan", "shadow_reranker_profile"} and isinstance(value, dict):
            out[alias] = dict(value)
            return
        if isinstance(value, dict):
            for key, child in value.items():
                next_key = f"{prefix}.{key}" if prefix else str(key)
                visit(next_key, child)
            return
        key = alias or prefix
        out[key] = value

    for key, value in dict(data or {}).items():
        visit(str(key), value)
    return out
