"""Overnight leader rotation research strategy."""

from .config import OLRConfig
from .artifact_store import OLR_FINAL_ARTIFACT_STAGE, OLR_STAGE1_ARTIFACT_STAGE
from .execution import (
    EXECUTION_CORE_VERSION,
    OLRAllocationPlan,
    OLREntryPlan,
    OLRExitPlan,
    OLRTradePlan,
    action_to_intent,
    normalize_action_prices,
    simulate_olr_trade,
    summarize_olr_outcomes_with_allocation,
)
from .research import (
    afternoon_selection_from_contexts,
    afternoon_selection_from_snapshot,
    build_afternoon_contexts,
    build_research_snapshot,
    daily_selection_from_snapshot,
    load_candidate_snapshot,
    run_afternoon_selection,
    run_daily_selection,
)

__all__ = [
    "OLRConfig",
    "OLR_FINAL_ARTIFACT_STAGE",
    "OLR_STAGE1_ARTIFACT_STAGE",
    "EXECUTION_CORE_VERSION",
    "OLRAllocationPlan",
    "OLREntryPlan",
    "OLRExitPlan",
    "OLRTradePlan",
    "action_to_intent",
    "afternoon_selection_from_contexts",
    "afternoon_selection_from_snapshot",
    "build_afternoon_contexts",
    "build_research_snapshot",
    "daily_selection_from_snapshot",
    "load_candidate_snapshot",
    "normalize_action_prices",
    "run_afternoon_selection",
    "run_daily_selection",
    "simulate_olr_trade",
    "summarize_olr_outcomes_with_allocation",
]
