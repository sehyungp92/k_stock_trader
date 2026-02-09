"""Nulrimok Smart Money Flow Score."""

import math
from dataclasses import dataclass
from typing import List

from .db import LRSDatabase
from ..config.constants import (
    FLOW_PERSISTENCE_WEIGHT, FLOW_INTENSITY_WEIGHT, FLOW_ACCEL_WEIGHT, FLOW_PERSISTENCE_MIN,
)
from ..config.switches import nulrimok_switches


@dataclass
class FlowScoreResult:
    persistence: float
    intensity_z: float
    accel_z: float
    score: float
    passes: bool


def _compute_adv_won(lrs: LRSDatabase, ticker: str, days: int = 20) -> float:
    """ADV in won = average daily traded value (typical_price * volume)."""
    bars = lrs.get_recent_bars(ticker, days)
    if not bars:
        return 0.0
    traded_values = [((b.high + b.low + b.close) / 3) * b.volume for b in bars]
    return sum(traded_values) / len(traded_values)


def _compute_accel(flow: List[float]) -> float:
    """Acceleration = MA5(flow) - MA20(flow)."""
    if len(flow) < 20:
        return 0.0
    ma5 = sum(flow[-5:]) / 5
    ma20 = sum(flow[-20:]) / 20
    return ma5 - ma20


def compute_flow_score(lrs: LRSDatabase, ticker: str, sector_tickers: list) -> FlowScoreResult:
    flow = lrs.get_smart_money_series(ticker, days=20)
    if len(flow) < 10:
        return FlowScoreResult(0, 0, 0, 0, False)

    persistence = sum(1 for x in flow[-10:] if x > 0) / 10.0

    adv_won = _compute_adv_won(lrs, ticker)
    if adv_won <= 0:
        return FlowScoreResult(persistence, 0, 0, 0, False)

    intensity = sum(flow[-5:]) / adv_won
    accel = _compute_accel(flow)

    # Compute sector intensities and accelerations for z-scoring
    sector_intensities = []
    sector_accels = []
    for t in sector_tickers:
        t_flow = lrs.get_smart_money_series(t, days=20)
        t_adv = _compute_adv_won(lrs, t)
        if t_flow and len(t_flow) >= 5 and t_adv > 0:
            sector_intensities.append(sum(t_flow[-5:]) / t_adv)
            sector_accels.append(_compute_accel(t_flow))

    if len(sector_intensities) > 2:
        i_mean = sum(sector_intensities) / len(sector_intensities)
        i_std = math.sqrt(sum((x - i_mean) ** 2 for x in sector_intensities) / len(sector_intensities)) or 1e-9
        intensity_z = (intensity - i_mean) / i_std

        a_mean = sum(sector_accels) / len(sector_accels)
        a_std = math.sqrt(sum((x - a_mean) ** 2 for x in sector_accels) / len(sector_accels)) or 1e-9
        accel_z = (accel - a_mean) / a_std
    else:
        intensity_z = 0.0
        accel_z = 0.0

    score = (FLOW_PERSISTENCE_WEIGHT * persistence +
             FLOW_INTENSITY_WEIGHT * max(0, min(1, intensity_z / 3 + 0.5)) +
             FLOW_ACCEL_WEIGHT * max(0, min(1, accel_z / 3 + 0.5)))

    # Compute sector median score for gating
    sector_scores = []
    for t in sector_tickers:
        t_flow = lrs.get_smart_money_series(t, days=20)
        t_adv = _compute_adv_won(lrs, t)
        if t_flow and len(t_flow) >= 10 and t_adv > 0:
            t_pers = sum(1 for x in t_flow[-10:] if x > 0) / 10.0
            t_int = sum(t_flow[-5:]) / t_adv
            if len(sector_intensities) > 2:
                t_int_z = (t_int - i_mean) / i_std
                t_acc = _compute_accel(t_flow)
                t_acc_z = (t_acc - a_mean) / a_std
            else:
                t_int_z, t_acc_z = 0.0, 0.0
            t_score = (FLOW_PERSISTENCE_WEIGHT * t_pers +
                       FLOW_INTENSITY_WEIGHT * max(0, min(1, t_int_z / 3 + 0.5)) +
                       FLOW_ACCEL_WEIGHT * max(0, min(1, t_acc_z / 3 + 0.5)))
            sector_scores.append(t_score)

    if sector_scores:
        sector_scores_sorted = sorted(sector_scores)
        median = sector_scores_sorted[len(sector_scores_sorted) // 2]
    else:
        median = 0.0

    # Use switch-configurable flow persistence threshold
    persistence_min = nulrimok_switches.flow_persistence_min
    passes = persistence >= persistence_min and score > median

    # Log would-block: passed permissive but would fail strict threshold
    if passes and persistence < FLOW_PERSISTENCE_MIN:
        nulrimok_switches.log_would_block(
            "FLOW",  # ticker not available here, logged at caller level
            "FLOW_PERSISTENCE",
            persistence,
            FLOW_PERSISTENCE_MIN,
        )

    return FlowScoreResult(persistence, intensity_z, accel_z, score, passes)
