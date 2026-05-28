from __future__ import annotations

import bisect
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime
from statistics import mean
from typing import Any, Iterable, Mapping


STRUCTURAL_CAMPAIGN_VERSION = "kalcb-structural-campaign-v1"
SQUEEZE_LOOKBACK_WINDOWS = 90


@dataclass(frozen=True, slots=True)
class KALCBCompressionBox:
    start_date: str
    end_date: str
    length: int
    high: float
    low: float
    mid: float
    height_pct: float
    atr_ratio: float
    containment: float
    squeeze_pct: float
    tier: str
    available: bool = True

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class KALCBStructuralCampaign:
    symbol: str
    trade_date: date
    sector: str
    rs_percentile: float
    stock_vs_universe_strength: float
    daily_trend_sign: int
    trend_alignment_detail: Mapping[str, float | int | str]
    compression_box: KALCBCompressionBox | None
    accumulation_score: float
    distribution_score: float
    sector_regime: str
    sector_leadership_pct: float
    prior_daily_breakout_watch: bool
    campaign_avwap: float | None
    campaign_box_high: float | None
    campaign_box_low: float | None
    campaign_box_mid: float | None
    campaign_breakout_level: float | None
    breakout_displacement: float | None
    campaign_state: str
    structural_campaign_score: float
    first30_confirmation_score: float
    selection_detail: Mapping[str, float]
    structural_reject_reasons: tuple[str, ...] = ()
    score_uses_ex_post_labels: bool = False

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["trade_date"] = self.trade_date.isoformat()
        payload["structural_reject_reasons"] = list(self.structural_reject_reasons)
        return payload


def compute_rs_percentiles(
    strength_by_symbol: Mapping[str, float],
) -> dict[str, float]:
    """Deterministic percentile rank in 0-100 space."""

    ordered = sorted(float(value) for value in strength_by_symbol.values())
    if not ordered:
        return {}
    total = len(ordered)
    return {
        str(symbol).zfill(6): 100.0 * bisect.bisect_right(ordered, float(value)) / total
        for symbol, value in strength_by_symbol.items()
    }


def build_structural_campaign(
    symbol: str,
    trade_date: date,
    daily_rows: Iterable[dict[str, Any]],
    *,
    sector: str = "UNKNOWN",
    rs_percentile: float = 50.0,
    stock_vs_universe_strength: float = 0.0,
    sector_daily_score_pct: float = 50.0,
    sector_participation: float = 0.0,
    market_heat_score: float = 50.0,
    daily_flow_rows: Iterable[dict[str, Any]] | None = None,
    daily_foreign_flow_rows: Iterable[dict[str, Any]] | None = None,
    daily_institutional_flow_rows: Iterable[dict[str, Any]] | None = None,
) -> KALCBStructuralCampaign:
    """Build a prior-daily structural campaign; same-day rows are ignored."""

    prior = _prior_rows(daily_rows, trade_date)
    flow = _prior_rows(daily_flow_rows or (), trade_date)
    foreign = _prior_rows(daily_foreign_flow_rows or (), trade_date)
    institutional = _prior_rows(daily_institutional_flow_rows or (), trade_date)
    if len(prior) < 20:
        detail = _empty_detail(float(rs_percentile), float(sector_daily_score_pct))
        return KALCBStructuralCampaign(
            symbol=str(symbol).zfill(6),
            trade_date=trade_date,
            sector=str(sector or "UNKNOWN").upper(),
            rs_percentile=float(rs_percentile),
            stock_vs_universe_strength=float(stock_vs_universe_strength),
            daily_trend_sign=0,
            trend_alignment_detail={},
            compression_box=None,
            accumulation_score=50.0,
            distribution_score=50.0,
            sector_regime=_sector_regime(sector_daily_score_pct),
            sector_leadership_pct=float(sector_daily_score_pct),
            prior_daily_breakout_watch=False,
            campaign_avwap=None,
            campaign_box_high=None,
            campaign_box_low=None,
            campaign_box_mid=None,
            campaign_breakout_level=None,
            breakout_displacement=None,
            campaign_state="none",
            structural_campaign_score=0.0,
            first30_confirmation_score=0.0,
            selection_detail=detail,
            structural_reject_reasons=("insufficient_daily_history",),
        )

    closes = [_num(row.get("close")) for row in prior]
    close = max(closes[-1], 1e-9)
    trend_sign, trend_detail, trend_quality = compute_daily_trend_sign(prior)
    box = detect_adaptive_compression_box(prior)
    accumulation, distribution = compute_accumulation_distribution(
        prior,
        daily_flow_rows=flow,
        daily_foreign_flow_rows=foreign,
        daily_institutional_flow_rows=institutional,
    )
    state, prior_watch, displacement = _campaign_state(close, box, trend_sign)
    detail = score_structural_campaign(
        rs_percentile=float(rs_percentile),
        trend_sign=trend_sign,
        trend_quality_pct=trend_quality,
        compression_box=box,
        accumulation_score=accumulation,
        sector_daily_score_pct=float(sector_daily_score_pct),
    )
    score = float(detail["structural_campaign_score"])
    reasons = _reject_reasons(
        rs_percentile=float(rs_percentile),
        trend_quality_pct=trend_quality,
        compression_box=box,
        accumulation_score=accumulation,
        sector_daily_score_pct=float(sector_daily_score_pct),
    )
    return KALCBStructuralCampaign(
        symbol=str(symbol).zfill(6),
        trade_date=trade_date,
        sector=str(sector or "UNKNOWN").upper(),
        rs_percentile=float(rs_percentile),
        stock_vs_universe_strength=float(stock_vs_universe_strength),
        daily_trend_sign=int(trend_sign),
        trend_alignment_detail=trend_detail,
        compression_box=box,
        accumulation_score=float(accumulation),
        distribution_score=float(distribution),
        sector_regime=_sector_regime(sector_daily_score_pct),
        sector_leadership_pct=float(sector_daily_score_pct),
        prior_daily_breakout_watch=prior_watch,
        campaign_avwap=None,
        campaign_box_high=box.high if box is not None and box.available else None,
        campaign_box_low=box.low if box is not None and box.available else None,
        campaign_box_mid=box.mid if box is not None and box.available else None,
        campaign_breakout_level=box.high if box is not None and box.available else None,
        breakout_displacement=displacement,
        campaign_state=state,
        structural_campaign_score=score,
        first30_confirmation_score=0.0,
        selection_detail=detail,
        structural_reject_reasons=tuple(reasons),
    )


def compute_daily_trend_sign(rows: Iterable[dict[str, Any]]) -> tuple[int, dict[str, float | int | str], float]:
    prior = list(rows)
    closes = [_num(row.get("close")) for row in prior]
    if len(closes) < 20:
        return 0, {}, 0.0
    close = max(closes[-1], 1e-9)
    sma20 = _sma(closes, 20)
    sma50 = _sma(closes, 50) if len(closes) >= 50 else sma20
    sma150 = _sma(closes, 150) if len(closes) >= 150 else sma50
    sma20_prior = _sma(closes[-25:-5], 20) if len(closes) >= 45 else sma20
    sma50_prior = _sma(closes[-65:-15], 50) if len(closes) >= 115 else sma50
    price_vs_sma20 = close / max(sma20, 1e-9) - 1.0
    price_vs_sma50 = close / max(sma50, 1e-9) - 1.0
    price_vs_sma150 = close / max(sma150, 1e-9) - 1.0
    ret20 = _return_pct(closes, 20)
    ret60 = _return_pct(closes, 60)
    sma20_rising = sma20 >= sma20_prior
    sma50_rising = sma50 >= sma50_prior
    if close >= sma20 >= sma50 >= sma150 * 0.98 and sma20_rising and ret20 >= 0.0:
        sign = 1
        tier = "up"
    elif close < sma20 and ret20 < 0.0 and (len(closes) < 50 or close < sma50):
        sign = -1
        tier = "down"
    else:
        sign = 0
        tier = "neutral"
    stack_quality = _bounded(50.0 + 260.0 * price_vs_sma20 + 160.0 * (sma20 / max(sma50, 1e-9) - 1.0), 0.0, 100.0)
    return_quality = _bounded(50.0 + 180.0 * ret20 + 80.0 * ret60, 0.0, 100.0)
    slope_quality = 100.0 if sma20_rising and sma50_rising else 62.5 if sma20_rising else 35.0
    quality = _bounded(0.45 * stack_quality + 0.35 * return_quality + 0.20 * slope_quality, 0.0, 100.0)
    return sign, {
        "trend_tier": tier,
        "sma20": float(sma20),
        "sma50": float(sma50),
        "sma150": float(sma150),
        "price_vs_sma20": float(price_vs_sma20),
        "price_vs_sma50": float(price_vs_sma50),
        "price_vs_sma150": float(price_vs_sma150),
        "sma20_rising": int(sma20_rising),
        "sma50_rising": int(sma50_rising),
        "return_20d": float(ret20),
        "return_60d": float(ret60),
    }, quality


def detect_adaptive_compression_box(rows: Iterable[dict[str, Any]]) -> KALCBCompressionBox | None:
    prior = list(rows)
    if len(prior) < 12:
        return None
    close = max(_num(prior[-1].get("close")), 1e-9)
    atr14 = _atr(prior, 14)
    atr50 = _atr(prior, 50) if len(prior) >= 50 else atr14
    true_ranges = _true_range_prefix(prior)
    highs = [_num(row.get("high")) for row in prior]
    lows = [_num(row.get("low")) for row in prior]
    candidates: list[tuple[float, KALCBCompressionBox]] = []
    for length in (8, 10, 15, 20):
        if len(prior) < length:
            continue
        window = prior[-length:]
        high = max(_num(row.get("high")) for row in window)
        low = min(_num(row.get("low")) for row in window)
        if high <= 0.0 or low <= 0.0 or high < low:
            continue
        height = max(high - low, 0.0)
        mid = (high + low) / 2.0
        height_pct = height / max(close, 1e-9)
        inner_low = low + 0.08 * height
        inner_high = high - 0.08 * height
        containment = sum(1 for row in window if inner_low <= _num(row.get("close")) <= inner_high) / max(len(window), 1)
        atr_ratio = height / max(atr50, 1e-9) if atr50 > 0.0 else min(height_pct / 0.12, 2.0)
        squeeze_pct = _squeeze_percentile(prior, length, atr_ratio, true_ranges, highs, lows)
        available = containment >= 0.45 and height_pct <= 0.22 and atr_ratio <= 8.0
        if not available:
            tier = "none"
        elif squeeze_pct <= 0.30 and containment >= 0.55:
            tier = "tight"
        elif squeeze_pct <= 0.65 or (height_pct <= 0.12 and containment >= 0.55):
            tier = "balanced"
        else:
            tier = "loose"
        box = KALCBCompressionBox(
            start_date=_row_date(window[0]).isoformat(),
            end_date=_row_date(window[-1]).isoformat(),
            length=len(window),
            high=float(high),
            low=float(low),
            mid=float(mid),
            height_pct=float(height_pct),
            atr_ratio=float(atr_ratio),
            containment=float(containment),
            squeeze_pct=float(squeeze_pct),
            tier=tier,
            available=bool(available),
        )
        score = (1.0 if available else 0.0) + containment + 0.60 * (1.0 - squeeze_pct) - 0.03 * abs(length - 15)
        candidates.append((score, box))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], -item[1].height_pct, item[1].length), reverse=True)
    return candidates[0][1]


def _squeeze_percentile(
    rows: list[dict[str, Any]],
    length: int,
    current_ratio: float,
    true_ranges: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> float:
    if len(rows) <= length:
        return _bounded(float(current_ratio) / 8.0, 0.0, 1.0)
    true_ranges = true_ranges or _true_range_prefix(rows)
    highs = highs or [_num(row.get("high")) for row in rows]
    lows = lows or [_num(row.get("low")) for row in rows]
    history: list[float] = []
    start = max(length, len(rows) - SQUEEZE_LOOKBACK_WINDOWS)
    for end in range(start, len(rows)):
        high = max(highs[end - length : end])
        low = min(lows[end - length : end])
        atr_len = 50 if end >= 50 else 14
        atr_start = max(1, end - atr_len)
        atr_count = max(end - atr_start, 0)
        atr = (true_ranges[end] - true_ranges[atr_start]) / atr_count if atr_count > 0 else 0.0
        if atr <= 0.0:
            continue
        history.append(max(high - low, 0.0) / max(atr, 1e-9))
    if len(history) < 4:
        return _bounded(float(current_ratio) / 8.0, 0.0, 1.0)
    ordered = sorted(history)
    return bisect.bisect_right(ordered, float(current_ratio)) / len(ordered)


def _true_range_prefix(rows: list[dict[str, Any]]) -> list[float]:
    prefix = [0.0] * (len(rows) + 1)
    if len(rows) < 2:
        return prefix
    prev_close = _num(rows[0].get("close"))
    for index, row in enumerate(rows[1:], start=1):
        high = _num(row.get("high"))
        low = _num(row.get("low"))
        true_range = max(high - low, abs(high - prev_close), abs(low - prev_close))
        prefix[index + 1] = prefix[index] + true_range
        prev_close = _num(row.get("close"))
    return prefix


def compute_accumulation_distribution(
    daily_rows: Iterable[dict[str, Any]],
    *,
    daily_flow_rows: Iterable[dict[str, Any]] | None = None,
    daily_foreign_flow_rows: Iterable[dict[str, Any]] | None = None,
    daily_institutional_flow_rows: Iterable[dict[str, Any]] | None = None,
) -> tuple[float, float]:
    prior = list(daily_rows)[-20:]
    if not prior:
        return 50.0, 50.0
    locs: list[float] = []
    volumes: list[float] = []
    for row in prior:
        high = _num(row.get("high"))
        low = _num(row.get("low"))
        close = _num(row.get("close"))
        locs.append(_bounded((close - low) / max(high - low, 1e-9), 0.0, 1.0))
        volumes.append(max(_num(row.get("volume")), 0.0))
    vol_sum = max(sum(max(value, 1.0) for value in volumes), 1e-9)
    weighted_loc = sum(loc * max(vol, 1.0) for loc, vol in zip(locs, volumes)) / vol_sum
    recent = mean(volumes[-5:]) if len(volumes) >= 5 else mean(volumes)
    base = mean(volumes) if volumes else 1.0
    volume_component = _bounded(0.5 + 0.25 * math.log(max(recent, 1.0) / max(base, 1.0)), 0.0, 1.0)
    ohlcv_score = 100.0 * (0.72 * weighted_loc + 0.28 * volume_component)
    foreign_values = _flow_series(daily_flow_rows or (), daily_foreign_flow_rows or (), keys=("foreign_net", "foreign", "foreign_flow"))
    inst_values = _flow_series(daily_flow_rows or (), daily_institutional_flow_rows or (), keys=("institutional_net", "inst_net", "institutional", "inst_flow"))
    flow_score = 50.0
    if foreign_values or inst_values:
        flow_score = _bounded(50.0 + 9.0 * _flow_z(foreign_values) + 8.0 * _flow_z(inst_values), 0.0, 100.0)
    accumulation = _bounded(0.62 * ohlcv_score + 0.38 * flow_score, 0.0, 100.0)
    distribution = _bounded(100.0 - accumulation + 8.0 * max(0.0, 0.35 - weighted_loc), 0.0, 100.0)
    return float(accumulation), float(distribution)


def score_structural_campaign(
    *,
    rs_percentile: float,
    trend_sign: int,
    trend_quality_pct: float,
    compression_box: KALCBCompressionBox | None,
    accumulation_score: float,
    sector_daily_score_pct: float,
) -> dict[str, float]:
    rs = float(rs_percentile)
    if rs >= 90.0:
        rs_score = 3.0
    elif rs >= 75.0:
        rs_score = 2.0
    elif rs >= 50.0:
        rs_score = 1.0
    else:
        rs_score = 0.0
    if trend_sign > 0 and trend_quality_pct >= 55.0:
        trend_score = 2.0
    elif trend_sign >= 0 and trend_quality_pct >= 42.0:
        trend_score = 1.0
    else:
        trend_score = 0.0
    if compression_box is not None and compression_box.available and compression_box.tier in {"tight", "balanced"}:
        compression_score = 2.0
    elif compression_box is not None and compression_box.available:
        compression_score = 1.0
    else:
        compression_score = 0.0
    if accumulation_score >= 68.0:
        accumulation_component = 2.0
    elif accumulation_score >= 52.0:
        accumulation_component = 1.0
    else:
        accumulation_component = 0.0
    sector_score = 1.0 if sector_daily_score_pct >= 60.0 else 0.5 if sector_daily_score_pct >= 45.0 else 0.0
    total = rs_score + trend_score + compression_score + accumulation_component + sector_score
    return {
        "rs_score": float(rs_score),
        "trend_score": float(trend_score),
        "compression_score_structural": float(compression_score),
        "accumulation_score_structural": float(accumulation_component),
        "sector_score": float(sector_score),
        "structural_campaign_score": float(total),
    }


def campaign_metadata(campaign: KALCBStructuralCampaign) -> dict[str, Any]:
    box = campaign.compression_box
    box_dict = box.to_json_dict() if box is not None else None
    selection_detail = dict(campaign.selection_detail)
    return {
        "structural_campaign_version": STRUCTURAL_CAMPAIGN_VERSION,
        "structural_campaign": campaign.to_json_dict(),
        "structural_campaign_score": campaign.structural_campaign_score,
        "first30_confirmation_score": campaign.first30_confirmation_score,
        "campaign_state": campaign.campaign_state,
        "campaign_state_score": _campaign_state_score(campaign.campaign_state),
        "campaign_box_high": campaign.campaign_box_high or 0.0,
        "campaign_box_low": campaign.campaign_box_low or 0.0,
        "campaign_box_mid": campaign.campaign_box_mid or 0.0,
        "campaign_box_range_pct": box.height_pct if box is not None else 0.0,
        "campaign_box_containment": box.containment if box is not None else 0.0,
        "campaign_avwap": campaign.campaign_avwap or 0.0,
        "campaign_breakout_level": campaign.campaign_breakout_level or 0.0,
        "campaign_breakout_displacement": campaign.breakout_displacement or 0.0,
        "campaign_selection_detail": selection_detail,
        "selection_detail": selection_detail,
        "stock_vs_universe_strength": campaign.stock_vs_universe_strength,
        "daily_trend_sign": campaign.daily_trend_sign,
        "trend_alignment_detail": dict(campaign.trend_alignment_detail),
        "accumulation_score_pct": campaign.accumulation_score,
        "distribution_score_pct": campaign.distribution_score,
        "sector_regime": campaign.sector_regime,
        "sector_leadership_pct": campaign.sector_leadership_pct,
        "sector_leadership": campaign.sector_leadership_pct >= 70.0,
        "prior_daily_breakout_watch": campaign.prior_daily_breakout_watch,
        "structural_score_uses_ex_post_labels": campaign.score_uses_ex_post_labels,
        "structural_reject_reasons": list(campaign.structural_reject_reasons),
        "compression_box": box_dict,
    }


def _campaign_state_score(state: str) -> float:
    return {
        "none": 0.0,
        "failed": 0.0,
        "compression": 1.0,
        "breakout_watch": 2.0,
        "prior_breakout": 3.0,
        "first30_confirmed": 4.0,
    }.get(str(state), 0.0)


def _campaign_state(
    close: float,
    box: KALCBCompressionBox | None,
    trend_sign: int,
) -> tuple[str, bool, float | None]:
    if box is None or not box.available:
        return ("failed" if trend_sign < 0 else "none"), False, None
    height = max(box.high - box.low, 1e-9)
    displacement = (close - box.high) / height
    if close > box.high * 1.002:
        return "prior_breakout", True, float(displacement)
    if close >= box.high - 0.25 * height:
        return "breakout_watch", True, float(displacement)
    return "compression", False, float(displacement)


def _reject_reasons(
    *,
    rs_percentile: float,
    trend_quality_pct: float,
    compression_box: KALCBCompressionBox | None,
    accumulation_score: float,
    sector_daily_score_pct: float,
) -> list[str]:
    reasons: list[str] = []
    if rs_percentile < 35.0:
        reasons.append("weak_relative_strength")
    if trend_quality_pct < 35.0:
        reasons.append("weak_trend")
    if compression_box is None or not compression_box.available:
        reasons.append("no_valid_compression_box")
    if accumulation_score < 40.0:
        reasons.append("weak_accumulation")
    if sector_daily_score_pct < 30.0:
        reasons.append("weak_sector_regime")
    return reasons


def _sector_regime(score_pct: float) -> str:
    score = float(score_pct)
    if score >= 70.0:
        return "leading"
    if score >= 50.0:
        return "neutral"
    if score > 0.0:
        return "weak"
    return "unknown"


def _empty_detail(rs_percentile: float, sector_daily_score_pct: float) -> dict[str, float]:
    return {
        "rs_score": 0.0,
        "trend_score": 0.0,
        "compression_score_structural": 0.0,
        "accumulation_score_structural": 0.0,
        "sector_score": 0.0,
        "structural_campaign_score": 0.0,
        "relative_strength_pct": float(rs_percentile),
        "sector_daily_score_pct": float(sector_daily_score_pct),
    }


def _prior_rows(rows: Iterable[dict[str, Any]], trade_date: date) -> list[dict[str, Any]]:
    out: list[tuple[date, dict[str, Any]]] = []
    for row in rows or ():
        parsed = _try_row_date(row)
        if parsed is None or parsed >= trade_date:
            continue
        out.append((parsed, dict(row)))
    out.sort(key=lambda item: item[0])
    return [row for _, row in out]


def _try_row_date(row: dict[str, Any]) -> date | None:
    try:
        return _row_date(row)
    except (TypeError, ValueError, AttributeError):
        return None


def _row_date(row: dict[str, Any]) -> date:
    raw = row.get("date") or row.get("trade_date") or row.get("timestamp")
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    text = str(raw)
    if "T" in text:
        text = text.split("T", 1)[0]
    if " " in text:
        text = text.split(" ", 1)[0]
    return date.fromisoformat(text[:10])


def _flow_series(
    aggregate_rows: Iterable[dict[str, Any]],
    specific_rows: Iterable[dict[str, Any]],
    *,
    keys: tuple[str, ...],
) -> list[float]:
    rows = list(specific_rows) or list(aggregate_rows)
    values: list[float] = []
    for row in rows:
        value = None
        for key in keys:
            if row.get(key) not in (None, ""):
                value = row.get(key)
                break
        values.append(_num(value))
    return values


def _flow_z(values: list[float]) -> float:
    if not values:
        return 0.0
    n = min(5, len(values))
    current = sum(values[-n:])
    if len(values) < max(12, n * 3):
        scale = mean(abs(value) for value in values) or 1.0
        return _bounded(current / max(scale * math.sqrt(n), 1e-9), -5.0, 5.0)
    sums = [sum(values[end - n : end]) for end in range(n, len(values) + 1)]
    history = sums[:-1][-60:]
    if len(history) < 8:
        return 0.0
    avg = mean(history)
    variance = mean((value - avg) ** 2 for value in history)
    std = math.sqrt(variance)
    return _bounded((sums[-1] - avg) / max(std, 1e-9), -5.0, 5.0)


def _atr(rows: list[dict[str, Any]], n: int) -> float:
    if len(rows) < 2:
        return 0.0
    sample = rows[-(n + 1) :] if len(rows) > n else rows
    prev_close = _num(sample[0].get("close"))
    trs: list[float] = []
    for row in sample[1:]:
        high = _num(row.get("high"))
        low = _num(row.get("low"))
        trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
        prev_close = _num(row.get("close"))
    return mean(trs) if trs else 0.0


def _return_pct(values: list[float], periods: int) -> float:
    if len(values) <= periods:
        return 0.0
    return values[-1] / max(values[-periods - 1], 1e-9) - 1.0


def _sma(values: list[float], n: int) -> float:
    sample = values[-n:] if len(values) >= n else values
    return mean(sample) if sample else 0.0


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _bounded(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
