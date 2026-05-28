from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from datetime import date, datetime
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from strategy_common.sector_map import normalize_sector, normalize_sector_map, normalize_symbol


SECTOR_DAILY_VERSION = "sector-daily-v2"


@dataclass(frozen=True, slots=True)
class SectorDailyMember:
    symbol: str
    sector: str
    ret_5d: float
    ret_20d: float
    ret_60d: float
    above_sma20: bool
    rel_volume: float = 1.0
    flow_5d: float = 0.0
    foreign_flow_5d: float = 0.0
    institutional_flow_5d: float = 0.0
    flow_agreement_5d: float = 0.0
    flow_available: bool = False
    trade_date: date | None = None

    @property
    def participation(self) -> float:
        return 1.0 if self.ret_5d >= 0.0 and self.rel_volume >= 1.0 else 0.0

    @property
    def member_score(self) -> float:
        acceleration = self.ret_5d - (self.ret_20d / 4.0)
        ret5_score = _bounded(0.5 + acceleration / 0.08, 0.0, 1.0)
        ret20_score = _bounded(0.5 + self.ret_20d / 0.20, 0.0, 1.0)
        ret60_score = _bounded(0.5 + self.ret_60d / 0.35, 0.0, 1.0)
        breadth_score = 1.0 if self.above_sma20 else 0.0
        rel_volume_score = _bounded(0.5 + 0.25 * math.log(max(self.rel_volume, 0.1)), 0.0, 1.0)
        if self.flow_available:
            flow_score = _bounded(
                0.5
                + 3.0 * self.flow_5d
                + 1.5 * self.foreign_flow_5d
                + 1.5 * self.institutional_flow_5d
                + 2.0 * self.flow_agreement_5d,
                0.0,
                1.0,
            )
        else:
            flow_score = 0.5
        return 100.0 * (
            0.18 * ret5_score
            + 0.26 * ret20_score
            + 0.12 * ret60_score
            + 0.18 * breadth_score
            + 0.12 * self.participation
            + 0.06 * rel_volume_score
            + 0.08 * flow_score
        )


@dataclass(frozen=True, slots=True)
class SectorDailySector:
    trade_date: date | None
    sector: str
    member_count: int
    ret_5d: float
    ret_20d: float
    ret_60d: float
    breadth_20d: float
    participation: float
    rel_volume: float
    flow_5d: float
    foreign_flow_5d: float
    institutional_flow_5d: float
    flow_agreement_5d: float
    raw_score: float
    score_pct: float
    regime: str


@dataclass(frozen=True, slots=True)
class SectorDailyFeature:
    symbol: str
    sector: str
    trade_date: date | None
    score_pct: float = 50.0
    ret_5d: float = 0.0
    ret_20d: float = 0.0
    ret_60d: float = 0.0
    breadth_20d: float = 0.5
    participation: float = 0.0
    rel_volume: float = 1.0
    flow_5d: float = 0.0
    foreign_flow_5d: float = 0.0
    institutional_flow_5d: float = 0.0
    flow_agreement_5d: float = 0.0
    effective_count: int = 0
    shrinkage_weight: float = 0.0
    regime: str = "UNKNOWN"
    raw_score: float = 50.0
    member_score: float = 50.0

    @classmethod
    def neutral(
        cls,
        symbol: str,
        *,
        sector: str = "UNKNOWN",
        trade_date: date | None = None,
    ) -> "SectorDailyFeature":
        return cls(
            symbol=normalize_symbol(symbol),
            sector=normalize_sector(sector),
            trade_date=trade_date,
        )

    def metadata(self, *, prefix: str = "") -> dict[str, float | int | str]:
        return sector_daily_metadata(self, prefix=prefix)


@dataclass(frozen=True, slots=True)
class SectorDailyPanel:
    features_by_key: dict[tuple[date | None, str], SectorDailyFeature]
    sectors_by_key: dict[tuple[date | None, str], SectorDailySector]
    version: str = SECTOR_DAILY_VERSION

    def feature_for(
        self,
        trade_date: date | None,
        symbol: str,
        *,
        sector: str = "UNKNOWN",
    ) -> SectorDailyFeature:
        normalized = normalize_symbol(symbol)
        return self.features_by_key.get(
            (trade_date, normalized),
            SectorDailyFeature.neutral(normalized, sector=sector, trade_date=trade_date),
        )


def build_sector_daily_panel(
    daily_by_symbol: Mapping[Any, Any],
    sector_map: Mapping[Any, Any],
    *,
    trade_dates: Iterable[date] | None = None,
    flow_by_symbol: Mapping[Any, Any] | None = None,
    foreign_flow_by_symbol: Mapping[Any, Any] | None = None,
    institutional_flow_by_symbol: Mapping[Any, Any] | None = None,
    symbols: Iterable[str] | None = None,
    min_history: int = 21,
    min_effective_members: int = 3,
    shrinkage_k: float = 3.0,
) -> SectorDailyPanel:
    normalized_sector_map = normalize_sector_map(sector_map)
    raw_symbols = symbols if symbols is not None else normalized_sector_map
    selected_symbols = tuple(sorted({normalize_symbol(symbol) for symbol in raw_symbols if normalize_symbol(symbol)}))
    selected_dates = tuple(sorted(set(trade_dates or _dates_from_daily_rows(daily_by_symbol))))
    daily_rows = _normalize_rows_by_symbol(daily_by_symbol)
    flow_rows = _normalize_rows_by_symbol(flow_by_symbol or {})
    foreign_rows = _normalize_rows_by_symbol(foreign_flow_by_symbol or {})
    institutional_rows = _normalize_rows_by_symbol(institutional_flow_by_symbol or {})

    features: dict[tuple[date | None, str], SectorDailyFeature] = {}
    sectors: dict[tuple[date | None, str], SectorDailySector] = {}
    for trade_day in selected_dates:
        members: list[SectorDailyMember] = []
        for symbol in selected_symbols:
            sector = normalized_sector_map.get(symbol)
            if not sector:
                continue
            member = member_from_daily_rows(
                symbol,
                sector,
                trade_day,
                daily_rows.get(symbol, ()),
                flow_rows=flow_rows.get(symbol, ()),
                foreign_flow_rows=foreign_rows.get(symbol, ()),
                institutional_flow_rows=institutional_rows.get(symbol, ()),
                min_history=min_history,
            )
            if member is not None:
                members.append(member)
        day_panel = score_sector_daily_members(
            members,
            trade_date=trade_day,
            min_effective_members=min_effective_members,
            shrinkage_k=shrinkage_k,
        )
        features.update(day_panel.features_by_key)
        sectors.update(day_panel.sectors_by_key)
    return SectorDailyPanel(features, sectors)


def score_sector_daily_members(
    members: Sequence[SectorDailyMember],
    *,
    trade_date: date | None = None,
    target_symbols: Iterable[str] | None = None,
    min_effective_members: int = 3,
    shrinkage_k: float = 3.0,
) -> SectorDailyPanel:
    normalized_members = tuple(
        SectorDailyMember(
            symbol=normalize_symbol(member.symbol),
            sector=normalize_sector(member.sector),
            ret_5d=float(member.ret_5d),
            ret_20d=float(member.ret_20d),
            ret_60d=float(member.ret_60d),
            above_sma20=bool(member.above_sma20),
            rel_volume=max(float(member.rel_volume), 0.0),
            flow_5d=float(member.flow_5d),
            foreign_flow_5d=float(member.foreign_flow_5d),
            institutional_flow_5d=float(member.institutional_flow_5d),
            flow_agreement_5d=float(member.flow_agreement_5d),
            flow_available=bool(member.flow_available),
            trade_date=member.trade_date or trade_date,
        )
        for member in members
        if normalize_symbol(member.symbol) and normalize_sector(member.sector) != "UNKNOWN"
    )
    targets = {normalize_symbol(symbol) for symbol in target_symbols or [member.symbol for member in normalized_members] if normalize_symbol(symbol)}
    by_symbol = {member.symbol: member for member in normalized_members}
    by_sector: dict[str, list[SectorDailyMember]] = {}
    for member in normalized_members:
        by_sector.setdefault(member.sector, []).append(member)

    sector_summaries = {
        sector: _summarize_members(items, trade_date=trade_date, sector=sector)
        for sector, items in by_sector.items()
    }
    pct_by_sector = _percentile_map({sector: summary.raw_score for sector, summary in sector_summaries.items()})
    sectors = {
        (trade_date, sector): SectorDailySector(
            trade_date=summary.trade_date,
            sector=summary.sector,
            member_count=summary.member_count,
            ret_5d=summary.ret_5d,
            ret_20d=summary.ret_20d,
            ret_60d=summary.ret_60d,
            breadth_20d=summary.breadth_20d,
            participation=summary.participation,
            rel_volume=summary.rel_volume,
            flow_5d=summary.flow_5d,
            foreign_flow_5d=summary.foreign_flow_5d,
            institutional_flow_5d=summary.institutional_flow_5d,
            flow_agreement_5d=summary.flow_agreement_5d,
            raw_score=summary.raw_score,
            score_pct=pct_by_sector.get(sector, 50.0),
            regime=_daily_regime(summary.raw_score, summary.ret_20d, summary.breadth_20d),
        )
        for sector, summary in sector_summaries.items()
    }
    sector_raw_values = [summary.raw_score for summary in sector_summaries.values()]

    features: dict[tuple[date | None, str], SectorDailyFeature] = {}
    for symbol in targets:
        member = by_symbol.get(symbol)
        if member is None:
            features[(trade_date, symbol)] = SectorDailyFeature.neutral(symbol, trade_date=trade_date)
            continue
        peers = [item for item in by_sector.get(member.sector, ()) if item.symbol != symbol]
        market_peers = [item for item in normalized_members if item.symbol != symbol]
        if not peers and not market_peers:
            features[(trade_date, symbol)] = SectorDailyFeature.neutral(symbol, sector=member.sector, trade_date=trade_date)
            continue
        peer_summary = _summarize_members(peers, trade_date=trade_date, sector=member.sector)
        market_summary = _summarize_members(market_peers, trade_date=trade_date, sector="MARKET")
        effective_count = len(peers)
        shrinkage_weight = 1.0 if effective_count >= int(min_effective_members) else effective_count / max(effective_count + float(shrinkage_k), 1e-9)
        raw_score = _blend(peer_summary.raw_score, market_summary.raw_score, shrinkage_weight)
        ret_20d = _blend(peer_summary.ret_20d, market_summary.ret_20d, shrinkage_weight)
        breadth = _blend(peer_summary.breadth_20d, market_summary.breadth_20d, shrinkage_weight)
        features[(trade_date, symbol)] = SectorDailyFeature(
            symbol=symbol,
            sector=member.sector,
            trade_date=trade_date,
            score_pct=_percentile_for_value(raw_score, sector_raw_values),
            ret_5d=_blend(peer_summary.ret_5d, market_summary.ret_5d, shrinkage_weight),
            ret_20d=ret_20d,
            ret_60d=_blend(peer_summary.ret_60d, market_summary.ret_60d, shrinkage_weight),
            breadth_20d=breadth,
            participation=_blend(peer_summary.participation, market_summary.participation, shrinkage_weight),
            rel_volume=_blend(peer_summary.rel_volume, market_summary.rel_volume, shrinkage_weight),
            flow_5d=_blend(peer_summary.flow_5d, market_summary.flow_5d, shrinkage_weight),
            foreign_flow_5d=_blend(peer_summary.foreign_flow_5d, market_summary.foreign_flow_5d, shrinkage_weight),
            institutional_flow_5d=_blend(peer_summary.institutional_flow_5d, market_summary.institutional_flow_5d, shrinkage_weight),
            flow_agreement_5d=_blend(peer_summary.flow_agreement_5d, market_summary.flow_agreement_5d, shrinkage_weight),
            effective_count=effective_count,
            shrinkage_weight=shrinkage_weight,
            regime=_daily_regime(raw_score, ret_20d, breadth),
            raw_score=raw_score,
            member_score=member.member_score,
        )
    return SectorDailyPanel(features, sectors)


def member_from_daily_rows(
    symbol: str,
    sector: str,
    trade_date: date,
    daily_rows: Sequence[Mapping[str, Any]],
    *,
    flow_rows: Sequence[Mapping[str, Any]] | None = None,
    foreign_flow_rows: Sequence[Mapping[str, Any]] | None = None,
    institutional_flow_rows: Sequence[Mapping[str, Any]] | None = None,
    min_history: int = 21,
) -> SectorDailyMember | None:
    rows = _prior_rows(daily_rows, trade_date)
    if len(rows) < int(min_history):
        return None
    if _bad_recent_ohlcv(rows[-20:]):
        return None
    closes = [_float(row.get("close")) for row in rows]
    volumes = [_float(row.get("volume")) for row in rows]
    latest_close = closes[-1]
    if latest_close <= 0.0 or volumes[-1] <= 0.0:
        return None
    sma20 = fmean(closes[-20:])
    prior_volume = fmean(volume for volume in volumes[-21:-1] if volume > 0.0) if len(volumes) >= 21 else 0.0
    flow = _flow_values(
        rows,
        _prior_rows(flow_rows or (), trade_date),
        _prior_rows(foreign_flow_rows or (), trade_date),
        _prior_rows(institutional_flow_rows or (), trade_date),
    )
    return SectorDailyMember(
        symbol=normalize_symbol(symbol),
        sector=normalize_sector(sector),
        trade_date=trade_date,
        ret_5d=_return_decimal(closes, 5),
        ret_20d=_return_decimal(closes, 20),
        ret_60d=_return_decimal(closes, 60),
        above_sma20=latest_close >= sma20,
        rel_volume=volumes[-1] / max(prior_volume, 1.0) if prior_volume > 0.0 else 1.0,
        flow_5d=flow["flow_5d"],
        foreign_flow_5d=flow["foreign_flow_5d"],
        institutional_flow_5d=flow["institutional_flow_5d"],
        flow_agreement_5d=flow["flow_agreement_5d"],
        flow_available=bool(flow["flow_available"]),
    )


def sector_daily_metadata(feature: SectorDailyFeature, *, prefix: str = "") -> dict[str, float | int | str]:
    base = {
        "sector_daily_score_pct": float(feature.score_pct),
        "sector_daily_ret_5d": float(feature.ret_5d),
        "sector_daily_ret_20d": float(feature.ret_20d),
        "sector_daily_ret_60d": float(feature.ret_60d),
        "sector_daily_breadth_20d": float(feature.breadth_20d),
        "sector_daily_participation": float(feature.participation),
        "sector_daily_rel_volume": float(feature.rel_volume),
        "sector_daily_flow_5d": float(feature.flow_5d),
        "sector_daily_foreign_flow_5d": float(feature.foreign_flow_5d),
        "sector_daily_institutional_flow_5d": float(feature.institutional_flow_5d),
        "sector_daily_flow_agreement_5d": float(feature.flow_agreement_5d),
        "sector_daily_effective_count": int(feature.effective_count),
        "sector_daily_shrinkage_weight": float(feature.shrinkage_weight),
        "sector_daily_regime": str(feature.regime),
        "sector_daily_version": SECTOR_DAILY_VERSION,
    }
    if not prefix:
        return base
    clean = str(prefix).strip("_")
    return {f"{clean}_{key}": value for key, value in base.items()}


def _summarize_members(
    members: Sequence[SectorDailyMember],
    *,
    trade_date: date | None,
    sector: str,
) -> SectorDailySector:
    items = tuple(members or ())
    if not items:
        return SectorDailySector(
            trade_date=trade_date,
            sector=normalize_sector(sector),
            member_count=0,
            ret_5d=0.0,
            ret_20d=0.0,
            ret_60d=0.0,
            breadth_20d=0.5,
            participation=0.0,
            rel_volume=1.0,
            flow_5d=0.0,
            foreign_flow_5d=0.0,
            institutional_flow_5d=0.0,
            flow_agreement_5d=0.0,
            raw_score=50.0,
            score_pct=50.0,
            regime="UNKNOWN",
        )
    raw_score = fmean(item.member_score for item in items)
    ret_20d = fmean(item.ret_20d for item in items)
    breadth = fmean(1.0 if item.above_sma20 else 0.0 for item in items)
    return SectorDailySector(
        trade_date=trade_date,
        sector=normalize_sector(sector),
        member_count=len(items),
        ret_5d=fmean(item.ret_5d for item in items),
        ret_20d=ret_20d,
        ret_60d=fmean(item.ret_60d for item in items),
        breadth_20d=breadth,
        participation=fmean(item.participation for item in items),
        rel_volume=fmean(max(item.rel_volume, 0.0) for item in items),
        flow_5d=fmean(item.flow_5d for item in items),
        foreign_flow_5d=fmean(item.foreign_flow_5d for item in items),
        institutional_flow_5d=fmean(item.institutional_flow_5d for item in items),
        flow_agreement_5d=fmean(item.flow_agreement_5d for item in items),
        raw_score=raw_score,
        score_pct=50.0,
        regime=_daily_regime(raw_score, ret_20d, breadth),
    )


def _flow_values(
    daily_rows: Sequence[Mapping[str, Any]],
    flow_rows: Sequence[Mapping[str, Any]],
    foreign_flow_rows: Sequence[Mapping[str, Any]],
    institutional_flow_rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    daily_by_date = {_row_date(row): row for row in daily_rows if _try_row_date(row) is not None}
    flow_by_date = {_row_date(row): row for row in flow_rows if _try_row_date(row) is not None}
    foreign_by_date = {_row_date(row): row for row in foreign_flow_rows if _try_row_date(row) is not None}
    inst_by_date = {_row_date(row): row for row in institutional_flow_rows if _try_row_date(row) is not None}
    if not (flow_by_date or foreign_by_date or inst_by_date):
        return _empty_flow_values()

    normalized: list[tuple[float, float, float, float]] = []
    for row_date in sorted(daily_by_date):
        daily = daily_by_date[row_date]
        volume = max(_float(daily.get("volume")), 1.0)
        flow_row = flow_by_date.get(row_date, {})
        foreign_row = foreign_by_date.get(row_date, {})
        inst_row = inst_by_date.get(row_date, {})
        foreign = _optional_flow_value(foreign_row, "foreign_net")
        if foreign is None:
            foreign = _optional_flow_value(flow_row, "foreign_net") or 0.0
        inst = _optional_flow_value(inst_row, "institutional_net", "inst_net")
        if inst is None:
            inst = _optional_flow_value(flow_row, "institutional_net", "inst_net") or 0.0
        foreign_norm = float(foreign) / volume
        inst_norm = float(inst) / volume
        agreement = min(max(foreign_norm, 0.0), max(inst_norm, 0.0))
        normalized.append((foreign_norm + inst_norm, foreign_norm, inst_norm, agreement))
    sample = normalized[-5:]
    if not sample:
        return _empty_flow_values()
    return {
        "flow_available": 1.0,
        "flow_5d": fmean(item[0] for item in sample),
        "foreign_flow_5d": fmean(item[1] for item in sample),
        "institutional_flow_5d": fmean(item[2] for item in sample),
        "flow_agreement_5d": fmean(item[3] for item in sample),
    }


def _empty_flow_values() -> dict[str, float]:
    return {
        "flow_available": 0.0,
        "flow_5d": 0.0,
        "foreign_flow_5d": 0.0,
        "institutional_flow_5d": 0.0,
        "flow_agreement_5d": 0.0,
    }


def _prior_rows(rows: Sequence[Mapping[str, Any]], trade_date: date) -> list[dict[str, Any]]:
    dated_rows: list[tuple[date, dict[str, Any]]] = []
    for row in rows or ():
        parsed = _try_row_date(row)
        if parsed is None or parsed >= trade_date:
            continue
        try:
            normalized = dict(row)
            normalized["date"] = parsed.isoformat()
            dated_rows.append((parsed, normalized))
        except (TypeError, ValueError):
            continue
    dated_rows.sort(key=lambda item: item[0])
    return [row for _, row in dated_rows]


def _try_row_date(row: Mapping[str, Any]) -> date | None:
    try:
        return _row_date(row)
    except (AttributeError, TypeError, ValueError):
        return None


def _row_date(row: Mapping[str, Any]) -> date:
    value = row.get("date") or row.get("trade_date") or row.get("timestamp")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value)
    if "T" in text:
        return datetime.fromisoformat(text).date()
    return date.fromisoformat(text[:10])


def _normalize_rows_by_symbol(rows_by_symbol: Mapping[Any, Any]) -> dict[str, tuple[Mapping[str, Any], ...]]:
    return {
        normalize_symbol(symbol): tuple(rows or ())
        for symbol, rows in dict(rows_by_symbol or {}).items()
        if normalize_symbol(symbol)
    }


def _dates_from_daily_rows(daily_by_symbol: Mapping[Any, Any]) -> tuple[date, ...]:
    dates: set[date] = set()
    for rows in dict(daily_by_symbol or {}).values():
        for row in rows or ():
            parsed = _try_row_date(row)
            if parsed is not None:
                dates.add(parsed)
    return tuple(sorted(dates))


def _bad_recent_ohlcv(rows: Sequence[Mapping[str, Any]]) -> bool:
    if not rows:
        return True
    for row in rows:
        open_ = _float(row.get("open"))
        high = _float(row.get("high"))
        low = _float(row.get("low"))
        close = _float(row.get("close"))
        volume = _float(row.get("volume"))
        if min(open_, high, low, close) <= 0.0 or volume <= 0.0:
            return True
        if high < low or close > high * 1.0001 or close < low * 0.9999:
            return True
    return False


def _return_decimal(closes: Sequence[float], lookback: int) -> float:
    if len(closes) <= lookback:
        return 0.0
    prior = float(closes[-1 - lookback])
    return float(closes[-1]) / prior - 1.0 if prior > 0.0 else 0.0


def _optional_flow_value(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in row and row.get(key) is not None:
            return _float(row.get(key))
    return None


def _percentile_map(raw: Mapping[str, float]) -> dict[str, float]:
    items = sorted((float(value), key) for key, value in raw.items())
    if not items:
        return {}
    if len(items) == 1:
        return {key: 50.0 for key in raw}
    ranks: dict[str, float] = {}
    denom = float(len(items) - 1)
    index = 0
    while index < len(items):
        value = items[index][0]
        end = index + 1
        while end < len(items) and items[end][0] == value:
            end += 1
        pct = 100.0 * ((index + end - 1) / 2.0) / denom
        for _, key in items[index:end]:
            ranks[key] = pct
        index = end
    return ranks


def _percentile_for_value(value: float, values: Sequence[float]) -> float:
    ordered = sorted(float(item) for item in values)
    if len(ordered) < 2:
        return 50.0
    value_float = float(value)
    left = bisect.bisect_left(ordered, value_float)
    right = bisect.bisect_right(ordered, value_float)
    rank = ((left + right - 1) / 2.0) if right > left else float(left)
    return _bounded(100.0 * rank / float(len(ordered) - 1), 0.0, 100.0)


def _daily_regime(raw_score: float, ret_20d: float, breadth_20d: float) -> str:
    if raw_score >= 62.0 and ret_20d >= 0.0 and breadth_20d >= 0.55:
        return "LEADING"
    if raw_score <= 42.0 and ret_20d <= 0.0 and breadth_20d <= 0.45:
        return "LAGGING"
    return "MIXED"


def _blend(primary: float, fallback: float, weight: float) -> float:
    bounded_weight = _bounded(float(weight), 0.0, 1.0)
    return bounded_weight * float(primary) + (1.0 - bounded_weight) * float(fallback)


def _bounded(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _float(value: Any, default: float = 0.0) -> float:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value_float if math.isfinite(value_float) else float(default)
