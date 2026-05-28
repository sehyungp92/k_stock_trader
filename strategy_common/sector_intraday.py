from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from datetime import date, time
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from strategy_common.clock import ensure_kst
from strategy_common.market import MarketBar
from strategy_common.sector_map import load_canonical_sector_map, normalize_sector, normalize_sector_map, normalize_symbol


SESSION_OPEN = time(9, 0)
FIRST30_CUTOFF = time(9, 30)
AFTERNOON_CUTOFF = time(14, 30)
SECTOR_INTRADAY_VERSION = "sector-intraday-v2"


@dataclass(frozen=True, slots=True)
class SectorIntradayMember:
    symbol: str
    sector: str
    ret: float
    vwap_ret: float
    close_location: float
    rel_volume: float = 1.0
    volume: float = 0.0
    bar_count: int = 0
    trade_date: date | None = None

    @property
    def member_score(self) -> float:
        return (
            float(self.ret)
            + 0.35 * float(self.vwap_ret)
            + 0.004 * (float(self.close_location) - 0.5)
            + 0.002 * math.log(max(float(self.rel_volume), 0.1))
        )


@dataclass(frozen=True, slots=True)
class SectorIntradaySector:
    trade_date: date | None
    sector: str
    cutoff_label: str
    member_count: int
    ret: float
    breadth: float
    rel_volume: float
    participation: float
    dispersion: float
    raw_score: float
    score_pct: float


@dataclass(frozen=True, slots=True)
class SectorIntradayFeature:
    symbol: str
    sector: str
    trade_date: date | None
    cutoff_label: str
    score_pct: float = 50.0
    ret: float = 0.0
    breadth: float = 0.5
    rel_volume: float = 1.0
    participation: float = 0.0
    effective_count: int = 0
    shrinkage_weight: float = 0.0
    raw_score: float = 0.0
    member_score: float = 0.0

    @classmethod
    def neutral(
        cls,
        symbol: str,
        *,
        sector: str = "UNKNOWN",
        trade_date: date | None = None,
        cutoff_label: str = "",
    ) -> "SectorIntradayFeature":
        return cls(
            symbol=normalize_symbol(symbol),
            sector=normalize_sector(sector),
            trade_date=trade_date,
            cutoff_label=cutoff_label,
        )

    def metadata(self, *, prefix: str = "") -> dict[str, float | int | str]:
        return sector_intraday_metadata(self, prefix=prefix)


@dataclass(frozen=True, slots=True)
class SectorIntradayPanel:
    features_by_key: dict[tuple[date | None, str], SectorIntradayFeature]
    sectors_by_key: dict[tuple[date | None, str], SectorIntradaySector]
    cutoff_label: str
    version: str = SECTOR_INTRADAY_VERSION

    def feature_for(
        self,
        trade_date: date | None,
        symbol: str,
        *,
        sector: str = "UNKNOWN",
    ) -> SectorIntradayFeature:
        normalized = normalize_symbol(symbol)
        return self.features_by_key.get(
            (trade_date, normalized),
            SectorIntradayFeature.neutral(normalized, sector=sector, trade_date=trade_date, cutoff_label=self.cutoff_label),
        )


def build_sector_intraday_panel(
    bars_by_key: Mapping[Any, Any],
    sector_map: Mapping[Any, Any],
    *,
    trade_dates: Iterable[date] | None = None,
    cutoff: time = AFTERNOON_CUTOFF,
    symbols: Iterable[str] | None = None,
    min_effective_members: int = 3,
    shrinkage_k: float = 3.0,
) -> SectorIntradayPanel:
    normalized_sector_map = normalize_sector_map(sector_map)
    raw_symbols = symbols if symbols is not None else normalized_sector_map
    selected_symbols = tuple(sorted({normalize_symbol(symbol) for symbol in raw_symbols if normalize_symbol(symbol)}))
    selected_dates = tuple(sorted(set(trade_dates or _dates_from_bars_by_key(bars_by_key))))
    cutoff_label = cutoff_label_for(cutoff)
    cutoff_bars = _cutoff_bars_by_key(bars_by_key, selected_dates, selected_symbols, cutoff)
    cutoff_volumes = {
        key: sum(max(float(bar.volume), 0.0) for bar in bars)
        for key, bars in cutoff_bars.items()
    }
    rel_volumes = _relative_cutoff_volume_map(cutoff_volumes)
    features: dict[tuple[date | None, str], SectorIntradayFeature] = {}
    sectors: dict[tuple[date | None, str], SectorIntradaySector] = {}

    for trade_day in selected_dates:
        members: list[SectorIntradayMember] = []
        for symbol in selected_symbols:
            sector = normalized_sector_map.get(symbol)
            if not sector:
                continue
            bars = cutoff_bars.get((trade_day, symbol), ())
            if not bars:
                continue
            rel_volume = rel_volumes.get((trade_day, symbol), 1.0)
            member = member_from_bars(symbol, sector, trade_day, bars, rel_volume=rel_volume)
            if member is not None:
                members.append(member)
        day_panel = score_sector_members(
            members,
            trade_date=trade_day,
            cutoff_label=cutoff_label,
            min_effective_members=min_effective_members,
            shrinkage_k=shrinkage_k,
        )
        features.update(day_panel.features_by_key)
        sectors.update(day_panel.sectors_by_key)

    return SectorIntradayPanel(features, sectors, cutoff_label=cutoff_label)


def score_sector_members(
    members: Sequence[SectorIntradayMember],
    *,
    trade_date: date | None = None,
    cutoff_label: str = "",
    target_symbols: Iterable[str] | None = None,
    min_effective_members: int = 3,
    shrinkage_k: float = 3.0,
) -> SectorIntradayPanel:
    normalized_members = tuple(
        SectorIntradayMember(
            symbol=normalize_symbol(member.symbol),
            sector=normalize_sector(member.sector),
            ret=float(member.ret),
            vwap_ret=float(member.vwap_ret),
            close_location=_bounded(float(member.close_location), 0.0, 1.0),
            rel_volume=max(float(member.rel_volume), 0.0),
            volume=max(float(member.volume), 0.0),
            bar_count=max(int(member.bar_count), 0),
            trade_date=member.trade_date or trade_date,
        )
        for member in members
        if normalize_symbol(member.symbol) and normalize_sector(member.sector) != "UNKNOWN"
    )
    targets = {normalize_symbol(symbol) for symbol in target_symbols or [member.symbol for member in normalized_members] if normalize_symbol(symbol)}
    by_symbol = {member.symbol: member for member in normalized_members}
    by_sector: dict[str, list[SectorIntradayMember]] = {}
    for member in normalized_members:
        by_sector.setdefault(member.sector, []).append(member)

    sector_summaries = {
        sector: _summarize_members(items, trade_date=trade_date, sector=sector, cutoff_label=cutoff_label)
        for sector, items in by_sector.items()
    }
    pct_by_sector = _percentile_map({sector: summary.raw_score for sector, summary in sector_summaries.items()})
    sectors = {
        (trade_date, sector): SectorIntradaySector(
            trade_date=summary.trade_date,
            sector=summary.sector,
            cutoff_label=summary.cutoff_label,
            member_count=summary.member_count,
            ret=summary.ret,
            breadth=summary.breadth,
            rel_volume=summary.rel_volume,
            participation=summary.participation,
            dispersion=summary.dispersion,
            raw_score=summary.raw_score,
            score_pct=pct_by_sector.get(sector, 50.0),
        )
        for sector, summary in sector_summaries.items()
    }
    sector_raw_values = [summary.raw_score for summary in sector_summaries.values()]

    features: dict[tuple[date | None, str], SectorIntradayFeature] = {}
    for symbol in targets:
        member = by_symbol.get(symbol)
        if member is None:
            features[(trade_date, symbol)] = SectorIntradayFeature.neutral(symbol, trade_date=trade_date, cutoff_label=cutoff_label)
            continue
        peers = [item for item in by_sector.get(member.sector, []) if item.symbol != symbol]
        market_peers = [item for item in normalized_members if item.symbol != symbol]
        peer_summary = _summarize_members(peers, trade_date=trade_date, sector=member.sector, cutoff_label=cutoff_label)
        market_peer_summary = _summarize_members(market_peers, trade_date=trade_date, sector="MARKET", cutoff_label=cutoff_label)
        effective_count = len(peers)
        shrinkage_weight = 1.0 if effective_count >= int(min_effective_members) else effective_count / max(effective_count + float(shrinkage_k), 1e-9)
        raw_score = _blend(peer_summary.raw_score, market_peer_summary.raw_score, shrinkage_weight)
        features[(trade_date, symbol)] = SectorIntradayFeature(
            symbol=symbol,
            sector=member.sector,
            trade_date=trade_date,
            cutoff_label=cutoff_label,
            score_pct=_percentile_for_value(raw_score, sector_raw_values),
            ret=_blend(peer_summary.ret, market_peer_summary.ret, shrinkage_weight),
            breadth=_blend(peer_summary.breadth, market_peer_summary.breadth, shrinkage_weight),
            rel_volume=_blend(peer_summary.rel_volume, market_peer_summary.rel_volume, shrinkage_weight),
            participation=_blend(peer_summary.participation, market_peer_summary.participation, shrinkage_weight),
            effective_count=effective_count,
            shrinkage_weight=shrinkage_weight,
            raw_score=raw_score,
            member_score=member.member_score,
        )
    return SectorIntradayPanel(features, sectors, cutoff_label=cutoff_label)


def member_from_bars(
    symbol: str,
    sector: str,
    trade_date: date,
    bars: Sequence[MarketBar],
    *,
    rel_volume: float = 1.0,
) -> SectorIntradayMember | None:
    ordered = tuple(sorted(bars, key=lambda bar: bar.timestamp))
    if not ordered:
        return None
    first = ordered[0]
    last = ordered[-1]
    open_ = max(float(first.open), 1e-9)
    high = max(float(bar.high) for bar in ordered)
    low = min(float(bar.low) for bar in ordered)
    volume = sum(max(float(bar.volume), 0.0) for bar in ordered)
    vwap_num = sum(((float(bar.high) + float(bar.low) + float(bar.close)) / 3.0) * max(float(bar.volume), 0.0) for bar in ordered)
    vwap = vwap_num / volume if volume > 0.0 else float(last.close)
    range_ = max(high - low, 0.0)
    return SectorIntradayMember(
        symbol=normalize_symbol(symbol),
        sector=normalize_sector(sector),
        trade_date=trade_date,
        ret=float(last.close) / open_ - 1.0,
        vwap_ret=float(last.close) / max(vwap, 1e-9) - 1.0,
        close_location=(float(last.close) - low) / range_ if range_ > 0.0 else 0.5,
        rel_volume=max(float(rel_volume), 0.0),
        volume=volume,
        bar_count=len(ordered),
    )


def completed_session_bars(bars: Any, trade_date: date, *, cutoff: time) -> tuple[MarketBar, ...]:
    out: list[MarketBar] = []
    for bar in sorted(tuple(bars or ()), key=lambda item: item.timestamp):
        ts = ensure_kst(bar.timestamp)
        if ts.date() != trade_date:
            continue
        if ts.time() < SESSION_OPEN or ts.time() >= cutoff:
            continue
        if not bool(bar.is_completed):
            continue
        out.append(bar)
    return tuple(out)


def sector_intraday_metadata(feature: SectorIntradayFeature, *, prefix: str = "") -> dict[str, float | int | str]:
    base = {
        "sector_intraday_score_pct": float(feature.score_pct),
        "sector_intraday_ret": float(feature.ret),
        "sector_intraday_breadth": float(feature.breadth),
        "sector_intraday_rel_volume": float(feature.rel_volume),
        "sector_intraday_participation": float(feature.participation),
        "sector_intraday_effective_count": int(feature.effective_count),
        "sector_intraday_shrinkage_weight": float(feature.shrinkage_weight),
    }
    if not prefix:
        return base
    clean = str(prefix).strip("_")
    return {f"{clean}_{key}": value for key, value in base.items()}


def cutoff_label_for(cutoff: time) -> str:
    return f"timestamp < {cutoff.hour:02d}:{cutoff.minute:02d} KST"


def _summarize_members(
    members: Sequence[SectorIntradayMember],
    *,
    trade_date: date | None,
    sector: str,
    cutoff_label: str,
) -> SectorIntradaySector:
    items = tuple(members or ())
    if not items:
        return SectorIntradaySector(
            trade_date=trade_date,
            sector=normalize_sector(sector),
            cutoff_label=cutoff_label,
            member_count=0,
            ret=0.0,
            breadth=0.5,
            rel_volume=1.0,
            participation=0.0,
            dispersion=0.0,
            raw_score=0.0,
            score_pct=50.0,
        )
    returns = [float(item.ret) for item in items]
    raw_scores = [float(item.member_score) for item in items]
    mean_ret = fmean(returns)
    return SectorIntradaySector(
        trade_date=trade_date,
        sector=normalize_sector(sector),
        cutoff_label=cutoff_label,
        member_count=len(items),
        ret=mean_ret,
        breadth=sum(1 for item in items if item.ret > 0.0) / len(items),
        rel_volume=fmean(max(float(item.rel_volume), 0.0) for item in items),
        participation=sum(1 for item in items if item.ret > 0.0 and item.rel_volume >= 1.0) / len(items),
        dispersion=(fmean((value - mean_ret) ** 2 for value in returns) ** 0.5) if len(returns) > 1 else 0.0,
        raw_score=fmean(raw_scores),
        score_pct=50.0,
    )


def _lookup_bars(bars_by_key: Mapping[Any, Any], trade_date: date, symbol: str) -> Any:
    key = (trade_date, symbol)
    if key in bars_by_key:
        return bars_by_key[key]
    if symbol in bars_by_key:
        return bars_by_key[symbol]
    return ()


def _dates_from_bars_by_key(bars_by_key: Mapping[Any, Any]) -> tuple[date, ...]:
    dates: set[date] = set()
    for key, bars in bars_by_key.items():
        if isinstance(key, tuple) and key and isinstance(key[0], date):
            dates.add(key[0])
            continue
        for bar in tuple(bars or ()):
            dates.add(ensure_kst(bar.timestamp).date())
    return tuple(sorted(dates))


def _cutoff_bars_by_key(
    bars_by_key: Mapping[Any, Any],
    trade_dates: Sequence[date],
    symbols: Sequence[str],
    cutoff: time,
) -> dict[tuple[date, str], tuple[MarketBar, ...]]:
    out: dict[tuple[date, str], tuple[MarketBar, ...]] = {}
    for trade_day in trade_dates:
        for symbol in symbols:
            bars = completed_session_bars(_lookup_bars(bars_by_key, trade_day, symbol), trade_day, cutoff=cutoff)
            if bars:
                out[(trade_day, symbol)] = bars
    return out


def _relative_cutoff_volume_map(cutoff_volumes: Mapping[tuple[date, str], float]) -> dict[tuple[date, str], float]:
    out: dict[tuple[date, str], float] = {}
    history_by_symbol: dict[str, list[float]] = {}
    for (trade_day, symbol), volume in sorted(cutoff_volumes.items()):
        current = max(float(volume), 0.0)
        history = history_by_symbol.setdefault(symbol, [])
        out[(trade_day, symbol)] = current / max(fmean(history[-20:]), 1.0) if history else 1.0
        if current > 0.0:
            history.append(current)
    return out


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


def _blend(primary: float, fallback: float, weight: float) -> float:
    bounded_weight = _bounded(float(weight), 0.0, 1.0)
    return bounded_weight * float(primary) + (1.0 - bounded_weight) * float(fallback)


def _bounded(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))
