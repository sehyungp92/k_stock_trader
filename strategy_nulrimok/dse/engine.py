"""Nulrimok Daily Selection Engine."""

import math
from datetime import date
from typing import Dict, List
from loguru import logger

from kis_core import sma, percentile_rank
from ..lrs.db import LRSDatabase
from ..lrs.regime import compute_regime
from ..lrs.flow import compute_flow_score
from ..lrs.avwap import compute_avwap_reference
from .artifact import WatchlistArtifact, TickerArtifact, PositionArtifact
from ..config.constants import (
    LEADER_TIER_A_PCT, LEADER_TIER_B_PCT, DAILY_RANK_FLOW_WEIGHT, DAILY_RANK_LEADER_WEIGHT,
    DAILY_RANK_SECTOR_WEIGHT, DAILY_RANK_AVWAP_WEIGHT, TRADABLE_TIER_A_PCT, TRADABLE_TIER_B_PCT,
    ACTIVE_SET_K, SECTOR_RANK_1, SECTOR_RANK_2, SECTOR_RANK_3, SECTOR_RANK_OTHER,
    SECTOR_FLOW_WEIGHT, SECTOR_BREADTH_WEIGHT, SECTOR_PART_WEIGHT, FLOW_PERSISTENCE_MIN,
    SIZE_TOP_20_MULT, SIZE_20_50_MULT, SIZE_50_80_MULT,
)
from ..config.switches import nulrimok_switches


class DailySelectionEngine:
    def __init__(self, lrs: LRSDatabase, universe: List[str]):
        self.lrs = lrs
        self.universe = universe

    def run(self, today: date, held_positions: List[dict]) -> WatchlistArtifact:
        logger.info(f"DSE: Starting for {today}")

        regime = compute_regime(self.lrs)
        logger.info(f"DSE: Regime tier={regime.tier}, score={regime.score:.2f}")

        if regime.tier == "C" and not nulrimok_switches.allow_tier_c_reduced:
            # Still compute flow reversal flags for held positions even in Tier C
            positions = [PositionArtifact(
                ticker=p["ticker"], entry_time=p.get("entry_time", ""), avg_price=p.get("avg_price", 0),
                qty=p.get("qty", 0), stop=p.get("stop", 0),
                flow_reversal_flag=self._check_flow_reversal(p["ticker"]),
                exit_at_open=self._check_flow_reversal(p["ticker"]),
            ) for p in held_positions]
            return WatchlistArtifact(date=today.isoformat(), regime_tier=regime.tier, regime_score=regime.score, positions=positions)

        sector_weights = self._compute_sector_weights()

        candidates = []
        for ticker in self.universe:
            artifact = self._process_ticker(ticker, regime, sector_weights)
            if artifact:
                candidates.append(artifact)

        candidates = self._rank_candidates(candidates)
        tradable = self._select_tradable(candidates, regime.tier)

        active_set = [c.ticker for c in tradable[:ACTIVE_SET_K]]
        overflow = [c.ticker for c in tradable[ACTIVE_SET_K:]]

        positions = [PositionArtifact(
            ticker=p["ticker"], entry_time=p.get("entry_time", ""), avg_price=p.get("avg_price", 0),
            qty=p.get("qty", 0), stop=p.get("stop", 0),
            flow_reversal_flag=self._check_flow_reversal(p["ticker"]),
            exit_at_open=self._check_flow_reversal(p["ticker"]),
        ) for p in held_positions]

        artifact = WatchlistArtifact(
            date=today.isoformat(), regime_tier=regime.tier, regime_score=regime.score,
            candidates=candidates, tradable=[c.ticker for c in tradable],
            active_set=active_set, overflow=overflow, positions=positions,
        )

        self.lrs.save_artifact(today, artifact.to_dict())
        logger.info(f"DSE: Complete. Tradable={len(tradable)}, Active={len(active_set)}")
        return artifact

    def _compute_sector_weights(self) -> Dict[str, float]:
        sectors = set(self.lrs.get_sector(t) for t in self.universe if self.lrs.get_sector(t))
        if not sectors:
            return {}

        sector_scores = {}
        for sector in sectors:
            members = self.lrs.get_sector_members(sector)
            if not members:
                sector_scores[sector] = 0.0
                continue

            # Flow trend: z-score of 20D foreign flow trend for sector basket
            flow_sums = []
            for t in members:
                flow = self.lrs.get_smart_money_series(t, 20)
                if len(flow) >= 20:
                    flow_sums.append(sum(flow[-20:]))
            flow_trend = sum(flow_sums) / len(flow_sums) if flow_sums else 0.0

            # Breadth: % members with close > SMA20
            above, total = 0, 0
            for t in members:
                closes = self.lrs.get_closes(t, 30)
                if len(closes) >= 20:
                    total += 1
                    if closes[-1] > sum(closes[-20:]) / 20:
                        above += 1
            breadth = above / total if total > 0 else 0.0

            # Participation: % members with persistence >= threshold
            participating, checked = 0, 0
            persistence_min = nulrimok_switches.flow_persistence_min
            for t in members:
                flow = self.lrs.get_smart_money_series(t, 10)
                if len(flow) >= 10:
                    checked += 1
                    pers = sum(1 for x in flow if x > 0) / 10.0
                    if pers >= persistence_min:
                        participating += 1
            participation = participating / checked if checked > 0 else 0.0

            sector_scores[sector] = (SECTOR_FLOW_WEIGHT * flow_trend +
                                     SECTOR_BREADTH_WEIGHT * breadth +
                                     SECTOR_PART_WEIGHT * participation)

        # Normalize flow_trend component via z-score across sectors
        raw_scores = list(sector_scores.values())
        if len(raw_scores) > 1:
            s_mean = sum(raw_scores) / len(raw_scores)
            s_std = math.sqrt(sum((x - s_mean) ** 2 for x in raw_scores) / len(raw_scores)) or 1e-9
            normalized = {s: (v - s_mean) / s_std for s, v in sector_scores.items()}
        else:
            normalized = {s: 0.0 for s in sector_scores}

        ranked = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
        weights = {}
        rank_values = [SECTOR_RANK_1, SECTOR_RANK_2, SECTOR_RANK_3]
        for i, (sector, _) in enumerate(ranked):
            weights[sector] = rank_values[i] if i < 3 else SECTOR_RANK_OTHER
        return weights

    def _process_ticker(self, ticker: str, regime, sector_weights: Dict[str, float]) -> TickerArtifact | None:
        artifact = TickerArtifact(ticker=ticker, regime_tier=regime.tier,
                                  regime_score=regime.score, risk_multiplier=regime.risk_mult)

        sector = self.lrs.get_sector(ticker)
        artifact.sector = sector or ""
        artifact.sector_rank_weight = sector_weights.get(sector, SECTOR_RANK_OTHER)

        sector_tickers = self.lrs.get_sector_members(sector) if sector else []
        flow = compute_flow_score(self.lrs, ticker, sector_tickers)
        artifact.flow_score, artifact.flow_persistence, artifact.flow_pass = flow.score, flow.persistence, flow.passes
        if not flow.passes:
            return None

        rs_pct = self._compute_rs_percentile(ticker)
        artifact.rs_percentile = rs_pct

        # Use switch-configurable RS thresholds
        if regime.tier == "A":
            threshold = nulrimok_switches.leader_tier_a_pct
            strict_threshold = LEADER_TIER_A_PCT
        else:
            threshold = nulrimok_switches.leader_tier_b_pct
            strict_threshold = LEADER_TIER_B_PCT

        artifact.leader_pass = rs_pct >= threshold

        # Log would-block: passed permissive but would fail strict threshold
        if artifact.leader_pass and rs_pct < strict_threshold:
            nulrimok_switches.log_would_block(
                ticker,
                "RS_PERCENTILE",
                rs_pct,
                strict_threshold,
                {"regime_tier": regime.tier},
            )

        if not artifact.leader_pass:
            return None

        trend_ok, sma50 = self._check_trend(ticker)
        artifact.trend_pass, artifact.sma50 = trend_ok, sma50
        if not trend_ok:
            return None

        avwap = compute_avwap_reference(self.lrs, ticker)
        artifact.anchor_date = avwap.anchor_date.isoformat() if avwap.anchor_date else None
        artifact.avwap_ref, artifact.band_lower, artifact.band_upper = avwap.avwap_ref, avwap.band_lower, avwap.band_upper
        artifact.acceptance_pass = avwap.acceptance_pass

        if avwap.avwap_ref > 0:
            last_close = self.lrs.get_closes(ticker, 1)
            if last_close:
                artifact.avwap_proximity = 1 - abs(last_close[0] - avwap.avwap_ref) / avwap.avwap_ref

        # Estimate ATR30m from daily ATR (approx 13 30m bars per day, scale by sqrt ratio)
        artifact.atr30m_est = self._estimate_atr30m(ticker)

        return artifact

    def _compute_rs_percentile(self, ticker: str) -> float:
        closes = self.lrs.get_closes(ticker, 60)
        if len(closes) < 20:
            return 0.0  # Truly insufficient data
        stock_return = (closes[-1] / closes[0]) - 1
        sector = self.lrs.get_sector(ticker)
        if not sector:
            return 50.0
        min_bars = min(len(closes), 60)
        sector_returns = []
        for t in self.lrs.get_sector_members(sector):
            t_closes = self.lrs.get_closes(t, 60)
            if len(t_closes) >= min_bars:
                sector_returns.append((t_closes[-1] / t_closes[0]) - 1)
        return percentile_rank(stock_return, sector_returns) if sector_returns else 50.0

    def _check_trend(self, ticker: str) -> tuple:
        closes = self.lrs.get_closes(ticker, 60)
        if len(closes) < 50:
            return False, 0.0
        sma50_values = sma(closes, 50)
        if not sma50_values:
            return False, 0.0
        current_sma50 = sma50_values[-1]
        slope = sma50_values[-1] - sma50_values[-6] if len(sma50_values) >= 6 else 0
        return (closes[-1] > current_sma50 and slope >= 0), current_sma50

    def _rank_candidates(self, candidates: List[TickerArtifact]) -> List[TickerArtifact]:
        candidates = [c for c in candidates if c]
        if not candidates:
            return []

        flow_scores = [c.flow_score for c in candidates]
        rs_pcts = [c.rs_percentile for c in candidates]
        avwap_proxs = [c.avwap_proximity for c in candidates]

        def norm(val, vals):
            min_v, max_v = min(vals), max(vals)
            return (val - min_v) / (max_v - min_v) if max_v != min_v else 0.5

        for c in candidates:
            c.daily_rank = (DAILY_RANK_FLOW_WEIGHT * norm(c.flow_score, flow_scores) +
                            DAILY_RANK_LEADER_WEIGHT * norm(c.rs_percentile, rs_pcts) +
                            DAILY_RANK_SECTOR_WEIGHT * c.sector_rank_weight +
                            DAILY_RANK_AVWAP_WEIGHT * norm(c.avwap_proximity, avwap_proxs))

        candidates.sort(key=lambda x: x.daily_rank, reverse=True)

        # Conviction sizing: set recommended_risk based on rank percentile
        n = len(candidates)
        for i, c in enumerate(candidates):
            pct = (i / n) * 100 if n > 0 else 50
            if pct < 20:
                c.recommended_risk = 0.005 * SIZE_TOP_20_MULT
            elif pct < 50:
                c.recommended_risk = 0.005 * SIZE_20_50_MULT
            else:
                c.recommended_risk = 0.005 * SIZE_50_80_MULT

        return candidates

    def _select_tradable(self, candidates: List[TickerArtifact], regime_tier: str) -> List[TickerArtifact]:
        if not candidates:
            return []
        cut = max(1, int(len(candidates) * (TRADABLE_TIER_A_PCT if regime_tier == "A" else TRADABLE_TIER_B_PCT)))
        tradable = candidates[:cut]
        for t in tradable:
            t.tradable = True
        return tradable

    def _check_flow_reversal(self, ticker: str) -> bool:
        flow = self.lrs.get_smart_money_series(ticker, 2)
        return len(flow) >= 2 and flow[-1] < 0 and flow[-2] < 0

    def _estimate_atr30m(self, ticker: str) -> float:
        """Estimate 30m ATR from daily bars (14-day ATR scaled to 30m)."""
        bars = self.lrs.get_recent_bars(ticker, 15)
        if len(bars) < 2:
            return 0.0
        tr_values = []
        for i in range(1, len(bars)):
            h, l, pc = bars[i].high, bars[i].low, bars[i - 1].close
            tr = max(h - l, abs(h - pc), abs(l - pc))
            tr_values.append(tr)
        if not tr_values:
            return 0.0
        atr_daily = sum(tr_values[-14:]) / min(14, len(tr_values))
        # Scale daily ATR to 30m: ~13 bars/day, ATR scales by sqrt(time)
        return atr_daily / math.sqrt(13)
