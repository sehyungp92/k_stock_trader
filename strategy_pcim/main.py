"""PCIM Strategy Main Orchestration."""

import asyncio
import os
from datetime import datetime, date, time
from typing import Dict, List, Optional
from loguru import logger

import yaml

from kis_core import KoreaInvestEnv, KoreaInvestAPI, RateBudget, build_kis_config_from_env
from oms_client import OMSClient, Intent, IntentType, Urgency, TimeHorizon, RiskPayload

from .config.constants import STRATEGY_ID, TIMING, PORTFOLIO, INTRADAY_HALT_KOSPI_DD_PCT, SIGNAL_EXTRACTION
from .config.switches import pcim_switches
from .external.youtube.watcher import YouTubeWatcher
from .external.youtube.models import ChannelConfig
from .external.transcript.fetcher import fetch_transcript
from .external.gemini.client import GeminiClient
from .external.gemini.extractor import SignalExtractor
from .pipeline.candidate import Candidate
from .pipeline.filters import apply_hard_filters, apply_gap_reversal_filter, compute_soft_multiplier
from .pipeline.gap_reversal import compute_gap_reversal_rate
from .pipeline.trend_gate import check_trend_gate
from .premarket.regime import compute_regime
from .premarket.bucketing import apply_bucketing
from .premarket.tier import apply_tier
from .premarket.sizing import compute_sizing
from .execution.bucket_a import check_bucket_a_trigger
from .execution.bucket_b import check_bucket_b_trigger
from .execution.vetoes import check_execution_veto
from .execution.orders import create_entry_intent, create_exit_intent, create_partial_exit_intent
from .positions.manager import PositionManager, PCIMPosition
from .positions.stops import check_stop_hit
from .positions.profit_taking import check_take_profit
from .positions.trailing import update_trailing_stop_eod
from .positions.time_exit import check_time_exit
from .analytics.hit_tracker import BucketAHitTracker


def load_config() -> dict:
    config_path = os.getenv("PCIM_CONFIG", "config/settings.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError(f"Config file {config_path} is empty or invalid")
    return cfg


def load_channels() -> List[ChannelConfig]:
    config_dir = os.path.dirname(os.getenv("PCIM_CONFIG", "config/settings.yaml"))
    path = os.path.join(config_dir, "influencers.yaml")
    with open(path) as f:
        data = yaml.safe_load(f)
    return [ChannelConfig(**ch) for ch in data.get("channels", [])]


def get_kst_now() -> datetime:
    from zoneinfo import ZoneInfo
    return datetime.now(tz=ZoneInfo("Asia/Seoul"))


def _log_entry_decision(c: Candidate, trigger_type: str, quote: dict, vol_ratio: float = 0.0):
    """Log decision snapshot for post-mortem analysis."""
    logger.info(
        f"ENTRY_DECISION | {c.symbol} | influencer={c.influencer_id} "
        f"bucket={c.bucket} tier={c.tier} "
        f"trigger={trigger_type} gap_pct={c.gap_pct:.4f} "
        f"conviction_score={c.conviction_score:.2f} influencers={c.influencer_count} "
        f"soft_mult={c.soft_mult:.2f} gap_rev_rate={c.gap_rev_rate:.2f} "
        f"raw_qty={c.raw_qty} final_qty={c.final_qty} "
        f"notional={c.final_notional:.0f} quote_last={quote.get('last', 0)} "
        f"vol_ratio={vol_ratio:.2f}"
    )


def consolidate_signals(candidates: List[Candidate]) -> List[Candidate]:
    """
    Consolidate signals when multiple influencers recommend same stock.
    - Average conviction scores
    - Track influencer count for priority boost
    """
    by_symbol: Dict[str, List[Candidate]] = {}
    for c in candidates:
        by_symbol.setdefault(c.symbol, []).append(c)

    consolidated = []
    boost = SIGNAL_EXTRACTION["CONSOLIDATION_BOOST"]

    for symbol, group in by_symbol.items():
        if len(group) == 1:
            consolidated.append(group[0])
        else:
            # Multiple influencers - boost conviction
            avg_conviction = sum(c.conviction_score for c in group) / len(group)
            # Boost: +0.05 per additional influencer, cap at 1.0
            boosted = min(1.0, avg_conviction + boost * (len(group) - 1))

            # Use first candidate as base, update conviction
            merged = group[0]
            merged.conviction_score = boosted
            merged.influencer_count = len(group)
            consolidated.append(merged)

            logger.info(
                f"CONSOLIDATE: {symbol} from {len(group)} influencers, "
                f"avg={avg_conviction:.2f} boosted={boosted:.2f}"
            )

    return consolidated


async def _cancel_and_handle_partial_fills(
    entry_submitted: Dict[str, int],
    position_manager: PositionManager,
    oms: OMSClient,
    api: KoreaInvestAPI,
    bucket_a_pending: Dict[str, int] = None,
    bucket_a_tracker: BucketAHitTracker = None,
):
    """Cancel unfilled entries at 10:00 and handle partial fills."""
    keep_pct = PORTFOLIO["KEEP_PARTIAL_FILL_PCT"]
    bucket_a_pending = bucket_a_pending or {}

    for symbol, intended_qty in entry_submitted.items():
        # Cancel any working entry orders
        cancel_intent = Intent(
            intent_type=IntentType.CANCEL_ORDERS,
            strategy_id=STRATEGY_ID,
            symbol=symbol,
            urgency=Urgency.HIGH,
            time_horizon=TimeHorizon.SWING,
            risk_payload=RiskPayload(rationale_code="10:00_cutoff"),
        )
        await oms.submit_intent(cancel_intent)

        pos = position_manager.get_position(symbol)
        if not pos or pos.status != "OPEN":
            continue

        # Check actual fill from OMS allocation
        oms_pos = await oms.get_position(symbol)
        actual_qty = oms_pos.get_allocation(STRATEGY_ID) if oms_pos else pos.remaining_qty

        if actual_qty < intended_qty:
            # Update position to reflect actual fill
            pos.qty = actual_qty
            pos.remaining_qty = actual_qty

            fill_pct = actual_qty / intended_qty if intended_qty > 0 else 0
            logger.info(f"{symbol}: Partial fill {fill_pct:.0%} ({actual_qty}/{intended_qty})")

            if fill_pct < keep_pct:
                exit_intent = create_exit_intent(symbol, actual_qty, "PARTIAL_FILL_EXIT", Urgency.HIGH)
                await oms.submit_intent(exit_intent)
                position_manager.close_position(symbol, "PARTIAL_FILL_EXIT")
                logger.info(f"{symbol}: Exiting dust position (fill {fill_pct:.0%} < {keep_pct:.0%})")

    # Track Bucket A misses (triggered but not filled by cutoff)
    if bucket_a_tracker:
        for symbol in list(bucket_a_pending.keys()):
            pos = position_manager.get_position(symbol)
            if not pos or pos.status != "OPEN":
                # Never filled → miss
                bucket_a_tracker.record_trigger(filled=False)
            del bucket_a_pending[symbol]

    entry_submitted.clear()


async def run_pcim():
    """Main PCIM strategy orchestration."""
    logger.info("Starting PCIM-Alpha v1.3.1")

    cfg = load_config()

    # Load conservative switches if CONSERVATIVE_MODE=true
    if os.getenv("CONSERVATIVE_MODE", "false").lower() == "true":
        pcim_switches.update_from_yaml("/app/config/conservative.yaml")
    pcim_switches.log_active_config()

    channels = load_channels()

    env = KoreaInvestEnv(build_kis_config_from_env())
    api = KoreaInvestAPI(env)

    # Connect to OMS service
    oms = OMSClient(os.environ.get("OMS_URL", "http://localhost:8000"), strategy_id=STRATEGY_ID)
    await oms.wait_ready()

    # Rate budget for REST calls (market data only - order flow goes via OMS)
    rate_budget = RateBudget()

    gemini_client = GeminiClient()
    signal_extractor = SignalExtractor(gemini_client)
    youtube_watcher = YouTubeWatcher(channels)
    position_manager = PositionManager()

    # Reconcile existing positions from OMS at startup
    today = get_kst_now().date()
    await position_manager.reconcile_from_oms(oms, api, today)
    logger.info(f"Startup: Reconciled {len(position_manager.get_open_positions())} positions from OMS")

    # Load Bucket A hit tracker for adaptive volume threshold
    state_dir = os.path.dirname(os.getenv("PCIM_CONFIG", "config/settings.yaml"))
    bucket_a_tracker = BucketAHitTracker.load(state_dir)
    bucket_a_pending: Dict[str, int] = {}  # symbol -> intended_qty for tracking fills

    # Runtime state
    candidates: List[Candidate] = []
    approved_watchlist: List[Candidate] = []
    regime = None
    kospi_prev_close = None
    kospi_closes = []
    intraday_halted = False
    entry_submitted: Dict[str, int] = {}  # symbol -> intended_qty
    cancel_done_today = False
    # Phase-completion flags to prevent repeated computation
    stats_done_today = False
    premarket_done_today = False
    import time as _time
    last_night_pipeline_ts = 0.0
    night_pipeline_interval = 3600.0  # seconds (1 hour)
    last_heartbeat_ts = 0.0
    heartbeat_interval = 30.0  # seconds

    while True:
        now = get_kst_now()
        today = now.date()
        now_ts = _time.time()

        # Periodic heartbeat
        if now_ts - last_heartbeat_ts > heartbeat_interval:
            open_positions = position_manager.get_open_positions()
            await oms.report_heartbeat(
                mode="RUNNING",
                symbols_hot=len(approved_watchlist),
                positions_count=len(open_positions),
                version="1.3.1",
            )
            last_heartbeat_ts = now_ts

        # =================================================================
        # NIGHT PIPELINE (20:00-06:00) - run every hour to catch late videos
        # =================================================================
        in_night_window = now.time() >= time(20, 0) or now.time() <= time(6, 0)
        if in_night_window and (now_ts - last_night_pipeline_ts >= night_pipeline_interval):
            last_night_pipeline_ts = now_ts
            logger.info("Night pipeline: Checking for new videos")

            new_videos = youtube_watcher.check_all_channels()
            for video in new_videos:
                raw_transcript = fetch_transcript(video.url)
                if not raw_transcript:
                    continue

                # Clean up transcript before extraction
                transcript = signal_extractor.punctuate_transcript(raw_transcript)
                result = signal_extractor.extract_signals(transcript, video_id=video.video_id)
                if not result or not result.signals:
                    continue

                for signal in result.signals:
                    # Use LLM-provided ticker if available, otherwise resolve
                    if signal.ticker:
                        symbol = signal.ticker
                    else:
                        symbol = api.resolve_symbol(signal.company_name)

                    if not symbol:
                        logger.warning(f"Could not resolve symbol for {signal.company_name}")
                        continue

                    candidates.append(Candidate(
                        influencer_id=video.influencer_id,
                        video_id=video.video_id,
                        symbol=symbol,
                        company_name=signal.company_name,
                        conviction_score=signal.conviction_score,
                    ))
                    logger.info(
                        f"RECOMMENDATION: influencer={video.influencer_id} "
                        f"channel={video.channel_name} symbol={symbol} "
                        f"company={signal.company_name} conviction={signal.conviction_score:.2f}"
                    )

            logger.info(f"Night pipeline: {len(candidates)} candidates")

            # Consolidate signals from multiple influencers recommending same stock
            if candidates:
                candidates = consolidate_signals(candidates)
                logger.info(f"After consolidation: {len(candidates)} unique symbols")

        # =================================================================
        # DAILY STATS REFRESH (by 06:00) - run once per day
        # =================================================================
        if now.time() < time(6, 0) and candidates and not stats_done_today:
            logger.info("Refreshing daily stats")
            stats_done_today = True
            acct = await oms.get_account_state()
            equity = acct.equity or 100_000_000

            for c in candidates:
                if c.is_rejected():
                    continue

                bars = api.get_daily_ohlcv(c.symbol, days=120)
                if not bars or len(bars) < 20:
                    c.reject_reason = "INSUFFICIENT_DATA"
                    continue

                closes = [b['close'] for b in bars]
                c.close_prev = closes[-1]
                c.sma20 = sum(closes[-20:]) / 20
                c.atr_20d = api.get_atr_20d(c.symbol)
                c.adtv_20d = api.get_adtv_20d(c.symbol)
                c.market_cap = api.get_market_cap(c.symbol)

                if not check_trend_gate(closes):
                    c.reject_reason = "TREND_GATE_FAIL"
                    continue
                c.pass_trend_gate = True

                has_earnings = api.earnings_within_days(c.symbol, 5)
                reject = apply_hard_filters(c, has_earnings)
                if reject:
                    c.reject_reason = reject
                    continue

                gap_result = compute_gap_reversal_rate(bars)
                c.gap_rev_rate = gap_result.rate
                c.gap_rev_events = gap_result.event_count
                c.gap_rev_insufficient = gap_result.insufficient_sample

                reject = apply_gap_reversal_filter(c)
                if reject:
                    c.reject_reason = reject
                    continue

                five_day_ret = (closes[-1] / closes[-5] - 1) if len(closes) >= 5 else 0
                c.soft_mult = compute_soft_multiplier(c, five_day_ret)

            kospi_bars = api.get_index_daily("KOSPI", days=120)
            kospi_closes = [b['close'] for b in kospi_bars]
            kospi_prev_close = kospi_closes[-1] if kospi_closes else None

        # =================================================================
        # APPROVAL WINDOW (08:00-08:30)
        # =================================================================
        if time(8, 0) <= now.time() <= time(8, 30):
            eligible = [c for c in candidates if not c.is_rejected()]

            if SIGNAL_EXTRACTION["HUMAN_APPROVAL_REQUIRED"]:
                logger.info(f"Approval window: {len(eligible)} eligible candidates awaiting human approval")
                # TODO: Implement actual approval mechanism (e.g., via API/UI)
                approved_watchlist = eligible
            else:
                # Auto-approve all eligible candidates
                if not approved_watchlist and eligible:
                    approved_watchlist = eligible
                    logger.info(f"Auto-approved {len(eligible)} candidates")

            # Compute regime
            if kospi_closes and regime is None:
                regime = compute_regime(kospi_closes)
                await oms.set_regime(regime.name)

        # =================================================================
        # PREMARKET CLASSIFICATION (08:40-09:00) - run once per day
        # =================================================================
        if time(8, 40) <= now.time() <= time(9, 0) and regime and not premarket_done_today:
            logger.info("Premarket classification")
            premarket_done_today = True
            acct = await oms.get_account_state()
            equity = acct.equity or 100_000_000

            for c in approved_watchlist:
                if c.is_rejected():
                    continue

                expected_open = api.get_expected_open(c.symbol)
                if not expected_open:
                    c.reject_reason = "NO_EXPECTED_OPEN"
                    continue

                c = apply_bucketing(c, expected_open, regime)
                if c.is_rejected():
                    continue
                c = apply_tier(c)
                if c.is_rejected():
                    continue
                c = compute_sizing(c, equity)
                if c.is_rejected():
                    continue
                c.priority_key = c.compute_priority_key()

            # Select under caps
            eligible = [c for c in approved_watchlist if not c.is_rejected()]
            eligible.sort(key=lambda x: x.priority_key or (99, 99, 99, 0))

            open_positions = position_manager.get_open_positions()
            max_slots = PORTFOLIO["MAX_OPEN_POSITIONS"] - len(open_positions)
            # Use mark-to-market for exposure instead of entry_price
            current_exposure = 0.0
            for p in open_positions:
                q = api.get_quote(p.symbol)
                current_exposure += p.remaining_qty * q.get('last', p.entry_price) if q else p.remaining_qty * p.entry_price
            max_exposure = regime.max_exposure * equity

            selected = []
            for c in eligible:
                if len(selected) >= max_slots:
                    logger.info(
                        f"PREMARKET_SELECT: {c.symbol} REJECTED max_positions "
                        f"(slots={len(selected)}/{max_slots})"
                    )
                    c.reject_reason = "MAX_POSITIONS"
                    continue
                if current_exposure + c.final_notional > max_exposure:
                    logger.info(
                        f"PREMARKET_SELECT: {c.symbol} REJECTED exposure_cap "
                        f"(cumulative={current_exposure:.0f}+{c.final_notional:.0f}={current_exposure+c.final_notional:.0f} > {max_exposure:.0f})"
                    )
                    c.reject_reason = "EXPOSURE_CAP"
                    continue
                logger.info(
                    f"PREMARKET_SELECT: {c.symbol} ACCEPTED notional={c.final_notional:.0f} "
                    f"cumulative_exposure={current_exposure+c.final_notional:.0f}/{max_exposure:.0f}"
                )
                selected.append(c)
                current_exposure += c.final_notional

            approved_watchlist = selected
            logger.info(f"Premarket selection complete: {len(selected)} candidates for execution")

        # =================================================================
        # EXECUTION WINDOW (09:01 until cutoff)
        # =================================================================
        # Use switch-configurable entry cutoff (default 10:30, conservative 10:00)
        cancel_at = pcim_switches.entry_cutoff
        strict_cutoff = TIMING["CANCEL_ENTRIES_AT"]

        # Log would-block if we're in the window between strict and permissive cutoff
        if time(strict_cutoff[0], strict_cutoff[1]) <= now.time() <= time(cancel_at[0], cancel_at[1]):
            pcim_switches.log_would_block(
                "TIMING",
                "ENTRY_CUTOFF",
                now.time().strftime("%H:%M"),
                f"{strict_cutoff[0]:02d}:{strict_cutoff[1]:02d}",
            )

        if time(9, 1) <= now.time() <= time(cancel_at[0], cancel_at[1]) and not intraday_halted:
            # Intraday halt check
            if kospi_prev_close:
                kospi_now = api.get_index_realtime("KOSPI")
                dd = (kospi_now - kospi_prev_close) / kospi_prev_close
                if dd <= INTRADAY_HALT_KOSPI_DD_PCT:
                    logger.warning(f"INTRADAY HALT: KOSPI DD {dd:.2%}")
                    intraday_halted = True
                    await asyncio.sleep(5)
                    continue

            # First, check pending orders for fills
            all_positions = await oms.get_all_positions()
            for symbol in list(position_manager.pending_orders.keys()):
                oms_pos = all_positions.get(symbol)
                alloc_qty = oms_pos.get_allocation(STRATEGY_ID) if oms_pos else 0
                alloc_obj = oms_pos.allocations.get(STRATEGY_ID) if oms_pos else None
                if alloc_qty > 0:
                    pending = position_manager.clear_pending(symbol)
                    if pending:
                        avg_price = alloc_obj.cost_basis if alloc_obj else 0.0
                        position_manager.add_position(PCIMPosition(
                            symbol=symbol,
                            entry_date=today,
                            entry_price=avg_price,
                            qty=alloc_qty,
                            atr_at_entry=pending['atr'],
                        ))
                        logger.info(f"{symbol}: Fill confirmed, position created @ {avg_price:.0f} qty={alloc_qty}")
                        # Track Bucket A hit (filled)
                        if symbol in bucket_a_pending:
                            bucket_a_tracker.record_trigger(filled=True)
                            del bucket_a_pending[symbol]

            for c in approved_watchlist:
                if c.is_rejected():
                    continue
                if position_manager.get_position(c.symbol):
                    continue
                if position_manager.was_submitted_today(c.symbol):
                    continue  # Idempotency: already submitted today

                if not rate_budget.try_consume("QUOTE"):
                    continue  # Skip this tick, retry next loop
                quote = api.get_quote(c.symbol)
                upper_limit = api.get_upper_limit_price(c.symbol, today)
                tick_size = api.get_tick_size(c.symbol)
                is_vi = api.is_in_vi(c.symbol)

                veto = check_execution_veto(quote, upper_limit, tick_size, is_vi)
                if veto:
                    bid = quote.get('bid', 0)
                    ask = quote.get('ask', 0)
                    last = quote.get('last', 0)
                    spread_pct = (ask - bid) / last if last > 0 else 0
                    upper_dist = (upper_limit - last) / tick_size if tick_size > 0 and upper_limit > 0 else 999
                    logger.info(
                        f"EXECUTION_VETO: {c.symbol} veto={veto} last={last:.0f} "
                        f"spread={spread_pct:.4f} upper_dist_ticks={upper_dist:.1f} vi={is_vi}"
                    )
                    continue

                # Bucket A trigger (after 09:03:05)
                if c.bucket == "A" and now.time() >= time(9, 3, 5) and rate_budget.try_consume("CHART"):
                    bar_3m = api.get_intraday_3m(c.symbol, "09:00", "09:03")
                    if bar_3m:
                        baseline = api.get_open_3m_baseline(c.symbol, 20)
                        # Use adaptive threshold based on hit-rate
                        adaptive_threshold = bucket_a_tracker.calibrated_threshold()
                        signal = check_bucket_a_trigger(bar_3m[-1], baseline, vol_threshold=adaptive_threshold)
                        if signal.triggered:
                            _log_entry_decision(c, "ORB", quote, signal.vol_ratio)
                            # Bucket A: 30-second fill timeout per spec
                            intent = create_entry_intent(c, quote['last'], urgency=Urgency.HIGH, expiry_ts=_time.time() + 30)
                            result = await oms.submit_intent(intent)
                            # EXECUTED means order submitted, not filled. Track as pending.
                            if result.status.name in ("EXECUTED", "APPROVED"):
                                position_manager.track_pending(c.symbol, intent.intent_id, c.final_qty, c.atr_20d)
                                entry_submitted[c.symbol] = c.final_qty
                                bucket_a_pending[c.symbol] = c.final_qty  # Track for hit-rate
                                c.reject_reason = "PENDING"

                # Bucket B trigger (after 09:10)
                if c.bucket == "B" and now.time() >= time(9, 10) and rate_budget.try_consume("CHART"):
                    bars_1m = api.get_intraday_1m(c.symbol, "09:00", now.strftime("%H:%M"))
                    if bars_1m:
                        signal = check_bucket_b_trigger(bars_1m)
                        if signal.triggered:
                            _log_entry_decision(c, "VWAP_RECLAIM", quote)
                            intent = create_entry_intent(c, quote['last'])
                            result = await oms.submit_intent(intent)
                            # EXECUTED means order submitted, not filled. Track as pending.
                            if result.status.name in ("EXECUTED", "APPROVED"):
                                position_manager.track_pending(c.symbol, intent.intent_id, c.final_qty, c.atr_20d)
                                entry_submitted[c.symbol] = c.final_qty
                                c.reject_reason = "PENDING"

        # =================================================================
        # 10:00 CANCEL + PARTIAL FILL HANDLING
        # =================================================================
        if (now.time() >= time(cancel_at[0], cancel_at[1])
                and not cancel_done_today and entry_submitted):
            await _cancel_and_handle_partial_fills(
                entry_submitted, position_manager, oms, api,
                bucket_a_pending=bucket_a_pending, bucket_a_tracker=bucket_a_tracker
            )
            cancel_done_today = True

        # =================================================================
        # POSITION MANAGEMENT (10:00+)
        # =================================================================
        if now.time() >= time(10, 0):
            for pos in position_manager.get_open_positions():
                quote = api.get_quote(pos.symbol)
                current_price = quote['last']

                if check_stop_hit(pos, current_price):
                    intent = create_exit_intent(pos.symbol, pos.remaining_qty, "STOP", Urgency.HIGH)
                    result = await oms.submit_intent(intent)
                    if result.status.name in ("EXECUTED", "APPROVED"):
                        position_manager.close_position(pos.symbol, "STOP")
                    else:
                        logger.warning(f"{pos.symbol}: Stop exit {result.status.name} - {result.message}")
                    continue

                should_tp, qty = check_take_profit(pos, current_price)
                if should_tp:
                    intent = create_partial_exit_intent(pos.symbol, qty, "TAKE_PROFIT")
                    result = await oms.submit_intent(intent)
                    if result.status.name in ("EXECUTED", "APPROVED"):
                        pos.tp_done = True
                        position_manager.reduce_position(pos.symbol, qty)
                    else:
                        logger.warning(f"{pos.symbol}: Take profit {result.status.name} - {result.message}")

                # Use KRX trading calendar for day count if available
                is_trading_day = getattr(api, 'is_trading_day', None)
                if check_time_exit(pos, today, is_trading_day):
                    intent = create_exit_intent(pos.symbol, pos.remaining_qty, "DAY15_EXIT")
                    result = await oms.submit_intent(intent)
                    if result.status.name in ("EXECUTED", "APPROVED"):
                        position_manager.close_position(pos.symbol, "DAY15_EXIT")
                    else:
                        logger.warning(f"{pos.symbol}: Time exit {result.status.name} - {result.message}")

        # =================================================================
        # EOD TRAILING UPDATE
        # =================================================================
        if time(15, 35) <= now.time() < time(15, 40):
            for pos in position_manager.get_open_positions():
                bars = api.get_daily_ohlcv(pos.symbol, days=30)
                if bars:
                    close_today = bars[-1]['close']
                    atr20 = api.get_atr_20d(pos.symbol)
                    update_trailing_stop_eod(pos, close_today, atr20)

        # Reset for next day
        if now.time() >= time(18, 0):
            candidates = []
            approved_watchlist = []
            intraday_halted = False
            entry_submitted.clear()
            bucket_a_pending.clear()
            cancel_done_today = False
            stats_done_today = False
            premarket_done_today = False
            last_night_pipeline_ts = 0.0
            position_manager.reset_daily_state()
            # Save and potentially reset Bucket A hit tracker
            bucket_a_tracker.reset_if_new_period(today)
            bucket_a_tracker.save(state_dir)
            logger.info(f"Bucket A adaptive threshold: {bucket_a_tracker.calibrated_threshold():.2f} "
                       f"(hit_rate={bucket_a_tracker.hit_rate:.2%})")

        await asyncio.sleep(5)

    await oms.close()


def main():
    """Entry point."""
    asyncio.run(run_pcim())


if __name__ == "__main__":
    main()
