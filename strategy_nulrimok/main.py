"""Nulrimok Strategy Main."""

import asyncio
from collections import defaultdict, deque
from datetime import datetime, date, time
from typing import Dict, Optional
from loguru import logger

import os
from kis_core import (
    KoreaInvestEnv, KoreaInvestAPI, RateBudget, RollingSMA, aggregate_bars,
    KISWebSocketClient, BaseSubscriptionManager, TickMessage,
    SectorExposure, SectorExposureConfig,
    filter_universe,
)
from oms_client import OMSClient

from .config.constants import (
    STRATEGY_ID, DSE_START, DSE_END, IEPE_START, IEPE_END, ACTIVE_SET_K,
    SECTOR_CAP_PCT, FLOW_EXIT_CHECK_START, FLOW_EXIT_START, FLOW_EXIT_END,
)
from .config.switches import nulrimok_switches
from .lrs.db import LRSDatabase
from .dse.engine import DailySelectionEngine
from .dse.artifact import WatchlistArtifact
from .iepe.entry import TickerEntryState, EntryState, process_entry
from .iepe.exits import PositionState, SetupType, manage_nulrimok_position, handle_flow_reversal_exits
from .iepe.rotation import rotate_active_set

VOL_HISTORY_LEN = 20
NEAR_BAND_PCT = 0.01  # Spec §7.3: 1.0% threshold for rotation protection
NEAR_BAND_PCT_RT = 0.005  # Tighter threshold for real-time WS detection (0.5%)
PENDING_FILL_MAX_CYCLES = 4  # Max cycles to wait for fill confirmation
DAILY_RISK_BUDGET_PCT = 0.04  # Max 4% of equity committed as total open risk


def compute_total_open_risk(
    position_states: Dict[str, "PositionState"],
    equity: float,
) -> float:
    """Compute total open risk across all positions as fraction of equity."""
    if equity <= 0:
        return 0.0
    total_risk = 0.0
    for pos in position_states.values():
        risk_per_share = max(pos.entry_price - pos.stop, 0.0)
        total_risk += pos.remaining_qty * risk_per_share
    return total_risk / equity


def load_config() -> dict:
    import os
    import yaml
    with open(os.getenv("NULRIMOK_CONFIG", "config/settings.yaml")) as f:
        return yaml.safe_load(f)


def get_kst_now() -> datetime:
    from zoneinfo import ZoneInfo
    return datetime.now(tz=ZoneInfo("Asia/Seoul"))


async def fetch_30m_bar(api, ticker: str, rate_budget: Optional[RateBudget] = None) -> Optional[dict]:
    if rate_budget and not rate_budget.try_consume("CHART"):
        return None
    try:
        bars_1m = api.get_minute_bars(ticker, minutes=30)
        if bars_1m is None or bars_1m.empty:
            return None
        aggregated = aggregate_bars(bars_1m.to_dict('records'), 30)
        if aggregated:
            bar = aggregated[-1]
            return {'timestamp': bar.timestamp, 'open': bar.open, 'high': bar.high,
                    'low': bar.low, 'close': bar.close, 'volume': bar.volume}
        return None
    except Exception:
        return None


def _build_sector_map(artifact: WatchlistArtifact) -> Dict[str, str]:
    """Build symbol-to-sector mapping from watchlist artifact."""
    sector_map = {}
    for ticker in artifact.all_tickers:
        ticker_art = artifact.get_ticker(ticker)
        if ticker_art and ticker_art.sector:
            sector_map[ticker] = ticker_art.sector
    return sector_map


def _sync_sector_exposure(
    sector_exposure: SectorExposure,
    position_states: Dict[str, PositionState],
) -> None:
    """Sync sector exposure from current position states."""
    positions = {
        ticker: (pos.qty, pos.entry_price)
        for ticker, pos in position_states.items()
    }
    sector_exposure.reconcile(positions)


def _increment_sessions_held(position_states: Dict[str, PositionState]) -> None:
    """Increment sessions_held for all held positions at day rollover."""
    for pos in position_states.values():
        pos.sessions_held += 1


async def _recover_positions(oms, position_states: Dict[str, PositionState]) -> None:
    """Recover position state from OMS allocations on startup/DSE."""
    try:
        allocations = await oms.get_strategy_allocations(STRATEGY_ID)
        for ticker, alloc in allocations.items():
            if ticker in position_states:
                # Already tracked locally, just sync qty
                position_states[ticker].remaining_qty = alloc.qty
            elif alloc.qty > 0:
                # Recovered position not in local state
                entry_ts = getattr(alloc, 'entry_ts', None) or get_kst_now()
                if isinstance(entry_ts, str):
                    from datetime import datetime as dt
                    entry_ts = dt.fromisoformat(entry_ts.replace('Z', '+00:00'))
                pos = PositionState(
                    ticker=ticker,
                    entry_time=entry_ts,
                    entry_price=alloc.cost_basis or 0.0,
                    qty=alloc.qty,
                    stop=getattr(alloc, 'soft_stop_px', 0.0) or 0.0,
                )
                pos.remaining_qty = alloc.qty
                pos.setup = SetupType.UNKNOWN
                # Estimate sessions_held from entry date
                days_held = (get_kst_now().date() - entry_ts.date()).days if hasattr(entry_ts, 'date') else 0
                pos.sessions_held = max(0, days_held)
                position_states[ticker] = pos
                logger.info(f"Recovered position: {ticker} qty={alloc.qty} from OMS")
    except Exception as e:
        logger.warning(f"Position recovery failed: {e}")


async def _reconcile_positions(oms, position_states: Dict[str, PositionState]) -> None:
    """Reconcile local position state against OMS allocations."""
    try:
        allocations = await oms.get_strategy_allocations(STRATEGY_ID)
        oms_tickers = {ticker: alloc for ticker, alloc in allocations.items() if alloc.qty > 0}

        # Check for positions closed externally
        for ticker in list(position_states.keys()):
            if ticker not in oms_tickers:
                logger.info(f"{ticker}: Position closed externally, removing from local state")
                del position_states[ticker]
            elif oms_tickers[ticker].qty < position_states[ticker].remaining_qty:
                # Partial exit happened externally
                position_states[ticker].remaining_qty = oms_tickers[ticker].qty
                logger.info(f"{ticker}: Updated remaining_qty to {oms_tickers[ticker].qty} from OMS")
    except Exception as e:
        logger.debug(f"Reconciliation check failed: {e}")


async def run_nulrimok():
    logger.info("Starting Nulrimok Strategy")
    cfg = load_config()

    env = KoreaInvestEnv(cfg["kis"])
    api = KoreaInvestAPI(env)

    # Connect to OMS service
    oms = OMSClient(os.environ.get("OMS_URL", "http://localhost:8000"), strategy_id=STRATEGY_ID)
    await oms.wait_ready()

    # Rate budget for REST calls (market data only - order flow goes via OMS)
    rate_budget = RateBudget()

    lrs = LRSDatabase(cfg.get("lrs_path", "lrs.db"))
    raw_universe = cfg.get("universe", [])
    filtered_universe, rejected = filter_universe(api, raw_universe)
    for r in rejected:
        logger.warning(f"Universe filter: {r['ticker']} rejected ({r['reason']})")
    dse = DailySelectionEngine(lrs, filtered_universe)

    artifact: Optional[WatchlistArtifact] = None
    entry_states: Dict[str, TickerEntryState] = {}
    position_states: Dict[str, PositionState] = {}
    sma_trackers: Dict[str, RollingSMA] = {}
    vol_histories: Dict[str, deque] = defaultdict(lambda: deque(maxlen=VOL_HISTORY_LEN))
    near_band_recently: Dict[str, bool] = {}
    last_near_band_time: Dict[str, datetime] = {}  # Track when price last entered band

    # Advisory sector exposure tracking (pct-based for Nulrimok).
    # The OMS RiskGateway is the authoritative source for sector caps.
    # This local tracker is a fast pre-filter to avoid unnecessary OMS round-trips.
    sector_config = SectorExposureConfig(
        mode="pct",
        max_sector_pct=SECTOR_CAP_PCT,
        unknown_sector_policy="allow",
    )
    sector_exposure: Optional[SectorExposure] = None

    # --- WebSocket for Active Monitoring Set ---
    ws_client = KISWebSocketClient(api)
    ws_subs: BaseSubscriptionManager | None = None
    active_set_ref: set = set()  # Reference for tick handler (updated when active_set changes)
    prev_active_set: set = set()
    ws_url = cfg.get("ws_url", "")

    def _on_nulrimok_tick(msg: TickMessage) -> None:
        """Real-time tick handler for band proximity detection."""
        nonlocal artifact, near_band_recently, last_near_band_time
        if msg.ticker not in active_set_ref:
            return
        if artifact is None:
            return
        ticker_art = artifact.get_ticker(msg.ticker)
        if ticker_art is None or ticker_art.avwap_ref <= 0:
            return
        # Check if price is within AVWAP band (0.5% for real-time)
        avwap = ticker_art.avwap_ref
        if abs(msg.price - avwap) / avwap < NEAR_BAND_PCT_RT:
            near_band_recently[msg.ticker] = True
            last_near_band_time[msg.ticker] = get_kst_now()

    if ws_url:
        if await ws_client.connect(ws_url):
            ws_client.on_tick(_on_nulrimok_tick)
            asyncio.create_task(ws_client.run())
            ws_subs = BaseSubscriptionManager(ws_client, max_regs=ACTIVE_SET_K)
            logger.info("Nulrimok WebSocket connected for Active Monitoring Set")
        else:
            logger.warning("Nulrimok WebSocket connect failed; using REST only")

    last_30m_boundary, last_rotation, dse_ran_today = None, datetime.min, False
    prev_trade_date: Optional[date] = None
    last_reconcile_cycle = 0
    reconcile_interval = 1  # Reconcile every cycle (~30m) to detect external closes promptly
    import time as _time
    last_heartbeat_ts = 0.0
    heartbeat_interval = 30.0  # seconds

    # Recover positions from OMS on startup
    await _recover_positions(oms, position_states)
    if position_states:
        logger.info(f"Startup: recovered {len(position_states)} positions from OMS")

    flow_exit_done_today = False

    while True:
        now = get_kst_now()
        today = now.date()

        # Periodic heartbeat
        now_ts = _time.time()
        if now_ts - last_heartbeat_ts > heartbeat_interval:
            active_set_count = len(artifact.active_set) if artifact else 0
            await oms.report_heartbeat(
                mode="RUNNING",
                symbols_hot=active_set_count,
                positions_count=len(position_states),
                version="1.0.1",
            )
            last_heartbeat_ts = now_ts

        # Day rollover: increment sessions_held
        if prev_trade_date and today > prev_trade_date and position_states:
            _increment_sessions_held(position_states)
            logger.info(f"Day rollover: incremented sessions_held for {len(position_states)} positions")
        prev_trade_date = today

        # DSE Phase
        if time(DSE_START[0], DSE_START[1]) <= now.time() <= time(DSE_END[0], DSE_END[1]) and not dse_ran_today:
            # Refresh position state from OMS before DSE
            await _recover_positions(oms, position_states)
            held = [{"ticker": t, "entry_time": p.entry_time.isoformat(), "avg_price": p.entry_price,
                     "qty": p.qty, "stop": p.stop} for t, p in position_states.items()]
            artifact = dse.run(today, held)
            entry_states = {t: TickerEntryState(ticker=t) for t in artifact.active_set}
            sma_trackers = {t: RollingSMA(period=5) for t in artifact.active_set}
            dse_ran_today = True

            # Initialize sector exposure with sector map from artifact
            sector_map = _build_sector_map(artifact)
            sector_exposure = SectorExposure(sector_map, sector_config)
            _sync_sector_exposure(sector_exposure, position_states)

            # Sync WS subscriptions with new active set
            if ws_subs and artifact:
                new_active = set(artifact.active_set)
                # Unsubscribe removed
                for t in prev_active_set - new_active:
                    await ws_subs.drop_tick(t)
                # Subscribe new
                for t in new_active - prev_active_set:
                    await ws_subs.ensure_tick(t)
                # Update reference for tick handler
                active_set_ref.clear()
                active_set_ref.update(new_active)
                prev_active_set = new_active.copy()
                logger.debug(f"WS subscriptions synced with active set ({len(new_active)} tickers)")

        if now.hour == 0 and now.minute < 5:
            dse_ran_today = False
            flow_exit_done_today = False

        # Pre-market flow reversal exit phase (08:55-09:01)
        # Execute flow reversal exits in tight window at session open
        if (time(FLOW_EXIT_CHECK_START[0], FLOW_EXIT_CHECK_START[1]) <= now.time() <= time(FLOW_EXIT_END[0], FLOW_EXIT_END[1], FLOW_EXIT_END[2])
                and artifact and not flow_exit_done_today):
            # Only execute in the precise 09:00:05-09:01:00 window
            if time(FLOW_EXIT_START[0], FLOW_EXIT_START[1], FLOW_EXIT_START[2]) <= now.time() <= time(FLOW_EXIT_END[0], FLOW_EXIT_END[1], FLOW_EXIT_END[2]):
                await handle_flow_reversal_exits(artifact.positions, oms, kis_api=api)
                flow_exit_done_today = True
                logger.info("Flow reversal exits executed for today")

        # IEPE Phase
        # Tier C: blocked unless allow_tier_c_reduced switch is on (0.25x sizing via regime.py)
        tier_c_blocked = artifact.regime_tier == "C" and not nulrimok_switches.allow_tier_c_reduced
        if (time(IEPE_START[0], IEPE_START[1]) <= now.time() <= time(IEPE_END[0], IEPE_END[1])
                and artifact and not tier_c_blocked):

            current_boundary = now.replace(minute=(now.minute // 30) * 30, second=0, microsecond=0)

            if last_30m_boundary is None or current_boundary > last_30m_boundary:
                last_30m_boundary = current_boundary
                last_reconcile_cycle += 1
                acct = await oms.get_account_state()
                equity = acct.equity or 100_000_000

                # Periodic reconciliation with OMS
                if last_reconcile_cycle >= reconcile_interval:
                    await _reconcile_positions(oms, position_states)
                    last_reconcile_cycle = 0

                for ticker in artifact.active_set:
                    ticker_artifact = artifact.get_ticker(ticker)
                    if not ticker_artifact or not ticker_artifact.tradable:
                        continue

                    bar = await fetch_30m_bar(api, ticker, rate_budget)
                    if not bar:
                        continue

                    close = bar['close']
                    volume = bar['volume']
                    sma5 = sma_trackers.get(ticker, RollingSMA(5)).update(close)

                    # Maintain rolling 30m volume history
                    vol_histories[ticker].append(volume)
                    vol_avg = (sum(vol_histories[ticker]) / len(vol_histories[ticker])
                               if vol_histories[ticker] else volume)

                    avwap = ticker_artifact.avwap_ref
                    if avwap > 0:
                        near_band_recently[ticker] = abs(close - avwap) / avwap < NEAR_BAND_PCT

                    entry_state = entry_states.setdefault(ticker, TickerEntryState(ticker=ticker))

                    # Handle PENDING_FILL: check if fill is confirmed
                    if entry_state.state == EntryState.PENDING_FILL:
                        try:
                            alloc_qty = await oms.get_allocation(ticker, STRATEGY_ID)
                            if alloc_qty > 0:
                                # Fill confirmed, get position for cost basis
                                oms_pos = await oms.get_position(ticker)
                                alloc = oms_pos.allocations.get(STRATEGY_ID) if oms_pos else None
                                cost_basis = alloc.cost_basis if alloc else close
                                # Create PositionState from OMS data
                                atr30m = ticker_artifact.atr30m_est or 0.0
                                if atr30m > 0:
                                    stop = max(avwap - 1.2 * atr30m, ticker_artifact.band_lower * 0.993)
                                else:
                                    stop = ticker_artifact.band_lower * 0.993
                                pos = PositionState(ticker, now, cost_basis, alloc_qty, stop)
                                pos.atr30m = atr30m
                                position_states[ticker] = pos
                                entry_state.state = EntryState.TRIGGERED
                                # Update sector exposure
                                if sector_exposure:
                                    sector_exposure.on_fill(ticker, alloc_qty, cost_basis)
                                logger.info(f"{ticker}: Fill confirmed, qty={alloc_qty}")
                            else:
                                entry_state.pending_fill_cycles += 1
                                if entry_state.pending_fill_cycles >= PENDING_FILL_MAX_CYCLES:
                                    logger.info(f"{ticker}: Fill timeout, resetting entry state")
                                    entry_state.reset()
                        except Exception:
                            entry_state.pending_fill_cycles += 1
                        continue

                    if ticker in position_states:
                        pos = position_states[ticker]
                        exit_id = await manage_nulrimok_position(
                            pos, bar, avwap,
                            vol_avg, now.time() >= time(15, 5), oms)
                        # Remove position after full exit (not partial)
                        if exit_id and pos.remaining_qty <= 0:
                            if sector_exposure:
                                sector_exposure.on_close(ticker, pos.qty, pos.entry_price)
                            del position_states[ticker]
                    else:
                        # Aggregate daily risk budget check
                        risk_mult = artifact.risk_mult if hasattr(artifact, 'risk_mult') else 1.0
                        budget = DAILY_RISK_BUDGET_PCT * risk_mult
                        if compute_total_open_risk(position_states, equity) >= budget:
                            continue

                        # Sector cap check before entry
                        # Estimate entry size for sector check (using close price as proxy)
                        est_qty = int(equity * 0.02 / close) if close > 0 else 0  # ~2% position
                        if sector_exposure and not sector_exposure.can_enter(ticker, est_qty, close, equity):
                            continue

                        # Entry submission: will transition to PENDING_FILL, position created on fill confirm
                        await process_entry(
                            entry_state, ticker_artifact, bar, sma5 or close,
                            vol_avg, now, equity, oms)

                daily_ranks = {t: artifact.get_ticker(t).daily_rank
                               for t in artifact.active_set if artifact.get_ticker(t)}
                old_active = set(artifact.active_set)
                artifact.active_set, artifact.overflow, last_rotation = rotate_active_set(
                    list(artifact.active_set), list(artifact.overflow), entry_states,
                    near_band_recently, daily_ranks, now, last_rotation)

                # Sync WS subscriptions after rotation
                if ws_subs:
                    new_active = set(artifact.active_set)
                    if new_active != old_active:
                        for t in old_active - new_active:
                            await ws_subs.drop_tick(t)
                        for t in new_active - old_active:
                            await ws_subs.ensure_tick(t)
                        active_set_ref.clear()
                        active_set_ref.update(new_active)
                        prev_active_set = new_active.copy()

        await asyncio.sleep(5)

    await oms.close()


def main():
    asyncio.run(run_nulrimok())


if __name__ == "__main__":
    main()
