"""
KMP Strategy Main Orchestration.
"""

from __future__ import annotations
import asyncio
from collections import deque
from datetime import datetime
from typing import Dict, Tuple
from loguru import logger

import os
from kis_core import (
    KoreaInvestEnv, KoreaInvestAPI, RateBudget, KISWebSocketClient, TickMessage, AskBidMessage,
    SectorExposure, SectorExposureConfig,
    filter_universe, build_kis_config_from_env,
)
from oms_client import OMSClient, Intent, IntentType, Urgency, TimeHorizon, RiskPayload, IntentStatus

from .config.constants import STRATEGY_ID, FLATTEN_TIME, RVOL_MIN
from .core.gates import is_past_entry_cutoff
from .config.universe_meta import load_universe_meta
from .core.state import SymbolState, State
from .core.scanner import scan_at_0915, apply_trend_anchor
from .core.fsm import alpha_step
from .core.exits import check_exit_conditions
from .core.reconcile import reconcile_exposure
from .adapters.program_regime import MarketProgramRegime, program_poll_task
from .adapters.ws_manager import SubscriptionManager, refresh_focus_list, release_non_position_slots
from .adapters.tick_dispatch import on_tick, on_ask_bid
from .premarket import compute_baselines


def load_config() -> dict:
    """Load configuration from environment/file."""
    import os
    import yaml

    config_path = os.getenv("KMP_CONFIG", "config/settings.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_kst_now() -> datetime:
    """Get current time in KST."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo

    return datetime.now(tz=ZoneInfo("Asia/Seoul"))


def market_open() -> bool:
    """Check if market is open."""
    now = get_kst_now()
    if now.weekday() >= 5:
        return False
    h, m = now.hour, now.minute
    if h < 9 or (h == 15 and m > 30) or h > 15:
        return False
    return True


# ---------------------------------------------------------------------------
# Breadth / regime gate
# ---------------------------------------------------------------------------

LEADER_RVOL_MIN = 1.5  # Spec §5 uses 1.5 for breadth (not RVOL_MIN=2.0)
RISK_OFF_DRAWDOWN = -0.01  # -1.0% KOSPI intraday drawdown triggers risk_off
CHOP_RANGE_THRESHOLD = 0.004  # 0.4% KOSPI intraday range = narrow/choppy
CHOP_BREADTH_LOOKBACK = 3  # Last N breadth readings must all be weak


def compute_regime_ok(
    states: Dict[str, SymbolState],
    candidates: list[str],
    last_prices: Dict[str, float],
    chop_detector: "ChopDetector | None" = None,
) -> Tuple[bool, int]:
    """
    Leader-breadth gate.

    Counts candidates that are surging, have RVol, and price >= VWAP.
    Spec §5: regime_ok = not chop AND leader_breadth_ok.

    Returns:
        (regime_ok, breadth_count) tuple.
    """
    breadth = 0
    for t in candidates:
        s = states.get(t)
        if s is None or s.fsm == State.DONE:
            continue
        price = last_prices.get(t, 0.0)
        if s.surge >= 3.0 and s.rvol_1m >= LEADER_RVOL_MIN and s.vwap > 0 and price >= s.vwap:
            breadth += 1

    breadth_ok = breadth >= 8

    # Feed breadth into chop detector and check chop
    is_chop = False
    if chop_detector is not None:
        chop_detector.add_breadth(breadth)
        is_chop = chop_detector.is_chop()

    return (breadth_ok and not is_chop), breadth


def compute_risk_off(kospi_price: float, kospi_prev_close: float) -> bool:
    """
    Check if KOSPI intraday drawdown exceeds risk_off threshold.

    risk_off blocks new entries AND forces exit of existing positions.
    """
    if kospi_prev_close <= 0 or kospi_price <= 0:
        return False
    drawdown = (kospi_price - kospi_prev_close) / kospi_prev_close
    return drawdown <= RISK_OFF_DRAWDOWN


class ChopDetector:
    """
    Detect range-bound / choppy market conditions.

    Chop = narrow KOSPI intraday range (<0.4%) AND weak leadership breadth
    (last 3 breadth readings all < 8).
    """

    def __init__(self):
        self._kospi_high: float = 0.0
        self._kospi_low: float = float('inf')
        self._kospi_open: float = 0.0
        self._breadth_history: deque = deque(maxlen=CHOP_BREADTH_LOOKBACK)

    def update_kospi(self, price: float) -> None:
        """Update KOSPI high/low tracking."""
        if self._kospi_open <= 0:
            self._kospi_open = price
        self._kospi_high = max(self._kospi_high, price)
        self._kospi_low = min(self._kospi_low, price)

    def add_breadth(self, breadth: int) -> None:
        """Record a breadth reading."""
        self._breadth_history.append(breadth)

    def is_chop(self) -> bool:
        """Check if current market is choppy."""
        if self._kospi_open <= 0 or len(self._breadth_history) < CHOP_BREADTH_LOOKBACK:
            return False

        # Narrow KOSPI range
        intraday_range = (self._kospi_high - self._kospi_low) / self._kospi_open
        narrow = intraday_range < CHOP_RANGE_THRESHOLD

        # Weak breadth: all recent readings < 8
        weak_breadth = all(b < 8 for b in self._breadth_history)

        return narrow and weak_breadth


# ---------------------------------------------------------------------------
# OMS position sync (ARMED → IN_POSITION, position closed → DONE)
# ---------------------------------------------------------------------------

async def _sync_positions(
    oms: OMSClient,
    states: Dict[str, SymbolState],
    candidates: list[str],
    last_prices: Dict[str, float],
    exposure: SectorExposure,
) -> None:
    """Sync FSM state from OMS positions each loop."""
    all_positions = await oms.get_all_positions()

    # Guard: if OMS returned empty but we have active positions, skip sync
    if not all_positions:
        has_active = any(
            states.get(t) and states[t].fsm in (State.ARMED, State.WAIT_ACCEPTANCE, State.IN_POSITION)
            for t in candidates
        )
        if has_active:
            logger.warning("OMS returned empty while positions active — skipping sync")
            return

    for ticker in candidates:
        s = states.get(ticker)
        if s is None:
            continue

        pos = all_positions.get(ticker)
        alloc_qty = pos.get_allocation(STRATEGY_ID) if pos else 0

        if alloc_qty > 0:
            if s.fsm in (State.ARMED, State.WAIT_ACCEPTANCE):
                import time as _time
                alloc = pos.allocations.get(STRATEGY_ID)
                s.entry_px = getattr(alloc, 'cost_basis', 0.0) or s.entry_px
                s.qty = alloc_qty
                s.max_fav = max(s.max_fav, last_prices.get(ticker, s.entry_px))
                s.trail_px = max(s.trail_px, s.structure_stop)
                s.entry_ts = _time.time()
                s.fsm = State.IN_POSITION
                # Update exposure: move from working → open
                exposure.on_fill(ticker, alloc_qty, s.entry_px)
                logger.info(f"{ticker}: Fill detected, IN_POSITION @ {s.entry_px:.0f} qty={s.qty}")
        elif alloc_qty == 0:
            if s.fsm == State.IN_POSITION:
                # Position closed externally (e.g. by OMS risk)
                exposure.on_close(ticker, s.qty, s.entry_px)
                s.fsm = State.DONE
                logger.info(f"{ticker}: Position closed externally, DONE")
            elif s.fsm == State.PENDING_EXIT:
                # Exit fill confirmed — position fully closed
                exposure.on_close(ticker, s.qty, s.entry_px)
                s.fsm = State.DONE
                logger.info(f"{ticker}: Exit fill confirmed, DONE")


# ---------------------------------------------------------------------------
# WS tick/ask-bid handlers (using shared KISWebSocketClient)
# ---------------------------------------------------------------------------

def make_tick_handler(
    states: Dict[str, SymbolState],
    last_prices: Dict[str, float],
):
    """Create a tick message handler for KISWebSocketClient."""

    def handler(msg: TickMessage) -> None:
        s = states.get(msg.ticker)
        if s is None:
            return
        on_tick(
            s, msg.price, msg.volume, msg.cum_vol, msg.cum_val,
            msg.vi_ref, msg.timestamp, s.or_locked,
        )
        last_prices[msg.ticker] = msg.price

    return handler


def make_askbid_handler(states: Dict[str, SymbolState]):
    """Create a bid/ask message handler for KISWebSocketClient."""

    def handler(msg: AskBidMessage) -> None:
        s = states.get(msg.ticker)
        if s is None:
            return
        on_ask_bid(s, msg.bid, msg.ask)

    return handler


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_kmp():
    """Main KMP strategy entry point."""
    logger.info("Starting KMP v2.3.4")

    cfg = load_config()

    # Initialize KIS API
    env = KoreaInvestEnv(build_kis_config_from_env())
    api = KoreaInvestAPI(env)

    # Rate budget for REST calls
    rate_budget = RateBudget()

    # Connect to OMS service
    oms = OMSClient(os.environ.get("OMS_URL", "http://localhost:8000"), strategy_id=STRATEGY_ID)
    await oms.wait_ready()

    # Initialize components
    program_regime = MarketProgramRegime()
    ws_client = KISWebSocketClient(api)

    # Load universe
    universe = cfg.get("universe", [])
    universe, rejected = filter_universe(api, universe)
    for r in rejected:
        logger.warning(f"Universe filter: {r['ticker']} rejected ({r['reason']})")
    states: Dict[str, SymbolState] = {t: SymbolState(code=t) for t in universe}

    # Load universe metadata (sector mapping)
    meta = load_universe_meta(cfg)
    sym_to_sector = {}
    for ticker, s in states.items():
        if ticker in meta:
            s.sector = meta[ticker].sector
            sym_to_sector[ticker] = meta[ticker].sector

    # Advisory sector exposure tracking (count-based for KMP).
    # The OMS RiskGateway is the authoritative source for sector caps.
    # This local tracker is a fast pre-filter to avoid unnecessary OMS round-trips.
    max_per_sector = cfg.get("max_per_sector", 1)
    sector_config = SectorExposureConfig(
        mode="count",
        max_positions_per_sector=max_per_sector,
        unknown_sector_policy="allow",
    )
    exposure = SectorExposure(sym_to_sector, sector_config)
    last_reconcile_ts = 0.0
    reconcile_interval = 2.0  # seconds

    # Track last known prices for focus-list proximity
    last_prices: Dict[str, float] = {}

    # Pre-market: fetch daily data for trend anchor
    logger.info("Fetching daily data for trend anchor...")
    daily_data = {}
    for ticker in universe:
        if not rate_budget.try_consume("CHART"):
            await asyncio.sleep(0.5)
        bars = api.get_daily_bars(ticker, days=80)
        if bars is not None and not bars.empty:
            daily_data[ticker] = bars.to_dict('records')

    apply_trend_anchor(states, daily_data)
    logger.info(f"Trend anchor applied. {sum(1 for s in states.values() if s.trend_ok)} tickers OK")

    # Fetch KOSPI prior close for risk_off detection
    kospi_prev_close = 0.0
    try:
        kospi_daily = api.get_index_daily("KOSPI", days=1)
        if kospi_daily and len(kospi_daily) > 0:
            kospi_prev_close = float(kospi_daily[-1].get('close', 0))
            logger.info(f"KOSPI prior close: {kospi_prev_close:.2f}")
    except Exception as e:
        logger.warning(f"Failed to fetch KOSPI prior close: {e}")
    chop_detector = ChopDetector()

    # Compute baselines from daily data (reuses daily_data fetched for trend anchor)
    baseline_15m, baseline_1m_vol = compute_baselines(daily_data)
    for t, s in states.items():
        s.avg_1m_vol = baseline_1m_vol.get(t, 0.0)

    # Start background tasks
    asyncio.create_task(program_poll_task(api, program_regime))

    # --- Connect WebSocket ---
    subs = None
    ws_url = env.ws_url
    if ws_url:
        if await ws_client.connect(ws_url):
            ws_client.on_tick(make_tick_handler(states, last_prices))
            ws_client.on_askbid(make_askbid_handler(states))
            asyncio.create_task(ws_client.run())
            subs = SubscriptionManager(ws_client)
            logger.info("WebSocket connected")
        else:
            logger.warning("WebSocket connect failed; falling back to REST polling")

    # Wait for market open
    while not market_open():
        logger.info("Waiting for market open...")
        await asyncio.sleep(60)

    # Wait for 09:15 scan
    while True:
        now = get_kst_now()
        if now.hour == 9 and now.minute >= 15:
            break
        await asyncio.sleep(1)

    # 09:15 scan
    logger.info("Running 09:15 scan...")
    candidates = await scan_at_0915(api, universe, baseline_15m, states, rate_budget=rate_budget)
    logger.info(f"Scan complete. {len(candidates)} candidates")

    # Subscribe to candidates (if WS available)
    if subs:
        for ticker in candidates[:20]:
            await subs.ensure_cnt(ticker)

    # Main loop
    logger.info("Entering main loop...")
    import time as _time
    last_heartbeat_ts = 0.0
    heartbeat_interval = 30.0  # seconds
    ws_slots_released = False  # Track if non-position WS slots released after entry cutoff

    while market_open():
        now = get_kst_now()

        # Flatten time check
        if (now.hour, now.minute) >= FLATTEN_TIME:
            logger.info("Flatten time reached")
            for s in states.values():
                if s.fsm == State.IN_POSITION:
                    result = await oms.submit_intent(Intent(
                        intent_type=IntentType.FLATTEN,
                        strategy_id=STRATEGY_ID,
                        symbol=s.code,
                        urgency=Urgency.HIGH,
                        time_horizon=TimeHorizon.INTRADAY,
                        risk_payload=RiskPayload(rationale_code="flatten_time"),
                    ))
                    if result.status.name in ("EXECUTED", "APPROVED"):
                        # Don't immediately mark DONE — wait for fill confirmation
                        s.fsm = State.PENDING_EXIT
                        logger.info(f"{s.code}: Flatten submitted, PENDING_EXIT")
                    else:
                        logger.warning(f"{s.code}: Flatten {result.status.name} - {result.message}")
            break

        # Refresh focus list (with near-trigger prioritisation)
        if subs:
            # After entry cutoff: release WS slots for non-position symbols (Nulrimok handoff)
            if is_past_entry_cutoff(now) and not ws_slots_released:
                await release_non_position_slots(states, subs)
                ws_slots_released = True
            # Continue maintaining focus for IN_POSITION symbols
            await refresh_focus_list(states, subs, last_prices)

        # Fetch KOSPI real-time price (shared by risk_off and chop detector)
        kospi_price = 0.0
        if rate_budget.try_consume("INDEX"):
            try:
                kospi_price = api.get_index_realtime("KOSPI")
            except Exception:
                pass
        if kospi_price > 0:
            chop_detector.update_kospi(kospi_price)

        # Risk-off: KOSPI intraday drawdown <= -1.0%
        risk_off = compute_risk_off(kospi_price, kospi_prev_close) if kospi_price > 0 else False

        # Breadth / regime gate (includes chop detection)
        regime_ok, breadth = compute_regime_ok(states, candidates, last_prices, chop_detector)
        is_chop = chop_detector.is_chop()

        # Get account state
        acct = await oms.get_account_state()
        equity = acct.equity or 100_000_000

        # Periodic heartbeat
        if _time.time() - last_heartbeat_ts > heartbeat_interval:
            hot_count = sum(1 for t in candidates if states.get(t) and states[t].fsm in (State.WATCH_BREAK, State.ARMED, State.WAIT_ACCEPTANCE))
            positions_count = sum(1 for s in states.values() if s.fsm == State.IN_POSITION)
            await oms.report_heartbeat(
                mode="RUNNING",
                symbols_hot=hot_count,
                symbols_warm=len(candidates) - hot_count,
                positions_count=positions_count,
                version="2.3.4",
            )
            last_heartbeat_ts = _time.time()

        # Periodic full reconciliation from OMS
        if _time.time() - last_reconcile_ts > reconcile_interval:
            await reconcile_exposure(oms, states, exposure, meta)
            last_reconcile_ts = _time.time()

        # Sync fills from OMS: ARMED → IN_POSITION / position closed → DONE
        await _sync_positions(oms, states, candidates, last_prices, exposure)

        # Process each candidate
        for ticker in candidates:
            s = states.get(ticker)
            if not s or s.fsm == State.DONE:
                continue

            # Use WS-sourced price if available, fall back to REST
            price = last_prices.get(ticker, 0.0)
            if price <= 0:
                if rate_budget.try_consume("QUOTE"):
                    price = api.get_last_price(ticker)
                    if price > 0:
                        last_prices[ticker] = price
            if price <= 0:
                continue

            # Derive ATR and 5m value from bar aggregators (fallback to estimate)
            atr_1m = s.atr_1m if s.atr_1m is not None else price * 0.01
            last_5m_value = s.last_5m_value if s.last_5m_value > 0 else s.value15 / 3

            # FSM step for non-position states (blocked by risk_off)
            if s.fsm != State.IN_POSITION and not risk_off:
                await alpha_step(
                    s=s,
                    price=price,
                    now_kst=now,
                    regime_ok=regime_ok,
                    prog_regime=program_regime.regime(),
                    prog_mult=program_regime.multiplier(),
                    equity=equity,
                    atr_1m=atr_1m,
                    last_5m_value=last_5m_value,
                    oms=oms,
                    exposure=exposure,
                    max_per_sector=max_per_sector,
                    regime_breadth_ok=(breadth >= 8),
                    not_chop=(not is_chop),
                )

            # Position management
            if s.fsm == State.IN_POSITION:
                should_exit, reason = check_exit_conditions(
                    s, price, program_regime.regime(), risk_off=risk_off
                )
                if should_exit:
                    logger.info(f"{ticker}: Exit triggered - {reason}")
                    result = await oms.submit_intent(Intent(
                        intent_type=IntentType.EXIT,
                        strategy_id=STRATEGY_ID,
                        symbol=ticker,
                        urgency=Urgency.HIGH,
                        time_horizon=TimeHorizon.INTRADAY,
                        risk_payload=RiskPayload(rationale_code=reason),
                    ))
                    if result.status.name in ("EXECUTED", "APPROVED"):
                        # Don't immediately mark DONE — wait for fill confirmation
                        s.fsm = State.PENDING_EXIT
                        logger.info(f"{ticker}: Exit submitted, PENDING_EXIT")
                    else:
                        logger.warning(f"{ticker}: Exit {result.status.name} - {result.message}")

        await asyncio.sleep(1)

    logger.info("KMP strategy shutdown")
    await oms.close()


def main():
    """Entry point."""
    asyncio.run(run_kmp())


if __name__ == "__main__":
    main()
