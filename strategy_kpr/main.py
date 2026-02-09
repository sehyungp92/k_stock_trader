"""KPR Strategy Main."""

import asyncio
import time as time_module
from datetime import datetime, time
from typing import Dict, Set
from loguru import logger

import os
from kis_core import (
    KoreaInvestEnv, KoreaInvestAPI, VWAPLedger, RateBudget, KISWebSocketClient,
    SectorExposure, SectorExposureConfig,
    filter_universe, build_kis_config_from_env,
)
from oms_client import OMSClient, Intent, IntentType, Urgency, TimeHorizon

from .config.switches import kpr_switches
from .config.constants import (
    STRATEGY_ID, COLD_POLL_SEC, VWAP_DEPTH_MIN, VWAP_DEPTH_MAX,
    WARM_POLL_DEFAULT, WARM_POLL_MICRO, FLOW_STALE_DEFAULT, FLOW_STALE_MICRO,
    MICRO_WINDOWS, ORDER_TIMEOUT_SEC, DRIFT_CHECK_INTERVAL, MAX_SECTOR_POSITIONS,
)
from .core.state import SymbolState, FSMState, Tier
from .core.fsm import alpha_step
from .core.exits import check_exits
from .core.drift import DriftMonitor
from .signals.investor import InvestorFlowProvider
from .signals.program import ProgramProvider
from .signals.micro import MicroPressureProvider
from .universe.tier_manager import UniverseManager, FeatureSet
from .adapters.ws_handler import (
    KPRSubscriptionManager, KPRTickState,
    make_kpr_tick_handler, sync_hot_subscriptions,
)


def load_config() -> dict:
    import os
    import yaml
    config_path = os.getenv("KPR_CONFIG", "config/settings.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError(f"Config file {config_path} is empty or invalid")
    if not cfg.get("universe"):
        raise ValueError(f"Config file {config_path} missing required 'universe' list")
    if not cfg.get("sector_map"):
        logger.warning(f"Config {config_path}: no 'sector_map' — sector tracking disabled")
    return cfg


def get_kst_now() -> datetime:
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo
    return datetime.now(tz=ZoneInfo("Asia/Seoul"))


def market_open() -> bool:
    now = get_kst_now()
    if now.weekday() >= 5:
        return False
    t = now.time()
    return time(9, 0) <= t <= time(15, 30)


def in_micro_window(now: datetime) -> bool:
    """Check if current time falls within a KRX micro window."""
    t = now.time()
    for (sh, sm), (eh, em) in MICRO_WINDOWS:
        if time(sh, sm) <= t <= time(eh, em):
            return True
    return False


async def run_kpr():
    logger.info("Starting KPR v4.3")
    cfg = load_config()

    # Load switches from YAML if configured (not default — only when SWITCHES_CONFIG is set)
    switches_path = os.getenv("SWITCHES_CONFIG")
    if switches_path:
        kpr_switches.update_from_yaml(switches_path)
    kpr_switches.log_active_config()

    env = KoreaInvestEnv(build_kis_config_from_env())
    api = KoreaInvestAPI(env)

    # Connect to OMS service
    oms = OMSClient(os.environ.get("OMS_URL", "http://localhost:8000"), strategy_id=STRATEGY_ID)
    await oms.wait_ready()

    # Rate budget for REST calls (market data only - order flow goes via OMS)
    rate_budget = RateBudget()
    investor_provider = InvestorFlowProvider(api, rate_budget)
    program_provider = ProgramProvider(api, rate_budget)
    micro_provider = MicroPressureProvider()
    universe_mgr = UniverseManager()

    await program_provider.probe()

    universe = cfg.get("universe", [])
    universe, rejected = filter_universe(api, universe)
    for r in rejected:
        logger.warning(f"Universe filter: {r['ticker']} rejected ({r['reason']})")
    states: Dict[str, SymbolState] = {t: SymbolState(code=t) for t in universe}
    vwap_ledgers: Dict[str, VWAPLedger] = {t: VWAPLedger() for t in universe}
    features: Dict[str, FeatureSet] = {t: FeatureSet() for t in universe}
    positions: Set[str] = set()

    # Advisory sector cap tracking (count-based for KPR).
    # The OMS RiskGateway is the authoritative source for sector caps.
    # This local tracker is a fast pre-filter to avoid unnecessary OMS round-trips.
    sector_map = cfg.get("sector_map", {})
    sector_config = SectorExposureConfig(
        mode="count",
        max_positions_per_sector=MAX_SECTOR_POSITIONS,
        unknown_sector_policy="allow",
    )
    sector_exposure = SectorExposure(sector_map, sector_config)
    for ticker, s in states.items():
        s.sector = sector_map.get(ticker, "")

    # Drift detection
    drift_monitor = DriftMonitor()
    last_drift_check = 0.0
    working_orders: Set[str] = set()  # Track orders awaiting fill

    # --- WebSocket for HOT tier ---
    ws_client = KISWebSocketClient(api)
    ws_subs: KPRSubscriptionManager | None = None
    tick_states: Dict[str, KPRTickState] = {}
    prev_hot_set: Set[str] = set()
    hot_set_ref: Set[str] = set()  # Reference updated when HOT tier changes
    ws_url = env.ws_url

    def _on_hot_tick(ticker: str, msg) -> None:
        """Feed HOT tier ticks to MicroPressureProvider."""
        micro_provider.on_tick(ticker, msg.price, msg.volume)

    if ws_url:
        if await ws_client.connect(ws_url):
            ws_client.on_tick(make_kpr_tick_handler(
                vwap_ledgers, tick_states, hot_set_ref,
                on_hot_tick=_on_hot_tick,
            ))
            asyncio.create_task(ws_client.run())
            ws_subs = KPRSubscriptionManager(ws_client)
            logger.info("KPR WebSocket connected for HOT tier")
        else:
            logger.warning("KPR WebSocket connect failed; HOT tier uses REST polling")

    # Track last processed bar timestamp per symbol (fixes VWAP double-counting)
    last_bar_ts: Dict[str, object] = {}
    # Track last poll time per symbol
    last_poll: Dict[str, datetime] = {}
    # Track session open price per symbol (for drop_from_open feature)
    day_open: Dict[str, float] = {}

    while not market_open():
        await asyncio.sleep(60)

    last_heartbeat_ts = 0.0
    heartbeat_interval = 30.0  # seconds

    while market_open():
        now = get_kst_now()
        now_ts = time_module.time()
        acct = await oms.get_account_state()
        equity = acct.equity or 100_000_000
        is_micro = in_micro_window(now)

        # --- Drift detection and reconciliation ---
        if now_ts - last_drift_check > DRIFT_CHECK_INTERVAL:
            last_drift_check = now_ts
            try:
                all_positions = await oms.get_all_positions()
                broker_positions = {
                    sym: pos.get_allocation(STRATEGY_ID)
                    for sym, pos in all_positions.items()
                    if pos.get_allocation(STRATEGY_ID) > 0
                }

                # Build local view
                local_positions = {
                    s.code: s.qty for s in states.values()
                    if s.fsm == FSMState.IN_POSITION and s.qty > 0
                }
                local_orders = {s.entry_order_id for s in states.values() if s.entry_order_id}

                # Check for drift
                events = drift_monitor.compute_drift(
                    local_positions, broker_positions, local_orders, set()
                )
                if drift_monitor.handle_drift(events):
                    # Reconcile: overwrite local with broker truth
                    for sym, qty in broker_positions.items():
                        s = states.get(sym)
                        if s:
                            s.qty = qty
                            if qty > 0 and s.fsm != FSMState.IN_POSITION:
                                s.fsm = FSMState.IN_POSITION
                                positions.add(sym)
                    # Reverse: zero out local positions missing from broker
                    for sym in list(local_positions.keys()):
                        if sym not in broker_positions:
                            s = states.get(sym)
                            if s and s.fsm == FSMState.IN_POSITION:
                                s.qty = 0
                                s.fsm = FSMState.DONE
                                positions.discard(sym)
                                sector_exposure.on_close(sym, local_positions[sym], 0)
                                logger.info(f"{sym}: Removed local phantom position")
                    drift_monitor.clear_after_reconcile()
            except Exception as e:
                logger.warning(f"Drift check failed: {e}")
                drift_monitor.block_on_oms_unavailable()

        # --- Order timeout detection ---
        for s in states.values():
            if s.entry_order_id and s.order_submit_ts > 0:
                order_age = now_ts - s.order_submit_ts
                if order_age > ORDER_TIMEOUT_SEC:
                    logger.info(f"{s.code}: Order timeout after {order_age:.0f}s")
                    try:
                        await oms.submit_intent(Intent(
                            intent_type=IntentType.CANCEL_ORDERS,
                            strategy_id=STRATEGY_ID,
                            symbol=s.code,
                            desired_qty=0,
                            urgency=Urgency.HIGH,
                            time_horizon=TimeHorizon.INTRADAY,
                        ))
                    except Exception as e:
                        logger.debug(f"{s.code}: Cancel failed: {e}")
                    s.entry_order_id = None
                    s.order_submit_ts = 0.0
                    working_orders.discard(s.code)

        # Periodic heartbeat
        if now_ts - last_heartbeat_ts > heartbeat_interval:
            await oms.report_heartbeat(
                mode="RUNNING",
                symbols_hot=len(universe_mgr.hot) if hasattr(universe_mgr, 'hot') else 0,
                symbols_warm=len(universe_mgr.warm) if hasattr(universe_mgr, 'warm') else 0,
                symbols_cold=len(universe) - len(getattr(universe_mgr, 'hot', set())) - len(getattr(universe_mgr, 'warm', set())),
                positions_count=len(positions),
                version="4.3.1",
            )
            last_heartbeat_ts = now_ts

        universe_mgr.rebalance(universe, states, features, positions)

        # Sync WS subscriptions with HOT tier changes
        if ws_subs:
            new_hot = universe_mgr.hot
            if new_hot != prev_hot_set:
                await sync_hot_subscriptions(ws_subs, prev_hot_set, new_hot)
                # Update hot_set_ref in-place for tick handler
                hot_set_ref.clear()
                hot_set_ref.update(new_hot)
                prev_hot_set = new_hot.copy()

        for ticker in universe:
            s = states[ticker]
            if s.fsm == FSMState.DONE:
                continue

            tier = universe_mgr.get_tier(ticker)

            # Targeted refresh for relevant symbols (reduces rate limit usage)
            if tier == Tier.HOT or s.fsm in (FSMState.SETUP_DETECTED, FSMState.ACCEPTING, FSMState.IN_POSITION):
                investor_provider.dispatch_refresh(ticker)

            # Micro-window aware polling intervals
            if tier == Tier.HOT:
                poll_interval = 15
            elif tier == Tier.WARM:
                poll_interval = WARM_POLL_MICRO if is_micro else WARM_POLL_DEFAULT
            else:
                poll_interval = COLD_POLL_SEC

            lp = last_poll.get(ticker)
            if lp and (now - lp).total_seconds() < poll_interval:
                continue
            last_poll[ticker] = now

            try:
                if not rate_budget.try_consume("CHART"):
                    continue  # Skip this tick, retry next loop
                bars = api.get_minute_bars(ticker, minutes=1)
                if bars is None or bars.empty:
                    continue
                bar = bars.iloc[-1].to_dict()

                # --- Deduplicate: only process new bars ---
                bar_ts = bar.get('timestamp') or bar.get('stck_bsop_date', '')
                if bar_ts == last_bar_ts.get(ticker):
                    continue
                last_bar_ts[ticker] = bar_ts

                # --- VWAP update (once per new bar, skip for HOT with tick updates) ---
                # HOT symbols get cumulative VWAP from ticks; bar update would double-count
                ts = tick_states.get(ticker)
                if not (ticker in hot_set_ref and ts and ts.last_tick_ts):
                    vwap_ledgers[ticker].update_from_bar(bar)
                vwap = vwap_ledgers[ticker].vwap

                close = float(bar.get('close', 0))

                # --- Feature update: in_vwap_band = 2-5% below VWAP ---
                if vwap > 0:
                    depth = (vwap - close) / vwap
                    features[ticker].in_vwap_band = VWAP_DEPTH_MIN <= depth <= VWAP_DEPTH_MAX
                else:
                    features[ticker].in_vwap_band = False

                # --- Tiering features (drop_from_open using session open, dist_to_vwap_band) ---
                if ticker not in day_open:
                    # First bar: capture session open from bar's open
                    session_open = float(bar.get('open', close))
                    if session_open > 0:
                        day_open[ticker] = session_open
                if ticker in day_open and day_open[ticker] > 0:
                    features[ticker].drop_from_open = (close - day_open[ticker]) / day_open[ticker]
                if vwap > 0:
                    depth = (vwap - close) / vwap
                    if depth < VWAP_DEPTH_MIN:
                        features[ticker].dist_to_vwap_band = VWAP_DEPTH_MIN - depth
                    elif depth > VWAP_DEPTH_MAX:
                        features[ticker].dist_to_vwap_band = depth - VWAP_DEPTH_MAX
                    else:
                        features[ticker].dist_to_vwap_band = 0.0

                # --- MicroPressure from bar ---
                micro_sig = micro_provider.update(ticker, bar)

                # --- Investor & Program signals (dynamic staleness for investor) ---
                flow_stale_threshold = FLOW_STALE_MICRO if is_micro else FLOW_STALE_DEFAULT
                investor_sig = await investor_provider.fetch(ticker, max_age=flow_stale_threshold)
                program_sig = await program_provider.fetch(ticker)

                # --- Exit checks (before entry FSM) ---
                if s.fsm == FSMState.IN_POSITION:
                    should_exit, reason, exit_qty = check_exits(
                        s, close, now, investor_sig, micro_sig,
                    )
                    if should_exit and exit_qty > 0:
                        logger.info(f"{ticker}: Exit {reason}, qty={exit_qty}")
                        from oms_client import RiskPayload
                        result = await oms.submit_intent(Intent(
                            intent_type=IntentType.EXIT,
                            strategy_id=STRATEGY_ID,
                            symbol=ticker,
                            desired_qty=exit_qty,
                            urgency=Urgency.HIGH,
                            time_horizon=TimeHorizon.INTRADAY,
                            risk_payload=RiskPayload(rationale_code=reason),
                        ))
                        # EXECUTED means order submitted, NOT filled.
                        # Transition to PENDING_EXIT; confirm via OMS allocation on next cycle.
                        if result.status.name in ("EXECUTED", "APPROVED"):
                            if reason == "partial_target":
                                s.partial_filled = True
                                s.trail_stop = max(s.trail_stop, s.entry_px)
                            s.fsm = FSMState.PENDING_EXIT
                            logger.info(f"{ticker}: Exit submitted, PENDING_EXIT")
                    continue

                # --- PENDING_EXIT: confirm exit fill from OMS allocation ---
                if s.fsm == FSMState.PENDING_EXIT:
                    try:
                        alloc_qty = await oms.get_allocation(ticker, STRATEGY_ID)
                        if alloc_qty <= 0:
                            # Fully exited
                            s.remaining_qty = 0
                            s.fsm = FSMState.DONE
                            positions.discard(ticker)
                            sector_exposure.on_close(ticker, s.qty, close)
                            logger.info(f"{ticker}: Exit fill confirmed, DONE")
                        elif alloc_qty < s.remaining_qty:
                            # Partial fill — update remaining and stay PENDING or go back
                            s.remaining_qty = alloc_qty
                            if s.partial_filled:
                                # Was a partial target exit, go back to IN_POSITION for more
                                s.fsm = FSMState.IN_POSITION
                                logger.info(f"{ticker}: Partial exit filled, remaining={alloc_qty}")
                        # else: no change yet, stay PENDING_EXIT
                    except Exception as e:
                        logger.warning(f"{ticker}: PENDING_EXIT check failed: {e}")
                    continue

                # --- Entry FSM ---
                investor_age = investor_provider.age_sec(ticker, now_ts)
                regime_ok = not acct.halt_new_entries
                intent_id = await alpha_step(
                    s, bar, vwap, now, investor_sig, micro_sig, program_sig,
                    program_provider.available or False,
                    regime_ok,
                    tier == Tier.HOT,
                    investor_provider.is_stale(ticker, flow_stale_threshold),
                    now.time() > time(14, 0),
                    close * 0.02,  # ATR proxy (until 1m bars provide real ATR)
                    equity, oms,
                    drift_monitor=drift_monitor,
                    sector_exposure=sector_exposure,
                    investor_age=investor_age,
                )

                if intent_id:
                    working_orders.add(ticker)

                if s.fsm == FSMState.IN_POSITION:
                    positions.add(ticker)
                else:
                    positions.discard(ticker)

            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}", exc_info=True)

        await asyncio.sleep(1)

    await oms.close()


def main():
    asyncio.run(run_kpr())


if __name__ == "__main__":
    main()
