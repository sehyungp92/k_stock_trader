"""KPR Strategy Main."""

import asyncio
import time as time_module
from datetime import datetime, time
from typing import Dict, Set
from loguru import logger

import os
from kis_core import (
    KoreaInvestEnv, KoreaInvestAPI, VWAPLedger, KISWebSocketClient,
    SectorExposure, SectorExposureConfig,
    filter_universe, build_kis_config_from_env,
    create_strategy_client,
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
from instrumentation.facade import InstrumentationKit
from instrumentation.src.drawdown import compute_drawdown_context
from instrumentation.src.mfe_mae import build_mfe_mae_context


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
    logger.add(
        "/app/data/logs/kpr_{time:YYYY-MM-DD}.log",
        rotation="00:00", retention="30 days", compression="gz",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    )
    logger.info("Starting KPR v4.3")
    cfg = load_config()
    experiment_cfg = cfg.get("experiment", {})

    # Load conservative switches if CONSERVATIVE_MODE=true
    if os.getenv("CONSERVATIVE_MODE", "false").lower() == "true":
        kpr_switches.update_from_yaml("/app/config/conservative.yaml")
    kpr_switches.log_active_config()

    env = KoreaInvestEnv(build_kis_config_from_env())
    api = KoreaInvestAPI(env)

    # Connect to OMS service
    oms = OMSClient(os.environ.get("OMS_URL", "http://localhost:8000"), strategy_id=STRATEGY_ID)
    await oms.wait_ready()

    # Instrumentation
    instr = InstrumentationKit.create(api, strategy_type="kpr")

    # Rate budget for REST calls (shared across containers via file-based coordination)
    rate_budget = create_strategy_client(
        STRATEGY_ID,
        state_file=os.environ.get("RATE_BUDGET_STATE_FILE"),
    )
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

    # MFE/MAE tracking dicts
    _mfe_prices: Dict[str, float] = {}
    _mae_prices: Dict[str, float] = {}

    # Last known prices for heartbeat enrichment
    _last_prices: Dict[str, float] = {}

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
    equity_warned = False

    while market_open():
        now = get_kst_now()
        now_ts = time_module.time()
        acct = await oms.get_account_state()
        equity = acct.equity or 100_000_000
        if acct.equity is not None and acct.equity <= 0 and not equity_warned:
            logger.warning(f"KPR: OMS equity=0 — entries will be DEFERRED until reconciliation")
            equity_warned = True
        elif acct.equity and acct.equity > 0 and equity_warned:
            logger.info(f"KPR: OMS equity loaded: {acct.equity:,.0f}")
            equity_warned = False
        is_micro = in_micro_window(now)

        # --- Drift detection and reconciliation ---
        if now_ts - last_drift_check > DRIFT_CHECK_INTERVAL:
            last_drift_check = now_ts
            try:
                all_positions = await oms.get_all_positions()

                # Exclude symbols with pending orders from drift detection —
                # their state is transitional and handled by fill confirmation loops
                pending_symbols = {
                    s.code for s in states.values()
                    if s.fsm in (FSMState.PENDING_ENTRY, FSMState.PENDING_EXIT)
                }

                broker_positions = {
                    sym: pos.get_allocation(STRATEGY_ID)
                    for sym, pos in all_positions.items()
                    if pos.get_allocation(STRATEGY_ID) > 0 and sym not in pending_symbols
                }

                # Build local view
                local_positions = {
                    s.code: s.qty for s in states.values()
                    if s.fsm == FSMState.IN_POSITION and s.qty > 0
                }

                # Check for drift (position-level only; order-level skipped
                # because OMS has no broker open-orders query endpoint)
                events = drift_monitor.compute_drift(
                    local_positions, broker_positions
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
                if instr:
                    instr.emit_error(
                        severity="critical",
                        error_type="drift_check_failed",
                        message=str(e),
                        context={"action": "drift_check"},
                    )
                logger.warning(f"Drift check failed: {e} — blocking new entries")
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
                    # Reset PENDING_ENTRY back to ACCEPTING so it can retry
                    if s.fsm == FSMState.PENDING_ENTRY:
                        s.fsm = FSMState.ACCEPTING
                        s._entry_signal_factors = None
                        s._entry_filter_decisions = None
                        s._pending_qty = 0

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
            hb_positions = []
            for ticker in positions:
                s = states.get(ticker)
                if s and s.fsm == FSMState.IN_POSITION:
                    px = _last_prices.get(ticker, s.entry_px)
                    hb_positions.append({
                        "pair": ticker, "side": "LONG", "qty": s.qty,
                        "entry_price": s.entry_px, "current_price": px,
                        "unrealized_pnl": round((px - s.entry_px) * s.qty),
                        "unrealized_pnl_pct": round((px / s.entry_px - 1) * 100, 2) if s.entry_px else 0,
                        "strategy_type": "kpr",
                    })
            hb_exposure = {}
            if hb_positions:
                hb_exposure = {
                    "total_positions": len(hb_positions),
                    "total_exposure_krw": round(sum(p["current_price"] * p["qty"] for p in hb_positions)),
                    "total_unrealized_pnl": round(sum(p["unrealized_pnl"] for p in hb_positions)),
                }
            instr.emit_heartbeat(
                active_positions=len(positions),
                positions=hb_positions, portfolio_exposure=hb_exposure,
            )
            instr.periodic_tick()
            instr.check_config_changes()
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
            if tier == Tier.HOT or s.fsm in (FSMState.SETUP_DETECTED, FSMState.ACCEPTING, FSMState.PENDING_ENTRY, FSMState.IN_POSITION):
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
                _last_prices[ticker] = close

                # --- MFE/MAE update for in-position symbols ---
                if ticker in _mfe_prices:
                    _mfe_prices[ticker] = max(_mfe_prices[ticker], close)
                    _mae_prices[ticker] = min(_mae_prices[ticker], close)

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
                            if instr:
                                instr.on_order_event(
                                    order_id=getattr(result, 'order_id', '') or '',
                                    pair=ticker, order_type="LIMIT", status="SUBMITTED",
                                    requested_qty=exit_qty, related_trade_id="",
                                )
                            if reason == "partial_target":
                                s.partial_filled = True
                                s.trail_stop = max(s.trail_stop, s.entry_px)
                            s._exit_reason = reason
                            s.fsm = FSMState.PENDING_EXIT
                            logger.info(f"{ticker}: Exit submitted, PENDING_EXIT")
                        else:
                            if instr:
                                instr.on_order_event(
                                    order_id=getattr(result, 'order_id', '') or '',
                                    pair=ticker, order_type="LIMIT", status="REJECTED",
                                    requested_qty=exit_qty, reject_reason=result.message or "",
                                )
                                instr.emit_error(
                                    severity="warning",
                                    error_type="exit_rejected",
                                    message=f"{result.status.name}: {result.message}",
                                    context={"symbol": ticker, "action": "exit", "reason": reason},
                                )
                            logger.warning(f"{ticker}: Exit {result.status.name} - {result.message}")
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
                            if instr:
                                mfe_mae = build_mfe_mae_context(
                                    entry_price=s.entry_px,
                                    stop_price=s.stop_level or s.entry_px * 0.98,
                                    max_fav_price=_mfe_prices.pop(ticker, 0),
                                    min_adverse_price=_mae_prices.pop(ticker, float('inf')),
                                )
                                instr.on_exit_fill(
                                    trade_id=f"KPR:{ticker}:{(s.entry_ts or now).strftime('%Y%m%d')}:{s.setup_type or 'drift'}",
                                    exit_price=close,
                                    exit_reason=getattr(s, '_exit_reason', 'unknown'),
                                    mfe_mae_context=mfe_mae,
                                )
                                if s.bid > 0 or s.ask > 0:
                                    instr.on_orderbook_context(
                                        pair=ticker,
                                        best_bid=s.bid, best_ask=s.ask,
                                        trade_context="exit",
                                        related_trade_id=f"KPR:{ticker}:{(s.entry_ts or now).strftime('%Y%m%d')}:{s.setup_type or 'drift'}",
                                    )
                            else:
                                _mfe_prices.pop(ticker, None)
                                _mae_prices.pop(ticker, None)
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

                # --- PENDING_ENTRY: confirm entry fill from OMS allocation ---
                if s.fsm == FSMState.PENDING_ENTRY:
                    try:
                        alloc_qty = await oms.get_allocation(ticker, STRATEGY_ID)
                        if alloc_qty > 0:
                            # Fill confirmed — get actual entry price from OMS
                            actual_price = close  # fallback
                            try:
                                oms_pos = await oms.get_position(ticker)
                                if oms_pos:
                                    alloc_obj = oms_pos.allocations.get(STRATEGY_ID)
                                    if alloc_obj and alloc_obj.cost_basis > 0:
                                        actual_price = alloc_obj.cost_basis
                            except Exception:
                                pass

                            s.fsm = FSMState.IN_POSITION
                            s.entry_px = actual_price
                            s.qty = alloc_qty
                            s.remaining_qty = alloc_qty
                            s.max_price = actual_price
                            s.trail_stop = 0.0
                            s.partial_filled = (alloc_qty < s._pending_qty)
                            s.entry_order_id = None
                            s.order_submit_ts = 0.0
                            working_orders.discard(ticker)

                            if sector_exposure:
                                sector_exposure.on_fill(s.code, alloc_qty, actual_price)
                            positions.add(ticker)
                            _mfe_prices[ticker] = actual_price
                            _mae_prices[ticker] = actual_price

                            # Emit on_entry_fill using pre-built signal context
                            if s._entry_signal_factors and instr:
                                portfolio_state = {
                                    "total_exposure_pct": acct.gross_exposure_pct if acct else 0.0,
                                    "num_positions": len(positions),
                                    "concurrent_positions_same_strategy": len(positions),
                                }
                                dd_ctx = compute_drawdown_context(acct.daily_pnl_pct if acct else 0.0)
                                import hashlib, json as _json
                                _sw_params = kpr_switches.to_params_dict()
                                _strat_params = {"confidence": s.confidence, "setup_type": s.setup_type, **_sw_params}
                                _param_set_id = hashlib.sha256(_json.dumps(_sw_params, sort_keys=True, default=str).encode()).hexdigest()[:12]
                                _fill_confirmed_at = time_module.time()
                                _exec_timeline = None
                                if s.signal_generated_at and s.oms_received_at and s.order_submitted_at:
                                    _exec_timeline = {
                                        "signal_generated_at": s.signal_generated_at,
                                        "oms_received_at": s.oms_received_at,
                                        "order_submitted_at": s.order_submitted_at,
                                        "fill_confirmed_at": _fill_confirmed_at,
                                        "signal_to_oms_ms": int((s.oms_received_at - s.signal_generated_at) * 1000),
                                        "oms_processing_ms": int((s.order_submitted_at - s.oms_received_at) * 1000),
                                        "broker_to_fill_ms": int((_fill_confirmed_at - s.order_submitted_at) * 1000),
                                        "total_latency_ms": int((_fill_confirmed_at - s.signal_generated_at) * 1000),
                                    }
                                instr.on_entry_fill(
                                    trade_id=f"KPR:{ticker}:{now.strftime('%Y%m%d')}:{s.setup_type or 'drift'}",
                                    symbol=ticker, entry_price=actual_price, qty=alloc_qty,
                                    signal=f"{s.setup_type or 'drift'}_reclaim",
                                    signal_id="kpr_mean_reversion",
                                    strategy_params=_strat_params,
                                    signal_factors=s._entry_signal_factors,
                                    filter_decisions=s._entry_filter_decisions,
                                    sizing_context=s.sizing_context,
                                    portfolio_state=portfolio_state,
                                    drawdown_context=dd_ctx,
                                    param_set_id=_param_set_id,
                                    experiment_id=experiment_cfg.get("experiment_id", ""),
                                    experiment_variant=experiment_cfg.get("experiment_variant", ""),
                                    execution_timeline=_exec_timeline,
                                )
                                if hasattr(s, 'bid') and (s.bid > 0 or s.ask > 0):
                                    instr.on_orderbook_context(
                                        pair=ticker,
                                        best_bid=s.bid, best_ask=s.ask,
                                        trade_context="entry",
                                        related_trade_id=f"KPR:{ticker}:{now.strftime('%Y%m%d')}:{s.setup_type or 'drift'}",
                                    )
                            # Clear pending fields
                            s._entry_signal_factors = None
                            s._entry_filter_decisions = None
                            s._pending_qty = 0
                            logger.info(f"{ticker}: Entry fill confirmed, IN_POSITION qty={alloc_qty} px={actual_price:.0f}")
                        # else: no allocation yet, stay PENDING_ENTRY
                    except Exception as e:
                        logger.warning(f"{ticker}: PENDING_ENTRY check failed: {e}")
                    continue

                # --- Entry FSM ---
                investor_age = investor_provider.age_sec(ticker, now_ts)
                regime_ok = not acct.halt_new_entries
                if not regime_ok and not getattr(alpha_step, '_halt_logged', False):
                    logger.warning("KPR: halt_new_entries active — all entries blocked")
                    alpha_step._halt_logged = True
                elif regime_ok and getattr(alpha_step, '_halt_logged', False):
                    alpha_step._halt_logged = False
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
                    instr=instr,
                    experiment_id=experiment_cfg.get("experiment_id", ""),
                    experiment_variant=experiment_cfg.get("experiment_variant", ""),
                )

                if intent_id:
                    working_orders.add(ticker)
                # positions.add and on_entry_fill now happen in PENDING_ENTRY confirmation
                if s.fsm not in (FSMState.IN_POSITION, FSMState.PENDING_ENTRY):
                    positions.discard(ticker)

            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}", exc_info=True)

        await asyncio.sleep(1)

    instr.build_daily_snapshot()
    instr.shutdown()
    await oms.close()


def main():
    asyncio.run(run_kpr())


if __name__ == "__main__":
    main()
