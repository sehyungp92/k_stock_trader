"""OMS FastAPI Server.

Exposes OMSCore over HTTP for multi-strategy deployment.
"""

from __future__ import annotations
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger

from kis_core import KoreaInvestEnv, KoreaInvestAPI, build_kis_config_from_env
from .oms_core import OMSCore
from .intent import Intent, IntentType, IntentStatus, IntentResult, Urgency, TimeHorizon, IntentConstraints, RiskPayload
from .state import StrategyAllocation
from .risk import RiskConfig
from .persistence import OMSPersistence


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------

def load_oms_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load OMS configuration from YAML file.

    Args:
        config_path: Path to config file. If None, searches in standard locations.

    Returns:
        Configuration dictionary. Empty dict if no config found.
    """
    # Default search paths
    search_paths = [
        config_path,
        os.environ.get("OMS_CONFIG_PATH"),
        "config/oms_config.yaml",
        "../config/oms_config.yaml",
        Path(__file__).parent.parent / "config" / "oms_config.yaml",
    ]

    for path in search_paths:
        if path is None:
            continue

        path = Path(path)
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
                logger.info(f"Loaded OMS config from {path}")
                return config
            except Exception as e:
                logger.warning(f"Failed to load config from {path}: {e}")

    logger.info("No OMS config file found, using defaults")
    return {}


def build_risk_config(config: Dict[str, Any]) -> RiskConfig:
    """
    Build RiskConfig from loaded configuration.

    Args:
        config: Configuration dictionary from load_oms_config()

    Returns:
        RiskConfig with values from config (or defaults if not specified)
    """
    risk_section = config.get("risk", {})
    regime_caps = config.get("regime_exposure_caps", {})
    strategy_budgets = config.get("strategy_budgets")

    return RiskConfig(
        # Daily circuit breakers
        daily_loss_warn_pct=risk_section.get("daily_loss_warn_pct", 0.02),
        daily_loss_halt_pct=risk_section.get("daily_loss_halt_pct", 0.03),
        # Exposure limits
        max_gross_exposure_pct=risk_section.get("max_gross_exposure_pct", 0.80),
        max_net_exposure_pct=risk_section.get("max_net_exposure_pct", 0.60),
        max_position_pct=risk_section.get("max_position_pct", 0.15),
        max_positions_count=risk_section.get("max_positions_count", 10),
        max_sector_pct=risk_section.get("max_sector_pct", 0.30),
        # Strategy budgets
        strategy_budgets=strategy_budgets,
        # Microstructure
        max_spread_bps=risk_section.get("max_spread_bps", 50.0),
        vi_cooldown_sec=risk_section.get("vi_cooldown_sec", 600.0),
        # Regime caps
        regime_exposure_caps=regime_caps if regime_caps else None,
    )


# ---------------------------------------------------------------------------
# Pydantic models for HTTP API
# ---------------------------------------------------------------------------

class IntentConstraintsModel(BaseModel):
    max_slippage_bps: Optional[float] = None
    max_spread_bps: Optional[float] = None
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    expiry_ts: Optional[float] = None


class RiskPayloadModel(BaseModel):
    entry_px: Optional[float] = None
    stop_px: Optional[float] = None
    hard_stop_px: Optional[float] = None
    rationale_code: str = ""
    confidence: str = "YELLOW"


class IntentRequest(BaseModel):
    intent_type: str
    strategy_id: str
    symbol: str
    desired_qty: Optional[int] = None
    target_qty: Optional[int] = None
    urgency: str = "NORMAL"
    time_horizon: str = "INTRADAY"
    constraints: IntentConstraintsModel = IntentConstraintsModel()
    risk_payload: RiskPayloadModel = RiskPayloadModel()
    signal_hash: Optional[str] = None


class IntentResultModel(BaseModel):
    intent_id: str
    status: str
    message: str = ""
    modified_qty: Optional[int] = None
    order_id: Optional[str] = None
    cooldown_until: Optional[float] = None


class AllocationInfo(BaseModel):
    strategy_id: str
    qty: int
    cost_basis: float
    entry_ts: Optional[datetime] = None
    soft_stop_px: Optional[float] = None
    time_stop_ts: Optional[float] = None


class PositionInfo(BaseModel):
    symbol: str
    real_qty: int
    avg_price: float
    allocations: Dict[str, AllocationInfo]
    hard_stop_px: Optional[float] = None
    entry_lock_owner: Optional[str] = None
    entry_lock_until: Optional[float] = None
    frozen: bool
    working_order_count: int


class AccountState(BaseModel):
    equity: float
    buyable_cash: float
    daily_pnl: float
    daily_pnl_pct: float
    safe_mode: bool
    halt_new_entries: bool
    flatten_in_progress: bool


class HealthResponse(BaseModel):
    status: str
    uptime_sec: float
    positions_count: int
    kis_circuit_breaker: Optional[str] = None
    recon_status: Optional[str] = None


class RegimeRequest(BaseModel):
    regime: str


class VICooldownRequest(BaseModel):
    symbol: str
    duration_sec: int


class StrategyHeartbeatRequest(BaseModel):
    mode: str = "RUNNING"
    symbols_hot: int = 0
    symbols_warm: int = 0
    symbols_cold: int = 0
    positions_count: int = 0
    last_error: Optional[str] = None
    version: Optional[str] = None


# ---------------------------------------------------------------------------
# Global OMS instance (singleton within the service)
# ---------------------------------------------------------------------------

_oms: Optional[OMSCore] = None
_start_time = time.time()


def get_oms() -> OMSCore:
    if _oms is None:
        raise HTTPException(status_code=503, detail="OMS not initialized")
    return _oms


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _oms
    logger.info("OMS Server starting...")

    # Load OMS configuration from file
    oms_config = load_oms_config()
    risk_config = build_risk_config(oms_config)

    if oms_config.get("strategy_budgets"):
        logger.info(f"Loaded strategy budgets: {list(oms_config['strategy_budgets'].keys())}")

    # Load KIS credentials from environment
    kis_config = build_kis_config_from_env()
    logger.info(f"Trading mode: {'PAPER' if kis_config['is_paper_trading'] else 'LIVE'}")
    env = KoreaInvestEnv(kis_config)
    api = KoreaInvestAPI(env)

    # Initialize persistence (optional - will degrade gracefully if Postgres unavailable)
    persistence = OMSPersistence()

    _oms = OMSCore(api, risk_config=risk_config, persistence=persistence)
    await _oms.start()

    logger.info("OMS Server ready")
    yield

    logger.info("OMS Server shutting down...")
    await _oms.shutdown()


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(title="OMS", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    oms = get_oms()
    cb_status = oms.adapter.api.get_circuit_breaker_status()
    cb_state = cb_status.get("state", "UNKNOWN")
    drift_count = sum(1 for p in oms.state.get_all_positions().values() if p.frozen)
    overall_status = "ok"
    if cb_state == "OPEN":
        overall_status = "degraded"
    elif drift_count > 0:
        overall_status = "warn"

    # Check reconciliation loop health
    recon_status = "WARN" if drift_count > 0 else "OK"
    if hasattr(oms, '_reconcile_task') and oms._reconcile_task and oms._reconcile_task.done():
        overall_status = "error"
        recon_status = "DEAD"

    # Check persistence health
    if oms.persistence and hasattr(oms.persistence, 'consecutive_failures'):
        if oms.persistence.consecutive_failures >= 5:
            if overall_status == "ok":
                overall_status = "degraded"
            recon_status = f"{recon_status},PERSIST_FAIL({oms.persistence.consecutive_failures})"

    return HealthResponse(
        status=overall_status,
        uptime_sec=time.time() - _start_time,
        positions_count=len(oms.state.get_all_positions()),
        kis_circuit_breaker=cb_state,
        recon_status=recon_status,
    )


# ---------------------------------------------------------------------------
# Intent Submission
# ---------------------------------------------------------------------------

@app.post("/api/v1/intents", response_model=IntentResultModel)
async def submit_intent(req: IntentRequest):
    oms = get_oms()

    intent = Intent(
        intent_type=IntentType[req.intent_type],
        strategy_id=req.strategy_id,
        symbol=req.symbol,
        desired_qty=req.desired_qty,
        target_qty=req.target_qty,
        urgency=Urgency[req.urgency],
        time_horizon=TimeHorizon[req.time_horizon],
        constraints=IntentConstraints(
            max_slippage_bps=req.constraints.max_slippage_bps,
            max_spread_bps=req.constraints.max_spread_bps,
            limit_price=req.constraints.limit_price,
            stop_price=req.constraints.stop_price,
            expiry_ts=req.constraints.expiry_ts,
        ),
        risk_payload=RiskPayload(
            entry_px=req.risk_payload.entry_px,
            stop_px=req.risk_payload.stop_px,
            hard_stop_px=req.risk_payload.hard_stop_px,
            rationale_code=req.risk_payload.rationale_code,
            confidence=req.risk_payload.confidence,
        ),
        signal_hash=req.signal_hash,
    )

    result = await oms.submit_intent(intent)

    return IntentResultModel(
        intent_id=result.intent_id,
        status=result.status.name,
        message=result.message,
        modified_qty=result.modified_qty,
        order_id=result.order_id,
        cooldown_until=result.cooldown_until,
    )


# ---------------------------------------------------------------------------
# State Queries
# ---------------------------------------------------------------------------

def _alloc_to_model(alloc: StrategyAllocation) -> AllocationInfo:
    return AllocationInfo(
        strategy_id=alloc.strategy_id,
        qty=alloc.qty,
        cost_basis=alloc.cost_basis,
        entry_ts=alloc.entry_ts,
        soft_stop_px=alloc.soft_stop_px,
        time_stop_ts=alloc.time_stop_ts,
    )


@app.get("/api/v1/positions", response_model=Dict[str, PositionInfo])
async def get_positions():
    oms = get_oms()
    result = {}
    for symbol, pos in oms.state.get_all_positions().items():
        result[symbol] = PositionInfo(
            symbol=pos.symbol,
            real_qty=pos.real_qty,
            avg_price=pos.avg_price,
            allocations={k: _alloc_to_model(v) for k, v in pos.allocations.items()},
            hard_stop_px=pos.hard_stop_px,
            entry_lock_owner=pos.entry_lock_owner,
            entry_lock_until=pos.entry_lock_until,
            frozen=pos.frozen,
            working_order_count=len(pos.working_orders),
        )
    return result


@app.get("/api/v1/positions/{symbol}", response_model=PositionInfo)
async def get_position(symbol: str):
    oms = get_oms()
    pos = oms.state.get_position(symbol)
    return PositionInfo(
        symbol=pos.symbol,
        real_qty=pos.real_qty,
        avg_price=pos.avg_price,
        allocations={k: _alloc_to_model(v) for k, v in pos.allocations.items()},
        hard_stop_px=pos.hard_stop_px,
        entry_lock_owner=pos.entry_lock_owner,
        entry_lock_until=pos.entry_lock_until,
        frozen=pos.frozen,
        working_order_count=len(pos.working_orders),
    )


@app.get("/api/v1/allocations/{strategy_id}", response_model=Dict[str, AllocationInfo])
async def get_allocations(strategy_id: str):
    oms = get_oms()
    allocs = oms.state.get_allocations_for_strategy(strategy_id.upper())
    return {symbol: _alloc_to_model(alloc) for symbol, alloc in allocs.items()}


# ---------------------------------------------------------------------------
# Strategy Heartbeat
# ---------------------------------------------------------------------------

@app.post("/api/v1/strategies/{strategy_id}/heartbeat")
async def strategy_heartbeat(strategy_id: str, req: StrategyHeartbeatRequest):
    """Receive heartbeat from a strategy, updating its state in the database."""
    oms = get_oms()
    if oms.persistence:
        await oms.persistence.update_strategy_state(
            strategy_id=strategy_id.upper(),
            mode=req.mode,
            symbols_hot=req.symbols_hot,
            symbols_warm=req.symbols_warm,
            symbols_cold=req.symbols_cold,
            positions_count=req.positions_count,
            last_error=req.last_error,
            version=req.version,
        )
    return {"status": "ok"}


@app.get("/api/v1/state/account", response_model=AccountState)
async def get_account_state(strategy_id: Optional[str] = None):
    oms = get_oms()
    equity = oms.state.equity

    # Apply capital allocation if strategy_id provided
    if strategy_id:
        budget = oms.risk.config.strategy_budgets.get(strategy_id.upper(), {})
        alloc_pct = budget.get("capital_allocation_pct", 1.0)
        equity = equity * alloc_pct

    return AccountState(
        equity=equity,
        buyable_cash=oms.state.buyable_cash,
        daily_pnl=oms.state.daily_pnl,
        daily_pnl_pct=oms.state.daily_pnl_pct,
        safe_mode=oms.risk.safe_mode,
        halt_new_entries=oms.risk.halt_new_entries,
        flatten_in_progress=oms.risk.flatten_in_progress,
    )


# ---------------------------------------------------------------------------
# Risk Controls
# ---------------------------------------------------------------------------

@app.post("/api/v1/risk/regime")
async def set_regime(req: RegimeRequest):
    oms = get_oms()
    oms.risk.set_regime(req.regime)
    # Persist regime change immediately
    if oms.persistence:
        from datetime import date
        await oms.persistence.update_daily_risk_portfolio(
            trade_date=date.today(),
            equity_krw=oms.state.equity,
            buyable_cash_krw=oms.state.buyable_cash,
            realized_pnl_krw=0,  # Updated on fills
            unrealized_pnl_krw=0,
            gross_exposure_krw=0,
            positions_count=len(oms.state.get_all_positions()),
            halted=oms.risk.halt_new_entries,
            safe_mode=oms.risk.safe_mode,
            regime=req.regime,
        )
    return {"status": "ok", "regime": req.regime}


@app.post("/api/v1/risk/vi-cooldown")
async def set_vi_cooldown(req: VICooldownRequest):
    oms = get_oms()
    pos = oms.state.get_position(req.symbol)
    pos.vi_cooldown_until = time.time() + req.duration_sec
    # Persist position state immediately
    if oms.persistence:
        await oms.persistence.sync_position(pos)
    return {"status": "ok"}


@app.post("/api/v1/risk/safe-mode")
async def set_safe_mode(enabled: bool = True):
    oms = get_oms()
    oms.risk.safe_mode = enabled
    # Persist safe_mode immediately via heartbeat
    if oms.persistence:
        drift_count = sum(
            1 for p in oms.state.get_all_positions().values()
            if p.frozen
        )
        await oms.persistence.heartbeat(
            equity_krw=oms.state.equity,
            buyable_cash_krw=oms.state.buyable_cash,
            daily_pnl_krw=oms.state.daily_pnl,
            daily_pnl_pct=oms.state.daily_pnl_pct,
            safe_mode=enabled,
            halt_new_entries=oms.risk.halt_new_entries,
            kis_connected=True,
            recon_status="WARN" if drift_count > 0 else "OK",
            drift_count=drift_count,
        )
    return {"status": "ok", "safe_mode": enabled}


# ---------------------------------------------------------------------------
# Admin / Operator
# ---------------------------------------------------------------------------

@app.post("/api/v1/admin/flatten-all")
async def flatten_all():
    oms = get_oms()
    await oms.flatten_all()
    return {"status": "ok"}


@app.post("/api/v1/admin/eod-cleanup")
async def eod_cleanup():
    oms = get_oms()
    await oms.eod_cleanup()
    return {"status": "ok"}


@app.post("/api/v1/admin/pause-strategy/{strategy_id}")
async def pause_strategy(strategy_id: str):
    oms = get_oms()
    oms.risk._paused_strategies.add(strategy_id.upper())
    # Persist paused state
    if oms.persistence:
        await oms.persistence.update_strategy_state(
            strategy_id=strategy_id.upper(),
            mode="PAUSED",
        )
    return {"status": "ok", "paused": strategy_id.upper()}


@app.post("/api/v1/admin/resume-strategy/{strategy_id}")
async def resume_strategy(strategy_id: str):
    oms = get_oms()
    oms.risk._paused_strategies.discard(strategy_id.upper())
    # Persist resumed state
    if oms.persistence:
        await oms.persistence.update_strategy_state(
            strategy_id=strategy_id.upper(),
            mode="RUNNING",
        )
    return {"status": "ok", "resumed": strategy_id.upper()}


class ResolveDriftRequest(BaseModel):
    symbol: str
    action: str  # "reassign" or "acknowledge"
    target_strategy_id: Optional[str] = None  # Required for "reassign"


@app.post("/api/v1/admin/resolve-drift")
async def resolve_drift(req: ResolveDriftRequest):
    """Resolve allocation drift by reassigning _UNKNOWN_ or acknowledging it."""
    oms = get_oms()
    pos = oms.state.get_position(req.symbol)
    unknown_alloc = pos.allocations.get("_UNKNOWN_")

    if not unknown_alloc or unknown_alloc.qty == 0:
        raise HTTPException(status_code=404, detail=f"No _UNKNOWN_ allocation for {req.symbol}")

    if req.action == "reassign":
        if not req.target_strategy_id:
            raise HTTPException(status_code=400, detail="target_strategy_id required for reassign")
        target_id = req.target_strategy_id.upper()
        oms.state.update_allocation(req.symbol, target_id, unknown_alloc.qty, cost_basis=pos.avg_price)
        unknown_alloc.qty = 0
        logger.info(f"Reassigned {req.symbol} _UNKNOWN_ to {target_id}")
    elif req.action == "acknowledge":
        unknown_alloc.qty = 0
        logger.info(f"Acknowledged and cleared _UNKNOWN_ for {req.symbol}")
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {req.action}")

    # Check if drift is resolved and unfreeze
    if pos.allocation_drift() == 0:
        pos.frozen = False
        logger.info(f"Unfroze {req.symbol} after drift resolution")

    if oms.persistence:
        await oms.persistence.log_recon(
            "ALLOCATION_DRIFT", symbol=req.symbol, action=f"RESOLVED_{req.action.upper()}",
            details=f"Admin resolved drift via {req.action}",
        )

    return {"status": "ok", "symbol": req.symbol, "frozen": pos.frozen}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    port = int(os.environ.get("OMS_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
