"""OMS FastAPI Server.

Exposes OMSCore over HTTP for multi-strategy deployment.
"""

from __future__ import annotations
import inspect
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from kis_core import KoreaInvestEnv, KoreaInvestAPI, build_kis_config_from_env
from .config_loader import (
    build_risk_config as _build_risk_config,
    configured_active_strategy_ids,
    effective_risk_config_payload,
    load_oms_sector_map,
    load_oms_config_with_source,
    missing_strategy_budgets,
    stable_mapping_hash,
)
from .oms_core import OMSCore
from .intent import Intent, IntentType, IntentStatus, IntentResult, Urgency, TimeHorizon, IntentConstraints, RiskPayload
from .state import StrategyAllocation, WorkingOrder
from .risk import RiskConfig
from .persistence import OMSPersistence
try:
    from instrumentation.src.deployment_logger import DeploymentLogger
    from instrumentation.src.lineage import context_from_env
    from instrumentation.src.oms_exporter import OMSEventEmitter
    from instrumentation.src.runtime_lineage import load_runtime_deployment_lineage
except Exception:  # pragma: no cover - OMS must run even if telemetry import fails
    DeploymentLogger = None  # type: ignore
    OMSEventEmitter = None  # type: ignore
    load_runtime_deployment_lineage = None  # type: ignore

    def context_from_env(**_: Any) -> None:  # type: ignore
        return None


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
    try:
        config, source = load_oms_config_with_source(config_path)
    except Exception as exc:
        logger.warning(f"Failed to load OMS config: {exc}")
        return {}
    if source is not None:
        logger.info(f"Loaded OMS config from {source}")
    else:
        logger.info("No OMS config file found, using defaults")
    return config


def build_risk_config(config: Dict[str, Any]) -> RiskConfig:
    """
    Build RiskConfig from loaded configuration.

    Args:
        config: Configuration dictionary from load_oms_config()

    Returns:
        RiskConfig with values from config (or defaults if not specified)
    """
    return _build_risk_config(config)


# ---------------------------------------------------------------------------
# Pydantic models for HTTP API
# ---------------------------------------------------------------------------

class IntentConstraintsModel(BaseModel):
    max_slippage_bps: Optional[float] = None
    max_spread_bps: Optional[float] = None
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    expiry_ts: Optional[float] = None
    execution_style: Optional[str] = None


class RiskPayloadModel(BaseModel):
    entry_px: Optional[float] = None
    stop_px: Optional[float] = None
    hard_stop_px: Optional[float] = None
    rationale_code: str = ""
    confidence: str = "YELLOW"


class IntentRequest(BaseModel):
    intent_id: Optional[str] = None
    idempotency_key: Optional[str] = None
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
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IntentResultModel(BaseModel):
    intent_id: str
    status: str
    message: str = ""
    modified_qty: Optional[int] = None
    order_id: Optional[str] = None
    cooldown_until: Optional[float] = None
    blocking_positions: Optional[List[Dict[str, Any]]] = None
    resource_conflict_type: Optional[str] = None
    oms_received_at: Optional[float] = None
    order_submitted_at: Optional[float] = None


class AllocationInfo(BaseModel):
    strategy_id: str
    qty: int
    cost_basis: float
    entry_ts: Optional[datetime] = None
    soft_stop_px: Optional[float] = None
    time_stop_ts: Optional[float] = None


class WorkingOrderInfo(BaseModel):
    order_id: str
    symbol: str
    side: str
    qty: int
    filled_qty: int
    remaining_qty: int
    price: float
    order_type: str
    status: str
    strategy_id: str
    intent_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    submit_ref: Optional[str] = None
    risk_stop_px: Optional[float] = None
    risk_hard_stop_px: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    submit_ts: Optional[float] = None
    cancel_after_sec: Optional[float] = None


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
    working_orders: List[WorkingOrderInfo] = Field(default_factory=list)


class AccountState(BaseModel):
    equity: float
    buyable_cash: float
    daily_pnl: float
    daily_pnl_pct: float
    safe_mode: bool
    halt_new_entries: bool
    flatten_in_progress: bool
    gross_exposure_pct: float = 0.0
    regime_exposure_cap: float = 1.0


class HealthResponse(BaseModel):
    status: str
    uptime_sec: float
    positions_count: int
    kis_circuit_breaker: Optional[str] = None
    recon_status: Optional[str] = None
    strategies: Optional[Dict[str, Any]] = None
    stop_protection_status: str = "unknown"
    unprotected_positions_count: int = 0
    active_stop_count: int = 0
    triggered_stop_count: int = 0
    stop_watcher_last_check_age_sec: Optional[float] = None
    stop_watcher_price_stale_count: int = 0
    idempotency_status: str = "unknown"


class RegimeRequest(BaseModel):
    regime: str


class VICooldownRequest(BaseModel):
    symbol: str
    duration_sec: int


class IdempotencyResolveRequest(BaseModel):
    status: str = "DEFERRED"
    reason: str
    order_id: Optional[str] = None


class StrategyHeartbeatRequest(BaseModel):
    mode: str = "RUNNING"
    symbols_hot: int = 0
    symbols_warm: int = 0
    symbols_cold: int = 0
    positions_count: int = 0
    last_error: Optional[str] = None
    version: Optional[str] = None
    pulse_verdict: Optional[str] = None
    pulse_md_ok_pct: Optional[float] = None
    pulse_signals_eval: Optional[int] = None


# ---------------------------------------------------------------------------
# Global OMS instance (singleton within the service)
# ---------------------------------------------------------------------------

_oms: Optional[OMSCore] = None
_start_time = time.time()
_strategy_heartbeats: Dict[str, dict] = {}  # strategy_id -> {ts, mode, version, pulse_verdict}


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
    logger.add(
        "/app/data/logs/oms_{time:YYYY-MM-DD}.log",
        rotation="00:00", retention="30 days", compression="gz",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    )
    logger.info("OMS Server starting...")

    # Load OMS configuration from file
    try:
        oms_config, oms_config_source = load_oms_config_with_source()
    except Exception as exc:
        logger.warning(f"Failed to load OMS config: {exc}")
        oms_config, oms_config_source = {}, None
    if oms_config_source is not None:
        logger.info(f"Loaded OMS config from {oms_config_source}")
    else:
        logger.info("No OMS config file found, using defaults")
    risk_config = build_risk_config(oms_config)
    active_strategy_ids = configured_active_strategy_ids(oms_config)

    if oms_config.get("strategy_budgets"):
        logger.info(f"Loaded strategy budgets: {list(oms_config['strategy_budgets'].keys())}")

    # Load KIS credentials from environment
    kis_config = build_kis_config_from_env()
    logger.info(f"Trading mode: {'PAPER' if kis_config['is_paper_trading'] else 'LIVE'}")
    missing_budgets = missing_strategy_budgets(risk_config, active_strategy_ids)
    if missing_budgets and os.environ.get("OMS_ALLOW_MISSING_STRATEGY_BUDGETS", "").lower() not in {"1", "true", "yes"}:
        raise RuntimeError(
            "OMS startup blocked: active strategies missing strategy_budgets: "
            + ", ".join(missing_budgets)
        )
    sector_map, sector_map_source = load_oms_sector_map(oms_config, config_source=oms_config_source)
    if not sector_map and os.environ.get("OMS_ALLOW_MISSING_SECTOR_MAP", "").lower() not in {"1", "true", "yes"}:
        raise RuntimeError("OMS startup blocked: approved sector map is missing or empty")
    if sector_map_source is not None:
        logger.info(f"Loaded OMS sector map from {sector_map_source} ({len(sector_map)} symbols)")
    else:
        logger.info(f"Loaded OMS sector map from inline config ({len(sector_map)} symbols)")
    env = KoreaInvestEnv(kis_config)
    api = KoreaInvestAPI(env)

    # Initialize persistence. Paper/live requires durable state unless explicitly
    # overridden for emergency operator workflows.
    oms_id = os.environ.get("OMS_ID", "primary")
    persistence = OMSPersistence(oms_id=oms_id)
    require_persistence = os.environ.get("OMS_ALLOW_NO_DATABASE", "").lower() not in {"1", "true", "yes"}
    logger.info(f"OMS instance: {oms_id}")

    assistant_event_dir = os.environ.get("ASSISTANT_EVENT_DATA_DIR", "instrumentation/data")
    lineage = _runtime_deployment_lineage(context_from_env(data_source_id="postgres_oms"), assistant_event_dir)
    event_emitter = None
    if assistant_event_dir.lower() not in {"", "off", "none", "disabled"} and OMSEventEmitter is not None:
        try:
            event_emitter = OMSEventEmitter(assistant_event_dir, lineage=lineage)
            if DeploymentLogger is not None:
                risk_snapshot = {
                    **effective_risk_config_payload(oms_config),
                    "sector_map_hash": stable_mapping_hash(sector_map),
                    "sector_map_size": len(sector_map),
                    "sector_map_source": str(sector_map_source or ""),
                    "stop_protection_modes_supported": ["OMS_WATCHER", "SYNTHETIC_ONLY"],
                    "default_stop_protection_mode": str(
                        (oms_config.get("risk") or {}).get("default_stop_protection_mode", "oms_watcher")
                    ),
                    "broker_native_stop_verified_at": None,
                    "idempotency_uniqueness_scope": "global_idempotency_key",
                    "reservation_rehydration_mode": "position_working_orders",
                }
                snapshot_event = DeploymentLogger(assistant_event_dir, lineage=lineage).emit_config_snapshot(
                    risk_config=risk_snapshot,
                    strategy_registry={"strategy_ids": list(active_strategy_ids), "producer": "oms_server"},
                    source_files=[path for path in (oms_config_source, sector_map_source) if path is not None],
                    environment={"OMS_ID": oms_id, "OMS_CONFIG_PATH": os.environ.get("OMS_CONFIG_PATH", "")},
                )
                _apply_config_snapshot_lineage(event_emitter, snapshot_event)
        except Exception:
            event_emitter = None
    _oms = OMSCore(
        api,
        risk_config=risk_config,
        persistence=persistence,
        event_emitter=event_emitter,
        sector_map=sector_map,
        require_persistence=require_persistence,
    )
    await _oms.start()

    logger.info("OMS Server ready")
    yield

    logger.info("OMS Server shutting down...")
    await _oms.shutdown()


def _apply_config_snapshot_lineage(event_emitter: Any, snapshot_event: Mapping[str, Any] | None) -> None:
    if event_emitter is None or not isinstance(snapshot_event, Mapping):
        return
    payload = snapshot_event.get("payload")
    if not isinstance(payload, Mapping):
        return
    updater = getattr(event_emitter, "update_lineage", None)
    if not callable(updater):
        return
    current = getattr(event_emitter, "lineage", None)
    updater(**_lineage_missing_overrides(current, payload))


def _runtime_deployment_lineage(lineage: Any, assistant_event_dir: str | Path) -> Any:
    if lineage is None or load_runtime_deployment_lineage is None:
        return lineage
    try:
        payload = load_runtime_deployment_lineage(assistant_event_dir)
    except Exception:
        return lineage
    if not payload:
        return lineage
    return lineage.with_overrides(**_lineage_authoritative_overrides(payload))


def _lineage_missing_overrides(lineage: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: value
        for field, value in _lineage_authoritative_overrides(payload).items()
        if not getattr(lineage, field, "")
    }


def _lineage_authoritative_overrides(payload: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "deployment_id",
        "strategy_version",
        "config_version",
        "portfolio_config_version",
        "risk_config_version",
        "allocation_version",
        "strategy_registry_version",
        "kis_resource_plan_hash",
        "portfolio_id",
        "account_alias",
        "portfolio_policy_hash",
        "code_sha",
    )
    return {
        field: payload.get(field)
        for field in fields
        if payload.get(field) not in (None, "")
    }


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(title="OMS", version="1.0.0", lifespan=lifespan)

_REQUIRED_STOP_HEALTH_FIELDS = (
    "unprotected_positions_count",
    "active_stop_count",
    "triggered_stop_count",
    "stop_watcher_price_stale_count",
)
_MAX_STOP_WATCHER_AGE_SEC = 60.0


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
    recon_status = "warn" if drift_count > 0 else "ok"
    if hasattr(oms, '_reconcile_task') and oms._reconcile_task and oms._reconcile_task.done():
        overall_status = "error"
        recon_status = "dead"

    # Check persistence health
    if oms.persistence and hasattr(oms.persistence, 'consecutive_failures'):
        if oms.persistence.consecutive_failures >= 5:
            if overall_status == "ok":
                overall_status = "degraded"
            recon_status = f"{recon_status},persist_fail({oms.persistence.consecutive_failures})"

    # Strategy liveness from in-memory heartbeat tracking
    strategies = {}
    for strat_id, info in _strategy_heartbeats.items():
        age = time.time() - info["ts"]
        strategies[strat_id] = {
            "last_heartbeat_sec_ago": round(age, 1),
            "status": "alive" if age < 300 else "stale",
            "mode": info["mode"],
            "version": info.get("version"),
            "pulse_verdict": info.get("pulse_verdict"),
        }
    if any(s["status"] == "stale" for s in strategies.values()):
        if overall_status == "ok":
            overall_status = "warn"

    idempotency_status = "ok"
    if getattr(oms, "require_persistence", False):
        connected = bool(
            oms.persistence is not None
            and callable(getattr(oms.persistence, "_is_connected", None))
            and oms.persistence._is_connected()
        )
        if not connected:
            idempotency_status = "error"
            overall_status = "error"
            recon_status = f"{recon_status},idempotency_error"
        elif getattr(oms.persistence, "consecutive_failures", 0) >= 5:
            idempotency_status = "degraded"
            if overall_status == "ok":
                overall_status = "degraded"
        else:
            health_getter = getattr(oms.persistence, "idempotency_health", None)
            if callable(health_getter):
                idem_health = health_getter()
                if inspect.isawaitable(idem_health):
                    idem_health = await idem_health
                idempotency_status = str((idem_health or {}).get("status") or "unknown").lower().strip()
                if idempotency_status == "error":
                    overall_status = "error"
                    recon_status = f"{recon_status},idempotency_error"
                elif idempotency_status != "ok":
                    if overall_status == "ok":
                        overall_status = "degraded"
                    recon_status = f"{recon_status},idempotency_{idempotency_status or 'unknown'}"

    stop_health = oms.stop_health_payload() if hasattr(oms, "stop_health_payload") else {}
    stop_status = str(stop_health.get("stop_protection_status") or "unknown")
    missing_stop_fields = [
        field
        for field in _REQUIRED_STOP_HEALTH_FIELDS
        if field not in stop_health or stop_health.get(field) is None
    ]
    stop_counts = {
        field: _health_int_or_none(stop_health, field)
        for field in _REQUIRED_STOP_HEALTH_FIELDS
        if field not in missing_stop_fields
    }
    invalid_stop_fields = [field for field, value in stop_counts.items() if value is None]
    unprotected_positions_count = stop_counts.get("unprotected_positions_count") or 0
    active_stop_count = stop_counts.get("active_stop_count") or 0
    triggered_stop_count = stop_counts.get("triggered_stop_count") or 0
    stop_watcher_price_stale_count = stop_counts.get("stop_watcher_price_stale_count") or 0
    stop_watcher_last_check_age_sec = stop_health.get("stop_watcher_last_check_age_sec")
    watcher_age_present = stop_watcher_last_check_age_sec is not None
    watcher_age = _health_float_or_none(stop_health, "stop_watcher_last_check_age_sec")
    watcher_age_invalid = watcher_age_present and watcher_age is None
    active_stop_lacks_watcher_check = active_stop_count > 0 and watcher_age is None
    active_stop_stale_watcher_check = (
        active_stop_count > 0
        and watcher_age is not None
        and watcher_age > _MAX_STOP_WATCHER_AGE_SEC
    )
    if missing_stop_fields:
        stop_status = "error"
        overall_status = "error"
        recon_status = f"{recon_status},stop_health_missing({','.join(missing_stop_fields)})"
    if invalid_stop_fields or watcher_age_invalid:
        invalid_fields = list(invalid_stop_fields)
        if watcher_age_invalid:
            invalid_fields.append("stop_watcher_last_check_age_sec")
        stop_status = "error"
        overall_status = "error"
        recon_status = f"{recon_status},stop_health_invalid({','.join(invalid_fields)})"
    if active_stop_lacks_watcher_check:
        stop_status = "error"
        overall_status = "error"
        recon_status = f"{recon_status},stop_watcher_missing"
    elif active_stop_stale_watcher_check:
        stop_status = "degraded"
        if overall_status == "ok":
            overall_status = "degraded"
        recon_status = f"{recon_status},stop_watcher_stale"
    if stop_watcher_price_stale_count > 0:
        if stop_status != "error":
            stop_status = "degraded"
        if overall_status in {"ok", "warn"}:
            overall_status = "degraded"
        recon_status = f"{recon_status},stop_price_stale({stop_watcher_price_stale_count})"
    if stop_status == "error":
        overall_status = "error"
        recon_status = f"{recon_status},stop_error"
    elif stop_status == "degraded" and overall_status == "ok":
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        uptime_sec=time.time() - _start_time,
        positions_count=len(oms.state.get_all_positions()),
        kis_circuit_breaker=cb_state,
        recon_status=recon_status,
        strategies=strategies if strategies else None,
        stop_protection_status=stop_status,
        unprotected_positions_count=unprotected_positions_count,
        active_stop_count=active_stop_count,
        triggered_stop_count=triggered_stop_count,
        stop_watcher_last_check_age_sec=watcher_age,
        stop_watcher_price_stale_count=stop_watcher_price_stale_count,
        idempotency_status=idempotency_status,
    )


def _health_int(payload: Mapping[str, Any], field: str) -> int:
    value = _health_int_or_none(payload, field)
    return value if value is not None else 0


def _health_int_or_none(payload: Mapping[str, Any], field: str) -> Optional[int]:
    try:
        value = int(payload.get(field))
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _health_float_or_none(payload: Mapping[str, Any], field: str) -> Optional[float]:
    try:
        value = float(payload.get(field))
    except (TypeError, ValueError):
        return None
    return value if value >= 0.0 else None


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
            execution_style=req.constraints.execution_style,
        ),
        risk_payload=RiskPayload(
            entry_px=req.risk_payload.entry_px,
            stop_px=req.risk_payload.stop_px,
            hard_stop_px=req.risk_payload.hard_stop_px,
            rationale_code=req.risk_payload.rationale_code,
            confidence=req.risk_payload.confidence,
        ),
        signal_hash=req.signal_hash,
        metadata=dict(req.metadata or {}),
        intent_id=req.intent_id or str(uuid.uuid4()),
        idempotency_key=req.idempotency_key,
    )

    result = await oms.submit_intent(intent)

    return IntentResultModel(
        intent_id=result.intent_id,
        status=result.status.name,
        message=result.message,
        modified_qty=result.modified_qty,
        order_id=result.order_id,
        cooldown_until=result.cooldown_until,
        blocking_positions=result.blocking_positions,
        resource_conflict_type=result.resource_conflict_type,
        oms_received_at=result.oms_received_at,
        order_submitted_at=result.order_submitted_at,
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


def _working_order_to_model(order: WorkingOrder) -> WorkingOrderInfo:
    filled_qty = max(int(order.filled_qty or 0), 0)
    qty = max(int(order.qty or 0), 0)
    return WorkingOrderInfo(
        order_id=order.order_id,
        symbol=order.symbol,
        side=order.side,
        qty=qty,
        filled_qty=filled_qty,
        remaining_qty=max(qty - filled_qty, 0),
        price=float(order.price or 0.0),
        order_type=order.order_type,
        status=order.status.name,
        strategy_id=order.strategy_id,
        intent_id=order.intent_id,
        idempotency_key=order.idempotency_key,
        submit_ref=order.submit_ref,
        risk_stop_px=order.risk_stop_px,
        risk_hard_stop_px=order.risk_hard_stop_px,
        created_at=order.created_at,
        updated_at=order.updated_at,
        submit_ts=order.submit_ts,
        cancel_after_sec=order.cancel_after_sec,
    )


@app.get("/api/v1/positions", response_model=Dict[str, PositionInfo])
async def get_positions():
    oms = get_oms()
    result = {}
    for symbol, pos in oms.state.get_all_positions().items():
        working_orders = [_working_order_to_model(order) for order in pos.working_orders]
        result[symbol] = PositionInfo(
            symbol=pos.symbol,
            real_qty=pos.real_qty,
            avg_price=pos.avg_price,
            allocations={k: _alloc_to_model(v) for k, v in pos.allocations.items()},
            hard_stop_px=pos.hard_stop_px,
            entry_lock_owner=pos.entry_lock_owner,
            entry_lock_until=pos.entry_lock_until,
            frozen=pos.frozen,
            working_order_count=len(working_orders),
            working_orders=working_orders,
        )
    return result


@app.get("/api/v1/positions/{symbol}", response_model=PositionInfo)
async def get_position(symbol: str):
    oms = get_oms()
    pos = oms.state.get_position(symbol)
    working_orders = [_working_order_to_model(order) for order in pos.working_orders]
    return PositionInfo(
        symbol=pos.symbol,
        real_qty=pos.real_qty,
        avg_price=pos.avg_price,
        allocations={k: _alloc_to_model(v) for k, v in pos.allocations.items()},
        hard_stop_px=pos.hard_stop_px,
        entry_lock_owner=pos.entry_lock_owner,
        entry_lock_until=pos.entry_lock_until,
        frozen=pos.frozen,
        working_order_count=len(working_orders),
        working_orders=working_orders,
    )


@app.get("/api/v1/working-orders", response_model=List[WorkingOrderInfo])
async def get_working_orders():
    oms = get_oms()
    return [_working_order_to_model(order) for order in oms.state.get_working_orders()]


@app.get("/api/v1/idempotency/pending")
async def get_pending_idempotency(stale_after_sec: float = 60.0):
    oms = get_oms()
    if not oms.persistence:
        return []
    lister = getattr(oms.persistence, "list_pending_idempotency", None)
    if not callable(lister):
        return []
    return await lister(stale_after_sec=stale_after_sec)


@app.post("/api/v1/idempotency/{key}/resolve", response_model=IntentResultModel)
async def resolve_idempotency(key: str, req: IdempotencyResolveRequest):
    oms = get_oms()
    if not req.reason.strip():
        raise HTTPException(status_code=400, detail="resolution reason is required")
    if not oms.persistence:
        raise HTTPException(status_code=503, detail="persistence unavailable")
    try:
        status = IntentStatus[req.status.upper()]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"unsupported status {req.status!r}")
    resolver = getattr(oms.persistence, "resolve_idempotency", None)
    if not callable(resolver):
        raise HTTPException(status_code=503, detail="idempotency resolver unavailable")
    result = await resolver(key, status=status, reason=req.reason, order_id=req.order_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"idempotency key not found: {key}")
    _emit_operator_event(
        oms,
        "IDEMPOTENCY_RESOLVE",
        payload={"idempotency_key": key, "status": status.name, "reason": req.reason, "order_id": req.order_id},
    )
    return IntentResultModel(
        intent_id=result.intent_id,
        status=result.status.name,
        message=result.message,
        modified_qty=result.modified_qty,
        order_id=result.order_id,
        cooldown_until=result.cooldown_until,
        blocking_positions=result.blocking_positions,
        resource_conflict_type=result.resource_conflict_type,
        oms_received_at=result.oms_received_at,
        order_submitted_at=result.order_submitted_at,
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
    _strategy_heartbeats[strategy_id.upper()] = {
        "ts": time.time(),
        "mode": req.mode,
        "version": req.version,
        "pulse_verdict": req.pulse_verdict,
    }
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

    # Compute gross exposure
    positions = oms.state.get_all_positions()
    gross = sum(
        p.real_qty * (p.avg_price or 0.0) + p.working_qty(side="BUY") * (p.avg_price or 0.0)
        for p in positions.values()
    )
    total_equity = max(oms.state.equity, 1.0)
    gross_pct = gross / total_equity
    regime_cap = oms.risk.config.regime_exposure_caps.get(oms.risk.config.current_regime, 1.0)

    return AccountState(
        equity=equity,
        buyable_cash=oms.state.buyable_cash,
        daily_pnl=oms.state.daily_pnl,
        daily_pnl_pct=oms.state.daily_pnl_pct,
        safe_mode=oms.risk.safe_mode,
        halt_new_entries=oms.risk.halt_new_entries,
        flatten_in_progress=oms.risk.flatten_in_progress,
        gross_exposure_pct=round(gross_pct, 4),
        regime_exposure_cap=regime_cap,
    )


# ---------------------------------------------------------------------------
# Risk Controls
# ---------------------------------------------------------------------------

def _emit_operator_event(oms: OMSCore, action: str, *, symbol: str = "", payload: Optional[Dict[str, Any]] = None) -> None:
    emitter = getattr(oms, "event_emitter", None)
    if emitter is None:
        return
    try:
        row = {
            "action": action,
            "manual": True,
            "source": "oms_server",
            "safe_mode": getattr(oms.risk, "safe_mode", False),
            "halt_new_entries": getattr(oms.risk, "halt_new_entries", False),
            **dict(payload or {}),
        }
        emitter.emit_reconciliation("RISK_CONTROL_CHANGE", symbol=symbol, payload=row)
        emitter.emit_portfolio_snapshot(oms, reason=f"risk_control:{action.lower()}")
    except Exception:
        pass


@app.post("/api/v1/risk/regime")
async def set_regime(req: RegimeRequest):
    oms = get_oms()
    previous = oms.risk.config.current_regime
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
    _emit_operator_event(oms, "SET_REGIME", payload={"before_value": previous, "after_value": req.regime})
    return {"status": "ok", "regime": req.regime}


@app.post("/api/v1/risk/vi-cooldown")
async def set_vi_cooldown(req: VICooldownRequest):
    oms = get_oms()
    pos = oms.state.get_position(req.symbol)
    previous = pos.vi_cooldown_until
    pos.vi_cooldown_until = time.time() + req.duration_sec
    # Persist position state immediately
    if oms.persistence:
        await oms.persistence.sync_position(pos)
    _emit_operator_event(
        oms,
        "SET_VI_COOLDOWN",
        symbol=req.symbol,
        payload={"before_value": previous, "after_value": pos.vi_cooldown_until, "duration_sec": req.duration_sec},
    )
    return {"status": "ok"}


@app.post("/api/v1/risk/safe-mode")
async def set_safe_mode(enabled: bool = True):
    oms = get_oms()
    previous = oms.risk.safe_mode
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
            recon_status="warn" if drift_count > 0 else "ok",
            drift_count=drift_count,
        )
    _emit_operator_event(oms, "SET_SAFE_MODE", payload={"before_value": previous, "after_value": enabled})
    return {"status": "ok", "safe_mode": enabled}


# ---------------------------------------------------------------------------
# Admin / Operator
# ---------------------------------------------------------------------------

@app.post("/api/v1/admin/flatten-all")
async def flatten_all():
    oms = get_oms()
    await oms.flatten_all()
    _emit_operator_event(oms, "FLATTEN_ALL")
    return {"status": "ok"}


@app.post("/api/v1/admin/eod-cleanup")
async def eod_cleanup():
    oms = get_oms()
    await oms.eod_cleanup()
    _emit_operator_event(oms, "EOD_CLEANUP")
    return {"status": "ok"}


@app.post("/api/v1/admin/pause-strategy/{strategy_id}")
async def pause_strategy(strategy_id: str):
    oms = get_oms()
    sid = strategy_id.upper()
    was_paused = sid in oms.risk._paused_strategies
    oms.risk._paused_strategies.add(sid)
    # Persist paused state
    if oms.persistence:
        await oms.persistence.update_strategy_state(
            strategy_id=strategy_id.upper(),
            mode="PAUSED",
        )
    _emit_operator_event(oms, "PAUSE_STRATEGY", payload={"strategy_id": sid, "before_value": was_paused, "after_value": True})
    return {"status": "ok", "paused": sid}


@app.post("/api/v1/admin/resume-strategy/{strategy_id}")
async def resume_strategy(strategy_id: str):
    oms = get_oms()
    sid = strategy_id.upper()
    was_paused = sid in oms.risk._paused_strategies
    oms.risk._paused_strategies.discard(sid)
    # Persist resumed state
    if oms.persistence:
        await oms.persistence.update_strategy_state(
            strategy_id=strategy_id.upper(),
            mode="RUNNING",
        )
    _emit_operator_event(oms, "RESUME_STRATEGY", payload={"strategy_id": sid, "before_value": was_paused, "after_value": False})
    return {"status": "ok", "resumed": sid}


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
        before_value = {"unknown_qty": unknown_alloc.qty, "target_qty": pos.get_allocation(target_id), "frozen": pos.frozen}
        oms.state.update_allocation(req.symbol, target_id, unknown_alloc.qty, cost_basis=pos.avg_price)
        unknown_alloc.qty = 0
        logger.info(f"Reassigned {req.symbol} _UNKNOWN_ to {target_id}")
    elif req.action == "acknowledge":
        before_value = {"unknown_qty": unknown_alloc.qty, "frozen": pos.frozen}
        unknown_alloc.qty = 0
        logger.info(f"Acknowledged and cleared _UNKNOWN_ for {req.symbol}")
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {req.action}")

    # Check if drift is resolved and unfreeze
    if pos.allocation_drift() == 0:
        pos.frozen = False
        logger.info(f"Unfroze {req.symbol} after drift resolution")

    if oms.persistence:
        if req.action == "reassign":
            target_alloc = pos.allocations.get(target_id)
            if target_alloc:
                await oms.persistence.sync_allocation(req.symbol, target_alloc)
        await oms.persistence.sync_allocation(req.symbol, unknown_alloc)
        await oms.persistence.sync_position(pos)
        await oms.persistence.log_recon(
            "ALLOCATION_DRIFT", symbol=req.symbol, action=f"RESOLVED_{req.action.upper()}",
            details=f"Admin resolved drift via {req.action}",
        )

    if unknown_alloc.qty == 0:
        pos.allocations.pop("_UNKNOWN_", None)

    _emit_operator_event(
        oms,
        "RESOLVE_DRIFT",
        symbol=req.symbol,
        payload={
            "drift_action": req.action,
            "target_strategy_id": req.target_strategy_id.upper() if req.target_strategy_id else "",
            "before_value": before_value,
            "after_value": {"drift": pos.allocation_drift(), "frozen": pos.frozen},
        },
    )
    return {"status": "ok", "symbol": req.symbol, "frozen": pos.frozen}


class CorrectAllocationRequest(BaseModel):
    symbol: str
    strategy_id: str
    new_qty: int


@app.post("/api/v1/admin/correct-allocation")
async def correct_allocation(req: CorrectAllocationRequest):
    """Admin: set a strategy's allocation to a specific value."""
    oms = get_oms()
    pos = oms.state.get_position(req.symbol)
    if req.strategy_id.upper() not in pos.allocations:
        raise HTTPException(status_code=404, detail=f"No allocation for {req.strategy_id} on {req.symbol}")
    if req.new_qty < 0:
        raise HTTPException(status_code=400, detail="new_qty cannot be negative")
    if req.new_qty > pos.real_qty:
        raise HTTPException(status_code=400, detail=f"new_qty ({req.new_qty}) > real_qty ({pos.real_qty})")
    result = await oms.correct_allocation(req.symbol, req.strategy_id.upper(), req.new_qty)
    _emit_operator_event(
        oms,
        "CORRECT_ALLOCATION",
        symbol=req.symbol,
        payload={
            "strategy_id": req.strategy_id.upper(),
            "before_value": result.get("old_qty"),
            "after_value": result.get("new_qty"),
            "drift": result.get("drift"),
            "frozen": result.get("frozen"),
        },
    )
    return {"status": "ok", **result}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    port = int(os.environ.get("OMS_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
