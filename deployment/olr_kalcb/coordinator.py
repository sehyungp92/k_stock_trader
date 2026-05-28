from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.engine import KALCBEngine, KALCBOMSLiveAdapter
from strategy_kalcb.models import KALCBDailySnapshot
from strategy_olr.config import OLRConfig
from strategy_olr.engine import OLREngine, OLROMSLiveAdapter
from strategy_olr.models import OLRDailySnapshot

from .readiness import load_strategy_artifact, required_stage_for

ARTIFACT_ONLY_DESCRIPTOR_MODES = {"artifact_only", "artifact_only_stage1"}
EXECUTION_DESCRIPTOR_MODES = {"dry_run", "paper", "live"}
DESCRIPTOR_MODES = ARTIFACT_ONLY_DESCRIPTOR_MODES | EXECUTION_DESCRIPTOR_MODES


@dataclass(slots=True)
class StrategyRuntimeDescriptor:
    strategy_id: str
    artifact_stage: str
    artifact_hash: str
    engine: Any
    snapshot: KALCBDailySnapshot | OLRDailySnapshot
    oms_adapter: Any | None = None
    priority: int = 100
    config_fingerprint: dict[str, Any] = field(default_factory=dict)


def create_strategy_descriptor(
    strategy_id: str,
    snapshot: KALCBDailySnapshot | OLRDailySnapshot,
    *,
    mode: str = "paper",
    oms_client: Any | None = None,
    kalcb_config: KALCBConfig | None = None,
    olr_config: OLRConfig | None = None,
    dry_run: bool | None = None,
    allow_unrouted_execution: bool = False,
    allow_unoptimized_defaults_for_artifact_only: bool = False,
    config_fingerprint: Mapping[str, Any] | None = None,
) -> StrategyRuntimeDescriptor:
    sid = strategy_id.upper().strip()
    mode_name = _normalize_descriptor_mode(mode)
    artifact_default_allowed = mode_name in ARTIFACT_ONLY_DESCRIPTOR_MODES and allow_unoptimized_defaults_for_artifact_only
    adapter_dry_run = bool(dry_run if dry_run is not None else mode_name not in {"paper", "live"})
    broker_capable_adapter_allowed = bool(allow_unrouted_execution or mode_name not in EXECUTION_DESCRIPTOR_MODES)
    if sid == "KALCB":
        if kalcb_config is None and mode_name in EXECUTION_DESCRIPTOR_MODES:
            raise ValueError("KALCB execution descriptor requires an explicit approved config")
        if kalcb_config is None and not artifact_default_allowed:
            raise ValueError("KALCB descriptor requires explicit config unless artifact-only defaults are explicitly allowed")
        cfg = kalcb_config or KALCBConfig()
        engine = KALCBEngine(config=cfg, candidate_snapshot=_ensure_kalcb(snapshot))
        adapter = KALCBOMSLiveAdapter(oms_client, config=cfg, dry_run=adapter_dry_run) if oms_client is not None and broker_capable_adapter_allowed else None
        return StrategyRuntimeDescriptor(
            sid,
            str(snapshot.metadata.get("artifact_stage") or ""),
            snapshot.artifact_hash,
            engine,
            snapshot,
            adapter,
            priority=10,
            config_fingerprint=dict(config_fingerprint or {}),
        )
    if sid == "OLR":
        if olr_config is None and mode_name in EXECUTION_DESCRIPTOR_MODES:
            raise ValueError("OLR execution descriptor requires an explicit approved config")
        if olr_config is None and not artifact_default_allowed:
            raise ValueError("OLR descriptor requires explicit config unless artifact-only defaults are explicitly allowed")
        cfg = olr_config or OLRConfig()
        engine = OLREngine(config=cfg, candidate_snapshot=_ensure_olr(snapshot))
        adapter = OLROMSLiveAdapter(oms_client, config=cfg, dry_run=adapter_dry_run) if oms_client is not None and broker_capable_adapter_allowed else None
        return StrategyRuntimeDescriptor(
            sid,
            str(snapshot.metadata.get("artifact_stage") or ""),
            snapshot.artifact_hash,
            engine,
            snapshot,
            adapter,
            priority=20,
            config_fingerprint=dict(config_fingerprint or {}),
        )
    raise ValueError(f"unsupported strategy_id={strategy_id!r}")


def create_strategy_descriptors(
    strategy_ids: list[str] | tuple[str, ...],
    *,
    trade_date: date,
    mode: str = "paper",
    artifact_roots: dict[str, str | Path] | None = None,
    oms_client: Any | None = None,
    allow_unrouted_execution: bool = False,
    kalcb_config: KALCBConfig | None = None,
    olr_config: OLRConfig | None = None,
    allow_unoptimized_defaults_for_artifact_only: bool = False,
) -> dict[str, StrategyRuntimeDescriptor]:
    descriptors: dict[str, StrategyRuntimeDescriptor] = {}
    for raw_sid in strategy_ids:
        sid = raw_sid.upper().strip()
        stage = required_stage_for(sid, mode)
        snapshot = load_strategy_artifact(sid, trade_date, stage, mode, artifact_roots=artifact_roots)
        descriptors[sid] = create_strategy_descriptor(
            sid,
            snapshot,
            mode=mode,
            oms_client=oms_client,
            allow_unrouted_execution=allow_unrouted_execution,
            kalcb_config=kalcb_config,
            olr_config=olr_config,
            allow_unoptimized_defaults_for_artifact_only=allow_unoptimized_defaults_for_artifact_only,
        )
    return descriptors


def _normalize_descriptor_mode(mode: str) -> str:
    mode_name = str(mode or "").strip().lower()
    if mode_name not in DESCRIPTOR_MODES:
        raise ValueError(f"unsupported runtime descriptor mode {mode!r}")
    return mode_name


def _ensure_kalcb(snapshot: Any) -> KALCBDailySnapshot:
    if not isinstance(snapshot, KALCBDailySnapshot):
        raise TypeError("KALCB descriptor requires KALCBDailySnapshot")
    return snapshot


def _ensure_olr(snapshot: Any) -> OLRDailySnapshot:
    if not isinstance(snapshot, OLRDailySnapshot):
        raise TypeError("OLR descriptor requires OLRDailySnapshot")
    return snapshot
