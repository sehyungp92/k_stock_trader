from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from strategy_kalcb.artifact_store import KALCBArtifactStore
from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.models import KALCBDailySnapshot
from strategy_kalcb.research import KALCB_FINAL_ARTIFACT_STAGE, candidate_config_fingerprint as kalcb_candidate_config_fingerprint
from strategy_olr.artifact_store import OLR_FINAL_ARTIFACT_STAGE, OLR_STAGE1_ARTIFACT_STAGE, OLRArtifactStore
from strategy_olr.config import OLRConfig
from strategy_olr.models import OLRDailySnapshot
from strategy_olr.research import final_candidate_config_fingerprint as olr_final_candidate_config_fingerprint

from .action_router import RoutedOMSAdapter, RuntimeActionRouter
from .coordinator import StrategyRuntimeDescriptor, create_strategy_descriptor
from .deployment_metadata import emit_deployment_metadata
from .dry_run_oms import RecordingOMSClient
from .hashing import canonical_json_hash, file_sha256
from .kis_resource_plan import (
    KISResourcePlan,
    build_kis_resource_plan,
    target_strategy_ids_for_bar,
)
from .portfolio import PortfolioArbitrationPolicy, PortfolioPolicyConfig
from .portfolio_context import PortfolioContextProvider, _coerce_account, _coerce_positions
from .readiness import DEFAULT_ARTIFACT_ROOTS, ArtifactReadinessFailure, load_strategy_artifact, required_stage_for, validate_strategy_artifacts
from .session_capture import PaperSessionRecorder
from .session_driver import RuntimeSessionDriver, handle_combined_bar

try:
    from instrumentation.src.lineage import LineageContext, get_code_sha
except Exception:  # pragma: no cover - instrumentation must be fail-open
    LineageContext = None  # type: ignore[assignment]
    get_code_sha = None  # type: ignore[assignment]

ARTIFACT_ONLY_MODES = {"artifact_only", "artifact_only_stage1"}
EXECUTION_MODES = {"dry_run", "paper", "live"}
RUNTIME_MODES = ARTIFACT_ONLY_MODES | EXECUTION_MODES
DEFAULT_STRATEGY_CONFIG_SOURCE = Path(
    os.environ.get("OLR_KALCB_BASELINE_MANIFEST", "data/live_readiness/olr_kalcb/2026-05-28/baseline_manifest.json")
)

REQUIRED_HEALTH_CHECKS_BY_MODE: dict[str, tuple[str, ...]] = {
    "artifact_only": (),
    "artifact_only_stage1": (),
    "dry_run": ("artifact_only_gate_passed", "market_data_ok", "risk_limits_loaded"),
    "paper": (
        "dry_run_gate_passed",
        "market_session_open",
        "kis_auth_ok",
        "market_data_ok",
        "account_ok",
        "order_route_enabled",
        "risk_limits_loaded",
        "kill_switch_ready",
        "oms_health_ok",
        "durable_stops_ok",
        "idempotency_reservation_ok",
        "portfolio_context_fresh",
        "paper_trading_approved",
    ),
    "live": (
        "paper_replay_gate_passed",
        "market_session_open",
        "kis_auth_ok",
        "market_data_ok",
        "account_ok",
        "order_route_enabled",
        "risk_limits_loaded",
        "kill_switch_ready",
        "oms_health_ok",
        "durable_stops_ok",
        "idempotency_reservation_ok",
        "portfolio_context_fresh",
        "live_capital_approved",
    ),
}

OMS_HARDENING_HEALTH_CHECKS = {"oms_health_ok", "durable_stops_ok", "idempotency_reservation_ok"}
OMS_HEALTH_PAYLOAD_KEYS = (
    "oms_health_payload",
    "raw_oms_health",
    "oms_health_body",
    "oms_health",
    "oms_health_gate_evidence",
)
_READY_HEALTH_STATES = {"ok", "ready", "healthy"}
_ACCEPTABLE_OMS_STATUS = _READY_HEALTH_STATES | {"warn"}
_REQUIRED_STOP_HEALTH_FIELDS = (
    "unprotected_positions_count",
    "active_stop_count",
    "triggered_stop_count",
    "stop_watcher_price_stale_count",
)
_MAX_STOP_WATCHER_AGE_SEC = 60.0


@dataclass(frozen=True, slots=True)
class RuntimePreflightCheck:
    name: str
    passed: bool
    detail: str = ""
    required: bool = True


@dataclass(frozen=True, slots=True)
class RuntimePreflightResult:
    mode: str
    trade_date: date
    checks: tuple[RuntimePreflightCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed or not check.required for check in self.checks)

    @property
    def failures(self) -> tuple[RuntimePreflightCheck, ...]:
        return tuple(check for check in self.checks if check.required and not check.passed)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "trade_date": self.trade_date.isoformat(),
            "passed": self.passed,
            "checks": [asdict(check) for check in self.checks],
        }


@dataclass(frozen=True, slots=True)
class RuntimeScheduleStep:
    name: str
    run_at_kst: str
    required: bool = True


@dataclass(frozen=True, slots=True)
class _RuntimeArtifactRequirement:
    strategy_id: str
    stage: str
    bucket: str
    path: Path
    snapshot: KALCBDailySnapshot | OLRDailySnapshot | None


@dataclass(frozen=True, slots=True)
class RuntimeSessionPlan:
    mode: str
    trade_date: date
    artifacts: dict[str, KALCBDailySnapshot | OLRDailySnapshot]
    artifact_failures: tuple[ArtifactReadinessFailure, ...]
    preflight: RuntimePreflightResult
    descriptors: dict[str, StrategyRuntimeDescriptor]
    drivers: dict[str, RuntimeSessionDriver]
    schedule: tuple[RuntimeScheduleStep, ...]
    portfolio_policy_hash: str | None = None
    portfolio_enabled: bool = True
    action_router: RuntimeActionRouter | None = None
    session_recorder: PaperSessionRecorder | None = None
    strategy_config_summaries: dict[str, dict[str, Any]] | None = None
    portfolio_policy_config: dict[str, Any] | None = None
    sector_map: dict[str, str] | None = None
    kis_resource_plan: KISResourcePlan | None = None
    kis_resource_plan_path: str | None = None
    strategy_configs: dict[str, KALCBConfig | OLRConfig] | None = None
    deployment_metadata_path: str | None = None
    deployment_metadata_contract_path: str | None = None
    deployment_metadata_environment: str | None = None
    runtime_entrypoint: str | None = None

    @property
    def ready_to_start(self) -> bool:
        if not self.preflight.passed:
            return False
        if self.mode in ARTIFACT_ONLY_MODES:
            return True
        return bool(self.drivers) and _closeout_capable(self.session_recorder)

    @property
    def kalcb_runtime_enabled(self) -> bool:
        return "KALCB" in self.drivers

    @property
    def olr_stage1_ready(self) -> bool:
        snapshot = self.artifacts.get("OLR")
        return bool(snapshot is not None and str((snapshot.metadata or {}).get("artifact_stage") or "") == OLR_STAGE1_ARTIFACT_STAGE)

    @property
    def olr_final_ready(self) -> bool:
        snapshot = self.artifacts.get("OLR")
        return bool(snapshot is not None and str((snapshot.metadata or {}).get("artifact_stage") or "") == OLR_FINAL_ARTIFACT_STAGE)

    @property
    def olr_runtime_enabled(self) -> bool:
        return "OLR" in self.drivers and self.olr_final_ready

    @property
    def resource_window_active(self) -> bool:
        return bool(self.kis_resource_plan is not None and self.kis_resource_plan.passed)

    @property
    def ready_for_kalcb_start(self) -> bool:
        return self.preflight.passed and self.kalcb_runtime_enabled and self.resource_window_active

    @property
    def ready_for_olr_final_generation(self) -> bool:
        return self.preflight.passed and ("OLR" in self.artifacts) and self.resource_window_active

    @property
    def ready_for_olr_start(self) -> bool:
        return self.preflight.passed and self.olr_runtime_enabled and self.resource_window_active

    @property
    def ready_for_closeout(self) -> bool:
        return _closeout_capable(self.session_recorder)

    def close_session(
        self,
        end_of_day_positions: Any,
        session_metrics: Mapping[str, Any] | None = None,
        *,
        closeout_reason: str = "normal_eod",
    ) -> Path:
        if not _closeout_capable(self.session_recorder):
            raise RuntimeError(f"{self.mode} runtime session has no closeout-capable recorder")
        return self.session_recorder.close_session(
            end_of_day_positions,
            session_metrics=session_metrics,
            closeout_reason=closeout_reason,
        )

    def enable_olr_final(
        self,
        snapshot: OLRDailySnapshot | None = None,
        *,
        artifact_root: str | Path | None = None,
        artifact_path: str | Path | None = None,
        initial_state: Any | None = None,
    ) -> RuntimeSessionDriver:
        if self.mode not in EXECUTION_MODES:
            raise RuntimeError(f"{self.mode} runtime plan cannot enable OLR final execution")
        if self.action_router is None or self.session_recorder is None:
            raise RuntimeError("OLR final enablement requires the existing routed runtime session")
        if "OLR" in self.drivers:
            raise RuntimeError("OLR final runtime is already enabled")
        if "OLR" not in dict(self.strategy_config_summaries or {}):
            raise RuntimeError("OLR final enablement requires approved OLR config captured at session start")
        final_snapshot, final_path = _load_olr_final_enablement_snapshot(
            self.trade_date,
            mode=self.mode,
            snapshot=snapshot,
            artifact_root=artifact_root,
            artifact_path=artifact_path,
        )
        olr_config = _runtime_config_for(self, "OLR")
        config_fingerprints = _approved_config_fingerprints(
            {"OLR": olr_config},
            {"OLR": dict((self.strategy_config_summaries or {}).get("OLR") or {})},
            sector_map=_canonical_sector_map(self.sector_map or {}),
        )
        binding_failures = _artifact_config_binding_failures(
            {"OLR": final_snapshot},
            config_fingerprints,
            mode=self.mode,
        )
        if binding_failures:
            raise ValueError(_artifact_failure_detail(binding_failures))
        configs = dict(self.strategy_configs or {})
        configs["OLR"] = olr_config
        artifacts = dict(self.artifacts)
        artifacts["OLR"] = final_snapshot
        resource_plan = _build_runtime_kis_resource_plan(
            tuple(self.strategy_config_summaries or ("KALCB", "OLR")),
            trade_date=self.trade_date,
            mode=self.mode,
            artifacts=artifacts,
            configs=configs,
            artifact_roots=_resource_artifact_roots_for_enablement(Path(final_path), artifact_root),
        )
        if not resource_plan.passed:
            raise ValueError(_resource_plan_failure_detail(resource_plan.failures))
        resource_plan_path = str(self.session_recorder.write_resource_plan(resource_plan.to_json_dict()))
        staged_row = _stage_runtime_artifact_row(
            self.session_recorder,
            _RuntimeArtifactRequirement(
                "OLR",
                OLR_FINAL_ARTIFACT_STAGE,
                "olr_final_snapshots",
                Path(final_path),
                final_snapshot,
            ),
            trade_date=self.trade_date,
            approved_config_fingerprints=config_fingerprints,
        )
        descriptor = create_strategy_descriptor(
            "OLR",
            final_snapshot,
            mode=self.mode,
            oms_client=self.action_router.oms_client,
            olr_config=olr_config,
            config_fingerprint=config_fingerprints.get("OLR"),
        )
        if initial_state is not None:
            descriptor.engine.state = _restore_strategy_state("OLR", initial_state)
        descriptor.oms_adapter = RoutedOMSAdapter(
            strategy_id="OLR",
            router=self.action_router,
            portfolio_adapter=descriptor.oms_adapter,
            dry_run=self.mode == "dry_run",
        )
        self.action_router.record_state_snapshot(
            "OLR",
            descriptor.engine.state,
            metadata={
                "record_reason": "runtime_session_pre_start",
                "mode": self.mode,
                "trade_date": self.trade_date.isoformat(),
                "artifact_stage": descriptor.artifact_stage,
                "artifact_hash": descriptor.artifact_hash,
                "enabled_during_session": True,
            },
        )
        portfolio_context = _shared_portfolio_context(self.drivers)
        driver = RuntimeSessionDriver(
            descriptor=descriptor,
            action_router=self.action_router,
            recorder=self.session_recorder,
            portfolio_context=portfolio_context,
            mode=self.mode,
        )
        self.artifacts["OLR"] = final_snapshot
        self.descriptors["OLR"] = descriptor
        self.drivers["OLR"] = driver
        if self.strategy_configs is not None:
            self.strategy_configs["OLR"] = olr_config
        object.__setattr__(self, "kis_resource_plan", resource_plan)
        object.__setattr__(self, "kis_resource_plan_path", resource_plan_path)
        _rewrite_runtime_manifest_for_enablement(
            self,
            staged_row=staged_row,
            resource_plan=resource_plan,
            resource_plan_path=resource_plan_path,
        )
        _emit_runtime_deployment_metadata(self)
        return driver

    async def handle_bar(self, bar: Any, *, target_strategy_ids: Sequence[str] | None = None) -> tuple[Any, ...]:
        if self.mode not in EXECUTION_MODES:
            raise RuntimeError(f"{self.mode} runtime plan cannot process market bars")
        if not self.ready_to_start or self.action_router is None:
            raise RuntimeError(f"{self.mode} runtime plan is not ready to process market bars")
        requested_targets = tuple(_normalize_strategy_id(item) for item in (target_strategy_ids or ()) if str(item).strip())
        disallowed_targets: tuple[str, ...] = ()
        targets = requested_targets
        if self.kis_resource_plan is not None:
            allowed_targets = target_strategy_ids_for_bar(
                self.kis_resource_plan,
                symbol=getattr(bar, "symbol", ""),
                timestamp=getattr(bar, "timestamp"),
                available_strategy_ids=tuple(self.drivers),
                held_or_pending_symbols=_held_or_pending_symbols(self.drivers),
            )
            if requested_targets:
                allowed_set = set(allowed_targets)
                targets = tuple(item for item in requested_targets if item in allowed_set)
                disallowed_targets = tuple(item for item in requested_targets if item not in allowed_set)
            else:
                targets = allowed_targets
        if disallowed_targets:
            _record_resource_route_suppression(
                self.session_recorder,
                self.kis_resource_plan,
                bar,
                requested_targets=disallowed_targets,
                reason_code="requested_target_not_in_active_resource_window",
            )
        if self.kis_resource_plan is not None and not targets:
            if self.session_recorder is not None and getattr(bar, "is_completed", False):
                self.session_recorder.record_market_bar(bar)
            if not disallowed_targets:
                _record_resource_route_suppression(
                    self.session_recorder,
                    self.kis_resource_plan,
                    bar,
                    requested_targets=requested_targets,
                    reason_code="no_resource_plan_target",
                )
            return ()
        return await handle_combined_bar(self.drivers, bar, target_strategy_ids=targets or None)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "trade_date": self.trade_date.isoformat(),
            "ready_to_start": self.ready_to_start,
            "ready_for_kalcb_start": self.ready_for_kalcb_start,
            "ready_for_olr_final_generation": self.ready_for_olr_final_generation,
            "ready_for_olr_start": self.ready_for_olr_start,
            "ready_for_closeout": self.ready_for_closeout,
            "kalcb_runtime_enabled": self.kalcb_runtime_enabled,
            "olr_stage1_ready": self.olr_stage1_ready,
            "olr_final_ready": self.olr_final_ready,
            "olr_runtime_enabled": self.olr_runtime_enabled,
            "resource_window_active": self.resource_window_active,
            "artifacts": {sid: _artifact_summary(snapshot) for sid, snapshot in self.artifacts.items()},
            "artifact_failures": [asdict(failure) for failure in self.artifact_failures],
            "preflight": self.preflight.to_json_dict(),
            "descriptors": {
                sid: {
                    "strategy_id": descriptor.strategy_id,
                    "artifact_stage": descriptor.artifact_stage,
                    "artifact_hash": descriptor.artifact_hash,
                    "has_oms_adapter": descriptor.oms_adapter is not None,
                    "priority": descriptor.priority,
                    "config_fingerprint": descriptor.config_fingerprint,
                }
                for sid, descriptor in self.descriptors.items()
            },
            "drivers": {
                sid: {
                    "strategy_id": driver.descriptor.strategy_id,
                    "mode": driver.mode,
                    "artifact_hash": driver.descriptor.artifact_hash,
                }
                for sid, driver in self.drivers.items()
            },
            "portfolio_policy_hash": self.portfolio_policy_hash,
            "portfolio_enabled": self.portfolio_enabled,
            "portfolio_policy_config": self.portfolio_policy_config,
            "strategy_configs": self.strategy_config_summaries or {},
            "sector_map": self.sector_map or {},
            "kis_resource_plan_hash": self.kis_resource_plan.plan_hash if self.kis_resource_plan is not None else "",
            "kis_resource_plan_path": self.kis_resource_plan_path or "",
            "kis_resource_plan": self.kis_resource_plan.to_json_dict() if self.kis_resource_plan is not None else None,
            "deployment_metadata_path": self.deployment_metadata_path or "",
            "has_action_router": self.action_router is not None,
            "has_session_recorder": self.session_recorder is not None,
            "closeout_capable": _closeout_capable(self.session_recorder),
            "schedule": [asdict(step) for step in self.schedule],
        }


def prepare_runtime_session(
    strategy_ids: Sequence[str],
    *,
    trade_date: date,
    mode: str = "artifact_only",
    completed_bar_source: str = "paced_rest",
    artifact_roots: dict[str, str | Path] | None = None,
    health_checks: Mapping[str, Any] | None = None,
    oms_client: Any | None = None,
    dry_run_oms_client: Any | None = None,
    session_recorder: PaperSessionRecorder | None = None,
    portfolio_config: PortfolioPolicyConfig | None = None,
    portfolio_enabled: bool = True,
    strategy_config_source: str | Path | Mapping[str, Any] | None = None,
    sector_map: Mapping[str, str] | None = None,
    initial_account_state: Any | None = None,
    initial_positions: Any | None = None,
    initial_strategy_states: Mapping[str, Any] | None = None,
    assistant_event_dir: str | Path | None = None,
    deployment_metadata_path: str | Path | None = None,
    deployment_metadata_contract_path: str | Path | None = None,
    deployment_metadata_environment: str | None = None,
    runtime_entrypoint: str | None = None,
) -> RuntimeSessionPlan:
    mode_name = _normalize_mode(mode)
    sids = tuple(dict.fromkeys(_normalize_strategy_id(strategy_id) for strategy_id in strategy_ids))
    if not sids:
        raise ValueError("at least one strategy_id is required")
    initial_states = {_normalize_strategy_id(key): value for key, value in dict(initial_strategy_states or {}).items()}
    artifacts, failures = _validate_strategy_artifacts_for_runtime(
        sids,
        trade_date=trade_date,
        mode=mode_name,
        artifact_roots=artifact_roots,
    )
    olr_final_required = _olr_final_required_for_runtime(sids, mode_name, artifacts)
    requires_portfolio = bool(portfolio_enabled and (len(sids) > 1 or mode_name in EXECUTION_MODES))
    portfolio = PortfolioArbitrationPolicy(portfolio_config) if requires_portfolio else None
    context_sector_map = _merged_sector_map(artifacts, sector_map)
    approved_sector_map, sector_map_failures = _approved_runtime_sector_map(sids, mode_name, sector_map)
    config_failures: list[ArtifactReadinessFailure] = []
    try:
        configs, config_summaries = _load_runtime_strategy_configs(
            sids,
            mode=mode_name,
            source=strategy_config_source,
        )
    except (FileNotFoundError, ValueError) as exc:
        if mode_name not in EXECUTION_MODES:
            raise
        configs = {}
        config_summaries = {}
        config_failures.extend(
            ArtifactReadinessFailure(sid, "approved_runtime_config", str(exc))
            for sid in sids
        )
    config_fingerprints = _approved_config_fingerprints(
        configs,
        config_summaries,
        sector_map=approved_sector_map,
    )
    for sid, fingerprint in config_fingerprints.items():
        config_summaries.setdefault(sid, {})["approved_config_fingerprint"] = fingerprint
    failures = [
        *failures,
        *sector_map_failures,
        *config_failures,
        *_artifact_config_binding_failures(artifacts, config_fingerprints, mode=mode_name),
    ]
    kis_resource_plan = _build_runtime_kis_resource_plan(
        sids,
        trade_date=trade_date,
        mode=mode_name,
        completed_bar_source=completed_bar_source,
        artifacts=artifacts,
        configs=configs,
        artifact_roots=artifact_roots,
    ) if mode_name in EXECUTION_MODES and not config_failures else None
    if kis_resource_plan is not None and not kis_resource_plan.passed:
        failures.extend(
            ArtifactReadinessFailure("KIS", "kis_resource_plan", failure)
            for failure in kis_resource_plan.failures
        )
    resource_plan_path = ""
    if session_recorder is not None and assistant_event_dir is not None:
        enable_export = getattr(session_recorder, "enable_assistant_export", None)
        if callable(enable_export):
            try:
                enable_export(assistant_event_dir, lineage=_runtime_assistant_lineage())
            except Exception:
                pass
    if kis_resource_plan is not None and session_recorder is not None:
        resource_plan_path = str(session_recorder.write_resource_plan(kis_resource_plan.to_json_dict()))
    effective_oms_client = _runtime_oms_client(
        mode_name,
        oms_client=oms_client,
        dry_run_oms_client=dry_run_oms_client,
        session_recorder=session_recorder,
        initial_account_state=initial_account_state,
        initial_positions=initial_positions,
    )
    initial_state_payload = _initial_runtime_state_payload(
        effective_oms_client,
        initial_account_state=initial_account_state,
        initial_positions=initial_positions,
    )
    artifact_bundle_available = _runtime_artifact_bundle_available(
        mode_name,
        sids,
        trade_date=trade_date,
        artifacts=artifacts,
        artifact_roots=artifact_roots,
        session_recorder=session_recorder,
        olr_final_required=olr_final_required,
    )
    preflight = run_runtime_preflight(
        mode_name,
        trade_date,
        artifact_failures=failures,
        health_checks=health_checks,
        requires_portfolio=requires_portfolio,
        portfolio_policy_loaded=portfolio is not None,
        oms_client_available=_oms_client_available(oms_client),
        dry_run_oms_available=_recording_oms_available(effective_oms_client),
        runtime_capture_available=session_recorder is not None,
        runtime_closeout_available=_closeout_capable(session_recorder),
        runtime_driver_available=True,
        runtime_initial_state_available=_initial_runtime_state_available(initial_state_payload),
        runtime_artifact_bundle_available=artifact_bundle_available,
        kis_resource_plan=kis_resource_plan,
        kis_resource_plan_hash_bound=bool(resource_plan_path and kis_resource_plan is not None and kis_resource_plan.plan_hash),
    )
    action_router = (
        RuntimeActionRouter(
            recorder=session_recorder,
            oms_client=effective_oms_client,
            portfolio_policy=portfolio,
            portfolio_enabled=portfolio is not None,
            dry_run=mode_name == "dry_run",
        )
        if preflight.passed and mode_name in EXECUTION_MODES and session_recorder is not None and _execution_oms_available(mode_name, effective_oms_client)
        else None
    )
    descriptors = (
        {
            sid: create_strategy_descriptor(
                sid,
                artifacts[sid],
                mode=mode_name,
                oms_client=effective_oms_client,
                kalcb_config=configs.get("KALCB") if sid == "KALCB" else None,
                olr_config=configs.get("OLR") if sid == "OLR" else None,
                config_fingerprint=config_fingerprints.get(sid),
            )
            for sid in sids
            if sid in artifacts and _artifact_is_executable_for_runtime(sid, artifacts[sid])
        }
        if preflight.passed and mode_name in EXECUTION_MODES
        else {}
    )
    portfolio_context = PortfolioContextProvider(effective_oms_client, sector_map=dict(context_sector_map)) if action_router is not None else None
    if portfolio_context is not None:
        if "initial_account_state" in initial_state_payload:
            portfolio_context.account_state = _coerce_account(initial_state_payload["initial_account_state"])
        if "initial_positions" in initial_state_payload:
            portfolio_context.positions = _coerce_positions(initial_state_payload["initial_positions"])
    drivers: dict[str, RuntimeSessionDriver] = {}
    if action_router is not None:
        staged_artifacts = _stage_runtime_artifacts(
            session_recorder,
            sids,
            trade_date=trade_date,
            artifacts=artifacts,
            artifact_roots=artifact_roots,
            approved_config_fingerprints=config_fingerprints,
            olr_final_required=olr_final_required,
        ) if session_recorder is not None else ()
        risk_config_payload, risk_config_source = _runtime_risk_config_payload()
        if session_recorder is not None:
            session_recorder.write_manifest(
                {
                    "mode": mode_name,
                    "strategy_ids": list(sids),
                    "strategy_configs": config_summaries,
                    "portfolio_enabled": portfolio is not None,
                    "portfolio_policy_config": asdict(portfolio.config) if portfolio is not None else None,
                    "portfolio_policy_hash": portfolio.policy_hash if portfolio is not None else None,
                    "sector_map": dict(context_sector_map),
                    "risk_config": risk_config_payload,
                    "risk_config_source": risk_config_source,
                    "staged_artifacts": staged_artifacts,
                    "kis_resource_plan_hash": kis_resource_plan.plan_hash if kis_resource_plan is not None else "",
                    "kis_resource_plan_path": resource_plan_path,
                    "kis_resource_plan_required": mode_name in EXECUTION_MODES,
                    **initial_state_payload,
                }
            )
        for descriptor in descriptors.values():
            initial_state = initial_states.get(descriptor.strategy_id)
            if initial_state is not None:
                descriptor.engine.state = _restore_strategy_state(descriptor.strategy_id, initial_state)
            descriptor.oms_adapter = RoutedOMSAdapter(
                strategy_id=descriptor.strategy_id,
                router=action_router,
                portfolio_adapter=descriptor.oms_adapter,
                dry_run=mode_name == "dry_run",
            )
            action_router.record_state_snapshot(
                descriptor.strategy_id,
                getattr(descriptor.engine, "state", None),
                metadata={
                    "record_reason": "runtime_session_pre_start",
                    "mode": mode_name,
                    "trade_date": trade_date.isoformat(),
                    "artifact_stage": descriptor.artifact_stage,
                    "artifact_hash": descriptor.artifact_hash,
                },
            )
            if portfolio_context is not None:
                drivers[descriptor.strategy_id] = RuntimeSessionDriver(
                    descriptor=descriptor,
                    action_router=action_router,
                    recorder=session_recorder,
                    portfolio_context=portfolio_context,
                    mode=mode_name,
                )
    preflight = _with_runtime_driver_check(preflight, mode_name, bool(drivers))
    plan = RuntimeSessionPlan(
        mode=mode_name,
        trade_date=trade_date,
        artifacts=artifacts,
        artifact_failures=tuple(failures),
        preflight=preflight,
        descriptors=descriptors,
        drivers=drivers,
        schedule=default_session_schedule(mode_name, sids),
        portfolio_policy_hash=portfolio.policy_hash if portfolio is not None else None,
        portfolio_enabled=bool(portfolio_enabled),
        action_router=action_router,
        session_recorder=session_recorder if mode_name in EXECUTION_MODES else None,
        strategy_config_summaries=config_summaries,
        portfolio_policy_config=asdict(portfolio.config) if portfolio is not None else None,
        sector_map=dict(context_sector_map),
        kis_resource_plan=kis_resource_plan,
        kis_resource_plan_path=resource_plan_path,
        strategy_configs=dict(configs),
        deployment_metadata_path=str(deployment_metadata_path or "") if _deployment_metadata_enabled(deployment_metadata_path) else "",
        deployment_metadata_contract_path=str(deployment_metadata_contract_path or ""),
        deployment_metadata_environment=deployment_metadata_environment or "",
        runtime_entrypoint=runtime_entrypoint or "",
    )
    _emit_runtime_deployment_metadata(plan)
    return plan


def run_runtime_preflight(
    mode: str,
    trade_date: date,
    *,
    artifact_failures: Sequence[ArtifactReadinessFailure] = (),
    health_checks: Mapping[str, Any] | None = None,
    requires_portfolio: bool = False,
    portfolio_policy_loaded: bool = False,
    oms_client_available: bool = False,
    dry_run_oms_available: bool = False,
    runtime_capture_available: bool = False,
    runtime_closeout_available: bool = False,
    runtime_driver_available: bool = False,
    runtime_initial_state_available: bool = False,
    runtime_artifact_bundle_available: bool = False,
    kis_resource_plan: KISResourcePlan | None = None,
    kis_resource_plan_hash_bound: bool = False,
) -> RuntimePreflightResult:
    mode_name = _normalize_mode(mode)
    checks = [
        RuntimePreflightCheck(
            "artifact_readiness",
            not artifact_failures,
            _artifact_failure_detail(artifact_failures) if artifact_failures else "ok",
        )
    ]
    if requires_portfolio:
        checks.append(
            RuntimePreflightCheck(
                "portfolio_policy_loaded",
                portfolio_policy_loaded,
                "ok" if portfolio_policy_loaded else "portfolio arbitration policy unavailable",
            )
        )
    if mode_name in {"paper", "live"}:
        checks.append(
            RuntimePreflightCheck(
                "oms_client_available",
                oms_client_available,
                "ok" if oms_client_available else "paper/live mode requires an OMS client",
            )
        )
    if mode_name == "dry_run":
        checks.append(
            RuntimePreflightCheck(
                "dry_run_oms_available",
                dry_run_oms_available,
                "ok" if dry_run_oms_available else "dry-run mode requires a recording OMS intent sink",
            )
        )
    if mode_name in EXECUTION_MODES:
        resource_failures = tuple(str(item) for item in kis_resource_plan.failures) if kis_resource_plan is not None else ()
        resource_plan_loaded = kis_resource_plan is not None
        resource_plan_passed = bool(kis_resource_plan is not None and kis_resource_plan.passed)
        kis_mode_ok = resource_plan_loaded and not any(item.startswith("kis_mode_mismatch") for item in resource_failures)
        rest_budget_ok = resource_plan_loaded and not any("rest" in item and "exceeded" in item for item in resource_failures)
        ws_budget_ok = resource_plan_loaded and not any(
            "ws" in item and ("exceeded" in item or "requires_explicit" in item) for item in resource_failures
        )
        checks.append(
            RuntimePreflightCheck(
                "runtime_evidence_capture",
                runtime_capture_available,
                "ok" if runtime_capture_available else f"{mode_name} mode requires session evidence capture",
            )
        )
        checks.append(
            RuntimePreflightCheck(
                "runtime_closeout_available",
                runtime_closeout_available,
                "ok" if runtime_closeout_available else f"{mode_name} mode requires a closeout-capable session recorder",
            )
        )
        checks.append(
            RuntimePreflightCheck(
                "runtime_driver_available",
                runtime_driver_available,
                "ok" if runtime_driver_available else f"{mode_name} mode requires RuntimeSessionDriver execution",
            )
        )
        checks.append(
            RuntimePreflightCheck(
                "runtime_initial_state_capture",
                runtime_initial_state_available,
                "ok" if runtime_initial_state_available else f"{mode_name} mode requires captured initial account and positions",
            )
        )
        checks.append(
            RuntimePreflightCheck(
                "runtime_artifact_bundle",
                runtime_artifact_bundle_available,
                "ok" if runtime_artifact_bundle_available else f"{mode_name} mode requires staged candidate artifact snapshots",
            )
        )
        checks.append(
            RuntimePreflightCheck(
                "kis_resource_plan_loaded",
                resource_plan_loaded,
                "ok" if resource_plan_loaded else f"{mode_name} mode requires a KIS resource plan",
            )
        )
        checks.append(
            RuntimePreflightCheck(
                "kis_resource_plan_passed",
                resource_plan_passed,
                "ok" if resource_plan_passed else _resource_plan_failure_detail(resource_failures),
            )
        )
        checks.append(
            RuntimePreflightCheck(
                "kis_mode_matches_runtime_mode",
                kis_mode_ok,
                "ok" if kis_mode_ok else "runtime mode does not match detected KIS paper/live mode",
            )
        )
        checks.append(
            RuntimePreflightCheck(
                "kis_rest_budget_ok",
                rest_budget_ok,
                "ok" if rest_budget_ok else "KIS REST demand exceeds approved resource plan",
            )
        )
        checks.append(
            RuntimePreflightCheck(
                "kis_ws_budget_ok",
                ws_budget_ok,
                "ok" if ws_budget_ok else "KIS WebSocket demand exceeds approved resource plan",
            )
        )
        checks.append(
            RuntimePreflightCheck(
                "kis_resource_plan_hash_bound",
                kis_resource_plan_hash_bound,
                "ok" if kis_resource_plan_hash_bound else "KIS resource plan hash is not bound to session evidence",
            )
        )
    provided_checks = health_checks or {}
    derived_oms_checks = (
        _derive_oms_hardening_health_checks(provided_checks)
        if mode_name in {"paper", "live"}
        else {}
    )
    for name in REQUIRED_HEALTH_CHECKS_BY_MODE[mode_name]:
        if name in derived_oms_checks:
            check = derived_oms_checks[name]
            if name in provided_checks:
                supplied = _coerce_health_check(name, provided_checks.get(name))
                if not supplied.passed:
                    check = RuntimePreflightCheck(
                        name,
                        False,
                        f"{check.detail}; supplied operator check failed: {supplied.detail}",
                    )
            checks.append(check)
        else:
            checks.append(_coerce_health_check(name, provided_checks.get(name)))
    return RuntimePreflightResult(mode_name, trade_date, tuple(checks))


def default_session_schedule(mode: str, strategy_ids: Sequence[str] = ("KALCB", "OLR")) -> tuple[RuntimeScheduleStep, ...]:
    mode_name = _normalize_mode(mode)
    sids = {_normalize_strategy_id(strategy_id) for strategy_id in strategy_ids}
    artifact_steps: list[RuntimeScheduleStep] = []
    if "KALCB" in sids:
        artifact_steps.append(RuntimeScheduleStep("kalcb_daily_artifact", "08:40"))
    if "OLR" in sids:
        artifact_steps.append(RuntimeScheduleStep("olr_stage1_artifact", "08:40"))
        if mode_name == "artifact_only":
            artifact_steps.append(RuntimeScheduleStep("olr_final_artifact", "14:35"))
    artifact_steps.append(
        RuntimeScheduleStep(
            "artifact_readiness",
            "14:36" if any(step.name == "olr_final_artifact" for step in artifact_steps) else "08:45",
        )
    )
    if mode_name in ARTIFACT_ONLY_MODES:
        return tuple(artifact_steps)
    steps = [
        *artifact_steps,
        RuntimeScheduleStep("kis_resource_plan", "08:45"),
    ]
    if "KALCB" in sids:
        steps.extend(
            (
                RuntimeScheduleStep("kalcb_runtime_preflight", "08:50"),
                RuntimeScheduleStep("kalcb_runtime_start", "09:00"),
            )
        )
    if "OLR" in sids:
        steps.extend(
            (
                RuntimeScheduleStep("olr_stage1_bar_acquisition_check", "14:20"),
                RuntimeScheduleStep("olr_final_artifact", "14:35"),
                RuntimeScheduleStep("olr_final_readiness", "14:36"),
                RuntimeScheduleStep("olr_runtime_enable", "14:36"),
            )
        )
    steps.extend(
        (
            RuntimeScheduleStep("session_capture_closeout", "15:35"),
            RuntimeScheduleStep("paper_replay_refresh", "16:10", required=mode_name in {"paper", "live"}),
        )
    )
    return tuple(steps)


def _validate_strategy_artifacts_for_runtime(
    strategy_ids: Sequence[str],
    *,
    trade_date: date,
    mode: str,
    artifact_roots: dict[str, str | Path] | None,
) -> tuple[dict[str, KALCBDailySnapshot | OLRDailySnapshot], list[ArtifactReadinessFailure]]:
    mode_name = _normalize_mode(mode)
    sids = tuple(_normalize_strategy_id(strategy_id) for strategy_id in strategy_ids)
    if mode_name not in EXECUTION_MODES or not {"KALCB", "OLR"}.issubset(set(sids)):
        return validate_strategy_artifacts(sids, trade_date=trade_date, mode=mode_name, artifact_roots=artifact_roots)
    artifacts: dict[str, KALCBDailySnapshot | OLRDailySnapshot] = {}
    failures: list[ArtifactReadinessFailure] = []
    for sid in sids:
        if sid == "KALCB":
            stage = KALCB_FINAL_ARTIFACT_STAGE
            try:
                artifacts[sid] = load_strategy_artifact(sid, trade_date, stage, mode_name, artifact_roots=artifact_roots)
            except Exception as exc:
                failures.append(ArtifactReadinessFailure(sid, stage, str(exc)))
        elif sid == "OLR":
            try:
                artifacts[sid] = load_strategy_artifact(
                    sid,
                    trade_date,
                    OLR_FINAL_ARTIFACT_STAGE,
                    mode_name,
                    artifact_roots=artifact_roots,
                )
            except Exception:
                try:
                    artifacts[sid] = load_strategy_artifact(
                        sid,
                        trade_date,
                        OLR_STAGE1_ARTIFACT_STAGE,
                        "artifact_only_stage1",
                        artifact_roots=artifact_roots,
                    )
                except Exception as stage1_exc:
                    failures.append(
                        ArtifactReadinessFailure(
                            sid,
                            OLR_STAGE1_ARTIFACT_STAGE,
                            f"OLR final artifact is unavailable and stage1 readiness failed: {stage1_exc}",
                        )
                    )
        else:
            stage = required_stage_for(sid, mode_name)
            try:
                artifacts[sid] = load_strategy_artifact(sid, trade_date, stage, mode_name, artifact_roots=artifact_roots)
            except Exception as exc:
                failures.append(ArtifactReadinessFailure(sid, stage, str(exc)))
    return artifacts, failures


def _olr_final_required_for_runtime(
    strategy_ids: Sequence[str],
    mode: str,
    artifacts: Mapping[str, KALCBDailySnapshot | OLRDailySnapshot],
) -> bool:
    sids = {_normalize_strategy_id(strategy_id) for strategy_id in strategy_ids}
    if mode not in EXECUTION_MODES or "OLR" not in sids:
        return True
    snapshot = artifacts.get("OLR")
    if snapshot is None:
        return True
    stage = str((snapshot.metadata or {}).get("artifact_stage") or "")
    return stage == OLR_FINAL_ARTIFACT_STAGE or "KALCB" not in sids


def _artifact_is_executable_for_runtime(strategy_id: str, snapshot: KALCBDailySnapshot | OLRDailySnapshot) -> bool:
    sid = _normalize_strategy_id(strategy_id)
    stage = str((snapshot.metadata or {}).get("artifact_stage") or "")
    if sid == "OLR":
        return stage == OLR_FINAL_ARTIFACT_STAGE
    return True


def _normalize_mode(mode: str) -> str:
    mode_name = str(mode or "").strip().lower()
    if mode_name not in RUNTIME_MODES:
        raise ValueError(f"unsupported runtime mode {mode!r}")
    return mode_name


def _normalize_strategy_id(strategy_id: str) -> str:
    sid = str(strategy_id or "").strip().upper()
    if not sid:
        raise ValueError("strategy_id cannot be blank")
    return sid


def _coerce_health_check(name: str, raw: Any) -> RuntimePreflightCheck:
    if isinstance(raw, Mapping):
        passed = bool(raw.get("passed"))
        detail = str(raw.get("detail") or ("ok" if passed else "failed"))
        return RuntimePreflightCheck(name, passed, detail)
    if isinstance(raw, tuple):
        passed = bool(raw[0]) if raw else False
        detail = str(raw[1]) if len(raw) > 1 else ("ok" if passed else "failed")
        return RuntimePreflightCheck(name, passed, detail)
    if raw is None:
        return RuntimePreflightCheck(name, False, "missing required preflight input")
    passed = bool(raw)
    return RuntimePreflightCheck(name, passed, "ok" if passed else "failed")


def _derive_oms_hardening_health_checks(provided_checks: Mapping[str, Any]) -> dict[str, RuntimePreflightCheck]:
    payload, source = _extract_oms_health_payload(provided_checks)
    if payload is None:
        detail = "missing raw OMS /health payload for hardening readiness evidence"
        if provided_checks.get("oms_health_payload_error"):
            detail = f"{detail}: {provided_checks['oms_health_payload_error']}"
        return {
            name: RuntimePreflightCheck(name, False, detail)
            for name in OMS_HARDENING_HEALTH_CHECKS
        }

    status = str(payload.get("status") or "").lower().strip()
    stop_status = str(payload.get("stop_protection_status") or "").lower().strip()
    idempotency_status = str(
        payload.get("idempotency_status")
        or payload.get("idempotency_health")
        or payload.get("reservation_reconcile_status")
        or ""
    ).lower().strip()
    missing_stop_fields = [
        field
        for field in _REQUIRED_STOP_HEALTH_FIELDS
        if field not in payload or payload.get(field) is None
    ]
    stop_counts = {
        field: _required_nonnegative_int(payload, field)
        for field in _REQUIRED_STOP_HEALTH_FIELDS
        if field not in missing_stop_fields
    }
    invalid_stop_fields = [field for field, value in stop_counts.items() if value is None]
    unprotected_count = stop_counts.get("unprotected_positions_count")
    active_stop_count = stop_counts.get("active_stop_count")
    stale_price_count = stop_counts.get("stop_watcher_price_stale_count")
    watcher_age = None
    watcher_age_missing = False
    watcher_age_stale = False
    if active_stop_count is not None and active_stop_count > 0:
        watcher_age = _required_nonnegative_float(payload, "stop_watcher_last_check_age_sec")
        watcher_age_missing = watcher_age is None
        watcher_age_stale = watcher_age is not None and watcher_age > _MAX_STOP_WATCHER_AGE_SEC

    oms_ok = bool(status in _ACCEPTABLE_OMS_STATUS)
    stop_ok = bool(
        not missing_stop_fields
        and not invalid_stop_fields
        and stop_status in _READY_HEALTH_STATES
        and unprotected_count == 0
        and stale_price_count == 0
        and not watcher_age_missing
        and not watcher_age_stale
    )
    idempotency_ok = bool(idempotency_status in _READY_HEALTH_STATES)
    stop_failure_detail = _stop_health_failure_detail(
        source=source,
        stop_status=stop_status,
        missing_fields=missing_stop_fields,
        invalid_fields=invalid_stop_fields,
        unprotected_count=unprotected_count,
        active_stop_count=active_stop_count,
        stale_price_count=stale_price_count,
        watcher_age=watcher_age,
        watcher_age_missing=watcher_age_missing,
        watcher_age_stale=watcher_age_stale,
    )

    return {
        "oms_health_ok": RuntimePreflightCheck(
            "oms_health_ok",
            oms_ok,
            (
                f"derived from OMS /health evidence '{source}': status={status or 'missing'}"
                if oms_ok
                else f"OMS /health status is {status or 'missing'} in evidence '{source}'"
            ),
        ),
        "durable_stops_ok": RuntimePreflightCheck(
            "durable_stops_ok",
            stop_ok,
            (
                f"derived from OMS /health evidence '{source}': stop_protection_status={stop_status}, "
                f"unprotected_positions_count={unprotected_count}, active_stop_count={active_stop_count}, "
                f"stop_watcher_last_check_age_sec={watcher_age}"
                if stop_ok
                else stop_failure_detail
            ),
        ),
        "idempotency_reservation_ok": RuntimePreflightCheck(
            "idempotency_reservation_ok",
            idempotency_ok,
            (
                f"derived from OMS /health evidence '{source}': idempotency_status={idempotency_status}"
                if idempotency_ok
                else f"OMS idempotency health is {idempotency_status or 'missing'} in evidence '{source}'"
            ),
        ),
    }


def _extract_oms_health_payload(provided_checks: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    for key in OMS_HEALTH_PAYLOAD_KEYS:
        raw = provided_checks.get(key)
        payload = _unwrap_oms_health_payload(raw)
        if payload is not None:
            return payload, key
    return None, ""


def _unwrap_oms_health_payload(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    for nested_key in ("payload", "raw_oms_health", "oms_health_payload", "health", "body"):
        nested = raw.get(nested_key)
        if isinstance(nested, Mapping):
            return dict(nested)
    if any(
        key in raw
        for key in (
            "status",
            "stop_protection_status",
            "idempotency_status",
            "idempotency_health",
            "reservation_reconcile_status",
        )
    ):
        return dict(raw)
    return None


def _required_nonnegative_int(payload: Mapping[str, Any], field: str) -> int | None:
    if field not in payload or payload.get(field) is None:
        return None
    try:
        value = int(payload[field])
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _required_nonnegative_float(payload: Mapping[str, Any], field: str) -> float | None:
    if field not in payload or payload.get(field) is None:
        return None
    try:
        value = float(payload[field])
    except (TypeError, ValueError):
        return None
    return value if value >= 0.0 else None


def _stop_health_failure_detail(
    *,
    source: str,
    stop_status: str,
    missing_fields: Sequence[str],
    invalid_fields: Sequence[str],
    unprotected_count: int | None,
    active_stop_count: int | None,
    stale_price_count: int | None,
    watcher_age: float | None,
    watcher_age_missing: bool,
    watcher_age_stale: bool,
) -> str:
    reasons: list[str] = []
    if stop_status not in _READY_HEALTH_STATES:
        reasons.append(f"stop_protection_status={stop_status or 'missing'}")
    if missing_fields:
        reasons.append(f"missing fields={','.join(missing_fields)}")
    if invalid_fields:
        reasons.append(f"invalid fields={','.join(invalid_fields)}")
    if unprotected_count not in {0, None}:
        reasons.append(f"unprotected_positions_count={unprotected_count}")
    if stale_price_count not in {0, None}:
        reasons.append(f"stop_watcher_price_stale_count={stale_price_count}")
    if watcher_age_missing:
        reasons.append("active stops require stop_watcher_last_check_age_sec")
    if watcher_age_stale:
        reasons.append(f"stop_watcher_last_check_age_sec={watcher_age}")
    if not reasons:
        reasons.append("durable stop health is incomplete")
    return f"OMS durable stop health failed in evidence '{source}': {'; '.join(reasons)}"


def _build_runtime_kis_resource_plan(
    strategy_ids: Sequence[str],
    *,
    trade_date: date,
    mode: str,
    completed_bar_source: str = "paced_rest",
    artifacts: Mapping[str, KALCBDailySnapshot | OLRDailySnapshot],
    configs: Mapping[str, KALCBConfig | OLRConfig],
    artifact_roots: Mapping[str, str | Path] | None,
) -> KISResourcePlan:
    roots = {**DEFAULT_ARTIFACT_ROOTS, **{key.upper(): Path(value) for key, value in dict(artifact_roots or {}).items()}}
    sids = {_normalize_strategy_id(strategy_id) for strategy_id in strategy_ids}
    kalcb_snapshot = artifacts.get("KALCB") if "KALCB" in sids else None
    olr_artifact = artifacts.get("OLR") if "OLR" in sids else None
    olr_stage1: OLRDailySnapshot | None = None
    olr_final: OLRDailySnapshot | None = None
    if "OLR" in sids:
        if olr_artifact is not None and str((olr_artifact.metadata or {}).get("artifact_stage") or "") == OLR_STAGE1_ARTIFACT_STAGE:
            olr_stage1 = olr_artifact
        elif olr_artifact is not None and str((olr_artifact.metadata or {}).get("artifact_stage") or "") == OLR_FINAL_ARTIFACT_STAGE:
            olr_final = olr_artifact
        if olr_stage1 is None:
            try:
                olr_stage1 = OLRArtifactStore(roots["OLR"]).load_snapshot(trade_date, artifact_stage=OLR_STAGE1_ARTIFACT_STAGE)
            except Exception:
                olr_stage1 = None
    return build_kis_resource_plan(
        trade_date=trade_date,
        mode=mode,
        kalcb_config=configs.get("KALCB") if isinstance(configs.get("KALCB"), KALCBConfig) else None,
        olr_config=configs.get("OLR") if isinstance(configs.get("OLR"), OLRConfig) else None,
        kalcb_snapshot=kalcb_snapshot if isinstance(kalcb_snapshot, KALCBDailySnapshot) else None,
        olr_stage1_snapshot=olr_stage1,
        olr_final_snapshot=olr_final,
        completed_bar_source=completed_bar_source,
    )


def _held_or_pending_symbols(drivers: Mapping[str, RuntimeSessionDriver]) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for strategy_id, driver in drivers.items():
        state = getattr(driver.descriptor.engine, "state", None)
        symbols: set[str] = set()
        for symbol, symbol_state in dict(getattr(state, "symbols", {}) or {}).items():
            if getattr(symbol_state, "position", None) is not None:
                symbols.add(str(symbol).zfill(6))
                continue
            for attr in ("pending_entry_order_id", "pending_exit_order_id"):
                if str(getattr(symbol_state, attr, "") or ""):
                    symbols.add(str(symbol).zfill(6))
                    break
        result[str(strategy_id).upper().strip()] = tuple(sorted(symbols))
    return result


def _load_olr_final_enablement_snapshot(
    trade_date: date,
    *,
    mode: str,
    snapshot: OLRDailySnapshot | None,
    artifact_root: str | Path | None,
    artifact_path: str | Path | None,
) -> tuple[OLRDailySnapshot, Path]:
    if artifact_root is None and artifact_path is None:
        raise ValueError("artifact_root or artifact_path is required to enable OLR final execution")
    root = Path(artifact_root) if artifact_root is not None else _olr_artifact_root_from_path(Path(artifact_path))
    path = OLRArtifactStore(root).path_for(trade_date, artifact_stage=OLR_FINAL_ARTIFACT_STAGE)
    if artifact_path is not None and Path(artifact_path).resolve(strict=False) != path.resolve(strict=False):
        raise ValueError("OLR final artifact_path must be the canonical OLR final artifact-store path")
    loaded = load_strategy_artifact(
        "OLR",
        trade_date,
        OLR_FINAL_ARTIFACT_STAGE,
        mode,
        artifact_roots={"OLR": root},
    )
    if snapshot is not None and loaded.artifact_hash != snapshot.artifact_hash:
        raise ValueError("OLR final snapshot does not match readiness-validated artifact evidence")
    return loaded, path


def _olr_artifact_root_from_path(path: Path) -> Path:
    return path.parent.parent if path.parent.name == "final" else path.parent


def _runtime_config_for(plan: RuntimeSessionPlan, strategy_id: str) -> KALCBConfig | OLRConfig:
    sid = _normalize_strategy_id(strategy_id)
    summary = dict((plan.strategy_config_summaries or {}).get(sid) or {})
    payload = dict(summary.get("payload") or {})
    if payload and sid == "KALCB":
        return KALCBConfig.from_mapping(payload)
    if payload and sid == "OLR":
        return OLRConfig.from_mapping(payload)
    cfg = dict(plan.strategy_configs or {}).get(sid)
    if isinstance(cfg, (KALCBConfig, OLRConfig)):
        return cfg
    if not payload:
        raise RuntimeError(f"{sid} approved runtime config payload is unavailable")
    raise ValueError(f"unsupported strategy_id={strategy_id!r}")


def _shared_portfolio_context(drivers: Mapping[str, RuntimeSessionDriver]) -> PortfolioContextProvider:
    contexts = {id(driver.portfolio_context): driver.portfolio_context for driver in drivers.values()}
    if not contexts:
        raise RuntimeError("OLR final enablement requires an existing runtime driver and portfolio context")
    if len(contexts) != 1:
        raise RuntimeError("OLR final enablement requires one shared PortfolioContextProvider")
    return next(iter(contexts.values()))


def _resource_artifact_roots_for_enablement(
    final_path: Path,
    artifact_root: str | Path | None,
) -> dict[str, Path]:
    if artifact_root is not None:
        return {"OLR": Path(artifact_root)}
    root = final_path.parent.parent if final_path.parent.name == "final" else final_path.parent
    return {"OLR": root}


def _rewrite_runtime_manifest_for_enablement(
    plan: RuntimeSessionPlan,
    *,
    staged_row: Mapping[str, Any],
    resource_plan: KISResourcePlan,
    resource_plan_path: str,
) -> None:
    recorder = plan.session_recorder
    if recorder is None:
        return
    manifest_path = recorder.paths.manifest
    existing = json.loads(manifest_path.read_text(encoding="utf-8") or "{}") if manifest_path.is_file() else {}
    staged = list(existing.get("staged_artifacts") or ())
    staged.append(dict(staged_row))
    strategy_ids = list(dict.fromkeys([*(existing.get("strategy_ids") or []), "OLR"]))
    recorder.write_manifest(
        {
            **existing,
            "strategy_ids": strategy_ids,
            "staged_artifacts": staged,
            "kis_resource_plan_hash": resource_plan.plan_hash,
            "kis_resource_plan_path": resource_plan_path,
            "olr_final_enabled_at": datetime.now(timezone.utc).isoformat(),
        }
    )


def _emit_runtime_deployment_metadata(plan: RuntimeSessionPlan) -> None:
    if not plan.deployment_metadata_path or plan.mode not in {"paper", "live"} or not plan.ready_to_start:
        return
    strategy_ids = _deployment_metadata_strategy_ids(plan)
    try:
        emit_deployment_metadata(
            plan.deployment_metadata_path,
            repo_root=_repo_root(),
            contract_path=plan.deployment_metadata_contract_path or None,
            mode=plan.mode,
            strategy_ids=strategy_ids,
            strategy_configs=_filter_mapping(plan.strategy_config_summaries or {}, strategy_ids),
            portfolio_policy_config=plan.portfolio_policy_config or {},
            strategy_artifacts=_filter_mapping(plan.artifacts, strategy_ids),
            initial_positions=_runtime_manifest_value(plan.session_recorder, "initial_positions", {}),
            kis_resource_plan_hash=plan.kis_resource_plan.plan_hash if plan.kis_resource_plan is not None else "",
            deployment_id=_runtime_deployment_id(plan),
            runtime_started_at_utc=_runtime_manifest_timestamp(plan.session_recorder),
            runtime_entrypoint=plan.runtime_entrypoint or "deployment.olr_kalcb.runtime:prepare_runtime_session",
            runtime_instance_id=_runtime_instance_id(plan),
            emission_environment=plan.deployment_metadata_environment or "",
        )
    except Exception as exc:
        _record_deployment_metadata_error(plan, exc)


def _runtime_deployment_id(plan: RuntimeSessionPlan) -> str:
    exporter = getattr(plan.session_recorder, "assistant_exporter", None)
    lineage = getattr(exporter, "current_lineage", None) or getattr(exporter, "base_lineage", None)
    return str(getattr(lineage, "deployment_id", "") or "")


def _runtime_assistant_lineage() -> Any | None:
    if LineageContext is None:
        return None
    try:
        code_sha = get_code_sha(_repo_root()) if callable(get_code_sha) else ""
        return LineageContext(code_sha=code_sha)
    except Exception:
        return None


def _runtime_instance_id(plan: RuntimeSessionPlan) -> str:
    recorder = plan.session_recorder
    if recorder is None:
        return ""
    try:
        return f"runtime:{canonical_json_hash({'session_root': str(recorder.paths.root), 'started_at': _runtime_manifest_timestamp(recorder)})[:16]}"
    except Exception:
        return ""


def _record_deployment_metadata_error(plan: RuntimeSessionPlan, exc: Exception) -> None:
    exporter = getattr(plan.session_recorder, "assistant_exporter", None)
    writer = getattr(exporter, "writer", None)
    if writer is None:
        return
    try:
        writer.write(
            "bot_error",
            {
                "record_type": "deployment_metadata_error",
                "component": "deployment_metadata",
                "severity": "warning",
                "mode": plan.mode,
                "deployment_metadata_path": plan.deployment_metadata_path or "",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            payload_key=f"deployment_metadata:{plan.deployment_metadata_path or ''}:{type(exc).__name__}",
            exchange_timestamp=datetime.now(timezone.utc),
            lineage=getattr(exporter, "current_lineage", None),
            scope="portfolio",
        )
    except Exception:
        return


def _deployment_metadata_strategy_ids(plan: RuntimeSessionPlan) -> tuple[str, ...]:
    active = tuple(str(strategy_id).upper().strip() for strategy_id in (plan.descriptors or {}) if str(strategy_id).strip())
    if active:
        return active
    return tuple(str(strategy_id).upper().strip() for strategy_id in (plan.strategy_config_summaries or plan.artifacts) if str(strategy_id).strip())


def _filter_mapping(mapping: Mapping[str, Any], strategy_ids: Sequence[str]) -> dict[str, Any]:
    wanted = {str(strategy_id).upper().strip() for strategy_id in strategy_ids}
    return {
        str(key).upper().strip(): value
        for key, value in dict(mapping or {}).items()
        if str(key).upper().strip() in wanted
    }


def _runtime_risk_config_payload() -> tuple[dict[str, Any], str]:
    try:
        from oms.config_loader import load_effective_risk_config_payload

        payload, source = load_effective_risk_config_payload()
        return payload, str(source or "")
    except Exception:
        try:
            from oms.risk import RiskConfig

            return asdict(RiskConfig()), ""
        except Exception:
            return {}, ""


def _runtime_manifest_timestamp(recorder: PaperSessionRecorder | None) -> str:
    generated_at = _runtime_manifest_value(recorder, "generated_at", "")
    if generated_at:
        return str(generated_at)
    return datetime.now(timezone.utc).isoformat()


def _runtime_manifest_value(recorder: PaperSessionRecorder | None, key: str, default: Any = None) -> Any:
    if recorder is None:
        return default
    manifest_path = recorder.paths.manifest
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8") or "{}")
            return manifest.get(key, default)
        except (OSError, json.JSONDecodeError):
            pass
    return default


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _record_resource_route_suppression(
    recorder: PaperSessionRecorder | None,
    plan: KISResourcePlan | None,
    bar: Any,
    *,
    requested_targets: Sequence[str] = (),
    reason_code: str,
) -> None:
    if recorder is None or plan is None:
        return
    recorder.append_jsonl(
        "subscription_events.jsonl",
        {
            "record_type": "market_data_route",
            "event_time": datetime.now(timezone.utc).isoformat(),
            "strategy_id": "NONE",
            "lease_name": "resource_plan_bar_route",
            "symbol": str(getattr(bar, "symbol", "") or "").zfill(6),
            "action": "suppressed",
            "registration_type": "completed_bar",
            "ws_used_before": 0,
            "ws_used_after": 0,
            "ws_budget": plan.limit_profile.ws_max_registrations,
            "kis_resource_plan_hash": plan.plan_hash,
            "reason_code": reason_code,
            "requested_strategy_ids": list(requested_targets),
        },
    )


def _runtime_oms_client(
    mode: str,
    *,
    oms_client: Any | None,
    dry_run_oms_client: Any | None,
    session_recorder: PaperSessionRecorder | None,
    initial_account_state: Any | None = None,
    initial_positions: Any | None = None,
) -> Any | None:
    if mode == "dry_run":
        if dry_run_oms_client is not None:
            return dry_run_oms_client
        if session_recorder is not None:
            kwargs: dict[str, Any] = {}
            if initial_account_state is not None:
                kwargs["account_state"] = _coerce_account(initial_account_state)
            if initial_positions is not None:
                kwargs["positions"] = _coerce_positions(initial_positions)
            return RecordingOMSClient(session_recorder, **kwargs)
        return None
    return oms_client


def _oms_client_available(client: Any | None) -> bool:
    return callable(getattr(client, "submit_intent", None))


def _recording_oms_available(client: Any | None) -> bool:
    return _oms_client_available(client) and bool(getattr(client, "record_only", False))


def _execution_oms_available(mode: str, client: Any | None) -> bool:
    return _recording_oms_available(client) if mode == "dry_run" else _oms_client_available(client)


def _closeout_capable(recorder: Any | None) -> bool:
    return callable(getattr(recorder, "close_session", None))


def _deployment_metadata_enabled(path: str | Path | None) -> bool:
    return str(path or "").strip().lower() not in {"", "off", "none", "disabled"}


def _artifact_failure_detail(failures: Sequence[ArtifactReadinessFailure]) -> str:
    return "; ".join(f"{failure.strategy_id}:{failure.stage}:{failure.detail}" for failure in failures)


def _resource_plan_failure_detail(failures: Sequence[str]) -> str:
    return "; ".join(failures) if failures else "KIS resource plan unavailable"


def _with_runtime_driver_check(
    preflight: RuntimePreflightResult,
    mode: str,
    runtime_driver_available: bool,
) -> RuntimePreflightResult:
    if mode not in EXECUTION_MODES:
        return preflight
    checks = [
        check
        for check in preflight.checks
        if check.name != "runtime_driver_available"
    ]
    checks.append(
        RuntimePreflightCheck(
            "runtime_driver_available",
            runtime_driver_available,
            "ok" if runtime_driver_available else f"{mode} mode requires RuntimeSessionDriver execution",
        )
    )
    return RuntimePreflightResult(preflight.mode, preflight.trade_date, tuple(checks))


def _load_runtime_strategy_configs(
    strategy_ids: Sequence[str],
    *,
    mode: str,
    source: str | Path | Mapping[str, Any] | None,
) -> tuple[dict[str, KALCBConfig | OLRConfig], dict[str, dict[str, Any]]]:
    sids = tuple(_normalize_strategy_id(strategy_id) for strategy_id in strategy_ids)
    if mode not in EXECUTION_MODES:
        return {}, {}
    manifest = _load_strategy_config_manifest(source)
    configs: dict[str, KALCBConfig | OLRConfig] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for sid in sids:
        record = _strategy_config_record(manifest, sid)
        if record is None:
            raise ValueError(f"missing approved optimized config for {sid}")
        else:
            source_path = str(record.get("path") or "")
            if not source_path:
                raise ValueError(f"approved optimized config record for {sid} is missing path")
            path = Path(source_path)
            if not path.is_absolute():
                path = Path.cwd() / path
            optimized = json.loads(path.read_text(encoding="utf-8") or "{}")
            payload = _config_mutation_payload(optimized)
            if not payload:
                raise ValueError(f"approved optimized config for {sid} has no mutations")
            actual_sha = file_sha256(path)
            source_sha = str(record.get("sha256") or actual_sha)
            if source_sha and source_sha != actual_sha:
                raise ValueError(f"approved optimized config SHA mismatch for {sid}: {path}")
            source_label = str(record.get("label") or "")
            source_manifest = str(manifest.get("_manifest_path") or "")
            source_path = str(path)
        cfg: KALCBConfig | OLRConfig
        if sid == "KALCB":
            cfg = KALCBConfig.from_mapping(payload)
        elif sid == "OLR":
            cfg = OLRConfig.from_mapping(payload)
        else:
            continue
        payload_hash = canonical_json_hash(payload)
        configs[sid] = cfg
        summaries[sid] = {
            "source_label": source_label,
            "source_path": source_path,
            "source_manifest": source_manifest,
            "source_sha256": source_sha,
            "payload": payload,
            "payload_hash": payload_hash,
            "mutation_hash": payload_hash,
            "hydrated_config_hash": canonical_json_hash(asdict(cfg)),
            "uses_defaults": not bool(payload),
        }
    return configs, summaries


def _approved_runtime_sector_map(
    strategy_ids: Sequence[str],
    mode: str,
    explicit: Mapping[str, str] | None,
) -> tuple[dict[str, str], tuple[ArtifactReadinessFailure, ...]]:
    mode_name = _normalize_mode(mode)
    sids = tuple(_normalize_strategy_id(strategy_id) for strategy_id in strategy_ids)
    if mode_name in EXECUTION_MODES and "KALCB" in sids and explicit is None:
        return {}, (
            ArtifactReadinessFailure(
                "KALCB",
                "approved_runtime_config",
                "KALCB execution requires the approved full sector_map; candidate-derived sector maps are non-promotional",
            ),
        )
    return _canonical_sector_map(explicit or {}), ()


def _approved_config_fingerprints(
    configs: Mapping[str, KALCBConfig | OLRConfig],
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    sector_map: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    fingerprints: dict[str, dict[str, Any]] = {}
    for sid, cfg in configs.items():
        summary = dict(summaries.get(sid) or {})
        payload = dict(summary.get("payload") or {})
        if sid == "KALCB":
            artifact_config_hash = kalcb_candidate_config_fingerprint(cfg, payload, _canonical_sector_map(sector_map))
            artifact_config_hash_kind = "kalcb_candidate_config_hash"
        elif sid == "OLR":
            artifact_config_hash = olr_final_candidate_config_fingerprint(cfg)
            artifact_config_hash_kind = "olr_final_candidate_config_hash"
        else:
            continue
        sector_fingerprint: dict[str, Any] = {}
        if sid == "KALCB":
            canonical_sector_map = _canonical_sector_map(sector_map)
            sector_fingerprint = {
                "sector_map_hash": canonical_json_hash(canonical_sector_map),
                "sector_map_size": len(canonical_sector_map),
                "sector_map_fingerprint_version": "kalcb-full-sector-map-v1",
            }
        fingerprints[sid] = {
            "strategy_id": sid,
            "optimized_config_path": str(summary.get("source_path") or ""),
            "optimized_config_sha256": str(summary.get("source_sha256") or ""),
            "mutation_payload_hash": str(summary.get("payload_hash") or ""),
            "hydrated_config_hash": str(summary.get("hydrated_config_hash") or ""),
            "artifact_config_hash": artifact_config_hash,
            "artifact_config_hash_kind": artifact_config_hash_kind,
            "fingerprint_version": "approved-optimized-config-v1",
            **sector_fingerprint,
        }
    return fingerprints


def _canonical_sector_map(sector_map: Mapping[str, str]) -> dict[str, str]:
    normalized = {
        str(symbol).zfill(6): str(sector or "").upper().strip()
        for symbol, sector in dict(sector_map or {}).items()
        if str(sector or "").strip()
    }
    return {
        symbol: normalized[symbol]
        for symbol in sorted(normalized)
    }


def _artifact_config_binding_failures(
    artifacts: Mapping[str, KALCBDailySnapshot | OLRDailySnapshot],
    approved_config_fingerprints: Mapping[str, Mapping[str, Any]],
    *,
    mode: str,
) -> list[ArtifactReadinessFailure]:
    if mode not in EXECUTION_MODES:
        return []
    failures: list[ArtifactReadinessFailure] = []
    for sid, snapshot in artifacts.items():
        fingerprint = dict(approved_config_fingerprints.get(sid) or {})
        expected = str(fingerprint.get("artifact_config_hash") or "")
        stage = str((snapshot.metadata or {}).get("artifact_stage") or "")
        if not expected:
            failures.append(ArtifactReadinessFailure(sid, stage, "approved config fingerprint is missing"))
            continue
        metadata = dict(snapshot.metadata or {})
        if sid == "KALCB":
            actual = str(metadata.get("candidate_config_hash") or "")
            if actual != expected:
                failures.append(
                    ArtifactReadinessFailure(
                        sid,
                        stage,
                        f"artifact candidate_config_hash does not match approved runtime config: {actual!r} != {expected!r}",
                    )
                )
            expected_sector_hash = str(fingerprint.get("sector_map_hash") or "")
            actual_sector_hash = str(metadata.get("sector_map_hash") or "")
            if actual_sector_hash != expected_sector_hash:
                failures.append(
                    ArtifactReadinessFailure(
                        sid,
                        stage,
                        f"artifact sector_map_hash does not match approved runtime sector map: {actual_sector_hash!r} != {expected_sector_hash!r}",
                    )
                )
            expected_sector_size = int(fingerprint.get("sector_map_size") or 0)
            try:
                actual_sector_size = int(metadata.get("sector_map_size"))
            except (TypeError, ValueError):
                actual_sector_size = -1
            if actual_sector_size != expected_sector_size:
                failures.append(
                    ArtifactReadinessFailure(
                        sid,
                        stage,
                        f"artifact sector_map_size does not match approved runtime sector map: {actual_sector_size!r} != {expected_sector_size!r}",
                    )
                )
        elif sid == "OLR":
            if stage != OLR_FINAL_ARTIFACT_STAGE:
                continue
            actual = str(metadata.get("final_candidate_config_hash") or metadata.get("candidate_config_hash") or "")
            if actual != expected:
                failures.append(
                    ArtifactReadinessFailure(
                        sid,
                        stage,
                        f"artifact final_candidate_config_hash does not match approved runtime config: {actual!r} != {expected!r}",
                    )
                )
    return failures


def _config_mutation_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    candidate: Any = raw
    while isinstance(candidate, Mapping):
        if "mutations" in candidate and isinstance(candidate.get("mutations"), Mapping):
            candidate = candidate["mutations"]
            continue
        if "payload" in candidate and isinstance(candidate.get("payload"), Mapping):
            candidate = candidate["payload"]
            continue
        break
    payload = dict(candidate or {}) if isinstance(candidate, Mapping) else {}
    metadata_only = {
        "config_hash",
        "file",
        "hash",
        "hydrated_config_hash",
        "mutation_hash",
        "path",
        "payload_hash",
        "sha256",
        "source_label",
        "source_manifest",
        "source_path",
        "source_sha256",
        "uses_defaults",
    }
    return {} if payload and set(payload).issubset(metadata_only) else payload


def _load_strategy_config_manifest(source: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source) if source is not None else DEFAULT_STRATEGY_CONFIG_SOURCE
    payload = json.loads(path.read_text(encoding="utf-8") or "{}")
    payload["_manifest_path"] = str(path)
    return payload


def _strategy_config_record(manifest: Mapping[str, Any], strategy_id: str) -> dict[str, Any] | None:
    sid = _normalize_strategy_id(strategy_id)
    for row in manifest.get("artifacts") or ():
        item = dict(row or {})
        label = str(item.get("label") or "").lower()
        path = str(item.get("path") or "").lower()
        if sid == "KALCB" and "kalcb" in label and "optimized_config" in label:
            return item
        if sid == "OLR" and "olr" in label and "optimized_config" in label:
            return item
        filename = path.replace("\\", "/").rsplit("/", 1)[-1]
        if sid.lower() in filename and filename.endswith("optimized_config.json"):
            return item
    return None


def _merged_sector_map(
    artifacts: Mapping[str, KALCBDailySnapshot | OLRDailySnapshot],
    explicit: Mapping[str, str] | None,
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for snapshot in artifacts.values():
        for candidate in tuple(snapshot.candidates or ()):
            sector = str(getattr(candidate, "sector", "") or "").upper().strip()
            if sector:
                mapping[str(getattr(candidate, "symbol", "")).zfill(6)] = sector
    for symbol, sector in dict(explicit or {}).items():
        normalized = str(sector or "").upper().strip()
        if normalized:
            mapping[str(symbol).zfill(6)] = normalized
    return mapping


def _initial_runtime_state_payload(
    client: Any | None,
    *,
    initial_account_state: Any | None = None,
    initial_positions: Any | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    state_proxy = getattr(client, "state", None)
    account = initial_account_state
    if account is None:
        account = getattr(client, "account_state", None)
    if account is None and state_proxy is not None:
        account = getattr(state_proxy, "_cached_account", None)
    if account is not None:
        payload["initial_account_state"] = _json_value(account)
    positions = initial_positions
    if positions is None:
        positions = getattr(client, "positions", None)
    if positions is None and state_proxy is not None:
        positions = getattr(state_proxy, "_cached_positions", None)
    if positions is not None:
        payload["initial_positions"] = _json_value(positions)
    return payload


def _initial_runtime_state_available(payload: Mapping[str, Any]) -> bool:
    if "initial_account_state" not in payload or "initial_positions" not in payload:
        return False
    account = payload.get("initial_account_state")
    if not isinstance(account, Mapping):
        return False
    cash = _float_or_zero(account.get("buyable_cash", account.get("cash")))
    equity = _float_or_zero(account.get("equity", account.get("buyable_cash", account.get("cash"))))
    return cash > 0.0 and equity > 0.0


def _runtime_artifact_bundle_available(
    mode: str,
    strategy_ids: Sequence[str],
    *,
    trade_date: date,
    artifacts: Mapping[str, KALCBDailySnapshot | OLRDailySnapshot],
    artifact_roots: Mapping[str, str | Path] | None,
    session_recorder: PaperSessionRecorder | None,
    olr_final_required: bool = True,
) -> bool:
    if mode not in EXECUTION_MODES:
        return True
    if session_recorder is None:
        return False
    try:
        requirements = _runtime_artifact_requirements(
            strategy_ids,
            trade_date=trade_date,
            artifacts=artifacts,
            artifact_roots=artifact_roots,
            olr_final_required=olr_final_required,
        )
    except Exception:
        return False
    return bool(requirements) and all(requirement.snapshot is not None and requirement.path.is_file() for requirement in requirements)


def _stage_runtime_artifacts(
    session_recorder: PaperSessionRecorder,
    strategy_ids: Sequence[str],
    *,
    trade_date: date,
    artifacts: Mapping[str, KALCBDailySnapshot | OLRDailySnapshot],
    artifact_roots: Mapping[str, str | Path] | None,
    approved_config_fingerprints: Mapping[str, Mapping[str, Any]] | None = None,
    olr_final_required: bool = True,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for requirement in _runtime_artifact_requirements(
        strategy_ids,
        trade_date=trade_date,
        artifacts=artifacts,
        artifact_roots=artifact_roots,
        olr_final_required=olr_final_required,
    ):
        if requirement.snapshot is None:
            raise FileNotFoundError(f"{requirement.strategy_id} {requirement.stage} artifact is missing")
        rows.append(
            _stage_runtime_artifact_row(
                session_recorder,
                requirement,
                trade_date=trade_date,
                approved_config_fingerprints=approved_config_fingerprints,
            )
        )
    return tuple(rows)


def _stage_runtime_artifact_row(
    session_recorder: PaperSessionRecorder,
    requirement: _RuntimeArtifactRequirement,
    *,
    trade_date: date,
    approved_config_fingerprints: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if requirement.snapshot is None:
        raise FileNotFoundError(f"{requirement.strategy_id} {requirement.stage} artifact is missing")
    target = session_recorder.copy_snapshot(requirement.path, requirement.bucket)
    row = {
        "record_type": "artifact_generation",
        "strategy_id": requirement.strategy_id,
        "trade_date": trade_date.isoformat(),
        "stage": requirement.stage,
        "artifact_hash": requirement.snapshot.artifact_hash,
        "source_fingerprint": requirement.snapshot.source_fingerprint,
        "candidate_count": len(requirement.snapshot.candidates),
        "bucket": requirement.bucket,
        "source_path": str(requirement.path),
        "session_path": str(target),
        "approved_config_fingerprint": dict((approved_config_fingerprints or {}).get(requirement.strategy_id) or {}),
    }
    session_recorder.append_jsonl("artifact_generation.jsonl", row)
    return row


def _runtime_artifact_requirements(
    strategy_ids: Sequence[str],
    *,
    trade_date: date,
    artifacts: Mapping[str, KALCBDailySnapshot | OLRDailySnapshot],
    artifact_roots: Mapping[str, str | Path] | None,
    olr_final_required: bool = True,
) -> tuple[_RuntimeArtifactRequirement, ...]:
    roots = {**DEFAULT_ARTIFACT_ROOTS, **{key.upper(): Path(value) for key, value in dict(artifact_roots or {}).items()}}
    requirements: list[_RuntimeArtifactRequirement] = []
    sids = {_normalize_strategy_id(strategy_id) for strategy_id in strategy_ids}
    if "KALCB" in sids:
        store = KALCBArtifactStore(roots["KALCB"])
        kalcb = artifacts.get("KALCB")
        requirements.append(
            _RuntimeArtifactRequirement(
                "KALCB",
                str((kalcb.metadata or {}).get("artifact_stage") or "") if kalcb is not None else "",
                "daily_snapshots",
                store.path_for(trade_date),
                kalcb,
            )
        )
    if "OLR" in sids:
        store = OLRArtifactStore(roots["OLR"])
        stage1 = store.load_snapshot(trade_date, artifact_stage=OLR_STAGE1_ARTIFACT_STAGE)
        olr_artifact = artifacts.get("OLR")
        final = olr_artifact if olr_artifact is not None and str((olr_artifact.metadata or {}).get("artifact_stage") or "") == OLR_FINAL_ARTIFACT_STAGE else None
        requirements.append(
            _RuntimeArtifactRequirement(
                "OLR",
                OLR_STAGE1_ARTIFACT_STAGE,
                "olr_stage1_snapshots",
                store.path_for(trade_date, artifact_stage=OLR_STAGE1_ARTIFACT_STAGE),
                stage1,
            )
        )
        if final is not None or olr_final_required:
            requirements.append(
                _RuntimeArtifactRequirement(
                    "OLR",
                    OLR_FINAL_ARTIFACT_STAGE,
                    "olr_final_snapshots",
                    store.path_for(trade_date, artifact_stage=OLR_FINAL_ARTIFACT_STAGE),
                    final,
                )
            )
    return tuple(requirements)


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _json_value(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    return value


def _restore_strategy_state(strategy_id: str, raw: Any) -> Any:
    sid = _normalize_strategy_id(strategy_id)
    if raw.__class__.__name__ in {"KALCBState", "OLRState"}:
        return raw
    if not isinstance(raw, Mapping):
        raise TypeError(f"initial strategy state for {sid} must be a state object or mapping")
    if sid == "KALCB":
        from strategy_kalcb.core.serializers import restore_state

        return restore_state(dict(raw))
    if sid == "OLR":
        from strategy_olr.core.serializers import restore_state

        return restore_state(dict(raw))
    raise ValueError(f"unsupported strategy_id={strategy_id!r}")


def _artifact_summary(snapshot: KALCBDailySnapshot | OLRDailySnapshot) -> dict[str, Any]:
    return {
        "strategy_id": snapshot.strategy_id,
        "trade_date": snapshot.trade_date.isoformat(),
        "artifact_stage": str((snapshot.metadata or {}).get("artifact_stage") or ""),
        "artifact_hash": snapshot.artifact_hash,
        "source_fingerprint": snapshot.source_fingerprint,
        "candidate_count": len(snapshot.candidates),
    }
