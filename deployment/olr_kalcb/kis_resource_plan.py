from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from strategy_kalcb.config import KALCBConfig
from strategy_kalcb.models import KALCBDailySnapshot
from strategy_olr.artifact_store import OLR_FINAL_ARTIFACT_STAGE, OLR_STAGE1_ARTIFACT_STAGE
from strategy_olr.config import OLRConfig
from strategy_olr.models import OLRDailySnapshot

from .hashing import canonical_json_hash
from .kis_limits import KISLimitProfile, kis_mode_matches_runtime, limit_profile_for_runtime


KST = ZoneInfo("Asia/Seoul")
RESOURCE_PLAN_VERSION = "olr-kalcb-kis-resource-plan-v1"
RESOURCE_PLAN_FILENAME = "kis_resource_plan.json"
DEFAULT_COMPLETED_BAR_SOURCE = "paced_rest"
OLR_FINAL_CUTOFF = time(14, 30)
OLR_RUNTIME_ENABLE = time(14, 36)
CLOSE_WINDOW_START = time(15, 15)
CLOSE_WINDOW_END = time(15, 35)
OLR_REST_SAFETY_BUFFER_S = 60.0


@dataclass(frozen=True, slots=True)
class CandidateSurface:
    strategy_id: str
    artifact_stage: str
    candidate_count: int
    active_symbols: tuple[str, ...]
    frontier_symbols: tuple[str, ...]
    final_symbols: tuple[str, ...]
    orderable_symbols: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True, slots=True)
class ResourceLeaseWindow:
    name: str
    starts_at_kst: str
    ends_at_kst: str
    strategy_id: str
    rest_symbols: tuple[str, ...]
    rest_endpoint_class: str
    rest_call_count: int
    rest_calls_per_5m_budget: int
    ws_symbols: tuple[str, ...]
    ws_regs_per_symbol: int
    ws_reg_count: int
    ws_reg_budget: int
    order_reserve: int
    oms_reserve: int
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True, slots=True)
class KISResourcePlan:
    trade_date: date
    mode: str
    limit_profile: KISLimitProfile
    candidate_surfaces: tuple[CandidateSurface, ...]
    lease_windows: tuple[ResourceLeaseWindow, ...]
    passed: bool
    failures: tuple[str, ...]
    warnings: tuple[str, ...]
    plan_hash: str
    version: str = RESOURCE_PLAN_VERSION

    def to_json_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return target


def build_kis_resource_plan(
    *,
    trade_date: date,
    mode: str,
    kalcb_config: KALCBConfig | None = None,
    olr_config: OLRConfig | None = None,
    kalcb_snapshot: KALCBDailySnapshot | None = None,
    olr_stage1_snapshot: OLRDailySnapshot | None = None,
    olr_final_snapshot: OLRDailySnapshot | None = None,
    completed_bar_source: str = DEFAULT_COMPLETED_BAR_SOURCE,
    limit_profile: KISLimitProfile | None = None,
    kis_is_paper: bool | None = None,
) -> KISResourcePlan:
    mode_name = str(mode or "").strip().lower()
    profile = limit_profile or limit_profile_for_runtime(mode_name, kis_is_paper=kis_is_paper)
    failures: list[str] = []
    warnings: list[str] = []
    surfaces: list[CandidateSurface] = []

    if not kis_mode_matches_runtime(mode_name, profile):
        expected = "paper" if mode_name == "paper" else "live"
        actual = "paper" if profile.kis_is_paper else "live"
        failures.append(f"kis_mode_mismatch:{expected}_runtime_detected_{actual}_kis_mode")

    kalcb_cfg = kalcb_config
    if kalcb_snapshot is not None:
        if kalcb_cfg is None:
            kalcb_cfg = KALCBConfig()
            warnings.append("kalcb_config_missing_using_default_non_promotional_profile")
        kalcb_surface, surface_warnings = extract_kalcb_candidate_surface(kalcb_snapshot, kalcb_cfg)
        surfaces.append(kalcb_surface)
        warnings.extend(surface_warnings)
        failures.extend(_kalcb_surface_failures(kalcb_surface, kalcb_cfg))

    olr_cfg = olr_config
    if olr_stage1_snapshot is not None or olr_final_snapshot is not None:
        if olr_cfg is None:
            olr_cfg = OLRConfig()
            warnings.append("olr_config_missing_using_default_non_promotional_profile")
        if olr_stage1_snapshot is not None:
            stage1_surface, stage1_warnings = extract_olr_candidate_surface(olr_stage1_snapshot, olr_cfg)
            surfaces.append(stage1_surface)
            warnings.extend(stage1_warnings)
            failures.extend(_olr_surface_failures(stage1_surface, olr_cfg))
        if olr_final_snapshot is not None:
            final_surface, final_warnings = extract_olr_candidate_surface(olr_final_snapshot, olr_cfg)
            surfaces.append(final_surface)
            warnings.extend(final_warnings)
            failures.extend(_olr_surface_failures(final_surface, olr_cfg))

    lease_windows = _build_lease_windows(
        profile=profile,
        surfaces=surfaces,
        kalcb_config=kalcb_cfg,
        completed_bar_source=completed_bar_source,
    )
    failures.extend(_lease_failures(profile, lease_windows))
    failures.extend(
        _olr_acquisition_failures(
            profile,
            lease_windows,
            completed_bar_source=completed_bar_source,
        )
    )

    draft = KISResourcePlan(
        trade_date=trade_date,
        mode=mode_name,
        limit_profile=profile,
        candidate_surfaces=tuple(surfaces),
        lease_windows=tuple(lease_windows),
        passed=not failures,
        failures=tuple(dict.fromkeys(failures)),
        warnings=tuple(dict.fromkeys(warnings)),
        plan_hash="",
    )
    return replace(draft, plan_hash=resource_plan_hash(draft))


def extract_kalcb_candidate_surface(
    snapshot: KALCBDailySnapshot,
    config: KALCBConfig,
) -> tuple[CandidateSurface, tuple[str, ...]]:
    metadata = dict(snapshot.metadata or {})
    candidate_symbols = _candidate_symbols(snapshot.candidates)
    frontier_symbols = _metadata_symbols(metadata, "frontier_symbols")
    active_symbols = _metadata_symbols(metadata, "active_symbols")
    active_symbols_present = "active_symbols" in metadata
    frontier_symbols_present = "frontier_symbols" in metadata
    frontier_rest_budget_present = "frontier_rest_budget_symbols_per_5m" in metadata
    warnings: list[str] = []
    if not active_symbols_present:
        warnings.append("kalcb_active_symbols_missing_from_artifact_metadata")
    if not frontier_symbols_present:
        warnings.append("kalcb_frontier_symbols_missing_from_artifact_metadata")
        frontier_symbols = candidate_symbols
    if bool(config.entry_plan_frontier_branch_universe):
        orderable_symbols = frontier_symbols
    else:
        orderable_symbols = active_symbols
    if not bool(config.entry_plan_require_initial_active) and not bool(config.entry_plan_frontier_branch_universe):
        warnings.append("kalcb_entry_plan_require_initial_active_false_runtime_routing_restricts_to_active_symbols")
    return (
        CandidateSurface(
            strategy_id="KALCB",
            artifact_stage=str(metadata.get("artifact_stage") or ""),
            candidate_count=len(candidate_symbols),
            active_symbols=active_symbols,
            frontier_symbols=frontier_symbols,
            final_symbols=frontier_symbols,
            orderable_symbols=orderable_symbols,
            metadata={
                "artifact_hash": snapshot.artifact_hash,
                "source_fingerprint": snapshot.source_fingerprint,
                "active_symbol_count": len(active_symbols),
                "active_symbols_present": active_symbols_present,
                "active_budget_source": str(metadata.get("active_budget_source") or ""),
                "frontier_symbol_count": len(frontier_symbols),
                "frontier_symbols_present": frontier_symbols_present,
                "overflow_symbol_count": int(metadata.get("overflow_symbol_count") or 0),
                "ws_budget": int(config.ws_budget),
                "frontier_size": int(config.frontier_size),
                "entry_plan_frontier_branch_universe": bool(config.entry_plan_frontier_branch_universe),
                "entry_plan_require_initial_active": bool(config.entry_plan_require_initial_active),
                "frontier_rest_budget_symbols_per_5m": metadata.get("frontier_rest_budget_symbols_per_5m"),
                "expected_frontier_rest_budget_symbols_per_5m": _kalcb_frontier_rest_budget_symbols_per_5m(config),
                "frontier_rest_budget_symbols_per_5m_present": frontier_rest_budget_present,
                "reported_active_symbol_count": metadata.get("active_symbol_count"),
                "reported_frontier_symbol_count": metadata.get("frontier_symbol_count"),
            },
        ),
        tuple(warnings),
    )


def extract_olr_candidate_surface(
    snapshot: OLRDailySnapshot,
    config: OLRConfig,
) -> tuple[CandidateSurface, tuple[str, ...]]:
    metadata = dict(snapshot.metadata or {})
    stage = _normalize_olr_stage(str(metadata.get("artifact_stage") or ""))
    symbols = _candidate_symbols(snapshot.candidates)
    tradable_symbols = tuple(
        symbol
        for symbol, candidate in zip(symbols, snapshot.candidates)
        if bool(getattr(candidate, "tradable", True))
    )
    is_final = stage == OLR_FINAL_ARTIFACT_STAGE
    warnings: list[str] = []
    if int(config.premarket_frontier_size) != int(config.research_top_long_count):
        warnings.append("olr_premarket_frontier_size_nonoperative_stage1_uses_research_top_long_count")
    return (
        CandidateSurface(
            strategy_id="OLR",
            artifact_stage=stage,
            candidate_count=len(symbols),
            active_symbols=(),
            frontier_symbols=symbols if stage == OLR_STAGE1_ARTIFACT_STAGE else (),
            final_symbols=symbols if is_final else (),
            orderable_symbols=tradable_symbols[: max(0, int(config.overnight_slot_count))] if is_final else (),
            metadata={
                "artifact_hash": snapshot.artifact_hash,
                "source_fingerprint": snapshot.source_fingerprint,
                "selected_symbol_count": len(symbols),
                "research_top_long_count": int(config.research_top_long_count),
                "afternoon_top_n": int(config.afternoon_top_n),
                "overnight_slot_count": int(config.overnight_slot_count),
                "premarket_frontier_size": int(config.premarket_frontier_size),
                "premarket_frontier_size_status": "non_operative_selector_uses_research_top_long_count",
            },
        ),
        tuple(warnings),
    )


def resource_plan_hash(plan: KISResourcePlan | Mapping[str, Any]) -> str:
    payload = plan.to_json_dict() if isinstance(plan, KISResourcePlan) else _json_value(dict(plan))
    payload.pop("plan_hash", None)
    return canonical_json_hash(payload)


def write_kis_resource_plan(plan: KISResourcePlan, session_root: str | Path) -> Path:
    return plan.write_json(Path(session_root) / RESOURCE_PLAN_FILENAME)


def load_kis_resource_plan(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8") or "{}")


def candidate_surface_for(plan: KISResourcePlan, strategy_id: str, stage: str | None = None) -> CandidateSurface | None:
    sid = str(strategy_id or "").upper().strip()
    for surface in plan.candidate_surfaces:
        if surface.strategy_id != sid:
            continue
        if stage is None or surface.artifact_stage == stage:
            return surface
    return None


def active_symbols_for(plan: KISResourcePlan, strategy_id: str) -> tuple[str, ...]:
    sid = str(strategy_id or "").upper().strip()
    if sid == "KALCB":
        surface = candidate_surface_for(plan, "KALCB")
        return surface.active_symbols if surface is not None else ()
    if sid == "OLR":
        surface = candidate_surface_for(plan, "OLR", OLR_FINAL_ARTIFACT_STAGE)
        return surface.final_symbols if surface is not None else ()
    return ()


def orderable_symbols_for(plan: KISResourcePlan, strategy_id: str) -> tuple[str, ...]:
    sid = str(strategy_id or "").upper().strip()
    if sid == "KALCB":
        surface = candidate_surface_for(plan, "KALCB")
    elif sid == "OLR":
        surface = candidate_surface_for(plan, "OLR", OLR_FINAL_ARTIFACT_STAGE)
    else:
        surface = None
    return surface.orderable_symbols if surface is not None else ()


def target_strategy_ids_for_bar(
    plan: KISResourcePlan,
    *,
    symbol: str,
    timestamp: datetime,
    available_strategy_ids: Sequence[str],
    held_or_pending_symbols: Mapping[str, Sequence[str]] | None = None,
) -> tuple[str, ...]:
    current = _kst_time(timestamp)
    normalized_symbol = _normalize_symbol(symbol)
    held = {str(key).upper().strip(): {_normalize_symbol(item) for item in values} for key, values in dict(held_or_pending_symbols or {}).items()}
    targets: list[str] = []
    for raw_sid in available_strategy_ids:
        sid = str(raw_sid or "").upper().strip()
        if sid == "KALCB":
            active = set(active_symbols_for(plan, "KALCB"))
            orderable = set(orderable_symbols_for(plan, "KALCB"))
            if _in_kalcb_entry_window(plan, current) and normalized_symbol in (orderable or active):
                targets.append(sid)
            elif normalized_symbol in held.get("KALCB", set()):
                targets.append(sid)
        elif sid == "OLR":
            orderable = set(orderable_symbols_for(plan, "OLR"))
            if _time_ge(current, OLR_RUNTIME_ENABLE) and normalized_symbol in (orderable | held.get("OLR", set())):
                targets.append(sid)
    return tuple(dict.fromkeys(targets))


def _build_lease_windows(
    *,
    profile: KISLimitProfile,
    surfaces: Sequence[CandidateSurface],
    kalcb_config: KALCBConfig | None,
    completed_bar_source: str,
) -> list[ResourceLeaseWindow]:
    windows: list[ResourceLeaseWindow] = []
    rest_budget = max(0, profile.rest_calls_per_5m - profile.order_rest_reserve_per_5m - profile.oms_reconcile_reserve_per_5m)
    ws_budget = max(0, profile.ws_max_registrations - profile.ws_reserved_execution_regs)
    kalcb_surface = _surface(surfaces, "KALCB")
    olr_stage1 = _surface(surfaces, "OLR", OLR_STAGE1_ARTIFACT_STAGE)
    olr_final = _surface(surfaces, "OLR", OLR_FINAL_ARTIFACT_STAGE)
    if kalcb_surface is not None and kalcb_config is not None:
        windows.append(
            ResourceLeaseWindow(
                name="kalcb_entry_discovery",
                starts_at_kst=_time_text(kalcb_config.session_open),
                ends_at_kst=_time_text(kalcb_config.entry_window_end),
                strategy_id="KALCB",
                rest_symbols=(),
                rest_endpoint_class="none",
                rest_call_count=0,
                rest_calls_per_5m_budget=rest_budget,
                ws_symbols=kalcb_surface.active_symbols,
                ws_regs_per_symbol=max(1, int(kalcb_config.ws_hot_regs_per_symbol)),
                ws_reg_count=len(kalcb_surface.active_symbols) * max(1, int(kalcb_config.ws_hot_regs_per_symbol)),
                ws_reg_budget=ws_budget,
                order_reserve=profile.order_rest_reserve_per_5m,
                oms_reserve=profile.oms_reconcile_reserve_per_5m,
                source="kalcb active_symbols from finalized artifact",
            )
        )
        windows.append(
            ResourceLeaseWindow(
                name="kalcb_position_management",
                starts_at_kst=_time_text(kalcb_config.entry_window_end),
                ends_at_kst=_time_text(kalcb_config.flatten_time),
                strategy_id="KALCB",
                rest_symbols=(),
                rest_endpoint_class="held_position_dynamic",
                rest_call_count=0,
                rest_calls_per_5m_budget=rest_budget,
                ws_symbols=(),
                ws_regs_per_symbol=max(1, int(kalcb_config.ws_hot_regs_per_symbol)),
                ws_reg_count=0,
                ws_reg_budget=ws_budget,
                order_reserve=profile.order_rest_reserve_per_5m,
                oms_reserve=profile.oms_reconcile_reserve_per_5m,
                source="dynamic held/pending KALCB symbols only",
                metadata={"max_positions": int(kalcb_config.max_positions)},
            )
        )
    if olr_stage1 is not None:
        acquisition_start = kalcb_config.entry_window_end if kalcb_config is not None else time(14, 20)
        rest_symbols = olr_stage1.frontier_symbols
        rest_calls = len(rest_symbols) if completed_bar_source == "paced_rest" else 0
        windows.append(
            ResourceLeaseWindow(
                name="olr_stage1_bar_acquisition",
                starts_at_kst=_time_text(acquisition_start),
                ends_at_kst=_time_text(OLR_FINAL_CUTOFF),
                strategy_id="OLR",
                rest_symbols=rest_symbols,
                rest_endpoint_class="completed_5m_chart",
                rest_call_count=rest_calls,
                rest_calls_per_5m_budget=rest_budget,
                ws_symbols=(),
                ws_regs_per_symbol=1,
                ws_reg_count=0,
                ws_reg_budget=ws_budget,
                order_reserve=profile.order_rest_reserve_per_5m,
                oms_reserve=profile.oms_reconcile_reserve_per_5m,
                source=completed_bar_source,
            )
        )
    if olr_final is not None:
        windows.append(
            ResourceLeaseWindow(
                name="olr_final_runtime",
                starts_at_kst=_time_text(OLR_RUNTIME_ENABLE),
                ends_at_kst=_time_text(CLOSE_WINDOW_END),
                strategy_id="OLR",
                rest_symbols=(),
                rest_endpoint_class="none",
                rest_call_count=0,
                rest_calls_per_5m_budget=rest_budget,
                ws_symbols=(),
                ws_regs_per_symbol=1,
                ws_reg_count=0,
                ws_reg_budget=ws_budget,
                order_reserve=profile.order_rest_reserve_per_5m,
                oms_reserve=profile.oms_reconcile_reserve_per_5m,
                source="final artifact orderable symbols; no persistent OLR stage1 websocket lease",
                metadata={"final_symbols": list(olr_final.final_symbols), "orderable_symbols": list(olr_final.orderable_symbols)},
            )
        )
    if kalcb_surface is not None or olr_final is not None:
        windows.append(
            ResourceLeaseWindow(
                name="close_order_window",
                starts_at_kst=_time_text(CLOSE_WINDOW_START),
                ends_at_kst=_time_text(CLOSE_WINDOW_END),
                strategy_id="OMS",
                rest_symbols=(),
                rest_endpoint_class="orders_and_reconciliation",
                rest_call_count=0,
                rest_calls_per_5m_budget=rest_budget,
                ws_symbols=(),
                ws_regs_per_symbol=1,
                ws_reg_count=0,
                ws_reg_budget=ws_budget,
                order_reserve=profile.order_rest_reserve_per_5m * 2,
                oms_reserve=profile.oms_reconcile_reserve_per_5m,
                source="shared close-window OMS reserve",
            )
        )
    return windows


def _kalcb_surface_failures(surface: CandidateSurface, config: KALCBConfig) -> list[str]:
    failures: list[str] = []
    if len(surface.active_symbols) > int(config.ws_budget):
        failures.append(f"kalcb_ws_budget_exceeded:active={len(surface.active_symbols)} ws_budget={config.ws_budget}")
    if len(surface.frontier_symbols) > int(config.frontier_size):
        failures.append(f"kalcb_frontier_size_exceeded:frontier={len(surface.frontier_symbols)} frontier_size={config.frontier_size}")
    if not set(surface.active_symbols).issubset(set(surface.frontier_symbols)):
        failures.append("kalcb_active_symbols_not_subset_of_frontier_symbols")
    if surface.metadata.get("active_symbols_present") is not True:
        failures.append("kalcb_active_symbols_missing_from_artifact_metadata")
    if surface.metadata.get("frontier_symbols_present") is not True:
        failures.append("kalcb_frontier_symbols_missing_from_artifact_metadata")
    if surface.candidate_count > 0 and not surface.active_symbols:
        failures.append("kalcb_active_symbols_empty_for_nonempty_artifact")
    if surface.metadata.get("frontier_rest_budget_symbols_per_5m_present") is not True:
        failures.append("kalcb_frontier_rest_budget_symbols_per_5m_missing")
    if str(surface.metadata.get("active_budget_source") or "") != "ws_budget":
        failures.append("kalcb_active_budget_source_missing_or_invalid")
    if surface.metadata.get("frontier_rest_budget_symbols_per_5m_present") is True:
        try:
            actual_rest_budget = int(surface.metadata.get("frontier_rest_budget_symbols_per_5m"))
            expected_rest_budget = int(surface.metadata.get("expected_frontier_rest_budget_symbols_per_5m"))
        except (TypeError, ValueError):
            failures.append("kalcb_frontier_rest_budget_symbols_per_5m_invalid")
        else:
            if actual_rest_budget != expected_rest_budget:
                failures.append("kalcb_frontier_rest_budget_symbols_per_5m_mismatch")
    if _metadata_count_mismatch(surface.metadata.get("reported_active_symbol_count"), len(surface.active_symbols)):
        failures.append("kalcb_active_symbol_count_mismatch")
    if _metadata_count_mismatch(surface.metadata.get("reported_frontier_symbol_count"), len(surface.frontier_symbols)):
        failures.append("kalcb_frontier_symbol_count_mismatch")
    if bool(config.entry_plan_frontier_branch_universe) and len(surface.orderable_symbols) > int(config.ws_budget):
        failures.append("kalcb_frontier_branch_universe_requires_explicit_ws_budget_for_all_orderable_symbols")
    return failures


def _olr_surface_failures(surface: CandidateSurface, config: OLRConfig) -> list[str]:
    failures: list[str] = []
    if surface.artifact_stage == OLR_STAGE1_ARTIFACT_STAGE and surface.candidate_count > int(config.research_top_long_count):
        failures.append(f"olr_stage1_count_exceeded:selected={surface.candidate_count} research_top_long_count={config.research_top_long_count}")
    if surface.artifact_stage == OLR_FINAL_ARTIFACT_STAGE:
        if surface.candidate_count > int(config.afternoon_top_n):
            failures.append(f"olr_final_count_exceeded:selected={surface.candidate_count} afternoon_top_n={config.afternoon_top_n}")
        if len(surface.orderable_symbols) > int(config.overnight_slot_count):
            failures.append(f"olr_orderable_count_exceeded:orderable={len(surface.orderable_symbols)} overnight_slot_count={config.overnight_slot_count}")
    return failures


def _lease_failures(profile: KISLimitProfile, windows: Sequence[ResourceLeaseWindow]) -> list[str]:
    failures: list[str] = []
    for window in windows:
        if window.ws_reg_count > window.ws_reg_budget:
            failures.append(f"kis_ws_budget_exceeded:{window.name}:required={window.ws_reg_count}:budget={window.ws_reg_budget}")
        if window.rest_call_count > 0:
            rest_capacity = _window_rest_capacity(window)
            if rest_capacity <= 0:
                failures.append(f"kis_rest_budget_exceeded:{window.name}:no_optional_rest_capacity_after_reserves")
            elif window.rest_call_count > rest_capacity:
                failures.append(f"kis_rest_budget_exceeded:{window.name}:required={window.rest_call_count}:budget={rest_capacity}")
        if window.order_reserve + window.oms_reserve >= profile.rest_calls_per_5m:
            failures.append(f"kis_rest_reserve_exceeds_capacity:{window.name}")
    return failures


def _olr_acquisition_failures(
    profile: KISLimitProfile,
    windows: Sequence[ResourceLeaseWindow],
    *,
    completed_bar_source: str,
) -> list[str]:
    if completed_bar_source != "paced_rest":
        return []
    failures: list[str] = []
    for window in windows:
        if window.name != "olr_stage1_bar_acquisition" or window.rest_call_count <= 0:
            continue
        available = max(0.0, _seconds_between(_parse_time(window.starts_at_kst), _parse_time(window.ends_at_kst)) - OLR_REST_SAFETY_BUFFER_S)
        required = float(window.rest_call_count) * float(profile.rest_min_interval_s)
        if required > available:
            failures.append(
                f"olr_stage1_rest_acquisition_window_exceeded:required_s={required:.3f}:available_s={available:.3f}"
            )
    return failures


def _surface(surfaces: Sequence[CandidateSurface], strategy_id: str, stage: str | None = None) -> CandidateSurface | None:
    for surface in surfaces:
        if surface.strategy_id != strategy_id:
            continue
        if stage is None or surface.artifact_stage == stage:
            return surface
    return None


def _candidate_symbols(candidates: Sequence[Any]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_normalize_symbol(getattr(candidate, "symbol", "")) for candidate in candidates if str(getattr(candidate, "symbol", "")).strip()))


def _metadata_symbols(metadata: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = metadata.get(key)
    if isinstance(value, str):
        raw = [part.strip() for part in value.split(",")]
    else:
        raw = list(value or ()) if isinstance(value, Sequence) else []
    return tuple(dict.fromkeys(_normalize_symbol(item) for item in raw if str(item).strip()))


def _metadata_count_mismatch(value: Any, actual: int) -> bool:
    if value in (None, ""):
        return False
    try:
        return int(value) != int(actual)
    except (TypeError, ValueError):
        return True


def _kalcb_frontier_rest_budget_symbols_per_5m(config: KALCBConfig) -> int:
    raw = int((5 * 60 / max(float(config.rest_min_interval_paper_s), 1e-9)) * max(min(float(config.frontier_rest_safety_fraction), 1.0), 0.01))
    return max(1, raw)


def _normalize_symbol(symbol: Any) -> str:
    return str(symbol or "").strip().zfill(6)


def _normalize_olr_stage(stage: str) -> str:
    value = str(stage or "").strip().lower()
    if value in {"stage1", "daily", OLR_STAGE1_ARTIFACT_STAGE}:
        return OLR_STAGE1_ARTIFACT_STAGE
    if value in {"final", "afternoon", OLR_FINAL_ARTIFACT_STAGE}:
        return OLR_FINAL_ARTIFACT_STAGE
    return stage


def _in_kalcb_entry_window(plan: KISResourcePlan, current: time) -> bool:
    for window in plan.lease_windows:
        if window.name == "kalcb_entry_discovery":
            return _time_ge(current, _parse_time(window.starts_at_kst)) and _time_lt(current, _parse_time(window.ends_at_kst))
    return False


def _kst_time(timestamp: datetime) -> time:
    ts = timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=KST)
    return ts.astimezone(KST).time().replace(tzinfo=None)


def _parse_time(value: str) -> time:
    hour, minute = str(value).split(":", 1)
    return time(int(hour), int(minute[:2]))


def _time_text(value: time) -> str:
    return f"{int(value.hour):02d}:{int(value.minute):02d}"


def _time_ge(left: time, right: time) -> bool:
    return (left.hour, left.minute, left.second) >= (right.hour, right.minute, right.second)


def _time_lt(left: time, right: time) -> bool:
    return (left.hour, left.minute, left.second) < (right.hour, right.minute, right.second)


def _seconds_between(start: time, end: time) -> float:
    start_s = start.hour * 3600 + start.minute * 60 + start.second
    end_s = end.hour * 3600 + end.minute * 60 + end.second
    return float(max(0, end_s - start_s))


def _window_rest_capacity(window: ResourceLeaseWindow) -> int:
    duration_s = _seconds_between(_parse_time(window.starts_at_kst), _parse_time(window.ends_at_kst))
    return int((duration_s / 300.0) * max(int(window.rest_calls_per_5m_budget), 0))


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, date):
        return value.isoformat()
    return value
